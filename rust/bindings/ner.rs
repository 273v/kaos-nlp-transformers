//! PyO3 wrapper for the GLiNER (zero-shot NER) extractor.
//!
//! ``PyNerBackend`` is the Python-side handle on ``OrtGlinerExtractor``.
//! The Python ``GLiNERExtractor`` (in ``kaos_nlp_transformers.ner``)
//! holds one of these and dispatches the heavy ``extract()`` call via
//! ``asyncio.to_thread`` from any async surface so the event loop
//! stays free across the ort.run().
//!
//! Result contract: returns a list of Python dicts per input text,
//! each dict carrying the byte-offset span, the decoded text, the
//! label, and a sigmoid-normalized score in ``[0, 1]``. The Python
//! wrapper then maps these into typed ``Entity`` dataclasses.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::bindings::util::map_backend_error;
use crate::core::device::Device;
use crate::core::error::BackendError;
use crate::core::model_registry::lookup_ner;
use crate::core::ner::{ExtractParams, NerExtractor, OrtGlinerExtractor};

/// Python-facing handle on a loaded GLiNER backend.
#[pyclass(name = "NerBackend", module = "kaos_nlp_transformers._rust.ner")]
pub(crate) struct PyNerBackend {
    inner: Arc<dyn NerExtractor>,
    model_id: String,
    device_str: String,
}

#[pymethods]
impl PyNerBackend {
    /// Load a registered GLiNER extractor.
    #[staticmethod]
    #[pyo3(signature = (model_id, *, device = "cpu", cache_dir = None))]
    fn load(
        py: Python<'_>,
        model_id: &str,
        device: &str,
        cache_dir: Option<&str>,
    ) -> PyResult<Self> {
        let model = lookup_ner(model_id).ok_or_else(|| {
            map_backend_error(py, BackendError::ModelNotRegistered(model_id.to_string()))
        })?;

        let dev = Device::parse(device).map_err(|e| map_backend_error(py, e))?;
        let cache: Option<PathBuf> = cache_dir.map(PathBuf::from);

        let backend = py
            .detach(|| OrtGlinerExtractor::load(model, &dev, cache.as_deref()))
            .map_err(|e| map_backend_error(py, e))?;

        let model_id = backend.model_id().to_string();
        let device_str = backend.device().to_string();

        Ok(Self {
            inner: Arc::new(backend),
            model_id,
            device_str,
        })
    }

    /// Run extraction over a batch of input texts against the given
    /// label list. Returns a Python list whose i-th element is a list
    /// of dicts (one per detected entity in input ``texts[i]``).
    ///
    /// Each entity dict has keys: ``start`` (byte offset, int),
    /// ``end`` (byte offset, int), ``text`` (the substring, str),
    /// ``label`` (str — the user's label string), ``score`` (float in
    /// ``[0, 1]``).
    #[pyo3(signature = (
        texts,
        labels,
        *,
        threshold = 0.5,
        max_width = 12,
        flat_ner = true,
        dup_label = false,
        multi_label = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn extract<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
        labels: Vec<String>,
        threshold: f32,
        max_width: usize,
        flat_ner: bool,
        dup_label: bool,
        multi_label: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let params = ExtractParams {
            threshold,
            max_width,
            flat_ner,
            dup_label,
            multi_label,
        };

        if texts.is_empty() {
            return Ok(PyList::empty(py));
        }

        let backend = self.inner.clone();
        let entities = py
            .detach(move || {
                let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
                backend.extract(&text_refs, &label_refs, params)
            })
            .map_err(|e| map_backend_error(py, e))?;

        // Marshal Vec<Vec<Entity>> → list of list of dicts.
        let outer = PyList::empty(py);
        for per_text in &entities {
            let inner = PyList::empty(py);
            for ent in per_text {
                let d = PyDict::new(py);
                d.set_item("start", ent.start)?;
                d.set_item("end", ent.end)?;
                d.set_item("text", ent.text.as_str())?;
                d.set_item("label", ent.label.as_str())?;
                d.set_item("score", ent.score)?;
                inner.append(d)?;
            }
            outer.append(inner)?;
        }

        Ok(outer)
    }

    #[getter]
    fn model_id(&self) -> &str {
        &self.model_id
    }

    #[getter]
    fn device(&self) -> &str {
        &self.device_str
    }
}

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "ner")?;
    m.add_class::<PyNerBackend>()?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_transformers._rust.ner", &m)?;
    Ok(())
}
