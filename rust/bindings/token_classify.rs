//! PyO3 wrapper for BERT-style token classifiers (`core::token_classify`).
//!
//! ``PyTokenClassifierBackend`` is the Python-side handle on
//! ``OrtTokenClassifier``. The Python ``PiiDetector`` wrapper holds
//! one and dispatches the heavy ``classify()`` call via
//! ``asyncio.to_thread`` from any async surface.
//!
//! Result contract: same per-text list-of-dicts shape as the GLiNER
//! binding (``core::ner``), so the Python layer can re-use the
//! ``Entity`` dataclass. Each dict has keys: ``start`` (char offset),
//! ``end`` (char offset), ``text`` (substring), ``label`` (PII
//! category, e.g. "PERSON" / "EMAIL_ADDRESS" — BIO prefix stripped),
//! ``score`` (softmax confidence, conservative min-across-span in
//! ``[0, 1]``).

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::bindings::util::map_backend_error;
use crate::core::device::Device;
use crate::core::error::BackendError;
use crate::core::model_registry::lookup_pii;
use crate::core::token_classify::{OrtTokenClassifier, TokenClassifier};

/// Python-facing handle on a loaded token-classification backend.
#[pyclass(
    name = "TokenClassifierBackend",
    module = "kaos_nlp_transformers._rust.token_classify"
)]
pub(crate) struct PyTokenClassifierBackend {
    inner: Arc<dyn TokenClassifier>,
    model_id: String,
    device_str: String,
    labels: Vec<String>,
}

#[pymethods]
impl PyTokenClassifierBackend {
    /// Load a registered token classifier (currently the PII model).
    #[staticmethod]
    #[pyo3(signature = (model_id, *, device = "cpu", cache_dir = None))]
    fn load(
        py: Python<'_>,
        model_id: &str,
        device: &str,
        cache_dir: Option<&str>,
    ) -> PyResult<Self> {
        // Only PII registry today; the trait surface is shared in case
        // future closed-label NER models slot in.
        let model = lookup_pii(model_id).ok_or_else(|| {
            map_backend_error(py, BackendError::ModelNotRegistered(model_id.to_string()))
        })?;

        let dev = Device::parse(device).map_err(|e| map_backend_error(py, e))?;
        let cache: Option<PathBuf> = cache_dir.map(PathBuf::from);

        let backend = py
            .detach(|| OrtTokenClassifier::load(model, &dev, cache.as_deref()))
            .map_err(|e| map_backend_error(py, e))?;

        let model_id = backend.model_id().to_string();
        let device_str = backend.device().to_string();
        let labels = backend.labels().to_vec();

        Ok(Self {
            inner: Arc::new(backend),
            model_id,
            device_str,
            labels,
        })
    }

    /// Run classification over a batch of texts. Returns a list whose
    /// i-th element is a list of entity dicts (one per detected span
    /// in ``texts[i]``). Each dict has ``start``, ``end`` (char
    /// offsets), ``text``, ``label``, ``score``.
    #[pyo3(signature = (texts, *, score_threshold = 0.5))]
    fn classify<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
        score_threshold: f32,
    ) -> PyResult<Bound<'py, PyList>> {
        if texts.is_empty() {
            return Ok(PyList::empty(py));
        }

        let backend = self.inner.clone();
        let entities = py
            .detach(move || {
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                backend.classify(&refs, score_threshold)
            })
            .map_err(|e| map_backend_error(py, e))?;

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

    /// The set of distinct entity-category labels this model emits
    /// (post-BIO strip).
    #[getter]
    fn labels(&self) -> Vec<String> {
        self.labels.clone()
    }
}

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "token_classify")?;
    m.add_class::<PyTokenClassifierBackend>()?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_transformers._rust.token_classify", &m)?;
    Ok(())
}
