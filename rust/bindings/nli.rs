//! PyO3 wrapper for the NLI (natural language inference) cross-encoder.
//!
//! ``PyNliBackend`` is the Python-side handle on ``OrtNliClassifier``.
//! The Python ``NliModel`` (in ``kaos_nlp_transformers.nli``) holds one
//! of these and dispatches the heavy `score()` call via
//! ``asyncio.to_thread`` from its async public surface so the event
//! loop stays free.
//!
//! Score contract: returns a float32 numpy array of shape
//! ``(n_pairs, 3)``. Each row is a softmax-normalized probability
//! triple in the canonical ``(entailment, neutral, contradiction)``
//! order — the order expected by the ``NLIScorer`` Protocol on the
//! kaos-llm-core side, regardless of how the underlying ONNX
//! checkpoint permuted its head.

use std::path::PathBuf;
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

use crate::bindings::util::map_backend_error;
use crate::core::device::Device;
use crate::core::error::BackendError;
use crate::core::model_registry::lookup_nli;
use crate::core::nli::{NliClassifier, OrtNliClassifier};

/// Python-facing handle on a loaded NLI cross-encoder backend.
#[pyclass(name = "NliBackend", module = "kaos_nlp_transformers._rust.nli")]
pub(crate) struct PyNliBackend {
    inner: Arc<dyn NliClassifier>,
    model_id: String,
    device_str: String,
}

#[pymethods]
impl PyNliBackend {
    /// Load a registered NLI cross-encoder.
    #[staticmethod]
    #[pyo3(signature = (model_id, *, device = "cpu", cache_dir = None))]
    fn load(
        py: Python<'_>,
        model_id: &str,
        device: &str,
        cache_dir: Option<&str>,
    ) -> PyResult<Self> {
        let model = lookup_nli(model_id).ok_or_else(|| {
            map_backend_error(py, BackendError::ModelNotRegistered(model_id.to_string()))
        })?;

        let dev = Device::parse(device).map_err(|e| map_backend_error(py, e))?;
        let cache: Option<PathBuf> = cache_dir.map(PathBuf::from);

        let backend = py
            .detach(|| OrtNliClassifier::load(model, &dev, cache.as_deref()))
            .map_err(|e| map_backend_error(py, e))?;

        let model_id = backend.model_id().to_string();
        let device_str = backend.device().to_string();

        Ok(Self {
            inner: Arc::new(backend),
            model_id,
            device_str,
        })
    }

    /// Score a batch of (premise, hypothesis) pairs. Returns a float32
    /// numpy array of shape ``(n_pairs, 3)``. Each row is a
    /// probability triple in canonical
    /// ``(entailment, neutral, contradiction)`` order.
    ///
    /// The Python wrapper calls this from inside ``asyncio.to_thread``
    /// so the event loop stays free across the heavy ort.run().
    #[pyo3(signature = (premises, hypotheses, batch_size = 16))]
    fn score<'py>(
        &self,
        py: Python<'py>,
        premises: Vec<String>,
        hypotheses: Vec<String>,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if premises.len() != hypotheses.len() {
            return Err(map_backend_error(
                py,
                BackendError::inference(format!(
                    "premises ({}) and hypotheses ({}) must have the same length",
                    premises.len(),
                    hypotheses.len()
                )),
            ));
        }

        if premises.is_empty() {
            let arr = ndarray::Array2::<f32>::zeros((0, 3));
            return Ok(arr.into_pyarray(py));
        }

        let backend = self.inner.clone();
        let probs = py
            .detach(move || {
                let pairs: Vec<(&str, &str)> = premises
                    .iter()
                    .zip(hypotheses.iter())
                    .map(|(p, h)| (p.as_str(), h.as_str()))
                    .collect();
                backend.score_pairs(&pairs, batch_size)
            })
            .map_err(|e| map_backend_error(py, e))?;

        // Flatten `Vec<[f32; 3]>` → row-major `(n, 3)` numpy.
        let n = probs.len();
        let mut flat = Vec::with_capacity(n * 3);
        for triple in &probs {
            flat.extend_from_slice(triple);
        }
        // `from_shape_vec` consumes the flat vector — `(n, 3)` × f32 always
        // matches the buffer size we just built, so the unwrap is safe.
        let arr = ndarray::Array2::from_shape_vec((n, 3), flat)
            .expect("flat buffer matches (n, 3) by construction");
        Ok(arr.into_pyarray(py))
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
    let m = PyModule::new(py, "nli")?;
    m.add_class::<PyNliBackend>()?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_transformers._rust.nli", &m)?;
    Ok(())
}
