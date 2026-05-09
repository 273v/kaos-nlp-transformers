//! PyO3 wrapper for the cross-encoder reranker.
//!
//! ``PyCrossEncoderBackend`` is the Python-side handle to
//! ``OrtCrossEncoder``. The Python ``CrossEncoderReranker`` (in
//! ``kaos_nlp_transformers.reranker``) holds one of these and
//! dispatches the (synchronous Rust call → async coroutine) wrap via
//! ``asyncio.to_thread`` so the event loop stays free.
//!
//! Score contract: returns sigmoid-normalized [0, 1] floats — same
//! contract the Python ``rerank()`` documented before KNT-601.

use std::path::PathBuf;
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::bindings::util::map_backend_error;
use crate::core::device::Device;
use crate::core::error::BackendError;
use crate::core::model_registry::lookup_reranker;
use crate::core::reranker::{CrossEncoder, OrtCrossEncoder};

/// Python-facing handle on a loaded cross-encoder backend.
#[pyclass(
    name = "CrossEncoderBackend",
    module = "kaos_nlp_transformers._rust.reranker"
)]
pub(crate) struct PyCrossEncoderBackend {
    inner: Arc<dyn CrossEncoder>,
    model_id: String,
    device_str: String,
}

#[pymethods]
impl PyCrossEncoderBackend {
    /// Load a registered cross-encoder reranker.
    #[staticmethod]
    #[pyo3(signature = (model_id, *, device = "cpu", cache_dir = None))]
    fn load(
        py: Python<'_>,
        model_id: &str,
        device: &str,
        cache_dir: Option<&str>,
    ) -> PyResult<Self> {
        let model = lookup_reranker(model_id).ok_or_else(|| {
            map_backend_error(py, BackendError::ModelNotRegistered(model_id.to_string()))
        })?;

        let dev = Device::parse(device).map_err(|e| map_backend_error(py, e))?;
        let cache: Option<PathBuf> = cache_dir.map(PathBuf::from);

        let backend = py
            .detach(|| OrtCrossEncoder::load(model, &dev, cache.as_deref()))
            .map_err(|e| map_backend_error(py, e))?;

        let model_id = backend.model_id().to_string();
        let device_str = backend.device().to_string();

        Ok(Self {
            inner: Arc::new(backend),
            model_id,
            device_str,
        })
    }

    /// Score a batch of (query, passage) pairs. Returns a float32
    /// numpy array of shape ``(n_pairs,)`` with sigmoid-normalized
    /// scores in [0, 1].
    ///
    /// The Python wrapper calls this from inside ``asyncio.to_thread``
    /// so the event loop stays free across the heavy ort.run().
    #[pyo3(signature = (queries, passages, batch_size = 32))]
    fn score<'py>(
        &self,
        py: Python<'py>,
        queries: Vec<String>,
        passages: Vec<String>,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if queries.len() != passages.len() {
            return Err(map_backend_error(
                py,
                BackendError::inference(format!(
                    "queries ({}) and passages ({}) must have the same length",
                    queries.len(),
                    passages.len()
                )),
            ));
        }

        if queries.is_empty() {
            let arr = ndarray::Array1::<f32>::zeros(0);
            return Ok(arr.into_pyarray(py));
        }

        let backend = self.inner.clone();
        let scores = py
            .detach(move || {
                let pairs: Vec<(&str, &str)> = queries
                    .iter()
                    .zip(passages.iter())
                    .map(|(q, p)| (q.as_str(), p.as_str()))
                    .collect();
                backend.score_pairs(&pairs, batch_size)
            })
            .map_err(|e| map_backend_error(py, e))?;

        let arr = ndarray::Array1::from_vec(scores);
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
    let m = PyModule::new(py, "reranker")?;
    m.add_class::<PyCrossEncoderBackend>()?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_transformers._rust.reranker", &m)?;
    Ok(())
}
