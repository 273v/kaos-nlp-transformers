//! PyO3 wrapper for the embedding backend.
//!
//! ``PyEmbeddingBackend`` is the Python-side handle to ``OrtBackend``.
//! The Python ``EmbeddingModel`` (in ``kaos_nlp_transformers.embedding``)
//! holds one of these as its ``_backend`` attribute when the Rust
//! path is selected.
//!
//! GIL discipline (audit KNT-602): every ``embed()`` call:
//!   1. takes ownership of ``Vec<String>`` from Python,
//!   2. enters ``py.detach(...)`` for the heavy
//!      tokenize → run → pool → normalize pipeline,
//!   3. re-acquires the GIL to convert the resulting ``Array2<f32>``
//!      into a ``PyArray2<f32>`` (zero-copy via from_owned_array).
//!
//! The ``Arc<dyn Backend + Send + Sync>`` lets multiple Python-side
//! ``EmbeddingModel`` instances share a backend (e.g., the
//! ``_load_rust_embedding_cached`` Python lru_cache will hand out the
//! same Arc to every caller asking for the same model).

use std::path::PathBuf;
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

use crate::bindings::util::map_backend_error;
use crate::core::backend::Backend;
use crate::core::device::Device;
use crate::core::error::BackendError;
use crate::core::model_registry::lookup_embedding;
use crate::core::ort_runtime::OrtBackend;

/// Python-facing handle on a loaded embedding backend.
#[pyclass(
    name = "EmbeddingBackend",
    module = "kaos_nlp_transformers._rust.embedding"
)]
pub(crate) struct PyEmbeddingBackend {
    inner: Arc<dyn Backend>,
}

#[pymethods]
impl PyEmbeddingBackend {
    /// Load a registered embedding model.
    ///
    /// Args:
    ///   model_id: HF Hub model id (must be in ``EMBEDDING_REGISTRY``).
    ///   device: ``"cpu"``, ``"cuda"``, ``"cuda:N"``, ``"openvino"``.
    ///   cache_dir: Optional override for the HF Hub cache.
    ///
    /// Raises:
    ///   ModelNotRegisteredError: model_id not in the Rust registry.
    ///   BackendNotInstalledError: device requires a feature flag the
    ///     wheel was not built with (e.g. cuda without --features gpu).
    ///   ModelLoadError: download / safetensors / session-build failure.
    #[staticmethod]
    #[pyo3(signature = (model_id, *, device = "cpu", cache_dir = None))]
    fn load(
        py: Python<'_>,
        model_id: &str,
        device: &str,
        cache_dir: Option<&str>,
    ) -> PyResult<Self> {
        let model = lookup_embedding(model_id).ok_or_else(|| {
            map_backend_error(py, BackendError::ModelNotRegistered(model_id.to_string()))
        })?;

        let dev = Device::parse(device).map_err(|e| map_backend_error(py, e))?;
        let cache: Option<PathBuf> = cache_dir.map(PathBuf::from);

        // Loading downloads weights + builds an ort Session — both
        // heavy I/O. Release the GIL while we do them.
        let backend = py
            .detach(|| OrtBackend::load(model, &dev, cache.as_deref()))
            .map_err(|e| map_backend_error(py, e))?;

        Ok(Self {
            inner: Arc::new(backend),
        })
    }

    /// Embed a batch of texts. Returns a ``(N, dim)`` float32 numpy array.
    ///
    /// The Python wrapper layer is responsible for re-checking shape +
    /// dim against the registry; this layer enforces it via the inner
    /// backend's own assertions and returns an EmbeddingError on mismatch.
    #[pyo3(signature = (texts, batch_size = 32))]
    fn embed<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if texts.is_empty() {
            // Matches the Python EmbeddingModel.embed contract: empty
            // input yields a (0, dim) array, no error.
            let arr = ndarray::Array2::<f32>::zeros((0, self.inner.dim()));
            return Ok(arr.into_pyarray(py));
        }

        // Hold an Arc clone so the closure is 'static-friendly without
        // borrowing &self across the allow_threads boundary.
        let backend = self.inner.clone();

        let arr = py
            .detach(move || {
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                backend.embed(&refs, batch_size)
            })
            .map_err(|e| map_backend_error(py, e))?;

        Ok(arr.into_pyarray(py))
    }

    /// Embedding dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// HF Hub model id this backend was loaded for.
    #[getter]
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    /// Device this backend runs on (``"cpu"``, ``"cuda:0"``, …).
    #[getter]
    fn device(&self) -> &str {
        self.inner.device()
    }

    /// Maximum sequence length the tokenizer applies as the
    /// truncation cap. Downstream consumers (chunkers) read this to
    /// size their chunks so embeddings don't silently truncate.
    #[getter]
    fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len()
    }

    /// Tokenize and return the non-pad token count per text. Useful
    /// for chunkers that need to decide whether a candidate chunk
    /// fits in ``max_seq_len`` before sending it to ``embed()``.
    /// No inference is run.
    fn count_tokens(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<usize>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let backend = self.inner.clone();
        py.detach(move || {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            backend.count_tokens(&refs)
        })
        .map_err(|e| map_backend_error(py, e))
    }
}

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "embedding")?;
    m.add_class::<PyEmbeddingBackend>()?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_transformers._rust.embedding", &m)?;
    Ok(())
}
