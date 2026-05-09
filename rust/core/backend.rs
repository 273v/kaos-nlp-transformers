//! Backend trait — the abstract surface every embedding inference
//! engine implements. Today there's exactly one implementation
//! (``ort_runtime::OrtBackend``); the trait is here so the PyO3
//! binding layer can hold ``Arc<dyn Backend + Send + Sync>`` and the
//! ``model2vec`` Python path stays separate.

use crate::core::error::Result;
use ndarray::Array2;

/// Embedding inference backend.
///
/// Every implementation must:
///   * Produce ``(batch, dim)`` `float32` arrays.
///   * L2-normalize the output rows when the registry entry's
///     ``normalize`` flag is true (KNT-101).
///   * Be ``Send + Sync`` so the PyO3 binding layer can hold it
///     across a ``py.allow_threads`` boundary.
pub trait Backend: Send + Sync {
    /// Embed a batch of texts. Output shape: ``(texts.len(), dim())``.
    fn embed(&self, texts: &[&str], batch_size: usize) -> Result<Array2<f32>>;

    /// Embedding dimension produced by this backend.
    fn dim(&self) -> usize;

    /// HF Hub model id this backend was loaded for.
    fn model_id(&self) -> &str;

    /// Device this backend runs on (``"cpu"``, ``"cuda:0"``, …).
    fn device(&self) -> &str;
}
