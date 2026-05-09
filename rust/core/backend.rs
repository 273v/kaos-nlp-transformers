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

    /// Maximum sequence length the backend's tokenizer applies as the
    /// truncation cap. Consumers that chunk text before embedding (e.g.
    /// ``kaos_content.chunking.EmbeddingChunker``) read this to size
    /// their chunks so nothing is silently truncated.
    fn max_seq_len(&self) -> usize;

    /// Tokenize a batch of texts and return the per-text token count.
    /// Does NOT run inference; just the tokenizer pass. Used by the
    /// downstream chunker to decide whether a candidate chunk fits in
    /// ``max_seq_len`` before materializing it for ``embed()``.
    fn count_tokens(&self, texts: &[&str]) -> Result<Vec<usize>>;
}
