//! Rust mirror of the Python ``REGISTRY`` and ``RERANKER_REGISTRY``
//! dicts. Source of truth for: which models we serve, what revision SHA
//! they pin to, what ONNX/tokenizer file names to fetch from the hub,
//! and what pooling/normalization to apply.
//!
//! Audit-01 KNT-003 contract: every entry has an explicit revision
//! SHA — never ``main``. The Rust loader passes this revision through
//! to ``hf-hub`` (unlike fastembed-rs's ``pull_from_hf`` which
//! hard-codes ``main`` and is structurally incompatible with KNT-003;
//! see docs/MIGRATION_0_2_0.md §15).
//!
//! When models change in the Python ``models.py``, the entries here
//! must be kept in sync. The integration test in ``rust/core/backend.rs``
//! cross-checks each row against the Python REGISTRY values.

use crate::core::pooling::Pooling;

/// A model entry — encodes everything the loader and runtime need
/// before the first ``Session::run``.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisteredModel {
    /// HF Hub model id (org/repo).
    pub model_id: &'static str,
    /// Pinned commit SHA — NEVER "main". Min 7 chars.
    pub revision: &'static str,
    /// Path of the ONNX file on the HF Hub repo (under the pinned revision).
    pub onnx_filename: &'static str,
    /// Path of the tokenizer.json on the HF Hub repo.
    pub tokenizer_filename: &'static str,
    /// Pooling strategy this model was trained for.
    pub pooling: Pooling,
    /// Whether to L2-normalize the pooled vectors. KNT-101 says we
    /// always do this, but the field is here for forward compatibility
    /// with cross-encoder rerankers (which return scalar logits, not
    /// normalized vectors).
    pub normalize: bool,
    /// Embedding dimension produced by this model.
    pub dim: usize,
    /// Maximum sequence length (truncation cap).
    pub max_seq_len: usize,
    /// SPDX-style license identifier (must be permissive).
    pub license: &'static str,
}

/// All embedding-family models the Rust backend serves. Mirror of
/// ``kaos_nlp_transformers.models.REGISTRY`` filtered to entries with
/// ``backend = "fastembed"`` (post-0.2.0 these become ``backend = "ort"``).
///
/// model2vec entries don't appear here — they're served by the Python
/// model2vec library on a separate code path. See
/// docs/MIGRATION_0_2_0.md §11 for the model2vec invariant.
pub const EMBEDDING_REGISTRY: &[RegisteredModel] = &[RegisteredModel {
    model_id: "BAAI/bge-small-en-v1.5",
    revision: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    // Xenova-converted ONNX — same one fastembed Python serves.
    // Path under the pinned revision is `onnx/model.onnx` for the
    // Xenova fork; for the BAAI repo direct, it's `model.onnx`.
    // We use BAAI directly because the registry pins BAAI's SHA.
    onnx_filename: "onnx/model.onnx",
    tokenizer_filename: "tokenizer.json",
    // BGE-family is CLS-pooled.
    pooling: Pooling::Cls,
    normalize: true,
    dim: 384,
    max_seq_len: 512,
    license: "MIT",
}];

/// All reranker-family models. Mirror of
/// ``kaos_nlp_transformers.models.RERANKER_REGISTRY``. Cross-encoders
/// return a single relevance score per (query, passage) pair, not a
/// vector; ``dim`` is recorded as 1 for shape symmetry with
/// ``RegisteredModel`` — the actual scoring contract is `[0, 1]` after
/// sigmoid (handled in ``core::reranker``).
pub const RERANKER_REGISTRY: &[RegisteredModel] = &[RegisteredModel {
    model_id: "BAAI/bge-reranker-base",
    revision: "2cfc18c9415c912f9d8155881c133215df768a70",
    onnx_filename: "onnx/model.onnx",
    tokenizer_filename: "tokenizer.json",
    pooling: Pooling::Cls, // unused for cross-encoder; classifier head reads CLS.
    normalize: false,      // sigmoid-normalized in core::reranker, not L2.
    dim: 1,
    max_seq_len: 512,
    license: "MIT",
}];

/// Look up an embedding model by id. Returns None if not registered.
pub fn lookup_embedding(model_id: &str) -> Option<&'static RegisteredModel> {
    EMBEDDING_REGISTRY.iter().find(|m| m.model_id == model_id)
}

/// Look up a reranker by id.
pub fn lookup_reranker(model_id: &str) -> Option<&'static RegisteredModel> {
    RERANKER_REGISTRY.iter().find(|m| m.model_id == model_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_registry_has_bge_small() {
        let m = lookup_embedding("BAAI/bge-small-en-v1.5").expect("bge-small in registry");
        assert_eq!(m.dim, 384);
        assert_eq!(m.pooling, Pooling::Cls);
        assert!(m.normalize);
    }

    #[test]
    fn unknown_model_returns_none() {
        assert!(lookup_embedding("foo/bar").is_none());
    }

    #[test]
    fn reranker_registry_has_bge_reranker() {
        let m = lookup_reranker("BAAI/bge-reranker-base").expect("bge-reranker in registry");
        assert_eq!(m.dim, 1);
        assert!(!m.normalize);
    }

    #[test]
    fn every_revision_is_a_full_sha() {
        // KNT-003 contract: revisions are full SHAs (40 hex chars), never "main"
        // or short SHAs. This catches drift if someone copy-pastes from a
        // shorter Python registry by accident.
        for m in EMBEDDING_REGISTRY.iter().chain(RERANKER_REGISTRY.iter()) {
            assert_ne!(m.revision, "main", "{} pinned to 'main'", m.model_id);
            assert_eq!(
                m.revision.len(),
                40,
                "{} revision must be a full 40-char SHA, got {}",
                m.model_id,
                m.revision
            );
            assert!(
                m.revision.chars().all(|c| c.is_ascii_hexdigit()),
                "{} revision must be hex",
                m.model_id
            );
        }
    }
}
