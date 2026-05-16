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

/// All NLI (natural language inference) cross-encoders. Mirror of
/// ``kaos_nlp_transformers.models.NLI_REGISTRY``. Same shape as
/// ``RERANKER_REGISTRY``: cross-encoder over ``(premise, hypothesis)``
/// pairs. The model output is a 3-way logit head;
/// ``core::nli::OrtNliClassifier`` applies softmax along axis 1 and
/// re-orders from the model's ``id2label`` permutation into the
/// canonical ``(entailment, neutral, contradiction)`` tuple expected
/// by the ``NLIScorer`` protocol on the kaos-llm-core side.
/// ``dim`` is recorded as 3 for the three-class output.
///
/// Default entry uses the ``Xenova/nli-deberta-v3-base`` ONNX
/// re-export of the Apache-2.0 ``cross-encoder/nli-deberta-v3-base``
/// upstream — Xenova ships the full quantization matrix; the upstream
/// only ships an AVX-512-VNNI quantized variant which fails on CPUs
/// without VNNI. License chain: weights are Apache-2.0 from upstream;
/// Xenova fork is a pure 🤗 Optimum re-export with no fine-tuning.
pub const NLI_REGISTRY: &[RegisteredModel] = &[RegisteredModel {
    model_id: "Xenova/nli-deberta-v3-base",
    // 2025-07-14 HEAD of main; verified via /api/models/{id} on 2026-05-15.
    revision: "80a99030ce45a69a39ea2a6f50756d03859ff521",
    // Use the portable quantized variant (244 MB) — matches the
    // bge-reranker-base precedent of "quantized by default" and is
    // the variant Transformers.js callers actually pull. The plain
    // int8 export (model_int8.onnx, 223 MB) is also available; the
    // quantized variant is chosen here because it's better tested
    // across the HF Optimum lineage.
    onnx_filename: "onnx/model_quantized.onnx",
    tokenizer_filename: "tokenizer.json",
    pooling: Pooling::Cls, // unused for NLI; classifier head reads CLS.
    normalize: false,      // softmax-normalized in core::nli, not L2.
    dim: 3,                // 3-class output: entailment / neutral / contradiction.
    max_seq_len: 512,
    license: "Apache-2.0",
}];

/// All GLiNER (zero-shot NER) checkpoints. Mirror of
/// ``kaos_nlp_transformers.models.NER_REGISTRY``. GLiNER is a
/// span-extraction model — input is the prompt
/// ``[ENT] label_1 [ENT] label_2 ... [SEP] text`` and the output is a
/// span-vs-label scoring tensor that ``core::ner`` decodes into
/// ``(start, end, label, score)`` tuples.
///
/// ``dim`` is recorded as 0 because the output shape is
/// ``(batch, n_spans, n_labels)`` rather than a fixed embedding
/// dimension; the per-call shape is decoded in ``core::ner``.
/// ``normalize`` is false (no L2 norm; the score head is its own
/// sigmoid-normalized contract).
///
/// License chain: upstream ``urchade/gliner_medium-v2.1`` is
/// Apache-2.0 (DeBERTa-v3-base backbone, 195M params); the
/// onnx-community fork is a pure ONNX re-export with no
/// fine-tuning. Excluded sibling: ``urchade/gliner_base`` and
/// ``onnx-community/gliner_base`` (CC-BY-NC 4.0 — non-commercial,
/// flagged in ``NER_EXCLUDED`` on the Python side).
pub const NER_REGISTRY: &[RegisteredModel] = &[
    RegisteredModel {
        model_id: "onnx-community/gliner_medium-v2.1",
        // Verified via /api/models/{id}/revision/{sha} on 2026-05-15.
        revision: "959437589dc623d4c0a93f6e2828213567929cde",
        // Use the fp32 variant (~746 MiB). The int8-quantized export
        // at ``onnx/model_quantized.onnx`` is severely degraded — its
        // sigmoid scores cap around 0.13 on inputs where the
        // PyTorch reference scores 0.99, so the default-threshold
        // 0.5 returns zero spans. (Cross-checked 2026-05-15 against
        // ``gliner.GLiNER.from_pretrained("urchade/gliner_medium-v2.1")``
        // — Python reference produced 0.9935/0.9772 on
        // "Barack Obama was born in Hawaii." while the same ONNX
        // session on the quantized export produced 0.131/0.111.)
        onnx_filename: "onnx/model.onnx",
        tokenizer_filename: "tokenizer.json",
        // GLiNER reads from per-token hidden states, not a CLS pooled
        // vector. The pooling field is unused by core::ner (we keep
        // CLS as a placeholder for shape symmetry with RegisteredModel).
        pooling: Pooling::Cls,
        normalize: false,
        dim: 0,
        max_seq_len: 384,
        license: "Apache-2.0",
    },
    RegisteredModel {
        model_id: "onnx-community/gliner_multi-v2.1",
        revision: "6ddaeb9413b0e71ad8457da1aab378a165b24058",
        // fp32 variant — same quantization concern as above. The
        // multilingual export is ~1.08 GiB fp32; consumers who need
        // a smaller footprint should switch to the upstream
        // urchade/* pytorch checkpoint via a separate code path.
        onnx_filename: "onnx/model.onnx",
        tokenizer_filename: "tokenizer.json",
        pooling: Pooling::Cls,
        normalize: false,
        dim: 0,
        max_seq_len: 384,
        license: "Apache-2.0",
    },
];

/// All PII (personally-identifiable-information) token-classifier
/// models. Mirror of ``kaos_nlp_transformers.models.PII_REGISTRY``.
/// Architecture: BERT-style token classifier with BIO encoding —
/// outputs ``(batch, seq, num_classes)`` logits;
/// ``core::token_classify`` decodes them into ``(start_char,
/// end_char, label, score)`` spans.
///
/// The class label set is **baked into the model** via
/// ``config.json::id2label`` (read at load time, not supplied at
/// inference like GLiNER). ``dim`` records the class count for
/// shape symmetry but the live shape is read from the loaded
/// session.
///
/// License chain: ``onnx-community/bert-small-pii-detection-ONNX``
/// declares ``license: apache-2.0`` on its card directly, with
/// ``base_model: gravitee-io/bert-small-pii-detection`` (also
/// Apache-2.0).
pub const PII_REGISTRY: &[RegisteredModel] = &[RegisteredModel {
    model_id: "onnx-community/bert-small-pii-detection-ONNX",
    // Verified via /api/models/{id}/revision/{sha} on 2026-05-16.
    revision: "6cb4e77c2b2c7f81e731b88cffa9b7a6fc675a4c",
    // int8-quantized variant — 27 MB on disk. BERT-small quantizes
    // cleanly (the quality-collapse pattern we saw on quantized
    // GLiNER is specific to the span-extraction head, not a
    // classification head). The new ``tests/scale/test_pii_quality_cuad.py``
    // gate catches accuracy regressions if this turns out wrong.
    onnx_filename: "onnx/model_int8.onnx",
    tokenizer_filename: "tokenizer.json",
    pooling: Pooling::Cls, // unused for token classification
    normalize: false,      // softmax-handled in core::token_classify
    // 49 classes = 1 (O) + 24 categories × 2 (B-/I-).
    dim: 49,
    // BERT-small has max_position_embeddings = 512.
    max_seq_len: 512,
    license: "Apache-2.0",
}];

/// Look up an embedding model by id. Returns None if not registered.
pub fn lookup_embedding(model_id: &str) -> Option<&'static RegisteredModel> {
    EMBEDDING_REGISTRY.iter().find(|m| m.model_id == model_id)
}

/// Look up a reranker by id.
pub fn lookup_reranker(model_id: &str) -> Option<&'static RegisteredModel> {
    RERANKER_REGISTRY.iter().find(|m| m.model_id == model_id)
}

/// Look up an NLI model by id.
pub fn lookup_nli(model_id: &str) -> Option<&'static RegisteredModel> {
    NLI_REGISTRY.iter().find(|m| m.model_id == model_id)
}

/// Look up a GLiNER (zero-shot NER) model by id.
pub fn lookup_ner(model_id: &str) -> Option<&'static RegisteredModel> {
    NER_REGISTRY.iter().find(|m| m.model_id == model_id)
}

/// Look up a PII (BERT token-classifier) model by id.
pub fn lookup_pii(model_id: &str) -> Option<&'static RegisteredModel> {
    PII_REGISTRY.iter().find(|m| m.model_id == model_id)
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
    fn nli_registry_has_deberta_v3() {
        let m = lookup_nli("Xenova/nli-deberta-v3-base").expect("nli-deberta-v3 in registry");
        assert_eq!(m.dim, 3);
        assert!(!m.normalize); // softmax-handled in core::nli
        assert_eq!(m.license, "Apache-2.0");
    }

    #[test]
    fn ner_registry_has_gliner_medium() {
        let m = lookup_ner("onnx-community/gliner_medium-v2.1").expect("gliner_medium in registry");
        assert_eq!(m.dim, 0); // span-vs-label tensor, decoded per-call
        assert!(!m.normalize);
        assert_eq!(m.license, "Apache-2.0");
        // fp32 model.onnx — int8 quantized export was tested and
        // rejected (scores cap around 0.13 vs the PyTorch reference's
        // 0.99). See model_registry.rs::NER_REGISTRY notes.
        assert_eq!(m.onnx_filename, "onnx/model.onnx");
    }

    #[test]
    fn ner_registry_has_gliner_multi() {
        let m = lookup_ner("onnx-community/gliner_multi-v2.1").expect("gliner_multi in registry");
        assert_eq!(m.dim, 0);
        assert_eq!(m.license, "Apache-2.0");
    }

    #[test]
    fn pii_registry_has_bert_small() {
        let m = lookup_pii("onnx-community/bert-small-pii-detection-ONNX")
            .expect("pii model in registry");
        assert_eq!(m.dim, 49); // O + 24 categories × 2 (B-/I-)
        assert_eq!(m.license, "Apache-2.0");
        assert_eq!(m.onnx_filename, "onnx/model_int8.onnx");
        assert_eq!(m.max_seq_len, 512);
    }

    #[test]
    fn every_revision_is_a_full_sha() {
        // KNT-003 contract: revisions are full SHAs (40 hex chars), never "main"
        // or short SHAs. This catches drift if someone copy-pastes from a
        // shorter Python registry by accident.
        for m in EMBEDDING_REGISTRY
            .iter()
            .chain(RERANKER_REGISTRY.iter())
            .chain(NLI_REGISTRY.iter())
            .chain(NER_REGISTRY.iter())
            .chain(PII_REGISTRY.iter())
        {
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
