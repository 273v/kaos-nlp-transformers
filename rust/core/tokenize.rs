//! Wrapper over the HF ``tokenizers`` crate.
//!
//! Two responsibilities:
//!
//! 1. **Loading**: from a ``tokenizer.json`` file path. The Python
//!    side never builds a tokenizer in code — every model in REGISTRY
//!    ships its tokenizer.json on the HF Hub at the pinned revision.
//!
//! 2. **Batch encoding**: produces ``(input_ids, attention_mask,
//!    token_type_ids)`` as ``Vec<Vec<i64>>`` triples. Uses
//!    ``BatchLongest`` padding by default (the per-batch policy ort
//!    uses today through fastembed). Truncation defaults to the
//!    model's max_position_embeddings (512 for bge-small).
//!
//! The shape contract: every returned vector is rectangular —
//! ``input_ids[i].len() == attention_mask[i].len() == seq_len`` for
//! all i, where seq_len is the longest tokenized sentence in the
//! batch (clamped to ``max_seq_len``).

use crate::core::error::{BackendError, Result};
use std::path::Path;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// A loaded tokenizer plus the policy we apply at encode time.
#[derive(Clone)]
pub struct TokenizerWrapper {
    inner: Tokenizer,
    /// Padding token id. Looked up from the tokenizer.json's added
    /// tokens. BERT-family default is 0.
    pub pad_id: u32,
    /// Maximum sequence length (truncation cap). 512 for bge-small.
    pub max_seq_len: usize,
}

/// Result of a batch encode. Shape: ``(batch_size, seq_len)`` for each.
pub struct EncodedBatch {
    /// Token ids, one row per input.
    pub input_ids: Vec<Vec<i64>>,
    /// 1 for real tokens, 0 for pad. Same shape as ``input_ids``.
    pub attention_mask: Vec<Vec<i64>>,
    /// Segment ids — all 0 for single-sentence inputs (the embedding
    /// case). Cross-encoder reranker uses non-zero values for the
    /// passage segment; that's a separate code path in
    /// ``core::reranker::encode_pair``.
    pub token_type_ids: Vec<Vec<i64>>,
    /// Batch dim (``input_ids.len()``).
    pub batch_size: usize,
    /// Sequence dim (length of every row after padding/truncation).
    pub seq_len: usize,
}

impl TokenizerWrapper {
    /// Load a tokenizer from a ``tokenizer.json`` file.
    pub fn from_file(path: impl AsRef<Path>, max_seq_len: usize) -> Result<Self> {
        let path_ref = path.as_ref();
        let mut inner = Tokenizer::from_file(path_ref).map_err(|e| BackendError::Io {
            path: path_ref.to_path_buf(),
            source_msg: e.to_string(),
        })?;

        // BERT-family pad_id is 0; we read it from the tokenizer's
        // padding config when present, otherwise default.
        let pad_id: u32 = inner.get_padding().map(|p| p.pad_id).unwrap_or(0);

        // Configure batch-longest padding to multiples of 1 (no
        // alignment), padding to the right, with the model's pad token.
        // This is what ort/fastembed does for BERT-family models.
        let pad_token = inner
            .get_padding()
            .map(|p| p.pad_token.clone())
            .unwrap_or_else(|| "[PAD]".to_string());

        inner.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token,
        }));

        inner
            .with_truncation(Some(TruncationParams {
                max_length: max_seq_len,
                ..Default::default()
            }))
            .map_err(BackendError::tokenization)?;

        Ok(Self {
            inner,
            pad_id,
            max_seq_len,
        })
    }

    /// Encode a batch of sentences. ``add_special_tokens`` is true for
    /// embedding / single-sentence cases (BERT [CLS] ... [SEP]).
    pub fn encode_batch(&self, texts: &[&str]) -> Result<EncodedBatch> {
        if texts.is_empty() {
            return Ok(EncodedBatch {
                input_ids: vec![],
                attention_mask: vec![],
                token_type_ids: vec![],
                batch_size: 0,
                seq_len: 0,
            });
        }

        // The tokenizers crate's encode_batch signature is
        // ``encode_batch_fast(inputs, add_special_tokens)``. Sentences
        // are passed as owned ``Vec<EncodeInput>`` so we map &str → owned.
        let inputs: Vec<tokenizers::EncodeInput<'_>> = texts
            .iter()
            .map(|s| tokenizers::EncodeInput::Single((*s).into()))
            .collect();

        let encodings = self
            .inner
            .encode_batch_fast(inputs, true)
            .map_err(BackendError::tokenization)?;

        let batch_size = encodings.len();
        // After BatchLongest padding every row is the same length.
        let seq_len = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);

        let mut input_ids = Vec::with_capacity(batch_size);
        let mut attention_mask = Vec::with_capacity(batch_size);
        let mut token_type_ids = Vec::with_capacity(batch_size);

        for enc in &encodings {
            input_ids.push(enc.get_ids().iter().map(|&u| u as i64).collect());
            attention_mask.push(enc.get_attention_mask().iter().map(|&u| u as i64).collect());
            token_type_ids.push(enc.get_type_ids().iter().map(|&u| u as i64).collect());
        }

        Ok(EncodedBatch {
            input_ids,
            attention_mask,
            token_type_ids,
            batch_size,
            seq_len,
        })
    }

    /// Encode a (query, passage) pair for cross-encoder scoring.
    /// Returns a single ``EncodedBatch`` with batch_size = 1 — caller
    /// stacks multiple pairs into a real batch.
    pub fn encode_pair(&self, query: &str, passage: &str) -> Result<EncodedBatch> {
        let enc = self
            .inner
            .encode_fast(
                tokenizers::EncodeInput::Dual(query.into(), passage.into()),
                true,
            )
            .map_err(BackendError::tokenization)?;

        let seq_len = enc.get_ids().len();
        Ok(EncodedBatch {
            input_ids: vec![enc.get_ids().iter().map(|&u| u as i64).collect()],
            attention_mask: vec![enc.get_attention_mask().iter().map(|&u| u as i64).collect()],
            token_type_ids: vec![enc.get_type_ids().iter().map(|&u| u as i64).collect()],
            batch_size: 1,
            seq_len,
        })
    }
}

#[cfg(test)]
mod tests {
    // Tokenizer loading tests need a real tokenizer.json on disk;
    // those are integration tests in P2.6 and they're network-gated
    // (requires HF Hub fetch of the bge-small tokenizer). Pure-logic
    // tests live here; the rest are gated by `--ignored`.

    #[test]
    fn shape_invariant_after_padding_is_obvious() {
        // Compile-time placeholder; the real shape invariants are
        // exercised in core/backend.rs integration tests once we have
        // a Session loaded.
    }
}
