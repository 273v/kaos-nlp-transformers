//! NLI (natural language inference) cross-encoder — softmax-normalized
//! three-class probabilities over (entailment, neutral, contradiction)
//! for ``(premise, hypothesis)`` pairs.
//!
//! Mechanically identical to the cross-encoder reranker
//! (``core::reranker``): tokenizer ``encode_pair`` produces the same
//! ``(input_ids, attention_mask, token_type_ids)`` shapes, ort's
//! ``Session::run`` returns the same ``"logits"`` output name. The
//! only differences are:
//!
//! 1. The logits tensor has shape ``(batch, 3)`` instead of
//!    ``(batch, 1)`` — three-class head instead of single-relevance.
//! 2. We softmax along axis 1 instead of sigmoid on column 0.
//! 3. We **re-order** the three-class output from the model's
//!    ``id2label`` permutation into the canonical
//!    ``(entailment, neutral, contradiction)`` tuple expected by the
//!    ``NLIScorer`` protocol on the kaos-llm-core side. The default
//!    registered model ``Xenova/nli-deberta-v3-base`` declares
//!    ``id2label = {0: contradiction, 1: entailment, 2: neutral}``;
//!    we hardcode that permutation here.
//!
//! When a second NLI checkpoint lands, the canonical fix is to read
//! the model's ``config.json`` at load time and build the permutation
//! dynamically — see the inline ``// TODO`` below. For 0.2.0a7 we
//! hardcode the one model we serve.

use crate::core::device::Device;
use crate::core::error::{BackendError, Result};
use crate::core::model_loader::{resolve_paths, ModelPaths};
use crate::core::model_registry::RegisteredModel;
use crate::core::tokenize::TokenizerWrapper;
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use std::path::Path;
use std::sync::Mutex;

/// NLI three-class probability triple. Canonical order:
/// ``(entailment, neutral, contradiction)`` regardless of how the
/// underlying ONNX checkpoint permuted its head. Matches the
/// ``NLIScore`` Protocol fields in
/// ``kaos_llm_core.programs.classify.nli``.
pub type NliProbs = [f32; 3];

/// NLI cross-encoder backend trait.
pub trait NliClassifier: Send + Sync {
    /// Score a batch of (premise, hypothesis) pairs. Returns
    /// softmax-normalized three-class probabilities, in the canonical
    /// (entailment, neutral, contradiction) order, one triple per pair.
    fn score_pairs(&self, pairs: &[(&str, &str)], batch_size: usize) -> Result<Vec<NliProbs>>;

    /// HF Hub model id this classifier was loaded for.
    fn model_id(&self) -> &str;

    /// Device this classifier runs on.
    fn device(&self) -> &str;
}

/// ort-backed NLI cross-encoder.
pub struct OrtNliClassifier {
    session: Mutex<Session>,
    tokenizer: TokenizerWrapper,
    model_id: String,
    device_str: String,
    /// True iff the loaded ONNX accepts a ``token_type_ids`` input.
    /// Same probe pattern as ``core::reranker::OrtCrossEncoder``.
    accepts_token_type_ids: bool,
    /// Permutation array. ``[i]`` is the column index in the raw ONNX
    /// logits that holds the i-th canonical class
    /// (0=entailment, 1=neutral, 2=contradiction).
    ///
    /// For ``Xenova/nli-deberta-v3-base`` (id2label =
    /// {0: contradiction, 1: entailment, 2: neutral}):
    /// permutation = [1, 2, 0].
    canonical_perm: [usize; 3],
}

impl OrtNliClassifier {
    /// Load an NLI cross-encoder model (e.g. Xenova/nli-deberta-v3-base).
    pub fn load(
        model: &RegisteredModel,
        device: &Device,
        cache_dir: Option<&Path>,
    ) -> Result<Self> {
        let ModelPaths { onnx, tokenizer } = resolve_paths(model, cache_dir)?;

        let builder = Session::builder()
            .map_err(|e| {
                BackendError::model_load(
                    model.model_id,
                    model.revision,
                    format!("Session::builder: {e}"),
                )
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                BackendError::model_load(
                    model.model_id,
                    model.revision,
                    format!("optimization level: {e}"),
                )
            })?;

        // KNT-NLI-002 (2026-05-16): saturate the intra-op thread
        // pool to ``available_parallelism()``. ort's default sized
        // to ~25% of cores on a 20-core host; explicit setting
        // roughly halves per-call latency. See rust/core/ner.rs for
        // the diagnostic numbers behind this change.
        let builder = configure_intra_threads(builder, model)?;

        let mut builder = configure_eps(builder, device, model)?;

        let session = builder.commit_from_file(&onnx).map_err(|e| {
            BackendError::model_load(
                model.model_id,
                model.revision,
                format!("commit_from_file({}): {e}", onnx.display()),
            )
        })?;

        let accepts_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        let tok = TokenizerWrapper::from_file(&tokenizer, model.max_seq_len)?;

        // TODO: when a second NLI checkpoint lands, read config.json
        // ``id2label`` at load time and derive ``canonical_perm``
        // dynamically. For 0.2.0a7 we serve a single model whose
        // permutation is fixed.
        let canonical_perm = match model.model_id {
            "Xenova/nli-deberta-v3-base" => {
                // id2label = {0: contradiction, 1: entailment, 2: neutral}
                // canonical = (entailment, neutral, contradiction)
                //           = (col 1,    col 2,   col 0)
                [1, 2, 0]
            }
            other => {
                return Err(BackendError::model_load(
                    other,
                    model.revision,
                    "no hardcoded canonical permutation for this NLI model; \
                     add an entry to OrtNliClassifier::load() and re-confirm \
                     id2label from the model's config.json"
                        .to_string(),
                ));
            }
        };

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: tok,
            model_id: model.model_id.to_string(),
            device_str: device.as_str(),
            accepts_token_type_ids,
            canonical_perm,
        })
    }
}

/// See ``rust/core/ner.rs::configure_intra_threads`` for rationale —
/// duplicated locally so each backend file stays self-contained.
fn configure_intra_threads(
    builder: ort::session::builder::SessionBuilder,
    model: &RegisteredModel,
) -> Result<ort::session::builder::SessionBuilder> {
    let override_n: Option<usize> = std::env::var("KAOS_NLP_TRANSFORMERS_INTRA_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0);
    match override_n {
        None => Ok(builder),
        Some(n) => builder.with_intra_threads(n).map_err(|e| {
            BackendError::model_load(
                model.model_id,
                model.revision,
                format!("with_intra_threads({n}): {e}"),
            )
        }),
    }
}

fn configure_eps(
    builder: ort::session::builder::SessionBuilder,
    device: &Device,
    _model: &RegisteredModel,
) -> Result<ort::session::builder::SessionBuilder> {
    match device {
        Device::Cpu => Ok(builder),
        Device::Cuda(_idx) => {
            if !device.is_compiled_in() {
                return Err(BackendError::BackendNotInstalled(
                    device
                        .install_extra_message()
                        .unwrap_or_else(|| "GPU feature not enabled".to_string()),
                ));
            }
            #[cfg(feature = "gpu")]
            {
                use ort::execution_providers::CUDAExecutionProvider;
                builder
                    .with_execution_providers([CUDAExecutionProvider::default().build()])
                    .map_err(|e| BackendError::Inference(format!("CUDA EP: {e}")))
            }
            #[cfg(not(feature = "gpu"))]
            {
                Ok(builder)
            }
        }
        Device::OpenVino => {
            if !device.is_compiled_in() {
                return Err(BackendError::BackendNotInstalled(
                    device
                        .install_extra_message()
                        .unwrap_or_else(|| "OpenVINO feature not enabled".to_string()),
                ));
            }
            #[cfg(feature = "openvino")]
            {
                use ort::execution_providers::OpenVINOExecutionProvider;
                builder
                    .with_execution_providers([OpenVINOExecutionProvider::default().build()])
                    .map_err(|e| BackendError::Inference(format!("OpenVINO EP: {e}")))
            }
            #[cfg(not(feature = "openvino"))]
            {
                Ok(builder)
            }
        }
    }
}

/// Stable softmax over a single row.
#[inline]
fn softmax_row(row: &[f32]) -> [f32; 3] {
    debug_assert_eq!(row.len(), 3);
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e0 = (row[0] - max).exp();
    let e1 = (row[1] - max).exp();
    let e2 = (row[2] - max).exp();
    let sum = e0 + e1 + e2;
    [e0 / sum, e1 / sum, e2 / sum]
}

impl NliClassifier for OrtNliClassifier {
    fn score_pairs(&self, pairs: &[(&str, &str)], batch_size: usize) -> Result<Vec<NliProbs>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let n = pairs.len();
        let mut scores: Vec<NliProbs> = Vec::with_capacity(n);

        for chunk in pairs.chunks(batch_size.max(1)) {
            let bs = chunk.len();
            let mut per_row_ids: Vec<Vec<i64>> = Vec::with_capacity(bs);
            let mut per_row_mask: Vec<Vec<i64>> = Vec::with_capacity(bs);
            let mut per_row_types: Vec<Vec<i64>> = Vec::with_capacity(bs);
            let mut max_seq = 0usize;
            for (premise, hypothesis) in chunk {
                let enc = self.tokenizer.encode_pair(premise, hypothesis)?;
                let ids = enc.input_ids.into_iter().next().unwrap_or_default();
                let mask = enc.attention_mask.into_iter().next().unwrap_or_default();
                let tids = enc.token_type_ids.into_iter().next().unwrap_or_default();
                max_seq = max_seq.max(ids.len());
                per_row_ids.push(ids);
                per_row_mask.push(mask);
                per_row_types.push(tids);
            }

            let pad_id = self.tokenizer.pad_id as i64;
            for row in per_row_ids.iter_mut() {
                row.resize(max_seq, pad_id);
            }
            for row in per_row_mask.iter_mut() {
                row.resize(max_seq, 0);
            }
            for row in per_row_types.iter_mut() {
                row.resize(max_seq, 0);
            }

            let mut flat_ids: Vec<i64> = Vec::with_capacity(bs * max_seq);
            let mut flat_mask: Vec<i64> = Vec::with_capacity(bs * max_seq);
            let mut flat_types: Vec<i64> = Vec::with_capacity(bs * max_seq);
            for row in &per_row_ids {
                flat_ids.extend_from_slice(row);
            }
            for row in &per_row_mask {
                flat_mask.extend_from_slice(row);
            }
            for row in &per_row_types {
                flat_types.extend_from_slice(row);
            }

            let shape = [bs as i64, max_seq as i64];
            let input_ids_tensor =
                TensorRef::from_array_view((shape.as_slice(), flat_ids.as_slice()))
                    .map_err(|e| BackendError::inference(format!("nli input_ids: {e}")))?;
            let attention_mask_tensor =
                TensorRef::from_array_view((shape.as_slice(), flat_mask.as_slice()))
                    .map_err(|e| BackendError::inference(format!("nli attention_mask: {e}")))?;
            let token_type_ids_tensor =
                TensorRef::from_array_view((shape.as_slice(), flat_types.as_slice()))
                    .map_err(|e| BackendError::inference(format!("nli token_type_ids: {e}")))?;

            let inputs = if self.accepts_token_type_ids {
                ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                    "token_type_ids" => token_type_ids_tensor,
                ]
            } else {
                let _ = token_type_ids_tensor;
                ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                ]
            };

            let (logits_data, logits_shape) = {
                let mut session = self
                    .session
                    .lock()
                    .map_err(|e| BackendError::inference(format!("session mutex: {e}")))?;
                let outputs = session
                    .run(inputs)
                    .map_err(|e| BackendError::inference(format!("nli Session::run: {e}")))?;

                let logits = outputs.get("logits").ok_or_else(|| {
                    BackendError::inference(
                        "ONNX NLI classifier has no 'logits' output — \
                         expected a 'logits' tensor of shape (batch, 3)."
                            .to_string(),
                    )
                })?;

                let (shape, slice) = logits
                    .try_extract_tensor::<f32>()
                    .map_err(|e| BackendError::inference(format!("extract logits: {e}")))?;

                (slice.to_vec(), shape.to_vec())
            };

            if logits_shape.len() != 2 || logits_shape[1] != 3 {
                return Err(BackendError::inference(format!(
                    "expected NLI logits shape (batch, 3), got {:?}",
                    logits_shape
                )));
            }
            let arr = Array2::from_shape_vec(
                (logits_shape[0] as usize, logits_shape[1] as usize),
                logits_data,
            )
            .map_err(|e| BackendError::inference(format!("logits reshape: {e}")))?;

            let perm = self.canonical_perm;
            for row in arr.outer_iter() {
                let raw = softmax_row(row.as_slice().unwrap_or(&[0.0, 0.0, 0.0]));
                scores.push([raw[perm[0]], raw[perm[1]], raw[perm[2]]]);
            }
        }

        Ok(scores)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn device(&self) -> &str {
        &self.device_str
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::model_registry::lookup_nli;

    #[test]
    fn softmax_row_sums_to_one() {
        let s = softmax_row(&[1.0, 2.0, 3.0]);
        let total: f32 = s.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
        // Monotone — bigger logit => bigger prob.
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn softmax_row_endpoints() {
        // Heavy peak on column 1: ~1.0 prob there, near-zero elsewhere.
        let s = softmax_row(&[-50.0, 0.0, -50.0]);
        assert!(s[1] > 0.999);
        assert!(s[0] < 1e-3);
        assert!(s[2] < 1e-3);
    }

    /// Live smoke for Xenova/nli-deberta-v3-base. Requires network or a
    /// populated cache. Run with:
    ///   cargo test --release -- --ignored nli_deberta_smoke
    #[test]
    #[ignore = "requires network or cached Xenova/nli-deberta-v3-base weights"]
    fn nli_deberta_smoke() {
        let model = lookup_nli("Xenova/nli-deberta-v3-base").expect("registered");
        let backend = OrtNliClassifier::load(model, &Device::Cpu, None).expect("load");

        let premise = "A man inspects the uniform of a figure in some East Asian country.";
        let pairs: &[(&str, &str)] = &[
            // Classic SNLI/MNLI evaluation triples.
            (premise, "The man is sleeping."), // contradiction
            (premise, "A man is checking a uniform."), // entailment-ish
            (premise, "The man is outdoors."), // neutral-ish
        ];

        let scores = backend.score_pairs(pairs, 3).expect("score");
        assert_eq!(scores.len(), 3);
        for s in &scores {
            let total: f32 = s.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-3,
                "probs should sum to ~1, got {:?}",
                s
            );
            for &v in s.iter() {
                assert!((0.0..=1.0).contains(&v), "prob {v} out of [0,1]");
            }
        }
        // Pair 0 should be heavy on contradiction (canonical index 2).
        assert!(
            scores[0][2] > scores[0][0],
            "expected pair 0 to favour contradiction over entailment: {:?}",
            scores[0]
        );
    }
}
