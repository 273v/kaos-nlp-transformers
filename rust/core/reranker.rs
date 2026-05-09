//! Cross-encoder reranker — sigmoid-normalized [0, 1] relevance
//! scoring for (query, passage) pairs.
//!
//! Distinct from embedding inference: cross-encoders concatenate the
//! query and passage with [SEP] tokens, run BERT, and read a single
//! logit off a classifier head. The output is a relevance score, not
//! a vector. Audit-06 KNT-501 retired the sentence-transformers path
//! for this task; KNT-601 ports it from Python ``fastembed.TextCrossEncoder``
//! to a direct Rust+ort implementation here.
//!
//! Sigmoid normalization centralizes here so the Python side
//! (``reranker.py``) becomes a thin async-thread dispatch wrapper
//! with no math.

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

/// Cross-encoder reranker. Holds an ort Session loaded against a
/// classification-head BERT model.
pub trait CrossEncoder: Send + Sync {
    /// Score a batch of (query, passage) pairs. Returns sigmoid-
    /// normalized scores in [0, 1], one per pair.
    fn score_pairs(&self, pairs: &[(&str, &str)], batch_size: usize) -> Result<Vec<f32>>;

    /// HF Hub model id this reranker was loaded for.
    fn model_id(&self) -> &str;

    /// Device this reranker runs on.
    fn device(&self) -> &str;
}

/// ort-backed cross-encoder.
pub struct OrtCrossEncoder {
    session: Mutex<Session>,
    tokenizer: TokenizerWrapper,
    model_id: String,
    device_str: String,
    /// True iff the loaded ONNX accepts a ``token_type_ids`` input.
    /// Some BERT-family exports omit it (the model embeds zero
    /// segment ids internally); supplying it then triggers an
    /// "Invalid input name" error from ort. See test
    /// ``bge_reranker_smoke`` for the regression that motivated this.
    accepts_token_type_ids: bool,
}

impl OrtCrossEncoder {
    /// Load a cross-encoder model (e.g. BAAI/bge-reranker-base).
    pub fn load(
        model: &RegisteredModel,
        device: &Device,
        cache_dir: Option<&Path>,
    ) -> Result<Self> {
        let ModelPaths { onnx, tokenizer } = resolve_paths(model, cache_dir)?;

        // Build session — same shape as embedding path; cross-encoder
        // op coverage is identical (BERT + Linear classifier head).
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

        let mut builder = configure_eps(builder, device, model)?;

        let session = builder.commit_from_file(&onnx).map_err(|e| {
            BackendError::model_load(
                model.model_id,
                model.revision,
                format!("commit_from_file({}): {e}", onnx.display()),
            )
        })?;

        // Probe inputs once at load time so we know whether to pass
        // token_type_ids per request.
        let accepts_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        let tok = TokenizerWrapper::from_file(&tokenizer, model.max_seq_len)?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: tok,
            model_id: model.model_id.to_string(),
            device_str: device.as_str(),
            accepts_token_type_ids,
        })
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

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl CrossEncoder for OrtCrossEncoder {
    fn score_pairs(&self, pairs: &[(&str, &str)], batch_size: usize) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let n = pairs.len();
        let mut scores: Vec<f32> = Vec::with_capacity(n);

        for chunk in pairs.chunks(batch_size.max(1)) {
            // Tokenize each pair and stack to a fixed (bs, seq_len)
            // matrix. Pair tokenization gives non-zero token_type_ids
            // for the passage segment — different from embedding path
            // where everything is zero.
            let bs = chunk.len();
            let mut per_row_ids: Vec<Vec<i64>> = Vec::with_capacity(bs);
            let mut per_row_mask: Vec<Vec<i64>> = Vec::with_capacity(bs);
            let mut per_row_types: Vec<Vec<i64>> = Vec::with_capacity(bs);
            let mut max_seq = 0usize;
            for (q, p) in chunk {
                let enc = self.tokenizer.encode_pair(q, p)?;
                // encode_pair returns batch_size=1; take row 0.
                let ids = enc.input_ids.into_iter().next().unwrap_or_default();
                let mask = enc.attention_mask.into_iter().next().unwrap_or_default();
                let tids = enc.token_type_ids.into_iter().next().unwrap_or_default();
                max_seq = max_seq.max(ids.len());
                per_row_ids.push(ids);
                per_row_mask.push(mask);
                per_row_types.push(tids);
            }

            // Pad each row to max_seq with pad_id / mask=0 / token_type=0.
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

            // Flatten.
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
                    .map_err(|e| BackendError::inference(format!("rerank input_ids: {e}")))?;
            let attention_mask_tensor =
                TensorRef::from_array_view((shape.as_slice(), flat_mask.as_slice()))
                    .map_err(|e| BackendError::inference(format!("rerank attention_mask: {e}")))?;
            let token_type_ids_tensor =
                TensorRef::from_array_view((shape.as_slice(), flat_types.as_slice()))
                    .map_err(|e| BackendError::inference(format!("rerank token_type_ids: {e}")))?;

            // Build the input map. Some BERT-family ONNX cross-encoder
            // exports omit token_type_ids; passing it would trigger
            // "Invalid input name" from ort. The accepts_token_type_ids
            // flag was probed at load time.
            let inputs = if self.accepts_token_type_ids {
                ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                    "token_type_ids" => token_type_ids_tensor,
                ]
            } else {
                let _ = token_type_ids_tensor; // computed but unused for this model
                ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                ]
            };

            // BAAI/bge-reranker-base ONNX export names its single-logit
            // output "logits" (shape (bs, 1)).
            let (logits_data, logits_shape) = {
                let mut session = self
                    .session
                    .lock()
                    .map_err(|e| BackendError::inference(format!("session mutex: {e}")))?;
                let outputs = session
                    .run(inputs)
                    .map_err(|e| BackendError::inference(format!("rerank Session::run: {e}")))?;

                let logits = outputs.get("logits").ok_or_else(|| {
                    BackendError::inference(
                        "ONNX cross-encoder has no 'logits' output — \
                         BAAI/bge-reranker-base ONNX must expose a 'logits' \
                         tensor of shape (batch, 1)."
                            .to_string(),
                    )
                })?;

                let (shape, slice) = logits
                    .try_extract_tensor::<f32>()
                    .map_err(|e| BackendError::inference(format!("extract logits: {e}")))?;

                (slice.to_vec(), shape.to_vec())
            };

            // Cross-encoder logits are (bs, 1) for BAAI/bge-reranker-base.
            // Some exports give (bs,) instead — handle both.
            let arr = match logits_shape.len() {
                1 => Array2::from_shape_vec((bs, 1), logits_data)
                    .map_err(|e| BackendError::inference(format!("logits 1D reshape: {e}")))?,
                2 => Array2::from_shape_vec(
                    (logits_shape[0] as usize, logits_shape[1] as usize),
                    logits_data,
                )
                .map_err(|e| BackendError::inference(format!("logits 2D reshape: {e}")))?,
                _ => {
                    return Err(BackendError::inference(format!(
                        "unexpected logits shape {:?}",
                        logits_shape
                    )))
                }
            };

            // Take the first column (single relevance logit) and apply sigmoid.
            for row in arr.outer_iter() {
                scores.push(sigmoid(row[0]));
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
    use crate::core::model_registry::lookup_reranker;

    #[test]
    fn sigmoid_endpoints() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(20.0) > 0.99);
        assert!(sigmoid(-20.0) < 0.01);
    }

    /// Live smoke for BAAI/bge-reranker-base. Requires network or a
    /// populated cache. Run with: `cargo test --release -- --ignored bge_reranker_smoke`
    #[test]
    #[ignore = "requires network or cached BAAI/bge-reranker-base weights"]
    fn bge_reranker_smoke() {
        let model = lookup_reranker("BAAI/bge-reranker-base").expect("registered");
        let backend = OrtCrossEncoder::load(model, &Device::Cpu, None).expect("load");

        // Semantic test: query about birds should score "robins are birds"
        // higher than "the moon is far".
        let query = "What is a robin?";
        let pairs: &[(&str, &str)] = &[
            (query, "Robins are small songbirds."),
            (query, "The moon is approximately 384,400 km from Earth."),
        ];

        let scores = backend.score_pairs(pairs, 2).expect("score");
        assert_eq!(scores.len(), 2);
        for &s in &scores {
            assert!((0.0..=1.0).contains(&s), "score {s} out of [0,1]");
        }
        assert!(
            scores[0] > scores[1],
            "expected birds-passage to outscore moon-passage: {:?}",
            scores
        );
    }
}
