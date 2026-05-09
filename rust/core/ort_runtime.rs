//! ort-backed implementation of ``Backend``.
//!
//! Pipeline per ``embed()`` call:
//!
//!   tokenize batch → stack to int64 tensors → Session::run →
//!   slice output → pool → optionally L2-normalize → return Array2.
//!
//! KNT-602 Send+Sync audit: ``ort::session::Session`` is ``Send +
//! Sync`` per ort 2.0.0-rc.10. ``TokenizerWrapper`` wraps a
//! ``tokenizers::Tokenizer`` which is also ``Send + Sync``. The whole
//! ``OrtBackend`` is therefore ``Send + Sync`` and safe to hold in
//! ``Arc`` across a ``py.allow_threads`` boundary.

use crate::core::backend::Backend;
use crate::core::device::Device;
use crate::core::error::{BackendError, Result};
use crate::core::model_loader::{resolve_paths, ModelPaths};
use crate::core::model_registry::RegisteredModel;
use crate::core::pooling::{l2_normalize, pool, Pooling};
use crate::core::tokenize::TokenizerWrapper;
use ndarray::{s, Array2, Array3, ArrayView3, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use std::path::Path;
use std::sync::Mutex;

/// ort-backed embedding inference.
pub struct OrtBackend {
    /// The loaded ort Session. Wrapped in Mutex because ort's
    /// Session::run takes &mut self in 2.0-rc.10. The mutex serializes
    /// concurrent embed() calls — fine for our use case (Python side
    /// is one-call-at-a-time per EmbeddingModel; concurrent users hold
    /// separate EmbeddingModel instances or wait at this Mutex).
    session: Mutex<Session>,
    tokenizer: TokenizerWrapper,
    pooling: Pooling,
    normalize: bool,
    dim: usize,
    model_id: String,
    device_str: String,
}

impl OrtBackend {
    /// Load a model from a registered model + device. Downloads (or
    /// looks up cached) the ONNX + tokenizer, builds an ort Session,
    /// configures EPs.
    pub fn load(
        model: &RegisteredModel,
        device: &Device,
        cache_dir: Option<&Path>,
    ) -> Result<Self> {
        // 1. Resolve model files (network or cache).
        let ModelPaths { onnx, tokenizer } = resolve_paths(model, cache_dir)?;

        // 2. Build the ort Session. SessionBuilder methods take `self`
        //    by value (returning Result<Self>) for chaining; commit_from_file
        //    takes &mut self and returns Session.
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

        // 3. Configure execution providers per device.
        let mut builder = configure_eps(builder, device, model)?;

        // 4. Commit the session from the ONNX file.
        let session = builder.commit_from_file(&onnx).map_err(|e| {
            BackendError::model_load(
                model.model_id,
                model.revision,
                format!("commit_from_file({}): {e}", onnx.display()),
            )
        })?;

        // 5. Load the tokenizer.
        let tok = TokenizerWrapper::from_file(&tokenizer, model.max_seq_len)?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: tok,
            pooling: model.pooling,
            normalize: model.normalize,
            dim: model.dim,
            model_id: model.model_id.to_string(),
            device_str: device.as_str(),
        })
    }
}

/// Configure ort EPs for a session builder. CPU is the default; CUDA
/// and OpenVINO require feature flags.
fn configure_eps(
    builder: ort::session::builder::SessionBuilder,
    device: &Device,
    model: &RegisteredModel,
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
            // The cuda EP is only available when --features gpu is set;
            // this branch is unreachable at runtime when feature is off.
            #[cfg(feature = "gpu")]
            {
                use ort::execution_providers::CUDAExecutionProvider;
                let _ = (_idx, model); // silence unused under cfg
                builder
                    .with_execution_providers([CUDAExecutionProvider::default().build()])
                    .map_err(|e| BackendError::Inference(format!("CUDA EP: {e}")))
            }
            #[cfg(not(feature = "gpu"))]
            {
                let _ = model;
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

impl Backend for OrtBackend {
    fn embed(&self, texts: &[&str], batch_size: usize) -> Result<Array2<f32>> {
        if texts.is_empty() {
            return Ok(Array2::zeros((0, self.dim)));
        }

        let n = texts.len();
        let mut output: Array2<f32> = Array2::zeros((n, self.dim));

        // Process in chunks of batch_size to bound peak memory.
        for (chunk_idx, chunk) in texts.chunks(batch_size.max(1)).enumerate() {
            let encoded = self.tokenizer.encode_batch(chunk)?;
            let bs = encoded.batch_size;
            let seq = encoded.seq_len;

            // Flatten the (bs, seq) Vec<Vec<i64>> into contiguous int64
            // buffers — required for ort's TensorRef::from_array_view.
            let mut flat_input_ids: Vec<i64> = Vec::with_capacity(bs * seq);
            let mut flat_attention_mask: Vec<i64> = Vec::with_capacity(bs * seq);
            let mut flat_token_type_ids: Vec<i64> = Vec::with_capacity(bs * seq);
            for row in &encoded.input_ids {
                flat_input_ids.extend_from_slice(row);
            }
            for row in &encoded.attention_mask {
                flat_attention_mask.extend_from_slice(row);
            }
            for row in &encoded.token_type_ids {
                flat_token_type_ids.extend_from_slice(row);
            }

            let shape = [bs as i64, seq as i64];
            let input_ids_tensor =
                TensorRef::from_array_view((shape.as_slice(), flat_input_ids.as_slice()))
                    .map_err(|e| BackendError::inference(format!("input_ids tensor: {e}")))?;
            let attention_mask_tensor =
                TensorRef::from_array_view((shape.as_slice(), flat_attention_mask.as_slice()))
                    .map_err(|e| BackendError::inference(format!("attention_mask tensor: {e}")))?;
            let token_type_ids_tensor =
                TensorRef::from_array_view((shape.as_slice(), flat_token_type_ids.as_slice()))
                    .map_err(|e| BackendError::inference(format!("token_type_ids tensor: {e}")))?;

            // BERT-family ONNX inputs are named exactly these.
            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ];

            // Run inference. We must extract the output tensor data into
            // an owned Vec INSIDE the lock scope, because SessionOutputs
            // borrows from the MutexGuard. After this block, `last_hidden_data`
            // is owned and the guard is dropped, freeing the session for
            // other threads.
            let (last_hidden_data, last_hidden_shape) = {
                let mut session = self
                    .session
                    .lock()
                    .map_err(|e| BackendError::inference(format!("session mutex poisoned: {e}")))?;
                let outputs = session
                    .run(inputs)
                    .map_err(|e| BackendError::inference(format!("Session::run: {e}")))?;

                // BERT-family ONNX exports name the embedding output
                // "last_hidden_state" (verified for BAAI/bge-small-en-v1.5
                // and sentence-transformers/all-MiniLM-L6-v2).
                let last_hidden = outputs.get("last_hidden_state").ok_or_else(|| {
                    BackendError::inference(
                        "ONNX model has no 'last_hidden_state' output — \
                         this BERT-family contract is required by the v0.2.0 \
                         embedding path. Re-export with the standard \
                         sentence-transformers ONNX shape."
                            .to_string(),
                    )
                })?;

                let (shape, slice) = last_hidden.try_extract_tensor::<f32>().map_err(|e| {
                    BackendError::inference(format!("extract last_hidden_state: {e}"))
                })?;

                // Copy to owned data so we can drop the lock.
                (slice.to_vec(), shape.to_vec())
            };

            if last_hidden_shape.len() != 3 {
                return Err(BackendError::inference(format!(
                    "expected 3D last_hidden_state, got shape {:?}",
                    last_hidden_shape
                )));
            }

            let hidden = last_hidden_shape[2] as usize;
            if hidden != self.dim {
                return Err(BackendError::inference(format!(
                    "model output dim={hidden} but registry expected {}",
                    self.dim
                )));
            }

            // Reshape owned data into a 3D ndarray view.
            let hidden_view: ArrayView3<'_, f32> =
                ArrayView3::from_shape((bs, seq, hidden), &last_hidden_data).map_err(|e| {
                    BackendError::inference(format!("reshape last_hidden_state: {e}"))
                })?;

            // Reconstitute the attention_mask as a 2D ndarray for pooling.
            let mask_arr: Array2<i64> = Array2::from_shape_vec((bs, seq), flat_attention_mask)
                .map_err(|e| BackendError::inference(format!("reshape attention_mask: {e}")))?;

            let mut pooled = pool(hidden_view, mask_arr.view(), self.pooling)?;
            if self.normalize {
                l2_normalize(&mut pooled);
            }

            // Copy chunk's pooled rows into the right slice of the output.
            let row_start = chunk_idx * batch_size;
            let row_end = row_start + bs;
            output.slice_mut(s![row_start..row_end, ..]).assign(&pooled);
        }

        Ok(output)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn device(&self) -> &str {
        &self.device_str
    }

    fn max_seq_len(&self) -> usize {
        self.tokenizer.max_seq_len
    }

    fn count_tokens(&self, texts: &[&str]) -> Result<Vec<usize>> {
        // The encode_batch path applies the model's standard padding +
        // truncation; counts include [CLS] and [SEP]. Pre-truncation
        // counts (i.e. how many tokens the input WOULD have been
        // without the cap) are not exposed by the tokenizers crate's
        // fast path; if a chunker wants pre-truncation counts it
        // should bypass max_seq_len at the tokenizer level. For our
        // chunker use case, post-truncation count is what we want
        // (it tells you "this chunk fits in N tokens").
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encoded = self.tokenizer.encode_batch(texts)?;
        // Count NON-PAD tokens per row by summing attention_mask.
        let counts: Vec<usize> = encoded
            .attention_mask
            .iter()
            .map(|mask| mask.iter().filter(|&&v| v != 0).count())
            .collect();
        Ok(counts)
    }
}

// Suppress unused-warning for Array3 (only used in the ArrayView3 path).
const _: fn() = || {
    let _: Option<Array3<f32>> = None;
    let _: Axis = Axis(0);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::model_registry::lookup_embedding;

    /// Integration test — requires either network access OR a populated
    /// HF cache for BAAI/bge-small-en-v1.5 at the pinned revision.
    /// Run with: `cargo test --release -- --ignored bge_small_smoke`
    #[test]
    #[ignore = "requires network or cached BAAI/bge-small-en-v1.5 weights"]
    fn bge_small_smoke() {
        let model = lookup_embedding("BAAI/bge-small-en-v1.5").expect("registered");
        let backend = OrtBackend::load(model, &Device::Cpu, None).expect("load");
        let texts = ["hello world", "the quick brown fox"];
        let out = backend
            .embed(&texts.iter().map(|s| *s).collect::<Vec<_>>(), 8)
            .expect("embed");
        assert_eq!(out.dim(), (2, 384));
        // Unit norm check.
        for row in out.axis_iter(Axis(0)) {
            let norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "row norm = {norm}");
        }
    }
}
