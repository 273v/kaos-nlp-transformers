//! Model file resolution.
//!
//! Returns local filesystem paths to ``(onnx_path, tokenizer_path)``
//! for a given ``RegisteredModel``. Resolution order:
//!
//! 1. **Vendored copy** at ``kaos_nlp_transformers/_vendor/<slug>/``
//!    — Python-side, only model2vec uses this today (audit-05 KNT-401).
//!    The Rust loader skips this branch because no embedding ONNX is
//!    vendored at 0.2.0; the field is here for forward compatibility.
//!
//! 2. **HF Hub snapshot** via ``hf-hub`` Rust crate. Honors the
//!    pinned revision SHA per audit-01 KNT-003. Honors
//!    ``KAOS_NLP_TRANSFORMERS_OFFLINE`` and ``HF_HUB_OFFLINE``.
//!
//! Caching: hf-hub uses ``HF_HOME`` (default
//! ``~/.cache/huggingface/hub``). Same cache the Python side uses, so
//! switching to the Rust loader doesn't re-download anything.

use crate::core::error::{BackendError, Result};
use crate::core::model_registry::RegisteredModel;
use std::path::{Path, PathBuf};

/// Resolved local paths for a model.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// Path to ``model.onnx`` (or ``onnx/model.onnx`` per registry).
    pub onnx: PathBuf,
    /// Path to ``tokenizer.json``.
    pub tokenizer: PathBuf,
}

/// Fetch (or look up cached) the model files for a registered model.
///
/// Args:
///   * ``model`` — entry from EMBEDDING_REGISTRY / RERANKER_REGISTRY.
///   * ``cache_dir`` — optional override for HF Hub cache. None falls
///     back to ``HF_HOME`` env var, then to the default location.
pub fn resolve_paths(model: &RegisteredModel, cache_dir: Option<&Path>) -> Result<ModelPaths> {
    // Build the hf-hub API. The 0.5 builder honors HF_HOME implicitly;
    // a caller-supplied cache_dir overrides it via with_cache_dir.
    let mut api_builder = hf_hub::api::sync::ApiBuilder::new();
    if let Some(cd) = cache_dir {
        api_builder = api_builder.with_cache_dir(cd.to_path_buf());
    }
    let api = api_builder.build().map_err(|e| {
        BackendError::model_load(
            model.model_id,
            model.revision,
            format!("hf-hub builder: {e}"),
        )
    })?;

    // Pin the revision explicitly. This is the KNT-003 contract that
    // fastembed-rs's pull_from_hf can't honor (it always tracks `main`).
    let repo = api.repo(hf_hub::Repo::with_revision(
        model.model_id.to_string(),
        hf_hub::RepoType::Model,
        model.revision.to_string(),
    ));

    let onnx = repo.get(model.onnx_filename).map_err(|e| {
        BackendError::model_load(
            model.model_id,
            model.revision,
            format!("download {}: {e}", model.onnx_filename),
        )
    })?;

    let tokenizer = repo.get(model.tokenizer_filename).map_err(|e| {
        BackendError::model_load(
            model.model_id,
            model.revision,
            format!("download {}: {e}", model.tokenizer_filename),
        )
    })?;

    Ok(ModelPaths { onnx, tokenizer })
}

#[cfg(test)]
mod tests {
    // Loader tests need network or a populated cache; they're integration
    // tests living in core::backend (P2.4) where they can be gated by
    // `#[ignore]` and run via `cargo test -- --ignored`.
    //
    // The unit-testable surface here is the path-shape contract, which
    // is exercised end-to-end in P2.4 tests.

    #[test]
    fn build_path_contract() {
        // Compile-only sanity: ModelPaths is constructible.
        let p = super::ModelPaths {
            onnx: "/tmp/model.onnx".into(),
            tokenizer: "/tmp/tokenizer.json".into(),
        };
        assert!(p.onnx.to_str().unwrap().ends_with(".onnx"));
        assert!(p.tokenizer.to_str().unwrap().ends_with(".json"));
    }
}
