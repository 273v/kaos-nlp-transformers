//! Error tree for the Rust core. Mirrors the Python ``errors.py``
//! hierarchy so the PyO3 binding layer can map Rust → Python
//! exceptions one-to-one (see ``rust/bindings/util.rs``).
//!
//! The variants below are deliberately narrow. Keep this discipline
//! as we add models — a new variant per actually-distinguishable
//! failure mode, not per call site.

use std::path::PathBuf;
use thiserror::Error;

/// All Rust-core errors. Mirrors the public Python error hierarchy:
///
/// * ``BackendError::ModelNotRegistered``  ↔ ``ModelNotRegisteredError``
/// * ``BackendError::ModelLoad``           ↔ ``ModelLoadError``
/// * ``BackendError::BackendNotInstalled`` ↔ ``BackendNotInstalledError``
/// * ``BackendError::Device``              ↔ ``DeviceNotReachableError``
/// * ``BackendError::Tokenization``        ↔ ``EmbeddingError`` (subcase)
/// * ``BackendError::Inference``           ↔ ``EmbeddingError`` (subcase)
/// * ``BackendError::Io``                  ↔ ``ModelLoadError`` (subcase)
#[derive(Debug, Error)]
pub enum BackendError {
    /// The model id is not in REGISTRY and unregistered models are forbidden.
    #[error("model {0:?} is not in the registry")]
    ModelNotRegistered(String),

    /// The backend (ort, tokenizers, hf-hub) failed to load the model.
    /// Wraps the underlying error string verbatim — callers should
    /// preserve the original message in their three-part error text.
    #[error("failed to load model {model_id:?} @ {revision:?}: {source_msg}")]
    ModelLoad {
        /// HF Hub model id.
        model_id: String,
        /// Pinned revision SHA.
        revision: String,
        /// Underlying error rendering (string, not boxed Error, so the
        /// type stays Send + Sync).
        source_msg: String,
    },

    /// A required runtime feature isn't compiled in (e.g. asking for
    /// ``cuda`` without ``--features gpu``).
    #[error("backend not installed: {0}")]
    BackendNotInstalled(String),

    /// A requested device is not reachable on this host.
    #[error("device {requested:?} is not reachable: {reason}")]
    Device {
        /// What the caller asked for (``"cuda"``, ``"cuda:1"``, …).
        requested: String,
        /// Human-readable reason from the device probe.
        reason: String,
    },

    /// Tokenization failed.
    #[error("tokenization failed: {0}")]
    Tokenization(String),

    /// ort session.run() or tensor-shape error.
    #[error("inference failed: {0}")]
    Inference(String),

    /// Filesystem I/O around model loading or vendored-path resolution.
    #[error("I/O error at {path:?}: {source_msg}")]
    Io {
        /// Path being accessed.
        path: PathBuf,
        /// Underlying io::Error rendering.
        source_msg: String,
    },
}

impl BackendError {
    /// Construct a ``ModelLoad`` error from any error type whose
    /// ``Display`` impl produces a useful message.
    pub fn model_load(
        model_id: impl Into<String>,
        revision: impl Into<String>,
        src: impl std::fmt::Display,
    ) -> Self {
        Self::ModelLoad {
            model_id: model_id.into(),
            revision: revision.into(),
            source_msg: src.to_string(),
        }
    }

    /// Construct a ``Tokenization`` error from any displayable source.
    pub fn tokenization(src: impl std::fmt::Display) -> Self {
        Self::Tokenization(src.to_string())
    }

    /// Construct an ``Inference`` error from any displayable source.
    pub fn inference(src: impl std::fmt::Display) -> Self {
        Self::Inference(src.to_string())
    }

    /// Construct an ``Io`` error from a path + io::Error.
    pub fn io(path: impl Into<PathBuf>, src: impl std::fmt::Display) -> Self {
        Self::Io {
            path: path.into(),
            source_msg: src.to_string(),
        }
    }
}

/// Convenience alias used throughout the core.
pub type Result<T> = std::result::Result<T, BackendError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_model_not_registered() {
        let e = BackendError::ModelNotRegistered("foo/bar".into());
        assert_eq!(e.to_string(), "model \"foo/bar\" is not in the registry");
    }

    #[test]
    fn display_model_load() {
        let e = BackendError::model_load("foo/bar", "abc123", "404 Not Found");
        let s = e.to_string();
        assert!(s.contains("foo/bar"));
        assert!(s.contains("abc123"));
        assert!(s.contains("404 Not Found"));
    }

    #[test]
    fn display_io() {
        let e = BackendError::io("/no/such/path", "no such file");
        let s = e.to_string();
        assert!(s.contains("/no/such/path"));
        assert!(s.contains("no such file"));
    }

    #[test]
    fn send_sync() {
        // Compile-time assertion: BackendError is Send + Sync. The
        // PyO3 binding layer holds these in Arc<dyn Backend + Send + Sync>.
        fn check<T: Send + Sync>() {}
        check::<BackendError>();
    }
}
