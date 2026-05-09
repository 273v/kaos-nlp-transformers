//! Shared PyO3 utilities — error mapping from ``BackendError`` to the
//! Python error hierarchy in ``kaos_nlp_transformers.errors``.
//!
//! Mapping rationale (mirrors the audit-01 KNT-001 / KNT-601
//! contract that public errors carry actionable triplet messages):
//!
//! * ``BackendError::ModelNotRegistered`` → ``ModelNotRegisteredError``
//! * ``BackendError::ModelLoad``           → ``ModelLoadError``
//! * ``BackendError::BackendNotInstalled`` → ``BackendNotInstalledError``
//! * ``BackendError::Device``              → ``ModelLoadError`` *(the
//!   typed ``DeviceNotReachableError`` lives Python-side and carries
//!   a ``LatentDevice`` payload — the backend layer reports device
//!   resolution failures as load failures, and the Python wrapper
//!   re-raises them with the rich payload when needed.)*
//! * ``BackendError::Tokenization``        → ``EmbeddingError``
//! * ``BackendError::Inference``           → ``EmbeddingError``
//! * ``BackendError::Io``                  → ``ModelLoadError``
//!
//! The Python ``KaosNLPTransformersError`` family exposes its
//! ``KaosCoreError(**details)`` shape; the ``message`` is preserved
//! verbatim so the three-part (what / fix / alternative) text the
//! Rust side composes survives the round-trip.

use crate::core::error::BackendError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Convert a ``BackendError`` to a ``PyErr`` of the appropriate
/// subclass under ``kaos_nlp_transformers.errors``.
///
/// Falls back to ``PyRuntimeError`` if the Python errors module is
/// unavailable for some reason — that's a defensive path; in normal
/// operation the package always imports cleanly.
pub(crate) fn map_backend_error(py: Python<'_>, err: BackendError) -> PyErr {
    let class_name = match &err {
        BackendError::ModelNotRegistered(_) => "ModelNotRegisteredError",
        BackendError::ModelLoad { .. } => "ModelLoadError",
        BackendError::BackendNotInstalled(_) => "BackendNotInstalledError",
        BackendError::Device { .. } => "ModelLoadError",
        BackendError::Tokenization(_) | BackendError::Inference(_) => "EmbeddingError",
        BackendError::Io { .. } => "ModelLoadError",
    };
    let message = err.to_string();

    match resolve_exception_class(py, class_name) {
        Ok(cls) => PyErr::from_value(
            cls.call1((message,))
                .unwrap_or_else(|_| cls.call0().unwrap()),
        ),
        Err(_) => PyRuntimeError::new_err(message),
    }
}

fn resolve_exception_class<'py>(py: Python<'py>, class_name: &str) -> PyResult<Bound<'py, PyAny>> {
    let module = py.import("kaos_nlp_transformers.errors")?;
    module.getattr(class_name)
}
