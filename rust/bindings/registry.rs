//! PyO3 wrapper for the model-registry helpers.
//!
//! Three pyfunctions:
//!
//! * ``capabilities() -> dict`` — compile-time + runtime capability
//!   snapshot. Replaces the Python-side ``_detect_onnx_providers``
//!   (P4.4). Build-time features ARE the truth source post-migration.
//!
//! * ``vendored_model_path(model_id: str) -> Optional[str]`` — moves
//!   the resolver from Python into Rust so the model2vec loader and
//!   any future Rust loader share one source of truth (KNT-401).
//!
//! * ``__version__`` — Cargo SemVer. Used by ``_version.py`` for the
//!   editable-install fallback path.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;

use crate::core::device::Capabilities;

/// Compile-time + runtime capability snapshot.
///
/// Returns a dict shaped like::
///
///   {
///     "cpu": True,
///     "cuda": <bool>,           # True iff cdylib was built with --features gpu
///     "openvino": <bool>,       # True iff cdylib was built with --features openvino
///     "build_features": ["std", "ndarray", ...],   # cargo features active in this build
///   }
///
/// The Python device-detection layer (``device.py``) consumes this
/// and reconciles it with OS-level GPU probes (nvidia-smi, /dev/kfd,
/// platform.machine) to populate ``LatentDevice`` entries.
#[pyfunction]
fn capabilities(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let caps = Capabilities::current();
    let d = PyDict::new(py);
    d.set_item("cpu", caps.cpu)?;
    d.set_item("cuda", caps.cuda)?;
    d.set_item("openvino", caps.openvino)?;

    // Surface a stable list of compile-time feature flags. Useful for
    // diagnostics ("which wheel did I install?") and for the MCP info
    // tool to render a one-liner.
    let features = PyList::empty(py);
    if caps.cuda {
        features.append("gpu")?;
    }
    if caps.openvino {
        features.append("openvino")?;
    }
    d.set_item("build_features", features)?;

    Ok(d)
}

/// Resolve a model_id to a vendored on-disk path under
/// ``kaos_nlp_transformers/_vendor/`` if one exists.
///
/// Audit-05 KNT-401: the wheel ships ``minishlab/potion-base-8M``
/// vendored so air-gapped installs work without the network. Returns
/// the absolute path string, or None if no vendored copy exists.
///
/// Detection mirrors the Python ``_vendored_model_path`` exactly:
/// the directory must exist AND contain a non-empty
/// ``model.safetensors``. Tries three slug shapes (full id, slash→dash,
/// last-segment) so callers can pass either ``"minishlab/potion-base-8M"``
/// or ``"potion-base-8M"``.
#[pyfunction]
fn vendored_model_path(py: Python<'_>, model_id: &str) -> PyResult<Option<String>> {
    // Resolve `kaos_nlp_transformers.__file__` to find the package's
    // installed location. Going through Python is the right move here:
    // the Rust cdylib doesn't know its own filesystem location at
    // compile time (could be in a wheel install, an editable build,
    // a venv, etc.), but Python imports always know the package path.
    let pkg = py.import("kaos_nlp_transformers")?;
    let pkg_file: String = pkg.getattr("__file__")?.extract()?;
    let pkg_root = match PathBuf::from(&pkg_file).parent() {
        Some(p) => p.to_path_buf(),
        None => return Ok(None),
    };
    let vendor_root = pkg_root.join("_vendor");
    if !vendor_root.is_dir() {
        return Ok(None);
    }

    let candidates: Vec<PathBuf> = vec![
        vendor_root.join(model_id),
        vendor_root.join(model_id.replace('/', "-")),
        vendor_root.join(model_id.split('/').next_back().unwrap_or(model_id)),
    ];

    for cand in candidates {
        let weights = cand.join("model.safetensors");
        if let Ok(meta) = std::fs::metadata(&weights) {
            if meta.is_file() && meta.len() > 0 {
                return Ok(Some(cand.to_string_lossy().into_owned()));
            }
        }
    }
    Ok(None)
}

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "registry")?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(capabilities, &m)?)?;
    m.add_function(wrap_pyfunction!(vendored_model_path, &m)?)?;

    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_transformers._rust.registry", &m)?;
    Ok(())
}
