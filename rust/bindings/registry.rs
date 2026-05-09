//! PyO3 wrapper for the model-registry helpers. Phase 1 stub —
//! concrete `capabilities()` and `vendored_model_path()` land in P3.1.

use pyo3::prelude::*;

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "registry")?;

    // Expose the cargo package version so callers can read it
    // straight from the Rust side. This unlocks the kaos-nlp-core-style
    // single-source-of-truth pattern where `kaos_nlp_transformers._version`
    // reads from `_rust.registry.__version__`.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_transformers._rust.registry", &m)?;
    Ok(())
}
