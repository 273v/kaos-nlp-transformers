//! PyO3 wrapper for the embedding backend. Phase 1 stub — concrete
//! `PyEmbeddingBackend` lands in P3.2.

use pyo3::prelude::*;

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "embedding")?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_transformers._rust.embedding", &m)?;
    Ok(())
}
