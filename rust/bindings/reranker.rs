//! PyO3 wrapper for the cross-encoder reranker. Phase 1 stub.

use pyo3::prelude::*;

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "reranker")?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_transformers._rust.reranker", &m)?;
    Ok(())
}
