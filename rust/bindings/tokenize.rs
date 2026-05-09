//! Thin PyO3 wrapper for the HF ``tokenizers`` crate.
//!
//! This module is mostly an internal/test surface — the production
//! embedding and reranker paths use ``core::tokenize::TokenizerWrapper``
//! directly, never going through Python. Exposed here so:
//!
//! * Test code can ``from kaos_nlp_transformers._rust.tokenize import
//!   PyHFTokenizer; tok = PyHFTokenizer.from_file(...); tok.encode([...])``
//!   to verify tokenization end-to-end without running a full model.
//!
//! * Future MCP tools that want to expose tokenization as a primitive
//!   have a ready entry point.

use crate::bindings::util::map_backend_error;
use crate::core::tokenize::TokenizerWrapper;
use pyo3::prelude::*;

/// Python-side wrapper around ``TokenizerWrapper``.
#[pyclass(name = "Tokenizer", module = "kaos_nlp_transformers._rust.tokenize")]
pub(crate) struct PyHFTokenizer {
    inner: TokenizerWrapper,
}

#[pymethods]
impl PyHFTokenizer {
    /// Load a tokenizer from a ``tokenizer.json`` path on disk.
    #[staticmethod]
    fn from_file(py: Python<'_>, path: &str, max_seq_len: usize) -> PyResult<Self> {
        let inner =
            TokenizerWrapper::from_file(path, max_seq_len).map_err(|e| map_backend_error(py, e))?;
        Ok(Self { inner })
    }

    /// Encode a batch of strings. Returns ``(input_ids, attention_mask,
    /// token_type_ids)`` as ``list[list[int]]`` triples (Python int64
    /// is the natural container).
    #[allow(clippy::type_complexity)]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
    ) -> PyResult<(Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<Vec<i64>>)> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let encoded = self
            .inner
            .encode_batch(&refs)
            .map_err(|e| map_backend_error(py, e))?;
        Ok((
            encoded.input_ids,
            encoded.attention_mask,
            encoded.token_type_ids,
        ))
    }

    /// Padding token id used by this tokenizer (BERT-family default 0).
    #[getter]
    fn pad_id(&self) -> u32 {
        self.inner.pad_id
    }

    /// Maximum sequence length applied by this tokenizer (truncation cap).
    #[getter]
    fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len
    }
}

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "tokenize")?;
    m.add_class::<PyHFTokenizer>()?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_transformers._rust.tokenize", &m)?;
    Ok(())
}
