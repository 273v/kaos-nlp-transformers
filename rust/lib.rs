//! kaos-nlp-transformers: Rust core for dense embeddings + cross-encoder
//! reranking.
//!
//! Audit-07 KNT-601 (0.2.0): the embedding/reranker backend stack moved
//! from the Python `fastembed` wrapper to a Rust cdylib that calls
//! [ort](https://github.com/pykeio/ort) directly. The Python public API
//! (`EmbeddingModel.load`, `EmbeddingModel.embed`, `CrossEncoderReranker`)
//! is preserved; the inference path goes through this crate's
//! `core::backend::Backend` trait.
//!
//! Layer cake:
//!
//! ```text
//!   Python:  EmbeddingModel.embed(texts)
//!     |
//!     v
//!   PyO3:    bindings::embedding::PyEmbeddingBackend.embed
//!     |   (py.allow_threads — GIL released for the batch)
//!     v
//!   Rust:    core::backend::Backend::embed
//!     |
//!     v
//!   ort:     Session::run on libonnxruntime (statically linked)
//! ```
//!
//! The `core::` modules are pure Rust (no PyO3) and are independently
//! testable via `cargo test`. The `bindings::` modules are the PyO3
//! wrappers and depend on the core layer.

// Many public core items are only consumed by the bindings layer, not within
// the crate itself. Allow dead_code at the crate root to avoid false positives.
#![allow(dead_code)]
// Core five lints — mirror kaos-nlp-core's selection.
#![warn(rust_2018_idioms)]
#![warn(rust_2021_compatibility)]
#![warn(unreachable_pub)]
#![warn(unused_qualifications)]
// Selected `clippy::pedantic` lints. Same carryover policy as kaos-nlp-core
// audit-01: these fire across cast/index sites and would gate every PR if
// hard-warn'd. Downgrade to allow; backfill incrementally per
// KNT-601-followup.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::redundant_clone)]

mod bindings;
pub mod core;

use pyo3::prelude::*;

/// The root Python module `kaos_nlp_transformers._rust`.
///
/// # Audit KNT-602 — Send+Sync + free-threaded Python
///
/// Declared `gil_used = false` per audit KNT-602 (mirror of kaos-nlp-core's
/// KNC-008) to declare free-threaded Python compatibility. Every
/// `#[pyclass]` registered below holds either:
///   - owned plain data (no interior mutability beyond `Mutex`/`RwLock`), or
///   - `Arc<T>` where `T: Send + Sync`.
///
/// Specifically:
///   - `bindings::embedding::PyEmbeddingBackend` → `Arc<dyn core::backend::Backend + Send + Sync>`
///   - `bindings::reranker::PyCrossEncoderBackend` → `Arc<dyn core::reranker::CrossEncoder + Send + Sync>`
///
/// `ort::Session` is `Send + Sync` as of ort 2.0.0-rc.10.
/// `tokenizers::Tokenizer` is `Send + Sync`.
///
/// This audit must be re-verified whenever a new `#[pyclass]` lands or
/// when ort/tokenizers pin bumps.
#[pymodule(gil_used = false)]
#[pyo3(name = "_rust")]
fn kaos_nlp_transformers_rust(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Submodules will be wired up in P3.4. The skeleton below is what
    // ships in P1; concrete bindings land progressively across Phase 3.
    bindings::embedding::register_module(m)?;
    bindings::reranker::register_module(m)?;
    bindings::registry::register_module(m)?;
    bindings::tokenize::register_module(m)?;

    // Set __path__ so Python treats this as a package (needed for
    // submodule imports like `from kaos_nlp_transformers._rust.embedding
    // import EmbeddingBackend`).
    m.setattr("__path__", pyo3::types::PyList::empty(py))?;
    m.setattr("__package__", "kaos_nlp_transformers._rust")?;

    Ok(())
}
