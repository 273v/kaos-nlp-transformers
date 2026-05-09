//! Pure-Rust core (no PyO3). Each submodule corresponds to a phase of
//! the migration plan; they land progressively in Phase 2.

pub mod backend;
pub mod device;
pub mod error;
pub mod model_loader;
pub mod model_registry;
pub mod ort_runtime;
pub mod pooling;
pub mod reranker;
pub mod tokenize;
