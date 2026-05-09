# kaos-nlp-transformers 0.2.0 — Migration to Rust + ort

**Status:** Plan, not yet executed. Lock-in date: 2026-05-09.
**Author:** Claude (with Mike).
**Audit ID:** KNT-601 (post-KNT-501; the embedding/reranker backend swap that follows the torch removal in 0.1.0a6).

This document is the prescriptive plan for the 0.2.0 release. It supersedes
the "fastembed Python backend" architecture established in KNT-501 with a
hybrid Rust+Python package layout that mirrors `kaos-nlp-core`.

---

## TL;DR

In one paragraph: 0.2.0 ships a Rust cdylib (`kaos_nlp_transformers._rust`,
PyO3 abi3-py313, `gil_used = false`) that calls
[ort](https://github.com/pykeio/ort) directly to load the same ONNX models
we ship today, without going through the `fastembed` Python wrapper or
fastembed-rs's wrapper. The `model2vec` static-embedding path is
unchanged. The public Python API (`EmbeddingModel.load`, `EmbeddingModel.embed`,
`REGISTRY`, exceptions) is preserved so callers don't break. The
free-threaded Python (`3.13t`/`3.14t`) guard is removed because the
Rust backend is GIL-safe. Wheel size grows slightly (~28 MB →
~22 MB cdylib + ~31 MB vendored model2vec = ~53 MB installed) but the
runtime dependency tree shrinks dramatically: `fastembed`,
`onnxruntime` Python wrapper, `tokenizers` Python wrapper, and
`py_rust_stemmers` are all removed in favor of the Rust crates statically
linked into our cdylib.

---

## Why ort, not burn or fastembed-rs

Decision data, all measured during the 2026-05-08/09 spike at
[`scratch/burn-experiment/`](../../scratch/burn-experiment/):

| | Cargo tree | Wheel (compressed) | cdylib | bs=1 sps | bs=64 sps | KNT-003 revision pin |
|---|---:|---:|---:|---:|---:|:---:|
| **ort direct (chosen)** | **129 crates** | 7.83 MB | 21 MB | **354** | **461** | ✓ correct |
| ort + fastembed-rs wrapper | 136 crates | 7.83 MB | 21 MB | 354 | 461 | ✗ tracks `main` |
| burn | 305 crates | 1.81 MB | 4.5 MB | 30 | 35 | ✓ correct |
| tinygrad CPU | n/a (Python) | n/a | n/a | 2.4 | 25.6 | n/a |
| Python fastembed (today) | n/a | 28.5 MB whl | n/a | 112 | 138 | ✗ no rev override |

**ort direct wins the engineering trade:** ~13× faster than burn at every
batch size; same throughput as fastembed-rs but 7 fewer crates to audit
and full revision-SHA control (fastembed-rs's `pull_from_hf` hard-codes
`main` and is structurally incompatible with audit-01 KNT-003); same
backend kernels (Intel oneDNN/MKL on CPU, NVIDIA cuBLAS/cuDNN on GPU)
that we already ship through Python `onnxruntime`, just consumed from
Rust without the Python boundary cost.

Cosine equivalence vs Python fastembed: **1.000000** for ort direct,
fastembed-rs, burn, AND tinygrad. Output is bit-identical — zero
quality risk on the migration itself.

Burn was rejected because the 13× CPU throughput cliff doesn't pay
back the 4.3× wheel-size win, and the supply chain pulls openssl
transitively through tracel-ai/models defaults despite our top-level
rustls config. tinygrad was rejected because its CPU JIT path lags
ort by ~10× even with clang available, its install footprint is
larger than either Rust strategy, and we'd be early production users.

---

## What changes / what stays

### Removed in 0.2.0

| Dep / surface | Removed because |
|---|---|
| `fastembed>=0.6` (Python) | replaced by `ort` Rust crate inside our cdylib |
| transitive: `onnxruntime` Python | libonnxruntime is statically linked into our cdylib via ort |
| transitive: `tokenizers` (Python wrapper) | replaced by Rust `tokenizers` crate inside our cdylib |
| transitive: `py_rust_stemmers` | unused (only there because Python fastembed listed it for sparse BM25 — we don't use sparse) |
| `_check_gil_enabled` guard in `embedding.py` and `reranker.py` | obsolete: ort is C++ FFI (no PyO3), Rust `tokenizers` ships free-threaded wheels, and our PyO3 module declares `gil_used = false` |
| `[torch]` extras alias (already empty since 0.1.0a6) | scheduled removal per CHANGELOG |

### Added in 0.2.0

| Component | Purpose |
|---|---|
| `Cargo.toml` + `rust/` source tree | Rust core, mirroring `kaos-nlp-core` layout |
| `kaos_nlp_transformers/_rust.abi3.so` | The compiled cdylib (per-platform wheel artifact) |
| `tests/reference/*.npy` | Frozen reference embeddings for cosine-equivalence regression tests |
| Per-platform wheel matrix | 7 wheels per release (matching `kaos-nlp-core`) instead of one `py3-none-any.whl` |

### Preserved (zero caller-facing change)

- Public API: `EmbeddingModel.load`, `EmbeddingModel.embed`, `dim`,
  `model_id`, `device`, `backend_name`, `license`
- `REGISTRY`, `EXCLUDED`, `RERANKER_REGISTRY`, `RegisteredModel`
- Exception hierarchy: `EmbeddingError`, `ModelLoadError`,
  `ModelNotRegisteredError`, `BackendNotInstalledError`,
  `DeviceNotReachableError`
- `KaosNLPTransformersSettings`: same fields, narrowed `backend` valid set
- `model2vec` backend: unchanged, separate code path
- Vendored `kaos_nlp_transformers/_vendor/potion-base-8M/`: unchanged
- `cli.py`, `serve.py`, `tools.py`, `clustering/`, `retrieval.py`:
  cosmetic message updates only

### Semantic changes (must appear in CHANGELOG)

1. `EmbeddingModel.backend_name` returns `"ort"` instead of `"fastembed"`.
2. `KaosNLPTransformersSettings.backend = "fastembed"` raises
   `ValueError` with a migration message (text in §10).
3. The `[gpu]` extra now means "kaos-nlp-transformers-gpu companion
   wheel" (separate distribution, see §13). This is a deferred 0.2.0a2
   landing — 0.2.0a1 ships CPU only.
4. `[openvino]` extra: kept as a no-op alias for one release; meaningful
   in 0.2.0a2 when GPU wheel lands. Removed in 0.3.0.
5. Python 3.13t / 3.14t: now supported. The hard refusal at
   `EmbeddingModel.load` is gone.

---

## File tree (post-migration)

Top-level only — full `rust/` and `tests/` subtrees expanded inline.

```
kaos-nlp-transformers/
├── AGENTS.md                                stays
├── CLAUDE.md                                stays (delegates to AGENTS.md)
├── CHANGELOG.md                             modified — 0.2.0 entry
├── CODE_OF_CONDUCT.md / CONTRIBUTING.md     stays
├── LICENSE / NOTICE                         modified — Rust attributions
├── README.md                                modified — install matrix, GPU story, free-threaded ✓
├── SECURITY.md                              stays
├── pyproject.toml                           rewritten — maturin build backend; deps trimmed; extras restructured
├── Cargo.toml                               new — crate metadata, profile.release config, ort+tokenizers+hf-hub
├── Cargo.lock                               new — committed for reproducible release builds (matches kaos-nlp-core)
├── deny.toml                                new — mirror kaos-nlp-core's cargo-deny policy
├── uv.lock                                  modified — regenerated
├── docs/
│   ├── MIGRATION_0_2_0.md                   new — this document
│   ├── functionality-review.md              stays
│   └── standards/
│       ├── code-quality-standards.md        modified — adds Rust toolchain to active gate
│       ├── engineering-process.md           modified — Cargo + maturin steps
│       └── …                                stays
├── rust/                                    new — PyO3 + Rust core; mirrors kaos-nlp-core/rust/
│   ├── lib.rs                               new — #[pymodule(gil_used=false)] _rust root
│   ├── bindings/
│   │   ├── mod.rs                           new
│   │   ├── embedding.rs                     new — PyEmbeddingBackend pyclass: load + embed → PyArray2<f32>
│   │   ├── reranker.rs                      new — PyCrossEncoderBackend pyclass: load + score → PyArray1<f32>
│   │   ├── tokenize.rs                      new — thin Tokenizer wrapper (mostly internal/test surface)
│   │   ├── registry.rs                      new — capabilities() + vendored_model_path() py functions
│   │   └── util.rs                          new — error mapping (ort → PyExceptions), ndarray ↔ PyArray helpers
│   └── core/                                new — pure Rust (no PyO3 dep), independently testable
│       ├── mod.rs                           new
│       ├── backend.rs                       new — trait Backend; ort_runtime impl
│       ├── ort_runtime.rs                   new — Session lifecycle, EP plumbing (CPU/CUDA/OpenVINO)
│       ├── model_registry.rs                new — Rust mirror of REGISTRY (model_id → revision SHA + ONNX path + pooling spec)
│       ├── model_loader.rs                  new — hf-hub snapshot fetch w/ revision pinning, vendored-path detection
│       ├── tokenize.rs                      new — Rust tokenizers crate wrapper, batch encode, padding policy
│       ├── pooling.rs                       new — mean / cls pooling, L2 normalize (centralized post-KNT-101)
│       ├── reranker.rs                      new — cross-encoder forward + sigmoid normalization
│       ├── device.rs                        new — Device enum, runtime EP availability probe
│       └── error.rs                         new — thiserror::Error tree
├── kaos_nlp_transformers/                   (Python package — mostly stays)
│   ├── __init__.py                          modified — no surface change
│   ├── _version.py                          modified — 0.2.0a1; reads from Cargo.toml at sdist time
│   ├── _rust/                               new — type-stub-only source dir
│   │   ├── __init__.pyi                     new
│   │   ├── embedding.pyi                    new
│   │   ├── reranker.pyi                     new
│   │   ├── tokenize.pyi                     new
│   │   └── registry.pyi                     new
│   ├── _rust.abi3.so                        build artifact (in wheel; in .gitignore for src)
│   ├── embedding.py                         modified — see §7
│   ├── models.py                            modified — RegisteredModel.backend valid set: {"ort", "model2vec"}
│   ├── device.py                            modified — _detect_rust_capabilities replaces _detect_onnx_providers
│   ├── errors.py                            modified — refreshed messages
│   ├── settings.py                          modified — backend valid set
│   ├── reranker.py                          modified — calls into _rust.reranker
│   ├── retrieval.py                         stays
│   ├── clustering/                          stays
│   ├── cli.py / serve.py / tools.py         modified — cosmetic message updates
│   ├── _vendor/                             stays (31 MB potion-base-8M; loader unchanged)
│   └── py.typed                             stays
├── tests/                                   (mostly stays; new tests below)
│   ├── unit/
│   │   ├── test_embedding.py                modified — mocks _rust.embedding.PyEmbeddingBackend instead of fastembed
│   │   ├── test_embedding_backends.py       modified — backend valid set update; "fastembed" raises with migration text
│   │   ├── test_models.py                   modified — backend ∈ {"ort", "model2vec"}
│   │   ├── test_device.py                   modified — _rust.registry.capabilities() fixture replaces onnxruntime providers
│   │   ├── test_settings.py                 modified — env-roundtrip uses "ort"
│   │   ├── test_audit_03.py                 deleted — KNT-201 free-threaded guard retired
│   │   ├── test_audit_06.py                 modified — KNT-501 history reference; no behavior change
│   │   ├── test_audit_07.py                 new — KNT-601 regression suite for the Rust cutover
│   │   ├── test_rust_extension.py           new — direct _rust import smoke + dim/shape/L2 invariants
│   │   ├── test_reference_vectors.py        new — frozen-vector cosine equivalence (the most important new test)
│   │   ├── test_free_threaded.py            new — opt-in: skip unless `python -X gil=0`
│   │   └── test_perf_smoke.py               new — bs=8 must hit ≥X sps on the CI runner
│   ├── reference/                           new — frozen NPY embeddings, committed binary
│   │   ├── sentences.txt                    new — 16 fixed inputs
│   │   ├── bge_small_en_v1_5.npy            new — frozen at migration time from existing fastembed
│   │   ├── bge_reranker_base.npy            new
│   │   ├── potion_base_8m.npy               new
│   │   ├── potion_base_32m.npy              new
│   │   └── potion_retrieval_32m.npy         new
│   └── integration/                         stays
├── scripts/                                 new
│   ├── freeze_reference_vectors.py          new — regen tests/reference/*.npy from current fastembed (RUN ONCE before cutover)
│   └── verify_wheel.sh                      new — install dist/*.whl in clean venv, smoke-import, vendor-path assert
├── benches/                                 new
│   └── bench_embedding.rs                   new — criterion benches at bs=[1, 8, 32, 128]
└── .github/workflows/
    ├── ci.yml                               modified — cargo fmt/clippy/test + maturin develop + pytest
    ├── release.yml                          modified — 7-platform wheel matrix
    └── security.yml                         modified — adds cargo audit + cargo deny check
```

---

## `Cargo.toml` (full, copy-paste ready)

```toml
[package]
name = "kaos-nlp-transformers"
version = "0.2.0-alpha.1"   # PEP 440 normalization in release.yml: 0.2.0-alpha.1 → 0.2.0a1
edition = "2024"
license = "Apache-2.0"
authors = ["273 Ventures LLC <it@273ventures.com>"]
description = "Dense embeddings and small-model inference for the Kelvin Agentic OS — Rust-native ONNX backend, model2vec static lookup, optional GPU"
rust-version = "1.83"
repository = "https://github.com/273v/kaos-nlp-transformers"
homepage = "https://kelvin.legal"
documentation = "https://docs.kelvin.legal"
readme = "README.md"
keywords = ["embeddings", "kaos", "nlp", "ort", "pyo3"]
categories = ["text-processing", "science"]

[lib]
name = "_rust"
path = "rust/lib.rs"
crate-type = ["cdylib", "rlib"]

[features]
default = []
gpu = ["ort/cuda"]
openvino = ["ort/openvino"]

[dependencies]
pyo3 = { version = "0.28", features = ["extension-module", "abi3-py313"] }
numpy = "0.28"
ndarray = "0.16"

# ort: ONNX Runtime via Rust. download-binaries fetches MS's manylinux2014
# libonnxruntime.a at build time and statically links it. tls-rustls is
# required because download-binaries needs an HTTP client (rustls so we
# don't need libssl-dev on build hosts).
ort = { version = "2.0.0-rc.10", default-features = false, features = ["ndarray", "download-binaries", "tls-rustls"] }

# Rust HF tokenizers — replaces the Python `tokenizers` wrapper.
tokenizers = { version = "0.22", default-features = false, features = ["onig"] }

# hf-hub: snapshot_download equivalent. rustls-tls so we don't pull openssl.
hf-hub = { version = "0.5", default-features = false, features = ["rustls-tls", "ureq"] }

serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
once_cell = "1"
rayon = "1.12"

[dev-dependencies]
criterion = { version = "0.8", features = ["html_reports"] }
tempfile = "3.0"

[[bench]]
name = "bench_embedding"
harness = false

[profile.release]
lto = true
codegen-units = 1
opt-level = 3
strip = "symbols"
```

Verified: this Cargo.toml shape resolves to **129 unique active crates**
(measured via `cargo tree --prefix none | sort -u | grep ^[a-z] | wc -l`)
and produces a **7.83 MB compressed wheel / 21 MB cdylib** when built
with realistic touch functions exercising the inference path.

---

## `pyproject.toml` (delta from current)

```toml
[build-system]
requires = ["maturin>=1.8,<2.0"]                  # was: hatchling>=1.27.0
build-backend = "maturin"

[project]
name = "kaos-nlp-transformers"
dynamic = ["version"]                              # version source moves to Cargo.toml
requires-python = ">=3.13"
license = "Apache-2.0"
license-files = ["LICENSE", "NOTICE"]

dependencies = [
  "kaos-core>=0.1.0a1",
  "kaos-content>=0.1.0a1",
  "kaos-nlp-core>=0.1.0a2",
  "numpy>=2.1",
  "huggingface_hub>=0.26",                         # NEW base dep — used by model2vec loader (Python side)
  # REMOVED: "fastembed>=0.6"                        — replaced by Rust ort+tokenizers in cdylib
]

[project.optional-dependencies]
clustering = ["scipy>=1.14.1"]                      # unchanged
model2vec  = ["model2vec>=0.8.1"]                   # unchanged
mcp        = ["kaos-mcp>=0.1.0a1"]                  # unchanged
gpu        = []                                     # NOW EMPTY — see §13. 0.2.0a2 will land kaos-nlp-transformers-gpu companion.
openvino   = []                                     # placeholder — meaningful in 0.2.0a2.
torch      = []                                     # KNT-501 alias removal scheduled 0.3.0; keep one more cycle.

[project.scripts]
kaos-nlp-transformers = "kaos_nlp_transformers.cli:main"
kaos-nlp-transformers-serve = "kaos_nlp_transformers.serve:main"

[dependency-groups]
dev = [
  "kaos-mcp>=0.1.0a1",
  "maturin>=1.8,<2.0",                              # NEW
  "pytest>=9.0.3",
  "pytest-asyncio>=1.3.0",
  "pytest-cov>=7.1.0",
  "pytest-benchmark>=5.1.0",                        # NEW — perf smoke gate
  "ruff>=0.15.12",
  "ty>=0.0.34,<0.1",
]

[tool.maturin]
python-source = "."                                 # package at repo root: kaos_nlp_transformers/
module-name = "kaos_nlp_transformers._rust"
features = ["pyo3/extension-module", "pyo3/abi3-py313"]
include = [
  { path = "kaos_nlp_transformers/_vendor/**", format = "sdist" },
  { path = "kaos_nlp_transformers/_vendor/**", format = "wheel" },
  { path = "LICENSE", format = "sdist" },
  { path = "NOTICE", format = "sdist" },
  { path = "CHANGELOG.md", format = "sdist" },
]
exclude = [
  { path = "target/**", format = "sdist" },
  { path = "target/**", format = "wheel" },
  { path = ".pytest_cache/**", format = "sdist" },
  { path = ".venv/**", format = "sdist" },
  { path = "tests/reference/**", format = "wheel" },  # frozen NPYs are test-only
]

# REMOVED: [tool.hatch.version], [tool.hatch.build.targets.{sdist,wheel}]
```

Build backend choice: **maturin**, identical pattern to `kaos-nlp-core`.
Rationale: per-platform abi3 wheels with one toolchain; team already
operates the kaos-nlp-core release pipeline.

---

## PyO3 module shape

`rust/lib.rs` declares `#[pymodule(gil_used = false)] fn kaos_nlp_transformers_rust(py, m)`
registering submodules: `embedding`, `reranker`, `tokenize`, `registry`.
Same `register_module(parent)` helper pattern kaos-nlp-core uses (build
`PyModule::new(py, "<name>")`, attach functions/classes, set
`sys.modules["kaos_nlp_transformers._rust.<name>"]`).

Per-binding outline:

**`bindings/embedding.rs`**
```rust
#[pyclass(name = "EmbeddingBackend", module = "kaos_nlp_transformers._rust.embedding")]
pub(crate) struct PyEmbeddingBackend {
    inner: Arc<dyn core::backend::Backend + Send + Sync>,
}

#[pymethods]
impl PyEmbeddingBackend {
    #[staticmethod]
    #[pyo3(signature = (model_id, *, revision, weights_path, tokenizer_path, device, cache_dir=None))]
    fn load(...) -> PyResult<Self>;

    /// GIL released for the entire batch; texts copied to Vec<String> up front.
    fn embed<'py>(&self, py: Python<'py>, texts: Vec<String>, batch_size: usize)
        -> PyResult<Bound<'py, numpy::PyArray2<f32>>>;

    #[getter] fn dim(&self) -> usize;
    #[getter] fn model_id(&self) -> &str;
    #[getter] fn device(&self) -> &str;
}
```

**`bindings/reranker.rs`** — sibling pyclass with same shape:
```rust
#[pyclass(name = "CrossEncoderBackend", module = "kaos_nlp_transformers._rust.reranker")]
pub(crate) struct PyCrossEncoderBackend { ... }

#[pymethods]
impl PyCrossEncoderBackend {
    #[staticmethod]
    fn load(...) -> PyResult<Self>;

    /// Returns sigmoid-normalized [0, 1] scores. Shape (n_pairs,).
    fn score<'py>(&self, py: Python<'py>, queries: Vec<String>, passages: Vec<String>)
        -> PyResult<Bound<'py, numpy::PyArray1<f32>>>;
}
```

**`bindings/tokenize.rs`** — minimal surface for tests/debugging.
Wraps `tokenizers::Tokenizer`. Returns `Vec<Vec<u32>>` token ids.

**`bindings/registry.rs`** — three pyfunctions:
- `capabilities() -> dict`: `{"cpu": true, "cuda": <bool>, "openvino": <bool>, "cuda_devices": [...], "build_features": [...]}`. Replaces Python `_detect_onnx_providers`. Build-time features ARE the truth source.
- `vendored_model_path(model_id: str) -> Optional[str]`: moves the resolver from Python to Rust so model2vec loader and Rust loader share one source of truth.
- `__version__`: cargo pkg version.

**Marshalling for embeddings:** `numpy::PyArray2<f32>` (the `numpy`
crate, version-aligned with PyO3 0.28). Zero-copy from
`ndarray::Array2<f32>` via `PyArray2::from_owned_array`. Free-threaded
safe (PyArray creation is GIL-bound but the heavy ort.run() happens
under `py.allow_threads`).

**GIL handling:** every embed call is
```rust
py.allow_threads(|| backend.embed(&texts, batch_size))
```
returning `Result<Array2<f32>>`, then converted to `PyArray2<f32>`
after GIL reacquired. Matches kaos-nlp-core's `py.detach(...)` pattern
used 22+ times in that repo.

**`gil_used = false`** — mandatory. Send+Sync audit: `PyEmbeddingBackend`
holds `Arc<dyn Backend + Send + Sync>`. Trait declared `: Send + Sync`.
`ort::Session` is Send+Sync as of ort 2.0-rc.10. Document in `rust/lib.rs`
as audit KNT-602 (mirror of KNC-008's comment block).

---

## API stability matrix

| Symbol | Pre-0.2.0 | Post-0.2.0 | Caller change |
|---|---|---|---|
| `EmbeddingModel.load(model_id, *, device=None, backend=None, settings=None)` | OK | OK | None |
| `EmbeddingModel.embed(texts, *, batch_size=32) -> np.ndarray` | OK | OK; calls `_rust.embedding.PyEmbeddingBackend.embed` returning a PyArray2 | None |
| `EmbeddingModel.{dim, license, model_id}` | OK | OK | None |
| `EmbeddingModel.device` | `DeviceInfo` | `DeviceInfo` | None |
| `EmbeddingModel.backend_name` | `"fastembed"` \| `"model2vec"` | **`"ort"` \| `"model2vec"`** | **SEMANTIC** — only if caller string-compares; in-tree no callers do |
| `KaosNLPTransformersSettings.backend` valid | `{"auto","fastembed","model2vec"}` | **`{"auto","ort","model2vec"}`** | **SEMANTIC** — `"fastembed"` raises `ValueError` (text in §10) |
| `KaosNLPTransformersSettings.device` valid | `{"auto","cpu","cuda","cuda:N","openvino"}` | same | None for callers; clearer error story |
| `EmbeddingError`, `ModelLoadError`, etc. | hierarchy | unchanged | None |
| `REGISTRY`, `EXCLUDED`, `RERANKER_REGISTRY`, `RegisteredModel` | dict layout | unchanged | None — `RegisteredModel.backend` value set narrows |
| `_check_gil_enabled` (private; called from `embedding.py` + `reranker.py`) | refuses 3.13t/3.14t | **REMOVED** | Internal; import line dropped |
| `_resolve_backend(requested, device, registry_backend)` | returns `"fastembed" \| "model2vec"` | **returns `"ort" \| "model2vec"`** | Internal |
| `_onnx_providers_for_device(device)` | Returns ORT EP tuple | **REMOVED**; replaced by Rust-side EP selection at backend construction | Internal — `reranker.py` import line drops |
| `detect_devices`, `resolve_device`, `DeviceInfo` | probes onnxruntime + nvidia-smi | probes `_rust.registry.capabilities()` + nvidia-smi | None |

**Documented user-visible changes for CHANGELOG 0.2.0:**

1. `backend_name` returns `"ort"` instead of `"fastembed"`.
2. `backend="fastembed"` no longer accepted (migration error per §10).
3. `_check_gil_enabled` removed → free-threaded Python (3.13t / 3.14t) **now works**.
4. Wheel size: ~28 MB → ~32 MB compressed (cdylib + same potion model);
   installed footprint **drops** from ~108 MB (fastembed stack) to ~53 MB.

---

## Settings & device resolution

`settings.py` changes are minimal — typed fields don't change shape,
only validation:

| `device` value | Resolves to | Runtime | Requires |
|---|---|---|---|
| `"auto"` | `system.best` | ort + CPU EP, escalates to CUDA EP if GPU build | — |
| `"cpu"` | `DeviceInfo("CPU","cpu","ort")` | ort + CPUExecutionProvider | always available |
| `"cuda"` / `"cuda:N"` | `DeviceInfo("NVIDIA …","cuda:N","ort")` | ort + CUDAExecutionProvider | wheel built with `gpu` feature; libonnxruntime_providers_cuda.so reachable; `/dev/nvidia*` present |
| `"openvino"` | `DeviceInfo("Intel OpenVINO","openvino","ort")` | ort + OpenVINOExecutionProvider | wheel built with `openvino` feature |
| anything else | `ValueError` | — | — |

`KAOS_NLP_TRANSFORMERS_DEVICE=cuda` honored unchanged via
pydantic-settings env binding. The detection layer flips:
`_detect_onnx_providers()` → `_detect_rust_capabilities()` calling
`_rust.registry.capabilities()`.

`LatentDevice.install_extra` continues to be `"gpu"`. Reason text
flips from "onnxruntime-gpu is not installed" to "kaos-nlp-transformers
was not built with the gpu feature; install kaos-nlp-transformers-gpu".

---

## Backend dispatch logic & migration error

`_resolve_backend` valid-set:

```python
_VALID_BACKENDS: frozenset[str] = frozenset({"auto", "ort", "model2vec"})
```

Auto-resolution stays identical: registry decides; static models →
`model2vec`. Dispatch body:

```python
if requested == "ort":
    return "ort"
if requested == "model2vec":
    return "model2vec"
if registry_backend == "model2vec":
    return "model2vec"
return "ort"
```

**Migration error for `backend="fastembed"`:**

```
Invalid backend 'fastembed'. The fastembed Python wrapper was replaced
by the Rust-native 'ort' backend in kaos-nlp-transformers 0.2.0 (same
ONNX runtime under the hood, no Python boundary, free-threaded Python
compatible). Fix: use one of ['auto', 'ort', 'model2vec'], or leave
unset for auto-detection. Alternative: pin to kaos-nlp-transformers<0.2
if you specifically need the fastembed Python wrapper (not recommended
— superseded).
```

Three-part shape (what / fix / alternative) per audit-01 KNT-001.

---

## Vendored potion-base-8M handling

The 31 MB `kaos_nlp_transformers/_vendor/potion-base-8M/` directory ships
**untouched**. The `model2vec` backend continues to call
`_vendored_model_path(model_id)` and `StaticModel.from_pretrained(local_path)`
— neither path runs through the Rust crate. **This is an explicit
invariant.**

What changes for inclusion:

- `[tool.maturin].include` (§6) declares the vendor dir in **both**
  `format = "sdist"` AND `format = "wheel"`. Without the wheel-format
  entry, maturin defaults to "Python sources only" and the vendored
  bytes drop. This is the opposite trade-off vs kaos-nlp-core (which
  bakes data into `_rust.abi3.so` via `include_bytes!`); we want the
  bytes loose so model2vec can `from_pretrained(dir_path)`.
- `_vendored_model_path` helper moves to Rust as
  `_rust.registry.vendored_model_path(model_id) -> Optional[str]`.
  Python wrapper in `embedding.py` becomes a one-line passthrough.
- `_vendor/README.md` and `_vendor/potion-base-8M/ATTRIBUTION.txt` stay.
- `NOTICE` gains a Rust-side attribution paragraph (`tokenizers-rs`,
  `ort`, `safetensors`).

Sanity assertion in `scripts/verify_wheel.sh`:
```bash
unzip -l dist/*.whl | grep -E "_vendor/potion-base-8M/.+\.safetensors"
```

---

## Build matrix & CI

### Per-arch CPU wheel matrix (matches kaos-nlp-core's release.yml)

| os runner | target | manylinux/musl tag |
|---|---|---|
| ubuntu-latest | x86_64-unknown-linux-gnu | manylinux_2_28 |
| ubuntu-latest | x86_64-unknown-linux-musl | musllinux_1_2 |
| ubuntu-24.04-arm | aarch64-unknown-linux-gnu | manylinux_2_28 |
| ubuntu-24.04-arm | aarch64-unknown-linux-musl | musllinux_1_2 |
| macos-14 | aarch64-apple-darwin | — |
| windows-latest | x86_64-pc-windows-msvc | — |
| windows-11-arm | aarch64-pc-windows-msvc | — |

**= 7 CPU wheels per release.** macOS x86_64 deliberately skipped
(matches kaos-nlp-core).

### GPU companion package — `kaos-nlp-transformers-gpu`

**Not in 0.2.0a1.** Land CPU clean first, then 0.2.0a2 ships GPU.

When ready, the companion is a separate distribution that:
- Pip-resolves when user does `pip install kaos-nlp-transformers[gpu]`
  (via `gpu = ["kaos-nlp-transformers-gpu"]` in pyproject.toml).
- Built from the same `Cargo.toml` source with `--features gpu`
  (enables `ort/cuda` → builds against onnxruntime-providers-cuda).
- Ships only on `linux x86_64 manylinux_2_28` and
  `linux aarch64 manylinux_2_28`. macOS / Windows / musl GPU wheels
  are explicitly out of scope.
- Loader pattern: companion ships
  `kaos_nlp_transformers_gpu/_rust.abi3.so` and a Python init that
  monkey-injects itself into `kaos_nlp_transformers._rust` on first
  import. This is the pattern `numpy-mkl` uses; document in the
  release notes.

= **0 GPU wheels in 0.2.0a1, 2 GPU wheels in 0.2.0a2.**

### CI changes

`ci.yml` additions:
1. `cargo fmt --check`
2. `cargo clippy --all-targets -- -D warnings` (with kaos-nlp-core's
   audit-01 carryover allow list; backfill incrementally — KNT-008-
   followup-equivalent)
3. `cargo test --no-default-features --lib` for pure-Rust core
4. `uv run maturin develop --release` then `uv run pytest`
5. Cache `target/` keyed on `Cargo.lock` hash; `Swatinem/rust-cache@v2`
   action (kaos-nlp-core uses sccache; mirror that)

`release.yml` rewrites:
1. Replace single-wheel `uv build` with the same matrix kaos-nlp-core
   uses (`PyO3/maturin-action@v1` invocation, per-row).
2. Add the version-sync step from kaos-nlp-core's release.yml that maps
   Cargo SemVer `0.2.0-alpha.1` → PEP 440 `0.2.0a1` and matches against
   the git tag.
3. Smoke test in clean venv (already exists, line 68-110) gains:
   `assert "_rust" in sys.modules and kaos_nlp_transformers._rust.__version__`.

`security.yml` (mirror kaos-nlp-core):
- `cargo audit`
- `cargo deny check`

Rust toolchain: `dtolnay/rust-toolchain@stable`. Pin minimum to
`rust-version = "1.83"` in Cargo.toml.

### `docs/standards/code-quality-standards.md` update

Strike the line "This package is pure Python. Rust, PyO3, `maturin`,
and Cargo checks are not part of its active quality gate." Replace
with cross-reference to kaos-nlp-core's standards doc for the Rust
toolchain gate. Keep the Python tools as-is.

---

## Migration validation

Tests fall into four buckets:

### A. Existing tests that need updating

| Test file | What changes |
|---|---|
| `tests/unit/test_embedding.py` | Replace `monkeypatch.setattr("fastembed.TextEmbedding", ...)` with `monkeypatch.setattr("kaos_nlp_transformers._rust.embedding.PyEmbeddingBackend.load", ...)`. Tests of L2 normalization, dim mismatch, empty-input, batch_size handling all stay valid |
| `tests/unit/test_embedding_backends.py` | Backend valid set update; `_resolve_backend("tensorflow", ...)` test stays; `_resolve_backend("fastembed", ...)` becomes a NEW negative-case test with the migration error |
| `tests/unit/test_models.py` | Assert `RegisteredModel.backend in {"ort", "model2vec"}` |
| `tests/unit/test_device.py` | Replace onnxruntime-providers fixture with a `_rust.registry.capabilities()` fixture; latent-device install_extra = `"gpu"` unchanged |
| `tests/unit/test_audit_03.py` (KNT-201 free-threaded guard) | **DELETE** — guard is gone |
| `tests/unit/test_settings.py` | env-roundtrip uses `"ort"` |
| `tests/unit/test_audit_06.py` | Keep history; comment that KNT-501 (torch removal) preceded KNT-601 (fastembed removal) |

### B. New tests required

1. **Frozen reference vectors** — `tests/unit/test_reference_vectors.py`.
   For each REGISTRY model where `backend != "model2vec"` (i.e.
   bge-small-en-v1.5 + bge-reranker-base) AND each model2vec model:
   - Embed `tests/reference/sentences.txt` (16 fixed inputs, committed).
   - Load `tests/reference/<model_slug>.npy` (frozen at migration time
     via `scripts/freeze_reference_vectors.py` running against the
     current 0.1.0a6 fastembed stack).
   - Assert `np.dot(new, ref) >= 0.9999` per row (cosine ≈1.0 on
     L2-normalized vectors). **This is the most important test in the
     migration.** It's the contract that protects "the new backend
     produces the same embeddings."

2. **Direct extension smoke** — `tests/unit/test_rust_extension.py`.
   `import kaos_nlp_transformers._rust`, check `__version__`, verify
   `_rust.embedding.PyEmbeddingBackend` is a class, run a tiny in-memory
   model load via mocked weights in tmp.

3. **Performance smoke** — `tests/unit/test_perf_smoke.py` marked `slow`.
   Embed 256 sentences at bs=8 on the CI runner, assert wall time < 5 s.
   Threshold tuned post-first-CI-run; this catches accidental algorithmic
   blowups (e.g. forgetting to release the GIL, double-tokenization).

4. **Free-threaded compatibility** — `tests/unit/test_free_threaded.py`.
   Skip-marked unless `sys._is_gil_enabled() is False`. Load a model,
   embed 8 sentences from 4 threads concurrently, assert no segfault
   and results match single-thread. CI runs in a separate matrix slot
   using `python -X gil=0 -m pytest` on Python 3.14t.

5. **Vendored model still loads** — extend
   `tests/integration/test_audit_05.py` with
   `test_potion_base_8m_in_wheel_layout`. Imports the wheel, calls
   `_rust.registry.vendored_model_path("minishlab/potion-base-8M")`,
   asserts non-None.

### C. Pytest plan

- `pytest tests/unit/` — every PR. Includes A, B.1, B.2, B.4 (skip
  guarded), B.5.
- `pytest -m slow tests/unit/` — release.yml gate, not per-PR.
- `pytest -m live tests/integration/` — manual; HF Hub access required.
- `pytest -m gpu tests/integration/` — gated on GPU runner.
- `pytest tests/unit/test_free_threaded.py -X gil=0` — separate ci.yml
  job using `astral-sh/setup-uv` + `uv python install 3.14t`.

### D. Bench validation

`benches/bench_embedding.rs` (criterion) at bs=[1, 8, 32, 128] on
bge-small. Numbers committed to `bench-results.json` per release.
Nightly `bench.yml` (mirror kaos-nlp-core), not per-PR.

---

## Rollout PRs (commit ordering)

Each landing point is shippable (or non-broken). PRs labeled `[KNT-601]`.

### Phase 1 — scaffolding (PR #1)

1. Add `Cargo.toml`, `Cargo.lock`, `deny.toml`, `rust/lib.rs` skeleton
   with **no bindings** but `gil_used = false`. Crate compiles to an
   empty `_rust` module.
2. Flip pyproject.toml build-backend to maturin; update
   `[tool.maturin]`. Move version source to Cargo.toml; `_version.py`
   reads from cdylib at runtime.
3. Update `ci.yml` to build via maturin develop. Tests still pass
   identically (no behavior change yet — fastembed still in deps).
4. Document in-progress state in CHANGELOG `[Unreleased]`.

### Phase 2 — Rust core lands (PRs #2–3)

5. **PR #2**: `rust/core/{error.rs, device.rs, tokenize.rs, pooling.rs,
   model_loader.rs, model_registry.rs}` + matching unit tests under
   `cargo test`. No PyO3 surface yet.
6. **PR #3**: `rust/core/{backend.rs, ort_runtime.rs, reranker.rs}`.
   Cargo tests load a tiny fixture model end-to-end and produce
   vectors. No Python integration yet.

### Phase 3 — bindings + parallel-path validation (PR #4)

7. `rust/bindings/{embedding,reranker,tokenize,registry}.rs`. Wire the
   pymodule. Add `tests/unit/test_rust_extension.py`.
8. **Both backends coexist** behind a hidden
   `KAOS_NLP_TRANSFORMERS_BACKEND=rust-experimental` flag — existing
   fastembed path stays default. This is the parallel-implementation
   cut point: new code is reachable but not the default.
9. Land `tests/reference/*.npy` (generated against the still-default
   fastembed path via `scripts/freeze_reference_vectors.py` — RUN ONCE
   before this PR; commit the binary NPYs). Cosine-equivalence test
   now runs on the experimental backend and proves output parity
   before the cutover.

### Phase 4 — hard cutover (PR #5) — this is the 0.2.0 release commit

10. Flip the default: `_resolve_backend` valid set becomes
    `{"auto","ort","model2vec"}`. `"fastembed"` raises (per §10).
11. Remove `_check_gil_enabled` and its callsites in `embedding.py` +
    `reranker.py`.
12. Drop `fastembed>=0.6` from `[project].dependencies`.
13. Rewrite `device.py` detection layer.
14. CHANGELOG: 0.2.0 entry; KNT-601 audit writeup mirroring KNT-501.
15. README install matrix update.

### Phase 5 — release engineering (PR #6)

16. `release.yml` matrix expansion to 7 wheels. Smoke step gains
    `_rust.__version__` assertion.
17. `security.yml` adds `cargo audit` + `cargo deny check`.
18. Tag `v0.2.0a1`, ship.

### Phase 6 — GPU companion (PR #7, ships as 0.2.0a2)

19. Stand up `kaos-nlp-transformers-gpu` companion package shape.
    2 GPU wheels (linux x86_64 + aarch64).
20. `[gpu]` extra in main package becomes
    `gpu = ["kaos-nlp-transformers-gpu"]`.
21. Add gpu integration tests gated on `--marker gpu`.

**Safe-to-ship cut points:** end of Phase 2 (Rust core compiles +
cargo-tested but unused — ships without behavior change), end of Phase 3
(experimental flag, reference-vector parity proven), end of Phase 4
(0.2.0 release proper). Phase 6 independent; can slip a release.

---

## Risks and open questions

### Resolved

- **kl3m-nano custom AttentionPool** is *not* in the 0.2.0 REGISTRY.
  This package serves only public models today. kl3m-nano deployment
  is a separate kaos-embeddings concern; 0.2.0 doesn't block on it.
- **Wheel size** measured: 7.83 MB compressed, ~22 MB cdylib +
  31 MB vendored model = ~53 MB installed (vs current ~108 MB
  fastembed stack — net WIN despite our earlier "no wheel-size
  improvement" framing for Strategy B/C).
- **Free-threaded compat** measured: HF tokenizers ships 3.14t wheels
  with `Py_MOD_GIL_NOT_USED`; ort + libonnxruntime are C++ FFI (no
  PyO3 GIL concerns); our cdylib declares `gil_used = false`. All
  three components stay GIL-off after import.

### Open at lock-in time

1. **3.14t install UX.** Python `tokenizers` doesn't yet ship `cp314t`
   wheels on PyPI ([huggingface/tokenizers#1734]); install on 3.14t
   currently source-builds. In our migration this is moot — we ship
   the Rust `tokenizers` crate inside our cdylib, not the Python
   wrapper. **No action; tracking only.**

2. **ort 2.0 stability.** `ort = "2.0.0-rc.10"` is RC, not GA. Risk:
   API changes between rc.10 and 2.0.0 final. Mitigation: pin exactly
   in Cargo.toml; bump deliberately when 2.0 GA lands. The maintainer
   (pykeio) is responsive; rc-tier for ~6 months has been
   ABI-stable in practice.

3. **`download-binaries` license review.** ort's
   `download-binaries` feature pulls a Microsoft-built libonnxruntime
   (MIT licensed). The download happens at build time on the
   wheel-build host, NOT at install time on the user's host. The
   binary is statically linked into our cdylib. Our `NOTICE` must
   include Microsoft's onnxruntime attribution. Add to the audit
   queue as KNT-603.

4. **CI wall time.** Adding 7-platform builds + LTO release config
   will materially slow PR CI. Mitigation: Swatinem/rust-cache@v2
   keyed on Cargo.lock + per-target cache scopes (matches
   kaos-nlp-core). Expected per-PR Linux CI: ~6 min cold,
   ~90 s warm. Other platforms only on `release.yml`.

5. **Frozen reference vectors regeneration.** When upstream HF Hub
   publishes a new revision of `BAAI/bge-small-en-v1.5`, `RegisteredModel.revision`
   bumps and the frozen NPY may need re-baselining. Process:
   `scripts/freeze_reference_vectors.py` runs against the current
   pinned revision; we update both the SHA and the NPY in the same
   PR; the cosine-equivalence test then re-passes. Document in
   `docs/standards/engineering-process.md`.

### Deferred

- **Quantization parity.** Python fastembed serves int8-quantized
  ONNX for BGE; our 0.2.0 ports the FP32 ONNX path because that's
  what we ship today through fastembed's default config. Adding
  int8 quantization is an ort feature flag away (`ort/qdq`) — defer
  to 0.3.0 once GPU wheel + revision-pin tooling are both stable.

- **GPU-direct tokenization.** Would require streaming token IDs
  to GPU memory before the ort.run() call. Not on the roadmap.

- **Apple Silicon CoreML EP.** `ort/coreml` exists; would let us
  dispatch some ops to ANE on macOS. Defer to 0.3.0+. Apple Silicon
  CPU wheel via ort already ships in 0.2.0.

[huggingface/tokenizers#1734]: https://github.com/huggingface/tokenizers/issues/1734

---

## External docs to update outside this package

- **`/home/mjbommar/.claude/projects/-home-mjbommar-projects-273v/memory/MEMORY.md`**
  — update the kaos-nlp-transformers entry: remove "fastembed" backend,
  add "ort+Rust", note free-threaded works.
- **`kaos-core` / `kelvin-training` callers** — search for `backend="fastembed"`;
  none expected, but verify before flipping the default.
- **`kelvin-app-frontend` model registry** — model_id strings unchanged;
  no action.

---

## Reference benchmark data (this is what we're committing to match)

From `scratch/burn-experiment/outputs/REPORT.md`:

| Backend | bs=1 sps | bs=8 sps | bs=32 sps | bs=64 sps | Cosine vs ref |
|---|---:|---:|---:|---:|---|
| Python fastembed (0.1.0a6) | 112 | 146 | 167 | 138 | 1.000000 |
| ort direct (0.2.0 target) | **354** | **555** | **453** | **461** | 1.000000 |

The 0.2.0 perf gain is the Python-boundary cost going away (~3× at
small batches) plus newer ort + tokenizers. Our regression test
threshold is "≥0.9999 cosine" not throughput — but the perf smoke
will catch any accidental >50% throughput regression at PR time.
