# Repository Agent Guidance

## Scope

This file is the canonical instruction file for coding agents working in
this repository. It applies to the whole repository unless a more
specific `AGENTS.md` is added in a subdirectory.

Keep agent-driven changes focused on the requested task. Preserve
existing user changes, avoid unrelated cleanup, and do not edit
generated files unless the requested change explicitly requires
regenerating them.

For contributor process, see [CONTRIBUTING.md](CONTRIBUTING.md). For
the durable engineering standards, link to these files instead of
duplicating their contents:

- [Python design and architecture](docs/standards/python-design-and-architecture.md)
- [Code quality standards](docs/standards/code-quality-standards.md)
- [Engineering process](docs/standards/engineering-process.md)
- [Tests, fixtures, and CI](docs/standards/tests-fixtures-ci.md)
- [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md)
- [SECURITY.md](SECURITY.md)

## Project Identity

- Distribution: `kaos-nlp-transformers`
- Import package: `kaos_nlp_transformers`
- Runtime: Python 3.13+ (free-threaded 3.13t / 3.14t supported)
- Package type: hybrid Rust + Python via PyO3 / maturin. abi3-py313
  wheels per OS/arch.
- Primary surface (Phase-5 retrieval stack):
  `EmbeddingModel.load` + `.embed` (dense embeddings),
  `CrossEncoderReranker.load` + `.rerank`, `EmbeddingRetriever`,
  `SemanticChunker`, `ExtractiveRanker`.
- Primary surface (Phase-8 small-model inference):
  `NliModel.load` + `.score(premise, hypotheses)` (NLI cross-encoder,
  satisfies `kaos_llm_core.programs.classify.NLIScorer` Protocol);
  `GLiNERExtractor.load` + `.extract(texts, labels)` (zero-shot NER
  via prompt-based span scoring); `PiiDetector.load` + `.detect(texts)`
  (closed-label BERT-small token classifier over 24 PII categories,
  ~17x faster than GLiNER at the closed-label task). All three share
  the `Entity` dataclass for span output.
- Shared admin surface: `detect_devices()`, `KaosNLPTransformersSettings`,
  and the curated model registries — `REGISTRY` / `EXCLUDED` (embedding),
  `RERANKER_REGISTRY` / `RERANKER_EXCLUDED`, `NLI_REGISTRY` /
  `NLI_EXCLUDED`, `NER_REGISTRY` / `NER_EXCLUDED`, `PII_REGISTRY` /
  `PII_EXCLUDED`.
- CLI surface: `kaos-nlp-transformers info` (diagnostic envelope),
  `kaos-nlp-transformers prefetch` (cache-warming with
  `--include {embedding,reranker,nli,ner,pii}` / `--model <id>` /
  `--dry-run` / `--json` / `--quiet`),
  `kaos-nlp-transformers-serve` (MCP server, requires `[mcp]`).

Runtime shape (KNT-601, 0.2.0+): embedding and cross-encoder inference
goes through an in-tree Rust cdylib
(`kaos_nlp_transformers._rust`) that calls libonnxruntime via the
[`ort`](https://github.com/pykeio/ort) crate, with HuggingFace
tokenizers (Rust) doing tokenization and `hf-hub` (Rust) handling
revision-pinned snapshot downloads. libonnxruntime is statically linked
into `_rust.abi3.so`; wheels carry no Python `onnxruntime` runtime
dep. PyTorch is not part of any in-tree runtime path.

Treat the public API exported from `kaos_nlp_transformers.__all__`,
documented README APIs, CLI entry points (`kaos-nlp-transformers`,
`kaos-nlp-transformers-serve`), settings fields, environment
variables, schemas, and MCP tool behavior as public contracts.

The package is intentionally dependency-light at base install:
`numpy`, `huggingface_hub`, and the KAOS runtime. The Rust-side
dependencies (ort, tokenizers, hf-hub, pyo3, ndarray, …) live in
`Cargo.toml` and ship bundled inside the cdylib at wheel-build time.

Current optional extras: `model2vec`, `mcp`. The `gpu` and `openvino`
extras are reserved-and-no-op aliases today; GPU acceleration ships
as a separate `kaos-nlp-transformers-gpu` companion wheel built with
`cargo build --features gpu` (ort/cuda EP). The `torch` extra is a
deprecated no-op alias scheduled for removal in 0.3.0. KNT-602
(0.2.0a3) retired the previous `clustering` extra; the
`SemanticDedupLevel` moved to `kaos-content[clustering]`.

## Setup

Use `uv` for environments, dependency resolution, builds, and tool
execution. Cargo + maturin for the Rust extension.

```bash
uv sync --group dev --extra model2vec
uv run maturin develop --release
uvx pre-commit install
```

The `maturin develop --release` step compiles `_rust.abi3.so` into
the active venv's site-packages; without it `from kaos_nlp_transformers
import EmbeddingModel` fails (the Python wrappers reach into
`_rust.embedding`, `_rust.reranker`, `_rust.registry`). Re-run after
any change under `rust/`. `ty`'s import resolution also depends on
the cdylib being present.

## Local Checks

Run the narrowest useful checks while developing, then run the
relevant gate before handing off:

```bash
uv run ruff format --check kaos_nlp_transformers tests
uv run ruff check kaos_nlp_transformers tests
uv run ty check kaos_nlp_transformers tests
uv run pytest tests/unit -q --no-cov
```

Use `ty`, not mypy. Inline suppressions use `# ty: ignore[...]` with
the narrowest applicable rule and a reason when the reason is not
obvious.

For Rust / PyO3 changes (anything under `rust/`), additionally run:

```bash
cargo fmt --check
cargo clippy --release --lib -- -D warnings
cargo test --release --lib
uv run maturin develop --release   # rebuild before pytest picks up the change
```

When packaging, metadata, README rendering, or release behavior
changes, also run:

```bash
cargo audit
cargo deny check
uv build
uvx --from twine twine check --strict dist/*
```

For optional surfaces, install the relevant extras and keep tests
opt-in:

```bash
uv sync --group dev --extra model2vec --extra mcp
```

The `[gpu]` and `[openvino]` extras are reserved aliases (no-op
today; companion-wheel pattern). To exercise the CUDA path locally,
build a GPU wheel with `cargo build --release --features gpu` and
install that wheel; do NOT add `onnxruntime-gpu` as a Python dep.

Live, GPU, OpenVINO, or model-download tests are opt-in and require
the environment to be prepared for network access, hardware, and
cache writes.

## Architecture Rules

- Keep the public API stable, typed, and explicit. `kaos_nlp_transformers.__all__`,
  documented classes and functions, CLI flags and JSON output, MCP
  tools and schemas, environment variables, and documented
  configuration are public contracts.
- Keep import-time work cheap. No network calls, model downloads,
  provider initialization, filesystem scans, logging setup, or
  expensive model loads at import time. Heavyweight work belongs
  behind explicit calls (`load`, `from_*`, CLI commands, MCP tool
  handlers).
- Prefer small composable modules over broad abstractions. Keep
  backend-specific logic in adapters so backend details do not leak
  through public APIs.
- Make behavior deterministic by default: stable ordering, pinned
  model revisions, explicit device/backend selection, typed settings,
  clear error paths.
- Keep optional dependencies behind extras and lazy imports. Optional
  extras fail with actionable package-extra guidance when the missing
  dep is hit.
- Keep public surfaces aligned with the documented architectural
  layers — the PyO3 binding layer (`rust/bindings/`) should stay thin
  (input conversion, error mapping, GIL release); domain logic
  (model load / pooling / tokenization / session execution) belongs
  in the Rust core (`rust/core/`).
- Preserve `py.typed` and typed public boundaries.

## NLP And Model Rules

- Keep the model registry license-reviewed and revision-pinned. Never
  add a model with revision `main`; use a concrete commit SHA and a
  compatible license.
- Respect every registry's exclusion list: `REGISTRY` / `EXCLUDED`
  (embedding), `RERANKER_REGISTRY` / `RERANKER_EXCLUDED`,
  `NLI_REGISTRY` / `NLI_EXCLUDED`, `NER_REGISTRY` / `NER_EXCLUDED`,
  `PII_REGISTRY` / `PII_EXCLUDED`. Do not bypass exclusion checks to make
  examples or tests pass.
- Keep all inference APIs stable: `EmbeddingModel.{load,embed,count_tokens,max_seq_len}`,
  `CrossEncoderReranker.{load,rerank}`, `NliModel.{load,score}`,
  `GLiNERExtractor.{load,extract}`, `PiiDetector.{load,detect,labels}`,
  `KaosNLPTransformersSettings`, and `detect_devices` are user-facing.
  `EmbeddingRetriever` is
  deprecated as of 0.2.0 (DeprecationWarning at construction) and
  scheduled for removal in 0.3.0; downstream callers should migrate
  to `kaos_content.indexing.SearchableDocument` /
  `kaos_content.indexing.SearchableCorpus`.
- Preserve embedding shape and dtype contracts. Embeddings remain
  deterministic `float32` numpy arrays with documented dimensions
  and normalization behavior.
- The Rust `ort` cdylib is the canonical inference backend for
  transformer-family models. `model2vec` is the static-numpy
  backend for registered static embedding models. Do not reintroduce
  `fastembed`, the Python `onnxruntime` wrapper,
  `sentence-transformers`, or PyTorch into the base runtime path.
- Device behavior must stay explicit and testable. Support
  documented values (`auto`, `cpu`, `cuda`, `cuda:N`, `openvino`)
  only when the runtime can actually use them — the cdylib's
  `_rust.registry.capabilities()` is the source of truth (`cuda` /
  `openvino` are True only when the wheel was built with
  `--features gpu` / `--features openvino`). Report latent devices
  without pretending they are reachable.
- Respect cache controls. `KaosNLPTransformersSettings.cache_dir`,
  `HF_HOME`, and backend cache keys must include the relevant model
  id, revision, device, backend, and cache directory so pinned
  updates do not reuse stale backends.
- Respect offline controls. `KaosNLPTransformersSettings.offline`,
  `KAOS_NLP_TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE`, and
  `TRANSFORMERS_OFFLINE` must prevent network/model-download paths.
  Scoped environment changes must be restored after load attempts.

## Testing

- New public API or behavior needs tests through the real entry
  point.
- Bug fixes need regression tests.
- Security-sensitive behavior needs accepted and rejected cases with
  realistic inputs.
- Unit tests must be offline-friendly: no network, no credentials,
  no GPU requirement, no live Hugging Face downloads, no dependence
  on a warm model cache.
- Use fakes, monkeypatching, vendored test fixtures, or the
  vendored `potion-base-8M` model path for offline coverage when
  possible.
- Tests that perform model downloads, live registry checks, public
  network access, or real backend inference outside the unit tier
  must be marked with the existing pytest markers (`integration`,
  `live`, `gpu`) and skipped when offline settings are enabled.
- Do not make CI depend on local hardware, local caches, private
  services, or ambient credentials. GPU, OpenVINO, live network, and
  large model-download tests require explicit opt-in.
- Fixtures must be small, redistributable, provenance-documented,
  and free of customer data, secrets, privileged content, and PII.

## Security

- Never commit secrets, tokens, private keys, `.env` files, customer
  data, PII, credentials, or privileged documents.
- Do not commit model caches, downloaded Hugging Face snapshots,
  ONNX Runtime caches, generated build outputs, virtual environments,
  coverage output, local tool state, or unrelated local files.
- Do not commit large generated artifacts unless they are
  intentionally shipped package assets with provenance, licensing,
  tests, and release-size review.
- Use `KaosNLPTransformersSettings` for package configuration.
  Preserve the `KAOS_NLP_TRANSFORMERS_` environment prefix and
  documented legacy fallbacks. Keep secrets in secret-aware types
  such as `SecretStr` and redact them in logs, exceptions, CLI
  output, MCP output, and serialized settings.
- `kaos-nlp-transformers-serve --http` requires an explicit
  `KAOS_NLP_TRANSFORMERS_HTTP_TOKEN` operator acknowledgement. Do
  not weaken this guard or imply that the token replaces real
  reverse-proxy authentication.
- Path-accepting tools must resolve paths against the configured
  workspace root and reject traversal outside that root.
- Do not add non-commercial, no-derivatives, GPL, AGPL,
  unknown-license, or ambiguous-license dependencies, models, or
  redistributed fixtures.
- Report suspected vulnerabilities through [SECURITY.md](SECURITY.md),
  not public issues.

## Commits, PRs, And Releases

- Use conventional commit style and sign commits with `git commit -s`
  for the Developer Certificate of Origin.
- Keep one logical change per PR. Match existing style, typing, error
  classes, settings patterns, and tests.
- Do not broaden dependency surfaces casually. New base dependencies
  require a strong reason, compatible license, tests, and
  documentation.
- Do not rely on undeclared transitive dependencies. Do not use
  private dependency APIs unless the risk is documented and covered
  by tests.
- PR descriptions should state what changed, why, how it was tested,
  and whether public API, CLI behavior, MCP schemas, package
  metadata, fixtures, or release artifacts changed.
- User-visible changes need a `CHANGELOG.md` entry under
  `[Unreleased]`.
- Do not change public defaults, model revisions, device selection,
  or offline/cache behavior casually.
- Do not perform live network, GPU, OpenVINO, model-download, or
  credentialed tests in PRs unless explicitly opted in.
- Do not hide missing optional dependencies by importing them at
  module import time or adding them to the base dependency set
  without review.
- Do not move public release tags or force-push shared branches.
