# Agent Guidance

This file is the canonical cross-tool guidance for AI agents working in the
public `kaos-nlp-transformers` repository. The PyPI distribution is
`kaos-nlp-transformers`; the Python import package is
`kaos_nlp_transformers`.

## Repository Shape

- This is a typed Python 3.13+ package for dense embeddings, transformer-backed
  inference, retrieval, reranking, device detection, and optional MCP tools.
- The public API is the surface exported from `kaos_nlp_transformers.__all__`,
  documented README APIs, CLI entry points, settings fields, environment
  variables, schemas, and MCP tool behavior.
- The package is intentionally dependency-light at base install time:
  `fastembed`/ONNX, `numpy`, and KAOS runtime dependencies. PyTorch is not part
  of any in-tree runtime path.
- Optional integrations belong behind extras and lazy imports. Current extras
  include `clustering`, `gpu`, `openvino`, `model2vec`, `mcp`, and the
  deprecated no-op `torch` compatibility alias.

## Local References

Use the repository docs that already exist:

- [README.md](README.md) for package purpose, install examples, public concepts,
  CLI usage, compatibility, and development commands.
- [CONTRIBUTING.md](CONTRIBUTING.md) for setup, quality gates, DCO sign-off, PR
  expectations, changelog policy, and security reporting.
- [Python design and architecture](docs/standards/python-design-and-architecture.md)
  for public API, dependency, settings, error, async, file/path, CLI, and docs
  standards.
- [Code quality standards](docs/standards/code-quality-standards.md) for ruff,
  ty, pytest, dependency hygiene, and definition of done.
- [Engineering process](docs/standards/engineering-process.md) for issue, PR,
  commit, release, hotfix, and security handling.
- [Tests, fixtures, and CI](docs/standards/tests-fixtures-ci.md) for test tiers,
  fixtures, CI expectations, and release gates.
- [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) for PR
  checklist expectations.
- [SECURITY.md](SECURITY.md) for private vulnerability reporting.

## Design Principles

- Keep APIs stable, typed, and explicit. Public API changes need tests, docs,
  changelog consideration, and conservative release judgment.
- Prefer small composable modules over broad abstractions. Keep provider-specific
  logic in adapters instead of leaking backend details through public APIs.
- Make behavior deterministic by default: stable ordering, pinned model
  revisions, explicit device/backend selection, typed settings, and clear error
  paths.
- Keep import-time side effects out of package modules. Do not perform network
  calls, model downloads, filesystem scans, provider initialization, logging
  setup, or expensive model loads at import time.
- Put all heavyweight or environment-sensitive work behind explicit calls such
  as `load`, `from_*`, `connect`, CLI commands, or MCP tool handlers.
- Treat security and privacy as defaults. Do not expose credentials, local
  paths, provider payloads, cache contents, model artifacts, or user text in
  logs, exceptions, JSON output, or test fixtures unless the API explicitly
  requires it and tests cover the behavior.

## NLP And Model Rules

- Keep model registries license-reviewed and revision-pinned. Never add a model
  with revision `main`; use a concrete commit SHA and a compatible license.
- Respect `REGISTRY`, `RERANKER_REGISTRY`, `EXCLUDED`, and
  `RERANKER_EXCLUDED`. Do not bypass exclusion checks to make examples or tests
  pass.
- Keep embedding and reranking APIs stable:
  `EmbeddingModel.load`, `EmbeddingModel.embed`, `EmbeddingRetriever`,
  `CrossEncoderReranker.load`, `CrossEncoderReranker.rerank`,
  `KaosNLPTransformersSettings`, and `detect_devices` are user-facing.
- Preserve embedding shape and dtype contracts. Embeddings should remain
  deterministic `float32` numpy arrays with documented dimensions and
  normalization behavior.
- `fastembed` is the ONNX backend for embedding and cross-encoder reranking.
  `model2vec` is the static-numpy backend for registered static embedding
  models. Do not reintroduce PyTorch, `sentence-transformers`, or transformer
  imports into the base runtime path.
- `model2vec` and other optional dependencies must be imported lazily and fail
  with actionable package-extra guidance when missing.
- Device behavior must stay explicit and testable. Support documented values
  such as `auto`, `cpu`, `cuda`, `cuda:N`, and `openvino` only when the runtime
  can actually use them. Report latent devices without pretending they are
  reachable.
- Respect cache controls. `KaosNLPTransformersSettings.cache_dir`, `HF_HOME`,
  and backend cache keys must include the relevant model id, revision, device,
  backend, and cache directory so pinned updates do not reuse stale backends.
- Respect offline controls. `KaosNLPTransformersSettings.offline`,
  `KAOS_NLP_TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE`, and
  `TRANSFORMERS_OFFLINE` must prevent network/model download paths. Scoped
  environment changes must be restored after load attempts.

## Tests And Network Discipline

- Unit tests must be offline-friendly: no network, no credentials, no GPU
  requirement, no live Hugging Face downloads, and no dependence on a warm model
  cache.
- Use fakes, monkeypatching, vendored test fixtures, or the vendored
  `potion-base-8M` model path for offline coverage when possible.
- Tests that perform model downloads, live registry checks, public network
  access, or real backend inference outside the unit tier must be clearly marked
  with the appropriate existing pytest markers (`integration`, `live`, `gpu`) and
  skipped when offline settings are enabled.
- Do not make CI depend on local hardware, local caches, private services, or
  ambient credentials. GPU, OpenVINO, live network, and large model-download
  tests require explicit opt-in.
- Fixtures must be small, redistributable, provenance-documented, and free of
  customer data, secrets, privileged content, and PII.

## Configuration, Credentials, And Serving

- Use `KaosNLPTransformersSettings` for package configuration. Preserve the
  `KAOS_NLP_TRANSFORMERS_` environment prefix and documented legacy fallbacks.
- Keep secrets in secret-aware types such as `SecretStr` and redact them in logs,
  exceptions, CLI output, MCP output, and serialized settings.
- `kaos-nlp-transformers-serve --http` requires an explicit
  `KAOS_NLP_TRANSFORMERS_HTTP_TOKEN` operator acknowledgement. Do not weaken this
  guard or imply that the token replaces real reverse-proxy authentication.
- Path-accepting tools must resolve paths against the configured workspace root
  and reject traversal outside that root.

## Local Setup And Checks

Base setup:

```bash
uv sync --group dev --extra clustering
uvx pre-commit install
```

For normal code changes, run:

```bash
uv run ruff format --check kaos_nlp_transformers tests
uv run ruff check kaos_nlp_transformers tests
uv run ty check kaos_nlp_transformers tests
uv run pytest tests/unit -q --no-cov
```

For docs-only changes, run checks that match the edited files. At minimum,
validate links and scan for private/local references in the changed docs. If code
or examples are changed, run the normal quality gate too.

For packaging, metadata, README rendering, or release behavior changes, also
run:

```bash
uv build
uvx --from twine twine check --strict dist/*
```

For optional surfaces, install the relevant extras and keep tests opt-in:

```bash
uv sync --group dev --extra clustering --extra model2vec
uv sync --group dev --extra gpu
uv sync --group dev --extra openvino
uv sync --group dev --extra mcp
```

Run live, GPU, OpenVINO, or model-download tests only when the task explicitly
requires them and the environment is prepared for network access, hardware, and
cache writes.

## Change Discipline

- Keep changes focused and avoid unrelated rewrites.
- Match existing style, typing, error classes, settings patterns, and tests.
- Do not broaden dependency surfaces casually. New base dependencies require a
  strong reason, compatible license, tests, and documentation.
- Do not rely on undeclared transitive dependencies.
- Do not use private dependency APIs unless the risk is documented and covered
  by tests.
- Update README, docs, and `CHANGELOG.md` when user-visible behavior, public
  API, CLI behavior, package metadata, fixtures, or release artifacts change.
- Use conventional commits and sign commits with DCO (`git commit -s`) when
  committing.

## Agents Must Not

- Do not commit secrets, tokens, private keys, `.env` files, customer data, PII,
  credentials, or privileged documents.
- Do not commit model caches, downloaded Hugging Face snapshots, ONNX Runtime
  caches, generated build outputs, virtual environments, coverage output, local
  tool state, or unrelated local files.
- Do not commit large generated artifacts unless they are intentionally shipped
  package assets with provenance, licensing, tests, and release-size review.
- Do not perform live network, GPU, OpenVINO, model-download, or credentialed
  tests unless explicitly opted in for the task.
- Do not add non-commercial, no-derivatives, GPL, AGPL, unknown-license, or
  ambiguous-license dependencies, models, or redistributed fixtures.
- Do not change public defaults, model revisions, device selection,
  offline/cache behavior, or error shapes casually.
- Do not hide missing optional dependencies by importing them at module import
  time or adding them to the base dependency set without review.
