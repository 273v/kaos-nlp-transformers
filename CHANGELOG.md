# Changelog

All notable changes to `kaos-nlp-transformers` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **CI: ``build + wheel smoke test`` shouldn't import from the
  workspace cwd.** The smoke step ran ``python -c "..."`` with the
  workflow's default cwd (the workspace, which contains the
  ``kaos_nlp_transformers/`` source tree). Python prepends
  ``sys.path[0] = cwd`` by default, so ``from
  kaos_nlp_transformers._rust.embedding import EmbeddingBackend``
  (called transitively from ``EmbeddingModel.load``) resolved
  against the *source* tree — which ships ``_rust/*.pyi`` stubs but
  NO ``_rust.abi3.so`` (the cdylib only exists in the wheel install)
  — and raised ``ModuleNotFoundError: No module named
  'kaos_nlp_transformers._rust.embedding'``. The wheel was correct;
  the test environment was wrong.

  Fix: set ``PYTHONSAFEPATH=1`` (Python 3.11+) on the step env so
  Python doesn't prepend cwd to sys.path, and ``cd /tmp/smoke``
  before the python probe so we're not even sitting in the source
  tree. Belt-and-suspenders.

  Also aligned with the ``release.yml`` pattern documented in the
  0.2.0a2 CHANGELOG: the smoke no longer reaches into
  ``_rust.<submodule>`` from a one-liner (that pattern is fragile
  even with PYTHONSAFEPATH because of the ``_rust.abi3.so`` vs
  ``_rust/`` namespace-package ambiguity on the wheel-install
  layout). The full ``_rust.<submodule>`` direct-import chain stays
  covered by every test-matrix leg via ``maturin develop`` +
  ``tests/unit/test_rust_extension.py``.

  Wiped ``/tmp/smoke`` before ``uv venv`` and added ``--reinstall``
  to ``uv pip install`` so the self-hosted runner doesn't reuse
  stale state.

  Files: ``.github/workflows/ci.yml``.

## [0.2.0a3] — 2026-05-10 — KNT-602 boundary fix (drop kaos-content dep)

KNT-602 Option A: restores the documented layer cake. kaos-content is
the consumer of kaos-nlp-transformers; never the inverse. The
``SemanticDedupLevel`` + the ``kaos-nlp-transformers-dedup-semantic``
MCP tool moved to kaos-content (released as 0.1.0a3). This package
now ships the embedding / reranker / device primitives only.

### Breaking changes

- **MCP tool removed**: ``kaos-nlp-transformers-dedup-semantic`` is no
  longer registered by ``register_transformers_tools``. **No
  deprecation cycle.** The replacement is
  ``kaos-content-dedup-semantic`` in kaos-content 0.1.0a3+ (mirror of
  the kaos-content CHANGELOG breaking-change entry). Rationale: a
  one-cycle deprecation shim would have to import
  ``SemanticDedupLevel`` from kaos-content, re-introducing the
  dep-cycle the move is fixing. The package is in the pre-1.0 alpha
  series (0.2.0a*), under which the cross-monorepo standards
  (``kaos-modules-auth/docs/oss/20-python-packaging/public-api-discipline.md``)
  permit breaking changes when documented. Downstream agents calling
  the old name should switch to ``kaos-content-dedup-semantic`` and
  install ``kaos-content[transformers,clustering]`` to enable the
  level. Existing code paths that imported
  ``kaos_nlp_transformers.clustering.SemanticDedupLevel`` should
  re-import from ``kaos_content.dedup.levels.semantic``.
- **Module removed**: ``kaos_nlp_transformers.clustering`` package
  is deleted. ``import kaos_nlp_transformers.clustering`` now raises
  ``ImportError``. ``test_audit_07.py`` pins the regression that the
  module stays gone.

### Removed

- **kaos-content from base dependencies** in ``pyproject.toml``. Pre-
  KNT-602 the package shipped ``kaos-content>=0.1.0a1`` as a base
  runtime dep (used only by the dedup level + MCP tool). Both moved
  out, so the dep is removed entirely from ``[project.dependencies]``.
  Downstream consumers of the base install no longer pull
  ``kaos-content`` and its transitive ``kaos-core`` / ``pydantic``
  surface unless they request it.
- **`[clustering]` extra** (``scipy>=1.14.1``). ``scipy`` moved with
  ``SemanticDedupLevel`` to ``kaos-content[clustering]``; that's now
  the canonical home.
- AGENTS.md / CHANGELOG / settings.py docstrings refreshed to drop
  references to the retired ``[clustering]`` extra and the
  in-package ``SemanticDedupLevel`` location.

### Added

- **`tests/unit/test_audit_07.py`** — regression tests pinning the
  KNT-602 boundary fix: no ``kaos_content`` imports anywhere in the
  package source, no ``clustering`` submodule importable, no
  ``kaos-nlp-transformers-dedup-semantic`` tool registered, no
  ``kaos-content`` dep in ``[project].dependencies``. Mirrors the
  KNT-001 ``test_no_kaos_ml_core_import_anywhere`` pattern.

## [0.2.0a2] — 2026-05-09 — release-pipeline fixes (no API change)

Re-roll of the 0.2.0a1 alpha; nothing landed on PyPI for 0.2.0a1 because
the wheel matrix tripped on two infrastructural issues that publish-pypi's
``needs: [sdist, wheels]`` gate correctly caught.

### Fixed

- **Drop musllinux from the wheel matrix.** The ``ort`` Rust crate
  uses ``download-binaries`` to fetch Microsoft's official
  ``libonnxruntime``, but Microsoft only publishes manylinux2014
  variants — there is no musllinux build to download. Both
  ``x86_64-unknown-linux-musl`` and ``aarch64-unknown-linux-musl``
  builds failed for this reason. Alpine / musl users will need a
  source-build wheel; tracking that as a 0.2.x follow-up.
- **Simplify the release-job smoke.** The previous smoke reached into
  ``kaos_nlp_transformers._rust.registry`` directly. On the manylinux
  CI runner this triggered a CPython namespace-package resolution
  ambiguity (the wheel ships ``_rust/*.pyi`` stubs alongside the
  cdylib at ``_rust.abi3.so``; on some Python builds the directory
  shadows the .so). The new smoke imports the public surface
  (``EmbeddingModel``, ``REGISTRY``, ``detect_devices``), which
  exercises the cdylib transitively without forcing the
  ambiguous-import path. The full ``_rust.<submodule>`` import chain
  is still gated by every ci.yml test-matrix leg via
  ``maturin develop`` + pytest.

## [0.2.0a1] — 2026-05-09 — KNT-601 Rust backend cutover

Audit-07 release. The Python ``fastembed`` wrapper is **retired
entirely**; embedding and reranker inference now go through an in-tree
Rust cdylib (``kaos_nlp_transformers._rust``) that calls
libonnxruntime via [ort](https://github.com/pykeio/ort). Same ONNX
models, same outputs (cosine ≥ 0.9999 vs frozen reference vectors),
but free-threaded Python compatible and one fewer Python boundary in
the inference path. Detailed plan:
[docs/MIGRATION_0_2_0.md](docs/MIGRATION_0_2_0.md).

### Removed

- **KNT-601 (HIGH) — fastembed Python wrapper retired.** The
  ``fastembed`` Python dep is gone, along with its transitive
  ``onnxruntime``, ``tokenizers`` (Python wrapper), and
  ``py_rust_stemmers``. Inference goes through the Rust cdylib's
  ``EmbeddingBackend`` / ``CrossEncoderBackend`` (ort + libonnxruntime
  + tokenizers Rust crate, all statically linked). Model coverage
  unchanged — ``BAAI/bge-small-en-v1.5`` (embedding) and
  ``BAAI/bge-reranker-base`` (reranker) load from the same pinned
  HF Hub revisions; outputs are bit-equivalent. Audit-01 KNT-003
  (revision pinning) is now correct by construction — the Rust loader
  passes ``revision`` to ``hf-hub`` explicitly, where fastembed used
  release-baked SHAs.
- **Audit-03 KNT-201 free-threaded guard removed.** The
  ``_check_gil_enabled`` refusal at ``EmbeddingModel.load`` /
  ``CrossEncoderReranker.load`` is gone. The Rust cdylib declares
  ``gil_used = false`` (audit KNT-602), and the Rust ``tokenizers``
  crate (statically linked into the cdylib) doesn't have the
  ``py_rust_stemmers`` SIGSEGV path. Free-threaded Python (3.13t /
  3.14t) loads cleanly.
- ``_load_fastembed_cached``, ``_load_cross_encoder_cached`` (both
  rewritten to call the Rust backend), and ``_onnx_providers_for_device``
  (replaced by Rust-side EP gating).

### Changed

- **Build backend: ``hatchling`` → ``maturin>=1.8``.** Per-platform
  abi3 wheels (cp313-abi3) for Linux x86_64 / aarch64 (manylinux +
  musllinux), macOS aarch64, Windows x86_64 / aarch64.
- ``EmbeddingModel.backend_name`` returns ``"ort"`` (was
  ``"fastembed"``) for the default registry path. ``"model2vec"``
  unchanged.
- ``RegisteredModel.backend`` valid set narrowed to ``{"ort", "model2vec"}``.
- ``KaosNLPTransformersSettings.backend`` valid set narrowed to
  ``{"auto", "ort", "model2vec"}``. Legacy values (``"fastembed"``,
  ``"sentence-transformers"``) raise ``ValueError`` with a migration
  message pointing at the Rust path.
- ``device.detect_devices()`` and ``device.SystemDevices`` now read
  the cdylib's compile-time capability flags via
  ``_rust.registry.capabilities()`` instead of asking the Python
  ``onnxruntime`` package for execution providers. The ``[gpu]``
  install hint flips from "install ``onnxruntime-gpu``" to
  "install the ``kaos-nlp-transformers-gpu`` companion wheel".
- Version source is now ``Cargo.toml [package].version``. Python
  ``__version__`` reads from installed package metadata
  (``importlib.metadata``); editable builds fall back to the
  cdylib's Cargo SemVer string.

### Added

- **``EmbeddingModel.embed(texts: Iterable[str])``.** Accepts any
  iterable (was ``list[str]`` only). Generators / lazy stream
  consumers can pass through without an explicit ``list()`` step.
- **``EmbeddingModel.max_seq_len: int``.** Surfaces the underlying
  tokenizer's truncation cap. Downstream chunkers
  (``kaos_content.chunking.EmbeddingChunker``) read this so chunks
  don't silently truncate at embed time.
- **``EmbeddingModel.count_tokens(texts) -> list[int]``.** Tokenizes
  without running inference; returns per-text non-pad token count.
- **Process-wide embedding cache (opt-in).** New setting
  ``KaosNLPTransformersSettings.embedding_cache_size`` (default 0 =
  off). When non-zero, an LRU keyed on
  ``(model_id, revision, blake2b(text))`` short-circuits repeated
  embedding requests. ~15 MB at 10K entries × 384-dim.
- **``[gpu]`` and ``[openvino]`` extras** preserved as pyproject keys
  for one release cycle. The 0.2.0a1 wheel is CPU-only; the 0.2.0a2
  release introduces a ``kaos-nlp-transformers-gpu`` companion
  package built with ``--features gpu`` (ort/cuda EP).
- ``deny.toml`` for cargo-deny supply-chain checks (license
  allowlist, advisory ignore list, multi-version warning).
- ``tests/reference/*.npy`` — frozen reference embeddings for the
  bit-equivalence regression test
  (``tests/unit/test_reference_vectors.py``). Per-row cosine ≥
  0.9999 vs the 0.1.0a6 fastembed output is the migration contract.

### Deprecated

- ``EmbeddingRetriever`` (text-only dense retriever). Use
  ``kaos_content.indexing.SearchableDocument(retrieval="embeddings")``
  for AST-grounded single-document retrieval, or the upcoming
  ``kaos_content.indexing.SearchableCorpus`` for cross-document
  retrieval. Both preserve ``block_ref`` / ``page`` / ``section_ref``
  provenance. Removal scheduled for 0.3.0; emits ``DeprecationWarning``
  in 0.2.0.

## [0.1.0a6] — 2026-05-08

Audit-06 release. One finding (KNT-501) closed: **PyTorch and
sentence-transformers are removed from the package entirely.** The cross-
encoder reranker now runs through `fastembed.TextCrossEncoder` (ONNX),
the same runtime as embedding does. Install footprint drops by ~1.4 GB.

### Removed

- **KNT-501 (HIGH) — torch + sentence-transformers backend retired.**
  Pre-0.1.0a6 the optional `[torch]` extra pulled in `torch`,
  `transformers`, and `sentence-transformers` (~1.4 GB) to power the
  cross-encoder reranker and any embedding model fastembed didn't
  natively support. Both surfaces now go through ONNX:
  - **Reranker:** `CrossEncoderReranker` was rewritten on top of
    `fastembed.rerank.cross_encoder.TextCrossEncoder`. Same model
    (`BAAI/bge-reranker-base`, MIT, pinned to the same SHA), same
    sigmoid-normalized `[0, 1]` scoring contract, same async-thread
    dispatch. The `_load_cross_encoder_cached` helper switched its
    backend import accordingly.
  - **Embedding:** the `sentence-transformers` branch in
    `EmbeddingModel.load` and `_resolve_backend` was deleted along with
    `_load_sentence_transformers_cached`. The valid backend set is now
    `{"auto", "fastembed", "model2vec"}`. GPU embedding goes through
    fastembed + onnxruntime-gpu via the existing
    `_onnx_providers_for_device` helper.
  - **Settings:** `KaosNLPTransformersSettings.backend` and
    `KaosNLPTransformersSettings.device` no longer accept
    `"sentence-transformers"`, `"mps"`, or `"xla"`. These raise
    `ValueError("Invalid backend ...")` at the resolve step rather than
    silently falling through. (Audit-02 KNT-107's strict-validation
    contract carries forward.)
  - **Device probe:** `_detect_torch_devices` was deleted. Reachable
    GPUs come from `_detect_reachable_gpus(onnx_providers)`, which
    cross-references `nvidia-smi` against
    `CUDAExecutionProvider`. `LatentDevice.install_extra` for unreachable
    NVIDIA / ROCm GPUs is now `"gpu"` (not `"torch"`).

  Test pins (audit-06): the live reranker integration suite
  (`tests/integration/test_reranker_live.py`, 8 tests) hits the real
  `fastembed.TextCrossEncoder` end-to-end — no mocks. Existing audit-01
  / audit-02 tests were updated to the new loader / valid-backend set.

### Changed

- **`[torch]` extra is now a no-op alias.** `pip install
  kaos-nlp-transformers[torch]` still resolves so existing CI and
  lockfiles don't error out, but it pulls in **zero** extra packages.
  New code should use `[gpu]` for CUDA acceleration. The alias is
  removed entirely in 0.3.0.
- **`RegisteredModel.backend` valid values** narrowed to
  `{"fastembed", "model2vec"}`. The reranker registry's only entry
  (`BAAI/bge-reranker-base`) flipped from `"sentence-transformers"` to
  `"fastembed"` to match the new loader.
- **`DeviceInfo.device`** no longer accepts `"mps"` / `"xla"`. The
  detector and resolver were trimmed accordingly.
- **`resolve_device` install hint** now says `[gpu]`, not `[torch]`,
  for unreachable CUDA devices.
- **Module docstrings + tool descriptions** updated across
  `__init__.py`, `errors.py`, `models.py`, `device.py`, `tools.py`,
  `embedding.py`, `reranker.py`, `settings.py` to reflect the
  fastembed-only inference path. Historical references to the retired
  surface are tagged `audit-06 KNT-501` for traceability.

### Why

PyTorch is the dominant Python install-size cost for the package and
the only reason `[torch]` existed was to power the cross-encoder
reranker (and a hypothetical "GPU = sentence-transformers" routing
that no real workload depended on). fastembed natively supports
`BAAI/bge-reranker-base` via `TextCrossEncoder` — same model, same
ONNX runtime as embedding. Routing both through the same backend
collapses the install matrix, shrinks the dependency tree by ~1.4 GB,
and removes the GIL-incompatible `transformers` chain from the
free-threaded-Python guard's reasoning.

The custom-model story is unchanged in shape: anyone shipping their
own sentence-transformers-trained model can convert it to ONNX +
register it via `RegisteredModel(backend="fastembed",
allow_unregistered=True)`. See `docs/CUSTOM_MODELS.md` (planned for
0.1.0b1) for the conversion recipe.

[0.1.0a6]: https://github.com/273v/kaos-nlp-transformers/compare/v0.1.0a5...v0.1.0a6

## [0.1.0a5] — 2026-05-08

### Added

- **Audit-05 KNT-401 — bundle `minishlab/potion-base-8M` inside the
  wheel.** A small static embedding model (~31 MB safetensors + tokenizer,
  pinned at SHA `bf8b056651a2c21b8d2565580b8569da283cab23`, MIT license)
  now ships under `kaos_nlp_transformers/_vendor/potion-base-8M/`. The
  loader checks the vendored path before calling
  `huggingface_hub.snapshot_download`, so:
  - `EmbeddingModel.load("minishlab/potion-base-8M")` works **offline /
    air-gapped** out of the box once `[model2vec]` is installed.
  - `HF_HUB_OFFLINE=1` no longer breaks this one model.
  - First-call latency drops from ~5s (download) to ~50 ms (filesystem).

  Wheel size: ~28 MB (was ~2 MB). This is a deliberate trade for
  offline-first deployments. Other model2vec models still download from
  HF on first use.

  Three new tests in `tests/integration/test_embed_model2vec.py`:
  - `test_vendored_path_detected_for_potion_base_8m` — directory probe
  - `test_vendored_path_returns_none_for_unvendored_models` — fallthrough guard
  - `test_vendored_path_loads_without_network` — `HF_HUB_OFFLINE=1` regression

  Loader emits an INFO line tagged `audit-05 KNT-401` recording which
  resolution path it took (vendored vs HF Hub), so an operator can
  audit-trace the source.

  REGISTRY entry added for `minishlab/potion-base-8M`: 8M params, 256-dim
  (PCA-reduced), MIT, backend=`model2vec`.

### Changed

- `_load_model2vec_cached` now resolves in two stages: vendored
  filesystem probe first (no network), HF Hub `snapshot_download`
  fallback. Existing `model2vec` callers see no API change; only the
  resolution order is updated.

## [0.1.0a3] — 2026-05-08

Hot-fix release for a hard SIGSEGV on free-threaded Python (3.13t / 3.14t).
One audit-03 finding (KNT-201) closed with five regression tests pinning
the runtime guard.

### Security / Correctness

- **KNT-201 (HIGH) — runtime guard against free-threaded Python.**
  ``import fastembed`` triggers SIGSEGV (exit 139) inside
  ``py_rust_stemmers``' module init under ``Py_GIL_DISABLED`` because the
  upstream Rust/PyO3 extension hasn't declared free-threaded support.
  ``EmbeddingModel.load`` and ``CrossEncoderReranker.load`` now check
  ``sys._is_gil_enabled()`` BEFORE attempting any backend import; if the
  interpreter is free-threaded they raise
  ``BackendNotInstalledError`` with a fix-and-track message instead of
  letting the segfault happen.

  **User-visible behavior change.** Code that previously crashed silently
  on Python 3.14t now raises a clean ``BackendNotInstalledError`` with:

  - what's wrong (fastembed's transitive py_rust_stemmers crashes
    under Py_GIL_DISABLED; sentence-transformers chain similarly
    affected via tokenizers + transformers)
  - how to fix (use the GIL-enabled build of Python 3.13 or 3.14, NOT
    the 3.14t free-threaded variant)
  - alternative (track upstream py_rust_stemmers / tokenizers
    free-threaded support; this guard is removed once those wheels
    declare ``Py_GIL_DISABLED``)

  Test pins:
  ``test_check_gil_enabled_passes_on_normal_build``,
  ``test_check_gil_enabled_refuses_on_free_threaded``,
  ``test_embedding_model_load_calls_the_guard``,
  ``test_cross_encoder_reranker_load_calls_the_guard``,
  ``test_is_free_threaded_python_helper``.

### Changed

- **Compatibility & status table** in README — Python 3.14t row flipped
  from "informational" to **NOT supported pending upstream**, with a
  link to the tracker. Same for any future ``Py_GIL_DISABLED`` build
  (3.13t, 3.15t).

[0.1.0a3]: https://github.com/273v/kaos-nlp-transformers/compare/v0.1.0a2...v0.1.0a3

## [0.1.0a2] — 2026-05-08

Audit-02 follow-up release. Seven findings (KNT-101..KNT-107) closed, all
with regression tests pinned in `tests/unit/test_audit_02.py` (24 new tests
covering normalization, validation, scoped offline, reranker governance,
semantic-dedup similarity, and backend strictness).

### Security / Correctness

- **KNT-101 (HIGH) — `EmbeddingModel.embed` enforces L2 normalization.**
  PRD §4 + §10 + the README all promised L2-normalized output, but the
  0.1.0a1 implementation only cast backend output to `float32`. fastembed +
  BGE happens to produce unit-norm vectors so direct cosine-via-dot-product
  consumers got correct scores in practice — but the contract was unenforced
  and a future registry entry (or the sentence-transformers / `[torch]`
  path) would silently violate it. Fix: pass
  `normalize_embeddings=True` to `SentenceTransformer.encode`, then apply
  an explicit `_l2_normalize` to the final array regardless of backend.
  All-zero rows return as zeros (no NaN). Cost: one `np.linalg.norm` +
  division per call (~1µs per 384-dim row), far below inference cost.
  **User-visible behavior change:** anyone consuming raw embedding magnitudes
  for non-cosine purposes (rare) will see a unit-norm result. Cosine
  consumers are unaffected. Test pin:
  `test_embed_returns_unit_norm_rows_for_arbitrary_backend_output`.
- **KNT-102 (HIGH) — `EmbeddingRetriever` input validation.** The 0.1.0a1
  constructor and `add_documents` validated `doc_ids` and `texts` lengths
  but not `external_ids` or `metadata_list` — a length mismatch silently
  corrupted the retriever and surfaced as wrong retrieval results downstream.
  Fix: extracted `_validate_parallel_lengths()` helper applied in BOTH
  `__init__` and `add_documents` BEFORE any internal-state mutation.
  Empty-list `[]` is now treated as "explicitly empty, must equal n=0",
  distinct from `None` (omitted → auto-fill defaults). `add_documents`
  builds the new list-extensions before calling `np.vstack`, so a backend
  exception during embedding doesn't leave a partially-updated retriever.
  **User-visible behavior change:** code that relied on the silent
  fall-through gets a `ValueError` with a specific field-name message.
  Test pins: `test_retriever_init_rejects_external_ids_length_mismatch`,
  `test_retriever_init_rejects_metadata_list_length_mismatch`,
  `test_retriever_init_rejects_explicit_empty_list`,
  `test_add_documents_validates_before_mutating`.
- **KNT-103 (HIGH) — scoped offline mode.** The audit-01 KNT-005 fix used
  `os.environ.setdefault()` which (1) refused to override
  `HF_HUB_OFFLINE=0` from the caller's shell, silently ignoring
  `offline=True`, and (2) once set to `"1"` never reverted, leaking
  offline policy to subsequent `offline=False` loads in the same process.
  Replaced with `_offline_env_scope` contextmanager that snapshot/restores
  both `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` around backend
  construction — restoration runs even on backend exception. The same
  context wraps `CrossEncoderReranker.load`, so the reranker honors
  `KaosNLPTransformersSettings.offline` exactly the same way.
  **User-visible behavior change:** long-running processes (FastAPI
  servers, agent loops) get reliable per-call offline policy. The
  `test_offline_setting_sets_hf_env_vars` test was rewritten to capture
  mid-scope state instead of post-call leftovers; see audit-02 test
  suite for the new restoration assertions. Test pins:
  `test_offline_env_scope_restores_on_clean_exit`,
  `test_offline_env_scope_overrides_hostile_zero`,
  `test_offline_env_scope_restores_on_exception`,
  `test_offline_false_is_noop`,
  `test_consecutive_offline_loads_dont_leak`.
- **KNT-104 (HIGH) — reranker registry parity.** `CrossEncoderReranker`
  was a top-level export but accepted any HuggingFace Hub model id with no
  license, revision, or offline gate — undermining the registry discipline
  embeddings enforced. Fix: added `RERANKER_REGISTRY` and `RERANKER_EXCLUDED`
  in `kaos_nlp_transformers.models` (same shape as `REGISTRY`/`EXCLUDED`,
  separate dicts so a reranker model id can never accidentally be used
  for embedding or vice versa). `CrossEncoderReranker.load` now enforces
  the registry gate (or `settings.allow_unregistered`), threads the pinned
  revision through `CrossEncoder(revision=...)`, and uses
  `_offline_env_scope`. Cache key keyed by `(model_id, revision, device,
  cache_dir)`. v0 reranker registry ships one entry: `BAAI/bge-reranker-base`
  (278M, MIT, SHA verified against HuggingFace API on 2026-05-08).
  Test pins: `test_reranker_registry_has_at_least_one_entry`,
  `test_reranker_load_rejects_unregistered`,
  `test_reranker_loader_signature_accepts_revision_and_cache_dir`,
  `test_reranker_excluded_blocks_load`.

### Changed

- **KNT-105 (MED) — `SemanticDedupLevel` reports real similarity.**
  The 0.1.0a1 code constructed `DedupCluster` without setting `similarity`,
  so every semantic cluster inherited the dataclass default `1.0`
  regardless of cluster tightness. Fix: compute mean pairwise cosine
  similarity over the cluster's L2-normalized embeddings (cheap — clusters
  are small) and pass to `DedupCluster(similarity=...)`. Result is clamped
  to `[0.0, 1.0]` for numeric jitter on near-1.0 values. Also: validate
  `distance_threshold` against the cosine distance domain `[0.0, 2.0]` at
  `__init__` time. Test pins: `test_semantic_dedup_threshold_validated`,
  `test_semantic_dedup_returns_real_similarity`,
  `test_semantic_dedup_threshold_monotonicity`.
- **KNT-106 (MED) — `EmbeddingRetriever.from_corpus` single-path.**
  The 0.1.0a1 implementation tried two embedding paths
  (`corpus.embed()` if present, else inline) and lost
  `device`/`backend`/`settings` policy on the first one. It also iterated
  the corpus twice (broken for generator-style corpora). Fix: drop the
  `corpus.embed` fallback entirely; materialize the corpus iterable once
  via `list(corpus)`; always embed inline through the loaded
  `EmbeddingModel.embed`. Single embedding code path is simpler and the
  loaded model's policy reaches every row. Test pins:
  `test_from_corpus_uses_loaded_model_not_corpus_embed`,
  `test_from_corpus_materializes_iterator_once`.
- **KNT-107 (LOW) — `_resolve_backend` strict validation.** Unknown
  backend strings (typos like `"tensorflow"`, `"FastEmbed"`,
  empty string) silently fell through to the auto path and picked
  a backend the user did not ask for. Now validated against
  `{"auto", "fastembed", "sentence-transformers"}` and raises `ValueError`
  with the valid set in the message. **User-visible behavior change:**
  misconfiguration fails loudly at the boundary, not silently inside the
  wrong backend. Test pin: `test_resolve_backend_rejects_unknown_backend`.

### Added

- **`RERANKER_REGISTRY` and `RERANKER_EXCLUDED`** top-level exports
  (KNT-104). 18 symbols total in `__all__` (was 16 in 0.1.0a1).
- **`_offline_env_scope` context manager** in
  `kaos_nlp_transformers.embedding` (private, but referenced from
  `reranker.py` for KNT-104 parity).
- **24 audit-02 regression tests** in `tests/unit/test_audit_02.py`,
  including two property-style invariants (retriever length-equality
  across random `add_documents` sequences; semantic-dedup membership
  monotonicity in `distance_threshold`).
- **PRD + CLAUDE.md status refresh** (KNT-108): both now reflect the
  shipped public API as of 0.1.0a2.

### Internal docs

- PRD `docs/internal/prd/kaos-nlp-transformers.md` status flipped from
  "Proposed" to "Shipped — 0.1.0a2 published to PyPI 2026-05-08; License
  Apache-2.0".
- `kaos-nlp-transformers/CLAUDE.md` "v0 surface" section rewritten to
  describe the actual 18-symbol public API and audit-01 + audit-02
  invariants.

## [0.1.0a1] — 2026-05-08

First public alpha. Pre-release audit pass `audit-01` (6 findings, all
fixed with regression tests in `tests/unit/test_audit_01.py`).

### Security

- **KNT-001 (HIGH) — upward `kaos_ml_core` import removed.**
  `EmbeddingRetriever.from_corpus()` previously fell back to
  `importlib.import_module("kaos_ml_core.features")` when the corpus
  did not expose an `.embed()` method. That made a Tier 3 package
  reach up into Tier 4 (the documented consumer), creating hidden
  runtime failures for anyone who installed only `kaos-nlp-transformers`.
  The method now uses the corpus's `.embed()` if present, otherwise
  embeds the unit texts directly with the loaded `EmbeddingModel`.
  Test pin: `tests/unit/test_audit_01.py::test_no_kaos_ml_core_import_anywhere`.
- **KNT-002 (HIGH) — `scipy` gated with an actionable install-hint.**
  `SemanticDedupLevel.find_clusters()` imported `scipy` without a
  declared dependency or extra. `scipy` is now declared under a new
  `[clustering]` extra and the import is wrapped in a try/except that
  raises `ImportError` with a fix message pointing at
  `pip install kaos-nlp-transformers[clustering]`. Test pin:
  `tests/unit/test_audit_01.py::test_semantic_dedup_raises_install_hint_when_scipy_missing`.
- **KNT-003 (HIGH) — `RegisteredModel.revision` threaded through loaders.**
  Both `_load_fastembed_cached` and `_load_sentence_transformers_cached`
  now accept `revision` as part of their cache key. The
  sentence-transformers loader passes it through to
  `SentenceTransformer(model_name, revision=...)`. fastembed maintains
  its own model registry with a fixed revision per release and does
  not expose a runtime revision override; the limitation is documented
  in the loader docstring and the cache-key invalidation still works
  if the registry's pinned SHA changes. Test pin:
  `tests/unit/test_audit_01.py::test_loader_signatures_accept_revision`.

### Changed

- **KNT-004 — settings injection on retriever and dedup factories.**
  `EmbeddingRetriever.from_texts`, `.from_corpus`, and
  `SemanticDedupLevel.__init__` now accept `device`, `backend`, and
  `settings` kwargs that are forwarded to `EmbeddingModel.load`. This
  closes the previous gap where outer-boundary cache/offline/device
  policy could not be injected past the factory layer. Test pins:
  `test_retriever_factories_accept_settings`,
  `test_semantic_dedup_accepts_settings`.
- **KNT-005 — `settings.offline` enforced at the load boundary.**
  `EmbeddingModel.load` now sets `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` (the documented huggingface_hub /
  transformers env vars) when the resolved `KaosNLPTransformersSettings`
  has `offline=True`. Both fastembed and sentence-transformers route
  through huggingface_hub and respect these flags — they will refuse
  to fetch missing models rather than silently downloading. Test pin:
  `test_offline_setting_sets_hf_env_vars`.
- **KNT-006 — top-level `__all__` ordering pinned.** `ruff RUF022`
  applied to the public surface; the resulting ordering (constants
  → classes → dunder → callables) is pinned in
  `test_top_level_all_passes_isort_check` so future drift is caught.

### Added

- **`[clustering]` extra** — `scipy>=1.14.1` for
  `SemanticDedupLevel`. Without it the rest of the package still
  installs cleanly; only the dedup level requires it.

### License

This release is the first to ship under the Apache License 2.0. Earlier
internal versions were proprietary.

[Unreleased]: https://github.com/273v/kaos-nlp-transformers/compare/v0.1.0a6...HEAD
[0.1.0a5]: https://github.com/273v/kaos-nlp-transformers/compare/v0.1.0a3...v0.1.0a5
[0.1.0a2]: https://github.com/273v/kaos-nlp-transformers/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/273v/kaos-nlp-transformers/releases/tag/v0.1.0a1
