# Changelog

All notable changes to `kaos-nlp-transformers` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/273v/kaos-nlp-transformers/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/273v/kaos-nlp-transformers/releases/tag/v0.1.0a1
