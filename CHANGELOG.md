# Changelog

All notable changes to `kaos-nlp-transformers` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/273v/kaos-nlp-transformers/compare/v0.1.0a2...HEAD
[0.1.0a2]: https://github.com/273v/kaos-nlp-transformers/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/273v/kaos-nlp-transformers/releases/tag/v0.1.0a1
