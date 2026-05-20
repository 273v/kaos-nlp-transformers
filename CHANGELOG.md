# Changelog

All notable changes to `kaos-nlp-transformers` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]


## [0.1.0a7] — 2026-05-20

WU-F.8 of the 0.1.0 GA plan
(`kaos-modules/docs/plans/2026-05-20-0.1.0-ga-plan.md`).
Decision #1 of the GA plan: stay on the 0.1.x line for ecosystem
consistency with the rest of the kaos-* DAG. The intervening
0.2.0aN PyPI releases (a1..a8) shipped the Phase-8 NLI / GLiNER /
PiiDetector inference stack and the KNT-601 hard cutover to the
in-tree Rust `ort` cdylib; that code is preserved in this release
under a 0.1.x version label.

Note: PEP 440 ordering means the prior 0.2.0a8 PyPI release will
still resolve as "newest" for callers without an explicit `<0.2`
ceiling. 0.1.x consumers should pin `kaos-nlp-transformers<0.2`.

### Changed

- Version label: source moves from `0.2.0-alpha.8` back to
  `0.1.0-alpha.7` (Cargo.toml single source of truth). PEP 440
  wheel metadata reads `0.1.0a7`.
- Bumped minimum `kaos-core` to `0.1.0a12,<0.2` (was `>=0.1.0a1`):
  catch up to current post-URI-redesign + Capability type API.
- Refreshed `uv.lock`: `kaos-core 0.1.0a10 -> 0.1.0a12`,
  `kaos-content 0.1.0a2 -> 0.1.0a12`, `kaos-nlp-core 0.1.0a6 -> 0.1.0a8`,
  `ruff 0.15.12 -> 0.15.13`, `ty 0.0.34 -> 0.0.36`,
  `maturin 1.13.1 -> 1.13.3`.

### Internal

- 237 unit tests pass with the rebuilt Rust cdylib at
  `0.1.0-alpha.7`. `ruff format / check / ty check` clean.


## [0.2.0a8] — 2026-05-16

Consolidates the work originally drafted for an in-flight 0.2.0a7 (NLI +
GLiNER + threading + Phase-8 scale/quality benchmarks, dated
2026-05-15) with the additional PII / MCP / prefetch work dated
2026-05-16. The intermediate 0.2.0a7 version was never published to
PyPI; releasing as a single a8 keeps the published-version history
contiguous with a6.

### Added

- **`PiiDetector` — closed-label PII detection** (`kaos_nlp_transformers.pii`).
  BERT-style token classifier complementing `GLiNERExtractor` for
  the standard PII redaction / compliance workflow. Default model:
  `onnx-community/bert-small-pii-detection-ONNX` (Apache-2.0 chain
  via upstream `gravitee-io/bert-small-pii-detection`; 28M params;
  27 MB int8 ONNX). Trained on `beki/privy` +
  `gretelai/synthetic_pii_finance_multilingual` + CoNLL-2003.
  Surfaces 24 PII categories (PERSON, EMAIL_ADDRESS, PHONE_NUMBER,
  CREDIT_CARD, US_SSN, US_ITIN, IBAN_CODE, FINANCIAL, …) with
  char-offset spans. Roughly 10× faster than running GLiNER
  zero-shot for the same closed-label set.
- **Shared `Entity` dataclass** — `PiiDetector.detect()` returns the
  same `Entity` shape as `GLiNERExtractor.extract()` so downstream
  redaction pipelines / `kaos_llm_core.programs.ner.GLiNERExtract`
  consume both extractors interchangeably.
- **`rust/core/token_classify.rs`** — new BERT-style token-classifier
  module (third inference pattern after NLI sentence-pair softmax,
  reranker sentence-pair sigmoid, GLiNER prompt-span). Tokenize with
  HuggingFace offset tracking → ort session → softmax-argmax per
  token → BIO decode → char offsets. Reads `id2label` from
  `config.json` at load time (fetched via hf-hub alongside
  ONNX + tokenizer). Output spans share `core::ner::Entity`.
- **`PII_REGISTRY` + `PII_EXCLUDED`** in `kaos_nlp_transformers.models`,
  pinning `onnx-community/bert-small-pii-detection-ONNX` and
  excluding `urchade/gliner_multi_pii-v1` (CC-BY-NC) +
  `ai4privacy/pii-masking-200k` (research-only training data).
- **Settings**: `default_pii_model` field on
  `KaosNLPTransformersSettings`, overridable via
  `KAOS_NLP_TRANSFORMERS_DEFAULT_PII_MODEL`.
- **CLI**: `kaos-nlp-transformers prefetch --include pii` and
  `kaos-nlp-transformers info` now show the PII registry; the
  `--include {embedding,reranker,nli,ner,pii}` enum gains the
  fifth member.
- **Live integration tests**: 10 tests in
  `tests/integration/test_pii_live.py` covering offset round-trip
  on multibyte text, 24-category label exposure, financial PII
  detection (SSN, credit card), batch independence.
- **Unit tests**: 10 tests in `tests/unit/test_pii.py` covering
  registry gating, threshold validation, `Entity` shape parity
  with GLiNER.


- **`NliModel` — natural-language-inference cross-encoder** (`kaos_nlp_transformers.nli`).
  Lands the Phase-8 lower half from
  `kaos-llm-core/docs/summarization-classification-plan.md` §4.2.3.
  Default model: `Xenova/nli-deberta-v3-base` (Apache-2.0 chain via
  upstream `cross-encoder/nli-deberta-v3-base`; 184M params; 244 MB
  `onnx/model_quantized.onnx`). Returns softmax-normalized
  three-class probabilities in the canonical
  `(entailment, neutral, contradiction)` order — `NliModel`
  satisfies `kaos_llm_core.programs.classify.nli.NLIScorer` at
  runtime so `ZeroShotNLIClassifier` is a drop-in consumer. The
  Rust core hard-codes the `id2label` permutation for the registered
  model; a future second model lands the dynamic `config.json`
  parse.
- **`GLiNERExtractor` — zero-shot NER via span extraction**
  (`kaos_nlp_transformers.ner`). Default model:
  `onnx-community/gliner_medium-v2.1` (Apache-2.0 chain via upstream
  `urchade/gliner_medium-v2.1`; 195M params; 746 MB fp32
  `onnx/model.onnx`). Implements the GLiNER prompt template
  `[<<ENT>>, label_1, <<ENT>>, label_2, ..., <<SEP>>, w_1, w_2, ...]`
  with word-level subword bookkeeping and span enumeration over
  `(start_word, width)` pairs — ported inline from the gline-rs
  reference (Apache-2.0) rather than added as a crate dep because
  gline-rs pins `ort 2.0.0-rc.9` / `tokenizers 0.21` / `ndarray 0.16`
  vs our 2.0.0-rc.10 / 0.23 / 0.17, and depends on a git-only
  sibling `orp`. The int8-quantized variant
  (`onnx/model_quantized.onnx`) was tested and rejected — its scores
  cap around 0.13 on PyTorch-reference 0.99 inputs, producing
  zero spans at the default threshold; the fp32 export is the
  default. `GLiNERExtractor` satisfies
  `kaos_llm_core.programs.ner.NerExtractor` at runtime.
- **`NER_REGISTRY` + `NER_EXCLUDED`** in
  `kaos_nlp_transformers.models`, pinning
  `onnx-community/gliner_medium-v2.1` and
  `onnx-community/gliner_multi-v2.1` (English + multilingual
  GLiNER), and excluding `urchade/gliner_base` +
  `onnx-community/gliner_base` (both CC-BY-NC 4.0 via upstream's
  weight licensing).
- **`NLI_REGISTRY` + `NLI_EXCLUDED`** similarly; the excluded entry
  records why `Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
  is blocked (NC training-data contamination).
- **Rust core**: `core::nli::OrtNliClassifier` +
  `core::nli::NliClassifier` trait;
  `core::ner::OrtGlinerExtractor` + `core::ner::NerExtractor` trait;
  PyO3 bindings at `bindings::nli` + `bindings::ner`.
- **Live integration tests**: `tests/integration/test_nli_live.py`
  (8 tests, all passing on the real ONNX) and
  `tests/integration/test_ner_live.py` (9 tests, all passing). The
  NLI tests verify the canonical
  `(entailment, neutral, contradiction)` permutation end-to-end; the
  NER tests verify byte-offset round-trip
  (`text[start:end] == entity.text`) and threshold monotonicity.
- **Settings**: `default_nli_model` and `default_ner_model` fields
  on `KaosNLPTransformersSettings`, overridable via
  `KAOS_NLP_TRANSFORMERS_DEFAULT_NLI_MODEL` /
  `KAOS_NLP_TRANSFORMERS_DEFAULT_NER_MODEL`.
- **`kaos-nlp-transformers prefetch` subcommand** + programmatic
  `kaos_nlp_transformers.cli.prefetch_models()`. Walks every
  registry (embedding / reranker / NLI / NER) and calls `.load()`
  to populate the HF Hub cache before first inference — useful in
  Dockerfile builds, CI cache-warming jobs, and air-gapped image
  preparation. Honors `--cache-dir`, `KAOS_NLP_TRANSFORMERS_CACHE_DIR`,
  `HF_HOME`. Filter with `--include {embedding,reranker,nli,ner}`
  (repeatable) or `--model <id>` (repeatable). Supports
  `--dry-run` and `--json` for tooling integration. Exits non-zero
  on any model load failure but continues through the rest of the
  batch so one bad row doesn't sink the whole prefetch.

### Changed

- Added `regex` to `[dependencies]` for the GLiNER word-level
  splitter (`\w+(?:[-_]\w+)*|\S`). Already a transitive dep through
  `tokenizers`; declaring it directly avoids relying on a transitive
  API surface.
- **`Entity.start` / `Entity.end` are codepoint offsets, not byte
  offsets** (KNT-NLI-003). The initial 0.2.0a7 build emitted byte
  offsets straight from the `regex` crate; that broke Python's
  char-indexed slicing on any contract containing curly quotes,
  em-dashes, or other multi-byte typographic punctuation. The
  Rust core now builds a byte→char map at split time, slices the
  source by byte (for `entity.text`) but exports char offsets in
  `Entity` so `source_text[e.start:e.end] == e.text` round-trips
  on all UTF-8 input. Caught by the new
  `tests/scale/test_ner_scale.py` benchmark on EDGAR; the unit
  ASCII test had been passing through the latent bug. New regression
  test `test_extract_offsets_roundtrip_on_multibyte_text` locks the
  invariant in.
- **Optional `KAOS_NLP_TRANSFORMERS_INTRA_THREADS` env-var override**
  added to all four ort backends (embedding, reranker, NLI, NER).
  When unset, ort picks its own intra-op thread count — which
  empirically beat every explicit setting we tried on a 20-core CPU
  host (`OMP_NUM_THREADS=1` had zero effect; explicit
  `intra_threads=20` was 80% slower than the default on the
  short-sequence GLiNER workload). The env var is kept for tuning
  unusual deployment shapes (long sequences, large batches, small
  cgroup quotas).


## [0.2.0a6] — 2026-05-15

### Changed

- **`SemanticChunker._pack` now routes adjacent-pair cosine through
  the pre-normalised fast path** in kaos-nlp-core 0.1.0a6+
  (`cosine_adjacent_normalized` instead of the generic
  `cosine_adjacent`). The `Embedder` protocol contract — and our
  canonical `EmbeddingModel` implementation — already guarantee
  unit-norm rows, so this is a free 1.5–7.4× speedup at every
  tested chunker shape (measured on Intel i7-12700K AVX2+FMA;
  cross-CPU envelope tracked in
  `docs/benchmarks/semantic-chunker-throughput-*.json`).
- **`ExtractiveRanker.rank` similarly routes cosine through the
  pre-normalised fast path** (`cosine_one_to_many_normalized`). The
  query-mode branch is direct; the centroid-mode branch first
  normalises the centroid in-place via
  `kaos_nlp_core.similarity.l2_normalize_in_place` because the mean
  of unit-norm rows is not itself unit-norm.
- **`Embedder` protocol docstring** explicitly states the unit-norm
  output contract that the canonical implementation already honours.
- Bumped `kaos-nlp-core` floor to `>=0.1.0a6` (was `>=0.1.0a5`).
  0.1.0a6 added the `cosine_*_normalized` and `l2_normalize_in_place`
  public surface this release consumes.

### Added

- **Throughput benches** at `tests/bench_semantic_chunker.py` and
  `tests/bench_extraction.py`. Measure end-to-end docs/sec through
  the full pipeline (embed + cosine + chunk-emit / rank) on the
  vendored model2vec embedder over USC / EDGAR / patents corpora.
  Marked `@pytest.mark.slow`; opt-in via `KAOS_NLP_SCALE_FIXTURES`.
  Honest numbers committed to
  `docs/benchmarks/semantic-chunker-throughput-*.json` and
  `extractive-ranker-throughput-*.json`.

### Perf envelope (measured)

Intel i7-12700K, model2vec / potion-base-8M embedder, single-core:

| Workload                                 | docs/sec | p50 ms/doc |
|------------------------------------------|---------:|-----------:|
| `SemanticChunker` over EDGAR (40 paras)  |     88.6 |       5.46 |
| `SemanticChunker` over USC (4 paras)     |    674.2 |       0.76 |
| `SemanticChunker` over patents (49 paras)|     64.5 |      13.27 |
| `ExtractiveRanker` centroid + k=10 EDGAR |     88.3 |       6.12 |
| `ExtractiveRanker` query + k=10 EDGAR    |     85.8 |       5.14 |
| `ExtractiveRanker` query + MMR (0.5)     |     83.4 |       6.84 |
| `ExtractiveRanker` centroid + k=20 USC   |    722.7 |       0.62 |

The cosine-dominated phase moved from one of the slower steps to a
sub-millisecond per-doc contribution; the throughput cap is now the
embedder inference time, which is exactly the right place for the
bottleneck to be.


## [0.2.0a5] — 2026-05-15

### Changed

- **`SemanticChunker._pack` now runs entirely in Rust** via
  ``kaos_nlp_core._rust.chunking.semantic_pack``. The greedy
  budget+topic-shift boundary scan moved from a Python loop into the
  Rust kernel; the Python wrapper now only marshals ``uint32`` offset
  + token arrays in, materialises Chunk objects from the returned
  group records out, and computes the adjacency cosine via the
  already-Rust ``cosine_adjacent``. Behaviour is bit-identical to the
  prior pure-Python loop (24 SemanticChunker tests + 5 scale tests
  pass unchanged).
- **`SemanticChunker` and `ExtractiveRanker` now route post-inference
  cosine + MMR through the Rust-backed
  :mod:`kaos_nlp_core.similarity` layer** (NumKong SIMD kernels).
  Previously these used numpy einsum / matmul in Python:
  - `SemanticChunker._pack` adjacent-pair cosine — now
    ``kaos_nlp_core.similarity.cosine_adjacent`` (one SIMD-dispatched
    call instead of a normalize + einsum pair).
  - `ExtractiveRanker.rank` centroid + query scoring — now
    ``kaos_nlp_core.similarity.cosine_one_to_many`` (17x numpy on
    1000-row x 768-d workloads).
  - `ExtractiveRanker.rank` MMR diversification — now
    ``kaos_nlp_core.similarity.mmr_select`` (67x numpy on
    1000-row x 768-d MMR with ``k=20``).
  - The local ``_cosine`` helper is retained as a no-op
    backwards-compat stub but is no longer called in the hot path.
  - Behavior contract: results agree with numpy reference within
    ``1e-5`` per cell (validated in
    ``kaos-nlp-core/tests/test_similarity.py`` and the bench grid).

### Documentation

- **Use, data-handling, and AI-authorship disclosure** added to the
  README. Confirms that inference is local (Rust cdylib + ONNX
  Runtime; no LLM provider transmission) once the model is cached.
  Notes that downstream `kaos-llm-core` Programs may transmit text
  to providers — sensitive-data callers should check that
  package's disclosure. AI-assisted authorship disclosure (Claude,
  Anthropic; human-reviewed) added.

### Added

- **`SemanticChunker`** — embedding-driven document chunker that
  implements the
  :class:`kaos_nlp_core.chunking.Chunker` protocol. Boundaries are
  placed where adjacent paragraph (or sentence) embeddings drop in
  cosine similarity below ``drop_threshold`` or where the running
  token count exceeds ``max_tokens``. Phase 5 of the cross-module
  summarization/classification plan.
- **`ExtractiveRanker`** — sentence-salience ranker with three modes:
  generic (centroid cosine), query-focused (query embedding cosine),
  and cross-encoder reranking. MMR diversity supported via the
  ``diversify`` parameter.
- **`ScoredSegment`** — frozen+slotted dataclass carrying ``text``,
  ``start``, ``end``, ``score``, ``rank``.
- **`ChunkerEmbedder` / `ExtractiveReranker`** — runtime-checkable
  Protocols defining the minimum interface
  :class:`SemanticChunker` / :class:`ExtractiveRanker` consume.
  Stubbing these in tests keeps the unit gate offline and never
  touches the Rust cdylib.
- All five new names are re-exported from
  ``kaos_nlp_transformers`` and listed in ``__all__``.
- Audit-07 KNT-700/701 extends ``test_audit_01`` to recognize the
  new public exports.

## [0.2.0a4] — 2026-05-11

### Fixed

- **CI: wheel-smoke step no longer shadowed by the workspace source
  tree.** ``python-source = "."`` means ``kaos_nlp_transformers/`` lives
  at the repo root. CPython's default ``sys.path[0] = cwd`` makes
  ``python -c "from kaos_nlp_transformers._rust import registry"`` (run
  from the repo workdir) pick up the source tree — which only ships
  ``_rust.pyi`` (a stub, not importable) after the single-file
  consolidation — shadowing the wheel install in ``/tmp/smoke``.
  Added ``PYTHONSAFEPATH=1`` (PEP 711, drops the implicit cwd entry)
  plus ``cd /tmp/smoke`` plus ``--reinstall`` to the wheel-smoke step,
  matching the canonical pattern documented in
  ``kaos-modules/docs/oss/40-ci-cd/hosted-runners.md``. This is the
  fix PR #1 (closed in favor of PR #7) originally carried; the
  type-stub consolidation half of #1 was the actual root-cause fix
  once paired with the PYTHONSAFEPATH guard. Files:
  ``.github/workflows/ci.yml``.
- **Tests: `_rust` submodule imports updated for the single-file
  `.pyi` consolidation.** Three test files (`tests/unit/test_rust_extension.py`,
  `tests/integration/test_embed_gpu.py`,
  `tests/integration/test_reranker_live.py`) still used the
  pre-refactor `from kaos_nlp_transformers._rust.<submodule> import
  <name>` pattern. After the stub-consolidation refactor, ty cannot
  resolve `_rust.<submodule>` as a module because the new `.pyi`
  exposes those names as classes inside `_rust.pyi` (the class-as-
  namespace pattern). The runtime PyO3 cdylib still exposes the
  submodules — the breakage was static-typing-only — but it surfaced
  as Lint + Pre-commit hooks CI failures on every PR. Tests now use
  the same `from kaos_nlp_transformers._rust import <submodule>`
  pattern the production code (device.py / embedding.py /
  reranker.py) was switched to in the refactor.

### Security

- **cargo-audit: ignore ``RUSTSEC-2024-0436`` (paste unmaintained) via
  ``.cargo/audit.toml``.** cargo-audit and cargo-deny consult separate
  advisory sources; ignoring the advisory in ``deny.toml`` is not
  enough on its own. The cargo-audit job in ``security.yml`` continued
  to fail (compounded by missing ``checks: write`` permission, which
  surfaces as ``Resource not accessible by integration`` when the
  audit-check action tries to annotate findings). Added an
  ``.cargo/audit.toml`` ignore list mirroring ``deny.toml``'s
  acknowledgement, and granted the workflow ``checks: write`` so the
  audit action's check-run creation succeeds. Drop the ignore in
  both files together once tokenizers migrates off ``paste``. Files:
  ``.cargo/audit.toml`` (new), ``.github/workflows/security.yml``.
- **cargo-deny: ignore ``RUSTSEC-2024-0436`` (paste unmaintained).**
  ``paste 1.0.15`` is pulled transitively via
  ``tokenizers 0.22.2 → macro_rules_attribute 0.2.2 → paste``. It's a
  proc-macro crate (compile-time only — nothing ships in the wheel at
  runtime) and the advisory itself notes "No safe upgrade is
  available" pending the ``pastey`` migration. The
  ``audit-KNT-601 §15`` block in ``deny.toml`` already documented
  this, but the ``ignore`` list was empty so ``cargo-deny`` failed
  on every CI run. Made the acknowledgement load-bearing. Drop the
  ignore once ``tokenizers`` migrates off ``paste``. Files:
  ``deny.toml``.
- **HuggingFace ``snapshot_download`` call now passes ``revision``
  explicitly (bandit B615).** ``_load_model2vec_cached`` previously
  built a ``snapshot_kwargs`` dict and called
  ``snapshot_download(**snapshot_kwargs)``. The revision pin was in
  the dict, so the behavior was already correct (audit-KNT-003), but
  bandit's B615 detector can't follow ``**kwargs`` unpacking and
  flagged the site as an unsafe download. Refactored to pass
  ``revision=`` (and the other args) explicitly as keyword arguments
  so the pin is statically visible. No behavior change — the
  registered SHA is what flows through either way. Files:
  ``kaos_nlp_transformers/embedding.py``.
### Changed

- **Type stubs for ``_rust`` consolidated into a single
  ``_rust.pyi`` sibling file.** Previously the wheel shipped per-
  submodule stubs in a ``kaos_nlp_transformers/_rust/`` subdirectory
  next to ``_rust.abi3.so``. CPython's namespace-package detector
  could ambiguously resolve ``_rust/`` as a package and shadow the
  cdylib, surfacing as ``ImportError: cannot import name '<sub>'
  from 'kaos_nlp_transformers._rust' (unknown location)`` on the
  wheel-install smoke test (PR #1 worked around it with
  ``PYTHONSAFEPATH=1`` + ``cd /tmp/smoke``). The new layout
  eliminates the shadow entirely:
  ``kaos_nlp_transformers/_rust.abi3.so`` (cdylib) +
  ``kaos_nlp_transformers/_rust.pyi`` (single-file stubs, with
  per-submodule types declared as nested classes). No ``_rust/``
  directory exists in the wheel.
- **Three internal call sites switched to attribute-style access
  through the parent ``_rust`` module.** ``device.py``,
  ``embedding.py``, and ``reranker.py`` previously imported via
  ``from kaos_nlp_transformers._rust.<sub> import X``. With the
  single-file stub, type checkers can't see ``<sub>`` as a real
  module (it's a nested class in the stub); the import-style form
  works at runtime but not at type-check time. Refactored to
  ``from kaos_nlp_transformers import _rust;
  X = _rust.<sub>.X`` — identical runtime semantics, fully resolved
  by ``ty``. No external API change.

Once this PR + PR #1 both land, the ``PYTHONSAFEPATH=1`` +
``cd /tmp/smoke`` workaround in ``.github/workflows/ci.yml``'s
smoke-test step can be reverted as a follow-up — the packaging-
level fix obsoletes the workaround.
- **bandit + vulture now run in both pre-commit and CI.** Two new
  hooks in ``.pre-commit-config.yaml`` (bandit + vulture), mirrored
  by two new jobs in ``security.yml`` (``bandit (static security)``
  + ``vulture (dead-code scan)``). Skip lists justified inline.
  Mirrors the rollout from kaos-core. **Depends on PR #3** (B615
  HF snapshot_download explicit revision) — bandit will fail on
  this branch's first run until #3 merges, then rebase clears it.
### Changed

- **uv.lock is now tracked in git.** Previously gitignored at v0.1.0a1
  because the ``[mcp]`` optional extra (and the ``kaos-mcp`` dev
  dependency) referenced a sibling not yet on PyPI; ``uv lock``
  couldn't resolve them. ``kaos-mcp`` shipped (0.1.0a2), so the
  original gating reason no longer applies. Tracking the lockfile
  gives reproducible local dev environments, lets Dependabot surface
  sibling-version bumps as PRs, and makes the supply-chain pin set
  publicly auditable. Mirrors the org-wide convention being adopted
  across all 16 kaos-* repos.

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

[Unreleased]: https://github.com/273v/kaos-nlp-transformers/compare/v0.2.0a4...HEAD
[0.2.0a4]: https://github.com/273v/kaos-nlp-transformers/compare/v0.2.0a3...v0.2.0a4
[0.1.0a5]: https://github.com/273v/kaos-nlp-transformers/compare/v0.1.0a3...v0.1.0a5
[0.1.0a2]: https://github.com/273v/kaos-nlp-transformers/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/273v/kaos-nlp-transformers/releases/tag/v0.1.0a1
