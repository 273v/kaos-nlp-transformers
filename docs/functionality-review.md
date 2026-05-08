# kaos-nlp-transformers Functionality Review

Date: 2026-05-08

Scope: `kaos-nlp-transformers` module, including embeddings, dense retrieval,
semantic deduplication, reranking, settings, packaging, CLI/server stubs, and
unit/integration tests.

## Summary

This module is a useful and mostly well-shaped home for neural NLP workloads
that should not live in `kaos-nlp-core`. The package split is sound: sparse and
lightweight NLP stays in `kaos-nlp-core`, while ONNX/torch-backed inference
lives here.

The main concern is that the public surface has grown past the original v0
contract without the same level of registry, offline, normalization, and test
discipline being applied everywhere. Embeddings are reasonably mature; dense
retrieval and semantic dedup are useful but need invariant hardening; reranking
is public but not yet governed by the registry/license/offline machinery.

## What Is Good

- The package boundary is right. Heavy ML dependencies are isolated from
  `kaos-nlp-core`, and optional imports are mostly lazy.
- `EmbeddingModel.load()` has a clear registry gate for embedding models and
  useful error messages for excluded or unregistered models.
- The model registry carries license, revision, dimension, backend, and notes.
  That is the right control point for a module that downloads third-party
  model artifacts.
- `EmbeddingRetriever` threads document IDs, external IDs, text, and metadata
  through `SearchHit`, which makes it composable with the sparse retriever
  surface.
- `SemanticDedupLevel` is in the right package. Keeping embedding-backed dedup
  out of `kaos-content` avoids dragging model runtimes into document plumbing.
- The tests include real integration checks for model download/inference and
  semantic behavior, not just shape-only assertions.
- The audit regression tests pin several important architecture fixes:
  no upward `kaos_ml_core` imports, scipy gating, revision threading, settings
  injection, offline env wiring, and top-level `__all__` ordering.

## What Is Bad Or Risky

### Embedding normalization contract is inconsistent

`README.md` and the PRD describe `EmbeddingModel.embed()` output as L2-normalized,
but `kaos_nlp_transformers/embedding.py` only casts backend output to `float32`.
It does not normalize rows, and the sentence-transformers path does not pass
`normalize_embeddings=True`.

Impact: direct dot-product consumers can get backend-dependent scores. It also
makes CPU/GPU parity less certain because fastembed and sentence-transformers
may not use identical default normalization behavior.

Recommendation: normalize in `EmbeddingModel.embed()` after shape validation, or
change the API docs to stop promising normalized vectors. The better default for
this module is to normalize centrally.

### Reranker bypasses the model governance layer

`CrossEncoderReranker` is exported from the top-level package, but it loads an
arbitrary model ID directly through sentence-transformers. It does not use a
registry, license audit, pinned revisions, cache directory, or offline settings.

Impact: this undermines the registry/license discipline that embeddings enforce.
It also means offline behavior and reproducibility are different for reranking
than for embedding.

Recommendation: introduce a task-aware registry, for example embedding vs.
reranker vs. zeroshot model specs, and route all model loading through shared
policy code.

### Offline enforcement is process-global and not reversible

`EmbeddingModel.load()` sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` when
settings say `offline=True`, but it uses `setdefault()`. That does not override
`HF_HUB_OFFLINE=0`, and once the variables are set to `1`, later calls with
`offline=False` remain affected in the same process.

Impact: settings changes are not reliably honored mid-process even though the
comment says they are. Tests only assert the happy path where env vars start
unset.

Recommendation: use backend-supported local-only options where available, or
wrap env mutation in a scoped context around backend construction. At minimum,
set explicit values and restore prior values after load.

### Retriever invariants are under-validated

`EmbeddingRetriever.__init__()` validates embedding rows against `doc_ids` and
`texts`, but does not validate `external_ids` or `metadata_list` lengths.
`add_documents()` also does not validate that `texts`, `doc_ids`, and optional
fields have matching lengths before mutating internal arrays and lists.

Impact: one bad call can leave the retriever in an internally inconsistent
state that fails later during retrieval or returns mismatched metadata.

Recommendation: centralize parallel-list validation and use it in constructor,
`from_texts()`, and `add_documents()`.

### `from_corpus()` loads too early and loses policy

`EmbeddingRetriever.from_corpus()` loads an `EmbeddingModel` before checking
whether the corpus can provide cached embeddings through `corpus.embed()`. It
also calls `corpus.embed(model=model_id, batch_size=batch_size)` without
forwarding cache/settings/device/backend policy.

Impact: it may download/load a model unnecessarily and can bypass the caller's
intended cache/offline/device behavior when the corpus has its own embedding
path.

Recommendation: materialize corpus units once, decide the embedding source
first, and pass a structured embedding policy rather than a loose subset of
arguments.

### Semantic dedup reports exact-match similarity

`SemanticDedupLevel` creates `DedupCluster` without setting `similarity`, so
clusters inherit `similarity=1.0` from `DedupCluster`. That is misleading for
semantic clusters. The code also does not guard NaN cosine distances from
zero-vector embeddings or validate threshold ranges.

Impact: reports can overstate semantic confidence, and pathological embeddings
can destabilize clustering.

Recommendation: compute mean pairwise cosine similarity for each returned
cluster, set it explicitly, and validate `distance_threshold`.

### Documentation is stale

The README still says v0 has one class with two methods and that the license is
proprietary. The package now exports `EmbeddingRetriever` and
`CrossEncoderReranker`, contains semantic dedup, and `pyproject.toml` declares
Apache-2.0.

Impact: users and agents will follow the wrong API and compliance story.

Recommendation: update README/CLAUDE/PRD status to reflect current public API,
extras, license, and planned-vs-shipped functionality.

### Invalid backend names silently fall through

`_resolve_backend()` treats any unknown backend string as if it were `"auto"`.

Impact: configuration mistakes silently choose a backend instead of failing
early.

Recommendation: validate backend settings with a literal/enum type or explicit
runtime check.

## What Is Missing

- MCP tools for `embed`, `rerank`, `list-models`, and eventually `zeroshot`.
- CLI commands beyond `info`, especially `embed`, `download`, and `list-models`.
- A real model registry for rerankers and future zero-shot models.
- A pre-download/cache validation path for offline deployments.
- Persistent vector index support and/or ANN integration for large corpora.
- A documented model expansion process that includes license verification,
  revision verification, backend load verification, and live tests.
- A common backend adapter layer that owns normalization, dtype, batching,
  error wrapping, revision/cache/offline policy, and instrumentation.
- Benchmarks for embedding throughput, retrieval latency, memory use, and
  semantic dedup scaling.

## Refactor Recommendations

1. Create a shared model-loading policy layer.

   Suggested shape: `ModelSpec(task, model_id, revision, license, dim, backend,
   params_m, notes)` plus task-specific registries. `EmbeddingModel`,
   `CrossEncoderReranker`, and future classifiers should all load through this
   layer.

2. Add backend adapters.

   Instead of branching directly on backend names inside `EmbeddingModel`, use
   small adapters with a common interface: `load(spec, policy)` and
   `embed(texts, batch_size)`. This is where normalization, dtype, revision,
   cache, offline mode, and backend-specific kwargs should live.

3. Harden `EmbeddingRetriever`.

   Make validation explicit, materialize corpus iterables once, avoid mutating
   internal state until new document embeddings and metadata are fully valid,
   and consider an immutable index object with a builder for appends.

4. Clarify async behavior.

   `EmbeddingRetriever.retrieve()` is async but performs embedding and matrix
   multiplication inline. Either make it synchronous or dispatch CPU/GPU-bound
   work through `asyncio.to_thread()` consistently.

5. Split shipped vs. planned docs.

   The README should describe what is actually public today. PRD/roadmap docs
   can keep the future phases.

## Missing Tests

### Unit tests

- `EmbeddingModel.embed()` normalization and zero-vector handling with fake
  backends.
- sentence-transformers call kwargs, especially `revision`, cache folder, and
  normalization policy.
- invalid `device`, invalid `backend`, invalid `batch_size`, and non-string
  input behavior.
- offline behavior when env vars are already set to conflicting values.
- cache key separation by model ID, revision, cache dir, provider, and device.
- `EmbeddingRetriever` constructor and `add_documents()` length mismatch tests
  for `external_ids`, `metadata_list`, `texts`, and `doc_ids`.
- `EmbeddingRetriever.from_collection()` and `from_corpus()` without live model
  downloads, using monkeypatched `EmbeddingModel.load()`.
- `CrossEncoderReranker` with a fake backend: ordering, truncation, empty input,
  score normalization, and backend exceptions.
- CLI `info --json` and non-JSON output.
- `serve.main()` stub behavior.

### Integration tests

- Live reranker smoke test, gated behind `[torch]` and network.
- Offline load test with a controlled empty cache that proves no network access
  is attempted.
- End-to-end dense retriever from a real KAOS corpus with provenance assertions.
- Semantic dedup on deterministic mocked embeddings and on a small real text
  fixture.
- CPU vs GPU parity with normalization asserted by norm and cosine similarity.

### Fuzzing / Property tests

Use Hypothesis for:

- Retriever invariants over random embedding matrices, document counts, and
  `top_k` values.
- Append sequences for `add_documents()` that verify internal arrays and
  metadata lists always stay the same length.
- Degenerate vectors: all-zero, duplicate vectors, very large values, NaN/Inf
  rejection or handling.
- Corpus grouping behavior with random units, missing group values, repeated
  document URIs, and repeated section refs.
- Semantic dedup threshold monotonicity: tightening the threshold should never
  increase clustered membership for a fixed distance matrix.

## Verification Performed During Review

Commands run:

```bash
uv run ruff check kaos_nlp_transformers tests
uv run ty check kaos_nlp_transformers tests
uv run pytest tests/unit -q
uv run pytest tests/unit --cov=kaos_nlp_transformers --cov-report=term-missing -q
uv run ruff format --check kaos_nlp_transformers tests
```

Results:

- `ruff check` passed.
- `ty check` passed.
- Unit tests passed: 63 tests.
- Unit coverage was 60%.
- `ruff format --check` failed because `tests/unit/test_audit_01.py` would be
  reformatted.

Live/network/GPU integration tests were not run as part of this review.

## Suggested Priority Order

1. Fix embedding normalization or update the documented contract.
2. Put rerankers behind the same registry/license/revision/offline policy as
   embeddings.
3. Make offline behavior scoped and test conflicting env-var cases.
4. Harden retriever input validation and append invariants.
5. Correct semantic dedup similarity reporting and degenerate-vector handling.
6. Refresh README/CLAUDE/PRD status and license text.
7. Add property tests for retrieval and clustering invariants.
