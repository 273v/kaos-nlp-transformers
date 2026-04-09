# kaos-nlp-transformers

Dense embeddings and small-model inference for the Kelvin Agentic OS.
The sibling package to `kaos-nlp-core` for ML workloads that need a
neural model rather than a sparse inverted index.

**Status:** v0 scaffold (proposed PRD + plan landed).

## What it does

v0 ships one class with two methods. That is the entire surface.

```python
from kaos_nlp_transformers import EmbeddingModel

model = EmbeddingModel.load("BAAI/bge-small-en-v1.5")
vecs = model.embed(["The court held that...", "The recipe calls for..."])
# vecs: np.ndarray, shape (2, 384), dtype float32, L2-normalized
```

The v0 registry contains exactly one supported model. Future phases
broaden the registry, add reranking (`BAAI/bge-reranker-v2-m3`), and
expose zero-shot NLI classification behind a `[torch]` extra.

## Why a separate package

`kaos-nlp-core` (sparse, Rust+PyO3, BM25 / MinHash / inverted index) is
the lightweight default for retrieval and search. Dense neural embeddings
require ONNX Runtime (and optionally torch + transformers), which would
roughly triple the install footprint. Splitting them into two packages
keeps the core platform lean.

`kaos-ml-core` consumes this package via its `[transformers]` extra to
produce dense feature matrices for the classification pipeline.

## Install

```bash
# Default — fastembed only (lightweight, ONNX, no torch)
uv add kaos-nlp-transformers

# With the heavy stack — needed for zero-shot NLI (Phase v1.2+)
uv add "kaos-nlp-transformers[torch]"
```

## Layout

```
kaos-nlp-transformers/
├── pyproject.toml             # hatchling build backend
├── kaos_nlp_transformers/
│   ├── __init__.py            # public API
│   ├── embedding.py           # EmbeddingModel (the v0 surface)
│   ├── models.py              # REGISTRY + EXCLUDED license audit
│   ├── settings.py            # KaosNLPTransformersSettings
│   ├── errors.py
│   ├── cli.py                 # `kaos-nlp-transformers info`
│   └── serve.py               # MCP server stub (v1.3)
└── tests/
    ├── unit/
    │   ├── test_models.py     # registry shape, license audit
    │   └── test_settings.py
    └── integration/
        └── test_embed_live.py # live download + embed
```

## License audit

The model registry is the binding contract — every entry passes a
license check at the point where it becomes loadable. Permissive only
(MIT, Apache-2.0, BSD). Hard exclusions in v0:

- `jinaai/jina-embeddings-v3` — CC-BY-NC 4.0
- `nvidia/NV-Embed-v1` / `v2` — CC-BY-NC 4.0
- `Qwen/Qwen3-Embedding-*` — MS MARCO training-data ambiguity

See `kaos_nlp_transformers.models.EXCLUDED` for the full list.

## Documentation

- **PRD** — `docs/internal/prd/kaos-nlp-transformers.md`
- **v0 plan + per-phase roadmap** — `docs/internal/plans/kaos-nlp-transformers-v0.md`
- **Consumer** — `kaos-ml-core` calls `EmbeddingModel.embed()` via its
  `embed_corpus()` shim

## Build

```bash
uv sync
ruff format kaos_nlp_transformers/ tests/
ruff check --fix kaos_nlp_transformers/ tests/
ty check kaos_nlp_transformers/ tests/
pytest tests/ -v
```

## License

LicenseRef-Proprietary © 273 Ventures LLC
