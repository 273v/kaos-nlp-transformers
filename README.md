# kaos-nlp-transformers

> **Part of [Kelvin Agentic OS](https://kelvin.legal) (KAOS)** — open agentic
> infrastructure for legal work, built by
> [273 Ventures](https://273ventures.com).
> See the [full KAOS package map](https://github.com/273v) for the rest of the stack.

[![PyPI - Version](https://img.shields.io/pypi/v/kaos-nlp-transformers)](https://pypi.org/project/kaos-nlp-transformers/)
[![Python](https://img.shields.io/pypi/pyversions/kaos-nlp-transformers)](https://pypi.org/project/kaos-nlp-transformers/)
[![License](https://img.shields.io/pypi/l/kaos-nlp-transformers)](https://github.com/273v/kaos-nlp-transformers/blob/main/LICENSE)
[![CI](https://github.com/273v/kaos-nlp-transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/273v/kaos-nlp-transformers/actions/workflows/ci.yml)

`kaos-nlp-transformers` is the dense-embedding and small-model inference
layer for KAOS — a typed Python API over [`fastembed`](https://github.com/qdrant/fastembed)
(ONNX-only, no PyTorch in the BASE install) that turns text into
float32 vectors and back. It ships a license-vetted model registry, a
shared `EmbeddingRetriever` for cosine similarity search, an optional
cross-encoder reranker, and a semantic-dedup level that plugs into
`kaos-content`'s deduplication framework.

It is dependency-light at the BASE: the install pulls in
`fastembed` (Apache-2.0) and the core KAOS runtime (`kaos-core`,
`kaos-content`, `kaos-nlp-core`) plus `numpy`. Optional extras layer
in the rest of the inference stack — `[torch]` for sentence-transformers
+ PyTorch, `[gpu]` for ONNX Runtime CUDA, `[openvino]` for Intel OpenVINO,
`[clustering]` for SciPy-backed semantic dedup, and `[mcp]` for the
forthcoming MCP tool surface.

## Install

```bash
uv add kaos-nlp-transformers
# or
pip install kaos-nlp-transformers
```

`kaos-nlp-transformers` requires Python **3.13** or newer. The default
install is fastembed-only (CPU + ONNX). Add the extras you need:

```bash
uv add "kaos-nlp-transformers[torch]"        # GPU inference / non-ONNX models
uv add "kaos-nlp-transformers[gpu]"          # NVIDIA CUDA via onnxruntime-gpu
uv add "kaos-nlp-transformers[openvino]"     # Intel CPU / GPU acceleration
uv add "kaos-nlp-transformers[clustering]"   # SemanticDedupLevel (scipy)
uv add "kaos-nlp-transformers[mcp]"          # MCP tool surface (planned 0.1.0a2+)
```

Platform coverage: any platform with a CPython 3.13+ wheel and ONNX
Runtime support — the `fastembed` wheel matrix covers Linux x86_64 +
aarch64 (manylinux), macOS x86_64 + arm64, and Windows x86_64.

## Quick start

```python
import numpy as np
from kaos_nlp_transformers import EmbeddingModel

# Load the v0 default model (BAAI/bge-small-en-v1.5, 33M params, MIT).
# First call downloads and caches; subsequent calls are O(1).
model = EmbeddingModel.load("BAAI/bge-small-en-v1.5")

# Embed a small batch. Returns a float32 numpy array of shape (N, dim).
texts = [
    "Force majeure clauses excuse performance.",
    "Indemnity caps the liability of the seller.",
]
vecs = model.embed(texts)
assert vecs.shape == (2, 384) and vecs.dtype == np.float32

# Cosine similarity over the L2-normalized rows.
def cosine(a, b):
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))

print(f"sim: {cosine(vecs[0], vecs[1]):.3f}")
# sim: 0.637   (similar legal-contract topic, distinct concepts)
```

For retrieval over a corpus, build an `EmbeddingRetriever`:

```python
import asyncio

from kaos_nlp_transformers import EmbeddingRetriever

retriever = EmbeddingRetriever.from_texts(
    texts=[
        "The buyer agrees to mediation in Delaware.",
        "All disputes shall be resolved by arbitration in New York.",
        "Force majeure clauses excuse performance.",
    ],
    doc_ids=[0, 1, 2],
)
hits = asyncio.run(retriever.retrieve("where do contract disputes go?", top_k=2))
for h in hits:
    print(f"{h.score:.3f}  {h.text}")
```

## Concepts

The package is built around a small set of typed primitives.

| Concept | What it is |
|---|---|
| **`EmbeddingModel`** | The single entry point for inference. `EmbeddingModel.load(model_id, *, device=None, backend=None, settings=None)` resolves the registry entry, picks a backend (`fastembed` by default, `sentence-transformers` for GPU / non-ONNX models), and returns an instance with an `.embed(texts, *, batch_size=32) -> np.ndarray` method. Backends are process-cached by `(model_id, revision, device, cache_dir)` so repeated `load()` calls are O(1). |
| **`RegisteredModel` / `REGISTRY` / `EXCLUDED`** | Curated, license-vetted model catalog. Each entry pins a HuggingFace Hub commit SHA (audit-01 KNT-003: revisions thread through the loader cache key). The `EXCLUDED` map names models intentionally rejected with their licensing reason — jina-v3 (CC-BY-NC), NV-Embed (CC-BY-NC), Qwen3-Embedding (MS MARCO ambiguity). v0 ships one entry: `BAAI/bge-small-en-v1.5` (33M, MIT). |
| **`EmbeddingRetriever`** | Brute-force cosine similarity search over a numpy matrix. `from_texts(...)` and `from_corpus(...)` factories. For corpora up to ~50K documents this is faster than FAISS overhead. Implements the `kaos_nlp_core.search.SearchHit` protocol. |
| **`CrossEncoderReranker`** | Optional second-pass reranker (cross-encoder / pair-scoring model, `[torch]` extra). Use to refine `EmbeddingRetriever` top-50 → top-10. |
| **`SemanticDedupLevel`** | Plug-in for `kaos-content`'s deduplication framework. Embeds documents, computes pairwise cosine distance with `scipy.spatial.distance.pdist`, and clusters with `scipy.cluster.hierarchy.fcluster`. Requires the `[clustering]` extra. |
| **`KaosNLPTransformersSettings`** | Typed settings (env prefix `KAOS_NLP_TRANSFORMERS_`): `default_model`, `cache_dir`, `offline`, `allow_unregistered`, `device`, `backend`, `profile`. Honors legacy `HF_HUB_OFFLINE` and `HF_HOME`. When `offline=True`, the load path sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` (audit-01 KNT-005). |
| **Device detection** | `detect_devices()` returns a `SystemDevices` snapshot (CUDA visible? MPS? OpenVINO? counts and names) so callers can route work appropriately. `EmbeddingModel.load(device="auto")` picks the best available; explicit `"cpu"` / `"cuda"` / `"cuda:0"` / `"mps"` / `"openvino"` are honored. |

## CLI

`kaos-nlp-transformers` ships a `kaos-nlp-transformers` administrative
CLI (`info` subcommand only in 0.1.0a1) plus a placeholder
`kaos-nlp-transformers-serve` for the future MCP server (the `[mcp]`
extra will wire it up):

```bash
kaos-nlp-transformers info --json    # version + registry + device snapshot
kaos-nlp-transformers-serve          # placeholder; MCP wiring in 0.1.0a2+
```

## Compatibility & status

| Aspect | |
|---|---|
| **Python** | 3.13, 3.14 (informational matrix entries for 3.14t free-threaded and 3.15-dev). Pure-Python `py3-none-any` wheel. |
| **OS** | Any platform with a CPython 3.13+ wheel and ONNX Runtime support — Linux x86_64 + aarch64 (manylinux), macOS x86_64 + arm64, Windows x86_64. |
| **Maturity** | Alpha. The public API is documented in `kaos_nlp_transformers.__all__`. |
| **Stability policy** | Pre-1.0: minor bumps may change behaviour. Every change is documented in [`CHANGELOG.md`](CHANGELOG.md). |
| **Test coverage** | 56 Python unit tests + 7 audit-01 regression tests (63 total). Live integration tests exercise real fastembed model downloads and a sentence-transformers GPU path; gated on the `live` and `gpu` markers respectively. |
| **Type checker** | Validated with [`ty`](https://docs.astral.sh/ty/), Astral's Python type checker. |

## Companion packages

`kaos-nlp-transformers` is one of the packages in the
[Kelvin Agentic OS](https://kelvin.legal). The broader stack:

| Package | Layer | What it does |
|---|---|---|
| [`kaos-core`](https://github.com/273v/kaos-core) | Core | Foundational runtime, MCP-native types, registries, execution engine, VFS |
| [`kaos-content`](https://github.com/273v/kaos-content) | Core | Typed document AST: Block/Inline, provenance, views |
| [`kaos-mcp`](https://github.com/273v/kaos-mcp) | Bridge | FastMCP server, `kaos` management CLI, MCP resource templates |
| [`kaos-pdf`](https://github.com/273v/kaos-pdf) | Extraction | PDF → AST with provenance |
| [`kaos-web`](https://github.com/273v/kaos-web) | Extraction | Web extraction, browser automation, search, domain intelligence |
| [`kaos-office`](https://github.com/273v/kaos-office) | Extraction | DOCX / PPTX / XLSX readers + writers to AST |
| [`kaos-tabular`](https://github.com/273v/kaos-tabular) | Extraction | DuckDB-powered SQL analytics |
| [`kaos-source`](https://github.com/273v/kaos-source) | Data | Government + financial data connectors (Federal Register, eCFR, EDGAR, GovInfo, PACER, GLEIF) |
| [`kaos-llm-client`](https://github.com/273v/kaos-llm-client) | LLM | Multi-provider LLM transport |
| [`kaos-llm-core`](https://github.com/273v/kaos-llm-core) | LLM | Typed LLM programming (Signatures, Programs, Optimizers) |
| [`kaos-nlp-core`](https://github.com/273v/kaos-nlp-core) | Primitives (Rust) | High-performance NLP primitives |
| [`kaos-nlp-transformers`](https://github.com/273v/kaos-nlp-transformers) | ML | Dense embeddings + retrieval |
| [`kaos-graph`](https://github.com/273v/kaos-graph) | Primitives (Rust) | Graph algorithms + RDF/SPARQL |
| [`kaos-ml-core`](https://github.com/273v/kaos-ml-core) | Primitives (Rust) | Classical ML on the document AST |
| [`kaos-citations`](https://github.com/273v/kaos-citations) | Legal | Legal citation extraction, resolution, verification |
| [`kaos-agents`](https://github.com/273v/kaos-agents) | Agentic | Agent runtime, memory, recipes |
| [`kaos-reference`](https://github.com/273v/kaos-reference) | Sample | Reference module for module authors |

Packages depend on `kaos-core`; everything else is opt-in. Mix and match the
ones you need.

## Development

```bash
git clone https://github.com/273v/kaos-nlp-transformers
cd kaos-nlp-transformers
uv sync --group dev --extra clustering
```

Install pre-commit hooks (recommended — they run the same checks as CI on
every commit, scoped to staged files):

```bash
uvx pre-commit install
uvx pre-commit run --all-files     # one-time full sweep
```

Manual QA commands (the same set CI runs):

```bash
uv run ruff format --check kaos_nlp_transformers tests
uv run ruff check kaos_nlp_transformers tests
uv run ty check kaos_nlp_transformers tests
uv run pytest tests/unit -q
```

## Build from source

```bash
uv build
uv pip install dist/*.whl
```

## Contributing

Issues and pull requests are welcome. By contributing you certify the
[Developer Certificate of Origin v1.1](https://developercertificate.org/) —
sign every commit with `git commit -s`. Please open an issue before starting
on a non-trivial change so we can align on scope.

## Security

For security issues, **please do not file a public issue**. Report privately
via [GitHub Private Vulnerability Reporting](https://github.com/273v/kaos-nlp-transformers/security/advisories/new)
or email **security@273ventures.com**. See [SECURITY.md](SECURITY.md) for the
full disclosure policy.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Copyright 2026 [273 Ventures LLC](https://273ventures.com).
Built for [kelvin.legal](https://kelvin.legal).
