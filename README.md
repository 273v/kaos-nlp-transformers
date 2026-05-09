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
layer for KAOS — a typed Python API over an in-tree Rust cdylib that
calls [`ort`](https://github.com/pykeio/ort) (libonnxruntime via Rust)
to turn text into float32 vectors and back. It ships a license-vetted
model registry, an optional cross-encoder reranker, and a semantic-dedup
level that plugs into `kaos-content`'s deduplication framework.

It is dependency-light at the BASE: the install pulls in only `numpy`,
`huggingface_hub`, and the core KAOS runtime (`kaos-core`,
`kaos-content`, `kaos-nlp-core`). **No PyTorch, no Python `fastembed`,
no Python `onnxruntime`** — the inference path is a Rust cdylib
(`kaos_nlp_transformers._rust`) shipped inside the wheel; libonnxruntime
is statically linked. Both embedding (`EmbeddingModel`) and
cross-encoder reranking (`CrossEncoderReranker`) run through the same
backend on CPU out of the box. Optional extras layer in adjacencies —
`[gpu]` for the GPU companion wheel (ort/cuda EP, NVIDIA),
`[openvino]` for Intel OpenVINO acceleration,
`[model2vec]` for the static-numpy lookup backend (~500x CPU speedup),
`[clustering]` for SciPy-backed semantic dedup, and `[mcp]` for the
MCP tool surface. Free-threaded Python (3.13t / 3.14t) is supported.

## Install

```bash
uv add kaos-nlp-transformers
# or
pip install kaos-nlp-transformers
```

`kaos-nlp-transformers` requires Python **3.13** or newer (free-threaded
3.13t / 3.14t supported). The default install is CPU-only via the Rust
ort backend. Add the extras you need:

```bash
uv add "kaos-nlp-transformers[gpu]"          # NVIDIA CUDA companion wheel (0.2.0a2)
uv add "kaos-nlp-transformers[openvino]"     # Intel CPU / GPU acceleration (0.2.0a2)
uv add "kaos-nlp-transformers[model2vec]"    # Static-numpy backend (~500x CPU)
uv add "kaos-nlp-transformers[clustering]"   # SemanticDedupLevel (scipy)
uv add "kaos-nlp-transformers[mcp]"          # MCP tool surface
```

> **0.2.0 migration note (KNT-601).** Audit KNT-601 retired the Python
> `fastembed` wrapper. Inference now goes through a Rust cdylib
> (`ort` + libonnxruntime, statically linked). Same models, same
> outputs (per-row cosine ≥ 0.9999 vs the prior backend). The
> `EmbeddingModel.load` / `EmbeddingModel.embed` /
> `CrossEncoderReranker` public API is unchanged. The `[gpu]` /
> `[openvino]` extras are no-op stubs in 0.2.0a1; the GPU companion
> wheel ships in 0.2.0a2. The `[torch]` no-op alias from KNT-501 is
> still preserved for one more cycle; removed in 0.3.0. The
> `EmbeddingRetriever` text-only retriever is deprecated in favor of
> `kaos_content.indexing.SearchableDocument` and
> `kaos_content.indexing.SearchableCorpus`; removal scheduled for 0.3.0.

Platform coverage: per-platform `cp313-abi3` wheels for Linux x86_64 +
aarch64 (manylinux + musllinux), macOS aarch64, Windows x86_64 +
aarch64. Free-threaded Python (3.13t / 3.14t) loads cleanly — no
`_check_gil_enabled` guard, no `py_rust_stemmers` SIGSEGV path.

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
| **`EmbeddingModel`** | The single entry point for inference. `EmbeddingModel.load(model_id, *, device=None, backend=None, settings=None)` resolves the registry entry, picks a backend (`fastembed` for ONNX models on CPU/GPU, `model2vec` for static lookup models), and returns an instance with an `.embed(texts, *, batch_size=32) -> np.ndarray` method. Backends are process-cached by `(model_id, revision, device, cache_dir)` so repeated `load()` calls are O(1). |
| **`RegisteredModel` / `REGISTRY` / `EXCLUDED`** | Curated, license-vetted model catalog. Each entry pins a HuggingFace Hub commit SHA (audit-01 KNT-003: revisions thread through the loader cache key). The `EXCLUDED` map names models intentionally rejected with their licensing reason — jina-v3 (CC-BY-NC), NV-Embed (CC-BY-NC), Qwen3-Embedding (MS MARCO ambiguity). v0 ships `BAAI/bge-small-en-v1.5` (33M, MIT, fastembed) plus three model2vec entries (`potion-base-8M`, `potion-base-32M`, `potion-retrieval-32M`). `potion-base-8M` is **vendored inside the wheel** (~28 MB), so it loads offline with no network. |
| **`EmbeddingRetriever`** | Brute-force cosine similarity search over a numpy matrix. `from_texts(...)` and `from_corpus(...)` factories. For corpora up to ~50K documents this is faster than FAISS overhead. Implements the `kaos_nlp_core.search.SearchHit` protocol. |
| **`CrossEncoderReranker`** | Optional second-pass reranker via `fastembed.TextCrossEncoder` (default `BAAI/bge-reranker-base`, MIT). No extra required for CPU; `[gpu]` accelerates on CUDA. Use to refine `EmbeddingRetriever` top-50 → top-10. Sigmoid-normalized scores in `[0, 1]`. |
| **`SemanticDedupLevel`** | Plug-in for `kaos-content`'s deduplication framework. Embeds documents, computes pairwise cosine distance with `scipy.spatial.distance.pdist`, and clusters with `scipy.cluster.hierarchy.fcluster`. Requires the `[clustering]` extra. |
| **`KaosNLPTransformersSettings`** | Typed settings (env prefix `KAOS_NLP_TRANSFORMERS_`): `default_model`, `default_reranker_model`, `cache_dir`, `offline`, `allow_unregistered`, `device`, `backend`, `profile`. Honors legacy `HF_HUB_OFFLINE` and `HF_HOME`. When `offline=True`, the load path sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` (audit-01 KNT-005). |
| **Device detection** | `detect_devices()` returns a `SystemDevices` snapshot (reachable accelerators + ONNX execution providers + latent GPUs the OS sees but the install can't drive). `EmbeddingModel.load(device="auto")` picks the best available; explicit `"cpu"` / `"cuda"` / `"cuda:0"` / `"openvino"` are honored. Audit-06 KNT-501 retired `mps` and `xla` alongside the torch backend. |

## CLI

`kaos-nlp-transformers` ships a `kaos-nlp-transformers` administrative
CLI (`info` subcommand) plus a `kaos-nlp-transformers-serve` MCP server
launcher that requires the `[mcp]` extra:

```bash
kaos-nlp-transformers info --json    # version + registry + device snapshot
kaos-nlp-transformers-serve          # stdio MCP server (requires [mcp])
```

## Compatibility & status

| Aspect | |
|---|---|
| **Python** | 3.13, 3.14 — GIL builds only. Free-threaded builds (3.13t / 3.14t / `Py_GIL_DISABLED`) are **not supported**: `EmbeddingModel.load` / `CrossEncoderReranker.load` raise `BackendNotInstalledError` because fastembed's transitive `py_rust_stemmers` and `tokenizers` C extensions segfault during module init without the GIL. Pending upstream `Py_GIL_DISABLED` declarations from those extensions; the guard is removed once that lands. Pure-Python `py3-none-any` wheel. |
| **OS** | Any platform with a CPython 3.13+ wheel and ONNX Runtime support — Linux x86_64 + aarch64 (manylinux), macOS x86_64 + arm64, Windows x86_64. |
| **Maturity** | Alpha. The public API is documented in `kaos_nlp_transformers.__all__`. |
| **Stability policy** | Pre-1.0: minor bumps may change behaviour. Every change is documented in [`CHANGELOG.md`](CHANGELOG.md). |
| **Test coverage** | 138 unit tests + 24 integration tests (162 total, 77% line coverage). Integration suite hits real fastembed embedding + cross-encoder reranker downloads — no mocks. GPU tests gated on the `gpu` marker; reranker live tests on `live`. |
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

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for setup, quality gates, pull request expectations, and engineering
standards. By contributing you agree to follow the
[project conduct expectations](CODE_OF_CONDUCT.md) and certify the
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
