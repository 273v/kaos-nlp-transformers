"""End-to-end throughput benchmark for :class:`SemanticChunker`.

Run with::

    KAOS_NLP_SCALE_FIXTURES=... \\
        uv run --no-sync pytest tests/bench_semantic_chunker.py --no-cov -s

Measures **docs/sec** -- the full pipeline (embed + cosine_adjacent +
semantic_pack + Chunk materialisation) -- across corpora that mirror
the production callsite (kelvin-* document ingest).

The bench uses the vendored ``potion-base-8M`` model2vec embedder for
offline runs (~0.5 ms / sentence on a modern x86 core, no GPU needed)
and the BGE ONNX path when ``KAOS_NLP_BENCH_BGE=1`` is set (slower,
~2-5 ms / sentence depending on hardware). The model2vec path is the
canonical default for kelvin-training and most kaos-content callers;
BGE coverage is opt-in because the wheel download alone is ~30 MB.

Workload shapes target real ingest sizes:

* ``n_docs=10`` -- single-batch ingest (REPL / dev iteration).
* ``n_docs=100`` -- typical "process this folder" run.
* ``n_docs=1000`` -- the largest batch we measure here; kelvin-training's
  end-of-day index rebuild sits around this size on a per-customer
  basis.

The output is a JSON report under ``docs/benchmarks/`` with median
docs/sec + ms/doc + total chunks emitted, plus the embedder backend
identifier so we can track drift when the upstream model changes.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_BENCH_DIR = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"


def _resolve_fixtures_dir() -> Path | None:
    env = os.environ.get("KAOS_NLP_SCALE_FIXTURES")
    if env:
        path = Path(env).expanduser().resolve()
        for name in ("edgar_agreements.jsonl", "usc.jsonl", "patents.jsonl"):
            if (path / name).exists():
                return path
    # Monorepo fallback: kaos-nlp-core/tests/fixtures sits beside us.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "kaos-modules" / "kaos-nlp-core" / "tests" / "fixtures"
        if (candidate / "edgar_agreements.jsonl").exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def fixtures_dir() -> Path:
    path = _resolve_fixtures_dir()
    if path is None:
        pytest.skip(
            "KAOS_NLP_SCALE_FIXTURES not set and monorepo fallback path "
            "kaos-modules/kaos-nlp-core/tests/fixtures missing. Bench "
            "requires the vendored EDGAR / USC / patents JSONL files."
        )
    return path


@pytest.fixture(scope="module")
def model2vec_embedder():
    """Load the vendored potion-base-8M model2vec embedder once per module.

    Returns ``None`` (causing tests to skip) if the ``[model2vec]`` extra
    isn't installed -- the bench is opt-in.
    """
    pytest.importorskip(
        "model2vec",
        reason="bench requires the [model2vec] extra for offline embedding",
    )
    from kaos_nlp_transformers import EmbeddingModel

    # potion-base-8M ships vendored under kaos_nlp_transformers/_vendor/.
    # EmbeddingModel.load auto-detects it from the registry.
    return EmbeddingModel.load("minishlab/potion-base-8M")


def _load_docs(fixtures_dir: Path, name: str, limit: int) -> list[str]:
    path = fixtures_dir / name
    docs: list[str] = []
    with path.open() as f:
        for line in f:
            if len(docs) >= limit:
                break
            row = json.loads(line)
            text = row.get("text") or row.get("body") or ""
            if text:
                docs.append(text)
    return docs


def _emit(name: str, payload: dict) -> None:
    if os.environ.get("KAOS_NLP_SCALE_NO_REPORT"):
        return
    try:
        _BENCH_DIR.mkdir(parents=True, exist_ok=True)
        (_BENCH_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    except Exception:
        pass


def _bench_chunker(chunker, docs: list[str]) -> dict[str, object]:
    """Run ``chunker.chunk(doc)`` over the doc list; return timing dict."""
    # Warm-up: first call pays model-load + dispatch overhead.
    chunker.chunk(docs[0])
    per_doc_ms: list[float] = []
    total_chunks = 0
    t0 = time.perf_counter_ns()
    for doc in docs:
        s = time.perf_counter_ns()
        chunks = chunker.chunk(doc)
        per_doc_ms.append((time.perf_counter_ns() - s) / 1e6)
        total_chunks += len(chunks)
    elapsed_s = (time.perf_counter_ns() - t0) / 1e9
    per_doc_ms.sort()
    n = len(per_doc_ms)
    return {
        "n_docs": n,
        "elapsed_s": round(elapsed_s, 4),
        "docs_per_sec": round(n / elapsed_s, 2) if elapsed_s > 0 else 0.0,
        "ms_per_doc_p50": round(per_doc_ms[n // 2], 3),
        "ms_per_doc_p95": round(per_doc_ms[min(n - 1, int(n * 0.95))], 3),
        "ms_per_doc_max": round(per_doc_ms[-1], 3),
        "total_chunks": total_chunks,
        "chunks_per_doc_avg": round(total_chunks / n, 2) if n else 0.0,
    }


@pytest.mark.parametrize(
    ("corpus", "n_docs"),
    [
        ("edgar_agreements.jsonl", 10),
        ("edgar_agreements.jsonl", 100),
        ("usc.jsonl", 100),
        ("patents.jsonl", 100),
    ],
)
def test_bench_semantic_chunker_model2vec(
    fixtures_dir, model2vec_embedder, corpus: str, n_docs: int
) -> None:
    """SemanticChunker throughput on the model2vec (potion-base-8M) path.

    This is the production-default embedder for kelvin-training and the
    fast-path measured here: every chunk's adjacent-pair cosine routes
    through ``kaos_nlp_core.similarity.cosine_adjacent_normalized``
    (kaos-nlp-core >= 0.1.0a6), which runs at AVX-512F / AVX2+FMA /
    NEON / scalar depending on host CPU.

    Asserts only the bench actually completes; the JSON artifact is the
    source of truth for tracking throughput drift across releases.
    """
    from kaos_nlp_transformers.chunking import SemanticChunker

    docs = _load_docs(fixtures_dir, corpus, n_docs)
    if not docs:
        pytest.skip(f"{corpus} returned 0 docs")

    chunker = SemanticChunker(
        embedder=model2vec_embedder,
        max_tokens=1024,
        drop_threshold=0.5,
        granularity="paragraph",
    )
    result = _bench_chunker(chunker, docs)
    result["embedder"] = "model2vec/potion-base-8M"
    result["corpus"] = corpus
    result["granularity"] = "paragraph"
    if os.environ.get("KAOS_NLP_BENCH_PRINT"):
        print(
            f"\n{corpus} n={n_docs} "
            f"docs/sec={result['docs_per_sec']:.1f} "
            f"ms/doc p50={result['ms_per_doc_p50']:.2f} "
            f"p95={result['ms_per_doc_p95']:.2f} "
            f"avg_chunks={result['chunks_per_doc_avg']:.1f}"
        )
    _emit(f"semantic-chunker-throughput-{corpus.replace('.jsonl', '')}-n{n_docs}", result)
