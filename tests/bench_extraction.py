"""End-to-end throughput benchmark for :class:`ExtractiveRanker`.

Run with::

    KAOS_NLP_SCALE_FIXTURES=... \\
        uv run --no-sync pytest tests/bench_extraction.py --no-cov -s

Measures **spans/sec** through ``rank()`` -- embedding + scoring +
MMR diversification + ordering -- across the two scoring modes the
production callsite uses:

1. **Query mode** -- ``query`` is provided; each candidate scored
   against the query via the new pre-normalised
   ``cosine_one_to_many_normalized`` fast path (kaos-nlp-core >=
   0.1.0a6).
2. **Centroid mode** -- ``query`` is ``None``; candidates scored
   against their own centroid. The centroid is the mean of unit-norm
   rows so we L2-normalise it before the fast-path call.

Both modes go through the SIMD-dispatched cosine + (optional) MMR
diversification path.

The bench uses the vendored ``potion-base-8M`` model2vec embedder
(offline, ~0.5 ms / sentence). BGE ONNX coverage is opt-in via
``KAOS_NLP_BENCH_BGE=1`` (slower but matches the 384-d production
shape that retrieval consumers typically hit).
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
        for name in ("edgar_agreements.jsonl", "usc.jsonl"):
            if (path / name).exists():
                return path
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
            "KAOS_NLP_SCALE_FIXTURES not set and monorepo fallback missing. "
            "Bench requires kaos-nlp-core/tests/fixtures/*.jsonl."
        )
    return path


@pytest.fixture(scope="module")
def model2vec_embedder():
    pytest.importorskip("model2vec", reason="bench requires the [model2vec] extra")
    from kaos_nlp_transformers import EmbeddingModel

    return EmbeddingModel.load("minishlab/potion-base-8M")


def _emit(name: str, payload: dict) -> None:
    if os.environ.get("KAOS_NLP_SCALE_NO_REPORT"):
        return
    try:
        _BENCH_DIR.mkdir(parents=True, exist_ok=True)
        (_BENCH_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    except Exception:
        pass


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


def _segment_docs(docs: list[str]) -> list[list]:
    """Pre-segment each doc into sentences for the ranker.

    ExtractiveRanker.rank takes a ``Sequence[Segment]``; we mirror the
    typical caller pattern of segmenting via
    ``kaos_nlp_core.segmentation.segment_sentences`` first.
    """
    from kaos_nlp_core.segmentation import segment_sentences

    return [segment_sentences(d) for d in docs]


def _bench_ranker(
    ranker,
    docs_segments: list[list],
    *,
    query: str | None,
    k: int,
    diversify: float,
) -> dict[str, object]:
    # Warm-up
    ranker.rank(docs_segments[0], query=query, top_k=k, diversify=diversify)
    per_doc_ms: list[float] = []
    total_spans = 0
    t0 = time.perf_counter_ns()
    for sents in docs_segments:
        if not sents:
            continue
        s = time.perf_counter_ns()
        out = ranker.rank(sents, query=query, top_k=k, diversify=diversify)
        per_doc_ms.append((time.perf_counter_ns() - s) / 1e6)
        total_spans += len(out)
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
        "total_spans": total_spans,
        "spans_per_doc_avg": round(total_spans / n, 2) if n else 0.0,
    }


@pytest.mark.parametrize(
    ("corpus", "n_docs", "mode", "k", "diversify"),
    [
        ("edgar_agreements.jsonl", 50, "centroid", 10, 0.0),
        ("edgar_agreements.jsonl", 50, "query", 10, 0.0),
        ("edgar_agreements.jsonl", 50, "query", 10, 0.5),  # MMR path
        ("usc.jsonl", 100, "centroid", 20, 0.0),
    ],
)
def test_bench_extractive_ranker_model2vec(
    fixtures_dir,
    model2vec_embedder,
    corpus: str,
    n_docs: int,
    mode: str,
    k: int,
    diversify: float,
) -> None:
    """ExtractiveRanker throughput across (corpus, mode, k, diversify).

    Routes the cosine-scoring step through the kaos-nlp-core 0.1.0a6
    pre-normalised fast path. With ``diversify > 0`` the MMR
    diversification kernel also engages.
    """
    from kaos_nlp_transformers.extraction import ExtractiveRanker

    docs = _load_docs(fixtures_dir, corpus, n_docs)
    if not docs:
        pytest.skip(f"{corpus} returned 0 docs")
    docs_segments = _segment_docs(docs)

    ranker = ExtractiveRanker(embedder=model2vec_embedder)
    query = "indemnification and limitation of liability" if mode == "query" else None
    result = _bench_ranker(ranker, docs_segments, query=query, k=k, diversify=diversify)
    result["embedder"] = "model2vec/potion-base-8M"
    result["corpus"] = corpus
    result["mode"] = mode
    result["k"] = k
    result["diversify"] = diversify

    if os.environ.get("KAOS_NLP_BENCH_PRINT"):
        print(
            f"\n{corpus} n={n_docs} mode={mode} k={k} diversify={diversify} "
            f"docs/sec={result['docs_per_sec']:.1f} "
            f"ms/doc p50={result['ms_per_doc_p50']:.2f} "
            f"p95={result['ms_per_doc_p95']:.2f}"
        )

    tag = f"{corpus.replace('.jsonl', '')}-{mode}-k{k}-div{int(diversify * 10)}"
    _emit(f"extractive-ranker-throughput-{tag}", result)
