"""Scale tests for :class:`SemanticChunker` with the real local embedder.

Unlike the unit-test stub embedders, these tests load the vendored
``minishlab/potion-base-8M`` model2vec model and run real embedding
inference against every paragraph of every document in the sampled
corpora.

What we validate:

1. **Real embeddings flow through the chunker without crashing** on
   ~150 long-form legal/patent documents and a 200-section USC
   sample. The embedder is the canonical
   :class:`~kaos_nlp_transformers.EmbeddingModel` — no stub.
2. **Offset round-trip holds on every chunk** produced. This is the
   same invariant as the deterministic chunker scale gate, exercised
   against embeddings that actually depend on text content.
3. **chunk_id determinism**: identical inputs produce identical
   chunks across two passes with the same embedder.
4. **Boundary placement responds to topic structure**: when the
   ``drop_threshold`` is meaningful (not 0.0), the embedding-aware
   chunker produces more chunks than the simple budget-only
   variant. This catches a silent fallback to "no semantic cut
   ever applied" regression.
5. **Throughput** is captured in a docs/benchmarks JSON for tracking.

These tests are slow (~30 s for the full sweep on a laptop, CPU
inference). Gated by ``pytest.mark.slow`` in conftest.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from kaos_nlp_core.chunking import (
    Chunk,
    validate_chunk_offsets,
)

from kaos_nlp_transformers import SemanticChunker

from .conftest import record_text

_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"


def _emit_report(name: str, payload: dict[str, Any]) -> None:
    if os.environ.get("KAOS_NLP_SCALE_NO_REPORT"):
        return
    try:
        _BENCH_DIR.mkdir(parents=True, exist_ok=True)
        (_BENCH_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Core exercise
# ---------------------------------------------------------------------------


def _exercise(
    chunker: Any,
    documents: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    total_chunks = 0
    total_docs = 0
    bad_offsets: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    chunk_token_counts: list[int] = []
    chunks_per_doc: list[int] = []
    elapsed_total = 0.0

    for doc_index, record in enumerate(documents):
        text = record_text(record)
        if not text.strip():
            continue
        doc_id = str(record.get("id", f"doc-{doc_index}"))
        total_docs += 1
        t0 = time.perf_counter()
        try:
            chunks: list[Chunk] = chunker.chunk(text, parent_id=doc_id)
        except Exception as exc:
            exceptions.append(
                {"doc_id": doc_id, "error_type": type(exc).__name__, "error": str(exc)[:200]}
            )
            continue
        elapsed_total += time.perf_counter() - t0
        chunks_per_doc.append(len(chunks))
        for chunk in chunks:
            total_chunks += 1
            chunk_token_counts.append(chunk.token_count)
            if not validate_chunk_offsets(text, chunk) and len(bad_offsets) < 5:
                bad_offsets.append(
                    {
                        "doc_id": doc_id,
                        "start": chunk.start,
                        "end": chunk.end,
                        "chunk_head": chunk.text[:60],
                    }
                )

    return {
        "label": label,
        "documents": total_docs,
        "chunks": total_chunks,
        "elapsed_seconds": round(elapsed_total, 3),
        "docs_per_sec": round(total_docs / elapsed_total, 2) if elapsed_total else 0.0,
        "chunks_per_sec": round(total_chunks / elapsed_total, 1) if elapsed_total else 0.0,
        "exceptions": exceptions,
        "bad_offset_examples": bad_offsets,
        "chunk_token_count_max": max(chunk_token_counts) if chunk_token_counts else 0,
        "chunk_token_count_mean": (
            round(sum(chunk_token_counts) / len(chunk_token_counts), 1) if chunk_token_counts else 0
        ),
        "chunks_per_doc_mean": (
            round(sum(chunks_per_doc) / len(chunks_per_doc), 1) if chunks_per_doc else 0
        ),
    }


def _assert_invariants(metrics: dict[str, Any]) -> None:
    bad = metrics["bad_offset_examples"]
    assert not bad, (
        f"{metrics['label']}: offset round-trip violated on {len(bad)} chunks. Examples: {bad[:3]}"
    )
    exceptions = metrics["exceptions"]
    assert not exceptions, (
        f"{metrics['label']}: SemanticChunker raised on {len(exceptions)} docs. "
        f"Examples: {exceptions[:3]}"
    )


# ---------------------------------------------------------------------------
# Per-corpus scale tests
# ---------------------------------------------------------------------------


def test_semantic_chunker_real_embedder_usc(
    local_embedder: Any, usc_sample: list[dict[str, Any]]
) -> None:
    chunker = SemanticChunker(
        embedder=local_embedder,
        max_tokens=1024,
        drop_threshold=0.4,
        granularity="paragraph",
    )
    metrics = _exercise(chunker, usc_sample, label="SemanticChunker@USC")
    _emit_report("semantic-chunker-scale-usc", metrics)
    _assert_invariants(metrics)
    assert metrics["chunks"] > 0


def test_semantic_chunker_real_embedder_edgar(
    local_embedder: Any, edgar_agreements: list[dict[str, Any]]
) -> None:
    chunker = SemanticChunker(
        embedder=local_embedder,
        max_tokens=1024,
        drop_threshold=0.4,
        granularity="paragraph",
    )
    metrics = _exercise(chunker, edgar_agreements, label="SemanticChunker@EDGAR")
    _emit_report("semantic-chunker-scale-edgar", metrics)
    _assert_invariants(metrics)
    assert metrics["chunks"] > 0


def test_semantic_chunker_real_embedder_patents(
    local_embedder: Any, patents: list[dict[str, Any]]
) -> None:
    chunker = SemanticChunker(
        embedder=local_embedder,
        max_tokens=1024,
        drop_threshold=0.4,
        granularity="paragraph",
    )
    metrics = _exercise(chunker, patents, label="SemanticChunker@Patents")
    _emit_report("semantic-chunker-scale-patents", metrics)
    _assert_invariants(metrics)
    assert metrics["chunks"] > 0


# ---------------------------------------------------------------------------
# Determinism with the real embedder
# ---------------------------------------------------------------------------


def test_semantic_chunker_determinism(
    local_embedder: Any, edgar_agreements: list[dict[str, Any]]
) -> None:
    """Two passes over the same documents must produce identical chunk_ids.

    Model2vec is a deterministic static lookup, so this also serves as
    a regression check that no nondeterministic step has crept in.
    """
    chunker = SemanticChunker(
        embedder=local_embedder,
        max_tokens=1024,
        drop_threshold=0.4,
    )
    sample = edgar_agreements[:10]
    drifts: list[str] = []
    for record in sample:
        text = record_text(record)
        doc_id = str(record.get("id", "edgar"))
        first = [c.chunk_id for c in chunker.chunk(text, parent_id=doc_id)]
        second = [c.chunk_id for c in chunker.chunk(text, parent_id=doc_id)]
        if first != second:
            drifts.append(doc_id)
    assert not drifts, f"SemanticChunker non-deterministic on docs: {drifts}"


# ---------------------------------------------------------------------------
# Semantic boundary placement actually responds to embeddings
# ---------------------------------------------------------------------------


def test_semantic_chunker_responds_to_threshold(
    local_embedder: Any, edgar_agreements: list[dict[str, Any]]
) -> None:
    """A meaningful ``drop_threshold`` must produce more chunks than 0.0.

    With ``drop_threshold=0.0`` no topic-shift cut ever fires; the
    chunker degenerates to a budget-only packer. With a meaningful
    threshold, embedding-aware cuts kick in. We assert the
    ``threshold > 0`` run produces strictly more chunks than the
    ``threshold == 0`` run on a 20-doc sample. This catches a silent
    fallback regression where, e.g., the threshold parameter is
    swallowed.
    """
    sample = edgar_agreements[:20]
    no_cut = SemanticChunker(
        embedder=local_embedder,
        max_tokens=10_000,  # very large, so budget never fires
        drop_threshold=0.0,
        granularity="paragraph",
    )
    semantic = SemanticChunker(
        embedder=local_embedder,
        max_tokens=10_000,
        drop_threshold=0.5,
        granularity="paragraph",
    )

    no_cut_total = 0
    semantic_total = 0
    for record in sample:
        text = record_text(record)
        if not text.strip():
            continue
        doc_id = str(record.get("id", "edgar"))
        no_cut_total += len(no_cut.chunk(text, parent_id=doc_id))
        semantic_total += len(semantic.chunk(text, parent_id=doc_id))

    assert semantic_total > no_cut_total, (
        f"SemanticChunker(drop_threshold=0.5) produced {semantic_total} chunks; "
        f"drop_threshold=0.0 produced {no_cut_total}. With real embeddings on "
        f"long contracts the threshold should produce more chunks."
    )
