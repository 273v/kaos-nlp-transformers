"""Live integration tests for ``CrossEncoderReranker``.

Hits a REAL Rust ``CrossEncoderBackend`` (ort + libonnxruntime,
BAAI/bge-reranker-base) — no mocks. Audit KNT-601 (0.2.0) retired the
fastembed Python wrapper entirely; this suite is the contract that
proves the migration end-to-end.

Skips when ``KAOS_NLP_TRANSFORMERS_OFFLINE=1`` or when the Rust
extension hasn't been built (e.g., editable install without
``maturin develop``).

Marked ``@pytest.mark.integration`` and ``@pytest.mark.live`` (network).
"""

from __future__ import annotations

import os

import pytest
from kaos_nlp_core.retrieval.protocol import RetrievalResult

pytestmark = [pytest.mark.integration, pytest.mark.live]


def _skip_if_offline() -> None:
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        pytest.skip("offline mode set")


def _skip_if_no_rust_extension() -> None:
    """Audit KNT-601 (0.2.0): cross-encoder reranking now goes through
    the in-tree Rust cdylib. Guard against an editable install where
    ``maturin develop`` was never run."""
    try:
        from kaos_nlp_transformers._rust.reranker import (  # noqa: F401
            CrossEncoderBackend,
        )
    except ImportError:
        pytest.skip(
            "kaos_nlp_transformers._rust extension is not built — "
            "run `uv run maturin develop --release` first."
        )


@pytest.fixture(scope="module")
def reranker():
    """Module-scoped reranker so the model is downloaded + loaded once.

    BAAI/bge-reranker-base is ~1 GB ONNX, so loading it per test would
    inflate suite runtime by tens of seconds.
    """
    _skip_if_offline()
    _skip_if_no_rust_extension()
    from kaos_nlp_transformers import CrossEncoderReranker

    return CrossEncoderReranker.load()  # default = BAAI/bge-reranker-base


def _legal_results() -> list[RetrievalResult]:
    """Three retrieval candidates: two legal, one off-topic."""
    return [
        RetrievalResult(
            text="All disputes shall be resolved by arbitration in New York.",
            score=0.5,
            doc_id="legal-1",
        ),
        RetrievalResult(
            text="Force majeure clauses excuse performance under defined conditions.",
            score=0.5,
            doc_id="legal-2",
        ),
        RetrievalResult(
            text="The recipe calls for two cups of flour and one cup of sugar.",
            score=0.5,
            doc_id="recipe-1",
        ),
    ]


# -- Load contract ---------------------------------------------------------


async def test_load_returns_real_reranker(reranker):
    """Load returns a CrossEncoderReranker bound to the registered model id."""
    from kaos_nlp_transformers import CrossEncoderReranker

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_id == "BAAI/bge-reranker-base"


async def test_load_uses_rust_cross_encoder(reranker):
    """Audit KNT-601 (0.2.0) contract: the underlying backend is the
    in-tree Rust ``CrossEncoderBackend`` (ort + libonnxruntime), not
    fastembed's TextCrossEncoder. Audit history: KNT-501 (0.1.0a6)
    retired sentence-transformers' CrossEncoder."""
    from kaos_nlp_transformers._rust.reranker import CrossEncoderBackend

    assert isinstance(reranker._backend, CrossEncoderBackend)


# -- Rerank contract -------------------------------------------------------


async def test_rerank_orders_legal_above_recipe_for_legal_query(reranker):
    """Real cross-encoder must put legal text above recipe text for a
    legal query. This is the headline behavior the reranker exists to
    provide; if it regresses, the migration is broken."""
    ranked = await reranker.rerank(
        query="where do contract disputes get resolved?",
        results=_legal_results(),
    )

    assert len(ranked) == 3
    # The two legal docs must rank above the recipe.
    doc_order = [r.result.doc_id for r in ranked]
    recipe_pos = doc_order.index("recipe-1")
    assert recipe_pos == 2, f"recipe should rank last, got order {doc_order}"


async def test_rerank_scores_in_unit_interval(reranker):
    """Sigmoid normalization is part of the public contract — every score
    must be in [0.0, 1.0]."""
    ranked = await reranker.rerank(
        query="arbitration procedures",
        results=_legal_results(),
    )
    for r in ranked:
        assert 0.0 <= r.rerank_score <= 1.0, (
            f"score {r.rerank_score} outside [0,1] for {r.result.doc_id}"
        )


async def test_rerank_produces_strictly_ordered_output(reranker):
    """rerank_score must be monotonically non-increasing in the returned list."""
    ranked = await reranker.rerank(
        query="contract law",
        results=_legal_results(),
    )
    scores = [r.rerank_score for r in ranked]
    assert scores == sorted(scores, reverse=True), (
        f"output not sorted by score descending: {scores}"
    )


async def test_rerank_top_k_truncates(reranker):
    ranked = await reranker.rerank(
        query="legal procedure",
        results=_legal_results(),
        top_k=2,
    )
    assert len(ranked) == 2


async def test_rerank_empty_results_returns_empty(reranker):
    """Empty input must short-circuit (no backend call) and return []."""
    ranked = await reranker.rerank(query="anything", results=[])
    assert ranked == []


async def test_rerank_preserves_result_payload(reranker):
    """The original RetrievalResult must round-trip through rerank
    unchanged — only ``rerank_score`` is added."""
    inputs = _legal_results()
    by_id = {r.doc_id: r for r in inputs}

    ranked = await reranker.rerank(query="arbitration", results=inputs)

    assert {r.result.doc_id for r in ranked} == {"legal-1", "legal-2", "recipe-1"}
    for r in ranked:
        original = by_id[r.result.doc_id]
        assert r.result.text == original.text
        assert r.result.score == original.score
