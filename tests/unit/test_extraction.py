"""Tests for :class:`kaos_nlp_transformers.ExtractiveRanker`.

Uses stub Embedder + Reranker so the gate stays offline.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pytest
from kaos_nlp_core.types import Segment

from kaos_nlp_transformers.extraction import (
    Embedder,
    ExtractiveRanker,
    Reranker,
    ScoredSegment,
)


class _SimpleEmbedder:
    """Embedder where each text gets a one-hot vector based on its index."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        materialized = list(texts)
        out = np.zeros((len(materialized), self.dim), dtype=np.float32)
        for index, text in enumerate(materialized):
            slot = hash(text) % self.dim
            out[index, slot] = 1.0
        return out


class _CentroidFavoringEmbedder:
    """All texts get the same vector — centroid scoring is uniform."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        materialized = list(texts)
        return np.ones((len(materialized), self.dim), dtype=np.float32)


class _QueryEmbedder:
    """Embedder where each text and query gets a sequential vector.

    Use this to test query-focused scoring: configure which text the
    query is "closest" to by hashing.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self._next = 0

    def embed(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        materialized = list(texts)
        out = np.zeros((len(materialized), self.dim), dtype=np.float32)
        for index, _text in enumerate(materialized):
            # Reverse order: first text high in slot 0, last in slot N-1.
            slot = index % self.dim
            out[index, slot] = 1.0
        return out


class _LinearReranker:
    """Reranker that returns scores in input order (high to low)."""

    def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
        return [1.0 - i * 0.1 for i in range(len(passages))]


def _segments(texts: list[str]) -> list[Segment]:
    """Build Segments with fake offsets just for tests."""
    out: list[Segment] = []
    cursor = 0
    for text in texts:
        out.append(Segment(text=text, start=cursor, end=cursor + len(text), confidence=1.0))
        cursor += len(text) + 1
    return out


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


def test_embedder_protocol() -> None:
    assert isinstance(_SimpleEmbedder(), Embedder)


def test_reranker_protocol() -> None:
    assert isinstance(_LinearReranker(), Reranker)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_no_segments(self) -> None:
        ranker = ExtractiveRanker(embedder=_SimpleEmbedder())
        assert ranker.rank([]) == []


# ---------------------------------------------------------------------------
# Centroid mode
# ---------------------------------------------------------------------------


class TestCentroidMode:
    def test_uniform_embedding_returns_all_in_input_order(self) -> None:
        ranker = ExtractiveRanker(embedder=_CentroidFavoringEmbedder())
        sents = _segments(["A.", "B.", "C."])
        result = ranker.rank(sents)
        assert len(result) == 3
        # All identical scores → numpy.argsort is stable → input order preserved.
        assert [s.text for s in result] == ["A.", "B.", "C."]

    def test_top_k(self) -> None:
        ranker = ExtractiveRanker(embedder=_CentroidFavoringEmbedder())
        result = ranker.rank(_segments(["A.", "B.", "C."]), top_k=2)
        assert len(result) == 2

    def test_each_segment_is_scored(self) -> None:
        ranker = ExtractiveRanker(embedder=_SimpleEmbedder())
        sents = _segments(["A.", "B.", "C."])
        result = ranker.rank(sents)
        assert all(isinstance(s, ScoredSegment) for s in result)
        assert all(s.rank == i for i, s in enumerate(result))


# ---------------------------------------------------------------------------
# Query mode (embedding-based)
# ---------------------------------------------------------------------------


class TestQueryEmbeddingMode:
    def test_top_k_with_query(self) -> None:
        ranker = ExtractiveRanker(embedder=_SimpleEmbedder())
        sents = _segments(["alpha", "beta", "gamma"])
        result = ranker.rank(sents, query="alpha relevance", top_k=1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Reranker mode
# ---------------------------------------------------------------------------


class TestRerankerMode:
    def test_uses_reranker(self) -> None:
        ranker = ExtractiveRanker(
            embedder=_SimpleEmbedder(),
            reranker=_LinearReranker(),
        )
        sents = _segments(["A.", "B.", "C.", "D."])
        result = ranker.rank(sents, query="q", use_reranker=True)
        # Linear reranker gives index 0 the top score.
        assert result[0].text == "A."
        assert result[0].score > result[1].score

    def test_use_reranker_requires_query(self) -> None:
        ranker = ExtractiveRanker(reranker=_LinearReranker())
        with pytest.raises(ValueError, match="requires both ``query``"):
            ranker.rank(_segments(["A."]), use_reranker=True)

    def test_use_reranker_requires_reranker(self) -> None:
        ranker = ExtractiveRanker(embedder=_SimpleEmbedder())
        with pytest.raises(ValueError, match="requires both"):
            ranker.rank(_segments(["A."]), query="x", use_reranker=True)


# ---------------------------------------------------------------------------
# Diversify (MMR)
# ---------------------------------------------------------------------------


class TestDiversify:
    def test_invalid_diversify_rejected(self) -> None:
        ranker = ExtractiveRanker(embedder=_SimpleEmbedder())
        with pytest.raises(ValueError, match=r"diversify must be in \[0, 1\]"):
            ranker.rank(_segments(["A."]), diversify=2.0)

    def test_mmr_returns_top_k(self) -> None:
        ranker = ExtractiveRanker(embedder=_SimpleEmbedder())
        sents = _segments(["A.", "B.", "C.", "D."])
        result = ranker.rank(sents, top_k=2, diversify=0.5)
        assert len(result) == 2

    def test_mmr_skipped_when_no_embeddings(self) -> None:
        # When use_reranker=True embeddings is None; MMR is silently
        # skipped and raw sort is used.
        ranker = ExtractiveRanker(
            embedder=_SimpleEmbedder(),
            reranker=_LinearReranker(),
        )
        result = ranker.rank(
            _segments(["A.", "B.", "C."]),
            query="q",
            use_reranker=True,
            diversify=0.5,
        )
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class TestConfigErrors:
    def test_no_embedder_no_reranker_rejected_for_basic_mode(self) -> None:
        ranker = ExtractiveRanker()
        with pytest.raises(ValueError, match="requires an embedder"):
            ranker.rank(_segments(["A."]))


# ---------------------------------------------------------------------------
# Public exposure
# ---------------------------------------------------------------------------


def test_exposed_from_package() -> None:
    import kaos_nlp_transformers

    assert kaos_nlp_transformers.ExtractiveRanker is ExtractiveRanker
    assert kaos_nlp_transformers.ScoredSegment is ScoredSegment
    for name in ("ExtractiveRanker", "ScoredSegment"):
        assert name in kaos_nlp_transformers.__all__
