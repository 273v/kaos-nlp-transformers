"""Tests for :class:`kaos_nlp_transformers.SemanticChunker`.

Uses a deterministic stub :class:`Embedder` so the gate stays offline
and never touches the real Rust cdylib or downloaded models.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest
from kaos_nlp_core.chunking import Chunk, Chunker, validate_chunk_offsets

from kaos_nlp_transformers.chunking import Embedder, SemanticChunker


class _ConstantStub:
    """Embedder that returns a fixed embedding for every text."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        materialized = list(texts)
        if not materialized:
            return np.zeros((0, self.dim), dtype=np.float32)
        # Same vector for every text → cosine similarity == 1.0 everywhere.
        vec = np.ones(self.dim, dtype=np.float32)
        return np.stack([vec for _ in materialized])

    def count_tokens(self, texts: Iterable[str]) -> list[int]:
        return [max(1, len(t) // 4) for t in texts]


class _PatternStub:
    """Embedder that produces structured cluster-aware vectors.

    Given a list of texts, returns vectors so that texts whose
    indices fall in the same ``cluster_size`` block are similar, and
    boundaries between blocks introduce a topic shift.
    """

    def __init__(self, dim: int = 8, cluster_size: int = 2) -> None:
        self.dim = dim
        self.cluster_size = cluster_size

    def embed(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        materialized = list(texts)
        if not materialized:
            return np.zeros((0, self.dim), dtype=np.float32)
        rng = np.random.default_rng(0)
        vectors: list[np.ndarray] = []
        for index in range(len(materialized)):
            cluster_id = index // self.cluster_size
            base = np.zeros(self.dim, dtype=np.float32)
            base[cluster_id % self.dim] = 1.0
            jitter = rng.normal(scale=0.01, size=self.dim).astype(np.float32)
            vectors.append(base + jitter)
        return np.stack(vectors)

    def count_tokens(self, texts: Iterable[str]) -> list[int]:
        return [max(1, len(t) // 4) for t in texts]


_LONG_TEXT = (
    "Topic A first paragraph. More about A.\n\n"
    "Topic A second paragraph. Still A.\n\n"
    "Topic B first paragraph. Different theme.\n\n"
    "Topic B second paragraph. More on B.\n"
)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_rejects_invalid_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            SemanticChunker(embedder=_ConstantStub(), max_tokens=0)

    def test_rejects_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"drop_threshold must be in \[0, 1\]"):
            SemanticChunker(embedder=_ConstantStub(), drop_threshold=1.5)

    def test_rejects_invalid_granularity(self) -> None:
        with pytest.raises(ValueError, match="granularity must be"):
            SemanticChunker(embedder=_ConstantStub(), granularity="word")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_implements_chunker_protocol() -> None:
    chunker = SemanticChunker(embedder=_ConstantStub())
    assert isinstance(chunker, Chunker)


def test_embedder_protocol_runtime_check() -> None:
    assert isinstance(_ConstantStub(), Embedder)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_text(self) -> None:
        chunker = SemanticChunker(embedder=_ConstantStub())
        assert chunker.chunk("") == []

    def test_whitespace_only(self) -> None:
        chunker = SemanticChunker(embedder=_ConstantStub())
        # Whitespace-only text has no paragraphs.
        assert chunker.chunk("   \n\n  \n") == []


# ---------------------------------------------------------------------------
# Behavior under different stubs
# ---------------------------------------------------------------------------


class TestSemanticChunking:
    def test_uniform_embedding_groups_all(self) -> None:
        # With identical vectors, no topic shift; the chunker only
        # cuts on token budget.
        chunker = SemanticChunker(
            embedder=_ConstantStub(),
            max_tokens=10_000,
            drop_threshold=0.0,
        )
        chunks = chunker.chunk(_LONG_TEXT, parent_id="doc-1")
        # No topic shifts + huge budget → single chunk.
        assert len(chunks) == 1
        assert chunks[0].metadata["chunker"] == "SemanticChunker"

    def test_pattern_embedding_creates_boundaries(self) -> None:
        # _PatternStub flips clusters every 2 paragraphs. With a
        # drop_threshold high enough to detect the shift, we expect
        # ≥ 2 chunks.
        chunker = SemanticChunker(
            embedder=_PatternStub(cluster_size=2),
            max_tokens=10_000,
            drop_threshold=0.5,
        )
        chunks = chunker.chunk(_LONG_TEXT, parent_id="doc-1")
        assert len(chunks) >= 2

    def test_max_tokens_enforced(self) -> None:
        # Tiny budget forces each paragraph into its own chunk.
        chunker = SemanticChunker(
            embedder=_ConstantStub(),
            max_tokens=1,
            drop_threshold=0.0,
        )
        chunks = chunker.chunk(_LONG_TEXT, parent_id="doc-1")
        # 4 paragraphs → 4 chunks.
        assert len(chunks) == 4

    def test_chunks_round_trip(self) -> None:
        chunker = SemanticChunker(embedder=_ConstantStub(), max_tokens=1000)
        chunks = chunker.chunk(_LONG_TEXT, parent_id="doc-1")
        for chunk in chunks:
            assert validate_chunk_offsets(_LONG_TEXT, chunk)

    def test_chunks_are_ordered(self) -> None:
        chunker = SemanticChunker(embedder=_PatternStub(), max_tokens=1000)
        chunks = chunker.chunk(_LONG_TEXT, parent_id="doc-1")
        starts = [c.start for c in chunks]
        assert starts == sorted(starts)

    def test_deterministic_for_fixed_stub(self) -> None:
        # The PatternStub uses a seeded RNG; same source → same chunks.
        chunker = SemanticChunker(embedder=_PatternStub(), max_tokens=1000)
        first = chunker.chunk(_LONG_TEXT, parent_id="doc-1")
        second = chunker.chunk(_LONG_TEXT, parent_id="doc-1")
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_parent_id_propagates(self) -> None:
        chunker = SemanticChunker(embedder=_ConstantStub(), max_tokens=1000)
        chunks = chunker.chunk(_LONG_TEXT, parent_id="doc-77")
        assert all(c.parent_id == "doc-77" for c in chunks)

    def test_sentence_granularity(self) -> None:
        chunker = SemanticChunker(
            embedder=_ConstantStub(),
            max_tokens=10_000,
            granularity="sentence",
        )
        chunks = chunker.chunk(_LONG_TEXT, parent_id="doc-1")
        assert all(c.metadata["granularity"] == "sentence" for c in chunks)


# ---------------------------------------------------------------------------
# Returned types
# ---------------------------------------------------------------------------


def test_returned_objects_are_chunks() -> None:
    chunker = SemanticChunker(embedder=_ConstantStub(), max_tokens=1000)
    chunks = chunker.chunk(_LONG_TEXT, parent_id="doc-1")
    for chunk in chunks:
        assert isinstance(chunk, Chunk)


# ---------------------------------------------------------------------------
# Top-level exposure
# ---------------------------------------------------------------------------


def test_semantic_chunker_exposed_from_package() -> None:
    import kaos_nlp_transformers

    assert kaos_nlp_transformers.SemanticChunker is SemanticChunker
    assert "SemanticChunker" in kaos_nlp_transformers.__all__
