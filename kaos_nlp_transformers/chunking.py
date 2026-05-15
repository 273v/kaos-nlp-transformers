"""Semantic chunking driven by sentence/paragraph embeddings.

A :class:`SemanticChunker` produces :class:`kaos_nlp_core.chunking.Chunk`
instances whose boundaries are placed at points of *topical drift*
rather than fixed token windows. The algorithm:

1. Segment the source into paragraph (or sentence) candidate units
   using :mod:`kaos_nlp_core.segmentation`.
2. Embed each candidate via an :class:`Embedder`.
3. Compute cosine similarity between adjacent embeddings.
4. Cut where similarity drops below ``drop_threshold`` (signal of a
   topic shift), or when the running token count exceeds
   ``max_tokens`` (hard ceiling).
5. Emit one :class:`Chunk` per resulting group.

The :class:`Embedder` :func:`typing.runtime_checkable` Protocol covers
both the in-tree :class:`EmbeddingModel` and any caller-supplied stub
(useful for offline tests). The class lives in this package — not in
``kaos-nlp-core`` — because it depends on neural embeddings; the
deterministic chunkers stay in ``kaos-nlp-core``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from kaos_nlp_core._rust.chunking import semantic_pack as _rust_semantic_pack
from kaos_nlp_core.chunking import Chunk, Chunker
from kaos_nlp_core.segmentation import segment_paragraphs, segment_sentences
from kaos_nlp_core.similarity import (
    cosine_adjacent_normalized as nlp_core_cosine_adjacent_normalized,
)
from kaos_nlp_core.types import Segment


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface consumed by :class:`SemanticChunker`.

    Implementations must return ``(N, dim)`` float arrays from
    :meth:`embed`. **Returned rows must be L2-normalised (unit-norm)**
    — the SemanticChunker hot path routes through
    :func:`kaos_nlp_core.similarity.cosine_adjacent_normalized`, which
    skips the per-row norm computation on the contract that the caller
    pre-normalises. The Rust-backed
    :class:`~kaos_nlp_transformers.EmbeddingModel` is the canonical
    implementation and L2-normalises every row via the
    ``normalize_embeddings`` ONNX path and a defensive ``_l2_normalize``
    pass; tests substitute stubs that match the same shape and the
    same contract.
    """

    def embed(
        self, texts: Iterable[str], *, batch_size: int = 32
    ) -> np.ndarray:  # pragma: no cover - protocol
        ...

    def count_tokens(self, texts: Iterable[str]) -> list[int]:  # pragma: no cover - protocol
        ...


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns 0.0 for zero norms.

    Kept as a private helper for backwards-compatibility with code that
    imported it before the SemanticChunker batched cosine_adjacent
    refactor. New code should use
    ``kaos_nlp_core.similarity.cosine_one_to_many`` or ``cosine`` for
    the Rust-backed SIMD path.
    """
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticChunker(Chunker):
    """Embedding-aware chunker.

    Args:
        embedder: :class:`Embedder` providing per-unit embeddings.
        max_tokens: Hard token-budget per chunk. Default ``1024``.
        drop_threshold: Adjacent-unit cosine similarity below this
            value is treated as a topic shift and inserts a chunk
            boundary. Default ``0.5``.
        granularity: ``"paragraph"`` (default) groups paragraph units;
            ``"sentence"`` groups sentence units. Paragraph
            granularity is faster and usually sufficient; sentence
            granularity gives tighter cuts at the cost of more
            embedding calls.

    Determinism:
        The chunker is deterministic for a fixed embedder revision.
        Identical inputs produce identical
        :attr:`Chunk.chunk_id` values across runs.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        max_tokens: int = 1024,
        drop_threshold: float = 0.5,
        granularity: str = "paragraph",
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        if not (0.0 <= drop_threshold <= 1.0):
            raise ValueError(f"drop_threshold must be in [0, 1], got {drop_threshold}")
        if granularity not in {"paragraph", "sentence"}:
            raise ValueError(f"granularity must be 'paragraph' or 'sentence', got {granularity!r}")
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.drop_threshold = drop_threshold
        self.granularity = granularity

    def _segment(self, text: str) -> list[Segment]:
        if self.granularity == "paragraph":
            return segment_paragraphs(text)
        return segment_sentences(text)

    def chunk(self, text: str, *, parent_id: str | None = None) -> list[Chunk]:
        if not text:
            return []
        units = self._segment(text)
        if not units:
            return []
        embeddings = self.embedder.embed([u.text for u in units])
        if embeddings.shape[0] != len(units):
            raise RuntimeError(
                f"embedder returned {embeddings.shape[0]} vectors for {len(units)} units"
            )
        token_counts = self.embedder.count_tokens([u.text for u in units])
        return self._pack(text, parent_id, units, embeddings, token_counts)

    def _pack(
        self,
        source: str,
        parent_id: str | None,
        units: Sequence[Segment],
        embeddings: np.ndarray,
        token_counts: Sequence[int],
    ) -> list[Chunk]:
        # Adjacent-pair cosine similarity through the Rust-backed
        # ``kaos_nlp_core.similarity.cosine_adjacent_normalized`` -- the
        # pre-normalised fast path. Skips the per-row norm computation
        # and the rsqrt finalisation; pure dot + clamp. Safe because the
        # ``Embedder`` protocol contract (and our canonical
        # ``EmbeddingModel`` implementation) guarantee unit-norm rows.
        # The Rust kernel dispatches once (AVX-512F / AVX2+FMA / NEON /
        # scalar) and runs the full pair loop inside the ISA-specific
        # path. ``adj_sim[i]`` = cosine(units[i], units[i+1]);
        # length = n - 1.
        n_units = embeddings.shape[0]
        if n_units >= 2:
            embeddings_f32 = (
                embeddings if embeddings.dtype == np.float32 else embeddings.astype(np.float32)
            )
            if not embeddings_f32.flags["C_CONTIGUOUS"]:
                embeddings_f32 = np.ascontiguousarray(embeddings_f32)
            adj_sim = nlp_core_cosine_adjacent_normalized(embeddings_f32)
        else:
            adj_sim = np.empty(0, dtype=np.float32)

        # The boundary scan (budget + topic-shift cuts) runs in Rust
        # via ``kaos_nlp_core._rust.chunking.semantic_pack``. The
        # Python side only materialises Chunk objects from the
        # returned group records — slicing the source text, summing
        # original token counts, and merging metadata.
        starts = np.empty(n_units, dtype=np.uint32)
        ends = np.empty(n_units, dtype=np.uint32)
        tokens_arr = np.empty(n_units, dtype=np.uint32)
        for i, u in enumerate(units):
            starts[i] = u.start
            ends[i] = u.end
            tokens_arr[i] = token_counts[i]

        (
            group_starts,
            group_ends,
            group_unit_starts,
            group_unit_ends,
            _group_token_sums,
        ) = _rust_semantic_pack(
            starts,
            ends,
            tokens_arr,
            adj_sim,
            self.max_tokens,
            self.drop_threshold,
        )

        chunks: list[Chunk] = []
        for gs, ge, us, ue in zip(
            group_starts.tolist(),
            group_ends.tolist(),
            group_unit_starts.tolist(),
            group_unit_ends.tolist(),
            strict=True,
        ):
            indices = tuple(range(us, ue))
            chunk_text = source[gs:ge]
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start=gs,
                    end=ge,
                    parent_id=parent_id,
                    token_count=sum(token_counts[i] for i in indices),
                    metadata={
                        "chunker": "SemanticChunker",
                        "granularity": self.granularity,
                        "unit_indices": indices,
                        "units": len(indices),
                    },
                )
            )
        return chunks


__all__ = [
    "Embedder",
    "SemanticChunker",
]
