"""Extractive ranking primitives.

An :class:`ExtractiveRanker` scores sentence-level
:class:`kaos_nlp_core.types.Segment` instances by their salience to a
document, returning a ranked list of
:class:`ScoredSegment` records. Three scoring modes:

- **Generic** (no query): each sentence is scored by cosine similarity
  to the document centroid (mean embedding). The highest-scoring
  sentences are the most "central" — a common extractive summarization
  baseline.
- **Query-focused** (query supplied, ``use_reranker=False``): each
  sentence is scored by cosine similarity to the query embedding.
- **Reranked** (query supplied, ``use_reranker=True``): scores come
  from a cross-encoder over (query, sentence) pairs. More accurate but
  slower than centroid / dot-product scoring.

Diversity (MMR) is applied as a post-filter when ``diversify > 0``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from kaos_nlp_core.similarity import (
    cosine_one_to_many_normalized as nlp_core_cosine_one_to_many_normalized,
)
from kaos_nlp_core.similarity import (
    l2_normalize_in_place as nlp_core_l2_normalize_in_place,
)
from kaos_nlp_core.similarity import (
    mmr_select as nlp_core_mmr_select,
)
from kaos_nlp_core.types import Segment


@runtime_checkable
class Embedder(Protocol):
    def embed(
        self, texts: Iterable[str], *, batch_size: int = 32
    ) -> np.ndarray:  # pragma: no cover - protocol
        ...


@runtime_checkable
class Reranker(Protocol):
    """Score ``(query, passage)`` pairs and return floats per passage.

    The signature mirrors the in-tree
    :class:`~kaos_nlp_transformers.CrossEncoderReranker` plus any
    caller-supplied stub for tests.
    """

    def score_pairs(
        self,
        query: str,
        passages: Sequence[str],
    ) -> list[float]:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True, slots=True)
class ScoredSegment:
    """A sentence segment with an attached relevance score.

    Attributes:
        text: Verbatim sentence text.
        start: Half-open start offset in the source.
        end: Half-open end offset (exclusive).
        score: Salience score. Range depends on the scoring mode:
            cosine similarity ``[-1, 1]``, cross-encoder scores can
            be arbitrary positive floats.
        rank: Position in the returned list (``0``-based).
    """

    text: str
    start: int
    end: int
    score: float
    rank: int


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Single-pair cosine. Used by callers that import the helper directly.

    Internal hot paths inside :class:`ExtractiveRanker` route through
    :func:`kaos_nlp_core.similarity.cosine_one_to_many` (the
    Rust+NumKong SIMD path) instead.
    """
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _to_contiguous_f32(arr: np.ndarray) -> np.ndarray:
    """Coerce an array to a C-contiguous float32 view (copy only if needed).

    The Rust similarity bindings require contiguous f32 buffers; this
    helper makes the boundary explicit and minimises hidden copies.
    """
    out = arr if arr.dtype == np.float32 else arr.astype(np.float32, copy=False)
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    return out


def _mmr_select(
    embeddings: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int,
    diversify: float,
) -> list[int]:
    """Maximal Marginal Relevance selection — vectorized.

    Iteratively picks the highest-MMR candidate where:
        MMR(i) = (1 - diversify) * scores[i]
               - diversify * max(cosine(emb[i], emb[picked_j]) for j)

    Implementation notes:

    - Embeddings are L2-normalized **once** up front so cosine
      similarity collapses to a single matmul along the way.
    - The "max sim against picked" vector is maintained incrementally:
      each new pick contributes one BLAS-1 dot-product per candidate,
      then we ``np.maximum`` into the running max-sim vector. This
      brings the loop from O(k·n) Python cosine calls (= O(k·n·d)
      ops in pure Python) down to O(k·n·d) BLAS — a 50-100x speedup
      for typical ``(k=50, n=2000, d=768)`` workloads, with identical
      arithmetic semantics.

    Returns the indices of the selected candidates in pick order.
    Caps ``top_k`` at the number of candidates available so callers
    asking for more picks than candidates get a graceful list cap
    rather than a misleading infinite-loop condition.
    """
    n = len(scores)
    if n == 0:
        return []
    cap = min(top_k, n)

    # L2-normalize once. Use a small epsilon to avoid divide-by-zero
    # for any all-zero embedding row.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = embeddings / norms

    # First pick: highest raw score.
    first = int(np.argmax(scores))
    picked: list[int] = [first]
    # Running max similarity of every candidate against the picked set.
    # Initialize with the cosine against the first pick.
    max_sim = normed @ normed[first]  # shape (n,), values in [-1, 1]
    # Mask the first pick out of contention.
    available = np.ones(n, dtype=bool)
    available[first] = False

    while len(picked) < cap and available.any():
        # MMR = (1 - diversify) * scores - diversify * max_sim
        mmr = (1.0 - diversify) * scores - diversify * max_sim
        # Pick the highest-MMR among still-available candidates.
        candidate_scores = np.where(available, mmr, -np.inf)
        best = int(np.argmax(candidate_scores))
        if not np.isfinite(candidate_scores[best]):
            break
        picked.append(best)
        available[best] = False
        # Update running max-sim with the newly-picked vector.
        new_sims = normed @ normed[best]
        max_sim = np.maximum(max_sim, new_sims)
    return picked


class ExtractiveRanker:
    """Rank sentence segments by salience.

    Args:
        embedder: :class:`Embedder` for centroid / query-embedding
            scoring.
        reranker: Optional :class:`Reranker` for query-focused mode.

    Methods:
        :meth:`rank` is the single entry point.
    """

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder = embedder
        self.reranker = reranker

    def rank(
        self,
        sentences: Sequence[Segment],
        *,
        query: str | None = None,
        top_k: int | None = None,
        diversify: float = 0.0,
        use_reranker: bool = False,
    ) -> list[ScoredSegment]:
        """Score and return ``sentences`` sorted by salience.

        Args:
            sentences: Candidate sentence segments. Empty input
                returns an empty list.
            query: Optional natural-language query. When ``None`` the
                ranker uses centroid scoring; when supplied the
                ranker uses query-embedding (default) or cross-encoder
                (when ``use_reranker=True``) scoring.
            top_k: Cap on returned sentences. ``None`` returns every
                sentence sorted by score descending.
            diversify: MMR parameter in ``[0, 1]``. ``0`` (default)
                pure salience; higher values penalize redundancy.
            use_reranker: Use the cross-encoder for query scoring.
                Requires ``reranker`` to be set and ``query`` to be
                supplied.

        Returns:
            List of :class:`ScoredSegment` in rank order.

        Raises:
            ValueError: When configuration is inconsistent
                (e.g., ``use_reranker=True`` without a reranker, or
                ``diversify`` out of range).
        """
        if not (0.0 <= diversify <= 1.0):
            raise ValueError(f"diversify must be in [0, 1], got {diversify}")
        if use_reranker and (query is None or self.reranker is None):
            raise ValueError("use_reranker=True requires both ``query`` and ``reranker`` to be set")
        if not sentences:
            return []
        texts = [s.text for s in sentences]

        if use_reranker:
            assert self.reranker is not None  # narrowed by check above
            assert query is not None
            raw_scores = self.reranker.score_pairs(query, texts)
            scores = np.array(raw_scores, dtype=np.float32)
            embeddings: np.ndarray | None = None
        else:
            if self.embedder is None:
                raise ValueError("ExtractiveRanker requires an embedder for non-reranker scoring")
            embeddings = self.embedder.embed(texts)
            embeddings = _to_contiguous_f32(embeddings)
            # Both branches route through the pre-normalised fast path
            # (``cosine_one_to_many_normalized``) -- it skips the per-row
            # ``‖row‖²`` and the rsqrt finalisation when the inputs are
            # already unit-norm. ``Embedder`` contract guarantees
            # unit-norm rows for ``embeddings`` and for the query embed;
            # the centroid is the only intermediate that we have to
            # L2-normalise ourselves (mean of unit-norm vectors is not
            # unit-norm).
            if query is not None:
                query_vec = self.embedder.embed([query])[0]
                query_vec = _to_contiguous_f32(query_vec)
                scores = nlp_core_cosine_one_to_many_normalized(query_vec, embeddings)
            else:
                centroid = embeddings.mean(axis=0).astype(np.float32, copy=False)
                centroid = _to_contiguous_f32(centroid)
                # Mean of unit-norm rows is not itself unit-norm; the
                # fast path requires unit-norm inputs on BOTH sides.
                nlp_core_l2_normalize_in_place(centroid)
                scores = nlp_core_cosine_one_to_many_normalized(centroid, embeddings)

        # Determine the order. Use MMR when diversify > 0 and we have
        # embeddings; otherwise sort by raw score.
        #
        # MMR runs through the Rust-backed
        # ``kaos_nlp_core.similarity.mmr_select`` — SIMD-dispatched
        # cosine sweeps with incremental max-sim maintenance.
        # Benchmarked at ~67x numpy on 1000-row x 768-d workloads.
        cap = top_k if top_k is not None else len(sentences)
        if diversify > 0.0 and embeddings is not None:
            scores_f32 = (
                scores if scores.dtype == np.float32 else scores.astype(np.float32, copy=False)
            )
            mmr = nlp_core_mmr_select(embeddings, scores_f32, k=cap, lambda_=1.0 - diversify)
            order = [int(i) for i in mmr.indices]
        else:
            order = list(np.argsort(-scores)[:cap])

        results: list[ScoredSegment] = []
        for rank, index in enumerate(order):
            seg = sentences[index]
            results.append(
                ScoredSegment(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    score=float(scores[index]),
                    rank=rank,
                )
            )
        return results


__all__ = [
    "Embedder",
    "ExtractiveRanker",
    "Reranker",
    "ScoredSegment",
]
