"""CrossEncoderReranker -- deterministic reranking via cross-encoder models.

Uses sentence-transformers ``CrossEncoder`` for passage-level relevance
scoring.  Zero LLM cost, deterministic, ~5ms per 100 candidates on GPU.

Requires the ``[torch]`` extra (``torch + sentence-transformers``).
Falls back gracefully with ``BackendNotInstalledError`` if not installed.

Default model: ``BAAI/bge-reranker-base`` (MIT, 0.3B params).

Example::

    from kaos_nlp_transformers.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker.load()
    ranked = await reranker.rerank("What does Rule 10b-5 prohibit?", results)
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any

from kaos_nlp_core.retrieval.protocol import RetrievalResult
from kaos_nlp_core.retrieval.reranker import RankedResult

from kaos_nlp_transformers.device import DeviceInfo, resolve_device
from kaos_nlp_transformers.errors import BackendNotInstalledError, ModelLoadError
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

logger = logging.getLogger(__name__)

# Default reranker model — MIT license, 0.3B params, CPU-friendly.
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    """Reranker using a cross-encoder model via sentence-transformers.

    Implements the ``Reranker`` protocol from kaos-nlp-core.
    Scores are sigmoid-normalized to [0.0, 1.0].
    """

    def __init__(
        self,
        _backend: Any,
        *,
        model_id: str = DEFAULT_RERANKER_MODEL,
        device: DeviceInfo | None = None,
    ) -> None:
        self._backend = _backend
        self._model_id = model_id
        self._device = device

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> DeviceInfo | None:
        return self._device

    @classmethod
    def load(
        cls,
        model_id: str | None = None,
        *,
        device: str | None = None,
        settings: KaosNLPTransformersSettings | None = None,
    ) -> CrossEncoderReranker:
        """Load a cross-encoder model for reranking.

        Args:
            model_id: HuggingFace model id. Defaults to
                ``BAAI/bge-reranker-base``.
            device: Device override ('auto', 'cpu', 'cuda', etc.).
            settings: Optional settings override.

        Raises:
            BackendNotInstalledError: If sentence-transformers is not
                installed.
            ModelLoadError: If the model fails to load.
        """
        s = settings if settings is not None else KaosNLPTransformersSettings()
        target = model_id or DEFAULT_RERANKER_MODEL
        req_device = device or s.device
        device_info = resolve_device(req_device)

        backend = _load_cross_encoder_cached(
            model_id=target,
            device=device_info.device,
        )
        logger.info(
            "Loaded reranker %s on %s (%s)",
            target,
            device_info.device,
            device_info.name,
        )
        return cls(backend, model_id=target, device=device_info)

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """Rerank results by cross-encoder relevance scoring.

        Pairs each result's text with the query and scores via the
        cross-encoder in a single batch.  Scores are sigmoid-normalized
        to [0.0, 1.0].

        The scoring is CPU/GPU-bound, so it's dispatched to a thread
        to avoid blocking the event loop.
        """
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        raw_scores = await asyncio.to_thread(
            self._backend.predict, pairs, show_progress_bar=False
        )

        # Sigmoid normalize to [0, 1]
        import numpy as np

        scores = 1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype=np.float64)))

        ranked = [
            RankedResult(result=r, rerank_score=float(s))
            for r, s in zip(results, scores, strict=True)
        ]
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked


@lru_cache(maxsize=4)
def _load_cross_encoder_cached(model_id: str, device: str):
    """Process-wide cache of loaded cross-encoder backends."""
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = (
            "sentence-transformers is not installed. "
            "Fix: install the torch extras via "
            "`pip install kaos-nlp-transformers[torch]`. "
            "Alternative: use JudgeReranker (LLM-based) from kaos-llm-core."
        )
        raise BackendNotInstalledError(msg) from exc

    try:
        return CrossEncoder(model_id, device=device)
    except Exception as exc:
        msg = (
            f"Failed to load reranker model {model_id!r} on device "
            f"{device!r}: {exc}. "
            "Fix: verify the model id is a valid HuggingFace cross-encoder. "
            f"Alternative: try device='cpu' or model='{DEFAULT_RERANKER_MODEL}'."
        )
        raise ModelLoadError(msg) from exc


__all__ = ["CrossEncoderReranker"]
