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
import importlib
from functools import lru_cache
from typing import Any

from kaos_core.logging import get_logger
from kaos_nlp_core.retrieval.protocol import RetrievalResult
from kaos_nlp_core.retrieval.reranker import RankedResult

from kaos_nlp_transformers.device import DeviceInfo, resolve_device
from kaos_nlp_transformers.embedding import _offline_env_scope
from kaos_nlp_transformers.errors import (
    BackendNotInstalledError,
    ModelLoadError,
    ModelNotRegisteredError,
)
from kaos_nlp_transformers.models import RERANKER_EXCLUDED, RERANKER_REGISTRY, RegisteredModel
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

logger = get_logger(__name__)

# Default reranker model — MIT license, 0.3B params, CPU-friendly. Pinned
# revision lives in RERANKER_REGISTRY (audit-02 KNT-104). Derives from the
# settings field default so a single env-var override
# (``KAOS_NLP_TRANSFORMERS_DEFAULT_RERANKER_MODEL``) updates every call site
# that does not pass an explicit ``model_id``. See
# docs/python/checklists/03-implement.md (settings-driven defaults).
DEFAULT_RERANKER_MODEL: str = KaosNLPTransformersSettings.model_fields[
    "default_reranker_model"
].default


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

        Audit-02 KNT-104: routes through the reranker registry for license /
        revision / offline policy parity with embeddings.

        Args:
            model_id: HuggingFace model id. Defaults to
                ``BAAI/bge-reranker-base`` (the v0 ``RERANKER_REGISTRY``
                default).
            device: Device override ('auto', 'cpu', 'cuda', etc.).
            settings: Optional settings override.

        Raises:
            ModelNotRegisteredError: If the model is in ``RERANKER_EXCLUDED``,
                or is not in ``RERANKER_REGISTRY`` and
                ``settings.allow_unregistered`` is false.
            BackendNotInstalledError: If sentence-transformers is not
                installed, OR if running on a free-threaded Python build
                (audit-03 KNT-201 — tokenizers / transformers chain not
                yet Py_GIL_DISABLED safe).
            ModelLoadError: If the model fails to load.
        """
        # Audit-03 KNT-201: same guard as EmbeddingModel.load — refuse
        # on free-threaded Python before attempting any backend import.
        from kaos_nlp_transformers.embedding import _check_gil_enabled

        _check_gil_enabled()

        s = settings if settings is not None else KaosNLPTransformersSettings()
        target = model_id or DEFAULT_RERANKER_MODEL

        # Registry gate, mirroring EmbeddingModel.load (audit-02 KNT-104).
        if target in RERANKER_EXCLUDED:
            reason = RERANKER_EXCLUDED[target]
            msg = (
                f"Reranker model {target!r} is excluded from the registry: "
                f"{reason}. Fix: pick a permissively-licensed alternative "
                "from kaos_nlp_transformers.models.RERANKER_REGISTRY. "
                "Alternative: if you have a commercial license arrangement, "
                "set KAOS_NLP_TRANSFORMERS_ALLOW_UNREGISTERED=true (use with care)."
            )
            raise ModelNotRegisteredError(msg)

        if target not in RERANKER_REGISTRY:
            if not s.allow_unregistered:
                available = ", ".join(sorted(RERANKER_REGISTRY.keys()))
                msg = (
                    f"Reranker model {target!r} is not in the v0 registry. "
                    f"Fix: choose one of [{available}]. "
                    "Alternative: set KAOS_NLP_TRANSFORMERS_ALLOW_UNREGISTERED=true "
                    "to bypass the registry (you are responsible for license review)."
                )
                raise ModelNotRegisteredError(msg)
            registered = RegisteredModel(
                model_id=target,
                revision="main",
                license="UNKNOWN",
                params_m=0,
                dim=1,
                backend="sentence-transformers",
                notes="unregistered reranker",
            )
        else:
            registered = RERANKER_REGISTRY[target]

        req_device = device or s.device
        device_info = resolve_device(req_device)
        cache_dir = str(s.cache_dir) if s.cache_dir else None

        # Audit-02 KNT-103: scoped offline env vars around backend
        # construction so the reranker honors KAOS_NLP_TRANSFORMERS_OFFLINE
        # exactly the way EmbeddingModel does.
        with _offline_env_scope(s.offline):
            backend = _load_cross_encoder_cached(
                model_id=registered.model_id,
                revision=registered.revision,
                device=device_info.device,
                cache_dir=cache_dir,
            )
        logger.info(
            "Loaded reranker %s @ %s on %s (%s)",
            registered.model_id,
            registered.revision,
            device_info.device,
            device_info.name,
        )
        return cls(backend, model_id=registered.model_id, device=device_info)

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
        raw_scores = await asyncio.to_thread(self._backend.predict, pairs, show_progress_bar=False)

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
def _load_cross_encoder_cached(
    model_id: str,
    revision: str,
    device: str,
    cache_dir: str | None = None,
):
    """Process-wide cache of loaded cross-encoder backends.

    Keyed by (model_id, revision, device, cache_dir) so a registry SHA
    bump invalidates the cached backend (audit-02 KNT-104).
    """
    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        msg = (
            "sentence-transformers is not installed. "
            "Fix: install the torch extras via "
            "`pip install kaos-nlp-transformers[torch]`. "
            "Alternative: use JudgeReranker (LLM-based) from kaos-llm-core."
        )
        raise BackendNotInstalledError(msg) from exc

    try:
        CrossEncoder = sentence_transformers.CrossEncoder
        kwargs: dict[str, Any] = {"device": device, "revision": revision}
        if cache_dir:
            kwargs["cache_folder"] = cache_dir
        return CrossEncoder(model_id, **kwargs)
    except Exception as exc:
        msg = (
            f"Failed to load reranker model {model_id!r} @ {revision} on "
            f"device {device!r}: {exc}. "
            "Fix: verify the model id is a valid HuggingFace cross-encoder. "
            f"Alternative: try device='cpu' or model='{DEFAULT_RERANKER_MODEL}'."
        )
        raise ModelLoadError(msg) from exc


__all__ = ["CrossEncoderReranker"]
