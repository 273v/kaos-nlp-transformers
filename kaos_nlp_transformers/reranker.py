"""CrossEncoderReranker — deterministic reranking via cross-encoder models.

Uses ``fastembed.rerank.cross_encoder.TextCrossEncoder`` for passage-
level relevance scoring. Zero LLM cost, deterministic. Runs on CPU
out of the box (ONNX Runtime); GPU acceleration via the ``[gpu]``
extra (``onnxruntime-gpu``).

Default model: ``BAAI/bge-reranker-base`` (MIT, 0.3B params,
~1 GB ONNX) — already in fastembed's native cross-encoder registry.

Audit-06 KNT-501: this module previously used
``sentence-transformers.CrossEncoder`` and required the ``[torch]``
extra (~1.4 GB of torch + transformers + sentence-transformers).
Migrated to fastembed's TextCrossEncoder which is ONNX-only and
already in the base dep tree (fastembed is a hard dep). Same model,
same scoring contract, ~1.4 GB lighter install. The free-threaded-
Python guard stays — ``py_rust_stemmers`` and ``tokenizers`` (still
in fastembed's transitive tree) crash on ``Py_GIL_DISABLED``.

Example::

    from kaos_nlp_transformers.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker.load()
    ranked = await reranker.rerank("What does Rule 10b-5 prohibit?", results)
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

import numpy as np
from kaos_core.logging import get_logger
from kaos_nlp_core.retrieval.protocol import RetrievalResult
from kaos_nlp_core.retrieval.reranker import RankedResult

from kaos_nlp_transformers.device import DeviceInfo, resolve_device
from kaos_nlp_transformers.embedding import _offline_env_scope, _onnx_providers_for_device
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
    """Reranker using a cross-encoder model via fastembed's ONNX path.

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

        Audit-06 KNT-501: uses fastembed's TextCrossEncoder (ONNX) instead of
        sentence-transformers. Same model, same scoring contract, ~1.4 GB
        lighter install. The ``[torch]`` extra is no longer required.

        Args:
            model_id: HuggingFace model id. Defaults to
                ``BAAI/bge-reranker-base`` (the v0 ``RERANKER_REGISTRY``
                default — natively supported by fastembed).
            device: Device override ('auto', 'cpu', 'cuda', etc.). GPU
                acceleration requires the ``[gpu]`` extra
                (``onnxruntime-gpu``).
            settings: Optional settings override.

        Raises:
            ModelNotRegisteredError: If the model is in ``RERANKER_EXCLUDED``,
                or is not in ``RERANKER_REGISTRY`` and
                ``settings.allow_unregistered`` is false.
            BackendNotInstalledError: If running on a free-threaded Python
                build (audit-03 KNT-201 — py_rust_stemmers / tokenizers
                still in fastembed's transitive dep tree are not yet
                Py_GIL_DISABLED safe).
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
            # Audit-06 KNT-501: backend changed from "sentence-transformers"
            # to "fastembed" since reranking now goes through the same
            # ONNX runtime as embedding does.
            registered = RegisteredModel(
                model_id=target,
                revision="main",
                license="UNKNOWN",
                params_m=0,
                dim=1,
                backend="fastembed",
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
            "Loaded reranker %s @ %s on %s (%s) via fastembed.TextCrossEncoder",
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

        # ``rerank_pairs`` is a generator (Iterable[float]). Materialize
        # inside the worker thread so the I/O / inference work doesn't
        # block the event loop. Audit-06 KNT-501: this replaces the old
        # ``backend.predict(pairs, show_progress_bar=False)`` call.
        def _score() -> list[float]:
            return list(self._backend.rerank_pairs(pairs))

        raw_scores = await asyncio.to_thread(_score)

        # Sigmoid normalize to [0, 1]
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
    """Process-wide cache of loaded fastembed TextCrossEncoder backends.

    Keyed by ``(model_id, revision, device, cache_dir)`` so a registry SHA
    bump invalidates the cached backend (audit-02 KNT-104).

    Audit-06 KNT-501: was sentence-transformers ``CrossEncoder``, now
    fastembed ``TextCrossEncoder``. Note that ``revision`` is part of the
    cache key (so a registry SHA change invalidates) but is NOT passed
    to fastembed — fastembed's reranker registry pins its own revisions
    per-release, like the embedding-side ``_load_fastembed_cached``.

    GPU support: device.startswith("cuda") routes ``CUDAExecutionProvider``
    via the existing ``_onnx_providers_for_device`` helper. CPU is the
    default.
    """
    try:
        from fastembed.rerank.cross_encoder import (
            TextCrossEncoder,  # type: ignore[import-not-found]
        )
    except ImportError as exc:
        msg = (
            "fastembed is not installed (or its rerank submodule is missing). "
            "Fix: reinstall via `pip install kaos-nlp-transformers` — fastembed "
            "is a hard dep at the base install. "
            "Alternative: use JudgeReranker (LLM-based) from kaos-llm-core."
        )
        raise BackendNotInstalledError(msg) from exc

    try:
        # Resolve a DeviceInfo-shaped object so we can reuse the existing
        # ``_onnx_providers_for_device`` helper that the embedding loader
        # uses. We only need the ``device`` field.
        device_info = DeviceInfo(name=device, device=device, backend="fastembed")
        providers = _onnx_providers_for_device(device_info)

        kwargs: dict[str, Any] = {"model_name": model_id}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if providers:
            kwargs["providers"] = list(providers)
        return TextCrossEncoder(**kwargs)
    except Exception as exc:
        msg = (
            f"Failed to load reranker model {model_id!r} @ {revision} on "
            f"device {device!r}: {exc}. "
            "Fix: verify the model id is in fastembed's reranker registry "
            "(`fastembed.rerank.cross_encoder.TextCrossEncoder.list_supported_models()`). "
            f"Alternative: try device='cpu' or model='{DEFAULT_RERANKER_MODEL}'."
        )
        raise ModelLoadError(msg) from exc


__all__ = ["CrossEncoderReranker"]
