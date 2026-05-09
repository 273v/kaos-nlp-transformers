"""CrossEncoderReranker — deterministic reranking via cross-encoder models.

Uses the Rust cdylib's ``CrossEncoderBackend`` for passage-level
relevance scoring. Zero LLM cost, deterministic. Runs on CPU out of
the box (libonnxruntime via ``ort``); GPU acceleration via the
``[gpu]`` companion wheel (``ort/cuda`` EP).

Default model: ``BAAI/bge-reranker-base`` (MIT, 0.3B params,
~1 GB ONNX).

Audit history:
* KNT-501 (0.1.0a6): retired sentence-transformers + torch (~1.4 GB).
  Cross-encoder moved to fastembed.TextCrossEncoder (ONNX) on the same
  ONNX runtime as embedding.
* KNT-601 (0.2.0): retired fastembed Python wrapper. Cross-encoder
  scoring now goes through the in-tree Rust backend
  (``_rust.reranker.CrossEncoderBackend``) which calls libonnxruntime
  directly via the ``ort`` Rust crate. Same model, same scoring
  contract (sigmoid-normalized [0, 1]), free-threaded Python compatible.

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

        Audit KNT-601 (0.2.0): uses the in-tree Rust cdylib's
        ``CrossEncoderBackend`` (ort + libonnxruntime) instead of
        ``fastembed.TextCrossEncoder``. Same model, same scoring
        contract (sigmoid-normalized [0, 1]), but free-threaded
        Python compatible and one fewer Python dep in the runtime tree.

        Args:
            model_id: HuggingFace model id. Defaults to
                ``BAAI/bge-reranker-base`` (the v0 ``RERANKER_REGISTRY``
                default).
            device: Device override ('auto', 'cpu', 'cuda', etc.). GPU
                acceleration requires the ``[gpu]`` companion wheel
                (ort/cuda EP).
            settings: Optional settings override.

        Raises:
            ModelNotRegisteredError: If the model is in ``RERANKER_EXCLUDED``,
                or is not in ``RERANKER_REGISTRY`` and
                ``settings.allow_unregistered`` is false.
            BackendNotInstalledError: If the Rust cdylib was not built
                with the requested device's feature flag (e.g. ``cuda``
                without ``--features gpu``).
            ModelLoadError: If the model fails to load.
        """
        # Audit KNT-601 (0.2.0): the audit-03 KNT-201 free-threaded
        # guard was retired alongside fastembed (see embedding.py).

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
            # Audit KNT-601 (0.2.0): cross-encoder reranking goes
            # through the in-tree Rust ``ort`` backend; the legacy
            # ``"fastembed"`` value is retired.
            registered = RegisteredModel(
                model_id=target,
                revision="main",
                license="UNKNOWN",
                params_m=0,
                dim=1,
                backend="ort",
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
            "Loaded reranker %s @ %s on %s (%s) via ort (Rust)",
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

        queries = [query] * len(results)
        passages = [r.text for r in results]

        # The Rust backend's ``score`` method already applies sigmoid;
        # it returns a (n_pairs,) float32 numpy array in [0, 1]. Run it
        # in a worker thread so the heavy ort.run() doesn't block the
        # event loop.
        def _score() -> np.ndarray:
            return self._backend.score(queries, passages)

        scores = await asyncio.to_thread(_score)

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
    """Process-wide cache of loaded Rust ``CrossEncoderBackend`` instances.

    Keyed by ``(model_id, revision, device, cache_dir)`` so a registry
    SHA bump invalidates the cached backend (audit-02 KNT-104).

    Audit KNT-601 (0.2.0): was ``fastembed.TextCrossEncoder``, now the
    Rust cdylib's ``CrossEncoderBackend`` (ort + libonnxruntime). The
    revision is honored at HF Hub fetch time by the Rust loader (no
    longer baked into the Python wrapper's release).
    """
    try:
        from kaos_nlp_transformers._rust.reranker import (
            CrossEncoderBackend,  # type: ignore[import-not-found]
        )
    except ImportError as exc:
        msg = (
            "kaos_nlp_transformers._rust extension is not built. "
            "Fix: run `uv run maturin develop --release` to compile the "
            "Rust cdylib for editable installs, or reinstall the package "
            "from a released wheel. "
            "Alternative: use JudgeReranker (LLM-based) from kaos-llm-core."
        )
        raise BackendNotInstalledError(msg) from exc

    try:
        # Drop the legacy DeviceInfo plumbing — the Rust loader takes
        # the device string directly.
        _ = DeviceInfo  # silence unused import (kept for type stability)
        return CrossEncoderBackend.load(model_id, device=device, cache_dir=cache_dir)
    except Exception as exc:
        # Pass through if the binding raised one of our own typed
        # errors (mapped via rust/bindings/util.rs::map_backend_error).
        if isinstance(exc, BackendNotInstalledError | ModelLoadError | ModelNotRegisteredError):
            raise
        msg = (
            f"Failed to load reranker model {model_id!r} @ {revision} on "
            f"device {device!r}: {exc}. "
            f"Fix: try device='cpu' or model='{DEFAULT_RERANKER_MODEL}'. "
            "Alternative: verify network access on first download "
            "(KAOS_NLP_TRANSFORMERS_OFFLINE)."
        )
        raise ModelLoadError(msg) from exc


# Suppress unused-import warning on ``Any`` after the lru_cache helper
# was rewritten. Kept in the import block in case future kwargs land.
_ = Any


__all__ = ["CrossEncoderReranker"]
