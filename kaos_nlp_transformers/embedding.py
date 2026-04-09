"""Dense embedding model — fastembed backend, registry-gated.

v0 surface: one class, two methods.

    EmbeddingModel.load(model_id) -> EmbeddingModel
    EmbeddingModel.embed(texts)   -> np.ndarray of shape (N, dim)

The registry check fires before the backend is invoked. Excluded models
raise ``ModelNotRegisteredError`` with the reason. Unregistered models
raise the same error unless ``allow_unregistered`` is set in settings.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from kaos_nlp_transformers.errors import (
    BackendNotInstalledError,
    EmbeddingError,
    ModelLoadError,
    ModelNotRegisteredError,
)
from kaos_nlp_transformers.models import EXCLUDED, REGISTRY, RegisteredModel
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings


class EmbeddingModel:
    """Dense embedding inference via the fastembed backend.

    v0 ships one supported model — ``BAAI/bge-small-en-v1.5`` (33M, MIT).
    Future phases broaden the registry. The class itself is a thin
    facade over fastembed; all heavy lifting happens in ONNX Runtime.
    """

    def __init__(self, registered: RegisteredModel, _backend: Any) -> None:
        # _backend is the loaded fastembed.TextEmbedding (or future
        # sentence-transformers.SentenceTransformer); we type it as Any
        # because the backend libraries are optional deps and we don't
        # want a hard import dependency in this module.
        self._registered = registered
        self._backend = _backend

    @property
    def model_id(self) -> str:
        return self._registered.model_id

    @property
    def dim(self) -> int:
        return self._registered.dim

    @property
    def license(self) -> str:
        return self._registered.license

    @classmethod
    def load(
        cls,
        model_id: str | None = None,
        *,
        settings: KaosNLPTransformersSettings | None = None,
    ) -> EmbeddingModel:
        """Load a registered model via fastembed.

        Args:
            model_id: HF Hub model id. Defaults to
                ``settings.default_model`` (``BAAI/bge-small-en-v1.5``).
            settings: Optional settings override. When None, a fresh
                ``KaosNLPTransformersSettings`` is constructed from the
                environment.

        Raises:
            ModelNotRegisteredError: If the model is in EXCLUDED, or
                if it's not in REGISTRY and ``allow_unregistered`` is
                false.
            BackendNotInstalledError: If fastembed is not installed.
            ModelLoadError: If fastembed fails to download or load the
                model (network error, corrupt cache, missing model).
        """
        s = settings if settings is not None else KaosNLPTransformersSettings()
        target = model_id or s.default_model

        if target in EXCLUDED:
            reason = EXCLUDED[target]
            msg = (
                f"Model {target!r} is excluded from the registry: {reason}. "
                "Fix: pick a permissively-licensed alternative from "
                "kaos_nlp_transformers.models.REGISTRY. "
                "Alternative: if you have a commercial license arrangement, "
                "set KAOS_NLP_TRANSFORMERS_ALLOW_UNREGISTERED=true (use with care)."
            )
            raise ModelNotRegisteredError(msg)

        if target not in REGISTRY:
            if not s.allow_unregistered:
                available = ", ".join(sorted(REGISTRY.keys()))
                msg = (
                    f"Model {target!r} is not in the v0 registry. "
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
                dim=0,
                backend="fastembed",
                notes="unregistered",
            )
        else:
            registered = REGISTRY[target]

        backend = _load_fastembed_cached(
            model_id=registered.model_id,
            cache_dir=str(s.cache_dir) if s.cache_dir else None,
        )
        return cls(registered, backend)

    def embed(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        """Run inference and return a (N, dim) float32 array.

        Args:
            texts: Input strings. Empty list returns a (0, dim) array.
            batch_size: Inference batch size passed to the backend.

        Raises:
            EmbeddingError: On backend exception or shape mismatch.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        try:
            vecs = list(self._backend.embed(texts, batch_size=batch_size))
        except Exception as exc:
            msg = (
                f"Embedding inference failed for model {self.model_id!r}: {exc}. "
                "Fix: verify the input texts are non-empty strings. "
                "Alternative: try a smaller batch_size if memory is constrained."
            )
            raise EmbeddingError(msg) from exc

        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            msg = (
                f"Backend returned shape {arr.shape} for {len(texts)} input texts. "
                f"Fix: this is a fastembed bug — file an issue with the model id "
                f"{self.model_id!r}."
            )
            raise EmbeddingError(msg)

        if self._registered.dim and arr.shape[1] != self._registered.dim:
            msg = (
                f"Backend returned dim={arr.shape[1]} for model {self.model_id!r}, "
                f"expected dim={self._registered.dim}. "
                "Fix: the registry entry may be wrong; verify against the model card."
            )
            raise EmbeddingError(msg)

        return arr


@lru_cache(maxsize=8)
def _load_fastembed_cached(model_id: str, cache_dir: str | None):
    """Process-wide cache of loaded fastembed backends.

    Loading a fastembed model parses the ONNX file and allocates
    runtime sessions, both of which are heavyweight. Caching here
    means repeated ``EmbeddingModel.load(same_id)`` calls in the same
    process are O(1). Settings can vary per-call without invalidating
    the cache because cache_dir + model_id together fully determine
    the loaded weights.
    """
    try:
        from fastembed import TextEmbedding  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = (
            "fastembed is not installed. "
            "Fix: install it via `uv add fastembed` or `pip install fastembed`. "
            "Alternative: install kaos-nlp-transformers with the default extras "
            "which include fastembed as a hard dep."
        )
        raise BackendNotInstalledError(msg) from exc

    try:
        return TextEmbedding(model_name=model_id, cache_dir=cache_dir)
    except Exception as exc:
        msg = (
            f"Failed to load model {model_id!r} via fastembed: {exc}. "
            "Fix: verify network access on first download, or set "
            "KAOS_NLP_TRANSFORMERS_OFFLINE=false. "
            f"Alternative: pre-download via `python -m fastembed download "
            f"--model {model_id}`, or check that the model id matches a fastembed-supported model."
        )
        raise ModelLoadError(msg) from exc


__all__ = ["EmbeddingModel"]
