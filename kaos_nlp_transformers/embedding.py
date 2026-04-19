"""Dense embedding model — multi-backend, device-aware.

v0 surface: one class, two methods.

    EmbeddingModel.load(model_id) -> EmbeddingModel
    EmbeddingModel.embed(texts)   -> np.ndarray of shape (N, dim)

Backends:
    - **fastembed** (default for CPU): ONNX Runtime, lightweight, fast cold start.
    - **sentence-transformers** (default for GPU): PyTorch, supports CUDA,
      ROCm, MPS, XLA/TPU. Install via ``pip install kaos-nlp-transformers[torch]``.

Device selection:
    - ``device="auto"`` (default): best GPU if torch+CUDA available, else CPU.
    - ``device="cpu"``: force CPU (fastembed).
    - ``device="cuda"`` / ``device="cuda:0"`` / ``device="cuda:1"``: specific GPU.
    - ``device="mps"`` / ``device="xla"`` / ``device="openvino"``: other accelerators.

The registry check fires before the backend is invoked. Excluded models
raise ``ModelNotRegisteredError`` with the reason. Unregistered models
raise the same error unless ``allow_unregistered`` is set in settings.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

import numpy as np
from kaos_core.logging import get_logger

from kaos_nlp_transformers.device import DeviceInfo, resolve_device
from kaos_nlp_transformers.errors import (
    BackendNotInstalledError,
    EmbeddingError,
    ModelLoadError,
    ModelNotRegisteredError,
)
from kaos_nlp_transformers.models import EXCLUDED, REGISTRY, RegisteredModel
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

logger = get_logger(__name__)


class EmbeddingModel:
    """Dense embedding inference with automatic backend and device selection.

    Supports fastembed (ONNX Runtime, CPU) and sentence-transformers
    (PyTorch, GPU/CPU). Device is auto-detected by default.
    """

    def __init__(
        self,
        registered: RegisteredModel,
        _backend: Any,
        *,
        device: DeviceInfo | None = None,
        backend_name: str = "fastembed",
    ) -> None:
        self._registered = registered
        self._backend = _backend
        self._device = device
        self._backend_name = backend_name

    @property
    def model_id(self) -> str:
        return self._registered.model_id

    @property
    def dim(self) -> int:
        return self._registered.dim

    @property
    def license(self) -> str:
        return self._registered.license

    @property
    def device(self) -> DeviceInfo | None:
        """The device this model is loaded on, or None for fastembed (ONNX)."""
        return self._device

    @property
    def backend_name(self) -> str:
        """Backend in use: 'fastembed' or 'sentence-transformers'."""
        return self._backend_name

    @classmethod
    def load(
        cls,
        model_id: str | None = None,
        *,
        device: str | None = None,
        backend: str | None = None,
        settings: KaosNLPTransformersSettings | None = None,
    ) -> EmbeddingModel:
        """Load a registered model with automatic backend and device selection.

        Args:
            model_id: HF Hub model id. Defaults to
                ``settings.default_model`` (``BAAI/bge-small-en-v1.5``).
            device: Device override. One of 'auto', 'cpu', 'cuda',
                'cuda:0', 'cuda:1', 'mps', 'xla', 'openvino'. Defaults
                to ``settings.device`` (which defaults to 'auto').
            backend: Backend override. One of 'auto', 'fastembed',
                'sentence-transformers'. Defaults to ``settings.backend``.
            settings: Optional settings override. When None, a fresh
                ``KaosNLPTransformersSettings`` is constructed from the
                environment.

        Raises:
            ModelNotRegisteredError: If the model is in EXCLUDED, or
                if it's not in REGISTRY and ``allow_unregistered`` is
                false.
            BackendNotInstalledError: If the required backend is not
                installed.
            ModelLoadError: If the backend fails to download or load the
                model.
        """
        s = settings if settings is not None else KaosNLPTransformersSettings()
        target = model_id or s.default_model
        req_device = device or s.device
        req_backend = backend or s.backend

        # --- Registry gate ---
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

        # --- Resolve device ---
        device_info = resolve_device(req_device)

        # --- Resolve backend ---
        effective_backend = _resolve_backend(req_backend, device_info, registered.backend)

        # --- Load backend ---
        cache_dir = str(s.cache_dir) if s.cache_dir else None

        if effective_backend == "sentence-transformers":
            st_backend = _load_sentence_transformers_cached(
                model_id=registered.model_id,
                device=device_info.device,
                cache_dir=cache_dir,
            )
            logger.info(
                "Loaded %s via sentence-transformers on %s (%s)",
                registered.model_id,
                device_info.device,
                device_info.name,
            )
            return cls(
                registered,
                st_backend,
                device=device_info,
                backend_name="sentence-transformers",
            )
        else:
            fe_backend = _load_fastembed_cached(
                model_id=registered.model_id,
                cache_dir=cache_dir,
                providers=_onnx_providers_for_device(device_info),
            )
            logger.info(
                "Loaded %s via fastembed on %s",
                registered.model_id,
                device_info.device,
            )
            return cls(
                registered,
                fe_backend,
                device=device_info,
                backend_name="fastembed",
            )

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
            if self._backend_name == "sentence-transformers":
                arr = self._backend.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                arr = np.asarray(arr, dtype=np.float32)
            else:
                vecs = list(self._backend.embed(texts, batch_size=batch_size))
                arr = np.asarray(vecs, dtype=np.float32)
        except Exception as exc:
            msg = (
                f"Embedding inference failed for model {self.model_id!r} "
                f"on {self._backend_name}: {exc}. "
                "Fix: verify the input texts are non-empty strings. "
                "Alternative: try a smaller batch_size if memory is constrained."
            )
            raise EmbeddingError(msg) from exc

        if arr.ndim != 2 or arr.shape[0] != len(texts):
            msg = (
                f"Backend returned shape {arr.shape} for {len(texts)} input texts. "
                f"Fix: this is a {self._backend_name} bug — file an issue with the "
                f"model id {self.model_id!r}."
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


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def _resolve_backend(requested: str, device: DeviceInfo, registry_backend: str) -> str:
    """Determine which backend to use given user preference and device.

    Returns 'fastembed' or 'sentence-transformers'.
    """
    if requested == "fastembed":
        return "fastembed"
    if requested == "sentence-transformers":
        return "sentence-transformers"

    # auto: let device + registry guide the choice
    if device.device != "cpu":
        # GPU → sentence-transformers (unless registry says fastembed-only)
        return device.backend
    # CPU → use whatever the registry says
    return registry_backend


def _onnx_providers_for_device(device: DeviceInfo) -> tuple[str, ...] | None:
    """Map DeviceInfo to ONNX Runtime execution providers, or None for default."""
    if device.device.startswith("cuda"):
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    if device.device == "openvino":
        return ("OpenVINOExecutionProvider", "CPUExecutionProvider")
    return None


# ---------------------------------------------------------------------------
# Backend loaders (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _load_fastembed_cached(
    model_id: str,
    cache_dir: str | None,
    providers: tuple[str, ...] | None = None,
):
    """Process-wide cache of loaded fastembed backends.

    Loading a fastembed model parses the ONNX file and allocates
    runtime sessions, both of which are heavyweight. Caching here
    means repeated ``EmbeddingModel.load(same_id)`` calls in the same
    process are O(1).
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
        kwargs: dict[str, Any] = {"model_name": model_id}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if providers:
            kwargs["providers"] = list(providers)
        return TextEmbedding(**kwargs)
    except Exception as exc:
        msg = (
            f"Failed to load model {model_id!r} via fastembed: {exc}. "
            "Fix: verify network access on first download, or set "
            "KAOS_NLP_TRANSFORMERS_OFFLINE=false. "
            f"Alternative: pre-download via `python -m fastembed download "
            f"--model {model_id}`, or check that the model id matches a "
            "fastembed-supported model."
        )
        raise ModelLoadError(msg) from exc


@lru_cache(maxsize=8)
def _load_sentence_transformers_cached(
    model_id: str,
    device: str,
    cache_dir: str | None = None,
):
    """Process-wide cache of loaded sentence-transformers backends.

    Keyed by (model_id, device, cache_dir) so different devices produce
    separate cached models.
    """
    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        msg = (
            "sentence-transformers is not installed. "
            "Fix: install the torch extras via "
            "`pip install kaos-nlp-transformers[torch]` or "
            "`uv pip install kaos-nlp-transformers[torch]`. "
            "Alternative: use device='cpu' to stay on fastembed."
        )
        raise BackendNotInstalledError(msg) from exc

    try:
        kwargs: dict[str, Any] = {"device": device}
        if cache_dir:
            kwargs["cache_folder"] = cache_dir
        SentenceTransformer = sentence_transformers.SentenceTransformer
        return SentenceTransformer(model_id, **kwargs)
    except Exception as exc:
        msg = (
            f"Failed to load model {model_id!r} via sentence-transformers "
            f"on device {device!r}: {exc}. "
            "Fix: verify the model id is a valid HuggingFace Hub model. "
            f"Alternative: try device='cpu' or a different model."
        )
        raise ModelLoadError(msg) from exc


__all__ = ["EmbeddingModel"]
