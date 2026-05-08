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
import os
from collections.abc import Iterator
from contextlib import contextmanager
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
        # Audit-03 KNT-201: refuse loudly on free-threaded Python builds.
        # fastembed pulls py_rust_stemmers (Rust/PyO3) for sparse BM25, and
        # py_rust_stemmers crashes during module init under Py_GIL_DISABLED
        # (SIGSEGV, exit 139). The sentence-transformers path has the same
        # exposure via the tokenizers + transformers stack. Better to fail
        # with a clear message at the API boundary than segfault inside
        # `import fastembed`.
        _check_gil_enabled()

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

        # Audit-02 KNT-103: scope HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE around
        # the backend construction so a process-wide env var doesn't leak
        # between two consecutive load() calls with different settings.offline
        # values. The 0.1.0a1 setdefault approach was buggy on two counts:
        #   1. setdefault refuses to override HF_HUB_OFFLINE=0 from the
        #      caller's shell, silently ignoring offline=True;
        #   2. once set to "1", it never reverted to "0" for subsequent
        #      offline=False loads in the same process.
        # The context manager snapshot/restores both vars even on backend
        # exception, so a long-running server can safely flip offline policy
        # per request.
        with _offline_env_scope(s.offline):
            if effective_backend == "sentence-transformers":
                st_backend = _load_sentence_transformers_cached(
                    model_id=registered.model_id,
                    revision=registered.revision,
                    device=device_info.device,
                    cache_dir=cache_dir,
                )
                logger.info(
                    "Loaded %s @ %s via sentence-transformers on %s (%s)",
                    registered.model_id,
                    registered.revision,
                    device_info.device,
                    device_info.name,
                )
                return cls(
                    registered,
                    st_backend,
                    device=device_info,
                    backend_name="sentence-transformers",
                )
            # fastembed pins revisions in its own model registry and does not
            # accept a runtime revision override (audit-01 KNT-003). The cache
            # key still includes registered.revision so that a registry change
            # to a different SHA invalidates the lru_cache entry, even though
            # fastembed itself loads the version baked into its release.
            fe_backend = _load_fastembed_cached(
                model_id=registered.model_id,
                revision=registered.revision,
                cache_dir=cache_dir,
                providers=_onnx_providers_for_device(device_info),
            )
            logger.info(
                "Loaded %s (registry revision %s; fastembed pins its own) on %s",
                registered.model_id,
                registered.revision,
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
                # normalize_embeddings=True keeps the sentence-transformers
                # path in lockstep with the contract; an extra explicit
                # _l2_normalize call below is then a defensive no-op for
                # this backend but guards future backend additions.
                arr = self._backend.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
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

        # Audit-02 KNT-101: enforce L2 normalization centrally regardless of
        # backend so the documented contract holds for every entry in REGISTRY,
        # not just the BGE-family fastembed default. fastembed+BGE already
        # returns unit-norm vectors so this is ~no-op there; sentence-transformers
        # was passed normalize_embeddings=True above so this is also a no-op
        # there. The cost is one np.linalg.norm + division per call (≈1µs per
        # row at 384-dim), which is far below inference cost (~1ms/row CPU).
        return _l2_normalize(arr)


# ---------------------------------------------------------------------------
# Free-threaded Python guard (audit-03 KNT-201)
# ---------------------------------------------------------------------------


def _is_free_threaded_python() -> bool:
    """True when running under a Py_GIL_DISABLED interpreter (3.13t / 3.14t).

    ``sys._is_gil_enabled()`` is part of the stable public-ish API since
    CPython 3.13 (PEP 703); on older builds without GIL-disable support
    the function does not exist, in which case we treat the interpreter
    as GIL-enabled.
    """
    import sys as _sys

    is_gil_enabled_fn = getattr(_sys, "_is_gil_enabled", None)
    if is_gil_enabled_fn is None:
        return False
    return not bool(is_gil_enabled_fn())


def _check_gil_enabled() -> None:
    """Raise :class:`BackendNotInstalledError` on free-threaded Python.

    The fastembed backend pulls ``py_rust_stemmers`` (Rust/PyO3) for its
    sparse BM25 path. As of 2026-05-08 ``py_rust_stemmers`` does not
    declare free-threaded support, and its module-init code crashes with
    SIGSEGV when loaded on Python 3.14t. The sentence-transformers
    backend has the same exposure via the ``tokenizers`` + ``transformers``
    chain.

    Audit-03 KNT-201: refuse at the API boundary so the user gets a
    clear error pointing at the upstream tracker rather than a hard
    segfault. When py_rust_stemmers / tokenizers / transformers ship
    Py_GIL_DISABLED support, this check is removed (no version pin
    needed; the upstream wheels resolve at runtime).
    """
    if _is_free_threaded_python():
        msg = (
            "kaos-nlp-transformers cannot load on a free-threaded Python "
            "build (3.13t / 3.14t / etc.). The fastembed backend's "
            "transitive dependency py_rust_stemmers crashes (SIGSEGV) "
            "during module init under Py_GIL_DISABLED, and the "
            "sentence-transformers backend has the same exposure via "
            "tokenizers + transformers. "
            "Fix: switch to the GIL-enabled build of Python 3.13 or 3.14 "
            "(`uv python install 3.14`, NOT 3.14t). "
            "Alternative: track upstream py_rust_stemmers / tokenizers "
            "free-threaded support; this guard is removed once those "
            "wheels declare Py_GIL_DISABLED."
        )
        raise BackendNotInstalledError(msg)


# ---------------------------------------------------------------------------
# Offline mode env-var scope (KNT-103)
# ---------------------------------------------------------------------------


_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


@contextmanager
def _offline_env_scope(offline: bool) -> Iterator[None]:
    """Snapshot/restore ``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE`` around
    a backend-construction block.

    When ``offline`` is True, both vars are set to ``"1"``; on exit the
    pre-call values (or absence) are restored even if the body raises.
    When ``offline`` is False, the function is a no-op — we deliberately do
    NOT force the vars to ``"0"`` because callers may have other reasons
    to keep huggingface_hub offline (firewall policy, etc.); we only
    promise that *our* offline mode does not leak across calls.
    """
    if not offline:
        yield
        return

    snapshot = {var: os.environ.get(var) for var in _OFFLINE_ENV_VARS}
    try:
        for var in _OFFLINE_ENV_VARS:
            os.environ[var] = "1"
        yield
    finally:
        for var, prior in snapshot.items():
            if prior is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prior


# ---------------------------------------------------------------------------
# L2 normalization (KNT-101)
# ---------------------------------------------------------------------------


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` with each row L2-normalized to unit length.

    All-zero rows are returned as zeros (no division by zero). Inputs that
    are already unit-norm are unchanged to within float32 epsilon — the
    division-by-norm round-trip introduces at most ~1e-7 of drift, which is
    irrelevant for cosine similarity and matches what the retriever's own
    re-normalization step produces.
    """
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Replace zero norms with 1.0 so the division leaves zero rows as-is.
    safe = np.where(norms == 0.0, 1.0, norms)
    return (arr / safe).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


_VALID_BACKENDS: frozenset[str] = frozenset({"auto", "fastembed", "sentence-transformers"})


def _resolve_backend(requested: str, device: DeviceInfo, registry_backend: str) -> str:
    """Determine which backend to use given user preference and device.

    Audit-02 KNT-107: ``requested`` is validated against the closed set
    ``{"auto", "fastembed", "sentence-transformers"}``. Unknown values
    (typos like ``"tensorflow"``) used to silently fall through to the
    auto path and pick a backend that the user did not ask for; now
    they raise ``ValueError`` with the valid set.

    Returns 'fastembed' or 'sentence-transformers'.
    """
    if requested not in _VALID_BACKENDS:
        msg = (
            f"Invalid backend {requested!r}. "
            f"Fix: use one of {sorted(_VALID_BACKENDS)}. "
            "Alternative: leave the setting unset to use the auto-detected "
            "backend (fastembed on CPU, sentence-transformers on GPU)."
        )
        raise ValueError(msg)

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
    revision: str,
    cache_dir: str | None,
    providers: tuple[str, ...] | None = None,
):
    """Process-wide cache of loaded fastembed backends.

    Loading a fastembed model parses the ONNX file and allocates
    runtime sessions, both of which are heavyweight. Caching here
    means repeated ``EmbeddingModel.load(same_id)`` calls in the same
    process are O(1).

    Note: ``revision`` is part of the cache key but NOT passed to
    fastembed. fastembed maintains its own model registry with a fixed
    revision per release; runtime revision override is not supported.
    Registry mismatches still invalidate the cache (different revision →
    different cache key → fresh load).
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
    revision: str,
    device: str,
    cache_dir: str | None = None,
):
    """Process-wide cache of loaded sentence-transformers backends.

    Keyed by (model_id, revision, device, cache_dir) so different
    devices and pinned revisions produce separate cached models.
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
        kwargs: dict[str, Any] = {"device": device, "revision": revision}
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
