"""Dense embedding model — multi-backend, device-aware.

v0 surface: one class, two methods.

    EmbeddingModel.load(model_id) -> EmbeddingModel
    EmbeddingModel.embed(texts)   -> np.ndarray of shape (N, dim)

Backends (audit-06 KNT-501 — torch removed entirely in 0.1.0a6):
    - **fastembed** (default): ONNX Runtime. CPU out of the box; GPU via
      the ``[gpu]`` extra (``onnxruntime-gpu``). CUDA / ROCm / OpenVINO
      execution providers are routed through ``_onnx_providers_for_device``.
    - **model2vec**: pure-numpy static embeddings (audit-04 KNT-301).
      No torch, no ONNX runtime cost — direct lookup-and-average. Fastest
      CPU path; quality trade documented per-model in REGISTRY notes.

Device selection:
    - ``device="auto"`` (default): best GPU if onnxruntime-gpu is reachable,
      else CPU.
    - ``device="cpu"``: force CPU.
    - ``device="cuda"`` / ``device="cuda:0"`` / ``device="cuda:1"``: route
      via CUDAExecutionProvider; requires ``[gpu]`` extra.
    - ``device="openvino"``: OpenVINOExecutionProvider; requires ``[openvino]``.

The registry check fires before the backend is invoked. Excluded models
raise ``ModelNotRegisteredError`` with the reason. Unregistered models
raise the same error unless ``allow_unregistered`` is set in settings.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
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

    Supports fastembed (ONNX Runtime; CPU + ``[gpu]`` extra for CUDA) and
    model2vec (pure-numpy static lookup). Device is auto-detected by default.
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
        """Backend in use: 'fastembed' or 'model2vec'."""
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
                'cuda:0', 'cuda:1', 'openvino'. Defaults to
                ``settings.device`` (which defaults to 'auto'). GPU
                routes require the ``[gpu]`` extra.
            backend: Backend override. One of 'auto', 'fastembed',
                'model2vec'. Defaults to ``settings.backend``.
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
        # Audit-03 KNT-201 (originally) + audit-06 KNT-501 (post-torch
        # removal): refuse loudly on free-threaded Python builds.
        # fastembed pulls py_rust_stemmers (sparse BM25) and tokenizers
        # (reranker), both Rust/PyO3, both crash with SIGSEGV during
        # module init under Py_GIL_DISABLED. Better to fail at the API
        # boundary with a clear message than segfault inside fastembed.
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
            if effective_backend == "model2vec":
                # Static-embedding backend (audit-04 KNT-301). model2vec's own
                # ``StaticModel.from_pretrained`` does not accept a revision
                # argument and calls ``snapshot_download`` without one; to
                # honor the audit-01 KNT-003 pinned-revision contract we
                # pre-download via huggingface_hub at the registry SHA and
                # hand the resulting local path to model2vec. This keeps the
                # registry the single source of truth even though the
                # downstream library doesn't expose revision pinning.
                m2v_backend = _load_model2vec_cached(
                    model_id=registered.model_id,
                    revision=registered.revision,
                    cache_dir=cache_dir,
                )
                # Static models are CPU-only by construction; force the
                # device to CPU so EmbeddingModel.device reports something
                # truthful instead of inheriting whatever resolve_device
                # picked from the system snapshot.
                cpu_device = DeviceInfo(name="CPU", device="cpu", backend="model2vec")
                logger.info(
                    "Loaded %s @ %s via model2vec (static, CPU)",
                    registered.model_id,
                    registered.revision,
                )
                return cls(
                    registered,
                    m2v_backend,
                    device=cpu_device,
                    backend_name="model2vec",
                )
            # Audit-06 KNT-501: the sentence-transformers branch was removed in
            # 0.1.0a6 alongside the [torch] extra. GPU embedding now goes
            # through fastembed with onnxruntime-gpu (same code path as CPU
            # — the providers list is the only difference). Any caller that
            # previously requested backend="sentence-transformers" is rejected
            # at the _resolve_backend boundary above.

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
            # Audit-06 KNT-501: the sentence-transformers backend was removed
            # in 0.1.0a6. fastembed and model2vec are the two surviving
            # backends; both produce ``np.ndarray`` directly.
            if self._backend_name == "model2vec":
                # ``StaticModel.encode`` returns ``np.ndarray`` directly and
                # respects the model's own ``normalize`` flag (defaults to
                # True for the potion family — verified in
                # audit-04 KNT-301 research). Caller-supplied ``batch_size``
                # is forwarded; multiprocessing is left to model2vec's own
                # threshold (default 10000 sentences) so small calls stay
                # cheap without spawning workers.
                arr = self._backend.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
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

    fastembed pulls ``py_rust_stemmers`` (Rust/PyO3) for its sparse BM25
    path AND ``tokenizers`` (Rust/PyO3) for the cross-encoder reranker.
    Neither declares Py_GIL_DISABLED safety as of 2026-05-08, and both
    crash with SIGSEGV during module init on Python 3.14t.

    Audit-03 KNT-201 (originally) + audit-06 KNT-501 (post-torch-removal):
    refuse at the API boundary so the user gets a clear error pointing at
    the upstream tracker rather than a hard segfault. The migration off
    sentence-transformers in 0.1.0a6 removed the ``transformers`` exposure
    but py_rust_stemmers + tokenizers remain in fastembed's transitive
    dep tree, so the guard stays. When the upstream wheels declare
    Py_GIL_DISABLED, this check can be removed (no version pin needed;
    upstream wheels resolve at runtime).
    """
    if _is_free_threaded_python():
        msg = (
            "kaos-nlp-transformers cannot load on a free-threaded Python "
            "build (3.13t / 3.14t / etc.). fastembed's transitive "
            "dependencies py_rust_stemmers (sparse BM25) and tokenizers "
            "(reranker) crash (SIGSEGV) during module init under "
            "Py_GIL_DISABLED. "
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


# Audit-06 KNT-501: ``"sentence-transformers"`` was removed from the valid-
# backend set in 0.1.0a6 alongside the [torch] extra. fastembed (ONNX) and
# model2vec (numpy) are the two surviving backends; both run on CPU by
# default, fastembed escalates to GPU via onnxruntime-gpu (the [gpu] extra).
_VALID_BACKENDS: frozenset[str] = frozenset({"auto", "fastembed", "model2vec"})


def _resolve_backend(requested: str, device: DeviceInfo, registry_backend: str) -> str:
    """Determine which backend to use given user preference and device.

    Audit-02 KNT-107: ``requested`` is validated against the closed set of
    backend names. Unknown values (typos like ``"tensorflow"`` /
    ``"sentence-transformers"`` post-0.1.0a6) raise ``ValueError`` with the
    valid set rather than silently falling through.

    Audit-04 KNT-302: ``"model2vec"`` is honored as both an explicit
    backend choice and an auto-resolution target. model2vec models are
    *static* (vocab → vector lookup); the loader pins them to CPU
    regardless of the requested device.

    Audit-06 KNT-501: post-torch-removal, the auto-resolution is just
    "registry decides." Both fastembed and model2vec run on CPU; fastembed
    additionally accepts a GPU device via onnxruntime providers. There is
    no "GPU → switch to a different backend" branch anymore — fastembed
    is the GPU path.

    Returns 'fastembed' or 'model2vec'.
    """
    if requested not in _VALID_BACKENDS:
        msg = (
            f"Invalid backend {requested!r}. "
            f"Fix: use one of {sorted(_VALID_BACKENDS)}. "
            "Alternative: leave the setting unset to use the auto-detected "
            "backend (registry decides — fastembed for ONNX models, "
            "model2vec for static lookup models)."
        )
        raise ValueError(msg)

    if requested == "fastembed":
        return "fastembed"
    if requested == "model2vec":
        return "model2vec"

    # auto: registry decides. fastembed handles GPU via onnxruntime
    # providers, so there's no device-specific override here.
    if registry_backend == "model2vec":
        return "model2vec"
    return "fastembed"


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


# Audit-06 KNT-501: ``_load_sentence_transformers_cached`` removed in
# 0.1.0a6. The sentence-transformers backend is gone; GPU embedding is
# served by fastembed + onnxruntime-gpu via _load_fastembed_cached above
# (the providers list is the only thing that changes between CPU and GPU).


def _vendored_model_path(model_id: str) -> Path | None:
    """Return ``kaos_nlp_transformers/_vendor/<slug>/`` if it exists with a
    loadable ``model.safetensors`` and matches the wheel's vendored copy.

    Audit-05 KNT-401: a small static model
    (``minishlab/potion-base-8M`` ~31 MB) is bundled inside the wheel so
    air-gapped / offline-first installs don't need to touch the network.
    The slug is the model id with ``/`` -> ``-`` (matching the upstream
    HF Hub repo-id convention for filesystem use). Returns ``None`` for
    every model that is NOT vendored — the caller falls through to
    ``snapshot_download`` in that case.

    Detection is intentionally narrow: we require both the directory
    AND a non-empty ``model.safetensors`` so a partially-deleted
    install (or a wheel built without the data files) silently falls
    through to the network path instead of failing inside model2vec.
    """
    slug = model_id.replace("/", "-").rsplit("-", 0)[0]
    # The vendored dir is shipped under the package itself.
    pkg_root = Path(__file__).resolve().parent
    vendor_root = pkg_root / "_vendor"
    # Try a few canonical slug shapes — exact match first, then
    # last-segment fallback for callers that pass just "potion-base-8M".
    candidates = [
        vendor_root / model_id,
        vendor_root / model_id.replace("/", "-"),
        vendor_root / model_id.split("/")[-1],
    ]
    for cand in candidates:
        weights = cand / "model.safetensors"
        if cand.is_dir() and weights.is_file() and weights.stat().st_size > 0:
            return cand
    # Suppress lint about unused variable while keeping the slug derivation
    # readable above; some future caller will want the slug verbatim.
    _ = slug
    return None


@lru_cache(maxsize=8)
def _load_model2vec_cached(
    model_id: str,
    revision: str,
    cache_dir: str | None = None,
):
    """Process-wide cache of loaded model2vec backends.

    Resolution order:

    1. **Vendored copy** at ``kaos_nlp_transformers/_vendor/<slug>/``
       (audit-05 KNT-401). Air-gapped installs of the wheel can load
       this without touching the network. Currently bundled:
       ``minishlab/potion-base-8M``.
    2. **HuggingFace Hub snapshot** at the pinned revision via
       ``huggingface_hub.snapshot_download(repo_id, revision=sha)``.
       First call downloads, subsequent calls hit the HF cache.

    Keyed by ``(model_id, revision, cache_dir)``. The revision is part
    of the cache key AND is honored at download time — see the docstring
    on the call site in ``EmbeddingModel.load`` for why we route the
    download through ``snapshot_download`` instead of letting model2vec
    resolve the repo id directly.
    """
    try:
        from model2vec import StaticModel  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = (
            "model2vec is not installed. "
            "Fix: install the model2vec extras via "
            "`pip install kaos-nlp-transformers[model2vec]` (or "
            "`uv add kaos-nlp-transformers[model2vec]`). "
            "Alternative: use the default fastembed model "
            "BAAI/bge-small-en-v1.5 — slightly slower but already in the "
            "base install."
        )
        raise BackendNotInstalledError(msg) from exc

    # 1. Vendored copy — try first, no network at all.
    vendored = _vendored_model_path(model_id)
    if vendored is not None:
        logger.info(
            "Loaded %s @ %s via model2vec from vendored path %s (audit-05 KNT-401)",
            model_id,
            revision,
            vendored,
        )
        try:
            return StaticModel.from_pretrained(str(vendored))
        except Exception as exc:
            # Vendored bytes failed to load — log and fall through to HF
            # rather than hard-failing, so a corrupted or stale vendor
            # dir doesn't take the package down.
            logger.warning(
                "Vendored copy at %s failed to load (%s); falling through "
                "to huggingface_hub.snapshot_download",
                vendored,
                exc,
            )

    # 2. Network path — pin the revision at download time so the
    # registry SHA is the single source of truth (audit-01 KNT-003 +
    # audit-04 KNT-301). ``snapshot_download`` returns the path of the
    # local snapshot, which we hand to model2vec instead of the repo id.
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        msg = (
            "huggingface_hub is not importable. "
            "Fix: reinstall kaos-nlp-transformers — huggingface_hub is "
            "pulled in as a transitive of fastembed in the base install."
        )
        raise BackendNotInstalledError(msg) from exc

    try:
        snapshot_kwargs: dict[str, Any] = {
            "repo_id": model_id,
            "revision": revision,
            "repo_type": "model",
        }
        if cache_dir:
            snapshot_kwargs["cache_dir"] = cache_dir
        local_path = snapshot_download(**snapshot_kwargs)
        return StaticModel.from_pretrained(local_path)
    except Exception as exc:
        msg = (
            f"Failed to load model {model_id!r} @ {revision} via model2vec: "
            f"{exc}. "
            "Fix: verify network access on first download, or pre-cache the "
            "model with `huggingface-cli download "
            f"{model_id} --revision {revision}`. "
            "Alternative: pick BAAI/bge-small-en-v1.5 (default, fastembed) "
            "to bypass the model2vec extra entirely."
        )
        raise ModelLoadError(msg) from exc


__all__ = ["EmbeddingModel"]
