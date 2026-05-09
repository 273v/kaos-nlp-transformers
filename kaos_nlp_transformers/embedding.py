"""Dense embedding model — multi-backend, device-aware.

v0 surface: one class, two methods.

    EmbeddingModel.load(model_id) -> EmbeddingModel
    EmbeddingModel.embed(texts)   -> np.ndarray of shape (N, dim)

Backends (audit KNT-601 — fastembed retired in 0.2.0):
    - **ort** (default): the in-tree Rust cdylib (``_rust.embedding``)
      calls libonnxruntime via the ``ort`` crate. CPU out of the box;
      GPU via the ``[gpu]`` companion wheel (ort/cuda EP).
    - **model2vec**: pure-numpy static embeddings (audit-04 KNT-301).
      No torch, no ONNX runtime cost — direct lookup-and-average. Fastest
      CPU path; quality trade documented per-model in REGISTRY notes.

Audit history:
    - KNT-501 (0.1.0a6): retired sentence-transformers + torch.
    - KNT-601 (0.2.0): retired fastembed Python wrapper. Same models,
      same outputs (cosine ≥ 0.9999 vs frozen reference vectors), but
      free-threaded Python compatible and Rust-native.

Device selection:
    - ``device="auto"`` (default): best GPU if available + GPU wheel
      installed, else CPU.
    - ``device="cpu"``: force CPU.
    - ``device="cuda"`` / ``device="cuda:0"`` / ``device="cuda:1"``:
      requires ``kaos-nlp-transformers-gpu`` companion wheel
      (ort/cuda EP).
    - ``device="openvino"``: requires ``[openvino]`` companion wheel.

The registry check fires before the backend is invoked. Excluded models
raise ``ModelNotRegisteredError`` with the reason. Unregistered models
raise the same error unless ``allow_unregistered`` is set in settings.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from collections.abc import Iterable, Iterator
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
        """Backend in use: ``'ort'`` or ``'model2vec'``."""
        return self._backend_name

    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length (in tokens) the underlying tokenizer
        applies as a truncation cap.

        For ``model2vec`` this is the static-lookup model's vocabulary
        cap (commonly very large; usually irrelevant since model2vec
        averages over present tokens). For ``ort`` this is the
        registry's ``max_seq_len`` (e.g. 512 for BAAI/bge-small-en-v1.5).

        Downstream consumers (kaos-content's ``EmbeddingChunker``) read
        this so chunks don't silently truncate at embed time. Audit
        KNT-601 (0.2.0) public-API addition.
        """
        if self._backend_name == "ort":
            return int(self._backend.max_seq_len)
        # model2vec: there's no inherent seq cap (it's vocab → vector
        # lookup). Report a large value so downstream chunkers don't
        # over-split. 1<<20 is "effectively unlimited" for any
        # practical document.
        return 1 << 20

    def count_tokens(self, texts: Iterable[str]) -> list[int]:
        """Tokenize ``texts`` and return per-text token counts.

        Does NOT run inference; only the tokenizer pass. Used by
        downstream chunkers to decide whether a candidate chunk fits in
        ``max_seq_len`` before sending it to ``embed()``. Audit KNT-601
        (0.2.0) public-API addition.

        For ``model2vec``, returns a coarse word-count approximation
        since model2vec uses its own internal tokenizer that produces
        equivalent semantics to whitespace splitting. (The Rust path
        uses the registered model's actual HF tokenizer.)
        """
        text_list: list[str] = list(texts)
        if not text_list:
            return []
        if self._backend_name == "ort":
            return list(self._backend.count_tokens(text_list))
        # model2vec: best-effort whitespace count (model2vec doesn't
        # expose its tokenizer cleanly through StaticModel).
        return [len(t.split()) for t in text_list]

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
        # Audit KNT-601 (0.2.0): the audit-03 KNT-201 free-threaded
        # guard was retired alongside fastembed. The Rust cdylib is
        # ``gil_used = false`` and the Python ``tokenizers`` /
        # ``py_rust_stemmers`` packages are no longer in the tree.
        # Free-threaded Python (3.13t/3.14t) loads cleanly.

        s = settings if settings is not None else KaosNLPTransformersSettings()
        target = model_id or s.default_model
        req_device = device or s.device
        req_backend = backend or s.backend

        # Audit KNT-601 (0.2.0): set the process-wide embedding cache
        # capacity from settings on first load. Subsequent calls
        # honor the cap that was set first; explicit shrinks via
        # KaosNLPTransformersSettings.embedding_cache_size on a later
        # load() will reduce capacity (and evict) if the new value is
        # smaller. Cache is disabled (size=0) by default.
        if s.embedding_cache_size > 0:
            _embed_cache_set_size(s.embedding_cache_size)

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
                backend="ort",
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
            # Audit KNT-601 (0.2.0): post-fastembed-removal, the embedding
            # path goes through the Rust ``_rust.embedding.PyEmbeddingBackend``
            # which calls libonnxruntime via the ``ort`` Rust crate.
            # Revision pinning is now KNT-003-compliant by construction
            # (the Rust loader passes ``revision`` to hf-hub explicitly,
            # unlike the legacy fastembed path which depended on
            # fastembed's release-baked SHA).
            rust_backend = _load_rust_embedding_cached(
                model_id=registered.model_id,
                revision=registered.revision,
                device=device_info.device,
                cache_dir=cache_dir,
            )
            logger.info(
                "Loaded %s @ %s via ort (Rust) on %s",
                registered.model_id,
                registered.revision,
                device_info.device,
            )
            return cls(
                registered,
                rust_backend,
                device=device_info,
                backend_name="ort",
            )

    def embed(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        """Run inference and return a (N, dim) float32 array.

        Args:
            texts: Input strings — any iterable (list, tuple, generator,
                ``iter_paragraph_units(...)``-style stream, etc.).
                Materialized internally so the backend can stack into
                tensors. Empty input returns a ``(0, dim)`` array.
                Audit KNT-601 (0.2.0) widened from ``list[str]``.
            batch_size: Inference batch size passed to the backend.

        Raises:
            EmbeddingError: On backend exception or shape mismatch.
        """
        # Materialize the iterable once. The Rust backend stacks the
        # batch into tensors, so an internal collection is unavoidable.
        # Always copy through ``list(...)`` so the type stays
        # ``list[str]`` regardless of whether the caller passed a list,
        # tuple, or generator (avoids ty's Iterable→list narrowing
        # complaint).
        text_list: list[str] = list(texts)
        if not text_list:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Audit KNT-601 (0.2.0): opt-in process-wide LRU cache. When
        # enabled, look up each text first; collect misses, embed only
        # those, splice cached and freshly-embedded rows back together
        # by original-position index so the output ordering matches the
        # input. Disabled by default (size 0) → straight passthrough.
        if _EMBED_CACHE_SIZE > 0:
            cached_rows: list[np.ndarray | None] = [None] * len(text_list)
            miss_indices: list[int] = []
            for i, t in enumerate(text_list):
                cached_rows[i] = _embed_cache_get(self.model_id, self._registered.revision, t)
                if cached_rows[i] is None:
                    miss_indices.append(i)
            # All hits: assemble the cached rows directly.
            if not miss_indices:
                # Every entry is non-None at this branch — narrow for ty.
                rows: list[np.ndarray] = [r for r in cached_rows if r is not None]
                return np.stack(rows, axis=0).astype(np.float32, copy=False)
            # Some misses: embed the missing texts only.
            miss_texts = [text_list[i] for i in miss_indices]
            miss_arr = self._embed_uncached(miss_texts, batch_size=batch_size)
            # Splice and write-back.
            for offset, original_idx in enumerate(miss_indices):
                row = miss_arr[offset]
                cached_rows[original_idx] = row
                _embed_cache_put(
                    self.model_id, self._registered.revision, text_list[original_idx], row
                )
            rows = [r for r in cached_rows if r is not None]
            return np.stack(rows, axis=0).astype(np.float32, copy=False)

        # Cache disabled — straight path through the backend.
        return self._embed_uncached(text_list, batch_size=batch_size)

    def _embed_uncached(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        """The embed body that talks directly to the backend.

        Factored out so the cache layer in ``embed()`` can route to it
        for misses while still going through the same shape/dim
        validation and L2-normalization contract.
        """
        try:
            # Audit KNT-601 (0.2.0): two surviving backends after the
            # fastembed retirement — ``ort`` (Rust, default) and
            # ``model2vec`` (Python static lookup). Both produce
            # ``np.ndarray`` directly.
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
                # ``ort`` path: ``PyEmbeddingBackend.embed`` returns a
                # (N, dim) float32 numpy array directly, already
                # L2-normalized inside Rust (audit KNT-101).
                arr = self._backend.embed(texts, batch_size=batch_size)
                arr = np.asarray(arr, dtype=np.float32)
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
# Free-threaded Python guard (audit-03 KNT-201) — RETIRED in 0.2.0
# ---------------------------------------------------------------------------
#
# Audit KNT-601 (0.2.0): the ``_check_gil_enabled`` guard was removed
# alongside the fastembed Python wrapper. fastembed transitively pulled
# ``py_rust_stemmers`` (no Py_GIL_DISABLED support) and an old
# ``tokenizers`` Python wrapper, both of which crashed under free-
# threaded interpreters. Post-migration the package's only Rust surface
# is the in-tree cdylib (``_rust.abi3.so``) which is built with
# ``gil_used = false`` (audit KNT-602), and the Rust ``tokenizers``
# crate is statically linked into that cdylib (no Python ``tokenizers``
# import at runtime). Free-threaded Python is now supported. Tests that
# regressed on the removed guard are deleted in P4.7.


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
# Embedding cache (KNT-601 — opt-in process-wide LRU)
# ---------------------------------------------------------------------------
#
# Process-wide LRU keyed on ``(model_id, revision, blake2b(text))`` →
# cached float32 vector. Disabled by default (size 0). Enabled via
# ``KaosNLPTransformersSettings.embedding_cache_size`` > 0.
#
# Threading: the cache uses a single Lock for both reads and writes.
# Embedding callers that go ``embed()`` → cache lookup → maybe-run →
# cache write all hold the lock briefly. Heavy ort.run() work happens
# OUTSIDE the lock (the lock is only held for the dict ops). This is
# correct for free-threaded Python (KNT-602).


_EMBED_CACHE_LOCK = threading.Lock()
_EMBED_CACHE: OrderedDict[tuple[str, str, bytes], np.ndarray] = OrderedDict()
_EMBED_CACHE_SIZE: int = 0


def _hash_text(text: str) -> bytes:
    """Hash a text into a stable cache key. blake2b is stdlib + fast."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def _embed_cache_get(model_id: str, revision: str, text: str) -> np.ndarray | None:
    """Return the cached vector for ``text`` or None on miss / disabled."""
    if _EMBED_CACHE_SIZE <= 0:
        return None
    key = (model_id, revision, _hash_text(text))
    with _EMBED_CACHE_LOCK:
        v = _EMBED_CACHE.get(key)
        if v is None:
            return None
        # LRU: move to end on hit.
        _EMBED_CACHE.move_to_end(key)
        return v


def _embed_cache_put(model_id: str, revision: str, text: str, vector: np.ndarray) -> None:
    """Store a vector for ``text``; evicts the oldest entry if at capacity."""
    if _EMBED_CACHE_SIZE <= 0:
        return
    key = (model_id, revision, _hash_text(text))
    with _EMBED_CACHE_LOCK:
        _EMBED_CACHE[key] = vector
        _EMBED_CACHE.move_to_end(key)
        while len(_EMBED_CACHE) > _EMBED_CACHE_SIZE:
            _EMBED_CACHE.popitem(last=False)


def _embed_cache_set_size(size: int) -> None:
    """Set the cache capacity. Sticky once non-zero (the FIRST non-zero
    setting wins for the process); shrinking via this function evicts
    LRU entries to fit. Calling with 0 is a no-op (the cache cannot be
    disabled mid-process; that would invalidate cached vectors held by
    callers). Audit KNT-601 (0.2.0)."""
    global _EMBED_CACHE_SIZE
    if size <= 0:
        return
    with _EMBED_CACHE_LOCK:
        if _EMBED_CACHE_SIZE == 0:
            _EMBED_CACHE_SIZE = size
        elif size < _EMBED_CACHE_SIZE:
            _EMBED_CACHE_SIZE = size
            while len(_EMBED_CACHE) > _EMBED_CACHE_SIZE:
                _EMBED_CACHE.popitem(last=False)


def _embed_cache_clear() -> None:
    """Test-only: clear the cache. Safe to call from any thread."""
    with _EMBED_CACHE_LOCK:
        _EMBED_CACHE.clear()


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


# Audit KNT-601 (0.2.0): ``"fastembed"`` was retired in favor of the
# Rust-native ``"ort"`` backend. The Python ``fastembed`` wrapper is no
# longer in the dependency tree; the Rust cdylib calls libonnxruntime
# directly via the ``ort`` Rust crate. ``"model2vec"`` (static numpy
# lookup) stays as the second valid backend on its separate code path.
# Pre-KNT-501 ``"sentence-transformers"`` was retired in 0.1.0a6.
_VALID_BACKENDS: frozenset[str] = frozenset({"auto", "ort", "model2vec"})

# Backwards-compat alias set: legacy values that should produce a
# clear migration error rather than a confusing "unknown backend" error.
_RETIRED_BACKENDS: dict[str, str] = {
    "fastembed": (
        "The Python fastembed wrapper was replaced by the Rust-native "
        "'ort' backend in kaos-nlp-transformers 0.2.0 (audit KNT-601 — "
        "same ONNX runtime under the hood, no Python boundary, "
        "free-threaded Python compatible). Fix: use one of "
        "['auto', 'ort', 'model2vec'], or leave the setting unset for "
        "auto-detection. Alternative: pin to kaos-nlp-transformers<0.2 "
        "if you specifically need the fastembed Python wrapper "
        "(not recommended — superseded)."
    ),
    "sentence-transformers": (
        "The sentence-transformers backend was retired in 0.1.0a6 "
        "(audit KNT-501). Fix: use 'auto', 'ort', or 'model2vec'."
    ),
}


def _resolve_backend(requested: str, device: DeviceInfo, registry_backend: str) -> str:
    """Determine which backend to use given user preference and device.

    Audit-02 KNT-107: ``requested`` is validated against the closed set of
    backend names. Unknown values raise ``ValueError`` with the valid set
    rather than silently falling through.

    Audit-04 KNT-302: ``"model2vec"`` is honored as both an explicit
    backend choice and an auto-resolution target. model2vec models are
    *static* (vocab → vector lookup); the loader pins them to CPU
    regardless of the requested device.

    Audit KNT-601 (0.2.0): post-fastembed-removal, the auto-resolution
    is "registry decides." Both ``ort`` and ``model2vec`` run on CPU;
    ``ort`` accepts GPU via the [gpu] companion wheel (ort/cuda EP).
    Legacy ``"fastembed"`` requests raise ``ValueError`` with the
    migration text from ``_RETIRED_BACKENDS``.

    Returns 'ort' or 'model2vec'.
    """
    if requested in _RETIRED_BACKENDS:
        msg = f"Invalid backend {requested!r}. {_RETIRED_BACKENDS[requested]}"
        raise ValueError(msg)
    if requested not in _VALID_BACKENDS:
        msg = (
            f"Invalid backend {requested!r}. "
            f"Fix: use one of {sorted(_VALID_BACKENDS)}. "
            "Alternative: leave the setting unset to use the auto-detected "
            "backend (registry decides — ort for ONNX models, "
            "model2vec for static lookup models)."
        )
        raise ValueError(msg)

    if requested == "ort":
        return "ort"
    if requested == "model2vec":
        return "model2vec"

    # auto: registry decides. ort handles GPU via the companion wheel
    # (ort/cuda EP), so there's no device-specific override here.
    if registry_backend == "model2vec":
        return "model2vec"
    return "ort"


# ---------------------------------------------------------------------------
# Backend loaders (cached)
# ---------------------------------------------------------------------------
#
# Audit KNT-601 (0.2.0): ``_onnx_providers_for_device`` was retired
# alongside fastembed. EP selection moved into the Rust backend's
# ``configure_eps`` (rust/core/ort_runtime.rs) which gates CUDA /
# OpenVINO behind the ``gpu`` / ``openvino`` cargo features.


@lru_cache(maxsize=8)
def _load_rust_embedding_cached(
    model_id: str,
    revision: str,
    device: str,
    cache_dir: str | None,
):
    """Process-wide cache of loaded Rust embedding backends.

    Audit KNT-601 (Phase 3): the Rust ``PyEmbeddingBackend.load`` is
    relatively heavy (download ONNX + tokenizer, build ort Session,
    optimization pass) so caching by ``(model_id, revision, device,
    cache_dir)`` matters for the long-running MCP server case where
    a process embeds many calls back-to-back.

    The ``revision`` is part of the cache key. Unlike fastembed-rs
    (which can't honor a runtime revision override), our Rust loader
    DOES pin the SHA at HF Hub fetch time — see the KNT-003 contract
    in ``rust/core/model_loader.rs::resolve_paths``.
    """
    try:
        from kaos_nlp_transformers._rust.embedding import (
            EmbeddingBackend,  # type: ignore[import-not-found]
        )
    except ImportError as exc:
        msg = (
            "kaos_nlp_transformers._rust extension is not built. "
            "Fix: run `uv run maturin develop --release` to compile the "
            "Rust cdylib for editable installs, or reinstall the package "
            "from a released wheel. "
            "Alternative: pin to kaos-nlp-transformers<0.2 if you need "
            "the legacy fastembed Python backend."
        )
        raise BackendNotInstalledError(msg) from exc

    try:
        # The Rust loader takes the device string directly; cache_dir
        # may be None to fall back to HF_HOME / system default.
        return EmbeddingBackend.load(model_id, device=device, cache_dir=cache_dir)
    except Exception as exc:
        # The Rust path raises through bindings/util.rs::map_backend_error,
        # which already surfaces as kaos_nlp_transformers.errors.*Error.
        # Wrap any other Python-side exception into ModelLoadError so the
        # public contract holds.
        if isinstance(exc, BackendNotInstalledError | ModelLoadError | ModelNotRegisteredError):
            raise
        # Cache key is (model_id, revision, device, cache_dir); revision
        # is part of the key but the Rust loader uses it via the registry
        # entry, not directly here.
        _ = revision
        msg = (
            f"Failed to load model {model_id!r} via the Rust ort backend: {exc}. "
            "Fix: verify network access on first download, or set "
            "KAOS_NLP_TRANSFORMERS_OFFLINE=false. "
            f"Alternative: unset KAOS_NLP_TRANSFORMERS_BACKEND to fall back "
            "to the fastembed default (Phase 3 only — Phase 4 retires this)."
        )
        raise ModelLoadError(msg) from exc


# Audit KNT-601 (0.2.0): ``_load_fastembed_cached`` retired. The Rust
# backend loader ``_load_rust_embedding_cached`` (defined above) is
# now the sole ONNX-backed embedding loader. Pre-KNT-501 (0.1.0a6)
# the sentence-transformers cached loader was retired in the same
# spirit. The legacy fastembed branch in ``EmbeddingModel.load`` is
# also gone; ``_resolve_backend`` only returns ``"ort"`` or
# ``"model2vec"``, never the retired backend names.


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
        from model2vec import StaticModel
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
