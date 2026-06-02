"""Disk-backed incremental embedding cache.

Layer decision: this module is I/O orchestration — it reads and writes
already-computed embedding vectors keyed by an already-computed content
hash. It runs no NLP/CPU kernel (no tokenization, pooling, or
similarity), so a stdlib + numpy Python implementation is appropriate
here. The compute-belongs-in-Rust rule applies to the inference kernels,
which stay in the Rust cdylib; the *key derivation* below deliberately
reuses ``embedding._hash_text`` so disk keys never diverge from the
in-process LRU keys.

What it is for
--------------
The in-process LRU (``KaosNLPTransformersSettings.embedding_cache_size``)
is lost on restart, so a long-running service re-embeds the same corpus
on every cold start. This disk cache persists
``(model_id, revision, dim, dtype, hash(text)) -> vector`` across process
restarts. On ``embed()`` only genuinely new (cache-miss) texts pay the
forward pass; their vectors are then written back. A growing corpus only
ever pays inference for texts it has never seen.

On-disk layout
--------------
Under the configured cache root each model/revision/shape gets its own
namespace directory, and each text gets one ``.npy`` file named by the
hex content hash::

    <cache_root>/
      <safe_model_id>/<revision>/<dim>-<dtype>/
        <hex_hash>.npy
        <hex_hash>.npy
        ...

``<safe_model_id>`` replaces filesystem-unsafe characters in the model
id. The ``<dim>-<dtype>`` segment namespaces by output shape and dtype,
and ``<revision>`` namespaces by the pinned model SHA, so a model,
revision, dimension, or dtype change can never serve a stale vector —
the lookup simply lands in a different directory and misses. Each
``.npy`` is the standard NumPy binary format, which records shape and
dtype and is portable across architectures.

Atomicity and concurrency
--------------------------
Each entry is written to a unique temporary file in the same directory
and then ``os.replace``-d into place, so a reader (this or another
process) never observes a half-written ``.npy``. A crash mid-write
leaves only an orphan ``*.tmp`` that the next write/read ignores; it is
never loaded. Concurrent writers of the same key race harmlessly — the
last ``os.replace`` wins and both wrote identical bytes for the same
content hash. Reads validate shape/dtype and treat any corrupt or
mismatched file as a miss (and best-effort delete it) rather than
returning a wrong vector.

Growth and pruning
-------------------
The cache is unbounded by design: an incremental corpus should keep
every vector it has paid for. There is no silent cap and no eviction.
An operator prunes by deleting files or whole namespace directories
under the cache root (e.g. ``rm -rf <cache_root>/<safe_model_id>``), or
by removing the root entirely; the next run repopulates only what is
used. This is intentionally separate from the HuggingFace snapshot cache
(``cache_dir`` / ``HF_HOME``) — that holds model weights, this holds
computed vectors.
"""

from __future__ import annotations

import contextlib
import os
import re
import secrets
from pathlib import Path

import numpy as np
from kaos_core.logging import get_logger

logger = get_logger(__name__)

# Filesystem-safe model-id slug: collapse anything outside a small safe
# set to ``_``. The revision SHA and the content hash are already safe;
# only the model id can contain ``/`` and other path-hostile characters.
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_segment(value: str) -> str:
    """Map an arbitrary string to a single filesystem-safe path segment."""
    cleaned = _UNSAFE_CHARS.sub("_", value).strip("._")
    return cleaned or "_"


class DiskEmbeddingCache:
    """Content-addressed, namespaced, on-disk store of embedding vectors.

    Construct one per ``(model_id, revision, dim, dtype)`` so the
    namespace directory is fixed for the instance's lifetime. The
    namespace is created lazily on first write, not at construction, to
    keep ``EmbeddingModel.load`` free of filesystem side effects when no
    vector is ever written.
    """

    def __init__(
        self,
        root: Path,
        *,
        model_id: str,
        revision: str,
        dim: int,
        dtype: str = "float32",
    ) -> None:
        self._root = Path(root)
        # Namespace by model/revision/shape so a pinned-revision bump or
        # a dim/dtype change lands in a fresh directory and never serves
        # stale vectors.
        self._namespace = (
            self._root
            / _safe_segment(model_id)
            / _safe_segment(revision)
            / f"{int(dim)}-{_safe_segment(dtype)}"
        )
        self._dim = int(dim)
        self._dtype = np.dtype(dtype)

    @property
    def namespace_dir(self) -> Path:
        """The directory this instance reads/writes (may not exist yet)."""
        return self._namespace

    def _path_for(self, key_hex: str) -> Path:
        return self._namespace / f"{key_hex}.npy"

    def get(self, key_hex: str) -> np.ndarray | None:
        """Return the cached 1-D vector for ``key_hex`` or None on miss.

        A file whose shape or dtype does not match this namespace's
        contract, or that fails to load, is treated as a miss and
        best-effort removed so a future write can replace it.
        """
        path = self._path_for(key_hex)
        if not path.is_file():
            return None
        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as exc:
            logger.warning("Disk cache entry %s failed to load (%s); ignoring", path, exc)
            self._discard(path)
            return None
        if arr.dtype != self._dtype or arr.ndim != 1 or arr.shape[0] != self._dim:
            logger.warning(
                "Disk cache entry %s shape/dtype mismatch (got %s %s, expected (%d,) %s); ignoring",
                path,
                arr.shape,
                arr.dtype,
                self._dim,
                self._dtype,
            )
            self._discard(path)
            return None
        return arr

    def put(self, key_hex: str, vector: np.ndarray) -> None:
        """Persist ``vector`` for ``key_hex`` via an atomic tmp+rename.

        A vector that does not match the namespace's shape/dtype contract
        is not written (it would only ever read back as a miss). Write
        failures are logged and swallowed: the disk cache is an
        optimization, never a correctness dependency.
        """
        vec = np.ascontiguousarray(vector, dtype=self._dtype)
        if vec.ndim != 1 or vec.shape[0] != self._dim:
            logger.warning(
                "Refusing to cache vector with shape %s (expected (%d,)) for key %s",
                vec.shape,
                self._dim,
                key_hex,
            )
            return
        try:
            self._namespace.mkdir(parents=True, exist_ok=True)
            final = self._path_for(key_hex)
            # Unique temp name in the same directory so os.replace is an
            # atomic same-filesystem rename. PID + random suffix keeps
            # concurrent writers from clobbering each other's temp file.
            tmp = self._namespace / f"{key_hex}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
            try:
                with tmp.open("wb") as fh:
                    np.save(fh, vec, allow_pickle=False)
                    fh.flush()
                    os.fsync(fh.fileno())
                tmp.replace(final)
            finally:
                # Clean up the temp file if the rename never happened.
                if tmp.exists():
                    self._discard(tmp)
        except Exception as exc:
            logger.warning("Disk cache write for key %s failed (%s); skipping", key_hex, exc)

    @staticmethod
    def _discard(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.unlink()


__all__ = ["DiskEmbeddingCache"]
