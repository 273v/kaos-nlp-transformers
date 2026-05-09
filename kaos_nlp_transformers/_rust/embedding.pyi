"""Stub for kaos_nlp_transformers._rust.embedding."""

from __future__ import annotations

from typing import Any

import numpy as np

class EmbeddingBackend:
    """Rust-backed embedding inference. Wraps an ort Session loaded
    from a registered model."""

    @staticmethod
    def load(
        model_id: str,
        *,
        device: str = "cpu",
        cache_dir: str | None = None,
    ) -> EmbeddingBackend: ...
    def embed(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray[Any, np.dtype[np.float32]]: ...
    @property
    def dim(self) -> int: ...
    @property
    def model_id(self) -> str: ...
    @property
    def device(self) -> str: ...
