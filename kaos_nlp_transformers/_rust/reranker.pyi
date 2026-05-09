"""Stub for kaos_nlp_transformers._rust.reranker."""

from __future__ import annotations

from typing import Any

import numpy as np

class CrossEncoderBackend:
    """Rust-backed cross-encoder inference. Wraps an ort Session loaded
    from a registered reranker model."""

    @staticmethod
    def load(
        model_id: str,
        *,
        device: str = "cpu",
        cache_dir: str | None = None,
    ) -> CrossEncoderBackend: ...
    def score(
        self,
        queries: list[str],
        passages: list[str],
        batch_size: int = 32,
    ) -> np.ndarray[Any, np.dtype[np.float32]]: ...
    @property
    def model_id(self) -> str: ...
    @property
    def device(self) -> str: ...
