"""Error hierarchy for kaos-nlp-transformers.

All exceptions inherit from ``KaosCoreError`` so they participate in
the agent-friendly triplet contract: every error message must answer
(1) what went wrong, (2) how to fix it, (3) alternative approach when
applicable.
"""

from __future__ import annotations

from kaos_core.exceptions import KaosCoreError


class KaosNLPTransformersError(KaosCoreError):
    """Base error for kaos-nlp-transformers."""


class ModelNotRegisteredError(KaosNLPTransformersError):
    """Model id is not in the registry and unregistered models are forbidden."""


class ModelLoadError(KaosNLPTransformersError):
    """Backend failed to load the model (download error, corrupt cache, etc.)."""


class EmbeddingError(KaosNLPTransformersError):
    """Inference failure (empty input, dim mismatch, backend exception)."""


class BackendNotInstalledError(KaosNLPTransformersError):
    """Required backend (fastembed or torch extras) is not installed."""


__all__ = [
    "BackendNotInstalledError",
    "EmbeddingError",
    "KaosNLPTransformersError",
    "ModelLoadError",
    "ModelNotRegisteredError",
]
