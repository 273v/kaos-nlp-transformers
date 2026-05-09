"""Error hierarchy for kaos-nlp-transformers.

All exceptions inherit from ``KaosCoreError`` so they participate in
the agent-friendly triplet contract: every error message must answer
(1) what went wrong, (2) how to fix it, (3) alternative approach when
applicable.

Errors that carry *structured* recovery information (e.g. an install
extra a caller should suggest) follow the kaos-core pattern from
``UnsafeURLError`` / ``ResponseSizeError``: typed attributes plus a
``details`` dict piped through ``KaosCoreError(**details)`` so MCP tool
callers can extract the recovery payload programmatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_core.exceptions import KaosCoreError

if TYPE_CHECKING:
    from kaos_nlp_transformers.device import LatentDevice


class KaosNLPTransformersError(KaosCoreError):
    """Base error for kaos-nlp-transformers."""


class ModelNotRegisteredError(KaosNLPTransformersError):
    """Model id is not in the registry and unregistered models are forbidden."""


class ModelLoadError(KaosNLPTransformersError):
    """Backend failed to load the model (download error, corrupt cache, etc.)."""


class EmbeddingError(KaosNLPTransformersError):
    """Inference failure (empty input, dim mismatch, backend exception)."""


class BackendNotInstalledError(KaosNLPTransformersError):
    """Required backend is not installed.

    Audit-06 KNT-501: post-torch-removal, this fires when fastembed
    is missing (base install only — should be impossible) or when an
    optional backend dep (``[model2vec]``, ``[gpu]`` for
    ``onnxruntime-gpu``) is requested but unavailable. The message is
    expected to carry an actionable install hint.
    """


class DeviceNotReachableError(KaosNLPTransformersError):
    """A requested accelerator is physically present but not reachable.

    Raised by ``device.resolve_device`` when the caller asks for e.g.
    ``cuda:0`` on a host where nvidia-smi sees the card but no Python
    binding is installed. Post-audit-06 KNT-501 the relevant binding is
    ``onnxruntime-gpu`` (the ``[gpu]`` extra); the legacy
    ``[torch]``-with-CUDA path was retired in 0.1.0a6. The
    ``install_extra`` detail gives an MCP tool / agent the exact extra
    to recommend — ``pip install kaos-nlp-transformers[{install_extra}]``.

    Attributes:
        requested: The device string the caller asked for ('cuda', 'cuda:1', …).
        kind: Accelerator family ('cuda', 'rocm', 'mps').
        install_extra: pyproject extra to install, or ``None`` if the fix
            is something other than an extra (driver missing, etc.).
        reason: Human-readable explanation lifted from the LatentDevice probe.
    """

    def __init__(
        self,
        *,
        requested: str,
        latent: LatentDevice,
        message: str | None = None,
        **extra: Any,
    ) -> None:
        self.requested = requested
        self.kind = latent.kind
        self.install_extra = latent.install_extra
        self.reason = latent.reason

        details: dict[str, Any] = {
            "requested": requested,
            "kind": latent.kind,
            "name": latent.name,
            "reason": latent.reason,
            "install_extra": latent.install_extra,
            **extra,
        }

        if message is None:
            install_hint = (
                f"pip install kaos-nlp-transformers[{latent.install_extra}]"
                if latent.install_extra
                else "(no single-extra fix available — see reason)"
            )
            message = (
                f"Device {requested!r} ({latent.name}) is physically present "
                f"but not reachable from this Python environment. "
                f"Fix: {install_hint}. "
                f"Reason: {latent.reason} "
                "Alternative: use device='cpu' or device='auto' to fall back."
            )
        super().__init__(message, **details)


__all__ = [
    "BackendNotInstalledError",
    "DeviceNotReachableError",
    "EmbeddingError",
    "KaosNLPTransformersError",
    "ModelLoadError",
    "ModelNotRegisteredError",
]
