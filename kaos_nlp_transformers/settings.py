"""Module settings for kaos-nlp-transformers.

Standard KAOS ``ModuleSettings`` pattern: env_prefix
``KAOS_NLP_TRANSFORMERS_``, ``mode="before"`` legacy fallback for
``HF_HUB_OFFLINE`` and ``HF_HOME``, ``extra="ignore"``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from kaos_core.config.module_settings import ModuleSettings
from pydantic import SecretStr, model_validator
from pydantic_settings import SettingsConfigDict


class KaosNLPTransformersSettings(ModuleSettings):
    """Typed settings for kaos-nlp-transformers."""

    default_model: str = "BAAI/bge-small-en-v1.5"
    """Default embedding model loaded by ``EmbeddingModel.load()`` and any
    consumer that does not pass an explicit ``model_id``. Must be present
    in ``REGISTRY`` (or ``allow_unregistered`` must be true).

    The whole package now derives its embedding-model default from this
    field — ``SemanticDedupLevel.__init__`` and the worker MCP tools pull
    ``model_fields["default_model"].default`` so a single environment
    override (``KAOS_NLP_TRANSFORMERS_DEFAULT_MODEL``) updates every call
    site that does not pass an explicit override."""

    default_reranker_model: str = "BAAI/bge-reranker-base"
    """Default cross-encoder reranker loaded by
    ``CrossEncoderReranker.load()``. Same single-source-of-truth pattern as
    ``default_model``: change this and every internal default updates."""

    cache_dir: Path | None = None
    offline: bool = False
    allow_unregistered: bool = False
    profile: str = "default"

    device: str = "auto"
    """Device for embedding inference.

    Values: 'auto' (detect best available), 'cpu', 'cuda', 'cuda:0',
    'cuda:1', 'mps', 'xla', 'openvino'. Default 'auto' selects the
    best GPU if torch is installed with GPU support, otherwise CPU.
    """

    backend: str = "auto"
    """Embedding backend preference.

    Values: 'auto' (device-dependent), 'fastembed', 'sentence-transformers'.
    Default 'auto' uses fastembed for CPU, sentence-transformers for GPU.
    """

    workspace_root: Path | None = None
    """Filesystem sandbox root for any tool that reads or writes files.

    Mirrors ``kaos_nlp_core.KaosNlpSettings.workspace_root``: ``None`` falls
    back to ``Path.cwd()`` at use time. Tool callers (CLI, MCP) MUST resolve
    user-supplied paths against this root and reject anything that escapes
    it. Set ``KAOS_NLP_TRANSFORMERS_WORKSPACE_ROOT`` to widen or pin the
    allowed area in production.
    """

    http_token: SecretStr | None = None
    """Operator-ack token for ``kaos-nlp-transformers-serve --http``.

    The value is not verified against incoming requests — kaos-mcp's
    current transport does not enforce bearer-token auth — but the
    *presence* of any non-empty value is required to start the HTTP
    transport. The semantics match ``kaos-nlp-core.http_token``: the
    operator confirms a reverse proxy is doing the actual authentication.
    ``SecretStr`` redacts the value in logs and ``model_dump`` output.
    """

    model_config = SettingsConfigDict(
        env_prefix="KAOS_NLP_TRANSFORMERS_",
        env_file=".env",
        extra="ignore",
    )

    @model_validator(mode="before")
    @classmethod
    def _legacy_env_fallbacks(cls, values: Any) -> Any:
        """Honor legacy HF_HUB_OFFLINE / HF_HOME / KAOS_PROFILE env vars."""
        if not isinstance(values, dict):
            return values

        if "offline" not in values or values.get("offline") is None:
            legacy_offline = os.environ.get("HF_HUB_OFFLINE", "").lower()
            if legacy_offline in ("1", "true", "yes"):
                values["offline"] = True

        if "cache_dir" not in values or values.get("cache_dir") is None:
            legacy_cache = os.environ.get("HF_HOME")
            if legacy_cache:
                values["cache_dir"] = Path(legacy_cache)

        if not values.get("profile"):
            legacy_profile = os.environ.get("KAOS_PROFILE")
            if legacy_profile:
                values["profile"] = legacy_profile

        return values


__all__ = ["KaosNLPTransformersSettings"]
