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
from pydantic import model_validator
from pydantic_settings import SettingsConfigDict


class KaosNLPTransformersSettings(ModuleSettings):
    """Typed settings for kaos-nlp-transformers."""

    default_model: str = "BAAI/bge-small-en-v1.5"
    cache_dir: Path | None = None
    offline: bool = False
    allow_unregistered: bool = False
    profile: str = "default"

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
