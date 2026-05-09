"""Stub for kaos_nlp_transformers._rust.registry."""

from __future__ import annotations

__version__: str

def capabilities() -> dict[str, bool | list[str]]:
    """Compile-time + runtime capability snapshot.

    Returns a dict shaped like::

        {
          "cpu": True,
          "cuda": False,
          "openvino": False,
          "build_features": [],
        }
    """

def vendored_model_path(model_id: str) -> str | None:
    """Return the absolute path of the vendored copy of ``model_id``,
    or None if no vendored copy ships in this wheel.
    """
