"""``_offline_env_scope`` must scope huggingface_hub's offline flag, not only the env.

huggingface_hub reads ``HF_HUB_OFFLINE`` once at import into
``huggingface_hub.constants.HF_HUB_OFFLINE``. These run in fresh interpreters because
import order is the bug: a first import inside an offline scope used to freeze offline
mode for the rest of the process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("huggingface_hub")


def _run(code: str) -> None:
    env = {
        k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_first_import_inside_offline_scope_does_not_leak():
    _run(
        """
        import os, sys
        from kaos_nlp_transformers.embedding import _offline_env_scope
        assert "huggingface_hub" not in sys.modules
        with _offline_env_scope(True):
            from huggingface_hub import constants
            assert constants.is_offline_mode()
        assert not constants.is_offline_mode()
        assert "HF_HUB_OFFLINE" not in os.environ
        """
    )


def test_scope_applies_when_hub_already_imported():
    _run(
        """
        from huggingface_hub import constants
        from kaos_nlp_transformers.embedding import _offline_env_scope
        assert not constants.is_offline_mode()
        with _offline_env_scope(True):
            assert constants.is_offline_mode()
        assert not constants.is_offline_mode()
        """
    )


def test_scope_restores_on_exception_and_respects_caller_offline():
    _run(
        """
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        from huggingface_hub import constants
        from kaos_nlp_transformers.embedding import _offline_env_scope
        try:
            with _offline_env_scope(True):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert constants.is_offline_mode() and os.environ["HF_HUB_OFFLINE"] == "1"
        with _offline_env_scope(False):
            assert constants.is_offline_mode()  # offline=False never forces online
        """
    )
