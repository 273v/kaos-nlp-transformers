"""Regression tests for audit-03 finding KNT-201.

Pins the runtime guard against free-threaded Python so a future refactor
can't silently re-introduce the SIGSEGV that ``import fastembed`` causes
on 3.14t (via the upstream py_rust_stemmers C extension that hasn't
declared Py_GIL_DISABLED support).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# -----------------------------------------------------------------------------
# KNT-201 — refuse to load on free-threaded Python
# -----------------------------------------------------------------------------


def test_check_gil_enabled_passes_on_normal_build():
    """On a GIL-enabled interpreter, the check is a no-op (returns None)."""
    from kaos_nlp_transformers.embedding import _check_gil_enabled

    # Don't monkeypatch — just verify it doesn't raise on the
    # interpreter the test suite is actually running under.
    _check_gil_enabled()  # MUST NOT raise


def test_check_gil_enabled_refuses_on_free_threaded(monkeypatch):
    """When ``sys._is_gil_enabled()`` reports False (free-threaded build),
    ``_check_gil_enabled`` raises ``BackendNotInstalledError`` with a
    message pointing at the upstream root cause and the recommended
    workaround."""
    import sys

    from kaos_nlp_transformers.embedding import _check_gil_enabled
    from kaos_nlp_transformers.errors import BackendNotInstalledError

    # Pretend the interpreter is free-threaded.
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: False, raising=False)

    with pytest.raises(BackendNotInstalledError) as exc_info:
        _check_gil_enabled()

    msg = str(exc_info.value)
    # Hard-rule the failure shape: the user must learn (1) what's wrong,
    # (2) how to fix, (3) where the upstream tracker lives.
    assert "free-threaded" in msg.lower()
    assert "py_rust_stemmers" in msg
    assert "3.13" in msg or "3.14" in msg


def test_embedding_model_load_calls_the_guard(monkeypatch):
    """``EmbeddingModel.load`` must invoke ``_check_gil_enabled`` BEFORE
    attempting any fastembed import — otherwise free-threaded users hit
    a SIGSEGV in py_rust_stemmers' module init instead of a clean error.
    """
    import sys

    from kaos_nlp_transformers import EmbeddingModel
    from kaos_nlp_transformers.errors import BackendNotInstalledError

    # Force the free-threaded code path. The poisoned _is_gil_enabled
    # has to fire before EmbeddingModel.load attempts to import the
    # backend layer; if the order is wrong, the test would crash with
    # an unrelated fastembed-loaded error before we got here.
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: False, raising=False)

    with pytest.raises(BackendNotInstalledError, match=r"free-threaded"):
        EmbeddingModel.load("BAAI/bge-small-en-v1.5")


def test_cross_encoder_reranker_load_calls_the_guard(monkeypatch):
    """``CrossEncoderReranker.load`` mirrors the guard so the
    ``[torch]`` extra path doesn't segfault on 3.14t either."""
    import sys

    from kaos_nlp_transformers.errors import BackendNotInstalledError
    from kaos_nlp_transformers.reranker import CrossEncoderReranker

    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: False, raising=False)

    with pytest.raises(BackendNotInstalledError, match=r"free-threaded"):
        CrossEncoderReranker.load()


def test_is_free_threaded_python_helper():
    """``_is_free_threaded_python`` reflects ``sys._is_gil_enabled()``
    when present, defaults to ``False`` on older builds without the API."""
    import sys

    from kaos_nlp_transformers.embedding import _is_free_threaded_python

    actual_gil_enabled = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    assert _is_free_threaded_python() == (not actual_gil_enabled)
