"""Shared test fixtures for unit tests.

Audit-03 KNT-201 added a free-threaded-Python guard at the top of
``EmbeddingModel.load`` and ``CrossEncoderReranker.load``: the load path
refuses to run on a Py_GIL_DISABLED interpreter (3.13t / 3.14t / etc.)
because fastembed's transitive ``py_rust_stemmers`` segfaults during
module init.

That guard is the first check in the load flow, so on a free-threaded
build it short-circuits every test that wanted to exercise some OTHER
check (registry gate, settings injection, offline scope, etc.). To keep
those tests testing what they say they test, we set ``sys._is_gil_enabled``
to ``True`` for the entire unit-test session by default.

The audit-03 tests in ``test_audit_03.py`` explicitly opt out via
``test_check_gil_enabled_passes_on_normal_build`` (which DOES want to
verify the unguarded path on the actual interpreter) and
``test_check_gil_enabled_refuses_on_free_threaded`` (which sets
``sys._is_gil_enabled`` False explicitly). Those tests use their own
local monkeypatch and are not affected by this fixture.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _force_gil_enabled_for_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``sys._is_gil_enabled() == True`` for every unit test that
    doesn't override it explicitly. This makes the unit suite portable
    across CPython 3.13 / 3.14 (GIL) and CPython 3.13t / 3.14t
    (free-threaded) — the load-path tests exercise the same assertions
    on both, with the audit-03 free-threaded guard tests being the only
    place that DOES exercise the no-GIL path.

    Tests that need the actual interpreter behavior can override by
    deleting the patch first, or by patching to a different value
    (the audit-03 tests do exactly that).
    """
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: True, raising=False)
