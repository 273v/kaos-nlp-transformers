"""Regression tests for audit-07 KNT-602 Option A.

KNT-602 moved ``SemanticDedupLevel`` + the ``kaos-nlp-transformers-dedup-semantic``
MCP tool from this package into kaos-content (which becomes the
canonical owner of AST-grounded integrations with kaos-nlp-transformers).
The change restores the documented layer cake — kaos-content is the
consumer of kaos-nlp-transformers, never the inverse.

These tests pin the boundary so a future refactor can't silently
re-introduce the cycle.
"""

from __future__ import annotations

import pathlib
import re

import pytest

pytestmark = pytest.mark.unit


def test_no_kaos_content_import_anywhere() -> None:
    """The package must not import ``kaos_content`` at any source path.

    Pre-KNT-602, ``kaos_nlp_transformers/clustering/semantic_dedup.py``
    and ``kaos_nlp_transformers/tools.py`` imported from
    ``kaos_content.dedup.types``. Both import sites moved to kaos-content
    in 0.2.0a3 (alongside the ``SemanticDedupLevel`` impl).
    """
    import kaos_nlp_transformers

    pkg_root = pathlib.Path(kaos_nlp_transformers.__file__).parent
    offenders: list[tuple[pathlib.Path, int, str]] = []
    pat = re.compile(r"\b(?:from|import)\s+kaos_content\b")
    for py in pkg_root.rglob("*.py"):
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            if pat.search(line):
                offenders.append((py.relative_to(pkg_root), lineno, line.strip()))
    assert not offenders, (
        "kaos_content imports found in kaos-nlp-transformers (would re-introduce "
        f"the KNT-602 boundary violation): {offenders}"
    )


def test_no_clustering_submodule() -> None:
    """The ``kaos_nlp_transformers.clustering`` submodule was removed.

    Importing it now must raise ImportError — the SemanticDedupLevel
    moved to ``kaos_content.dedup.levels.semantic`` in kaos-content
    0.1.0a3.
    """
    with pytest.raises(ImportError):
        # Bypass any cached sentinel from prior tests.
        import importlib

        importlib.import_module("kaos_nlp_transformers.clustering")


def test_no_dedup_semantic_tool_registered() -> None:
    """``register_transformers_tools`` must not register the old
    ``kaos-nlp-transformers-dedup-semantic`` tool — it moved to
    kaos-content as ``kaos-content-dedup-semantic``.
    """
    pytest.importorskip("kaos_core")
    from kaos_core import KaosRuntime

    from kaos_nlp_transformers.tools import register_transformers_tools

    runtime = KaosRuntime()
    register_transformers_tools(runtime)
    names = runtime.tools.list_tools()
    assert "kaos-nlp-transformers-dedup-semantic" not in names, (
        "the old dedup-semantic tool was removed in 0.2.0a3 (KNT-602); "
        f"it must not be registered. Got tools: {sorted(names)}"
    )


def test_pyproject_does_not_pin_kaos_content_in_base_deps() -> None:
    """Base ``[project].dependencies`` must not include kaos-content.

    KNT-602 moved kaos-content out of the base dependency list in
    0.2.0a3 — it stays only in the dev group (test environment) since
    integration tests cover the consumer surface.
    """
    pkg_root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = pkg_root / "pyproject.toml"
    text = pyproject.read_text()

    # Locate the [project] table's dependencies = [...] block.
    match = re.search(
        r"^\[project\]\s*$.*?^dependencies\s*=\s*\[(.*?)\]",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match, "could not locate [project].dependencies in pyproject.toml"
    base_deps = match.group(1)

    # Pull only quoted PEP 508 strings (the actual dep specs) — match a
    # mention in a comment doesn't count as a regression.
    actual_deps = re.findall(r'"([^"\n]+)"', base_deps)
    offenders = [d for d in actual_deps if d.startswith(("kaos-content", "kaos_content"))]
    assert not offenders, (
        "kaos-content is back in [project].dependencies — KNT-602 boundary fix "
        f"regressed. Offending entries: {offenders}"
    )
