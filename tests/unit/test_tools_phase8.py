"""Unit tests for the Phase-8 MCP tools (nli-classify / ner-extract /
pii-detect).

These tests cover:

1. **Metadata shape** — tool name, MCP-safe hyphenation, read-only
   annotations, input-schema completeness.
2. **Parameter validation** — every error path that fires before any
   model is loaded (empty/missing/wrong-type inputs).
3. **Registry-driven loading** — the tools' default model resolves
   from the settings (so a single env-var override updates the MCP
   surface).

Real-inference smoke for these tools lives in
``tests/integration/test_tools_phase8_live.py`` (skipped offline).
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit

kaos_core = pytest.importorskip("kaos_core")

from kaos_core import KaosRuntime  # noqa: E402
from kaos_core.base.tool import KaosTool  # noqa: E402

from kaos_nlp_transformers.tools import register_transformers_tools  # noqa: E402


def _make_runtime() -> KaosRuntime:
    return KaosRuntime()


def _get_tool(runtime: KaosRuntime, name: str) -> KaosTool:
    tool = runtime.tools.get_tool(name)
    assert tool is not None, f"register_transformers_tools must register {name}"
    return tool


# ---------------------------------------------------------------------------
# Metadata shape — each tool registered, MCP-safe name, read-only annotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name",
    [
        "kaos-nlp-transformers-nli-classify",
        "kaos-nlp-transformers-ner-extract",
        "kaos-nlp-transformers-pii-detect",
    ],
)
def test_phase8_tool_registered_and_readonly(tool_name: str) -> None:
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, tool_name)
    md = tool.metadata
    assert md.name == tool_name
    assert md.module_name == "kaos-nlp-transformers"
    assert md.version
    assert md.annotations is not None
    # All three inference tools are read-only and idempotent — they
    # do not mutate any registry or cache state at runtime (model
    # download is a side effect of .load(), not of the tool call,
    # and is bounded by the registry's pinned SHAs).
    assert md.annotations.readOnlyHint is True
    assert md.annotations.idempotentHint is True
    assert md.annotations.openWorldHint is False
    # MCP-safe hyphenated naming.
    assert "_" not in md.name
    # Input schema should have at least one parameter for every
    # inference tool.
    assert len(md.input_schema) >= 1


def test_nli_classify_input_schema_has_premise_and_hypotheses() -> None:
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-nli-classify")
    names = {p.name for p in tool.metadata.input_schema}
    assert "premise" in names
    assert "hypotheses" in names
    assert "model_id" in names


def test_ner_extract_input_schema_has_texts_labels_threshold() -> None:
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-ner-extract")
    names = {p.name for p in tool.metadata.input_schema}
    assert {"texts", "labels", "threshold", "model_id"}.issubset(names)


def test_pii_detect_input_schema_has_texts_threshold() -> None:
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-pii-detect")
    names = {p.name for p in tool.metadata.input_schema}
    assert {"texts", "score_threshold", "model_id"}.issubset(names)


# ---------------------------------------------------------------------------
# Parameter validation — offline paths (no model loads)
# ---------------------------------------------------------------------------


def _run(tool: KaosTool, inputs: dict):
    return asyncio.run(tool.execute(inputs))


class TestNliClassifyValidation:
    def test_missing_premise(self) -> None:
        runtime = _make_runtime()
        register_transformers_tools(runtime)
        tool = _get_tool(runtime, "kaos-nlp-transformers-nli-classify")
        result = _run(tool, {"hypotheses": ["x"]})
        assert result.isError is True
        assert "premise" in result.require_text().lower()

    def test_empty_hypotheses(self) -> None:
        runtime = _make_runtime()
        register_transformers_tools(runtime)
        tool = _get_tool(runtime, "kaos-nlp-transformers-nli-classify")
        result = _run(tool, {"premise": "Hello.", "hypotheses": []})
        assert result.isError is True
        assert "hypotheses" in result.require_text().lower()

    def test_non_string_hypothesis_rejected(self) -> None:
        runtime = _make_runtime()
        register_transformers_tools(runtime)
        tool = _get_tool(runtime, "kaos-nlp-transformers-nli-classify")
        result = _run(tool, {"premise": "x", "hypotheses": [1, "ok"]})
        assert result.isError is True
        assert "string" in result.require_text().lower()


class TestNerExtractValidation:
    def test_missing_texts(self) -> None:
        runtime = _make_runtime()
        register_transformers_tools(runtime)
        tool = _get_tool(runtime, "kaos-nlp-transformers-ner-extract")
        result = _run(tool, {"labels": ["person"]})
        assert result.isError is True
        assert "texts" in result.require_text().lower()

    def test_missing_labels(self) -> None:
        runtime = _make_runtime()
        register_transformers_tools(runtime)
        tool = _get_tool(runtime, "kaos-nlp-transformers-ner-extract")
        result = _run(tool, {"texts": ["Alice walks home."]})
        assert result.isError is True
        assert "labels" in result.require_text().lower()

    def test_non_string_label_rejected(self) -> None:
        runtime = _make_runtime()
        register_transformers_tools(runtime)
        tool = _get_tool(runtime, "kaos-nlp-transformers-ner-extract")
        result = _run(tool, {"texts": ["x"], "labels": ["person", 42]})
        assert result.isError is True
        assert "string" in result.require_text().lower()


class TestPiiDetectValidation:
    def test_missing_texts(self) -> None:
        runtime = _make_runtime()
        register_transformers_tools(runtime)
        tool = _get_tool(runtime, "kaos-nlp-transformers-pii-detect")
        result = _run(tool, {})
        assert result.isError is True
        assert "texts" in result.require_text().lower()

    def test_non_string_text_rejected(self) -> None:
        runtime = _make_runtime()
        register_transformers_tools(runtime)
        tool = _get_tool(runtime, "kaos-nlp-transformers-pii-detect")
        result = _run(tool, {"texts": ["ok", 7]})
        assert result.isError is True
        assert "string" in result.require_text().lower()


# ---------------------------------------------------------------------------
# Info tool surfaces all five registries now
# ---------------------------------------------------------------------------


def test_info_tool_surfaces_all_five_registries() -> None:
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-info")
    result = _run(tool, {})
    assert result.isError is False
    payload = result.structuredContent
    assert isinstance(payload, dict)
    # All five registry families surface their structured shape.
    for key in (
        "embedding_models",
        "reranker_models",
        "nli_models",
        "ner_models",
        "pii_models",
    ):
        assert key in payload, f"info tool missing {key}"
        section = payload[key]
        assert "registered" in section
        assert "excluded" in section
        assert isinstance(section["registered"], list)


def test_info_tool_surfaces_default_models_for_every_family() -> None:
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-info")
    result = _run(tool, {})
    assert result.isError is False
    settings = result.structuredContent["settings"]
    # The settings envelope should expose every default_* field.
    for key in (
        "default_model",
        "default_reranker_model",
        "default_nli_model",
        "default_ner_model",
        "default_pii_model",
    ):
        assert key in settings, f"info tool settings missing {key}"
        assert settings[key]  # non-empty
