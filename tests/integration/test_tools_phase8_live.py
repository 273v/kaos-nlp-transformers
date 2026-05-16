"""Live MCP integration tests for the Phase-8 tools.

Hits real Rust backends through the MCP tool wrappers. Verifies the
JSON-RPC envelope shape end-to-end:
``tool.execute(inputs) -> ToolResult`` with ``isError=False``,
``structuredContent`` containing the expected fields.

Skips when ``KAOS_NLP_TRANSFORMERS_OFFLINE=1`` or when the Rust
extension hasn't been built.

Marked ``@pytest.mark.integration`` and ``@pytest.mark.live``.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.live]

kaos_core = pytest.importorskip("kaos_core")

from kaos_core import KaosRuntime  # noqa: E402
from kaos_core.base.tool import KaosTool  # noqa: E402

from kaos_nlp_transformers.tools import register_transformers_tools  # noqa: E402


def _skip_if_offline() -> None:
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        pytest.skip("offline mode set")


def _skip_if_no_rust_extension() -> None:
    try:
        from kaos_nlp_transformers._rust import ner, nli, token_classify  # noqa: F401
    except ImportError:
        pytest.skip("kaos_nlp_transformers._rust extension is not built")


@pytest.fixture(scope="module")
def runtime() -> KaosRuntime:
    _skip_if_offline()
    _skip_if_no_rust_extension()
    rt = KaosRuntime()
    register_transformers_tools(rt)
    return rt


def _get(runtime: KaosRuntime, name: str) -> KaosTool:
    tool = runtime.tools.get_tool(name)
    assert tool is not None
    return tool


def _run(tool: KaosTool, inputs: dict):
    return asyncio.run(tool.execute(inputs))


# ---------------------------------------------------------------------------
# NLI classify
# ---------------------------------------------------------------------------


def test_nli_classify_live(runtime: KaosRuntime) -> None:
    tool = _get(runtime, "kaos-nlp-transformers-nli-classify")
    result = _run(
        tool,
        {
            "premise": "Acme Corp shall pay rent of $5,000/month.",
            "hypotheses": [
                "This text is about a lease agreement.",
                "This text is about employment.",
            ],
        },
    )
    assert result.isError is False, result.require_text()
    payload = result.structuredContent
    assert isinstance(payload, dict)
    assert payload["model_id"]
    assert payload["device"]
    rows = payload["scores"]
    assert len(rows) == 2
    for row in rows:
        # softmax over (entailment, neutral, contradiction) should sum ~1
        total = row["entailment"] + row["neutral"] + row["contradiction"]
        assert 0.99 <= total <= 1.01
    # argmax-related fields populated
    assert payload["argmax_hypothesis"] in (
        "This text is about a lease agreement.",
        "This text is about employment.",
    )
    assert 0.0 <= payload["argmax_entailment"] <= 1.0


# ---------------------------------------------------------------------------
# NER extract
# ---------------------------------------------------------------------------


def test_ner_extract_live(runtime: KaosRuntime) -> None:
    tool = _get(runtime, "kaos-nlp-transformers-ner-extract")
    result = _run(
        tool,
        {
            "texts": ["Barack Obama was born in Hawaii."],
            "labels": ["person", "place"],
        },
    )
    assert result.isError is False, result.require_text()
    payload = result.structuredContent
    assert isinstance(payload, dict)
    assert payload["n_texts"] == 1
    assert payload["n_entities_total"] >= 1
    entities = payload["entities"][0]
    labels = {e["label"] for e in entities}
    assert "person" in labels or "place" in labels
    # Char-offset round trip per entity.
    src = "Barack Obama was born in Hawaii."
    for e in entities:
        assert src[e["start"] : e["end"]] == e["text"]


# ---------------------------------------------------------------------------
# PII detect
# ---------------------------------------------------------------------------


def test_pii_detect_live(runtime: KaosRuntime) -> None:
    tool = _get(runtime, "kaos-nlp-transformers-pii-detect")
    src = "Contact Jennifer Stacey at jen.stacey@galera.com today."
    result = _run(tool, {"texts": [src]})
    assert result.isError is False, result.require_text()
    payload = result.structuredContent
    assert isinstance(payload, dict)
    assert payload["n_texts"] == 1
    assert payload["n_entities_total"] >= 1
    labels = {e["label"] for e in payload["entities"][0]}
    assert "PERSON" in labels or "EMAIL_ADDRESS" in labels
    # Available label vocab is exposed.
    assert isinstance(payload["available_labels"], list)
    assert "PERSON" in payload["available_labels"]
    # Char offsets round-trip.
    for e in payload["entities"][0]:
        assert src[e["start"] : e["end"]] == e["text"]


# ---------------------------------------------------------------------------
# Info tool surfaces all five families
# ---------------------------------------------------------------------------


def test_info_tool_lists_pii_family_live(runtime: KaosRuntime) -> None:
    tool = _get(runtime, "kaos-nlp-transformers-info")
    result = _run(tool, {})
    assert result.isError is False
    payload = result.structuredContent
    pii_ids = {m["model_id"] for m in payload["pii_models"]["registered"]}
    assert "onnx-community/bert-small-pii-detection-ONNX" in pii_ids
    nli_ids = {m["model_id"] for m in payload["nli_models"]["registered"]}
    assert "Xenova/nli-deberta-v3-base" in nli_ids
    ner_ids = {m["model_id"] for m in payload["ner_models"]["registered"]}
    assert "onnx-community/gliner_medium-v2.1" in ner_ids
