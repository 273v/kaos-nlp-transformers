"""Unit tests for the kaos-nlp-transformers MCP tool surface."""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit

# kaos-core is required for register_transformers_tools. The package can be
# used standalone without it (Python API only), and the tools-module test
# exists to confirm the kaos-core path stays wired. If the dev env somehow
# runs without kaos-core, skip cleanly.
kaos_core = pytest.importorskip("kaos_core")


from kaos_core import KaosRuntime  # noqa: E402
from kaos_core.base.tool import KaosTool  # noqa: E402

from kaos_nlp_transformers.device import DeviceInfo, LatentDevice, SystemDevices  # noqa: E402
from kaos_nlp_transformers.tools import register_transformers_tools  # noqa: E402


def _make_runtime() -> KaosRuntime:
    return KaosRuntime()


def _get_info_tool(runtime: KaosRuntime) -> KaosTool:
    """Resolve the info tool with a typed assertion so ty can prove non-None."""
    tool = runtime.tools.get_tool("kaos-nlp-transformers-info")
    assert tool is not None, "register_transformers_tools must register the info tool"
    return tool


def test_register_transformers_tools_returns_count():
    runtime = _make_runtime()
    n = register_transformers_tools(runtime)
    assert n == 5


def test_register_transformers_tools_names():
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    expected = {
        "kaos-nlp-transformers-info",
        "kaos-nlp-transformers-embed",
        "kaos-nlp-transformers-retrieve",
        "kaos-nlp-transformers-rerank",
        "kaos-nlp-transformers-dedup-semantic",
    }
    assert expected.issubset(set(runtime.tools.list_tools()))


def _get_tool(runtime: KaosRuntime, name: str) -> KaosTool:
    """Resolve any registered tool with a typed assertion."""
    tool = runtime.tools.get_tool(name)
    assert tool is not None, f"register_transformers_tools must register {name}"
    return tool


def test_info_tool_metadata_shape():
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_info_tool(runtime)
    md = tool.metadata
    assert md.name == "kaos-nlp-transformers-info"
    assert md.module_name == "kaos-nlp-transformers"
    assert md.version  # _VERSION resolves
    assert md.annotations is not None
    assert md.annotations.readOnlyHint is True
    assert md.annotations.idempotentHint is True
    assert md.annotations.openWorldHint is False
    # Validates against the MCP-safe hyphenated naming requirement.
    assert md.name == "kaos-nlp-transformers-info"


def _stub_system_devices(monkeypatch, *, reachable, latent=()):
    """Force ``device.get_system_devices()`` to return a fixed SystemDevices.

    The InfoTool now reads through the cached snapshot, not the raw probe, so
    we stub at that layer. Downstream call sites (CLI, EmbeddingModel.load)
    flow through the same cache, so a single stub covers all of them.
    """
    sd = SystemDevices(
        devices=tuple(reachable),
        onnx_providers=("CPUExecutionProvider",),
        latent_devices=tuple(latent),
    )
    monkeypatch.setattr("kaos_nlp_transformers.device.get_system_devices", lambda: sd)


def test_info_tool_execute_cpu_only_box(monkeypatch):
    cpu = DeviceInfo(name="CPU", device="cpu", backend="fastembed")
    _stub_system_devices(monkeypatch, reachable=[cpu])

    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_info_tool(runtime)

    result = asyncio.run(tool.execute({}, None))
    assert result.isError is False
    payload = result.structuredContent
    assert payload is not None
    assert payload["module"] == "kaos-nlp-transformers"
    assert payload["resolved_device"]["device"] == "cpu"
    assert len(payload["reachable_devices"]) == 1
    assert payload["latent_devices"] == []
    assert any(
        m["model_id"] == "BAAI/bge-small-en-v1.5" for m in payload["embedding_models"]["registered"]
    )
    # Summary line goes to TextContent — agents see this without parsing JSON.
    assert "device=cpu" in result.require_text()


def test_info_tool_execute_surfaces_latent_devices(monkeypatch):
    """The motivating UX: an agent calling info on a GPU box with the base
    install MUST see the latent devices and the install hint."""
    cpu = DeviceInfo(name="CPU", device="cpu", backend="fastembed")
    latent = [
        LatentDevice(
            name="NVIDIA GeForce RTX 5070 Ti",
            kind="cuda",
            reason="torch is not installed",
            install_extra="torch",
            detail={"index": 0, "memory_mb": 16303},
        ),
        LatentDevice(
            name="NVIDIA GeForce RTX 4070 Ti SUPER",
            kind="cuda",
            reason="torch is not installed",
            install_extra="torch",
            detail={"index": 1, "memory_mb": 16376},
        ),
    ]
    _stub_system_devices(monkeypatch, reachable=[cpu], latent=latent)

    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_info_tool(runtime)

    result = asyncio.run(tool.execute({}, None))
    payload = result.structuredContent
    assert payload is not None
    assert len(payload["latent_devices"]) == 2
    first = payload["latent_devices"][0]
    assert first["kind"] == "cuda"
    assert first["install_extra"] == "torch"
    assert first["install_hint"] == "pip install kaos-nlp-transformers[torch]"
    assert first["detail"]["memory_mb"] == 16303
    # Summary names the latent extras so agents can lift them out of the
    # text channel without parsing structured content.
    summary = result.require_text()
    assert "latent=2" in summary
    assert "torch" in summary


def test_info_tool_execute_includes_reranker_registry(monkeypatch):
    cpu = DeviceInfo(name="CPU", device="cpu", backend="fastembed")
    _stub_system_devices(monkeypatch, reachable=[cpu])

    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_info_tool(runtime)

    result = asyncio.run(tool.execute({}, None))
    payload = result.structuredContent
    assert payload is not None
    rerankers = payload["reranker_models"]["registered"]
    assert any(r["model_id"] == "BAAI/bge-reranker-base" for r in rerankers)


# -- embed -----------------------------------------------------------------


def test_embed_tool_rejects_empty_texts():
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-embed")

    result = asyncio.run(tool.execute({"texts": []}, None))
    assert result.isError is True
    msg = result.require_text()
    assert "non-empty array" in msg
    assert "Fix:" in msg
    assert "Alternative:" in msg


def test_embed_tool_rejects_non_string_elements():
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-embed")

    result = asyncio.run(tool.execute({"texts": ["ok", 42]}, None))
    assert result.isError is True
    assert "must be a string" in result.require_text()


def test_embed_tool_enforces_cap(monkeypatch):
    """Cap protects the MCP channel from oversized payloads."""
    monkeypatch.setattr("kaos_nlp_transformers.tools._MAX_EMBED_TEXTS", 3)
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-embed")

    result = asyncio.run(tool.execute({"texts": ["a", "b", "c", "d"]}, None))
    assert result.isError is True
    assert "Too many texts" in result.require_text()
    assert "cap 3" in result.require_text()


@pytest.mark.live
def test_embed_tool_happy_path():
    """Live test — exercises the real fastembed CPU path."""
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-embed")

    result = asyncio.run(tool.execute({"texts": ["hello", "world"]}, None))
    assert result.isError is False, result.require_text()
    payload = result.structuredContent
    assert payload is not None
    assert payload["dim"] == 384
    assert payload["shape"] == [2, 384]
    assert len(payload["embeddings"]) == 2
    assert len(payload["embeddings"][0]) == 384


# -- retrieve --------------------------------------------------------------


def test_retrieve_tool_rejects_empty_query():
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-retrieve")

    result = asyncio.run(tool.execute({"query": "", "documents": ["a"]}, None))
    assert result.isError is True
    assert "non-empty string" in result.require_text()


def test_retrieve_tool_rejects_malformed_documents():
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-retrieve")

    # Number is neither a string nor an object with a `text` field.
    result = asyncio.run(tool.execute({"query": "q", "documents": [123]}, None))
    assert result.isError is True
    assert "documents[0]" in result.require_text()


@pytest.mark.live
def test_retrieve_tool_happy_path():
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-retrieve")

    result = asyncio.run(
        tool.execute(
            {
                "query": "where do disputes go?",
                "documents": [
                    "The buyer agrees to mediation in Delaware.",
                    "All disputes shall be resolved by arbitration in New York.",
                    "Force majeure clauses excuse performance.",
                ],
                "top_k": 2,
            },
            None,
        )
    )
    assert result.isError is False, result.require_text()
    payload = result.structuredContent
    assert payload is not None
    assert len(payload["hits"]) == 2
    # First hit should be one of the dispute-resolution docs (0 or 1), not 2.
    assert payload["hits"][0]["doc_id"] in {0, 1}


# -- rerank ----------------------------------------------------------------


def test_rerank_tool_friendly_error_when_torch_missing(monkeypatch):
    """Without [torch], the tool surfaces a structured install hint, not a stack trace."""
    from kaos_nlp_transformers.errors import BackendNotInstalledError

    def _raise(*args, **kwargs):
        raise BackendNotInstalledError(
            "sentence-transformers is not installed. "
            "Fix: install the torch extras via `pip install "
            "kaos-nlp-transformers[torch]`. "
            "Alternative: use JudgeReranker (LLM-based) from kaos-llm-core."
        )

    monkeypatch.setattr(
        "kaos_nlp_transformers.reranker.CrossEncoderReranker.load",
        classmethod(lambda cls, *a, **kw: _raise()),
    )

    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-rerank")

    result = asyncio.run(
        tool.execute({"query": "q", "candidates": ["a", "b"]}, None),
    )
    assert result.isError is True
    msg = result.require_text()
    assert "kaos-nlp-transformers[torch]" in msg
    assert "Fix:" in msg


def test_rerank_tool_rejects_empty_candidates():
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-rerank")

    result = asyncio.run(tool.execute({"query": "q", "candidates": []}, None))
    assert result.isError is True
    assert "non-empty array" in result.require_text()


# -- dedup-semantic --------------------------------------------------------


def test_dedup_tool_requires_two_documents():
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-dedup-semantic")

    result = asyncio.run(
        tool.execute(
            {"documents": [{"doc_id": "a", "text": "x"}]},
            None,
        )
    )
    assert result.isError is True
    assert "at least 2 entries" in result.require_text()


def test_dedup_tool_rejects_non_string_doc_id():
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-dedup-semantic")

    result = asyncio.run(
        tool.execute(
            {
                "documents": [
                    {"doc_id": 1, "text": "x"},
                    {"doc_id": 2, "text": "y"},
                ]
            },
            None,
        )
    )
    assert result.isError is True
    assert "string `doc_id` and `text`" in result.require_text()


@pytest.mark.live
def test_dedup_tool_happy_path():
    """Live test — needs scipy + fastembed model. Gated on `live` marker."""
    pytest.importorskip("scipy")
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-dedup-semantic")

    result = asyncio.run(
        tool.execute(
            {
                "documents": [
                    {"doc_id": "a", "text": "Force majeure clauses excuse performance."},
                    {"doc_id": "b", "text": "Force majeure provisions excuse performance."},
                    {"doc_id": "c", "text": "Indemnity caps the liability of the seller."},
                ],
                "distance_threshold": 0.15,
            },
            None,
        )
    )
    assert result.isError is False, result.require_text()
    payload = result.structuredContent
    assert payload is not None
    # The two near-duplicates should land in the same cluster, the third alone.
    assert len(payload["clusters"]) == 1
    cluster = payload["clusters"][0]
    assert set(cluster["member_doc_ids"]) == {"a", "b"}
    assert 0.0 <= cluster["similarity"] <= 1.0


def test_dedup_tool_rejects_oversized_input(monkeypatch):
    monkeypatch.setattr("kaos_nlp_transformers.tools._MAX_DEDUP_DOCS", 3)
    runtime = _make_runtime()
    register_transformers_tools(runtime)
    tool = _get_tool(runtime, "kaos-nlp-transformers-dedup-semantic")

    docs = [{"doc_id": str(i), "text": "x"} for i in range(4)]
    result = asyncio.run(tool.execute({"documents": docs}, None))
    assert result.isError is True
    assert "Too many documents" in result.require_text()
