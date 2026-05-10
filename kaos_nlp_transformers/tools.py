"""MCP tool registration for kaos-nlp-transformers.

Mirrors ``kaos_nlp_core.tools.register_nlp_tools`` in shape: a single
``register_transformers_tools(runtime)`` entry point that defines its
``KaosTool`` subclasses *inside* the function so ``kaos-core`` can stay a
soft dependency at import time. The package still works without
``kaos-core`` for the pure-Python embedding API; only the MCP surface
requires it.

Tool surface (v0):

    kaos-nlp-transformers-info
        Diagnostic envelope: registered models, resolved device, reachable
        and latent accelerators with install hints. Read-only, idempotent.

    kaos-nlp-transformers-embed
        Encode a list of texts into dense float32 vectors using the
        configured embedding model. Read-only.

    kaos-nlp-transformers-retrieve
        Build an inline EmbeddingRetriever over a passed list of documents
        and return the top-k cosine-similar hits for a query. Read-only.

    kaos-nlp-transformers-rerank
        Score (query, candidate) pairs with a cross-encoder reranker
        backed by the in-tree Rust ort cdylib and return them sorted
        by relevance. No extra required for CPU; the
        ``kaos-nlp-transformers-gpu`` companion wheel adds CUDA.
        Audit-06 KNT-501 retired the legacy ``[torch]`` requirement.

KNT-602 Option A (0.2.0a3): the ``kaos-nlp-transformers-dedup-semantic``
tool moved to kaos-content as ``kaos-content-dedup-semantic`` along
with the ``SemanticDedupLevel`` implementation. Install
``kaos-content[transformers,clustering]`` and use
``kaos_content.tools.register_content_tools`` to expose it.

Each tool catches package-level exceptions
(``ModelLoadError`` / ``EmbeddingError`` / ``DeviceNotReachableError`` /
``BackendNotInstalledError`` / ``ModelNotRegisteredError``) and converts
them into ``ToolResult.create_error`` with the three-part message contract
from ``docs/python/design/errors.md`` — what went wrong, how to fix it,
and an alternative path.
"""

from __future__ import annotations

from typing import Any, cast

from kaos_nlp_transformers._version import __version__

_MODULE = "kaos-nlp-transformers"
_VERSION = __version__

# Cap the size of payloads that round-trip through MCP. 1000 texts at 384-dim
# float32 is ~1.5 MB which is comfortably under MCP transport limits; tools
# that produce vectors enforce this so an agent that requests too many in one
# call gets a precise error instead of timing out the channel.
_MAX_EMBED_TEXTS: int = 1000
_MAX_RETRIEVE_DOCS: int = 5000
_MAX_RERANK_CANDIDATES: int = 1000


def register_transformers_tools(runtime: Any) -> int:
    """Register kaos-nlp-transformers MCP tools with a ``KaosRuntime``.

    Args:
        runtime: A ``kaos_core.KaosRuntime`` instance.

    Returns:
        Number of tools registered.

    Raises:
        ImportError: If ``kaos-core`` is not installed. The MCP server
            entry point (``serve.py``) catches this and prints a friendly
            install hint; library callers see the underlying error.
    """
    try:
        from kaos_core.base.context import KaosContext
        from kaos_core.base.tool import KaosTool
        from kaos_core.types.annotations import ToolAnnotations
        from kaos_core.types.enums import ToolCapability, ToolCategory
        from kaos_core.types.metadata import ToolMetadata
        from kaos_core.types.parameters import ParameterSchema
        from kaos_core.types.results import ToolResult
    except ImportError as exc:
        msg = (
            "kaos-core is required for MCP tool registration. "
            "Fix: pip install kaos-core. "
            "Alternative: use the Python API (EmbeddingModel, EmbeddingRetriever, "
            "CrossEncoderReranker) directly without the MCP surface."
        )
        raise ImportError(msg) from exc

    # All five tools are read-only, idempotent, and live in a closed world
    # (no outbound HTTP except first-time HuggingFace model download, which
    # itself is bounded by the registry). One shared annotations object keeps
    # the audit trail consistent across the surface.
    _RO_ANNOTATIONS = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    # ── 1. kaos-nlp-transformers-info ────────────────────────────────

    class InfoTool(KaosTool):
        """Diagnostic envelope: registered models, device map, settings."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-transformers-info",
                display_name="Inference Diagnostics",
                description=(
                    "Return diagnostics for kaos-nlp-transformers: registered "
                    "embedding and reranker models with pinned revisions, the "
                    "currently-resolved device + backend, all reachable "
                    "accelerators, and any LATENT accelerators that are "
                    "physically present but not reachable from this Python "
                    "install. Each latent entry carries an install_extra "
                    "field naming the pyproject extra "
                    "(`pip install kaos-nlp-transformers[<extra>]`) that "
                    "would convert it to reachable. Read-only, idempotent, "
                    "no network egress."
                ),
                category=ToolCategory.UTILITY,
                capability=ToolCapability.ANALYZE,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_RO_ANNOTATIONS,
                input_schema=[],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            # Local imports keep the registration cheap and let test stubs
            # monkeypatch the device probe without import-time side effects.
            # ``get_system_devices`` is the cached snapshot — running the
            # info tool repeatedly in a long-running MCP server should not
            # re-exec nvidia-smi each call.
            from kaos_nlp_transformers.device import get_system_devices, resolve_device
            from kaos_nlp_transformers.models import (
                EXCLUDED,
                REGISTRY,
                RERANKER_EXCLUDED,
                RERANKER_REGISTRY,
            )
            from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

            settings = KaosNLPTransformersSettings.from_context(context)
            system = get_system_devices()
            chosen = resolve_device(settings.device, system)

            payload: dict[str, Any] = {
                "module": _MODULE,
                "version": _VERSION,
                "settings": {
                    "default_model": settings.default_model,
                    "default_reranker_model": settings.default_reranker_model,
                    "device": settings.device,
                    "backend": settings.backend,
                    "offline": settings.offline,
                    "allow_unregistered": settings.allow_unregistered,
                    "profile": settings.profile,
                    "cache_dir": str(settings.cache_dir) if settings.cache_dir else None,
                    "workspace_root": (
                        str(settings.workspace_root) if settings.workspace_root else None
                    ),
                },
                "resolved_device": {
                    "name": chosen.name,
                    "device": chosen.device,
                    "backend": chosen.backend,
                    "memory_mb": chosen.memory_mb,
                },
                "reachable_devices": [
                    {
                        "name": d.name,
                        "device": d.device,
                        "backend": d.backend,
                        "memory_mb": d.memory_mb,
                    }
                    for d in system.devices
                ],
                "latent_devices": [
                    {
                        "name": d.name,
                        "kind": d.kind,
                        "reason": d.reason,
                        "install_extra": d.install_extra,
                        "install_hint": (
                            f"pip install kaos-nlp-transformers[{d.install_extra}]"
                            if d.install_extra
                            else None
                        ),
                        "detail": d.detail,
                    }
                    for d in system.latent_devices
                ],
                "onnx_providers": list(system.onnx_providers),
                "embedding_models": {
                    "registered": [
                        {
                            "model_id": m.model_id,
                            "revision": m.revision,
                            "license": m.license,
                            "params_m": m.params_m,
                            "dim": m.dim,
                            "backend": m.backend,
                            "notes": m.notes,
                        }
                        for m in REGISTRY.values()
                    ],
                    "excluded": [{"model_id": k, "reason": v} for k, v in sorted(EXCLUDED.items())],
                },
                "reranker_models": {
                    "registered": [
                        {
                            "model_id": m.model_id,
                            "revision": m.revision,
                            "license": m.license,
                            "params_m": m.params_m,
                            "backend": m.backend,
                            "notes": m.notes,
                        }
                        for m in RERANKER_REGISTRY.values()
                    ],
                    "excluded": [
                        {"model_id": k, "reason": v} for k, v in sorted(RERANKER_EXCLUDED.items())
                    ],
                },
            }

            n_latent = len(payload["latent_devices"])
            n_reachable = len(payload["reachable_devices"])
            summary_parts = [
                f"device={chosen.device} ({chosen.name})",
                f"backend={chosen.backend}",
                f"reachable={n_reachable}",
            ]
            if n_latent:
                hints = ", ".join(
                    str(d["install_extra"]) for d in payload["latent_devices"] if d["install_extra"]
                )
                summary_parts.append(f"latent={n_latent} (extras: {hints})")
            summary = "; ".join(summary_parts)

            return ToolResult.create_success(payload, summary=summary)

    # ── 2. kaos-nlp-transformers-embed ───────────────────────────────

    class EmbedTool(KaosTool):
        """Encode texts into dense float32 vectors."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-transformers-embed",
                display_name="Embed Texts",
                description=(
                    "Encode a list of texts into dense float32 embeddings "
                    "using the configured model (default "
                    "BAAI/bge-small-en-v1.5, 384-dim, MIT). Returns rows in "
                    "input order with L2-normalized vectors so cosine "
                    "similarity reduces to a dot product. For retrieval, "
                    "prefer kaos-nlp-transformers-retrieve which avoids "
                    "round-tripping the embedding matrix through MCP. "
                    f"Hard-cap: {_MAX_EMBED_TEXTS} texts per call."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.TRANSFORM,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="texts",
                        type="array",
                        description="Input texts to embed.",
                        constraints={"items": {"type": "string"}, "minItems": 1},
                    ),
                    ParameterSchema(
                        name="model_id",
                        type="string",
                        description=(
                            "Override the embedding model (must be present in "
                            "REGISTRY unless allow_unregistered is set)."
                        ),
                        required=False,
                        default=None,
                    ),
                    ParameterSchema(
                        name="batch_size",
                        type="integer",
                        description="Inference batch size (default 32).",
                        required=False,
                        default=32,
                        constraints={"minimum": 1, "maximum": 512},
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            import asyncio

            from kaos_nlp_transformers.embedding import EmbeddingModel
            from kaos_nlp_transformers.errors import (
                BackendNotInstalledError,
                DeviceNotReachableError,
                EmbeddingError,
                ModelLoadError,
                ModelNotRegisteredError,
            )
            from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

            texts = inputs.get("texts")
            if not isinstance(texts, list) or not texts:
                return ToolResult.create_error(
                    "Parameter 'texts' is required and must be a non-empty array. "
                    'Fix: pass `{"texts": ["…"]}` with at least one string. '
                    "Alternative: call kaos-nlp-transformers-info to see the "
                    "configured default model and the resolved device."
                )
            if any(not isinstance(t, str) for t in texts):
                return ToolResult.create_error(
                    "Every element of 'texts' must be a string. "
                    "Fix: cast or filter non-string elements client-side. "
                    "Alternative: split the call so each batch is homogeneous."
                )
            if len(texts) > _MAX_EMBED_TEXTS:
                return ToolResult.create_error(
                    f"Too many texts: {len(texts)} (cap {_MAX_EMBED_TEXTS}). "
                    "Fix: split the call into batches of at most "
                    f"{_MAX_EMBED_TEXTS} texts. "
                    "Alternative: build a persistent index with "
                    "kaos-nlp-transformers-retrieve and query by text instead "
                    "of returning the matrix."
                )

            settings = KaosNLPTransformersSettings.from_context(context)
            model_id = inputs.get("model_id") or settings.default_model
            batch_size = int(inputs.get("batch_size") or 32)

            try:
                model = EmbeddingModel.load(model_id, settings=settings)
                # ``embed`` is sync + CPU/GPU-bound; dispatch to a thread so we
                # don't block the event loop in a stdio MCP server.
                arr = await asyncio.to_thread(model.embed, texts, batch_size=batch_size)
            except (
                ModelNotRegisteredError,
                ModelLoadError,
                EmbeddingError,
                DeviceNotReachableError,
                BackendNotInstalledError,
            ) as exc:
                return ToolResult.create_error(str(exc))
            except Exception as exc:
                return ToolResult.create_error(
                    f"Embedding failed for model {model_id!r}: {exc}. "
                    "Fix: call kaos-nlp-transformers-info to confirm the model "
                    "is registered and the device is reachable. "
                    "Alternative: try device='cpu' to bypass GPU issues."
                )

            payload = {
                "model_id": model.model_id,
                "dim": model.dim,
                "backend": model.backend_name,
                "device": model.device.device if model.device else "unknown",
                "shape": [int(arr.shape[0]), int(arr.shape[1])],
                "embeddings": arr.tolist(),
            }
            return ToolResult.create_success(
                payload,
                summary=(
                    f"Embedded {arr.shape[0]} text(s) → {arr.shape[1]}-dim "
                    f"via {model.backend_name} on "
                    f"{model.device.device if model.device else 'unknown'}"
                ),
            )

    # ── 3. kaos-nlp-transformers-retrieve ────────────────────────────

    class RetrieveTool(KaosTool):
        """Inline cosine-similarity retrieval over a list of documents."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-transformers-retrieve",
                display_name="Embed and Retrieve",
                description=(
                    "Build an inline EmbeddingRetriever over a list of "
                    "documents and return the top-k cosine-similar hits for "
                    "a query. Both query and corpus are embedded with the "
                    "configured model. For best quality on adversarial "
                    "queries, follow this with "
                    "kaos-nlp-transformers-rerank. v0 keeps the index "
                    "in-call (no persistence); persistent corpora belong on "
                    f"kaos-llm-core. Hard-cap: {_MAX_RETRIEVE_DOCS} docs."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.QUERY,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="query",
                        type="string",
                        description="Query text.",
                    ),
                    ParameterSchema(
                        name="documents",
                        type="array",
                        description=(
                            "List of documents to index. Each item can be a "
                            "string OR an object with `text` plus optional "
                            "`doc_id` (int), `external_id` (str), and "
                            "`metadata` (object)."
                        ),
                        constraints={"minItems": 1},
                    ),
                    ParameterSchema(
                        name="top_k",
                        type="integer",
                        description="Number of hits to return (default 10).",
                        required=False,
                        default=10,
                        constraints={"minimum": 1, "maximum": 1000},
                    ),
                    ParameterSchema(
                        name="model_id",
                        type="string",
                        description="Override the embedding model.",
                        required=False,
                        default=None,
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            import asyncio

            from kaos_nlp_transformers.errors import (
                BackendNotInstalledError,
                DeviceNotReachableError,
                EmbeddingError,
                ModelLoadError,
                ModelNotRegisteredError,
            )
            from kaos_nlp_transformers.retrieval import EmbeddingRetriever
            from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

            query = inputs.get("query")
            documents = inputs.get("documents")
            if not isinstance(query, str) or not query.strip():
                return ToolResult.create_error(
                    "Parameter 'query' is required and must be a non-empty string. "
                    'Fix: pass `{"query": "…"}`. '
                    "Alternative: use kaos-nlp-transformers-embed if you only "
                    "need vectors and not retrieval."
                )
            if not isinstance(documents, list) or not documents:
                return ToolResult.create_error(
                    "Parameter 'documents' is required and must be a non-empty array. "
                    'Fix: pass `{"documents": ["…"]}` (strings or objects '
                    "with a `text` field). "
                    "Alternative: call kaos-nlp-transformers-info to confirm "
                    "the package is configured before retrieval."
                )
            if len(documents) > _MAX_RETRIEVE_DOCS:
                return ToolResult.create_error(
                    f"Too many documents: {len(documents)} (cap {_MAX_RETRIEVE_DOCS}). "
                    "Fix: pre-shard the corpus and run multiple retrieve "
                    "calls. "
                    "Alternative: build a persistent index in kaos-llm-core "
                    "for corpora above this scale."
                )

            # Normalize documents into parallel arrays. Strings get auto doc_ids.
            texts: list[str] = []
            doc_ids: list[int] = []
            external_ids: list[str | None] = []
            metadata_list: list[dict[str, Any]] = []
            for idx, raw_item in enumerate(documents):
                if isinstance(raw_item, str):
                    texts.append(raw_item)
                    doc_ids.append(idx)
                    external_ids.append(None)
                    metadata_list.append({})
                    continue
                if isinstance(raw_item, dict):
                    item = cast(dict[str, Any], raw_item)
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        texts.append(text_value)
                        doc_ids.append(int(item.get("doc_id", idx)))
                        ext = item.get("external_id")
                        external_ids.append(ext if isinstance(ext, str) else None)
                        md = item.get("metadata")
                        metadata_list.append(
                            cast(dict[str, Any], md) if isinstance(md, dict) else {}
                        )
                        continue
                return ToolResult.create_error(
                    f"documents[{idx}] is not a string or object with a "
                    "`text` field. "
                    "Fix: use either a plain string or "
                    '`{"text": "…", "doc_id": 0, "external_id": '
                    '"…", "metadata": {…}}`. '
                    "Alternative: drop the malformed item client-side."
                )

            settings = KaosNLPTransformersSettings.from_context(context)
            model_id = inputs.get("model_id") or settings.default_model
            top_k = int(inputs.get("top_k") or 10)

            try:
                retriever = await asyncio.to_thread(
                    EmbeddingRetriever.from_texts,
                    texts=texts,
                    doc_ids=doc_ids,
                    external_ids=external_ids,
                    metadata_list=metadata_list,
                    model_id=model_id,
                    settings=settings,
                )
                hits = await retriever.retrieve(query, top_k=top_k)
            except (
                ModelNotRegisteredError,
                ModelLoadError,
                EmbeddingError,
                DeviceNotReachableError,
                BackendNotInstalledError,
            ) as exc:
                return ToolResult.create_error(str(exc))
            except Exception as exc:
                return ToolResult.create_error(
                    f"Retrieval failed: {exc}. "
                    "Fix: verify documents are non-empty strings. "
                    "Alternative: use kaos-nlp-transformers-embed to isolate "
                    "whether the failure is at the embedding or the search step."
                )

            payload = {
                "query": query,
                "model_id": model_id,
                "num_documents": len(texts),
                "top_k": top_k,
                "hits": [
                    {
                        "doc_id": h.doc_id,
                        "score": h.score,
                        "text": h.text,
                        "external_id": h.external_id,
                        "metadata": h.metadata,
                    }
                    for h in hits
                ],
            }
            return ToolResult.create_success(
                payload, summary=f"{len(hits)} hit(s) over {len(texts)} doc(s)"
            )

    # ── 4. kaos-nlp-transformers-rerank ──────────────────────────────

    class RerankTool(KaosTool):
        """Cross-encoder reranking of (query, candidate) pairs."""

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="kaos-nlp-transformers-rerank",
                display_name="Cross-Encoder Rerank",
                description=(
                    "Score (query, candidate) pairs with a cross-encoder "
                    "reranker (default BAAI/bge-reranker-base, MIT) via "
                    "fastembed.TextCrossEncoder (ONNX) and return them "
                    "sorted by relevance. Sigmoid-normalized scores in "
                    "[0, 1]. Pair with kaos-nlp-transformers-retrieve: "
                    "take its top-50 hits, rerank to top-10. CPU works "
                    "out of the box; install [gpu] for CUDA acceleration. "
                    f"Hard-cap: {_MAX_RERANK_CANDIDATES} candidates per call."
                ),
                category=ToolCategory.TEXT,
                capability=ToolCapability.QUERY,
                module_name=_MODULE,
                version=_VERSION,
                annotations=_RO_ANNOTATIONS,
                input_schema=[
                    ParameterSchema(
                        name="query",
                        type="string",
                        description="Query text.",
                    ),
                    ParameterSchema(
                        name="candidates",
                        type="array",
                        description=(
                            "Candidate texts to rerank. Each item is either "
                            "a string OR an object with `text` plus optional "
                            "`doc_id`, `external_id`, `metadata`."
                        ),
                        constraints={"minItems": 1},
                    ),
                    ParameterSchema(
                        name="top_k",
                        type="integer",
                        description="Truncate to top-k after reranking (default: keep all).",
                        required=False,
                        default=None,
                        constraints={"minimum": 1},
                    ),
                    ParameterSchema(
                        name="model_id",
                        type="string",
                        description="Override the reranker model.",
                        required=False,
                        default=None,
                    ),
                ],
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            from kaos_nlp_core.retrieval.protocol import RetrievalResult

            from kaos_nlp_transformers.errors import (
                BackendNotInstalledError,
                DeviceNotReachableError,
                ModelLoadError,
                ModelNotRegisteredError,
            )
            from kaos_nlp_transformers.reranker import CrossEncoderReranker
            from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

            query = inputs.get("query")
            candidates = inputs.get("candidates")
            if not isinstance(query, str) or not query.strip():
                return ToolResult.create_error(
                    "Parameter 'query' is required and must be a non-empty string. "
                    'Fix: pass `{"query": "…"}`. '
                    "Alternative: skip reranking and use the score from "
                    "kaos-nlp-transformers-retrieve directly."
                )
            if not isinstance(candidates, list) or not candidates:
                return ToolResult.create_error(
                    "Parameter 'candidates' is required and must be a non-empty array. "
                    'Fix: pass `{"candidates": ["…"]}`. '
                    "Alternative: use kaos-nlp-transformers-retrieve to "
                    "produce candidates from a corpus first."
                )
            if len(candidates) > _MAX_RERANK_CANDIDATES:
                return ToolResult.create_error(
                    f"Too many candidates: {len(candidates)} "
                    f"(cap {_MAX_RERANK_CANDIDATES}). "
                    "Fix: pre-filter to the retriever's top-N first. "
                    "Alternative: split the call into batches and merge "
                    "client-side."
                )

            # ``RetrievalResult`` (kaos-nlp-core) uses ``doc_id: str`` and
            # has no ``external_id`` field — provenance flows through
            # ``metadata`` instead. Coerce/lift caller payload here so the
            # rerank step is decoupled from upstream id types.
            results: list[RetrievalResult] = []
            for idx, raw_item in enumerate(candidates):
                if isinstance(raw_item, str):
                    results.append(
                        RetrievalResult(text=raw_item, score=0.0, doc_id=str(idx), metadata={})
                    )
                    continue
                if isinstance(raw_item, dict):
                    item = cast(dict[str, Any], raw_item)
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        md_raw = item.get("metadata")
                        md: dict[str, Any] = (
                            dict(cast(dict[str, Any], md_raw)) if isinstance(md_raw, dict) else {}
                        )
                        ext = item.get("external_id")
                        if isinstance(ext, str):
                            # Preserve external_id round-trip via metadata so
                            # callers can correlate ranked output back to their
                            # own ids without us widening the kaos-nlp-core
                            # RetrievalResult type.
                            md.setdefault("external_id", ext)
                        results.append(
                            RetrievalResult(
                                text=text_value,
                                score=float(item.get("score", 0.0)),
                                doc_id=str(item.get("doc_id", idx)),
                                metadata=md,
                            )
                        )
                        continue
                return ToolResult.create_error(
                    f"candidates[{idx}] is not a string or object with a "
                    "`text` field. "
                    "Fix: use a plain string or "
                    '`{"text": "…", "doc_id": 0}`. '
                    "Alternative: drop the malformed item client-side."
                )

            settings = KaosNLPTransformersSettings.from_context(context)
            model_id = inputs.get("model_id") or settings.default_reranker_model
            top_k_raw = inputs.get("top_k")
            top_k = int(top_k_raw) if top_k_raw is not None else None

            try:
                reranker = CrossEncoderReranker.load(model_id, settings=settings)
                ranked = await reranker.rerank(query, results, top_k=top_k)
            except BackendNotInstalledError as exc:
                # Friendly fallthrough: a backend dep is missing (most
                # commonly fastembed itself, or onnxruntime-gpu when the
                # caller asked for CUDA). The exception message already
                # carries the install hint.
                return ToolResult.create_error(str(exc))
            except (
                ModelNotRegisteredError,
                ModelLoadError,
                DeviceNotReachableError,
            ) as exc:
                return ToolResult.create_error(str(exc))
            except Exception as exc:
                return ToolResult.create_error(
                    f"Reranking failed for model {model_id!r}: {exc}. "
                    "Fix: call kaos-nlp-transformers-info to confirm the "
                    "reranker is registered and the device is reachable. "
                    "Alternative: skip reranking and trust the retriever's "
                    "cosine score."
                )

            payload = {
                "query": query,
                "model_id": model_id,
                "ranked": [
                    {
                        "doc_id": r.result.doc_id,
                        "rerank_score": r.rerank_score,
                        "retriever_score": r.result.score,
                        "text": r.result.text,
                        # external_id, if the caller supplied one, was tucked
                        # into metadata above; surface it back at the top
                        # level so round-tripping stays clean.
                        "external_id": r.result.metadata.get("external_id"),
                        "metadata": {
                            k: v for k, v in r.result.metadata.items() if k != "external_id"
                        },
                    }
                    for r in ranked
                ],
            }
            return ToolResult.create_success(
                payload, summary=f"Reranked {len(candidates)} → kept {len(ranked)}"
            )

    # NOTE: kaos-nlp-transformers-dedup-semantic was REMOVED in 0.2.0a3
    # (KNT-602 Option A). The MCP tool moved to kaos-content as
    # `kaos-content-dedup-semantic`; the SemanticDedupLevel impl moved
    # to `kaos_content.dedup.levels.semantic`. No deprecation cycle —
    # see kaos-nlp-transformers CHANGELOG and kaos-content CHANGELOG
    # 0.1.0a3 for the breaking-change rationale.

    # ── Registration ─────────────────────────────────────────────────

    tool_classes: list[type[KaosTool]] = [
        InfoTool,
        EmbedTool,
        RetrieveTool,
        RerankTool,
    ]

    count = 0
    for cls in tool_classes:
        runtime.tools.register_tool(cls())
        count += 1
    return count


__all__ = ["register_transformers_tools"]
