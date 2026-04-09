# kaos-nlp-transformers Development Notes

## Purpose

Sibling to `kaos-nlp-core`. Hosts heavy ML model dependencies
(`fastembed` in v0; future `[torch]` extra for `sentence-transformers`,
`transformers`, `torch`) without polluting the core platform's dep tree.

Sole job in v0: produce dense embeddings for `kaos-ml-core` to consume
as feature matrices via the `embed_corpus()` shim.

## Architecture

Pure Python (no Rust crate, hatchling build backend, not Maturin).
fastembed already wraps ONNX Runtime in C++/Rust under the hood —
adding a second native crate on top would gain nothing.

```
Application code / kaos-ml-core / MCP clients
    ↓
Python API (kaos_nlp_transformers/)
    EmbeddingModel · Reranker (v1.1) · ZeroShotClassifier (v1.2)
    models · settings · errors · adapters (v1.3)
    ↓
Backend layer (fastembed default; sentence-transformers via [torch] extra)
    ↓
Native runtimes — ONNX Runtime · (torch when [torch] enabled)
```

## Dependencies

- **Hard deps:** `kaos-core`, `kaos-content`, `numpy`, `fastembed`
- **`[torch]` extra:** `torch>=2.5`, `transformers>=4.45`,
  `sentence-transformers>=5.0` — required for zero-shot NLI
  classification (v1.2). Default install is fastembed-only.
- **`[mcp]` extra:** `kaos-mcp` (v1.3 MCP tools)

## v0 surface

One class, two methods. Nothing else.

```python
from kaos_nlp_transformers import EmbeddingModel

model = EmbeddingModel.load("BAAI/bge-small-en-v1.5")
vecs = model.embed(["text 1", "text 2"])  # → np.float32 (2, 384)
```

The v0 model registry has exactly one entry:
`BAAI/bge-small-en-v1.5` (33M, MIT). Phase v1.0 broadens to bge-m3,
arctic-embed-m, mxbai-xsmall, nomic-embed-v1.5.

## Hard rules

1. **Never add a model to the registry without a license check.** The
   exclusion list in `kaos_nlp_transformers.models.EXCLUDED` is binding.
   Currently flags: jina-v3 (CC-BY-NC), NV-Embed (CC-BY-NC), Qwen3-Embedding
   (MS MARCO ambiguity).
2. **Always pin model revisions to a commit SHA.** Never load `main`.
   Min 7-char SHA enforced by the registry shape test.
3. **Never depend on `transformers` or `torch` outside the `[torch]`
   extra.** The default install must remain fastembed-only.
4. **Never bypass `KAOS_NLP_TRANSFORMERS_OFFLINE=true`.** Offline mode
   must refuse network access — even for first-time downloads.
5. **Never read `os.environ` in inference internals.** Settings are
   loaded at the edge and passed in.
6. **Live integration tests are the quality bar.** Mocked tests are
   supplementary. Per the platform-wide no-fake-tests rule.
7. **Never add a model to the registry without verifying it loads in
   fastembed.** Half the HF Hub is "released" but doesn't actually run
   without custom code paths.
8. **Never add AGPL/GPL dependencies.** This is a proprietary codebase.

## QA Sequence (mandatory)

```bash
ruff format kaos_nlp_transformers/ tests/
ruff check --fix kaos_nlp_transformers/ tests/
ty check kaos_nlp_transformers/ tests/
pytest tests/ -v
```

## Documentation

- PRD: `docs/internal/prd/kaos-nlp-transformers.md`
- v0 plan + per-phase roadmap: `docs/internal/plans/kaos-nlp-transformers-v0.md`
- Sibling package: `docs/internal/prd/kaos-ml-core.md` (consumer)

When adding a new MCP tool (Phase v1.3+), also update:
`docs/index.md`, `docs/architecture.md`, `docs/reference/mcp-inventory.md`,
and `_KNOWN_TOOL_COUNTS` in `kaos-mcp/kaos_mcp/management/status.py`.
