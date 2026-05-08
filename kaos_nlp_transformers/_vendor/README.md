# Vendored model artifacts

This directory ships pre-downloaded model weights and tokenizers inside
the `kaos-nlp-transformers` wheel so that the loader can resolve them
without touching the network. It is **not** part of the public Python
API — nothing in `kaos_nlp_transformers/__init__.py` exports it.

## Currently vendored

### `potion-base-8M/`

- **Source**: `https://huggingface.co/minishlab/potion-base-8M`
- **Revision**: `bf8b056651a2c21b8d2565580b8569da283cab23` (pinned in
  `kaos_nlp_transformers/models.py` REGISTRY)
- **License**: MIT (Copyright (c) MinishLab — see ATTRIBUTION.txt)
- **On-disk size**: ~31 MB (safetensors + tokenizer + small JSONs;
  the `onnx/model.onnx` from upstream is **NOT** vendored — model2vec
  reads safetensors)
- **Vendored at**: 2026-05-08

### Why this directory exists

The wheel ships ~33 MB instead of ~2 MB so that:

1. Air-gapped / firewalled deployments work without HF cache priming
2. First-call latency is zero (no 5-second snapshot download)
3. Reproducibility is locked to the wheel SHA — registry pin and
   on-disk bytes always agree

The trade-off is a 16× wheel-size increase. If this becomes a
distribution-cost complaint, the canonical industry pattern is to
extract the vendored model to a separate companion package
(`kaos-nlp-transformers-static-models`) and keep this directory empty
in the main wheel — that's the spaCy / `en_core_web_sm` shape. Ship
date for that split is TBD; for now the bundled wheel is the simpler
ship.

### Loader integration

`kaos_nlp_transformers.embedding._load_model2vec_cached` checks for a
vendored copy at `kaos_nlp_transformers/_vendor/<slugified-model-id>/`
**before** calling `huggingface_hub.snapshot_download`. If the directory
exists with a non-empty `model.safetensors`, the loader uses it
directly; otherwise it falls through to the network path. See the
loader's docstring + the `LOAD_SOURCE` tag emitted at INFO level in the
log.

### Adding more models

To vendor a new model:

1. Confirm it has a permissive license (MIT / Apache-2.0 / BSD).
2. Confirm the wheel-size impact is justified (anything pushing past
   PyPI's 100 MB default cap requires a limit-raise request first).
3. Add the REGISTRY entry in `models.py` with the pinned revision SHA.
4. Run `huggingface_hub.snapshot_download(repo_id, revision=sha,
   allow_patterns=[...])` into `kaos_nlp_transformers/_vendor/<slug>/`
   — only the files the loader actually reads, never `.gitattributes`,
   never the upstream README beyond what we attribute below.
5. Append the model to the "Currently vendored" list above + add an
   ATTRIBUTION.txt entry alongside the model files.
6. Verify the wheel build via `uv build` and inspect the wheel's
   contents (`unzip -l dist/*.whl | grep _vendor`).
