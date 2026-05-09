"""Freeze reference embeddings + reranker scores for the migration
contract test (audit KNT-601).

Run this against the **current 0.1.0a6 fastembed stack** (the
incumbent, not yet migrated). Writes ``tests/reference/<slug>.npy``
files that ``test_reference_vectors.py`` then compares against — the
contract is "Rust backend output cosine ≥ 0.9999 vs the frozen NPYs."

Run once before flipping the default backend in P4.1:

    uv run python scripts/freeze_reference_vectors.py

The script is intentionally conservative — it imports fastembed
directly rather than going through ``EmbeddingModel.load`` so that a
half-migrated tree (where ``_resolve_backend`` is mid-flip) still
produces baseline numbers. Each output is a deterministic
``(N, dim)`` float32 numpy array.

Maintainers' note: re-run this when a registry entry's revision SHA
changes (i.e., we pin a new model version) and commit the updated
NPYs in the same PR. The cosine-equivalence test will then re-pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "tests" / "reference"
SENTENCES_PATH = REFERENCE_DIR / "sentences.txt"


def _slug(model_id: str) -> str:
    """Normalize a model id into a safe filename stem."""
    return model_id.lower().replace("/", "_").replace("-", "_").replace(".", "_")


def _load_sentences() -> list[str]:
    text = SENTENCES_PATH.read_text(encoding="utf-8")
    sentences = [line.strip() for line in text.splitlines() if line.strip()]
    if not sentences:
        msg = f"no sentences in {SENTENCES_PATH}"
        raise SystemExit(msg)
    return sentences


def _freeze_fastembed_embedding(model_id: str, sentences: list[str]) -> np.ndarray:
    """Freeze a fastembed text-embedding model's outputs."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        msg = "fastembed must be installed to freeze reference vectors. uv sync."
        raise SystemExit(msg) from exc

    model = TextEmbedding(model_name=model_id)
    vecs = list(model.embed(sentences, batch_size=8))
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(sentences):
        msg = f"unexpected fastembed output shape {arr.shape}"
        raise SystemExit(msg)
    # L2 normalize (matches the contract centralized in pooling.rs).
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms == 0.0, 1.0, norms)
    return arr.astype(np.float32, copy=False)


def _freeze_model2vec(model_id: str, revision: str, sentences: list[str]) -> np.ndarray:
    """Freeze a model2vec (static lookup) embedding model's outputs.

    model2vec doesn't go through ort, but we want frozen NPYs for these
    too so the test_reference_vectors test covers both code paths.
    """
    try:
        from huggingface_hub import snapshot_download
        from model2vec import StaticModel
    except ImportError as exc:
        msg = "model2vec extra must be installed: uv sync --extra model2vec"
        raise SystemExit(msg) from exc

    local_path = snapshot_download(repo_id=model_id, revision=revision, repo_type="model")
    model = StaticModel.from_pretrained(local_path)
    arr = np.asarray(model.encode(sentences, show_progress_bar=False), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms == 0.0, 1.0, norms)
    return arr.astype(np.float32, copy=False)


def _freeze_fastembed_reranker(model_id: str, sentences: list[str]) -> np.ndarray:
    """Freeze the BGE reranker's sigmoid-normalized scores for a fixed
    set of (query, passage) pairs constructed from the sentence list.

    Pairing strategy: sentences[0] is the query, sentences[1..] are
    candidate passages. Produces (N-1,) scores in [0, 1].
    """
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as exc:
        msg = "fastembed must be installed."
        raise SystemExit(msg) from exc

    model = TextCrossEncoder(model_name=model_id)
    query = sentences[0]
    pairs = [(query, p) for p in sentences[1:]]
    raw = list(model.rerank_pairs(pairs))
    arr = np.asarray(raw, dtype=np.float64)
    # fastembed's TextCrossEncoder.rerank_pairs returns raw logits;
    # apply sigmoid to match the Rust path's [0, 1] contract.
    scores = 1.0 / (1.0 + np.exp(-arr))
    return scores.astype(np.float32, copy=False)


def main() -> None:
    sentences = _load_sentences()
    print(f"[+] {len(sentences)} sentences loaded from {SENTENCES_PATH}")

    # Mirror REGISTRY: every fastembed-backed entry needs a frozen
    # NPY. Hard-coded here (rather than imported from REGISTRY) so a
    # half-migrated REGISTRY can still be re-baselined.
    embedding_models: list[tuple[str, str]] = [
        ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
    ]
    model2vec_models: list[tuple[str, str]] = [
        ("minishlab/potion-retrieval-32M", "6fc8051fab2a1e0ee76689cf08c853792ac285e7"),
        ("minishlab/potion-base-8M", "bf8b056651a2c21b8d2565580b8569da283cab23"),
        ("minishlab/potion-base-32M", "1e5a03f8eeb2c98b928fbbd846f22f816360919f"),
    ]
    reranker_models: list[tuple[str, str]] = [
        ("BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70"),
    ]

    for model_id, _rev in embedding_models:
        print(f"[+] freezing fastembed embedding {model_id} ...")
        arr = _freeze_fastembed_embedding(model_id, sentences)
        out = REFERENCE_DIR / f"{_slug(model_id)}.npy"
        np.save(out, arr)
        print(f"    wrote {out} shape={arr.shape}")

    for model_id, rev in model2vec_models:
        print(f"[+] freezing model2vec {model_id} @ {rev[:8]} ...")
        try:
            arr = _freeze_model2vec(model_id, rev, sentences)
        except SystemExit as e:
            print(f"    SKIP: {e}")
            continue
        out = REFERENCE_DIR / f"{_slug(model_id)}.npy"
        np.save(out, arr)
        print(f"    wrote {out} shape={arr.shape}")

    for model_id, _rev in reranker_models:
        print(f"[+] freezing fastembed reranker {model_id} ...")
        arr = _freeze_fastembed_reranker(model_id, sentences)
        out = REFERENCE_DIR / f"{_slug(model_id)}.npy"
        np.save(out, arr)
        print(f"    wrote {out} shape={arr.shape}")

    print()
    print("Done. Commit the resulting tests/reference/*.npy alongside this script.")


if __name__ == "__main__":
    sys.exit(main())
