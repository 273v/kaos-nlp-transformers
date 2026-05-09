"""Frozen-vector parity test (audit KNT-601).

The most important test in the migration. Asserts that the Rust ort
backend produces output bit-equivalent (cosine ≥ 0.9999) to the
frozen reference NPYs that ``scripts/freeze_reference_vectors.py``
captured against the 0.1.0a6 fastembed stack.

The test runs against the experimental Rust path
(``KAOS_NLP_TRANSFORMERS_RUST_EXPERIMENTAL=1``) BEFORE Phase 4 flips
the default. After Phase 4 the env var becomes a no-op (the gate is
removed and ort is the default) but the parity assertion stays
valid because the underlying ONNX session is identical to what the
NPYs were frozen against.

The test is offline-friendly only when the HF cache is populated;
flagged with the ``live`` marker otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
SENTENCES_PATH = REFERENCE_DIR / "sentences.txt"


def _slug(model_id: str) -> str:
    return model_id.lower().replace("/", "_").replace("-", "_").replace(".", "_")


def _sentences() -> list[str]:
    return [line.strip() for line in SENTENCES_PATH.read_text().splitlines() if line.strip()]


def _load_ref(model_id: str) -> np.ndarray:
    npy = REFERENCE_DIR / f"{_slug(model_id)}.npy"
    if not npy.exists():
        pytest.skip(f"frozen reference NPY missing: {npy}")
    return np.load(npy)


@pytest.fixture(autouse=True)
def _enable_rust_experimental(monkeypatch):
    """Phase 4 (KNT-601): the experimental flag is now a no-op
    since ``ort`` is the default backend. The fixture stays for
    cache-clearing isolation between tests; the env var is harmless.
    """
    monkeypatch.setenv("KAOS_NLP_TRANSFORMERS_RUST_EXPERIMENTAL", "1")
    from kaos_nlp_transformers import embedding as _embedding

    _embedding._load_rust_embedding_cached.cache_clear()
    _embedding._embed_cache_clear()
    yield
    _embedding._load_rust_embedding_cached.cache_clear()
    _embedding._embed_cache_clear()


@pytest.mark.live
def test_bge_small_rust_matches_frozen():
    """The Rust ort path must produce bit-equivalent embeddings to
    the fastembed-frozen reference for BAAI/bge-small-en-v1.5.

    Audit KNT-601: this is the bit-equivalence regression test. If
    cosine ever drops below 0.9999 the migration introduced a numerical
    bug; investigate before flipping the default backend.
    """
    from kaos_nlp_transformers import EmbeddingModel

    model_id = "BAAI/bge-small-en-v1.5"
    ref = _load_ref(model_id)

    em = EmbeddingModel.load(model_id)
    assert em.backend_name == "ort", (
        f"experimental flag did not route to Rust backend: backend_name={em.backend_name!r}"
    )

    sentences = _sentences()
    out = em.embed(sentences)
    assert out.shape == ref.shape, f"shape mismatch: out={out.shape} ref={ref.shape}"
    assert out.dtype == np.float32

    # Both arrays are L2-normalized in their producers; cosine = dot.
    sims = np.einsum("ij,ij->i", out, ref)
    min_sim = float(sims.min())
    assert min_sim >= 0.9999, (
        f"per-row cosine ≥ 0.9999 is the migration contract; got min={min_sim:.6f}, "
        f"all sims={sims.tolist()}"
    )


def test_reference_npys_present():
    """The frozen reference NPYs must be committed to the tree.
    This test runs offline — fails if scripts/freeze_reference_vectors.py
    was not run before commit, or the NPYs were accidentally removed."""
    expected = [
        "baai_bge_small_en_v1_5.npy",
        "minishlab_potion_retrieval_32m.npy",
        "minishlab_potion_base_8m.npy",
        "minishlab_potion_base_32m.npy",
        "baai_bge_reranker_base.npy",
    ]
    for name in expected:
        path = REFERENCE_DIR / name
        assert path.exists(), (
            f"frozen reference {path} missing — re-run scripts/freeze_reference_vectors.py"
        )
        assert path.stat().st_size > 0


def test_reference_sentences_non_empty():
    s = _sentences()
    assert len(s) >= 8, "expected ≥ 8 sentences for the parity test"
    # Distinct sentences only (otherwise the parity test is degenerate).
    assert len(set(s)) == len(s), "tests/reference/sentences.txt has duplicates"


# Ensure the env-var-fixture above only fires for tests in this module
# — avoid leaking the flag into the rest of the suite where the
# default fastembed path must keep working.
_ = os
