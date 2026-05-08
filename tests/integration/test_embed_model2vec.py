"""Live integration tests for the model2vec backend (audit-04 KNT-301).

These exercise the real network path: download the pinned snapshot from
huggingface.co and run a real encode through ``StaticModel``. They
require the ``[model2vec]`` extra and unrestricted network egress on
first run; subsequent runs are served from the HF cache.

Run them explicitly with::

    uv run pytest tests/integration/test_embed_model2vec.py -m live

Tier rationale (per docs/python/checklists/04-test.md):
    - ``unit``: see test_models.py / test_embedding_backends.py — pure
      logic, no network.
    - ``integration``: this file (real download + encode contract).
    - ``live``: subset of integration that depends on network access.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.live]


def _skip_if_no_model2vec() -> None:
    pytest.importorskip("model2vec")


def _skip_if_offline() -> None:
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip("offline mode set")


# -- potion-retrieval-32M --------------------------------------------------


def test_potion_retrieval_load_and_embed():
    """Real download + encode contract for the retrieval-tuned potion."""
    _skip_if_no_model2vec()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load("minishlab/potion-retrieval-32M")
    assert model.backend_name == "model2vec"
    assert model.dim == 512
    assert model.device is not None
    # Static models are CPU-only by construction (audit-04 KNT-302). The
    # registry might be loaded on a GPU box; we explicitly pin device=cpu
    # at load time, so this is a contract regression guard.
    assert model.device.device == "cpu"

    texts = [
        "The court held that the plaintiff's claim was barred by the statute of limitations.",
        "The recipe calls for two cups of flour and one teaspoon of salt.",
        "The company reported quarterly earnings of $1.2 billion.",
    ]
    vecs = model.embed(texts)
    assert vecs.shape == (3, 512)
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_potion_retrieval_semantic_ordering():
    """Quality smoke: legal-pair similarity must exceed legal-vs-recipe.

    This is the same shape of assertion the existing GPU integration suite
    uses (``test_gpu_semantic_ordering``); it catches a backend that loads
    but produces nonsense. Not a benchmark — quality numbers come from
    docs/benchmarks/, not this assertion.
    """
    _skip_if_no_model2vec()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load("minishlab/potion-retrieval-32M")
    texts = [
        "The court held that the defendant breached the contract.",
        "The judge ruled that the agreement was violated.",
        "The recipe calls for two tablespoons of olive oil.",
    ]
    vecs = model.embed(texts)

    sim_legal_pair = float(np.dot(vecs[0], vecs[1]))
    sim_legal_recipe = float(np.dot(vecs[0], vecs[2]))
    assert sim_legal_pair > sim_legal_recipe + 0.05, (
        f"static retrieval model failed semantic ordering: "
        f"legal-pair={sim_legal_pair:.3f} vs legal-recipe={sim_legal_recipe:.3f}"
    )


# -- potion-base-32M -------------------------------------------------------


def test_potion_base_load_and_embed():
    _skip_if_no_model2vec()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load("minishlab/potion-base-32M")
    assert model.backend_name == "model2vec"
    assert model.dim == 512

    vecs = model.embed(["hello world", "another short string"])
    assert vecs.shape == (2, 512)
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


# -- empty-input contract --------------------------------------------------


def test_empty_input_returns_zero_rows():
    """``embed([])`` must return ``(0, dim)`` regardless of backend."""
    _skip_if_no_model2vec()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load("minishlab/potion-retrieval-32M")
    vecs = model.embed([])
    assert vecs.shape == (0, 512)
    assert vecs.dtype == np.float32


# -- backend cache parity --------------------------------------------------


def test_repeated_loads_hit_lru_cache():
    """Second ``EmbeddingModel.load`` for the same id is O(1) — same
    underlying backend object. Audit-04 keeps the lru_cache invariant the
    other backends already enforce."""
    _skip_if_no_model2vec()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    a = EmbeddingModel.load("minishlab/potion-retrieval-32M")
    b = EmbeddingModel.load("minishlab/potion-retrieval-32M")
    assert a._backend is b._backend
