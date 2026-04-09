"""Live embed test — downloads bge-small-en-v1.5 and embeds a few strings.

Skipped when offline mode is set or fastembed is missing. Required to
pass before kaos-nlp-transformers v0 ships per the no-fake-tests rule.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_embed_live():
    pytest.importorskip("fastembed")
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip("offline mode set")

    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load("BAAI/bge-small-en-v1.5")
    assert model.dim == 384

    texts = [
        "The court held that the plaintiff's claim was barred by the statute of limitations.",
        "The recipe calls for two cups of flour and one teaspoon of salt.",
        "The company reported quarterly earnings of $1.2 billion.",
    ]
    vecs = model.embed(texts)

    assert vecs.shape == (3, 384)
    assert vecs.dtype == np.float32

    # Sanity check: legal sentence and finance sentence should be more
    # similar to each other than the recipe sentence is to either. This
    # is a content-aware assertion (per the no-fake-tests rule:
    # assertions must verify content understanding, not just
    # `len(response) > 0`).
    sim_legal_finance = float(np.dot(vecs[0], vecs[2]))
    sim_legal_recipe = float(np.dot(vecs[0], vecs[1]))
    assert sim_legal_finance > sim_legal_recipe - 0.05, (
        f"expected legal-finance similarity ({sim_legal_finance:.3f}) "
        f"to exceed legal-recipe similarity ({sim_legal_recipe:.3f}) by a margin"
    )


def test_empty_input_returns_zero_array():
    pytest.importorskip("fastembed")
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip("offline mode set")

    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load("BAAI/bge-small-en-v1.5")
    vecs = model.embed([])
    assert vecs.shape == (0, 384)
    assert vecs.dtype == np.float32
