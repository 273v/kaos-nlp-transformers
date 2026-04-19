"""Integration tests for GPU embedding — requires CUDA GPUs.

Skipped when torch is not installed or CUDA is not available.
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def _skip_if_no_gpu():
    try:
        torch = importlib.import_module("torch")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("torch not installed")


def _skip_if_offline():
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        pytest.skip("offline mode set")


# -- GPU load + embed ------------------------------------------------------


def test_gpu_auto_selects_cuda():
    _skip_if_no_gpu()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load(device="auto")
    assert model.backend_name == "sentence-transformers"
    assert model.device is not None
    assert model.device.device.startswith("cuda")


def test_gpu_embed_shape_and_dtype():
    _skip_if_no_gpu()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load(device="cuda:0")
    texts = ["hello world", "legal contract"]
    vecs = model.embed(texts)
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32


def test_gpu_embed_empty_input():
    _skip_if_no_gpu()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load(device="cuda:0")
    vecs = model.embed([])
    assert vecs.shape == (0, 384)


def test_gpu_embeddings_match_cpu():
    """Verify GPU and CPU produce near-identical embeddings."""
    _skip_if_no_gpu()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    cpu_model = EmbeddingModel.load(device="cpu")
    gpu_model = EmbeddingModel.load(device="cuda:0")

    texts = [
        "The Securities Exchange Act of 1934 regulates securities trading.",
        "A recipe for chocolate cake requires cocoa powder.",
        "The Internal Revenue Code imposes a tax on income.",
    ]
    cpu_vecs = cpu_model.embed(texts)
    gpu_vecs = gpu_model.embed(texts)

    # Embeddings should be near-identical (same model, different runtime)
    for i in range(len(texts)):
        cos_sim = float(
            np.dot(cpu_vecs[i], gpu_vecs[i])
            / (np.linalg.norm(cpu_vecs[i]) * np.linalg.norm(gpu_vecs[i]))
        )
        assert cos_sim > 0.999, f"text[{i}] cosine sim {cos_sim:.6f} < 0.999"


def test_gpu_semantic_ordering():
    """GPU embeddings preserve semantic similarity ordering."""
    _skip_if_no_gpu()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load(device="cuda:0")
    texts = [
        "The court held that the defendant breached the contract.",
        "The judge ruled that the agreement was violated.",
        "The recipe calls for two tablespoons of olive oil.",
    ]
    vecs = model.embed(texts)

    # Legal sentences should be more similar to each other than to recipe
    sim_legal = float(np.dot(vecs[0], vecs[1]))
    sim_cross = float(np.dot(vecs[0], vecs[2]))
    assert sim_legal > sim_cross + 0.05


def test_gpu_batch_sizes():
    """Various batch sizes produce identical results."""
    _skip_if_no_gpu()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load(device="cuda:0")
    texts = [f"sentence number {i} about legal matters" for i in range(50)]

    vecs_bs8 = model.embed(texts, batch_size=8)
    vecs_bs32 = model.embed(texts, batch_size=32)
    vecs_bs128 = model.embed(texts, batch_size=128)

    np.testing.assert_allclose(vecs_bs8, vecs_bs32, atol=1e-5)
    np.testing.assert_allclose(vecs_bs8, vecs_bs128, atol=1e-5)


def test_explicit_device_selection():
    """Explicit cuda:0 and cuda:1 load on different devices."""
    _skip_if_no_gpu()
    _skip_if_offline()

    torch = importlib.import_module("torch")

    if torch.cuda.device_count() < 2:
        pytest.skip("need 2 GPUs for this test")

    from kaos_nlp_transformers import EmbeddingModel

    model0 = EmbeddingModel.load(device="cuda:0")
    model1 = EmbeddingModel.load(device="cuda:1")

    assert model0.device is not None
    assert model1.device is not None
    assert model0.device.device == "cuda:0"
    assert model1.device.device == "cuda:1"
    assert model0.device.name != model1.device.name or model0.device.device != model1.device.device

    # Both produce valid embeddings
    vecs0 = model0.embed(["test"])
    vecs1 = model1.embed(["test"])
    assert vecs0.shape == vecs1.shape == (1, 384)


def test_force_cpu_ignores_gpu():
    """device='cpu' forces fastembed even when GPU is available."""
    _skip_if_no_gpu()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load(device="cpu")
    assert model.backend_name == "fastembed"
    assert model.device is not None
    assert model.device.device == "cpu"


def test_force_fastembed_backend():
    """backend='fastembed' forces fastembed even on GPU device."""
    _skip_if_no_gpu()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load(device="cpu", backend="fastembed")
    assert model.backend_name == "fastembed"
