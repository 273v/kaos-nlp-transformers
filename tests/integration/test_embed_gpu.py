"""Integration tests for GPU embedding — requires onnxruntime-gpu + CUDA.

Audit-06 KNT-501: post-torch-removal, the GPU on-ramp is the ``[gpu]``
extra (onnxruntime-gpu) and fastembed via ``CUDAExecutionProvider``. The
old torch-based skip + sentence-transformers backend assertions were
retired alongside the SE backend.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def _has_cuda_provider() -> bool:
    """True iff onnxruntime + a working CUDAExecutionProvider are installed.

    Checking ``get_available_providers`` is more accurate than checking
    ``import onnxruntime_gpu`` because the gpu wheel installs as
    ``onnxruntime`` with the CUDA provider added; there is no
    ``onnxruntime_gpu`` import name.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


def _has_nvidia_gpu() -> bool:
    """True iff nvidia-smi reports at least one GPU."""
    from kaos_nlp_transformers.device import _run_nvidia_smi

    return bool(_run_nvidia_smi())


def _skip_if_no_gpu():
    if not _has_nvidia_gpu():
        pytest.skip("no NVIDIA GPU on this host")
    if not _has_cuda_provider():
        pytest.skip(
            "onnxruntime-gpu / CUDAExecutionProvider not installed — "
            "install via `pip install kaos-nlp-transformers[gpu]`"
        )


def _skip_if_offline():
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        pytest.skip("offline mode set")


# -- GPU load + embed ------------------------------------------------------


def test_gpu_auto_selects_cuda():
    _skip_if_no_gpu()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load(device="auto")
    # Audit-06 KNT-501: the GPU embedding backend is fastembed (ONNX) via
    # onnxruntime-gpu, not sentence-transformers.
    assert model.backend_name == "fastembed"
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

    from kaos_nlp_transformers.device import _run_nvidia_smi

    if len(_run_nvidia_smi()) < 2:
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
    """backend='fastembed' forces fastembed even on GPU device.

    Post-audit-06 this is functionally a no-op (fastembed is the only
    GPU-capable backend now), but the explicit setter still resolves
    cleanly and is useful for pinning behavior in pipelines."""
    _skip_if_no_gpu()
    _skip_if_offline()
    from kaos_nlp_transformers import EmbeddingModel

    model = EmbeddingModel.load(device="cpu", backend="fastembed")
    assert model.backend_name == "fastembed"


# -- Latent-device contract on a real GPU box ------------------------------


@pytest.mark.gpu
def test_gpu_box_has_no_latent_devices():
    """On a host with onnxruntime-gpu, the OS-level probe must not
    double-count NVIDIA GPUs as latent. This is the contract that makes
    `kaos-nlp-transformers info` correct on actual GPU machines, not just
    a CPU-only fallback box where it happens to look right."""
    _skip_if_no_gpu()
    from kaos_nlp_transformers.device import _reset_cache_for_tests, detect_devices

    _reset_cache_for_tests()
    system = detect_devices()
    assert system.has_gpu is True
    # The reconciliation step in _detect_latent_devices subtracts reachable
    # GPUs of each kind from the OS-probe candidates, so a fully
    # CUDAExecutionProvider-aware box should report zero latents.
    assert system.has_latent_gpu is False, (
        f"expected zero latent devices on a CUDA-reachable host; "
        f"got {[(d.name, d.kind) for d in system.latent_devices]}"
    )


@pytest.mark.gpu
def test_info_tool_resolves_to_cuda_on_gpu_box():
    """End-to-end through the MCP info tool: device='auto' on a GPU box
    must pick CUDA, not CPU, with no latent_devices noise."""
    import asyncio

    _skip_if_no_gpu()
    from kaos_core import KaosRuntime

    from kaos_nlp_transformers.device import _reset_cache_for_tests
    from kaos_nlp_transformers.tools import register_transformers_tools

    _reset_cache_for_tests()
    runtime = KaosRuntime()
    register_transformers_tools(runtime)
    tool = runtime.tools.get_tool("kaos-nlp-transformers-info")
    assert tool is not None

    result = asyncio.run(tool.execute({}, None))
    assert result.isError is False
    payload = result.structuredContent
    assert payload is not None
    assert payload["resolved_device"]["device"].startswith("cuda")
    # Audit-06 KNT-501: GPU backend is fastembed (onnxruntime-gpu), not SE.
    assert payload["resolved_device"]["backend"] == "fastembed"
    assert payload["latent_devices"] == []
