"""Unit tests for device detection and resolution."""

from __future__ import annotations

import logging

import pytest

from kaos_nlp_transformers.device import (
    DeviceInfo,
    LatentDevice,
    SystemDevices,
    _detect_latent_devices,
    _kind_of,
    _reset_cache_for_tests,
    _run_nvidia_smi,
    detect_devices,
    resolve_device,
)
from kaos_nlp_transformers.errors import DeviceNotReachableError

pytestmark = pytest.mark.unit


# -- DeviceInfo -----------------------------------------------------------


def test_device_info_fields():
    d = DeviceInfo(
        name="Test GPU", device="cuda:0", backend="sentence-transformers", memory_mb=16000
    )
    assert d.name == "Test GPU"
    assert d.device == "cuda:0"
    assert d.backend == "sentence-transformers"
    assert d.memory_mb == 16000


def test_device_info_defaults():
    d = DeviceInfo(name="CPU", device="cpu", backend="fastembed")
    assert d.memory_mb == 0


# -- SystemDevices ---------------------------------------------------------


def test_system_devices_best_returns_first():
    gpu = DeviceInfo(name="GPU", device="cuda:0", backend="sentence-transformers", memory_mb=8000)
    cpu = DeviceInfo(name="CPU", device="cpu", backend="fastembed")
    sys = SystemDevices(devices=(gpu, cpu))
    assert sys.best == gpu


def test_system_devices_best_fallback_cpu():
    sys = SystemDevices(devices=())
    assert sys.best.device == "cpu"
    assert sys.best.backend == "fastembed"


def test_system_devices_has_gpu():
    gpu = DeviceInfo(name="GPU", device="cuda:0", backend="sentence-transformers", memory_mb=8000)
    cpu = DeviceInfo(name="CPU", device="cpu", backend="fastembed")
    assert SystemDevices(devices=(gpu, cpu)).has_gpu is True
    assert SystemDevices(devices=(cpu,)).has_gpu is False
    assert SystemDevices(devices=()).has_gpu is False


def test_system_devices_gpu_devices():
    gpu0 = DeviceInfo(
        name="GPU0", device="cuda:0", backend="sentence-transformers", memory_mb=16000
    )
    gpu1 = DeviceInfo(name="GPU1", device="cuda:1", backend="sentence-transformers", memory_mb=8000)
    cpu = DeviceInfo(name="CPU", device="cpu", backend="fastembed")
    sys = SystemDevices(devices=(gpu0, gpu1, cpu))
    assert len(sys.gpu_devices) == 2
    assert sys.gpu_devices[0] == gpu0


def test_system_devices_cpu_device():
    gpu = DeviceInfo(name="GPU", device="cuda:0", backend="sentence-transformers")
    cpu = DeviceInfo(name="CPU", device="cpu", backend="fastembed")
    sys = SystemDevices(devices=(gpu, cpu))
    assert sys.cpu_device == cpu


def test_system_devices_cpu_device_fallback():
    sys = SystemDevices(devices=())
    assert sys.cpu_device.device == "cpu"


def test_system_devices_has_latent_gpu():
    cpu = DeviceInfo(name="CPU", device="cpu", backend="fastembed")
    latent = LatentDevice(name="GPU", kind="cuda", reason="r", install_extra="torch")
    assert SystemDevices(devices=(cpu,), latent_devices=(latent,)).has_latent_gpu is True
    assert SystemDevices(devices=(cpu,)).has_latent_gpu is False


# -- LatentDevice ----------------------------------------------------------


def test_latent_device_fields():
    ld = LatentDevice(
        name="NVIDIA RTX 5070 Ti",
        kind="cuda",
        reason="torch not installed",
        install_extra="torch",
        detail={"index": 0, "memory_mb": 16303},
    )
    assert ld.kind == "cuda"
    assert ld.install_extra == "torch"
    assert ld.detail["memory_mb"] == 16303


# -- _kind_of --------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cuda", "cuda"),
        ("cuda:0", "cuda"),
        ("cuda:7", "cuda"),
        ("mps", "mps"),
        ("rocm", "rocm"),
        ("cpu", None),
        ("openvino", None),
        ("xla", None),
        ("nonsense", None),
    ],
)
def test_kind_of(device, expected):
    assert _kind_of(device) == expected


# -- _run_nvidia_smi (parsing) ---------------------------------------------


class _FakeCompleted:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def test_run_nvidia_smi_parses_two_gpus(monkeypatch):
    """Parsing the canonical csv,noheader,nounits output."""
    sample = (
        "0, GPU-aaaa-1111, NVIDIA GeForce RTX 5070 Ti, 16303, 595.58.03\n"
        "1, GPU-bbbb-2222, NVIDIA GeForce RTX 4070 Ti SUPER, 16376, 595.58.03\n"
    )
    monkeypatch.setattr(
        "kaos_nlp_transformers.device.shutil.which", lambda _name: "/usr/bin/nvidia-smi"
    )
    monkeypatch.setattr(
        "kaos_nlp_transformers.device.subprocess.run",
        lambda *a, **kw: _FakeCompleted(stdout=sample, returncode=0),
    )
    rows = _run_nvidia_smi()
    assert len(rows) == 2
    assert rows[0]["name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert rows[0]["memory_mb"] == "16303"
    assert rows[0]["uuid"] == "GPU-aaaa-1111"
    assert rows[1]["index"] == "1"


def test_run_nvidia_smi_returns_empty_when_smi_missing(monkeypatch):
    monkeypatch.setattr("kaos_nlp_transformers.device.shutil.which", lambda _name: None)
    assert _run_nvidia_smi() == []


def test_run_nvidia_smi_returns_empty_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        "kaos_nlp_transformers.device.shutil.which", lambda _name: "/usr/bin/nvidia-smi"
    )
    monkeypatch.setattr(
        "kaos_nlp_transformers.device.subprocess.run",
        lambda *a, **kw: _FakeCompleted(stdout="", returncode=9),
    )
    assert _run_nvidia_smi() == []


def test_run_nvidia_smi_returns_empty_on_timeout(monkeypatch):
    import subprocess

    def _raise(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=2.0)

    monkeypatch.setattr(
        "kaos_nlp_transformers.device.shutil.which", lambda _name: "/usr/bin/nvidia-smi"
    )
    monkeypatch.setattr("kaos_nlp_transformers.device.subprocess.run", _raise)
    assert _run_nvidia_smi() == []


# -- _detect_latent_devices (reconciliation) -------------------------------


def _stub_probes(monkeypatch, *, nvidia=(), rocm=(), apple=()):
    monkeypatch.setattr("kaos_nlp_transformers.device._probe_nvidia", lambda: list(nvidia))
    monkeypatch.setattr("kaos_nlp_transformers.device._probe_rocm", lambda: list(rocm))
    monkeypatch.setattr("kaos_nlp_transformers.device._probe_apple", lambda: list(apple))


def test_reconcile_no_torch_two_nvidia_gpus(monkeypatch):
    """OS sees 2 NVIDIAs, torch sees 0 — both surface as latent."""
    cand = [
        LatentDevice(name="GPU0", kind="cuda", reason="x", install_extra="torch"),
        LatentDevice(name="GPU1", kind="cuda", reason="x", install_extra="torch"),
    ]
    _stub_probes(monkeypatch, nvidia=cand)
    out = _detect_latent_devices(reachable=[], onnx_providers=["CPUExecutionProvider"])
    assert [d.name for d in out] == ["GPU0", "GPU1"]


def test_reconcile_torch_sees_both_no_latents(monkeypatch):
    """OS sees 2 NVIDIAs and torch already reaches both — latent list is empty."""
    cand = [
        LatentDevice(name="GPU0", kind="cuda", reason="x", install_extra="torch"),
        LatentDevice(name="GPU1", kind="cuda", reason="x", install_extra="torch"),
    ]
    _stub_probes(monkeypatch, nvidia=cand)
    reachable = [
        DeviceInfo(name="GPU0", device="cuda:0", backend="sentence-transformers", memory_mb=16000),
        DeviceInfo(name="GPU1", device="cuda:1", backend="sentence-transformers", memory_mb=16000),
    ]
    out = _detect_latent_devices(reachable=reachable, onnx_providers=[])
    assert out == []


def test_reconcile_onnx_cuda_provider_suppresses_latents(monkeypatch):
    """onnxruntime-gpu sees the GPUs even without torch — not latent."""
    cand = [LatentDevice(name="GPU0", kind="cuda", reason="x", install_extra="torch")]
    _stub_probes(monkeypatch, nvidia=cand)
    out = _detect_latent_devices(
        reachable=[],
        onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert out == []


def test_reconcile_apple_no_torch_yields_latent(monkeypatch):
    cand = [LatentDevice(name="Apple MPS", kind="mps", reason="x", install_extra="torch")]
    _stub_probes(monkeypatch, apple=cand)
    out = _detect_latent_devices(reachable=[], onnx_providers=[])
    assert len(out) == 1
    assert out[0].kind == "mps"


def test_reconcile_no_probes_no_latents(monkeypatch):
    _stub_probes(monkeypatch)  # all empty
    assert _detect_latent_devices(reachable=[], onnx_providers=[]) == []


# -- detect_devices (top-level + logging) ----------------------------------


def test_detect_devices_returns_system_devices():
    _reset_cache_for_tests()
    sys = detect_devices()
    assert isinstance(sys, SystemDevices)
    # Must always have at least CPU
    assert len(sys.devices) >= 1
    assert sys.cpu_device.device == "cpu"


@pytest.fixture
def kaos_caplog(caplog):
    """Attach caplog's handler directly to the kaos device logger.

    kaos_core.logging configures the ``kaos`` root logger with
    ``propagate = False``, so the default caplog plumbing (which captures
    via the root logger) sees nothing. Hook the handler in place for the
    duration of the test.
    """
    target = logging.getLogger("kaos.nlp_transformers.device")
    target.addHandler(caplog.handler)
    prior_level = target.level
    target.setLevel(logging.DEBUG)
    try:
        yield caplog
    finally:
        target.removeHandler(caplog.handler)
        target.setLevel(prior_level)


def test_detect_devices_warns_when_only_latent(monkeypatch, kaos_caplog):
    """Promotion of silent CPU fallback to WARNING — the regression this fix prevents."""
    cand = [LatentDevice(name="NV GPU", kind="cuda", reason="r", install_extra="torch")]
    monkeypatch.setattr("kaos_nlp_transformers.device._detect_torch_devices", lambda: [])
    monkeypatch.setattr("kaos_nlp_transformers.device._detect_onnx_providers", lambda: [])
    _stub_probes(monkeypatch, nvidia=cand)
    _reset_cache_for_tests()

    sys = detect_devices()

    assert sys.has_gpu is False
    assert sys.has_latent_gpu is True
    msgs = [r.getMessage() for r in kaos_caplog.records if r.levelno == logging.WARNING]
    assert any("latent accelerator" in m and "kaos-nlp-transformers[torch]" in m for m in msgs)


def test_detect_devices_no_warning_when_gpu_reachable(monkeypatch, kaos_caplog):
    """Regression guard: existing GPU-reachable path stays at INFO, not WARNING."""
    monkeypatch.setattr(
        "kaos_nlp_transformers.device._detect_torch_devices",
        lambda: [
            DeviceInfo(
                name="GPU", device="cuda:0", backend="sentence-transformers", memory_mb=16000
            )
        ],
    )
    monkeypatch.setattr("kaos_nlp_transformers.device._detect_onnx_providers", lambda: [])
    _stub_probes(monkeypatch)
    _reset_cache_for_tests()

    detect_devices()

    warnings_only = [r for r in kaos_caplog.records if r.levelno >= logging.WARNING]
    assert warnings_only == []


def test_detect_devices_no_warning_when_clean_cpu_box(monkeypatch, kaos_caplog):
    monkeypatch.setattr("kaos_nlp_transformers.device._detect_torch_devices", lambda: [])
    monkeypatch.setattr("kaos_nlp_transformers.device._detect_onnx_providers", lambda: [])
    _stub_probes(monkeypatch)  # nothing latent either
    _reset_cache_for_tests()

    sys = detect_devices()

    assert sys.has_gpu is False
    assert sys.has_latent_gpu is False
    warnings_only = [r for r in kaos_caplog.records if r.levelno >= logging.WARNING]
    assert warnings_only == []


# -- resolve_device --------------------------------------------------------


def _make_system() -> SystemDevices:
    return SystemDevices(
        devices=(
            DeviceInfo(
                name="Big GPU", device="cuda:0", backend="sentence-transformers", memory_mb=16000
            ),
            DeviceInfo(
                name="Small GPU", device="cuda:1", backend="sentence-transformers", memory_mb=8000
            ),
            DeviceInfo(name="CPU", device="cpu", backend="fastembed"),
        ),
        onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
    )


def test_resolve_auto():
    sys = _make_system()
    d = resolve_device("auto", sys)
    assert d.device == "cuda:0"


def test_resolve_cpu():
    sys = _make_system()
    d = resolve_device("cpu", sys)
    assert d.device == "cpu"


def test_resolve_cuda_bare():
    sys = _make_system()
    d = resolve_device("cuda", sys)
    assert d.device == "cuda:0"


def test_resolve_cuda_indexed():
    sys = _make_system()
    d = resolve_device("cuda:1", sys)
    assert d.device == "cuda:1"
    assert d.name == "Small GPU"


def test_resolve_unavailable_raises():
    """No latent of this kind → plain ValueError."""
    sys = SystemDevices(devices=(DeviceInfo(name="CPU", device="cpu", backend="fastembed"),))
    with pytest.raises(ValueError, match="not available"):
        resolve_device("cuda:0", sys)


def test_resolve_cuda_unreachable_raises_typed_error():
    """Latent GPU present + cuda requested → DeviceNotReachableError with install_extra."""
    sys = SystemDevices(
        devices=(DeviceInfo(name="CPU", device="cpu", backend="fastembed"),),
        latent_devices=(
            LatentDevice(
                name="NVIDIA RTX 5070 Ti",
                kind="cuda",
                reason="torch not installed",
                install_extra="torch",
                detail={"index": 0},
            ),
        ),
    )
    with pytest.raises(DeviceNotReachableError) as exc_info:
        resolve_device("cuda:0", sys)
    err = exc_info.value
    assert err.requested == "cuda:0"
    assert err.kind == "cuda"
    assert err.install_extra == "torch"
    assert "kaos-nlp-transformers[torch]" in str(err)
    assert err.details["install_extra"] == "torch"
    assert err.details["name"] == "NVIDIA RTX 5070 Ti"


def test_resolve_mps_unreachable_raises_typed_error():
    sys = SystemDevices(
        devices=(DeviceInfo(name="CPU", device="cpu", backend="fastembed"),),
        latent_devices=(
            LatentDevice(
                name="Apple MPS",
                kind="mps",
                reason="torch not installed",
                install_extra="torch",
            ),
        ),
    )
    with pytest.raises(DeviceNotReachableError) as exc_info:
        resolve_device("mps", sys)
    assert exc_info.value.kind == "mps"


def test_resolve_openvino_available():
    sys = SystemDevices(
        devices=(DeviceInfo(name="CPU", device="cpu", backend="fastembed"),),
        onnx_providers=("OpenVINOExecutionProvider", "CPUExecutionProvider"),
    )
    d = resolve_device("openvino", sys)
    assert d.device == "openvino"
    assert d.backend == "fastembed"


def test_resolve_openvino_unavailable():
    sys = SystemDevices(
        devices=(DeviceInfo(name="CPU", device="cpu", backend="fastembed"),),
        onnx_providers=("CPUExecutionProvider",),
    )
    with pytest.raises(ValueError, match="OpenVINO"):
        resolve_device("openvino", sys)


# -- detect_devices (live) -------------------------------------------------


def test_detect_devices_onnx_providers():
    sys = detect_devices()
    assert isinstance(sys.onnx_providers, tuple)
    # CPUExecutionProvider should always be there if onnxruntime is installed
    try:
        import onnxruntime as _ort  # noqa: F401

        assert "CPUExecutionProvider" in sys.onnx_providers
    except ImportError:
        pass
