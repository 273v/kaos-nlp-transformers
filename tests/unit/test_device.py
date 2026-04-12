"""Unit tests for device detection and resolution."""

from __future__ import annotations

import pytest

from kaos_nlp_transformers.device import (
    DeviceInfo,
    SystemDevices,
    detect_devices,
    resolve_device,
)

pytestmark = pytest.mark.unit


# -- DeviceInfo -----------------------------------------------------------


def test_device_info_fields():
    d = DeviceInfo(name="Test GPU", device="cuda:0", backend="sentence-transformers", memory_mb=16000)
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
    gpu0 = DeviceInfo(name="GPU0", device="cuda:0", backend="sentence-transformers", memory_mb=16000)
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


# -- resolve_device --------------------------------------------------------


def _make_system() -> SystemDevices:
    return SystemDevices(
        devices=(
            DeviceInfo(name="Big GPU", device="cuda:0", backend="sentence-transformers", memory_mb=16000),
            DeviceInfo(name="Small GPU", device="cuda:1", backend="sentence-transformers", memory_mb=8000),
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
    sys = SystemDevices(
        devices=(DeviceInfo(name="CPU", device="cpu", backend="fastembed"),)
    )
    with pytest.raises(ValueError, match="not available"):
        resolve_device("cuda:0", sys)


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


def test_detect_devices_returns_system_devices():
    sys = detect_devices()
    assert isinstance(sys, SystemDevices)
    # Must always have at least CPU
    assert len(sys.devices) >= 1
    assert sys.cpu_device.device == "cpu"


def test_detect_devices_onnx_providers():
    sys = detect_devices()
    assert isinstance(sys.onnx_providers, tuple)
    # CPUExecutionProvider should always be there if onnxruntime is installed
    try:
        import onnxruntime as _ort  # noqa: F401

        assert "CPUExecutionProvider" in sys.onnx_providers
    except ImportError:
        pass
