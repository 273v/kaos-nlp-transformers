"""Hardware detection and device routing for embedding inference.

Probes the runtime environment for available accelerators (CUDA, ROCm,
OpenVINO, MPS, XLA/TPU) and recommends a backend + device string.

The detection is lazy and cached — the first call to ``detect_device()``
probes the system; subsequent calls return the cached result.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Detected hardware accelerator."""

    name: str
    """Human-readable device name (e.g. 'NVIDIA GeForce RTX 5070 Ti')."""

    device: str
    """PyTorch-style device string: 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps', 'xla'."""

    backend: str
    """Recommended embedding backend: 'fastembed' or 'sentence-transformers'."""

    memory_mb: int = 0
    """Device memory in MB (0 for CPU or unknown)."""


@dataclass(frozen=True, slots=True)
class SystemDevices:
    """All detected accelerators on the system."""

    devices: tuple[DeviceInfo, ...] = ()
    """All detected devices, ordered by preference (best GPU first, CPU last)."""

    onnx_providers: tuple[str, ...] = ()
    """Available ONNX Runtime execution providers."""

    @property
    def best(self) -> DeviceInfo:
        """Return the highest-priority device (first GPU, or CPU)."""
        if self.devices:
            return self.devices[0]
        return DeviceInfo(name="CPU", device="cpu", backend="fastembed")

    @property
    def has_gpu(self) -> bool:
        return any(d.device != "cpu" for d in self.devices)

    @property
    def gpu_devices(self) -> tuple[DeviceInfo, ...]:
        return tuple(d for d in self.devices if d.device != "cpu")

    @property
    def cpu_device(self) -> DeviceInfo:
        for d in self.devices:
            if d.device == "cpu":
                return d
        return DeviceInfo(name="CPU", device="cpu", backend="fastembed")


def _detect_torch_devices() -> list[DeviceInfo]:
    """Probe PyTorch for CUDA, ROCm, MPS, and XLA devices."""
    devices: list[DeviceInfo] = []

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return devices

    # CUDA / ROCm (ROCm presents as CUDA in PyTorch)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            # torch >=2.11 renamed total_mem → total_memory
            mem_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            mem_mb = mem_bytes // (1024 * 1024)
            devices.append(
                DeviceInfo(
                    name=props.name,
                    device=f"cuda:{i}",
                    backend="sentence-transformers",
                    memory_mb=mem_mb,
                )
            )

    # Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(
            DeviceInfo(
                name="Apple MPS",
                device="mps",
                backend="sentence-transformers",
            )
        )

    # XLA / TPU
    try:
        xm = importlib.import_module("torch_xla.core.xla_model")

        xla_device = xm.xla_device()  # type: ignore[no-untyped-call]
        devices.append(
            DeviceInfo(
                name=f"XLA ({getattr(xla_device, 'type', 'unknown')})",
                device="xla",
                backend="sentence-transformers",
            )
        )
    except (ImportError, RuntimeError):
        pass

    return devices


def _detect_onnx_providers() -> list[str]:
    """Return available ONNX Runtime execution providers."""
    try:
        ort: Any = importlib.import_module("onnxruntime")

        return list(ort.get_available_providers())
    except ImportError:
        return []


def detect_devices() -> SystemDevices:
    """Detect all available hardware accelerators.

    Returns a ``SystemDevices`` with GPU devices ordered by memory
    (largest first), followed by CPU.
    """
    gpu_devices = _detect_torch_devices()
    onnx_providers = _detect_onnx_providers()

    # Sort GPU devices by memory descending (prefer bigger GPUs)
    gpu_devices.sort(key=lambda d: d.memory_mb, reverse=True)

    # Always include CPU as fallback
    cpu = DeviceInfo(name="CPU", device="cpu", backend="fastembed")
    all_devices = (*gpu_devices, cpu)

    result = SystemDevices(
        devices=all_devices,
        onnx_providers=tuple(onnx_providers),
    )

    if gpu_devices:
        names = ", ".join(f"{d.name} ({d.device}, {d.memory_mb}MB)" for d in gpu_devices)
        logger.info("Detected GPU devices: %s", names)
    else:
        logger.debug("No GPU devices detected; using CPU")
    logger.debug("ONNX Runtime providers: %s", onnx_providers)

    return result


def resolve_device(requested: str, system: SystemDevices | None = None) -> DeviceInfo:
    """Resolve a user-requested device string to a concrete DeviceInfo.

    Args:
        requested: One of 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1',
            'mps', 'xla', 'openvino'.
        system: Pre-detected system devices. If None, runs detection.

    Returns:
        The best matching DeviceInfo.

    Raises:
        ValueError: If the requested device is not available.
    """
    if system is None:
        system = detect_devices()

    if requested == "auto":
        return system.best

    if requested == "cpu":
        return system.cpu_device

    # Match by device string prefix (e.g. "cuda" matches "cuda:0")
    for d in system.devices:
        if d.device == requested:
            return d

    # "cuda" without index → first CUDA device
    if requested == "cuda":
        for d in system.devices:
            if d.device.startswith("cuda"):
                return d

    # OpenVINO special case — not a PyTorch device
    if requested == "openvino":
        if "OpenVINOExecutionProvider" in system.onnx_providers:
            return DeviceInfo(
                name="Intel OpenVINO",
                device="openvino",
                backend="fastembed",
            )
        msg = (
            "OpenVINO requested but OpenVINOExecutionProvider not available. "
            "Fix: install optimum[openvino] or onnxruntime with OpenVINO support."
        )
        raise ValueError(msg)

    msg = (
        f"Device {requested!r} not available. "
        f"Available: {', '.join(d.device for d in system.devices)}. "
        "Fix: use 'auto' for automatic detection, or install the required "
        "backend (e.g. `pip install kaos-nlp-transformers[torch]` for CUDA)."
    )
    raise ValueError(msg)


# Module-level cache
_cached_system: SystemDevices | None = None


def get_system_devices() -> SystemDevices:
    """Return cached system device detection (runs once per process)."""
    global _cached_system
    if _cached_system is None:
        _cached_system = detect_devices()
    return _cached_system


__all__ = [
    "DeviceInfo",
    "SystemDevices",
    "detect_devices",
    "get_system_devices",
    "resolve_device",
]
