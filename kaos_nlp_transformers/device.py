"""Hardware detection and device routing for embedding inference.

Two layers of probing run on first call:

1. **Reachable** devices — accelerators that the *current Python install* can
   actually drive. Audit-06 KNT-501: post-torch-removal, this is exactly
   "what does ONNX Runtime see?" — ``CUDAExecutionProvider`` + the matching
   nvidia-smi enumeration is the CUDA path; ``OpenVINOExecutionProvider``
   is the OpenVINO path; ``CoreMLExecutionProvider`` (when added) covers
   Apple Silicon. These end up in ``SystemDevices.devices`` and are
   returned by ``resolve_device``.

2. **Latent** devices — accelerators that are physically present on the host
   but **not** reachable from this Python install (e.g. an NVIDIA GPU on a
   box where ``onnxruntime-gpu`` was not installed). These end up in
   ``SystemDevices.latent_devices`` with a typed ``install_extra`` hint so
   callers (CLI, MCP info tool, agents) can recommend the exact
   ``pip install kaos-nlp-transformers[<extra>]`` to recover. Post-0.1.0a6
   the only install hint we emit is ``"gpu"`` (onnxruntime-gpu) — the
   ``"torch"`` extra is gone.

The OS-level probes are deliberately import-free: NVIDIA goes through
``/dev/nvidia*`` plus a one-shot ``nvidia-smi`` exec; AMD ROCm through
``/dev/kfd``; Apple via ``platform.machine()``. They never import
onnxruntime, so they work on a fresh ``pip install
kaos-nlp-transformers`` (fastembed-only) base box.

Detection is cached at the module level — first call probes, subsequent
calls return the cached result.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kaos_core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reachable device types (existing surface, unchanged)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """A reachable hardware accelerator (or CPU fallback)."""

    name: str
    """Human-readable device name (e.g. 'NVIDIA GeForce RTX 5070 Ti')."""

    device: str
    """Device string: 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'openvino'.
    Audit-06 KNT-501: 'mps' and 'xla' were retired alongside the torch backend."""

    backend: str
    """Recommended embedding backend — always 'fastembed' post-audit-06.
    Field kept for forward compatibility (e.g. 'model2vec' for static-only
    devices, if we ever surface those as DeviceInfo entries)."""

    memory_mb: int = 0
    """Device memory in MB (0 for CPU or unknown)."""


@dataclass(frozen=True, slots=True)
class LatentDevice:
    """A physically-present accelerator that is NOT currently reachable.

    Latent devices are surfaced so that an operator (or an agent calling
    the MCP info tool) can install the correct extra and convert the
    latent device into a reachable one. The ``install_extra`` field is the
    machine-readable hint — use it directly in a recommendation like
    ``pip install kaos-nlp-transformers[{install_extra}]``.
    """

    name: str
    """Human-readable device name from the OS-level probe."""

    kind: str
    """Accelerator family: 'cuda' | 'rocm' | 'mps' | 'xla' (matches DeviceInfo.device prefix)."""

    reason: str
    """Why this device is not reachable (e.g. 'torch is not installed')."""

    install_extra: str | None
    """The pyproject extra to install ('torch', 'gpu', 'openvino') or ``None``
    if the fix is not a single extra (e.g. driver missing)."""

    detail: dict[str, Any] = field(default_factory=dict)
    """Free-form OS-probe payload (uuid, memory_mb, driver_version, …)."""


@dataclass(frozen=True, slots=True)
class SystemDevices:
    """All accelerators known to the runtime — reachable and latent."""

    devices: tuple[DeviceInfo, ...] = ()
    """Reachable devices, ordered by preference (best GPU first, CPU last)."""

    onnx_providers: tuple[str, ...] = ()
    """Available ONNX Runtime execution providers."""

    latent_devices: tuple[LatentDevice, ...] = ()
    """Physically-present accelerators that are not reachable from this install."""

    @property
    def best(self) -> DeviceInfo:
        """Return the highest-priority reachable device (first GPU, or CPU)."""
        if self.devices:
            return self.devices[0]
        return DeviceInfo(name="CPU", device="cpu", backend="fastembed")

    @property
    def has_gpu(self) -> bool:
        """True iff at least one reachable GPU is present."""
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

    @property
    def has_latent_gpu(self) -> bool:
        """True iff at least one OS-detected GPU is unreachable from this install."""
        return len(self.latent_devices) > 0


# ---------------------------------------------------------------------------
# Reachable-device probes (onnxruntime — Python-level)
# ---------------------------------------------------------------------------


def _detect_onnx_providers() -> list[str]:
    """Return available ONNX Runtime execution providers.

    Audit-06 KNT-501: post-torch-removal, onnxruntime is the only Python-
    level reachable-device probe. CUDA / OpenVINO / CoreML / ROCm all
    surface here as execution-provider names. The matching OS-level
    probes (nvidia-smi, /dev/kfd, platform.machine) backfill the
    "physically present but Python-unreachable" latent-device list.
    """
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except ImportError:
        return []


def _detect_reachable_gpus(onnx_providers: list[str]) -> list[DeviceInfo]:
    """Build the reachable-GPU list from the ONNX providers + nvidia-smi.

    Audit-06 KNT-501: replaces the old ``_detect_torch_devices``. The
    pattern: if ``CUDAExecutionProvider`` is in ``onnx_providers``,
    every NVIDIA GPU surfaced by ``nvidia-smi`` is a reachable
    ``cuda:N`` device. fastembed (or any onnxruntime-gpu consumer)
    can target it directly. If onnxruntime-gpu is NOT installed, the
    GPUs go to the latent list with ``install_extra="gpu"``.
    """
    devices: list[DeviceInfo] = []
    if "CUDAExecutionProvider" not in onnx_providers:
        return devices
    for row in _run_nvidia_smi():
        try:
            mem_mb = int(row.get("memory_mb", "0"))
        except ValueError:
            mem_mb = 0
        try:
            idx = int(row["index"])
        except (KeyError, ValueError):
            idx = len(devices)
        devices.append(
            DeviceInfo(
                name=row.get("name", "NVIDIA GPU"),
                device=f"cuda:{idx}",
                backend="fastembed",
                memory_mb=mem_mb,
            )
        )
    return devices


# ---------------------------------------------------------------------------
# OS-level latent-device probes (no torch/onnx imports — see module docstring)
# ---------------------------------------------------------------------------


# Default subprocess timeout for nvidia-smi / rocm-smi. Both should respond in
# tens of milliseconds on a healthy host; the 2.0s ceiling tolerates a stuck
# driver without blocking module import for long.
_SMI_TIMEOUT_S: float = 2.0


def _nvidia_devices_present() -> bool:
    """Cheap presence test — true iff /dev/nvidia* device nodes exist.

    The driver creates /dev/nvidia0 .. /dev/nvidiaN and /dev/nvidiactl when
    a card is bound. Checking the filesystem avoids forking nvidia-smi on
    boxes that have neither a driver nor a card.
    """
    if not Path("/dev").is_dir():
        return False
    try:
        return any(p.name.startswith("nvidia") for p in Path("/dev").iterdir())
    except OSError:
        return False


def _run_nvidia_smi() -> list[dict[str, str]]:
    """Return one dict per GPU from nvidia-smi, or [] on any failure.

    The query format is fixed and machine-readable: ``--format=csv,noheader,nounits``
    so memory.total comes back as an integer-valued MiB string with no
    surrounding noise. Any non-zero exit, timeout, or parse failure returns []
    rather than raising — the OS probe is best-effort and never blocks
    detection of reachable devices.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        result = subprocess.run(
            [
                smi,
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT_S,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0:
        return []

    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        idx, uuid, name, mem_mib, driver = parts
        rows.append(
            {
                "index": idx,
                "uuid": uuid,
                "name": name,
                "memory_mb": mem_mib,
                "driver_version": driver,
            }
        )
    return rows


def _probe_nvidia() -> list[LatentDevice]:
    """OS-level NVIDIA GPU probe — independent of torch/onnxruntime."""
    if not _nvidia_devices_present():
        return []
    rows = _run_nvidia_smi()
    if not rows:
        return []

    out: list[LatentDevice] = []
    for row in rows:
        try:
            mem_mb = int(row.get("memory_mb", "0"))
        except ValueError:
            mem_mb = 0
        out.append(
            LatentDevice(
                name=row["name"],
                kind="cuda",
                reason=(
                    "onnxruntime-gpu is not installed; the GPU is visible to "
                    "the driver but not to this Python process. Audit-06 "
                    "KNT-501: torch is no longer required — the GPU on-ramp "
                    "is the [gpu] extra (onnxruntime-gpu)."
                ),
                install_extra="gpu",
                detail={
                    "index": int(row["index"]) if row["index"].isdigit() else row["index"],
                    "uuid": row["uuid"],
                    "memory_mb": mem_mb,
                    "driver_version": row["driver_version"],
                },
            )
        )
    return out


def _probe_rocm() -> list[LatentDevice]:
    """OS-level AMD ROCm GPU probe.

    /dev/kfd is created by amdkfd when ROCm is installed and a supported
    card is present. We don't attempt to enumerate cards by name without
    rocm-smi because the AMD identifier path is messier than NVIDIA's; one
    bucket entry is enough to drive the install hint.
    """
    if not Path("/dev/kfd").exists():
        return []
    return [
        LatentDevice(
            name="AMD ROCm GPU",
            kind="rocm",
            reason=(
                "/dev/kfd is present (ROCm driver loaded) but onnxruntime "
                "with ROCm support is not in this environment. Audit-06 "
                "KNT-501: ROCm now requires onnxruntime-rocm, not torch."
            ),
            install_extra="gpu",
            detail={},
        )
    ]


def _probe_apple() -> list[LatentDevice]:
    """OS-level Apple Silicon (MPS) probe — fires only on arm64 macOS."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    return [
        LatentDevice(
            name="Apple Silicon (MPS)",
            kind="mps",
            reason=(
                "Running on Apple Silicon. Audit-06 KNT-501: torch + MPS is "
                "no longer the GPU on-ramp; install onnxruntime with the "
                "CoreMLExecutionProvider for Apple-Silicon acceleration."
            ),
            install_extra=None,
            detail={"machine": platform.machine(), "system": platform.system()},
        )
    ]


def _detect_latent_devices(
    reachable: list[DeviceInfo], onnx_providers: list[str]
) -> list[LatentDevice]:
    """Run all OS-level probes and reconcile against reachable devices.

    A probe-detected GPU is considered LATENT only if no reachable GPU of the
    same kind already covers it. We compare by ``kind`` (cuda/rocm/mps) and
    by **count**, not by name match — when torch sees the GPUs, it does so
    fully, and we don't want false-positive latents in that case.

    The reconciliation also folds in the onnxruntime-gpu detection: if torch
    isn't installed but ``CUDAExecutionProvider`` is in onnx_providers, we
    still say the NVIDIA GPU is reachable (via fastembed), so it's not
    latent. The current registry default is fastembed, so this matters.
    """
    candidates: list[LatentDevice] = []
    candidates.extend(_probe_nvidia())
    candidates.extend(_probe_rocm())
    candidates.extend(_probe_apple())

    if not candidates:
        return []

    # Counts of reachable devices by kind, to subtract from the candidate list.
    reachable_by_kind: dict[str, int] = {}
    for d in reachable:
        if d.device.startswith("cuda"):
            reachable_by_kind["cuda"] = reachable_by_kind.get("cuda", 0) + 1
        elif d.device == "mps":
            reachable_by_kind["mps"] = reachable_by_kind.get("mps", 0) + 1
        # ROCm presents as CUDA in PyTorch — already counted above.

    # If onnxruntime-gpu is present, fastembed can drive NVIDIA cards even
    # without torch. Treat the NVIDIA group as reachable in that case.
    if "CUDAExecutionProvider" in onnx_providers and reachable_by_kind.get("cuda", 0) == 0:
        # Pretend torch saw them so we don't list every GPU as latent — the
        # actual reachable_devices list won't gain entries (we only build
        # those from torch), but the latent list correctly excludes them.
        reachable_by_kind["cuda"] = sum(1 for c in candidates if c.kind == "cuda")

    out: list[LatentDevice] = []
    seen_per_kind: dict[str, int] = {}
    for cand in candidates:
        used = seen_per_kind.get(cand.kind, 0)
        if used < reachable_by_kind.get(cand.kind, 0):
            seen_per_kind[cand.kind] = used + 1
            continue
        out.append(cand)
    return out


# ---------------------------------------------------------------------------
# Top-level detection
# ---------------------------------------------------------------------------


def detect_devices() -> SystemDevices:
    """Detect all available hardware accelerators — reachable and latent.

    Returns a ``SystemDevices`` with reachable GPU devices ordered by memory
    (largest first), CPU as fallback, and latent devices populated from the
    OS-level probes. Logs at WARNING level when a latent GPU exists but no
    reachable GPU does — the silent-CPU-fallback failure mode this guard
    was built to fix.
    """
    onnx_providers = _detect_onnx_providers()
    gpu_devices = _detect_reachable_gpus(onnx_providers)
    latent = _detect_latent_devices(gpu_devices, onnx_providers)

    # Sort GPU devices by memory descending (prefer bigger GPUs)
    gpu_devices.sort(key=lambda d: d.memory_mb, reverse=True)

    # Always include CPU as fallback
    cpu = DeviceInfo(name="CPU", device="cpu", backend="fastembed")
    all_devices = (*gpu_devices, cpu)

    result = SystemDevices(
        devices=all_devices,
        onnx_providers=tuple(onnx_providers),
        latent_devices=tuple(latent),
    )

    if gpu_devices:
        names = ", ".join(f"{d.name} ({d.device}, {d.memory_mb}MB)" for d in gpu_devices)
        logger.info("Detected GPU devices: %s", names)
    elif latent:
        # The motivating fix: on a GPU box where the user installed only the
        # base package, the OS probes find the GPU but the Python probes
        # don't. Don't bury this at debug — the user is paying for silicon
        # they're not using.
        hints = "; ".join(
            f"{d.name} ({d.kind}) — pip install kaos-nlp-transformers[{d.install_extra}]"
            if d.install_extra
            else f"{d.name} ({d.kind}) — {d.reason}"
            for d in latent
        )
        logger.warning(
            "Detected %d latent accelerator(s) NOT reachable from this Python "
            "environment; falling back to CPU. Install hints: %s",
            len(latent),
            hints,
        )
    else:
        logger.debug("No GPU devices detected; using CPU")
    logger.debug("ONNX Runtime providers: %s", onnx_providers)

    return result


def resolve_device(requested: str, system: SystemDevices | None = None) -> DeviceInfo:
    """Resolve a user-requested device string to a concrete reachable DeviceInfo.

    Args:
        requested: One of 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1',
            'mps', 'xla', 'openvino'.
        system: Pre-detected system devices. If None, runs detection.

    Returns:
        The best matching reachable DeviceInfo.

    Raises:
        DeviceNotReachableError: If the requested device matches a *latent*
            device — physically present but not reachable. Carries an
            ``install_extra`` detail so callers can recommend the fix.
        ValueError: If the requested device is not present at all (typo,
            wrong index, OpenVINO without the provider, etc.).
    """
    # Local import — DeviceNotReachableError lives in errors.py and errors.py
    # has no other reason to depend on device.py, so a top-level import would
    # introduce a needless cycle.
    from kaos_nlp_transformers.errors import DeviceNotReachableError

    if system is None:
        # Use the cached snapshot so repeated EmbeddingModel.load() calls in a
        # long-running process (MCP server, retrieval pipeline) don't re-fork
        # nvidia-smi for every load. Tests that need a fresh probe call
        # ``_reset_cache_for_tests`` and then ``detect_devices`` directly.
        system = get_system_devices()

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

    # Latent-device check: if a CUDA / ROCm / MPS device of the right kind
    # was detected at the OS level but isn't reachable, raise the typed
    # DeviceNotReachableError instead of a generic ValueError so agents can
    # extract the install hint structurally.
    requested_kind = _kind_of(requested)
    if requested_kind is not None:
        for latent in system.latent_devices:
            if latent.kind == requested_kind:
                raise DeviceNotReachableError(requested=requested, latent=latent)

    msg = (
        f"Device {requested!r} not available. "
        f"Available: {', '.join(d.device for d in system.devices)}. "
        "Fix: use 'auto' for automatic detection, or install the required "
        "backend (e.g. `pip install kaos-nlp-transformers[gpu]` for CUDA "
        "via onnxruntime-gpu — audit-06 KNT-501 retired the [torch] extra)."
    )
    raise ValueError(msg)


def _kind_of(device: str) -> str | None:
    """Map a requested device string to its accelerator kind, or None for CPU/unknown."""
    if device.startswith("cuda"):
        return "cuda"
    if device == "mps":
        return "mps"
    if device == "rocm":
        return "rocm"
    return None


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------


_cached_system: SystemDevices | None = None


def get_system_devices() -> SystemDevices:
    """Return cached system device detection (runs once per process)."""
    global _cached_system
    if _cached_system is None:
        _cached_system = detect_devices()
    return _cached_system


def _reset_cache_for_tests() -> None:
    """Clear the module-level cache. Test-only — do not call from app code."""
    global _cached_system
    _cached_system = None


__all__ = [
    "DeviceInfo",
    "LatentDevice",
    "SystemDevices",
    "detect_devices",
    "get_system_devices",
    "resolve_device",
]
