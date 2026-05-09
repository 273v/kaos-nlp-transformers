//! Device enumeration for the Rust core.
//!
//! Two layers, exactly mirroring the Python ``device.py`` model:
//!
//! 1. **Compile-time capabilities** — what feature flags were enabled
//!    when this cdylib was built. ``cuda`` requires ``--features gpu``;
//!    ``openvino`` requires ``--features openvino``. Exposed to Python
//!    via ``_rust.registry.capabilities()`` (P3.1).
//!
//! 2. **Runtime device selection** — when the caller asks for ``cuda``,
//!    we check the compile-time gate AND verify the EP is actually
//!    available in the loaded ort runtime. Failure modes map to
//!    ``BackendError::{BackendNotInstalled, Device}``.
//!
//! The Python-side device probe (nvidia-smi, /dev/kfd, etc.) stays in
//! ``device.py``; this layer is the "what does the Rust extension
//! support, given how it was compiled" answer.

use crate::core::error::{BackendError, Result};

/// A reachable device the Rust backend can target.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU. Always available.
    Cpu,
    /// NVIDIA GPU at index N. Requires ``--features gpu``.
    Cuda(u32),
    /// Intel OpenVINO. Requires ``--features openvino``.
    OpenVino,
}

impl Device {
    /// Parse a device string from the Python side. Accepts ``"cpu"``,
    /// ``"cuda"``, ``"cuda:N"``, ``"openvino"``. Unknown values
    /// return a ``Device`` error so the binding layer can emit a
    /// ``DeviceNotReachableError`` / ``ValueError`` upstream.
    pub fn parse(spec: &str) -> Result<Self> {
        match spec {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda(0)),
            "openvino" => Ok(Self::OpenVino),
            other if other.starts_with("cuda:") => {
                let idx_str = &other[5..];
                idx_str
                    .parse::<u32>()
                    .map(Self::Cuda)
                    .map_err(|e| BackendError::Device {
                        requested: spec.to_string(),
                        reason: format!("could not parse cuda device index: {e}"),
                    })
            }
            _ => Err(BackendError::Device {
                requested: spec.to_string(),
                reason: format!(
                    "unknown device {spec:?}; valid: 'cpu', 'cuda', 'cuda:N', 'openvino'"
                ),
            }),
        }
    }

    /// Render the device back to its canonical string form (matches
    /// what ``DeviceInfo.device`` carries on the Python side).
    pub fn as_str(&self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::Cuda(n) => format!("cuda:{n}"),
            Self::OpenVino => "openvino".to_string(),
        }
    }

    /// True iff the cdylib was compiled with the feature flag this
    /// device requires. Runtime EP-availability is a *separate* check
    /// performed when the ort Session is built (see
    /// ``ort_runtime::OrtBackend::load`` in P2.4) — a wheel built with
    /// ``--features gpu`` on a host without an NVIDIA driver will
    /// compile fine but fail at session-build time.
    pub fn is_compiled_in(&self) -> bool {
        match self {
            Self::Cpu => true,
            Self::Cuda(_) => cfg!(feature = "gpu"),
            Self::OpenVino => cfg!(feature = "openvino"),
        }
    }

    /// Failure-mode message when ``is_compiled_in`` returns false.
    /// Three-part shape (what / fix / alternative) so the PyO3
    /// binding can preserve it verbatim through to the Python
    /// ``BackendNotInstalledError``.
    pub fn install_extra_message(&self) -> Option<String> {
        match self {
            Self::Cpu => None,
            Self::Cuda(_) => Some(
                "CUDA execution provider not compiled into this wheel. \
                 Fix: install the kaos-nlp-transformers-gpu companion package \
                 (`pip install kaos-nlp-transformers[gpu]`). \
                 Alternative: use device='cpu' or device='auto'."
                    .to_string(),
            ),
            Self::OpenVino => Some(
                "OpenVINO execution provider not compiled into this wheel. \
                 Fix: install the kaos-nlp-transformers-gpu[openvino] \
                 companion. Alternative: use device='cpu'."
                    .to_string(),
            ),
        }
    }
}

/// Compile-time capability snapshot. Exposed to Python by
/// ``bindings::registry::capabilities()``.
#[derive(Debug, Clone)]
pub struct Capabilities {
    /// Always true; CPU is the baseline.
    pub cpu: bool,
    /// True iff the cdylib was built with ``--features gpu``.
    pub cuda: bool,
    /// True iff the cdylib was built with ``--features openvino``.
    pub openvino: bool,
}

impl Capabilities {
    /// Snapshot the cdylib's compile-time feature flags.
    pub fn current() -> Self {
        Self {
            cpu: true,
            cuda: cfg!(feature = "gpu"),
            openvino: cfg!(feature = "openvino"),
        }
    }
}

impl Default for Capabilities {
    fn default() -> Self {
        Self::current()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu() {
        assert_eq!(Device::parse("cpu").unwrap(), Device::Cpu);
    }

    #[test]
    fn parse_cuda_no_index() {
        assert_eq!(Device::parse("cuda").unwrap(), Device::Cuda(0));
    }

    #[test]
    fn parse_cuda_with_index() {
        assert_eq!(Device::parse("cuda:3").unwrap(), Device::Cuda(3));
    }

    #[test]
    fn parse_openvino() {
        assert_eq!(Device::parse("openvino").unwrap(), Device::OpenVino);
    }

    #[test]
    fn parse_invalid() {
        let err = Device::parse("mps").unwrap_err();
        match err {
            BackendError::Device { requested, .. } => assert_eq!(requested, "mps"),
            _ => panic!("expected Device error"),
        }
    }

    #[test]
    fn parse_cuda_bad_index() {
        let err = Device::parse("cuda:not-a-number").unwrap_err();
        assert!(matches!(err, BackendError::Device { .. }));
    }

    #[test]
    fn round_trip_string() {
        assert_eq!(Device::parse("cpu").unwrap().as_str(), "cpu");
        assert_eq!(Device::parse("cuda:1").unwrap().as_str(), "cuda:1");
        assert_eq!(Device::parse("openvino").unwrap().as_str(), "openvino");
    }

    #[test]
    fn cpu_always_compiled() {
        assert!(Device::Cpu.is_compiled_in());
    }

    #[test]
    fn capabilities_cpu_baseline() {
        let caps = Capabilities::current();
        assert!(caps.cpu);
        // cuda / openvino reflect feature flags — verify they line up
        // with the corresponding cfg!() values.
        assert_eq!(caps.cuda, cfg!(feature = "gpu"));
        assert_eq!(caps.openvino, cfg!(feature = "openvino"));
    }
}
