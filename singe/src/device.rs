use std::fmt::{self, Display, Formatter};

/// Device placement for tensor storage and execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// Host CPU execution and memory.
    #[default]
    Cpu,
    /// CUDA device by zero-based device index.
    Cuda(i32),
    /// Shape-only tensors that record metadata without allocating memory or running kernels.
    Meta,
}

impl Device {
    /// Returns the CUDA device id, or `None` for CPU.
    #[must_use]
    pub fn cuda_id(self) -> Option<i32> {
        match self {
            Self::Cuda(id) => Some(id),
            Self::Cpu | Self::Meta => None,
        }
    }

    /// Returns whether this is the CPU device.
    #[must_use]
    pub const fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Returns whether this is a CUDA device.
    #[must_use]
    pub const fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    /// Returns whether this is the meta device, whose tensors carry metadata but no values.
    #[must_use]
    pub const fn is_meta(self) -> bool {
        matches!(self, Self::Meta)
    }
}

impl Display for Device {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Meta => write!(f, "meta"),
            Self::Cuda(id) => {
                if *id == 0 {
                    write!(f, "cuda")
                } else {
                    write!(f, "cuda:{id}")
                }
            }
        }
    }
}
