use std::{cell::Cell, marker::PhantomData, rc::Rc};

/// Precision permitted for provider computation, independently of graph rewrites.
/// FP16/BF16 matmul and reductions accumulate in FP32 under both policies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ComputePrecision {
    /// Preserve storage input precision and use the prescribed accumulator type.
    #[default]
    Full,
    /// Permit TF32 input rounding for FP32 matrix operations.
    /// Providers may still select full precision.
    /// This does not permit FP16 accumulation.
    AllowTf32,
    /// Permit TF32 and approximations in generated kernels: fused multiply-add, approximate division, square root, and transcendental functions, and flushing denormals to zero.
    /// Generated kernels compile with NVRTC's `--use_fast_math`, so their results differ from the CPU reference by more than rounding order.
    Relaxed,
}

impl ComputePrecision {
    /// Returns whether FP32 matrix operations may round their inputs to TF32.
    #[must_use]
    pub const fn allows_tf32(self) -> bool {
        matches!(self, Self::AllowTf32 | Self::Relaxed)
    }
}

/// Repeatability on the same device, software versions, inputs, and execution configuration.
/// This does not promise identical bits across different devices.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Determinism {
    /// Permits implementations without a checked determinism guarantee.
    #[default]
    Allow,
    /// Select only implementations with a checked determinism contract, or fail.
    Require,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudnnGraphKind {
    Matmul,
    NonMatmul,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NumericalPolicy {
    pub(crate) precision: ComputePrecision,
    pub(crate) determinism: Determinism,
    // The precision of FP32 convolutions and the other cuDNN graphs that are not matmuls, apart from `precision`: TF32 by default, as PyTorch's `torch.backends.cudnn.allow_tf32` is on while its matmuls' TF32 is off.
    // A model whose outputs are thresholded or are themselves the signal, such as SuperPoint's keypoints or DAC's reconstruction, runs its convolutions at full precision (`with_convolution_precision`), since TF32 engines round apart from PyTorch's.
    pub(crate) cudnn_precision: ComputePrecision,
}

impl NumericalPolicy {
    const DEFAULT: Self = Self {
        precision: ComputePrecision::Full,
        determinism: Determinism::Allow,
        cudnn_precision: ComputePrecision::AllowTf32,
    };
}

impl Default for NumericalPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

thread_local! {
    static CURRENT: Cell<NumericalPolicy> = const { Cell::new(NumericalPolicy::DEFAULT) };
}

impl NumericalPolicy {
    #[cfg(feature = "cuda")]
    #[must_use]
    pub const fn precision(self) -> ComputePrecision {
        self.precision
    }

    #[cfg(feature = "cuda")]
    #[must_use]
    pub const fn determinism(self) -> Determinism {
        self.determinism
    }

    /// Read at provider/cache boundaries, never during logical inference.
    pub(crate) fn current() -> Self {
        CURRENT.with(Cell::get)
    }

    #[cfg(feature = "cuda")]
    pub(crate) const fn for_cudnn(self, kind: CudnnGraphKind) -> Self {
        match kind {
            CudnnGraphKind::Matmul => self,
            CudnnGraphKind::NonMatmul => Self {
                precision: self.cudnn_precision,
                ..self
            },
        }
    }
}

// Runs `f` with eager FP32 convolutions on this thread at `precision`, such as `AllowTf32` for PyTorch's default speed.
pub fn with_convolution_precision<T>(precision: ComputePrecision, f: impl FnOnce() -> T) -> T {
    let _scope = NumericalScope::enter(NumericalPolicy {
        cudnn_precision: precision,
        ..NumericalPolicy::current()
    });
    f()
}

/// Bridges an instance's explicit configuration to eager backend dispatch and native preparation.
/// The guard cannot move to another thread.
pub struct NumericalScope {
    previous: NumericalPolicy,
    _thread: PhantomData<Rc<()>>,
}

impl NumericalScope {
    pub(crate) fn enter(policy: NumericalPolicy) -> Self {
        let previous = NumericalPolicy::current();
        CURRENT.with(|current| current.set(policy));
        Self {
            previous,
            _thread: PhantomData,
        }
    }
}

impl Drop for NumericalScope {
    fn drop(&mut self) {
        CURRENT.with(|current| current.set(self.previous));
    }
}

#[cfg(test)]
mod tests;
