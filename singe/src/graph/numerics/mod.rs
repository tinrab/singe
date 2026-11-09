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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NumericalPolicy {
    pub(crate) precision: ComputePrecision,
    pub(crate) determinism: Determinism,
}

thread_local! {
    static CURRENT: Cell<NumericalPolicy> = const { Cell::new(NumericalPolicy {
        precision: ComputePrecision::Full,
        determinism: Determinism::Allow,
    }) };
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
