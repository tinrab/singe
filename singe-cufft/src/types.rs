use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_cufft_sys as sys;

use singe_core::impl_enum_display;

/// Sign of the Fourier exponent for a complex-to-complex transform.
///
/// cuFFT leaves both directions unnormalized; a forward transform followed by an inverse scales values by the transform element count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum Direction {
    /// Uses the negative sign in the Fourier exponent.
    Forward = sys::CUFFT_FORWARD,
    /// Uses the positive sign in the Fourier exponent.
    Inverse = sys::CUFFT_INVERSE.cast_signed(),
}

impl_enum_display!(Direction, {
    Direction::Forward => "CUFFT_FORWARD",
    Direction::Inverse => "CUFFT_INVERSE",
});

singe_core::native_enum! {
    /// Input and output data kinds and precision of a transform.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum TransformType: sys::cufftType_t {
        /// Real to complex (interleaved).
        RealToComplex = CUFFT_R2C,
        /// Complex (interleaved) to real.
        ComplexToReal = CUFFT_C2R,
        /// Complex to complex (interleaved).
        ComplexToComplex = CUFFT_C2C,
        /// Double to double-complex (interleaved).
        DoubleRealToDoubleComplex = CUFFT_D2Z,
        /// Double-complex (interleaved) to double.
        DoubleComplexToDoubleReal = CUFFT_Z2D,
        /// Double-complex to double-complex (interleaved).
        DoubleComplexToDoubleComplex = CUFFT_Z2Z,
    }
}

impl_enum_display!(TransformType, {
    TransformType::RealToComplex => "CUFFT_R2C",
    TransformType::ComplexToReal => "CUFFT_C2R",
    TransformType::ComplexToComplex => "CUFFT_C2C",
    TransformType::DoubleRealToDoubleComplex => "CUFFT_D2Z",
    TransformType::DoubleComplexToDoubleReal => "CUFFT_Z2D",
    TransformType::DoubleComplexToDoubleComplex => "CUFFT_Z2Z",
});

singe_core::native_enum! {
    /// Integer properties controlling native FFT planning and kernel generation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PlanProperty: sys::cufftProperty_t {
        /// Uses the Bluestein algorithm for all transforms in the plan.
        BluesteinOnly = NVFFT_PLAN_PROPERTY_INT64_BLUESTEIN_ONLY,
        /// Disables fused multiply-add operations.
        DisableFma = NVFFT_PLAN_PROPERTY_INT64_DISABLE_FMA,
        /// Enables runtime link-time-optimized kernels when nonzero; zero is the default.
        ///
        /// Set before planning.
        /// cuFFT does not allow changing this property after planning.
        PatientJit = NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT,
        /// Limits the number of host threads available to the native plan.
        MaxNumHostThreads = NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS,
    }
}

impl_enum_display!(PlanProperty, {
    PlanProperty::BluesteinOnly => "NVFFT_PLAN_PROPERTY_INT64_BLUESTEIN_ONLY",
    PlanProperty::DisableFma => "NVFFT_PLAN_PROPERTY_INT64_DISABLE_FMA",
    PlanProperty::PatientJit => "NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT",
    PlanProperty::MaxNumHostThreads => "NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS",
});
