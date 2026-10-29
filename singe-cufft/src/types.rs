use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_cufft_sys as sys;

use singe_core::{impl_enum_conversion, impl_enum_display};

/// Sign of the Fourier exponent for a complex-to-complex transform.
///
/// cuFFT leaves both directions unnormalized; a forward transform followed by
/// an inverse scales values by the transform element count.
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

/// The cuFFT library supports complex- and real-data transforms. The `cufftType` data type is an enumeration of the types of transform data supported by cuFFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum TransformType {
    /// Real to complex (interleaved).
    RealToComplex = sys::cufftType_t::CUFFT_R2C as _,
    /// Complex (interleaved) to real.
    ComplexToReal = sys::cufftType_t::CUFFT_C2R as _,
    /// Complex to complex (interleaved).
    ComplexToComplex = sys::cufftType_t::CUFFT_C2C as _,
    /// Double to double-complex (interleaved).
    DoubleRealToDoubleComplex = sys::cufftType_t::CUFFT_D2Z as _,
    /// Double-complex (interleaved) to double.
    DoubleComplexToDoubleReal = sys::cufftType_t::CUFFT_Z2D as _,
    /// Double-complex to double-complex (interleaved).
    DoubleComplexToDoubleComplex = sys::cufftType_t::CUFFT_Z2Z as _,
}

impl_enum_conversion!(sys::cufftType_t, TransformType);

impl_enum_display!(TransformType, {
    TransformType::RealToComplex => "CUFFT_R2C",
    TransformType::ComplexToReal => "CUFFT_C2R",
    TransformType::ComplexToComplex => "CUFFT_C2C",
    TransformType::DoubleRealToDoubleComplex => "CUFFT_D2Z",
    TransformType::DoubleComplexToDoubleReal => "CUFFT_Z2D",
    TransformType::DoubleComplexToDoubleComplex => "CUFFT_Z2Z",
});

/// Integer properties controlling native FFT planning and kernel generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PlanProperty {
    /// Uses the Bluestein algorithm for all transforms in the plan.
    BluesteinOnly = sys::cufftProperty_t::NVFFT_PLAN_PROPERTY_INT64_BLUESTEIN_ONLY as _,
    /// Disables fused multiply-add operations.
    DisableFma = sys::cufftProperty_t::NVFFT_PLAN_PROPERTY_INT64_DISABLE_FMA as _,
    /// Enables runtime link-time-optimized kernels when nonzero; zero is the default.
    ///
    /// Set before planning. cuFFT does not allow changing this property after planning.
    PatientJit = sys::cufftProperty_t::NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT as _,
    /// Limits the number of host threads available to the native plan.
    MaxNumHostThreads = sys::cufftProperty_t::NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS as _,
}

impl_enum_conversion!(sys::cufftProperty_t, PlanProperty);

impl_enum_display!(PlanProperty, {
    PlanProperty::BluesteinOnly => "NVFFT_PLAN_PROPERTY_INT64_BLUESTEIN_ONLY",
    PlanProperty::DisableFma => "NVFFT_PLAN_PROPERTY_INT64_DISABLE_FMA",
    PlanProperty::PatientJit => "NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT",
    PlanProperty::MaxNumHostThreads => "NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS",
});
