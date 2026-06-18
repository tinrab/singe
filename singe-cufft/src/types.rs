use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_cufft_sys as sys;

use singe_core::{impl_enum_conversion, impl_enum_display};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum Direction {
    Forward = sys::CUFFT_FORWARD,
    Inverse = sys::CUFFT_INVERSE as i32,
}

impl_enum_display!(Direction, {
    Direction::Forward => "CUFFT_FORWARD",
    Direction::Inverse => "CUFFT_INVERSE",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum TransformType {
    RealToComplex = sys::cufftType_t::CUFFT_R2C as _,
    ComplexToReal = sys::cufftType_t::CUFFT_C2R as _,
    ComplexToComplex = sys::cufftType_t::CUFFT_C2C as _,
    DoubleRealToDoubleComplex = sys::cufftType_t::CUFFT_D2Z as _,
    DoubleComplexToDoubleReal = sys::cufftType_t::CUFFT_Z2D as _,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PlanProperty {
    PatientJit = sys::cufftProperty_t::NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT as _,
    MaxNumHostThreads = sys::cufftProperty_t::NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS as _,
}

impl_enum_conversion!(sys::cufftProperty_t, PlanProperty);

impl_enum_display!(PlanProperty, {
    PlanProperty::PatientJit => "NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT",
    PlanProperty::MaxNumHostThreads => "NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS",
});
