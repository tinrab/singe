use num_enum::{IntoPrimitive, TryFromPrimitive};

use serde::{Deserialize, Serialize};

use singe_core::{impl_enum_conversion, impl_enum_display};

use singe_cudnn_sys as sys;

use crate::{error::Result, utility::check_range};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Indicates whether Tensor Core operations are permitted for a cuDNN operation.
#[repr(u32)]
#[non_exhaustive]
pub enum MathType {
    /// Tensor Core operations are not used on pre-NVIDIA A100 GPU devices.
    /// On A100 GPU architecture devices, Tensor Core TF32 operation is permitted.
    Default = sys::cudnnMathType_t::CUDNN_DEFAULT_MATH as _,
    /// Permits Tensor Core operations without actively down-converting tensors
    /// to use Tensor Cores.
    TensorOp = sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH as _,
    /// Permits Tensor Core operations and actively down-converts tensors to use
    /// Tensor Cores.
    TensorOpAllowConversion = sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION as _,
    /// Restricted to only kernels that use FMA instructions.
    ///
    /// On pre-NVIDIA A100 GPU devices, [`MathType::Default`] and [`MathType::Fma`] have the same behavior: Tensor Core kernels are not selected.
    /// With NVIDIA Ampere architecture and CUDA toolkit 11, [`MathType::Default`] permits TF32 Tensor Core operation and [`MathType::Fma`] does not.
    /// The TF32 behavior for [`MathType::Default`] and the other Tensor Core math types can be explicitly disabled by the environment variable `NVIDIA_TF32_OVERRIDE=0`.
    Fma = sys::cudnnMathType_t::CUDNN_FMA_MATH as _,
}

impl_enum_conversion!(sys::cudnnMathType_t, MathType);

impl_enum_display!(MathType, {
    Self::Default => "CUDNN_DEFAULT_MATH",
    Self::TensorOp => "CUDNN_TENSOR_OP_MATH",
    Self::TensorOpAllowConversion => "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION",
    Self::Fma => "CUDNN_FMA_MATH",
});

/// Whether a native operation propagates NaN values.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
#[repr(u32)]
#[non_exhaustive]
pub enum NanPropagation {
    /// Requests that NaN values are not propagated.
    NotPropagate = 0,
    /// Requests propagation of NaN values.
    Propagate = 1,
}

impl_enum_display!(NanPropagation, {
    Self::NotPropagate => "CUDNN_NOT_PROPAGATE_NAN",
    Self::Propagate => "CUDNN_PROPAGATE_NAN",
});

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Indicates whether computed results are reproducible.
#[repr(u32)]
#[non_exhaustive]
pub enum Determinism {
    /// Results are not guaranteed to be reproducible.
    NonDeterministic = sys::cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC as _,
    /// Results are guaranteed to be reproducible.
    Deterministic = sys::cudnnDeterminism_t::CUDNN_DETERMINISTIC as _,
}

impl_enum_conversion!(sys::cudnnDeterminism_t, Determinism);

impl_enum_display!(Determinism, {
    Self::NonDeterministic => "CUDNN_NON_DETERMINISTIC",
    Self::Deterministic => "CUDNN_DETERMINISTIC",
});

/// An exact native fraction with its supplied numerator and denominator preserved.
/// Construction does not normalize the value; operation geometry validates denominators before use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Fraction {
    numerator: i64,
    denominator: i64,
}

impl Fraction {
    /// Preserves the supplied numerator and denominator without validation or normalization.
    #[must_use]
    pub const fn new(numerator: i64, denominator: i64) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Represents an integer exactly with denominator one.
    #[must_use]
    pub const fn whole(value: i64) -> Self {
        Self::new(value, 1)
    }

    /// Returns the stored numerator without normalization.
    #[must_use]
    pub const fn numerator(self) -> i64 {
        self.numerator
    }

    /// Returns the stored denominator, which has not necessarily been validated.
    #[must_use]
    pub const fn denominator(self) -> i64 {
        self.denominator
    }

    pub(crate) fn validate(self, name: &str) -> Result<()> {
        check_range!(name, self.denominator > 0)
    }

    pub(crate) fn raw(self) -> sys::cudnnFraction_t {
        sys::cudnnFraction_t {
            numerator: self.numerator,
            denominator: self.denominator,
        }
    }
}

impl From<i32> for Fraction {
    fn from(value: i32) -> Self {
        Self::whole(i64::from(value))
    }
}

impl From<i64> for Fraction {
    fn from(value: i64) -> Self {
        Self::whole(value)
    }
}

impl From<(i32, i32)> for Fraction {
    fn from((numerator, denominator): (i32, i32)) -> Self {
        Self::new(i64::from(numerator), i64::from(denominator))
    }
}

impl From<(i64, i64)> for Fraction {
    fn from((numerator, denominator): (i64, i64)) -> Self {
        Self::new(numerator, denominator)
    }
}
