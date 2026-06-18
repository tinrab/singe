use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cudnn_sys as sys;

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

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
#[repr(u32)]
#[non_exhaustive]
pub enum NanPropagation {
    NotPropagate = 0,
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
