use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_core::impl_enum_display;
use singe_cudnn_sys as sys;

/// Describes an operator used with tensor reduction operation.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
#[repr(u32)]
#[non_exhaustive]
pub enum ReduceTensorOperator {
    /// The operation to be performed is addition.
    Add = sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD as _,
    /// The operation to be performed is multiplication.
    Mul = sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MUL as _,
    /// The operation to be performed is a minimum comparison.
    Min = sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MIN as _,
    /// The operation to be performed is a maximum comparison.
    Max = sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MAX as _,
    /// The operation to be performed is a maximum comparison of absolute values.
    AbsoluteMax = sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_AMAX as _,
    /// The operation to be performed is addition of absolute values.
    Avg = sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_AVG as _,
    /// The operation to be performed is addition of absolute values.
    Norm1 = sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_NORM1 as _,
    /// The operation to be performed is a square root of the sum of squares.
    Norm2 = sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_NORM2 as _,
    /// The operation to be performed is multiplication, not including elements of value zero.
    MulNoZeros = sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS as _,
}

impl_enum_display!(ReduceTensorOperator, {
    Self::Add => "CUDNN_REDUCE_TENSOR_ADD",
    Self::Mul => "CUDNN_REDUCE_TENSOR_MUL",
    Self::Min => "CUDNN_REDUCE_TENSOR_MIN",
    Self::Max => "CUDNN_REDUCE_TENSOR_MAX",
    Self::AbsoluteMax => "CUDNN_REDUCE_TENSOR_AMAX",
    Self::Avg => "CUDNN_REDUCE_TENSOR_AVG",
    Self::Norm1 => "CUDNN_REDUCE_TENSOR_NORM1",
    Self::Norm2 => "CUDNN_REDUCE_TENSOR_NORM2",
    Self::MulNoZeros => "CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS",
});

/// Used to indicate whether indices are to be computed by tensor reduction operation.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
#[repr(u32)]
#[non_exhaustive]
pub enum ReduceTensorIndices {
    /// Do not compute indices.
    None = sys::cudnnReduceTensorIndices_t::CUDNN_REDUCE_TENSOR_NO_INDICES as _,
    /// Compute indices. The resulting indices are relative, and flattened.
    Flattened = sys::cudnnReduceTensorIndices_t::CUDNN_REDUCE_TENSOR_FLATTENED_INDICES as _,
}

impl_enum_display!(ReduceTensorIndices, {
    Self::None => "CUDNN_REDUCE_TENSOR_NO_INDICES",
    Self::Flattened => "CUDNN_REDUCE_TENSOR_FLATTENED_INDICES",
});

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
#[repr(u32)]
#[non_exhaustive]
pub enum IndicesType {
    U32 = sys::cudnnIndicesType_t::CUDNN_32BIT_INDICES as _,
    U64 = sys::cudnnIndicesType_t::CUDNN_64BIT_INDICES as _,
    U16 = sys::cudnnIndicesType_t::CUDNN_16BIT_INDICES as _,
    U8 = sys::cudnnIndicesType_t::CUDNN_8BIT_INDICES as _,
}

impl_enum_display!(IndicesType, {
    Self::U32 => "CUDNN_32BIT_INDICES",
    Self::U64 => "CUDNN_64BIT_INDICES",
    Self::U16 => "CUDNN_16BIT_INDICES",
    Self::U8 => "CUDNN_8BIT_INDICES",
});
