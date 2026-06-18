use std::fmt::Debug;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_core::{impl_enum_display, try_enum_from_raw};
use singe_cudnn_sys as sys;

pub use singe_cuda::types::{
    Complex32, Complex64, bf16, f4e2m1, f6e2m3, f6e3m2, f8e4m3, f8e5m2, f8ue8m0, f16,
};

use crate::error::Error;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Data type used by tensor and filter descriptors.
#[repr(u32)]
#[non_exhaustive]
pub enum DataType {
    /// The data is a 32-bit single-precision floating-point (`float`).
    F32 = sys::cudnnDataType_t::CUDNN_DATA_FLOAT as _,
    /// The data is a 64-bit double-precision floating-point (`double`).
    F64 = sys::cudnnDataType_t::CUDNN_DATA_DOUBLE as _,
    /// The data is a 16-bit floating-point.
    F16 = sys::cudnnDataType_t::CUDNN_DATA_HALF as _,
    /// The data is an 8-bit signed integer.
    I8 = sys::cudnnDataType_t::CUDNN_DATA_INT8 as _,
    /// The data is a 32-bit signed integer.
    I32 = sys::cudnnDataType_t::CUDNN_DATA_INT32 as _,
    /// The data is an 8-bit unsigned integer.
    U8 = sys::cudnnDataType_t::CUDNN_DATA_UINT8 as _,
    /// The data is a 16-bit quantity, with 7 mantissa bits, 8 exponent bits, and 1 sign bit.
    BF16 = sys::cudnnDataType_t::CUDNN_DATA_BFLOAT16 as _,
    /// The data is a 64-bit signed integer.
    I64 = sys::cudnnDataType_t::CUDNN_DATA_INT64 as _,
    /// The data is a boolean (`bool`).
    ///
    /// For [`BackendAttributeType::Boolean`](crate::attribute::BackendAttributeType::Boolean),
    /// elements are packed: one byte contains 8 boolean elements.
    /// Within each byte, elements are indexed from the least significant bit to the most significant bit.
    /// For example, a 1-dimensional tensor of 8 elements containing `01001111` has value 1 for elements 0 through 3, 0 for elements 4 and 5, 1 for element 6, and 0 for element 7.
    ///
    /// Tensors with more than 8 elements use more bytes, ordered from least significant to most significant byte.
    /// CUDA is little-endian, so the least significant byte has the lower memory address.
    /// For example, 16 elements containing `01001111 11111100` have value 1 for elements 0 through 3, 0 for elements 4 and 5, 1 for element 6, 0 for element 7, 0 for elements 8 and 9, and 1 for elements 10 through 15.
    Boolean = sys::cudnnDataType_t::CUDNN_DATA_BOOLEAN as _,
    /// The data is an 8-bit quantity, with 3 mantissa bits, 4 exponent bits, and 1 sign bit.
    F8E4M3 = sys::cudnnDataType_t::CUDNN_DATA_FP8_E4M3 as _,
    /// The data is an 8-bit quantity, with 2 mantissa bits, 5 exponent bits, and 1 sign bit.
    F8E5M2 = sys::cudnnDataType_t::CUDNN_DATA_FP8_E5M2 as _,
    /// The data type is a higher throughput but lower precision compute type (compared to [`DataType::F32`]) used for FP8 tensor core operations.
    FastFloatForFp8 = sys::cudnnDataType_t::CUDNN_DATA_FAST_FLOAT_FOR_FP8 as _,
    F8E8M0 = sys::cudnnDataType_t::CUDNN_DATA_FP8_E8M0 as _,
    F4E2M1 = sys::cudnnDataType_t::CUDNN_DATA_FP4_E2M1 as _,
    I4 = sys::cudnnDataType_t::CUDNN_DATA_INT4 as _,
    U4 = sys::cudnnDataType_t::CUDNN_DATA_UINT4 as _,
    U32 = sys::cudnnDataType_t::CUDNN_DATA_UINT32 as _,
    ComplexF32 = sys::cudnnDataType_t::CUDNN_DATA_COMPLEX_FP32 as _,
    ComplexF64 = sys::cudnnDataType_t::CUDNN_DATA_COMPLEX_FP64 as _,
}

impl_enum_display!(DataType, {
    Self::F32 => "CUDNN_DATA_FLOAT",
    Self::F64 => "CUDNN_DATA_DOUBLE",
    Self::F16 => "CUDNN_DATA_HALF",
    Self::I8 => "CUDNN_DATA_INT8",
    Self::I32 => "CUDNN_DATA_INT32",
    Self::U8 => "CUDNN_DATA_UINT8",
    Self::BF16 => "CUDNN_DATA_BFLOAT16",
    Self::I64 => "CUDNN_DATA_INT64",
    Self::Boolean => "CUDNN_DATA_BOOLEAN",
    Self::F8E4M3 => "CUDNN_DATA_FP8_E4M3",
    Self::F8E5M2 => "CUDNN_DATA_FP8_E5M2",
    Self::FastFloatForFp8 => "CUDNN_DATA_FAST_FLOAT_FOR_FP8",
    Self::F8E8M0 => "CUDNN_DATA_FP8_E8M0",
    Self::F4E2M1 => "CUDNN_DATA_FP4_E2M1",
    Self::I4 => "CUDNN_DATA_INT4",
    Self::U4 => "CUDNN_DATA_UINT4",
    Self::U32 => "CUDNN_DATA_UINT32",
    Self::ComplexF32 => "CUDNN_DATA_COMPLEX_FP32",
    Self::ComplexF64 => "CUDNN_DATA_COMPLEX_FP64",
});

impl DataType {
    pub const fn type_name(self) -> &'static str {
        match self {
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Boolean => "bool",
            Self::F8E4M3 => "f8e4m3",
            Self::F8E5M2 => "f8e5m2",
            Self::F8E8M0 => "f8ue8m0",
            Self::F4E2M1 => "f4e2m1",
            Self::FastFloatForFp8 => "fast_float_for_fp8",
            Self::I4 => "i4",
            Self::U4 => "u4",
            Self::ComplexF32 => "complex_f32",
            Self::ComplexF64 => "complex_f64",
        }
    }

    pub const fn size(self) -> i64 {
        match self {
            DataType::I8
            | DataType::U8
            | DataType::Boolean
            | DataType::F8E4M3
            | DataType::F8E5M2
            | DataType::F8E8M0
            | DataType::F4E2M1
            | DataType::I4
            | DataType::U4
            | DataType::FastFloatForFp8 => 1,
            DataType::F16 | DataType::BF16 => 2,
            DataType::F32 | DataType::I32 | DataType::U32 => 4,
            DataType::F64 | DataType::I64 | DataType::ComplexF32 => 8,
            DataType::ComplexF64 => 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TensorDataType {
    DataType(DataType),
    Int8x4,
    Uint8x4,
    Int8x32,
}

impl From<DataType> for TensorDataType {
    fn from(value: DataType) -> Self {
        Self::DataType(value)
    }
}

impl From<DataType> for sys::cudnnDataType_t {
    fn from(value: DataType) -> Self {
        match value {
            DataType::F32 => Self::CUDNN_DATA_FLOAT,
            DataType::F64 => Self::CUDNN_DATA_DOUBLE,
            DataType::F16 => Self::CUDNN_DATA_HALF,
            DataType::I8 => Self::CUDNN_DATA_INT8,
            DataType::I32 => Self::CUDNN_DATA_INT32,
            DataType::U8 => Self::CUDNN_DATA_UINT8,
            DataType::BF16 => Self::CUDNN_DATA_BFLOAT16,
            DataType::I64 => Self::CUDNN_DATA_INT64,
            DataType::Boolean => Self::CUDNN_DATA_BOOLEAN,
            DataType::F8E4M3 => Self::CUDNN_DATA_FP8_E4M3,
            DataType::F8E5M2 => Self::CUDNN_DATA_FP8_E5M2,
            DataType::FastFloatForFp8 => Self::CUDNN_DATA_FAST_FLOAT_FOR_FP8,
            DataType::F8E8M0 => Self::CUDNN_DATA_FP8_E8M0,
            DataType::F4E2M1 => Self::CUDNN_DATA_FP4_E2M1,
            DataType::I4 => Self::CUDNN_DATA_INT4,
            DataType::U4 => Self::CUDNN_DATA_UINT4,
            DataType::U32 => Self::CUDNN_DATA_UINT32,
            DataType::ComplexF32 => Self::CUDNN_DATA_COMPLEX_FP32,
            DataType::ComplexF64 => Self::CUDNN_DATA_COMPLEX_FP64,
        }
    }
}

impl TryFrom<sys::cudnnDataType_t> for DataType {
    type Error = Error;

    fn try_from(value: sys::cudnnDataType_t) -> Result<Self, Self::Error> {
        try_enum_from_raw("cudnn data type", value as u32).map_err(|error| {
            Error::InvalidEnumValue {
                name: error.name.into(),
                value: error.value,
            }
        })
    }
}

impl TryFrom<TensorDataType> for sys::cudnnDataType_t {
    type Error = Error;

    fn try_from(value: TensorDataType) -> Result<Self, Self::Error> {
        Ok(match value {
            TensorDataType::DataType(data_type) => data_type.into(),
            TensorDataType::Int8x4 => Self::CUDNN_DATA_INT8x4,
            TensorDataType::Uint8x4 => Self::CUDNN_DATA_UINT8x4,
            TensorDataType::Int8x32 => Self::CUDNN_DATA_INT8x32,
        })
    }
}

pub trait DataTypeLike: Clone + Copy + Debug + 'static {
    fn data_type() -> DataType;

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType>;
}

impl DataTypeLike for f32 {
    fn data_type() -> DataType {
        DataType::F32
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for f64 {
    fn data_type() -> DataType {
        DataType::F64
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for f16 {
    fn data_type() -> DataType {
        DataType::F16
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for bf16 {
    fn data_type() -> DataType {
        DataType::BF16
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for f8e4m3 {
    fn data_type() -> DataType {
        DataType::F8E4M3
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for f8e5m2 {
    fn data_type() -> DataType {
        DataType::F8E5M2
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for f8ue8m0 {
    fn data_type() -> DataType {
        DataType::F8E8M0
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for f6e2m3 {
    fn data_type() -> DataType {
        DataType::FastFloatForFp8
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for f6e3m2 {
    fn data_type() -> DataType {
        DataType::FastFloatForFp8
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for f4e2m1 {
    fn data_type() -> DataType {
        DataType::F4E2M1
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for bool {
    fn data_type() -> DataType {
        DataType::Boolean
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for i8 {
    fn data_type() -> DataType {
        DataType::I8
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        match V::WIDTH {
            4 => Some(TensorDataType::Int8x4),
            32 => Some(TensorDataType::Int8x32),
            _ => None,
        }
    }
}

impl DataTypeLike for u8 {
    fn data_type() -> DataType {
        DataType::U8
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        match V::WIDTH {
            4 => Some(TensorDataType::Uint8x4),
            _ => None,
        }
    }
}

impl DataTypeLike for i32 {
    fn data_type() -> DataType {
        DataType::I32
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for i64 {
    fn data_type() -> DataType {
        DataType::I64
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for u32 {
    fn data_type() -> DataType {
        DataType::U32
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for Complex32 {
    fn data_type() -> DataType {
        DataType::ComplexF32
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

impl DataTypeLike for Complex64 {
    fn data_type() -> DataType {
        DataType::ComplexF64
    }

    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        let _ = V::WIDTH;
        None
    }
}

pub trait DataTypeVectorization: Default + Clone + Copy + Debug + 'static {
    const WIDTH: usize;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Vec4;

impl DataTypeVectorization for Vec4 {
    const WIDTH: usize = 4;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Vec32;

impl DataTypeVectorization for Vec32 {
    const WIDTH: usize = 32;
}
