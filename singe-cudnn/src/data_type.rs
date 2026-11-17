//! Tensor storage data types, packed encodings, and scalar type associations.

mod floating_private {
    #[expect(
        unnameable_types,
        reason = "seals `FloatingElement`, so callers cannot name or implement it"
    )]
    pub trait Sealed {}
}

use std::fmt::Debug;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_core::{impl_enum_display, try_enum_from_raw};
pub use singe_cuda::types::{
    Complex32, Complex64, bf16, f4e2m1, f8e4m3, f8e5m2, f8ue5m3, f8ue8m0, f16,
};
use singe_cuda::view::NoUninitDeviceRepr;
use singe_cudnn_sys as sys;

use crate::error::Error;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Data type used by tensor and filter descriptors.
#[repr(u32)]
#[non_exhaustive]
pub enum DataType {
    /// Unsigned FP8 with 5-bit exponent and 3-bit mantissa, for block scaling.
    F8UE5M3 = sys::cudnnDataType_t::CUDNN_DATA_FP8_E5M3.0 as _,
    /// The data is a 32-bit single-precision floating-point (`float`).
    F32 = sys::cudnnDataType_t::CUDNN_DATA_FLOAT.0 as _,
    /// The data is a 64-bit double-precision floating-point (`double`).
    F64 = sys::cudnnDataType_t::CUDNN_DATA_DOUBLE.0 as _,
    /// The data is a 16-bit floating-point.
    F16 = sys::cudnnDataType_t::CUDNN_DATA_HALF.0 as _,
    /// The data is an 8-bit signed integer.
    I8 = sys::cudnnDataType_t::CUDNN_DATA_INT8.0 as _,
    /// The data is a 32-bit signed integer.
    I32 = sys::cudnnDataType_t::CUDNN_DATA_INT32.0 as _,
    /// The data is an 8-bit unsigned integer.
    U8 = sys::cudnnDataType_t::CUDNN_DATA_UINT8.0 as _,
    /// The data is a 16-bit quantity, with 7 mantissa bits, 8 exponent bits, and 1 sign bit.
    Bf16 = sys::cudnnDataType_t::CUDNN_DATA_BFLOAT16.0 as _,
    /// The data is a 64-bit signed integer.
    I64 = sys::cudnnDataType_t::CUDNN_DATA_INT64.0 as _,
    /// Bit-packed booleans, with eight elements per byte.
    /// Element zero occupies the least significant bit.
    /// Use [`Self::ByteBoolean`] for Rust `bool` storage.
    Boolean = sys::cudnnDataType_t::CUDNN_DATA_BOOLEAN.0 as _,
    /// One boolean per byte, matching Rust `bool` storage.
    /// Use [`Self::Boolean`] for bit-packed boolean tensors.
    ByteBoolean = sys::cudnnDataType_t::CUDNN_DATA_BYTE_BOOLEAN.0 as _,
    /// The data is an 8-bit quantity, with 3 mantissa bits, 4 exponent bits, and 1 sign bit.
    F8E4M3 = sys::cudnnDataType_t::CUDNN_DATA_FP8_E4M3.0 as _,
    /// The data is an 8-bit quantity, with 2 mantissa bits, 5 exponent bits, and 1 sign bit.
    F8E5M2 = sys::cudnnDataType_t::CUDNN_DATA_FP8_E5M2.0 as _,
    /// The data type is a higher throughput but lower precision compute type (compared to [`DataType::F32`]) used for FP8 tensor core operations.
    FastFloatForFp8 = sys::cudnnDataType_t::CUDNN_DATA_FAST_FLOAT_FOR_FP8.0 as _,
    /// Pure-exponent scale format (8-bit exponent, 0-bit mantissa) for block scaling.
    F8UE8M0 = sys::cudnnDataType_t::CUDNN_DATA_FP8_E8M0.0 as _,
    /// FP4 with 2-bit exponent, 1-bit mantissa.
    F4E2M1 = sys::cudnnDataType_t::CUDNN_DATA_FP4_E2M1.0 as _,
    /// 4-bit signed integer.
    I4 = sys::cudnnDataType_t::CUDNN_DATA_INT4.0 as _,
    /// 4-bit unsigned integer.
    U4 = sys::cudnnDataType_t::CUDNN_DATA_UINT4.0 as _,
    /// 32-bit unsigned integer.
    U32 = sys::cudnnDataType_t::CUDNN_DATA_UINT32.0 as _,
    /// Complex 32-bit floating point.
    ComplexF32 = sys::cudnnDataType_t::CUDNN_DATA_COMPLEX_FP32.0 as _,
    /// Complex 64-bit floating point.
    ComplexF64 = sys::cudnnDataType_t::CUDNN_DATA_COMPLEX_FP64.0 as _,
}

impl_enum_display!(DataType, {
    Self::F8UE5M3 => "CUDNN_DATA_FP8_E5M3",
    Self::F32 => "CUDNN_DATA_FLOAT",
    Self::F64 => "CUDNN_DATA_DOUBLE",
    Self::F16 => "CUDNN_DATA_HALF",
    Self::I8 => "CUDNN_DATA_INT8",
    Self::I32 => "CUDNN_DATA_INT32",
    Self::U8 => "CUDNN_DATA_UINT8",
    Self::Bf16 => "CUDNN_DATA_BFLOAT16",
    Self::I64 => "CUDNN_DATA_INT64",
    Self::Boolean => "CUDNN_DATA_BOOLEAN",
    Self::ByteBoolean => "CUDNN_DATA_BYTE_BOOLEAN",
    Self::F8E4M3 => "CUDNN_DATA_FP8_E4M3",
    Self::F8E5M2 => "CUDNN_DATA_FP8_E5M2",
    Self::FastFloatForFp8 => "CUDNN_DATA_FAST_FLOAT_FOR_FP8",
    Self::F8UE8M0 => "CUDNN_DATA_FP8_E8M0",
    Self::F4E2M1 => "CUDNN_DATA_FP4_E2M1",
    Self::I4 => "CUDNN_DATA_INT4",
    Self::U4 => "CUDNN_DATA_UINT4",
    Self::U32 => "CUDNN_DATA_UINT32",
    Self::ComplexF32 => "CUDNN_DATA_COMPLEX_FP32",
    Self::ComplexF64 => "CUDNN_DATA_COMPLEX_FP64",
});

impl DataType {
    /// Native tensor storage bits per element, including packed element types.
    ///
    /// # Panics
    ///
    /// Panics if the native byte size cannot be represented as `usize`.
    /// Every current variant has a byte size between 1 and 16, so valid variants fit supported host targets.
    #[must_use]
    pub fn storage_bits(self) -> usize {
        match self {
            Self::Boolean => 1,
            Self::F4E2M1 | Self::I4 | Self::U4 => 4,
            _ => {
                usize::try_from(self.size()).expect("native data type size fits the host size") * 8
            }
        }
    }

    /// Returns the stable diagnostic name of the native scalar or packed encoding.
    #[must_use]
    pub const fn type_name(self) -> &'static str {
        match self {
            Self::F8UE5M3 => "f8ue5m3",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Boolean => "bool",
            Self::ByteBoolean => "byte_bool",
            Self::F8E4M3 => "f8e4m3",
            Self::F8E5M2 => "f8e5m2",
            Self::F8UE8M0 => "f8ue8m0",
            Self::F4E2M1 => "f4e2m1",
            Self::FastFloatForFp8 => "fast_float_for_fp8",
            Self::I4 => "i4",
            Self::U4 => "u4",
            Self::ComplexF32 => "complex_f32",
            Self::ComplexF64 => "complex_f64",
        }
    }

    /// Returns the byte width of one scalar representation, rounding packed sub-byte encodings up to one byte.
    ///
    /// Use [`Self::storage_bits`] for the actual bits per packed tensor element.
    #[must_use]
    pub const fn size(self) -> i64 {
        match self {
            Self::F8UE5M3
            | Self::I8
            | Self::U8
            | Self::Boolean
            | Self::ByteBoolean
            | Self::F8E4M3
            | Self::F8E5M2
            | Self::F8UE8M0
            | Self::F4E2M1
            | Self::I4
            | Self::U4
            | Self::FastFloatForFp8 => 1,
            Self::F16 | Self::Bf16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::ComplexF32 => 8,
            Self::ComplexF64 => 16,
        }
    }
}

/// Native scalar or vectorized storage encoding for legacy tensor descriptors.
/// Known Rust encodings convert to native values infallibly; native values require decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TensorDataType {
    /// A scalar or packed encoding without legacy channel vectorization.
    DataType(DataType),
    /// Four signed 8-bit lanes per legacy vectorized storage unit.
    Int8x4,
    /// Four unsigned 8-bit lanes per legacy vectorized storage unit.
    Uint8x4,
    /// Thirty-two signed 8-bit lanes per legacy vectorized storage unit.
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
            DataType::F8UE5M3 => Self::CUDNN_DATA_FP8_E5M3,
            DataType::F32 => Self::CUDNN_DATA_FLOAT,
            DataType::F64 => Self::CUDNN_DATA_DOUBLE,
            DataType::F16 => Self::CUDNN_DATA_HALF,
            DataType::I8 => Self::CUDNN_DATA_INT8,
            DataType::I32 => Self::CUDNN_DATA_INT32,
            DataType::U8 => Self::CUDNN_DATA_UINT8,
            DataType::Bf16 => Self::CUDNN_DATA_BFLOAT16,
            DataType::I64 => Self::CUDNN_DATA_INT64,
            DataType::Boolean => Self::CUDNN_DATA_BOOLEAN,
            DataType::ByteBoolean => Self::CUDNN_DATA_BYTE_BOOLEAN,
            DataType::F8E4M3 => Self::CUDNN_DATA_FP8_E4M3,
            DataType::F8E5M2 => Self::CUDNN_DATA_FP8_E5M2,
            DataType::FastFloatForFp8 => Self::CUDNN_DATA_FAST_FLOAT_FOR_FP8,
            DataType::F8UE8M0 => Self::CUDNN_DATA_FP8_E8M0,
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
        Ok(try_enum_from_raw("DataType", value.0)?)
    }
}

impl From<TensorDataType> for sys::cudnnDataType_t {
    fn from(value: TensorDataType) -> Self {
        match value {
            TensorDataType::DataType(data_type) => data_type.into(),
            TensorDataType::Int8x4 => Self::CUDNN_DATA_INT8x4,
            TensorDataType::Uint8x4 => Self::CUDNN_DATA_UINT8x4,
            TensorDataType::Int8x32 => Self::CUDNN_DATA_INT8x32,
        }
    }
}

impl TryFrom<sys::cudnnDataType_t> for TensorDataType {
    type Error = Error;

    fn try_from(native: sys::cudnnDataType_t) -> Result<Self, Self::Error> {
        match native {
            sys::cudnnDataType_t::CUDNN_DATA_INT8x4 => Ok(Self::Int8x4),
            sys::cudnnDataType_t::CUDNN_DATA_UINT8x4 => Ok(Self::Uint8x4),
            sys::cudnnDataType_t::CUDNN_DATA_INT8x32 => Ok(Self::Int8x32),
            _ => Ok(Self::DataType(native.try_into()?)),
        }
    }
}

/// A Rust element representation matching a cuDNN datatype and vectorization.
///
/// # Safety
///
/// The reported datatype, scalar encoding, alignment and valid values must match cuDNN.
/// Values must contain initialized bytes.
/// Packed encodings describe scalar values here; ordinary Rust arrays of those values are not native packed tensor storage.
/// Consumers must use byte-oriented storage for packed tensors.
/// Rust `bool` uses [`DataType::ByteBoolean`]; bit-packed [`DataType::Boolean`] uses encoded bytes.
pub unsafe trait TensorElement: NoUninitDeviceRepr + Debug {
    /// Returns the native scalar encoding represented by this Rust element type.
    fn data_type() -> DataType;

    /// Native encoding for the requested legacy vectorization, or `None` when unsupported.
    /// Scalar-only representations use this default.
    #[must_use]
    fn tensor_data_type<V: DataTypeVectorization>() -> Option<TensorDataType> {
        None
    }
}

// SAFETY: `f32` is the IEEE binary32 element of `CUDNN_DATA_FLOAT`, with the same size, alignment and every bit pattern valid.
unsafe impl TensorElement for f32 {
    fn data_type() -> DataType {
        DataType::F32
    }
}

// SAFETY: `f64` is the IEEE binary64 element of `CUDNN_DATA_DOUBLE`, with the same size, alignment and every bit pattern valid.
unsafe impl TensorElement for f64 {
    fn data_type() -> DataType {
        DataType::F64
    }
}

// SAFETY: `f16` is the two-byte IEEE binary16 element of `CUDNN_DATA_HALF`, and every bit pattern is a valid value.
unsafe impl TensorElement for f16 {
    fn data_type() -> DataType {
        DataType::F16
    }
}

// SAFETY: `bf16` is the two-byte bfloat16 element of `CUDNN_DATA_BFLOAT16`, and every bit pattern is a valid value.
unsafe impl TensorElement for bf16 {
    fn data_type() -> DataType {
        DataType::Bf16
    }
}

// SAFETY: `f8e4m3` is one byte holding the E4M3 encoding of `CUDNN_DATA_FP8_E4M3`, and every byte is a valid value.
unsafe impl TensorElement for f8e4m3 {
    fn data_type() -> DataType {
        DataType::F8E4M3
    }
}

// SAFETY: `f8e5m2` is one byte holding the E5M2 encoding of `CUDNN_DATA_FP8_E5M2`, and every byte is a valid value.
unsafe impl TensorElement for f8e5m2 {
    fn data_type() -> DataType {
        DataType::F8E5M2
    }
}

// SAFETY: `f8ue5m3` is one byte holding the unsigned E5M3 encoding of `CUDNN_DATA_FP8_E5M3`, and every byte is a valid value.
unsafe impl TensorElement for f8ue5m3 {
    fn data_type() -> DataType {
        DataType::F8UE5M3
    }
}

// SAFETY: `f8ue8m0` is one byte holding the E8M0 scale encoding of `CUDNN_DATA_FP8_E8M0`, and every byte is a valid value.
unsafe impl TensorElement for f8ue8m0 {
    fn data_type() -> DataType {
        DataType::F8UE8M0
    }
}

// SAFETY: `f4e2m1` is one byte whose low nibble is one E2M1 scalar; packed `CUDNN_DATA_FP4_E2M1` tensors use byte storage, as the trait requires.
unsafe impl TensorElement for f4e2m1 {
    fn data_type() -> DataType {
        DataType::F4E2M1
    }
}

// SAFETY: Rust `bool` is one byte holding 0 or 1, the layout of `CUDNN_DATA_BYTE_BOOLEAN`; cuDNN writes only 0 or 1 to it.
unsafe impl TensorElement for bool {
    fn data_type() -> DataType {
        DataType::ByteBoolean
    }
}

// SAFETY: `i8` is the element of `CUDNN_DATA_INT8`, and four or thirty-two lanes of it are the `INT8x4` and `INT8x32` vector encodings.
unsafe impl TensorElement for i8 {
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

// SAFETY: `u8` is the element of `CUDNN_DATA_UINT8`, and four lanes of it are the `UINT8x4` vector encoding.
unsafe impl TensorElement for u8 {
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

// SAFETY: `i32` is the element of `CUDNN_DATA_INT32`, with the same size, alignment and every bit pattern valid.
unsafe impl TensorElement for i32 {
    fn data_type() -> DataType {
        DataType::I32
    }
}

// SAFETY: `i64` is the element of `CUDNN_DATA_INT64`, with the same size, alignment and every bit pattern valid.
unsafe impl TensorElement for i64 {
    fn data_type() -> DataType {
        DataType::I64
    }
}

// SAFETY: `u32` is the element of `CUDNN_DATA_UINT32`, with the same size, alignment and every bit pattern valid.
unsafe impl TensorElement for u32 {
    fn data_type() -> DataType {
        DataType::U32
    }
}

// SAFETY: `Complex32` is `#[repr(C)]` real and imaginary `f32`s, the interleaved layout of `CUDNN_DATA_COMPLEX_FP32`.
unsafe impl TensorElement for Complex32 {
    fn data_type() -> DataType {
        DataType::ComplexF32
    }
}

// SAFETY: `Complex64` is `#[repr(C)]` real and imaginary `f64`s, the interleaved layout of `CUDNN_DATA_COMPLEX_FP64`.
unsafe impl TensorElement for Complex64 {
    fn data_type() -> DataType {
        DataType::ComplexF64
    }
}

/// Type-level lane width for legacy vectorized tensor encodings.
pub trait DataTypeVectorization: Default + Clone + Copy + Debug + 'static {
    /// Number of scalar lanes in each native vectorized storage unit.
    const WIDTH: usize;
}

/// Selects a four-lane legacy vectorized tensor encoding.
#[derive(Debug, Default, Clone, Copy)]
pub struct Vec4;

impl DataTypeVectorization for Vec4 {
    const WIDTH: usize = 4;
}

/// Selects a thirty-two-lane legacy vectorized tensor encoding.
#[derive(Debug, Default, Clone, Copy)]
pub struct Vec32;

impl DataTypeVectorization for Vec32 {
    const WIDTH: usize = 32;
}

/// A floating tensor element and its native host scaling type.
/// Half and bfloat16 tensors use f32 coefficients; f64 tensors use f64.
pub trait FloatingElement: TensorElement + floating_private::Sealed {
    /// Host scalar type used for native alpha and beta coefficients.
    type Scale: TensorElement;
}

macro_rules! floating_element {
    ($element:ty, $scale:ty) => {
        impl floating_private::Sealed for $element {}
        impl FloatingElement for $element {
            type Scale = $scale;
        }
    };
}

floating_element!(f32, f32);

floating_element!(f64, f64);

floating_element!(f16, f32);

floating_element!(bf16, f32);
