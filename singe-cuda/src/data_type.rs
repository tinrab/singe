use std::fmt::Debug;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda_sys::library_types::cudaDataType_t;

use crate::types::{
    Complex32, Complex64, bf16, f4e2m1, f6e2m3, f6e3m2, f8e4m3, f8e5m2, f8ue8m0, f16,
};

/// Rust wrapper for CUDA's data type enum.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum DataType {
    /// 16-bit real half precision floating-point (IEEE 754-2008 binary16).
    F16 = cudaDataType_t::CUDA_R_16F as _,
    /// 32-bit complex (2x16-bit half precision floats).
    ComplexF16 = cudaDataType_t::CUDA_C_16F as _,
    /// 16-bit real bfloat16 floating-point.
    Bf16 = cudaDataType_t::CUDA_R_16BF as _,
    /// 32-bit complex (2x16-bit bfloat16 floats).
    ComplexBf16 = cudaDataType_t::CUDA_C_16BF as _,
    /// 32-bit real single precision floating-point (IEEE 754 binary32).
    F32 = cudaDataType_t::CUDA_R_32F as _,
    /// 64-bit complex (2x32-bit single precision floats).
    ComplexF32 = cudaDataType_t::CUDA_C_32F as _,
    /// 64-bit real double precision floating-point (IEEE 754 binary64).
    F64 = cudaDataType_t::CUDA_R_64F as _,
    /// 128-bit complex (2x64-bit double precision floats).
    ComplexF64 = cudaDataType_t::CUDA_C_64F as _,
    /// 4-bit real signed integer.
    I4 = cudaDataType_t::CUDA_R_4I as _,
    /// 8-bit complex (2x4-bit signed integers).
    ComplexI4 = cudaDataType_t::CUDA_C_4I as _,
    /// 4-bit real unsigned integer.
    U4 = cudaDataType_t::CUDA_R_4U as _,
    /// 8-bit complex (2x4-bit unsigned integers).
    ComplexU4 = cudaDataType_t::CUDA_C_4U as _,
    /// 8-bit real signed integer.
    I8 = cudaDataType_t::CUDA_R_8I as _,
    /// 16-bit complex (2x8-bit signed integers).
    ComplexI8 = cudaDataType_t::CUDA_C_8I as _,
    /// 8-bit real unsigned integer.
    U8 = cudaDataType_t::CUDA_R_8U as _,
    /// 16-bit complex (2x8-bit unsigned integers).
    ComplexU8 = cudaDataType_t::CUDA_C_8U as _,
    /// 16-bit real signed integer.
    I16 = cudaDataType_t::CUDA_R_16I as _,
    /// 32-bit complex (2x16-bit signed integers).
    ComplexI16 = cudaDataType_t::CUDA_C_16I as _,
    /// 16-bit real unsigned integer.
    U16 = cudaDataType_t::CUDA_R_16U as _,
    /// 32-bit complex (2x16-bit unsigned integers).
    ComplexU16 = cudaDataType_t::CUDA_C_16U as _,
    /// 32-bit real signed integer.
    I32 = cudaDataType_t::CUDA_R_32I as _,
    /// 64-bit complex (2x32-bit signed integers).
    ComplexI32 = cudaDataType_t::CUDA_C_32I as _,
    /// 32-bit real unsigned integer.
    U32 = cudaDataType_t::CUDA_R_32U as _,
    /// 64-bit complex (2x32-bit unsigned integers).
    ComplexU32 = cudaDataType_t::CUDA_C_32U as _,
    /// 64-bit real signed integer.
    I64 = cudaDataType_t::CUDA_R_64I as _,
    /// 128-bit complex (2x64-bit signed integers).
    ComplexI64 = cudaDataType_t::CUDA_C_64I as _,
    /// 64-bit real unsigned integer.
    U64 = cudaDataType_t::CUDA_R_64U as _,
    /// 128-bit complex (2x64-bit unsigned integers).
    ComplexU64 = cudaDataType_t::CUDA_C_64U as _,
    /// 8-bit real floating point in E4M3 format.
    F8E4M3 = cudaDataType_t::CUDA_R_8F_E4M3 as _,
    /// 8-bit real floating point in E5M2 format.
    F8E5M2 = cudaDataType_t::CUDA_R_8F_E5M2 as _,
    /// 8-bit real floating point in E8M0 format (unsigned exponent, zero mantissa bits).
    F8UE8M0 = cudaDataType_t::CUDA_R_8F_UE8M0 as _,
    /// 6-bit real floating point in E2M3 format (2-bit exponent, 3-bit mantissa).
    F6E2M3 = cudaDataType_t::CUDA_R_6F_E2M3 as _,
    /// 6-bit real floating point in E3M2 format (3-bit exponent, 2-bit mantissa).
    F6E3M2 = cudaDataType_t::CUDA_R_6F_E3M2 as _,
    /// 4-bit real floating point in E2M1 format (2-bit exponent, 1-bit mantissa).
    F4E2M1 = cudaDataType_t::CUDA_R_4F_E2M1 as _,
}

impl_enum_conversion!(DataType, cudaDataType_t);

impl DataType {
    pub const fn size_of(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 | Self::I16 | Self::U16 => 2,
            Self::ComplexF16
            | Self::ComplexBf16
            | Self::F32
            | Self::I32
            | Self::U32
            | Self::I8
            | Self::U8 => 4,
            Self::ComplexF32 | Self::F64 | Self::I64 | Self::U64 => 8,
            Self::ComplexF64 => 16,
            Self::I4 | Self::U4 | Self::F4E2M1 => 1,
            Self::ComplexI4 | Self::ComplexU4 => 1,
            Self::F8E4M3 | Self::F8E5M2 | Self::F8UE8M0 => 1,
            Self::F6E2M3 | Self::F6E3M2 => 1,
            Self::ComplexI8 | Self::ComplexU8 => 2,
            Self::ComplexI16 | Self::ComplexU16 => 4,
            Self::ComplexI32 | Self::ComplexU32 => 8,
            Self::ComplexI64 | Self::ComplexU64 => 16,
        }
    }
}

impl_enum_display!(DataType, {
    Self::F16 => "CUDA_R_16F",
    Self::ComplexF16 => "CUDA_C_16F",
    Self::Bf16 => "CUDA_R_16BF",
    Self::ComplexBf16 => "CUDA_C_16BF",
    Self::F32 => "CUDA_R_32F",
    Self::ComplexF32 => "CUDA_C_32F",
    Self::F64 => "CUDA_R_64F",
    Self::ComplexF64 => "CUDA_C_64F",
    Self::I4 => "CUDA_R_4I",
    Self::ComplexI4 => "CUDA_C_4I",
    Self::U4 => "CUDA_R_4U",
    Self::ComplexU4 => "CUDA_C_4U",
    Self::I8 => "CUDA_R_8I",
    Self::ComplexI8 => "CUDA_C_8I",
    Self::U8 => "CUDA_R_8U",
    Self::ComplexU8 => "CUDA_C_8U",
    Self::I16 => "CUDA_R_16I",
    Self::ComplexI16 => "CUDA_C_16I",
    Self::U16 => "CUDA_R_16U",
    Self::ComplexU16 => "CUDA_C_16U",
    Self::I32 => "CUDA_R_32I",
    Self::ComplexI32 => "CUDA_C_32I",
    Self::U32 => "CUDA_R_32U",
    Self::ComplexU32 => "CUDA_C_32U",
    Self::I64 => "CUDA_R_64I",
    Self::ComplexI64 => "CUDA_C_64I",
    Self::U64 => "CUDA_R_64U",
    Self::ComplexU64 => "CUDA_C_64U",
    Self::F8E4M3 => "CUDA_R_8F_E4M3",
    Self::F8E5M2 => "CUDA_R_8F_E5M2",
    Self::F8UE8M0 => "CUDA_R_8F_UE8M0",
    Self::F6E2M3 => "CUDA_R_6F_E2M3",
    Self::F6E3M2 => "CUDA_R_6F_E3M2",
    Self::F4E2M1 => "CUDA_R_4F_E2M1",
});

pub trait DataTypeLike: Clone + Copy + Default + Debug + 'static {
    fn data_type() -> DataType;

    fn is_complex() -> bool;

    fn rust_type_name() -> &'static str;
}

macro_rules! impl_data_type {
    ($ty:ty, $data_type:ident, $is_complex:expr) => {
        impl DataTypeLike for $ty {
            fn data_type() -> DataType {
                DataType::$data_type
            }

            fn is_complex() -> bool {
                $is_complex
            }

            fn rust_type_name() -> &'static str {
                // TODO: `type_name::<T>()`?
                stringify!($ty)
            }
        }
    };
}

impl_data_type!(f32, F32, false);
impl_data_type!(f64, F64, false);
impl_data_type!(f16, F16, false);
impl_data_type!(bf16, Bf16, false);
impl_data_type!(f8e4m3, F8E4M3, false);
impl_data_type!(f8e5m2, F8E5M2, false);
impl_data_type!(f8ue8m0, F8UE8M0, false);
impl_data_type!(f6e2m3, F6E2M3, false);
impl_data_type!(f6e3m2, F6E3M2, false);
impl_data_type!(f4e2m1, F4E2M1, false);
impl_data_type!(i8, I8, false);
impl_data_type!(u8, U8, false);
impl_data_type!(i32, I32, false);
impl_data_type!(u32, U32, false);
impl_data_type!(Complex32, ComplexF32, true);
impl_data_type!(Complex64, ComplexF64, true);

#[cfg(test)]
mod tests {
    use super::{
        DataType, DataTypeLike, bf16, f4e2m1, f6e2m3, f6e3m2, f8e4m3, f8e5m2, f8ue8m0, f16,
    };

    #[test]
    fn test_low_precision_module_reexports_expected_types() {
        let _ = f16::from_f32(1.0);
        let _ = bf16::from_f32(1.0);
        let _ = f8e4m3::from_bits(0);
        let _ = f8e5m2::from_bits(0);
        let _ = f8ue8m0::from_bits(0);
        let _ = f6e2m3::from_bits(0);
        let _ = f6e3m2::from_bits(0);
        let _ = f4e2m1::from_bits(0);
    }

    #[test]
    fn test_data_type_like_maps_low_precision_storage_types() {
        assert_eq!(f16::data_type(), DataType::F16);
        assert_eq!(bf16::data_type(), DataType::Bf16);
        assert_eq!(f8e4m3::data_type(), DataType::F8E4M3);
        assert_eq!(f8e5m2::data_type(), DataType::F8E5M2);
        assert_eq!(f8ue8m0::data_type(), DataType::F8UE8M0);
        assert_eq!(f6e2m3::data_type(), DataType::F6E2M3);
        assert_eq!(f6e3m2::data_type(), DataType::F6E3M2);
        assert_eq!(f4e2m1::data_type(), DataType::F4E2M1);
    }
}
