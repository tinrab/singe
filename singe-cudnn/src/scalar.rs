use serde::{Deserialize, Serialize};
use singe_cuda::types::*;

use crate::data_type::DataType;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ScalarValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F16(u16),
    Bf16(u16),
    F8E4M3(u8),
    F8E5M2(u8),
    F8UE8M0(u8),
    F4E2M1(u8),
}

impl ScalarValue {
    pub const fn data_type(self) -> DataType {
        match self {
            Self::I32(_) => DataType::I32,
            Self::I64(_) => DataType::I64,
            Self::F32(_) => DataType::F32,
            Self::F16(_) => DataType::F16,
            Self::Bf16(_) => DataType::BF16,
            Self::F8E4M3(_) => DataType::F8E4M3,
            Self::F8E5M2(_) => DataType::F8E5M2,
            Self::F8UE8M0(_) => DataType::F8E8M0,
            Self::F4E2M1(_) => DataType::F4E2M1,
        }
    }

    pub const fn type_name(self) -> &'static str {
        match self {
            Self::I32(_) => "i32",
            Self::I64(_) => "i64",
            Self::F32(_) => "f32",
            Self::F16(_) => "f16",
            Self::Bf16(_) => "bf16",
            Self::F8E4M3(_) => "f8e4m3",
            Self::F8E5M2(_) => "f8e5m2",
            Self::F8UE8M0(_) => "f8ue8m0",
            Self::F4E2M1(_) => "f4e2m1",
        }
    }

    pub const fn f16_bits(bits: u16) -> Self {
        Self::F16(bits)
    }

    pub const fn bf16_bits(bits: u16) -> Self {
        Self::Bf16(bits)
    }

    pub const fn f8e4m3_bits(bits: u8) -> Self {
        Self::F8E4M3(bits)
    }

    pub const fn f8e5m2_bits(bits: u8) -> Self {
        Self::F8E5M2(bits)
    }

    pub const fn f8ue8m0_bits(bits: u8) -> Self {
        Self::F8UE8M0(bits)
    }

    pub const fn f4e2m1_bits(bits: u8) -> Self {
        Self::F4E2M1(bits)
    }

    pub const fn from_f8e4m3(value: f8e4m3) -> Self {
        Self::F8E4M3(value.to_bits())
    }

    pub const fn from_f8e5m2(value: f8e5m2) -> Self {
        Self::F8E5M2(value.to_bits())
    }

    pub const fn from_f8ue8m0(value: f8ue8m0) -> Self {
        Self::F8UE8M0(value.to_bits())
    }

    pub const fn from_f4e2m1(value: f4e2m1) -> Self {
        Self::F4E2M1(value.to_bits())
    }
}
