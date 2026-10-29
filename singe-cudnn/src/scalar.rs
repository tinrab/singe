pub(crate) mod float;
mod storage;

pub(crate) use storage::ScalarStorage;

use serde::{Deserialize, Serialize};

use singe_cuda::types::{
    Complex32, Complex64, bf16, f4e2m1, f8e4m3, f8e5m2, f8ue5m3, f8ue8m0, f16,
};

use crate::{
    data_type::DataType,
    error::{Error, Result},
};

/// Native values for runtime by-value tensors, compile-time constants and reducer identities.
/// Complex values store the real component before the imaginary component in native memory.
/// Human-readable serialization encodes nonfinite float components as hexadecimal bit
/// strings, retaining infinities and NaN payloads. Finite components remain numbers.
/// Equality compares the encoding and native bytes, including NaN payloads and signed zeros.
/// Four-bit scalars use the low nibble; padding bits do not affect native values or serialization.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ScalarValue {
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 32-bit integer.
    I32(i32),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 64-bit integer.
    I64(i64),
    /// A single bit-packed boolean element.
    Boolean(bool),
    /// A byte-addressed boolean, matching Rust `bool` storage.
    ByteBoolean(bool),
    /// The low four bits contain one signed two's-complement element.
    I4(#[serde(with = "nibble")] u8),
    /// The low four bits contain one unsigned element.
    U4(#[serde(with = "nibble")] u8),
    /// IEEE 754 single-precision floating-point value.
    F32(#[serde(with = "float::f32_bits")] f32),
    /// IEEE 754 double-precision floating-point value.
    F64(#[serde(with = "float::f64_bits")] f64),
    /// Complex value with single-precision real and imaginary components.
    ComplexF32 {
        /// Real component.
        #[serde(with = "float::f32_bits")]
        re: f32,
        /// Imaginary component.
        #[serde(with = "float::f32_bits")]
        im: f32,
    },
    /// Complex value with double-precision real and imaginary components.
    ComplexF64 {
        /// Real component.
        #[serde(with = "float::f64_bits")]
        re: f64,
        /// Imaginary component.
        #[serde(with = "float::f64_bits")]
        im: f64,
    },
    /// Raw IEEE 754 half-precision encoding.
    F16(u16),
    /// Raw 16-bit brain floating-point encoding.
    Bf16(u16),
    /// Raw 8-bit floating-point encoding with 4 exponent bits and 3 mantissa bits.
    F8E4M3(u8),
    /// Raw 8-bit floating-point encoding with 5 exponent bits and 2 mantissa bits.
    F8E5M2(u8),
    /// Unsigned 8-bit floating-point with 5 exponent bits and 3 mantissa bits.
    F8UE5M3(u8),
    /// Raw unsigned 8-bit floating-point encoding with 8 exponent bits and no mantissa bits.
    F8UE8M0(u8),
    /// Raw 4-bit floating-point encoding with 2 exponent bits and 1 mantissa bit, stored in the low nibble.
    F4E2M1(#[serde(with = "nibble")] u8),
}

impl PartialEq for ScalarValue {
    fn eq(&self, other: &Self) -> bool {
        self.data_type() == other.data_type() && self.native_bytes() == other.native_bytes()
    }
}

impl Eq for ScalarValue {}

pub(crate) const SCALAR_STORAGE_BYTES: usize = 16;
const PACKED_SCALAR_MASK: u8 = 0x0f;

impl ScalarValue {
    const fn canonical(self) -> Self {
        match self {
            Self::I4(bits) => Self::I4(bits & PACKED_SCALAR_MASK),
            Self::U4(bits) => Self::U4(bits & PACKED_SCALAR_MASK),
            Self::F4E2M1(bits) => Self::F4E2M1(bits & PACKED_SCALAR_MASK),
            value => value,
        }
    }

    /// Native scalar bytes in host byte order, padded to the largest supported scalar.
    pub(crate) fn native_bytes(self) -> [u8; SCALAR_STORAGE_BYTES] {
        let mut bytes = [0; SCALAR_STORAGE_BYTES];
        match self.canonical() {
            Self::I8(value) => bytes[0] = value.to_ne_bytes()[0],
            Self::U8(value)
            | Self::I4(value)
            | Self::U4(value)
            | Self::F8E4M3(value)
            | Self::F8E5M2(value)
            | Self::F8UE5M3(value)
            | Self::F8UE8M0(value)
            | Self::F4E2M1(value) => bytes[0] = value,
            Self::Boolean(value) | Self::ByteBoolean(value) => bytes[0] = u8::from(value),
            Self::I64(value) => bytes[..8].copy_from_slice(&value.to_ne_bytes()),
            Self::F64(value) => bytes[..8].copy_from_slice(&value.to_ne_bytes()),
            Self::I32(value) => bytes[..4].copy_from_slice(&value.to_ne_bytes()),
            Self::U32(value) => bytes[..4].copy_from_slice(&value.to_ne_bytes()),
            Self::F32(value) => bytes[..4].copy_from_slice(&value.to_ne_bytes()),
            Self::ComplexF32 { re, im } => {
                bytes[..4].copy_from_slice(&re.to_ne_bytes());
                bytes[4..8].copy_from_slice(&im.to_ne_bytes());
            }
            Self::ComplexF64 { re, im } => {
                bytes[..8].copy_from_slice(&re.to_ne_bytes());
                bytes[8..16].copy_from_slice(&im.to_ne_bytes());
            }
            Self::F16(value) | Self::Bf16(value) => {
                bytes[..2].copy_from_slice(&value.to_ne_bytes());
            }
        }
        bytes
    }

    /// Returns the native data type represented by this scalar encoding.
    #[must_use]
    pub const fn data_type(self) -> DataType {
        match self {
            Self::I8(_) => DataType::I8,
            Self::U8(_) => DataType::U8,
            Self::I32(_) => DataType::I32,
            Self::U32(_) => DataType::U32,
            Self::I64(_) => DataType::I64,
            Self::Boolean(_) => DataType::Boolean,
            Self::ByteBoolean(_) => DataType::ByteBoolean,
            Self::I4(_) => DataType::I4,
            Self::U4(_) => DataType::U4,
            Self::F32(_) => DataType::F32,
            Self::F64(_) => DataType::F64,
            Self::ComplexF32 { .. } => DataType::ComplexF32,
            Self::ComplexF64 { .. } => DataType::ComplexF64,
            Self::F16(_) => DataType::F16,
            Self::Bf16(_) => DataType::BF16,
            Self::F8E4M3(_) => DataType::F8E4M3,
            Self::F8E5M2(_) => DataType::F8E5M2,
            Self::F8UE5M3(_) => DataType::F8UE5M3,
            Self::F8UE8M0(_) => DataType::F8UE8M0,
            Self::F4E2M1(_) => DataType::F4E2M1,
        }
    }

    /// The multiplicative identity encoded in `data_type`.
    ///
    /// # Errors
    /// Returns an error for data types without a scalar encoding.
    pub fn one(data_type: DataType) -> Result<Self> {
        Ok(match data_type {
            DataType::I8 => Self::I8(1),
            DataType::U8 => Self::U8(1),
            DataType::I4 => Self::I4(1),
            DataType::U4 => Self::U4(1),
            DataType::Boolean => Self::Boolean(true),
            DataType::ByteBoolean => Self::ByteBoolean(true),
            DataType::I32 => Self::I32(1),
            DataType::U32 => Self::U32(1),
            DataType::I64 => Self::I64(1),
            DataType::F16 => Self::f16_bits(0x3c00),
            DataType::BF16 => Self::bf16_bits(0x3f80),
            DataType::F32 => Self::F32(1.0),
            DataType::F64 => Self::F64(1.0),
            DataType::ComplexF32 => Self::ComplexF32 { re: 1.0, im: 0.0 },
            DataType::ComplexF64 => Self::ComplexF64 { re: 1.0, im: 0.0 },
            DataType::F8E4M3 => Self::F8E4M3(0x38),
            DataType::F8E5M2 => Self::F8E5M2(0x3c),
            DataType::F8UE5M3 => Self::F8UE5M3(0x78),
            DataType::F8UE8M0 => Self::F8UE8M0(0x7f),
            DataType::F4E2M1 => Self::F4E2M1(0x02),
            DataType::FastFloatForFp8 => return Err(Error::UnsupportedDataType),
        })
    }

    /// Returns the diagnostic name of this scalar's native data type.
    #[must_use]
    pub const fn type_name(self) -> &'static str {
        self.data_type().type_name()
    }

    /// Constructs a signed four-bit encoding, ignoring the unused high nibble.
    #[must_use]
    pub const fn i4_bits(bits: u8) -> Self {
        Self::I4(bits).canonical()
    }

    /// Constructs an unsigned four-bit encoding, ignoring the unused high nibble.
    #[must_use]
    pub const fn u4_bits(bits: u8) -> Self {
        Self::U4(bits).canonical()
    }

    /// Constructs a half-precision scalar from its raw IEEE 754 bits.
    #[must_use]
    pub const fn f16_bits(bits: u16) -> Self {
        Self::F16(bits)
    }

    /// Constructs a brain floating-point scalar from its raw 16-bit encoding.
    #[must_use]
    pub const fn bf16_bits(bits: u16) -> Self {
        Self::Bf16(bits)
    }

    /// Constructs an E4M3 scalar from its raw 8-bit encoding.
    #[must_use]
    pub const fn f8e4m3_bits(bits: u8) -> Self {
        Self::F8E4M3(bits)
    }

    /// Constructs an E5M2 scalar from its raw 8-bit encoding.
    #[must_use]
    pub const fn f8e5m2_bits(bits: u8) -> Self {
        Self::F8E5M2(bits)
    }

    /// Constructs an unsigned E5M3 scalar from its raw bits.
    #[must_use]
    pub const fn f8ue5m3_bits(bits: u8) -> Self {
        Self::F8UE5M3(bits)
    }

    /// Constructs an unsigned E8M0 scalar from its raw 8-bit encoding.
    #[must_use]
    pub const fn f8ue8m0_bits(bits: u8) -> Self {
        Self::F8UE8M0(bits)
    }

    /// Constructs an FP4 encoding, ignoring the unused high nibble.
    #[must_use]
    pub const fn f4e2m1_bits(bits: u8) -> Self {
        Self::F4E2M1(bits).canonical()
    }

    /// Preserves the native bits of an E4M3 value in a scalar.
    #[must_use]
    pub const fn from_f8e4m3(value: f8e4m3) -> Self {
        Self::F8E4M3(value.to_bits())
    }

    /// Preserves the native bits of an E5M2 value in a scalar.
    #[must_use]
    pub const fn from_f8e5m2(value: f8e5m2) -> Self {
        Self::F8E5M2(value.to_bits())
    }

    /// Constructs an unsigned E5M3 scalar.
    #[must_use]
    pub const fn from_f8ue5m3(value: f8ue5m3) -> Self {
        Self::F8UE5M3(value.to_bits())
    }

    /// Preserves the native bits of an unsigned E8M0 value in a scalar.
    #[must_use]
    pub const fn from_f8ue8m0(value: f8ue8m0) -> Self {
        Self::F8UE8M0(value.to_bits())
    }

    /// Constructs an E2M1 scalar from the low four bits of the supplied value.
    #[must_use]
    pub const fn from_f4e2m1(value: f4e2m1) -> Self {
        Self::f4e2m1_bits(value.to_bits())
    }
}

mod nibble {
    use serde::{Deserialize, Deserializer, Serializer};

    use super::PACKED_SCALAR_MASK;

    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "Serde custom serializers require a borrowed field value"
    )]
    pub fn serialize<S: Serializer>(bits: &u8, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u8(*bits & PACKED_SCALAR_MASK)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u8, D::Error> {
        Ok(u8::deserialize(deserializer)? & PACKED_SCALAR_MASK)
    }
}

macro_rules! impl_scalar_from {
    ($($ty:ty => $convert:expr),* $(,)?) => {
        $(impl From<$ty> for ScalarValue {
            fn from(value: $ty) -> Self { ($convert)(value) }
        })*
    };
}

impl_scalar_from! {
    i8 => ScalarValue::I8,
    u8 => ScalarValue::U8,
    i32 => ScalarValue::I32,
    u32 => ScalarValue::U32,
    i64 => ScalarValue::I64,
    bool => ScalarValue::ByteBoolean,
    f32 => ScalarValue::F32,
    f64 => ScalarValue::F64,
    f16 => |value: f16| ScalarValue::F16(value.to_bits()),
    bf16 => |value: bf16| ScalarValue::Bf16(value.to_bits()),
    f8e4m3 => ScalarValue::from_f8e4m3,
    f8e5m2 => ScalarValue::from_f8e5m2,
    f8ue5m3 => ScalarValue::from_f8ue5m3,
    f8ue8m0 => ScalarValue::from_f8ue8m0,
    f4e2m1 => ScalarValue::from_f4e2m1,
    Complex32 => |value: Complex32| ScalarValue::ComplexF32 { re: value.re, im: value.im },
    Complex64 => |value: Complex64| ScalarValue::ComplexF64 { re: value.re, im: value.im },
}

/// Rounds a scalar coefficient to the precision used by non-f64 native operations.
#[expect(
    clippy::cast_possible_truncation,
    reason = "native f32 computation requires rounding the caller's f64 coefficient"
)]
pub(crate) fn native_f32(value: f64) -> f32 {
    value as f32
}
