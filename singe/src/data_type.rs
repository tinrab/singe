use std::fmt::{self, Display, Formatter};

pub use half::{bf16, f16};
use num_enum::{IntoPrimitive, TryFromPrimitive};

use crate::error::{Error, Result};

/// Element data type carried by tensors and storage allocations.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, TryFromPrimitive, IntoPrimitive,
)]
#[repr(u8)]
pub enum DataType {
    /// Boolean values.
    Bool = 0,
    /// Unsigned 8-bit integer.
    U8 = 1,
    /// Signed 8-bit integer.
    I8 = 2,
    /// Unsigned 16-bit integer.
    U16 = 3,
    /// Signed 16-bit integer.
    I16 = 4,
    /// Unsigned 32-bit integer.
    U32 = 5,
    /// Signed 32-bit integer.
    I32 = 6,
    /// Unsigned 64-bit integer.
    U64 = 7,
    /// Signed 64-bit integer.
    I64 = 8,
    /// IEEE half precision float.
    F16 = 9,
    /// bfloat16 floating-point value.
    Bf16 = 10,
    /// IEEE single precision float.
    F32 = 11,
    /// IEEE double precision float.
    F64 = 12,
    /// FP8 E4M3 low-precision float.
    F8E4M3 = 13,
    /// FP8 E5M2 low-precision float.
    F8E5M2 = 14,
    /// FP4 E2M1 low-precision float.
    F4E2M1 = 15,
}

impl DataType {
    /// Returns the byte width of one element.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 | Self::F4E2M1 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// Returns whether this data type is floating-point.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F16
                | Self::Bf16
                | Self::F32
                | Self::F64
                | Self::F8E4M3
                | Self::F8E5M2
                | Self::F4E2M1
        )
    }

    /// Returns whether this data type is an integer type.
    #[must_use]
    pub const fn is_int(self) -> bool {
        matches!(
            self,
            Self::U8
                | Self::I8
                | Self::U16
                | Self::I16
                | Self::U32
                | Self::I32
                | Self::U64
                | Self::I64
        )
    }

    /// Returns whether values of this type are signed.
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::F16
                | Self::Bf16
                | Self::F32
                | Self::F64
                | Self::F8E4M3
                | Self::F8E5M2
                | Self::F4E2M1
        )
    }

    /// Returns whether this is a low-precision floating-point format.
    #[must_use]
    pub const fn is_lowp(self) -> bool {
        matches!(
            self,
            Self::F16 | Self::Bf16 | Self::F8E4M3 | Self::F8E5M2 | Self::F4E2M1
        )
    }

    /// Orders data types for promotion and in-place storage: Boolean, then integer, then floating point.
    #[must_use]
    pub const fn category(self) -> u8 {
        if self.is_float() {
            2
        } else if matches!(self, Self::Bool) {
            0
        } else {
            1
        }
    }

    /// Returns the data type an operation on operands of `self` and `other` computes in, by `PyTorch`'s rules.
    ///
    /// A zero-dimensional operand defers to a dimensioned one of the same or a higher category; otherwise the two promote with [`Self::promote_types`].
    ///
    /// # Errors
    ///
    /// Returns an error if the two types have no common type.
    pub fn promote(self, self_rank: i64, other: Self, other_rank: i64) -> Result<Self> {
        match (self_rank > 0, other_rank > 0) {
            (true, false) if other.category() <= self.category() => Ok(self),
            (false, true) if self.category() <= other.category() => Ok(other),
            // A zero-dimensional operand of a higher category than the dimensioned one decides the type.
            (true, false) => Ok(other),
            (false, true) => Ok(self),
            _ => self.promote_types(other),
        }
    }

    /// Returns the smallest data type both `self` and `other` convert to without losing their category, as `torch.promote_types` does.
    ///
    /// # Errors
    ///
    /// Returns an error if the two types have no common type: an FP8 or FP4 type with another type, or `U16`, `U32`, or `U64` with another integer or Boolean type.
    pub fn promote_types(self, other: Self) -> Result<Self> {
        if self == other {
            return Ok(self);
        }

        let byte_float =
            |data_type: Self| matches!(data_type, Self::F8E4M3 | Self::F8E5M2 | Self::F4E2M1);
        if byte_float(self) || byte_float(other) {
            return Err(Error::unsupported_data_type(
                "promotion of an fp8 or fp4 type",
            ));
        }

        match (self.is_float(), other.is_float()) {
            (true, true) => {
                return Ok(if self == Self::F64 || other == Self::F64 {
                    Self::F64
                } else {
                    // Every other pair of distinct floating types is F32 with one of them, or F16 with BF16.
                    Self::F32
                });
            }
            (true, false) => return Ok(self),
            (false, true) => return Ok(other),
            (false, false) => {}
        }

        let wide_unsigned =
            |data_type: Self| matches!(data_type, Self::U16 | Self::U32 | Self::U64);
        if wide_unsigned(self) || wide_unsigned(other) {
            return Err(Error::unsupported_data_type(
                "promotion of u16, u32, or u64 with another type",
            ));
        }
        Ok(match (self, other) {
            (Self::Bool, integer) | (integer, Self::Bool) => integer,
            (Self::U8, Self::I8) | (Self::I8, Self::U8) => Self::I16,
            (Self::U8, signed) | (signed, Self::U8) => signed,
            _ if self.size_bytes() >= other.size_bytes() => self,
            _ => other,
        })
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool => write!(f, "bool"),
            Self::U8 => write!(f, "u8"),
            Self::I8 => write!(f, "i8"),
            Self::U16 => write!(f, "u16"),
            Self::I16 => write!(f, "i16"),
            Self::U32 => write!(f, "u32"),
            Self::I32 => write!(f, "i32"),
            Self::U64 => write!(f, "u64"),
            Self::I64 => write!(f, "i64"),
            Self::F16 => write!(f, "f16"),
            Self::Bf16 => write!(f, "bf16"),
            Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"),
            Self::F8E4M3 => write!(f, "f8e4m3"),
            Self::F8E5M2 => write!(f, "f8e5m2"),
            Self::F4E2M1 => write!(f, "f4e2m1"),
        }
    }
}
