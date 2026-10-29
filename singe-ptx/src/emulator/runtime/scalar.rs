use half::{bf16, f16};

use crate::{
    ast::ScalarType,
    emulator::runtime::buffer::{Argument, KernelArg},
};

pub mod sealed {
    use crate::ast::ScalarType;

    pub trait Scalar: Copy + Send + Sync + 'static {
        const TYPE: ScalarType;
        const SIZE: usize;

        fn encode(self, bytes: &mut Vec<u8>);

        fn decode(bytes: &[u8]) -> Self;
    }
}

/// Element type of a [`Buffer`](super::Buffer), transferred as little-endian bytes.
///
/// Implemented for `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`, `f64`,
/// [`F16`](crate::emulator::F16), and [`Bf16`](crate::emulator::Bf16). This trait is
/// sealed.
pub trait PtxScalar: sealed::Scalar {}

macro_rules! scalar {
    ($($ty:ty => $ptx:ident),* $(,)?) => { $(
        impl sealed::Scalar for $ty {
            const TYPE: ScalarType = ScalarType::$ptx;
            const SIZE: usize = size_of::<Self>();
            fn encode(self, bytes: &mut Vec<u8>) { bytes.extend_from_slice(&self.to_le_bytes()); }
            fn decode(bytes: &[u8]) -> Self {
                let mut value = [0; size_of::<Self>()];
                value.copy_from_slice(bytes);
                Self::from_le_bytes(value)
            }
        }

        impl PtxScalar for $ty {}
        impl From<$ty> for KernelArg {
            fn from(value: $ty) -> Self {
                let mut bytes = Vec::with_capacity(size_of::<$ty>());
                sealed::Scalar::encode(value, &mut bytes);
                Self { value: Argument::Scalar(ScalarType::$ptx, bytes) }
            }
        }
    )* };
}

scalar! { i8 => S8, u8 => U8, i16 => S16, u16 => U16, i32 => S32, u32 => U32, i64 => S64, u64 => U64, f32 => F32, f64 => F64 }

macro_rules! half_scalar {
    ($ty:ty, $ptx:ident) => {
        impl sealed::Scalar for $ty {
            const TYPE: ScalarType = ScalarType::$ptx;
            const SIZE: usize = 2;
            fn encode(self, bytes: &mut Vec<u8>) {
                bytes.extend_from_slice(&self.to_bits().to_le_bytes());
            }

            fn decode(bytes: &[u8]) -> Self {
                Self::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]))
            }
        }

        impl PtxScalar for $ty {}
        impl From<$ty> for KernelArg {
            fn from(value: $ty) -> Self {
                Self {
                    value: Argument::Scalar(
                        ScalarType::$ptx,
                        value.to_bits().to_le_bytes().to_vec(),
                    ),
                }
            }
        }
    };
}

half_scalar!(f16, F16);

half_scalar!(bf16, Bf16);
