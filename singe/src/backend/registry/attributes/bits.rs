use std::hash::{Hash, Hasher};

use crate::{
    backend::registry::attributes::PackedFormat, data_type::DataType, device::Device,
    scalar::Scalar, shape::Shape,
};

/// Equality and hashing that compare floating values by their bits.
///
/// Attributes key graph deduplication and fingerprints, so two attributes are equal exactly when every field has the same representation, including NaN payloads and zero signs.
pub trait BitHash {
    /// Feeds this value's bits to `state`.
    fn bit_hash<H: Hasher>(&self, state: &mut H);

    /// Returns whether this value and `other` have the same bits.
    fn bit_eq(&self, other: &Self) -> bool;
}

macro_rules! exact_bit_hash {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl BitHash for $ty {
                fn bit_hash<H: Hasher>(&self, state: &mut H) {
                    self.hash(state);
                }

                fn bit_eq(&self, other: &Self) -> bool {
                    self == other
                }
            }
        )+
    };
}

exact_bit_hash!(
    bool,
    i64,
    u64,
    usize,
    String,
    DataType,
    Device,
    Shape,
    PackedFormat
);

impl BitHash for f32 {
    fn bit_hash<H: Hasher>(&self, state: &mut H) {
        self.to_bits().hash(state);
    }

    fn bit_eq(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl BitHash for f64 {
    fn bit_hash<H: Hasher>(&self, state: &mut H) {
        self.to_bits().hash(state);
    }

    fn bit_eq(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl BitHash for Scalar {
    fn bit_hash<H: Hasher>(&self, state: &mut H) {
        match *self {
            Self::Bool(value) => (0_u8, u64::from(value)).hash(state),
            Self::Signed(value) => (1_u8, value.cast_unsigned()).hash(state),
            Self::Unsigned(value) => (2_u8, value).hash(state),
            Self::Float(value) => (3_u8, value.to_bits()).hash(state),
        }
    }

    fn bit_eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Bool(lhs), Self::Bool(rhs)) => lhs == rhs,
            (Self::Signed(lhs), Self::Signed(rhs)) => lhs == rhs,
            (Self::Unsigned(lhs), Self::Unsigned(rhs)) => lhs == rhs,
            (Self::Float(lhs), Self::Float(rhs)) => lhs.to_bits() == rhs.to_bits(),
            _ => false,
        }
    }
}

impl<T: BitHash> BitHash for Option<T> {
    fn bit_hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Some(value) => {
                true.hash(state);
                value.bit_hash(state);
            }
            None => false.hash(state),
        }
    }

    fn bit_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Some(lhs), Some(rhs)) => lhs.bit_eq(rhs),
            (None, None) => true,
            _ => false,
        }
    }
}

impl<T: BitHash> BitHash for Vec<T> {
    fn bit_hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for value in self {
            value.bit_hash(state);
        }
    }

    fn bit_eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|(lhs, rhs)| lhs.bit_eq(rhs))
    }
}

macro_rules! tuple_bit_hash {
    ($(($($name:ident $index:tt),+)),+ $(,)?) => {
        $(
            impl<$($name: BitHash),+> BitHash for ($($name,)+) {
                fn bit_hash<H: Hasher>(&self, state: &mut H) {
                    $(self.$index.bit_hash(state);)+
                }

                fn bit_eq(&self, other: &Self) -> bool {
                    $(self.$index.bit_eq(&other.$index))&&+
                }
            }
        )+
    };
}

tuple_bit_hash!((A 0, B 1), (A 0, B 1, C 2), (A 0, B 1, C 2, D 3));
