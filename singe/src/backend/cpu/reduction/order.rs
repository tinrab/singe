use std::cmp::Ordering;

use crate::{data_type::DataType, error::Result, tensor::Tensor};

// Every element read exactly for comparison: integers, Bools, and `U64` as `i128`, and floating values as `F64`, which holds every narrower float exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Key {
    Int(i128),
    Float(f64),
}

impl Key {
    pub fn is_nan(self) -> bool {
        matches!(self, Self::Float(value) if value.is_nan())
    }

    // NaN orders above every other value, as PyTorch's sorts place it.
    pub fn compare(self, other: Self) -> Ordering {
        match (self, other) {
            (Self::Int(lhs), Self::Int(rhs)) => lhs.cmp(&rhs),
            (Self::Float(lhs), Self::Float(rhs)) => match (lhs.is_nan(), rhs.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => lhs.partial_cmp(&rhs).unwrap_or(Ordering::Equal),
            },
            (Self::Int(lhs), Self::Float(rhs)) => Self::Float(lhs as f64).compare(Self::Float(rhs)),
            (Self::Float(lhs), Self::Int(rhs)) => Self::Float(lhs).compare(Self::Float(rhs as f64)),
        }
    }
}

// Returns the tensor's elements as keys, in row-major order.
pub fn keys(input: &Tensor) -> Result<Vec<Key>> {
    let input = input.contiguous()?;
    Ok(match input.data_type() {
        DataType::U64 => input
            .to_vec::<u64>()?
            .into_iter()
            .map(|value| Key::Int(i128::from(value)))
            .collect(),
        data_type if data_type.is_float() => input
            .cast(DataType::F64)?
            .to_vec::<f64>()?
            .into_iter()
            .map(Key::Float)
            .collect(),
        _ => input
            .cast(DataType::I64)?
            .to_vec::<i64>()?
            .into_iter()
            .map(|value| Key::Int(i128::from(value)))
            .collect(),
    })
}
