use crate::error::{Error, Result};
use singe_core::checked_int;

pub fn to_i32(value: impl TryInto<i32>, name: &str) -> Result<i32> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_i64(value: impl TryInto<i64>, name: &str) -> Result<i64> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_u64(value: impl TryInto<u64>, name: &str) -> Result<u64> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_i32_vec<T: TryInto<i32> + Copy>(values: &[T], name: &str) -> Result<Vec<i32>> {
    values
        .iter()
        .copied()
        .map(|value| to_i32(value, name))
        .collect()
}

pub fn to_usize(bytes: impl TryInto<usize>, name: &str) -> Result<usize> {
    checked_int(bytes, name, |name| Error::OutOfRange { name })
}

macro_rules! check_range {
    ($name:expr, $expr:expr $(,)?) => {
        if !{ $expr } {
            Err(Error::OutOfRange {
                name: { $name }.into(),
            })
        } else {
            Ok(())
        }
    };
}

pub(crate) use check_range;
