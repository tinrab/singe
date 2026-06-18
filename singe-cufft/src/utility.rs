use crate::error::{Error, Result};
use singe_core::checked_int;

pub fn to_i32(value: impl TryInto<i32>, name: &str) -> Result<i32> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_i64(value: impl TryInto<i64>, name: &str) -> Result<i64> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_usize(value: impl TryInto<usize>, name: &str) -> Result<usize> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_i32_vec<T: TryInto<i32> + Copy>(values: &[T], name: &str) -> Result<Vec<i32>> {
    values
        .iter()
        .copied()
        .map(|value| to_i32(value, name))
        .collect()
}

pub fn to_i64_vec<T: TryInto<i64> + Copy>(values: &[T], name: &str) -> Result<Vec<i64>> {
    values
        .iter()
        .copied()
        .map(|value| to_i64(value, name))
        .collect()
}

pub fn optional_to_i32_vec<T: TryInto<i32> + Copy>(
    values: Option<&[T]>,
    rank: usize,
    name: &str,
) -> Result<Vec<i32>> {
    let Some(values) = values else {
        return Ok(Vec::new());
    };

    if values.len() != rank {
        return Err(Error::LengthMismatch {
            name: name.into(),
            expected: rank,
            actual: values.len(),
        });
    }

    values
        .iter()
        .copied()
        .map(|value| to_i32(value, name))
        .collect()
}

pub fn optional_to_i64_vec<T: TryInto<i64> + Copy>(
    values: Option<&[T]>,
    rank: usize,
    name: &str,
) -> Result<Vec<i64>> {
    let Some(values) = values else {
        return Ok(Vec::new());
    };

    if values.len() != rank {
        return Err(Error::LengthMismatch {
            name: name.into(),
            expected: rank,
            actual: values.len(),
        });
    }

    values
        .iter()
        .copied()
        .map(|value| to_i64(value, name))
        .collect()
}
