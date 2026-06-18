use crate::{
    error::{Error, Result},
    types::Mode,
};
use singe_core::checked_int;

pub fn to_i32(value: u32, name: &str) -> Result<i32> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_u32(value: usize, name: &str) -> Result<u32> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_i64(value: impl TryInto<i64>, name: &str) -> Result<i64> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_usize(value: u64, name: &str) -> Result<usize> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_i64_vec<T: TryInto<i64> + Copy>(values: &[T], name: &str) -> Result<Vec<i64>> {
    values
        .iter()
        .copied()
        .map(|value| to_i64(value, name))
        .collect()
}

pub fn modes_to_i32_vec(modes: &[Mode]) -> Vec<i32> {
    modes.iter().map(|mode| mode.as_raw()).collect()
}
