use crate::error::{Error, Result};
use singe_core::checked_int;

pub fn to_i64(value: impl TryInto<i64>, name: &str) -> Result<i64> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_usize(value: impl TryInto<usize>, name: &str) -> Result<usize> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_i32(value: impl TryInto<i32>, name: &str) -> Result<i32> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}
