use crate::error::{Error, Result};
use singe_core::checked_int;

pub fn to_u32(value: usize, name: &str) -> Result<u32> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_u64(value: usize, name: &str) -> Result<u64> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}
