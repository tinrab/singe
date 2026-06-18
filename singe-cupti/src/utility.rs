use std::mem::size_of;

use singe_core::checked_int;

use crate::error::{Error, Result};

pub fn to_u64(value: impl TryInto<u64>, name: &str) -> Result<u64> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_u32(value: impl TryInto<u32>, name: &str) -> Result<u32> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_usize(value: impl TryInto<usize>, name: &str) -> Result<usize> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn checked_byte_len<T>(count: usize, name: &str) -> Result<usize> {
    count
        .checked_mul(size_of::<T>())
        .ok_or(Error::OutOfRange { name: name.into() })
}
