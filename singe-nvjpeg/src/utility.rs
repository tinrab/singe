use singe_core::{checked_arithmetic, checked_conversions};
use singe_cuda::stream::{Stream as CudaStream, StreamBinding};

use crate::{
    error::{Error, Result},
    image::Size,
};

checked_conversions!(Error {
    to_u32 -> u32,
    to_i32 -> i32,
    to_usize -> usize,
    to_u64 -> u64,
});

pub fn to_positive_i32(value: impl TryInto<i32>, name: &str) -> Result<i32> {
    let value = to_i32(value, name)?;
    if value <= 0 {
        return Err(Error::OutOfRange {
            name: name.to_owned().into(),
        });
    }
    Ok(value)
}

pub fn to_non_zero_u64(value: impl TryInto<u64>, name: &str) -> Result<u64> {
    let value = to_u64(value, name)?;
    if value == 0 {
        return Err(Error::Empty {
            name: name.to_owned().into(),
        });
    }
    Ok(value)
}

pub fn checked_row_bytes(size: Size, bytes_per_pixel: usize) -> Result<usize> {
    checked_mul(size.width as usize, bytes_per_pixel, "row bytes")
}

checked_arithmetic!(Error { checked_mul });

pub const fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

pub fn ceil_to_multiple_u32(value: u32, multiple: u32, name: &str) -> Result<u32> {
    ceil_div_u32(value, multiple)
        .checked_mul(multiple)
        .ok_or(Error::OutOfRange {
            name: name.to_owned().into(),
        })
}

/// Completes borrowed codec work before releasing participants on every path.
pub fn complete<T>(stream: &CudaStream, execute: impl FnOnce() -> Result<T>) -> Result<T> {
    StreamBinding::Borrowed(stream.to_borrowed()).sync_scope(execute)
}
