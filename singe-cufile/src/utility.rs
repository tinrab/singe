use std::ptr;

use singe_core::{checked_arithmetic, checked_conversions};
use singe_cufile_sys as sys;

use crate::{
    error::{Error, Result},
    types::StatisticsLevel,
};

checked_conversions!(Error {
    to_i64 -> i64,
    to_u64 -> u64,
    to_i32 -> i32,
    to_u32 -> u32,
    to_usize -> usize,
    to_statistics_level -> StatisticsLevel,
});

checked_arithmetic!(Error { checked_byte_len });

pub fn checked_io_size(buffer_size: usize, size: usize, buffer_offset: u64) -> Result<usize> {
    let offset = to_usize(buffer_offset, "buffer_offset")?;
    let available = buffer_size
        .checked_sub(offset)
        .ok_or_else(|| Error::out_of_range("buffer_offset"))?;
    if size > available {
        return Err(Error::out_of_range("size"));
    }
    Ok(size)
}

pub fn null_fs_ops() -> *const sys::CUfileFSOps_t {
    ptr::null()
}
