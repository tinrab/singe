use std::ptr;

use singe_core::checked_conversions;
use singe_nvml_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
};

checked_conversions!(Error {
    to_i32 -> i32,
    to_u32 -> u32,
    to_usize -> usize,
});

// NVML fills at most a fixed array's capacity, so a larger reported count has no more valid entries.
pub(crate) fn reported_entries<T>(entries: &[T], count: impl TryInto<usize>) -> &[T] {
    let count = count
        .try_into()
        .map_or(entries.len(), |count| count.min(entries.len()));
    &entries[..count]
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the assertion limits NVML structure sizes to their 24-bit size field"
)]
pub(crate) const fn struct_version<T>(version: u32) -> u32 {
    assert!(
        size_of::<T>() <= 0x00ff_ffff,
        "NVML structure size exceeds 24 bits"
    );
    assert!(version <= 0xff, "NVML structure version exceeds eight bits");
    (size_of::<T>() as u32) | (version << 24)
}

pub(crate) fn query_u32_list(
    function: impl Fn(*mut u32, *mut u32) -> sys::nvmlReturn_t,
) -> Result<Vec<u32>> {
    let mut count = 0;
    let status = function(&raw mut count, ptr::null_mut());
    if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
        return Ok(Vec::new());
    }

    if status != sys::nvmlReturn_t::NVML_SUCCESS
        && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
    {
        return Err(status.into());
    }

    let mut values = vec![0u32; to_usize(count, "NVML list length")?];
    try_ffi!(function(&raw mut count, values.as_mut_ptr()))?;
    values.truncate(to_usize(count, "NVML list length")?);
    Ok(values)
}

pub(crate) fn query_sized_raw(
    function: impl Fn(*mut u32, *mut u8) -> sys::nvmlReturn_t,
) -> Result<Vec<u8>> {
    let mut size = 0;
    let status = function(&raw mut size, ptr::null_mut());
    if status == sys::nvmlReturn_t::NVML_SUCCESS && size == 0 {
        return Ok(Vec::new());
    }

    if status != sys::nvmlReturn_t::NVML_SUCCESS
        && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
    {
        return Err(status.into());
    }

    let mut buffer = vec![0u8; to_usize(size, "NVML buffer size")?];
    try_ffi!(function(&raw mut size, buffer.as_mut_ptr()))?;
    buffer.truncate(to_usize(size, "NVML buffer size")?);
    Ok(buffer)
}
