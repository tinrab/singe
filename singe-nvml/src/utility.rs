use std::{ffi::c_ulong, ptr, sync::LazyLock, thread::available_parallelism};

use singe_core::string_from_c_chars;
use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    try_ffi,
    types::{ClockRangeI32, ProcessInfo, UtilizationCounter},
};

pub fn option_u64_from_not_available(value: u64) -> Option<u64> {
    if value == sys::NVML_VALUE_NOT_AVAILABLE as u64 {
        None
    } else {
        Some(value)
    }
}

pub const fn struct_version<T>(version: u32) -> u32 {
    (size_of::<T>() as u32) | (version << 24)
}

static ULONG_BITMASK_COUNT: LazyLock<usize> = LazyLock::new(|| {
    let cpu_count = available_parallelism()
        .map(|count| count.get())
        .unwrap_or(64);
    let bits_per_ulong = c_ulong::BITS as usize;
    cpu_count.div_ceil(bits_per_ulong)
});

pub fn ulong_bitmask_count() -> usize {
    *ULONG_BITMASK_COUNT
}

pub fn mask255_bits(mask: sys::nvmlMask255_t) -> Vec<u32> {
    mask.mask
        .into_iter()
        .enumerate()
        .flat_map(|(word_index, word)| {
            (0..u32::BITS).filter_map(move |bit_index| {
                let bit = 1u32 << bit_index;
                (word & bit != 0).then_some((word_index as u32) * u32::BITS + bit_index)
            })
        })
        .collect()
}

pub fn device_string_query(
    device: Device,
    length: usize,
    function: unsafe extern "C" fn(sys::nvmlDevice_t, *mut i8, u32) -> sys::nvmlReturn_t,
) -> Result<String> {
    let mut buffer = vec![0i8; length];
    unsafe {
        try_ffi!(function(
            device.as_raw(),
            buffer.as_mut_ptr(),
            buffer.len() as u32
        ))?;
    }
    Ok(string_from_c_chars(&buffer))
}

pub fn query_u32_list(
    function: impl Fn(*mut u32, *mut u32) -> sys::nvmlReturn_t,
) -> Result<Vec<u32>> {
    let mut count = 0;
    let status = function(&raw mut count, ptr::null_mut());
    if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
        return Ok(Vec::new());
    }
    if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
        return Err(status.into());
    }

    let mut values = vec![0u32; count as usize];
    try_ffi!(function(&raw mut count, values.as_mut_ptr()))?;
    values.truncate(count as usize);
    Ok(values)
}

pub fn query_sized_raw(
    function: impl Fn(*mut u32, *mut u8) -> sys::nvmlReturn_t,
) -> Result<Vec<u8>> {
    let mut size = 0;
    let status = function(&raw mut size, ptr::null_mut());
    if status == sys::nvmlReturn_t::NVML_SUCCESS && size == 0 {
        return Ok(Vec::new());
    }
    if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
        return Err(status.into());
    }

    let mut buffer = vec![0u8; size as usize];
    try_ffi!(function(&raw mut size, buffer.as_mut_ptr()))?;
    buffer.truncate(size as usize);
    Ok(buffer)
}

pub fn query_process_info_list(
    device: Device,
    function: unsafe extern "C" fn(
        sys::nvmlDevice_t,
        *mut u32,
        *mut sys::nvmlProcessInfo_t,
    ) -> sys::nvmlReturn_t,
) -> Result<Vec<ProcessInfo>> {
    let mut count = 0;
    let status = unsafe { function(device.as_raw(), &raw mut count, ptr::null_mut()) };
    if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
        return Ok(Vec::new());
    }
    if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
        return Err(status.into());
    }

    let mut infos = vec![sys::nvmlProcessInfo_t::default(); count as usize];
    unsafe {
        try_ffi!(function(
            device.as_raw(),
            &raw mut count,
            infos.as_mut_ptr()
        ))?;
    }
    infos.truncate(count as usize);
    Ok(infos.into_iter().map(Into::into).collect())
}

pub fn device_utilization_counter(
    device: Device,
    function: unsafe extern "C" fn(sys::nvmlDevice_t, *mut u32, *mut u32) -> sys::nvmlReturn_t,
) -> Result<UtilizationCounter> {
    let mut utilization = 0;
    let mut sampling_period_us = 0;
    unsafe {
        try_ffi!(function(
            device.as_raw(),
            &raw mut utilization,
            &raw mut sampling_period_us,
        ))?;
    }
    Ok(UtilizationCounter {
        utilization,
        sampling_period_us,
    })
}

pub fn device_ulong_bitmask_list(
    function: impl Fn(u32, *mut c_ulong) -> sys::nvmlReturn_t,
) -> Result<Vec<u64>> {
    let count = ulong_bitmask_count();
    let mut values = vec![0 as c_ulong; count];
    try_ffi!(function(count as u32, values.as_mut_ptr()))?;
    Ok(values.into_iter().collect())
}

pub fn device_clock_offset_range(
    device: Device,
    function: unsafe extern "C" fn(sys::nvmlDevice_t, *mut i32, *mut i32) -> sys::nvmlReturn_t,
) -> Result<ClockRangeI32> {
    let mut min = 0;
    let mut max = 0;
    unsafe {
        try_ffi!(function(device.as_raw(), &raw mut min, &raw mut max))?;
    }
    Ok(ClockRangeI32 { min, max })
}
