use std::ffi::CString;

use singe_core::string_from_c_chars;
use singe_cufile_sys as sys;

use crate::{
    driver::{Slab, split_slabs},
    error::{Error, Result},
    try_ffi,
    types::{BoolConfigParameter, SizeConfigParameter, StringConfigParameter},
    utility::{to_i32, to_usize},
};

/// Returns a size-valued configuration parameter.
///
/// Units depend on `param`.
///
/// # Errors
///
/// Returns an error if cuFile cannot query `param` or the reported value cannot be represented as `usize`.
pub fn usize_value(param: SizeConfigParameter) -> Result<usize> {
    let mut value = 0;
    // SAFETY: `cuFileGetParameterSizeT` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuFileGetParameterSizeT(param.into(), &raw mut value,))?;
    }
    to_usize(value, "configuration size")
}

/// Sets a size-valued configuration override for the next driver load.
///
/// Units depend on `param`.
///
/// # Errors
///
/// Returns an error if the driver is already open or cuFile rejects the parameter or value.
pub fn set_usize_value(param: SizeConfigParameter, value: usize) -> Result<()> {
    // SAFETY: `cuFileSetParameterSizeT` receives only plain values.
    unsafe {
        try_ffi!(sys::cuFileSetParameterSizeT(param.into(), value as _))?;
    }
    Ok(())
}

/// Returns a boolean configuration parameter.
///
/// # Errors
///
/// Returns an error if cuFile cannot query `param`.
pub fn bool_value(param: BoolConfigParameter) -> Result<bool> {
    let mut value = false;
    // SAFETY: `cuFileGetParameterBool` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuFileGetParameterBool(param.into(), &raw mut value))?;
    }
    Ok(value)
}

/// Sets a boolean configuration override for the next driver load.
///
/// # Errors
///
/// Returns an error if the driver is already open or cuFile rejects the parameter or value.
pub fn set_bool(param: BoolConfigParameter, value: bool) -> Result<()> {
    // SAFETY: `cuFileSetParameterBool` receives only plain values.
    unsafe {
        try_ffi!(sys::cuFileSetParameterBool(param.into(), value))?;
    }
    Ok(())
}

/// Reads a string configuration parameter into a buffer of `capacity` bytes.
///
/// The capacity includes space for the terminating NUL byte.
/// The returned string excludes that terminator.
///
/// # Errors
///
/// Returns an error if `capacity` is zero or exceeds the `i32` range, or cuFile rejects the parameter or buffer size.
pub fn string_with_capacity(param: StringConfigParameter, capacity: usize) -> Result<String> {
    if capacity == 0 {
        return Err(Error::out_of_range("capacity"));
    }

    let mut buffer = vec![0; capacity];
    // SAFETY: `cuFileGetParameterString` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuFileGetParameterString(
            param.into(),
            buffer.as_mut_ptr(),
            to_i32(buffer.len(), "capacity")?,
        ))?;
    }
    Ok(string_from_c_chars(&buffer))
}

/// Sets a string configuration override for the next driver load.
///
/// # Errors
///
/// Returns an error if `value` contains an interior NUL byte, the driver is already open, or cuFile rejects the parameter or value.
pub fn set_string(param: StringConfigParameter, value: impl AsRef<str>) -> Result<()> {
    let value = CString::new(value.as_ref())?;
    // SAFETY: `cuFileSetParameterString` receives pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cuFileSetParameterString(param.into(), value.as_ptr()))?;
    }
    Ok(())
}

/// Returns the permitted minimum and maximum for a size-valued parameter.
///
/// Both values use the units of `param`.
///
/// # Errors
///
/// Returns an error if cuFile cannot query the limits or either limit cannot be represented as `usize`.
pub fn usize_value_min_max(param: SizeConfigParameter) -> Result<(usize, usize)> {
    let mut min = 0;
    let mut max = 0;
    // SAFETY: `cuFileGetParameterMinMaxValue` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuFileGetParameterMinMaxValue(
            param.into(),
            &raw mut min,
            &raw mut max,
        ))?;
    }
    Ok((
        to_usize(min, "minimum size")?,
        to_usize(max, "maximum size")?,
    ))
}

/// Sets the slab sizes and counts for the POSIX memory pool.
///
/// Slab sizes are in KB.
/// The override is applied at driver load time.
///
/// # Errors
///
/// Returns an error if the slab count exceeds the `i32` range, the driver is already open, or cuFile rejects the size/count pairs.
pub fn set_posix_pool_slab_array(slabs: &[Slab]) -> Result<()> {
    let (sizes, counts, len) = split_slabs(slabs)?;
    // SAFETY: `cuFileSetParameterPosixPoolSlabArray` receives pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cuFileSetParameterPosixPoolSlabArray(
            sizes.as_ptr(),
            counts.as_ptr(),
            len,
        ))?;
    }
    Ok(())
}

/// Returns `len` slab size/count pairs for the POSIX memory pool.
///
/// `len` must equal the number of configured slab classes; sizes are in KB.
///
/// # Errors
///
/// Returns an error if `len` exceeds the `i32` range, cuFile rejects the requested length, or a returned size or count cannot be represented as `usize`.
pub fn posix_pool_slab_array(len: usize) -> Result<Vec<Slab>> {
    slab_array(len, sys::cuFileGetParameterPosixPoolSlabArray)
}

/// Sets the slab sizes and counts for the GPU bounce-buffer pool.
///
/// Slab sizes are in KB.
/// The override is applied at driver load time.
///
/// # Errors
///
/// Returns an error if the slab count exceeds the `i32` range, the driver is already open, or cuFile rejects the size/count pairs.
pub fn set_gpu_bounce_buffer_slab_array(slabs: &[Slab]) -> Result<()> {
    let (sizes, counts, len) = split_slabs(slabs)?;
    // SAFETY: `cuFileSetParameterGpuBounceBufferSlabArray` receives pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cuFileSetParameterGpuBounceBufferSlabArray(
            sizes.as_ptr(),
            counts.as_ptr(),
            len,
        ))?;
    }
    Ok(())
}

/// Returns `len` slab size/count pairs for the GPU bounce-buffer pool.
///
/// `len` must equal the number of configured slab classes; sizes are in KB.
///
/// # Errors
///
/// Returns an error if `len` exceeds the `i32` range, cuFile rejects the requested length, or a returned size or count cannot be represented as `usize`.
pub fn gpu_bounce_buffer_slab_array(len: usize) -> Result<Vec<Slab>> {
    slab_array(len, sys::cuFileGetParameterGpuBounceBufferSlabArray)
}

fn slab_array(
    len: usize,
    get: unsafe extern "C" fn(*mut sys::size_t, *mut sys::size_t, i32) -> sys::CUfileError_t,
) -> Result<Vec<Slab>> {
    let raw_len = to_i32(len, "len")?;
    let mut sizes = vec![0; len];
    let mut counts = vec![0; len];
    // SAFETY: `get` receives two output buffers of `len` elements each.
    unsafe {
        try_ffi!(get(sizes.as_mut_ptr(), counts.as_mut_ptr(), raw_len))?;
    }
    sizes
        .into_iter()
        .zip(counts)
        .map(|(size, count)| {
            Ok(Slab::new(
                to_usize(size, "slab size")?,
                to_usize(count, "slab count")?,
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_slab_array_round_trip_shape() -> Result<()> {
        let slabs = Vec::<Slab>::new();
        let (sizes, counts, len) = split_slabs(&slabs)?;
        assert!(sizes.is_empty());
        assert!(counts.is_empty());
        assert_eq!(len, 0);
        Ok(())
    }
}
