use std::ffi::CString;

use singe_core::string_from_c_chars;
use singe_cufile_sys as sys;

use crate::{
    driver::{Slab, split_slabs},
    error::{Error, Result},
    try_ffi,
    types::{BoolConfigParameter, SizeConfigParameter, StringConfigParameter},
    utility::to_i32,
};

pub fn usize_value(param: SizeConfigParameter) -> Result<usize> {
    let mut value = 0;
    unsafe {
        try_ffi!(sys::cuFileGetParameterSizeT(param.into(), &raw mut value,))?;
    }
    Ok(value as usize)
}

pub fn set_usize_value(param: SizeConfigParameter, value: usize) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuFileSetParameterSizeT(param.into(), value as _))?;
    }
    Ok(())
}

pub fn bool_value(param: BoolConfigParameter) -> Result<bool> {
    let mut value = false;
    unsafe {
        try_ffi!(sys::cuFileGetParameterBool(param.into(), &raw mut value))?;
    }
    Ok(value)
}

pub fn set_bool(param: BoolConfigParameter, value: bool) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuFileSetParameterBool(param.into(), value))?;
    }
    Ok(())
}

pub fn string_with_capacity(param: StringConfigParameter, capacity: usize) -> Result<String> {
    if capacity == 0 {
        return Err(Error::OutOfRange {
            name: "capacity".into(),
        });
    }

    let mut buffer = vec![0; capacity];
    unsafe {
        try_ffi!(sys::cuFileGetParameterString(
            param.into(),
            buffer.as_mut_ptr(),
            to_i32(buffer.len(), "capacity")?,
        ))?;
    }
    Ok(string_from_c_chars(&buffer))
}

pub fn set_string(param: StringConfigParameter, value: impl AsRef<str>) -> Result<()> {
    let value = CString::new(value.as_ref())?;
    unsafe {
        try_ffi!(sys::cuFileSetParameterString(param.into(), value.as_ptr()))?;
    }
    Ok(())
}

pub fn usize_value_min_max(param: SizeConfigParameter) -> Result<(usize, usize)> {
    let mut min = 0;
    let mut max = 0;
    unsafe {
        try_ffi!(sys::cuFileGetParameterMinMaxValue(
            param.into(),
            &raw mut min,
            &raw mut max,
        ))?;
    }
    Ok((min as usize, max as usize))
}

pub fn set_posix_pool_slab_array(slabs: &[Slab]) -> Result<()> {
    let (sizes, counts, len) = split_slabs(slabs)?;
    unsafe {
        try_ffi!(sys::cuFileSetParameterPosixPoolSlabArray(
            sizes.as_ptr(),
            counts.as_ptr(),
            len,
        ))?;
    }
    Ok(())
}

pub fn posix_pool_slab_array(len: usize) -> Result<Vec<Slab>> {
    slab_array(len, sys::cuFileGetParameterPosixPoolSlabArray)
}

pub fn set_gpu_bounce_buffer_slab_array(slabs: &[Slab]) -> Result<()> {
    let (sizes, counts, len) = split_slabs(slabs)?;
    unsafe {
        try_ffi!(sys::cuFileSetParameterGpuBounceBufferSlabArray(
            sizes.as_ptr(),
            counts.as_ptr(),
            len,
        ))?;
    }
    Ok(())
}

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
    unsafe {
        try_ffi!(get(sizes.as_mut_ptr(), counts.as_mut_ptr(), raw_len))?;
    }
    Ok(sizes
        .into_iter()
        .zip(counts)
        .map(|(size, count)| Slab::new(size as usize, count as usize))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slab_array_round_trip_shape() -> Result<()> {
        let slabs = Vec::<Slab>::new();
        let (sizes, counts, len) = split_slabs(&slabs)?;
        assert!(sizes.is_empty());
        assert!(counts.is_empty());
        assert_eq!(len, 0);
        Ok(())
    }
}
