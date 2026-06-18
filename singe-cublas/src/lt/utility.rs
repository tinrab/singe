use std::ffi::c_void;

use singe_cublas_sys as sys;

use crate::{error::Result, try_ffi, utility::to_usize};

pub fn set_attribute(
    mut call: impl FnMut(*const c_void, core::ffi::c_ulong) -> sys::cublasStatus_t,
    value: *const c_void,
    size: usize,
) -> Result<()> {
    try_ffi!(call(value, size as _))?;
    Ok(())
}

pub fn read_attribute(
    mut call: impl FnMut(
        *mut c_void,
        core::ffi::c_ulong,
        *mut core::ffi::c_ulong,
    ) -> sys::cublasStatus_t,
    value: *mut c_void,
    size: usize,
    name: &str,
) -> Result<usize> {
    let mut written = 0;
    try_ffi!(call(value, size as _, &raw mut written))?;
    to_usize(written, name)
}
