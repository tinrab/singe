use core::ffi;
use std::ffi::c_void;

use singe_cublas_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    utility::to_usize,
};

pub fn set_attribute(
    mut call: impl FnMut(*const c_void, ffi::c_ulong) -> sys::cublasStatus_t,
    value: *const c_void,
    size: usize,
) -> Result<()> {
    try_ffi!(call(value, size as _))?;
    Ok(())
}

pub fn read_attribute(
    mut call: impl FnMut(*mut c_void, ffi::c_ulong, *mut ffi::c_ulong) -> sys::cublasStatus_t,
    value: *mut c_void,
    size: usize,
    name: &str,
) -> Result<usize> {
    let mut written = 0;
    try_ffi!(call(value, size as _, &raw mut written))?;
    to_usize(written, name)
}

pub fn attribute_enum<T: TryFrom<u32>>(value: u32, name: &str) -> Result<T> {
    T::try_from(value).map_err(|_| Error::InvalidAttributeValue {
        name: name.to_owned().into(),
        value,
    })
}
