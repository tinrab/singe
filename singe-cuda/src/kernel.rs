use std::{ffi::CStr, ptr};

use singe_cuda_sys::driver;

use crate::{
    context::Context,
    error::{Error, Result},
    try_ffi,
    types::FunctionAttribute,
};

/// Raw CUDA kernel handle adapter.
///
/// Implementors define how to query names, attributes, and mutable attributes
/// for a specific CUDA kernel handle type.
pub trait KernelHandle {
    /// Raw CUDA handle type accepted by the corresponding driver calls.
    type RawHandle: Copy;

    /// Calls the CUDA API that returns the kernel name for `raw`.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid kernel handle for `device_id`, and `name` must be
    /// valid for CUDA to write a pointer-sized result.
    unsafe fn raw_name(
        raw: Self::RawHandle,
        name: *mut *const i8,
        device_id: i32,
    ) -> driver::CUresult;

    /// Calls the CUDA API that returns a kernel attribute for `raw`.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid kernel handle for `device_id`, and `value` must be
    /// valid for CUDA to write an `i32` result.
    unsafe fn raw_attribute(
        value: *mut i32,
        attribute: driver::CUfunction_attribute,
        raw: Self::RawHandle,
        device_id: i32,
    ) -> driver::CUresult;

    /// Calls the CUDA API that sets a kernel attribute for `raw`.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid kernel handle for `device_id`, and the selected
    /// attribute/value pair must be accepted by CUDA for that handle.
    unsafe fn set_attribute(
        raw: Self::RawHandle,
        attribute: driver::CUfunction_attribute,
        value: i32,
        device_id: i32,
    ) -> driver::CUresult;
}

#[derive(Debug)]
pub struct ModuleKernelHandle;

impl KernelHandle for ModuleKernelHandle {
    type RawHandle = driver::CUfunction;

    unsafe fn raw_name(
        raw: Self::RawHandle,
        name: *mut *const i8,
        _device_id: i32,
    ) -> driver::CUresult {
        unsafe { driver::cuFuncGetName(name, raw) }
    }

    unsafe fn raw_attribute(
        value: *mut i32,
        attribute: driver::CUfunction_attribute,
        raw: Self::RawHandle,
        _device_id: i32,
    ) -> driver::CUresult {
        unsafe { driver::cuFuncGetAttribute(value, attribute, raw) }
    }

    unsafe fn set_attribute(
        raw: Self::RawHandle,
        attribute: driver::CUfunction_attribute,
        value: i32,
        _device_id: i32,
    ) -> driver::CUresult {
        unsafe { driver::cuFuncSetAttribute(raw, attribute, value) }
    }
}

#[derive(Debug)]
pub struct LibraryKernelHandle;

impl KernelHandle for LibraryKernelHandle {
    type RawHandle = driver::CUkernel;

    unsafe fn raw_name(
        raw: Self::RawHandle,
        name: *mut *const i8,
        _device_id: i32,
    ) -> driver::CUresult {
        unsafe { driver::cuKernelGetName(name, raw) }
    }

    unsafe fn raw_attribute(
        value: *mut i32,
        attribute: driver::CUfunction_attribute,
        raw: Self::RawHandle,
        device_id: i32,
    ) -> driver::CUresult {
        unsafe { driver::cuKernelGetAttribute(value, attribute, raw, device_id) }
    }

    unsafe fn set_attribute(
        raw: Self::RawHandle,
        attribute: driver::CUfunction_attribute,
        value: i32,
        device_id: i32,
    ) -> driver::CUresult {
        unsafe { driver::cuKernelSetAttribute(attribute, value, raw, device_id) }
    }
}

pub fn name<H: KernelHandle>(ctx: &Context, raw: H::RawHandle) -> Result<String> {
    ctx.bind()?;
    let mut name = ptr::null();
    unsafe {
        try_ffi!(H::raw_name(raw, &raw mut name, ctx.device().id()))?;
        if name.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(CStr::from_ptr(name).to_string_lossy().into_owned())
    }
}

pub fn attribute<H: KernelHandle>(
    ctx: &Context,
    raw: H::RawHandle,
    attribute: FunctionAttribute,
) -> Result<i32> {
    ctx.bind()?;
    let mut value = 0;
    unsafe {
        try_ffi!(H::raw_attribute(
            &raw mut value,
            attribute.into(),
            raw,
            ctx.device().id(),
        ))?;
    }
    Ok(value)
}

pub fn set_attribute<H: KernelHandle>(
    ctx: &Context,
    raw: H::RawHandle,
    attribute: FunctionAttribute,
    value: i32,
) -> Result<()> {
    ctx.bind()?;
    unsafe {
        try_ffi!(H::set_attribute(
            raw,
            attribute.into(),
            value,
            ctx.device().id(),
        ))?;
    }
    Ok(())
}
