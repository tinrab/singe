pub mod launch;

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

/// Adapter for querying and configuring context-specific CUDA module functions (`CUfunction`).
#[derive(Debug)]
pub struct ModuleKernelHandle;

impl KernelHandle for ModuleKernelHandle {
    type RawHandle = driver::CUfunction;

    unsafe fn raw_name(
        raw: Self::RawHandle,
        name: *mut *const i8,
        _device_id: i32,
    ) -> driver::CUresult {
        // SAFETY: `cuFuncGetName` receives only plain values.
        unsafe { driver::cuFuncGetName(name, raw) }
    }

    unsafe fn raw_attribute(
        value: *mut i32,
        attribute: driver::CUfunction_attribute,
        raw: Self::RawHandle,
        _device_id: i32,
    ) -> driver::CUresult {
        // SAFETY: `cuFuncGetAttribute` receives only plain values.
        unsafe { driver::cuFuncGetAttribute(value, attribute, raw) }
    }

    unsafe fn set_attribute(
        raw: Self::RawHandle,
        attribute: driver::CUfunction_attribute,
        value: i32,
        _device_id: i32,
    ) -> driver::CUresult {
        // SAFETY: `cuFuncSetAttribute` receives only plain values.
        unsafe { driver::cuFuncSetAttribute(raw, attribute, value) }
    }
}

/// Adapter for querying CUDA library kernels (`CUkernel`) and configuring them for a selected device.
#[derive(Debug)]
pub struct LibraryKernelHandle;

impl KernelHandle for LibraryKernelHandle {
    type RawHandle = driver::CUkernel;

    unsafe fn raw_name(
        raw: Self::RawHandle,
        name: *mut *const i8,
        _device_id: i32,
    ) -> driver::CUresult {
        // SAFETY: `cuKernelGetName` receives only plain values.
        unsafe { driver::cuKernelGetName(name, raw) }
    }

    unsafe fn raw_attribute(
        value: *mut i32,
        attribute: driver::CUfunction_attribute,
        raw: Self::RawHandle,
        device_id: i32,
    ) -> driver::CUresult {
        // SAFETY: `cuKernelGetAttribute` receives only plain values.
        unsafe { driver::cuKernelGetAttribute(value, attribute, raw, device_id) }
    }

    unsafe fn set_attribute(
        raw: Self::RawHandle,
        attribute: driver::CUfunction_attribute,
        value: i32,
        device_id: i32,
    ) -> driver::CUresult {
        // SAFETY: `cuKernelSetAttribute` receives only plain values.
        unsafe { driver::cuKernelSetAttribute(attribute, value, raw, device_id) }
    }
}

/// Returns the name of a raw CUDA kernel handle.
///
/// # Safety
/// `raw` must be a live handle associated with `context` for the entire call.
///
/// # Errors
///
/// Returns an error if `context` cannot be bound, the adapter's native name query fails, or CUDA returns a null name pointer. Invalid UTF-8 in a nonnull native name is replaced in the returned string.
pub unsafe fn name<H: KernelHandle>(context: &Context, raw: H::RawHandle) -> Result<String> {
    context.bind()?;
    let mut name = ptr::null();
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(H::raw_name(raw, &raw mut name, context.device().id()))?;
        if name.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(CStr::from_ptr(name).to_string_lossy().into_owned())
    }
}

/// Returns an attribute of a raw CUDA kernel handle.
///
/// # Safety
/// `raw` must be a live handle associated with `context` for the entire call.
///
/// # Errors
///
/// Returns an error if `context` cannot be bound or the adapter's native attribute query fails for the kernel and selected device.
pub unsafe fn attribute<H: KernelHandle>(
    context: &Context,
    raw: H::RawHandle,
    attribute: FunctionAttribute,
) -> Result<i32> {
    context.bind()?;
    let mut value = 0;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(H::raw_attribute(
            &raw mut value,
            attribute.into(),
            raw,
            context.device().id(),
        ))?;
    }
    Ok(value)
}

/// Sets an attribute on a raw CUDA kernel handle.
///
/// # Safety
/// `raw` must be a live handle associated with `context` for the entire call.
///
/// # Errors
///
/// Returns an error if `context` cannot be bound or the adapter's native setter rejects the attribute or value for the kernel and selected device.
pub unsafe fn set_attribute<H: KernelHandle>(
    context: &Context,
    raw: H::RawHandle,
    attribute: FunctionAttribute,
    value: i32,
) -> Result<()> {
    context.bind()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(H::set_attribute(
            raw,
            attribute.into(),
            value,
            context.device().id(),
        ))?;
    }
    Ok(())
}
