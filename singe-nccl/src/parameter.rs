//! Read-only access to NCCL runtime parameters.

use std::{
    ffi::{CStr, CString},
    ptr, slice,
};

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    utility::to_usize,
};

/// A parameter handle owned by NCCL's parameter system.
/// The handle must not be freed by the caller.
#[derive(Debug)]
pub struct Parameter {
    handle: sys::ncclParamHandle_t,
}

macro_rules! integer_getters {
    ($($name:ident: $ty:ty => $ffi:ident),+ $(,)?) => {
        $(
            /// Read the value of the bound parameter as the requested integer type.
            ///
            /// # Errors
            ///
            /// Returns an error if NCCL cannot read this parameter using the requested integer type.
            pub fn $name(&self) -> Result<$ty> {
                let mut value = 0;
                // SAFETY: the `ffi` entry point receives live handles owned by the wrappers and valid output locations.
                unsafe { try_ffi!(sys::$ffi(self.handle, &raw mut value))?; }
                Ok(value)
            }
        )+
    };
}

impl Parameter {
    /// Look up the parameter identified by `key` and return a handle to it.
    ///
    /// # Errors
    ///
    /// Returns an error if NCCL cannot bind `key` or returns a null parameter handle.
    pub fn create(key: &CStr) -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: `ncclParamBind` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclParamBind(&raw mut handle, key.as_ptr()))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle })
    }

    integer_getters! {
        i8: i8 => ncclParamGetI8,
        i16: i16 => ncclParamGetI16,
        i32: i32 => ncclParamGetI32,
        i64: i64 => ncclParamGetI64,
        u8: u8 => ncclParamGetU8,
        u16: u16 => ncclParamGetU16,
        u32: u32 => ncclParamGetU32,
        u64: u64 => ncclParamGetU64,
    }

    /// Read the value of the bound parameter as a string.
    /// The returned string is copied before another NCCL call can invalidate its storage.
    ///
    /// # Errors
    ///
    /// Returns an error if NCCL cannot retrieve the string value or returns a null string pointer.
    pub fn string(&self) -> Result<CString> {
        let mut value = ptr::null();
        // SAFETY: `ncclParamGetStr` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclParamGetStr(self.handle, &raw mut value))?;
        }
        if value.is_null() {
            return Err(Error::NullPointer);
        }
        // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
        Ok(unsafe { CStr::from_ptr(value) }.to_owned())
    }
}

/// Get all registered parameter keys.
/// By default, the results include only parameters published in NCCL documentation.
/// Setting `NCCL_PARAM_DUMP_ALL=true` will include all parameters.
/// The returned strings are owned copies of NCCL's thread-local table.
///
/// # Errors
///
/// Returns an error if NCCL cannot enumerate parameter keys, the returned count cannot be represented as `usize`, or a nonempty table or key pointer is null.
pub fn keys() -> Result<Vec<CString>> {
    let mut table = ptr::null_mut();
    let mut len = 0;
    // SAFETY: `ncclParamGetAllParameterKeys` receives valid output locations.
    unsafe {
        try_ffi!(sys::ncclParamGetAllParameterKeys(
            &raw mut table,
            &raw mut len
        ))?;
    }
    let len = to_usize(len, "parameter count")?;
    if len == 0 {
        return Ok(Vec::new());
    }
    if table.is_null() {
        return Err(Error::NullPointer);
    }
    // SAFETY: NCCL returned a non-null table of `len` entries that it keeps alive.
    unsafe { slice::from_raw_parts(table, len) }
        .iter()
        .map(|&key| {
            if key.is_null() {
                return Err(Error::NullPointer);
            }
            // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
            Ok(unsafe { CStr::from_ptr(key) }.to_owned())
        })
        .collect()
}
