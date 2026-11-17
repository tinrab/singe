use std::{
    ffi::{CStr, CString, c_char},
    slice,
};

use singe_core::checked_conversions;

use crate::error::{Error, Result};

checked_conversions!(Error {
    to_u64 -> u64,
    to_u32 -> u32,
    to_usize -> usize,
});

pub(crate) fn optional_cstr<'a>(ptr: *const c_char) -> Option<&'a CStr> {
    if ptr.is_null() {
        None
    } else {
        // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
        Some(unsafe { CStr::from_ptr(ptr) })
    }
}

pub(crate) fn copy_string(value: *const c_char) -> Result<CString> {
    if value.is_null() {
        return Err(Error::NullHandle);
    }
    // SAFETY: `value` was checked for null, and every caller passes a NUL-terminated string CUPTI keeps alive until this copy.
    Ok(unsafe { CStr::from_ptr(value) }.to_owned())
}

pub(crate) fn copy_strings(values: *const *const c_char, count: u64) -> Result<Vec<CString>> {
    let count = to_usize(count, "string count")?;
    if count == 0 {
        return Ok(Vec::new());
    }

    if values.is_null() {
        return Err(Error::NullHandle);
    }
    // SAFETY: every caller passes a non-null table of `count` string pointers that CUPTI returned and keeps alive until this copy.
    unsafe { slice::from_raw_parts(values, count) }
        .iter()
        .map(|&value| copy_string(value))
        .collect()
}
