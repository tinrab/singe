#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::{
    ffi::{CStr, CString, NulError},
    path::Path,
    slice,
};

/// Converts a filesystem path to a C string.
///
/// On Unix, paths are converted from their raw OS bytes. On other platforms,
/// paths are converted as UTF-8. Interior NUL bytes are reported as
/// [`NulError`].
///
/// # Errors
///
/// Returns an error if the converted path contains an interior NUL byte.
pub fn path_to_cstring(path: &Path) -> Result<CString, NulError> {
    #[cfg(unix)]
    {
        CString::new(path.as_os_str().as_bytes())
    }

    #[cfg(not(unix))]
    {
        CString::new(path.as_os_str().to_string_lossy().as_bytes())
    }
}

/// Converts a fixed C character buffer to a Rust string.
///
/// Bytes after the first NUL are ignored and invalid UTF-8 is replaced with the
/// Unicode replacement character. If the buffer has no NUL terminator, the whole
/// bounded buffer is converted.
#[must_use]
pub fn string_from_c_chars(buffer: &[i8]) -> String {
    // SAFETY: `i8` and `u8` have the same size and alignment, so the buffer can be viewed as bytes.
    let bytes = unsafe { slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), buffer.len()) };
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Converts a C string pointer to a Rust string.
///
/// A null pointer is treated as an empty string. Otherwise `ptr` must point to a
/// valid NUL-terminated C string for the duration of the call.
///
/// # Safety
///
/// `ptr` must be either null or a valid pointer accepted by [`CStr::from_ptr`].
#[must_use]
pub unsafe fn string_from_c_ptr(ptr: *const i8) -> String {
    if ptr.is_null() {
        return String::new();
    }

    // SAFETY: the caller upholds this function's safety contract.
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

/// Converts a library-owned static C string to a `&'static str`.
///
/// Returns `fallback` for a null pointer or invalid UTF-8.
///
/// # Safety
///
/// `ptr` must be null or point to a NUL-terminated string with static storage
/// duration, such as the strings returned by CUDA library error-name functions.
#[must_use]
pub unsafe fn static_str_from_c_ptr(ptr: *const i8, fallback: &'static str) -> &'static str {
    if ptr.is_null() {
        return fallback;
    }

    // SAFETY: the caller guarantees a NUL-terminated string with static storage.
    unsafe { CStr::from_ptr(ptr) }.to_str().unwrap_or(fallback)
}

/// Copies a Rust string into a fixed C character buffer.
///
/// At most `N - 1` bytes are copied and the remaining buffer contents are left
/// unchanged. To guarantee a trailing NUL, zero-initialize the destination before
/// calling this function.
pub fn copy_string_to_c_chars<const N: usize>(buffer: &mut [i8; N], value: &str) {
    let bytes = value.as_bytes();
    let len = bytes.len().min(N.saturating_sub(1));
    for (slot, byte) in buffer.iter_mut().zip(bytes.iter().copied()).take(len) {
        *slot = byte.cast_signed();
    }
}
