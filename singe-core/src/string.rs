#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::{
    ffi::{CStr, CString, NulError, c_char},
    path::Path,
    slice,
};

/// Converts a filesystem path to a C string.
///
/// On Unix, paths are converted from their raw OS bytes.
/// On other platforms, paths are converted as UTF-8.
/// Interior NUL bytes are reported as [`NulError`].
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
/// Bytes after the first NUL are ignored and invalid UTF-8 is replaced with the Unicode replacement character.
/// If the buffer has no NUL terminator, the whole bounded buffer is converted.
#[must_use]
pub fn string_from_c_chars(buffer: &[c_char]) -> String {
    // SAFETY: `c_char` and `u8` have the same size and alignment, so the buffer can be viewed as bytes.
    let bytes = unsafe { slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), buffer.len()) };
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Converts a C string pointer to a Rust string.
///
/// A null pointer is treated as an empty string.
/// Otherwise `ptr` must point to a valid NUL-terminated C string for the duration of the call.
///
/// # Safety
///
/// `ptr` must be either null or a valid pointer accepted by [`CStr::from_ptr`].
#[must_use]
pub unsafe fn string_from_c_ptr(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }

    // SAFETY: `ptr` is non-null here, and the caller guarantees it points to a NUL-terminated string valid for this call.
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

/// Converts a library-owned static C string to a `&'static str`.
///
/// Returns `fallback` for a null pointer or invalid UTF-8.
///
/// # Safety
///
/// `ptr` must be null or point to a NUL-terminated string with static storage duration, such as the strings returned by CUDA library error-name functions.
#[must_use]
pub unsafe fn static_str_from_c_ptr(ptr: *const c_char, fallback: &'static str) -> &'static str {
    if ptr.is_null() {
        return fallback;
    }

    // SAFETY: `ptr` is non-null here, and the caller guarantees a NUL-terminated string with static storage.
    unsafe { CStr::from_ptr(ptr) }.to_str().unwrap_or(fallback)
}

/// Copies a Rust string into a fixed C character buffer and terminates it with a NUL.
///
/// At most `N - 1` bytes are copied, followed by a NUL; bytes after the NUL are left unchanged.
/// An empty buffer is left unchanged.
pub fn copy_string_to_c_chars<const N: usize>(buffer: &mut [c_char; N], value: &str) {
    let Some(capacity) = N.checked_sub(1) else {
        return;
    };

    let bytes = value.as_bytes();
    let len = bytes.len().min(capacity);
    for (slot, byte) in buffer.iter_mut().zip(bytes.iter().copied()).take(len) {
        *slot = c_char::from_ne_bytes([byte]);
    }

    buffer[len] = 0;
}
