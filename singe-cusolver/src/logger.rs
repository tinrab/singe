//! Process-wide cuSOLVER logger configuration.

use std::path::Path;

use singe_core::path_to_cstring;
use singe_cusolver_sys as sys;

use crate::{error::Result, try_ffi};

/// Installs the cuSOLVER logger callback.
///
/// # Safety
///
/// `callback`, if present, must remain valid for use by cuSOLVER and must
/// follow the callback ABI expected by the library.
///
/// # Errors
///
/// Returns an error if cuSOLVER rejects the callback.
pub unsafe fn set_callback(callback: sys::cusolverDnLoggerCallback_t) -> Result<()> {
    // SAFETY: `cusolverDnLoggerSetCallback` receives only plain values.
    unsafe {
        try_ffi!(sys::cusolverDnLoggerSetCallback(callback))?;
    }
    Ok(())
}

/// Sets the cuSOLVER logger verbosity level.
///
/// # Errors
///
/// Returns an error if cuSOLVER rejects the logging level.
pub fn set_level(level: i32) -> Result<()> {
    // SAFETY: `cusolverDnLoggerSetLevel` receives only plain values.
    unsafe {
        try_ffi!(sys::cusolverDnLoggerSetLevel(level))?;
    }
    Ok(())
}

/// Sets the cuSOLVER logger mask.
///
/// # Errors
///
/// Returns an error if cuSOLVER rejects the logging mask.
pub fn set_mask(mask: i32) -> Result<()> {
    // SAFETY: `cusolverDnLoggerSetMask` receives only plain values.
    unsafe {
        try_ffi!(sys::cusolverDnLoggerSetMask(mask))?;
    }
    Ok(())
}

/// Sets the FILE handle used for cuSOLVER logging.
///
/// Once registered, the file handle must remain open until another handle is
/// installed or logging is disabled.
///
/// # Safety
///
/// `file` must be a valid `FILE` handle for as long as cuSOLVER may write to it.
///
/// # Errors
///
/// Returns an error if cuSOLVER rejects the file handle.
pub unsafe fn set_file(file: *mut sys::FILE) -> Result<()> {
    // SAFETY: `cusolverDnLoggerSetFile` receives only plain values.
    unsafe {
        try_ffi!(sys::cusolverDnLoggerSetFile(file))?;
    }
    Ok(())
}

/// Sets the cuSOLVER logging output file by path.
///
/// # Errors
///
/// Returns an error if `path` cannot be converted to a C string or if
/// cuSOLVER cannot open the log file.
pub fn open_file(path: impl AsRef<Path>) -> Result<()> {
    let path = path_to_cstring(path.as_ref())?;
    // SAFETY: `cusolverDnLoggerOpenFile` receives pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnLoggerOpenFile(path.as_ptr()))?;
    }
    Ok(())
}

/// Disables cuSOLVER logging for the current process.
///
/// # Errors
///
/// Returns an error if cuSOLVER cannot disable logging.
pub fn disable() -> Result<()> {
    // SAFETY: `cusolverDnLoggerForceDisable` takes no arguments.
    unsafe {
        try_ffi!(sys::cusolverDnLoggerForceDisable())?;
    }
    Ok(())
}
