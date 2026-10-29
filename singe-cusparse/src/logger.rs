//! Process-wide cuSPARSE logger configuration.

use std::path::Path;

use singe_core::path_to_cstring;
use singe_cusparse_sys as sys;

use crate::{error::Result, try_ffi};

/// Experimental: sets the logging callback function.
///
/// The callback must use the cuSPARSE logger callback ABI.
///
/// # Safety
///
/// `callback`, if present, must remain valid for use by cuSPARSE and must
/// follow the callback ABI expected by the library.
///
/// # Errors
///
/// Returns an error if cuSPARSE rejects the callback.
pub unsafe fn set_callback(callback: sys::cusparseLoggerCallback_t) -> Result<()> {
    // SAFETY: `cusparseLoggerSetCallback` receives only plain values.
    unsafe {
        try_ffi!(sys::cusparseLoggerSetCallback(callback))?;
    }
    Ok(())
}

/// Experimental: sets the logging level.
///
/// # Errors
///
/// Returns an error if cuSPARSE rejects the logging level.
pub fn set_level(level: i32) -> Result<()> {
    // SAFETY: `cusparseLoggerSetLevel` receives only plain values.
    unsafe {
        try_ffi!(sys::cusparseLoggerSetLevel(level))?;
    }
    Ok(())
}

/// Experimental: sets the logging mask.
///
/// # Errors
///
/// Returns an error if cuSPARSE rejects the logging mask.
pub fn set_mask(mask: i32) -> Result<()> {
    // SAFETY: `cusparseLoggerSetMask` receives only plain values.
    unsafe {
        try_ffi!(sys::cusparseLoggerSetMask(mask))?;
    }
    Ok(())
}

/// Experimental: sets the logging output file.
/// Once registered, the provided file handle must remain open until another
/// file handle is registered.
///
/// # Safety
///
/// `file` must be a valid `FILE` handle for as long as cuSPARSE may write to it.
///
/// # Errors
///
/// Returns an error if cuSPARSE rejects the file handle.
pub unsafe fn set_file(file: *mut sys::FILE) -> Result<()> {
    // SAFETY: `cusparseLoggerSetFile` receives only plain values.
    unsafe {
        try_ffi!(sys::cusparseLoggerSetFile(file))?;
    }
    Ok(())
}

/// Experimental: sets the logging output file by path.
///
/// # Errors
///
/// Returns an error if `path` cannot be converted to a C string or if
/// cuSPARSE cannot open the log file.
pub fn open_file(path: impl AsRef<Path>) -> Result<()> {
    let path = path_to_cstring(path.as_ref())?;
    // SAFETY: `cusparseLoggerOpenFile` receives pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseLoggerOpenFile(path.as_ptr()))?;
    }
    Ok(())
}

/// Disables cuSPARSE logging.
///
/// # Errors
///
/// Returns an error if cuSPARSE cannot disable logging.
pub fn disable() -> Result<()> {
    // SAFETY: `cusparseLoggerForceDisable` takes no arguments.
    unsafe {
        try_ffi!(sys::cusparseLoggerForceDisable())?;
    }
    Ok(())
}
