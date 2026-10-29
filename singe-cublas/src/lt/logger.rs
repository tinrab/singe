//! Process-wide cuBLASLt logger configuration.

use std::path::Path;

use singe_core::path_to_cstring;
use singe_cublas_sys as sys;

use crate::{error::Result, lt::LoggerCallback, try_ffi};

/// Experimental: sets the cuBLASLt logging callback.
///
/// # Errors
///
/// Returns an error if cuBLASLt rejects the callback.
pub fn set_callback(callback: LoggerCallback) -> Result<()> {
    // SAFETY: `cublasLtLoggerSetCallback` receives only plain values.
    unsafe {
        try_ffi!(sys::cublasLtLoggerSetCallback(callback))?;
    }
    Ok(())
}

/// Experimental: sets the cuBLASLt logging level.
///
/// # Errors
///
/// Returns an error if `level` is not accepted by cuBLASLt.
pub fn set_level(level: i32) -> Result<()> {
    // SAFETY: `cublasLtLoggerSetLevel` receives only plain values.
    unsafe {
        try_ffi!(sys::cublasLtLoggerSetLevel(level))?;
    }
    Ok(())
}

/// Experimental: sets the cuBLASLt logging mask.
///
/// # Errors
///
/// Returns an error if cuBLASLt rejects `mask`.
pub fn set_mask(mask: i32) -> Result<()> {
    // SAFETY: `cublasLtLoggerSetMask` receives only plain values.
    unsafe {
        try_ffi!(sys::cublasLtLoggerSetMask(mask))?;
    }
    Ok(())
}

/// Experimental: opens `path` and uses it as the cuBLASLt logging output.
///
/// # Errors
///
/// Returns an error if `path` contains an interior NUL byte or if cuBLASLt cannot open the file.
pub fn open_file(path: impl AsRef<Path>) -> Result<()> {
    let path = path_to_cstring(path.as_ref())?;
    // SAFETY: `cublasLtLoggerOpenFile` receives pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasLtLoggerOpenFile(path.as_ptr()))?;
    }
    Ok(())
}
