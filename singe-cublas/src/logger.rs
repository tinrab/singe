//! Process-wide cuBLAS logger configuration.

use std::{path::Path, ptr};

use singe_core::path_to_cstring;
use singe_cublas_sys as sys;

use crate::{error::Result, try_ffi};

/// Installs a custom logger callback through cuBLAS.
///
/// # Safety
///
/// `callback`, if present, must remain valid for use by cuBLAS and must follow
/// the callback ABI expected by the library.
///
/// # Errors
///
/// Returns an error if cuBLAS rejects the callback.
pub unsafe fn set_callback(callback: sys::cublasLogCallback) -> Result<()> {
    // SAFETY: `cublasSetLoggerCallback` receives only plain values.
    unsafe {
        try_ffi!(sys::cublasSetLoggerCallback(callback))?;
    }
    Ok(())
}

/// Configures cuBLAS logging at runtime.
/// Logging can also be configured with environment variables checked by `libcublas`.
///
/// # Errors
///
/// Returns an error if `file` contains an interior NUL byte or if cuBLAS
/// rejects the logging configuration.
pub fn configure(enabled: bool, stdout: bool, stderr: bool, file: Option<&Path>) -> Result<()> {
    let file = file.map(path_to_cstring).transpose()?;

    // SAFETY: `cublasLoggerConfigure` receives pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasLoggerConfigure(
            enabled.into(),
            stdout.into(),
            stderr.into(),
            file.as_ref().map_or(ptr::null(), |file| file.as_ptr()),
        ))?;
    }
    Ok(())
}
