//! Process-wide cuTENSOR logger configuration.

use std::path::Path;

use singe_core::path_to_cstring;
use singe_cutensor_sys as sys;

use crate::{
    error::Result,
    try_ffi,
    types::{LoggerLevel, LoggerMask},
};

/// Sets the logging callback.
///
/// # Safety
///
/// `callback` must remain valid for every later cuTENSOR log event that may
/// call it and must follow cuTENSOR's callback requirements.
///
/// # Errors
///
/// Returns an error if cuTENSOR rejects the callback.
pub unsafe fn set_callback(callback: sys::cutensorLoggerCallback_t) -> Result<()> {
    // SAFETY: `cutensorLoggerSetCallback` receives only plain values.
    unsafe {
        try_ffi!(sys::cutensorLoggerSetCallback(callback))?;
    }
    Ok(())
}

/// Sets the logging level.
///
/// # Errors
///
/// Returns an error if cuTENSOR rejects the logging level.
pub fn set_level(level: LoggerLevel) -> Result<()> {
    // SAFETY: `cutensorLoggerSetLevel` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::cutensorLoggerSetLevel(level.as_raw()))?;
    }
    Ok(())
}

/// Sets the log mask.
///
/// # Errors
///
/// Returns an error if cuTENSOR rejects the logging mask.
pub fn set_mask(mask: LoggerMask) -> Result<()> {
    // SAFETY: `cutensorLoggerSetMask` receives only plain values.
    unsafe {
        try_ffi!(sys::cutensorLoggerSetMask(mask.bits()))?;
    }
    Ok(())
}

/// Sets the logging output file.
///
/// # Safety
///
/// `file` must be a valid C `FILE` pointer for every later cuTENSOR log write
/// that may use it, or a null pointer if cuTENSOR accepts that for the active
/// logger configuration.
///
/// # Errors
///
/// Returns an error if cuTENSOR rejects the file pointer.
pub unsafe fn set_file(file: *mut sys::FILE) -> Result<()> {
    // SAFETY: `cutensorLoggerSetFile` receives only plain values.
    unsafe {
        try_ffi!(sys::cutensorLoggerSetFile(file))?;
    }
    Ok(())
}

/// Sets the logging output file by path.
///
/// # Errors
///
/// Returns an error if `path` cannot be converted to a C string or if
/// cuTENSOR cannot open the log file.
pub fn open_file(path: impl AsRef<Path>) -> Result<()> {
    let path = path_to_cstring(path.as_ref())?;
    // SAFETY: `cutensorLoggerOpenFile` receives pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cutensorLoggerOpenFile(path.as_ptr()))?;
    }
    Ok(())
}

/// Disables logging for the entire run.
///
/// # Errors
///
/// Returns an error if cuTENSOR cannot disable logging.
pub fn disable() -> Result<()> {
    // SAFETY: `cutensorLoggerForceDisable` takes no arguments.
    unsafe {
        try_ffi!(sys::cutensorLoggerForceDisable())?;
    }
    Ok(())
}
