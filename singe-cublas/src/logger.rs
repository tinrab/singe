//! Process-wide cuBLAS logger configuration.

use std::{ffi::c_char, path::Path, ptr};

use singe_core::{
    logger::{self, CallbackSlot},
    path_to_cstring, report_cleanup_failure,
};

use crate::{error::Result, sys, try_ffi};

/// A cuBLAS log message, borrowed for the duration of the callback.
///
/// cuBLAS passes only the message, so the level is always 0 and the function name is empty.
pub type LogRecord<'a> = logger::LogRecord<'a, ()>;

/// A cuBLAS log message that owns its strings.
pub type OwnedLogRecord = logger::OwnedLogRecord<()>;

static CALLBACK: CallbackSlot<()> = CallbackSlot::new("cublas");

/// Keeps a callback installed by [`set_callback`]; dropping it removes the callback unless a newer one replaced it.
#[derive(Debug)]
#[must_use = "dropping the guard removes the callback"]
pub struct CallbackGuard {
    generation: u64,
}

impl Drop for CallbackGuard {
    fn drop(&mut self) {
        // SAFETY: clearing the native callback passes no pointers.
        let result = CALLBACK.release(self.generation, || unsafe {
            try_ffi!(sys::cublasSetLoggerCallback(None))
        });
        if let Err(error) = result {
            report_cleanup_failure("clear", "cublas logger callback", &error);
        }
    }
}

/// Sends cuBLAS's log messages to `callback` until the returned guard drops, and turns cuBLAS logging on.
///
/// cuBLAS calls it on the thread that logs, and it must not call cuBLAS.
/// A panic in `callback` is caught, and the message it was handling is lost.
///
/// # Errors
///
/// Returns an error if cuBLAS rejects the callback.
pub fn set_callback(
    callback: impl for<'a> Fn(LogRecord<'a>) + Send + Sync + 'static,
) -> Result<CallbackGuard> {
    // SAFETY: the trampoline has the signature cuBLAS expects and only reads `CALLBACK`.
    let generation = CALLBACK.install(callback, || unsafe {
        try_ffi!(sys::cublasSetLoggerCallback(Some(trampoline)))
    })?;
    Ok(CallbackGuard { generation })
}

unsafe extern "C" fn trampoline(message: *const c_char) {
    // SAFETY: cuBLAS passes a message that stays valid for the duration of the callback.
    unsafe { CALLBACK.dispatch(0, ptr::null(), message) };
}

/// Turns cuBLAS logging on and writes the log to the file at `path`.
///
/// # Errors
///
/// Returns an error if `path` contains an interior NUL byte or cuBLAS rejects the configuration.
pub fn open_file(path: &Path) -> Result<()> {
    let path = path_to_cstring(path)?;
    // SAFETY: `cublasLoggerConfigure` copies the path during the call.
    unsafe { try_ffi!(sys::cublasLoggerConfigure(1, 0, 0, path.as_ptr())) }
}

/// Turns cuBLAS logging off.
///
/// # Errors
///
/// Returns an error if cuBLAS rejects the configuration.
pub fn disable() -> Result<()> {
    // SAFETY: a null file name is allowed, and the other arguments are flags passed by value.
    unsafe { try_ffi!(sys::cublasLoggerConfigure(0, 0, 0, ptr::null())) }
}
