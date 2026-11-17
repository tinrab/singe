//! Process-wide cuBLASLt logger configuration.

use std::{
    ffi::{c_char, c_int},
    fmt::{self, Display, Formatter},
    path::Path,
};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{
    impl_enum_display,
    logger::{self, CallbackSlot},
    path_to_cstring, report_cleanup_failure,
};

use crate::{error::Result, sys, try_ffi};

/// cuBLASLt logger verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum LogLevel {
    /// No messages.
    Off = 0,
    /// Error messages.
    Error = 1,
    /// Internal trace messages.
    Trace = 2,
    /// Performance hints.
    Hints = 3,
    /// Informational messages.
    Info = 4,
    /// API call traces.
    ApiTrace = 5,
}

impl_enum_display!(LogLevel, {
    Self::Off => "off",
    Self::Error => "error",
    Self::Trace => "trace",
    Self::Hints => "hints",
    Self::Info => "info",
    Self::ApiTrace => "api trace",
});

bitflags::bitflags! {
    /// cuBLASLt logger message classes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct LogMask: i32 {
        /// Error messages.
        const ERROR = 1;
        /// Internal trace messages.
        const TRACE = 2;
        /// Performance hints.
        const HINTS = 4;
        /// Informational messages.
        const INFO = 8;
        /// API call traces.
        const API_TRACE = 16;
    }
}

impl Display for LogMask {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// A cuBLASLt log message, borrowed for the duration of the callback.
pub type LogRecord<'a> = logger::LogRecord<'a, LogLevel>;

/// A cuBLASLt log message that owns its strings.
pub type OwnedLogRecord = logger::OwnedLogRecord<LogLevel>;

static CALLBACK: CallbackSlot<LogLevel> = CallbackSlot::new("cublaslt");

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
            try_ffi!(sys::cublasLtLoggerSetCallback(None))
        });
        if let Err(error) = result {
            report_cleanup_failure("clear", "cublaslt logger callback", &error);
        }
    }
}

/// Sends cuBLASLt's log messages to `callback` until the returned guard drops.
///
/// cuBLASLt calls it on the thread that logs, and it must not call cuBLASLt.
/// A panic in `callback` is caught, and the message it was handling is lost.
///
/// # Errors
///
/// Returns an error if cuBLASLt rejects the callback.
pub fn set_callback(
    callback: impl for<'a> Fn(LogRecord<'a>) + Send + Sync + 'static,
) -> Result<CallbackGuard> {
    // SAFETY: the trampoline has the signature cuBLASLt expects and only reads `CALLBACK`.
    let generation = CALLBACK.install(callback, || unsafe {
        try_ffi!(sys::cublasLtLoggerSetCallback(Some(trampoline)))
    })?;
    Ok(CallbackGuard { generation })
}

unsafe extern "C" fn trampoline(level: c_int, function: *const c_char, message: *const c_char) {
    // SAFETY: cuBLASLt passes strings that stay valid for the duration of the callback.
    unsafe { CALLBACK.dispatch(level, function, message) };
}

/// Sets the most verbose level cuBLASLt logs.
///
/// # Errors
///
/// Returns an error if cuBLASLt rejects the level.
pub fn set_level(level: LogLevel) -> Result<()> {
    // SAFETY: `cublasLtLoggerSetLevel` takes the level by value and reads no caller memory.
    unsafe { try_ffi!(sys::cublasLtLoggerSetLevel(level.into())) }
}

/// Sets which message classes cuBLASLt logs.
///
/// # Errors
///
/// Returns an error if cuBLASLt rejects the mask.
pub fn set_mask(mask: LogMask) -> Result<()> {
    // SAFETY: `cublasLtLoggerSetMask` takes the mask by value and reads no caller memory.
    unsafe { try_ffi!(sys::cublasLtLoggerSetMask(mask.bits())) }
}

/// Writes cuBLASLt's log to the file at `path`.
///
/// # Errors
///
/// Returns an error if `path` contains an interior NUL byte or cuBLASLt cannot open the file.
pub fn open_file(path: &Path) -> Result<()> {
    let path = path_to_cstring(path)?;
    // SAFETY: `cublasLtLoggerOpenFile` reads the path only during the call.
    unsafe { try_ffi!(sys::cublasLtLoggerOpenFile(path.as_ptr())) }
}

/// Turns cuBLASLt logging off for the rest of the process.
///
/// # Errors
///
/// Returns an error if cuBLASLt cannot disable logging.
pub fn disable() -> Result<()> {
    // SAFETY: `cublasLtLoggerForceDisable` takes no arguments and only changes process-wide logger state.
    unsafe { try_ffi!(sys::cublasLtLoggerForceDisable()) }
}
