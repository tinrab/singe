//! Process-wide cuSPARSE logger configuration.

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

/// cuSPARSE logger verbosity level.
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
    /// cuSPARSE logger message classes.
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

/// A cuSPARSE log message, borrowed for the duration of the callback.
pub type LogRecord<'a> = logger::LogRecord<'a, LogLevel>;

/// A cuSPARSE log message that owns its strings.
pub type OwnedLogRecord = logger::OwnedLogRecord<LogLevel>;

static CALLBACK: CallbackSlot<LogLevel> = CallbackSlot::new("cusparse");

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
            try_ffi!(sys::cusparseLoggerSetCallback(None))
        });
        if let Err(error) = result {
            report_cleanup_failure("clear", "cusparse logger callback", &error);
        }
    }
}

/// Sends cuSPARSE's log messages to `callback` until the returned guard drops.
///
/// cuSPARSE calls it on the thread that logs, and it must not call cuSPARSE.
/// A panic in `callback` is caught, and the message it was handling is lost.
///
/// # Errors
///
/// Returns an error if cuSPARSE rejects the callback.
pub fn set_callback(
    callback: impl for<'a> Fn(LogRecord<'a>) + Send + Sync + 'static,
) -> Result<CallbackGuard> {
    // SAFETY: the trampoline has the signature cuSPARSE expects and only reads `CALLBACK`.
    let generation = CALLBACK.install(callback, || unsafe {
        try_ffi!(sys::cusparseLoggerSetCallback(Some(trampoline)))
    })?;
    Ok(CallbackGuard { generation })
}

unsafe extern "C" fn trampoline(level: c_int, function: *const c_char, message: *const c_char) {
    // SAFETY: cuSPARSE passes strings that stay valid for the duration of the callback.
    unsafe { CALLBACK.dispatch(level, function, message) };
}

/// Sets the most verbose level cuSPARSE logs.
///
/// # Errors
///
/// Returns an error if cuSPARSE rejects the level.
pub fn set_level(level: LogLevel) -> Result<()> {
    // SAFETY: the level is a valid enum value and the call touches no caller memory.
    unsafe { try_ffi!(sys::cusparseLoggerSetLevel(level.into())) }
}

/// Sets which message classes cuSPARSE logs.
///
/// # Errors
///
/// Returns an error if cuSPARSE rejects the mask.
pub fn set_mask(mask: LogMask) -> Result<()> {
    // SAFETY: the mask is a plain integer and the call touches no caller memory.
    unsafe { try_ffi!(sys::cusparseLoggerSetMask(mask.bits())) }
}

/// Writes cuSPARSE's log to the file at `path`, replacing its contents.
///
/// # Errors
///
/// Returns an error if `path` contains an interior NUL byte or cuSPARSE cannot open the file.
pub fn open_file(path: &Path) -> Result<()> {
    let path = path_to_cstring(path)?;
    // SAFETY: `cusparseLoggerOpenFile` reads the path only during the call.
    unsafe { try_ffi!(sys::cusparseLoggerOpenFile(path.as_ptr())) }
}

/// Turns cuSPARSE logging off for the rest of the process.
///
/// # Errors
///
/// Returns an error if cuSPARSE cannot disable logging.
pub fn disable() -> Result<()> {
    // SAFETY: `cusparseLoggerForceDisable` takes no arguments.
    unsafe { try_ffi!(sys::cusparseLoggerForceDisable()) }
}
