//! Process-wide cuSOLVER logger configuration.

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

/// cuSOLVER logger verbosity level.
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
    /// cuSOLVER logger message classes.
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

/// A cuSOLVER log message, borrowed for the duration of the callback.
pub type LogRecord<'a> = logger::LogRecord<'a, LogLevel>;

/// A cuSOLVER log message that owns its strings.
pub type OwnedLogRecord = logger::OwnedLogRecord<LogLevel>;

static CALLBACK: CallbackSlot<LogLevel> = CallbackSlot::new("cusolver");

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
            try_ffi!(sys::cusolverDnLoggerSetCallback(None))
        });
        if let Err(error) = result {
            report_cleanup_failure("clear", "cusolver logger callback", &error);
        }
    }
}

/// Sends cuSOLVER's log messages to `callback` until the returned guard drops.
///
/// cuSOLVER calls it on the thread that logs, and it must not call cuSOLVER.
/// A panic in `callback` is caught, and the message it was handling is lost.
///
/// # Errors
///
/// Returns an error if cuSOLVER rejects the callback.
pub fn set_callback(
    callback: impl for<'a> Fn(LogRecord<'a>) + Send + Sync + 'static,
) -> Result<CallbackGuard> {
    // SAFETY: the trampoline has the signature cuSOLVER expects and only reads `CALLBACK`.
    let generation = CALLBACK.install(callback, || unsafe {
        try_ffi!(sys::cusolverDnLoggerSetCallback(Some(trampoline)))
    })?;
    Ok(CallbackGuard { generation })
}

unsafe extern "C" fn trampoline(level: c_int, function: *const c_char, message: *const c_char) {
    // SAFETY: cuSOLVER passes strings that stay valid for the duration of the callback.
    unsafe { CALLBACK.dispatch(level, function, message) };
}

/// Sets the most verbose level cuSOLVER logs.
///
/// # Errors
///
/// Returns an error if cuSOLVER rejects the level.
pub fn set_level(level: LogLevel) -> Result<()> {
    // SAFETY: the call takes the level by value and touches no Rust memory.
    unsafe { try_ffi!(sys::cusolverDnLoggerSetLevel(level.into())) }
}

/// Sets which message classes cuSOLVER logs.
///
/// # Errors
///
/// Returns an error if cuSOLVER rejects the mask.
pub fn set_mask(mask: LogMask) -> Result<()> {
    // SAFETY: the call takes the mask by value and touches no Rust memory.
    unsafe { try_ffi!(sys::cusolverDnLoggerSetMask(mask.bits())) }
}

/// Writes cuSOLVER's log to the file at `path`, replacing its contents.
///
/// # Errors
///
/// Returns an error if `path` contains an interior NUL byte or cuSOLVER cannot open the file.
pub fn open_file(path: &Path) -> Result<()> {
    let path = path_to_cstring(path)?;
    // SAFETY: `cusolverDnLoggerOpenFile` reads the path only during the call.
    unsafe { try_ffi!(sys::cusolverDnLoggerOpenFile(path.as_ptr())) }
}

/// Turns cuSOLVER logging off for the rest of the process.
///
/// # Errors
///
/// Returns an error if cuSOLVER cannot disable logging.
pub fn disable() -> Result<()> {
    // SAFETY: `cusolverDnLoggerForceDisable` takes no arguments.
    unsafe { try_ffi!(sys::cusolverDnLoggerForceDisable()) }
}
