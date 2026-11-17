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

/// cuDSS logger verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum LogLevel {
    /// Disable logger output by level.
    Off = 0,
    /// Error messages.
    Error = 1,
    /// Internal trace messages.
    Trace = 2,
    /// Hint messages.
    Hints = 3,
    /// Informational messages.
    Info = 4,
    /// API call trace messages.
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
    /// cuDSS logger message class mask.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct LogMask: i32 {
        /// Error messages.
        const ERROR = 1;
        /// Internal trace messages.
        const TRACE = 2;
        /// Hint messages.
        const HINTS = 4;
        /// Informational messages.
        const INFO = 8;
        /// API call trace messages.
        const API_TRACE = 16;
    }
}

impl Display for LogMask {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// A cuDSS log message, borrowed for the duration of the callback.
pub type LogRecord<'a> = logger::LogRecord<'a, LogLevel>;

/// A cuDSS log message that owns its strings.
pub type OwnedLogRecord = logger::OwnedLogRecord<LogLevel>;

static CALLBACK: CallbackSlot<LogLevel> = CallbackSlot::new("cudss");

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
            try_ffi!(sys::cudssLoggerSetCallback(None))
        });
        if let Err(error) = result {
            report_cleanup_failure("clear", "cudss logger callback", &error);
        }
    }
}

/// Sends cuDSS's log messages to `callback` until the returned guard drops.
///
/// cuDSS calls it on the thread that logs, and it must not call cuDSS.
/// A panic in `callback` is caught, and the message it was handling is lost.
///
/// Logger state is process-wide.
/// In MGMN mode, each process has independent logger state and must install its own callback if needed.
///
/// # Errors
///
/// Returns an error if cuDSS rejects the callback.
pub fn set_callback(
    callback: impl for<'a> Fn(LogRecord<'a>) + Send + Sync + 'static,
) -> Result<CallbackGuard> {
    // SAFETY: the trampoline has the signature cuDSS expects and only reads `CALLBACK`.
    let generation = CALLBACK.install(callback, || unsafe {
        try_ffi!(sys::cudssLoggerSetCallback(Some(trampoline)))
    })?;
    Ok(CallbackGuard { generation })
}

unsafe extern "C" fn trampoline(level: c_int, function: *const c_char, message: *const c_char) {
    // SAFETY: cuDSS passes strings that stay valid for the duration of the callback.
    unsafe { CALLBACK.dispatch(level, function, message) };
}

/// Sets the most verbose level cuDSS logs.
///
/// # Errors
///
/// Returns an error if cuDSS rejects the level.
pub fn set_level(level: LogLevel) -> Result<()> {
    // SAFETY: the level is a `#[repr(i32)]` value cuDSS defines, and setting it touches no Rust-owned memory.
    unsafe { try_ffi!(sys::cudssLoggerSetLevel(level.into())) }
}

/// Sets which message classes cuDSS logs.
///
/// # Errors
///
/// Returns an error if cuDSS rejects the mask.
pub fn set_mask(mask: LogMask) -> Result<()> {
    // SAFETY: the mask is an integer built from cuDSS's mask bits, and setting it touches no Rust-owned memory.
    unsafe { try_ffi!(sys::cudssLoggerSetMask(mask.bits())) }
}

/// Writes cuDSS's log to the file at `path`, replacing its contents.
///
/// Multiple calls close the previous file.
/// Parent directories must already exist.
///
/// In MGMN mode, each process should open a different file.
/// Multiple processes writing to the same file concurrently is not safe.
///
/// # Errors
///
/// Returns an error if `path` contains an interior NUL byte or cuDSS cannot open the file.
pub fn open_file(path: &Path) -> Result<()> {
    let path = path_to_cstring(path)?;
    // SAFETY: `cudssLoggerOpenFile` reads the path only during the call.
    unsafe { try_ffi!(sys::cudssLoggerOpenFile(path.as_ptr())) }
}

/// Permanently and irreversibly disables all logging for the entire process.
///
/// Once called, logging cannot be re-enabled for the lifetime of the process.
/// Existing file sinks and registered callbacks are disabled immediately, and this takes precedence over every other logging configuration.
///
/// # Errors
///
/// Returns an error if cuDSS cannot disable logging.
pub fn disable() -> Result<()> {
    // SAFETY: `cudssLoggerForceDisable` takes no arguments.
    unsafe { try_ffi!(sys::cudssLoggerForceDisable()) }
}
