//! Process-wide cuTENSOR logger configuration.

use std::{
    ffi::c_char,
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

/// cuTENSOR logger verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum LogLevel {
    /// No messages.
    Off = 0,
    /// Error messages.
    Error = 1,
    /// Performance traces.
    PerformanceTrace = 2,
    /// Performance hints.
    PerformanceHints = 3,
    /// Kernel-selection heuristic traces.
    HeuristicsTrace = 4,
    /// API call traces.
    ApiTrace = 5,
}

impl_enum_display!(LogLevel, {
    Self::Off => "off",
    Self::Error => "error",
    Self::PerformanceTrace => "performance trace",
    Self::PerformanceHints => "performance hints",
    Self::HeuristicsTrace => "heuristics trace",
    Self::ApiTrace => "api trace",
});

bitflags::bitflags! {
    /// cuTENSOR logger message classes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct LogMask: i32 {
        /// Error messages.
        const ERROR = 1;
        /// Performance traces.
        const PERFORMANCE_TRACE = 2;
        /// Performance hints.
        const PERFORMANCE_HINTS = 4;
        /// Kernel-selection heuristic traces.
        const HEURISTICS_TRACE = 8;
        /// API call traces.
        const API_TRACE = 16;
    }
}

impl Display for LogMask {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// A cuTENSOR log message, borrowed for the duration of the callback.
pub type LogRecord<'a> = logger::LogRecord<'a, LogLevel>;

/// A cuTENSOR log message that owns its strings.
pub type OwnedLogRecord = logger::OwnedLogRecord<LogLevel>;

static CALLBACK: CallbackSlot<LogLevel> = CallbackSlot::new("cutensor");

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
            try_ffi!(sys::cutensorLoggerSetCallback(None))
        });
        if let Err(error) = result {
            report_cleanup_failure("clear", "cutensor logger callback", &error);
        }
    }
}

/// Sends cuTENSOR's log messages to `callback` until the returned guard drops.
///
/// cuTENSOR calls it on the thread that logs, and it must not call cuTENSOR.
/// A panic in `callback` is caught, and the message it was handling is lost.
///
/// # Errors
///
/// Returns an error if cuTENSOR rejects the callback.
pub fn set_callback(
    callback: impl for<'a> Fn(LogRecord<'a>) + Send + Sync + 'static,
) -> Result<CallbackGuard> {
    // SAFETY: the trampoline has the signature cuTENSOR expects and only reads `CALLBACK`.
    let generation = CALLBACK.install(callback, || unsafe {
        try_ffi!(sys::cutensorLoggerSetCallback(Some(trampoline)))
    })?;
    Ok(CallbackGuard { generation })
}

unsafe extern "C" fn trampoline(level: i32, function: *const c_char, message: *const c_char) {
    // SAFETY: cuTENSOR passes strings that stay valid for the duration of the callback.
    unsafe { CALLBACK.dispatch(level, function, message) };
}

/// Sets the most verbose level cuTENSOR logs.
///
/// # Errors
///
/// Returns an error if cuTENSOR rejects the level.
pub fn set_level(level: LogLevel) -> Result<()> {
    // SAFETY: the level is a valid `LogLevel` value, and the call touches only cuTENSOR's global logger state.
    unsafe { try_ffi!(sys::cutensorLoggerSetLevel(level.into())) }
}

/// Sets which message classes cuTENSOR logs.
///
/// # Errors
///
/// Returns an error if cuTENSOR rejects the mask.
pub fn set_mask(mask: LogMask) -> Result<()> {
    // SAFETY: the mask holds only `LogMask` bits, and the call touches only cuTENSOR's global logger state.
    unsafe { try_ffi!(sys::cutensorLoggerSetMask(mask.bits())) }
}

/// Writes cuTENSOR's log to the file at `path`, replacing its contents.
///
/// # Errors
///
/// Returns an error if `path` contains an interior NUL byte or cuTENSOR cannot open the file.
pub fn open_file(path: &Path) -> Result<()> {
    let path = path_to_cstring(path)?;
    // SAFETY: `path` is a nul-terminated string that outlives the call, which reads it only during the call.
    unsafe { try_ffi!(sys::cutensorLoggerOpenFile(path.as_ptr())) }
}

/// Turns cuTENSOR logging off for the rest of the process.
///
/// # Errors
///
/// Returns an error if cuTENSOR cannot disable logging.
pub fn disable() -> Result<()> {
    // SAFETY: `cutensorLoggerForceDisable` takes no arguments.
    unsafe { try_ffi!(sys::cutensorLoggerForceDisable()) }
}
