use std::{
    borrow::Cow,
    ffi::CStr,
    os::raw::c_char,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_display, path_to_cstring};
use singe_cudss_sys as sys;

use crate::{error::Result, try_ffi};

/// Global cuDSS logger controls.
///
/// cuDSS exposes process-wide logger state, so this type is a lightweight handle rather than an owned logger instance.
#[derive(Debug, Clone, Copy, Default)]
pub struct Logger;

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

impl LogLevel {
    /// Returns the raw cuDSS log level value.
    pub const fn raw(self) -> i32 {
        self as i32
    }
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
        /// Disable all message classes.
        const OFF = 0;
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

/// A borrowed cuDSS logger message.
///
/// The string data is only valid for the duration of the callback.
/// Store an owned copy if the message must outlive the callback invocation.
#[derive(Debug, Clone)]
pub struct LogRecord<'a> {
    raw_level: i32,
    function: Cow<'a, str>,
    message: Cow<'a, str>,
}

impl<'a> LogRecord<'a> {
    /// Returns the raw cuDSS log level value.
    pub const fn raw_level(&self) -> i32 {
        self.raw_level
    }

    /// Returns the typed log level, or `None` for an unknown raw level.
    pub fn level(&self) -> Option<LogLevel> {
        LogLevel::try_from(self.raw_level).ok()
    }

    /// Returns the cuDSS function name associated with this message.
    pub fn function(&self) -> &str {
        &self.function
    }

    /// Returns the logger message text.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Copies this borrowed record into an owned record.
    pub fn into_owned(self) -> OwnedLogRecord {
        self.into()
    }
}

/// An owned cuDSS logger message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedLogRecord {
    raw_level: i32,
    function: String,
    message: String,
}

impl OwnedLogRecord {
    /// Returns the raw cuDSS log level value.
    pub const fn raw_level(&self) -> i32 {
        self.raw_level
    }

    /// Returns the typed log level, or `None` for an unknown raw level.
    pub fn level(&self) -> Option<LogLevel> {
        LogLevel::try_from(self.raw_level).ok()
    }

    /// Returns the cuDSS function name associated with this message.
    pub fn function(&self) -> &str {
        &self.function
    }

    /// Returns the logger message text.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<LogRecord<'_>> for OwnedLogRecord {
    fn from(record: LogRecord<'_>) -> Self {
        Self {
            raw_level: record.raw_level,
            function: record.function.into_owned(),
            message: record.message.into_owned(),
        }
    }
}

/// RAII guard for a logger callback installed through [`Logger::set_callback`].
///
/// Dropping the guard clears the callback if it is still the active callback.
#[derive(Debug)]
pub struct CallbackGuard {
    generation: u64,
}

type Callback = Arc<dyn for<'a> Fn(LogRecord<'a>) + Send + Sync + 'static>;

static CALLBACK: Mutex<Option<(u64, Callback)>> = Mutex::new(None);
static CALLBACK_GENERATION: AtomicU64 = AtomicU64::new(1);

impl Logger {
    /// Returns a handle to the process-wide cuDSS logger controls.
    pub const fn global() -> Self {
        Self
    }

    /// Sets a custom callback function to receive log messages.
    ///
    /// The callback is invoked synchronously from the calling thread. It must
    /// not call cuDSS APIs because doing so may deadlock.
    ///
    /// Logger state is process-wide. In MGMN mode, each process has independent
    /// logger state and must install its own callback if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot install the callback.
    pub fn set_callback(
        self,
        callback: impl for<'a> Fn(LogRecord<'a>) + Send + Sync + 'static,
    ) -> Result<CallbackGuard> {
        let generation = CALLBACK_GENERATION.fetch_add(1, Ordering::Relaxed);
        {
            let mut current = CALLBACK
                .lock()
                .expect("cudss logger callback lock poisoned");
            *current = Some((generation, Arc::new(callback)));
        }

        unsafe {
            if let Err(error) = try_ffi!(sys::cudssLoggerSetCallback(Some(callback_trampoline))) {
                let mut current = CALLBACK
                    .lock()
                    .expect("cudss logger callback lock poisoned");
                if current
                    .as_ref()
                    .is_some_and(|(current_generation, _)| *current_generation == generation)
                {
                    *current = None;
                }
                return Err(error);
            }
        }

        Ok(CallbackGuard { generation })
    }

    /// Clears the currently installed logger callback.
    ///
    /// Logger state is process-wide. In MGMN mode, this only clears the
    /// callback in the current process.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot clear the callback.
    pub fn clear_callback(self) -> Result<()> {
        {
            let mut current = CALLBACK
                .lock()
                .expect("cudss logger callback lock poisoned");
            *current = None;
        }

        unsafe {
            try_ffi!(sys::cudssLoggerSetCallback(None))?;
        }
        Ok(())
    }

    /// Opens a file and redirects log output to it.
    ///
    /// The file is opened in write mode, truncating any existing content.
    /// Multiple calls close the previous file. Parent directories must already
    /// exist.
    ///
    /// In MGMN mode, each process should open a different file. Multiple
    /// processes writing to the same file concurrently is not safe.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` cannot be converted to a C string or if cuDSS
    /// cannot open the file.
    pub fn open_file(self, path: impl AsRef<Path>) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        unsafe {
            try_ffi!(sys::cudssLoggerOpenFile(path.as_ptr()))?;
        }
        Ok(())
    }

    /// Sets the logger verbosity level.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the level.
    pub fn set_level(self, level: LogLevel) -> Result<()> {
        unsafe {
            try_ffi!(sys::cudssLoggerSetLevel(level.raw()))?;
        }
        Ok(())
    }

    /// Sets the logger message class mask.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the mask.
    pub fn set_mask(self, mask: LogMask) -> Result<()> {
        unsafe {
            try_ffi!(sys::cudssLoggerSetMask(mask.bits()))?;
        }
        Ok(())
    }

    /// Permanently and irreversibly disables all logging for the entire process.
    ///
    /// Once called, logging cannot be re-enabled for the lifetime of the
    /// process. Existing file sinks and registered callbacks are disabled
    /// immediately, and this takes precedence over every other logging
    /// configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot disable logging.
    pub fn disable(self) -> Result<()> {
        unsafe {
            try_ffi!(sys::cudssLoggerForceDisable())?;
        }
        Ok(())
    }
}

impl CallbackGuard {
    /// Returns the callback generation guarded by this value.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Clears the callback guarded by this value.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot clear the callback.
    pub fn clear(mut self) -> Result<()> {
        let result = clear_callback_generation(self.generation);
        self.generation = 0;
        result
    }
}

impl Drop for CallbackGuard {
    fn drop(&mut self) {
        let should_clear = {
            let current = CALLBACK
                .lock()
                .expect("cudss logger callback lock poisoned");
            current
                .as_ref()
                .is_some_and(|(generation, _)| *generation == self.generation)
        };

        if should_clear {
            let _ = clear_callback_generation(self.generation);
        }
    }
}

fn clear_callback_generation(generation: u64) -> Result<()> {
    let should_clear = {
        let mut current = CALLBACK
            .lock()
            .expect("cudss logger callback lock poisoned");
        if current
            .as_ref()
            .is_some_and(|(current_generation, _)| *current_generation == generation)
        {
            *current = None;
            true
        } else {
            false
        }
    };

    if should_clear {
        unsafe {
            try_ffi!(sys::cudssLoggerSetCallback(None))?;
        }
    }
    Ok(())
}

extern "C" fn callback_trampoline(
    raw_level: i32,
    function_name: *const c_char,
    message: *const c_char,
) {
    let callback = {
        let current = CALLBACK
            .lock()
            .expect("cudss logger callback lock poisoned");
        current.as_ref().map(|(_, callback)| Arc::clone(callback))
    };

    if let Some(callback) = callback {
        callback(LogRecord {
            raw_level,
            function: c_string_lossy(function_name),
            message: c_string_lossy(message),
        });
    }
}

fn c_string_lossy<'a>(ptr: *const c_char) -> Cow<'a, str> {
    if ptr.is_null() {
        Cow::Borrowed("")
    } else {
        unsafe { CStr::from_ptr(ptr).to_string_lossy() }
    }
}
