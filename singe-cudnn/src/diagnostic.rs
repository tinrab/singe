//! cuDNN error reporting: callbacks, last-error messages and remote kernel errors.

use std::{
    ffi::{CStr, CString, c_char, c_void},
    fmt::{self, Display, Formatter},
    ptr, result,
    time::{Duration, SystemTime},
};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{
    LibraryVersion, UnknownEnumValue, impl_enum_display,
    logger::{self, CallbackSlot},
    report_cleanup_failure, string_from_c_chars,
};

use crate::{
    context::Context,
    error::{Result, Status},
    sys, try_ffi, unpack_cudnn_version,
};

singe_core::native_enum! {
    /// Remote kernel error query mode used by [`query_runtime_error`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum RuntimeErrorQueryMode: sys::cudnnErrQueryMode_t {
        /// Read the error storage location regardless of the kernel completion status.
        RawCode = CUDNN_ERRQUERY_RAWCODE,
        /// Report whether all tasks in the configured cuDNN stream have completed.
        /// If that is the case, report the remote kernel error code.
        NonBlocking = CUDNN_ERRQUERY_NONBLOCKING,
        /// Wait for all tasks in the configured cuDNN stream before reporting the remote kernel error code.
        Blocking = CUDNN_ERRQUERY_BLOCKING,
    }
}

singe_core::native_enum! {
    /// Severity passed to the configured cuDNN logging callback.
    /// The numerical values match the values accepted by the `CUDNN_LOGLEVEL_DBG` environment variable.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Severity: sys::cudnnSeverity_t {
        /// Fatal error severity.
        Fatal = CUDNN_SEV_FATAL,
        /// Error severity.
        Error = CUDNN_SEV_ERROR,
        /// Warning severity.
        Warning = CUDNN_SEV_WARNING,
        /// Informational severity.
        Info = CUDNN_SEV_INFO,
    }
}

impl TryFrom<i32> for Severity {
    type Error = UnknownEnumValue;

    fn try_from(value: i32) -> result::Result<Self, UnknownEnumValue> {
        u32::try_from(value)
            .ok()
            .and_then(|value| Self::try_from_primitive(value).ok())
            .ok_or_else(|| UnknownEnumValue::new("Severity", value))
    }
}

bitflags::bitflags! {
    /// cuDNN message severities passed to the logging callback; fatal messages are always passed.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct LogMask: u32 {
        /// Error messages.
        const ERROR = 0b0010;
        /// Warning messages.
        const WARNING = 0b0100;
        /// Informational messages, including API call logs.
        const INFO = 0b1000;
    }
}

impl Display for LogMask {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// Metadata cuDNN passes with each log message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DebugInfo {
    /// cuDNN library version that logged the message, or `None` when cuDNN passed an undecodable value.
    pub version: Option<LibraryVersion>,
    /// Status of the logged API call.
    pub status: Status,
    /// Wall-clock time of the message.
    pub time: SystemTime,
    /// Time since cuDNN started logging, in whole seconds.
    pub since_start: Duration,
    /// Address of the cuDNN handle used by the logged call, for correlating messages.
    pub handle: usize,
    /// Address of the CUDA stream used by the logged call, for correlating messages.
    pub stream: usize,
    /// Process ID.
    pub process_id: u64,
    /// Thread ID.
    pub thread_id: u64,
    /// CUDA device ordinal.
    pub device: i32,
}

impl DebugInfo {
    fn from_raw(raw: &sys::cudnnDebug_t) -> Self {
        Self {
            version: unpack_cudnn_version(u64::from(raw.cudnn_version)),
            status: Status::from(raw.cudnnStatus),
            time: SystemTime::UNIX_EPOCH
                + Duration::from_secs(u64::from(raw.time_sec))
                + Duration::from_micros(u64::from(raw.time_usec)),
            since_start: Duration::from_secs(u64::from(raw.time_delta)),
            handle: raw.handle.addr(),
            stream: raw.stream.addr(),
            process_id: raw.pid,
            thread_id: raw.tid,
            device: raw.cudaDeviceId,
        }
    }
}

/// A cuDNN log message with its metadata, borrowed for the duration of the callback.
pub type LogRecord<'a> = logger::LogRecord<'a, Severity, DebugInfo>;

/// A cuDNN log message with its metadata that owns its strings.
pub type OwnedLogRecord = logger::OwnedLogRecord<Severity, DebugInfo>;

static CALLBACK: CallbackSlot<Severity, DebugInfo> = CallbackSlot::new("cudnn");

/// Keeps a callback installed by [`set_callback`]; dropping it restores cuDNN's default callback and previous mask unless a newer callback replaced it.
#[derive(Debug)]
#[must_use = "dropping the guard removes the callback"]
pub struct CallbackGuard {
    generation: u64,
    previous_mask: u32,
}

impl Drop for CallbackGuard {
    fn drop(&mut self) {
        // SAFETY: a null callback selects cuDNN's default callback, which reads no user data.
        let result = CALLBACK.release(self.generation, || unsafe {
            try_ffi!(sys::cudnnSetCallback(
                self.previous_mask,
                ptr::null_mut(),
                None
            ))
        });
        if let Err(error) = result {
            report_cleanup_failure("clear", "cudnn logger callback", &error);
        }
    }
}

/// Sends cuDNN's log messages of the severities in `mask`, and every fatal message, to `callback` until the returned guard drops.
///
/// cuDNN calls it on the thread that logs, and it must not call cuDNN.
/// A panic in `callback` is caught, and the message it was handling is lost.
///
/// # Errors
///
/// Returns an error if cuDNN cannot report its current mask or rejects the callback.
pub fn set_callback(
    mask: LogMask,
    callback: impl for<'a> Fn(LogRecord<'a>) + Send + Sync + 'static,
) -> Result<CallbackGuard> {
    let mut previous_mask = 0;
    let generation = CALLBACK.install(callback, || -> Result<()> {
        let mut user_data = ptr::null_mut();
        let mut previous_callback = None;
        // SAFETY: `cudnnGetCallback` writes only into the local output locations.
        unsafe {
            try_ffi!(sys::cudnnGetCallback(
                &raw mut previous_mask,
                &raw mut user_data,
                &raw mut previous_callback,
            ))?;
        }

        // SAFETY: the trampoline has the signature cuDNN expects, ignores the user data, and only reads `CALLBACK`.
        unsafe {
            try_ffi!(sys::cudnnSetCallback(
                mask.bits(),
                ptr::null_mut(),
                Some(trampoline)
            ))
        }
    })?;
    Ok(CallbackGuard {
        generation,
        previous_mask,
    })
}

unsafe extern "C" fn trampoline(
    severity: sys::cudnnSeverity_t,
    _user_data: *mut c_void,
    debug: *const sys::cudnnDebug_t,
    message: *const c_char,
) {
    let debug = if debug.is_null() {
        DebugInfo::from_raw(&sys::cudnnDebug_t::default())
    } else {
        // SAFETY: cuDNN passes a record that stays valid for the duration of the callback.
        DebugInfo::from_raw(unsafe { &*debug })
    };

    // SAFETY: cuDNN passes a message that stays valid for the duration of the callback.
    let joined = unsafe { join_lines(message) };
    let message = joined.as_ref().map_or(message, |joined| joined.as_ptr());
    let severity = i32::try_from(severity.0).unwrap_or(i32::MAX);
    // SAFETY: `message` is null, cuDNN's string, or `joined`, each valid for this call.
    unsafe { CALLBACK.dispatch_with(severity, ptr::null(), message, debug) };
}

// cuDNN ends each line of a message with a NUL byte and the whole message with an empty line.
// The caller passes null or such a message, valid for this call.
unsafe fn join_lines(message: *const c_char) -> Option<CString> {
    if message.is_null() {
        return None;
    }

    let mut joined = Vec::new();
    let mut line = message;
    loop {
        // SAFETY: `line` starts a NUL-terminated line before the message's final empty line.
        let bytes = unsafe { CStr::from_ptr(line) }.to_bytes();
        if bytes.is_empty() {
            break;
        }

        if !joined.is_empty() {
            joined.push(b'\n');
        }

        joined.extend_from_slice(bytes);
        // SAFETY: the next line starts after this line's NUL byte, still within the message.
        line = unsafe { line.add(bytes.len() + 1) };
    }
    CString::new(joined).ok()
}

/// Returns the last encountered cuDNN error message in the current thread as a string.
/// Inside the cuDNN library, the messages are stored in thread local buffers.
/// The error is cleared after this call retrieves it.
/// Native failure conversion captures this message in [`crate::error::Error::Cudnn`] automatically.
/// Uses the C++ frontend's 65,535-byte message capacity, including the terminator.
#[must_use]
pub fn last_error_string() -> String {
    // The capacity is 65,535 bytes, `u16::MAX`, so it converts to both length types without a check.
    let mut buffer = vec![0_i8; usize::from(u16::MAX)];
    // SAFETY: `buffer` is a live allocation of exactly the length passed, and cuDNN writes at most that many bytes, NUL-terminated.
    unsafe {
        sys::cudnnGetLastErrorString(buffer.as_mut_ptr(), sys::size_t::from(u16::MAX));
    }
    string_from_c_chars(&buffer)
}

/// cuDNN library functions perform extensive input argument checking before launching GPU kernels.
/// The last step is to verify that the GPU kernel actually started.
/// A kernel launch failure is reported as [`Status::ExecutionFailed`] by the corresponding wrapper call.
/// Typically, after a GPU kernel starts, no runtime checks are performed by the kernel itself - numerical results are simply written to output buffers.
///
/// When persistent spatial batch normalization is selected, the algorithm may encounter numerical overflows where ordinary spatial batch normalization performs just fine, albeit at a slower speed.
/// Invoke [`query_runtime_error`] to verify that numerical overflows did not occur during kernel execution.
/// Those issues are reported by the kernel that performs computations.
///
/// Use [`query_runtime_error`] in polling and blocking software control flows.
/// There are two polling modes ([`RuntimeErrorQueryMode::RawCode`] and [`RuntimeErrorQueryMode::NonBlocking`]) and one blocking mode [`RuntimeErrorQueryMode::Blocking`].
///
/// [`RuntimeErrorQueryMode::RawCode`] reads the error storage location regardless of the kernel completion status.
/// The kernel might not even start and the error storage (allocated per cuDNN handle) might be used by an earlier call.
///
/// [`RuntimeErrorQueryMode::NonBlocking`] checks whether all tasks in the configured cuDNN stream are completed.
/// If tasks in that stream are pending, [`query_runtime_error`] returns immediately with [`Status::RuntimeInProgress`].
/// Otherwise, it returns the remote kernel error code.
///
/// In blocking mode ([`RuntimeErrorQueryMode::Blocking`]), this waits for all tasks to drain from the configured cuDNN stream before reporting the remote kernel error code.
/// The blocking flavor can be further adjusted by calling [`Device::set_flags`](singe_cuda::device::Device::set_flags) with the [`ContextFlags::SCHEDULE_SPIN`](singe_cuda::context::ContextFlags::SCHEDULE_SPIN), [`ContextFlags::SCHEDULE_YIELD`](singe_cuda::context::ContextFlags::SCHEDULE_YIELD), or [`ContextFlags::SCHEDULE_BLOCKING_SYNC`](singe_cuda::context::ContextFlags::SCHEDULE_BLOCKING_SYNC) flag.
///
/// Do not use [`RuntimeErrorQueryMode::NonBlocking`] or [`RuntimeErrorQueryMode::Blocking`] if [`Context::set_stream`] is called between functions that report runtime kernel errors and [`query_runtime_error`].
///
/// The result can be `Ok`, [`Status::RuntimeInProgress`], or [`Status::RuntimeFpOverflow`].
/// The remote kernel error is automatically cleared by [`query_runtime_error`].
///
/// Use [`query_runtime_error`] after batch normalization calls that use persistent spatial mode.
///
/// # Errors
///
/// Returns an error if cuDNN rejects the input arguments, if the stream synchronization or query used to retrieve the remote error fails, or if the device cannot access zero-copy memory to report kernel errors.
pub fn query_runtime_error(context: &Context, mode: RuntimeErrorQueryMode) -> Result<Status> {
    let mut status = sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS;
    // SAFETY: `context` owns a live handle, `status` is a writable local, and the null tag is the only value cuDNN accepts for it.
    unsafe {
        try_ffi!(sys::cudnnQueryRuntimeError(
            context.as_raw(),
            &raw mut status,
            mode.into(),
            ptr::null_mut(),
        ))?;
    }

    Ok(Status::from(status))
}

impl_enum_display!(RuntimeErrorQueryMode, {
    RuntimeErrorQueryMode::RawCode => "CUDNN_ERRQUERY_RAWCODE",
    RuntimeErrorQueryMode::NonBlocking => "CUDNN_ERRQUERY_NONBLOCKING",
    RuntimeErrorQueryMode::Blocking => "CUDNN_ERRQUERY_BLOCKING",
});

impl_enum_display!(Severity, {
    Severity::Fatal => "CUDNN_SEV_FATAL",
    Severity::Error => "CUDNN_SEV_ERROR",
    Severity::Warning => "CUDNN_SEV_WARNING",
    Severity::Info => "CUDNN_SEV_INFO",
});
