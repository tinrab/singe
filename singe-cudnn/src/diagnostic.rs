//! cuDNN error reporting: callbacks, last-error messages and remote kernel errors.

use std::ptr;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display, string_from_c_chars};
use singe_cudnn_sys as sys;

use crate::{
    context::Context,
    error::{Result, Status},
    try_ffi,
};

/// Remote kernel error query mode used by [`query_runtime_error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum RuntimeErrorQueryMode {
    /// Read the error storage location regardless of the kernel completion status.
    RawCode = sys::cudnnErrQueryMode_t::CUDNN_ERRQUERY_RAWCODE as _,
    /// Report whether all tasks in the configured cuDNN stream have completed.
    /// If that is the case, report the remote kernel error code.
    NonBlocking = sys::cudnnErrQueryMode_t::CUDNN_ERRQUERY_NONBLOCKING as _,
    /// Wait for all tasks in the configured cuDNN stream before reporting the remote kernel error code.
    Blocking = sys::cudnnErrQueryMode_t::CUDNN_ERRQUERY_BLOCKING as _,
}

impl_enum_conversion!(sys::cudnnErrQueryMode_t, RuntimeErrorQueryMode);

/// Severity passed to the configured cuDNN logging callback.
/// The numerical values match the values accepted by the `CUDNN_LOGLEVEL_DBG` environment variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Severity {
    ///< Fatal error severity. @since cuDNN 9.0.0
    Fatal = sys::cudnnSeverity_t::CUDNN_SEV_FATAL as _,
    ///< Error severity. @since cuDNN 9.0.0
    Error = sys::cudnnSeverity_t::CUDNN_SEV_ERROR as _,
    ///< Warning severity. @since cuDNN 9.0.0
    Warning = sys::cudnnSeverity_t::CUDNN_SEV_WARNING as _,
    ///< Informational severity. @since cuDNN 9.0.0
    Info = sys::cudnnSeverity_t::CUDNN_SEV_INFO as _,
}

impl_enum_conversion!(sys::cudnnSeverity_t, Severity);

/// Native diagnostic record passed to a cuDNN logging callback.
pub type DebugInfo = sys::cudnnDebug_t;

/// Optional native cuDNN logging callback using the upstream ABI.
pub type Callback = sys::cudnnCallback_t;

/// Queries the internal state of cuDNN error reporting.
///
/// # Errors
///
/// Returns an error if cuDNN cannot query the callback state.
pub fn callback() -> Result<(u32, *mut (), Callback)> {
    let mut mask = 0u32;
    let mut user_data = ptr::null_mut();
    let mut callback = None;
    // SAFETY: `cudnnGetCallback` receives valid output locations.
    unsafe {
        try_ffi!(sys::cudnnGetCallback(
            &raw mut mask,
            &raw mut user_data,
            &raw mut callback,
        ))?;
    }
    Ok((mask, user_data.cast(), callback))
}

/// Sets the internal state of cuDNN error reporting.
///
/// # Safety
///
/// `callback`, if present, must follow the cuDNN callback ABI. `user_data`
/// must remain valid for any callback invocation that may use it.
///
/// # Errors
///
/// Returns an error if cuDNN rejects the callback state.
pub unsafe fn set_callback(mask: u32, user_data: *mut (), callback: Callback) -> Result<()> {
    // SAFETY: `cudnnSetCallback` receives only plain values.
    unsafe { try_ffi!(sys::cudnnSetCallback(mask, user_data.cast(), callback)) }
}

/// Returns the last encountered cuDNN error message in the current thread as a string.
/// Inside the cuDNN library, the messages are stored in thread local buffers.
/// The error is cleared after this call retrieves it.
/// Native failure conversion captures this message in [`crate::error::Error::Cudnn`] automatically.
/// Uses the C++ frontend's 65,535-byte message capacity, including the terminator.
#[must_use]
pub fn last_error_string() -> String {
    let mut buffer = vec![0_i8; 65_535];
    // SAFETY: `cudnnGetLastErrorString` receives valid output locations.
    unsafe {
        sys::cudnnGetLastErrorString(buffer.as_mut_ptr(), buffer.len() as _);
    }
    string_from_c_chars(&buffer)
}

/// cuDNN library functions perform extensive input argument checking before launching GPU kernels.
/// The last step is to verify that the GPU kernel actually started.
/// A kernel launch failure is reported as [`Status::ExecutionFailed`] by the corresponding wrapper call.
/// Typically, after a GPU kernel starts, no runtime checks are performed by the kernel itself - numerical results are simply written to output buffers.
///
/// When persistent spatial batch normalization is selected, the algorithm may encounter
/// numerical overflows where ordinary spatial batch normalization performs just fine, albeit at a
/// slower speed.
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
/// Use [`query_runtime_error`] after batch normalization calls that use
/// persistent spatial mode.
///
/// # Errors
///
/// Returns an error if cuDNN rejects the input arguments, if the stream
/// synchronization or query used to retrieve the remote error fails, or if the
/// device cannot access zero-copy memory to report kernel errors.
pub fn query_runtime_error(context: &Context, mode: RuntimeErrorQueryMode) -> Result<Status> {
    let mut status = sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS;
    // SAFETY: `cudnnQueryRuntimeError` receives live handles owned by the wrappers and valid output locations.
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
