use std::{borrow::Cow, ffi, result::Result as StdResult};

use singe_core::{static_str_from_c_ptr, status_enum};
use singe_cuda::error::Error as CudaError;
use singe_nccl_sys as sys;
use thiserror::Error;

/// An error from CUDA, NCCL, or validation of wrapper state and arguments.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// A CUDA operation failed.
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    /// An NCCL operation returned a failure status.
    #[error("nccl error ({code}): {message}")]
    Nccl {
        /// The NCCL status code.
        code: Status,
        /// A description of the NCCL status.
        message: String,
    },

    /// A string contains an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native communicator handle is null.
    #[error("unexpected null handle")]
    NullHandle,

    /// A native API returned or received an unexpected null pointer.
    #[error("unexpected null pointer")]
    NullPointer,

    /// An argument or computed quantity is outside its supported range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// The argument or quantity outside its supported range.
        name: Cow<'static, str>,
    },

    /// Buffer lengths or per-rank counts do not match the operation.
    #[error("`{name}` length mismatch")]
    LengthMismatch {
        /// The buffers or per-rank values with incompatible lengths.
        name: Cow<'static, str>,
    },

    /// An allocation cannot be transferred while other owners retain it.
    #[error("resource is shared and cannot be consumed")]
    OwnershipShared,

    /// An operation requires the device primary CUDA context.
    #[error("nccl requires the device primary cuda context")]
    UnsupportedContext,

    /// Resources belong to different CUDA contexts.
    #[error("resource belongs to a different cuda context")]
    ContextMismatch,

    /// An operation cannot run inside an open NCCL group.
    #[error("operation cannot run inside an open nccl group")]
    GroupActive,

    /// The stream belongs to another CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// The communicator state does not permit the operation.
    #[error("communicator is in {state} state")]
    InvalidCommunicatorState {
        /// The communicator state that prevented the operation.
        state: String,
    },

    /// The builder mode does not support the requested creation operation.
    #[error("communicator builder mode does not support {operation}")]
    InvalidBuilderMode {
        /// The operation incompatible with the selected builder mode.
        operation: String,
    },
}

/// An NCCL wrapper result with [`enum@Error`] as its error type.
pub type Result<T> = StdResult<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

impl From<sys::ncclResult_t> for Error {
    fn from(code: sys::ncclResult_t) -> Self {
        Self::from(Status::from(code))
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        Self::Nccl {
            code: status,
            message: status.description().to_string(),
        }
    }
}

impl From<ffi::NulError> for Error {
    fn from(_: ffi::NulError) -> Self {
        Self::InteriorNul
    }
}

status_enum! {
    /// Status codes returned by NCCL operations.
    pub enum Status: u32 => sys::ncclResult_t, unknown = "UNKNOWN_NCCL_STATUS" {
        /// Operation succeeded.
        Success = ncclSuccess,
        /// A call to a CUDA function failed.
        UnhandledCudaError = ncclUnhandledCudaError,
        /// A call to the system failed.
        SystemError = ncclSystemError,
        /// An internal check failed.
        /// Usually caused by an NCCL bug or memory corruption.
        InternalError = ncclInternalError,
        /// An argument has an invalid value.
        InvalidArgument = ncclInvalidArgument,
        /// The call to NCCL is incorrect.
        /// This usually reflects a programming error.
        InvalidUsage = ncclInvalidUsage,
        /// A call failed possibly due to a network error or a remote process exiting prematurely.
        RemoteError = ncclRemoteError,
        /// An NCCL operation on the communicator is being enqueued and progressed in the background.
        ///
        /// Whenever an operation returns an error (neither [`Status::Success`] nor [`Status::InProgress`]),
        /// NCCL may print a more detailed message when the `NCCL_DEBUG` environment variable is set to `WARN`.
        InProgress = ncclInProgress,
        /// The operation timed out.
        Timeout = ncclTimeout,
    }
}

impl Status {
    /// Library description of this status.
    #[must_use]
    pub fn description(self) -> &'static str {
        let Ok(raw) = sys::ncclResult_t::try_from(self) else {
            return "unknown nccl error";
        };
        // SAFETY: `ncclGetErrorString` returns null or a static NUL-terminated string.
        unsafe { static_str_from_c_ptr(sys::ncclGetErrorString(raw), "unknown nccl error") }
    }
}

/// Converts a raw NCCL status-returning expression into a crate [`Result`](crate::error::Result).
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_nccl_sys::ncclResult_t::ncclSuccess {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_converts_from_raw_code() {
        let status = Status::from(sys::ncclResult_t::ncclInvalidArgument);
        assert_eq!(status, Status::InvalidArgument);
        assert_eq!(status.raw(), sys::ncclResult_t::ncclInvalidArgument as u32);
    }

    #[test]
    fn test_status_display_uses_c_name() {
        assert_eq!(Status::InvalidArgument.to_string(), "ncclInvalidArgument");
    }

    #[test]
    fn test_unknown_status_keeps_raw_code() {
        let status = Status::Unknown(9999);
        assert_eq!(status.raw(), 9999);
        assert_eq!(status.to_string(), "UNKNOWN_NCCL_STATUS(9999)");
    }
}
