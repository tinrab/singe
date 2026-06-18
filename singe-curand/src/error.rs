//! Error and result types for safe cuRAND wrappers.

use crate::types::Status;
use singe_cuda::error::Error as CudaError;
use singe_curand_sys as sys;
use thiserror::Error;

/// Error returned by safe cuRAND wrapper operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// CUDA runtime or driver wrapper error.
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    /// cuRAND reported an unsuccessful status.
    #[error("curand error ({code}): {message}")]
    Curand {
        /// Typed cuRAND status code.
        code: Status,
        /// Human-readable status description.
        message: String,
    },

    /// cuRAND returned a null handle where a valid handle was expected.
    #[error("unexpected null handle")]
    NullHandle,

    /// A device-only operation was requested for a host generator.
    #[error("generator is not a device generator")]
    NotDeviceGenerator,

    /// A host-only operation was requested for a device generator.
    #[error("generator is not a host generator")]
    NotHostGenerator,

    /// A stream from another CUDA context was passed to a generator.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,
}

/// Result alias used by `singe-curand`.
pub type Result<T> = std::result::Result<T, Error>;

impl From<sys::curandStatus_t> for Error {
    fn from(status: sys::curandStatus_t) -> Self {
        Self::from(Status::from(status))
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        Self::Curand {
            code: status,
            message: status_message(status),
        }
    }
}

/// Converts a raw cuRAND status-returning expression into a crate [`Result`].
///
/// The expression is evaluated once. `CURAND_STATUS_SUCCESS` becomes `Ok(())`;
/// every other status becomes [`Error::Curand`].
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_curand_sys::curandStatus_t::CURAND_STATUS_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}

fn status_message(status: Status) -> String {
    let Ok(raw) = sys::curandStatus_t::try_from(status) else {
        return String::from("unknown curand error");
    };

    match raw {
        sys::curandStatus_t::CURAND_STATUS_SUCCESS => "success",
        sys::curandStatus_t::CURAND_STATUS_VERSION_MISMATCH => "version mismatch",
        sys::curandStatus_t::CURAND_STATUS_NOT_INITIALIZED => "not initialized",
        sys::curandStatus_t::CURAND_STATUS_ALLOCATION_FAILED => "allocation failed",
        sys::curandStatus_t::CURAND_STATUS_TYPE_ERROR => "type error",
        sys::curandStatus_t::CURAND_STATUS_OUT_OF_RANGE => "out of range",
        sys::curandStatus_t::CURAND_STATUS_LENGTH_NOT_MULTIPLE => "length is not a multiple",
        sys::curandStatus_t::CURAND_STATUS_DOUBLE_PRECISION_REQUIRED => "double precision required",
        sys::curandStatus_t::CURAND_STATUS_LAUNCH_FAILURE => "launch failure",
        sys::curandStatus_t::CURAND_STATUS_PREEXISTING_FAILURE => "preexisting failure",
        sys::curandStatus_t::CURAND_STATUS_INITIALIZATION_FAILED => "initialization failed",
        sys::curandStatus_t::CURAND_STATUS_ARCH_MISMATCH => "architecture mismatch",
        sys::curandStatus_t::CURAND_STATUS_INTERNAL_ERROR => "internal error",
    }
    .to_string()
}
