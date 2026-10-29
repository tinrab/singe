//! Error and result types for safe cuRAND wrappers.

use std::{borrow::Cow, result::Result as StdResult};

use singe_core::status_enum;
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

    /// A size could not be represented by the native interface.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the size or count that could not be represented.
        name: Cow<'static, str>,
    },
}

/// Result alias used by `singe-curand`.
pub type Result<T> = StdResult<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

impl From<sys::curandStatus_t> for Error {
    fn from(status: sys::curandStatus_t) -> Self {
        Self::from(Status::from(status))
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        Self::Curand {
            code: status,
            message: status.description().to_string(),
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

impl Status {
    /// Static description of this status.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::VersionMismatch => "version mismatch",
            Self::NotInitialized => "not initialized",
            Self::AllocationFailed => "allocation failed",
            Self::TypeError => "type error",
            Self::OutOfRange => "out of range",
            Self::LengthNotMultiple => "length is not a multiple",
            Self::DoublePrecisionRequired => "double precision required",
            Self::LaunchFailure => "launch failure",
            Self::PreexistingFailure => "preexisting failure",
            Self::InitializationFailed => "initialization failed",
            Self::ArchitectureMismatch => "architecture mismatch",
            Self::InternalError => "internal error",
            Self::Unknown(_) => "unknown curand error",
        }
    }
}

status_enum! {
    /// Result status returned by cuRAND host API calls.
    pub enum Status: u32 => sys::curandStatus_t, unknown = "UNKNOWN_CURAND_STATUS" {
        /// The operation completed successfully.
        Success = CURAND_STATUS_SUCCESS,
        /// The cuRAND header version does not match the linked library version.
        VersionMismatch = CURAND_STATUS_VERSION_MISMATCH,
        /// The generator has not been initialized.
        NotInitialized = CURAND_STATUS_NOT_INITIALIZED,
        /// cuRAND could not allocate required memory.
        AllocationFailed = CURAND_STATUS_ALLOCATION_FAILED,
        /// The generator or argument has the wrong type.
        TypeError = CURAND_STATUS_TYPE_ERROR,
        /// An argument is outside the range accepted by cuRAND.
        OutOfRange = CURAND_STATUS_OUT_OF_RANGE,
        /// The requested length is not a multiple of the quasirandom dimension.
        LengthNotMultiple = CURAND_STATUS_LENGTH_NOT_MULTIPLE,
        /// The selected operation requires GPU double-precision support.
        DoublePrecisionRequired = CURAND_STATUS_DOUBLE_PRECISION_REQUIRED,
        /// A cuRAND kernel launch failed.
        LaunchFailure = CURAND_STATUS_LAUNCH_FAILURE,
        /// A previous CUDA failure was already pending when cuRAND was entered.
        PreexistingFailure = CURAND_STATUS_PREEXISTING_FAILURE,
        /// cuRAND could not initialize CUDA.
        InitializationFailed = CURAND_STATUS_INITIALIZATION_FAILED,
        /// The GPU architecture does not support the requested feature.
        ArchitectureMismatch = CURAND_STATUS_ARCH_MISMATCH,
        /// cuRAND reported an internal library error.
        InternalError = CURAND_STATUS_INTERNAL_ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_converts_from_raw_code() {
        let status = Status::from(sys::curandStatus_t::CURAND_STATUS_NOT_INITIALIZED);
        assert_eq!(status, Status::NotInitialized);
        assert_eq!(
            status.raw(),
            sys::curandStatus_t::CURAND_STATUS_NOT_INITIALIZED as u32
        );
    }

    #[test]
    fn test_status_display_uses_c_name() {
        assert_eq!(
            Status::NotInitialized.to_string(),
            "CURAND_STATUS_NOT_INITIALIZED"
        );
    }

    #[test]
    fn test_unknown_status_keeps_raw_code() {
        let status = Status::Unknown(9999);
        assert_eq!(status.raw(), 9999);
        assert_eq!(status.to_string(), "UNKNOWN_CURAND_STATUS(9999)");
    }
}
