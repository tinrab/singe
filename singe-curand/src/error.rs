//! Error and result types for safe cuRAND wrappers.

use std::{borrow::Cow, result::Result as StdResult};

use singe_cuda::error::Error as CudaError;
use singe_curand_sys as sys;
use thiserror::Error;

singe_core::wrapper_status! {
    library = "curand",
    error = Error::Curand,
    raw = sys::curandStatus_t,
    success = CURAND_STATUS_SUCCESS,
    absence = sys::take_absence,
    unknown = "UNKNOWN_CURAND_STATUS",
    /// Result status returned by cuRAND host API calls.
    pub enum Status: u32 {
        /// The operation completed successfully.
        Success = CURAND_STATUS_SUCCESS => "success",
        /// The cuRAND header version does not match the linked library version.
        VersionMismatch = CURAND_STATUS_VERSION_MISMATCH => "version mismatch",
        /// The generator has not been initialized.
        NotInitialized = CURAND_STATUS_NOT_INITIALIZED => "not initialized",
        /// cuRAND could not allocate required memory.
        AllocationFailed = CURAND_STATUS_ALLOCATION_FAILED => "allocation failed",
        /// The generator or argument has the wrong type.
        TypeError = CURAND_STATUS_TYPE_ERROR => "type error",
        /// An argument is outside the range accepted by cuRAND.
        OutOfRange = CURAND_STATUS_OUT_OF_RANGE => "out of range",
        /// The requested length is not a multiple of the quasirandom dimension.
        LengthNotMultiple = CURAND_STATUS_LENGTH_NOT_MULTIPLE => "length is not a multiple",
        /// The selected operation requires GPU double-precision support.
        DoublePrecisionRequired = CURAND_STATUS_DOUBLE_PRECISION_REQUIRED => "double precision required",
        /// A cuRAND kernel launch failed.
        LaunchFailure = CURAND_STATUS_LAUNCH_FAILURE => "launch failure",
        /// A previous CUDA failure was already pending when cuRAND was entered.
        PreexistingFailure = CURAND_STATUS_PREEXISTING_FAILURE => "preexisting failure",
        /// cuRAND could not initialize CUDA.
        InitializationFailed = CURAND_STATUS_INITIALIZATION_FAILED => "initialization failed",
        /// The GPU architecture does not support the requested feature.
        ArchitectureMismatch = CURAND_STATUS_ARCH_MISMATCH => "architecture mismatch",
        /// cuRAND reported an internal library error.
        InternalError = CURAND_STATUS_INTERNAL_ERROR => "internal error",
    }
}

/// Error returned by safe cuRAND wrapper operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// cuRAND or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// CUDA runtime or driver wrapper error.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// cuRAND reported an unsuccessful status.
    #[error("curand error ({code}): {}", code.description())]
    Curand {
        /// Typed cuRAND status code.
        code: Status,
    },

    /// cuRAND returned a null handle where a valid handle was expected.
    #[error("unexpected null handle")]
    NullHandle,

    /// cuRAND returned a null pointer where a valid pointer was expected.
    #[error("`{name}` is null")]
    NullPointer {
        /// Name of the pointer.
        name: Cow<'static, str>,
    },

    /// A supplied stream is associated with a different CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// A native enum value has no typed variant in this crate.
    #[error("unknown `{name}` value {value}")]
    UnknownEnumValue {
        /// Name of the typed enum.
        name: Cow<'static, str>,
        /// Unrecognized native value.
        value: i64,
    },

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

impl From<singe_core::UnknownEnumValue> for Error {
    fn from(error: singe_core::UnknownEnumValue) -> Self {
        Self::UnknownEnumValue {
            name: error.name.into(),
            value: error.value,
        }
    }
}

// Converts a raw cuRAND status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
