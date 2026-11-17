use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_cuda::error::Error as CudaError;
use thiserror::Error;

use crate::sys;

singe_core::wrapper_status! {
    library = "cudss",
    error = Error::Cudss,
    raw = sys::cudssStatus_t,
    success = CUDSS_STATUS_SUCCESS,
    absence = sys::take_absence,
    unknown = "UNKNOWN_CUDSS_STATUS",
    /// Host-side cuDSS status code.
    ///
    /// Device-side asynchronous failures are reported through [`DataParameter::Info`](crate::types::DataParameter::Info) rather than this status type.
    pub enum Status: u32 {
        /// Operation completed successfully.
        Success = CUDSS_STATUS_SUCCESS => "success",
        /// A required cuDSS object was not initialized.
        NotInitialized = CUDSS_STATUS_NOT_INITIALIZED => "library not initialized",
        /// Resource allocation failed.
        AllocFailed = CUDSS_STATUS_ALLOC_FAILED => "allocation failed",
        /// An invalid value was passed to cuDSS.
        InvalidValue = CUDSS_STATUS_INVALID_VALUE => "invalid value",
        /// The requested operation or parameter is not supported.
        NotSupported = CUDSS_STATUS_NOT_SUPPORTED => "not supported",
        /// GPU program execution failed.
        ExecutionFailed = CUDSS_STATUS_EXECUTION_FAILED => "execution failed",
        /// An internal cuDSS operation failed.
        InternalError = CUDSS_STATUS_INTERNAL_ERROR => "internal error",
        /// Iterative refinement did not reach the requested tolerance.
        ///
        /// The refined solution remains available.
        IrFailed = CUDSS_STATUS_IR_FAILED => "iterative refinement failed",
    }
}

/// Error type returned by safe cuDSS wrappers.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// cuDSS or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// CUDA wrapper error.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// cuDSS returned a non-success status.
    #[error("cudss error ({code}): {}", code.description())]
    Cudss {
        /// Native cuDSS status.
        code: Status,
    },

    /// A string passed to a C API contained an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// cuDSS returned a null handle after a successful creation status.
    #[error("unexpected null handle")]
    NullHandle,

    /// A provided slice length does not match the length required by cuDSS.
    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        /// Name of the argument or field with the mismatch.
        name: Cow<'static, str>,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// A buffer or table is shorter than the operation requires.
    #[error("`{name}` is too small (requires {required}, got {actual})")]
    BufferTooSmall {
        /// Name of the buffer or table.
        name: Cow<'static, str>,
        /// Required length.
        required: usize,
        /// Supplied length.
        actual: usize,
    },

    /// A native enum value has no typed variant in this crate.
    #[error("unknown `{name}` value {value}")]
    UnknownEnumValue {
        /// Name of the typed enum or parameter.
        name: Cow<'static, str>,
        /// Unrecognized native value.
        value: i64,
    },

    /// A value is outside the range the target API type or operation accepts.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the argument or field with the invalid value.
        name: Cow<'static, str>,
    },

    /// A byte count returned by cuDSS did not match the destination size.
    #[error("`{name}` size mismatch (expected {expected} bytes, got {actual})")]
    SizeMismatch {
        /// Name of the queried value.
        name: Cow<'static, str>,
        /// Expected byte count.
        expected: usize,
        /// Actual byte count.
        actual: usize,
    },

    /// A data type is not valid for the requested operation.
    #[error("unsupported data type for `{name}`")]
    UnsupportedDataType {
        /// Name of the operation or field with the unsupported type.
        name: Cow<'static, str>,
    },

    /// Device memory handler allocator name exceeded cuDSS' fixed buffer.
    #[error("allocator name is too long (max {max} bytes, got {actual})")]
    AllocatorNameTooLong {
        /// Maximum accepted byte length.
        max: usize,
        /// Actual byte length.
        actual: usize,
    },

    /// A device list does not match the CUDA device an operation runs on.
    #[error("`{name}` does not match the cuda device")]
    DeviceMismatch {
        /// Name of the device list or device.
        name: Cow<'static, str>,
    },

    /// A library object belongs to a different cuDSS context.
    #[error("object belongs to a different cudss context")]
    ContextMismatch,

    /// A supplied stream is associated with a different CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// A completed solver phase reported a nonzero `Info` value.
    #[error("`{operation}` failed with {info}")]
    OperationFailed {
        /// Name of the solver operation.
        operation: Cow<'static, str>,
        /// Nonzero cuDSS `Info` value reported after the operation.
        info: i32,
    },
}

/// Result type used by `singe-cudss`.
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

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

// Converts a raw cuDSS status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
