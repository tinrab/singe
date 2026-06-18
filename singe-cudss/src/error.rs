use std::{ffi::NulError, fmt};

use singe_cuda::error::Error as CudaError;
use singe_cudss_sys as sys;
use thiserror::Error;

/// Host-side cuDSS status code.
///
/// Device-side asynchronous failures are reported through [`DataParameter::Info`](crate::types::DataParameter::Info) rather than this status type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    /// Operation completed successfully.
    Success,
    /// A required cuDSS object was not initialized.
    NotInitialized,
    /// Resource allocation failed.
    AllocFailed,
    /// An invalid value was passed to cuDSS.
    InvalidValue,
    /// The requested operation or parameter is not supported.
    NotSupported,
    /// GPU program execution failed.
    ExecutionFailed,
    /// An internal cuDSS operation failed.
    InternalError,
    /// Iterative refinement did not reach the requested tolerance.
    ///
    /// The refined solution remains available.
    IrFailed,
    /// A status code not known to this wrapper.
    Unknown(u32),
}

impl Status {
    /// Returns a short lowercase description of the status.
    pub const fn description(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::NotInitialized => "library not initialized",
            Self::AllocFailed => "allocation failed",
            Self::InvalidValue => "invalid value",
            Self::NotSupported => "not supported",
            Self::ExecutionFailed => "execution failed",
            Self::InternalError => "internal error",
            Self::IrFailed => "iterative refinement failed",
            Self::Unknown(_) => "unknown cudss error",
        }
    }

    /// Returns the raw cuDSS status code.
    pub const fn raw(self) -> u32 {
        match self {
            Self::Success => sys::cudssStatus_t::CUDSS_STATUS_SUCCESS as _,
            Self::NotInitialized => sys::cudssStatus_t::CUDSS_STATUS_NOT_INITIALIZED as _,
            Self::AllocFailed => sys::cudssStatus_t::CUDSS_STATUS_ALLOC_FAILED as _,
            Self::InvalidValue => sys::cudssStatus_t::CUDSS_STATUS_INVALID_VALUE as _,
            Self::NotSupported => sys::cudssStatus_t::CUDSS_STATUS_NOT_SUPPORTED as _,
            Self::ExecutionFailed => sys::cudssStatus_t::CUDSS_STATUS_EXECUTION_FAILED as _,
            Self::InternalError => sys::cudssStatus_t::CUDSS_STATUS_INTERNAL_ERROR as _,
            Self::IrFailed => sys::cudssStatus_t::CUDSS_STATUS_IR_FAILED as _,
            Self::Unknown(code) => code,
        }
    }
}

impl TryFrom<u32> for Status {
    type Error = u32;

    fn try_from(code: u32) -> std::result::Result<Self, Self::Error> {
        match code {
            code if code == sys::cudssStatus_t::CUDSS_STATUS_SUCCESS as u32 => Ok(Self::Success),
            code if code == sys::cudssStatus_t::CUDSS_STATUS_NOT_INITIALIZED as u32 => {
                Ok(Self::NotInitialized)
            }
            code if code == sys::cudssStatus_t::CUDSS_STATUS_ALLOC_FAILED as u32 => {
                Ok(Self::AllocFailed)
            }
            code if code == sys::cudssStatus_t::CUDSS_STATUS_INVALID_VALUE as u32 => {
                Ok(Self::InvalidValue)
            }
            code if code == sys::cudssStatus_t::CUDSS_STATUS_NOT_SUPPORTED as u32 => {
                Ok(Self::NotSupported)
            }
            code if code == sys::cudssStatus_t::CUDSS_STATUS_EXECUTION_FAILED as u32 => {
                Ok(Self::ExecutionFailed)
            }
            code if code == sys::cudssStatus_t::CUDSS_STATUS_INTERNAL_ERROR as u32 => {
                Ok(Self::InternalError)
            }
            code if code == sys::cudssStatus_t::CUDSS_STATUS_IR_FAILED as u32 => Ok(Self::IrFailed),
            code => Err(code),
        }
    }
}

impl From<sys::cudssStatus_t> for Status {
    fn from(status: sys::cudssStatus_t) -> Self {
        Self::try_from(status as u32).unwrap_or_else(Self::Unknown)
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("CUDSS_STATUS_SUCCESS"),
            Self::NotInitialized => f.write_str("CUDSS_STATUS_NOT_INITIALIZED"),
            Self::AllocFailed => f.write_str("CUDSS_STATUS_ALLOC_FAILED"),
            Self::InvalidValue => f.write_str("CUDSS_STATUS_INVALID_VALUE"),
            Self::NotSupported => f.write_str("CUDSS_STATUS_NOT_SUPPORTED"),
            Self::ExecutionFailed => f.write_str("CUDSS_STATUS_EXECUTION_FAILED"),
            Self::InternalError => f.write_str("CUDSS_STATUS_INTERNAL_ERROR"),
            Self::IrFailed => f.write_str("CUDSS_STATUS_IR_FAILED"),
            Self::Unknown(code) => write!(f, "UNKNOWN_CUDSS_STATUS({code})"),
        }
    }
}

/// Error type returned by safe cuDSS wrappers.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// CUDA wrapper error.
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    /// cuDSS returned a non-success status.
    #[error("cudss error ({code}): {message}")]
    Cudss {
        /// Raw cuDSS status mapped into the wrapper status type.
        code: Status,
        /// Human-readable status description.
        message: String,
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
        name: String,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// A numeric value could not be represented by the target API type.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the argument or field with the invalid value.
        name: String,
    },

    /// A byte count returned by cuDSS did not match the destination size.
    #[error("`{name}` size mismatch (expected {expected}, got {actual})")]
    SizeMismatch {
        /// Name of the queried value.
        name: String,
        /// Expected byte count.
        expected: usize,
        /// Actual byte count.
        actual: usize,
    },

    /// A descriptor dimension is negative or otherwise invalid.
    #[error("invalid matrix dimension `{name}`")]
    InvalidMatrixDimension {
        /// Name of the invalid dimension.
        name: String,
    },

    /// A data type is not valid for the requested operation.
    #[error("unsupported data type for `{name}`")]
    UnsupportedDataType {
        /// Name of the operation or field with the unsupported type.
        name: String,
    },

    /// A dense matrix leading dimension is invalid for its shape and layout.
    #[error("invalid leading dimension")]
    InvalidLeadingDimension,

    /// Device memory handler allocator name exceeded cuDSS' fixed buffer.
    #[error("allocator name is too long (max {max} bytes, got {actual})")]
    AllocatorNameTooLong {
        /// Maximum accepted byte length.
        max: usize,
        /// Actual byte length.
        actual: usize,
    },

    /// A numeric value returned by cuDSS could not be represented by the target
    /// Rust type.
    #[error("integer conversion failed for `{0}`")]
    IntegerConversion(String),

    /// A raw handle cannot be extracted because wrapper objects still share it.
    #[error("handle is shared")]
    HandleShared,

    /// A CUDA stream belongs to a different CUDA context than the cuDSS handle.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,
}

/// Result type used by `singe-cudss`.
pub type Result<T> = std::result::Result<T, Error>;

impl From<sys::cudssStatus_t> for Error {
    fn from(status: sys::cudssStatus_t) -> Self {
        let code = Status::from(status);
        Self::Cudss {
            code,
            message: code.description().to_string(),
        }
    }
}

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

#[macro_export]
/// Converts a raw cuDSS status-returning expression into a wrapper result.
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_cudss_sys::cudssStatus_t::CUDSS_STATUS_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}
