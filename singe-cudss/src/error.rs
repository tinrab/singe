use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_core::status_enum;
use singe_cuda::error::Error as CudaError;
use singe_cudss_sys as sys;
use thiserror::Error;

status_enum! {
    /// Host-side cuDSS status code.
    ///
    /// Device-side asynchronous failures are reported through [`DataParameter::Info`](crate::types::DataParameter::Info) rather than this status type.
    pub enum Status: u32 => sys::cudssStatus_t, unknown = "UNKNOWN_CUDSS_STATUS" {
        /// Operation completed successfully.
        Success = CUDSS_STATUS_SUCCESS,
        /// A required cuDSS object was not initialized.
        NotInitialized = CUDSS_STATUS_NOT_INITIALIZED,
        /// Resource allocation failed.
        AllocFailed = CUDSS_STATUS_ALLOC_FAILED,
        /// An invalid value was passed to cuDSS.
        InvalidValue = CUDSS_STATUS_INVALID_VALUE,
        /// The requested operation or parameter is not supported.
        NotSupported = CUDSS_STATUS_NOT_SUPPORTED,
        /// GPU program execution failed.
        ExecutionFailed = CUDSS_STATUS_EXECUTION_FAILED,
        /// An internal cuDSS operation failed.
        InternalError = CUDSS_STATUS_INTERNAL_ERROR,
        /// Iterative refinement did not reach the requested tolerance.
        ///
        /// The refined solution remains available.
        IrFailed = CUDSS_STATUS_IR_FAILED,
    }
}

impl Status {
    /// Returns a short lowercase description of the status.
    #[must_use]
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
        name: Cow<'static, str>,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// A numeric value could not be represented by the target API type.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the argument or field with the invalid value.
        name: Cow<'static, str>,
    },

    /// A byte count returned by cuDSS did not match the destination size.
    #[error("`{name}` size mismatch (expected {expected}, got {actual})")]
    SizeMismatch {
        /// Name of the queried value.
        name: Cow<'static, str>,
        /// Expected byte count.
        expected: usize,
        /// Actual byte count.
        actual: usize,
    },

    /// A descriptor dimension is negative or otherwise invalid.
    #[error("invalid matrix dimension `{name}`")]
    InvalidMatrixDimension {
        /// Name of the invalid dimension.
        name: Cow<'static, str>,
    },

    /// A data type is not valid for the requested operation.
    #[error("unsupported data type for `{name}`")]
    UnsupportedDataType {
        /// Name of the operation or field with the unsupported type.
        name: Cow<'static, str>,
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

    /// A native enum parameter value is not recognized by this version.
    #[error("unknown value {value} for `{name}`")]
    UnknownParameterValue {
        /// Name of the parameter whose enum value was not recognized.
        name: Cow<'static, str>,
        /// Unrecognized native enum discriminant.
        value: u32,
    },

    /// A raw handle cannot be extracted because wrapper objects still share it.
    #[error("resource is shared and cannot be consumed")]
    OwnershipShared,

    /// A buffer, matrix, or stream belongs to another CUDA context.
    #[error("resource belongs to a different cuda context")]
    ContextMismatch,

    /// A completed numeric solver phase reported a matrix failure.
    #[error("numerical solver failure (info {info})")]
    NumericalFailure {
        /// Nonzero cuDSS `Info` value reported after the numerical phase.
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

/// Converts a raw cuDSS status-returning expression into a crate [`Result`](crate::error::Result).
#[macro_export]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_converts_from_raw_code() {
        let status = Status::from(sys::cudssStatus_t::CUDSS_STATUS_INVALID_VALUE);
        assert_eq!(status, Status::InvalidValue);
        assert_eq!(
            status.raw(),
            sys::cudssStatus_t::CUDSS_STATUS_INVALID_VALUE as u32
        );
    }

    #[test]
    fn test_status_display_uses_c_name() {
        assert_eq!(
            Status::InvalidValue.to_string(),
            "CUDSS_STATUS_INVALID_VALUE"
        );
    }

    #[test]
    fn test_unknown_status_keeps_raw_code() {
        let status = Status::Unknown(9999);
        assert_eq!(status.raw(), 9999);
        assert_eq!(status.to_string(), "UNKNOWN_CUDSS_STATUS(9999)");
    }
}
