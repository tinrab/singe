use std::{borrow::Cow, ffi::NulError, result};

use singe_core::{static_str_from_c_ptr, status_enum};
use singe_cuda::error::Error as CudaError;
use singe_cusparse_sys as sys;
use thiserror::Error;

status_enum! {
    /// [`Status`] is the cuSPARSE status code returned by cuSPARSE operations.
    pub enum Status: u32 => sys::cusparseStatus_t, unknown = "UNKNOWN_CUSPARSE_STATUS" {
        /// The operation completed successfully.
        Success = CUSPARSE_STATUS_SUCCESS,
        /// The cuSPARSE library was not initialized.
        /// Common causes include a missing [`Context::create`](crate::context::Context::create) call, a CUDA runtime error reported by cuSPARSE, or an error in the hardware setup.
        /// The error also applies when a matrix or vector descriptor is not initialized.
        NotInitialized = CUSPARSE_STATUS_NOT_INITIALIZED,
        /// Resource allocation failed inside the cuSPARSE library.
        /// Usually caused by a device memory allocation (`cudaMalloc()`) or host memory allocation failure.
        AllocFailed = CUSPARSE_STATUS_ALLOC_FAILED,
        /// An unsupported value or parameter was passed to the operation, such as a negative vector size.
        InvalidValue = CUSPARSE_STATUS_INVALID_VALUE,
        /// The operation requires a feature absent from the device architecture.
        ArchitectureMismatch = CUSPARSE_STATUS_ARCH_MISMATCH,
        /// An access to GPU memory space failed, which is usually caused by a
        /// failure to bind a texture or an invalid device pointer.
        /// To correct: check that all device pointers are valid and that no texture
        /// is bound that could interfere with the operation.
        MappingError = CUSPARSE_STATUS_MAPPING_ERROR,
        /// The GPU program failed to execute.
        /// A kernel launch failure on the GPU is a common cause.
        ExecutionFailed = CUSPARSE_STATUS_EXECUTION_FAILED,
        /// An internal cuSPARSE operation failed.
        /// Also check that memory passed to the operation is not released before the operation completes.
        InternalError = CUSPARSE_STATUS_INTERNAL_ERROR,
        /// The matrix type is not supported by this operation.
        /// Usually caused by passing an invalid matrix descriptor to the operation.
        /// Check that the fields in [`MatrixDescriptor`](crate::matrix::MatrixDescriptor) were set correctly.
        MatrixTypeNotSupported = CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
        /// A zero pivot was encountered during the factorization.
        /// The operation cannot proceed because a diagonal element is zero.
        /// To correct: ensure the input matrix is non-singular.
        ZeroPivot = CUSPARSE_STATUS_ZERO_PIVOT,
        /// The operation or data type combination is currently not supported.
        NotSupported = CUSPARSE_STATUS_NOT_SUPPORTED,
        /// The resources for the computation, such as GPU global or shared memory, are not sufficient to complete the operation.
        /// The error can also indicate that the current computation mode, such as sparse matrix index bit size, cannot handle the given input.
        InsufficientResources = CUSPARSE_STATUS_INSUFFICIENT_RESOURCES,
    }
}

impl Status {
    /// Library description of this status.
    #[must_use]
    pub fn description(self) -> &'static str {
        let Ok(raw) = sys::cusparseStatus_t::try_from(self) else {
            return "unknown cusparse error";
        };
        // SAFETY: `cusparseGetErrorString` returns null or a static NUL-terminated string.
        unsafe { static_str_from_c_ptr(sys::cusparseGetErrorString(raw), "unknown cusparse error") }
    }
}

/// CUDA and cuSPARSE status failures and errors validating wrapper inputs or native output.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// CUDA context, stream, or memory operation failed.
    #[error("cuda error: {0}")]
    Cuda(
        /// Underlying CUDA error.
        #[from]
        CudaError,
    ),

    /// cuSPARSE returned a non-success status.
    #[error("cusparse error ({code}): {message}")]
    Cusparse {
        /// Native cuSPARSE status code.
        code: Status,
        /// Description associated with the native status.
        message: String,
    },

    /// A string contains an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native operation returned a null handle where a live resource was required.
    #[error("unexpected null handle")]
    NullHandle,

    /// A count, size, dimension, or layout exceeds the supported representation or range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Value or layout whose required range was exceeded.
        name: Cow<'static, str>,
    },

    /// A borrowed storage range is smaller than the required extent.
    #[error("`{name}` length mismatch (expected at least {expected}, got {actual})")]
    LengthMismatch {
        /// Storage range being validated.
        name: Cow<'static, str>,
        /// Minimum required element or byte count.
        expected: usize,
        /// Available element or byte count.
        actual: usize,
    },

    /// A native descriptor attribute contains an unrecognized typed value.
    #[error("invalid value {value} for attribute `{name}`")]
    InvalidAttributeValue {
        /// Attribute being decoded.
        name: Cow<'static, str>,
        /// Unrecognized native attribute value.
        value: u32,
    },

    /// Input scalars use different host or device pointer modes.
    #[error("scalar pointer modes do not match")]
    ScalarPointerModeMismatch,

    /// An operand or plan is associated with a different execution context.
    #[error("plan belongs to a different cusparse context")]
    PlanContextMismatch,

    /// A supplied stream is associated with a different CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,
}

/// Result of a cuSPARSE wrapper operation.
pub type Result<T> = result::Result<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

impl From<sys::cusparseStatus_t> for Error {
    fn from(status: sys::cusparseStatus_t) -> Self {
        let code = Status::from(status);
        Self::from(code)
    }
}

impl From<Status> for Error {
    fn from(code: Status) -> Self {
        Self::Cusparse {
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

/// Converts a raw cuSPARSE status-returning expression into a crate [`Result`](crate::error::Result).
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
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
        let status = Status::from(sys::cusparseStatus_t::CUSPARSE_STATUS_INVALID_VALUE);
        assert_eq!(status, Status::InvalidValue);
        assert_eq!(
            status.raw(),
            sys::cusparseStatus_t::CUSPARSE_STATUS_INVALID_VALUE as u32
        );
    }

    #[test]
    fn test_status_display_uses_c_name() {
        assert_eq!(
            Status::InvalidValue.to_string(),
            "CUSPARSE_STATUS_INVALID_VALUE"
        );
    }

    #[test]
    fn test_unknown_status_keeps_raw_code() {
        let status = Status::Unknown(9999);
        assert_eq!(status.raw(), 9999);
        assert_eq!(status.to_string(), "UNKNOWN_CUSPARSE_STATUS(9999)");
    }
}
