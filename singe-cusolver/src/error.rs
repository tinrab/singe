use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_core::status_enum;
use singe_cuda::error::Error as CudaError;
use singe_cusolver_sys as sys;
use thiserror::Error;

status_enum! {
    /// This is a status type returned by the library functions and it can have the following values.
    pub enum Status: u32 => sys::cusolverStatus_t, unknown = "UNKNOWN_CUSOLVER_STATUS" {
        /// The operation completed successfully.
        Success = CUSOLVER_STATUS_SUCCESS,
        /// The cuSOLVER library was not initialized.
        NotInitialized = CUSOLVER_STATUS_NOT_INITIALIZED,
        /// Resource allocation failed.
        AllocFailed = CUSOLVER_STATUS_ALLOC_FAILED,
        /// An invalid value was provided.
        InvalidValue = CUSOLVER_STATUS_INVALID_VALUE,
        /// The current device architecture is not supported.
        ArchitectureMismatch = CUSOLVER_STATUS_ARCH_MISMATCH,
        /// Memory or resource mapping failed.
        MappingError = CUSOLVER_STATUS_MAPPING_ERROR,
        /// Kernel execution failed.
        ExecutionFailed = CUSOLVER_STATUS_EXECUTION_FAILED,
        /// An internal cuSOLVER error occurred.
        InternalError = CUSOLVER_STATUS_INTERNAL_ERROR,
        /// The matrix type is not supported by the operation.
        MatrixTypeNotSupported = CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
        /// The requested operation is not supported.
        NotSupported = CUSOLVER_STATUS_NOT_SUPPORTED,
        /// A zero pivot was encountered.
        ZeroPivot = CUSOLVER_STATUS_ZERO_PIVOT,
        /// The cuSOLVER license is invalid.
        InvalidLicense = CUSOLVER_STATUS_INVALID_LICENSE,
        /// IRS parameters were not initialized.
        IrsParamsNotInitialized = CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED,
        /// IRS parameters are invalid.
        IrsParamsInvalid = CUSOLVER_STATUS_IRS_PARAMS_INVALID,
        /// IRS precision parameters are invalid.
        IrsParamsInvalidPrec = CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC,
        /// IRS refinement parameters are invalid.
        IrsParamsInvalidRefine = CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE,
        /// IRS max-iteration parameters are invalid.
        IrsParamsInvalidMaxIter = CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER,
        /// An internal IRS error occurred.
        IrsInternalError = CUSOLVER_STATUS_IRS_INTERNAL_ERROR,
        /// The requested IRS configuration is not supported.
        IrsNotSupported = CUSOLVER_STATUS_IRS_NOT_SUPPORTED,
        /// IRS configuration values are out of range.
        IrsOutOfRange = CUSOLVER_STATUS_IRS_OUT_OF_RANGE,
        /// GMRES refinement does not support the requested number of right-hand sides.
        IrsNrhsNotSupportedForRefineGmres = CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES,
        /// IRS info structures were not initialized.
        IrsInfosNotInitialized = CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED,
        /// IRS info structures were not destroyed before reuse.
        IrsInfosNotDestroyed = CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED,
        /// The IRS system matrix is singular.
        IrsMatrixSingular = CUSOLVER_STATUS_IRS_MATRIX_SINGULAR,
        /// The provided workspace is invalid.
        InvalidWorkspace = CUSOLVER_STATUS_INVALID_WORKSPACE,
    }
}

impl Status {
    /// Returns a lowercase description of the native status.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::NotInitialized => "library not initialized",
            Self::AllocFailed => "allocation failed",
            Self::InvalidValue => "invalid value",
            Self::ArchitectureMismatch => "architecture mismatch",
            Self::MappingError => "mapping error",
            Self::ExecutionFailed => "execution failed",
            Self::InternalError => "internal error",
            Self::MatrixTypeNotSupported => "matrix type not supported",
            Self::NotSupported => "not supported",
            Self::ZeroPivot => "zero pivot",
            Self::InvalidLicense => "invalid license",
            Self::IrsParamsNotInitialized => "irs params not initialized",
            Self::IrsParamsInvalid => "irs params invalid",
            Self::IrsParamsInvalidPrec => "irs params invalid precision",
            Self::IrsParamsInvalidRefine => "irs params invalid refinement",
            Self::IrsParamsInvalidMaxIter => "irs params invalid maxiter",
            Self::IrsInternalError => "irs internal error",
            Self::IrsNotSupported => "irs not supported",
            Self::IrsOutOfRange => "irs out of range",
            Self::IrsNrhsNotSupportedForRefineGmres => "irs nrhs not supported for refine gmres",
            Self::IrsInfosNotInitialized => "irs infos not initialized",
            Self::IrsInfosNotDestroyed => "irs infos not destroyed",
            Self::IrsMatrixSingular => "irs matrix singular",
            Self::InvalidWorkspace => "invalid workspace",
            Self::Unknown(_) => "unknown cusolver error",
        }
    }
}

/// Errors from cuSOLVER, CUDA, and checked solver argument validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// A CUDA operation failed.
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    /// cuSOLVER returned a non-success native status.
    #[error("cusolver error ({code}): {message}")]
    Cusolver {
        /// Native status returned by cuSOLVER.
        code: Status,
        /// Description of the native status.
        message: String,
    },

    /// A native string argument contains an interior nul byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native resource handle is unexpectedly null.
    #[error("unexpected null handle")]
    NullHandle,

    /// An argument or checked conversion is outside the supported range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the out-of-range argument or conversion.
        name: Cow<'static, str>,
    },

    /// A matrix leading dimension is too small for its logical rows.
    #[error("invalid matrix leading dimension")]
    InvalidLeadingDimension,

    /// Matrix dimensions, capacity, or checked storage-size arithmetic are invalid.
    #[error("invalid matrix shape")]
    InvalidMatrixShape,

    /// A vector does not have the required extent.
    #[error("invalid vector shape")]
    InvalidVectorShape,

    /// An eigensolver value or index interval is invalid.
    #[error("invalid eigensolver range")]
    InvalidEigenRange,

    /// The requested singular-vector modes are invalid for the operation.
    #[error("invalid svd mode")]
    InvalidSvdMode,

    /// The iterative-refinement precision configuration is unsupported or inconsistent.
    #[error("invalid precision configuration")]
    InvalidPrecisionConfiguration,

    /// The supplied workspace is smaller than the required capacity.
    #[error("insufficient workspace size: {actual} provided (required {required})")]
    InsufficientWorkspaceSize {
        /// Required workspace capacity in the units used by the operation.
        required: usize,
        /// Supplied workspace capacity in the units used by the operation.
        actual: usize,
    },

    /// The iterative-refinement residual history is invalid.
    #[error("invalid residual history")]
    InvalidResidualHistory,

    /// A stream belongs to another CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// A device allocation belongs to another CUDA context.
    #[error("memory belongs to a different cuda context")]
    MemoryContextMismatch,

    /// A completed solver operation reported a nonzero device status.
    #[error("solver failed with device status {info}")]
    ComputationFailed {
        /// Nonzero solver device status; its interpretation depends on the routine.
        info: i32,
    },

    /// A solver reported more eigenvalues than the permitted maximum.
    #[error("invalid eigenvalue count: {actual} exceeds {maximum}")]
    InvalidEigenvalueCount {
        /// Eigenvalue count reported by the completed solver.
        actual: usize,
        /// Maximum permitted eigenvalue count.
        maximum: usize,
    },
}

/// Result of a cuSOLVER wrapper operation.
pub type Result<T> = StdResult<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

impl From<sys::cusolverStatus_t> for Error {
    fn from(code: sys::cusolverStatus_t) -> Self {
        let status = Status::from(code);
        Self::from(status)
    }
}

impl From<Status> for Error {
    fn from(code: Status) -> Self {
        Self::Cusolver {
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

/// Converts a raw cuSOLVER status-returning expression into a crate [`Result`](crate::error::Result).
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
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
        let status = Status::from(sys::cusolverStatus_t::CUSOLVER_STATUS_INVALID_VALUE);
        assert_eq!(status, Status::InvalidValue);
        assert_eq!(
            status.raw(),
            sys::cusolverStatus_t::CUSOLVER_STATUS_INVALID_VALUE as u32
        );
    }

    #[test]
    fn test_status_display_uses_c_name() {
        assert_eq!(
            Status::InvalidValue.to_string(),
            "CUSOLVER_STATUS_INVALID_VALUE"
        );
    }

    #[test]
    fn test_unknown_status_keeps_raw_code() {
        let status = Status::Unknown(9999);
        assert_eq!(status.raw(), 9999);
        assert_eq!(status.to_string(), "UNKNOWN_CUSOLVER_STATUS(9999)");
    }
}
