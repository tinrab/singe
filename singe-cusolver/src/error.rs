//! Native statuses and checked solver errors.

use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_cuda::error::Error as CudaError;
use singe_cusolver_sys as sys;
use thiserror::Error;

singe_core::wrapper_status! {
    library = "cusolver",
    error = Error::Cusolver,
    raw = sys::cusolverStatus_t,
    success = CUSOLVER_STATUS_SUCCESS,
    absence = sys::take_absence,
    unknown = "UNKNOWN_CUSOLVER_STATUS",
    /// This is a status type returned by the library functions and it can have the following values.
    pub enum Status: u32 {
        /// The operation completed successfully.
        Success = CUSOLVER_STATUS_SUCCESS => "success",
        /// The cuSOLVER library was not initialized.
        NotInitialized = CUSOLVER_STATUS_NOT_INITIALIZED => "library not initialized",
        /// Resource allocation failed.
        AllocFailed = CUSOLVER_STATUS_ALLOC_FAILED => "allocation failed",
        /// An invalid value was provided.
        InvalidValue = CUSOLVER_STATUS_INVALID_VALUE => "invalid value",
        /// The current device architecture is not supported.
        ArchitectureMismatch = CUSOLVER_STATUS_ARCH_MISMATCH => "architecture mismatch",
        /// Memory or resource mapping failed.
        MappingError = CUSOLVER_STATUS_MAPPING_ERROR => "mapping error",
        /// Kernel execution failed.
        ExecutionFailed = CUSOLVER_STATUS_EXECUTION_FAILED => "execution failed",
        /// An internal cuSOLVER error occurred.
        InternalError = CUSOLVER_STATUS_INTERNAL_ERROR => "internal error",
        /// The matrix type is not supported by the operation.
        MatrixTypeNotSupported = CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED => "matrix type not supported",
        /// The requested operation is not supported.
        NotSupported = CUSOLVER_STATUS_NOT_SUPPORTED => "not supported",
        /// A zero pivot was encountered.
        ZeroPivot = CUSOLVER_STATUS_ZERO_PIVOT => "zero pivot",
        /// The cuSOLVER license is invalid.
        InvalidLicense = CUSOLVER_STATUS_INVALID_LICENSE => "invalid license",
        /// IRS parameters were not initialized.
        IrsParamsNotInitialized = CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED => "irs params not initialized",
        /// IRS parameters are invalid.
        IrsParamsInvalid = CUSOLVER_STATUS_IRS_PARAMS_INVALID => "irs params invalid",
        /// IRS precision parameters are invalid.
        IrsParamsInvalidPrec = CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC => "irs params invalid precision",
        /// IRS refinement parameters are invalid.
        IrsParamsInvalidRefine = CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE => "irs params invalid refinement",
        /// IRS max-iteration parameters are invalid.
        IrsParamsInvalidMaxIter = CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER => "irs params invalid maxiter",
        /// An internal IRS error occurred.
        IrsInternalError = CUSOLVER_STATUS_IRS_INTERNAL_ERROR => "irs internal error",
        /// The requested IRS configuration is not supported.
        IrsNotSupported = CUSOLVER_STATUS_IRS_NOT_SUPPORTED => "irs not supported",
        /// IRS configuration values are out of range.
        IrsOutOfRange = CUSOLVER_STATUS_IRS_OUT_OF_RANGE => "irs out of range",
        /// GMRES refinement does not support the requested number of right-hand sides.
        IrsNrhsNotSupportedForRefineGmres = CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES => "irs nrhs not supported for refine gmres",
        /// IRS info structures were not initialized.
        IrsInfosNotInitialized = CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED => "irs infos not initialized",
        /// IRS info structures were not destroyed before reuse.
        IrsInfosNotDestroyed = CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED => "irs infos not destroyed",
        /// The IRS system matrix is singular.
        IrsMatrixSingular = CUSOLVER_STATUS_IRS_MATRIX_SINGULAR => "irs matrix singular",
        /// The provided workspace is invalid.
        InvalidWorkspace = CUSOLVER_STATUS_INVALID_WORKSPACE => "invalid workspace",
    }
}

/// Errors from cuSOLVER, CUDA, and checked solver argument validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// cuSOLVER or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// A CUDA operation failed.
    #[error(transparent)]
    Cuda(CudaError),

    /// cuSOLVER returned a non-success native status.
    #[error("cusolver error ({code}): {}", code.description())]
    Cusolver {
        /// Native status returned by cuSOLVER.
        code: Status,
    },

    /// A native string argument contains an interior nul byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native resource handle is unexpectedly null.
    #[error("unexpected null handle")]
    NullHandle,

    /// A native enum value has no typed variant in this crate.
    #[error("unknown `{name}` value {value}")]
    UnknownEnumValue {
        /// Name of the typed enum.
        name: Cow<'static, str>,
        /// Unrecognized native value.
        value: i64,
    },

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

    /// A buffer or workspace is smaller than the operation requires.
    #[error("`{name}` is too small (requires {required}, got {actual})")]
    BufferTooSmall {
        /// Name of the buffer.
        name: Cow<'static, str>,
        /// Required capacity in the units used by the operation.
        required: usize,
        /// Supplied capacity in the units used by the operation.
        actual: usize,
    },

    /// The iterative-refinement residual history is invalid.
    #[error("invalid residual history")]
    InvalidResidualHistory,

    /// Residual history was read from a report that did not request it before the solve.
    #[error("residual history was not requested")]
    ResidualHistoryNotRequested,

    /// A stream belongs to another CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// A completed solver operation reported a nonzero device status.
    #[error("`{operation}` failed with {info}")]
    OperationFailed {
        /// Name of the solver operation.
        operation: Cow<'static, str>,
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

impl From<CudaError> for Error {
    fn from(error: CudaError) -> Self {
        match error {
            CudaError::StreamContextMismatch => Self::StreamContextMismatch,
            error => Self::Cuda(error),
        }
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

// Converts a raw cuSOLVER status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
