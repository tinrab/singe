use std::{ffi::NulError, fmt};

use singe_cuda::error::Error as CudaError;
use singe_cusolver_sys as sys;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    /// The operation completed successfully.
    Success,
    /// The cuSOLVER library was not initialized.
    NotInitialized,
    /// Resource allocation failed.
    AllocFailed,
    /// An invalid value was provided.
    InvalidValue,
    /// The current device architecture is not supported.
    ArchitectureMismatch,
    /// Memory or resource mapping failed.
    MappingError,
    /// Kernel execution failed.
    ExecutionFailed,
    /// An internal cuSOLVER error occurred.
    InternalError,
    /// The matrix type is not supported by the operation.
    MatrixTypeNotSupported,
    /// The requested operation is not supported.
    NotSupported,
    /// A zero pivot was encountered.
    ZeroPivot,
    /// The cuSOLVER license is invalid.
    InvalidLicense,
    /// IRS parameters were not initialized.
    IrsParamsNotInitialized,
    /// IRS parameters are invalid.
    IrsParamsInvalid,
    /// IRS precision parameters are invalid.
    IrsParamsInvalidPrec,
    /// IRS refinement parameters are invalid.
    IrsParamsInvalidRefine,
    /// IRS max-iteration parameters are invalid.
    IrsParamsInvalidMaxIter,
    /// An internal IRS error occurred.
    IrsInternalError,
    /// The requested IRS configuration is not supported.
    IrsNotSupported,
    /// IRS configuration values are out of range.
    IrsOutOfRange,
    /// GMRES refinement does not support the requested number of right-hand sides.
    IrsNrhsNotSupportedForRefineGmres,
    /// IRS info structures were not initialized.
    IrsInfosNotInitialized,
    /// IRS info structures were not destroyed before reuse.
    IrsInfosNotDestroyed,
    /// The IRS system matrix is singular.
    IrsMatrixSingular,
    /// The provided workspace is invalid.
    InvalidWorkspace,
    /// A status code not known to this version of Singe.
    Unknown(u32),
}

impl Status {
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

    pub const fn raw(self) -> u32 {
        match self {
            Self::Success => sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS as _,
            Self::NotInitialized => sys::cusolverStatus_t::CUSOLVER_STATUS_NOT_INITIALIZED as _,
            Self::AllocFailed => sys::cusolverStatus_t::CUSOLVER_STATUS_ALLOC_FAILED as _,
            Self::InvalidValue => sys::cusolverStatus_t::CUSOLVER_STATUS_INVALID_VALUE as _,
            Self::ArchitectureMismatch => sys::cusolverStatus_t::CUSOLVER_STATUS_ARCH_MISMATCH as _,
            Self::MappingError => sys::cusolverStatus_t::CUSOLVER_STATUS_MAPPING_ERROR as _,
            Self::ExecutionFailed => sys::cusolverStatus_t::CUSOLVER_STATUS_EXECUTION_FAILED as _,
            Self::InternalError => sys::cusolverStatus_t::CUSOLVER_STATUS_INTERNAL_ERROR as _,
            Self::MatrixTypeNotSupported => {
                sys::cusolverStatus_t::CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED as _
            }
            Self::NotSupported => sys::cusolverStatus_t::CUSOLVER_STATUS_NOT_SUPPORTED as _,
            Self::ZeroPivot => sys::cusolverStatus_t::CUSOLVER_STATUS_ZERO_PIVOT as _,
            Self::InvalidLicense => sys::cusolverStatus_t::CUSOLVER_STATUS_INVALID_LICENSE as _,
            Self::IrsParamsNotInitialized => {
                sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED as _
            }
            Self::IrsParamsInvalid => {
                sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_PARAMS_INVALID as _
            }
            Self::IrsParamsInvalidPrec => {
                sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC as _
            }
            Self::IrsParamsInvalidRefine => {
                sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE as _
            }
            Self::IrsParamsInvalidMaxIter => {
                sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER as _
            }
            Self::IrsInternalError => {
                sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_INTERNAL_ERROR as _
            }
            Self::IrsNotSupported => sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_NOT_SUPPORTED as _,
            Self::IrsOutOfRange => sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_OUT_OF_RANGE as _,
            Self::IrsNrhsNotSupportedForRefineGmres => {
                sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES as _
            }
            Self::IrsInfosNotInitialized => {
                sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED as _
            }
            Self::IrsInfosNotDestroyed => {
                sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED as _
            }
            Self::IrsMatrixSingular => {
                sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_MATRIX_SINGULAR as _
            }
            Self::InvalidWorkspace => sys::cusolverStatus_t::CUSOLVER_STATUS_INVALID_WORKSPACE as _,
            Self::Unknown(code) => code,
        }
    }
}

impl TryFrom<u32> for Status {
    type Error = u32;

    fn try_from(code: u32) -> std::result::Result<Self, Self::Error> {
        match code {
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS as u32 => {
                Ok(Self::Success)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_NOT_INITIALIZED as u32 => {
                Ok(Self::NotInitialized)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_ALLOC_FAILED as u32 => {
                Ok(Self::AllocFailed)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_INVALID_VALUE as u32 => {
                Ok(Self::InvalidValue)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_ARCH_MISMATCH as u32 => {
                Ok(Self::ArchitectureMismatch)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_MAPPING_ERROR as u32 => {
                Ok(Self::MappingError)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_EXECUTION_FAILED as u32 => {
                Ok(Self::ExecutionFailed)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_INTERNAL_ERROR as u32 => {
                Ok(Self::InternalError)
            }
            code if code
                == sys::cusolverStatus_t::CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED as u32 =>
            {
                Ok(Self::MatrixTypeNotSupported)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_NOT_SUPPORTED as u32 => {
                Ok(Self::NotSupported)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_ZERO_PIVOT as u32 => {
                Ok(Self::ZeroPivot)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_INVALID_LICENSE as u32 => {
                Ok(Self::InvalidLicense)
            }
            code if code
                == sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED as u32 =>
            {
                Ok(Self::IrsParamsNotInitialized)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_PARAMS_INVALID as u32 => {
                Ok(Self::IrsParamsInvalid)
            }
            code if code
                == sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC as u32 =>
            {
                Ok(Self::IrsParamsInvalidPrec)
            }
            code if code
                == sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE as u32 =>
            {
                Ok(Self::IrsParamsInvalidRefine)
            }
            code if code
                == sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER as u32 =>
            {
                Ok(Self::IrsParamsInvalidMaxIter)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_INTERNAL_ERROR as u32 => {
                Ok(Self::IrsInternalError)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_NOT_SUPPORTED as u32 => {
                Ok(Self::IrsNotSupported)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_OUT_OF_RANGE as u32 => {
                Ok(Self::IrsOutOfRange)
            }
            code if code
                == sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES
                    as u32 =>
            {
                Ok(Self::IrsNrhsNotSupportedForRefineGmres)
            }
            code if code
                == sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED as u32 =>
            {
                Ok(Self::IrsInfosNotInitialized)
            }
            code if code
                == sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED as u32 =>
            {
                Ok(Self::IrsInfosNotDestroyed)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_IRS_MATRIX_SINGULAR as u32 => {
                Ok(Self::IrsMatrixSingular)
            }
            code if code == sys::cusolverStatus_t::CUSOLVER_STATUS_INVALID_WORKSPACE as u32 => {
                Ok(Self::InvalidWorkspace)
            }
            code => Err(code),
        }
    }
}

impl From<sys::cusolverStatus_t> for Status {
    fn from(status: sys::cusolverStatus_t) -> Self {
        Self::try_from(status as u32).unwrap_or_else(Self::Unknown)
    }
}

impl TryFrom<Status> for sys::cusolverStatus_t {
    type Error = Status;

    fn try_from(status: Status) -> std::result::Result<Self, Self::Error> {
        match status {
            Status::Success => Ok(Self::CUSOLVER_STATUS_SUCCESS),
            Status::NotInitialized => Ok(Self::CUSOLVER_STATUS_NOT_INITIALIZED),
            Status::AllocFailed => Ok(Self::CUSOLVER_STATUS_ALLOC_FAILED),
            Status::InvalidValue => Ok(Self::CUSOLVER_STATUS_INVALID_VALUE),
            Status::ArchitectureMismatch => Ok(Self::CUSOLVER_STATUS_ARCH_MISMATCH),
            Status::MappingError => Ok(Self::CUSOLVER_STATUS_MAPPING_ERROR),
            Status::ExecutionFailed => Ok(Self::CUSOLVER_STATUS_EXECUTION_FAILED),
            Status::InternalError => Ok(Self::CUSOLVER_STATUS_INTERNAL_ERROR),
            Status::MatrixTypeNotSupported => Ok(Self::CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED),
            Status::NotSupported => Ok(Self::CUSOLVER_STATUS_NOT_SUPPORTED),
            Status::ZeroPivot => Ok(Self::CUSOLVER_STATUS_ZERO_PIVOT),
            Status::InvalidLicense => Ok(Self::CUSOLVER_STATUS_INVALID_LICENSE),
            Status::IrsParamsNotInitialized => Ok(Self::CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED),
            Status::IrsParamsInvalid => Ok(Self::CUSOLVER_STATUS_IRS_PARAMS_INVALID),
            Status::IrsParamsInvalidPrec => Ok(Self::CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC),
            Status::IrsParamsInvalidRefine => Ok(Self::CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE),
            Status::IrsParamsInvalidMaxIter => Ok(Self::CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER),
            Status::IrsInternalError => Ok(Self::CUSOLVER_STATUS_IRS_INTERNAL_ERROR),
            Status::IrsNotSupported => Ok(Self::CUSOLVER_STATUS_IRS_NOT_SUPPORTED),
            Status::IrsOutOfRange => Ok(Self::CUSOLVER_STATUS_IRS_OUT_OF_RANGE),
            Status::IrsNrhsNotSupportedForRefineGmres => {
                Ok(Self::CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES)
            }
            Status::IrsInfosNotInitialized => Ok(Self::CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED),
            Status::IrsInfosNotDestroyed => Ok(Self::CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED),
            Status::IrsMatrixSingular => Ok(Self::CUSOLVER_STATUS_IRS_MATRIX_SINGULAR),
            Status::InvalidWorkspace => Ok(Self::CUSOLVER_STATUS_INVALID_WORKSPACE),
            Status::Unknown(_) => Err(status),
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("CUSOLVER_STATUS_SUCCESS"),
            Self::NotInitialized => f.write_str("CUSOLVER_STATUS_NOT_INITIALIZED"),
            Self::AllocFailed => f.write_str("CUSOLVER_STATUS_ALLOC_FAILED"),
            Self::InvalidValue => f.write_str("CUSOLVER_STATUS_INVALID_VALUE"),
            Self::ArchitectureMismatch => f.write_str("CUSOLVER_STATUS_ARCH_MISMATCH"),
            Self::MappingError => f.write_str("CUSOLVER_STATUS_MAPPING_ERROR"),
            Self::ExecutionFailed => f.write_str("CUSOLVER_STATUS_EXECUTION_FAILED"),
            Self::InternalError => f.write_str("CUSOLVER_STATUS_INTERNAL_ERROR"),
            Self::MatrixTypeNotSupported => {
                f.write_str("CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED")
            }
            Self::NotSupported => f.write_str("CUSOLVER_STATUS_NOT_SUPPORTED"),
            Self::ZeroPivot => f.write_str("CUSOLVER_STATUS_ZERO_PIVOT"),
            Self::InvalidLicense => f.write_str("CUSOLVER_STATUS_INVALID_LICENSE"),
            Self::IrsParamsNotInitialized => {
                f.write_str("CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED")
            }
            Self::IrsParamsInvalid => f.write_str("CUSOLVER_STATUS_IRS_PARAMS_INVALID"),
            Self::IrsParamsInvalidPrec => f.write_str("CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC"),
            Self::IrsParamsInvalidRefine => {
                f.write_str("CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE")
            }
            Self::IrsParamsInvalidMaxIter => {
                f.write_str("CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER")
            }
            Self::IrsInternalError => f.write_str("CUSOLVER_STATUS_IRS_INTERNAL_ERROR"),
            Self::IrsNotSupported => f.write_str("CUSOLVER_STATUS_IRS_NOT_SUPPORTED"),
            Self::IrsOutOfRange => f.write_str("CUSOLVER_STATUS_IRS_OUT_OF_RANGE"),
            Self::IrsNrhsNotSupportedForRefineGmres => {
                f.write_str("CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES")
            }
            Self::IrsInfosNotInitialized => {
                f.write_str("CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED")
            }
            Self::IrsInfosNotDestroyed => f.write_str("CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED"),
            Self::IrsMatrixSingular => f.write_str("CUSOLVER_STATUS_IRS_MATRIX_SINGULAR"),
            Self::InvalidWorkspace => f.write_str("CUSOLVER_STATUS_INVALID_WORKSPACE"),
            Self::Unknown(code) => write!(f, "UNKNOWN_CUSOLVER_STATUS({code})"),
        }
    }
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    #[error("cusolver error ({code}): {message}")]
    Cusolver { code: Status, message: String },

    #[error("string contains interior nul byte")]
    InteriorNul,

    #[error("unexpected null handle")]
    NullHandle,

    #[error("`{name}` is out of range")]
    OutOfRange { name: String },

    #[error("invalid matrix leading dimension")]
    InvalidLeadingDimension,

    #[error("invalid matrix shape")]
    InvalidMatrixShape,

    #[error("invalid vector shape")]
    InvalidVectorShape,

    #[error("invalid eigensolver range")]
    InvalidEigenRange,

    #[error("invalid svd mode")]
    InvalidSvdMode,

    #[error("invalid precision configuration")]
    InvalidPrecisionConfiguration,

    #[error("insufficient workspace size: {actual} provided (required {required})")]
    InsufficientWorkspaceSize { required: usize, actual: usize },

    #[error("invalid residual history")]
    InvalidResidualHistory,

    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,
}

pub type Result<T> = std::result::Result<T, Error>;

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
