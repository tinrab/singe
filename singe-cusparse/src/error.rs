use std::{
    ffi::{CStr, NulError},
    fmt, result,
};

use singe_cuda::error::Error as CudaError;
use singe_cusparse_sys as sys;
use thiserror::Error;

/// [`Status`] is the cuSPARSE status code returned by cuSPARSE operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    /// The operation completed successfully.
    Success,
    /// The cuSPARSE library was not initialized.
    /// Common causes include a missing [`Context::create`](crate::context::Context::create) call, a CUDA runtime error reported by cuSPARSE, or an error in the hardware setup.
    /// The error also applies when a matrix or vector descriptor is not initialized.
    NotInitialized,
    /// Resource allocation failed inside the cuSPARSE library.
    /// Usually caused by a device memory allocation (`cudaMalloc()`) or host memory allocation failure.
    AllocFailed,
    /// An unsupported value or parameter was passed to the operation, such as a negative vector size.
    InvalidValue,
    /// The operation requires a feature absent from the device architecture.
    ArchitectureMismatch,
    /// An access to GPU memory space failed, which is usually caused by a
    /// failure to bind a texture or an invalid device pointer.
    /// To correct: check that all device pointers are valid and that no texture
    /// is bound that could interfere with the operation.
    MappingError,
    /// The GPU program failed to execute.
    /// A kernel launch failure on the GPU is a common cause.
    ExecutionFailed,
    /// An internal cuSPARSE operation failed.
    /// Also check that memory passed to the operation is not released before the operation completes.
    InternalError,
    /// The matrix type is not supported by this operation.
    /// Usually caused by passing an invalid matrix descriptor to the operation.
    /// Check that the fields in [`MatrixDescriptor`](crate::matrix::MatrixDescriptor) were set correctly.
    MatrixTypeNotSupported,
    /// A zero pivot was encountered during the factorization.
    /// The operation cannot proceed because a diagonal element is zero.
    /// To correct: ensure the input matrix is non-singular.
    ZeroPivot,
    /// The operation or data type combination is currently not supported.
    NotSupported,
    /// The resources for the computation, such as GPU global or shared memory, are not sufficient to complete the operation.
    /// The error can also indicate that the current computation mode, such as sparse matrix index bit size, cannot handle the given input.
    InsufficientResources,
    /// A status code not known to this version of Singe.
    Unknown(u32),
}

impl Status {
    pub fn description(self) -> String {
        match sys::cusparseStatus_t::try_from(self.raw()) {
            Ok(status) => cusparse_error_string(status),
            Err(_) => String::from("unknown cusparse error"),
        }
    }

    pub const fn raw(self) -> u32 {
        match self {
            Self::Success => sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS as _,
            Self::NotInitialized => sys::cusparseStatus_t::CUSPARSE_STATUS_NOT_INITIALIZED as _,
            Self::AllocFailed => sys::cusparseStatus_t::CUSPARSE_STATUS_ALLOC_FAILED as _,
            Self::InvalidValue => sys::cusparseStatus_t::CUSPARSE_STATUS_INVALID_VALUE as _,
            Self::ArchitectureMismatch => sys::cusparseStatus_t::CUSPARSE_STATUS_ARCH_MISMATCH as _,
            Self::MappingError => sys::cusparseStatus_t::CUSPARSE_STATUS_MAPPING_ERROR as _,
            Self::ExecutionFailed => sys::cusparseStatus_t::CUSPARSE_STATUS_EXECUTION_FAILED as _,
            Self::InternalError => sys::cusparseStatus_t::CUSPARSE_STATUS_INTERNAL_ERROR as _,
            Self::MatrixTypeNotSupported => {
                sys::cusparseStatus_t::CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED as _
            }
            Self::ZeroPivot => sys::cusparseStatus_t::CUSPARSE_STATUS_ZERO_PIVOT as _,
            Self::NotSupported => sys::cusparseStatus_t::CUSPARSE_STATUS_NOT_SUPPORTED as _,
            Self::InsufficientResources => {
                sys::cusparseStatus_t::CUSPARSE_STATUS_INSUFFICIENT_RESOURCES as _
            }
            Self::Unknown(code) => code,
        }
    }
}

impl TryFrom<u32> for Status {
    type Error = u32;

    fn try_from(code: u32) -> result::Result<Self, u32> {
        match code {
            code if code == sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS as u32 => {
                Ok(Self::Success)
            }
            code if code == sys::cusparseStatus_t::CUSPARSE_STATUS_NOT_INITIALIZED as u32 => {
                Ok(Self::NotInitialized)
            }
            code if code == sys::cusparseStatus_t::CUSPARSE_STATUS_ALLOC_FAILED as u32 => {
                Ok(Self::AllocFailed)
            }
            code if code == sys::cusparseStatus_t::CUSPARSE_STATUS_INVALID_VALUE as u32 => {
                Ok(Self::InvalidValue)
            }
            code if code == sys::cusparseStatus_t::CUSPARSE_STATUS_ARCH_MISMATCH as u32 => {
                Ok(Self::ArchitectureMismatch)
            }
            code if code == sys::cusparseStatus_t::CUSPARSE_STATUS_MAPPING_ERROR as u32 => {
                Ok(Self::MappingError)
            }
            code if code == sys::cusparseStatus_t::CUSPARSE_STATUS_EXECUTION_FAILED as u32 => {
                Ok(Self::ExecutionFailed)
            }
            code if code == sys::cusparseStatus_t::CUSPARSE_STATUS_INTERNAL_ERROR as u32 => {
                Ok(Self::InternalError)
            }
            code if code
                == sys::cusparseStatus_t::CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED as u32 =>
            {
                Ok(Self::MatrixTypeNotSupported)
            }
            code if code == sys::cusparseStatus_t::CUSPARSE_STATUS_ZERO_PIVOT as u32 => {
                Ok(Self::ZeroPivot)
            }
            code if code == sys::cusparseStatus_t::CUSPARSE_STATUS_NOT_SUPPORTED as u32 => {
                Ok(Self::NotSupported)
            }
            code if code
                == sys::cusparseStatus_t::CUSPARSE_STATUS_INSUFFICIENT_RESOURCES as u32 =>
            {
                Ok(Self::InsufficientResources)
            }
            code => Err(code),
        }
    }
}

impl From<sys::cusparseStatus_t> for Status {
    fn from(status: sys::cusparseStatus_t) -> Self {
        Self::try_from(status as u32).unwrap_or_else(Self::Unknown)
    }
}

impl TryFrom<Status> for sys::cusparseStatus_t {
    type Error = Status;

    fn try_from(status: Status) -> result::Result<Self, Status> {
        match status {
            Status::Success => Ok(Self::CUSPARSE_STATUS_SUCCESS),
            Status::NotInitialized => Ok(Self::CUSPARSE_STATUS_NOT_INITIALIZED),
            Status::AllocFailed => Ok(Self::CUSPARSE_STATUS_ALLOC_FAILED),
            Status::InvalidValue => Ok(Self::CUSPARSE_STATUS_INVALID_VALUE),
            Status::ArchitectureMismatch => Ok(Self::CUSPARSE_STATUS_ARCH_MISMATCH),
            Status::MappingError => Ok(Self::CUSPARSE_STATUS_MAPPING_ERROR),
            Status::ExecutionFailed => Ok(Self::CUSPARSE_STATUS_EXECUTION_FAILED),
            Status::InternalError => Ok(Self::CUSPARSE_STATUS_INTERNAL_ERROR),
            Status::MatrixTypeNotSupported => Ok(Self::CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED),
            Status::ZeroPivot => Ok(Self::CUSPARSE_STATUS_ZERO_PIVOT),
            Status::NotSupported => Ok(Self::CUSPARSE_STATUS_NOT_SUPPORTED),
            Status::InsufficientResources => Ok(Self::CUSPARSE_STATUS_INSUFFICIENT_RESOURCES),
            Status::Unknown(_) => Err(status),
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("CUSPARSE_STATUS_SUCCESS"),
            Self::NotInitialized => f.write_str("CUSPARSE_STATUS_NOT_INITIALIZED"),
            Self::AllocFailed => f.write_str("CUSPARSE_STATUS_ALLOC_FAILED"),
            Self::InvalidValue => f.write_str("CUSPARSE_STATUS_INVALID_VALUE"),
            Self::ArchitectureMismatch => f.write_str("CUSPARSE_STATUS_ARCH_MISMATCH"),
            Self::MappingError => f.write_str("CUSPARSE_STATUS_MAPPING_ERROR"),
            Self::ExecutionFailed => f.write_str("CUSPARSE_STATUS_EXECUTION_FAILED"),
            Self::InternalError => f.write_str("CUSPARSE_STATUS_INTERNAL_ERROR"),
            Self::MatrixTypeNotSupported => {
                f.write_str("CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED")
            }
            Self::ZeroPivot => f.write_str("CUSPARSE_STATUS_ZERO_PIVOT"),
            Self::NotSupported => f.write_str("CUSPARSE_STATUS_NOT_SUPPORTED"),
            Self::InsufficientResources => f.write_str("CUSPARSE_STATUS_INSUFFICIENT_RESOURCES"),
            Self::Unknown(code) => write!(f, "UNKNOWN_CUSPARSE_STATUS({code})"),
        }
    }
}

fn cusparse_error_string(status: sys::cusparseStatus_t) -> String {
    unsafe {
        let c_ptr = sys::cusparseGetErrorString(status);
        if c_ptr.is_null() {
            String::from("unknown cusparse error")
        } else {
            CStr::from_ptr(c_ptr).to_string_lossy().into_owned()
        }
    }
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    #[error("cusparse error ({code}): {message}")]
    Cusparse { code: Status, message: String },

    #[error("string contains interior nul byte")]
    InteriorNul,

    #[error("unexpected null handle")]
    NullHandle,

    #[error("`{name}` is out of range")]
    OutOfRange { name: String },

    #[error("scalar pointer modes do not match")]
    ScalarPointerModeMismatch,

    #[error("plan belongs to a different cusparse context")]
    PlanContextMismatch,

    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,
}

pub type Result<T> = result::Result<T, Error>;

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
            message: code.description(),
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
        if status != singe_cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}
