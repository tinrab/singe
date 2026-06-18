use std::{
    ffi::{self, CStr},
    fmt, result,
};

use singe_cuda::error::Error as CudaError;
use thiserror::Error;

use singe_cublas_sys as sys;

/// cuBLAS status returned by library calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    /// The operation completed successfully.
    Success,
    /// The cuBLAS library was not initialized.
    /// Common causes include a missing prior [`Context::create`](crate::context::Context::create) call, an error in the CUDA runtime called by cuBLAS, or an error in the hardware setup.
    /// To correct: call [`Context::create`](crate::context::Context::create) before the operation and check that the hardware, driver, and cuBLAS library are correctly installed.
    NotInitialized,
    /// Resource allocation failed inside the cuBLAS library.
    /// Usually caused by a device memory allocation failure.
    /// To correct: before the operation, deallocate previously allocated memory as much as possible.
    AllocFailed,
    /// An unsupported value or parameter was passed to the operation, such as a negative vector size.
    /// To correct: ensure that all the parameters being passed have valid values.
    InvalidValue,
    /// The operation requires a feature absent from the device architecture; usually caused by compute capability lower than 5.0.
    /// To correct: compile and run the application on a device with appropriate compute capability.
    ArchitectureMismatch,
    /// An access to GPU memory space failed, which is usually caused by a failure to bind a texture.
    /// To correct: before the operation, unbind any previously bound textures.
    MappingError,
    /// The GPU program failed to execute.
    /// A kernel launch failure on the GPU is a common cause.
    /// To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.
    ExecutionFailed,
    /// An internal cuBLAS operation failed.
    /// Usually caused by an asynchronous memory copy failure.
    /// To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.
    /// Also check that memory passed to the operation is not released before the operation completes.
    InternalError,
    /// The requested operation is not supported.
    NotSupported,
    /// The requested operation requires a license, and an error was detected
    /// when checking the current licensing.
    /// This error can happen if the license is not present or expired, or if
    /// `NVIDIA_LICENSE_FILE` is not set properly.
    LicenseError,
    /// A status returned by a newer cuBLAS library than this crate knows.
    Unknown(u32),
}

impl Status {
    pub const fn description(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::NotInitialized => "not initialized",
            Self::AllocFailed => "allocation failed",
            Self::InvalidValue => "invalid value",
            Self::ArchitectureMismatch => "architecture mismatch",
            Self::MappingError => "mapping error",
            Self::ExecutionFailed => "execution failed",
            Self::InternalError => "internal error",
            Self::NotSupported => "not supported",
            Self::LicenseError => "license error",
            Self::Unknown(_) => "unknown cublas error",
        }
    }

    pub const fn raw(self) -> u32 {
        match self {
            Self::Success => sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS as _,
            Self::NotInitialized => sys::cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED as _,
            Self::AllocFailed => sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED as _,
            Self::InvalidValue => sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE as _,
            Self::ArchitectureMismatch => sys::cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH as _,
            Self::MappingError => sys::cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR as _,
            Self::ExecutionFailed => sys::cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED as _,
            Self::InternalError => sys::cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR as _,
            Self::NotSupported => sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED as _,
            Self::LicenseError => sys::cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR as _,
            Self::Unknown(code) => code,
        }
    }
}

impl From<sys::cublasStatus_t> for Status {
    fn from(status: sys::cublasStatus_t) -> Self {
        match status {
            sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS => Self::Success,
            sys::cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Self::NotInitialized,
            sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED => Self::AllocFailed,
            sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => Self::InvalidValue,
            sys::cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => Self::ArchitectureMismatch,
            sys::cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR => Self::MappingError,
            sys::cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => Self::ExecutionFailed,
            sys::cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR => Self::InternalError,
            sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED => Self::NotSupported,
            sys::cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR => Self::LicenseError,
        }
    }
}

impl TryFrom<Status> for sys::cublasStatus_t {
    type Error = Status;

    fn try_from(status: Status) -> result::Result<Self, Self::Error> {
        match status {
            Status::Success => Ok(Self::CUBLAS_STATUS_SUCCESS),
            Status::NotInitialized => Ok(Self::CUBLAS_STATUS_NOT_INITIALIZED),
            Status::AllocFailed => Ok(Self::CUBLAS_STATUS_ALLOC_FAILED),
            Status::InvalidValue => Ok(Self::CUBLAS_STATUS_INVALID_VALUE),
            Status::ArchitectureMismatch => Ok(Self::CUBLAS_STATUS_ARCH_MISMATCH),
            Status::MappingError => Ok(Self::CUBLAS_STATUS_MAPPING_ERROR),
            Status::ExecutionFailed => Ok(Self::CUBLAS_STATUS_EXECUTION_FAILED),
            Status::InternalError => Ok(Self::CUBLAS_STATUS_INTERNAL_ERROR),
            Status::NotSupported => Ok(Self::CUBLAS_STATUS_NOT_SUPPORTED),
            Status::LicenseError => Ok(Self::CUBLAS_STATUS_LICENSE_ERROR),
            Status::Unknown(_) => Err(status),
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("CUBLAS_STATUS_SUCCESS"),
            Self::NotInitialized => f.write_str("CUBLAS_STATUS_NOT_INITIALIZED"),
            Self::AllocFailed => f.write_str("CUBLAS_STATUS_ALLOC_FAILED"),
            Self::InvalidValue => f.write_str("CUBLAS_STATUS_INVALID_VALUE"),
            Self::ArchitectureMismatch => f.write_str("CUBLAS_STATUS_ARCH_MISMATCH"),
            Self::MappingError => f.write_str("CUBLAS_STATUS_MAPPING_ERROR"),
            Self::ExecutionFailed => f.write_str("CUBLAS_STATUS_EXECUTION_FAILED"),
            Self::InternalError => f.write_str("CUBLAS_STATUS_INTERNAL_ERROR"),
            Self::NotSupported => f.write_str("CUBLAS_STATUS_NOT_SUPPORTED"),
            Self::LicenseError => f.write_str("CUBLAS_STATUS_LICENSE_ERROR"),
            Self::Unknown(code) => write!(f, "UNKNOWN_CUBLAS_STATUS({code})"),
        }
    }
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    #[error("cublas error ({code}): {message}")]
    Cublas { code: Status, message: String },

    #[error("string contains interior nul byte")]
    InteriorNul,

    #[error("unexpected null handle")]
    NullHandle,

    #[error("`{name}` is out of range")]
    OutOfRange { name: String },

    #[error("unexpected attribute size: expected {expected} bytes, got {actual}")]
    AttributeSizeMismatch { expected: usize, actual: usize },

    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("invalid vector increment")]
    InvalidIncrement,

    #[error("invalid matrix leading dimension")]
    InvalidLeadingDimension,

    #[error("invalid matrix shape")]
    InvalidMatrixShape,

    #[error("invalid vector shape")]
    InvalidVectorShape,

    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    #[error("operation requires host pointer mode")]
    RequiresHostPointerMode,

    #[error("scalar pointer modes do not match")]
    ScalarPointerModeMismatch,
}

pub type Result<T> = result::Result<T, Error>;

impl From<sys::cublasStatus_t> for Error {
    fn from(status: sys::cublasStatus_t) -> Self {
        let message = unsafe {
            let c_ptr = sys::cublasGetStatusString(status);
            if c_ptr.is_null() {
                String::from("unknown cublas error")
            } else {
                CStr::from_ptr(c_ptr).to_string_lossy().into_owned()
            }
        };

        Self::Cublas {
            code: status.into(),
            message,
        }
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        match sys::cublasStatus_t::try_from(status) {
            Ok(status) => status.into(),
            Err(code) => Self::Cublas {
                code,
                message: code.description().into(),
            },
        }
    }
}

impl From<ffi::NulError> for Error {
    fn from(_: ffi::NulError) -> Self {
        Self::InteriorNul
    }
}

#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}
