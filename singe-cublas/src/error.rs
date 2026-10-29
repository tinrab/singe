use std::{
    borrow::Cow,
    ffi::{self, CStr},
    result,
};

use singe_core::status_enum;
use singe_cublas_sys as sys;
use singe_cuda::error::Error as CudaError;
use thiserror::Error;

status_enum! {
    /// cuBLAS status returned by library calls.
    pub enum Status: u32 => sys::cublasStatus_t, unknown = "UNKNOWN_CUBLAS_STATUS" {
        /// The operation completed successfully.
        Success = CUBLAS_STATUS_SUCCESS,
        /// The cuBLAS library was not initialized.
        /// Common causes include a missing prior [`Context::create`](crate::context::Context::create) call, an error in the CUDA runtime called by cuBLAS, or an error in the hardware setup.
        /// To correct: call [`Context::create`](crate::context::Context::create) before the operation and check that the hardware, driver, and cuBLAS library are correctly installed.
        NotInitialized = CUBLAS_STATUS_NOT_INITIALIZED,
        /// Resource allocation failed inside the cuBLAS library.
        /// Usually caused by a device memory allocation failure.
        /// To correct: before the operation, deallocate previously allocated memory as much as possible.
        AllocFailed = CUBLAS_STATUS_ALLOC_FAILED,
        /// An unsupported value or parameter was passed to the operation, such as a negative vector size.
        /// To correct: ensure that all the parameters being passed have valid values.
        InvalidValue = CUBLAS_STATUS_INVALID_VALUE,
        /// The operation requires a feature absent from the device architecture; usually caused by compute capability lower than 5.0.
        /// To correct: compile and run the application on a device with appropriate compute capability.
        ArchitectureMismatch = CUBLAS_STATUS_ARCH_MISMATCH,
        /// An access to GPU memory space failed, which is usually caused by a failure to bind a texture.
        /// To correct: before the operation, unbind any previously bound textures.
        MappingError = CUBLAS_STATUS_MAPPING_ERROR,
        /// The GPU program failed to execute.
        /// A kernel launch failure on the GPU is a common cause.
        /// To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.
        ExecutionFailed = CUBLAS_STATUS_EXECUTION_FAILED,
        /// An internal cuBLAS operation failed.
        /// Usually caused by an asynchronous memory copy failure.
        /// To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.
        /// Also check that memory passed to the operation is not released before the operation completes.
        InternalError = CUBLAS_STATUS_INTERNAL_ERROR,
        /// The requested operation is not supported.
        NotSupported = CUBLAS_STATUS_NOT_SUPPORTED,
        /// The requested operation requires a license, and an error was detected
        /// when checking the current licensing.
        /// This error can happen if the license is not present or expired, or if
        /// `NVIDIA_LICENSE_FILE` is not set properly.
        LicenseError = CUBLAS_STATUS_LICENSE_ERROR,
    }
}

impl Status {
    /// Returns a short description of this status.
    #[must_use]
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
}

/// An error from CUDA, cuBLAS, or validation of wrapper arguments.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// A CUDA operation failed.
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    /// A cuBLAS operation returned a failure status.
    #[error("cublas error ({code}): {message}")]
    Cublas {
        /// The native cuBLAS status.
        code: Status,
        /// The status description reported by cuBLAS.
        message: String,
    },

    /// A string contains an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native handle is null.
    #[error("unexpected null handle")]
    NullHandle,

    /// A value or computed size cannot be represented by the required integer type.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// The argument or computed quantity that is out of range.
        name: Cow<'static, str>,
    },

    /// The native attribute size differs from the requested Rust representation.
    #[error("unexpected attribute size: expected {expected} bytes, got {actual}")]
    AttributeSizeMismatch {
        /// The size of the requested Rust representation, in bytes.
        expected: usize,
        /// The attribute size returned by cuBLAS, in bytes.
        actual: usize,
    },

    /// A native attribute value is not a recognized variant of the requested enum.
    #[error("invalid value {value} for attribute `{name}`")]
    InvalidAttributeValue {
        /// The attribute being decoded.
        name: Cow<'static, str>,
        /// The unrecognized native value.
        value: u32,
    },

    /// An argument has a different number of elements than required.
    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        /// The argument with the incorrect length.
        name: Cow<'static, str>,
        /// The required number of elements.
        expected: usize,
        /// The supplied number of elements.
        actual: usize,
    },

    /// A vector increment is invalid for the requested operation.
    #[error("invalid vector increment")]
    InvalidIncrement,

    /// A matrix leading dimension is too small for the requested layout.
    #[error("invalid matrix leading dimension")]
    InvalidLeadingDimension,

    /// Matrix storage, dimensions, or batch lengths do not match the operation.
    #[error("invalid matrix shape")]
    InvalidMatrixShape,

    /// Vector storage is too short or vector lengths are incompatible.
    #[error("invalid vector shape")]
    InvalidVectorShape,

    /// A stream belongs to a different CUDA context from the handle.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// The operation requires the handle to use host scalar pointers.
    #[error("operation requires host pointer mode")]
    RequiresHostPointerMode,

    /// The scalar arguments use different pointer modes.
    #[error("scalar pointer modes do not match")]
    ScalarPointerModeMismatch,
}

/// A cuBLAS wrapper result with [`enum@Error`] as its error type.
pub type Result<T> = result::Result<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

impl From<sys::cublasStatus_t> for Error {
    fn from(status: sys::cublasStatus_t) -> Self {
        // SAFETY: `cublasGetStatusString` returns null or a static NUL-terminated string.
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

/// Converts a raw cuBLAS status-returning expression into a crate [`Result`](crate::error::Result).
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
