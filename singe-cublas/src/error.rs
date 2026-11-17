use std::{borrow::Cow, ffi::NulError, result};

use singe_cublas_sys as sys;
use singe_cuda::{data_type::DataType, error::Error as CudaError};
use thiserror::Error;

singe_core::wrapper_status! {
    library = "cublas",
    error = Error::Cublas,
    raw = sys::cublasStatus_t,
    success = CUBLAS_STATUS_SUCCESS,
    absence = sys::take_absence,
    unknown = "UNKNOWN_CUBLAS_STATUS",
    description = sys::cublasGetStatusString,
    /// cuBLAS status returned by library calls.
    pub enum Status: u32 {
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
        /// The requested operation requires a license, and an error was detected when checking the current licensing.
        /// This error can happen if the license is not present or expired, or if `NVIDIA_LICENSE_FILE` is not set properly.
        LicenseError = CUBLAS_STATUS_LICENSE_ERROR,
    }
}

/// An error from CUDA, cuBLAS, or validation of wrapper arguments.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// cuBLAS or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// A CUDA operation failed.
    #[error(transparent)]
    Cuda(CudaError),

    /// A cuBLAS or cuBLASLt operation returned a failure status.
    #[error("cublas error ({code}): {}", code.description())]
    Cublas {
        /// The native cuBLAS status.
        code: Status,
    },

    /// A string contains an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native handle is null.
    #[error("unexpected null handle")]
    NullHandle,

    /// A native enum or attribute value has no typed variant in this crate.
    #[error("unknown `{name}` value {value}")]
    UnknownEnumValue {
        /// Name of the typed enum or attribute.
        name: Cow<'static, str>,
        /// Unrecognized native value.
        value: i64,
    },

    /// A value or computed size cannot be represented by the required integer type.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// The argument or computed quantity that is out of range.
        name: Cow<'static, str>,
    },

    /// A native value's size differs from the requested Rust representation.
    #[error("`{name}` size mismatch (expected {expected} bytes, got {actual})")]
    SizeMismatch {
        /// The value being read.
        name: Cow<'static, str>,
        /// The size of the requested Rust representation, in bytes.
        expected: usize,
        /// The size returned by cuBLAS, in bytes.
        actual: usize,
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

    /// A buffer holds fewer elements or bytes than the operation requires.
    #[error("`{name}` is too small (requires {required}, got {actual})")]
    BufferTooSmall {
        /// Name of the buffer.
        name: Cow<'static, str>,
        /// Required length.
        required: usize,
        /// Supplied length.
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

    /// The scalar arguments use different pointer modes.
    #[error("scalar pointer modes do not match")]
    ScalarPointerModeMismatch,

    /// A data type argument, or the Rust type of a value, differs from the data type the operation requires.
    #[error("`{name}` is {actual}, but the operation requires {expected}")]
    DataTypeMismatch {
        /// The data type argument or value.
        name: Cow<'static, str>,
        /// The required data type.
        expected: DataType,
        /// The supplied data type.
        actual: DataType,
    },
}

/// A cuBLAS wrapper result with [`enum@Error`] as its error type.
pub type Result<T> = result::Result<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }

    pub(crate) fn buffer_too_small(
        name: impl Into<Cow<'static, str>>,
        required: usize,
        actual: usize,
    ) -> Self {
        Self::BufferTooSmall {
            name: name.into(),
            required,
            actual,
        }
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

// Converts a raw cuBLAS or cuBLASLt status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
