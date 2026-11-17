use std::{borrow::Cow, ffi::NulError, result};

use singe_cuda::error::Error as CudaError;
use singe_cusparse_sys as sys;
use thiserror::Error;

singe_core::wrapper_status! {
    library = "cusparse",
    error = Error::Cusparse,
    raw = sys::cusparseStatus_t,
    success = CUSPARSE_STATUS_SUCCESS,
    absence = sys::take_absence,
    unknown = "UNKNOWN_CUSPARSE_STATUS",
    description = sys::cusparseGetErrorString,
    /// [`Status`] is the cuSPARSE status code returned by cuSPARSE operations.
    pub enum Status: u32 {
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
        /// An access to GPU memory space failed, which is usually caused by a failure to bind a texture or an invalid device pointer.
        /// To correct: check that all device pointers are valid and that no texture is bound that could interfere with the operation.
        MappingError = CUSPARSE_STATUS_MAPPING_ERROR,
        /// The GPU program failed to execute.
        /// A kernel launch failure on the GPU is a common cause.
        ExecutionFailed = CUSPARSE_STATUS_EXECUTION_FAILED,
        /// An internal cuSPARSE operation failed.
        /// Also check that memory passed to the operation is not released before the operation completes.
        InternalError = CUSPARSE_STATUS_INTERNAL_ERROR,
        /// The matrix type is not supported by this operation.
        /// Usually caused by passing an invalid matrix descriptor to the operation.
        /// Check the matrix type set on the descriptor.
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

/// CUDA and cuSPARSE status failures and errors validating wrapper inputs or native output.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// cuSPARSE or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// A CUDA context, stream, or memory operation failed.
    #[error(transparent)]
    Cuda(CudaError),

    /// cuSPARSE returned a failure status.
    #[error("cusparse error ({code}): {}", code.description())]
    Cusparse {
        /// Native cuSPARSE status.
        code: Status,
    },

    /// A string contains an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native call returned a null handle where a live resource was required.
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

    /// A count, size, dimension, or layout exceeds the supported representation or range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Value or layout whose required range was exceeded.
        name: Cow<'static, str>,
    },

    /// A buffer is smaller than the extent the operation reads or writes.
    #[error("`{name}` is too small (requires {required}, got {actual})")]
    BufferTooSmall {
        /// Name of the buffer.
        name: Cow<'static, str>,
        /// Required length.
        required: usize,
        /// Supplied length.
        actual: usize,
    },

    /// Input scalars use different host or device pointer modes.
    #[error("scalar pointer modes do not match")]
    ScalarPointerModeMismatch,

    /// A cuSPARSE object belongs to a different context.
    #[error("`{name}` belongs to a different cusparse context")]
    ContextMismatch {
        /// Name of the object.
        name: Cow<'static, str>,
    },

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

    pub(crate) fn context_mismatch(name: impl Into<Cow<'static, str>>) -> Self {
        Self::ContextMismatch { name: name.into() }
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

// Converts a raw cuSPARSE status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
