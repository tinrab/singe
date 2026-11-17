use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_cuda::error::Error as CudaError;
use singe_nvimagecodec_sys as sys;
use thiserror::Error;

use crate::types::ProcessingStatus;

singe_core::wrapper_status! {
    library = "nvimagecodec",
    error = Error::Nvimagecodec,
    raw = sys::nvimgcodecStatus_t,
    success = NVIMGCODEC_STATUS_SUCCESS,
    absence = sys::take_absence,
    unknown = "UNKNOWN_NVIMGCODEC_STATUS",
    /// Status codes returned by nvImageCodec calls.
    pub enum Status: u32 {
        /// The operation succeeded.
        Success = NVIMGCODEC_STATUS_SUCCESS => "success",
        /// The library has not been initialized.
        NotInitialized = NVIMGCODEC_STATUS_NOT_INITIALIZED => "library not initialized",
        /// An argument is invalid.
        InvalidParameter = NVIMGCODEC_STATUS_INVALID_PARAMETER => "invalid parameter",
        /// The compressed image stream is malformed.
        BadCodeStream = NVIMGCODEC_STATUS_BAD_CODESTREAM => "bad code stream",
        /// The compressed image stream uses unsupported features.
        CodeStreamUnsupported = NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED => "code stream unsupported",
        /// A required allocation failed.
        AllocatorFailure = NVIMGCODEC_STATUS_ALLOCATOR_FAILURE => "allocator failure",
        /// The requested operation failed during execution.
        ExecutionFailed = NVIMGCODEC_STATUS_EXECUTION_FAILED => "execution failed",
        /// The device architecture does not support this operation.
        ArchitectureMismatch = NVIMGCODEC_STATUS_ARCH_MISMATCH => "architecture mismatch",
        /// An internal implementation error occurred.
        InternalError = NVIMGCODEC_STATUS_INTERNAL_ERROR => "internal error",
        /// No supported implementation is available for the requested operation.
        ImplementationUnsupported = NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED => "implementation unsupported",
        /// A dependency required by the operation is unavailable.
        MissedDependencies = NVIMGCODEC_STATUS_MISSED_DEPENDENCIES => "missed dependencies",
        /// An extension reported that the library has not been initialized.
        ExtensionNotInitialized = NVIMGCODEC_STATUS_EXTENSION_NOT_INITIALIZED => "extension not initialized",
        /// An extension reported that an argument is invalid.
        ExtensionInvalidParameter = NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER => "extension invalid parameter",
        /// An extension reported that the compressed image stream is malformed.
        ExtensionBadCodeStream = NVIMGCODEC_STATUS_EXTENSION_BAD_CODE_STREAM => "extension bad code stream",
        /// An extension reported that the compressed image stream uses unsupported features.
        ExtensionCodeStreamUnsupported = NVIMGCODEC_STATUS_EXTENSION_CODESTREAM_UNSUPPORTED => "extension code stream unsupported",
        /// An extension reported that a required allocation failed.
        ExtensionAllocatorFailure = NVIMGCODEC_STATUS_EXTENSION_ALLOCATOR_FAILURE => "extension allocator failure",
        /// An extension reported that the device architecture does not support this operation.
        ExtensionArchitectureMismatch = NVIMGCODEC_STATUS_EXTENSION_ARCH_MISMATCH => "extension architecture mismatch",
        /// An extension reported that an internal implementation error occurred.
        ExtensionInternalError = NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR => "extension internal error",
        /// An extension reported that no supported implementation is available for the requested operation.
        ExtensionImplementationNotSupported = NVIMGCODEC_STATUS_EXTENSION_IMPLEMENTATION_NOT_SUPPORTED => "extension implementation not supported",
        /// An extension reported that the compressed image stream is incomplete.
        ExtensionIncompleteBitstream = NVIMGCODEC_STATUS_EXTENSION_INCOMPLETE_BITSTREAM => "extension incomplete bitstream",
        /// An extension reported that the requested operation failed during execution.
        ExtensionExecutionFailed = NVIMGCODEC_STATUS_EXTENSION_EXECUTION_FAILED => "extension execution failed",
        /// An extension reported that a CUDA API call failed.
        ExtensionCudaCallError = NVIMGCODEC_STATUS_EXTENSION_CUDA_CALL_ERROR => "extension cuda call error",
    }
}

/// Errors returned by this crate.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// nvImageCodec or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// An operation needs to write an image whose pixels are read-only.
    #[error("image samples are read-only")]
    ReadOnlySamples,

    /// An operation needs to read an image whose pixels are not initialized.
    #[error("image samples have not been initialized")]
    UninitializedSamples,

    /// A stream belongs to a different CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// An nvImageCodec object belongs to a different instance.
    #[error("`{name}` belongs to a different nvimagecodec context")]
    ContextMismatch {
        /// Name of the object.
        name: Cow<'static, str>,
    },

    /// A CUDA operation failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// nvImageCodec returned an error status.
    #[error("nvimagecodec error ({code}): {}", code.description())]
    Nvimagecodec {
        /// Status returned by nvImageCodec.
        code: Status,
    },

    /// A string passed to nvImageCodec contains a NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// nvImageCodec returned a null handle.
    #[error("unexpected null handle")]
    NullHandle,

    /// A required pointer is null.
    #[error("`{name}` is null")]
    NullPointer {
        /// Name of the argument or value.
        name: Cow<'static, str>,
    },

    /// A native enum value has no typed variant in this crate.
    #[error("unknown `{name}` value {value}")]
    UnknownEnumValue {
        /// Name of the typed enum.
        name: Cow<'static, str>,
        /// Unrecognized native value.
        value: i64,
    },

    /// A value, or a size computed from it, is out of range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the argument or value.
        name: Cow<'static, str>,
    },

    /// A collection, string, or dimension that must not be empty is empty.
    #[error("`{name}` is empty")]
    Empty {
        /// Name of the argument or value.
        name: Cow<'static, str>,
    },

    /// The requested operation or format is not supported.
    #[error("`{name}` is unsupported")]
    Unsupported {
        /// Name of the argument or value.
        name: Cow<'static, str>,
    },

    /// An option name or value cannot be written in nvImageCodec's option syntax.
    #[error("invalid option `{name}`")]
    InvalidOption {
        /// Name of the argument or value.
        name: Cow<'static, str>,
    },

    /// Collections that must match in length have different lengths.
    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        /// Name of the argument or value.
        name: Cow<'static, str>,
        /// Expected number of entries.
        expected: usize,
        /// Actual number of entries.
        actual: usize,
    },

    /// A buffer or stride is smaller than the extent the operation reads or writes.
    #[error("`{name}` is too small (requires {required}, got {actual})")]
    BufferTooSmall {
        /// Name of the buffer or stride.
        name: Cow<'static, str>,
        /// Required size in bytes.
        required: usize,
        /// Supplied size in bytes.
        actual: usize,
    },

    /// nvImageCodec processed a batch but reported a failure for one image.
    #[error("`{operation}` failed with {status}")]
    OperationFailed {
        /// The failed image, named by its index in the batch.
        operation: Cow<'static, str>,
        /// Status reported for the image.
        status: ProcessingStatus,
    },
}

/// Result type returned by this crate.
pub type Result<T> = StdResult<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
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

// Converts a raw nvImageCodec status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
