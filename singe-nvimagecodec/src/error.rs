use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_core::status_enum;
use singe_cuda::error::Error as CudaError;
use singe_nvimagecodec_sys as sys;
use thiserror::Error;

use crate::types::ProcessingStatus;

status_enum! {
    /// Status codes returned by nvImageCodec calls.
    pub enum Status: u32 => sys::nvimgcodecStatus_t, unknown = "UNKNOWN_NVIMGCODEC_STATUS", unknown_description = "unknown nvimagecodec error" {
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
    // Its library would not load.
    #[error(transparent)]
    Load(#[from] singe_nvimagecodec_sys::LoadError),

    /// An operation needs to write an image whose pixels are read-only.
    #[error("image samples are read-only")]
    ReadOnlySamples,

    /// An operation needs to read an image whose pixels are not initialized.
    #[error("image samples have not been initialized")]
    UninitializedSamples,

    /// A stream belongs to a different CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// Values used together belong to different nvImageCodec instances.
    #[error("resources belong to different nvimagecodec instances")]
    InstanceMismatch,

    /// A CUDA operation failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// nvImageCodec returned an error status.
    #[error("nvimagecodec error ({code}): {message}")]
    Nvimagecodec {
        /// Status returned by nvImageCodec.
        code: Status,
        /// Description of the status.
        message: String,
    },

    /// A string passed to nvImageCodec contains a NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// nvImageCodec returned a null handle.
    #[error("unexpected null handle")]
    NullHandle,

    /// A required pointer is null.
    #[error("unexpected null pointer for `{name}`")]
    NullPointer {
        /// Name of the argument or value.
        name: Cow<'static, str>,
    },

    /// A value cannot be consumed while something else still shares it.
    #[error("resource is shared and cannot be consumed")]
    OwnershipShared,

    /// nvImageCodec reported a version that cannot be decoded.
    #[error("invalid version number")]
    InvalidVersion,

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

    /// nvImageCodec processed a batch but reported a failure for one image.
    #[error("image {index} failed with nvimagecodec processing status {status}")]
    ImageStatus {
        /// Index of the image in the batch.
        index: usize,
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

impl From<sys::nvimgcodecStatus_t> for Error {
    fn from(status: sys::nvimgcodecStatus_t) -> Self {
        if status == sys::nvimgcodecStatus_t::NVIMGCODEC_STATUS_NOT_INITIALIZED
            && let Err(error) = singe_nvimagecodec_sys::load()
        {
            return Self::Load(error);
        }

        Self::from(Status::from(status))
    }
}

impl From<Status> for Error {
    fn from(code: Status) -> Self {
        Self::Nvimagecodec {
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

/// Evaluates a native nvImageCodec call and converts its status into a [`Result`](crate::error::Result).
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_nvimagecodec_sys::nvimgcodecStatus_t::NVIMGCODEC_STATUS_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}
