//! Error and status types.

use std::{borrow::Cow, result::Result as StdResult};

use singe_cuda::error::Error as CudaError;
use singe_nvjpeg_sys as sys;
use thiserror::Error;

#[cfg(feature = "k2")]
use crate::k2::error::Status as K2Status;

singe_core::wrapper_status! {
    library = "nvjpeg",
    error = Error::Nvjpeg,
    raw = sys::nvjpegStatus_t,
    success = NVJPEG_STATUS_SUCCESS,
    absence = sys::take_absence,
    unknown = "UNKNOWN_NVJPEG_STATUS",
    /// Status codes returned by nvJPEG calls.
    pub enum Status: u32 {
        /// The call succeeded.
        ///
        /// Many calls are asynchronous, so some errors appear only after synchronizing.
        Success = NVJPEG_STATUS_SUCCESS => "success",
        /// The library handle was not initialized.
        NotInitialized = NVJPEG_STATUS_NOT_INITIALIZED => "not initialized",
        /// An argument is invalid, such as null input data or an image index out of range.
        InvalidParameter = NVJPEG_STATUS_INVALID_PARAMETER => "invalid parameter",
        /// The JPEG data cannot be parsed; check the data and its size.
        BadJpeg = NVJPEG_STATUS_BAD_JPEG => "bad jpeg",
        /// nvJPEG does not support decoding this JPEG.
        JpegNotSupported = NVJPEG_STATUS_JPEG_NOT_SUPPORTED => "jpeg not supported",
        /// A custom allocation or free callback returned a nonzero code.
        AllocatorFailure = NVJPEG_STATUS_ALLOCATOR_FAILURE => "allocator failure",
        /// Work on the device failed.
        ExecutionFailed = NVJPEG_STATUS_EXECUTION_FAILED => "execution failed",
        /// The device does not support the requested backend, JPEG features, or output format.
        ArchitectureMismatch = NVJPEG_STATUS_ARCH_MISMATCH => "architecture mismatch",
        /// An internal nvJPEG error occurred.
        InternalError = NVJPEG_STATUS_INTERNAL_ERROR => "internal error",
        /// No implementation supports the request.
        ImplementationNotSupported = NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED => "implementation not supported",
        /// The JPEG data is incomplete.
        IncompleteBitstream = NVJPEG_STATUS_INCOMPLETE_BITSTREAM => "incomplete bitstream",
    }
}

/// Errors returned by this crate.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// nvJPEG, nvJPEG 2000, or one of their symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// A CUDA call failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// nvJPEG returned an error status.
    #[error("nvjpeg error ({code}): {}", code.description())]
    Nvjpeg {
        /// Status returned by nvJPEG.
        code: Status,
    },

    /// nvJPEG 2000 returned an error status.
    #[cfg(feature = "k2")]
    #[error("nvjpeg2k error ({code}): {}", code.description())]
    Nvjpeg2k {
        /// Status returned by nvJPEG 2000.
        code: K2Status,
    },

    /// nvJPEG returned a null handle.
    #[error("unexpected null handle")]
    NullHandle,

    /// A stream belongs to a different CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// An nvJPEG object was created with a different nvJPEG handle.
    #[error("`{name}` belongs to a different nvjpeg context")]
    ContextMismatch {
        /// Name of the object.
        name: Cow<'static, str>,
    },

    /// The operation needs a parsed bitstream, but none was parsed successfully.
    #[error("bitstream has no completed parse")]
    BitstreamNotParsed,

    /// A required pointer is null.
    #[error("`{name}` is null")]
    NullPointer {
        /// Name of the pointer.
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

    /// A value does not fit the integer type that nvJPEG uses for it.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the argument or field with the invalid value.
        name: Cow<'static, str>,
    },

    /// A slice has a different length than the operation requires.
    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        /// Name of the argument or field with the mismatch.
        name: Cow<'static, str>,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// A buffer is smaller than the image plane it holds.
    #[error("`{name}` is too small (requires {required}, got {actual})")]
    BufferTooSmall {
        /// Name of the buffer with the invalid size.
        name: Cow<'static, str>,
        /// Required size in bytes.
        required: usize,
        /// Actual size in bytes.
        actual: usize,
    },

    /// The decode settings are not supported by the chosen way of decoding.
    #[error("unsupported decode configuration: {reason}")]
    UnsupportedDecodeConfiguration {
        /// Reason the configuration is unsupported.
        reason: Cow<'static, str>,
    },

    /// The encode settings are not supported by the chosen way of encoding.
    #[error("unsupported encode configuration: {reason}")]
    UnsupportedEncodeConfiguration {
        /// Reason the configuration is unsupported.
        reason: Cow<'static, str>,
    },

    /// An input that must not be empty is empty.
    #[error("`{name}` is empty")]
    Empty {
        /// Name of the empty input.
        name: Cow<'static, str>,
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

// Converts a raw nvJPEG status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
