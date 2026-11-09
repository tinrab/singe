//! Error and status types.

use std::{borrow::Cow, io, result::Result as StdResult};

use singe_core::status_enum;
use singe_cuda::error::Error as CudaError;
use singe_nvjpeg_sys as sys;
#[cfg(feature = "k2")]
use singe_nvjpeg_sys::k2 as k2_sys;
use thiserror::Error;

#[cfg(feature = "k2")]
use crate::k2::error::Status as K2Status;

/// Errors returned by this crate.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    // Its library would not load.
    #[error(transparent)]
    Load(#[from] singe_nvjpeg_sys::LoadError),

    /// A lock was poisoned by a thread that panicked while holding it.
    #[error("`{name}` lock is poisoned")]
    LockPoisoned {
        /// What the lock guards.
        name: &'static str,
    },

    /// A CUDA call failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// Reading or writing a file failed.
    #[error(transparent)]
    Io(#[from] io::Error),

    /// nvJPEG returned an error status.
    #[error("nvjpeg error ({code}): {message}")]
    Nvjpeg {
        /// Status returned by nvJPEG.
        code: Status,
        /// Human-readable status description.
        message: String,
    },

    /// nvJPEG 2000 returned an error status.
    #[cfg(feature = "k2")]
    #[error("nvjpeg2k error ({code}): {message}")]
    Nvjpeg2k {
        /// Status returned by nvJPEG 2000.
        code: K2Status,
        /// Human-readable status description.
        message: String,
    },

    /// nvJPEG returned a null handle.
    #[error("unexpected null handle")]
    NullHandle,

    /// A stream belongs to a different CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// Values used together were created with different nvJPEG handles.
    #[error("resource belongs to a different nvjpeg handle")]
    HandleMismatch,

    /// The operation needs a parsed bitstream, but none was parsed successfully.
    #[error("bitstream has no completed parse")]
    BitstreamNotParsed,

    /// A required pointer is null.
    #[error("`{name}` is null")]
    NullPointer {
        /// Name of the pointer.
        name: Cow<'static, str>,
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
    #[error("`{name}` buffer too small (required {required} bytes, got {actual} bytes)")]
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
        reason: String,
    },

    /// The encode settings are not supported by the chosen way of encoding.
    #[error("unsupported encode configuration: {reason}")]
    UnsupportedEncodeConfiguration {
        /// Reason the configuration is unsupported.
        reason: String,
    },

    /// An input that must not be empty is empty.
    #[error("`{name}` is empty")]
    Empty {
        /// Name of the empty input.
        name: Cow<'static, str>,
    },

    /// A string passed to nvJPEG contains a NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,
}

/// Result type returned by this crate.
pub type Result<T> = StdResult<T, Error>;

status_enum! {
    /// Status codes returned by nvJPEG calls.
    pub enum Status: u32 => sys::nvjpegStatus_t, unknown = "UNKNOWN_NVJPEG_STATUS", unknown_description = "unknown nvjpeg error" {
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

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

impl From<sys::nvjpegStatus_t> for Error {
    fn from(status: sys::nvjpegStatus_t) -> Self {
        if status == sys::nvjpegStatus_t::NVJPEG_STATUS_NOT_INITIALIZED
            && let Err(error) = singe_nvjpeg_sys::load()
        {
            return Self::Load(error);
        }

        Self::from(Status::from(status))
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        Self::Nvjpeg {
            code: status,
            message: status.description().into(),
        }
    }
}

#[cfg(feature = "k2")]
impl From<k2_sys::nvjpeg2kStatus_t> for Error {
    fn from(status: k2_sys::nvjpeg2kStatus_t) -> Self {
        if status == k2_sys::nvjpeg2kStatus_t::NVJPEG2K_STATUS_NOT_INITIALIZED
            && let Err(error) = singe_nvjpeg_sys::load_k2()
        {
            return Self::Load(error);
        }

        Self::from(K2Status::from(status))
    }
}

#[cfg(feature = "k2")]
impl From<K2Status> for Error {
    fn from(status: K2Status) -> Self {
        Self::Nvjpeg2k {
            code: status,
            message: status.description().into(),
        }
    }
}

/// Evaluates a native nvJPEG call and converts its status into a [`Result`].
///
/// The expression is evaluated once.
/// [`Status::Success`] becomes `Ok(())`; every other status becomes [`Error::Nvjpeg`].
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_nvjpeg_sys::nvjpegStatus_t::NVJPEG_STATUS_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}
