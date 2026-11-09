//! Errors and nvTIFF status codes.

use std::{borrow::Cow, ffi::NulError, io, result::Result as StdResult};

use singe_core::status_enum;
use singe_cuda::error::Error as CudaError;
use singe_nvtiff_sys as sys;
use thiserror::Error;

/// Error returned by nvTIFF wrappers.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    // Its library would not load.
    #[error(transparent)]
    Load(#[from] singe_nvtiff_sys::LoadError),

    /// CUDA call failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// Writing an encoded TIFF file failed.
    #[error(transparent)]
    Io(#[from] io::Error),

    /// nvTIFF call failed.
    #[error("nvtiff error ({code}): {message}")]
    Nvtiff {
        /// Status returned by nvTIFF.
        code: Status,
        /// Description of the status.
        message: String,
    },

    /// nvTIFF returned a null handle.
    #[error("unexpected null handle")]
    NullHandle,

    /// Pointer was null.
    #[error("unexpected null pointer in `{name}`")]
    NullPointer {
        /// Name of the argument or field.
        name: Cow<'static, str>,
    },

    /// A stream belongs to a different CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// Value does not fit the type nvTIFF expects.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the argument or field.
        name: Cow<'static, str>,
    },

    /// Length differs from the one nvTIFF expects.
    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        /// Name of the argument or field.
        name: Cow<'static, str>,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// Buffer is smaller than the data written to it.
    #[error("`{name}` buffer too small (required {required} bytes, got {actual} bytes)")]
    BufferTooSmall {
        /// Name of the buffer.
        name: Cow<'static, str>,
        /// Required size in bytes.
        required: usize,
        /// Actual size in bytes.
        actual: usize,
    },

    /// String contains a NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,
}

/// Result type of nvTIFF wrappers.
pub type Result<T> = StdResult<T, Error>;

impl Error {
    /// Returns whether this error means an optional tag or `GeoTIFF` key is absent.
    #[must_use]
    pub fn is_missing_metadata(&self) -> bool {
        matches!(
            self,
            Self::Nvtiff {
                code: Status::TagNotFound | Status::TiffNotSupported,
                ..
            }
        )
    }

    /// Returns whether nvTIFF does not support the TIFF file.
    #[must_use]
    pub fn is_unsupported_tiff(&self) -> bool {
        matches!(
            self,
            Self::Nvtiff {
                code: Status::TiffNotSupported,
                ..
            }
        )
    }
}

status_enum! {
    /// Status returned by nvTIFF functions.
    pub enum Status: u32 => sys::nvtiffStatus_t, unknown = "UNKNOWN_NVTIFF_STATUS", unknown_description = "unknown nvtiff error" {
        /// The call succeeded.
        ///
        /// Asynchronous work can still fail later and report the error on a later call.
        Success = NVTIFF_STATUS_SUCCESS => "success",
        /// The handle was not initialized.
        NotInitialized = NVTIFF_STATUS_NOT_INITIALIZED => "not initialized",
        /// An argument was invalid, such as a null pointer or an unknown enum value.
        InvalidParameter = NVTIFF_STATUS_INVALID_PARAMETER => "invalid parameter",
        /// The TIFF data is corrupt and cannot be parsed.
        BadTiff = NVTIFF_STATUS_BAD_TIFF => "bad tiff",
        /// nvTIFF does not support decoding this TIFF file.
        TiffNotSupported = NVTIFF_STATUS_TIFF_NOT_SUPPORTED => "tiff not supported",
        /// A custom allocator failed.
        AllocatorFailure = NVTIFF_STATUS_ALLOCATOR_FAILURE => "allocator failure",
        /// GPU work failed.
        ExecutionFailed = NVTIFF_STATUS_EXECUTION_FAILED => "execution failed",
        /// The device does not support the requested work.
        ArchitectureMismatch = NVTIFF_STATUS_ARCH_MISMATCH => "architecture mismatch",
        /// nvTIFF hit an internal error.
        InternalError = NVTIFF_STATUS_INTERNAL_ERROR => "internal error",
        /// nvTIFF cannot load nvCOMP.
        NvcompNotFound = NVTIFF_STATUS_NVCOMP_NOT_FOUND => "nvcomp not found",
        /// nvTIFF cannot load nvJPEG.
        NvjpegNotFound = NVTIFF_STATUS_NVJPEG_NOT_FOUND => "nvjpeg not found",
        /// The tag or `GeoTIFF` key is absent.
        TagNotFound = NVTIFF_STATUS_TAG_NOT_FOUND => "tag not found",
        /// An argument is out of range.
        ParameterOutOfBounds = NVTIFF_STATUS_PARAMETER_OUT_OF_BOUNDS => "parameter out of bounds",
        /// nvTIFF cannot load nvJPEG 2000.
        Nvjpeg2kNotFound = NVTIFF_STATUS_NVJPEG2K_NOT_FOUND => "nvjpeg2k not found",
        /// Each region can be decoded alone, but not together in one batch.
        BatchIncompatible = NVTIFF_STATUS_BATCH_INCOMPATIBLE => "batch incompatible",
    }
}

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

impl From<sys::nvtiffStatus_t> for Error {
    fn from(status: sys::nvtiffStatus_t) -> Self {
        if status == sys::nvtiffStatus_t::NVTIFF_STATUS_NOT_INITIALIZED
            && let Err(error) = singe_nvtiff_sys::load()
        {
            return Self::Load(error);
        }

        Self::from(Status::from(status))
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        Self::Nvtiff {
            code: status,
            message: status.description().into(),
        }
    }
}

/// Converts an nvTIFF status into a [`Result`].
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_nvtiff_sys::nvtiffStatus_t::NVTIFF_STATUS_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_converts_from_raw_code() {
        let status = Status::from(sys::nvtiffStatus_t::NVTIFF_STATUS_INVALID_PARAMETER);
        assert_eq!(status, Status::InvalidParameter);
        assert_eq!(
            status.raw(),
            sys::nvtiffStatus_t::NVTIFF_STATUS_INVALID_PARAMETER as u32
        );
    }

    #[test]
    fn test_status_display_uses_c_name() {
        assert_eq!(
            Status::InvalidParameter.to_string(),
            "NVTIFF_STATUS_INVALID_PARAMETER"
        );
    }

    #[test]
    fn test_unknown_status_keeps_raw_code() {
        let status = Status::Unknown(9999);
        assert_eq!(status.raw(), 9999);
        assert_eq!(status.to_string(), "UNKNOWN_NVTIFF_STATUS(9999)");
    }
}
