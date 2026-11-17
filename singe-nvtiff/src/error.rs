//! Errors and nvTIFF status codes.

use std::{borrow::Cow, ffi::NulError, io, result::Result as StdResult};

use singe_cuda::error::Error as CudaError;
use singe_nvtiff_sys as sys;
use thiserror::Error;

singe_core::wrapper_status! {
    library = "nvtiff",
    error = Error::Nvtiff,
    raw = sys::nvtiffStatus_t,
    success = NVTIFF_STATUS_SUCCESS,
    absence = sys::take_absence,
    unknown = "UNKNOWN_NVTIFF_STATUS",
    /// Status returned by nvTIFF functions.
    pub enum Status: u32 {
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

/// Error returned by nvTIFF wrappers.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// nvTIFF or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// CUDA call failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// Writing an encoded TIFF file failed.
    #[error(transparent)]
    Io(#[from] io::Error),

    /// nvTIFF call failed.
    #[error("nvtiff error ({code}): {}", code.description())]
    Nvtiff {
        /// Status returned by nvTIFF.
        code: Status,
    },

    /// nvTIFF returned a null handle.
    #[error("unexpected null handle")]
    NullHandle,

    /// An nvTIFF object belongs to a different CUDA context.
    #[error("`{name}` belongs to a different cuda context")]
    ContextMismatch {
        /// Name of the object.
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
    #[error("`{name}` is too small (requires {required}, got {actual})")]
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

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

// Converts a raw nvTIFF status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
