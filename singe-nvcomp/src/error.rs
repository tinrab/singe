use std::{borrow::Cow, io, result::Result as StdResult};

use singe_core::{static_str_from_c_ptr, status_enum};
use singe_cuda::{
    error::Error as CudaError,
    view::{AnyBitPatternDeviceRepr, DeviceRepr, ZeroableDeviceRepr},
};
use singe_nvcomp_sys as sys;
use thiserror::Error;

/// Errors returned by this crate.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    // Its library would not load.
    #[error(transparent)]
    Load(#[from] singe_nvcomp_sys::LoadError),

    /// An nvCOMP manager failed or threw a C++ exception.
    #[error("nvcomp manager error: {message}")]
    Compressor {
        /// Failure or exception message from the manager.
        message: String,
    },

    /// Reading from or writing to a stream failed.
    #[error(transparent)]
    Io(#[from] io::Error),

    /// A CUDA call failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// nvCOMP returned an error status.
    #[error("nvcomp error ({code}): {message}")]
    Nvcomp {
        /// Status returned by nvCOMP.
        code: Status,
        /// nvCOMP's description of the status.
        message: String,
    },

    /// A value does not fit the integer type that nvCOMP uses for it.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the argument or field with the invalid value.
        name: Cow<'static, str>,
    },

    /// An argument has an invalid value.
    #[error("`{name}` is invalid")]
    InvalidValue {
        /// Name of the invalid argument or field.
        name: Cow<'static, str>,
    },

    /// A size exceeds the limit nvCOMP documents for the codec.
    #[error("`{name}` exceeds maximum supported size (max {max}, got {actual})")]
    SizeLimitExceeded {
        /// Name of the argument or field with the invalid value.
        name: Cow<'static, str>,
        /// Largest supported size.
        max: usize,
        /// Size that was passed.
        actual: usize,
    },

    /// A stream belongs to a different CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

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

    /// A chunk's byte count differs from the expected value.
    #[error("chunk {index} `{name}` length mismatch (expected {expected}, got {actual})")]
    ChunkLengthMismatch {
        /// Index of the chunk in the batch.
        index: usize,
        /// Name of the mismatched field.
        name: Cow<'static, str>,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// A device buffer does not meet nvCOMP's alignment requirement.
    #[error("`{name}` does not satisfy {alignment}-byte alignment")]
    AlignmentMismatch {
        /// Name of the buffer or table with the invalid alignment.
        name: Cow<'static, str>,
        /// Required byte alignment.
        alignment: usize,
    },

    /// A batched operation reported an error for one chunk.
    #[error("chunk {index} failed with nvcomp status {status}")]
    ChunkStatus {
        /// Index of the chunk in the batch.
        index: usize,
        /// Status that nvCOMP wrote for the chunk.
        status: Status,
    },
}

/// Result type returned by this crate.
pub type Result<T> = StdResult<T, Error>;

/// Evaluates a native nvCOMP call and converts its status into a [`Result`].
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_nvcomp_sys::nvcompStatus_t::nvcompSuccess {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}

status_enum! {
    /// Status codes returned by nvCOMP calls.
    pub enum Status: u32 => sys::nvcompStatus_t, unknown = "UNKNOWN_NVCOMP_STATUS" {
        /// The operation succeeded.
        Success = nvcompSuccess,
        /// An argument has an invalid value.
        InvalidValue = nvcompErrorInvalidValue,
        /// The requested operation or configuration is unsupported.
        NotSupported = nvcompErrorNotSupported,
        /// The input could not be decompressed.
        CannotDecompress = nvcompErrorCannotDecompress,
        /// A stored checksum does not match the computed checksum.
        BadChecksum = nvcompErrorBadChecksum,
        /// The requested checksum verification could not be performed.
        CannotVerifyChecksums = nvcompErrorCannotVerifyChecksums,
        /// The output buffer cannot hold the produced data.
        OutputBufferTooSmall = nvcompErrorOutputBufferTooSmall,
        /// The compressed-stream header has an invalid length.
        WrongHeaderLength = nvcompErrorWrongHeaderLength,
        /// A buffer does not satisfy the required alignment.
        Alignment = nvcompErrorAlignment,
        /// An input chunk exceeds the supported size.
        ChunkSizeTooLarge = nvcompErrorChunkSizeTooLarge,
        /// The input could not be compressed.
        CannotCompress = nvcompErrorCannotCompress,
        /// The input byte count is invalid for the selected codec or data representation.
        WrongInputLength = nvcompErrorWrongInputLength,
        /// The batch exceeds the supported number of chunks.
        BatchSizeTooLarge = nvcompErrorBatchSizeTooLarge,
        /// The requested subchunk count exceeds the supported limit.
        SubChunkCountTooLarge = nvcompErrorSubChunkCountTooLarge,
        /// The requested subchunk count is below the supported minimum.
        SubChunkCountTooSmall = nvcompErrorSubChunkCountTooSmall,
        /// The output buffer alignment is weaker than the required alignment.
        OutputBufferAlignmentTooSmall = nvcompErrorOutputBufferAlignmentTooSmall,
        /// A CUDA operation failed.
        CudaError = nvcompErrorCudaError,
        /// An internal nvCOMP error occurred.
        Internal = nvcompErrorInternal,
    }
}

impl Status {
    /// Returns nvCOMP's description of this status.
    #[must_use]
    pub fn description(self) -> &'static str {
        let Ok(raw) = sys::nvcompStatus_t::try_from(self) else {
            return "unknown nvcomp error";
        };
        // SAFETY: `nvcompGetStatusString` returns null or a static NUL-terminated string.
        unsafe { static_str_from_c_ptr(sys::nvcompGetStatusString(raw), "unknown nvcomp error") }
    }
}

/// Per-chunk status that batched nvCOMP operations write to device memory.
///
/// It has the same layout as the native status, so a device buffer of these can be passed to nvCOMP directly.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Default,
    DeviceRepr,
    ZeroableDeviceRepr,
    AnyBitPatternDeviceRepr,
)]
#[repr(transparent)]
pub struct DeviceStatus(u32);

impl DeviceStatus {
    /// Wraps a raw status code.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the raw status code.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Returns the native status, or `None` if the code is not a known nvCOMP status.
    #[must_use]
    pub fn try_as_raw(self) -> Option<sys::nvcompStatus_t> {
        sys::nvcompStatus_t::try_from(self.0).ok()
    }

    /// Converts the code to a [`Status`].
    #[must_use]
    pub fn status(self) -> Status {
        self.into()
    }
}

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

impl From<DeviceStatus> for Status {
    fn from(status: DeviceStatus) -> Self {
        Self::from(status.raw())
    }
}

impl From<sys::nvcompStatus_t> for Error {
    fn from(status: sys::nvcompStatus_t) -> Self {
        if status == sys::nvcompStatus_t::nvcompErrorInternal
            && let Err(error) = singe_nvcomp_sys::load()
        {
            return Self::Load(error);
        }

        Self::from(Status::from(status))
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        Self::Nvcomp {
            code: status,
            message: status.description().to_string(),
        }
    }
}
