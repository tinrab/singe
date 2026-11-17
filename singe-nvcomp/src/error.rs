use std::{borrow::Cow, io, result::Result as StdResult};

use singe_cuda::{
    error::Error as CudaError,
    view::{AnyBitPatternDeviceRepr, DeviceRepr, ZeroableDeviceRepr},
};
use singe_nvcomp_sys as sys;
use thiserror::Error;

singe_core::wrapper_status! {
    library = "nvcomp",
    error = Error::Nvcomp,
    raw = sys::nvcompStatus_t,
    success = nvcompSuccess,
    absence = sys::take_absence,
    unknown = "UNKNOWN_NVCOMP_STATUS",
    description = sys::nvcompGetStatusString,
    /// Status codes returned by nvCOMP calls.
    pub enum Status: u32 {
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

/// Errors returned by this crate.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// nvCOMP or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// An nvCOMP manager failed or threw a C++ exception.
    #[cfg(feature = "manager")]
    #[error("nvcomp manager error: {message}")]
    Compressor {
        /// Failure or exception message from the manager.
        message: String,
    },

    /// Reading from or writing to a stream failed.
    #[error(transparent)]
    Io(#[from] io::Error),

    /// A null handle was passed where a valid handle was expected.
    #[error("unexpected null handle")]
    NullHandle,

    /// A CUDA call failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// nvCOMP returned an error status.
    #[error("nvcomp error ({code}): {}", code.description())]
    Nvcomp {
        /// Status returned by nvCOMP.
        code: Status,
    },

    /// A native enum value has no typed variant in this crate.
    #[error("unknown `{name}` value {value}")]
    UnknownEnumValue {
        /// Name of the typed enum.
        name: Cow<'static, str>,
        /// Unrecognized native value.
        value: i64,
    },

    /// A value does not fit the integer type that nvCOMP uses for it.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the argument or field with the invalid value.
        name: Cow<'static, str>,
    },

    /// An input that must not be empty is empty.
    #[error("`{name}` is empty")]
    Empty {
        /// Name of the empty input.
        name: Cow<'static, str>,
    },

    /// A size or count that must be positive is zero.
    #[error("`{name}` must be positive")]
    NotPositive {
        /// Name of the argument or field.
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

    /// A buffer is smaller than the extent the operation reads or writes.
    #[error("`{name}` is too small (requires {required}, got {actual})")]
    BufferTooSmall {
        /// Name of the buffer.
        name: Cow<'static, str>,
        /// Required size in bytes.
        required: usize,
        /// Supplied size in bytes.
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

    /// A batched operation reported an error for one chunk.
    #[error("`{operation}` failed with {status}")]
    OperationFailed {
        /// The failed chunk, named by its index in the batch.
        operation: Cow<'static, str>,
        /// Status that nvCOMP wrote for the chunk.
        status: Status,
    },
}

/// Result type returned by this crate.
pub type Result<T> = StdResult<T, Error>;

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
        sys::nvcompStatus_t::try_from(self.status()).ok()
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

impl From<singe_core::UnknownEnumValue> for Error {
    fn from(error: singe_core::UnknownEnumValue) -> Self {
        Self::UnknownEnumValue {
            name: error.name.into(),
            value: error.value,
        }
    }
}

impl From<DeviceStatus> for Status {
    fn from(status: DeviceStatus) -> Self {
        Self::from(status.raw())
    }
}

// Converts a raw nvCOMP status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
