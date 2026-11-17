use std::{borrow::Cow, ffi, result::Result as StdResult};

use singe_cuda::error::Error as CudaError;
use singe_nccl_sys as sys;
use thiserror::Error;

singe_core::wrapper_status! {
    library = "nccl",
    error = Error::Nccl,
    raw = sys::ncclResult_t,
    success = ncclSuccess,
    absence = sys::take_absence,
    unknown = "UNKNOWN_NCCL_STATUS",
    description = sys::ncclGetErrorString,
    /// Status codes returned by NCCL operations.
    pub enum Status: u32 {
        /// Operation succeeded.
        Success = ncclSuccess,
        /// A call to a CUDA function failed.
        UnhandledCudaError = ncclUnhandledCudaError,
        /// A call to the system failed.
        SystemError = ncclSystemError,
        /// An internal check failed.
        /// Usually caused by an NCCL bug or memory corruption.
        InternalError = ncclInternalError,
        /// An argument has an invalid value.
        InvalidArgument = ncclInvalidArgument,
        /// The call to NCCL is incorrect.
        /// This usually reflects a programming error.
        InvalidUsage = ncclInvalidUsage,
        /// A call failed possibly due to a network error or a remote process exiting prematurely.
        RemoteError = ncclRemoteError,
        /// An NCCL operation on the communicator is being enqueued and progressed in the background.
        ///
        /// Whenever an operation returns an error (neither [`Status::Success`] nor [`Status::InProgress`]), NCCL may print a more detailed message when the `NCCL_DEBUG` environment variable is set to `WARN`.
        InProgress = ncclInProgress,
        /// The operation timed out.
        Timeout = ncclTimeout,
    }
}

/// An error from CUDA, NCCL, or validation of wrapper state and arguments.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// NCCL or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// A CUDA operation failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// An NCCL operation returned a failure status.
    #[error("nccl error ({code}): {}", code.description())]
    Nccl {
        /// The NCCL status code.
        code: Status,
    },

    /// A string contains an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native communicator handle is null.
    #[error("unexpected null handle")]
    NullHandle,

    /// A native API returned or received an unexpected null pointer.
    #[error("`{name}` is null")]
    NullPointer {
        /// Name of the pointer or required value.
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

    /// An argument or computed quantity is outside its supported range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// The argument or quantity outside its supported range.
        name: Cow<'static, str>,
    },

    /// Buffer lengths or per-rank counts do not match the operation.
    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        /// The buffers or per-rank values with incompatible lengths.
        name: Cow<'static, str>,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },

    /// A list of buffers or ranks is empty.
    #[error("`{name}` is empty")]
    Empty {
        /// Name of the list.
        name: Cow<'static, str>,
    },

    /// An operation requires the device primary CUDA context.
    #[error("nccl requires the device primary cuda context")]
    UnsupportedContext,

    /// An NCCL object belongs to a different communicator.
    #[error("`{name}` belongs to a different nccl context")]
    ContextMismatch {
        /// Name of the object.
        name: Cow<'static, str>,
    },

    /// An operation cannot run inside an open NCCL group.
    #[error("operation cannot run inside an open nccl group")]
    GroupActive,

    /// The stream belongs to another CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// The communicator state does not permit the operation.
    #[error("communicator is in {state} state")]
    InvalidCommunicatorState {
        /// The communicator state that prevented the operation.
        state: String,
    },
}

/// An NCCL wrapper result with [`enum@Error`] as its error type.
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

impl From<ffi::NulError> for Error {
    fn from(_: ffi::NulError) -> Self {
        Self::InteriorNul
    }
}

// Converts a raw NCCL status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
