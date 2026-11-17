use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_cuda::{data_type::DataType, error::Error as CudaError};
use singe_cufft_sys as sys;
use thiserror::Error;

use crate::types::{Direction, TransformType};

singe_core::wrapper_status! {
    library = "cufft",
    error = Error::Cufft,
    raw = sys::cufftResult_t,
    success = CUFFT_SUCCESS,
    absence = sys::take_absence,
    unknown = "UNKNOWN_CUFFT_STATUS",
    /// Status returned by a cuFFT call.
    ///
    /// Every status except [`Status::Success`] means the call failed.
    pub enum Status: u32 {
        /// The cuFFT operation was successful.
        Success = CUFFT_SUCCESS => "success",
        /// cuFFT was passed an invalid plan handle.
        InvalidPlan = CUFFT_INVALID_PLAN => "invalid plan",
        /// cuFFT failed to allocate GPU or CPU memory.
        AllocFailed = CUFFT_ALLOC_FAILED => "allocation failed",
        /// The cuFFT type provided is unsupported.
        InvalidType = CUFFT_INVALID_TYPE => "invalid transform type",
        /// User specified an invalid pointer or parameter.
        InvalidValue = CUFFT_INVALID_VALUE => "invalid value",
        /// Driver or internal cuFFT library error.
        InternalError = CUFFT_INTERNAL_ERROR => "internal error",
        /// Failed to execute an FFT on the GPU.
        ExecFailed = CUFFT_EXEC_FAILED => "execution failed",
        /// The cuFFT library failed to initialize.
        SetupFailed = CUFFT_SETUP_FAILED => "library setup failed",
        /// User specified an invalid transform size.
        InvalidSize = CUFFT_INVALID_SIZE => "invalid size",
        /// Not currently in use.
        UnalignedData = CUFFT_UNALIGNED_DATA => "unaligned data",
        /// Execution of a plan was on different GPU than plan creation.
        InvalidDevice = CUFFT_INVALID_DEVICE => "invalid device",
        /// No workspace has been provided prior to plan execution.
        NoWorkspace = CUFFT_NO_WORKSPACE => "insufficient workspace",
        /// Function does not implement functionality for parameters given.
        NotImplemented = CUFFT_NOT_IMPLEMENTED => "not implemented",
        /// Operation is not supported for parameters given.
        NotSupported = CUFFT_NOT_SUPPORTED => "not supported",
        /// cuFFT is unable to find a dependency.
        MissingDependency = CUFFT_MISSING_DEPENDENCY => "missing dependency",
        /// An NVRTC failure was encountered during a cuFFT operation.
        NvrtcFailure = CUFFT_NVRTC_FAILURE => "nvrtc failure",
        /// An nvJitLink failure was encountered during a cuFFT operation.
        NvjitlinkFailure = CUFFT_NVJITLINK_FAILURE => "nvjitlink failure",
        /// An NVSHMEM failure was encountered during a cuFFT operation.
        NvshmemFailure = CUFFT_NVSHMEM_FAILURE => "nvshmem failure",
    }
}

/// Errors from cuFFT, CUDA operations, and wrapper argument validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// cuFFT or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// A lock was poisoned by a thread that panicked while holding it.
    #[error("`{name}` lock is poisoned")]
    LockPoisoned {
        /// What the lock guards.
        name: Cow<'static, str>,
    },

    /// A distributed allocation still retains the plan, preventing configuration changes.
    #[error("`{name}` is still shared")]
    OwnershipShared {
        /// Name of the shared resource.
        name: Cow<'static, str>,
    },

    /// A CUDA operation failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// cuFFT returned a failure status.
    #[error("cufft error ({code}): {}", code.description())]
    Cufft {
        /// Native cuFFT status, preserving unrecognized status codes.
        code: Status,
    },

    /// A callback symbol or linker input name contains an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// The native API returned a null handle or descriptor.
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

    /// A value, storage extent, or configuration is outside the supported range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Argument or configuration that could not be represented or accepted.
        name: Cow<'static, str>,
    },

    /// An embedding array has a different length from the transform rank.
    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        /// Embedding array whose length is incorrect.
        name: Cow<'static, str>,
        /// Number of dimensions required by the transform.
        expected: usize,
        /// Number of embedding extents supplied.
        actual: usize,
    },

    /// A buffer or work area is smaller than the configured transform requires.
    #[error("`{name}` is too small (requires {required}, got {actual})")]
    BufferTooSmall {
        /// Name of the buffer.
        name: Cow<'static, str>,
        /// Required number of elements, or bytes for a work area or distributed storage.
        required: usize,
        /// Supplied number of elements, or bytes for a work area or distributed storage.
        actual: usize,
    },

    /// A buffer's element type differs from the data type the layout configures.
    #[error("`{name}` data type mismatch (expected `{expected}`, got `{actual}`)")]
    DataTypeMismatch {
        /// Name of the buffer.
        name: Cow<'static, str>,
        /// Data type the layout configures.
        expected: DataType,
        /// Element type of the supplied buffer.
        actual: DataType,
    },

    /// A distributed allocation belongs to another native plan.
    #[error("distributed allocation belongs to another plan")]
    PlanMismatch,

    /// A stream belongs to a different CUDA context from the plan.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// The plan has no transform recorded by one of its planning methods.
    #[error("plan has no configured transform")]
    PlanNotConfigured,

    /// The operand element types select a different transform from the configured one.
    #[error("operands select a `{actual}` transform, but the plan is configured for `{expected}`")]
    TransformTypeMismatch {
        /// Transform the plan is configured for.
        expected: TransformType,
        /// Transform the operand element types select.
        actual: TransformType,
    },

    /// A real transform was asked to run in the direction it does not support.
    #[error("`{kind}` transforms do not run in direction `{direction}`")]
    UnsupportedDirection {
        /// Transform the operand element types select.
        kind: TransformType,
        /// Requested direction.
        direction: Direction,
    },
}

/// Result of a cuFFT wrapper operation.
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

// Converts a raw cuFFT status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
