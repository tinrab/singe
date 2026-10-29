use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_core::status_enum;
use singe_cuda::error::Error as CudaError;
use singe_cufft_sys as sys;
use thiserror::Error;

status_enum! {
    /// All cuFFT Library return values except for `cufftResult::CUFFT_SUCCESS` indicate that the current API call failed and the user should reconfigure to correct the problem. The possible return values are defined as follows.
    pub enum Status: u32 => sys::cufftResult_t, unknown = "UNKNOWN_CUFFT_STATUS" {
        /// The cuFFT operation was successful.
        Success = CUFFT_SUCCESS,
        /// cuFFT was passed an invalid plan handle.
        InvalidPlan = CUFFT_INVALID_PLAN,
        /// cuFFT failed to allocate GPU or CPU memory.
        AllocFailed = CUFFT_ALLOC_FAILED,
        /// The cuFFT type provided is unsupported.
        InvalidType = CUFFT_INVALID_TYPE,
        /// User specified an invalid pointer or parameter.
        InvalidValue = CUFFT_INVALID_VALUE,
        /// Driver or internal cuFFT library error.
        InternalError = CUFFT_INTERNAL_ERROR,
        /// Failed to execute an FFT on the GPU.
        ExecFailed = CUFFT_EXEC_FAILED,
        /// The cuFFT library failed to initialize.
        SetupFailed = CUFFT_SETUP_FAILED,
        /// User specified an invalid transform size.
        InvalidSize = CUFFT_INVALID_SIZE,
        /// Not currently in use.
        UnalignedData = CUFFT_UNALIGNED_DATA,
        /// Execution of a plan was on different GPU than plan creation.
        InvalidDevice = CUFFT_INVALID_DEVICE,
        /// No workspace has been provided prior to plan execution.
        NoWorkspace = CUFFT_NO_WORKSPACE,
        /// Function does not implement functionality for parameters given.
        NotImplemented = CUFFT_NOT_IMPLEMENTED,
        /// Operation is not supported for parameters given.
        NotSupported = CUFFT_NOT_SUPPORTED,
        /// cuFFT is unable to find a dependency.
        MissingDependency = CUFFT_MISSING_DEPENDENCY,
        /// An NVRTC failure was encountered during a cuFFT operation.
        NvrtcFailure = CUFFT_NVRTC_FAILURE,
        /// An nvJitLink failure was encountered during a cuFFT operation.
        NvjitlinkFailure = CUFFT_NVJITLINK_FAILURE,
        /// An NVSHMEM failure was encountered during a cuFFT operation.
        NvshmemFailure = CUFFT_NVSHMEM_FAILURE,
    }
}

impl Status {
    /// Returns a short description of this status, including unrecognized native codes.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::InvalidPlan => "invalid plan",
            Self::AllocFailed => "allocation failed",
            Self::InvalidType => "invalid transform type",
            Self::InvalidValue => "invalid value",
            Self::InternalError => "internal error",
            Self::ExecFailed => "execution failed",
            Self::SetupFailed => "library setup failed",
            Self::InvalidSize => "invalid size",
            Self::UnalignedData => "unaligned data",
            Self::InvalidDevice => "invalid device",
            Self::NoWorkspace => "insufficient workspace",
            Self::NotImplemented => "not implemented",
            Self::NotSupported => "not supported",
            Self::MissingDependency => "missing dependency",
            Self::NvrtcFailure => "nvrtc failure",
            Self::NvjitlinkFailure => "nvjitlink failure",
            Self::NvshmemFailure => "nvshmem failure",
            Self::Unknown(_) => "unknown cufft error",
        }
    }
}

/// Errors from cuFFT, CUDA operations, and wrapper argument validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// A distributed allocation still retains the plan, preventing configuration changes.
    #[error("resource is shared and cannot be consumed")]
    OwnershipShared,
    /// A CUDA operation failed.
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    /// cuFFT returned a failure status.
    #[error("cufft error ({code}): {message}")]
    Cufft {
        /// Native cuFFT status, preserving unrecognized status codes.
        code: Status,
        /// Text describing the native status.
        message: String,
    },

    /// A callback symbol or linker input name contains an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// The native API returned a null handle or descriptor.
    #[error("unexpected null handle")]
    NullHandle,

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

    /// A stream or memory region belongs to a different CUDA context from the plan.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// The builder has no transform shape.
    #[error("plan builder is missing a transform definition")]
    MissingPlanDefinition,
}

/// Result of a cuFFT wrapper operation.
pub type Result<T> = StdResult<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

impl From<sys::cufftResult_t> for Error {
    fn from(status: sys::cufftResult_t) -> Self {
        let code = Status::from(status);
        Self::Cufft {
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

/// Converts a raw cuFFT status-returning expression into a crate [`Result`](crate::error::Result).
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_cufft_sys::cufftResult_t::CUFFT_SUCCESS {
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
        let status = Status::from(sys::cufftResult_t::CUFFT_INVALID_VALUE);
        assert_eq!(status, Status::InvalidValue);
        assert_eq!(status.raw(), sys::cufftResult_t::CUFFT_INVALID_VALUE as u32);
    }

    #[test]
    fn test_status_display_uses_c_name() {
        assert_eq!(Status::InvalidValue.to_string(), "CUFFT_INVALID_VALUE");
    }

    #[test]
    fn test_unknown_status_keeps_raw_code() {
        let status = Status::Unknown(9999);
        assert_eq!(status.raw(), 9999);
        assert_eq!(status.to_string(), "UNKNOWN_CUFFT_STATUS(9999)");
    }
}
