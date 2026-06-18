use std::{ffi::NulError, fmt};

use singe_cuda::error::Error as CudaError;
use singe_cufft_sys as sys;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    Success,
    InvalidPlan,
    AllocFailed,
    InvalidType,
    InvalidValue,
    InternalError,
    ExecFailed,
    SetupFailed,
    InvalidSize,
    UnalignedData,
    InvalidDevice,
    NoWorkspace,
    NotImplemented,
    NotSupported,
    MissingDependency,
    NvrtcFailure,
    NvjitlinkFailure,
    NvshmemFailure,
    Unknown(u32),
}

impl Status {
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

    pub const fn raw(self) -> u32 {
        match self {
            Self::Success => sys::cufftResult_t::CUFFT_SUCCESS as _,
            Self::InvalidPlan => sys::cufftResult_t::CUFFT_INVALID_PLAN as _,
            Self::AllocFailed => sys::cufftResult_t::CUFFT_ALLOC_FAILED as _,
            Self::InvalidType => sys::cufftResult_t::CUFFT_INVALID_TYPE as _,
            Self::InvalidValue => sys::cufftResult_t::CUFFT_INVALID_VALUE as _,
            Self::InternalError => sys::cufftResult_t::CUFFT_INTERNAL_ERROR as _,
            Self::ExecFailed => sys::cufftResult_t::CUFFT_EXEC_FAILED as _,
            Self::SetupFailed => sys::cufftResult_t::CUFFT_SETUP_FAILED as _,
            Self::InvalidSize => sys::cufftResult_t::CUFFT_INVALID_SIZE as _,
            Self::UnalignedData => sys::cufftResult_t::CUFFT_UNALIGNED_DATA as _,
            Self::InvalidDevice => sys::cufftResult_t::CUFFT_INVALID_DEVICE as _,
            Self::NoWorkspace => sys::cufftResult_t::CUFFT_NO_WORKSPACE as _,
            Self::NotImplemented => sys::cufftResult_t::CUFFT_NOT_IMPLEMENTED as _,
            Self::NotSupported => sys::cufftResult_t::CUFFT_NOT_SUPPORTED as _,
            Self::MissingDependency => sys::cufftResult_t::CUFFT_MISSING_DEPENDENCY as _,
            Self::NvrtcFailure => sys::cufftResult_t::CUFFT_NVRTC_FAILURE as _,
            Self::NvjitlinkFailure => sys::cufftResult_t::CUFFT_NVJITLINK_FAILURE as _,
            Self::NvshmemFailure => sys::cufftResult_t::CUFFT_NVSHMEM_FAILURE as _,
            Self::Unknown(code) => code,
        }
    }
}

impl From<sys::cufftResult_t> for Status {
    fn from(status: sys::cufftResult_t) -> Self {
        match status {
            sys::cufftResult_t::CUFFT_SUCCESS => Self::Success,
            sys::cufftResult_t::CUFFT_INVALID_PLAN => Self::InvalidPlan,
            sys::cufftResult_t::CUFFT_ALLOC_FAILED => Self::AllocFailed,
            sys::cufftResult_t::CUFFT_INVALID_TYPE => Self::InvalidType,
            sys::cufftResult_t::CUFFT_INVALID_VALUE => Self::InvalidValue,
            sys::cufftResult_t::CUFFT_INTERNAL_ERROR => Self::InternalError,
            sys::cufftResult_t::CUFFT_EXEC_FAILED => Self::ExecFailed,
            sys::cufftResult_t::CUFFT_SETUP_FAILED => Self::SetupFailed,
            sys::cufftResult_t::CUFFT_INVALID_SIZE => Self::InvalidSize,
            sys::cufftResult_t::CUFFT_UNALIGNED_DATA => Self::UnalignedData,
            sys::cufftResult_t::CUFFT_INVALID_DEVICE => Self::InvalidDevice,
            sys::cufftResult_t::CUFFT_NO_WORKSPACE => Self::NoWorkspace,
            sys::cufftResult_t::CUFFT_NOT_IMPLEMENTED => Self::NotImplemented,
            sys::cufftResult_t::CUFFT_NOT_SUPPORTED => Self::NotSupported,
            sys::cufftResult_t::CUFFT_MISSING_DEPENDENCY => Self::MissingDependency,
            sys::cufftResult_t::CUFFT_NVRTC_FAILURE => Self::NvrtcFailure,
            sys::cufftResult_t::CUFFT_NVJITLINK_FAILURE => Self::NvjitlinkFailure,
            sys::cufftResult_t::CUFFT_NVSHMEM_FAILURE => Self::NvshmemFailure,
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("CUFFT_SUCCESS"),
            Self::InvalidPlan => f.write_str("CUFFT_INVALID_PLAN"),
            Self::AllocFailed => f.write_str("CUFFT_ALLOC_FAILED"),
            Self::InvalidType => f.write_str("CUFFT_INVALID_TYPE"),
            Self::InvalidValue => f.write_str("CUFFT_INVALID_VALUE"),
            Self::InternalError => f.write_str("CUFFT_INTERNAL_ERROR"),
            Self::ExecFailed => f.write_str("CUFFT_EXEC_FAILED"),
            Self::SetupFailed => f.write_str("CUFFT_SETUP_FAILED"),
            Self::InvalidSize => f.write_str("CUFFT_INVALID_SIZE"),
            Self::UnalignedData => f.write_str("CUFFT_UNALIGNED_DATA"),
            Self::InvalidDevice => f.write_str("CUFFT_INVALID_DEVICE"),
            Self::NoWorkspace => f.write_str("CUFFT_NO_WORKSPACE"),
            Self::NotImplemented => f.write_str("CUFFT_NOT_IMPLEMENTED"),
            Self::NotSupported => f.write_str("CUFFT_NOT_SUPPORTED"),
            Self::MissingDependency => f.write_str("CUFFT_MISSING_DEPENDENCY"),
            Self::NvrtcFailure => f.write_str("CUFFT_NVRTC_FAILURE"),
            Self::NvjitlinkFailure => f.write_str("CUFFT_NVJITLINK_FAILURE"),
            Self::NvshmemFailure => f.write_str("CUFFT_NVSHMEM_FAILURE"),
            Self::Unknown(code) => write!(f, "UNKNOWN_CUFFT_STATUS({code})"),
        }
    }
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    #[error("cufft error ({code}): {message}")]
    Cufft { code: Status, message: String },

    #[error("string contains interior nul byte")]
    InteriorNul,

    #[error("unexpected null handle")]
    NullHandle,

    #[error("`{name}` is out of range")]
    OutOfRange { name: String },

    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    #[error("plan builder is missing a transform definition")]
    MissingPlanDefinition,
}

pub type Result<T> = std::result::Result<T, Error>;

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
