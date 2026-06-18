#![allow(deprecated)]

use std::{
    ffi::{CStr, NulError},
    fmt,
};

use singe_nvml_sys as sys;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    Success,
    Uninitialized,
    InvalidArgument,
    NotSupported,
    NoPermission,
    #[deprecated]
    AlreadyInitialized,
    NotFound,
    InsufficientSize,
    InsufficientPower,
    DriverNotLoaded,
    Timeout,
    IrqIssue,
    LibraryNotFound,
    FunctionNotFound,
    CorruptedInforom,
    GpuIsLost,
    ResetRequired,
    OperatingSystem,
    LibRmVersionMismatch,
    InUse,
    Memory,
    NoData,
    VgpuEccNotSupported,
    InsufficientResources,
    FreqNotSupported,
    ArgumentVersionMismatch,
    Deprecated,
    NotReady,
    GpuNotFound,
    InvalidState,
    ResetTypeNotSupported,
    UnknownError,
    Unknown(u32),
}

impl Status {
    pub const fn raw(self) -> u32 {
        match self {
            Self::Success => sys::nvmlReturn_t::NVML_SUCCESS as _,
            Self::Uninitialized => sys::nvmlReturn_t::NVML_ERROR_UNINITIALIZED as _,
            Self::InvalidArgument => sys::nvmlReturn_t::NVML_ERROR_INVALID_ARGUMENT as _,
            Self::NotSupported => sys::nvmlReturn_t::NVML_ERROR_NOT_SUPPORTED as _,
            Self::NoPermission => sys::nvmlReturn_t::NVML_ERROR_NO_PERMISSION as _,
            Self::AlreadyInitialized => sys::nvmlReturn_t::NVML_ERROR_ALREADY_INITIALIZED as _,
            Self::NotFound => sys::nvmlReturn_t::NVML_ERROR_NOT_FOUND as _,
            Self::InsufficientSize => sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE as _,
            Self::InsufficientPower => sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_POWER as _,
            Self::DriverNotLoaded => sys::nvmlReturn_t::NVML_ERROR_DRIVER_NOT_LOADED as _,
            Self::Timeout => sys::nvmlReturn_t::NVML_ERROR_TIMEOUT as _,
            Self::IrqIssue => sys::nvmlReturn_t::NVML_ERROR_IRQ_ISSUE as _,
            Self::LibraryNotFound => sys::nvmlReturn_t::NVML_ERROR_LIBRARY_NOT_FOUND as _,
            Self::FunctionNotFound => sys::nvmlReturn_t::NVML_ERROR_FUNCTION_NOT_FOUND as _,
            Self::CorruptedInforom => sys::nvmlReturn_t::NVML_ERROR_CORRUPTED_INFOROM as _,
            Self::GpuIsLost => sys::nvmlReturn_t::NVML_ERROR_GPU_IS_LOST as _,
            Self::ResetRequired => sys::nvmlReturn_t::NVML_ERROR_RESET_REQUIRED as _,
            Self::OperatingSystem => sys::nvmlReturn_t::NVML_ERROR_OPERATING_SYSTEM as _,
            Self::LibRmVersionMismatch => {
                sys::nvmlReturn_t::NVML_ERROR_LIB_RM_VERSION_MISMATCH as _
            }
            Self::InUse => sys::nvmlReturn_t::NVML_ERROR_IN_USE as _,
            Self::Memory => sys::nvmlReturn_t::NVML_ERROR_MEMORY as _,
            Self::NoData => sys::nvmlReturn_t::NVML_ERROR_NO_DATA as _,
            Self::VgpuEccNotSupported => sys::nvmlReturn_t::NVML_ERROR_VGPU_ECC_NOT_SUPPORTED as _,
            Self::InsufficientResources => {
                sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_RESOURCES as _
            }
            Self::FreqNotSupported => sys::nvmlReturn_t::NVML_ERROR_FREQ_NOT_SUPPORTED as _,
            Self::ArgumentVersionMismatch => {
                sys::nvmlReturn_t::NVML_ERROR_ARGUMENT_VERSION_MISMATCH as _
            }
            Self::Deprecated => sys::nvmlReturn_t::NVML_ERROR_DEPRECATED as _,
            Self::NotReady => sys::nvmlReturn_t::NVML_ERROR_NOT_READY as _,
            Self::GpuNotFound => sys::nvmlReturn_t::NVML_ERROR_GPU_NOT_FOUND as _,
            Self::InvalidState => sys::nvmlReturn_t::NVML_ERROR_INVALID_STATE as _,
            Self::ResetTypeNotSupported => {
                sys::nvmlReturn_t::NVML_ERROR_RESET_TYPE_NOT_SUPPORTED as _
            }
            Self::UnknownError => sys::nvmlReturn_t::NVML_ERROR_UNKNOWN as _,
            Self::Unknown(code) => code,
        }
    }

    pub fn description(self) -> String {
        let Ok(raw) = sys::nvmlReturn_t::try_from(self) else {
            return String::from("unknown nvml error");
        };

        unsafe {
            let ptr = sys::nvmlErrorString(raw);
            if ptr.is_null() {
                String::from("unknown nvml error")
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }
}

impl From<sys::nvmlReturn_t> for Status {
    fn from(status: sys::nvmlReturn_t) -> Self {
        Self::from(status as u32)
    }
}

impl From<u32> for Status {
    fn from(code: u32) -> Self {
        match code {
            x if x == sys::nvmlReturn_t::NVML_SUCCESS as u32 => Self::Success,
            x if x == sys::nvmlReturn_t::NVML_ERROR_UNINITIALIZED as u32 => Self::Uninitialized,
            x if x == sys::nvmlReturn_t::NVML_ERROR_INVALID_ARGUMENT as u32 => {
                Self::InvalidArgument
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_NOT_SUPPORTED as u32 => Self::NotSupported,
            x if x == sys::nvmlReturn_t::NVML_ERROR_NO_PERMISSION as u32 => Self::NoPermission,
            x if x == sys::nvmlReturn_t::NVML_ERROR_ALREADY_INITIALIZED as u32 => {
                Self::AlreadyInitialized
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_NOT_FOUND as u32 => Self::NotFound,
            x if x == sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE as u32 => {
                Self::InsufficientSize
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_POWER as u32 => {
                Self::InsufficientPower
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_DRIVER_NOT_LOADED as u32 => {
                Self::DriverNotLoaded
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_TIMEOUT as u32 => Self::Timeout,
            x if x == sys::nvmlReturn_t::NVML_ERROR_IRQ_ISSUE as u32 => Self::IrqIssue,
            x if x == sys::nvmlReturn_t::NVML_ERROR_LIBRARY_NOT_FOUND as u32 => {
                Self::LibraryNotFound
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_FUNCTION_NOT_FOUND as u32 => {
                Self::FunctionNotFound
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_CORRUPTED_INFOROM as u32 => {
                Self::CorruptedInforom
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_GPU_IS_LOST as u32 => Self::GpuIsLost,
            x if x == sys::nvmlReturn_t::NVML_ERROR_RESET_REQUIRED as u32 => Self::ResetRequired,
            x if x == sys::nvmlReturn_t::NVML_ERROR_OPERATING_SYSTEM as u32 => {
                Self::OperatingSystem
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_LIB_RM_VERSION_MISMATCH as u32 => {
                Self::LibRmVersionMismatch
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_IN_USE as u32 => Self::InUse,
            x if x == sys::nvmlReturn_t::NVML_ERROR_MEMORY as u32 => Self::Memory,
            x if x == sys::nvmlReturn_t::NVML_ERROR_NO_DATA as u32 => Self::NoData,
            x if x == sys::nvmlReturn_t::NVML_ERROR_VGPU_ECC_NOT_SUPPORTED as u32 => {
                Self::VgpuEccNotSupported
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_RESOURCES as u32 => {
                Self::InsufficientResources
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_FREQ_NOT_SUPPORTED as u32 => {
                Self::FreqNotSupported
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_ARGUMENT_VERSION_MISMATCH as u32 => {
                Self::ArgumentVersionMismatch
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_DEPRECATED as u32 => Self::Deprecated,
            x if x == sys::nvmlReturn_t::NVML_ERROR_NOT_READY as u32 => Self::NotReady,
            x if x == sys::nvmlReturn_t::NVML_ERROR_GPU_NOT_FOUND as u32 => Self::GpuNotFound,
            x if x == sys::nvmlReturn_t::NVML_ERROR_INVALID_STATE as u32 => Self::InvalidState,
            x if x == sys::nvmlReturn_t::NVML_ERROR_RESET_TYPE_NOT_SUPPORTED as u32 => {
                Self::ResetTypeNotSupported
            }
            x if x == sys::nvmlReturn_t::NVML_ERROR_UNKNOWN as u32 => Self::UnknownError,
            code => Self::Unknown(code),
        }
    }
}

impl TryFrom<Status> for sys::nvmlReturn_t {
    type Error = Status;

    fn try_from(status: Status) -> std::result::Result<Self, Self::Error> {
        match status {
            Status::Success => Ok(Self::NVML_SUCCESS),
            Status::Uninitialized => Ok(Self::NVML_ERROR_UNINITIALIZED),
            Status::InvalidArgument => Ok(Self::NVML_ERROR_INVALID_ARGUMENT),
            Status::NotSupported => Ok(Self::NVML_ERROR_NOT_SUPPORTED),
            Status::NoPermission => Ok(Self::NVML_ERROR_NO_PERMISSION),
            Status::AlreadyInitialized => Ok(Self::NVML_ERROR_ALREADY_INITIALIZED),
            Status::NotFound => Ok(Self::NVML_ERROR_NOT_FOUND),
            Status::InsufficientSize => Ok(Self::NVML_ERROR_INSUFFICIENT_SIZE),
            Status::InsufficientPower => Ok(Self::NVML_ERROR_INSUFFICIENT_POWER),
            Status::DriverNotLoaded => Ok(Self::NVML_ERROR_DRIVER_NOT_LOADED),
            Status::Timeout => Ok(Self::NVML_ERROR_TIMEOUT),
            Status::IrqIssue => Ok(Self::NVML_ERROR_IRQ_ISSUE),
            Status::LibraryNotFound => Ok(Self::NVML_ERROR_LIBRARY_NOT_FOUND),
            Status::FunctionNotFound => Ok(Self::NVML_ERROR_FUNCTION_NOT_FOUND),
            Status::CorruptedInforom => Ok(Self::NVML_ERROR_CORRUPTED_INFOROM),
            Status::GpuIsLost => Ok(Self::NVML_ERROR_GPU_IS_LOST),
            Status::ResetRequired => Ok(Self::NVML_ERROR_RESET_REQUIRED),
            Status::OperatingSystem => Ok(Self::NVML_ERROR_OPERATING_SYSTEM),
            Status::LibRmVersionMismatch => Ok(Self::NVML_ERROR_LIB_RM_VERSION_MISMATCH),
            Status::InUse => Ok(Self::NVML_ERROR_IN_USE),
            Status::Memory => Ok(Self::NVML_ERROR_MEMORY),
            Status::NoData => Ok(Self::NVML_ERROR_NO_DATA),
            Status::VgpuEccNotSupported => Ok(Self::NVML_ERROR_VGPU_ECC_NOT_SUPPORTED),
            Status::InsufficientResources => Ok(Self::NVML_ERROR_INSUFFICIENT_RESOURCES),
            Status::FreqNotSupported => Ok(Self::NVML_ERROR_FREQ_NOT_SUPPORTED),
            Status::ArgumentVersionMismatch => Ok(Self::NVML_ERROR_ARGUMENT_VERSION_MISMATCH),
            Status::Deprecated => Ok(Self::NVML_ERROR_DEPRECATED),
            Status::NotReady => Ok(Self::NVML_ERROR_NOT_READY),
            Status::GpuNotFound => Ok(Self::NVML_ERROR_GPU_NOT_FOUND),
            Status::InvalidState => Ok(Self::NVML_ERROR_INVALID_STATE),
            Status::ResetTypeNotSupported => Ok(Self::NVML_ERROR_RESET_TYPE_NOT_SUPPORTED),
            Status::UnknownError => Ok(Self::NVML_ERROR_UNKNOWN),
            Status::Unknown(_) => Err(status),
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("NVML_SUCCESS"),
            Self::Uninitialized => f.write_str("NVML_ERROR_UNINITIALIZED"),
            Self::InvalidArgument => f.write_str("NVML_ERROR_INVALID_ARGUMENT"),
            Self::NotSupported => f.write_str("NVML_ERROR_NOT_SUPPORTED"),
            Self::NoPermission => f.write_str("NVML_ERROR_NO_PERMISSION"),
            Self::AlreadyInitialized => f.write_str("NVML_ERROR_ALREADY_INITIALIZED"),
            Self::NotFound => f.write_str("NVML_ERROR_NOT_FOUND"),
            Self::InsufficientSize => f.write_str("NVML_ERROR_INSUFFICIENT_SIZE"),
            Self::InsufficientPower => f.write_str("NVML_ERROR_INSUFFICIENT_POWER"),
            Self::DriverNotLoaded => f.write_str("NVML_ERROR_DRIVER_NOT_LOADED"),
            Self::Timeout => f.write_str("NVML_ERROR_TIMEOUT"),
            Self::IrqIssue => f.write_str("NVML_ERROR_IRQ_ISSUE"),
            Self::LibraryNotFound => f.write_str("NVML_ERROR_LIBRARY_NOT_FOUND"),
            Self::FunctionNotFound => f.write_str("NVML_ERROR_FUNCTION_NOT_FOUND"),
            Self::CorruptedInforom => f.write_str("NVML_ERROR_CORRUPTED_INFOROM"),
            Self::GpuIsLost => f.write_str("NVML_ERROR_GPU_IS_LOST"),
            Self::ResetRequired => f.write_str("NVML_ERROR_RESET_REQUIRED"),
            Self::OperatingSystem => f.write_str("NVML_ERROR_OPERATING_SYSTEM"),
            Self::LibRmVersionMismatch => f.write_str("NVML_ERROR_LIB_RM_VERSION_MISMATCH"),
            Self::InUse => f.write_str("NVML_ERROR_IN_USE"),
            Self::Memory => f.write_str("NVML_ERROR_MEMORY"),
            Self::NoData => f.write_str("NVML_ERROR_NO_DATA"),
            Self::VgpuEccNotSupported => f.write_str("NVML_ERROR_VGPU_ECC_NOT_SUPPORTED"),
            Self::InsufficientResources => f.write_str("NVML_ERROR_INSUFFICIENT_RESOURCES"),
            Self::FreqNotSupported => f.write_str("NVML_ERROR_FREQ_NOT_SUPPORTED"),
            Self::ArgumentVersionMismatch => f.write_str("NVML_ERROR_ARGUMENT_VERSION_MISMATCH"),
            Self::Deprecated => f.write_str("NVML_ERROR_DEPRECATED"),
            Self::NotReady => f.write_str("NVML_ERROR_NOT_READY"),
            Self::GpuNotFound => f.write_str("NVML_ERROR_GPU_NOT_FOUND"),
            Self::InvalidState => f.write_str("NVML_ERROR_INVALID_STATE"),
            Self::ResetTypeNotSupported => f.write_str("NVML_ERROR_RESET_TYPE_NOT_SUPPORTED"),
            Self::UnknownError => f.write_str("NVML_ERROR_UNKNOWN"),
            Self::Unknown(code) => write!(f, "unknown nvml status ({code})"),
        }
    }
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    #[error("nvml error ({code}): {message}")]
    Nvml { code: Status, message: String },

    #[error("string contains interior nul byte")]
    InteriorNul,

    #[error("unexpected null handle")]
    NullHandle,

    #[error("unknown field value type ({0})")]
    UnknownFieldValueType(u32),

    #[error("unexpected `{name}` field value type ({value})")]
    UnexpectedFieldValueType { name: String, value: String },

    #[error("unknown `{name}` value ({value})")]
    UnknownEnumValue { name: String, value: u32 },

    #[error("negative `{name}` value ({value})")]
    NegativeValue { name: String, value: i64 },

    #[error("`{name}` returned no values")]
    EmptyOutput { name: String },
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

impl From<sys::nvmlReturn_t> for Error {
    fn from(code: sys::nvmlReturn_t) -> Self {
        Self::from(Status::from(code))
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        Self::Nvml {
            code: status,
            message: status.description(),
        }
    }
}

#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let err = { $expr };
        if err != singe_nvml_sys::nvmlReturn_t::NVML_SUCCESS {
            Err($crate::error::Error::from(err))
        } else {
            Ok(())
        }
    }};
}
