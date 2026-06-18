#![allow(deprecated)]

use std::ffi::{CStr, NulError};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::error::Error as CudaError;
use singe_cupti_sys as sys;
use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    #[error("cupti error ({code}): {message}")]
    Cupti { code: Status, message: String },

    #[error("string contains interior nul byte")]
    InteriorNul,

    #[error("unexpected null handle")]
    NullHandle,

    #[error("`{name}` is out of range")]
    OutOfRange { name: String },

    #[error("unexpected length for `{name}`: expected {expected}, got {actual}")]
    LengthMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("invalid attribute: `{name}`")]
    InvalidAttribute { name: String },

    #[error("`{name}` lock poisoned")]
    LockPoisoned { name: String },
}

pub type Result<T> = std::result::Result<T, Error>;

#[allow(deprecated)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Status {
    Success = sys::CUptiResult::CUPTI_SUCCESS as _,
    InvalidParameter = sys::CUptiResult::CUPTI_ERROR_INVALID_PARAMETER as _,
    InvalidDevice = sys::CUptiResult::CUPTI_ERROR_INVALID_DEVICE as _,
    InvalidContext = sys::CUptiResult::CUPTI_ERROR_INVALID_CONTEXT as _,
    InvalidEventDomainId = sys::CUptiResult::CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID as _,
    InvalidEventId = sys::CUptiResult::CUPTI_ERROR_INVALID_EVENT_ID as _,
    InvalidEventName = sys::CUptiResult::CUPTI_ERROR_INVALID_EVENT_NAME as _,
    InvalidOperation = sys::CUptiResult::CUPTI_ERROR_INVALID_OPERATION as _,
    OutOfMemory = sys::CUptiResult::CUPTI_ERROR_OUT_OF_MEMORY as _,
    Hardware = sys::CUptiResult::CUPTI_ERROR_HARDWARE as _,
    ParameterSizeNotSufficient = sys::CUptiResult::CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT as _,
    ApiNotImplemented = sys::CUptiResult::CUPTI_ERROR_API_NOT_IMPLEMENTED as _,
    MaxLimitReached = sys::CUptiResult::CUPTI_ERROR_MAX_LIMIT_REACHED as _,
    NotReady = sys::CUptiResult::CUPTI_ERROR_NOT_READY as _,
    NotCompatible = sys::CUptiResult::CUPTI_ERROR_NOT_COMPATIBLE as _,
    NotInitialized = sys::CUptiResult::CUPTI_ERROR_NOT_INITIALIZED as _,
    InvalidMetricId = sys::CUptiResult::CUPTI_ERROR_INVALID_METRIC_ID as _,
    InvalidMetricName = sys::CUptiResult::CUPTI_ERROR_INVALID_METRIC_NAME as _,
    QueueEmpty = sys::CUptiResult::CUPTI_ERROR_QUEUE_EMPTY as _,
    InvalidHandle = sys::CUptiResult::CUPTI_ERROR_INVALID_HANDLE as _,
    InvalidStream = sys::CUptiResult::CUPTI_ERROR_INVALID_STREAM as _,
    InvalidKind = sys::CUptiResult::CUPTI_ERROR_INVALID_KIND as _,
    InvalidEventValue = sys::CUptiResult::CUPTI_ERROR_INVALID_EVENT_VALUE as _,
    Disabled = sys::CUptiResult::CUPTI_ERROR_DISABLED as _,
    InvalidModule = sys::CUptiResult::CUPTI_ERROR_INVALID_MODULE as _,
    InvalidMetricValue = sys::CUptiResult::CUPTI_ERROR_INVALID_METRIC_VALUE as _,
    HardwareBusy = sys::CUptiResult::CUPTI_ERROR_HARDWARE_BUSY as _,
    NotSupported = sys::CUptiResult::CUPTI_ERROR_NOT_SUPPORTED as _,
    UnifiedMemoryProfilingNotSupported =
        sys::CUptiResult::CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED as _,
    UnifiedMemoryProfilingNotSupportedOnDevice =
        sys::CUptiResult::CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE as _,
    UnifiedMemoryProfilingNotSupportedOnNonPeerToPeerDevices =
        sys::CUptiResult::CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES as _,
    UnifiedMemoryProfilingNotSupportedWithMultiProcessService =
        sys::CUptiResult::CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS as _,
    CudaDynamicParallelismTracingNotSupported =
        sys::CUptiResult::CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED as _,
    VirtualizedDeviceNotSupported =
        sys::CUptiResult::CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED as _,
    CudaCompilerNotCompatible = sys::CUptiResult::CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE as _,
    InsufficientPrivileges = sys::CUptiResult::CUPTI_ERROR_INSUFFICIENT_PRIVILEGES as _,
    OldProfilerApiInitialized = sys::CUptiResult::CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED as _,
    OpenAccUndefinedRoutine = sys::CUptiResult::CUPTI_ERROR_OPENACC_UNDEFINED_ROUTINE as _,
    #[allow(deprecated)]
    LegacyProfilerNotSupported = sys::CUptiResult::CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED as _,
    MultipleSubscribersNotSupported =
        sys::CUptiResult::CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED as _,
    VirtualizedDeviceInsufficientPrivileges =
        sys::CUptiResult::CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES as _,
    ConfidentialComputingNotSupported =
        sys::CUptiResult::CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED as _,
    CmpDeviceNotSupported = sys::CUptiResult::CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED as _,
    MigDeviceNotSupported = sys::CUptiResult::CUPTI_ERROR_MIG_DEVICE_NOT_SUPPORTED as _,
    SliDeviceNotSupported = sys::CUptiResult::CUPTI_ERROR_SLI_DEVICE_NOT_SUPPORTED as _,
    WslDeviceNotSupported = sys::CUptiResult::CUPTI_ERROR_WSL_DEVICE_NOT_SUPPORTED as _,
    InvalidChipName = sys::CUptiResult::CUPTI_ERROR_INVALID_CHIP_NAME as _,
    HesTraceNotSupportedOnMultiProcessService =
        sys::CUptiResult::CUPTI_ERROR_HES_TRACE_NOT_SUPPORTED_ON_MPS as _,
    Unknown = sys::CUptiResult::CUPTI_ERROR_UNKNOWN as _,
    ForceInt = sys::CUptiResult::CUPTI_ERROR_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUptiResult, Status);

impl Status {
    pub fn description(self) -> String {
        self.to_string()
            .trim_start_matches("CUPTI_ERROR_")
            .trim_start_matches("CUPTI_")
            .replace('_', " ")
            .to_ascii_lowercase()
    }
}

impl_enum_display!(Status, {
    Self::Success => "CUPTI_SUCCESS",
    Self::InvalidParameter => "CUPTI_ERROR_INVALID_PARAMETER",
    Self::InvalidDevice => "CUPTI_ERROR_INVALID_DEVICE",
    Self::InvalidContext => "CUPTI_ERROR_INVALID_CONTEXT",
    Self::InvalidEventDomainId => "CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID",
    Self::InvalidEventId => "CUPTI_ERROR_INVALID_EVENT_ID",
    Self::InvalidEventName => "CUPTI_ERROR_INVALID_EVENT_NAME",
    Self::InvalidOperation => "CUPTI_ERROR_INVALID_OPERATION",
    Self::OutOfMemory => "CUPTI_ERROR_OUT_OF_MEMORY",
    Self::Hardware => "CUPTI_ERROR_HARDWARE",
    Self::ParameterSizeNotSufficient => "CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT",
    Self::ApiNotImplemented => "CUPTI_ERROR_API_NOT_IMPLEMENTED",
    Self::MaxLimitReached => "CUPTI_ERROR_MAX_LIMIT_REACHED",
    Self::NotReady => "CUPTI_ERROR_NOT_READY",
    Self::NotCompatible => "CUPTI_ERROR_NOT_COMPATIBLE",
    Self::NotInitialized => "CUPTI_ERROR_NOT_INITIALIZED",
    Self::InvalidMetricId => "CUPTI_ERROR_INVALID_METRIC_ID",
    Self::InvalidMetricName => "CUPTI_ERROR_INVALID_METRIC_NAME",
    Self::QueueEmpty => "CUPTI_ERROR_QUEUE_EMPTY",
    Self::InvalidHandle => "CUPTI_ERROR_INVALID_HANDLE",
    Self::InvalidStream => "CUPTI_ERROR_INVALID_STREAM",
    Self::InvalidKind => "CUPTI_ERROR_INVALID_KIND",
    Self::InvalidEventValue => "CUPTI_ERROR_INVALID_EVENT_VALUE",
    Self::Disabled => "CUPTI_ERROR_DISABLED",
    Self::InvalidModule => "CUPTI_ERROR_INVALID_MODULE",
    Self::InvalidMetricValue => "CUPTI_ERROR_INVALID_METRIC_VALUE",
    Self::HardwareBusy => "CUPTI_ERROR_HARDWARE_BUSY",
    Self::NotSupported => "CUPTI_ERROR_NOT_SUPPORTED",
    Self::UnifiedMemoryProfilingNotSupported => "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED",
    Self::UnifiedMemoryProfilingNotSupportedOnDevice => "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE",
    Self::UnifiedMemoryProfilingNotSupportedOnNonPeerToPeerDevices => "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES",
    Self::UnifiedMemoryProfilingNotSupportedWithMultiProcessService => "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS",
    Self::CudaDynamicParallelismTracingNotSupported => "CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED",
    Self::VirtualizedDeviceNotSupported => "CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED",
    Self::CudaCompilerNotCompatible => "CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE",
    Self::InsufficientPrivileges => "CUPTI_ERROR_INSUFFICIENT_PRIVILEGES",
    Self::OldProfilerApiInitialized => "CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED",
    Self::OpenAccUndefinedRoutine => "CUPTI_ERROR_OPENACC_UNDEFINED_ROUTINE",
    Self::LegacyProfilerNotSupported => "CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED",
    Self::MultipleSubscribersNotSupported => "CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED",
    Self::VirtualizedDeviceInsufficientPrivileges => "CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES",
    Self::ConfidentialComputingNotSupported => "CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED",
    Self::CmpDeviceNotSupported => "CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED",
    Self::MigDeviceNotSupported => "CUPTI_ERROR_MIG_DEVICE_NOT_SUPPORTED",
    Self::SliDeviceNotSupported => "CUPTI_ERROR_SLI_DEVICE_NOT_SUPPORTED",
    Self::WslDeviceNotSupported => "CUPTI_ERROR_WSL_DEVICE_NOT_SUPPORTED",
    Self::InvalidChipName => "CUPTI_ERROR_INVALID_CHIP_NAME",
    Self::HesTraceNotSupportedOnMultiProcessService => "CUPTI_ERROR_HES_TRACE_NOT_SUPPORTED_ON_MPS",
    Self::Unknown => "CUPTI_ERROR_UNKNOWN",
    Self::ForceInt => "CUPTI_ERROR_FORCE_INT",
});

impl From<sys::CUptiResult> for Error {
    fn from(code: sys::CUptiResult) -> Self {
        let status = Status::try_from(code as u32).unwrap_or(Status::Unknown);
        let message = unsafe {
            let mut message_ptr = std::ptr::null();
            let error_result = sys::cuptiGetErrorMessage(code, &mut message_ptr);
            if error_result == sys::CUptiResult::CUPTI_SUCCESS && !message_ptr.is_null() {
                CStr::from_ptr(message_ptr).to_string_lossy().into_owned()
            } else {
                format!("cupti error code {:?}", code)
            }
        };

        Self::Cupti {
            code: status,
            message,
        }
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        sys::CUptiResult::from(status).into()
    }
}

impl Status {
    /// Returns CUPTI's symbolic string for this status.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the status is not a valid CUPTI result code.
    pub fn get_result_string(self) -> Result<String> {
        let mut message = std::ptr::null();
        unsafe {
            crate::try_ffi!(sys::cuptiGetResultString(self.into(), &mut message))?;
            if message.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(CStr::from_ptr(message).to_string_lossy().into_owned())
        }
    }

    /// Returns CUPTI's descriptive error message for this status.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the status is not a valid CUPTI result code.
    pub fn get_error_message(self) -> Result<String> {
        let mut message = std::ptr::null();
        unsafe {
            crate::try_ffi!(sys::cuptiGetErrorMessage(self.into(), &mut message))?;
            if message.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(CStr::from_ptr(message).to_string_lossy().into_owned())
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
        if status != singe_cupti_sys::CUptiResult::CUPTI_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}
