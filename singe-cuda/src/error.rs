#![allow(deprecated)]

use std::{
    ffi::{CStr, NulError},
    fmt::{self, Display, Formatter},
    io,
};

use num_enum::{FromPrimitive, IntoPrimitive};
use thiserror::Error;

use singe_cuda_sys::{nvrtc, nvvm, runtime};

use crate::nvrtc::Status as NvrtcStatus;
use crate::nvvm::Status as NvvmStatus;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Status {
    Success = runtime::cudaError_t::CUDA_SUCCESS as _,
    InvalidValue = runtime::cudaError_t::CUDA_ERROR_INVALID_VALUE as _,
    OutOfMemory = runtime::cudaError_t::CUDA_ERROR_OUT_OF_MEMORY as _,
    NotInitialized = runtime::cudaError_t::CUDA_ERROR_NOT_INITIALIZED as _,
    Deinitialized = runtime::cudaError_t::CUDA_ERROR_DEINITIALIZED as _,
    ProfilerDisabled = runtime::cudaError_t::CUDA_ERROR_PROFILER_DISABLED as _,
    #[deprecated]
    ProfilerNotInitialized = runtime::cudaError_t::CUDA_ERROR_PROFILER_NOT_INITIALIZED as _,
    #[deprecated]
    ProfilerAlreadyStarted = runtime::cudaError_t::CUDA_ERROR_PROFILER_ALREADY_STARTED as _,
    #[deprecated]
    ProfilerAlreadyStopped = runtime::cudaError_t::CUDA_ERROR_PROFILER_ALREADY_STOPPED as _,
    StubLibrary = runtime::cudaError_t::CUDA_ERROR_STUB_LIBRARY as _,
    CallRequiresNewerDriver = runtime::cudaError_t::CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER as _,
    DeviceUnavailable = runtime::cudaError_t::CUDA_ERROR_DEVICE_UNAVAILABLE as _,
    NoDevice = runtime::cudaError_t::CUDA_ERROR_NO_DEVICE as _,
    InvalidDevice = runtime::cudaError_t::CUDA_ERROR_INVALID_DEVICE as _,
    DeviceNotLicensed = runtime::cudaError_t::CUDA_ERROR_DEVICE_NOT_LICENSED as _,
    InvalidImage = runtime::cudaError_t::CUDA_ERROR_INVALID_IMAGE as _,
    InvalidContext = runtime::cudaError_t::CUDA_ERROR_INVALID_CONTEXT as _,
    #[deprecated]
    ContextAlreadyCurrent = runtime::cudaError_t::CUDA_ERROR_CONTEXT_ALREADY_CURRENT as _,
    MapFailed = runtime::cudaError_t::CUDA_ERROR_MAP_FAILED as _,
    UnmapFailed = runtime::cudaError_t::CUDA_ERROR_UNMAP_FAILED as _,
    ArrayIsMapped = runtime::cudaError_t::CUDA_ERROR_ARRAY_IS_MAPPED as _,
    AlreadyMapped = runtime::cudaError_t::CUDA_ERROR_ALREADY_MAPPED as _,
    NoBinaryForGpu = runtime::cudaError_t::CUDA_ERROR_NO_BINARY_FOR_GPU as _,
    AlreadyAcquired = runtime::cudaError_t::CUDA_ERROR_ALREADY_ACQUIRED as _,
    NotMapped = runtime::cudaError_t::CUDA_ERROR_NOT_MAPPED as _,
    NotMappedAsArray = runtime::cudaError_t::CUDA_ERROR_NOT_MAPPED_AS_ARRAY as _,
    NotMappedAsPointer = runtime::cudaError_t::CUDA_ERROR_NOT_MAPPED_AS_POINTER as _,
    EccUncorrectable = runtime::cudaError_t::CUDA_ERROR_ECC_UNCORRECTABLE as _,
    UnsupportedLimit = runtime::cudaError_t::CUDA_ERROR_UNSUPPORTED_LIMIT as _,
    ContextAlreadyInUse = runtime::cudaError_t::CUDA_ERROR_CONTEXT_ALREADY_IN_USE as _,
    PeerAccessUnsupported = runtime::cudaError_t::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED as _,
    InvalidPtx = runtime::cudaError_t::CUDA_ERROR_INVALID_PTX as _,
    InvalidGraphicsContext = runtime::cudaError_t::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT as _,
    NvlinkUncorrectable = runtime::cudaError_t::CUDA_ERROR_NVLINK_UNCORRECTABLE as _,
    JitCompilerNotFound = runtime::cudaError_t::CUDA_ERROR_JIT_COMPILER_NOT_FOUND as _,
    UnsupportedPtxVersion = runtime::cudaError_t::CUDA_ERROR_UNSUPPORTED_PTX_VERSION as _,
    JitCompilationDisabled = runtime::cudaError_t::CUDA_ERROR_JIT_COMPILATION_DISABLED as _,
    UnsupportedExecAffinity = runtime::cudaError_t::CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY as _,
    UnsupportedDevsideSync = runtime::cudaError_t::CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC as _,
    Contained = runtime::cudaError_t::CUDA_ERROR_CONTAINED as _,
    InvalidSource = runtime::cudaError_t::CUDA_ERROR_INVALID_SOURCE as _,
    FileNotFound = runtime::cudaError_t::CUDA_ERROR_FILE_NOT_FOUND as _,
    SharedObjectSymbolNotFound =
        runtime::cudaError_t::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND as _,
    SharedObjectInitFailed = runtime::cudaError_t::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED as _,
    OperatingSystem = runtime::cudaError_t::CUDA_ERROR_OPERATING_SYSTEM as _,
    InvalidHandle = runtime::cudaError_t::CUDA_ERROR_INVALID_HANDLE as _,
    IllegalState = runtime::cudaError_t::CUDA_ERROR_ILLEGAL_STATE as _,
    LossyQuery = runtime::cudaError_t::CUDA_ERROR_LOSSY_QUERY as _,
    NotFound = runtime::cudaError_t::CUDA_ERROR_NOT_FOUND as _,
    NotReady = runtime::cudaError_t::CUDA_ERROR_NOT_READY as _,
    IllegalAddress = runtime::cudaError_t::CUDA_ERROR_ILLEGAL_ADDRESS as _,
    LaunchOutOfResources = runtime::cudaError_t::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES as _,
    LaunchTimeout = runtime::cudaError_t::CUDA_ERROR_LAUNCH_TIMEOUT as _,
    LaunchIncompatibleTexturing =
        runtime::cudaError_t::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING as _,
    PeerAccessAlreadyEnabled = runtime::cudaError_t::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED as _,
    PeerAccessNotEnabled = runtime::cudaError_t::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED as _,
    PrimaryContextActive = runtime::cudaError_t::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE as _,
    ContextIsDestroyed = runtime::cudaError_t::CUDA_ERROR_CONTEXT_IS_DESTROYED as _,
    Assert = runtime::cudaError_t::CUDA_ERROR_ASSERT as _,
    TooManyPeers = runtime::cudaError_t::CUDA_ERROR_TOO_MANY_PEERS as _,
    HostMemoryAlreadyRegistered =
        runtime::cudaError_t::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED as _,
    HostMemoryNotRegistered = runtime::cudaError_t::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED as _,
    HardwareStackError = runtime::cudaError_t::CUDA_ERROR_HARDWARE_STACK_ERROR as _,
    IllegalInstruction = runtime::cudaError_t::CUDA_ERROR_ILLEGAL_INSTRUCTION as _,
    MisalignedAddress = runtime::cudaError_t::CUDA_ERROR_MISALIGNED_ADDRESS as _,
    InvalidAddressSpace = runtime::cudaError_t::CUDA_ERROR_INVALID_ADDRESS_SPACE as _,
    InvalidPc = runtime::cudaError_t::CUDA_ERROR_INVALID_PC as _,
    LaunchFailed = runtime::cudaError_t::CUDA_ERROR_LAUNCH_FAILED as _,
    CooperativeLaunchTooLarge = runtime::cudaError_t::CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE as _,
    TensorMemoryLeak = runtime::cudaError_t::CUDA_ERROR_TENSOR_MEMORY_LEAK as _,
    NotPermitted = runtime::cudaError_t::CUDA_ERROR_NOT_PERMITTED as _,
    NotSupported = runtime::cudaError_t::CUDA_ERROR_NOT_SUPPORTED as _,
    SystemNotReady = runtime::cudaError_t::CUDA_ERROR_SYSTEM_NOT_READY as _,
    SystemDriverMismatch = runtime::cudaError_t::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH as _,
    CompatNotSupportedOnDevice =
        runtime::cudaError_t::CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE as _,
    MpsConnectionFailed = runtime::cudaError_t::CUDA_ERROR_MPS_CONNECTION_FAILED as _,
    MpsRpcFailure = runtime::cudaError_t::CUDA_ERROR_MPS_RPC_FAILURE as _,
    MpsServerNotReady = runtime::cudaError_t::CUDA_ERROR_MPS_SERVER_NOT_READY as _,
    MpsMaxClientsReached = runtime::cudaError_t::CUDA_ERROR_MPS_MAX_CLIENTS_REACHED as _,
    MpsMaxConnectionsReached = runtime::cudaError_t::CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED as _,
    MpsClientTerminated = runtime::cudaError_t::CUDA_ERROR_MPS_CLIENT_TERMINATED as _,
    CdpNotSupported = runtime::cudaError_t::CUDA_ERROR_CDP_NOT_SUPPORTED as _,
    CdpVersionMismatch = runtime::cudaError_t::CUDA_ERROR_CDP_VERSION_MISMATCH as _,
    StreamCaptureUnsupported = runtime::cudaError_t::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED as _,
    StreamCaptureInvalidated = runtime::cudaError_t::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED as _,
    StreamCaptureMerge = runtime::cudaError_t::CUDA_ERROR_STREAM_CAPTURE_MERGE as _,
    StreamCaptureUnmatched = runtime::cudaError_t::CUDA_ERROR_STREAM_CAPTURE_UNMATCHED as _,
    StreamCaptureUnjoined = runtime::cudaError_t::CUDA_ERROR_STREAM_CAPTURE_UNJOINED as _,
    StreamCaptureIsolation = runtime::cudaError_t::CUDA_ERROR_STREAM_CAPTURE_ISOLATION as _,
    StreamCaptureImplicit = runtime::cudaError_t::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT as _,
    CapturedEvent = runtime::cudaError_t::CUDA_ERROR_CAPTURED_EVENT as _,
    StreamCaptureWrongThread = runtime::cudaError_t::CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD as _,
    Timeout = runtime::cudaError_t::CUDA_ERROR_TIMEOUT as _,
    GraphExecUpdateFailure = runtime::cudaError_t::CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE as _,
    ExternalDevice = runtime::cudaError_t::CUDA_ERROR_EXTERNAL_DEVICE as _,
    InvalidClusterSize = runtime::cudaError_t::CUDA_ERROR_INVALID_CLUSTER_SIZE as _,
    FunctionNotLoaded = runtime::cudaError_t::CUDA_ERROR_FUNCTION_NOT_LOADED as _,
    InvalidResourceType = runtime::cudaError_t::CUDA_ERROR_INVALID_RESOURCE_TYPE as _,
    InvalidResourceConfiguration =
        runtime::cudaError_t::CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION as _,
    KeyRotation = runtime::cudaError_t::CUDA_ERROR_KEY_ROTATION as _,
    StreamDetached = runtime::cudaError_t::CUDA_ERROR_STREAM_DETACHED as _,
    UnknownError = runtime::cudaError_t::CUDA_ERROR_UNKNOWN as _,
    #[num_enum(catch_all)]
    Unknown(u32),
}

impl From<runtime::cudaError_t> for Status {
    fn from(value: runtime::cudaError_t) -> Self {
        Self::from(value as u32)
    }
}

impl From<Status> for runtime::cudaError_t {
    fn from(value: Status) -> Self {
        if let Status::Unknown(_) = value {
            return runtime::cudaError_t::CUDA_ERROR_UNKNOWN;
        }
        let value: u32 = value.into();
        unsafe { core::mem::transmute::<u32, Self>(value) }
    }
}

impl Status {
    pub fn description(self) -> String {
        if let Self::Unknown(_) = self {
            return String::from("unknown cuda error");
        }
        unsafe {
            let ptr = runtime::cudaGetErrorString(self.into());
            if ptr.is_null() {
                String::from("unknown cuda error")
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }
}

impl Display for Status {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("CUDA_SUCCESS"),
            Self::InvalidValue => f.write_str("CUDA_ERROR_INVALID_VALUE"),
            Self::OutOfMemory => f.write_str("CUDA_ERROR_OUT_OF_MEMORY"),
            Self::NotInitialized => f.write_str("CUDA_ERROR_NOT_INITIALIZED"),
            Self::Deinitialized => f.write_str("CUDA_ERROR_DEINITIALIZED"),
            Self::ProfilerDisabled => f.write_str("CUDA_ERROR_PROFILER_DISABLED"),
            Self::ProfilerNotInitialized => f.write_str("CUDA_ERROR_PROFILER_NOT_INITIALIZED"),
            Self::ProfilerAlreadyStarted => f.write_str("CUDA_ERROR_PROFILER_ALREADY_STARTED"),
            Self::ProfilerAlreadyStopped => f.write_str("CUDA_ERROR_PROFILER_ALREADY_STOPPED"),
            Self::StubLibrary => f.write_str("CUDA_ERROR_STUB_LIBRARY"),
            Self::CallRequiresNewerDriver => f.write_str("CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER"),
            Self::DeviceUnavailable => f.write_str("CUDA_ERROR_DEVICE_UNAVAILABLE"),
            Self::NoDevice => f.write_str("CUDA_ERROR_NO_DEVICE"),
            Self::InvalidDevice => f.write_str("CUDA_ERROR_INVALID_DEVICE"),
            Self::DeviceNotLicensed => f.write_str("CUDA_ERROR_DEVICE_NOT_LICENSED"),
            Self::InvalidImage => f.write_str("CUDA_ERROR_INVALID_IMAGE"),
            Self::InvalidContext => f.write_str("CUDA_ERROR_INVALID_CONTEXT"),
            Self::ContextAlreadyCurrent => f.write_str("CUDA_ERROR_CONTEXT_ALREADY_CURRENT"),
            Self::MapFailed => f.write_str("CUDA_ERROR_MAP_FAILED"),
            Self::UnmapFailed => f.write_str("CUDA_ERROR_UNMAP_FAILED"),
            Self::ArrayIsMapped => f.write_str("CUDA_ERROR_ARRAY_IS_MAPPED"),
            Self::AlreadyMapped => f.write_str("CUDA_ERROR_ALREADY_MAPPED"),
            Self::NoBinaryForGpu => f.write_str("CUDA_ERROR_NO_BINARY_FOR_GPU"),
            Self::AlreadyAcquired => f.write_str("CUDA_ERROR_ALREADY_ACQUIRED"),
            Self::NotMapped => f.write_str("CUDA_ERROR_NOT_MAPPED"),
            Self::NotMappedAsArray => f.write_str("CUDA_ERROR_NOT_MAPPED_AS_ARRAY"),
            Self::NotMappedAsPointer => f.write_str("CUDA_ERROR_NOT_MAPPED_AS_POINTER"),
            Self::EccUncorrectable => f.write_str("CUDA_ERROR_ECC_UNCORRECTABLE"),
            Self::UnsupportedLimit => f.write_str("CUDA_ERROR_UNSUPPORTED_LIMIT"),
            Self::ContextAlreadyInUse => f.write_str("CUDA_ERROR_CONTEXT_ALREADY_IN_USE"),
            Self::PeerAccessUnsupported => f.write_str("CUDA_ERROR_PEER_ACCESS_UNSUPPORTED"),
            Self::InvalidPtx => f.write_str("CUDA_ERROR_INVALID_PTX"),
            Self::InvalidGraphicsContext => f.write_str("CUDA_ERROR_INVALID_GRAPHICS_CONTEXT"),
            Self::NvlinkUncorrectable => f.write_str("CUDA_ERROR_NVLINK_UNCORRECTABLE"),
            Self::JitCompilerNotFound => f.write_str("CUDA_ERROR_JIT_COMPILER_NOT_FOUND"),
            Self::UnsupportedPtxVersion => f.write_str("CUDA_ERROR_UNSUPPORTED_PTX_VERSION"),
            Self::JitCompilationDisabled => f.write_str("CUDA_ERROR_JIT_COMPILATION_DISABLED"),
            Self::UnsupportedExecAffinity => f.write_str("CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY"),
            Self::UnsupportedDevsideSync => f.write_str("CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC"),
            Self::Contained => f.write_str("CUDA_ERROR_CONTAINED"),
            Self::InvalidSource => f.write_str("CUDA_ERROR_INVALID_SOURCE"),
            Self::FileNotFound => f.write_str("CUDA_ERROR_FILE_NOT_FOUND"),
            Self::SharedObjectSymbolNotFound => {
                f.write_str("CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND")
            }
            Self::SharedObjectInitFailed => f.write_str("CUDA_ERROR_SHARED_OBJECT_INIT_FAILED"),
            Self::OperatingSystem => f.write_str("CUDA_ERROR_OPERATING_SYSTEM"),
            Self::InvalidHandle => f.write_str("CUDA_ERROR_INVALID_HANDLE"),
            Self::IllegalState => f.write_str("CUDA_ERROR_ILLEGAL_STATE"),
            Self::LossyQuery => f.write_str("CUDA_ERROR_LOSSY_QUERY"),
            Self::NotFound => f.write_str("CUDA_ERROR_NOT_FOUND"),
            Self::NotReady => f.write_str("CUDA_ERROR_NOT_READY"),
            Self::IllegalAddress => f.write_str("CUDA_ERROR_ILLEGAL_ADDRESS"),
            Self::LaunchOutOfResources => f.write_str("CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"),
            Self::LaunchTimeout => f.write_str("CUDA_ERROR_LAUNCH_TIMEOUT"),
            Self::LaunchIncompatibleTexturing => {
                f.write_str("CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING")
            }
            Self::PeerAccessAlreadyEnabled => f.write_str("CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED"),
            Self::PeerAccessNotEnabled => f.write_str("CUDA_ERROR_PEER_ACCESS_NOT_ENABLED"),
            Self::PrimaryContextActive => f.write_str("CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE"),
            Self::ContextIsDestroyed => f.write_str("CUDA_ERROR_CONTEXT_IS_DESTROYED"),
            Self::Assert => f.write_str("CUDA_ERROR_ASSERT"),
            Self::TooManyPeers => f.write_str("CUDA_ERROR_TOO_MANY_PEERS"),
            Self::HostMemoryAlreadyRegistered => {
                f.write_str("CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED")
            }
            Self::HostMemoryNotRegistered => f.write_str("CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED"),
            Self::HardwareStackError => f.write_str("CUDA_ERROR_HARDWARE_STACK_ERROR"),
            Self::IllegalInstruction => f.write_str("CUDA_ERROR_ILLEGAL_INSTRUCTION"),
            Self::MisalignedAddress => f.write_str("CUDA_ERROR_MISALIGNED_ADDRESS"),
            Self::InvalidAddressSpace => f.write_str("CUDA_ERROR_INVALID_ADDRESS_SPACE"),
            Self::InvalidPc => f.write_str("CUDA_ERROR_INVALID_PC"),
            Self::LaunchFailed => f.write_str("CUDA_ERROR_LAUNCH_FAILED"),
            Self::CooperativeLaunchTooLarge => {
                f.write_str("CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE")
            }
            Self::TensorMemoryLeak => f.write_str("CUDA_ERROR_TENSOR_MEMORY_LEAK"),
            Self::NotPermitted => f.write_str("CUDA_ERROR_NOT_PERMITTED"),
            Self::NotSupported => f.write_str("CUDA_ERROR_NOT_SUPPORTED"),
            Self::SystemNotReady => f.write_str("CUDA_ERROR_SYSTEM_NOT_READY"),
            Self::SystemDriverMismatch => f.write_str("CUDA_ERROR_SYSTEM_DRIVER_MISMATCH"),
            Self::CompatNotSupportedOnDevice => {
                f.write_str("CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE")
            }
            Self::MpsConnectionFailed => f.write_str("CUDA_ERROR_MPS_CONNECTION_FAILED"),
            Self::MpsRpcFailure => f.write_str("CUDA_ERROR_MPS_RPC_FAILURE"),
            Self::MpsServerNotReady => f.write_str("CUDA_ERROR_MPS_SERVER_NOT_READY"),
            Self::MpsMaxClientsReached => f.write_str("CUDA_ERROR_MPS_MAX_CLIENTS_REACHED"),
            Self::MpsMaxConnectionsReached => f.write_str("CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED"),
            Self::MpsClientTerminated => f.write_str("CUDA_ERROR_MPS_CLIENT_TERMINATED"),
            Self::CdpNotSupported => f.write_str("CUDA_ERROR_CDP_NOT_SUPPORTED"),
            Self::CdpVersionMismatch => f.write_str("CUDA_ERROR_CDP_VERSION_MISMATCH"),
            Self::StreamCaptureUnsupported => f.write_str("CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED"),
            Self::StreamCaptureInvalidated => f.write_str("CUDA_ERROR_STREAM_CAPTURE_INVALIDATED"),
            Self::StreamCaptureMerge => f.write_str("CUDA_ERROR_STREAM_CAPTURE_MERGE"),
            Self::StreamCaptureUnmatched => f.write_str("CUDA_ERROR_STREAM_CAPTURE_UNMATCHED"),
            Self::StreamCaptureUnjoined => f.write_str("CUDA_ERROR_STREAM_CAPTURE_UNJOINED"),
            Self::StreamCaptureIsolation => f.write_str("CUDA_ERROR_STREAM_CAPTURE_ISOLATION"),
            Self::StreamCaptureImplicit => f.write_str("CUDA_ERROR_STREAM_CAPTURE_IMPLICIT"),
            Self::CapturedEvent => f.write_str("CUDA_ERROR_CAPTURED_EVENT"),
            Self::StreamCaptureWrongThread => f.write_str("CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD"),
            Self::Timeout => f.write_str("CUDA_ERROR_TIMEOUT"),
            Self::GraphExecUpdateFailure => f.write_str("CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE"),
            Self::ExternalDevice => f.write_str("CUDA_ERROR_EXTERNAL_DEVICE"),
            Self::InvalidClusterSize => f.write_str("CUDA_ERROR_INVALID_CLUSTER_SIZE"),
            Self::FunctionNotLoaded => f.write_str("CUDA_ERROR_FUNCTION_NOT_LOADED"),
            Self::InvalidResourceType => f.write_str("CUDA_ERROR_INVALID_RESOURCE_TYPE"),
            Self::InvalidResourceConfiguration => {
                f.write_str("CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION")
            }
            Self::KeyRotation => f.write_str("CUDA_ERROR_KEY_ROTATION"),
            Self::StreamDetached => f.write_str("CUDA_ERROR_STREAM_DETACHED"),
            Self::UnknownError => f.write_str("CUDA_ERROR_UNKNOWN"),
            Self::Unknown(code) => write!(f, "UNKNOWN_CUDA_STATUS({code})"),
        }
    }
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("cuda error ({code}): {message}")]
    Cuda { code: Status, message: String },

    #[error("nvrtc error ({code}): {message}")]
    Nvrtc { code: NvrtcStatus, message: String },

    #[error("nvvm error ({code}): {message}")]
    Nvvm { code: NvvmStatus, message: String },

    #[error("string contains interior nul byte")]
    InteriorNul,

    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("device not found")]
    DeviceNotFound,

    #[error("invalid memory access")]
    InvalidMemoryAccess,
    #[error("invalid memory allocation request")]
    InvalidMemoryAllocationRequest,
    #[error("unexpected null handle")]
    NullHandle,
    #[error("invalid value")]
    InvalidValue,

    #[error("`{name}` must be nonzero")]
    ZeroValue { name: String },

    #[error("`{name}` is out of range")]
    OutOfRange { name: String },

    #[error("operation graph required")]
    GraphOperationRequired,
    #[error("graph dependency mismatch")]
    GraphDependencyMismatch,
    #[error("graph node belongs to a different graph")]
    GraphNodeMismatch,
    #[error("graph belongs to a different cuda context")]
    GraphContextMismatch,

    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<runtime::cudaError_t> for Error {
    fn from(code: runtime::cudaError_t) -> Self {
        let message = unsafe {
            let c_ptr = runtime::cudaGetErrorString(code);
            if c_ptr.is_null() {
                String::from("unknown cuda error")
            } else {
                CStr::from_ptr(c_ptr).to_string_lossy().into_owned()
            }
        };

        Self::Cuda {
            code: Status::from(code),
            message,
        }
    }
}

impl From<Status> for Error {
    fn from(code: Status) -> Self {
        Self::Cuda {
            code,
            message: code.description(),
        }
    }
}

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

impl From<nvrtc::nvrtcResult> for Error {
    fn from(code: nvrtc::nvrtcResult) -> Self {
        let code = NvrtcStatus::from(code);
        Self::from(code)
    }
}

impl From<NvrtcStatus> for Error {
    fn from(code: NvrtcStatus) -> Self {
        Self::Nvrtc {
            code,
            message: code.description(),
        }
    }
}

impl From<nvvm::nvvmResult> for Error {
    fn from(code: nvvm::nvvmResult) -> Self {
        let code = NvvmStatus::from(code);
        Self::from(code)
    }
}

impl From<NvvmStatus> for Error {
    fn from(code: NvvmStatus) -> Self {
        Self::Nvvm {
            code,
            message: code.description(),
        }
    }
}

#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let err = { $expr };
        if err != singe_cuda_sys::runtime::cudaError_t::CUDA_SUCCESS {
            Err($crate::error::Error::from(err))
        } else {
            Ok(())
        }
    }};
}

#[macro_export]
macro_rules! try_nvrtc {
    ($expr:expr) => {{
        let err = { $expr };
        if err != singe_cuda_sys::nvrtc::nvrtcResult::NVRTC_SUCCESS {
            Err($crate::error::Error::from(err))
        } else {
            Ok(())
        }
    }};
}

#[macro_export]
macro_rules! try_nvvm {
    ($expr:expr) => {{
        let err = { $expr };
        if err != singe_cuda_sys::nvvm::nvvmResult::NVVM_SUCCESS {
            Err($crate::error::Error::from(err))
        } else {
            Ok(())
        }
    }};
}
