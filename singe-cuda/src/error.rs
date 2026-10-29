#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

use std::{
    borrow::Cow,
    error,
    ffi::{CStr, NulError},
    io,
    result::Result as StdResult,
    sync::Arc,
};

use oxide_artifacts::ArtifactError;
use singe_core::{static_str_from_c_ptr, status_enum};
use singe_cuda_sys::{nvrtc, nvvm, runtime};
use thiserror::Error;

#[cfg(feature = "nvjitlink_13_4")]
use crate::nvjitlink::Status as NvJitLinkStatus;
use crate::{
    kernel::launch::LaunchContractError, nvrtc::Status as NvrtcStatus, nvvm::Status as NvvmStatus,
};

status_enum! {
    /// Error codes
    pub enum Status: u32 => runtime::cudaError_t, unknown = "UNKNOWN_CUDA_STATUS" {
        /// The API call returned with no errors.
        ///
        /// In the case of query calls, this also means that the operation being queried is complete (see `cuEventQuery` and `cuStreamQuery`).
        Success = CUDA_SUCCESS,
        /// This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
        InvalidValue = CUDA_ERROR_INVALID_VALUE,
        /// The API call failed because it was unable to allocate enough memory or other resources to perform the requested operation.
        OutOfMemory = CUDA_ERROR_OUT_OF_MEMORY,
        /// This indicates that the CUDA driver has not been initialized with `cuInit` or that initialization has failed.
        NotInitialized = CUDA_ERROR_NOT_INITIALIZED,
        /// This indicates that the CUDA driver is in the process of shutting down.
        Deinitialized = CUDA_ERROR_DEINITIALIZED,
        /// This indicates profiler is not initialized for this run.
        ///
        /// This can happen when the application is running with external profiling tools like visual profiler.
        ProfilerDisabled = CUDA_ERROR_PROFILER_DISABLED,
        /// Deprecated: This error return is deprecated as of CUDA 5.0. It is no longer an error  to attempt to enable/disable the profiling via ::cuProfilerStart or  ::cuProfilerStop without initialization.
        #[deprecated]
        ProfilerNotInitialized = CUDA_ERROR_PROFILER_NOT_INITIALIZED,
        /// Deprecated: This error return is deprecated as of CUDA 5.0. It is no longer an error  to call cuProfilerStart() when profiling is already enabled.
        #[deprecated]
        ProfilerAlreadyStarted = CUDA_ERROR_PROFILER_ALREADY_STARTED,
        /// Deprecated: This error return is deprecated as of CUDA 5.0. It is no longer an error  to call cuProfilerStop() when profiling is already disabled.
        #[deprecated]
        ProfilerAlreadyStopped = CUDA_ERROR_PROFILER_ALREADY_STOPPED,
        /// This indicates that the CUDA driver that the application has loaded is a stub library.
        ///
        /// Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.
        StubLibrary = CUDA_ERROR_STUB_LIBRARY,
        /// This indicates that the API call requires a newer CUDA driver than the one currently installed.
        ///
        /// Users should install an updated NVIDIA CUDA driver to allow the API call to succeed.
        CallRequiresNewerDriver = CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER,
        /// This indicates that requested CUDA device is unavailable at the current time.
        ///
        /// Devices are often unavailable due to use of `CUcomputemode_enum::CU_COMPUTEMODE_EXCLUSIVE_PROCESS` or `CUcomputemode_enum::CU_COMPUTEMODE_PROHIBITED`.
        DeviceUnavailable = CUDA_ERROR_DEVICE_UNAVAILABLE,
        /// This indicates that no CUDA-capable devices were detected by the installed CUDA driver.
        NoDevice = CUDA_ERROR_NO_DEVICE,
        /// This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device.
        InvalidDevice = CUDA_ERROR_INVALID_DEVICE,
        /// This error indicates that the Grid license is not applied.
        DeviceNotLicensed = CUDA_ERROR_DEVICE_NOT_LICENSED,
        /// This indicates that the device kernel image is invalid.
        ///
        /// This can also indicate an invalid CUDA module.
        InvalidImage = CUDA_ERROR_INVALID_IMAGE,
        /// This most frequently indicates that there is no context bound to the current thread.
        ///
        /// This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See `cuCtxGetApiVersion` for more details. This can also be returned if the green context passed to an API call was not converted to a `CUcontext` using `cuCtxFromGreenCtx` API.
        InvalidContext = CUDA_ERROR_INVALID_CONTEXT,
        /// This indicated that the context being supplied as a parameter to the API call was already the active context.
        #[deprecated]
        ContextAlreadyCurrent = CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
        /// This indicates that a map or register operation has failed.
        MapFailed = CUDA_ERROR_MAP_FAILED,
        /// This indicates that an unmap or unregister operation has failed.
        UnmapFailed = CUDA_ERROR_UNMAP_FAILED,
        /// This indicates that the specified array is currently mapped and thus cannot be destroyed.
        ArrayIsMapped = CUDA_ERROR_ARRAY_IS_MAPPED,
        /// This indicates that the resource is already mapped.
        AlreadyMapped = CUDA_ERROR_ALREADY_MAPPED,
        /// This indicates that there is no kernel image available that is suitable for the device.
        ///
        /// This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.
        NoBinaryForGpu = CUDA_ERROR_NO_BINARY_FOR_GPU,
        /// This indicates that a resource has already been acquired.
        AlreadyAcquired = CUDA_ERROR_ALREADY_ACQUIRED,
        /// This indicates that a resource is not mapped.
        NotMapped = CUDA_ERROR_NOT_MAPPED,
        /// This indicates that a mapped resource is not available for access as an array.
        NotMappedAsArray = CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
        /// This indicates that a mapped resource is not available for access as a pointer.
        NotMappedAsPointer = CUDA_ERROR_NOT_MAPPED_AS_POINTER,
        /// This indicates that an uncorrectable ECC error was detected during execution.
        EccUncorrectable = CUDA_ERROR_ECC_UNCORRECTABLE,
        /// This indicates that the [CUlimit](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html#group__cuda__types_1gae24c2d4214af24139020f1aecaf32665) passed to the API call is not supported by the active device.
        UnsupportedLimit = CUDA_ERROR_UNSUPPORTED_LIMIT,
        /// This indicates that the `CUcontext` passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread.
        ContextAlreadyInUse = CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
        /// This indicates that peer access is not supported across the given devices.
        PeerAccessUnsupported = CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
        /// This indicates that a PTX JIT compilation failed.
        InvalidPtx = CUDA_ERROR_INVALID_PTX,
        /// This indicates an error with OpenGL or DirectX context.
        InvalidGraphicsContext = CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
        /// This indicates that an uncorrectable NVLink error was detected during the execution.
        NvlinkUncorrectable = CUDA_ERROR_NVLINK_UNCORRECTABLE,
        /// This indicates that the PTX JIT compiler library was not found.
        JitCompilerNotFound = CUDA_ERROR_JIT_COMPILER_NOT_FOUND,
        /// This indicates that the provided PTX was compiled with an unsupported toolchain.
        UnsupportedPtxVersion = CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
        /// This indicates that the PTX JIT compilation was disabled.
        JitCompilationDisabled = CUDA_ERROR_JIT_COMPILATION_DISABLED,
        /// This indicates that the [CUexecAffinityType](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html#group__cuda__types_1ga14e6345acf2bda65be91eda77cf03f5c) passed to the API call is not supported by the active device.
        UnsupportedExecAffinity = CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY,
        /// This indicates that the code to be compiled by the PTX JIT contains unsupported call to cudaDeviceSynchronize.
        UnsupportedDevsideSync = CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC,
        /// This indicates that an exception occurred on the device that is now contained by the GPU's error containment capability.
        ///
        /// Common causes are - a. Certain types of invalid accesses of peer GPU memory over nvlink b. Certain classes of hardware errors This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
        Contained = CUDA_ERROR_CONTAINED,
        /// This indicates that the device kernel source is invalid.
        ///
        /// This includes compilation/linker errors encountered in device code or user error.
        InvalidSource = CUDA_ERROR_INVALID_SOURCE,
        /// This indicates that the file specified was not found.
        FileNotFound = CUDA_ERROR_FILE_NOT_FOUND,
        /// This indicates that initialization of a shared object failed.
        SharedObjectInitFailed = CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
        /// This indicates that an OS call failed.
        OperatingSystem = CUDA_ERROR_OPERATING_SYSTEM,
        /// This indicates that a resource handle passed to the API call was not valid.
        ///
        /// Resource handles are opaque types like `CUstream` and `CUevent`.
        InvalidHandle = CUDA_ERROR_INVALID_HANDLE,
        /// This indicates that a resource required by the API call is not in a valid state to perform the requested operation.
        IllegalState = CUDA_ERROR_ILLEGAL_STATE,
        /// This indicates an attempt was made to introspect an object in a way that would discard semantically important information.
        ///
        /// This is either due to the object using funtionality newer than the API version used to introspect it or omission of optional return arguments.
        LossyQuery = CUDA_ERROR_LOSSY_QUERY,
        /// This indicates that a named symbol was not found.
        ///
        /// Examples of symbols are global/constant variable names, driver function names, texture names, and surface names.
        NotFound = CUDA_ERROR_NOT_FOUND,
        /// This indicates that asynchronous operations issued previously have not completed yet.
        ///
        /// This result is not actually an error, but must be indicated differently than `cudaError_enum::CUDA_SUCCESS` (which indicates completion). Calls that may return this value include `cuEventQuery` and `cuStreamQuery`.
        NotReady = CUDA_ERROR_NOT_READY,
        /// While executing a kernel, the device encountered a load or store instruction on an invalid memory address.
        ///
        /// This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
        IllegalAddress = CUDA_ERROR_ILLEGAL_ADDRESS,
        /// This indicates that a launch did not occur because it did not have appropriate resources.
        ///
        /// This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count. Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too many arguments and can also result in this error.
        LaunchOutOfResources = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
        /// This indicates that the device kernel took too long to execute.
        ///
        /// This can only occur if timeouts are enabled - see the device attribute `CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT` for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
        LaunchTimeout = CUDA_ERROR_LAUNCH_TIMEOUT,
        /// This error indicates that a call to `cuCtxEnablePeerAccess` is trying to re-enable peer access to a context which has already had peer access to it enabled.
        PeerAccessAlreadyEnabled = CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
        /// This error indicates that `cuCtxDisablePeerAccess` is trying to disable peer access which has not been enabled yet via `cuCtxEnablePeerAccess`.
        PeerAccessNotEnabled = CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
        /// This error indicates that the primary context for the specified device has already been initialized.
        PrimaryContextActive = CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
        /// This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized.
        ContextIsDestroyed = CUDA_ERROR_CONTEXT_IS_DESTROYED,
        /// A device-side assert triggered during kernel execution.
        ///
        /// The context cannot be used anymore, and must be destroyed. All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.
        Assert = CUDA_ERROR_ASSERT,
        /// This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to `cuCtxEnablePeerAccess`.
        TooManyPeers = CUDA_ERROR_TOO_MANY_PEERS,
        /// This error indicates that the pointer passed to `cuMemHostUnregister` does not correspond to any currently registered memory region.
        HostMemoryNotRegistered = CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
        /// While executing a kernel, the device encountered a stack error.
        ///
        /// This can be due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
        HardwareStackError = CUDA_ERROR_HARDWARE_STACK_ERROR,
        /// While executing a kernel, the device encountered an illegal instruction.
        ///
        /// This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
        IllegalInstruction = CUDA_ERROR_ILLEGAL_INSTRUCTION,
        /// While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned.
        ///
        /// This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
        MisalignedAddress = CUDA_ERROR_MISALIGNED_ADDRESS,
        /// While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space.
        ///
        /// This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
        InvalidAddressSpace = CUDA_ERROR_INVALID_ADDRESS_SPACE,
        /// While executing a kernel, the device program counter wrapped its address space.
        ///
        /// This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
        InvalidPc = CUDA_ERROR_INVALID_PC,
        /// An exception occurred on the device while executing a kernel.
        ///
        /// Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
        LaunchFailed = CUDA_ERROR_LAUNCH_FAILED,
        /// This error indicates that the number of blocks launched per grid for a kernel that was launched via either `cuLaunchCooperativeKernel` or `cuLaunchCooperativeKernelMultiDevice` exceeds the maximum number of blocks as allowed by `cuOccupancyMaxActiveBlocksPerMultiprocessor` or `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` times the number of multiprocessors as specified by the device attribute `CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.
        CooperativeLaunchTooLarge = CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
        /// An exception occurred on the device while exiting a kernel using tensor memory: the tensor memory was not completely deallocated.
        ///
        /// This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
        TensorMemoryLeak = CUDA_ERROR_TENSOR_MEMORY_LEAK,
        /// This error indicates that the attempted operation is not permitted.
        NotPermitted = CUDA_ERROR_NOT_PERMITTED,
        /// This error indicates that the attempted operation is not supported on the current system or device.
        NotSupported = CUDA_ERROR_NOT_SUPPORTED,
        /// This error indicates that the system is not yet ready to start any CUDA work.
        ///
        /// To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.
        SystemNotReady = CUDA_ERROR_SYSTEM_NOT_READY,
        /// This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver.
        ///
        /// Refer to the compatibility documentation for supported versions.
        SystemDriverMismatch = CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
        /// This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
        MpsConnectionFailed = CUDA_ERROR_MPS_CONNECTION_FAILED,
        /// This error indicates that the remote procedural call between the MPS server and the MPS client failed.
        MpsRpcFailure = CUDA_ERROR_MPS_RPC_FAILURE,
        /// This error indicates that the MPS server is not ready to accept new MPS client requests.
        ///
        /// This error can be returned when the MPS server is in the process of recovering from a fatal failure.
        MpsServerNotReady = CUDA_ERROR_MPS_SERVER_NOT_READY,
        /// This error indicates that the hardware resources required to create MPS client have been exhausted.
        MpsMaxClientsReached = CUDA_ERROR_MPS_MAX_CLIENTS_REACHED,
        /// This error indicates the hardware resources required to support device connections have been exhausted.
        MpsMaxConnectionsReached = CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED,
        /// This error indicates that the MPS client has been terminated by the server.
        ///
        /// To continue using CUDA, the process must be terminated and relaunched.
        MpsClientTerminated = CUDA_ERROR_MPS_CLIENT_TERMINATED,
        /// This error indicates that the module is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it.
        CdpNotSupported = CUDA_ERROR_CDP_NOT_SUPPORTED,
        /// This error indicates that a module contains an unsupported interaction between different versions of CUDA Dynamic Parallelism.
        CdpVersionMismatch = CUDA_ERROR_CDP_VERSION_MISMATCH,
        /// This error indicates that the operation is not permitted when the stream is capturing.
        StreamCaptureUnsupported = CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
        /// This error indicates that the current capture sequence on the stream has been invalidated due to a previous error.
        StreamCaptureInvalidated = CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
        /// This error indicates that the operation would have resulted in a merge of two independent capture sequences.
        StreamCaptureMerge = CUDA_ERROR_STREAM_CAPTURE_MERGE,
        /// This error indicates that the capture was not initiated in this stream.
        StreamCaptureUnmatched = CUDA_ERROR_STREAM_CAPTURE_UNMATCHED,
        /// This error indicates that the capture sequence contains a fork that was not joined to the primary stream.
        StreamCaptureUnjoined = CUDA_ERROR_STREAM_CAPTURE_UNJOINED,
        /// This error indicates that a dependency would have been created which crosses the capture sequence boundary.
        ///
        /// Only implicit in-stream ordering dependencies are allowed to cross the boundary.
        StreamCaptureIsolation = CUDA_ERROR_STREAM_CAPTURE_ISOLATION,
        /// This error indicates a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.
        StreamCaptureImplicit = CUDA_ERROR_STREAM_CAPTURE_IMPLICIT,
        /// This error indicates that the operation is not permitted on an event which was last recorded in a capturing stream.
        CapturedEvent = CUDA_ERROR_CAPTURED_EVENT,
        /// A stream capture sequence not initiated with the `CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED` argument to cuStreamBeginCapture was passed to `cuStreamEndCapture` in a different thread.
        StreamCaptureWrongThread = CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD,
        /// This error indicates that the timeout specified for the wait operation has lapsed.
        Timeout = CUDA_ERROR_TIMEOUT,
        /// This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.
        GraphExecUpdateFailure = CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE,
        /// This indicates that an error has occurred in a device outside of GPU.
        ///
        /// It can be a synchronous error with respect to CUDA API or an asynchronous error from the external device. In case of asynchronous error, it means that if CUDA was waiting for an external device's signal before consuming shared data, the external device signaled an error indicating that the data is not valid for consumption. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. In case of synchronous error, it means that one or more external devices have encountered an error and cannot complete the operation.
        ExternalDevice = CUDA_ERROR_EXTERNAL_DEVICE,
        /// Indicates a kernel launch error due to cluster misconfiguration.
        InvalidClusterSize = CUDA_ERROR_INVALID_CLUSTER_SIZE,
        /// Indiciates a function handle is not loaded when calling an API that requires a loaded function.
        FunctionNotLoaded = CUDA_ERROR_FUNCTION_NOT_LOADED,
        /// This error indicates one or more resources passed in are not valid resource types for the operation.
        InvalidResourceType = CUDA_ERROR_INVALID_RESOURCE_TYPE,
        /// This error indicates that an error happened during the key rotation sequence.
        KeyRotation = CUDA_ERROR_KEY_ROTATION,
        /// This error indicates that the requested operation is not permitted because the stream is in a detached state.
        ///
        /// This can occur if the green context associated with the stream has been destroyed, limiting the stream's operational capabilities.
        StreamDetached = CUDA_ERROR_STREAM_DETACHED,
        /// The API call failed because of a hardware resource required to bind memory to a multicast object is unavailable.
        MulticastResourceFull = CUDA_ERROR_MULTICAST_RESOURCE_FULL,
        /// This indicates that the Loader version is insufficient for fatbin.
        InsufficientLoaderVersion = CUDA_ERROR_INSUFFICIENT_LOADER_VERSION,
        /// The GPU fabric is not ready within the bounded wait while the fabric manager probe is still in progress (or not converging in time).
        ///
        /// Applications may retry after a delay; for the initialization wait budget, see environment variables such as CUDA_FABRIC_INIT_TIMEOUT_MS. The CUDA Runtime uses the same value as ::cudaErrorFabricNotReady.
        FabricNotReady = CUDA_ERROR_FABRIC_NOT_READY,
        /// This indicates that an unknown internal error has occurred.
        UnknownError = CUDA_ERROR_UNKNOWN,
    }
}

impl Status {
    /// Library description of this status.
    #[must_use]
    pub fn description(self) -> &'static str {
        let Ok(raw) = runtime::cudaError_t::try_from(self) else {
            return "unknown cuda error";
        };
        // SAFETY: `cudaGetErrorString` returns null or a static NUL-terminated string.
        unsafe { static_str_from_c_ptr(runtime::cudaGetErrorString(raw), "unknown cuda error") }
    }
}

/// CUDA wrapper failures, including native status codes, compiler diagnostics, resource ownership, and submission tracking.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// The requested GPU architecture cannot be represented or supported by this operation.
    #[error("unsupported gpu architecture `{target}`")]
    UnsupportedArchitecture {
        /// Requested GPU architecture name.
        target: String,
    },
    /// No embedded artifact with the requested name matches the target compute capability.
    #[error(
        "no compatible embedded artifact named `{name}` for compute capability {major}.{minor}"
    )]
    NoCompatibleArtifact {
        /// Requested embedded artifact name.
        name: Cow<'static, str>,
        /// Target compute capability major component.
        major: u32,
        /// Target compute capability minor component.
        minor: u32,
    },
    /// Embedded artifact selection, decoding, or loading failed.
    #[error(transparent)]
    Artifact(
        /// Underlying artifact error.
        #[from]
        ArtifactError,
    ),
    /// A per-thread default stream cannot be retained as a stream shared across threads.
    #[error("per-thread default streams cannot be retained as shared streams")]
    ThreadLocalStreamUnsupported,
    /// The foreign readiness stream and Singe stream refer to different native CUDA streams.
    #[error("the singe and foreign readiness streams must refer to the same native stream")]
    ForeignStreamMismatch,
    /// The submitted native operation returned an error.
    #[error("native operation failed: {0}")]
    Operation(
        /// Underlying native operation error.
        #[source]
        Box<dyn error::Error + Send + Sync>,
    ),
    /// An error was retained while waiting for submission completion.
    #[error("submission completion failed: {0}")]
    Completion(
        /// Shared error retained by the submission completion state.
        #[source]
        Arc<Self>,
    ),
    /// The worker responsible for tracking CUDA completion stopped.
    #[error("cuda completion worker stopped")]
    CompletionWorkerStopped,
    /// Destroying a retained submission resource panicked.
    #[error("submission resource destruction panicked")]
    ResourceDestructionPanicked,
    /// Retiring completed submission state panicked.
    #[error("completion retirement panicked")]
    CompletionPanicked,
    /// The device cannot provide the concurrent managed access required for managed host access.
    #[error("managed host access requires concurrent managed access support")]
    ManagedHostAccessUnsupported,
    /// CUDA access was requested while the calling thread already held a host or enqueue access guard.
    #[error("nested cuda access while a host or enqueue access guard is held")]
    ReentrantAccess,
    /// A resource mutex was poisoned by an earlier panic.
    #[error("`{name}` resource lock is poisoned")]
    ResourcePoisoned {
        /// Name of the resource whose lock is poisoned.
        name: &'static str,
    },
    /// Submission access tracking was poisoned by an earlier panic.
    #[error("submission access tracking is poisoned")]
    AccessTrackingPoisoned,
    /// Kernel argument, launch, or memory-access validation failed.
    #[error(transparent)]
    LaunchContract(
        /// Kernel launch contract violation.
        #[from]
        LaunchContractError,
    ),
    /// A compiler operation failed with a diagnostic message.
    #[error("compiler error: {message}")]
    Compiler {
        /// Diagnostic text returned by the compiler integration.
        message: String,
    },
    /// The native device linker returned a failure status and description.
    #[cfg(feature = "nvjitlink_13_4")]
    #[error("nvjitlink error ({code}): {message}")]
    NvJitLink {
        /// Native device-linker status.
        code: NvJitLinkStatus,
        /// Description of the linker status.
        message: String,
    },
    /// A native CUDA operation returned a failure status and description.
    #[error("cuda error ({code}): {message}")]
    Cuda {
        /// Native CUDA status.
        code: Status,
        /// Description of the CUDA status.
        message: String,
    },

    /// A native NVRTC operation returned a failure status and description.
    #[error("nvrtc error ({code}): {message}")]
    Nvrtc {
        /// Native NVRTC status.
        code: NvrtcStatus,
        /// Description of the NVRTC status.
        message: String,
    },

    /// NVVM compilation returned a failure status and description.
    #[error("nvvm error ({code}): {message}")]
    Nvvm {
        /// Native NVVM status.
        code: NvvmStatus,
        /// Description of the NVVM status.
        message: String,
    },

    /// A string intended for a native NUL-terminated interface contains an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// An operating-system I/O operation failed.
    #[error("io error: {0}")]
    Io(
        /// Underlying I/O error.
        #[from]
        io::Error,
    ),

    /// The requested CUDA device could not be found.
    #[error("device not found")]
    DeviceNotFound,

    /// The requested memory access does not satisfy the resource bounds or access contract.
    #[error("invalid memory access")]
    InvalidMemoryAccess,
    /// The requested memory extent cannot be allocated or represented by the allocation calculation.
    #[error("invalid memory allocation request")]
    InvalidMemoryAllocationRequest,
    /// Another owner still retains the allocation, preventing exclusive ownership transfer or release.
    #[error("allocation is still retained by another owner")]
    AllocationInUse,
    /// The memory allocation belongs to another CUDA context.
    #[error("memory belongs to a different cuda context")]
    MemoryContextMismatch,
    /// A native handle or pointer expected to be non-null was null.
    #[error("unexpected null handle")]
    NullHandle,
    /// An argument or returned value violates the operation's value requirements.
    #[error("invalid value")]
    InvalidValue,

    /// A named argument is zero where a nonzero value is required.
    #[error("`{name}` must be nonzero")]
    ZeroValue {
        /// Name of the argument that must be nonzero.
        name: Cow<'static, str>,
    },

    /// A named argument or computed extent exceeds the supported range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the argument or computed extent that exceeded its range.
        name: Cow<'static, str>,
    },

    /// The requested operation requires an operation graph.
    #[error("operation graph required")]
    GraphOperationRequired,
    /// Graph dependency relationships do not match the required graph state.
    #[error("graph dependency mismatch")]
    GraphDependencyMismatch,
    /// A graph node belongs to a different graph.
    #[error("graph node belongs to a different graph")]
    GraphNodeMismatch,
    /// The graph belongs to a different CUDA context.
    #[error("graph belongs to a different cuda context")]
    GraphContextMismatch,

    /// The stream belongs to a different CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,
}

/// Result of a CUDA wrapper operation, using the crate's typed `Error`.
pub type Result<T> = StdResult<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

impl From<runtime::cudaError_t> for Error {
    fn from(code: runtime::cudaError_t) -> Self {
        // SAFETY: `cudaGetErrorString` returns null or a static NUL-terminated string; null is handled.
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
            message: code.description().to_owned(),
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
            message: code.description().to_string(),
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
            message: code.description().to_string(),
        }
    }
}

/// Converts a raw CUDA status-returning expression into a crate [`Result`](crate::error::Result).
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_cuda_sys::runtime::cudaError_t::CUDA_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}

/// Converts a raw NVRTC status-returning expression into a crate [`Result`](crate::error::Result).
#[macro_export]
macro_rules! try_nvrtc {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_cuda_sys::nvrtc::nvrtcResult::NVRTC_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}

/// Converts a raw NVVM status-returning expression into a crate [`Result`](crate::error::Result).
#[macro_export]
macro_rules! try_nvvm {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_cuda_sys::nvvm::nvvmResult::NVVM_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}
