use std::fmt::{self, Display, Formatter};

use bitflags::{bitflags, parser};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cupti_sys as sys;

use crate::types::ProcessId;

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// CUPTI channel type reported by activity records.
    pub enum ChannelType: sys::CUpti_ChannelType {
        /// Invalid value decoded from CUPTI.
        Invalid = CUPTI_CHANNEL_TYPE_INVALID,
        /// Channel is used for standard work launch and tracking.
        Compute = CUPTI_CHANNEL_TYPE_COMPUTE,
        /// Channel is used by an asynchronous copy engine For confidential compute configurations, work launch and completion are done using the copy engines.
        AsynchronousMemoryCopy = CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY,
        /// Channel is used for memory decompression operations.
        Decompress = CUPTI_CHANNEL_TYPE_DECOMP,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_CHANNEL_TYPE_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Source and destination kind for an activity memory-copy record.
    pub enum ActivityMemoryCopyKind: sys::CUpti_ActivityMemcpyKind {
        /// The memory copy kind is not known.
        Unknown = CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN,
        /// A host to device memory copy.
        HostToDevice = CUPTI_ACTIVITY_MEMCPY_KIND_HTOD,
        /// A device to host memory copy.
        DeviceToHost = CUPTI_ACTIVITY_MEMCPY_KIND_DTOH,
        /// A host to device array memory copy.
        HostToArray = CUPTI_ACTIVITY_MEMCPY_KIND_HTOA,
        /// A device array to host memory copy.
        ArrayToHost = CUPTI_ACTIVITY_MEMCPY_KIND_ATOH,
        /// A device array to device array memory copy.
        ArrayToArray = CUPTI_ACTIVITY_MEMCPY_KIND_ATOA,
        /// A device array to device memory copy.
        ArrayToDevice = CUPTI_ACTIVITY_MEMCPY_KIND_ATOD,
        /// A device to device array memory copy.
        DeviceToArray = CUPTI_ACTIVITY_MEMCPY_KIND_DTOA,
        /// A device to device memory copy on the same device.
        DeviceToDevice = CUPTI_ACTIVITY_MEMCPY_KIND_DTOD,
        /// A host to host memory copy.
        HostToHost = CUPTI_ACTIVITY_MEMCPY_KIND_HTOH,
        /// A peer to peer memory copy across different devices.
        PeerToPeer = CUPTI_ACTIVITY_MEMCPY_KIND_PTOP,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Memory kind reported by activity memory records.
    pub enum ActivityMemoryKind: sys::CUpti_ActivityMemoryKind {
        /// The memory kind is unknown.
        Unknown = CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN,
        /// The memory is pageable.
        Pageable = CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE,
        /// The memory is pinned.
        Pinned = CUPTI_ACTIVITY_MEMORY_KIND_PINNED,
        /// The memory is on the device.
        Device = CUPTI_ACTIVITY_MEMORY_KIND_DEVICE,
        /// The memory is an array.
        Array = CUPTI_ACTIVITY_MEMORY_KIND_ARRAY,
        /// The memory is managed.
        Managed = CUPTI_ACTIVITY_MEMORY_KIND_MANAGED,
        /// The memory is device static.
        DeviceStatic = CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC,
        /// The memory is managed static.
        ManagedStatic = CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_MEMORY_KIND_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Operation type reported by [`crate::activity::ActivityMemory`].
    pub enum ActivityMemoryOperationType: sys::CUpti_ActivityMemoryOperationType {
        /// The operation is invalid.
        Invalid = CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_INVALID,
        /// Memory is allocated.
        Allocation = CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION,
        /// Memory is released.
        Release = CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Memory pool type reported by [`crate::activity::ActivityMemoryPool`].
    pub enum ActivityMemoryPoolType: sys::CUpti_ActivityMemoryPoolType {
        /// The operation is invalid.
        Invalid = CUPTI_ACTIVITY_MEMORY_POOL_TYPE_INVALID,
        /// Memory pool is local to the process.
        Local = CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL,
        /// Memory pool is imported by the process.
        Imported = CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_MEMORY_POOL_TYPE_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Memory pool operation reported by [`crate::activity::ActivityMemoryPool`].
    pub enum ActivityMemoryPoolOperationType: sys::CUpti_ActivityMemoryPoolOperationType {
        /// The operation is invalid.
        Invalid = CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_INVALID,
        /// Memory pool is created.
        Created = CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_CREATED,
        /// Memory pool is destroyed.
        Destroyed = CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_DESTROYED,
        /// Memory pool is trimmed.
        Trimmed = CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Launch mode used for device graph execution.
    pub enum DeviceGraphLaunchMode: sys::CUpti_DeviceGraphLaunchMode {
        /// Invalid value decoded from CUPTI.
        Invalid = CUPTI_DEVICE_GRAPH_LAUNCH_MODE_INVALID,
        /// Fire And Forget value decoded from CUPTI.
        FireAndForget = CUPTI_DEVICE_GRAPH_LAUNCH_MODE_FIRE_AND_FORGET,
        /// Tail value decoded from CUPTI.
        Tail = CUPTI_DEVICE_GRAPH_LAUNCH_MODE_TAIL,
        /// Fire And Forget As Sibling value decoded from CUPTI.
        FireAndForgetAsSibling = CUPTI_DEVICE_GRAPH_LAUNCH_MODE_FIRE_AND_FORGET_AS_SIBLING,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Compute API family reported by activity records.
    pub enum ActivityComputeApiKind: sys::CUpti_ActivityComputeApiKind {
        /// The compute API is not known.
        Unknown = CUPTI_ACTIVITY_COMPUTE_API_UNKNOWN,
        /// The compute APIs are for CUDA.
        Cuda = CUPTI_ACTIVITY_COMPUTE_API_CUDA,
        /// The compute APIs are for CUDA running in MPS (Multi-Process Service) environment.
        CudaMultiProcessService = CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_COMPUTE_API_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// CUDA-in-Graphics mode associated with a CUDA context.
    pub enum ContextCigMode: sys::CUpti_ContextCigMode {
        /// Regular (non-CIG) mode.
        None = CUPTI_CONTEXT_CIG_MODE_NONE,
        /// CIG mode.
        Cig = CUPTI_CONTEXT_CIG_MODE_CIG,
        /// CIG fallback mode.
        CigFallback = CUPTI_CONTEXT_CIG_MODE_CIG_FALLBACK,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_CONTEXT_CIG_MODE_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Operation reported by a compute-engine context-switch record.
    pub enum ComputeEngineContextSwitchOperationType: sys::CUpti_ComputeEngineCtxSwitchOperationType {
        /// Invalid value decoded from CUPTI.
        Invalid = CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_INVALID,
        /// The start of the CUDA context switch operation.
        Start = CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_START,
        /// The end of the CUDA context switch operation.
        End = CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_END,
        /// Number of entries defined for this CUPTI enum.
        Count = CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_COUNT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Synchronization operation reported by [`crate::activity::ActivitySynchronization`].
    pub enum ActivitySynchronizationType: sys::CUpti_ActivitySynchronizationType {
        /// Unknown data.
        Unknown = CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN,
        /// Event synchronize API.
        EventSynchronize = CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE,
        /// Stream wait event API.
        StreamWaitEvent = CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT,
        /// Stream synchronize API.
        StreamSynchronize = CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE,
        /// Context synchronize API.
        ContextSynchronize = CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Stream flag reported by [`crate::activity::ActivityStream`].
    pub enum ActivityStreamFlag: sys::CUpti_ActivityStreamFlag {
        /// Unknown data.
        Unknown = CUPTI_ACTIVITY_STREAM_CREATE_FLAG_UNKNOWN,
        /// Default stream.
        Default = CUPTI_ACTIVITY_STREAM_CREATE_FLAG_DEFAULT,
        /// Non-blocking stream.
        NonBlocking = CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING,
        /// Null stream.
        Null = CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NULL,
        /// Stream create Mask.
        Mask = CUPTI_ACTIVITY_STREAM_CREATE_MASK,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_STREAM_CREATE_FLAG_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Partitioned global cache configuration reported by kernel activity records.
    pub enum ActivityPartitionedGlobalCacheConfig: sys::CUpti_ActivityPartitionedGlobalCacheConfig {
        /// Partitioned global cache config unknown.
        Unknown = CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_UNKNOWN,
        /// Partitioned global cache not supported.
        NotSupported = CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED,
        /// Partitioned global cache config off.
        Off = CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF,
        /// Partitioned global cache config on.
        On = CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// CUDA kernel launch type reported by kernel activity records.
    pub enum ActivityLaunchType: sys::CUpti_ActivityLaunchType {
        /// The kernel was launched via a regular kernel call.
        Regular = CUPTI_ACTIVITY_LAUNCH_TYPE_REGULAR,
        /// The kernel was launched via API `cudaLaunchCooperativeKernel()` or `cuLaunchCooperativeKernel()`.
        CooperativeSingleDevice = CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_SINGLE_DEVICE,
        /// The kernel was launched via API `cudaLaunchCooperativeKernelMultiDevice()` or `cuLaunchCooperativeKernelMultiDevice()`.
        CooperativeMultiDevice = CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_MULTI_DEVICE,
        /// The kernel was launched as a CBL commandlist.
        CblCommandlist = CUPTI_ACTIVITY_LAUNCH_TYPE_CBL_COMMANDLIST,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Shared-memory limit configuration for occupancy-related kernel metadata.
    pub enum FunctionSharedMemoryLimitConfig: sys::CUpti_FuncShmemLimitConfig {
        /// The shared memory limit config is default.
        Default = CUPTI_FUNC_SHMEM_LIMIT_DEFAULT,
        /// User has opted for a higher dynamic shared memory limit using function attribute ‘cudaFuncAttributeMaxDynamicSharedMemorySize’ for runtime API or `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` for driver API.
        OptIn = CUPTI_FUNC_SHMEM_LIMIT_OPTIN,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_FUNC_SHMEM_LIMIT_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Kind of named object referenced by activity records.
    pub enum ActivityObjectKind: sys::CUpti_ActivityObjectKind {
        /// The object kind is not known.
        Unknown = CUPTI_ACTIVITY_OBJECT_UNKNOWN,
        /// A process.
        Process = CUPTI_ACTIVITY_OBJECT_PROCESS,
        /// A thread.
        Thread = CUPTI_ACTIVITY_OBJECT_THREAD,
        /// A device.
        Device = CUPTI_ACTIVITY_OBJECT_DEVICE,
        /// A context.
        Context = CUPTI_ACTIVITY_OBJECT_CONTEXT,
        /// A stream.
        Stream = CUPTI_ACTIVITY_OBJECT_STREAM,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_OBJECT_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Kind of CUPTI overhead reported by [`crate::activity::ActivityOverhead`].
    pub enum ActivityOverheadKind: sys::CUpti_ActivityOverheadKind {
        /// The overhead kind is not known.
        Unknown = CUPTI_ACTIVITY_OVERHEAD_UNKNOWN,
        /// Compiler overhead.
        DriverCompiler = CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER,
        /// Activity buffer flush overhead.
        BufferFlush = CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH,
        /// CUPTI instrumentation overhead.
        Instrumentation = CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION,
        /// CUPTI resource creation and destruction overhead.
        Resource = CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE,
        /// CUDA Runtime triggered module loading overhead.
        RuntimeTriggeredModuleLoading = CUPTI_ACTIVITY_OVERHEAD_RUNTIME_TRIGGERED_MODULE_LOADING,
        /// Lazy function loading overhead.
        LazyFunctionLoading = CUPTI_ACTIVITY_OVERHEAD_LAZY_FUNCTION_LOADING,
        /// Overhead due to lack of command buffer space.
        ///
        /// See [`crate::activity::ActivityOverhead`] for the decoded overhead payload.
        CommandBufferFull = CUPTI_ACTIVITY_OVERHEAD_COMMAND_BUFFER_FULL,
        /// Overhead due to activity buffer request.
        ActivityBufferRequest = CUPTI_ACTIVITY_OVERHEAD_ACTIVITY_BUFFER_REQUEST,
        /// Overhead due to UVM activity initialization.
        UvmActivityInit = CUPTI_ACTIVITY_OVERHEAD_UVM_ACTIVITY_INIT,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_OVERHEAD_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// External correlation namespace used by CUPTI correlation records.
    pub enum ExternalCorrelationKind: sys::CUpti_ExternalCorrelationKind {
        /// Invalid value decoded from CUPTI.
        Invalid = CUPTI_EXTERNAL_CORRELATION_KIND_INVALID,
        /// The external API is unknown to CUPTI.
        Unknown = CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN,
        /// The external API is `OpenACC`.
        OpenAcc = CUPTI_EXTERNAL_CORRELATION_KIND_OPENACC,
        /// The external API is custom0.
        Custom0 = CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,
        /// The external API is custom1.
        Custom1 = CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1,
        /// The external API is custom2.
        Custom2 = CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2,
        /// Add new kinds before this line.
        Size = CUPTI_EXTERNAL_CORRELATION_KIND_SIZE,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_EXTERNAL_CORRELATION_KIND_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Kind of environment data carried by [`crate::activity::ActivityEnvironment`].
    pub enum ActivityEnvironmentKind: sys::CUpti_ActivityEnvironmentKind {
        /// Unknown data.
        Unknown = CUPTI_ACTIVITY_ENVIRONMENT_UNKNOWN,
        /// The environment data is related to speed.
        Speed = CUPTI_ACTIVITY_ENVIRONMENT_SPEED,
        /// The environment data is related to temperature.
        Temperature = CUPTI_ACTIVITY_ENVIRONMENT_TEMPERATURE,
        /// The environment data is related to power.
        Power = CUPTI_ACTIVITY_ENVIRONMENT_POWER,
        /// The environment data is related to cooling.
        Cooling = CUPTI_ACTIVITY_ENVIRONMENT_COOLING,
        /// Number of entries defined for this CUPTI enum.
        Count = CUPTI_ACTIVITY_ENVIRONMENT_COUNT,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_ENVIRONMENT_KIND_FORCE_INT,
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    /// Clock-throttle reason flags reported by CUPTI environment records.
    pub struct EnvironmentClocksThrottleReasons: u32 {
        /// No clock throttling.
        const NONE = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_NONE.0;
        /// Nothing is running on the GPU and the clocks are dropping to idle state.
        const GPU_IDLE = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_GPU_IDLE.0;
        /// The GPU clocks are limited by a user specified limit.
        const USER_DEFINED_CLOCKS = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_USER_DEFINED_CLOCKS.0;
        /// A software power scaling algorithm is reducing the clocks below requested clocks.
        const SOFTWARE_POWER_CAP = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_SW_POWER_CAP.0;
        /// Hardware slowdown to reduce the clock by a factor of two or more is engaged.
        ///
        /// This is an indicator of one of the following: 1) Temperature is too high, 2) External power brake assertion is being triggered (e.g. by the system power supply), 3) Change in power state.
        const HARDWARE_SLOWDOWN = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN.0;
        /// Some unspecified factor is reducing the clocks.
        const UNKNOWN = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_UNKNOWN.0;
        /// Throttle reason is not supported for this GPU.
        const UNSUPPORTED = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_UNSUPPORTED.0;
    }
}

impl Display for EnvironmentClocksThrottleReasons {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Preemption activity kind.
    pub enum ActivityPreemptionKind: sys::CUpti_ActivityPreemptionKind {
        /// The preemption kind is not known.
        Unknown = CUPTI_ACTIVITY_PREEMPTION_KIND_UNKNOWN,
        /// Preemption to save CDP block.
        Save = CUPTI_ACTIVITY_PREEMPTION_KIND_SAVE,
        /// Preemption to restore CDP block.
        Restore = CUPTI_ACTIVITY_PREEMPTION_KIND_RESTORE,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_PREEMPTION_KIND_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Stall reason reported by PC-sampling activity records.
    pub enum ActivityPcSamplingStallReason: sys::CUpti_ActivityPCSamplingStallReason {
        /// Invalid reason.
        Invalid = CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID,
        /// No stall, instruction is selected for issue.
        None = CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE,
        /// Warp is blocked because next instruction is not yet available, because of instruction cache miss, or because of branching effects.
        InstructionFetch = CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH,
        /// Instruction is waiting on an arithmetic dependency.
        ExecutionDependency = CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY,
        /// Warp is blocked because it is waiting for a memory access to complete.
        MemoryDependency = CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY,
        /// Texture sub-system is fully utilized or has too many outstanding requests.
        Texture = CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE,
        /// Warp is blocked as it is waiting at __`syncthreads()` or at memory barrier.
        Sync = CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC,
        /// Warp is blocked waiting for **constant** memory and immediate memory access to complete.
        ConstantMemoryDependency = CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY,
        /// Compute operation cannot be performed due to the required resources not being available.
        PipeBusy = CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY,
        /// Warp is blocked because there are too many pending memory operations.
        MemoryThrottle = CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE,
        /// Warp was ready to issue, but some other warp issued instead.
        NotSelected = CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED,
        /// Miscellaneous reasons.
        Other = CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER,
        /// Sleeping.
        Sleeping = CUPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_PC_SAMPLING_STALL_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Access type reported for unified-memory page-fault counters.
    pub enum ActivityUnifiedMemoryAccessType: sys::CUpti_ActivityUnifiedMemoryAccessType {
        /// The unified memory access type is not known.
        Unknown = CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_UNKNOWN,
        /// The page fault was triggered by read memory instruction.
        Read = CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_READ,
        /// The page fault was triggered by write memory instruction.
        Write = CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_WRITE,
        /// The page fault was triggered by atomic memory instruction.
        Atomic = CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_ATOMIC,
        /// The page fault was triggered by memory prefetch operation.
        Prefetch = CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_PREFETCH,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Migration cause reported for unified-memory transfer counters.
    pub enum ActivityUnifiedMemoryMigrationCause: sys::CUpti_ActivityUnifiedMemoryMigrationCause {
        /// The unified memory migration cause is not known.
        Unknown = CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_UNKNOWN,
        /// The unified memory migrated due to an explicit call from the user e.g.
        ///
        /// cudaMemPrefetchAsync.
        User = CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_USER,
        /// The unified memory migrated to guarantee data coherence e.g.
        ///
        /// CPU/GPU faults on Pascal+ and kernel launch on pre-Pascal GPUs.
        Coherence = CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_COHERENCE,
        /// The unified memory was speculatively migrated by the UVM driver before being accessed by the destination processor to improve performance.
        Prefetch = CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_PREFETCH,
        /// The unified memory migrated to the CPU because it was evicted to make room for another block of memory on the GPU.
        Eviction = CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_EVICTION,
        /// The unified memory migrated to another processor because of access counter notifications.
        ///
        /// Only frequently accessed pages are migrated between CPU and GPU, or between peer GPUs.
        AccessCounters = CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_ACCESS_COUNTERS,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Remote-map cause reported for unified-memory counters.
    pub enum ActivityUnifiedMemoryRemoteMapCause: sys::CUpti_ActivityUnifiedMemoryRemoteMapCause {
        /// The cause of mapping to remote memory was unknown.
        Unknown = CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_UNKNOWN,
        /// Mapping to remote memory was added to maintain data coherence.
        Coherence = CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_COHERENCE,
        /// Mapping to remote memory was added to prevent further thrashing.
        Thrashing = CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_THRASHING,
        /// Mapping to remote memory was added to enforce the hints specified by the programmer or by performance heuristics of the UVM driver.
        Policy = CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_POLICY,
        /// Mapping to remote memory was added because there is no more memory available on the processor and eviction was not possible.
        OutOfMemory = CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_OUT_OF_MEMORY,
        /// Mapping to remote memory was added after the memory was evicted to make room for another block of memory on the GPU.
        Eviction = CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_EVICTION,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// SASS instruction class reported by instruction activity records.
    pub enum ActivityInstructionClass: sys::CUpti_ActivityInstructionClass {
        /// The instruction class is not known.
        Unknown = CUPTI_ACTIVITY_INSTRUCTION_CLASS_UNKNOWN,
        /// Represents a 32 bit floating point operation.
        Float32 = CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_32,
        /// Represents a 64 bit floating point operation.
        Float64 = CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_64,
        /// Represents an integer operation.
        Integer = CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTEGER,
        /// Represents a bit conversion operation.
        BitConversion = CUPTI_ACTIVITY_INSTRUCTION_CLASS_BIT_CONVERSION,
        /// Represents a control flow instruction.
        ControlFlow = CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONTROL_FLOW,
        /// Represents a global load-store instruction.
        Global = CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL,
        /// Represents a shared load-store instruction.
        Shared = CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED,
        /// Represents a local load-store instruction.
        Local = CUPTI_ACTIVITY_INSTRUCTION_CLASS_LOCAL,
        /// Represents a generic load-store instruction.
        Generic = CUPTI_ACTIVITY_INSTRUCTION_CLASS_GENERIC,
        /// Represents a surface load-store instruction.
        Surface = CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE,
        /// Represents a constant load instruction.
        Constant = CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONSTANT,
        /// Represents a texture load-store instruction.
        Texture = CUPTI_ACTIVITY_INSTRUCTION_CLASS_TEXTURE,
        /// Represents a global atomic instruction.
        GlobalAtomic = CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL_ATOMIC,
        /// Represents a shared atomic instruction.
        SharedAtomic = CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED_ATOMIC,
        /// Represents a surface atomic instruction.
        SurfaceAtomic = CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE_ATOMIC,
        /// Represents a inter-thread communication instruction.
        InterThreadCommunication = CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTER_THREAD_COMMUNICATION,
        /// Represents a barrier instruction.
        Barrier = CUPTI_ACTIVITY_INSTRUCTION_CLASS_BARRIER,
        /// Represents some miscellaneous instructions which do not fit in the above classification.
        Miscellaneous = CUPTI_ACTIVITY_INSTRUCTION_CLASS_MISCELLANEOUS,
        /// Represents a 16 bit floating point operation.
        Float16 = CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_16,
        /// Represents uniform instruction.
        Uniform = CUPTI_ACTIVITY_INSTRUCTION_CLASS_UNIFORM,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_INSTRUCTION_CLASS_KIND_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Confidential-compute key-rotation event type.
    pub enum ConfidentialComputeRotationEventType: sys::CUpti_ConfidentialComputeRotationEventType {
        /// Invalid value decoded from CUPTI.
        Invalid = CUPTI_CONFIDENTIAL_COMPUTE_INVALID_ROTATION_EVENT,
        /// This channel has been blocked from accepting new CUDA work so a key rotation can be done.
        ChannelBlocked = CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_BLOCKED,
        /// This channel remains blocked and all queued CUDA work has completed.
        ///
        /// Other clients or channels may cause delays in starting the key rotation.
        ChannelDrained = CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_DRAINED,
        /// Key rotations have completed and this channel is unblocked.
        ChannelUnblocked = CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_UNBLOCKED,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_CONFIDENTIAL_COMPUTE_EVENT_TYPE_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// `OpenACC` event kind reported by `OpenACC` activity records.
    pub enum OpenAccEventKind: sys::CUpti_OpenAccEventKind {
        /// Invalid value decoded from CUPTI.
        Invalid = CUPTI_OPENACC_EVENT_KIND_INVALID,
        /// Device Init value decoded from CUPTI.
        DeviceInit = CUPTI_OPENACC_EVENT_KIND_DEVICE_INIT,
        /// Device Shutdown value decoded from CUPTI.
        DeviceShutdown = CUPTI_OPENACC_EVENT_KIND_DEVICE_SHUTDOWN,
        /// Runtime Shutdown value decoded from CUPTI.
        RuntimeShutdown = CUPTI_OPENACC_EVENT_KIND_RUNTIME_SHUTDOWN,
        /// Enqueue Launch value decoded from CUPTI.
        EnqueueLaunch = CUPTI_OPENACC_EVENT_KIND_ENQUEUE_LAUNCH,
        /// Enqueue Upload value decoded from CUPTI.
        EnqueueUpload = CUPTI_OPENACC_EVENT_KIND_ENQUEUE_UPLOAD,
        /// Enqueue Download value decoded from CUPTI.
        EnqueueDownload = CUPTI_OPENACC_EVENT_KIND_ENQUEUE_DOWNLOAD,
        /// Wait value decoded from CUPTI.
        Wait = CUPTI_OPENACC_EVENT_KIND_WAIT,
        /// Implicit Wait value decoded from CUPTI.
        ImplicitWait = CUPTI_OPENACC_EVENT_KIND_IMPLICIT_WAIT,
        /// Compute Construct value decoded from CUPTI.
        ComputeConstruct = CUPTI_OPENACC_EVENT_KIND_COMPUTE_CONSTRUCT,
        /// Update value decoded from CUPTI.
        Update = CUPTI_OPENACC_EVENT_KIND_UPDATE,
        /// Enter Data value decoded from CUPTI.
        EnterData = CUPTI_OPENACC_EVENT_KIND_ENTER_DATA,
        /// Exit Data value decoded from CUPTI.
        ExitData = CUPTI_OPENACC_EVENT_KIND_EXIT_DATA,
        /// Create value decoded from CUPTI.
        Create = CUPTI_OPENACC_EVENT_KIND_CREATE,
        /// Delete value decoded from CUPTI.
        Delete = CUPTI_OPENACC_EVENT_KIND_DELETE,
        /// Alloc value decoded from CUPTI.
        Alloc = CUPTI_OPENACC_EVENT_KIND_ALLOC,
        /// Free value decoded from CUPTI.
        Free = CUPTI_OPENACC_EVENT_KIND_FREE,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_OPENACC_EVENT_KIND_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// `OpenACC` construct kind reported by `OpenACC` activity records.
    pub enum OpenAccConstructKind: sys::CUpti_OpenAccConstructKind {
        /// Unknown value decoded from CUPTI.
        Unknown = CUPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN,
        /// Parallel value decoded from CUPTI.
        Parallel = CUPTI_OPENACC_CONSTRUCT_KIND_PARALLEL,
        /// Kernels value decoded from CUPTI.
        Kernels = CUPTI_OPENACC_CONSTRUCT_KIND_KERNELS,
        /// Loop value decoded from CUPTI.
        Loop = CUPTI_OPENACC_CONSTRUCT_KIND_LOOP,
        /// Data value decoded from CUPTI.
        Data = CUPTI_OPENACC_CONSTRUCT_KIND_DATA,
        /// Enter Data value decoded from CUPTI.
        EnterData = CUPTI_OPENACC_CONSTRUCT_KIND_ENTER_DATA,
        /// Exit Data value decoded from CUPTI.
        ExitData = CUPTI_OPENACC_CONSTRUCT_KIND_EXIT_DATA,
        /// Host Data value decoded from CUPTI.
        HostData = CUPTI_OPENACC_CONSTRUCT_KIND_HOST_DATA,
        /// Atomic value decoded from CUPTI.
        Atomic = CUPTI_OPENACC_CONSTRUCT_KIND_ATOMIC,
        /// Declare value decoded from CUPTI.
        Declare = CUPTI_OPENACC_CONSTRUCT_KIND_DECLARE,
        /// Init value decoded from CUPTI.
        Init = CUPTI_OPENACC_CONSTRUCT_KIND_INIT,
        /// Shutdown value decoded from CUPTI.
        Shutdown = CUPTI_OPENACC_CONSTRUCT_KIND_SHUTDOWN,
        /// Set value decoded from CUPTI.
        Set = CUPTI_OPENACC_CONSTRUCT_KIND_SET,
        /// Update value decoded from CUPTI.
        Update = CUPTI_OPENACC_CONSTRUCT_KIND_UPDATE,
        /// Routine value decoded from CUPTI.
        Routine = CUPTI_OPENACC_CONSTRUCT_KIND_ROUTINE,
        /// Wait value decoded from CUPTI.
        Wait = CUPTI_OPENACC_CONSTRUCT_KIND_WAIT,
        /// Runtime Api value decoded from CUPTI.
        RuntimeApi = CUPTI_OPENACC_CONSTRUCT_KIND_RUNTIME_API,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_OPENACC_CONSTRUCT_KIND_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// OpenMP event kind reported by OpenMP activity records.
    pub enum OpenMultiProcessingEventKind: sys::CUpti_OpenMpEventKind {
        /// Invalid value decoded from CUPTI.
        Invalid = CUPTI_OPENMP_EVENT_KIND_INVALID,
        /// Parallel value decoded from CUPTI.
        Parallel = CUPTI_OPENMP_EVENT_KIND_PARALLEL,
        /// Task value decoded from CUPTI.
        Task = CUPTI_OPENMP_EVENT_KIND_TASK,
        /// Thread value decoded from CUPTI.
        Thread = CUPTI_OPENMP_EVENT_KIND_THREAD,
        /// Idle utilization level.
        Idle = CUPTI_OPENMP_EVENT_KIND_IDLE,
        /// Wait Barrier value decoded from CUPTI.
        WaitBarrier = CUPTI_OPENMP_EVENT_KIND_WAIT_BARRIER,
        /// Wait Taskwait value decoded from CUPTI.
        WaitTaskwait = CUPTI_OPENMP_EVENT_KIND_WAIT_TASKWAIT,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_OPENMP_EVENT_KIND_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// JIT input/output entry type reported by [`crate::activity::ActivityJit`].
    pub enum ActivityJitEntryType: sys::CUpti_ActivityJitEntryType {
        /// Invalid value decoded from CUPTI.
        Invalid = CUPTI_ACTIVITY_JIT_ENTRY_INVALID,
        /// PTX to CUBIN.
        ParallelThreadExecutionToCudaBinary = CUPTI_ACTIVITY_JIT_ENTRY_PTX_TO_CUBIN,
        /// NVVM-IR to PTX.
        NvidiaVirtualMachineIntermediateRepresentationToParallelThreadExecution = CUPTI_ACTIVITY_JIT_ENTRY_NVVM_IR_TO_PTX,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_JIT_ENTRY_TYPE_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// JIT operation type reported by [`crate::activity::ActivityJit`].
    pub enum ActivityJitOperationType: sys::CUpti_ActivityJitOperationType {
        /// Invalid value decoded from CUPTI.
        Invalid = CUPTI_ACTIVITY_JIT_OPERATION_INVALID,
        /// Loaded from the compute cache.
        CacheLoad = CUPTI_ACTIVITY_JIT_OPERATION_CACHE_LOAD,
        /// Stored in the compute cache.
        CacheStore = CUPTI_ACTIVITY_JIT_OPERATION_CACHE_STORE,
        /// JIT compilation.
        Compile = CUPTI_ACTIVITY_JIT_OPERATION_COMPILE,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_JIT_OPERATION_TYPE_FORCE_INT,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Automatic boost state reported for a CUDA context.
pub struct ActivityAutoBoostState {
    /// Whether automatic boost is enabled.
    pub enabled: bool,
    /// Process that requested the automatic boost state.
    pub process_id: ProcessId,
}

impl From<sys::CUpti_ActivityAutoBoostState> for ActivityAutoBoostState {
    fn from(value: sys::CUpti_ActivityAutoBoostState) -> Self {
        Self {
            enabled: value.enabled != 0,
            process_id: ProcessId::from(value.pid),
        }
    }
}

impl_enum_display!(ChannelType, {
    Self::Invalid => "CUPTI_CHANNEL_TYPE_INVALID",
    Self::Compute => "CUPTI_CHANNEL_TYPE_COMPUTE",
    Self::AsynchronousMemoryCopy => "CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY",
    Self::Decompress => "CUPTI_CHANNEL_TYPE_DECOMP",
    Self::ForceInt => "CUPTI_CHANNEL_TYPE_FORCE_INT",
});

impl_enum_display!(ActivityMemoryCopyKind, {
    Self::Unknown => "CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN",
    Self::HostToDevice => "CUPTI_ACTIVITY_MEMCPY_KIND_HTOD",
    Self::DeviceToHost => "CUPTI_ACTIVITY_MEMCPY_KIND_DTOH",
    Self::HostToArray => "CUPTI_ACTIVITY_MEMCPY_KIND_HTOA",
    Self::ArrayToHost => "CUPTI_ACTIVITY_MEMCPY_KIND_ATOH",
    Self::ArrayToArray => "CUPTI_ACTIVITY_MEMCPY_KIND_ATOA",
    Self::ArrayToDevice => "CUPTI_ACTIVITY_MEMCPY_KIND_ATOD",
    Self::DeviceToArray => "CUPTI_ACTIVITY_MEMCPY_KIND_DTOA",
    Self::DeviceToDevice => "CUPTI_ACTIVITY_MEMCPY_KIND_DTOD",
    Self::HostToHost => "CUPTI_ACTIVITY_MEMCPY_KIND_HTOH",
    Self::PeerToPeer => "CUPTI_ACTIVITY_MEMCPY_KIND_PTOP",
    Self::ForceInt => "CUPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT",
});

impl_enum_display!(ActivityMemoryKind, {
    Self::Unknown => "CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN",
    Self::Pageable => "CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE",
    Self::Pinned => "CUPTI_ACTIVITY_MEMORY_KIND_PINNED",
    Self::Device => "CUPTI_ACTIVITY_MEMORY_KIND_DEVICE",
    Self::Array => "CUPTI_ACTIVITY_MEMORY_KIND_ARRAY",
    Self::Managed => "CUPTI_ACTIVITY_MEMORY_KIND_MANAGED",
    Self::DeviceStatic => "CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC",
    Self::ManagedStatic => "CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC",
    Self::ForceInt => "CUPTI_ACTIVITY_MEMORY_KIND_FORCE_INT",
});

impl_enum_display!(ActivityMemoryOperationType, {
    Self::Invalid => "CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_INVALID",
    Self::Allocation => "CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION",
    Self::Release => "CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE",
    Self::ForceInt => "CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_FORCE_INT",
});

impl_enum_display!(ActivityMemoryPoolType, {
    Self::Invalid => "CUPTI_ACTIVITY_MEMORY_POOL_TYPE_INVALID",
    Self::Local => "CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL",
    Self::Imported => "CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED",
    Self::ForceInt => "CUPTI_ACTIVITY_MEMORY_POOL_TYPE_FORCE_INT",
});

impl_enum_display!(ActivityMemoryPoolOperationType, {
    Self::Invalid => "CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_INVALID",
    Self::Created => "CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_CREATED",
    Self::Destroyed => "CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_DESTROYED",
    Self::Trimmed => "CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED",
    Self::ForceInt => "CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_FORCE_INT",
});

impl_enum_display!(DeviceGraphLaunchMode, {
    Self::Invalid => "CUPTI_DEVICE_GRAPH_LAUNCH_MODE_INVALID",
    Self::FireAndForget => "CUPTI_DEVICE_GRAPH_LAUNCH_MODE_FIRE_AND_FORGET",
    Self::Tail => "CUPTI_DEVICE_GRAPH_LAUNCH_MODE_TAIL",
    Self::FireAndForgetAsSibling => "CUPTI_DEVICE_GRAPH_LAUNCH_MODE_FIRE_AND_FORGET_AS_SIBLING",
});

impl_enum_display!(ActivityComputeApiKind, {
    Self::Unknown => "CUPTI_ACTIVITY_COMPUTE_API_UNKNOWN",
    Self::Cuda => "CUPTI_ACTIVITY_COMPUTE_API_CUDA",
    Self::CudaMultiProcessService => "CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS",
    Self::ForceInt => "CUPTI_ACTIVITY_COMPUTE_API_FORCE_INT",
});

impl_enum_display!(ContextCigMode, {
    Self::None => "CUPTI_CONTEXT_CIG_MODE_NONE",
    Self::Cig => "CUPTI_CONTEXT_CIG_MODE_CIG",
    Self::CigFallback => "CUPTI_CONTEXT_CIG_MODE_CIG_FALLBACK",
    Self::ForceInt => "CUPTI_CONTEXT_CIG_MODE_FORCE_INT",
});

impl_enum_display!(ComputeEngineContextSwitchOperationType, {
    Self::Invalid => "CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_INVALID",
    Self::Start => "CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_START",
    Self::End => "CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_END",
    Self::Count => "CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_COUNT",
});

impl_enum_display!(ActivitySynchronizationType, {
    Self::Unknown => "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN",
    Self::EventSynchronize => "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE",
    Self::StreamWaitEvent => "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT",
    Self::StreamSynchronize => "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE",
    Self::ContextSynchronize => "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE",
    Self::ForceInt => "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_FORCE_INT",
});

impl_enum_display!(ActivityStreamFlag, {
    Self::Unknown => "CUPTI_ACTIVITY_STREAM_CREATE_FLAG_UNKNOWN",
    Self::Default => "CUPTI_ACTIVITY_STREAM_CREATE_FLAG_DEFAULT",
    Self::NonBlocking => "CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING",
    Self::Null => "CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NULL",
    Self::Mask => "CUPTI_ACTIVITY_STREAM_CREATE_MASK",
    Self::ForceInt => "CUPTI_ACTIVITY_STREAM_CREATE_FLAG_FORCE_INT",
});

impl_enum_display!(ActivityPartitionedGlobalCacheConfig, {
    Self::Unknown => "CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_UNKNOWN",
    Self::NotSupported => "CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED",
    Self::Off => "CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF",
    Self::On => "CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON",
    Self::ForceInt => "CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_FORCE_INT",
});

impl_enum_display!(ActivityLaunchType, {
    Self::Regular => "CUPTI_ACTIVITY_LAUNCH_TYPE_REGULAR",
    Self::CooperativeSingleDevice => "CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_SINGLE_DEVICE",
    Self::CooperativeMultiDevice => "CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_MULTI_DEVICE",
    Self::CblCommandlist => "CUPTI_ACTIVITY_LAUNCH_TYPE_CBL_COMMANDLIST",
});

impl_enum_display!(FunctionSharedMemoryLimitConfig, {
    Self::Default => "CUPTI_FUNC_SHMEM_LIMIT_DEFAULT",
    Self::OptIn => "CUPTI_FUNC_SHMEM_LIMIT_OPTIN",
    Self::ForceInt => "CUPTI_FUNC_SHMEM_LIMIT_FORCE_INT",
});

impl_enum_display!(ActivityObjectKind, {
    Self::Unknown => "CUPTI_ACTIVITY_OBJECT_UNKNOWN",
    Self::Process => "CUPTI_ACTIVITY_OBJECT_PROCESS",
    Self::Thread => "CUPTI_ACTIVITY_OBJECT_THREAD",
    Self::Device => "CUPTI_ACTIVITY_OBJECT_DEVICE",
    Self::Context => "CUPTI_ACTIVITY_OBJECT_CONTEXT",
    Self::Stream => "CUPTI_ACTIVITY_OBJECT_STREAM",
    Self::ForceInt => "CUPTI_ACTIVITY_OBJECT_FORCE_INT",
});

impl_enum_display!(ActivityOverheadKind, {
    Self::Unknown => "CUPTI_ACTIVITY_OVERHEAD_UNKNOWN",
    Self::DriverCompiler => "CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER",
    Self::BufferFlush => "CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH",
    Self::Instrumentation => "CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION",
    Self::Resource => "CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE",
    Self::RuntimeTriggeredModuleLoading => "CUPTI_ACTIVITY_OVERHEAD_RUNTIME_TRIGGERED_MODULE_LOADING",
    Self::LazyFunctionLoading => "CUPTI_ACTIVITY_OVERHEAD_LAZY_FUNCTION_LOADING",
    Self::CommandBufferFull => "CUPTI_ACTIVITY_OVERHEAD_COMMAND_BUFFER_FULL",
    Self::ActivityBufferRequest => "CUPTI_ACTIVITY_OVERHEAD_ACTIVITY_BUFFER_REQUEST",
    Self::UvmActivityInit => "CUPTI_ACTIVITY_OVERHEAD_UVM_ACTIVITY_INIT",
    Self::ForceInt => "CUPTI_ACTIVITY_OVERHEAD_FORCE_INT",
});

impl_enum_display!(ExternalCorrelationKind, {
    Self::Invalid => "CUPTI_EXTERNAL_CORRELATION_KIND_INVALID",
    Self::Unknown => "CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN",
    Self::OpenAcc => "CUPTI_EXTERNAL_CORRELATION_KIND_OPENACC",
    Self::Custom0 => "CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0",
    Self::Custom1 => "CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1",
    Self::Custom2 => "CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2",
    Self::Size => "CUPTI_EXTERNAL_CORRELATION_KIND_SIZE",
    Self::ForceInt => "CUPTI_EXTERNAL_CORRELATION_KIND_FORCE_INT",
});

impl_enum_display!(ActivityEnvironmentKind, {
    Self::Unknown => "CUPTI_ACTIVITY_ENVIRONMENT_UNKNOWN",
    Self::Speed => "CUPTI_ACTIVITY_ENVIRONMENT_SPEED",
    Self::Temperature => "CUPTI_ACTIVITY_ENVIRONMENT_TEMPERATURE",
    Self::Power => "CUPTI_ACTIVITY_ENVIRONMENT_POWER",
    Self::Cooling => "CUPTI_ACTIVITY_ENVIRONMENT_COOLING",
    Self::Count => "CUPTI_ACTIVITY_ENVIRONMENT_COUNT",
    Self::ForceInt => "CUPTI_ACTIVITY_ENVIRONMENT_KIND_FORCE_INT",
});

impl_enum_display!(ActivityPreemptionKind, {
    Self::Unknown => "CUPTI_ACTIVITY_PREEMPTION_KIND_UNKNOWN",
    Self::Save => "CUPTI_ACTIVITY_PREEMPTION_KIND_SAVE",
    Self::Restore => "CUPTI_ACTIVITY_PREEMPTION_KIND_RESTORE",
    Self::ForceInt => "CUPTI_ACTIVITY_PREEMPTION_KIND_FORCE_INT",
});

impl_enum_display!(ActivityPcSamplingStallReason, {
    Self::Invalid => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID",
    Self::None => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE",
    Self::InstructionFetch => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH",
    Self::ExecutionDependency => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY",
    Self::MemoryDependency => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY",
    Self::Texture => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE",
    Self::Sync => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC",
    Self::ConstantMemoryDependency => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY",
    Self::PipeBusy => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY",
    Self::MemoryThrottle => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE",
    Self::NotSelected => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED",
    Self::Other => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER",
    Self::Sleeping => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING",
    Self::ForceInt => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_FORCE_INT",
});

impl_enum_display!(ActivityUnifiedMemoryAccessType, {
    Self::Unknown => "CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_UNKNOWN",
    Self::Read => "CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_READ",
    Self::Write => "CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_WRITE",
    Self::Atomic => "CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_ATOMIC",
    Self::Prefetch => "CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_PREFETCH",
});

impl_enum_display!(ActivityUnifiedMemoryMigrationCause, {
    Self::Unknown => "CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_UNKNOWN",
    Self::User => "CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_USER",
    Self::Coherence => "CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_COHERENCE",
    Self::Prefetch => "CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_PREFETCH",
    Self::Eviction => "CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_EVICTION",
    Self::AccessCounters => "CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_ACCESS_COUNTERS",
});

impl_enum_display!(ActivityUnifiedMemoryRemoteMapCause, {
    Self::Unknown => "CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_UNKNOWN",
    Self::Coherence => "CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_COHERENCE",
    Self::Thrashing => "CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_THRASHING",
    Self::Policy => "CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_POLICY",
    Self::OutOfMemory => "CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_OUT_OF_MEMORY",
    Self::Eviction => "CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_EVICTION",
});

impl_enum_display!(ActivityInstructionClass, {
    Self::Unknown => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_UNKNOWN",
    Self::Float32 => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_32",
    Self::Float64 => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_64",
    Self::Integer => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTEGER",
    Self::BitConversion => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_BIT_CONVERSION",
    Self::ControlFlow => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONTROL_FLOW",
    Self::Global => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL",
    Self::Shared => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED",
    Self::Local => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_LOCAL",
    Self::Generic => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_GENERIC",
    Self::Surface => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE",
    Self::Constant => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONSTANT",
    Self::Texture => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_TEXTURE",
    Self::GlobalAtomic => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL_ATOMIC",
    Self::SharedAtomic => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED_ATOMIC",
    Self::SurfaceAtomic => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE_ATOMIC",
    Self::InterThreadCommunication => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTER_THREAD_COMMUNICATION",
    Self::Barrier => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_BARRIER",
    Self::Miscellaneous => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_MISCELLANEOUS",
    Self::Float16 => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_16",
    Self::Uniform => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_UNIFORM",
    Self::ForceInt => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_KIND_FORCE_INT",
});

impl_enum_display!(ConfidentialComputeRotationEventType, {
    Self::Invalid => "CUPTI_CONFIDENTIAL_COMPUTE_INVALID_ROTATION_EVENT",
    Self::ChannelBlocked => "CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_BLOCKED",
    Self::ChannelDrained => "CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_DRAINED",
    Self::ChannelUnblocked => "CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_UNBLOCKED",
    Self::ForceInt => "CUPTI_CONFIDENTIAL_COMPUTE_EVENT_TYPE_FORCE_INT",
});

impl_enum_display!(OpenAccEventKind, {
    Self::Invalid => "CUPTI_OPENACC_EVENT_KIND_INVALID",
    Self::DeviceInit => "CUPTI_OPENACC_EVENT_KIND_DEVICE_INIT",
    Self::DeviceShutdown => "CUPTI_OPENACC_EVENT_KIND_DEVICE_SHUTDOWN",
    Self::RuntimeShutdown => "CUPTI_OPENACC_EVENT_KIND_RUNTIME_SHUTDOWN",
    Self::EnqueueLaunch => "CUPTI_OPENACC_EVENT_KIND_ENQUEUE_LAUNCH",
    Self::EnqueueUpload => "CUPTI_OPENACC_EVENT_KIND_ENQUEUE_UPLOAD",
    Self::EnqueueDownload => "CUPTI_OPENACC_EVENT_KIND_ENQUEUE_DOWNLOAD",
    Self::Wait => "CUPTI_OPENACC_EVENT_KIND_WAIT",
    Self::ImplicitWait => "CUPTI_OPENACC_EVENT_KIND_IMPLICIT_WAIT",
    Self::ComputeConstruct => "CUPTI_OPENACC_EVENT_KIND_COMPUTE_CONSTRUCT",
    Self::Update => "CUPTI_OPENACC_EVENT_KIND_UPDATE",
    Self::EnterData => "CUPTI_OPENACC_EVENT_KIND_ENTER_DATA",
    Self::ExitData => "CUPTI_OPENACC_EVENT_KIND_EXIT_DATA",
    Self::Create => "CUPTI_OPENACC_EVENT_KIND_CREATE",
    Self::Delete => "CUPTI_OPENACC_EVENT_KIND_DELETE",
    Self::Alloc => "CUPTI_OPENACC_EVENT_KIND_ALLOC",
    Self::Free => "CUPTI_OPENACC_EVENT_KIND_FREE",
    Self::ForceInt => "CUPTI_OPENACC_EVENT_KIND_FORCE_INT",
});

impl_enum_display!(OpenAccConstructKind, {
    Self::Unknown => "CUPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN",
    Self::Parallel => "CUPTI_OPENACC_CONSTRUCT_KIND_PARALLEL",
    Self::Kernels => "CUPTI_OPENACC_CONSTRUCT_KIND_KERNELS",
    Self::Loop => "CUPTI_OPENACC_CONSTRUCT_KIND_LOOP",
    Self::Data => "CUPTI_OPENACC_CONSTRUCT_KIND_DATA",
    Self::EnterData => "CUPTI_OPENACC_CONSTRUCT_KIND_ENTER_DATA",
    Self::ExitData => "CUPTI_OPENACC_CONSTRUCT_KIND_EXIT_DATA",
    Self::HostData => "CUPTI_OPENACC_CONSTRUCT_KIND_HOST_DATA",
    Self::Atomic => "CUPTI_OPENACC_CONSTRUCT_KIND_ATOMIC",
    Self::Declare => "CUPTI_OPENACC_CONSTRUCT_KIND_DECLARE",
    Self::Init => "CUPTI_OPENACC_CONSTRUCT_KIND_INIT",
    Self::Shutdown => "CUPTI_OPENACC_CONSTRUCT_KIND_SHUTDOWN",
    Self::Set => "CUPTI_OPENACC_CONSTRUCT_KIND_SET",
    Self::Update => "CUPTI_OPENACC_CONSTRUCT_KIND_UPDATE",
    Self::Routine => "CUPTI_OPENACC_CONSTRUCT_KIND_ROUTINE",
    Self::Wait => "CUPTI_OPENACC_CONSTRUCT_KIND_WAIT",
    Self::RuntimeApi => "CUPTI_OPENACC_CONSTRUCT_KIND_RUNTIME_API",
    Self::ForceInt => "CUPTI_OPENACC_CONSTRUCT_KIND_FORCE_INT",
});

impl_enum_display!(OpenMultiProcessingEventKind, {
    Self::Invalid => "CUPTI_OPENMP_EVENT_KIND_INVALID",
    Self::Parallel => "CUPTI_OPENMP_EVENT_KIND_PARALLEL",
    Self::Task => "CUPTI_OPENMP_EVENT_KIND_TASK",
    Self::Thread => "CUPTI_OPENMP_EVENT_KIND_THREAD",
    Self::Idle => "CUPTI_OPENMP_EVENT_KIND_IDLE",
    Self::WaitBarrier => "CUPTI_OPENMP_EVENT_KIND_WAIT_BARRIER",
    Self::WaitTaskwait => "CUPTI_OPENMP_EVENT_KIND_WAIT_TASKWAIT",
    Self::ForceInt => "CUPTI_OPENMP_EVENT_KIND_FORCE_INT",
});

impl_enum_display!(ActivityJitEntryType, {
    Self::Invalid => "CUPTI_ACTIVITY_JIT_ENTRY_INVALID",
    Self::ParallelThreadExecutionToCudaBinary => "CUPTI_ACTIVITY_JIT_ENTRY_PTX_TO_CUBIN",
    Self::NvidiaVirtualMachineIntermediateRepresentationToParallelThreadExecution => "CUPTI_ACTIVITY_JIT_ENTRY_NVVM_IR_TO_PTX",
    Self::ForceInt => "CUPTI_ACTIVITY_JIT_ENTRY_TYPE_FORCE_INT",
});

impl_enum_display!(ActivityJitOperationType, {
    Self::Invalid => "CUPTI_ACTIVITY_JIT_OPERATION_INVALID",
    Self::CacheLoad => "CUPTI_ACTIVITY_JIT_OPERATION_CACHE_LOAD",
    Self::CacheStore => "CUPTI_ACTIVITY_JIT_OPERATION_CACHE_STORE",
    Self::Compile => "CUPTI_ACTIVITY_JIT_OPERATION_COMPILE",
    Self::ForceInt => "CUPTI_ACTIVITY_JIT_OPERATION_TYPE_FORCE_INT",
});

/// The execution model of a kernel function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionExecutionModel {
    /// The execution model is unknown.
    Unknown,
    /// Single instruction, multiple threads execution.
    Simt,
    /// Tile execution.
    Tile,
    /// Hybrid execution.
    Hybrid,
}

impl From<u32> for FunctionExecutionModel {
    fn from(value: u32) -> Self {
        match value {
            value if value == sys::CUpti_FuncExecutionModel::CUPTI_FUNC_EXECUTION_MODEL_SIMT.0 => {
                Self::Simt
            }
            value if value == sys::CUpti_FuncExecutionModel::CUPTI_FUNC_EXECUTION_MODEL_TILE.0 => {
                Self::Tile
            }
            value
                if value == sys::CUpti_FuncExecutionModel::CUPTI_FUNC_EXECUTION_MODEL_HYBRID.0 =>
            {
                Self::Hybrid
            }
            _ => Self::Unknown,
        }
    }
}
