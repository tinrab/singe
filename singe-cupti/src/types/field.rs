use std::fmt::{self, Display, Formatter};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cupti_sys as sys;

macro_rules! activity_field_enum {
    ($name:ident, $raw:path, { $($(#[$variant_attr:meta])* $variant:ident = $value:ident,)+ }) => {
        singe_core::native_enum! {
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
            #[repr(u32)]
            #[non_exhaustive]
            /// CUPTI activity field identifier enum.
            #[allow(clippy::doc_markdown, reason = "native field tables preserve upstream C declarations in their documentation")]
            pub enum $name: $raw {
                $(
                    #[doc = concat!("CUPTI activity field `", stringify!($value), "`.")]
                    $(#[$variant_attr])*
                    $variant = $value,
                )+
            }
        }

        impl From<$name> for ActivityFieldId {
            fn from(value: $name) -> Self {
                let raw: u32 = value.into();
                ActivityFieldId(raw.cast_signed())
            }
        }

        impl_enum_display!($name, {
            $(Self::$variant => stringify!($value),)+
        });
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Raw activity field identifier used in activity field selection.
pub struct ActivityFieldId(i32);

impl ActivityFieldId {
    /// Returns the raw value decoded from CUPTI.
    #[must_use]
    pub const fn get(self) -> i32 {
        self.0
    }

    pub(crate) const fn as_raw(self) -> i32 {
        self.0
    }
}

impl Display for ActivityFieldId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

activity_field_enum!(ActivityApiField, sys::CUpti_ActivityApiFieldIds, {
    Kind = API_FIELD_KIND,
    CallbackId = API_FIELD_CBID,
    Start = API_FIELD_START,
    End = API_FIELD_END,
    ProcessId = API_FIELD_PROCESS_ID,
    ThreadId = API_FIELD_THREAD_ID,
    CorrelationId = API_FIELD_CORRELATION_ID,
    ReturnValue = API_FIELD_RETURN_VALUE,
});

activity_field_enum!(ActivityDeviceField, sys::CUpti_ActivityDeviceFieldIds, {
    Kind = DEVICE_FIELD_KIND,
    Flags = DEVICE_FIELD_FLAGS,
    GlobalMemoryBandwidth = DEVICE_FIELD_GLOBAL_MEMORY_BANDWIDTH,
    GlobalMemorySize = DEVICE_FIELD_GLOBAL_MEMORY_SIZE,
    ConstantMemorySize = DEVICE_FIELD_CONSTANT_MEMORY_SIZE,
    L2CacheSize = DEVICE_FIELD_L2_CACHE_SIZE,
    NumThreadsPerWarp = DEVICE_FIELD_NUM_THREADS_PER_WARP,
    CoreClockRate = DEVICE_FIELD_CORE_CLOCK_RATE,
    NumMemoryCopyEngines = DEVICE_FIELD_NUM_MEMCPY_ENGINES,
    NumMultiprocessors = DEVICE_FIELD_NUM_MULTIPROCESSORS,
    MaxIpc = DEVICE_FIELD_MAX_IPC,
    MaxWarpsPerMultiprocessor = DEVICE_FIELD_MAX_WARPS_PER_MULTIPROCESSOR,
    MaxBlocksPerMultiprocessor = DEVICE_FIELD_MAX_BLOCKS_PER_MULTIPROCESSOR,
    MaxSharedMemoryPerMultiprocessor = DEVICE_FIELD_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
    MaxRegistersPerMultiprocessor = DEVICE_FIELD_MAX_REGISTERS_PER_MULTIPROCESSOR,
    MaxRegistersPerBlock = DEVICE_FIELD_MAX_REGISTERS_PER_BLOCK,
    MaxSharedMemoryPerBlock = DEVICE_FIELD_MAX_SHARED_MEMORY_PER_BLOCK,
    MaxThreadsPerBlock = DEVICE_FIELD_MAX_THREADS_PER_BLOCK,
    MaxBlockDimX = DEVICE_FIELD_MAX_BLOCK_DIM_X,
    MaxBlockDimY = DEVICE_FIELD_MAX_BLOCK_DIM_Y,
    MaxBlockDimZ = DEVICE_FIELD_MAX_BLOCK_DIM_Z,
    MaxGridDimX = DEVICE_FIELD_MAX_GRID_DIM_X,
    MaxGridDimY = DEVICE_FIELD_MAX_GRID_DIM_Y,
    MaxGridDimZ = DEVICE_FIELD_MAX_GRID_DIM_Z,
    ComputeCapabilityMajor = DEVICE_FIELD_COMPUTE_CAPABILITY_MAJOR,
    ComputeCapabilityMinor = DEVICE_FIELD_COMPUTE_CAPABILITY_MINOR,
    Id = DEVICE_FIELD_ID,
    EccEnabled = DEVICE_FIELD_ECC_ENABLED,
    Uuid = DEVICE_FIELD_UUID,
    Name = DEVICE_FIELD_NAME,
    IsCudaVisible = DEVICE_FIELD_IS_CUDA_VISIBLE,
    IsMigEnabled = DEVICE_FIELD_IS_MIG_ENABLED,
    GpuInstanceId = DEVICE_FIELD_GPU_INSTANCE_ID,
    ComputeInstanceId = DEVICE_FIELD_COMPUTE_INSTANCE_ID,
    MigUuid = DEVICE_FIELD_MIG_UUID,
    IsNonUniformMemoryAccessNode = DEVICE_FIELD_IS_NUMA_NODE,
    NonUniformMemoryAccessId = DEVICE_FIELD_NUMA_ID,
    NumTpcs = DEVICE_FIELD_NUM_TPCS,
    Max = DEVICE_FIELD_MAX,
});

activity_field_enum!(ActivityMemoryCopyField, sys::CUpti_ActivityMemcpyFieldIds, {
        /// uint32_t sourceGraphId; The unique graph ID of the node from which the memcpy node was instantiated or last updated.
        ///
        /// This field will be 0 if the memcpy is not launched through graph launch APIs.
        SourceGraphId = MEMCPY_FIELD_SOURCE_GRAPH_ID,
        /// uint64_t sourceGraphNodeId; The unique graph node ID of the node from which the memcpy node was instantiated or last updated.
        ///
        /// This field will be 0 if the memcpy is not launched through graph launch APIs.
        SourceGraphNodeId = MEMCPY_FIELD_SOURCE_GRAPH_NODE_ID,
    Kind = MEMCPY_FIELD_KIND,
    CopyKind = MEMCPY_FIELD_COPY_KIND,
    SourceKind = MEMCPY_FIELD_SRC_KIND,
    DestinationKind = MEMCPY_FIELD_DST_KIND,
    Flags = MEMCPY_FIELD_FLAGS,
    Bytes = MEMCPY_FIELD_BYTES,
    Start = MEMCPY_FIELD_START,
    End = MEMCPY_FIELD_END,
    DeviceId = MEMCPY_FIELD_DEVICE_ID,
    ContextId = MEMCPY_FIELD_CONTEXT_ID,
    StreamId = MEMCPY_FIELD_STREAM_ID,
    CorrelationId = MEMCPY_FIELD_CORRELATION_ID,
    GraphNodeId = MEMCPY_FIELD_GRAPH_NODE_ID,
    GraphId = MEMCPY_FIELD_GRAPH_ID,
    ChannelId = MEMCPY_FIELD_CHANNEL_ID,
    ChannelType = MEMCPY_FIELD_CHANNEL_TYPE,
    IsDeviceLaunched = MEMCPY_FIELD_IS_DEVICE_LAUNCHED,
    CopyCount = MEMCPY_FIELD_COPY_COUNT,
});

activity_field_enum!(ActivityPeerMemoryCopyField, sys::CUpti_ActivityMemcpy2FieldIds, {
    Kind = MEMCPY2_FIELD_KIND,
    CopyKind = MEMCPY2_FIELD_COPY_KIND,
    SourceKind = MEMCPY2_FIELD_SRC_KIND,
    DestinationKind = MEMCPY2_FIELD_DST_KIND,
    Flags = MEMCPY2_FIELD_FLAGS,
    Bytes = MEMCPY2_FIELD_BYTES,
    Start = MEMCPY2_FIELD_START,
    End = MEMCPY2_FIELD_END,
    DeviceId = MEMCPY2_FIELD_DEVICE_ID,
    ContextId = MEMCPY2_FIELD_CONTEXT_ID,
    StreamId = MEMCPY2_FIELD_STREAM_ID,
    SourceDeviceId = MEMCPY2_FIELD_SRC_DEVICE_ID,
    SourceContextId = MEMCPY2_FIELD_SRC_CONTEXT_ID,
    DestinationDeviceId = MEMCPY2_FIELD_DST_DEVICE_ID,
    DestinationContextId = MEMCPY2_FIELD_DST_CONTEXT_ID,
    CorrelationId = MEMCPY2_FIELD_CORRELATION_ID,
    GraphNodeId = MEMCPY2_FIELD_GRAPH_NODE_ID,
    GraphId = MEMCPY2_FIELD_GRAPH_ID,
    ChannelId = MEMCPY2_FIELD_CHANNEL_ID,
    ChannelType = MEMCPY2_FIELD_CHANNEL_TYPE,
    Max = MEMCPY2_FIELD_MAX,
});

activity_field_enum!(ActivityMemorySetField, sys::CUpti_ActivityMemsetFieldIds, {
        /// uint32_t sourceGraphId; The unique graph ID of the node from which the memset node was instantiated or last updated.
        ///
        /// This field will be 0 if the memset is not launched through graph launch APIs.
        SourceGraphId = MEMSET_FIELD_SOURCE_GRAPH_ID,
        /// uint64_t sourceGraphNodeId; The unique graph node ID of the node from which the memset node was instantiated or last updated.
        ///
        /// This field will be 0 if the memset is not launched through graph launch APIs.
        SourceGraphNodeId = MEMSET_FIELD_SOURCE_GRAPH_NODE_ID,
    Kind = MEMSET_FIELD_KIND,
    Value = MEMSET_FIELD_VALUE,
    Bytes = MEMSET_FIELD_BYTES,
    Start = MEMSET_FIELD_START,
    End = MEMSET_FIELD_END,
    DeviceId = MEMSET_FIELD_DEVICE_ID,
    ContextId = MEMSET_FIELD_CONTEXT_ID,
    StreamId = MEMSET_FIELD_STREAM_ID,
    CorrelationId = MEMSET_FIELD_CORRELATION_ID,
    Flags = MEMSET_FIELD_FLAGS,
    MemoryKind = MEMSET_FIELD_MEMORY_KIND,
    GraphNodeId = MEMSET_FIELD_GRAPH_NODE_ID,
    GraphId = MEMSET_FIELD_GRAPH_ID,
    ChannelId = MEMSET_FIELD_CHANNEL_ID,
    ChannelType = MEMSET_FIELD_CHANNEL_TYPE,
    IsDeviceLaunched = MEMSET_FIELD_IS_DEVICE_LAUNCHED,
});

activity_field_enum!(ActivityMemoryField, sys::CUpti_ActivityMemoryFieldIds, {
    Kind = MEMORY_FIELD_KIND,
    OperationType = MEMORY_FIELD_OPERATION_TYPE,
    MemoryKind = MEMORY_FIELD_MEMORY_KIND,
    CorrelationId = MEMORY_FIELD_CORRELATION_ID,
    Address = MEMORY_FIELD_ADDRESS,
    Bytes = MEMORY_FIELD_BYTES,
    Timestamp = MEMORY_FIELD_TIMESTAMP,
    ProcessId = MEMORY_FIELD_PROCESS_ID,
    DeviceId = MEMORY_FIELD_DEVICE_ID,
    ContextId = MEMORY_FIELD_CONTEXT_ID,
    StreamId = MEMORY_FIELD_STREAM_ID,
    IsAsynchronous = MEMORY_FIELD_IS_ASYNC,
    PoolType = MEMORY_FIELD_POOL_TYPE,
    PoolAddress = MEMORY_FIELD_POOL_ADDRESS,
    PoolReleaseThreshold = MEMORY_FIELD_POOL_RELEASE_THRESHOLD,
    PoolSize = MEMORY_FIELD_POOL_SIZE,
    PoolProcessId = MEMORY_FIELD_POOL_PROCESS_ID,
    PoolUtilizedSize = MEMORY_FIELD_POOL_UTILIZED_SIZE,
    Source = MEMORY_FIELD_SOURCE,
});

activity_field_enum!(
    ActivityMemoryPoolField,
    sys::CUpti_ActivityMemoryPoolFieldIds,
    {
        Kind = MEMORY_POOL_FIELD_KIND,
        OperationType = MEMORY_POOL_FIELD_OPERATION_TYPE,
        PoolType = MEMORY_POOL_FIELD_POOL_TYPE,
        CorrelationId = MEMORY_POOL_FIELD_CORRELATION_ID,
        ProcessId = MEMORY_POOL_FIELD_PROCESS_ID,
        DeviceId = MEMORY_POOL_FIELD_DEVICE_ID,
        MinBytesToKeep = MEMORY_POOL_FIELD_MIN_BYTES_TO_KEEP,
        Address = MEMORY_POOL_FIELD_ADDRESS,
        Size = MEMORY_POOL_FIELD_SIZE,
        ReleaseThreshold = MEMORY_POOL_FIELD_RELEASE_THRESHOLD,
        Timestamp = MEMORY_POOL_FIELD_TIMESTAMP,
        UtilizedSize = MEMORY_POOL_FIELD_UTILIZED_SIZE,
        IsManagedPool = MEMORY_POOL_FIELD_IS_MANAGED_POOL,
    }
);

activity_field_enum!(ActivityGraphTraceField, sys::CUpti_ActivityGraphTraceFieldIds, {
    Kind = GRAPH_TRACE_FIELD_KIND,
    CorrelationId = GRAPH_TRACE_FIELD_CORRELATION_ID,
    Start = GRAPH_TRACE_FIELD_START,
    End = GRAPH_TRACE_FIELD_END,
    DeviceId = GRAPH_TRACE_FIELD_DEVICE_ID,
    GraphId = GRAPH_TRACE_FIELD_GRAPH_ID,
    ContextId = GRAPH_TRACE_FIELD_CONTEXT_ID,
    StreamId = GRAPH_TRACE_FIELD_STREAM_ID,
    EndDeviceId = GRAPH_TRACE_FIELD_END_DEVICE_ID,
    EndContextId = GRAPH_TRACE_FIELD_END_CONTEXT_ID,
});

activity_field_enum!(
    ActivityDeviceGraphTraceField,
    sys::CUpti_ActivityDeviceGraphTraceFieldIds,
    {
        Kind = DEVICE_GRAPH_TRACE_FIELD_KIND,
        DeviceId = DEVICE_GRAPH_TRACE_FIELD_DEVICE_ID,
        Start = DEVICE_GRAPH_TRACE_FIELD_START,
        End = DEVICE_GRAPH_TRACE_FIELD_END,
        GraphId = DEVICE_GRAPH_TRACE_FIELD_GRAPH_ID,
        LauncherGraphId = DEVICE_GRAPH_TRACE_FIELD_LAUNCHER_GRAPH_ID,
        DeviceLaunchMode = DEVICE_GRAPH_TRACE_FIELD_DEVICE_LAUNCH_MODE,
        ContextId = DEVICE_GRAPH_TRACE_FIELD_CONTEXT_ID,
        StreamId = DEVICE_GRAPH_TRACE_FIELD_STREAM_ID,
    }
);

activity_field_enum!(
    ActivityGraphHostNodeField,
    sys::CUpti_ActivityGraphHostNodeFieldIds,
    {
        Kind = GRAPH_HOST_NODE_FIELD_KIND,
        StreamId = GRAPH_HOST_NODE_FIELD_STREAM_ID,
        ContextId = GRAPH_HOST_NODE_FIELD_CONTEXT_ID,
        DeviceId = GRAPH_HOST_NODE_FIELD_DEVICE_ID,
        CorrelationId = GRAPH_HOST_NODE_FIELD_CORRELATION_ID,
        GraphId = GRAPH_HOST_NODE_FIELD_GRAPH_ID,
        GraphNodeId = GRAPH_HOST_NODE_FIELD_GRAPH_NODE_ID,
        ProcessId = GRAPH_HOST_NODE_FIELD_PROCESS_ID,
        ThreadId = GRAPH_HOST_NODE_FIELD_THREAD_ID,
        Start = GRAPH_HOST_NODE_FIELD_START,
        End = GRAPH_HOST_NODE_FIELD_END,
    }
);

activity_field_enum!(ActivityHostLaunchField, sys::CUpti_ActivityHostLaunchFieldIds, {
    Kind = HOST_LAUNCH_FIELD_KIND,
    StreamId = HOST_LAUNCH_FIELD_STREAM_ID,
    ContextId = HOST_LAUNCH_FIELD_CONTEXT_ID,
    DeviceId = HOST_LAUNCH_FIELD_DEVICE_ID,
    CorrelationId = HOST_LAUNCH_FIELD_CORRELATION_ID,
    ProcessId = HOST_LAUNCH_FIELD_PROCESS_ID,
    ThreadId = HOST_LAUNCH_FIELD_THREAD_ID,
    Start = HOST_LAUNCH_FIELD_START,
    End = HOST_LAUNCH_FIELD_END,
});

activity_field_enum!(
    ActivityComputeEngineContextSwitchField,
    sys::CUpti_ActivityComputeEngineCtxSwitchFieldIds,
    {
        Kind = COMPUTE_ENGINE_CTX_SWITCH_FIELD_KIND,
        ContextId = COMPUTE_ENGINE_CTX_SWITCH_FIELD_CONTEXT_ID,
        Timestamp = COMPUTE_ENGINE_CTX_SWITCH_FIELD_TIMESTAMP,
        OperationType = COMPUTE_ENGINE_CTX_SWITCH_FIELD_OPERATION_TYPE,
    }
);

activity_field_enum!(
    ActivityGreenContextField,
    sys::CUpti_ActivityGreenContextFieldIds,
    {
        /// uint32_t workqueueChannelCount; The number of HW channels assigned to this green context’s work queue.
        WorkqueueChannelCount = GREEN_CONTEXT_FIELD_WORKQUEUE_CHANNEL_COUNT,
        /// uint32_t workqueueChannelIds\[32\]; Array of HW channel IDs assigned to this green context’s work queue.
        WorkqueueChannelIds = GREEN_CONTEXT_FIELD_WORKQUEUE_CHANNEL_IDS,
        /// uint64_t workqueueResourceId; The work queue resource ID associated with the green context.
        ///
        /// Refer.
        WorkqueueResourceId = GREEN_CONTEXT_FIELD_WORKQUEUE_RESOURCE_ID,
        /// uint32_t workqueueConcurrencyLimit; The concurrency limit for the work queue associated with the green context.
        WorkqueueConcurrencyLimit = GREEN_CONTEXT_FIELD_WORKQUEUE_CONCURRENCY_LIMIT,
        /// uint32_t workqueueSharingScope; The sharing scope for the work queue associated with the green context.
        WorkqueueSharingScope = GREEN_CONTEXT_FIELD_WORKQUEUE_SHARING_SCOPE,
        Kind = GREEN_CONTEXT_FIELD_KIND,
        ContextId = GREEN_CONTEXT_FIELD_CONTEXT_ID,
        ParentContextId = GREEN_CONTEXT_FIELD_PARENT_CONTEXT_ID,
        DeviceId = GREEN_CONTEXT_FIELD_DEVICE_ID,
        NumMultiprocessors = GREEN_CONTEXT_FIELD_NUM_MULTIPROCESSORS,
        NumTpcs = GREEN_CONTEXT_FIELD_NUM_TPCS,
        LogicalTpcMaskSize = GREEN_CONTEXT_FIELD_LOGICAL_TPC_MASK_SIZE,
        LogicalTpcMask = GREEN_CONTEXT_FIELD_LOGICAL_TPC_MASK,
    }
);

activity_field_enum!(ActivityNameField, sys::CUpti_ActivityNameFieldIds, {
    Kind = NAME_FIELD_KIND,
    ObjectKind = NAME_FIELD_OBJECT_KIND,
    ObjectId = NAME_FIELD_OBJECT_ID,
    Name = NAME_FIELD_NAME,
});

activity_field_enum!(ActivityMarkerField, sys::CUpti_ActivityMarkerFieldIds, {
    Kind = MARKER_FIELD_KIND,
    Flags = MARKER_FIELD_FLAGS,
    Timestamp = MARKER_FIELD_TIMESTAMP,
    Id = MARKER_FIELD_ID,
    ProcessId = MARKER_FIELD_PROCESS_ID,
    ThreadId = MARKER_FIELD_THREAD_ID,
    Name = MARKER_FIELD_NAME,
    Domain = MARKER_FIELD_DOMAIN,
});

activity_field_enum!(
    ActivityMarkerDataField,
    sys::CUpti_ActivityMarkerDataFieldIds,
    {
        Kind = MARKER_DATA_FIELD_KIND,
        Flags = MARKER_DATA_FIELD_FLAGS,
        Id = MARKER_DATA_FIELD_ID,
        PayloadKind = MARKER_DATA_FIELD_PAYLOAD_KIND,
        Payload = MARKER_DATA_FIELD_PAYLOAD,
        Color = MARKER_DATA_FIELD_COLOR,
        Category = MARKER_DATA_FIELD_CATEGORY,
        CuptiDomainId = MARKER_DATA_FIELD_CUPTI_DOMAIN_ID,
    }
);

activity_field_enum!(
    ActivityExternalCorrelationField,
    sys::CUpti_ActivityExternalCorrelationFieldIds,
    {
        Kind = EXTERNAL_CORRELATION_FIELD_KIND,
        ExternalKind = EXTERNAL_CORRELATION_FIELD_EXTERNAL_KIND,
        ExternalId = EXTERNAL_CORRELATION_FIELD_EXTERNAL_ID,
        CorrelationId = EXTERNAL_CORRELATION_FIELD_CORRELATION_ID,
    }
);

activity_field_enum!(ActivityCudaEventField, sys::CUpti_ActivityCudaEventFieldIds, {
    Kind = CUDA_EVENT_FIELD_KIND,
    CorrelationId = CUDA_EVENT_FIELD_CORRELATION_ID,
    ContextId = CUDA_EVENT_FIELD_CONTEXT_ID,
    StreamId = CUDA_EVENT_FIELD_STREAM_ID,
    EventId = CUDA_EVENT_FIELD_EVENT_ID,
    DeviceId = CUDA_EVENT_FIELD_DEVICE_ID,
    DeviceTimestamp = CUDA_EVENT_FIELD_DEVICE_TIMESTAMP,
    CudaEventSyncId = CUDA_EVENT_FIELD_CUDA_EVENT_SYNC_ID,
});

activity_field_enum!(ActivityOverheadField, sys::CUpti_ActivityOverheadFieldIds, {
    Kind = OVERHEAD_FIELD_KIND,
    OverheadKind = OVERHEAD_FIELD_OVERHEAD_KIND,
    ProcessId = OVERHEAD_FIELD_PROCESS_ID,
    ThreadId = OVERHEAD_FIELD_THREAD_ID,
    Start = OVERHEAD_FIELD_START,
    End = OVERHEAD_FIELD_END,
    CorrelationId = OVERHEAD_FIELD_CORRELATION_ID,
    OverheadData = OVERHEAD_FIELD_OVERHEAD_DATA,
});

activity_field_enum!(ActivityUvmCounterField, sys::CUpti_ActivityUvmCounterFieldIds, {
    Kind = UVM_COUNTER_FIELD_KIND,
    CounterKind = UVM_COUNTER_FIELD_COUNTER_KIND,
    Value = UVM_COUNTER_FIELD_VALUE,
    Start = UVM_COUNTER_FIELD_START,
    End = UVM_COUNTER_FIELD_END,
    Address = UVM_COUNTER_FIELD_ADDRESS,
    SourceId = UVM_COUNTER_FIELD_SRC_ID,
    DestinationId = UVM_COUNTER_FIELD_DST_ID,
    StreamId = UVM_COUNTER_FIELD_STREAM_ID,
    ProcessId = UVM_COUNTER_FIELD_PROCESS_ID,
    Flags = UVM_COUNTER_FIELD_FLAGS,
    Processors = UVM_COUNTER_FIELD_PROCESSORS,
    Max = UVM_COUNTER_FIELD_MAX,
});

activity_field_enum!(ActivityContextField, sys::CUpti_ActivityContextFieldIds, {
    Kind = CONTEXT_FIELD_KIND,
    ContextId = CONTEXT_FIELD_CONTEXT_ID,
    DeviceId = CONTEXT_FIELD_DEVICE_ID,
    ComputeApiKind = CONTEXT_FIELD_COMPUTE_API_KIND,
    NullStreamId = CONTEXT_FIELD_NULL_STREAM_ID,
    ParentContextId = CONTEXT_FIELD_PARENT_CONTEXT_ID,
    IsGreenContext = CONTEXT_FIELD_IS_GREEN_CONTEXT,
    NumMultiprocessors = CONTEXT_FIELD_NUM_MULTIPROCESSORS,
    CigMode = CONTEXT_FIELD_CIG_MODE,
    ProcessId = CONTEXT_FIELD_PROCESS_ID,
});

activity_field_enum!(ActivityKernelField, sys::CUpti_ActivityKernelFieldIds, {
        /// uint32_t sourceGraphId; The unique graph ID of the node from which the kernel node was instantiated or last updated.
        ///
        /// This field will be 0 if the kernel is not launched through graph launch APIs.
        SourceGraphId = KERNEL_FIELD_SOURCE_GRAPH_ID,
        /// uint64_t sourceGraphNodeId; The unique graph node ID of the node from which the kernel node was instantiated or last updated.
        ///
        /// This field will be 0 if the kernel is not launched through graph launch APIs.
        SourceGraphNodeId = KERNEL_FIELD_SOURCE_GRAPH_NODE_ID,
        /// uint32_t executionModel; The execution model of the kernel function.
        ///
        /// See also
        ///
        /// [`FunctionExecutionModel`](crate::types::FunctionExecutionModel) for possible values.
        ExecutionModel = KERNEL_FIELD_EXECUTION_MODEL,
    Kind = KERNEL_FIELD_KIND,
    CacheConfigRequested = KERNEL_FIELD_CACHE_CONFIG_REQUESTED,
    CacheConfigExecuted = KERNEL_FIELD_CACHE_CONFIG_EXECUTED,
    SharedMemoryConfig = KERNEL_FIELD_SHARED_MEMORY_CONFIG,
    RegistersPerThread = KERNEL_FIELD_REGISTERS_PER_THREAD,
    PartitionedGlobalCacheRequested = KERNEL_FIELD_PARTITIONED_GLOBAL_CACHE_REQUESTED,
    PartitionedGlobalCacheExecuted = KERNEL_FIELD_PARTITIONED_GLOBAL_CACHE_EXECUTED,
    Start = KERNEL_FIELD_START,
    End = KERNEL_FIELD_END,
    Completed = KERNEL_FIELD_COMPLETED,
    DeviceId = KERNEL_FIELD_DEVICE_ID,
    ContextId = KERNEL_FIELD_CONTEXT_ID,
    StreamId = KERNEL_FIELD_STREAM_ID,
    GridX = KERNEL_FIELD_GRID_X,
    GridY = KERNEL_FIELD_GRID_Y,
    GridZ = KERNEL_FIELD_GRID_Z,
    BlockX = KERNEL_FIELD_BLOCK_X,
    BlockY = KERNEL_FIELD_BLOCK_Y,
    BlockZ = KERNEL_FIELD_BLOCK_Z,
    StaticSharedMemory = KERNEL_FIELD_STATIC_SHARED_MEMORY,
    DynamicSharedMemory = KERNEL_FIELD_DYNAMIC_SHARED_MEMORY,
    LocalMemoryPerThread = KERNEL_FIELD_LOCAL_MEMORY_PER_THREAD,
    CorrelationId = KERNEL_FIELD_CORRELATION_ID,
    GridId = KERNEL_FIELD_GRID_ID,
    Name = KERNEL_FIELD_NAME,
    Queued = KERNEL_FIELD_QUEUED,
    Submitted = KERNEL_FIELD_SUBMITTED,
    LaunchType = KERNEL_FIELD_LAUNCH_TYPE,
    IsSharedMemoryCarveoutRequested = KERNEL_FIELD_IS_SHARED_MEMORY_CARVEOUT_REQUESTED,
    SharedMemoryCarveoutRequested = KERNEL_FIELD_SHARED_MEMORY_CARVEOUT_REQUESTED,
    SharedMemoryExecuted = KERNEL_FIELD_SHARED_MEMORY_EXECUTED,
    GraphNodeId = KERNEL_FIELD_GRAPH_NODE_ID,
    SharedMemoryLimitConfig = KERNEL_FIELD_SHMEM_LIMIT_CONFIG,
    GraphId = KERNEL_FIELD_GRAPH_ID,
    AccessPolicyWindow = KERNEL_FIELD_ACCESS_POLICY_WINDOW,
    ChannelId = KERNEL_FIELD_CHANNEL_ID,
    ChannelType = KERNEL_FIELD_CHANNEL_TYPE,
    ClusterX = KERNEL_FIELD_CLUSTER_X,
    ClusterY = KERNEL_FIELD_CLUSTER_Y,
    ClusterZ = KERNEL_FIELD_CLUSTER_Z,
    ClusterSchedulingPolicy = KERNEL_FIELD_CLUSTER_SCHEDULING_POLICY,
    LocalMemoryTotal = KERNEL_FIELD_LOCAL_MEMORY_TOTAL,
    MaxPotentialClusterSize = KERNEL_FIELD_MAX_POTENTIAL_CLUSTER_SIZE,
    MaxActiveClusters = KERNEL_FIELD_MAX_ACTIVE_CLUSTERS,
    IsDeviceLaunched = KERNEL_FIELD_IS_DEVICE_LAUNCHED,
    LaunchPriority = KERNEL_FIELD_LAUNCH_PRIORITY,
});

activity_field_enum!(
    ActivityMemoryDecompressField,
    sys::CUpti_ActivityMemDecompressFieldIds,
    {
        Kind = MEM_DECOMPRESS_FIELD_KIND,
        DeviceId = MEM_DECOMPRESS_FIELD_DEVICE_ID,
        ContextId = MEM_DECOMPRESS_FIELD_CONTEXT_ID,
        StreamId = MEM_DECOMPRESS_FIELD_STREAM_ID,
        ChannelId = MEM_DECOMPRESS_FIELD_CHANNEL_ID,
        ChannelType = MEM_DECOMPRESS_FIELD_CHANNEL_TYPE,
        CorrelationId = MEM_DECOMPRESS_FIELD_CORRELATION_ID,
        NumberOfOperations = MEM_DECOMPRESS_FIELD_NUMBER_OF_OPERATIONS,
        SourceBytes = MEM_DECOMPRESS_FIELD_SOURCE_BYTES,
        Start = MEM_DECOMPRESS_FIELD_START,
        End = MEM_DECOMPRESS_FIELD_END,
    }
);

activity_field_enum!(
    ActivityConfidentialComputeRotationField,
    sys::CUpti_ActivityConfidentialComputeRotationFieldIds,
    {
        Kind = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_KIND,
        EventType = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_EVENT_TYPE,
        DeviceId = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_DEVICE_ID,
        ContextId = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_CONTEXT_ID,
        ChannelId = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_CHANNEL_ID,
        ChannelType = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_CHANNEL_TYPE,
        Timestamp = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_TIMESTAMP,
        Max = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_MAX,
    }
);

activity_field_enum!(ActivityJitField, sys::CUpti_ActivityJitFieldIds, {
    Kind = JIT_FIELD_KIND,
    EntryType = JIT_FIELD_ENTRY_TYPE,
    OperationType = JIT_FIELD_OPERATION_TYPE,
    DeviceId = JIT_FIELD_DEVICE_ID,
    Start = JIT_FIELD_START,
    End = JIT_FIELD_END,
    CorrelationId = JIT_FIELD_CORRELATION_ID,
    OperationCorrelationId = JIT_FIELD_OPERATION_CORRELATION_ID,
    CacheSize = JIT_FIELD_CACHE_SIZE,
    CachePath = JIT_FIELD_CACHE_PATH,
    ProcessId = JIT_FIELD_PROCESS_ID,
    ThreadId = JIT_FIELD_THREAD_ID,
    Max = JIT_FIELD_MAX,
});

activity_field_enum!(ActivityStreamField, sys::CUpti_ActivityStreamFieldIds, {
    Kind = STREAM_FIELD_KIND,
    ContextId = STREAM_FIELD_CONTEXT_ID,
    StreamId = STREAM_FIELD_STREAM_ID,
    Priority = STREAM_FIELD_PRIORITY,
    Flag = STREAM_FIELD_FLAG,
    CorrelationId = STREAM_FIELD_CORRELATION_ID,
});

activity_field_enum!(
    ActivitySynchronizationField,
    sys::CUpti_ActivitySynchronizationFieldIds,
    {
        Kind = SYNCHRONIZATION_FIELD_KIND,
        SynchronizationType = SYNCHRONIZATION_FIELD_TYPE,
        Start = SYNCHRONIZATION_FIELD_START,
        End = SYNCHRONIZATION_FIELD_END,
        CorrelationId = SYNCHRONIZATION_FIELD_CORRELATION_ID,
        ContextId = SYNCHRONIZATION_FIELD_CONTEXT_ID,
        StreamId = SYNCHRONIZATION_FIELD_STREAM_ID,
        CudaEventId = SYNCHRONIZATION_FIELD_CUDA_EVENT_ID,
        CudaEventSyncId = SYNCHRONIZATION_FIELD_CUDA_EVENT_SYNC_ID,
        ReturnValue = SYNCHRONIZATION_FIELD_RETURN_VALUE,
    }
);
