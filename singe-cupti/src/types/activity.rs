use std::{
    fmt::{self, Display, Formatter},
    mem::size_of,
    ptr,
};

use bitflags::{bitflags, parser};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cupti_sys as sys;

use crate::{
    callback_id::{DriverCallbackId, RuntimeCallbackId},
    error::{Result, Status},
    types::{ActivityFieldId, DeviceId, MultipleSubscriberState},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// Controls whether an activity flush is requested normally or forced.
pub enum ActivityFlushFlag {
    /// Requests the default activity buffer flush behavior.
    Default,
    /// Requests a forced activity buffer flush.
    Forced,
}

impl From<ActivityFlushFlag> for u32 {
    fn from(value: ActivityFlushFlag) -> Self {
        match value {
            ActivityFlushFlag::Default => 0,
            ActivityFlushFlag::Forced => {
                sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED.0
            }
        }
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// CUPTI activity record kind.
    ///
    /// Each value identifies the shape of activity record that CUPTI can emit.
    pub enum ActivityKind: sys::CUpti_ActivityKind {
        /// The activity record is invalid.
        Invalid = CUPTI_ACTIVITY_KIND_INVALID,
        /// A host-to-host, host-to-device, or device-to-device memory copy record.
        ///
        /// CUPTI emits this as [`crate::activity::ActivityMemoryCopy`].
        /// Use [`ActivityKind::PeerMemoryCopy`] for peer-to-peer copies.
        MemoryCopy = CUPTI_ACTIVITY_KIND_MEMCPY,
        /// A GPU memory set record emitted as [`crate::activity::ActivityMemorySet`].
        MemorySet = CUPTI_ACTIVITY_KIND_MEMSET,
        /// A serialized GPU kernel execution record emitted as [`crate::activity::ActivityKernel`].
        ///
        /// This kind serializes kernel execution.
        /// Prefer [`ActivityKind::ConcurrentKernel`] when kernel concurrency should be preserved.
        Kernel = CUPTI_ACTIVITY_KIND_KERNEL,
        /// A CUDA driver API call record emitted as [`crate::activity::ActivityApi`].
        Driver = CUPTI_ACTIVITY_KIND_DRIVER,
        /// A CUDA runtime API call record emitted as [`crate::activity::ActivityApi`].
        Runtime = CUPTI_ACTIVITY_KIND_RUNTIME,
        /// CUDA device information emitted as [`crate::activity::ActivityDevice`].
        Device = CUPTI_ACTIVITY_KIND_DEVICE,
        /// CUDA context information emitted as [`crate::activity::ActivityContext`].
        Context = CUPTI_ACTIVITY_KIND_CONTEXT,
        /// A concurrent GPU kernel execution record emitted as [`crate::activity::ActivityKernel`].
        ConcurrentKernel = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
        /// Resource names assigned through NVTX APIs, emitted as [`crate::activity::ActivityName`].
        Name = CUPTI_ACTIVITY_KIND_NAME,
        /// An NVTX marker record emitted as [`crate::activity::ActivityMarker`].
        Marker = CUPTI_ACTIVITY_KIND_MARKER,
        /// Optional NVTX marker payload emitted as [`crate::activity::ActivityMarkerPayload`].
        ///
        /// Enable [`ActivityKind::Marker`] as well to receive the associated marker records.
        MarkerData = CUPTI_ACTIVITY_KIND_MARKER_DATA,
        /// Legacy source location record emitted as [`crate::activity::ActivitySourceLocator`].
        ///
        /// This kind is unsupported starting with CUDA 13.0; use the SASS Metric APIs instead.
        SourceLocator = CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR,
        /// Legacy source-level global access record emitted as [`ActivityGlobalAccess`](crate::activity::ActivityGlobalAccess).
        ///
        /// This kind is unsupported starting with CUDA 13.0; use the SASS Metric APIs instead.
        GlobalAccess = CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS,
        /// Legacy source-level branch record emitted as [`crate::activity::ActivityBranch`].
        ///
        /// This kind is unsupported starting with CUDA 13.0; use the SASS Metric APIs instead.
        Branch = CUPTI_ACTIVITY_KIND_BRANCH,
        /// CUPTI, compiler, CUDA driver, or other overhead emitted as [`crate::activity::ActivityOverhead`].
        Overhead = CUPTI_ACTIVITY_KIND_OVERHEAD,
        /// CUDA Dynamic Parallelism kernel record emitted as [`crate::activity::ActivityCudaDynamicParallelismKernel`].
        ///
        /// This kind is controlled through [`ActivityKind::ConcurrentKernel`] rather than enabled directly.
        CudaDynamicParallelismKernel = CUPTI_ACTIVITY_KIND_CDP_KERNEL,
        /// CUDA Dynamic Parallelism preemption record emitted as [`crate::activity::ActivityPreemption`].
        Preemption = CUPTI_ACTIVITY_KIND_PREEMPTION,
        /// GPU power, clock, thermal, and related environment data emitted as [`crate::activity::ActivityEnvironment`].
        Environment = CUPTI_ACTIVITY_KIND_ENVIRONMENT,
        /// A peer-to-peer memory copy record emitted as [`crate::activity::ActivityPeerMemoryCopy`].
        PeerMemoryCopy = CUPTI_ACTIVITY_KIND_MEMCPY2,
        /// Legacy source-level instruction execution record emitted as [`crate::activity::ActivityInstructionExecution`].
        ///
        /// This kind is unsupported starting with CUDA 13.0; use the SASS Metric APIs instead.
        InstructionExecution = CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION,
        /// Unified Memory counter record emitted as [`crate::activity::ActivityUnifiedMemoryCounter`].
        UnifiedMemoryCounter = CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER,
        /// Device function record emitted as [`crate::activity::ActivityFunction`].
        Function = CUPTI_ACTIVITY_KIND_FUNCTION,
        /// CUDA module record emitted as [`crate::activity::ActivityModule`].
        ///
        /// This kind is populated from module callbacks rather than directly enabled or disabled.
        Module = CUPTI_ACTIVITY_KIND_MODULE,
        /// Device attribute record emitted as [`crate::activity::ActivityDeviceAttribute`].
        ///
        /// This kind stores values collected through [`DeviceAttribute`](crate::types::DeviceAttribute) or CUDA device attributes.
        DeviceAttribute = CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE,
        /// Legacy source-level shared access record emitted as [`crate::activity::ActivitySharedAccess`].
        ///
        /// This kind is unsupported starting with CUDA 13.0; use the SASS Metric APIs instead.
        SharedAccess = CUPTI_ACTIVITY_KIND_SHARED_ACCESS,
        /// Legacy PC sampling record emitted as [`crate::activity::ActivityPcSampling`].
        ///
        /// This kind serializes kernels and is unsupported starting with CUDA 13.0; use the PC Sampling API instead.
        PcSampling = CUPTI_ACTIVITY_KIND_PC_SAMPLING,
        /// Legacy PC sampling summary record emitted as [`crate::activity::ActivityPcSamplingRecordInfo`].
        ///
        /// This kind is unsupported starting with CUDA 13.0; use the PC Sampling API instead.
        PcSamplingRecordInfo = CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO,
        /// Legacy SASS/source correlation record emitted as [`crate::activity::ActivityInstructionCorrelation`].
        ///
        /// CUPTI emits these records when source-level analysis or PC sampling is enabled.
        /// This kind is unsupported starting with CUDA 13.0; use the SASS Metric APIs instead.
        InstructionCorrelation = CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION,
        /// `OpenACC` data event record emitted as [`crate::activity::ActivityOpenAccTransfer`].
        OpenAccData = CUPTI_ACTIVITY_KIND_OPENACC_DATA,
        /// `OpenACC` launch event record emitted as [`crate::activity::ActivityOpenAccLaunch`].
        OpenAccLaunch = CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH,
        /// `OpenACC` miscellaneous event record emitted as [`crate::activity::ActivityOpenAccOther`].
        OpenAccOther = CUPTI_ACTIVITY_KIND_OPENACC_OTHER,
        /// CUDA event handle record emitted as [`crate::activity::ActivityCudaEvent`].
        CudaEvent = CUPTI_ACTIVITY_KIND_CUDA_EVENT,
        /// CUDA stream record emitted as [`crate::activity::ActivityStream`].
        Stream = CUPTI_ACTIVITY_KIND_STREAM,
        /// CUDA synchronization record emitted as [`crate::activity::ActivitySynchronization`].
        Synchronization = CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
        /// External API correlation record emitted as [`crate::activity::ActivityExternalCorrelation`].
        ExternalCorrelation = CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
        /// `NVLink` topology record emitted as [`crate::activity::ActivityNvLink`].
        NvLink = CUPTI_ACTIVITY_KIND_NVLINK,
        /// Legacy memory allocation record emitted as [`crate::activity::ActivityLegacyMemory`].
        Memory = CUPTI_ACTIVITY_KIND_MEMORY,
        /// PCI topology record emitted as [`crate::activity::ActivityPcie`].
        Pcie = CUPTI_ACTIVITY_KIND_PCIE,
        /// OpenMP parallel event record emitted as [`crate::activity::ActivityOpenMultiProcessing`].
        OpenMultiProcessing = CUPTI_ACTIVITY_KIND_OPENMP,
        /// Internal CUDA driver launch record emitted as [`crate::activity::ActivityApi`].
        InternalLaunchApi = CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API,
        /// Memory allocation record emitted as [`crate::activity::ActivityMemory`].
        Memory2 = CUPTI_ACTIVITY_KIND_MEMORY2,
        /// Memory pool record emitted as [`crate::activity::ActivityMemoryPool`].
        MemoryPool = CUPTI_ACTIVITY_KIND_MEMORY_POOL,
        /// CUDA graph trace record emitted as [`crate::activity::ActivityGraphTrace`].
        GraphTrace = CUPTI_ACTIVITY_KIND_GRAPH_TRACE,
        /// JIT operation record emitted as [`crate::activity::ActivityJit`].
        Jit = CUPTI_ACTIVITY_KIND_JIT,
        /// Device graph trace record emitted as [`crate::activity::ActivityDeviceGraphTrace`].
        ///
        /// Enable [`ActivityKind::GraphTrace`] and device graph tracing to receive these records.
        DeviceGraphTrace = CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE,
        /// Memory decompression record emitted as [`crate::activity::ActivityMemoryDecompress`].
        MemoryDecompress = CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS,
        /// Confidential-computing overhead record emitted as [`crate::activity::ActivityConfidentialComputeRotation`].
        ConfidentialComputeRotation = CUPTI_ACTIVITY_KIND_CONFIDENTIAL_COMPUTE_ROTATION,
        /// CUDA graph host-node execution record emitted as [`crate::activity::ActivityGraphHostNode`].
        GraphHostNode = CUPTI_ACTIVITY_KIND_GRAPH_HOST_NODE,
        /// Compute-engine context switch record emitted as [`crate::activity::ActivityComputeEngineContextSwitch`].
        ComputeEngineCtxSwitch = CUPTI_ACTIVITY_KIND_COMPUTE_ENGINE_CTX_SWITCH,
        /// Host function launch record emitted as [`crate::activity::ActivityHostLaunch`].
        HostLaunch = CUPTI_ACTIVITY_KIND_HOST_LAUNCH,
        /// Green context allocation record emitted as [`crate::activity::ActivityGreenContext`].
        GreenContext = CUPTI_ACTIVITY_KIND_GREEN_CONTEXT,
        /// Count of supported activity kinds.
        Count = CUPTI_ACTIVITY_KIND_COUNT,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_KIND_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Activity API configuration attribute.
    pub enum ActivityAttribute: sys::CUpti_ActivityAttribute {
        /// Per-context device buffer size, in bytes, for concurrent kernel, memory copy, and memory set activity data.
        ///
        /// The attribute value is a `usize`.
        /// Larger buffers reduce dropped timestamps but reserve more device memory.
        /// The default is 3,200,000 bytes.
        /// This only affects future buffer allocations, so set it before CUDA initialization or context creation when possible.
        ///
        /// Device buffer count is controlled by [`ActivityAttribute::DeviceBufferPoolLimit`], and pre-allocation is controlled by [`ActivityAttribute::DeviceBufferPreAllocateValue`].
        DeviceBufferSize = CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
        /// Per-context device buffer size, in bytes, for CUDA Dynamic Parallelism activity data.
        ///
        /// The attribute value is a `usize`.
        /// Larger buffers reduce flushes but reserve more device memory.
        /// The default is 8 MiB and only future allocations are affected.
        DeviceBufferSizeCudaDynamicParallelism = CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP,
        /// Maximum number of activity device buffers CUPTI may allocate per context.
        ///
        /// The attribute value is a `usize`.
        /// Raising the limit can help high-rate kernel, memory copy, and memory set workloads at the cost of a larger memory footprint.
        /// The default is 250 and only future allocations are affected.
        DeviceBufferPoolLimit = CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
        /// Deprecated profiling semaphore pool size.
        ///
        /// The attribute value is a `usize`.
        /// CUPTI no longer uses profiling semaphore pools starting with CUDA 12.3.
        #[deprecated]
        ProfilingSemaphorePoolSize = CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE,
        /// Deprecated maximum number of profiling semaphore pools per context.
        ///
        /// The attribute value is a `usize`.
        /// CUPTI no longer uses profiling semaphore pools starting with CUDA 12.3.
        #[deprecated]
        ProfilingSemaphorePoolLimit = CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT,
        /// Controls whether callers provide zero-initialized activity buffers.
        ///
        /// The attribute value is a `u8`.
        /// Zero means CUPTI clears buffers before filling them; nonzero means the buffer callback must return an already-zeroed buffer.
        /// The default is 0.
        ZeroedOutActivityBuffer = CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER,
        /// Number of activity device buffers to pre-allocate during context initialization.
        ///
        /// The attribute value is a `usize` and must be lower than [`ActivityAttribute::DeviceBufferPoolLimit`].
        /// The default is 3.
        DeviceBufferPreAllocateValue = CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE,
        /// Deprecated number of profiling semaphore pools to pre-allocate during context initialization.
        ///
        /// The attribute value is a `usize`.
        /// CUPTI no longer uses profiling semaphore pools starting with CUDA 12.3.
        #[deprecated]
        ProfilingSemaphorePreAllocateValue = CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_PRE_ALLOCATE_VALUE,
        /// Chooses page-locked host memory for activity profiling buffers.
        ///
        /// The attribute value is a `u8`.
        /// Zero selects device memory, nonzero selects pinned host memory.
        /// Pinned host buffers are not supported on confidential-computing devices.
        MemAllocationTypeHostPinned = CUPTI_ACTIVITY_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED,
        /// Stores activity records in per-thread activity buffers.
        ///
        /// The attribute value is a `u8`.
        /// Set it before registering activity buffer callbacks and before enabling activity kinds.
        /// The default is 1.
        PerThreadActivityBuffer = CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER,
        /// Per-context device buffer size, in bytes, for device graph activity data.
        ///
        /// The attribute value is a `usize`.
        /// Larger buffers reduce flushes but reserve more device memory.
        /// The default is 16 MiB and only future allocations are affected.
        DeviceBufferSizeDeviceGraphs = CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_DEVICE_GRAPHS,
        /// Enables user-defined activity records.
        ///
        /// The attribute value is a `u8`.
        /// When enabled, CUPTI emits only fields selected for collection by the user.
        /// This requires CUPTI API version 13.2 or newer and must be set before enabling any activity kinds.
        UserDefinedRecords = CUPTI_ACTIVITY_ATTR_USER_DEFINED_RECORDS,
        /// Reports or configures multiple-subscriber support.
        ///
        /// The attribute value is a `u8`: 0 disables multiple subscribers, 1 enables them, and 2 means no earlier subscriber has chosen a value.
        /// Query through the activity attribute API but is selected during subscription.
        MultipleSubscriberState = CUPTI_ACTIVITY_ATTR_MULTIPLE_SUBSCRIBER_STATE,
        /// Enables the Hardware Events System globally for activity collection.
        ///
        /// The attribute value is a `u8`.
        /// Set it before any subscriber enables activity kinds.
        EnableHes = CUPTI_ACTIVITY_ATTR_ENABLE_HES,
        /// Enables allocation source library tracking for a subscriber.
        ///
        /// The attribute value is a `u8`.
        /// It cannot be combined with the older allocation-source enable API when multiple subscribers are allowed.
        EnableAllocationSourceTracking = CUPTI_ACTIVITY_ATTR_ENABLE_ALLOCATION_SOURCE_TRACKING,
        /// Enables kernel latency timestamp collection for a subscriber.
        ///
        /// The attribute value is a `u8`.
        /// The default is disabled.
        EnableKernelLatencyTimestamps = CUPTI_ACTIVITY_ATTR_ENABLE_KERNEL_LATENCY_TIMESTAMPS,
        /// Enables synchronization records even for synchronization calls that return errors.
        ///
        /// The attribute value is a `u8`.
        /// The default is disabled.
        EnableAllSyncRecords = CUPTI_ACTIVITY_ATTR_ENABLE_ALL_SYNC_RECORDS,
        /// Enables device timestamp collection for CUDA event activity records.
        ///
        /// The attribute value is a `u8`.
        /// The default is disabled.
        EnableCudaEventDeviceTimestamps = CUPTI_ACTIVITY_ATTR_ENABLE_CUDA_EVENT_DEVICE_TIMESTAMPS,
        /// Enables kernel launch attribute collection for kernel activity records.
        ///
        /// The attribute value is a `u8`.
        /// The default is disabled.
        EnableKernelLaunchAttributes = CUPTI_ACTIVITY_ATTR_ENABLE_KERNEL_LAUNCH_ATTRIBUTES,
        /// Enables device graph trace collection for graph activity records.
        ///
        /// The attribute value is a `u8`.
        /// The default is disabled.
        EnableDeviceGraphTrace = CUPTI_ACTIVITY_ATTR_ENABLE_DEVICE_GRAPH_TRACE,
        /// Enables graph-level tracing in multi-subscriber mode.
        ///
        /// The attribute value is a `u8`.
        /// The default is disabled.
        EnableMultiSubscriberGraphLevelTrace = CUPTI_ACTIVITY_ATTR_ENABLE_MULTI_SUBSCRIBER_GRAPH_TRACE,
        /// Selects the thread ID source used in activity records.
        ///
        /// The attribute value is an [`ActivityThreadIdType`].
        /// The default is [`ActivityThreadIdType::Default`].
        ThreadIdType = CUPTI_ACTIVITY_ATTR_THREAD_ID_TYPE,
        /// Registers or removes the timestamp callback used for activity records.
        ///
        /// The attribute value is a CUPTI timestamp callback pointer; null unregisters the callback and restores CUPTI's default CPU timer.
        TimestampCallback = CUPTI_ACTIVITY_ATTR_TIMESTAMP_CALLBACK,
        /// Configures CUDA in Graphics (CIG) mode for activity collection.
        ///
        /// The attribute value is a `u8`.
        CigMode = CUPTI_ACTIVITY_ATTR_ENABLE_CIG_MODE,
        /// Reserved integer sentinel from CUPTI.
        DeviceBufferForceInt = CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_FORCE_INT,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// Decoded value for an [`ActivityAttribute`] query.
pub enum ActivityAttributeValue {
    /// Bytes scalar value returned by CUPTI.
    Bytes(usize),
    /// Number of entries defined for this CUPTI enum.
    Count(usize),
    /// Boolean enabled state returned by CUPTI.
    Enabled(bool),
    /// Multiple Subscriber State value decoded from CUPTI.
    MultipleSubscriberState(MultipleSubscriberState),
    /// Thread ID Type value decoded from CUPTI.
    ThreadIdType(ActivityThreadIdType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// Writable value for a supported [`ActivityAttribute`].
pub enum ActivityAttributeSetting {
    /// Bytes scalar value returned by CUPTI.
    Bytes(usize),
    /// Number of entries defined for this CUPTI enum.
    Count(usize),
    /// Boolean enabled state returned by CUPTI.
    Enabled(bool),
    /// Thread ID Type value decoded from CUPTI.
    ThreadIdType(ActivityThreadIdType),
}

/// Per-callback enable/disable options for API activity kinds.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ApiCallbackFilter {
    /// Records are generated only for the callback IDs listed.
    DriverAllow(Vec<DriverCallbackId>),
    /// Records are generated only for the runtime callback IDs listed.
    RuntimeAllow(Vec<RuntimeCallbackId>),
    /// Records are generated for all callback IDs except those listed.
    DriverDeny(Vec<DriverCallbackId>),
    /// Records are generated for all runtime callback IDs except those listed.
    RuntimeDeny(Vec<RuntimeCallbackId>),
}

/// Field selection and callback filtering for subscriber-scoped activity collection.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ActivityOptions {
    /// Fields CUPTI records; an empty list selects every field.
    ///
    /// A nonempty selection makes every activity record completed afterwards in this process decode to [`crate::activity::ActivityRecordPayload::Unsupported`], since it shapes user-defined records.
    pub fields: Vec<ActivityFieldId>,
    /// Allowlist or denylist of callback IDs, supported only for driver and runtime API activity kinds.
    pub callback_filter: Option<ApiCallbackFilter>,
}

impl ActivityOptions {
    pub(crate) fn validate_kind(&self, kind: ActivityKind) -> Result<()> {
        let valid = match self.callback_filter {
            Some(ApiCallbackFilter::DriverAllow(_) | ApiCallbackFilter::DriverDeny(_)) => {
                kind == ActivityKind::Driver
            }
            Some(ApiCallbackFilter::RuntimeAllow(_) | ApiCallbackFilter::RuntimeDeny(_)) => {
                kind == ActivityKind::Runtime
            }
            None => true,
        };
        if valid {
            Ok(())
        } else {
            Err(Status::InvalidKind.into())
        }
    }

    pub(crate) fn with_raw<R>(&self, f: impl FnOnce(&mut sys::CUpti_ActivityConfig) -> R) -> R {
        let mut field_ids = self
            .fields
            .iter()
            .map(|field| field.as_raw())
            .collect::<Vec<_>>();
        let field_ids_ptr = if field_ids.is_empty() {
            ptr::null_mut()
        } else {
            field_ids.as_mut_ptr()
        };
        let field_selection = sys::CUpti_ActivityFieldSelection {
            structSize: size_of::<sys::CUpti_ActivityFieldSelection>() as u64,
            numFields: field_ids.len() as u64,
            pFieldIds: field_ids_ptr,
        };
        let (mut cbids, enable) = match &self.callback_filter {
            Some(ApiCallbackFilter::DriverAllow(ids)) => {
                (ids.iter().map(|id| id.get()).collect::<Vec<_>>(), 1)
            }
            Some(ApiCallbackFilter::RuntimeAllow(ids)) => {
                (ids.iter().map(|id| id.get()).collect::<Vec<_>>(), 1)
            }
            Some(ApiCallbackFilter::DriverDeny(ids)) => {
                (ids.iter().map(|id| id.get()).collect::<Vec<_>>(), 0)
            }
            Some(ApiCallbackFilter::RuntimeDeny(ids)) => {
                (ids.iter().map(|id| id.get()).collect::<Vec<_>>(), 0)
            }
            None => (Vec::new(), 0),
        };
        let mut filter = sys::CUpti_ActivityApiCbidOptions {
            structSize: size_of::<sys::CUpti_ActivityApiCbidOptions>() as _,
            pCbids: cbids.as_mut_ptr(),
            numCbids: cbids.len() as _,
            enable,
        };
        let mut options = sys::CUpti_ActivityApiOptions {
            structSize: size_of::<sys::CUpti_ActivityApiOptions>() as _,
            pCbidOptions: &raw mut filter,
        };
        let mut config = sys::CUpti_ActivityConfig {
            structSize: size_of::<sys::CUpti_ActivityConfig>() as u64,
            fieldSelection: field_selection,
            pKindOptions: if self.callback_filter.is_some() {
                (&raw mut options).cast()
            } else {
                ptr::null_mut()
            },
        };
        f(&mut config)
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Deprecated unified-memory counter scope.
    pub enum ActivityUnifiedMemoryCounterScope: sys::CUpti_ActivityUnifiedMemoryCounterScope {
        /// The unified memory counter scope is not known.
        Unknown = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_UNKNOWN,
        /// Collect unified memory counter for single process on one device.
        ProcessSingleDevice = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE,
        /// Collect unified memory counter for single process across all devices.
        ProcessAllDevices = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES,
        /// Number of entries defined for this CUPTI enum.
        Count = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_COUNT,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Unified-memory counter kind.
    pub enum ActivityUnifiedMemoryCounterKind: sys::CUpti_ActivityUnifiedMemoryCounterKind {
        /// The unified memory counter kind is not known.
        Unknown = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_UNKNOWN,
        /// Number of bytes transferred from host to device.
        BytesTransferHostToDevice = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD,
        /// Number of bytes transferred from device to host.
        BytesTransferDeviceToHost = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH,
        /// Number of CPU page faults, this is only supported on 64 bit Linux and Mac platforms.
        CentralProcessingUnitPageFaultCount = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
        /// Number of GPU page faults, this is only supported on devices with compute capability 6.0 and higher and 64 bit Linux platforms.
        GpuPageFault = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT,
        /// Thrashing occurs when data is frequently accessed by multiple processors and has to be constantly migrated around to achieve data locality.
        ///
        /// In this case the overhead of migration may exceed the benefits of locality.
        /// This is only supported on 64 bit Linux platforms.
        Thrashing = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING,
        /// Throttling is a prevention technique used by the driver to avoid further thrashing.
        ///
        /// Here, the driver doesn’t service the fault for one of the contending processors for a specific period of time, so that the other processor can run at full-speed.
        /// This is only supported on 64 bit Linux platforms.
        Throttling = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING,
        /// In case throttling does not help, the driver tries to pin the memory to a processor for a specific period of time.
        ///
        /// One of the contending processors will have slow access to the memory, while the other will have fast access.
        /// This is only supported on 64 bit Linux platforms.
        RemoteMap = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP,
        /// Number of bytes transferred from one device to another device.
        ///
        /// This is only supported on 64 bit Linux platforms.
        BytesTransferDeviceToDevice = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD,
        /// Number of entries defined for this CUPTI enum.
        Count = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_COUNT,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_FORCE_INT,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Configuration for unified-memory activity counters.
pub struct ActivityUnifiedMemoryCounterConfig {
    /// Scope over which the unified memory counter is collected.
    pub scope: ActivityUnifiedMemoryCounterScope,
    /// Unified memory counter kind to configure.
    pub kind: ActivityUnifiedMemoryCounterKind,
    /// Device identifier used when the selected scope targets one device.
    pub device_id: DeviceId,
    /// Whether the selected unified memory counter is enabled.
    pub enable: bool,
}

impl ActivityUnifiedMemoryCounterConfig {
    /// Creates a unified memory counter configuration.
    #[must_use]
    pub const fn new(kind: ActivityUnifiedMemoryCounterKind, enable: bool) -> Self {
        Self {
            scope: ActivityUnifiedMemoryCounterScope::ProcessAllDevices,
            kind,
            device_id: DeviceId(0),
            enable,
        }
    }

    pub(crate) fn to_raw(self) -> sys::CUpti_ActivityUnifiedMemoryCounterConfig {
        #[expect(
            deprecated,
            reason = "wraps deprecated upstream API values that remain part of the supported surface"
        )]
        sys::CUpti_ActivityUnifiedMemoryCounterConfig {
            scope: self.scope.into(),
            kind: self.kind.into(),
            deviceId: self.device_id.as_raw(),
            enable: u32::from(self.enable),
        }
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Thread identifier source used by CUPTI activity records.
    pub enum ActivityThreadIdType: sys::CUpti_ActivityThreadIdType {
        /// Default type Windows uses API `GetCurrentThreadId()` Linux/Mac/Android/QNX use POSIX pthread API `pthread_self()`.
        Default = CUPTI_ACTIVITY_THREAD_ID_TYPE_DEFAULT,
        /// This type is based on the system API available on the underlying platform and thread-id obtained is supposed to be unique for the process lifetime.
        ///
        /// Windows uses API `GetCurrentThreadId()` Linux uses syscall `SYS_gettid` Mac uses syscall `SYS_thread_selfid` Android/QNX use `gettid()`.
        System = CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM,
        /// Add new enums before this field.
        Size = CUPTI_ACTIVITY_THREAD_ID_TYPE_SIZE,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_ACTIVITY_THREAD_ID_TYPE_FORCE_INT,
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    /// Flags attached to CUPTI activity records.
    pub struct ActivityFlags: u32 {
        /// Indicates the activity record has no flags.
        const NONE = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_NONE.0;
        /// Indicates the activity represents a device that supports concurrent kernel execution.
        ///
        /// Valid for [`ActivityKind::Device`] records.
        const DEVICE_CONCURRENT_KERNELS = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_DEVICE_CONCURRENT_KERNELS.0;
        /// CUPTI flag `DEVICE_ATTRIBUTE_CUDEVICE`.
        const DEVICE_ATTRIBUTE_CUDEVICE = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_DEVICE_ATTRIBUTE_CUDEVICE.0;
        /// Indicates the activity represents a region start marker.
        ///
        /// Valid for [`ActivityKind::Marker`] records.
        const MARKER_START = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_START.0;
        /// Indicates the activity represents a region end marker.
        ///
        /// Valid for [`ActivityKind::Marker`] records.
        const MARKER_END = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_END.0;
        /// Indicates the activity represents an attempt to acquire a user defined synchronization object.
        ///
        /// Valid for [`ActivityKind::Marker`] records.
        const MARKER_SYNC_ACQUIRE = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE.0;
        /// Indicates the activity represents success in acquiring the user defined synchronization object.
        ///
        /// Valid for [`ActivityKind::Marker`] records.
        const MARKER_SYNC_ACQUIRE_SUCCESS = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_SUCCESS.0;
        /// Indicates the activity represents failure in acquiring the user defined synchronization object.
        ///
        /// Valid for [`ActivityKind::Marker`] records.
        const MARKER_SYNC_ACQUIRE_FAILED = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_FAILED.0;
        /// Indicates the activity represents releasing a reservation on user defined synchronization object.
        ///
        /// Valid for [`ActivityKind::Marker`] records.
        const MARKER_SYNC_RELEASE = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_SYNC_RELEASE.0;
        /// Mask for the number of bytes requested by each thread in [`ActivityGlobalAccess`](crate::activity::ActivityGlobalAccess).
        const GLOBAL_ACCESS_KIND_SIZE_MASK = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_SIZE_MASK.0;
        /// If bit in this flag is set, the access was load, else it is a store access.
        ///
        /// Valid for [`ActivityGlobalAccess`](crate::activity::ActivityGlobalAccess) records.
        const GLOBAL_ACCESS_KIND_LOAD = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_LOAD.0;
        /// If this bit in flag is set, the load access was cached else it is uncached.
        ///
        /// Valid for [`ActivityGlobalAccess`](crate::activity::ActivityGlobalAccess) records.
        const GLOBAL_ACCESS_KIND_CACHED = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_CACHED.0;
        /// Mask for the [`ActivityInstructionClass`](crate::types::ActivityInstructionClass) bits stored in instruction execution and correlation records.
        const INSTRUCTION_CLASS_MASK = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_INSTRUCTION_CLASS_MASK.0;
    }
}

impl Display for ActivityFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

impl_enum_display!(ActivityKind, {
    Self::Invalid => "CUPTI_ACTIVITY_KIND_INVALID",
    Self::MemoryCopy => "CUPTI_ACTIVITY_KIND_MEMCPY",
    Self::MemorySet => "CUPTI_ACTIVITY_KIND_MEMSET",
    Self::Kernel => "CUPTI_ACTIVITY_KIND_KERNEL",
    Self::Driver => "CUPTI_ACTIVITY_KIND_DRIVER",
    Self::Runtime => "CUPTI_ACTIVITY_KIND_RUNTIME",
    Self::Device => "CUPTI_ACTIVITY_KIND_DEVICE",
    Self::Context => "CUPTI_ACTIVITY_KIND_CONTEXT",
    Self::ConcurrentKernel => "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
    Self::Name => "CUPTI_ACTIVITY_KIND_NAME",
    Self::Marker => "CUPTI_ACTIVITY_KIND_MARKER",
    Self::MarkerData => "CUPTI_ACTIVITY_KIND_MARKER_DATA",
    Self::SourceLocator => "CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR",
    Self::GlobalAccess => "CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS",
    Self::Branch => "CUPTI_ACTIVITY_KIND_BRANCH",
    Self::Overhead => "CUPTI_ACTIVITY_KIND_OVERHEAD",
    Self::CudaDynamicParallelismKernel => "CUPTI_ACTIVITY_KIND_CDP_KERNEL",
    Self::Preemption => "CUPTI_ACTIVITY_KIND_PREEMPTION",
    Self::Environment => "CUPTI_ACTIVITY_KIND_ENVIRONMENT",
    Self::PeerMemoryCopy => "CUPTI_ACTIVITY_KIND_MEMCPY2",
    Self::InstructionExecution => "CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION",
    Self::UnifiedMemoryCounter => "CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER",
    Self::Function => "CUPTI_ACTIVITY_KIND_FUNCTION",
    Self::Module => "CUPTI_ACTIVITY_KIND_MODULE",
    Self::DeviceAttribute => "CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE",
    Self::SharedAccess => "CUPTI_ACTIVITY_KIND_SHARED_ACCESS",
    Self::PcSampling => "CUPTI_ACTIVITY_KIND_PC_SAMPLING",
    Self::PcSamplingRecordInfo => "CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO",
    Self::InstructionCorrelation => "CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION",
    Self::OpenAccData => "CUPTI_ACTIVITY_KIND_OPENACC_DATA",
    Self::OpenAccLaunch => "CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH",
    Self::OpenAccOther => "CUPTI_ACTIVITY_KIND_OPENACC_OTHER",
    Self::CudaEvent => "CUPTI_ACTIVITY_KIND_CUDA_EVENT",
    Self::Stream => "CUPTI_ACTIVITY_KIND_STREAM",
    Self::Synchronization => "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    Self::ExternalCorrelation => "CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION",
    Self::NvLink => "CUPTI_ACTIVITY_KIND_NVLINK",
    Self::Memory => "CUPTI_ACTIVITY_KIND_MEMORY",
    Self::Pcie => "CUPTI_ACTIVITY_KIND_PCIE",
    Self::OpenMultiProcessing => "CUPTI_ACTIVITY_KIND_OPENMP",
    Self::InternalLaunchApi => "CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API",
    Self::Memory2 => "CUPTI_ACTIVITY_KIND_MEMORY2",
    Self::MemoryPool => "CUPTI_ACTIVITY_KIND_MEMORY_POOL",
    Self::GraphTrace => "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    Self::Jit => "CUPTI_ACTIVITY_KIND_JIT",
    Self::DeviceGraphTrace => "CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE",
    Self::MemoryDecompress => "CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS",
    Self::ConfidentialComputeRotation => "CUPTI_ACTIVITY_KIND_CONFIDENTIAL_COMPUTE_ROTATION",
    Self::GraphHostNode => "CUPTI_ACTIVITY_KIND_GRAPH_HOST_NODE",
    Self::ComputeEngineCtxSwitch => "CUPTI_ACTIVITY_KIND_COMPUTE_ENGINE_CTX_SWITCH",
    Self::HostLaunch => "CUPTI_ACTIVITY_KIND_HOST_LAUNCH",
    Self::GreenContext => "CUPTI_ACTIVITY_KIND_GREEN_CONTEXT",
    Self::Count => "CUPTI_ACTIVITY_KIND_COUNT",
    Self::ForceInt => "CUPTI_ACTIVITY_KIND_FORCE_INT",
});

impl_enum_display!(ActivityAttribute, {
    Self::DeviceBufferSize => "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE",
    Self::DeviceBufferSizeCudaDynamicParallelism => "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP",
    Self::DeviceBufferPoolLimit => "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT",
    Self::ProfilingSemaphorePoolSize => "CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE",
    Self::ProfilingSemaphorePoolLimit => "CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT",
    Self::ZeroedOutActivityBuffer => "CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER",
    Self::DeviceBufferPreAllocateValue => "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE",
    Self::ProfilingSemaphorePreAllocateValue => "CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_PRE_ALLOCATE_VALUE",
    Self::MemAllocationTypeHostPinned => "CUPTI_ACTIVITY_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED",
    Self::PerThreadActivityBuffer => "CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER",
    Self::DeviceBufferSizeDeviceGraphs => "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_DEVICE_GRAPHS",
    Self::UserDefinedRecords => "CUPTI_ACTIVITY_ATTR_USER_DEFINED_RECORDS",
    Self::MultipleSubscriberState => "CUPTI_ACTIVITY_ATTR_MULTIPLE_SUBSCRIBER_STATE",
    Self::EnableHes => "CUPTI_ACTIVITY_ATTR_ENABLE_HES",
    Self::EnableAllocationSourceTracking => "CUPTI_ACTIVITY_ATTR_ENABLE_ALLOCATION_SOURCE_TRACKING",
    Self::EnableKernelLatencyTimestamps => "CUPTI_ACTIVITY_ATTR_ENABLE_KERNEL_LATENCY_TIMESTAMPS",
    Self::EnableAllSyncRecords => "CUPTI_ACTIVITY_ATTR_ENABLE_ALL_SYNC_RECORDS",
    Self::EnableCudaEventDeviceTimestamps => "CUPTI_ACTIVITY_ATTR_ENABLE_CUDA_EVENT_DEVICE_TIMESTAMPS",
    Self::EnableKernelLaunchAttributes => "CUPTI_ACTIVITY_ATTR_ENABLE_KERNEL_LAUNCH_ATTRIBUTES",
    Self::EnableDeviceGraphTrace => "CUPTI_ACTIVITY_ATTR_ENABLE_DEVICE_GRAPH_TRACE",
    Self::EnableMultiSubscriberGraphLevelTrace => "CUPTI_ACTIVITY_ATTR_ENABLE_MULTI_SUBSCRIBER_GRAPH_TRACE",
    Self::ThreadIdType => "CUPTI_ACTIVITY_ATTR_THREAD_ID_TYPE",
    Self::TimestampCallback => "CUPTI_ACTIVITY_ATTR_TIMESTAMP_CALLBACK",
    Self::CigMode => "CUPTI_ACTIVITY_ATTR_ENABLE_CIG_MODE",
    Self::DeviceBufferForceInt => "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_FORCE_INT",
});

impl_enum_display!(ActivityUnifiedMemoryCounterScope, {
    Self::Unknown => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_UNKNOWN",
    Self::ProcessSingleDevice => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE",
    Self::ProcessAllDevices => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES",
    Self::Count => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_COUNT",
    Self::ForceInt => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_FORCE_INT",
});

impl_enum_display!(ActivityUnifiedMemoryCounterKind, {
    Self::Unknown => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_UNKNOWN",
    Self::BytesTransferHostToDevice => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD",
    Self::BytesTransferDeviceToHost => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH",
    Self::CentralProcessingUnitPageFaultCount => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT",
    Self::GpuPageFault => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT",
    Self::Thrashing => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING",
    Self::Throttling => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING",
    Self::RemoteMap => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP",
    Self::BytesTransferDeviceToDevice => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD",
    Self::Count => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_COUNT",
    Self::ForceInt => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_FORCE_INT",
});

impl_enum_display!(ActivityThreadIdType, {
    Self::Default => "CUPTI_ACTIVITY_THREAD_ID_TYPE_DEFAULT",
    Self::System => "CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM",
    Self::Size => "CUPTI_ACTIVITY_THREAD_ID_TYPE_SIZE",
    Self::ForceInt => "CUPTI_ACTIVITY_THREAD_ID_TYPE_FORCE_INT",
});

impl_enum_display!(ActivityFlushFlag, {
    Self::Default => "CUPTI_ACTIVITY_FLAG_NONE",
    Self::Forced => "CUPTI_ACTIVITY_FLAG_FLUSH_FORCED",
});

impl Display for ActivityAttributeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes(value) => write!(f, "Bytes({value})"),
            Self::Count(value) => write!(f, "Count({value})"),
            Self::Enabled(value) => write!(f, "Enabled({value})"),
            Self::MultipleSubscriberState(value) => write!(f, "MultipleSubscriberState({value})"),
            Self::ThreadIdType(value) => write!(f, "ThreadIdType({value})"),
        }
    }
}

impl Display for ActivityAttributeSetting {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes(value) => write!(f, "Bytes({value})"),
            Self::Count(value) => write!(f, "Count({value})"),
            Self::Enabled(value) => write!(f, "Enabled({value})"),
            Self::ThreadIdType(value) => write!(f, "ThreadIdType({value})"),
        }
    }
}
