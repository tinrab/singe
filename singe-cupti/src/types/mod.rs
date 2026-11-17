#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

mod activity;
mod callback;
mod device;
mod field;
mod ids;
mod metric;
mod record;

pub(crate) use self::callback::SubscriberHandle;
pub use self::{
    activity::{
        ActivityAttribute, ActivityAttributeSetting, ActivityAttributeValue, ActivityFlags,
        ActivityFlushFlag, ActivityKind, ActivityOptions, ActivityThreadIdType,
        ActivityUnifiedMemoryCounterConfig, ActivityUnifiedMemoryCounterKind,
        ActivityUnifiedMemoryCounterScope, ApiCallbackFilter,
    },
    callback::{
        ApiCallbackSite, CallbackDomain, CallbackId, CallbackIdResource, CallbackIdState,
        CallbackIdSync, MultipleSubscriberState,
    },
    device::{DeviceAttribute, DeviceType, DeviceVirtualizationMode, LinkFlags, PcieDeviceType},
    field::{
        ActivityApiField, ActivityComputeEngineContextSwitchField,
        ActivityConfidentialComputeRotationField, ActivityContextField, ActivityCudaEventField,
        ActivityDeviceField, ActivityDeviceGraphTraceField, ActivityExternalCorrelationField,
        ActivityFieldId, ActivityGraphHostNodeField, ActivityGraphTraceField,
        ActivityGreenContextField, ActivityHostLaunchField, ActivityJitField, ActivityKernelField,
        ActivityMarkerDataField, ActivityMarkerField, ActivityMemoryCopyField,
        ActivityMemoryDecompressField, ActivityMemoryField, ActivityMemoryPoolField,
        ActivityMemorySetField, ActivityNameField, ActivityOverheadField,
        ActivityPeerMemoryCopyField, ActivityStreamField, ActivitySynchronizationField,
        ActivityUvmCounterField,
    },
    ids::{
        ActivityMarkerId, ChannelId, ComputeInstanceId, ContextId, CorrelationId, CudaDevice,
        CudaDeviceAttribute, CudaEventId, CudaEventSyncId, CuptiDomainId, DeviceId,
        ExecutableGraphId, ExternalCorrelationId, FunctionId, GpuInstanceId, GraphId, GraphNodeId,
        GridId, JitOperationCorrelationId, ModuleId, NonUniformMemoryAccessId, NpuDomainId,
        OpenAccAsynchronousId, OpenAccDeviceNumber, OpenAccDeviceTypeId, PcieBridgeId, PcieBusId,
        PcieDomainId, PcieHardwareDeviceId, PcieVendorId, ProcessId, SourceLocatorId, StreamId,
        ThreadId, UnifiedMemoryProcessorId,
    },
    metric::{MetricValue, MetricValueKind, MetricValueUtilizationLevel},
    record::{
        ActivityAutoBoostState, ActivityComputeApiKind, ActivityEnvironmentKind,
        ActivityInstructionClass, ActivityJitEntryType, ActivityJitOperationType,
        ActivityLaunchType, ActivityMemoryCopyKind, ActivityMemoryKind,
        ActivityMemoryOperationType, ActivityMemoryPoolOperationType, ActivityMemoryPoolType,
        ActivityObjectKind, ActivityOverheadKind, ActivityPartitionedGlobalCacheConfig,
        ActivityPcSamplingStallReason, ActivityPreemptionKind, ActivityStreamFlag,
        ActivitySynchronizationType, ActivityUnifiedMemoryAccessType,
        ActivityUnifiedMemoryMigrationCause, ActivityUnifiedMemoryRemoteMapCause, ChannelType,
        ComputeEngineContextSwitchOperationType, ConfidentialComputeRotationEventType,
        ContextCigMode, DeviceGraphLaunchMode, EnvironmentClocksThrottleReasons,
        ExternalCorrelationKind, FunctionExecutionModel, FunctionSharedMemoryLimitConfig,
        OpenAccConstructKind, OpenAccEventKind, OpenMultiProcessingEventKind,
    },
};
pub use crate::callback_id::{DriverCallbackId, RuntimeCallbackId};
