use std::ffi::CStr;

use singe_cuda::dim::Dim3;

use crate::{
    error::Result,
    sys,
    types::{
        ActivityComputeApiKind, ActivityFlags, ActivityKind, ActivityMarkerId, ActivityObjectKind,
        ActivityOverheadKind, ActivityPreemptionKind, ActivityStreamFlag,
        ActivitySynchronizationType, CallbackId, ChannelId, ChannelType,
        ComputeEngineContextSwitchOperationType, ConfidentialComputeRotationEventType,
        ContextCigMode, ContextId, CorrelationId, CudaEventId, CudaEventSyncId, CuptiDomainId,
        DeviceGraphLaunchMode, DeviceId, ExternalCorrelationId, ExternalCorrelationKind, GraphId,
        GraphNodeId, GridId, MetricValue, MetricValueKind, MetricValueUtilizationLevel,
        OpenAccAsynchronousId, OpenAccConstructKind, OpenAccDeviceNumber, OpenAccDeviceTypeId,
        OpenAccEventKind, OpenMultiProcessingEventKind, ProcessId, StreamId, ThreadId,
    },
    utility::optional_cstr,
};

/// Decoded activity record for a driver or runtime API invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityApi {
    /// The activity record kind, must be [`ActivityKind::Driver`], [`ActivityKind::Runtime`], or [`ActivityKind::InternalLaunchApi`].
    pub kind: ActivityKind,
    /// The ID of the driver or runtime function.
    pub callback_id: CallbackId,
    /// The start timestamp for the function, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the function.
    pub start: u64,
    /// The end timestamp for the function, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the function.
    pub end: u64,
    /// The ID of the process where the driver or runtime CUDA function is executing.
    pub process_id: ProcessId,
    /// The ID of the thread where the driver or runtime CUDA function is executing.
    pub thread_id: ThreadId,
    /// The correlation ID of the driver or runtime CUDA function.
    /// Each function invocation is assigned a unique correlation ID that is identical to the correlation ID in the memcpy, memset, or kernel activity record that is associated with this function.
    pub correlation_id: CorrelationId,
    /// Native return status: a `CUresult` code for a driver call or a `cudaError_t` code for a runtime call.
    pub return_value: u32,
}

impl ActivityApi {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityAPI) -> Result<Self> {
        Ok(Self {
            kind: ActivityKind::try_from(value.kind)?,
            callback_id: CallbackId::from(value.cbid),
            start: value.start,
            end: value.end,
            process_id: ProcessId::from(value.processId),
            thread_id: ThreadId::from(value.threadId),
            correlation_id: CorrelationId::from(value.correlationId),
            return_value: value.returnValue,
        })
    }
}

/// Decoded CUDA stream activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityStream {
    /// The ID of the context where the stream was created.
    pub context_id: ContextId,
    /// A unique stream ID to identify the stream.
    pub stream_id: StreamId,
    /// The clamped priority for the stream.
    pub priority: u32,
    /// Flags associated with the stream.
    pub flag: ActivityStreamFlag,
    /// The correlation ID of the API to which this result is associated.
    pub correlation_id: CorrelationId,
}

impl ActivityStream {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityStream) -> Result<Self> {
        Ok(Self {
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            priority: value.priority,
            flag: ActivityStreamFlag::try_from(value.flag)?,
            correlation_id: CorrelationId::from(value.correlationId),
        })
    }
}

/// Decoded CUDA synchronization activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivitySynchronization {
    /// Kind of CUDA synchronization operation recorded.
    pub synchronization_type: ActivitySynchronizationType,
    /// The start timestamp for the function, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the function.
    pub start: u64,
    /// The end timestamp for the function, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the function.
    pub end: u64,
    /// The correlation ID of the API to which this result is associated.
    pub correlation_id: CorrelationId,
    /// The ID of the context for which the synchronization API is called.
    /// In case of context synchronization API it is the context id for which the API is called.
    /// In case of stream/event synchronization it is the ID of the context where the stream/event was created.
    pub context_id: ContextId,
    /// The compute stream for which the synchronization API is called.
    /// A raw value of `u32::MAX` means the field does not apply to this record, as for context and event synchronization.
    pub stream_id: StreamId,
    /// The event ID for which the synchronization API is called.
    /// A raw value of `u32::MAX` means the field does not apply to this record, as for context and stream synchronization.
    pub cuda_event_id: CudaEventId,
    /// A unique ID to associate event synchronization records with the latest CUDA Event record.
    /// Similar field is added in [`ActivityCudaEvent`] to associate synchronization record to the CUDA Event record.
    ///
    /// The same CUDA event can be used multiple times, so the event id will not be unique to correlate the synchronization record with the latest CUDA Event record.
    /// This field will be unique and can be used to do the required correlation.
    ///
    /// A raw value of `u32::MAX` means the field does not apply to this record.
    /// Valid only for synchronization records related to CUDA Events.
    pub cuda_event_sync_id: CudaEventSyncId,
    /// The return value for the synchronization record.
    /// Use [`crate::activity::set_all_sync_records_enabled`] to enable or disable collection of synchronization records with return value being non-zero.
    /// This is a raw CUDA driver result code.
    pub return_value: u32,
}

impl ActivitySynchronization {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivitySynchronization2) -> Result<Self> {
        Ok(Self {
            synchronization_type: ActivitySynchronizationType::try_from(value.type_)?,
            start: value.start,
            end: value.end,
            correlation_id: CorrelationId::from(value.correlationId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            cuda_event_id: CudaEventId::from(value.cudaEventId),
            cuda_event_sync_id: CudaEventSyncId::from(value.cudaEventSyncId),
            return_value: value.returnValue,
        })
    }
}

/// Decoded CUDA graph execution interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityGraphTrace {
    /// The correlation ID of the graph launch.
    /// Each graph launch is assigned a unique correlation ID that is identical to the correlation ID in the driver API activity record that launched the graph.
    pub correlation_id: CorrelationId,
    /// The start timestamp for the graph execution, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the graph.
    pub start: u64,
    /// The end timestamp for the graph execution, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the graph.
    pub end: u64,
    /// The ID of the device where the first node of the graph is executed.
    /// If this is `INT_MAX`, then the start is on the host.
    pub device_id: DeviceId,
    /// The unique ID of the graph that is launched.
    pub graph_id: GraphId,
    /// The ID of the context where the first node of the graph is executed.
    /// If this is `INT_MAX`, then the start is on the host.
    pub context_id: ContextId,
    /// The ID of the stream where the graph is being launched.
    pub stream_id: StreamId,
    /// The ID of the device where last node of the graph is executed.
    pub end_device_id: DeviceId,
    /// The ID of the context where the last node of the graph is executed.
    pub end_context_id: ContextId,
}

impl ActivityGraphTrace {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityGraphTrace2) -> Self {
        Self {
            correlation_id: CorrelationId::from(value.correlationId),
            start: value.start,
            end: value.end,
            device_id: DeviceId::from(value.deviceId),
            graph_id: GraphId::from(value.graphId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            end_device_id: DeviceId::from(value.endDeviceId),
            end_context_id: ContextId::from(value.endContextId),
        }
    }
}

/// Decoded device graph execution interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityDeviceGraphTrace {
    /// The ID of the device where the first node of the graph is executed.
    pub device_id: DeviceId,
    /// The start timestamp for the graph execution, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the graph.
    pub start: u64,
    /// The end timestamp for the graph execution, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the graph.
    pub end: u64,
    /// The unique ID of the graph that is launched.
    pub graph_id: GraphId,
    /// The unique ID of the graph that has launched this graph.
    pub launcher_graph_id: GraphId,
    /// The type of launch.
    /// See [`DeviceGraphLaunchMode`].
    pub device_launch_mode: DeviceGraphLaunchMode,
    /// The ID of the context where the first node of the graph is executed.
    pub context_id: ContextId,
    /// The ID of the stream where the graph is being launched.
    pub stream_id: StreamId,
}

impl ActivityDeviceGraphTrace {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityDeviceGraphTrace) -> Self {
        Self {
            device_id: DeviceId::from(value.deviceId),
            start: value.start,
            end: value.end,
            graph_id: GraphId::from(value.graphId),
            launcher_graph_id: GraphId::from(value.launcherGraphId),
            device_launch_mode: DeviceGraphLaunchMode::try_from(value.deviceLaunchMode)
                .unwrap_or(DeviceGraphLaunchMode::Invalid),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
        }
    }
}

/// Decoded execution of a CUDA graph host node on the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityGraphHostNode {
    /// The ID of the stream that waits for the graph host node to finish.
    pub stream_id: StreamId,
    /// The ID of the CUDA context to which the waiting CUDA stream belongs.
    pub context_id: ContextId,
    /// The ID of the CUDA device to which the CUDA context and stream belong.
    /// The graph host node is executing on the CPU, but it is associated with the CUDA device through the CUDA context and stream.
    pub device_id: DeviceId,
    /// The correlation ID of the graph host node operation.
    /// Each operation is assigned a unique correlation ID that is identical to the correlation ID in the driver API activity record that launched the operation.
    pub correlation_id: CorrelationId,
    /// The unique ID of the graph that executed this host node through graph launch.
    pub graph_id: GraphId,
    /// The unique ID of the graph node that executed this host node through graph launch.
    pub graph_node_id: GraphNodeId,
    /// The ID of the process where the host node is executing.
    pub process_id: ProcessId,
    /// The ID of the thread where the host node is executing.
    pub thread_id: ThreadId,
    /// The start timestamp.
    /// A value of 0 means the start time is unknown.
    pub start: u64,
    /// The end timestamp.
    /// A value of 0 means the end time is unknown.
    pub end: u64,
}

impl ActivityGraphHostNode {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityGraphHostNode) -> Self {
        Self {
            stream_id: StreamId::from(value.streamId),
            context_id: ContextId::from(value.contextId),
            device_id: DeviceId::from(value.deviceId),
            correlation_id: CorrelationId::from(value.correlationId),
            graph_id: GraphId::from(value.graphId),
            graph_node_id: GraphNodeId::from(value.graphNodeId),
            process_id: ProcessId::from(value.processId),
            thread_id: ThreadId::from(value.threadId),
            start: value.start,
            end: value.end,
        }
    }
}

/// Decoded host launch activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityHostLaunch {
    /// The ID of the CUDA context to which the waiting CUDA stream belongs.
    pub stream_id: StreamId,
    /// The ID of the CUDA context to which the waiting CUDA stream belongs.
    pub context_id: ContextId,
    /// The ID of the CUDA device to which the CUDA context and stream belong.
    /// The host function is executing on the CPU, but it is associated with the CUDA device through the CUDA context and stream.
    pub device_id: DeviceId,
    /// The correlation ID of the host launch operation.
    /// Each operation is assigned a unique correlation ID that is identical to the correlation ID in the driver API activity record that launched the operation.
    pub correlation_id: CorrelationId,
    /// The ID of the process where the host function is executing.
    pub process_id: ProcessId,
    /// The ID of the thread where the host function is executing.
    pub thread_id: ThreadId,
    /// The start timestamp.
    /// A value of 0 means the start time is unknown.
    pub start: u64,
    /// The end timestamp.
    /// A value of 0 means the end time is unknown.
    pub end: u64,
}

impl ActivityHostLaunch {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityHostLaunch) -> Self {
        Self {
            stream_id: StreamId::from(value.streamId),
            context_id: ContextId::from(value.contextId),
            device_id: DeviceId::from(value.deviceId),
            correlation_id: CorrelationId::from(value.correlationId),
            process_id: ProcessId::from(value.processId),
            thread_id: ThreadId::from(value.threadId),
            start: value.start,
            end: value.end,
        }
    }
}

/// Decoded CUDA compute-engine context switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityComputeEngineContextSwitch {
    /// The ID of the CUDA context.
    pub context_id: ContextId,
    /// The timestamp at which the [`ComputeEngineContextSwitchOperationType`] occurs.
    pub timestamp: u64,
    /// The type of the Compute Engine Context switch operation.
    /// [`ComputeEngineContextSwitchOperationType::Start`] indicates the start of the context switch operation.
    /// [`ComputeEngineContextSwitchOperationType::End`] indicates the end of the context switch operation.
    pub operation_type: ComputeEngineContextSwitchOperationType,
}

impl ActivityComputeEngineContextSwitch {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityComputeEngineCtxSwitch) -> Result<Self> {
        Ok(Self {
            context_id: ContextId::from(value.contextId),
            timestamp: value.timestamp,
            operation_type: ComputeEngineContextSwitchOperationType::try_from(value.operationType)?,
        })
    }
}

/// Decoded activity record for a green context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityGreenContext {
    /// The context ID of the green context.
    pub context_id: ContextId,
    /// The ID of the parent context.
    pub parent_context_id: ContextId,
    /// The device ID associated with the green context.
    pub device_id: DeviceId,
    /// The number of TPCs allocated to the green context.
    pub num_tpcs: u32,
    /// The number of multiprocessors (SMs) allocated to the green context.
    pub num_multiprocessors: u16,
    /// Allocated logical texture-processing cluster (TPC) identifiers as 32-bit bitmask words.
    ///
    /// Only the first `logical_tpc_mask_size` words are valid; unused entries are zero.
    /// Word `j` covers TPC identifiers `32 * j..32 * (j + 1)`.
    /// For example, a first word of 66 selects TPCs 1 and 6.
    /// The 32-word array can represent up to 1,024 TPCs.
    pub logical_tpc_mask: [u32; 32],
    /// Number of valid 32-bit words in `logical_tpc_mask`.
    pub logical_tpc_mask_size: u8,
    /// The work queue resource ID associated with the green context.
    /// This is a unique identifier for the work queue resource used by this green context.
    pub workqueue_resource_id: u64,
    /// The concurrency limit for the work queue associated with the green context.
    /// This defines the maximum number of concurrent operations allowed on this work queue.
    pub workqueue_concurrency_limit: u32,
    /// The sharing scope for the work queue associated with the green context.
    /// This defines how the work queue can be shared across different contexts or processes.
    /// See the CUDA `CUdevWorkqueueConfigScope` values for possible sharing scopes.
    pub workqueue_sharing_scope: u32,
    /// The number of HW channels assigned to this green context's work queue.
    /// This reflects the actual channel assignment at green context creation time.
    /// A value of 0 indicates that channel information is not available.
    pub workqueue_channel_count: u32,
    /// Array of HW channel IDs assigned to this green context's work queue.
    /// Valid entries are specified by workqueueChannelCount; unused entries are zeroed.
    /// The array supports up to 32 channel IDs.
    pub workqueue_channel_ids: [u32; 32],
}

impl ActivityGreenContext {
    /// Array of HW channel IDs assigned to this green context's work queue.
    /// Valid entries are specified by `workqueue_channel_count`.
    #[must_use]
    pub fn workqueue_channels(&self) -> &[u32] {
        &self.workqueue_channel_ids
            [..(self.workqueue_channel_count as usize).min(self.workqueue_channel_ids.len())]
    }

    pub(crate) fn from_raw(value: &sys::CUpti_ActivityGreenContext3) -> Self {
        Self {
            workqueue_resource_id: value.workqueueResourceId,
            workqueue_concurrency_limit: value.workqueueConcurrencyLimit,
            workqueue_sharing_scope: value.workqueueSharingScope,
            workqueue_channel_count: value.workqueueChannelCount,
            workqueue_channel_ids: value.workqueueChannelIds,
            context_id: ContextId::from(value.contextId),
            parent_context_id: ContextId::from(value.parentContextId),
            device_id: DeviceId::from(value.deviceId),
            num_tpcs: value.numTpcs,
            num_multiprocessors: value.numMultiprocessors,
            logical_tpc_mask: value.logicalTpcMask,
            logical_tpc_mask_size: value.logicalTpcMaskSize,
        }
    }

    /// Returns the valid prefix of the logical TPC bitmask.
    ///
    /// The native word count is clamped to the 32-word storage capacity.
    #[must_use]
    pub fn logical_tpc_mask_words(&self) -> &[u32] {
        let len = usize::from(self.logical_tpc_mask_size).min(self.logical_tpc_mask.len());
        &self.logical_tpc_mask[..len]
    }

    /// Returns whether `logical_tpc_id` is set in the valid bitmask prefix.
    ///
    /// Returns `false` when the identifier falls outside the reported mask.
    #[must_use]
    pub fn has_logical_tpc(&self, logical_tpc_id: u32) -> bool {
        let word = logical_tpc_id / 32;
        let bit = logical_tpc_id % 32;
        self.logical_tpc_mask_words()
            .get(word as usize)
            .is_some_and(|mask| (mask & (1u32 << bit)) != 0)
    }
}

/// Process, thread, device, context, or stream identity decoded from the native object-kind tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityObjectId {
    /// Unrecognized or unsupported object kind.
    Unknown,
    /// Process identity.
    Process {
        /// Process identifier.
        process_id: ProcessId,
    },
    /// Thread identity, including its process.
    Thread {
        /// Process identifier.
        process_id: ProcessId,
        /// Thread identifier within the process.
        thread_id: ThreadId,
    },
    /// CUDA device identity.
    Device {
        /// CUDA device identifier.
        device_id: DeviceId,
    },
    /// CUDA context identity, including its device.
    Context {
        /// CUDA device identifier.
        device_id: DeviceId,
        /// CUDA context identifier on the device.
        context_id: ContextId,
    },
    /// CUDA stream identity, including its context and device.
    Stream {
        /// CUDA device identifier.
        device_id: DeviceId,
        /// CUDA context identifier on the device.
        context_id: ContextId,
        /// CUDA stream identifier in the context.
        stream_id: StreamId,
    },
}

impl ActivityObjectId {
    pub(crate) fn from_raw(
        kind: ActivityObjectKind,
        value: sys::CUpti_ActivityObjectKindId,
    ) -> Self {
        match kind {
            ActivityObjectKind::Process => {
                // SAFETY: the type tag matched above selects this union field.
                let value = unsafe { value.pt };
                Self::Process {
                    process_id: ProcessId::from(value.processId),
                }
            }
            ActivityObjectKind::Thread => {
                // SAFETY: the type tag matched above selects this union field.
                let value = unsafe { value.pt };
                Self::Thread {
                    process_id: ProcessId::from(value.processId),
                    thread_id: ThreadId::from(value.threadId),
                }
            }
            ActivityObjectKind::Device => {
                // SAFETY: the type tag matched above selects this union field.
                let value = unsafe { value.dcs };
                Self::Device {
                    device_id: DeviceId::from(value.deviceId),
                }
            }
            ActivityObjectKind::Context => {
                // SAFETY: the type tag matched above selects this union field.
                let value = unsafe { value.dcs };
                Self::Context {
                    device_id: DeviceId::from(value.deviceId),
                    context_id: ContextId::from(value.contextId),
                }
            }
            ActivityObjectKind::Stream => {
                // SAFETY: the type tag matched above selects this union field.
                let value = unsafe { value.dcs };
                Self::Stream {
                    device_id: DeviceId::from(value.deviceId),
                    context_id: ContextId::from(value.contextId),
                    stream_id: StreamId::from(value.streamId),
                }
            }
            _ => Self::Unknown,
        }
    }
}

/// Decoded activity record providing a name.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityName<'a> {
    /// The kind of activity object being named.
    pub object_kind: ActivityObjectKind,
    /// Activity object identifier, decoded according to `object_kind`.
    pub object_id: ActivityObjectId,
    /// Name associated with the recorded object.
    /// `None` represents a null native pointer.
    pub name: Option<&'a CStr>,
}

impl ActivityName<'_> {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityName) -> Result<Self> {
        let object_kind = ActivityObjectKind::try_from(value.objectKind)?;
        let name = if value.name.is_null() {
            None
        } else {
            // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
            Some(unsafe { CStr::from_ptr(value.name) })
        };
        Ok(Self {
            object_kind,
            object_id: ActivityObjectId::from_raw(object_kind, value.objectId),
            name,
        })
    }
}

/// Decoded instantaneous, start, or end marker.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMarker<'a> {
    /// The flags associated with the marker.
    pub flags: ActivityFlags,
    /// The timestamp for the marker, in nanoseconds.
    /// A value of 0 indicates that timestamp information could not be collected for the marker.
    pub timestamp: u64,
    /// The marker ID.
    pub id: ActivityMarkerId,
    /// The kind of activity object associated with this marker.
    pub object_kind: ActivityObjectKind,
    /// Activity object identifier, decoded according to `object_kind`.
    pub object_id: ActivityObjectId,
    /// Marker name for an instantaneous or start marker; `None` for an end marker.
    pub name: Option<&'a CStr>,
    /// Marker domain name; `None` selects the default domain.
    pub domain: Option<&'a CStr>,
}

impl ActivityMarker<'_> {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityMarker2) -> Result<Self> {
        let object_kind = ActivityObjectKind::try_from(value.objectKind)?;
        let name = if value.name.is_null() {
            None
        } else {
            // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
            Some(unsafe { CStr::from_ptr(value.name) })
        };
        let domain = if value.domain.is_null() {
            None
        } else {
            // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
            Some(unsafe { CStr::from_ptr(value.domain) })
        };
        Ok(Self {
            flags: ActivityFlags::from_bits_truncate(value.flags.0),
            timestamp: value.timestamp,
            id: ActivityMarkerId::from(value.id),
            object_kind,
            object_id: ActivityObjectId::from_raw(object_kind, value.objectId),
            name,
            domain,
        })
    }
}

/// Correlation between a CUDA activity and an external record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityExternalCorrelation {
    /// The kind of external API this record correlated to.
    pub external_kind: ExternalCorrelationKind,
    /// The correlation ID of the associated non-CUDA API record.
    /// The exact field in the associated external record depends on [`Self::external_kind`].
    pub external_id: ExternalCorrelationId,
    /// The correlation ID of the associated CUDA driver or runtime API record.
    pub correlation_id: CorrelationId,
}

impl ActivityExternalCorrelation {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityExternalCorrelation) -> Result<Self> {
        Ok(Self {
            external_kind: ExternalCorrelationKind::try_from(value.externalKind)?,
            external_id: ExternalCorrelationId::from(value.externalId),
            correlation_id: CorrelationId::from(value.correlationId),
        })
    }
}

/// Decoded activity record providing detailed information for a marker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivityMarkerPayload {
    /// The flags associated with the marker.
    pub flags: ActivityFlags,
    /// The marker ID.
    pub id: ActivityMarkerId,
    /// Defines the payload format for the value associated with the marker.
    pub payload_kind: MetricValueKind,
    /// The payload value.
    pub payload: MetricValue,
    /// The color for the marker.
    pub color: u32,
    /// The category for the marker.
    pub category: u32,
    /// CUPTI maintained domain id required for NVTX extended payloads.
    /// To parse the payload correctly, the domain id must be used to identify the payload attributes as they are domain specific.
    pub cupti_domain_id: CuptiDomainId,
}

impl ActivityMarkerPayload {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityMarkerData2) -> Result<Self> {
        let payload_kind = MetricValueKind::try_from(value.payloadKind)?;
        Ok(Self {
            flags: ActivityFlags::from_bits_truncate(value.flags.0),
            id: ActivityMarkerId::from(value.id),
            payload_kind,
            payload: metric_value_from_raw_kind(payload_kind, value.payload)?,
            color: value.color,
            category: value.category,
            cupti_domain_id: CuptiDomainId::from(value.cuptiDomainId),
        })
    }
}

fn metric_value_from_raw_kind(
    kind: MetricValueKind,
    value: sys::CUpti_MetricValue,
) -> Result<MetricValue> {
    // SAFETY: every field of this union is a plain number, so reading the raw 64-bit pattern is always valid.
    Ok(unsafe {
        match kind {
            MetricValueKind::Double => MetricValue::Double(value.metricValueDouble),
            MetricValueKind::Uint64 | MetricValueKind::ForceInt => {
                MetricValue::Uint64(value.metricValueUint64)
            }
            MetricValueKind::Percent => MetricValue::Percent(value.metricValuePercent),
            MetricValueKind::Throughput => MetricValue::Throughput(value.metricValueThroughput),
            MetricValueKind::Int64 => MetricValue::Int64(value.metricValueInt64),
            MetricValueKind::UtilizationLevel => MetricValue::UtilizationLevel(
                MetricValueUtilizationLevel::try_from(value.metricValueUtilizationLevel)?,
            ),
            MetricValueKind::NvtxExtendedPayload => {
                MetricValue::NvtxExtendedPayload(value.metricValueNvtxExtendedPayload)
            }
        }
    })
}

/// Decoded CUDA event activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityCudaEvent {
    /// The correlation ID of the API to which this result is associated.
    pub correlation_id: CorrelationId,
    /// The ID of the context where the event was recorded.
    pub context_id: ContextId,
    /// The compute stream where the event was recorded.
    pub stream_id: StreamId,
    /// A unique event ID to identify the event record.
    pub event_id: CudaEventId,
    /// The ID of the device where the event was recorded.
    pub device_id: DeviceId,
    /// The device-side timestamp on CUDA event record.
    /// Timestamp is in nanoseconds.
    /// Collection of this field is disabled by default.
    /// It can be enabled by calling [`crate::activity::set_cuda_event_device_timestamps_enabled`].
    pub device_timestamp: u64,
    /// A unique ID to associate event synchronization records with the latest CUDA Event record.
    pub cuda_event_sync_id: CudaEventSyncId,
}

impl ActivityCudaEvent {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityCudaEvent2) -> Self {
        Self {
            correlation_id: CorrelationId::from(value.correlationId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            event_id: CudaEventId::from(value.eventId),
            device_id: DeviceId::from(value.deviceId),
            device_timestamp: value.deviceTimestamp,
            cuda_event_sync_id: CudaEventSyncId::from(value.cudaEventSyncId),
        }
    }
}

/// Decoded activity record for a context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityContext {
    /// The context ID.
    pub context_id: ContextId,
    /// The device ID.
    pub device_id: DeviceId,
    /// The compute API kind.
    pub compute_api_kind: ActivityComputeApiKind,
    /// Identifier of the default stream in this context.
    pub null_stream_id: StreamId,
    /// The ID of the parent context.
    /// It would be 0 if context does not have parent.
    pub parent_context_id: ContextId,
    /// Whether this is a CUDA green context.
    pub is_green_context: bool,
    /// Number of multiprocessors assigned to the green context.
    /// Valid only when `is_green_context` is `true`.
    pub num_multiprocessors: u16,
    /// This field indicates the CIG mode.
    pub cig_mode: ContextCigMode,
    /// The ID of the process associated with the context.
    pub process_id: ProcessId,
}

impl ActivityContext {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityContext4) -> Result<Self> {
        Ok(Self {
            context_id: ContextId::from(value.contextId),
            device_id: DeviceId::from(value.deviceId),
            compute_api_kind: ActivityComputeApiKind::try_from(u32::from(value.computeApiKind))
                .unwrap_or(ActivityComputeApiKind::Unknown),
            null_stream_id: StreamId::from(value.nullStreamId),
            parent_context_id: ContextId::from(value.parentContextId),
            is_green_context: value.isGreenContext != 0,
            num_multiprocessors: value.numMultiprocessors,
            cig_mode: ContextCigMode::try_from(value.cigMode)?,
            process_id: ProcessId::from(value.processId),
        })
    }
}

/// Decoded OpenMP activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOpenMultiProcessing {
    /// `OpenMP` event kind.
    pub event_kind: OpenMultiProcessingEventKind,
    /// Version number.
    pub version: u32,
    /// Identifier of the thread associated with the OpenMP event.
    pub thread_id: ThreadId,
    /// CUPTI start timestamp.
    pub start: u64,
    /// CUPTI end timestamp.
    pub end: u64,
    /// The ID of the process where the OpenMP activity is executing.
    pub process_id: ProcessId,
    /// The ID of the thread where the OpenMP activity is executing.
    pub cu_thread_id: ThreadId,
}

impl ActivityOpenMultiProcessing {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityOpenMp) -> Result<Self> {
        Ok(Self {
            event_kind: OpenMultiProcessingEventKind::try_from(value.eventKind)?,
            version: value.version,
            thread_id: ThreadId::from(value.threadId),
            start: value.start,
            end: value.end,
            process_id: ProcessId::from(value.cuProcessId),
            cu_thread_id: ThreadId::from(value.cuThreadId),
        })
    }
}

/// Common metadata for `OpenACC` data, launch, and other activity records.
///
/// Optional strings borrow the source activity record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOpenAccBase<'a> {
    /// Kind of `OpenACC` event.
    pub event_kind: OpenAccEventKind,
    /// Enclosing `OpenACC` construct.
    /// PGI runtimes older than 16.1 report an unknown construct.
    pub parent_construct: OpenAccConstructKind,
    /// Native record version number.
    pub version: u32,
    /// Whether the event is implicit, such as a wait implied by a synchronous data construct.
    pub implicit: bool,
    /// Device type reported by the `OpenACC` runtime.
    pub device_type: OpenAccDeviceTypeId,
    /// Device number reported by the `OpenACC` runtime.
    pub device_number: OpenAccDeviceNumber,
    /// Identifier of the thread associated with the event.
    pub thread_id: ThreadId,
    /// Value of the directive's `async` clause.
    pub asynchronous_id: OpenAccAsynchronousId,
    /// Internal asynchronous queue number used by the runtime.
    pub asynchronous_map: u64,
    /// Source line of the directive or construct start; zero means unknown.
    pub line_number: u32,
    /// Source line of the construct end; zero means unknown.
    pub end_line_number: u32,
    /// First source line of the enclosing function; zero means unknown.
    pub function_line_number: u32,
    /// Last source line of the enclosing function; zero means unknown.
    pub function_end_line_number: u32,
    /// CUPTI start timestamp for the event.
    pub start: u64,
    /// CUPTI end timestamp for the event.
    pub end: u64,
    /// Associated CUDA device identifier; valid only for the NVIDIA device type.
    pub cuda_device_id: DeviceId,
    /// Associated CUDA context identifier; valid only for the NVIDIA device type.
    pub cuda_context_id: ContextId,
    /// Associated CUDA stream identifier; valid only for the NVIDIA device type.
    pub cuda_stream_id: StreamId,
    /// Identifier of the process executing the `OpenACC` activity.
    pub process_id: ProcessId,
    /// Identifier of the thread executing the `OpenACC` activity.
    pub cuda_thread_id: ThreadId,
    /// `OpenACC` correlation identifier; valid only for the NVIDIA device type.
    ///
    /// A nonzero identifier matches the preceding external-correlation record of the `OpenACC` kind.
    pub external_id: ExternalCorrelationId,
    /// Borrowed source file name or path; `None` means the native pointer is null.
    pub source_file: Option<&'a CStr>,
    /// Borrowed name of the function containing the event; `None` means the native pointer is null.
    pub function_name: Option<&'a CStr>,
}

macro_rules! open_acc_base_from_raw {
    ($value:expr) => {{
        ActivityOpenAccBase {
            event_kind: OpenAccEventKind::try_from($value.eventKind)?,
            parent_construct: OpenAccConstructKind::try_from($value.parentConstruct)?,
            version: $value.version,
            implicit: $value.implicit != 0,
            device_type: OpenAccDeviceTypeId::from($value.deviceType),
            device_number: OpenAccDeviceNumber::from($value.deviceNumber),
            thread_id: ThreadId::from($value.threadId),
            asynchronous_id: OpenAccAsynchronousId::from($value.async_),
            asynchronous_map: $value.asyncMap,
            line_number: $value.lineNo,
            end_line_number: $value.endLineNo,
            function_line_number: $value.funcLineNo,
            function_end_line_number: $value.funcEndLineNo,
            start: $value.start,
            end: $value.end,
            cuda_device_id: DeviceId::from($value.cuDeviceId),
            cuda_context_id: ContextId::from($value.cuContextId),
            cuda_stream_id: StreamId::from($value.cuStreamId),
            process_id: ProcessId::from($value.cuProcessId),
            cuda_thread_id: ThreadId::from($value.cuThreadId),
            external_id: ExternalCorrelationId::from(u64::from($value.externalId)),
            source_file: optional_cstr($value.srcFile),
            function_name: optional_cstr($value.funcName),
        }
    }};
}

/// Decoded activity record for `OpenACC` data.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOpenAccTransfer<'a> {
    /// Common `OpenACC` metadata for this data-transfer event.
    pub base: ActivityOpenAccBase<'a>,
    /// Number of bytes.
    pub bytes: u64,
    /// Host pointer if available.
    pub host_ptr: u64,
    /// Device pointer if available.
    pub device_ptr: u64,
    /// Name of the variable associated with the event, when known.
    /// `None` represents a null native pointer.
    pub variable_name: Option<&'a CStr>,
}

impl ActivityOpenAccTransfer<'_> {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityOpenAccData) -> Result<Self> {
        Ok(Self {
            base: open_acc_base_from_raw!(value),
            bytes: value.bytes,
            host_ptr: value.hostPtr,
            device_ptr: value.devicePtr,
            variable_name: optional_cstr(value.varName),
        })
    }
}

/// Decoded activity record for `OpenACC` launch.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOpenAccLaunch<'a> {
    /// Common `OpenACC` metadata for this kernel-launch event.
    pub base: ActivityOpenAccBase<'a>,
    /// The number of gangs created for this kernel launch.
    pub num_gangs: u64,
    /// The number of workers created for this kernel launch.
    pub num_workers: u64,
    /// The number of vector lanes created for this kernel launch.
    pub vector_length: u64,
    /// Name of the launched kernel, when known.
    /// `None` represents a null native pointer.
    pub kernel_name: Option<&'a CStr>,
}

impl ActivityOpenAccLaunch<'_> {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityOpenAccLaunch) -> Result<Self> {
        Ok(Self {
            base: open_acc_base_from_raw!(value),
            num_gangs: value.numGangs,
            num_workers: value.numWorkers,
            vector_length: value.vectorLength,
            kernel_name: optional_cstr(value.kernelName),
        })
    }
}

/// An `OpenACC` event represented entirely by the common activity metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOpenAccOther<'a> {
    /// Common metadata describing this `OpenACC` event.
    pub base: ActivityOpenAccBase<'a>,
}

impl ActivityOpenAccOther<'_> {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityOpenAccOther) -> Result<Self> {
        Ok(Self {
            base: open_acc_base_from_raw!(value),
        })
    }
}

/// Decoded preemption of a CUDA Dynamic Parallelism kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPreemption {
    /// kind of the preemption.
    pub preemption_kind: ActivityPreemptionKind,
    /// The timestamp of the preemption, in nanoseconds.
    /// A value of 0 indicates that timestamp information could not be collected for the preemption.
    pub timestamp: u64,
    /// The grid-id of the block that is preempted.
    pub grid_id: GridId,
    /// Coordinates of the thread block affected by the preemption.
    pub block: Dim3,
}

impl ActivityPreemption {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityPreemption) -> Result<Self> {
        Ok(Self {
            preemption_kind: ActivityPreemptionKind::try_from(value.preemptionKind)?,
            timestamp: value.timestamp,
            grid_id: GridId::from(value.gridId),
            block: Dim3 {
                x: value.blockX,
                y: value.blockY,
                z: value.blockZ,
            },
        })
    }
}

/// Decoded confidential-computing encryption-key rotation event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityConfidentialComputeRotation {
    /// Type of event [`ConfidentialComputeRotationEventType`].
    pub event_type: ConfidentialComputeRotationEventType,
    /// Device ID.
    pub device_id: DeviceId,
    /// Context ID.
    pub context_id: ContextId,
    /// Channel ID.
    pub channel_id: ChannelId,
    /// Channel Type [`ChannelType`].
    pub channel_type: ChannelType,
    /// Timestamp in nanoseconds.
    pub timestamp: u64,
}

impl ActivityConfidentialComputeRotation {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityConfidentialComputeRotation) -> Result<Self> {
        Ok(Self {
            event_type: ConfidentialComputeRotationEventType::try_from(value.eventType)?,
            device_id: DeviceId::from(value.deviceId),
            context_id: ContextId::from(value.contextId),
            channel_id: ChannelId::from(value.channelId),
            channel_type: ChannelType::try_from(value.channelType)?,
            timestamp: value.timestamp,
        })
    }
}

/// Decoded activity record for CUPTI and driver overheads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOverhead {
    /// The kind of overhead, CUPTI, DRIVER, COMPILER etc.
    pub overhead_kind: ActivityOverheadKind,
    /// The kind of activity object that the overhead is associated with.
    pub object_kind: ActivityObjectKind,
    /// Activity object identifier, decoded according to `object_kind`.
    pub object_id: ActivityObjectId,
    /// The start timestamp for the overhead, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the overhead.
    pub start: u64,
    /// The end timestamp for the overhead, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the overhead.
    pub end: u64,
    /// The correlation ID of the overhead operation to which records belong to.
    /// This ID is identical to the correlation ID in the driver or runtime API activity record that launched the overhead operation.
    /// In some cases, it can be zero, such as for [`ActivityOverheadKind::BufferFlush`] records.
    pub correlation_id: CorrelationId,
    /// Whether the native record includes an additional overhead payload.
    /// This flag records its presence without exposing its pointer.
    pub has_overhead_data: bool,
}

impl ActivityOverhead {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityOverhead3) -> Result<Self> {
        let object_kind = ActivityObjectKind::try_from(value.objectKind)?;
        Ok(Self {
            overhead_kind: ActivityOverheadKind::try_from(value.overheadKind)?,
            object_kind,
            object_id: ActivityObjectId::from_raw(object_kind, value.objectId),
            start: value.start,
            end: value.end,
            correlation_id: CorrelationId::from(value.correlationId),
            has_overhead_data: !value.overheadData.is_null(),
        })
    }
}
