use std::{
    alloc::{Layout, LayoutError, alloc_zeroed, dealloc, handle_alloc_error},
    ffi,
    ffi::CStr,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
    mem::{self, size_of},
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    ptr,
    result::Result as StdResult,
    slice,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use singe_cuda::dim::Dim3;

use crate::{
    callback::Subscriber,
    context::Context,
    error::{Error, Result},
    sys, try_ffi,
    types::{
        ActivityAttribute, ActivityAttributeSetting, ActivityAttributeValue,
        ActivityComputeApiKind, ActivityConfig, ActivityEnvironmentKind, ActivityFlags,
        ActivityFlushFlag, ActivityJitEntryType, ActivityJitOperationType, ActivityKind,
        ActivityLaunchType, ActivityMarkerId, ActivityMemoryCopyKind, ActivityMemoryKind,
        ActivityMemoryOperationType, ActivityMemoryPoolOperationType, ActivityMemoryPoolType,
        ActivityObjectKind, ActivityOverheadKind, ActivityPartitionedGlobalCacheConfig,
        ActivityPcSamplingConfig, ActivityPcSamplingStallReason, ActivityPreemptionKind,
        ActivityStreamFlag, ActivitySynchronizationType, ActivityThreadIdType,
        ActivityUnifiedMemoryAccessType, ActivityUnifiedMemoryCounterConfig,
        ActivityUnifiedMemoryCounterKind, ActivityUnifiedMemoryMigrationCause,
        ActivityUnifiedMemoryRemoteMapCause, CallbackId, ChannelId, ChannelType,
        ComputeEngineContextSwitchOperationType, ComputeInstanceId,
        ConfidentialComputeRotationEventType, ContextCigMode, ContextId, CorrelationId, CudaDevice,
        CudaDeviceAttribute, CudaEventId, CudaEventSyncId, CuptiDomainId, DeviceAttribute,
        DeviceGraphLaunchMode, DeviceId, DeviceType, EnvironmentClocksThrottleReasons,
        EventDomainId, EventId, ExternalCorrelationId, ExternalCorrelationKind,
        FunctionExecutionModel, FunctionId, FunctionSharedMemoryLimitConfig, GpuInstanceId,
        GraphId, GraphNodeId, GridId, JitOperationCorrelationId, LinkFlags, MetricId, MetricValue,
        MetricValueKind, MetricValueUtilizationLevel, ModuleId, MultipleSubscriberState,
        NonUniformMemoryAccessId, NpuDomainId, OpenAccAsynchronousId, OpenAccConstructKind,
        OpenAccDeviceNumber, OpenAccDeviceTypeId, OpenAccEventKind, OpenMultiProcessingEventKind,
        PcieBridgeId, PcieBusId, PcieDeviceType, PcieDomainId, PcieHardwareDeviceId, PcieVendorId,
        ProcessId, SourceLocatorId, StreamId, ThreadId, UnifiedMemoryProcessorId,
    },
    utility::{to_u32, to_u64, to_usize},
};

type TimestampCallback = dyn FnMut() -> u64 + Send + 'static;

static TIMESTAMP_CALLBACK: Mutex<Option<TimestampCallbackState>> = Mutex::new(None);

type ActivityBufferCompleteCallback = dyn for<'a> FnMut(ActivityBuffer<'a>) + Send + 'static;

static ACTIVITY_BUFFER_CALLBACKS: Mutex<Option<ActivityBufferCallbackState>> = Mutex::new(None);
static NEXT_ACTIVITY_CALLBACK_OWNER_ID: AtomicU64 = AtomicU64::new(1);

const ACTIVITY_BUFFER_ALIGNMENT: usize = 8;

struct ActivityBufferCallbackState {
    owner_id: Option<u64>,
    buffer_size: usize,
    max_num_records: usize,
    completed: Box<ActivityBufferCompleteCallback>,
}

struct TimestampCallbackState {
    owner_id: u64,
    callback: Box<TimestampCallback>,
}

/// Borrowed bytes from a completed CUPTI activity buffer.
///
/// Record views and borrowed payload fields cannot outlive the source buffer.
#[derive(Debug, Clone, Copy)]
pub struct ActivityBuffer<'a> {
    bytes: &'a [u8],
}

impl<'a> ActivityBuffer<'a> {
    /// Returns the valid bytes in the completed activity buffer.
    #[must_use]
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Returns the number of valid activity-buffer bytes.
    #[must_use]
    pub const fn len(self) -> usize {
        self.bytes.len()
    }

    /// Returns whether the activity buffer contains no valid bytes.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.bytes.is_empty()
    }

    /// Returns an iterator over records in this completed activity buffer.
    ///
    /// Records and their borrowed payload fields retain the buffer lifetime. Native parsing errors are yielded by the iterator.
    #[must_use]
    pub fn records(self) -> ActivityRecords<'a> {
        ActivityRecords {
            buffer: self.bytes.as_ptr().cast_mut(),
            valid_size_bytes: self.bytes.len() as u64,
            current: ptr::null_mut(),
            finished: false,
            _marker: PhantomData,
        }
    }
}

/// Owns a byte copy of a completed CUPTI activity buffer.
///
/// Decoded records borrow this owner. Native pointer-backed strings and binary images are not deep-copied with the record bytes.
#[derive(Debug, Clone)]
pub struct ActivityBufferOwned {
    bytes: Vec<u8>,
}

impl ActivityBufferOwned {
    /// Returns the raw CUPTI activity bytes captured from a completed buffer.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Transfers ownership of the copied activity-record bytes.
    ///
    /// Native pointer-backed data referenced by records is not deep-copied.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Returns the number of valid activity-buffer bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns whether the activity buffer contains no valid bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Returns an iterator over records in this completed activity buffer.
    ///
    /// Records and their borrowed payload fields retain the buffer lifetime. Native parsing errors are yielded by the iterator.
    #[must_use]
    pub fn records(&self) -> ActivityRecords<'_> {
        ActivityRecords {
            buffer: self.bytes.as_ptr().cast_mut(),
            valid_size_bytes: self.bytes.len() as u64,
            current: ptr::null_mut(),
            finished: false,
            _marker: PhantomData,
        }
    }
}

/// Guard for the current Rust activity-buffer callbacks.
///
/// Dropping the guard clears its callbacks if this registration still owns them. Dropping a replaced registration leaves the newer callbacks installed.
#[derive(Debug)]
pub struct ActivityCallbackRegistration {
    owner_id: u64,
}

impl Drop for ActivityCallbackRegistration {
    fn drop(&mut self) {
        clear_activity_callbacks_if_owner(self.owner_id);
    }
}

/// Guard for the current Rust timestamp callback.
///
/// Dropping the guard removes its callback only if this registration still owns it.
#[derive(Debug)]
pub struct TimestampCallbackRegistration {
    owner_id: u64,
}

impl Drop for TimestampCallbackRegistration {
    fn drop(&mut self) {
        let Ok(mut callback) = TIMESTAMP_CALLBACK.lock() else {
            return;
        };
        if callback
            .as_ref()
            .is_some_and(|callback| callback.owner_id == self.owner_id)
        {
            *callback = None;
        }
    }
}

/// Borrowed CUPTI activity record with kind-checked payload decoders.
///
/// Copies retain the same borrow lifetime. Unrecognized kinds remain available through [`Self::raw_kind`].
#[derive(Debug, Clone, Copy)]
pub struct ActivityRecord<'a> {
    ptr: *const sys::CUpti_Activity,
    _marker: PhantomData<&'a sys::CUpti_Activity>,
}

impl<'a> ActivityRecord<'a> {
    /// Returns the record kind, or `None` for kinds newer than these bindings.
    #[must_use]
    pub fn kind(self) -> Option<ActivityKind> {
        ActivityKind::try_from(self.raw_kind()).ok()
    }

    /// Returns the raw `CUpti_ActivityKind` value of the record header.
    #[must_use]
    pub fn raw_kind(self) -> u32 {
        // SAFETY: every CUPTI record starts with a `u32` kind header, and the record stays
        // borrowed for `'a`; `read_unaligned` makes no alignment assumption.
        unsafe { ptr::read_unaligned(self.ptr.cast::<u32>()) }
    }

    /// Decodes [`ActivityMemoryDecompress`] for [`ActivityKind::MemoryDecompress`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn memory_decompress(self) -> Option<ActivityMemoryDecompress> {
        if self.kind() != Some(ActivityKind::MemoryDecompress) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMemDecompress>()) };
        Some(ActivityMemoryDecompress::from_raw(record))
    }

    /// Decodes [`ActivityMemoryCopy`] for [`ActivityKind::MemoryCopy`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn memory_copy(self) -> Option<ActivityMemoryCopy> {
        if self.kind() != Some(ActivityKind::MemoryCopy) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMemcpy7>()) };
        Some(ActivityMemoryCopy::from_raw(record))
    }

    /// Decodes [`ActivityPeerMemoryCopy`] for [`ActivityKind::PeerMemoryCopy`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn peer_memory_copy(self) -> Option<ActivityPeerMemoryCopy> {
        if self.kind() != Some(ActivityKind::PeerMemoryCopy) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMemcpyPtoP4>()) };
        Some(ActivityPeerMemoryCopy::from_raw(record))
    }

    /// Decodes [`ActivityMemorySet`] for [`ActivityKind::MemorySet`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn memory_set(self) -> Option<ActivityMemorySet> {
        if self.kind() != Some(ActivityKind::MemorySet) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMemset5>()) };
        Some(ActivityMemorySet::from_raw(record))
    }

    /// Decodes [`ActivityKernel`] for [`ActivityKind::Kernel`], [`ActivityKind::ConcurrentKernel`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn kernel(self) -> Option<ActivityKernel<'a>> {
        if !matches!(
            self.kind(),
            Some(ActivityKind::Kernel | ActivityKind::ConcurrentKernel)
        ) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityKernel13>()) };
        Some(ActivityKernel::from_raw(record))
    }

    /// Decodes [`ActivityCudaDynamicParallelismKernel`] for [`ActivityKind::CudaDynamicParallelismKernel`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn cuda_dynamic_parallelism_kernel(
        self,
    ) -> Option<ActivityCudaDynamicParallelismKernel<'a>> {
        if self.kind() != Some(ActivityKind::CudaDynamicParallelismKernel) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityCdpKernel>()) };
        Some(ActivityCudaDynamicParallelismKernel::from_raw(record))
    }

    /// Decodes [`ActivityApi`] for [`ActivityKind::Driver`], [`ActivityKind::Runtime`], [`ActivityKind::InternalLaunchApi`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn api(self) -> Option<ActivityApi> {
        if !matches!(
            self.kind(),
            Some(ActivityKind::Driver | ActivityKind::Runtime | ActivityKind::InternalLaunchApi)
        ) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityAPI>()) };
        Some(ActivityApi::from_raw(record))
    }

    /// Decodes [`ActivityStream`] for [`ActivityKind::Stream`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn stream(self) -> Option<ActivityStream> {
        if self.kind() != Some(ActivityKind::Stream) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityStream>()) };
        Some(ActivityStream::from_raw(record))
    }

    /// Decodes [`ActivitySynchronization`] for [`ActivityKind::Synchronization`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn synchronization(self) -> Option<ActivitySynchronization> {
        if self.kind() != Some(ActivityKind::Synchronization) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivitySynchronization2>()) };
        Some(ActivitySynchronization::from_raw(record))
    }

    /// Decodes [`ActivityMemory`] for [`ActivityKind::Memory2`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn memory(self) -> Option<ActivityMemory<'a>> {
        if self.kind() != Some(ActivityKind::Memory2) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMemory4>()) };
        Some(ActivityMemory::from_raw(record))
    }

    /// Decodes [`ActivityLegacyMemory`] for [`ActivityKind::Memory`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn legacy_memory(self) -> Option<ActivityLegacyMemory<'a>> {
        if self.kind() != Some(ActivityKind::Memory) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMemory>()) };
        Some(ActivityLegacyMemory::from_raw(record))
    }

    /// Decodes [`ActivityMemoryPool`] for [`ActivityKind::MemoryPool`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn memory_pool(self) -> Option<ActivityMemoryPool> {
        if self.kind() != Some(ActivityKind::MemoryPool) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMemoryPool3>()) };
        Some(ActivityMemoryPool::from_raw(record))
    }

    /// Decodes [`ActivityGraphTrace`] for [`ActivityKind::GraphTrace`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn graph_trace(self) -> Option<ActivityGraphTrace> {
        if self.kind() != Some(ActivityKind::GraphTrace) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityGraphTrace2>()) };
        Some(ActivityGraphTrace::from_raw(record))
    }

    /// Decodes [`ActivityDeviceGraphTrace`] for [`ActivityKind::DeviceGraphTrace`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn device_graph_trace(self) -> Option<ActivityDeviceGraphTrace> {
        if self.kind() != Some(ActivityKind::DeviceGraphTrace) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityDeviceGraphTrace>()) };
        Some(ActivityDeviceGraphTrace::from_raw(record))
    }

    /// Decodes [`ActivityGraphHostNode`] for [`ActivityKind::GraphHostNode`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn graph_host_node(self) -> Option<ActivityGraphHostNode> {
        if self.kind() != Some(ActivityKind::GraphHostNode) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityGraphHostNode>()) };
        Some(ActivityGraphHostNode::from_raw(record))
    }

    /// Decodes [`ActivityHostLaunch`] for [`ActivityKind::HostLaunch`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn host_launch(self) -> Option<ActivityHostLaunch> {
        if self.kind() != Some(ActivityKind::HostLaunch) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityHostLaunch>()) };
        Some(ActivityHostLaunch::from_raw(record))
    }

    /// Decodes [`ActivityComputeEngineContextSwitch`] for [`ActivityKind::ComputeEngineCtxSwitch`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn compute_engine_context_switch(self) -> Option<ActivityComputeEngineContextSwitch> {
        if self.kind() != Some(ActivityKind::ComputeEngineCtxSwitch) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityComputeEngineCtxSwitch>())
        };
        Some(ActivityComputeEngineContextSwitch::from_raw(record))
    }

    /// Decodes [`ActivityGreenContext`] for [`ActivityKind::GreenContext`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn green_context(self) -> Option<ActivityGreenContext> {
        if self.kind() != Some(ActivityKind::GreenContext) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityGreenContext3>()) };
        Some(ActivityGreenContext::from_raw(&record))
    }

    /// Decodes [`ActivityName`] for [`ActivityKind::Name`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn name(self) -> Option<ActivityName<'a>> {
        if self.kind() != Some(ActivityKind::Name) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityName>()) };
        Some(ActivityName::from_raw(record))
    }

    /// Decodes [`ActivityMarker`] for [`ActivityKind::Marker`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn marker(self) -> Option<ActivityMarker<'a>> {
        if self.kind() != Some(ActivityKind::Marker) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMarker2>()) };
        Some(ActivityMarker::from_raw(record))
    }

    /// Decodes [`ActivityExternalCorrelation`] for [`ActivityKind::ExternalCorrelation`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn external_correlation(self) -> Option<ActivityExternalCorrelation> {
        if self.kind() != Some(ActivityKind::ExternalCorrelation) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityExternalCorrelation>())
        };
        Some(ActivityExternalCorrelation::from_raw(record))
    }

    /// Decodes [`ActivityMarkerPayload`] for [`ActivityKind::MarkerData`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn marker_data(self) -> Option<ActivityMarkerPayload> {
        if self.kind() != Some(ActivityKind::MarkerData) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMarkerData2>()) };
        Some(ActivityMarkerPayload::from_raw(record))
    }

    /// Decodes [`ActivityCudaEvent`] for [`ActivityKind::CudaEvent`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn cuda_event(self) -> Option<ActivityCudaEvent> {
        if self.kind() != Some(ActivityKind::CudaEvent) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityCudaEvent2>()) };
        Some(ActivityCudaEvent::from_raw(record))
    }

    /// Decodes [`ActivityContext`] for [`ActivityKind::Context`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn context(self) -> Option<ActivityContext> {
        if self.kind() != Some(ActivityKind::Context) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityContext4>()) };
        Some(ActivityContext::from_raw(record))
    }

    /// Decodes [`ActivityEvent`] for [`ActivityKind::Event`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn event(self) -> Option<ActivityEvent> {
        if self.kind() != Some(ActivityKind::Event) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityEvent>()) };
        Some(ActivityEvent::from_raw(record))
    }

    /// Decodes [`ActivityEventInstance`] for [`ActivityKind::EventInstance`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn event_instance(self) -> Option<ActivityEventInstance> {
        if self.kind() != Some(ActivityKind::EventInstance) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityEventInstance>()) };
        Some(ActivityEventInstance::from_raw(record))
    }

    /// Decodes [`ActivityMetric`] for [`ActivityKind::Metric`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn metric(self) -> Option<ActivityMetric> {
        if self.kind() != Some(ActivityKind::Metric) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMetric>()) };
        Some(ActivityMetric::from_raw(record))
    }

    /// Decodes [`ActivityMetricInstance`] for [`ActivityKind::MetricInstance`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn metric_instance(self) -> Option<ActivityMetricInstance> {
        if self.kind() != Some(ActivityKind::MetricInstance) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMetricInstance>()) };
        Some(ActivityMetricInstance::from_raw(record))
    }

    /// Decodes [`ActivityInstantaneousEvent`] for [`ActivityKind::InstantaneousEvent`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn instantaneous_event(self) -> Option<ActivityInstantaneousEvent> {
        if self.kind() != Some(ActivityKind::InstantaneousEvent) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityInstantaneousEvent>())
        };
        Some(ActivityInstantaneousEvent::from_raw(record))
    }

    /// Decodes [`ActivityInstantaneousEventInstance`] for [`ActivityKind::InstantaneousEventInstance`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn instantaneous_event_instance(self) -> Option<ActivityInstantaneousEventInstance> {
        if self.kind() != Some(ActivityKind::InstantaneousEventInstance) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe {
            ptr::read_unaligned(
                self.ptr
                    .cast::<sys::CUpti_ActivityInstantaneousEventInstance>(),
            )
        };
        Some(ActivityInstantaneousEventInstance::from_raw(record))
    }

    /// Decodes [`ActivityInstantaneousMetric`] for [`ActivityKind::InstantaneousMetric`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn instantaneous_metric(self) -> Option<ActivityInstantaneousMetric> {
        if self.kind() != Some(ActivityKind::InstantaneousMetric) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityInstantaneousMetric>())
        };
        Some(ActivityInstantaneousMetric::from_raw(record))
    }

    /// Decodes [`ActivityInstantaneousMetricInstance`] for [`ActivityKind::InstantaneousMetricInstance`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn instantaneous_metric_instance(self) -> Option<ActivityInstantaneousMetricInstance> {
        if self.kind() != Some(ActivityKind::InstantaneousMetricInstance) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe {
            ptr::read_unaligned(
                self.ptr
                    .cast::<sys::CUpti_ActivityInstantaneousMetricInstance>(),
            )
        };
        Some(ActivityInstantaneousMetricInstance::from_raw(record))
    }

    /// Decodes [`ActivityGlobalAccess`] for [`ActivityKind::GlobalAccess`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn global_access(self) -> Option<ActivityGlobalAccess> {
        if self.kind() != Some(ActivityKind::GlobalAccess) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityGlobalAccess3>()) };
        Some(ActivityGlobalAccess::from_raw(record))
    }

    /// Decodes [`ActivityBranch`] for [`ActivityKind::Branch`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn branch(self) -> Option<ActivityBranch> {
        if self.kind() != Some(ActivityKind::Branch) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityBranch2>()) };
        Some(ActivityBranch::from_raw(record))
    }

    /// Decodes [`ActivityInstructionExecution`] for [`ActivityKind::InstructionExecution`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn instruction_execution(self) -> Option<ActivityInstructionExecution> {
        if self.kind() != Some(ActivityKind::InstructionExecution) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityInstructionExecution>())
        };
        Some(ActivityInstructionExecution::from_raw(record))
    }

    /// Decodes [`ActivityPcSampling`] for [`ActivityKind::PcSampling`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn pc_sampling(self) -> Option<ActivityPcSampling> {
        if self.kind() != Some(ActivityKind::PcSampling) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityPCSampling3>()) };
        Some(ActivityPcSampling::from_raw(record))
    }

    /// Decodes [`ActivityPcSamplingRecordInfo`] for [`ActivityKind::PcSamplingRecordInfo`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn pc_sampling_record_info(self) -> Option<ActivityPcSamplingRecordInfo> {
        if self.kind() != Some(ActivityKind::PcSamplingRecordInfo) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityPCSamplingRecordInfo>())
        };
        Some(ActivityPcSamplingRecordInfo::from_raw(record))
    }

    /// Decodes [`ActivityUnifiedMemoryCounter`] for [`ActivityKind::UnifiedMemoryCounter`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn unified_memory_counter(self) -> Option<ActivityUnifiedMemoryCounter> {
        if self.kind() != Some(ActivityKind::UnifiedMemoryCounter) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityUnifiedMemoryCounter3>())
        };
        Some(ActivityUnifiedMemoryCounter::from_raw(record))
    }

    /// Decodes [`ActivityDeviceAttribute`] for [`ActivityKind::DeviceAttribute`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn device_attribute(self) -> Option<ActivityDeviceAttribute> {
        if self.kind() != Some(ActivityKind::DeviceAttribute) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityDeviceAttribute>()) };
        Some(ActivityDeviceAttribute::from_raw(record))
    }

    /// Decodes [`ActivityEnvironment`] for [`ActivityKind::Environment`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn environment(self) -> Option<ActivityEnvironment> {
        if self.kind() != Some(ActivityKind::Environment) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityEnvironment>()) };
        Some(ActivityEnvironment::from_raw(record))
    }

    /// Decodes [`ActivityDevice`] for [`ActivityKind::Device`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn device(self) -> Option<ActivityDevice<'a>> {
        if self.kind() != Some(ActivityKind::Device) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityDevice6>()) };
        Some(ActivityDevice::from_raw(record))
    }

    /// Decodes [`ActivityJit`] for [`ActivityKind::Jit`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn jit(self) -> Option<ActivityJit<'a>> {
        if self.kind() != Some(ActivityKind::Jit) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityJit2>()) };
        Some(ActivityJit::from_raw(record))
    }

    /// Decodes [`ActivityNvLink`] for [`ActivityKind::NvLink`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn nvlink(self) -> Option<ActivityNvLink<'a>> {
        if self.kind() != Some(ActivityKind::NvLink) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityNvLink5>()) };
        Some(ActivityNvLink::from_raw(record))
    }

    /// Decodes [`ActivityPcie`] for [`ActivityKind::Pcie`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn pcie(self) -> Option<ActivityPcie> {
        if self.kind() != Some(ActivityKind::Pcie) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityPcie>()) };
        Some(ActivityPcie::from_raw(record))
    }

    /// Decodes [`ActivityInstructionCorrelation`] for [`ActivityKind::InstructionCorrelation`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn instruction_correlation(self) -> Option<ActivityInstructionCorrelation> {
        if self.kind() != Some(ActivityKind::InstructionCorrelation) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityInstructionCorrelation>())
        };
        Some(ActivityInstructionCorrelation::from_raw(record))
    }

    /// Decodes [`ActivityOpenMultiProcessing`] for [`ActivityKind::OpenMultiProcessing`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn open_multi_processing(self) -> Option<ActivityOpenMultiProcessing> {
        if self.kind() != Some(ActivityKind::OpenMultiProcessing) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityOpenMp>()) };
        Some(ActivityOpenMultiProcessing::from_raw(record))
    }

    /// Decodes [`ActivityOpenAccTransfer`] for [`ActivityKind::OpenAccData`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn open_acc_data(self) -> Option<ActivityOpenAccTransfer<'a>> {
        if self.kind() != Some(ActivityKind::OpenAccData) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityOpenAccData>()) };
        Some(ActivityOpenAccTransfer::from_raw(record))
    }

    /// Decodes [`ActivityOpenAccLaunch`] for [`ActivityKind::OpenAccLaunch`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn open_acc_launch(self) -> Option<ActivityOpenAccLaunch<'a>> {
        if self.kind() != Some(ActivityKind::OpenAccLaunch) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityOpenAccLaunch>()) };
        Some(ActivityOpenAccLaunch::from_raw(record))
    }

    /// Decodes [`ActivityOpenAccOther`] for [`ActivityKind::OpenAccOther`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn open_acc_other(self) -> Option<ActivityOpenAccOther<'a>> {
        if self.kind() != Some(ActivityKind::OpenAccOther) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityOpenAccOther>()) };
        Some(ActivityOpenAccOther::from_raw(record))
    }

    /// Decodes [`ActivityPreemption`] for [`ActivityKind::Preemption`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn preemption(self) -> Option<ActivityPreemption> {
        if self.kind() != Some(ActivityKind::Preemption) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityPreemption>()) };
        Some(ActivityPreemption::from_raw(record))
    }

    /// Decodes [`ActivityFunction`] for [`ActivityKind::Function`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn function(self) -> Option<ActivityFunction<'a>> {
        if self.kind() != Some(ActivityKind::Function) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityFunction>()) };
        Some(ActivityFunction::from_raw(record))
    }

    /// Decodes [`ActivityModule`] for [`ActivityKind::Module`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn module(self) -> Option<ActivityModule<'a>> {
        if self.kind() != Some(ActivityKind::Module) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityModule>()) };
        Some(ActivityModule::from_raw(record))
    }

    /// Decodes [`ActivitySharedAccess`] for [`ActivityKind::SharedAccess`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn shared_access(self) -> Option<ActivitySharedAccess> {
        if self.kind() != Some(ActivityKind::SharedAccess) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivitySharedAccess>()) };
        Some(ActivitySharedAccess::from_raw(record))
    }

    /// Decodes [`ActivityConfidentialComputeRotation`] for [`ActivityKind::ConfidentialComputeRotation`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn confidential_compute_rotation(self) -> Option<ActivityConfidentialComputeRotation> {
        if self.kind() != Some(ActivityKind::ConfidentialComputeRotation) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record = unsafe {
            ptr::read_unaligned(
                self.ptr
                    .cast::<sys::CUpti_ActivityConfidentialComputeRotation>(),
            )
        };
        Some(ActivityConfidentialComputeRotation::from_raw(record))
    }

    /// Decodes [`ActivitySourceLocator`] for [`ActivityKind::SourceLocator`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn source_locator(self) -> Option<ActivitySourceLocator<'a>> {
        if self.kind() != Some(ActivityKind::SourceLocator) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivitySourceLocator>()) };
        Some(ActivitySourceLocator::from_raw(record))
    }

    /// Decodes [`ActivityOverhead`] for [`ActivityKind::Overhead`] records.
    ///
    /// Returns `None` for other activity kinds. Borrowed payload fields retain the source record lifetime.
    #[must_use]
    pub fn overhead(self) -> Option<ActivityOverhead> {
        if self.kind() != Some(ActivityKind::Overhead) {
            return None;
        }
        // SAFETY: the kind check above means CUPTI wrote a complete record of this type at `self.ptr`; `read_unaligned` makes no alignment assumption.
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityOverhead3>()) };
        Some(ActivityOverhead::from_raw(record))
    }

    /// Decodes this record into a tagged payload.
    ///
    /// Kinds without a supported decoder produce [`ActivityRecordPayload::Unsupported`] with the original raw kind. Borrowed payload fields retain the source record lifetime.
    #[expect(
        clippy::too_many_lines,
        reason = "activity decoding enumerates the supported native activity record kinds"
    )]
    pub fn decode(self) -> ActivityRecordPayload<'a> {
        let raw_kind = self.raw_kind();
        let Some(kind) = self.kind() else {
            return ActivityRecordPayload::Unsupported { raw_kind };
        };

        macro_rules! decode_record {
            ($method:ident, $variant:ident) => {
                self.$method()
                    .map(ActivityRecordPayload::$variant)
                    .unwrap_or(ActivityRecordPayload::Unsupported { raw_kind })
            };
        }

        match kind {
            ActivityKind::Driver | ActivityKind::Runtime | ActivityKind::InternalLaunchApi => {
                decode_record!(api, Api)
            }
            ActivityKind::MemoryCopy => decode_record!(memory_copy, MemoryCopy),
            ActivityKind::PeerMemoryCopy => decode_record!(peer_memory_copy, PeerMemoryCopy),
            ActivityKind::MemorySet => decode_record!(memory_set, MemorySet),
            ActivityKind::MemoryDecompress => decode_record!(memory_decompress, MemoryDecompress),
            ActivityKind::Kernel | ActivityKind::ConcurrentKernel => decode_record!(kernel, Kernel),
            ActivityKind::CudaDynamicParallelismKernel => decode_record!(
                cuda_dynamic_parallelism_kernel,
                CudaDynamicParallelismKernel
            ),
            ActivityKind::Stream => decode_record!(stream, Stream),
            ActivityKind::Synchronization => decode_record!(synchronization, Synchronization),
            ActivityKind::Memory2 => decode_record!(memory, Memory),
            ActivityKind::Memory => decode_record!(legacy_memory, LegacyMemory),
            ActivityKind::MemoryPool => decode_record!(memory_pool, MemoryPool),
            ActivityKind::GraphTrace => decode_record!(graph_trace, GraphTrace),
            ActivityKind::DeviceGraphTrace => {
                decode_record!(device_graph_trace, DeviceGraphTrace)
            }
            ActivityKind::GraphHostNode => decode_record!(graph_host_node, GraphHostNode),
            ActivityKind::HostLaunch => decode_record!(host_launch, HostLaunch),
            ActivityKind::ComputeEngineCtxSwitch => {
                decode_record!(compute_engine_context_switch, ComputeEngineContextSwitch)
            }
            ActivityKind::GreenContext => decode_record!(green_context, GreenContext),
            ActivityKind::Name => decode_record!(name, Name),
            ActivityKind::Marker => decode_record!(marker, Marker),
            ActivityKind::ExternalCorrelation => {
                decode_record!(external_correlation, ExternalCorrelation)
            }
            ActivityKind::MarkerData => decode_record!(marker_data, MarkerData),
            ActivityKind::CudaEvent => decode_record!(cuda_event, CudaEvent),
            ActivityKind::Context => decode_record!(context, Context),
            ActivityKind::Event => decode_record!(event, Event),
            ActivityKind::EventInstance => decode_record!(event_instance, EventInstance),
            ActivityKind::Metric => decode_record!(metric, Metric),
            ActivityKind::MetricInstance => decode_record!(metric_instance, MetricInstance),
            ActivityKind::InstantaneousEvent => {
                decode_record!(instantaneous_event, InstantaneousEvent)
            }
            ActivityKind::InstantaneousEventInstance => {
                decode_record!(instantaneous_event_instance, InstantaneousEventInstance)
            }
            ActivityKind::InstantaneousMetric => {
                decode_record!(instantaneous_metric, InstantaneousMetric)
            }
            ActivityKind::InstantaneousMetricInstance => {
                decode_record!(instantaneous_metric_instance, InstantaneousMetricInstance)
            }
            ActivityKind::GlobalAccess => decode_record!(global_access, GlobalAccess),
            ActivityKind::Branch => decode_record!(branch, Branch),
            ActivityKind::InstructionExecution => {
                decode_record!(instruction_execution, InstructionExecution)
            }
            ActivityKind::PcSampling => {
                decode_record!(pc_sampling, PcSampling)
            }
            ActivityKind::PcSamplingRecordInfo => {
                decode_record!(pc_sampling_record_info, PcSamplingRecordInfo)
            }
            ActivityKind::UnifiedMemoryCounter => {
                decode_record!(unified_memory_counter, UnifiedMemoryCounter)
            }
            ActivityKind::DeviceAttribute => decode_record!(device_attribute, DeviceAttribute),
            ActivityKind::Environment => decode_record!(environment, Environment),
            ActivityKind::Device => decode_record!(device, Device),
            ActivityKind::Jit => decode_record!(jit, Jit),
            ActivityKind::NvLink => decode_record!(nvlink, NvLink),
            ActivityKind::Pcie => decode_record!(pcie, Pcie),
            ActivityKind::InstructionCorrelation => {
                decode_record!(instruction_correlation, InstructionCorrelation)
            }
            ActivityKind::OpenMultiProcessing => {
                decode_record!(open_multi_processing, OpenMultiProcessing)
            }
            ActivityKind::OpenAccData => {
                decode_record!(open_acc_data, OpenAccData)
            }
            ActivityKind::OpenAccLaunch => {
                decode_record!(open_acc_launch, OpenAccLaunch)
            }
            ActivityKind::OpenAccOther => {
                decode_record!(open_acc_other, OpenAccOther)
            }
            ActivityKind::Preemption => decode_record!(preemption, Preemption),
            ActivityKind::Function => decode_record!(function, Function),
            ActivityKind::Module => decode_record!(module, Module),
            ActivityKind::SharedAccess => decode_record!(shared_access, SharedAccess),
            ActivityKind::ConfidentialComputeRotation => {
                decode_record!(confidential_compute_rotation, ConfidentialComputeRotation)
            }
            ActivityKind::SourceLocator => decode_record!(source_locator, SourceLocator),
            ActivityKind::Overhead => decode_record!(overhead, Overhead),
            _ => ActivityRecordPayload::Unsupported { raw_kind },
        }
    }
}

fn memory_copy_kind_from_raw(value: u8) -> ActivityMemoryCopyKind {
    ActivityMemoryCopyKind::try_from(u32::from(value)).unwrap_or(ActivityMemoryCopyKind::Unknown)
}

fn memory_kind_from_raw(value: impl Into<u32>) -> ActivityMemoryKind {
    ActivityMemoryKind::try_from(value.into()).unwrap_or(ActivityMemoryKind::Unknown)
}

fn partitioned_global_cache_config_from_raw(
    value: sys::CUpti_ActivityPartitionedGlobalCacheConfig,
) -> ActivityPartitionedGlobalCacheConfig {
    ActivityPartitionedGlobalCacheConfig::try_from(value as u32)
        .unwrap_or(ActivityPartitionedGlobalCacheConfig::Unknown)
}

fn launch_type_from_raw(value: u8) -> ActivityLaunchType {
    ActivityLaunchType::try_from(u32::from(value)).unwrap_or(ActivityLaunchType::Regular)
}

fn function_shared_memory_limit_config_from_raw(
    value: sys::CUpti_FuncShmemLimitConfig,
) -> FunctionSharedMemoryLimitConfig {
    FunctionSharedMemoryLimitConfig::try_from(value as u32)
        .unwrap_or(FunctionSharedMemoryLimitConfig::Default)
}

/// Decoded activity record payload, selected by its native activity kind.
///
/// Borrowed fields retain the source record lifetime; copying this enum does not extend that lifetime.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ActivityRecordPayload<'a> {
    /// Activity kind for which this wrapper has no decoded payload.
    Api(ActivityApi),
    /// Activity kind for which this wrapper has no decoded payload.
    Kernel(ActivityKernel<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    CudaDynamicParallelismKernel(ActivityCudaDynamicParallelismKernel<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    MemoryCopy(ActivityMemoryCopy),
    /// Activity kind for which this wrapper has no decoded payload.
    PeerMemoryCopy(ActivityPeerMemoryCopy),
    /// Activity kind for which this wrapper has no decoded payload.
    MemorySet(ActivityMemorySet),
    /// Activity kind for which this wrapper has no decoded payload.
    MemoryDecompress(ActivityMemoryDecompress),
    /// Activity kind for which this wrapper has no decoded payload.
    Stream(ActivityStream),
    /// Activity kind for which this wrapper has no decoded payload.
    Synchronization(ActivitySynchronization),
    /// Activity kind for which this wrapper has no decoded payload.
    Memory(ActivityMemory<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    LegacyMemory(ActivityLegacyMemory<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    MemoryPool(ActivityMemoryPool),
    /// Activity kind for which this wrapper has no decoded payload.
    GraphTrace(ActivityGraphTrace),
    /// Activity kind for which this wrapper has no decoded payload.
    DeviceGraphTrace(ActivityDeviceGraphTrace),
    /// Activity kind for which this wrapper has no decoded payload.
    GraphHostNode(ActivityGraphHostNode),
    /// Activity kind for which this wrapper has no decoded payload.
    HostLaunch(ActivityHostLaunch),
    /// Activity kind for which this wrapper has no decoded payload.
    ComputeEngineContextSwitch(ActivityComputeEngineContextSwitch),
    /// Activity kind for which this wrapper has no decoded payload.
    GreenContext(ActivityGreenContext),
    /// Activity kind for which this wrapper has no decoded payload.
    Name(ActivityName<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    Marker(ActivityMarker<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    ExternalCorrelation(ActivityExternalCorrelation),
    /// Activity kind for which this wrapper has no decoded payload.
    MarkerData(ActivityMarkerPayload),
    /// Activity kind for which this wrapper has no decoded payload.
    CudaEvent(ActivityCudaEvent),
    /// Activity kind for which this wrapper has no decoded payload.
    Context(ActivityContext),
    /// Activity kind for which this wrapper has no decoded payload.
    Event(ActivityEvent),
    /// Activity kind for which this wrapper has no decoded payload.
    EventInstance(ActivityEventInstance),
    /// Activity kind for which this wrapper has no decoded payload.
    Metric(ActivityMetric),
    /// Activity kind for which this wrapper has no decoded payload.
    MetricInstance(ActivityMetricInstance),
    /// Activity kind for which this wrapper has no decoded payload.
    InstantaneousEvent(ActivityInstantaneousEvent),
    /// Activity kind for which this wrapper has no decoded payload.
    InstantaneousEventInstance(ActivityInstantaneousEventInstance),
    /// Activity kind for which this wrapper has no decoded payload.
    InstantaneousMetric(ActivityInstantaneousMetric),
    /// Activity kind for which this wrapper has no decoded payload.
    InstantaneousMetricInstance(ActivityInstantaneousMetricInstance),
    /// Activity kind for which this wrapper has no decoded payload.
    GlobalAccess(ActivityGlobalAccess),
    /// Activity kind for which this wrapper has no decoded payload.
    Branch(ActivityBranch),
    /// Activity kind for which this wrapper has no decoded payload.
    InstructionExecution(ActivityInstructionExecution),
    /// Activity kind for which this wrapper has no decoded payload.
    PcSampling(ActivityPcSampling),
    /// Activity kind for which this wrapper has no decoded payload.
    PcSamplingRecordInfo(ActivityPcSamplingRecordInfo),
    /// Activity kind for which this wrapper has no decoded payload.
    UnifiedMemoryCounter(ActivityUnifiedMemoryCounter),
    /// Activity kind for which this wrapper has no decoded payload.
    DeviceAttribute(ActivityDeviceAttribute),
    /// Activity kind for which this wrapper has no decoded payload.
    Environment(ActivityEnvironment),
    /// Activity kind for which this wrapper has no decoded payload.
    Device(ActivityDevice<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    Jit(ActivityJit<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    NvLink(ActivityNvLink<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    Pcie(ActivityPcie),
    /// Activity kind for which this wrapper has no decoded payload.
    InstructionCorrelation(ActivityInstructionCorrelation),
    /// Activity kind for which this wrapper has no decoded payload.
    OpenMultiProcessing(ActivityOpenMultiProcessing),
    /// Activity kind for which this wrapper has no decoded payload.
    OpenAccData(ActivityOpenAccTransfer<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    OpenAccLaunch(ActivityOpenAccLaunch<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    OpenAccOther(ActivityOpenAccOther<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    Preemption(ActivityPreemption),
    /// Activity kind for which this wrapper has no decoded payload.
    Function(ActivityFunction<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    Module(ActivityModule<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    SharedAccess(ActivitySharedAccess),
    /// Activity kind for which this wrapper has no decoded payload.
    ConfidentialComputeRotation(ActivityConfidentialComputeRotation),
    /// Activity kind for which this wrapper has no decoded payload.
    SourceLocator(ActivitySourceLocator<'a>),
    /// Activity kind for which this wrapper has no decoded payload.
    Overhead(ActivityOverhead),
    /// Activity kind for which this wrapper has no decoded payload.
    Unsupported {
        /// Unrecognized native activity-kind value, preserved without decoding its payload.
        raw_kind: u32,
    },
}

/// Decoded activity record for a driver or runtime API invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityApi {
    /// The activity record kind, must be [`ActivityKind::Driver`], [`ActivityKind::Runtime`], or [`ActivityKind::InternalLaunchApi`].
    pub kind: ActivityKind,
    /// The ID of the driver or runtime function.
    pub callback_id: CallbackId,
    /// The start timestamp for the function, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the function.
    pub start: u64,
    /// The end timestamp for the function, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the function.
    pub end: u64,
    /// The ID of the process where the driver or runtime CUDA function is executing.
    pub process_id: ProcessId,
    /// The ID of the thread where the driver or runtime CUDA function is executing.
    pub thread_id: ThreadId,
    /// The correlation ID of the driver or runtime CUDA function. Each function invocation is assigned a unique correlation ID that is identical to the correlation ID in the memcpy, memset, or kernel activity record that is associated with this function.
    pub correlation_id: CorrelationId,
    /// Native return status: a `CUresult` code for a driver call or a `cudaError_t` code for a runtime call.
    pub return_value: u32,
}

impl ActivityApi {
    fn from_raw(value: sys::CUpti_ActivityAPI) -> Self {
        Self {
            kind: ActivityKind::from(value.kind),
            callback_id: CallbackId::from(value.cbid),
            start: value.start,
            end: value.end,
            process_id: ProcessId::from(value.processId),
            thread_id: ThreadId::from(value.threadId),
            correlation_id: CorrelationId::from(value.correlationId),
            return_value: value.returnValue,
        }
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
    fn from_raw(value: sys::CUpti_ActivityStream) -> Self {
        Self {
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            priority: value.priority,
            flag: ActivityStreamFlag::from(value.flag),
            correlation_id: CorrelationId::from(value.correlationId),
        }
    }
}

/// Decoded CUDA synchronization activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivitySynchronization {
    /// Kind of CUDA synchronization operation recorded.
    pub synchronization_type: ActivitySynchronizationType,
    /// The start timestamp for the function, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the function.
    pub start: u64,
    /// The end timestamp for the function, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the function.
    pub end: u64,
    /// The correlation ID of the API to which this result is associated.
    pub correlation_id: CorrelationId,
    /// The ID of the context for which the synchronization API is called. In case of context synchronization API it is the context id for which the API is called. In case of stream/event synchronization it is the ID of the context where the stream/event was created.
    pub context_id: ContextId,
    /// The compute stream for which the synchronization API is called. A `CUPTI_SYNCHRONIZATION_INVALID_VALUE` value indicate the field is not applicable for this record. Not valid for `cuCtxSynchronize`, `cuEventSynchronize`.
    pub stream_id: StreamId,
    /// The event ID for which the synchronization API is called. A `CUPTI_SYNCHRONIZATION_INVALID_VALUE` value indicate the field is not applicable for this record. Not valid for `cuCtxSynchronize`, `cuStreamSynchronize`.
    pub cuda_event_id: CudaEventId,
    /// A unique ID to associate event synchronization records with the latest CUDA Event record. Similar field is added in [`ActivityCudaEvent`] to associate synchronization record to the CUDA Event record.
    ///
    /// The same CUDA event can be used multiple times, so the event id will not be unique to correlate the synchronization record with the latest CUDA Event record. This field will be unique and can be used to do the required correlation.
    ///
    /// A `CUPTI_SYNCHRONIZATION_INVALID_VALUE` value indicates that the field is not applicable for this record. Valid only for synchronization records related to CUDA Events.
    pub cuda_event_sync_id: CudaEventSyncId,
    /// The return value for the synchronization record. Use [`enable_all_sync_records`] API to enable/disable collection of synchronization records with return value being non-zero. This will be a `CUresult` value.
    pub return_value: u32,
}

impl ActivitySynchronization {
    fn from_raw(value: sys::CUpti_ActivitySynchronization2) -> Self {
        Self {
            synchronization_type: ActivitySynchronizationType::from(value.type_),
            start: value.start,
            end: value.end,
            correlation_id: CorrelationId::from(value.correlationId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            cuda_event_id: CudaEventId::from(value.cudaEventId),
            cuda_event_sync_id: CudaEventSyncId::from(value.cudaEventSyncId),
            return_value: value.returnValue,
        }
    }
}

/// Memory-pool configuration attached to a memory operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemoryPoolConfig {
    /// Native pool type used to decode `data`.
    pub pool_type: ActivityMemoryPoolType,
    /// The base address of the memory pool.
    pub address: u64,
    /// Tagged local, imported, or uninterpreted pool properties.
    pub data: ActivityMemoryPoolProperties,
}

/// Memory-pool properties decoded according to the native pool-type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityMemoryPoolProperties {
    /// Properties of a locally owned memory pool.
    Local {
        /// Memory-pool size in bytes.
        size: u64,
        /// Memory-pool release threshold in bytes.
        release_threshold: u64,
        /// Utilized memory-pool capacity in bytes.
        utilized_size: u64,
    },
    /// Properties of a memory pool imported from another process.
    Imported {
        /// Identifier of the process from which the pool was imported.
        process_id: ProcessId,
    },
    /// Raw union bits and ancillary values for an unrecognized or invalid pool type.
    ///
    /// The native pool tag does not establish a semantic interpretation for these values.
    Unknown {
        /// Preserved 64-bit pool union representation.
        raw_value: u64,
        /// Preserved native release-threshold field; validity depends on the unrecognized pool type.
        release_threshold: u64,
        /// Preserved native utilization field; validity depends on the unrecognized pool type.
        utilized_size: u64,
    },
}

impl ActivityMemoryPoolConfig {
    fn from_raw(value: sys::CUpti_ActivityMemory4__bindgen_ty_1) -> Self {
        let pool_type = ActivityMemoryPoolType::from(value.memoryPoolType);
        let data = match pool_type {
            ActivityMemoryPoolType::Local => ActivityMemoryPoolProperties::Local {
                // SAFETY: the type tag matched above selects this union field.
                size: unsafe { value.pool.size },
                release_threshold: value.releaseThreshold,
                utilized_size: value.utilizedSize,
            },
            ActivityMemoryPoolType::Imported => ActivityMemoryPoolProperties::Imported {
                // SAFETY: the type tag matched above selects this union field.
                process_id: ProcessId::from(unsafe { value.pool.processId }),
            },
            ActivityMemoryPoolType::Invalid | ActivityMemoryPoolType::ForceInt => {
                ActivityMemoryPoolProperties::Unknown {
                    // SAFETY: both union fields are `u64`, so reading `size` as raw bits is valid for any pool type.
                    raw_value: unsafe { value.pool.size },
                    release_threshold: value.releaseThreshold,
                    utilized_size: value.utilizedSize,
                }
            }
        };

        Self {
            pool_type,
            address: value.address,
            data,
        }
    }
}

/// Decoded memory allocation or release activity.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemory<'a> {
    /// The memory operation requested by the user, [`ActivityMemoryOperationType`].
    pub operation_type: ActivityMemoryOperationType,
    /// The memory kind requested by the user, [`ActivityMemoryKind`].
    pub memory_kind: ActivityMemoryKind,
    /// The correlation ID of the memory operation. Each memory operation is assigned a unique correlation ID that is identical to the correlation ID in the driver and runtime API activity record that launched the memory operation.
    pub correlation_id: CorrelationId,
    /// The virtual address of the allocation.
    pub address: u64,
    /// The number of bytes of memory allocated.
    pub bytes: u64,
    /// The start timestamp for the memory operation, in nanoseconds.
    pub timestamp: u64,
    /// The program counter of the memory operation.
    pub pc: u64,
    /// The ID of the process to which this record belongs to.
    pub process_id: ProcessId,
    /// The ID of the device where the memory operation is taking place.
    pub device_id: DeviceId,
    /// Context identifier; preserves `CUPTI_INVALID_CONTEXT_ID` when the native context is null.
    pub context_id: ContextId,
    /// The ID of the stream. If memory operation is not async, `stream_id` is set to `CUPTI_INVALID_STREAM_ID`.
    pub stream_id: StreamId,
    /// Variable name shared by records for the same symbol. `None` represents a null native pointer.
    pub name: Option<&'a CStr>,
    /// Whether the memory operation uses an asynchronous memory API.
    pub is_asynchronous: bool,
    /// The memory pool configuration used for the memory operations.
    pub memory_pool_config: ActivityMemoryPoolConfig,
    /// Shared object or binary from which the allocation request originates. `None` represents a null native pointer.
    pub source: Option<&'a CStr>,
}

impl ActivityMemory<'_> {
    fn from_raw(value: sys::CUpti_ActivityMemory4) -> Self {
        let name = if value.name.is_null() {
            None
        } else {
            // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
            Some(unsafe { CStr::from_ptr(value.name) })
        };
        let source = if value.source.is_null() {
            None
        } else {
            // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
            Some(unsafe { CStr::from_ptr(value.source) })
        };

        Self {
            operation_type: ActivityMemoryOperationType::from(value.memoryOperationType),
            memory_kind: ActivityMemoryKind::from(value.memoryKind),
            correlation_id: CorrelationId::from(value.correlationId),
            address: value.address,
            bytes: value.bytes,
            timestamp: value.timestamp,
            pc: value.PC,
            process_id: ProcessId::from(value.processId),
            device_id: DeviceId::from(value.deviceId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            name,
            is_asynchronous: value.isAsync != 0,
            memory_pool_config: ActivityMemoryPoolConfig::from_raw(value.memoryPoolConfig),
            source,
        }
    }
}

/// Decoded legacy memory-allocation lifetime record.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityLegacyMemory<'a> {
    /// The memory kind requested by the user.
    pub memory_kind: ActivityMemoryKind,
    /// The virtual address of the allocation.
    pub address: u64,
    /// The number of bytes of memory allocated.
    pub bytes: u64,
    /// The start timestamp for the memory operation, i.e. the time when memory was allocated, in nanoseconds.
    pub start: u64,
    /// The end timestamp for the memory operation, i.e. the time when memory was freed, in nanoseconds. This will be 0 if memory is not freed in the application.
    pub end: u64,
    /// The program counter of the allocation of memory.
    pub allocation_pc: u64,
    /// The program counter of the freeing of memory. This will be 0 if memory is not freed in the application.
    pub free_pc: u64,
    /// The ID of the process to which this record belongs to.
    pub process_id: ProcessId,
    /// The ID of the device where the memory allocation is taking place.
    pub device_id: DeviceId,
    /// Context identifier; preserves `CUPTI_INVALID_CONTEXT_ID` when the native context is null.
    pub context_id: ContextId,
    /// Variable name shared by records for the same symbol. `None` represents a null native pointer.
    pub name: Option<&'a CStr>,
}

impl ActivityLegacyMemory<'_> {
    fn from_raw(value: sys::CUpti_ActivityMemory) -> Self {
        Self {
            memory_kind: ActivityMemoryKind::from(value.memoryKind),
            address: value.address,
            bytes: value.bytes,
            start: value.start,
            end: value.end,
            allocation_pc: value.allocPC,
            free_pc: value.freePC,
            process_id: ProcessId::from(value.processId),
            device_id: DeviceId::from(value.deviceId),
            context_id: ContextId::from(value.contextId),
            name: optional_cstr(value.name),
        }
    }
}

/// Decoded memory-pool creation, destruction, or trimming activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemoryPool {
    /// The memory operation requested by the user, [`ActivityMemoryPoolOperationType`].
    pub operation_type: ActivityMemoryPoolOperationType,
    /// The type of the memory pool, [`ActivityMemoryPoolType`].
    pub pool_type: ActivityMemoryPoolType,
    /// The correlation ID of the memory pool operation. Each memory pool operation is assigned a unique correlation ID that is identical to the correlation ID in the driver and runtime API activity record that launched the memory operation.
    pub correlation_id: CorrelationId,
    /// The ID of the process to which this record belongs to.
    pub process_id: ProcessId,
    /// The ID of the device where the memory pool is created.
    pub device_id: DeviceId,
    /// The minimum bytes to keep of the memory pool. `min_bytes_to_keep` is valid for [`ActivityMemoryPoolOperationType::Trimmed`], [`ActivityMemoryPoolOperationType`].
    pub min_bytes_to_keep: u64,
    /// The virtual address of the allocation.
    pub address: u64,
    /// The size of the memory pool operation in bytes. `size` is valid for [`ActivityMemoryPoolType::Local`], [`ActivityMemoryPoolType`].
    pub size: u64,
    /// The release threshold of the memory pool. `release_threshold` is valid for [`ActivityMemoryPoolType::Local`], [`ActivityMemoryPoolType`].
    pub release_threshold: u64,
    /// The start timestamp for the memory operation, in nanoseconds.
    pub timestamp: u64,
    /// The utilized size of the memory pool. `utilized_size` is valid for [`ActivityMemoryPoolType::Local`], [`ActivityMemoryPoolType`].
    pub utilized_size: u64,
    /// Whether the pool uses managed memory (`true`) rather than pinned memory (`false`).
    pub is_managed_pool: bool,
}

impl ActivityMemoryPool {
    fn from_raw(value: sys::CUpti_ActivityMemoryPool3) -> Self {
        Self {
            operation_type: ActivityMemoryPoolOperationType::from(value.memoryPoolOperationType),
            pool_type: ActivityMemoryPoolType::from(value.memoryPoolType),
            correlation_id: CorrelationId::from(value.correlationId),
            process_id: ProcessId::from(value.processId),
            device_id: DeviceId::from(value.deviceId),
            min_bytes_to_keep: value.minBytesToKeep,
            address: value.address,
            size: value.size,
            release_threshold: value.releaseThreshold,
            timestamp: value.timestamp,
            utilized_size: value.utilizedSize,
            is_managed_pool: value.isManagedPool != 0,
        }
    }
}

/// Decoded CUDA graph execution interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityGraphTrace {
    /// The correlation ID of the graph launch. Each graph launch is assigned a unique correlation ID that is identical to the correlation ID in the driver API activity record that launched the graph.
    pub correlation_id: CorrelationId,
    /// The start timestamp for the graph execution, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the graph.
    pub start: u64,
    /// The end timestamp for the graph execution, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the graph.
    pub end: u64,
    /// The ID of the device where the first node of the graph is executed. If this is `INT_MAX`, then the start is on the host.
    pub device_id: DeviceId,
    /// The unique ID of the graph that is launched.
    pub graph_id: GraphId,
    /// The ID of the context where the first node of the graph is executed. If this is `INT_MAX`, then the start is on the host.
    pub context_id: ContextId,
    /// The ID of the stream where the graph is being launched.
    pub stream_id: StreamId,
    /// The ID of the device where last node of the graph is executed.
    pub end_device_id: DeviceId,
    /// The ID of the context where the last node of the graph is executed.
    pub end_context_id: ContextId,
}

impl ActivityGraphTrace {
    fn from_raw(value: sys::CUpti_ActivityGraphTrace2) -> Self {
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
    /// The start timestamp for the graph execution, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the graph.
    pub start: u64,
    /// The end timestamp for the graph execution, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the graph.
    pub end: u64,
    /// The unique ID of the graph that is launched.
    pub graph_id: GraphId,
    /// The unique ID of the graph that has launched this graph.
    pub launcher_graph_id: GraphId,
    /// The type of launch. See [`DeviceGraphLaunchMode`].
    pub device_launch_mode: DeviceGraphLaunchMode,
    /// The ID of the context where the first node of the graph is executed.
    pub context_id: ContextId,
    /// The ID of the stream where the graph is being launched.
    pub stream_id: StreamId,
}

impl ActivityDeviceGraphTrace {
    fn from_raw(value: sys::CUpti_ActivityDeviceGraphTrace) -> Self {
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
    /// The ID of the CUDA device to which the CUDA context and stream belong. The graph host node is executing on the CPU, but it is associated with the CUDA device through the CUDA context and stream.
    pub device_id: DeviceId,
    /// The correlation ID of the graph host node operation. Each operation is assigned a unique correlation ID that is identical to the correlation ID in the driver API activity record that launched the operation.
    pub correlation_id: CorrelationId,
    /// The unique ID of the graph that executed this host node through graph launch.
    pub graph_id: GraphId,
    /// The unique ID of the graph node that executed this host node through graph launch.
    pub graph_node_id: GraphNodeId,
    /// The ID of the process where the host node is executing.
    pub process_id: ProcessId,
    /// The ID of the thread where the host node is executing.
    pub thread_id: ThreadId,
    /// The start timestamp. A value of `CUPTI_TIMESTAMP_UNKNOWN` indicates that the start time is unknown.
    pub start: u64,
    /// The end timestamp. A value of `CUPTI_TIMESTAMP_UNKNOWN` indicates that the start time is unknown.
    pub end: u64,
}

impl ActivityGraphHostNode {
    fn from_raw(value: sys::CUpti_ActivityGraphHostNode) -> Self {
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
    /// The ID of the CUDA device to which the CUDA context and stream belong. The host function is executing on the CPU, but it is associated with the CUDA device through the CUDA context and stream.
    pub device_id: DeviceId,
    /// The correlation ID of the host launch operation. Each operation is assigned a unique correlation ID that is identical to the correlation ID in the driver API activity record that launched the operation.
    pub correlation_id: CorrelationId,
    /// The ID of the process where the host function is executing.
    pub process_id: ProcessId,
    /// The ID of the thread where the host function is executing.
    pub thread_id: ThreadId,
    /// The start timestamp. A value of `CUPTI_TIMESTAMP_UNKNOWN` indicates that the start time is unknown.
    pub start: u64,
    /// End timestamp; `CUPTI_TIMESTAMP_UNKNOWN` indicates that the end time is unavailable.
    pub end: u64,
}

impl ActivityHostLaunch {
    fn from_raw(value: sys::CUpti_ActivityHostLaunch) -> Self {
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
    /// The type of the Compute Engine Context switch operation. [`ComputeEngineContextSwitchOperationType::Start`] indicates the start of the context switch operation. [`ComputeEngineContextSwitchOperationType::End`] indicates the end of the context switch operation.
    pub operation_type: ComputeEngineContextSwitchOperationType,
}

impl ActivityComputeEngineContextSwitch {
    fn from_raw(value: sys::CUpti_ActivityComputeEngineCtxSwitch) -> Self {
        Self {
            context_id: ContextId::from(value.contextId),
            timestamp: value.timestamp,
            operation_type: ComputeEngineContextSwitchOperationType::from(value.operationType),
        }
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
    /// Only the first `logical_tpc_mask_size` words are valid; unused entries are zero. Word `j` covers TPC identifiers `32 * j..32 * (j + 1)`. For example, a first word of 66 selects TPCs 1 and 6. The 32-word array can represent up to 1,024 TPCs.
    pub logical_tpc_mask: [u32; 32],
    /// Number of valid 32-bit words in `logical_tpc_mask`.
    pub logical_tpc_mask_size: u8,
    /// The work queue resource ID associated with the green context.
    ///  This is a unique identifier for the work queue resource used by this green context.
    pub workqueue_resource_id: u64,
    /// The concurrency limit for the work queue associated with the green context.
    ///  This defines the maximum number of concurrent operations allowed on this work queue.
    pub workqueue_concurrency_limit: u32,
    /// The sharing scope for the work queue associated with the green context.
    ///  This defines how the work queue can be shared across different contexts or processes.
    ///  See the CUDA `CUdevWorkqueueConfigScope` values for possible sharing scopes.
    pub workqueue_sharing_scope: u32,
    /// The number of HW channels assigned to this green context's work queue.
    ///  This reflects the actual channel assignment at green context creation time.
    ///  A value of 0 indicates that channel information is not available.
    pub workqueue_channel_count: u32,
    /// Array of HW channel IDs assigned to this green context's work queue.
    ///  Valid entries are specified by workqueueChannelCount; unused entries are zeroed.
    ///  The array supports up to 32 channel IDs.
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

    fn from_raw(value: &sys::CUpti_ActivityGreenContext3) -> Self {
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
    fn from_raw(kind: ActivityObjectKind, value: sys::CUpti_ActivityObjectKindId) -> Self {
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
    /// Name associated with the recorded object. `None` represents a null native pointer.
    pub name: Option<&'a CStr>,
}

impl ActivityName<'_> {
    fn from_raw(value: sys::CUpti_ActivityName) -> Self {
        let object_kind = ActivityObjectKind::from(value.objectKind);
        let name = if value.name.is_null() {
            None
        } else {
            // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
            Some(unsafe { CStr::from_ptr(value.name) })
        };
        Self {
            object_kind,
            object_id: ActivityObjectId::from_raw(object_kind, value.objectId),
            name,
        }
    }
}

/// Decoded instantaneous, start, or end marker.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMarker<'a> {
    /// The flags associated with the marker.
    ///
    /// See also: `CUpti_ActivityFlag`
    pub flags: ActivityFlags,
    /// The timestamp for the marker, in nanoseconds. A value of 0 indicates that timestamp information could not be collected for the marker.
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
    fn from_raw(value: sys::CUpti_ActivityMarker2) -> Self {
        let object_kind = ActivityObjectKind::from(value.objectKind);
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
        Self {
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
            timestamp: value.timestamp,
            id: ActivityMarkerId::from(value.id),
            object_kind,
            object_id: ActivityObjectId::from_raw(object_kind, value.objectId),
            name,
            domain,
        }
    }
}

/// Correlation between a CUDA activity and an external record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityExternalCorrelation {
    /// The kind of external API this record correlated to.
    pub external_kind: ExternalCorrelationKind,
    /// The correlation ID of the associated non-CUDA API record. The exact field in the associated external record depends on that record's activity kind (
    ///
    /// See also: `external_kind`).
    pub external_id: ExternalCorrelationId,
    /// The correlation ID of the associated CUDA driver or runtime API record.
    pub correlation_id: CorrelationId,
}

impl ActivityExternalCorrelation {
    fn from_raw(value: sys::CUpti_ActivityExternalCorrelation) -> Self {
        Self {
            external_kind: ExternalCorrelationKind::from(value.externalKind),
            external_id: ExternalCorrelationId::from(value.externalId),
            correlation_id: CorrelationId::from(value.correlationId),
        }
    }
}

fn metric_value_from_raw_kind(kind: MetricValueKind, value: sys::CUpti_MetricValue) -> MetricValue {
    // SAFETY: every field of this union is a plain number, so reading the raw 64-bit pattern is always valid.
    unsafe {
        match kind {
            MetricValueKind::Double => MetricValue::Double(value.metricValueDouble),
            MetricValueKind::Uint64 | MetricValueKind::ForceInt => {
                MetricValue::Uint64(value.metricValueUint64)
            }
            MetricValueKind::Percent => MetricValue::Percent(value.metricValuePercent),
            MetricValueKind::Throughput => MetricValue::Throughput(value.metricValueThroughput),
            MetricValueKind::Int64 => MetricValue::Int64(value.metricValueInt64),
            MetricValueKind::UtilizationLevel => MetricValue::UtilizationLevel(
                MetricValueUtilizationLevel::from(value.metricValueUtilizationLevel),
            ),
            MetricValueKind::NvtxExtendedPayload => {
                MetricValue::NvtxExtendedPayload(value.metricValueNvtxExtendedPayload)
            }
        }
    }
}

/// Native 64-bit metric value bits, interpreted using an explicit [`MetricValueKind`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MetricValueRaw {
    bits: u64,
}

impl Debug for MetricValueRaw {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MetricValueRaw")
            .field(&format_args!("{:#x}", self.bits))
            .finish()
    }
}

impl MetricValueRaw {
    fn from_raw(value: sys::CUpti_MetricValue) -> Self {
        Self {
            // SAFETY: every field of this union is a plain number, so reading the raw 64-bit pattern is always valid.
            bits: unsafe { value.metricValueUint64 },
        }
    }

    /// Interprets the native metric bits according to `kind`.
    ///
    /// Choose the kind reported by the metric metadata. The native integer sentinel is interpreted as an unsigned 64-bit value.
    #[must_use]
    pub fn value_as(self, kind: MetricValueKind) -> MetricValue {
        metric_value_from_raw_kind(
            kind,
            sys::CUpti_MetricValue {
                metricValueUint64: self.bits,
            },
        )
    }
}

/// Decoded activity record providing detailed information for a marker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivityMarkerPayload {
    /// The flags associated with the marker.
    ///
    /// See also: `CUpti_ActivityFlag`
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
    /// CUPTI maintained domain id required for NVTX extended payloads. To parse the payload correctly, the domain id must be used to identify the payload attributes as they are domain specific.
    pub cupti_domain_id: CuptiDomainId,
}

impl ActivityMarkerPayload {
    fn from_raw(value: sys::CUpti_ActivityMarkerData2) -> Self {
        let payload_kind = MetricValueKind::from(value.payloadKind);
        Self {
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
            id: ActivityMarkerId::from(value.id),
            payload_kind,
            payload: metric_value_from_raw_kind(payload_kind, value.payload),
            color: value.color,
            category: value.category,
            cupti_domain_id: CuptiDomainId::from(value.cuptiDomainId),
        }
    }
}

/// Decoded activity record for a CUPTI metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMetric {
    /// The metric ID.
    pub id: MetricId,
    /// The metric value.
    pub value: MetricValueRaw,
    /// The correlation ID of the metric. Use of this ID is user-defined, but typically this ID value will equal the correlation ID of the kernel for which the metric was gathered.
    pub correlation_id: CorrelationId,
    /// The properties of this metric.
    ///
    /// See also: `CUpti_ActivityFlag`
    pub flags: ActivityFlags,
}

impl ActivityMetric {
    fn from_raw(value: sys::CUpti_ActivityMetric) -> Self {
        Self {
            id: MetricId::from(value.id),
            value: MetricValueRaw::from_raw(value.value),
            correlation_id: CorrelationId::from(value.correlationId),
            flags: ActivityFlags::from_bits_truncate(u32::from(value.flags)),
        }
    }
}

/// Decoded activity record for a CUPTI metric with instance information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMetricInstance {
    /// The metric ID.
    pub id: MetricId,
    /// The metric value.
    pub value: MetricValueRaw,
    /// The metric domain instance.
    pub instance: u32,
    /// The correlation ID of the metric. Use of this ID is user-defined, but typically this ID value will equal the correlation ID of the kernel for which the metric was gathered.
    pub correlation_id: CorrelationId,
    /// The properties of this metric.
    ///
    /// See also: `CUpti_ActivityFlag`
    pub flags: ActivityFlags,
}

impl ActivityMetricInstance {
    fn from_raw(value: sys::CUpti_ActivityMetricInstance) -> Self {
        Self {
            id: MetricId::from(value.id),
            value: MetricValueRaw::from_raw(value.value),
            instance: value.instance,
            correlation_id: CorrelationId::from(value.correlationId),
            flags: ActivityFlags::from_bits_truncate(u32::from(value.flags)),
        }
    }
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
    /// The device-side timestamp on CUDA event record. Timestamp is in nanoseconds. Collection of this field is disabled by default. It can be enabled by calling CUPTI API [`enable_cuda_event_device_timestamulti_process_service`].
    pub device_timestamp: u64,
    /// A unique ID to associate event synchronization records with the latest CUDA Event record.
    pub cuda_event_sync_id: CudaEventSyncId,
}

impl ActivityCudaEvent {
    fn from_raw(value: sys::CUpti_ActivityCudaEvent2) -> Self {
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
    ///
    /// See also: [`ActivityComputeApiKind`].
    pub compute_api_kind: ActivityComputeApiKind,
    /// Identifier of the default stream in this context.
    pub null_stream_id: StreamId,
    /// The ID of the parent context. It would be 0 if context does not have parent.
    pub parent_context_id: ContextId,
    /// Whether this is a CUDA green context.
    pub is_green_context: bool,
    /// Number of multiprocessors assigned to the green context. Valid only when `is_green_context` is `true`.
    pub num_multiprocessors: u16,
    /// This field indicates the CIG mode.
    pub cig_mode: ContextCigMode,
    /// The ID of the process associated with the context.
    pub process_id: ProcessId,
}

impl ActivityContext {
    fn from_raw(value: sys::CUpti_ActivityContext4) -> Self {
        Self {
            context_id: ContextId::from(value.contextId),
            device_id: DeviceId::from(value.deviceId),
            compute_api_kind: ActivityComputeApiKind::try_from(u32::from(value.computeApiKind))
                .unwrap_or(ActivityComputeApiKind::Unknown),
            null_stream_id: StreamId::from(value.nullStreamId),
            parent_context_id: ContextId::from(value.parentContextId),
            is_green_context: value.isGreenContext != 0,
            num_multiprocessors: value.numMultiprocessors,
            cig_mode: ContextCigMode::from(value.cigMode),
            process_id: ProcessId::from(value.processId),
        }
    }
}

/// Decoded activity record for a CUPTI event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityEvent {
    /// The event ID.
    pub id: EventId,
    /// The event value.
    pub value: u64,
    /// The event domain ID.
    pub domain_id: EventDomainId,
    /// The correlation ID of the event. Use of this ID is user-defined, but typically this ID value will equal the correlation ID of the kernel for which the event was gathered.
    pub correlation_id: CorrelationId,
}

impl ActivityEvent {
    fn from_raw(value: sys::CUpti_ActivityEvent) -> Self {
        Self {
            id: EventId::from(value.id),
            value: value.value,
            domain_id: EventDomainId::from(value.domain),
            correlation_id: CorrelationId::from(value.correlationId),
        }
    }
}

/// Decoded activity record for a CUPTI event with instance information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityEventInstance {
    /// The event ID.
    pub id: EventId,
    /// The event domain ID.
    pub domain_id: EventDomainId,
    /// The event domain instance.
    pub instance: u32,
    /// The event value.
    pub value: u64,
    /// The correlation ID of the event. Use of this ID is user-defined, but typically this ID value will equal the correlation ID of the kernel for which the event was gathered.
    pub correlation_id: CorrelationId,
}

impl ActivityEventInstance {
    fn from_raw(value: sys::CUpti_ActivityEventInstance) -> Self {
        Self {
            id: EventId::from(value.id),
            domain_id: EventDomainId::from(value.domain),
            instance: value.instance,
            value: value.value,
            correlation_id: CorrelationId::from(value.correlationId),
        }
    }
}

/// Decoded activity record for an instantaneous CUPTI event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityInstantaneousEvent {
    /// The event ID.
    pub id: EventId,
    /// The event value.
    pub value: u64,
    /// The timestamp at which event is sampled.
    pub timestamp: u64,
    /// The device id.
    pub device_id: DeviceId,
}

impl ActivityInstantaneousEvent {
    fn from_raw(value: sys::CUpti_ActivityInstantaneousEvent) -> Self {
        Self {
            id: EventId::from(value.id),
            value: value.value,
            timestamp: value.timestamp,
            device_id: DeviceId::from(value.deviceId),
        }
    }
}

/// Decoded activity record for an instantaneous CUPTI event with event domain instance information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityInstantaneousEventInstance {
    /// The event ID.
    pub id: EventId,
    /// The event value.
    pub value: u64,
    /// The timestamp at which event is sampled.
    pub timestamp: u64,
    /// The device id.
    pub device_id: DeviceId,
    /// The event domain instance.
    pub instance: u8,
}

impl ActivityInstantaneousEventInstance {
    fn from_raw(value: sys::CUpti_ActivityInstantaneousEventInstance) -> Self {
        Self {
            id: EventId::from(value.id),
            value: value.value,
            timestamp: value.timestamp,
            device_id: DeviceId::from(value.deviceId),
            instance: value.instance,
        }
    }
}

/// Decoded activity record for an instantaneous CUPTI metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityInstantaneousMetric {
    /// The metric ID.
    pub id: MetricId,
    /// The metric value.
    pub value: MetricValueRaw,
    /// The timestamp at which metric is sampled.
    pub timestamp: u64,
    /// The device id.
    pub device_id: DeviceId,
    /// The properties of this metric.
    ///
    /// See also: `CUpti_ActivityFlag`
    pub flags: ActivityFlags,
}

impl ActivityInstantaneousMetric {
    fn from_raw(value: sys::CUpti_ActivityInstantaneousMetric) -> Self {
        Self {
            id: MetricId::from(value.id),
            value: MetricValueRaw::from_raw(value.value),
            timestamp: value.timestamp,
            device_id: DeviceId::from(value.deviceId),
            flags: ActivityFlags::from_bits_truncate(u32::from(value.flags)),
        }
    }
}

/// Decoded instantaneous activity record for a CUPTI metric with instance information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityInstantaneousMetricInstance {
    /// The metric ID.
    pub id: MetricId,
    /// The metric value.
    pub value: MetricValueRaw,
    /// The timestamp at which metric is sampled.
    pub timestamp: u64,
    /// The device id.
    pub device_id: DeviceId,
    /// The properties of this metric.
    ///
    /// See also: `CUpti_ActivityFlag`
    pub flags: ActivityFlags,
    /// The metric domain instance.
    pub instance: u8,
}

impl ActivityInstantaneousMetricInstance {
    fn from_raw(value: sys::CUpti_ActivityInstantaneousMetricInstance) -> Self {
        Self {
            id: MetricId::from(value.id),
            value: MetricValueRaw::from_raw(value.value),
            timestamp: value.timestamp,
            device_id: DeviceId::from(value.deviceId),
            flags: ActivityFlags::from_bits_truncate(u32::from(value.flags)),
            instance: value.instance,
        }
    }
}

/// Decoded activity record for source-level global access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityGlobalAccess {
    /// The properties of this global access.
    pub flags: ActivityFlags,
    /// The ID for source locator.
    pub source_locator_id: SourceLocatorId,
    /// The correlation ID of the kernel to which this result is associated.
    pub correlation_id: CorrelationId,
    /// Correlation ID with global/device function name.
    pub function_id: FunctionId,
    /// The number of times this instruction was executed per warp. It will be incremented when at least one of thread among warp is active with predicate and condition code evaluating to true.
    pub executed: u32,
    /// The pc offset for the access.
    pub pc_offset: u64,
    /// This increments each time when this instruction is executed by number of threads that executed this instruction with predicate and condition code evaluating to true.
    pub threads_executed: u64,
    /// The total number of 32 bytes transactions to L2 cache generated by this access.
    pub l2_transactions: u64,
    /// The minimum number of L2 transactions possible based on the access pattern.
    pub theoretical_l2_transactions: u64,
}

impl ActivityGlobalAccess {
    fn from_raw(value: sys::CUpti_ActivityGlobalAccess3) -> Self {
        Self {
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
            source_locator_id: SourceLocatorId::from(value.sourceLocatorId),
            correlation_id: CorrelationId::from(value.correlationId),
            function_id: FunctionId::from(value.functionId),
            executed: value.executed,
            pc_offset: value.pcOffset,
            threads_executed: value.threadsExecuted,
            l2_transactions: value.l2_transactions,
            theoretical_l2_transactions: value.theoreticalL2Transactions,
        }
    }
}

/// Decoded source-level branch execution statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityBranch {
    /// The ID for source locator.
    pub source_locator_id: SourceLocatorId,
    /// The correlation ID of the kernel to which this result is associated.
    pub correlation_id: CorrelationId,
    /// Correlation ID with global/device function name.
    pub function_id: FunctionId,
    /// The pc offset for the branch.
    pub pc_offset: u32,
    /// Number of times this branch diverged.
    pub diverged: u32,
    /// This increments each time when this instruction is executed by number of threads that executed this instruction.
    pub threads_executed: u64,
    /// The number of times this instruction was executed per warp. It will be incremented regardless of predicate or condition code.
    pub executed: u32,
}

impl ActivityBranch {
    fn from_raw(value: sys::CUpti_ActivityBranch2) -> Self {
        Self {
            source_locator_id: SourceLocatorId::from(value.sourceLocatorId),
            correlation_id: CorrelationId::from(value.correlationId),
            function_id: FunctionId::from(value.functionId),
            pc_offset: value.pcOffset,
            diverged: value.diverged,
            threads_executed: value.threadsExecuted,
            executed: value.executed,
        }
    }
}

/// Decoded source-level instruction execution statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityInstructionExecution {
    /// The properties of this instruction execution.
    pub flags: ActivityFlags,
    /// The ID for source locator.
    pub source_locator_id: SourceLocatorId,
    /// The correlation ID of the kernel to which this result is associated.
    pub correlation_id: CorrelationId,
    /// Correlation ID with global/device function name.
    pub function_id: FunctionId,
    /// The pc offset for the instruction.
    pub pc_offset: u32,
    /// This increments each time when this instruction is executed by number of threads that executed this instruction, regardless of predicate or condition code.
    pub threads_executed: u64,
    /// This increments each time when this instruction is executed by number of threads that executed this instruction with predicate and condition code evaluating to true.
    pub not_predicated_off_threads_executed: u64,
    /// The number of times this instruction was executed per warp. It will be incremented regardless of predicate or condition code.
    pub executed: u32,
}

impl ActivityInstructionExecution {
    fn from_raw(value: sys::CUpti_ActivityInstructionExecution) -> Self {
        Self {
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
            source_locator_id: SourceLocatorId::from(value.sourceLocatorId),
            correlation_id: CorrelationId::from(value.correlationId),
            function_id: FunctionId::from(value.functionId),
            pc_offset: value.pcOffset,
            threads_executed: value.threadsExecuted,
            not_predicated_off_threads_executed: value.notPredOffThreadsExecuted,
            executed: value.executed,
        }
    }
}

/// Decoded activity record for PC sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPcSampling {
    /// The properties of this instruction.
    pub flags: ActivityFlags,
    /// The ID for source locator.
    pub source_locator_id: SourceLocatorId,
    /// The correlation ID of the kernel to which this result is associated.
    pub correlation_id: CorrelationId,
    /// Correlation ID with global/device function name.
    pub function_id: FunctionId,
    /// Number of times the PC was sampled with the `stall_reason` in the record. These samples indicate that no instruction was issued in that cycle from the warp scheduler from where the warp was sampled. Field is valid for devices with compute capability 6.0 and higher.
    pub latency_samples: u32,
    /// Number of times the PC was sampled with the `stall_reason` in the record. The same PC can be sampled with different stall reasons. The count includes `latency_samples`.
    pub samples: u32,
    /// Current stall reason. Includes one of the reasons from [`ActivityPcSamplingStallReason`].
    pub stall_reason: ActivityPcSamplingStallReason,
    /// The pc offset for the instruction.
    pub pc_offset: u64,
}

impl ActivityPcSampling {
    fn from_raw(value: sys::CUpti_ActivityPCSampling3) -> Self {
        Self {
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
            source_locator_id: SourceLocatorId::from(value.sourceLocatorId),
            correlation_id: CorrelationId::from(value.correlationId),
            function_id: FunctionId::from(value.functionId),
            latency_samples: value.latencySamples,
            samples: value.samples,
            stall_reason: ActivityPcSamplingStallReason::from(value.stallReason),
            pc_offset: value.pcOffset,
        }
    }
}

/// Status and sample counts for PC sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPcSamplingRecordInfo {
    /// The correlation ID of the kernel to which this result is associated.
    pub correlation_id: CorrelationId,
    /// Number of times the PC was sampled for this kernel instance including all dropped samples.
    pub total_samples: u64,
    /// Number of samples that were dropped by hardware due to backpressure/overflow.
    pub dropped_samples: u64,
    /// Sampling period in terms of number of cycles .
    pub sampling_period_in_cycles: u64,
}

impl ActivityPcSamplingRecordInfo {
    fn from_raw(value: sys::CUpti_ActivityPCSamplingRecordInfo) -> Self {
        Self {
            correlation_id: CorrelationId::from(value.correlationId),
            total_samples: value.totalSamples,
            dropped_samples: value.droppedSamples,
            sampling_period_in_cycles: value.samplingPeriodInCycles,
        }
    }
}

/// Decoded Unified Memory counter activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityUnifiedMemoryCounter {
    /// The Unified Memory counter kind.
    pub counter_kind: ActivityUnifiedMemoryCounterKind,
    /// Value of the counter For `counter_kind` [`ActivityUnifiedMemoryCounterKind::BytesTransferHostToDevice`], [`ActivityUnifiedMemoryCounterKind::BytesTransferDeviceToHost`], [`ActivityUnifiedMemoryCounterKind::Thrashing`] and [`ActivityUnifiedMemoryCounterKind::RemoteMap`], it is the size of the memory region in bytes. For `counter_kind` [`ActivityUnifiedMemoryCounterKind::GpuPageFault`], it is the number of page fault groups for the same page. For `counter_kind` [`ActivityUnifiedMemoryCounterKind::CentralProcessingUnitPageFaultCount`], it is the program counter for the instruction that caused fault.
    pub value: u64,
    /// The start timestamp of the counter, in nanoseconds. For `counter_kind` [`ActivityUnifiedMemoryCounterKind::BytesTransferHostToDevice`] and [`ActivityUnifiedMemoryCounterKind::BytesTransferDeviceToHost`], timestamp is captured when activity starts on GPU. For `counter_kind` [`ActivityUnifiedMemoryCounterKind::GpuPageFault`] and [`ActivityUnifiedMemoryCounterKind::CentralProcessingUnitPageFaultCount`], timestamp is captured when CUDA driver started processing the fault. For `counter_kind` [`ActivityUnifiedMemoryCounterKind::Thrashing`], timestamp is captured when CUDA driver detected thrashing of memory region. For `counter_kind` [`ActivityUnifiedMemoryCounterKind::Throttling`], timestamp is captured when throttling operation was started by CUDA driver. For `counter_kind` [`ActivityUnifiedMemoryCounterKind::RemoteMap`], timestamp is captured when CUDA driver has pushed all required operations to the processor specified by `destination_id`.
    pub start: u64,
    /// The end timestamp of the counter, in nanoseconds. Ignore this field if `counter_kind` is [`ActivityUnifiedMemoryCounterKind::CentralProcessingUnitPageFaultCount`] or [`ActivityUnifiedMemoryCounterKind::Thrashing`] or [`ActivityUnifiedMemoryCounterKind::RemoteMap`]. For `counter_kind` [`ActivityUnifiedMemoryCounterKind::BytesTransferHostToDevice`] and [`ActivityUnifiedMemoryCounterKind::BytesTransferDeviceToHost`], timestamp is captured when activity finishes on GPU. For `counter_kind` [`ActivityUnifiedMemoryCounterKind::GpuPageFault`], timestamp is captured when CUDA driver queues the replay of faulting memory accesses on the GPU For `counter_kind` [`ActivityUnifiedMemoryCounterKind::Throttling`], timestamp is captured when throttling operation was finished by CUDA driver.
    pub end: u64,
    /// This is the virtual base address of the page/s being transferred. For cpu and gpu faults, the virtual address for the page that faulted.
    pub address: u64,
    /// The ID of the source CPU/device involved in the memory transfer, page fault, thrashing, throttling or remote map operation. For `counter_kind` [`ActivityUnifiedMemoryCounterKind::Thrashing`], it is a bitwise OR of the device IDs fighting for the memory region, ONLY if there are less than 32 devices. Ignore this field if `counter_kind` is [`ActivityUnifiedMemoryCounterKind::CentralProcessingUnitPageFaultCount`].
    pub source_id: UnifiedMemoryProcessorId,
    /// The ID of the destination CPU/device involved in the memory transfer or remote map operation. Ignore this field if `counter_kind` is [`ActivityUnifiedMemoryCounterKind::GpuPageFault`] or [`ActivityUnifiedMemoryCounterKind::CentralProcessingUnitPageFaultCount`] or [`ActivityUnifiedMemoryCounterKind::Thrashing`] or [`ActivityUnifiedMemoryCounterKind::Throttling`].
    pub destination_id: UnifiedMemoryProcessorId,
    /// The ID of the stream causing the transfer. This value of this field is invalid.
    pub stream_id: StreamId,
    /// The ID of the process to which this record belongs to.
    pub process_id: ProcessId,
    /// Counter-specific access type, migration cause, remote-map cause, activity flags, or preserved raw bits, selected by `counter_kind`.
    pub flags: ActivityUnifiedMemoryCounterFlags,
    /// Device-participation bitset for thrashing counters; other counter kinds do not use this field.
    ///
    /// Each word covers 64 device identifiers: `processors[0]` covers devices 0 through 63, `processors[1]` covers devices 64 through 127, and so on.
    pub processors: [u64; 5],
}

/// Meaning of Unified Memory flags selected by the counter kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityUnifiedMemoryCounterFlags {
    /// Memory access type for a GPU page fault.
    AccessType(ActivityUnifiedMemoryAccessType),
    /// Migration cause for host-to-device, device-to-host, or device-to-device transfers.
    MigrationCause(ActivityUnifiedMemoryMigrationCause),
    /// Reason for a remote mapping.
    RemoteMapCause(ActivityUnifiedMemoryRemoteMapCause),
    /// Activity flags for thrashing or throttling counters.
    Activity(ActivityFlags),
    /// Uninterpreted flags for a counter kind without a typed flag interpretation.
    Raw(u32),
}

impl ActivityUnifiedMemoryCounterFlags {
    fn from_raw(kind: ActivityUnifiedMemoryCounterKind, flags: u32) -> Self {
        match kind {
            ActivityUnifiedMemoryCounterKind::GpuPageFault => Self::AccessType(
                ActivityUnifiedMemoryAccessType::try_from(flags)
                    .unwrap_or(ActivityUnifiedMemoryAccessType::Unknown),
            ),
            ActivityUnifiedMemoryCounterKind::BytesTransferHostToDevice
            | ActivityUnifiedMemoryCounterKind::BytesTransferDeviceToHost
            | ActivityUnifiedMemoryCounterKind::BytesTransferDeviceToDevice => {
                Self::MigrationCause(
                    ActivityUnifiedMemoryMigrationCause::try_from(flags)
                        .unwrap_or(ActivityUnifiedMemoryMigrationCause::Unknown),
                )
            }
            ActivityUnifiedMemoryCounterKind::RemoteMap => Self::RemoteMapCause(
                ActivityUnifiedMemoryRemoteMapCause::try_from(flags)
                    .unwrap_or(ActivityUnifiedMemoryRemoteMapCause::Unknown),
            ),
            ActivityUnifiedMemoryCounterKind::Thrashing
            | ActivityUnifiedMemoryCounterKind::Throttling => {
                Self::Activity(ActivityFlags::from_bits_truncate(flags))
            }
            ActivityUnifiedMemoryCounterKind::Unknown
            | ActivityUnifiedMemoryCounterKind::CentralProcessingUnitPageFaultCount
            | ActivityUnifiedMemoryCounterKind::Count
            | ActivityUnifiedMemoryCounterKind::ForceInt => Self::Raw(flags),
        }
    }
}

impl ActivityUnifiedMemoryCounter {
    fn from_raw(value: sys::CUpti_ActivityUnifiedMemoryCounter3) -> Self {
        let counter_kind = ActivityUnifiedMemoryCounterKind::from(value.counterKind);
        Self {
            counter_kind,
            value: value.value,
            start: value.start,
            end: value.end,
            address: value.address,
            source_id: UnifiedMemoryProcessorId::from(value.srcId),
            destination_id: UnifiedMemoryProcessorId::from(value.dstId),
            stream_id: StreamId::from(value.streamId),
            process_id: ProcessId::from(value.processId),
            flags: ActivityUnifiedMemoryCounterFlags::from_raw(counter_kind, value.flags),
            processors: value.processors,
        }
    }
}

/// Whether a device attribute belongs to CUPTI or the CUDA driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityDeviceAttributeKind {
    /// CUPTI-defined device attribute.
    Cupti(DeviceAttribute),
    /// CUDA driver device attribute.
    Cuda(CudaDeviceAttribute),
}

/// Native device-attribute value bits with typed numeric accessors.
///
/// The attribute selector determines which accessor is meaningful.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivityDeviceAttributeValue {
    bits: u64,
}

impl Debug for ActivityDeviceAttributeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ActivityDeviceAttributeValue")
            .field(&format_args!("{:#x}", self.bits))
            .finish()
    }
}

impl ActivityDeviceAttributeValue {
    fn from_raw(value: sys::CUpti_ActivityDeviceAttribute__bindgen_ty_2) -> Self {
        Self {
            // SAFETY: every field of this union is a plain number, so reading the raw 64-bit pattern is always valid.
            bits: unsafe { value.vUint64 },
        }
    }

    /// Interprets all 64 value bits as an IEEE 754 `f64`.
    ///
    /// Use this interpretation only when the selected attribute reports a double-precision value.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        f64::from_bits(self.bits)
    }

    /// Returns the low 32 bits as an unsigned integer.
    ///
    /// # Panics
    ///
    /// Panics if the internal four-byte slice has an unexpected length. Its fixed range makes this condition unreachable.
    #[must_use]
    pub fn as_u32(self) -> u32 {
        u32::from_le_bytes(
            self.bits.to_le_bytes()[..4]
                .try_into()
                .expect("four low register bytes"),
        )
    }

    /// Returns the preserved 64-bit unsigned integer representation.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.bits
    }

    /// Interprets the low 32 bits as a signed integer without numeric conversion.
    #[must_use]
    pub fn as_i32(self) -> i32 {
        self.as_u32().cast_signed()
    }

    /// Interprets all 64 bits as a signed integer without numeric conversion.
    #[must_use]
    pub fn as_i64(self) -> i64 {
        self.bits.cast_signed()
    }
}

/// Decoded activity record for a device attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityDeviceAttribute {
    /// The flags associated with the device.
    ///
    /// See also: `CUpti_ActivityFlag`
    pub flags: ActivityFlags,
    /// The ID of the device that this attribute applies to.
    pub device_id: DeviceId,
    /// Attribute selector decoded as either a CUPTI attribute or a CUDA driver attribute using the native device-attribute flag.
    pub attribute: ActivityDeviceAttributeKind,
    /// Raw attribute value. Interpret it using the selected CUPTI or CUDA driver attribute and its native value type.
    pub value: ActivityDeviceAttributeValue,
}

impl ActivityDeviceAttribute {
    fn from_raw(value: sys::CUpti_ActivityDeviceAttribute) -> Self {
        let flags = ActivityFlags::from_bits_truncate(value.flags as u32);
        let attribute = if flags.contains(ActivityFlags::DEVICE_ATTRIBUTE_CUDEVICE) {
            // SAFETY: the device-attribute flag checked above selects this union field.
            ActivityDeviceAttributeKind::Cuda(CudaDeviceAttribute::from(unsafe {
                value.attribute.cu
            }))
        } else {
            // SAFETY: the device-attribute flag checked above selects this union field.
            ActivityDeviceAttributeKind::Cupti(DeviceAttribute::from(unsafe {
                value.attribute.cupti
            }))
        };
        Self {
            flags,
            device_id: DeviceId::from(value.deviceId),
            attribute,
            value: ActivityDeviceAttributeValue::from_raw(value.value),
        }
    }
}

/// Sampled GPU and memory clocks and PCI Express link state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityEnvironmentSpeed {
    /// Streaming-multiprocessor clock frequency in MHz.
    pub sm_clock: u32,
    /// Memory clock frequency in MHz.
    pub memory_clock: u32,
    /// PCI Express link generation.
    pub pcie_link_gen: u32,
    /// PCI Express link width.
    pub pcie_link_width: u32,
    /// Reasons the device clocks are being throttled.
    pub clocks_throttle_reasons: EnvironmentClocksThrottleReasons,
}

/// Sampled GPU power consumption and the power-management threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityEnvironmentPower {
    /// Power consumed by the GPU and associated circuitry, in milliwatts.
    pub power: u32,
    /// Power-management threshold in milliwatts.
    pub power_limit: u32,
}

/// Sampled environmental value decoded according to the environment-kind tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityEnvironmentReading {
    /// Unrecognized or unsupported environmental sample kind.
    Unknown,
    /// Clock frequencies, PCI Express link state, and clock-throttling reasons.
    Speed(ActivityEnvironmentSpeed),
    /// GPU temperature sample.
    Temperature {
        /// GPU temperature in degrees Celsius.
        gpu_temperature: u32,
    },
    /// Power-consumption and power-limit sample.
    Power(ActivityEnvironmentPower),
    /// Fan-speed sample.
    Cooling {
        /// Fan speed as a percentage of its maximum.
        fan_speed: u32,
    },
}

/// Decoded activity record for CUPTI environmental data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityEnvironment {
    /// The ID of the device.
    pub device_id: DeviceId,
    /// The timestamp when this sample was retrieved, in nanoseconds. A value of 0 indicates that timestamp information could not be collected for the marker.
    pub timestamp: u64,
    /// The kind of data reported in this record.
    pub environment_kind: ActivityEnvironmentKind,
    /// Environmental reading selected by `environment_kind`.
    pub data: ActivityEnvironmentReading,
}

impl ActivityEnvironment {
    fn from_raw(value: sys::CUpti_ActivityEnvironment) -> Self {
        let environment_kind = ActivityEnvironmentKind::from(value.environmentKind);
        // SAFETY: each arm reads the union field that the matched tag selects.
        let data = unsafe {
            match environment_kind {
                ActivityEnvironmentKind::Speed => {
                    let speed = value.data.speed;
                    ActivityEnvironmentReading::Speed(ActivityEnvironmentSpeed {
                        sm_clock: speed.smClock,
                        memory_clock: speed.memoryClock,
                        pcie_link_gen: speed.pcieLinkGen,
                        pcie_link_width: speed.pcieLinkWidth,
                        clocks_throttle_reasons:
                            EnvironmentClocksThrottleReasons::from_bits_truncate(
                                speed.clocksThrottleReasons as u32,
                            ),
                    })
                }
                ActivityEnvironmentKind::Temperature => ActivityEnvironmentReading::Temperature {
                    gpu_temperature: value.data.temperature.gpuTemperature,
                },
                ActivityEnvironmentKind::Power => {
                    let power = value.data.power;
                    ActivityEnvironmentReading::Power(ActivityEnvironmentPower {
                        power: power.power,
                        power_limit: power.powerLimit,
                    })
                }
                ActivityEnvironmentKind::Cooling => ActivityEnvironmentReading::Cooling {
                    fan_speed: value.data.cooling.fanSpeed,
                },
                _ => ActivityEnvironmentReading::Unknown,
            }
        };
        Self {
            device_id: DeviceId::from(value.deviceId),
            timestamp: value.timestamp,
            environment_kind,
            data,
        }
    }
}

fn uuid_from_raw(value: sys::CUuuid) -> [u8; 16] {
    value.bytes.map(|byte| byte as u8)
}

/// Decoded activity record for a device.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "CUPTI reports independent device capabilities as boolean fields"
)]
pub struct ActivityDevice<'a> {
    /// The flags associated with the device.
    ///
    /// See also: `CUpti_ActivityFlag`
    pub flags: ActivityFlags,
    /// The global memory bandwidth available on the device, in `kBytes`/sec.
    pub global_memory_bandwidth: u64,
    /// The amount of global memory on the device, in bytes.
    pub global_memory_size: u64,
    /// The amount of constant memory on the device, in bytes.
    pub constant_memory_size: u32,
    /// The size of the L2 cache on the device, in bytes.
    pub l2_cache_size: u32,
    /// The number of threads per warp on the device.
    pub num_threads_per_warp: u32,
    /// The core clock rate of the device, in `kHz`.
    pub core_clock_rate: u32,
    /// Number of memory copy engines on the device.
    pub num_memory_copy_engines: u32,
    /// Number of multiprocessors on the device.
    pub num_multiprocessors: u32,
    /// The maximum "instructions per cycle" possible on each device multiprocessor.
    pub max_instructions_per_cycle: u32,
    /// Maximum number of warps that can be present on a multiprocessor at any given time.
    pub max_warps_per_multiprocessor: u32,
    /// Maximum number of blocks that can be present on a multiprocessor at any given time.
    pub max_blocks_per_multiprocessor: u32,
    /// Maximum amount of shared memory available per multiprocessor, in bytes.
    pub max_shared_memory_per_multiprocessor: u32,
    /// Maximum number of 32-bit registers available per multiprocessor.
    pub max_registers_per_multiprocessor: u32,
    /// Maximum number of registers that can be allocated to a block.
    pub max_registers_per_block: u32,
    /// Maximum amount of shared memory that can be assigned to a block, in bytes.
    pub max_shared_memory_per_block: u32,
    /// Maximum number of threads allowed in a block.
    pub max_threads_per_block: u32,
    /// Maximum thread-block extent in each dimension.
    pub max_block_dim: Dim3,
    /// Maximum grid extent in each dimension.
    pub max_grid_dim: Dim3,
    /// Compute capability for the device, major number.
    pub compute_capability_major: u32,
    /// Compute capability for the device, minor number.
    pub compute_capability_minor: u32,
    /// The device ID.
    pub device_id: DeviceId,
    /// Whether error-correcting code (ECC) is enabled for the device.
    pub ecc_enabled: bool,
    /// The device UUID. This value is the globally unique immutable alphanumeric identifier of the device.
    pub uuid: [u8; 16],
    /// Device name. This borrowed view does not transfer ownership of the native name allocation. `None` represents a null native pointer.
    pub name: Option<&'a CStr>,
    /// Whether the device is visible to CUDA under the `CUDA_VISIBLE_DEVICES` environment setting.
    pub is_cuda_visible: bool,
    /// Whether Multi-Instance GPU (MIG) mode is enabled on the device.
    pub is_mig_enabled: bool,
    /// GPU Instance id for MIG enabled devices. If mig mode is disabled value is set to `UINT32_MAX`
    pub gpu_instance_id: GpuInstanceId,
    /// Compute Instance id for MIG enabled devices. If mig mode is disabled value is set to `UINT32_MAX`
    pub compute_instance_id: ComputeInstanceId,
    /// The MIG UUID. This value is the globally unique immutable alphanumeric identifier of the device.
    pub mig_uuid: [u8; 16],
    /// Whether this GPU is a non-uniform memory access (NUMA) node.
    pub is_non_uniform_memory_access_node: bool,
    /// NUMA node identifier of GPU memory; preserves the native invalid identifier when the GPU is not a NUMA node.
    pub non_uniform_memory_access_id: NonUniformMemoryAccessId,
    /// Number of TPCs on the device.
    pub num_tpcs: u32,
}

impl ActivityDevice<'_> {
    fn from_raw(value: sys::CUpti_ActivityDevice6) -> Self {
        let name = if value.name.is_null() {
            None
        } else {
            // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
            Some(unsafe { CStr::from_ptr(value.name) })
        };
        Self {
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
            global_memory_bandwidth: value.globalMemoryBandwidth,
            global_memory_size: value.globalMemorySize,
            constant_memory_size: value.constantMemorySize,
            l2_cache_size: value.l2CacheSize,
            num_threads_per_warp: value.numThreadsPerWarp,
            core_clock_rate: value.coreClockRate,
            num_memory_copy_engines: value.numMemcpyEngines,
            num_multiprocessors: value.numMultiprocessors,
            max_instructions_per_cycle: value.maxIPC,
            max_warps_per_multiprocessor: value.maxWarpsPerMultiprocessor,
            max_blocks_per_multiprocessor: value.maxBlocksPerMultiprocessor,
            max_shared_memory_per_multiprocessor: value.maxSharedMemoryPerMultiprocessor,
            max_registers_per_multiprocessor: value.maxRegistersPerMultiprocessor,
            max_registers_per_block: value.maxRegistersPerBlock,
            max_shared_memory_per_block: value.maxSharedMemoryPerBlock,
            max_threads_per_block: value.maxThreadsPerBlock,
            max_block_dim: Dim3 {
                x: value.maxBlockDimX,
                y: value.maxBlockDimY,
                z: value.maxBlockDimZ,
            },
            max_grid_dim: Dim3 {
                x: value.maxGridDimX,
                y: value.maxGridDimY,
                z: value.maxGridDimZ,
            },
            compute_capability_major: value.computeCapabilityMajor,
            compute_capability_minor: value.computeCapabilityMinor,
            device_id: DeviceId::from(value.id),
            ecc_enabled: value.eccEnabled != 0,
            uuid: uuid_from_raw(value.uuid),
            name,
            is_cuda_visible: value.isCudaVisible != 0,
            is_mig_enabled: value.isMigEnabled != 0,
            gpu_instance_id: GpuInstanceId::from(value.gpuInstanceId),
            compute_instance_id: ComputeInstanceId::from(value.computeInstanceId),
            mig_uuid: uuid_from_raw(value.migUuid),
            is_non_uniform_memory_access_node: value.isNumaNode != 0,
            non_uniform_memory_access_id: NonUniformMemoryAccessId::from(value.numaId),
            num_tpcs: value.numTpcs,
        }
    }
}

/// Decoded JIT compilation or compute-cache load or store activity.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityJit<'a> {
    /// The JIT entry type.
    pub entry_type: ActivityJitEntryType,
    /// The JIT operation type.
    pub operation_type: ActivityJitOperationType,
    /// The device ID.
    pub device_id: DeviceId,
    /// The start timestamp for the JIT operation, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the JIT operation.
    pub start: u64,
    /// The end timestamp for the JIT operation, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the JIT operation.
    pub end: u64,
    /// The correlation ID of the JIT operation to which records belong to. Each JIT operation is assigned a unique correlation ID that is identical to the correlation ID in the driver or runtime API activity record that launched the JIT operation.
    pub correlation_id: CorrelationId,
    /// The correlation ID to correlate JIT compilation, load and store operations. Each JIT compilation unit is assigned a unique correlation ID at the time of the JIT compilation. This correlation id can be used to find the matching JIT cache load/store records.
    pub operation_correlation_id: JitOperationCorrelationId,
    /// The size of compute cache.
    pub cache_size: u64,
    /// Path at which the compiled binary is cached. `None` represents a null native pointer.
    pub cache_path: Option<&'a CStr>,
    /// The ID of the process where the JIT operation is executing.
    pub process_id: ProcessId,
    /// The ID of the thread where the JIT operation is executing.
    pub thread_id: ThreadId,
}

impl ActivityJit<'_> {
    fn from_raw(value: sys::CUpti_ActivityJit2) -> Self {
        let cache_path = if value.cachePath.is_null() {
            None
        } else {
            // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
            Some(unsafe { CStr::from_ptr(value.cachePath) })
        };
        Self {
            entry_type: ActivityJitEntryType::from(value.jitEntryType),
            operation_type: ActivityJitOperationType::from(value.jitOperationType),
            device_id: DeviceId::from(value.deviceId),
            start: value.start,
            end: value.end,
            correlation_id: CorrelationId::from(value.correlationId),
            operation_correlation_id: JitOperationCorrelationId::from(
                value.jitOperationCorrelationId,
            ),
            cache_size: value.cacheSize,
            cache_path,
            process_id: ProcessId::from(value.processId),
            thread_id: ThreadId::from(value.threadId),
        }
    }
}

/// NPU index and domain identifier used by topology records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityNpuId {
    /// Index of the NPU endpoint.
    pub index: u32,
    /// Domain containing the NPU endpoint.
    pub domain_id: NpuDomainId,
}

/// GPU or NPU identity decoded according to a topology endpoint type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityTopologyDeviceId {
    /// Unrecognized topology endpoint type.
    Unknown,
    /// GPU UUID.
    Gpu {
        /// Native 16-byte GPU UUID.
        uuid: [u8; 16],
    },
    /// NPU index and domain.
    Npu(ActivityNpuId),
}

/// Decoded `NVLink` connection information.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityNvLink<'a> {
    /// `NVLink` version.
    pub version: u32,
    /// Device type of endpoint zero.
    pub device0_type: DeviceType,
    /// Device type of endpoint one.
    pub device1_type: DeviceType,
    /// Decoded identifier of endpoint zero: a GPU UUID, an NPU identifier, or an unknown device kind.
    pub device0_id: ActivityTopologyDeviceId,
    /// Decoded identifier of endpoint one: a GPU UUID, an NPU identifier, or an unknown device kind.
    pub device1_id: ActivityTopologyDeviceId,
    /// Flag gives capabilities of the link
    ///
    /// See also: `CUpti_LinkFlag`
    pub flags: LinkFlags,
    /// Number of physical `NVLinks` present between two devices.
    pub physical_link_count: u32,
    /// Borrowed endpoint-zero port identifiers, one per physical link. `None` indicates a null native pointer or zero links.
    pub device0_ports: Option<&'a [u32]>,
    /// Borrowed endpoint-one port identifiers, one per physical link. `None` indicates a null native pointer or zero links.
    pub device1_ports: Option<&'a [u32]>,
    /// Bandwidth of `NVLink` in kbytes/sec.
    pub bandwidth: u64,
    /// Whether an `NVSwitch` is an intermediate node in this connection.
    pub nvswitch_connected: bool,
}

impl ActivityNvLink<'_> {
    fn from_raw(value: sys::CUpti_ActivityNvLink5) -> Self {
        let device0_type = DeviceType::from(value.typeDev0);
        let device1_type = DeviceType::from(value.typeDev1);
        let port_count = value.physicalNvLinkCount as usize;
        let device0_ports = if value.portDev0.is_null() || port_count == 0 {
            None
        } else {
            // SAFETY: CUPTI stores `port_count` port entries behind each non-null port pointer, valid while the record is borrowed.
            Some(unsafe { slice::from_raw_parts(value.portDev0, port_count) })
        };
        let device1_ports = if value.portDev1.is_null() || port_count == 0 {
            None
        } else {
            // SAFETY: CUPTI stores `port_count` port entries behind each non-null port pointer, valid while the record is borrowed.
            Some(unsafe { slice::from_raw_parts(value.portDev1, port_count) })
        };
        Self {
            version: value.nvlinkVersion,
            device0_type,
            device1_type,
            device0_id: nvlink_device0_id_from_raw(device0_type, value.idDev0),
            device1_id: nvlink_device1_id_from_raw(device1_type, value.idDev1),
            flags: LinkFlags::from_bits_truncate(value.flag),
            physical_link_count: value.physicalNvLinkCount,
            device0_ports,
            device1_ports,
            bandwidth: value.bandwidth,
            nvswitch_connected: value.nvswitchConnected != 0,
        }
    }
}

fn nvlink_device0_id_from_raw(
    device_type: DeviceType,
    value: sys::CUpti_ActivityNvLink5__bindgen_ty_1,
) -> ActivityTopologyDeviceId {
    // SAFETY: each arm reads the union field that the matched tag selects.
    unsafe {
        match device_type {
            DeviceType::Gpu => ActivityTopologyDeviceId::Gpu {
                uuid: uuid_from_raw(value.uuidDev),
            },
            DeviceType::Npu => ActivityTopologyDeviceId::Npu(ActivityNpuId {
                index: value.npu.index,
                domain_id: NpuDomainId::from(value.npu.domainId),
            }),
            _ => ActivityTopologyDeviceId::Unknown,
        }
    }
}

fn nvlink_device1_id_from_raw(
    device_type: DeviceType,
    value: sys::CUpti_ActivityNvLink5__bindgen_ty_2,
) -> ActivityTopologyDeviceId {
    // SAFETY: each arm reads the union field that the matched tag selects.
    unsafe {
        match device_type {
            DeviceType::Gpu => ActivityTopologyDeviceId::Gpu {
                uuid: uuid_from_raw(value.uuidDev),
            },
            DeviceType::Npu => ActivityTopologyDeviceId::Npu(ActivityNpuId {
                index: value.npu.index,
                domain_id: NpuDomainId::from(value.npu.domainId),
            }),
            _ => ActivityTopologyDeviceId::Unknown,
        }
    }
}

/// GPU identity and peer-access information for PCI Express topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPcieGpuAttributes {
    /// Native 16-byte GPU UUID.
    pub uuid: [u8; 16],
    /// Native fixed-size array of CUDA device handles with peer-to-peer access capability.
    pub peer_devices: [CudaDevice; 32],
}

/// Bridge identity and downstream bus information for PCI Express topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPcieBridgeAttributes {
    /// Downstream bus number used to locate devices and bridges connected below this bridge.
    pub secondary_bus: PcieBusId,
    /// Hardware device identifier of the bridge.
    pub device_id: PcieHardwareDeviceId,
    /// Vendor identifier of the bridge.
    pub vendor_id: PcieVendorId,
}

/// CUDA device handle or bridge identifier for a PCI Express topology record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityPcieDeviceId {
    /// CUDA device handle for a GPU endpoint.
    Gpu(CudaDevice),
    /// Identifier for a PCI Express bridge.
    Bridge(PcieBridgeId),
}

/// GPU or bridge properties decoded from a PCI Express device-type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityPcieAttributes {
    /// GPU UUID and peer-access information.
    Gpu(ActivityPcieGpuAttributes),
    /// Bridge vendor, device, and downstream bus information.
    Bridge(ActivityPcieBridgeAttributes),
}

macro_rules! impl_debug_display {
    ($($name:ty),+ $(,)?) => {
        $(
            impl Display for $name {
                fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    write!(f, "{self:?}")
                }
            }
        )+
    };
}

impl_debug_display!(
    ActivityRecordPayload<'_>,
    ActivityMemoryPoolProperties,
    ActivityObjectId,
    ActivityUnifiedMemoryCounterFlags,
    ActivityDeviceAttributeKind,
    ActivityEnvironmentReading,
    ActivityTopologyDeviceId,
    ActivityPcieDeviceId,
    ActivityPcieAttributes,
);

/// Decoded PCI Express device topology information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPcie {
    /// PCI Express device-type tag used to decode identity and attributes.
    pub device_type: PcieDeviceType,
    /// A unique identifier for GPU or Bridge in Topology.
    pub id: ActivityPcieDeviceId,
    /// Domain for the GPU or Bridge, required to identify which PCIE bus it belongs to in multiple NUMA systems.
    pub domain: PcieDomainId,
    /// PCIE Generation of GPU or Bridge.
    pub pcie_generationeration: u16,
    /// Link rate of the GPU or bridge in gigatransfers per second (GT/s).
    pub link_rate: u16,
    /// Link width of the GPU or bridge.
    pub link_width: u16,
    /// Upstream bus ID for the GPU or PCI bridge. Required to identify which bus it is connected to in the topology.
    pub upstream_bus: u16,
    /// GPU or bridge properties selected by `device_type`.
    pub attributes: ActivityPcieAttributes,
}

impl ActivityPcie {
    fn from_raw(value: sys::CUpti_ActivityPcie) -> Self {
        let device_type = PcieDeviceType::from(value.type_);
        // SAFETY: each arm reads the union field that the matched tag selects.
        let (id, attributes) = unsafe {
            match device_type {
                PcieDeviceType::Gpu => {
                    let attributes = value.attr.gpuAttr;
                    (
                        ActivityPcieDeviceId::Gpu(CudaDevice::from(value.id.devId)),
                        ActivityPcieAttributes::Gpu(ActivityPcieGpuAttributes {
                            uuid: uuid_from_raw(attributes.uuidDev),
                            peer_devices: attributes.peerDev.map(CudaDevice::from),
                        }),
                    )
                }
                PcieDeviceType::Bridge => {
                    let attributes = value.attr.bridgeAttr;
                    (
                        ActivityPcieDeviceId::Bridge(PcieBridgeId::from(value.id.bridgeId)),
                        ActivityPcieAttributes::Bridge(ActivityPcieBridgeAttributes {
                            secondary_bus: PcieBusId::from(attributes.secondaryBus),
                            device_id: PcieHardwareDeviceId::from(attributes.deviceId),
                            vendor_id: PcieVendorId::from(attributes.vendorId),
                        }),
                    )
                }
                PcieDeviceType::ForceInt => (
                    ActivityPcieDeviceId::Bridge(PcieBridgeId::from(value.id.bridgeId)),
                    ActivityPcieAttributes::Bridge(ActivityPcieBridgeAttributes {
                        secondary_bus: PcieBusId::from(0),
                        device_id: PcieHardwareDeviceId::from(0),
                        vendor_id: PcieVendorId::from(0),
                    }),
                ),
            }
        };
        Self {
            device_type,
            id,
            domain: PcieDomainId::from(value.domain),
            pcie_generationeration: value.pcieGeneration,
            link_rate: value.linkRate,
            link_width: value.linkWidth,
            upstream_bus: value.upstreamBus,
            attributes,
        }
    }
}

/// Correlation between a device instruction and a source location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityInstructionCorrelation {
    /// The properties of this instruction.
    pub flags: ActivityFlags,
    /// The ID for source locator.
    pub source_locator_id: SourceLocatorId,
    /// Correlation ID with global/device function name.
    pub function_id: FunctionId,
    /// The pc offset for the instruction.
    pub pc_offset: u32,
}

impl ActivityInstructionCorrelation {
    fn from_raw(value: sys::CUpti_ActivityInstructionCorrelation) -> Self {
        Self {
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
            source_locator_id: SourceLocatorId::from(value.sourceLocatorId),
            function_id: FunctionId::from(value.functionId),
            pc_offset: value.pcOffset,
        }
    }
}

/// Decoded OpenMP activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOpenMultiProcessing {
    /// CUPTI OpenMP event kind (
    ///
    /// See also: [`OpenMultiProcessingEventKind`]).
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
    fn from_raw(value: sys::CUpti_ActivityOpenMp) -> Self {
        Self {
            event_kind: OpenMultiProcessingEventKind::from(value.eventKind),
            version: value.version,
            thread_id: ThreadId::from(value.threadId),
            start: value.start,
            end: value.end,
            process_id: ProcessId::from(value.cuProcessId),
            cu_thread_id: ThreadId::from(value.cuThreadId),
        }
    }
}

fn optional_cstr<'a>(ptr: *const ffi::c_char) -> Option<&'a CStr> {
    if ptr.is_null() {
        None
    } else {
        // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
        Some(unsafe { CStr::from_ptr(ptr) })
    }
}

/// Common metadata for `OpenACC` data, launch, and other activity records.
///
/// Optional strings borrow the source activity record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOpenAccBase<'a> {
    /// Kind of `OpenACC` event.
    pub event_kind: OpenAccEventKind,
    /// Enclosing `OpenACC` construct. PGI runtimes older than 16.1 report an unknown construct.
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
            event_kind: OpenAccEventKind::from($value.eventKind),
            parent_construct: OpenAccConstructKind::from($value.parentConstruct),
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
    /// Name of the variable associated with the event, when known. `None` represents a null native pointer.
    pub variable_name: Option<&'a CStr>,
}

impl ActivityOpenAccTransfer<'_> {
    fn from_raw(value: sys::CUpti_ActivityOpenAccData) -> Self {
        Self {
            base: open_acc_base_from_raw!(value),
            bytes: value.bytes,
            host_ptr: value.hostPtr,
            device_ptr: value.devicePtr,
            variable_name: optional_cstr(value.varName),
        }
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
    /// Name of the launched kernel, when known. `None` represents a null native pointer.
    pub kernel_name: Option<&'a CStr>,
}

impl ActivityOpenAccLaunch<'_> {
    fn from_raw(value: sys::CUpti_ActivityOpenAccLaunch) -> Self {
        Self {
            base: open_acc_base_from_raw!(value),
            num_gangs: value.numGangs,
            num_workers: value.numWorkers,
            vector_length: value.vectorLength,
            kernel_name: optional_cstr(value.kernelName),
        }
    }
}

/// An `OpenACC` event represented entirely by the common activity metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOpenAccOther<'a> {
    /// Common metadata describing this `OpenACC` event.
    pub base: ActivityOpenAccBase<'a>,
}

impl ActivityOpenAccOther<'_> {
    fn from_raw(value: sys::CUpti_ActivityOpenAccOther) -> Self {
        Self {
            base: open_acc_base_from_raw!(value),
        }
    }
}

/// Decoded preemption of a CUDA Dynamic Parallelism kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPreemption {
    /// kind of the preemption.
    pub preemption_kind: ActivityPreemptionKind,
    /// The timestamp of the preemption, in nanoseconds. A value of 0 indicates that timestamp information could not be collected for the preemption.
    pub timestamp: u64,
    /// The grid-id of the block that is preempted.
    pub grid_id: GridId,
    /// Coordinates of the thread block affected by the preemption.
    pub block: Dim3,
}

impl ActivityPreemption {
    fn from_raw(value: sys::CUpti_ActivityPreemption) -> Self {
        Self {
            preemption_kind: ActivityPreemptionKind::from(value.preemptionKind),
            timestamp: value.timestamp,
            grid_id: GridId::from(value.gridId),
            block: Dim3 {
                x: value.blockX,
                y: value.blockY,
                z: value.blockZ,
            },
        }
    }
}

/// Decoded device-function identity and metadata.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityFunction<'a> {
    /// ID to uniquely identify the record.
    pub id: FunctionId,
    /// The ID of the context where the function is launched.
    pub context_id: ContextId,
    /// The module ID in which this global/device function is present.
    pub module_id: ModuleId,
    /// The function's unique symbol index in the module.
    pub function_index: u32,
    /// Function name shared by records for the same kernel. `None` represents a null native pointer.
    pub name: Option<&'a CStr>,
}

impl ActivityFunction<'_> {
    fn from_raw(value: sys::CUpti_ActivityFunction) -> Self {
        let name = if value.name.is_null() {
            None
        } else {
            // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
            Some(unsafe { CStr::from_ptr(value.name) })
        };
        Self {
            id: FunctionId::from(value.id),
            context_id: ContextId::from(value.contextId),
            module_id: ModuleId::from(value.moduleId),
            function_index: value.functionIndex,
            name,
        }
    }
}

/// Decoded activity record for a CUDA module.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityModule<'a> {
    /// The ID of the context where the module is loaded.
    pub context_id: ContextId,
    /// The module ID.
    pub id: ModuleId,
    /// Borrowed CUDA binary image bytes, when the native image pointer is present. `None` represents a null native pointer.
    pub cubin: Option<&'a [u8]>,
}

impl ActivityModule<'_> {
    fn from_raw(value: sys::CUpti_ActivityModule) -> Self {
        let cubin = if value.cubin.is_null() || value.cubinSize == 0 {
            None
        } else {
            // SAFETY: CUPTI stores `cubinSize` bytes behind the non-null cubin pointer, valid while the record is borrowed.
            Some(unsafe {
                slice::from_raw_parts(value.cubin.cast::<u8>(), value.cubinSize as usize)
            })
        };
        Self {
            context_id: ContextId::from(value.contextId),
            id: ModuleId::from(value.id),
            cubin,
        }
    }
}

/// Decoded activity record for source-level shared access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivitySharedAccess {
    /// The properties of this shared access.
    pub flags: ActivityFlags,
    /// The ID for source locator.
    pub source_locator_id: SourceLocatorId,
    /// The correlation ID of the kernel to which this result is associated.
    pub correlation_id: CorrelationId,
    /// Correlation ID with global/device function name.
    pub function_id: FunctionId,
    /// The pc offset for the access.
    pub pc_offset: u32,
    /// This increments each time when this instruction is executed by number of threads that executed this instruction with predicate and condition code evaluating to true.
    pub threads_executed: u64,
    /// The total number of shared memory transactions generated by this access.
    pub shared_transactions: u64,
    /// The minimum number of shared memory transactions possible based on the access pattern.
    pub theoretical_shared_transactions: u64,
    /// The number of times this instruction was executed per warp. It will be incremented when at least one of thread among warp is active with predicate and condition code evaluating to true.
    pub executed: u32,
}

impl ActivitySharedAccess {
    fn from_raw(value: sys::CUpti_ActivitySharedAccess) -> Self {
        Self {
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
            source_locator_id: SourceLocatorId::from(value.sourceLocatorId),
            correlation_id: CorrelationId::from(value.correlationId),
            function_id: FunctionId::from(value.functionId),
            pc_offset: value.pcOffset,
            threads_executed: value.threadsExecuted,
            shared_transactions: value.sharedTransactions,
            theoretical_shared_transactions: value.theoreticalSharedTransactions,
            executed: value.executed,
        }
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
    fn from_raw(value: sys::CUpti_ActivityConfidentialComputeRotation) -> Self {
        Self {
            event_type: ConfidentialComputeRotationEventType::from(value.eventType),
            device_id: DeviceId::from(value.deviceId),
            context_id: ContextId::from(value.contextId),
            channel_id: ChannelId::from(value.channelId),
            channel_type: ChannelType::from(value.channelType),
            timestamp: value.timestamp,
        }
    }
}

/// Decoded source file and line location.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivitySourceLocator<'a> {
    /// The ID for the source path, will be used in all the source level results.
    pub id: SourceLocatorId,
    /// The line number in the source .
    pub line_number: u32,
    /// Source file path, when available. `None` represents a null native pointer.
    pub file_name: Option<&'a CStr>,
}

impl ActivitySourceLocator<'_> {
    fn from_raw(value: sys::CUpti_ActivitySourceLocator) -> Self {
        let file_name = if value.fileName.is_null() {
            None
        } else {
            // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
            Some(unsafe { CStr::from_ptr(value.fileName) })
        };
        Self {
            id: SourceLocatorId::from(value.id),
            line_number: value.lineNumber,
            file_name,
        }
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
    /// The start timestamp for the overhead, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the overhead.
    pub start: u64,
    /// The end timestamp for the overhead, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the overhead.
    pub end: u64,
    /// The correlation ID of the overhead operation to which records belong to. This ID is identical to the correlation ID in the driver or runtime API activity record that launched the overhead operation. In some cases, it can be zero, such as for [`ActivityOverheadKind::BufferFlush`] records.
    pub correlation_id: CorrelationId,
    /// Whether the native record includes an additional overhead payload. This flag records its presence without exposing its pointer.
    pub has_overhead_data: bool,
}

impl ActivityOverhead {
    fn from_raw(value: sys::CUpti_ActivityOverhead3) -> Self {
        let object_kind = ActivityObjectKind::from(value.objectKind);
        Self {
            overhead_kind: ActivityOverheadKind::from(value.overheadKind),
            object_kind,
            object_id: ActivityObjectId::from_raw(object_kind, value.objectId),
            start: value.start,
            end: value.end,
            correlation_id: CorrelationId::from(value.correlationId),
            has_overhead_data: !value.overheadData.is_null(),
        }
    }
}

/// Decoded CUDA kernel execution activity.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityKernel<'a> {
    /// Execution model of the kernel function.
    pub execution_model: FunctionExecutionModel,
    /// Kernel name shared by records for the same kernel. `None` represents a null native pointer.
    pub name: Option<&'a CStr>,
    /// The ID of the device where the kernel is executing.
    pub device_id: DeviceId,
    /// The ID of the context where the kernel is executing.
    pub context_id: ContextId,
    /// The ID of the stream where the kernel is executing.
    pub stream_id: StreamId,
    /// The correlation ID of the kernel. Each kernel execution is assigned a unique correlation ID that is identical to the correlation ID in the driver or runtime API activity record that launched the kernel.
    pub correlation_id: CorrelationId,
    /// The grid ID of the kernel. Each kernel is assigned a unique grid ID at runtime.
    pub grid_id: GridId,
    /// Grid dimensions in thread blocks.
    pub grid: Dim3,
    /// Thread-block dimensions in threads.
    pub block: Dim3,
    /// Thread-block cluster dimensions; valid for compute capability 9.0 and later.
    pub cluster: Dim3,
    /// The start timestamp for the kernel execution, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the kernel.
    pub start: u64,
    /// The end timestamp for the kernel execution, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the kernel.
    pub end: u64,
    /// The completed timestamp for the kernel execution, in nanoseconds.  It represents the completion of all its child kernels and the kernel itself. A value of `CUPTI_TIMESTAMP_UNKNOWN` indicates that the completion time is unknown.
    pub completed: u64,
    /// The timestamp when the kernel is queued up in the command buffer, in nanoseconds. A value of `CUPTI_TIMESTAMP_UNKNOWN` indicates that the queued time could not be collected for the kernel. This timestamp is not collected by default. Use API [`enable_latency_timestamulti_process_service`]() to enable collection.
    ///
    /// Command buffer is a buffer written by CUDA driver to send commands like kernel launch, memory copy etc to the GPU. All launches of CUDA kernels are asynchronous with respect to the host, the host requests the launch by writing commands into the command buffer, then returns without checking the GPU's progress.
    pub queued: u64,
    /// The timestamp when the command buffer containing the kernel launch is submitted to the GPU, in nanoseconds. A value of `CUPTI_TIMESTAMP_UNKNOWN` indicates that the submitted time could not be collected for the kernel. This timestamp is not collected by default. Use API [`enable_latency_timestamulti_process_service`]() to enable collection.
    pub submitted: u64,
    /// The static shared memory allocated for the kernel, in bytes.
    pub static_shared_memory: u32,
    /// The dynamic shared memory reserved for the kernel, in bytes.
    pub dynamic_shared_memory: u32,
    /// The amount of local memory reserved for each thread, in bytes.
    pub local_memory_per_thread: u32,
    /// The total amount of local memory reserved for the kernel, in bytes.
    pub local_memory_total: u64,
    /// The number of registers required for each thread executing the kernel.
    pub registers_per_thread: u16,
    /// The shared memory configuration used for the kernel. The value is one of the `CUsharedconfig` enumeration values from `cuda.h`.
    pub shared_memory_config: u8,
    /// Shared memory size set by the driver.
    pub shared_memory_executed: u32,
    /// Requested shared-memory carveout percentage, or `None` when no carveout was requested.
    pub shared_memory_carveout_requested: Option<u8>,
    /// The partitioned global caching requested for the kernel. Partitioned global caching is required to enable caching on certain chips, such as devices with compute capability 5.2.
    pub partitioned_global_cache_requested: ActivityPartitionedGlobalCacheConfig,
    /// The partitioned global caching executed for the kernel. Partitioned global caching is required to enable caching on certain chips, such as devices with compute capability 5.2. Partitioned global caching can be automatically disabled if the occupancy requirement of the launch cannot support caching.
    pub partitioned_global_cache_executed: ActivityPartitionedGlobalCacheConfig,
    /// Whether the kernel uses a regular launch or a single-device or multi-device cooperative launch.
    pub launch_type: ActivityLaunchType,
    /// The shared memory limit config for the kernel. This field shows whether user has opted for a higher per block limit of dynamic shared memory.
    pub shared_memory_limit_config: FunctionSharedMemoryLimitConfig,
    /// Graph node ID in the source graph from which the executable graph was instantiated.
    pub source_graph_node_id: GraphNodeId,
    /// ID of the source graph from which the executable graph was instantiated.
    pub source_graph_id: GraphId,
    /// The unique ID of the graph node that launched this kernel through graph launch APIs. This field will be 0 if the kernel is not launched through graph launch APIs.
    pub graph_node_id: GraphNodeId,
    /// The unique ID of the graph that launched this kernel through graph launch APIs. This field will be 0 if the kernel is not launched through graph launch APIs.
    pub graph_id: GraphId,
    /// The ID of the HW channel on which the kernel is launched.
    pub channel_id: ChannelId,
    /// The type of the channel.
    pub channel_type: ChannelType,
    /// The maximum cluster size for the kernel.
    pub max_potential_cluster_size: u32,
    /// The maximum clusters that could co-exist on the target device for the kernel.
    pub max_active_clusters: u32,
    /// Whether the kernel belongs to a device-launched graph.
    pub is_device_launched: bool,
    /// The launch priority of the kernel.
    pub priority: i32,
}

impl ActivityKernel<'_> {
    fn from_raw(value: sys::CUpti_ActivityKernel13) -> Self {
        let name = if value.name.is_null() {
            None
        } else {
            // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
            Some(unsafe { CStr::from_ptr(value.name) })
        };

        Self {
            execution_model: FunctionExecutionModel::from(value.executionModel),
            name,
            device_id: DeviceId::from(value.deviceId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            correlation_id: CorrelationId::from(value.correlationId),
            grid_id: GridId::from(value.gridId),
            grid: Dim3 {
                x: value.gridX as u32,
                y: value.gridY as u32,
                z: value.gridZ as u32,
            },
            block: Dim3 {
                x: value.blockX as u32,
                y: value.blockY as u32,
                z: value.blockZ as u32,
            },
            cluster: Dim3 {
                x: value.clusterX,
                y: value.clusterY,
                z: value.clusterZ,
            },
            start: value.start,
            end: value.end,
            completed: value.completed,
            queued: value.queued,
            submitted: value.submitted,
            static_shared_memory: value.staticSharedMemory as u32,
            dynamic_shared_memory: value.dynamicSharedMemory as u32,
            local_memory_per_thread: value.localMemoryPerThread,
            local_memory_total: value.localMemoryTotal_v2,
            registers_per_thread: value.registersPerThread,
            shared_memory_config: value.sharedMemoryConfig,
            shared_memory_executed: value.sharedMemoryExecuted,
            shared_memory_carveout_requested: (value.isSharedMemoryCarveoutRequested != 0)
                .then_some(value.sharedMemoryCarveoutRequested),
            partitioned_global_cache_requested: partitioned_global_cache_config_from_raw(
                value.partitionedGlobalCacheRequested,
            ),
            partitioned_global_cache_executed: partitioned_global_cache_config_from_raw(
                value.partitionedGlobalCacheExecuted,
            ),
            launch_type: launch_type_from_raw(value.launchType),
            shared_memory_limit_config: function_shared_memory_limit_config_from_raw(
                value.shmemLimitConfig,
            ),
            source_graph_node_id: value.sourceGraphNodeId.into(),
            source_graph_id: value.sourceGraphId.into(),
            graph_node_id: GraphNodeId::from(value.graphNodeId),
            graph_id: GraphId::from(value.graphId),
            channel_id: ChannelId::from(value.channelID),
            channel_type: ChannelType::from(value.channelType),
            max_potential_cluster_size: value.maxPotentialClusterSize,
            max_active_clusters: value.maxActiveClusters,
            is_device_launched: value.isDeviceLaunched != 0,
            priority: value.priority,
        }
    }
}

/// Decoded CUDA Dynamic Parallelism kernel execution activity.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityCudaDynamicParallelismKernel<'a> {
    /// Kernel name shared by records for the same kernel. `None` represents a null native pointer.
    pub name: Option<&'a CStr>,
    /// The ID of the device where the kernel is executing.
    pub device_id: DeviceId,
    /// The ID of the context where the kernel is executing.
    pub context_id: ContextId,
    /// The ID of the stream where the kernel is executing.
    pub stream_id: StreamId,
    /// The correlation ID of the kernel. Each kernel execution is assigned a unique correlation ID that is identical to the correlation ID in the driver API activity record that launched the kernel.
    pub correlation_id: CorrelationId,
    /// The grid ID of the kernel. Each kernel execution is assigned a unique grid ID.
    pub grid_id: GridId,
    /// The grid ID of the parent kernel.
    pub parent_grid_id: GridId,
    /// Coordinates of the parent thread block that launched this kernel.
    pub parent_block: Dim3,
    /// Grid dimensions in thread blocks.
    pub grid: Dim3,
    /// Thread-block dimensions in threads.
    pub block: Dim3,
    /// The start timestamp for the kernel execution, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the kernel.
    pub start: u64,
    /// The end timestamp for the kernel execution, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the kernel.
    pub end: u64,
    /// The timestamp when kernel is marked as completed, in nanoseconds. A value of `CUPTI_TIMESTAMP_UNKNOWN` indicates that the completion time is unknown.
    pub completed: u64,
    /// The timestamp when kernel is queued up, in nanoseconds. A value of `CUPTI_TIMESTAMP_UNKNOWN` indicates that the queued time is unknown.
    pub queued: u64,
    /// The timestamp when kernel is submitted to the gpu, in nanoseconds. A value of `CUPTI_TIMESTAMP_UNKNOWN` indicates that the submission time is unknown.
    pub submitted: u64,
    /// The static shared memory allocated for the kernel, in bytes.
    pub static_shared_memory: u32,
    /// The dynamic shared memory reserved for the kernel, in bytes.
    pub dynamic_shared_memory: u32,
    /// The amount of local memory reserved for each thread, in bytes.
    pub local_memory_per_thread: u32,
    /// The total amount of local memory reserved for the kernel, in bytes.
    pub local_memory_total: u32,
    /// The number of registers required for each thread executing the kernel.
    pub registers_per_thread: u16,
    /// The shared memory configuration used for the kernel. The value is one of the `CUsharedconfig` enumeration values from `cuda.h`.
    pub shared_memory_config: u8,
}

impl ActivityCudaDynamicParallelismKernel<'_> {
    fn from_raw(value: sys::CUpti_ActivityCdpKernel) -> Self {
        Self {
            name: optional_cstr(value.name),
            device_id: DeviceId::from(value.deviceId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            correlation_id: CorrelationId::from(value.correlationId),
            grid_id: GridId::from(value.gridId),
            parent_grid_id: GridId::from(value.parentGridId),
            parent_block: Dim3 {
                x: value.parentBlockX,
                y: value.parentBlockY,
                z: value.parentBlockZ,
            },
            grid: Dim3 {
                x: value.gridX as u32,
                y: value.gridY as u32,
                z: value.gridZ as u32,
            },
            block: Dim3 {
                x: value.blockX as u32,
                y: value.blockY as u32,
                z: value.blockZ as u32,
            },
            start: value.start,
            end: value.end,
            completed: value.completed,
            queued: value.queued,
            submitted: value.submitted,
            static_shared_memory: value.staticSharedMemory as u32,
            dynamic_shared_memory: value.dynamicSharedMemory as u32,
            local_memory_per_thread: value.localMemoryPerThread,
            local_memory_total: value.localMemoryTotal,
            registers_per_thread: value.registersPerThread,
            shared_memory_config: value.sharedMemoryConfig,
        }
    }
}

/// Decoded device memory-decompression activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemoryDecompress {
    /// The ID of the device.
    pub device_id: DeviceId,
    /// The ID of the context.
    pub context_id: ContextId,
    /// The ID of the stream.
    pub stream_id: StreamId,
    /// The ID of the HW channel on which the memory copy is occurring.
    pub channel_id: ChannelId,
    /// The type of the channel.
    pub channel_type: ChannelType,
    /// The correlation ID of the decompression operations. Each operation is assigned a unique correlation ID that is identical to the correlation ID in the driver API activity record that launched the operation.
    pub correlation_id: CorrelationId,
    /// The number of operations in the batch.
    pub number_of_operations: u32,
    /// The number of bytes to be read and decompressed in the batch operation.
    pub source_bytes: u64,
    /// The start timestamp. A value of `CUPTI_TIMESTAMP_UNKNOWN` indicates that the start time is unknown.
    pub start: u64,
    /// The end timestamp. A value of `CUPTI_TIMESTAMP_UNKNOWN` indicates that the start time is unknown.
    pub end: u64,
}

impl ActivityMemoryDecompress {
    fn from_raw(value: sys::CUpti_ActivityMemDecompress) -> Self {
        Self {
            device_id: DeviceId::from(value.deviceId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            channel_id: ChannelId::from(value.channelID),
            channel_type: ChannelType::from(value.channelType),
            correlation_id: CorrelationId::from(value.correlationId),
            number_of_operations: value.numberOfOperations,
            source_bytes: value.sourceBytes,
            start: value.start,
            end: value.end,
        }
    }
}

/// Decoded activity record for memory copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemoryCopy {
    /// Kind of memory copy.
    pub copy_kind: ActivityMemoryCopyKind,
    /// Memory kind read by the copy.
    pub source_kind: ActivityMemoryKind,
    /// Memory kind written by the copy.
    pub destination_kind: ActivityMemoryKind,
    /// The flags associated with the memory copy.
    ///
    /// See also: `CUpti_ActivityFlag`
    pub flags: ActivityFlags,
    /// The number of bytes transferred by the memory copy.
    pub bytes: u64,
    /// The start timestamp for the memory copy, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the memory copy.
    pub start: u64,
    /// The end timestamp for the memory copy, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the memory copy.
    pub end: u64,
    /// The ID of the device where the memory copy is occurring.
    pub device_id: DeviceId,
    /// The ID of the context where the memory copy is occurring.
    pub context_id: ContextId,
    /// The ID of the stream where the memory copy is occurring.
    pub stream_id: StreamId,
    /// The correlation ID of the memory copy. Each memory copy is assigned a unique correlation ID that is identical to the correlation ID in the driver API activity record that launched the memory copy.
    pub correlation_id: CorrelationId,
    /// The runtime correlation ID of the memory copy. Each memory copy is assigned a unique runtime correlation ID that is identical to the correlation ID in the runtime API activity record that launched the memory copy.
    pub runtime_correlation_id: CorrelationId,
    /// Graph node ID in the source graph from which the executable graph was instantiated.
    pub source_graph_node_id: GraphNodeId,
    /// ID of the source graph from which the executable graph was instantiated.
    pub source_graph_id: GraphId,
    /// The unique ID of the graph node that executed this memcpy through graph launch. This field will be 0 if the memcpy is not done through graph launch.
    pub graph_node_id: GraphNodeId,
    /// The unique ID of the graph that executed this memcpy through graph launch. This field will be 0 if the memcpy is not done through graph launch.
    pub graph_id: GraphId,
    /// The ID of the HW channel on which the memory copy is occurring.
    pub channel_id: ChannelId,
    /// The type of the channel.
    pub channel_type: ChannelType,
    /// Whether the copy belongs to a device-launched graph.
    pub is_device_launched: bool,
    /// The total number of memcopy operations traced in this record. This field is valid for memcpy operations happening using `MemcpyBatchAsync` APIs in CUDA. In `MemcpyBatchAsync` APIs, multiple memcpy operations are batched together for optimization purposes based on certain heuristics. For other memcpy operations, this field will be 1.
    pub copy_count: u64,
}

impl ActivityMemoryCopy {
    fn from_raw(value: sys::CUpti_ActivityMemcpy7) -> Self {
        Self {
            copy_kind: memory_copy_kind_from_raw(value.copyKind),
            source_kind: memory_kind_from_raw(u32::from(value.srcKind)),
            destination_kind: memory_kind_from_raw(u32::from(value.dstKind)),
            flags: ActivityFlags::from_bits_truncate(u32::from(value.flags)),
            bytes: value.bytes,
            start: value.start,
            end: value.end,
            device_id: DeviceId::from(value.deviceId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            correlation_id: CorrelationId::from(value.correlationId),
            runtime_correlation_id: CorrelationId::from(value.runtimeCorrelationId),
            source_graph_node_id: value.sourceGraphNodeId.into(),
            source_graph_id: value.sourceGraphId.into(),
            graph_node_id: GraphNodeId::from(value.graphNodeId),
            graph_id: GraphId::from(value.graphId),
            channel_id: ChannelId::from(value.channelID),
            channel_type: ChannelType::from(value.channelType),
            is_device_launched: value.isDeviceLaunched != 0,
            copy_count: value.copyCount,
        }
    }
}

/// Decoded activity record for peer-to-peer memory copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPeerMemoryCopy {
    /// Kind of memory copy.
    pub copy_kind: ActivityMemoryCopyKind,
    /// Memory kind read by the copy.
    pub source_kind: ActivityMemoryKind,
    /// Memory kind written by the copy.
    pub destination_kind: ActivityMemoryKind,
    /// The flags associated with the memory copy.
    ///
    /// See also: `CUpti_ActivityFlag`
    pub flags: ActivityFlags,
    /// The number of bytes transferred by the memory copy.
    pub bytes: u64,
    /// The start timestamp for the memory copy, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the memory copy.
    pub start: u64,
    /// The end timestamp for the memory copy, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the memory copy.
    pub end: u64,
    /// The ID of the device where the memory copy is occurring.
    pub device_id: DeviceId,
    /// The ID of the context where the memory copy is occurring.
    pub context_id: ContextId,
    /// The ID of the stream where the memory copy is occurring.
    pub stream_id: StreamId,
    /// The ID of the device where memory is being copied from.
    pub source_device_id: DeviceId,
    /// The ID of the context owning the memory being copied from.
    pub source_context_id: ContextId,
    /// The ID of the device where memory is being copied to.
    pub destination_device_id: DeviceId,
    /// The ID of the context owning the memory being copied to.
    pub destination_context_id: ContextId,
    /// The correlation ID of the memory copy. Each memory copy is assigned a unique correlation ID that is identical to the correlation ID in the driver and runtime API activity record that launched the memory copy.
    pub correlation_id: CorrelationId,
    /// The unique ID of the graph node that executed the memcpy through graph launch. This field will be 0 if memcpy is not done using graph launch.
    pub graph_node_id: GraphNodeId,
    /// The unique ID of the graph that executed this memcpy through graph launch. This field will be 0 if the memcpy is not done through graph launch.
    pub graph_id: GraphId,
    /// The ID of the HW channel on which the memory copy is occurring.
    pub channel_id: ChannelId,
    /// The type of the channel.
    pub channel_type: ChannelType,
}

impl ActivityPeerMemoryCopy {
    fn from_raw(value: sys::CUpti_ActivityMemcpyPtoP4) -> Self {
        Self {
            copy_kind: memory_copy_kind_from_raw(value.copyKind),
            source_kind: memory_kind_from_raw(u32::from(value.srcKind)),
            destination_kind: memory_kind_from_raw(u32::from(value.dstKind)),
            flags: ActivityFlags::from_bits_truncate(u32::from(value.flags)),
            bytes: value.bytes,
            start: value.start,
            end: value.end,
            device_id: DeviceId::from(value.deviceId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            source_device_id: DeviceId::from(value.srcDeviceId),
            source_context_id: ContextId::from(value.srcContextId),
            destination_device_id: DeviceId::from(value.dstDeviceId),
            destination_context_id: ContextId::from(value.dstContextId),
            correlation_id: CorrelationId::from(value.correlationId),
            graph_node_id: GraphNodeId::from(value.graphNodeId),
            graph_id: GraphId::from(value.graphId),
            channel_id: ChannelId::from(value.channelID),
            channel_type: ChannelType::from(value.channelType),
        }
    }
}

/// Decoded device memory-set activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemorySet {
    /// The value being assigned to memory by the memory set.
    pub value: u32,
    /// The number of bytes being set by the memory set.
    pub bytes: u64,
    /// The start timestamp for the memory set, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the memory set.
    pub start: u64,
    /// The end timestamp for the memory set, in nanoseconds. A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the memory set.
    pub end: u64,
    /// The ID of the device where the memory set is occurring.
    pub device_id: DeviceId,
    /// The ID of the context where the memory set is occurring.
    pub context_id: ContextId,
    /// The ID of the stream where the memory set is occurring.
    pub stream_id: StreamId,
    /// The correlation ID of the memory set. Each memory set is assigned a unique correlation ID that is identical to the correlation ID in the driver API activity record that launched the memory set.
    pub correlation_id: CorrelationId,
    /// The flags associated with the memset.
    ///
    /// See also: `CUpti_ActivityFlag`
    pub flags: ActivityFlags,
    /// The memory kind of the memory set
    ///
    /// See also: [`ActivityMemoryKind`].
    pub memory_kind: ActivityMemoryKind,
    /// Graph node ID in the source graph from which the executable graph was instantiated.
    pub source_graph_node_id: GraphNodeId,
    /// ID of the source graph from which the executable graph was instantiated.
    pub source_graph_id: GraphId,
    /// The unique ID of the graph node that executed this memset through graph launch. This field will be 0 if the memset is not executed through graph launch.
    pub graph_node_id: GraphNodeId,
    /// The unique ID of the graph that executed this memset through graph launch. This field will be 0 if the memset is not executed through graph launch.
    pub graph_id: GraphId,
    /// The ID of the HW channel on which the memory set is occurring.
    pub channel_id: ChannelId,
    /// The type of the channel.
    pub channel_type: ChannelType,
    /// Whether the memory set belongs to a device-launched graph.
    pub is_device_launched: bool,
}

impl ActivityMemorySet {
    fn from_raw(value: sys::CUpti_ActivityMemset5) -> Self {
        Self {
            value: value.value,
            bytes: value.bytes,
            start: value.start,
            end: value.end,
            device_id: DeviceId::from(value.deviceId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            correlation_id: CorrelationId::from(value.correlationId),
            flags: ActivityFlags::from_bits_truncate(u32::from(value.flags)),
            memory_kind: memory_kind_from_raw(u32::from(value.memoryKind)),
            source_graph_node_id: value.sourceGraphNodeId.into(),
            source_graph_id: value.sourceGraphId.into(),
            graph_node_id: GraphNodeId::from(value.graphNodeId),
            graph_id: GraphId::from(value.graphId),
            channel_id: ChannelId::from(value.channelID),
            channel_type: ChannelType::from(value.channelType),
            is_device_launched: value.isDeviceLaunched != 0,
        }
    }
}

/// Iterator over borrowed records in a completed activity buffer.
///
/// Stops at the end of the buffer. A native iteration error is yielded once as `Err`, after which iteration stops.
#[derive(Debug)]
pub struct ActivityRecords<'a> {
    buffer: *mut u8,
    valid_size_bytes: u64,
    current: *mut sys::CUpti_Activity,
    finished: bool,
    _marker: PhantomData<&'a [u8]>,
}

impl<'a> Iterator for ActivityRecords<'a> {
    type Item = Result<ActivityRecord<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // SAFETY: `cuptiActivityGetNextRecord` receives valid output locations.
        let status = unsafe {
            sys::cuptiActivityGetNextRecord(
                self.buffer,
                self.valid_size_bytes,
                &raw mut self.current,
            )
        };

        match status {
            sys::CUptiResult::CUPTI_SUCCESS => Some(Ok(ActivityRecord {
                ptr: self.current.cast_const(),
                _marker: PhantomData,
            })),
            sys::CUptiResult::CUPTI_ERROR_QUEUE_EMPTY => {
                self.finished = true;
                None
            }
            status => {
                self.finished = true;
                Some(Err(Error::from(status)))
            }
        }
    }
}

/// Requested activity-buffer capacity and record limit for completion callbacks.
#[derive(Debug, Clone, Copy)]
pub struct ActivityBufferCallbackConfig {
    /// Buffer allocation size in bytes.
    ///
    /// When CUPTI requests a buffer, a zero size or an invalid allocation layout supplies no buffer.
    pub buffer_size: usize,
    /// Maximum records CUPTI may write per buffer; zero means no record-count limit.
    pub max_num_records: usize,
}

impl ActivityBufferCallbackConfig {
    /// Records a requested buffer capacity in bytes with no record-count limit.
    ///
    /// The allocation layout is checked when CUPTI requests a buffer.
    #[must_use]
    pub const fn new(buffer_size: usize) -> Self {
        Self {
            buffer_size,
            max_num_records: 0,
        }
    }

    /// Sets the maximum records per buffer; zero removes the record-count limit.
    #[must_use]
    pub fn with_max_num_records(mut self, max_num_records: usize) -> Self {
        self.max_num_records = max_num_records;
        self
    }
}

/// Collects byte copies of completed activity buffers behind a shared mutex.
///
/// Buffers accumulate until [`Self::take_buffers`] removes them. Dropping the collector clears its callbacks if it still owns the current registration.
#[derive(Debug)]
pub struct ActivityCollector {
    owner_id: u64,
    buffers: Arc<Mutex<Vec<ActivityBufferOwned>>>,
}

impl ActivityCollector {
    /// Registers CUPTI activity callbacks and stores completed buffers in memory.
    ///
    /// CUPTI supports one process-wide activity buffer callback.
    /// Creating a new collector replaces the previous callback owner.
    ///
    /// # Errors
    ///
    /// Returns an error if the activity callback-state mutex is poisoned or native callback registration fails.
    pub fn create(config: ActivityBufferCallbackConfig) -> Result<Self> {
        let owner_id = NEXT_ACTIVITY_CALLBACK_OWNER_ID.fetch_add(1, Ordering::Relaxed);
        let buffers = Arc::new(Mutex::new(Vec::new()));
        let callback_buffers = Arc::clone(&buffers);

        register_callbacks_with_owner(Some(owner_id), config, move |buffer| {
            if let Ok(mut callback_buffers) = callback_buffers.lock() {
                callback_buffers.push(ActivityBufferOwned {
                    bytes: buffer.bytes().to_vec(),
                });
            }
        })?;

        Ok(Self { owner_id, buffers })
    }

    /// Globally enables collection of `kind`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI is uninitialized, the activity kind is invalid or unsupported, or collection conflicts with the current profiling configuration.
    pub fn enable(&self, kind: ActivityKind) -> Result<()> {
        enable_global(kind)
    }

    /// Globally disables collection of `kind`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI is uninitialized or rejects the activity kind.
    pub fn disable(&self, kind: ActivityKind) -> Result<()> {
        disable_global(kind)
    }

    /// Flushes activity buffers using the requested completion policy.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot flush the process-wide activity buffers.
    pub fn flush_all(&self, flag: ActivityFlushFlag) -> Result<()> {
        flush_all(flag)
    }

    /// Returns the number of completed buffers currently stored by the collector.
    ///
    /// # Errors
    ///
    /// Returns an error if the collector-buffer mutex is poisoned.
    pub fn len(&self) -> Result<usize> {
        Ok(self
            .buffers
            .lock()
            .map_err(|_| Error::LockPoisoned {
                name: "activity collector buffers".into(),
            })?
            .len())
    }

    /// Returns whether the collector currently stores no completed buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if the collector-buffer mutex is poisoned.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Removes all completed buffers from this collector and returns them.
    ///
    /// # Errors
    ///
    /// Returns an error if the collector-buffer mutex is poisoned.
    pub fn take_buffers(&self) -> Result<Vec<ActivityBufferOwned>> {
        Ok(mem::take(&mut *self.buffers.lock().map_err(|_| {
            Error::LockPoisoned {
                name: "activity collector buffers".into(),
            }
        })?))
    }
}

impl Drop for ActivityCollector {
    fn drop(&mut self) {
        clear_activity_callbacks_if_owner(self.owner_id);
    }
}

/// Scope of an activity-buffer flush: the process, a context, or a stream.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ActivityFlushTarget<'a> {
    /// Flush buffers across the process.
    Global,
    /// Flush buffers for the supplied CUDA context.
    Context(&'a Context),
    /// Flush buffers for one stream in the supplied CUDA context.
    Stream {
        /// CUDA context containing the stream to flush.
        context: &'a Context,
        /// CUPTI identifier of the stream to flush.
        stream_id: StreamId,
    },
}

impl Display for ActivityFlushTarget<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Global => write!(f, "global"),
            Self::Context(_) => write!(f, "context"),
            Self::Stream { .. } => write!(f, "stream"),
        }
    }
}

impl ActivityFlushTarget<'_> {
    fn context_and_stream_id(self) -> Result<(sys::CUcontext, u32)> {
        match self {
            Self::Global => Ok((ptr::null_mut(), 0)),
            Self::Context(context) => Ok((context.as_raw(), 0)),
            Self::Stream { context, stream_id } => {
                Ok((context.as_raw(), to_u32(stream_id.as_raw(), "stream_id")?))
            }
        }
    }
}

impl Context {
    /// Enables collection of an activity kind for this context.
    ///
    /// Context-specific activity settings override global activity settings.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
    /// - Returns [`crate::error::Status::NotCompatible`] if the activity kind cannot be enabled in the current configuration.
    pub fn enable_activity(&self, kind: ActivityKind) -> Result<()> {
        // SAFETY: `cuptiActivityEnableContext` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cuptiActivityEnableContext(self.as_raw(), kind.into()))?;
        }
        Ok(())
    }

    /// Disables collection of an activity kind for this context.
    ///
    /// Context-specific activity settings override global activity settings.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
    pub fn disable_activity(&self, kind: ActivityKind) -> Result<()> {
        // SAFETY: `cuptiActivityDisableContext` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cuptiActivityDisableContext(self.as_raw(), kind.into()))?;
        }
        Ok(())
    }

    /// Configures legacy PC sampling for this context.
    ///
    /// For Pascal and older GPUs, configure PC sampling before enabling [`crate::types::ActivityKind::PcSampling`]. Starting with CUDA 13.0 this legacy profiler path is unsupported.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidOperation`] if event collection is configured in a way that conflicts with PC sampling.
    /// - Returns [`crate::error::Status::InvalidParameter`] if the PC sampling configuration is invalid.
    /// - Returns [`crate::error::Status::NotSupported`] if the system or device does not support legacy PC sampling.
    pub fn configure_pc_sampling(&self, config: &mut ActivityPcSamplingConfig) -> Result<()> {
        let mut config = config.to_raw();
        // SAFETY: `cuptiActivityConfigurePCSampling` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cuptiActivityConfigurePCSampling(
                self.as_raw(),
                &raw mut config,
            ))?;
        }
        Ok(())
    }

    /// Returns and resets the number of activity records dropped for a stream in this context.
    ///
    /// Dropped records indicate insufficient activity buffer space, including device-side buffers for CUDA Dynamic Parallelism records.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
    pub fn num_dropped_records(&self, stream_id: StreamId) -> Result<u64> {
        let mut dropped = 0;
        let stream_id = to_u32(stream_id.as_raw(), "stream_id")?;
        // SAFETY: `cuptiActivityGetNumDroppedRecords` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cuptiActivityGetNumDroppedRecords(
                self.as_raw(),
                stream_id,
                &raw mut dropped,
            ))?;
        }
        Ok(dropped)
    }
}

/// Globally enables collection of an activity kind.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
/// - Returns [`crate::error::Status::NotCompatible`] if the activity kind cannot be enabled in the current configuration.
pub fn enable_global(kind: ActivityKind) -> Result<()> {
    // SAFETY: `cuptiActivityEnable` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityEnable(kind.into()))?;
    }
    Ok(())
}

/// Globally disables collection of an activity kind.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
pub fn disable_global(kind: ActivityKind) -> Result<()> {
    // SAFETY: `cuptiActivityDisable` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityDisable(kind.into()))?;
    }
    Ok(())
}

/// Globally enables collection of an activity kind and asks CUPTI to emit existing resource records when supported.
///
/// Register activity buffer callbacks before calling this; otherwise CUPTI can enable the kind without delivering records. Resource dumping is mainly useful for late attach to existing devices, contexts, streams, `NVLink`, and `PCIe` records.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
/// - Returns [`crate::error::Status::NotCompatible`] if the activity kind cannot be enabled in the current configuration.
/// - Returns [`crate::error::Status::UnknownError`] if CUPTI reports an internal error.
pub fn enable_global_and_dump(kind: ActivityKind) -> Result<()> {
    // SAFETY: `cuptiActivityEnableAndDump` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableAndDump(kind.into()))?;
    }
    Ok(())
}

/// Enables collection of an activity kind for a subscriber.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
/// - Returns [`crate::error::Status::NotCompatible`] if the activity kind cannot be enabled in the current configuration.
pub fn enable(
    subscriber: &Subscriber,
    kind: ActivityKind,
    config: &mut ActivityConfig,
) -> Result<()> {
    let subscriber = subscriber.as_raw()?;
    config.validate_kind(kind)?;
    // SAFETY: `cuptiActivityEnable_v2` receives only plain values.
    config.with_raw(|config| unsafe {
        try_ffi!(sys::cuptiActivityEnable_v2(subscriber, kind.into(), config,))?;
        Ok(())
    })
}

/// Disables collection of an activity kind for a subscriber.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
pub fn disable(
    subscriber: &Subscriber,
    kind: ActivityKind,
    config: &mut ActivityConfig,
) -> Result<()> {
    let subscriber = subscriber.as_raw()?;
    config.validate_kind(kind)?;
    // SAFETY: `cuptiActivityDisable_v2` receives only plain values.
    config.with_raw(|config| unsafe {
        try_ffi!(sys::cuptiActivityDisable_v2(
            subscriber,
            kind.into(),
            config,
        ))?;
        Ok(())
    })
}

/// Enables collection of an activity kind for a subscriber and asks CUPTI to emit existing resource records when supported.
///
/// Register activity buffer callbacks before calling this; otherwise CUPTI can enable the kind without delivering records.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
/// - Returns [`crate::error::Status::NotCompatible`] if the activity kind cannot be enabled in the current configuration.
/// - Returns [`crate::error::Status::UnknownError`] if CUPTI reports an internal error.
pub fn enable_and_dump(
    subscriber: &Subscriber,
    kind: ActivityKind,
    config: &mut ActivityConfig,
) -> Result<()> {
    let subscriber = subscriber.as_raw()?;
    config.validate_kind(kind)?;
    // SAFETY: `cuptiActivityEnableAndDump_v2` receives only plain values.
    config.with_raw(|config| unsafe {
        try_ffi!(sys::cuptiActivityEnableAndDump_v2(
            subscriber,
            kind.into(),
            config,
        ))?;
        Ok(())
    })
}

/// Returns the activity kinds enabled for `subscriber`.
///
/// # Errors
///
/// Returns an error if the subscriber handle is absent, native enumeration rejects the subscriber or fails, or a reported count cannot be represented as `usize`.
pub fn enabled_kinds(subscriber: &Subscriber) -> Result<Vec<ActivityKind>> {
    let mut buffer_size = 0u32;
    let mut enabled_count = 0u32;
    let subscriber = subscriber.as_raw()?;
    // SAFETY: `cuptiActivityGetEnabledKinds` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuptiActivityGetEnabledKinds(
            subscriber,
            ptr::null_mut(),
            &raw mut buffer_size,
            &raw mut enabled_count,
        ))?;
    }

    if enabled_count == 0 {
        return Ok(Vec::new());
    }

    let len = to_usize(enabled_count, "enabled_activity_kinds_count")?;
    let mut kinds = vec![sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INVALID; len];
    let mut buffer_size = enabled_count;
    let mut enabled_count = 0u32;
    // SAFETY: `cuptiActivityGetEnabledKinds` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuptiActivityGetEnabledKinds(
            subscriber,
            kinds.as_mut_ptr(),
            &raw mut buffer_size,
            &raw mut enabled_count,
        ))?;
    }
    kinds.truncate(to_usize(enabled_count, "enabled_activity_kinds_count")?);
    Ok(kinds.into_iter().map(ActivityKind::from).collect())
}

/// Returns the byte size of CUPTI's activity record struct for an activity kind and CUPTI API version.
///
/// Pass `0` for `version` to query the current runtime version. Nonzero versions must use CUPTI's `xxyyzz` version format and be at least 13.0.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
/// - Returns [`crate::error::Status::NotSupported`] if the requested nonzero CUPTI version is older than 13.0.
pub fn struct_size(kind: ActivityKind, version: u32) -> Result<usize> {
    let mut size = 0;
    // SAFETY: `cuptiActivityGetStructSize` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuptiActivityGetStructSize(
            kind.into(),
            version,
            &raw mut size
        ))?;
    }
    to_usize(size, "activity_struct_size")
}

/// Configures Unified Memory counters before enabling [`crate::types::ActivityKind::UnifiedMemoryCounter`].
///
/// Call this after CUDA driver initialization and before collecting Unified Memory counter activity.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if one of the Unified Memory counter configurations is invalid.
/// - Returns [`crate::error::Status::UnifiedMemoryProfilingNotSupported`] if the platform does not support Unified Memory counters.
/// - Returns [`crate::error::Status::UnifiedMemoryProfilingNotSupportedOnDevice`] if the device does not support Unified Memory counters.
/// - Returns [`crate::error::Status::UnifiedMemoryProfilingNotSupportedOnNonPeerToPeerDevices`] if the multi-GPU topology does not support Unified Memory counters.
pub fn configure_unified_memory_counter(
    config: &mut [ActivityUnifiedMemoryCounterConfig],
) -> Result<()> {
    let count = to_u32(config.len(), "unified_memory_counter_config_count")?;
    let mut config = config
        .iter()
        .copied()
        .map(ActivityUnifiedMemoryCounterConfig::to_raw)
        .collect::<Vec<_>>();
    // SAFETY: `cuptiActivityConfigureUnifiedMemoryCounter` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuptiActivityConfigureUnifiedMemoryCounter(
            config.as_mut_ptr(),
            count,
        ))?;
    }
    Ok(())
}

/// Controls whether synchronization activity records are emitted for all CUDA event and stream queries.
///
/// When disabled, CUPTI may skip records for queries whose CUDA event or stream operation has not completed.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn enable_all_sync_records(enable: bool) -> Result<()> {
    // SAFETY: `cuptiActivityEnableAllSyncRecords` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableAllSyncRecords(u8::from(enable)))?;
    }
    Ok(())
}

/// Controls whether memory allocation activity records include the source shared library path.
///
/// This requires [`crate::types::ActivityKind::Memory2`] and adds runtime overhead.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn enable_allocation_source(enable: bool) -> Result<()> {
    // SAFETY: `cuptiActivityEnableAllocationSource` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableAllocationSource(u8::from(enable)))?;
    }
    Ok(())
}

/// Controls whether CUDA event activity records include device timestamps.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn enable_cuda_event_device_timestamulti_process_service(enable: bool) -> Result<()> {
    // SAFETY: `cuptiActivityEnableCudaEventDeviceTimestamps` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableCudaEventDeviceTimestamps(u8::from(
            enable
        )))?;
    }
    Ok(())
}

/// Controls whether graph activity collection includes device graph trace records.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn enable_device_graph(enable: bool) -> Result<()> {
    // SAFETY: `cuptiActivityEnableDeviceGraph` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableDeviceGraph(u8::from(enable)))?;
    }
    Ok(())
}

/// Enables or disables tracing for a specific CUDA driver callback ID.
///
/// The broad [`crate::types::ActivityKind::Driver`] setting can override this finer-grained setting.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn enable_driver_api(callback_id: CallbackId, enable: bool) -> Result<()> {
    // SAFETY: `cuptiActivityEnableDriverApi` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableDriverApi(
            callback_id.as_raw(),
            u8::from(enable)
        ))?;
    }
    Ok(())
}

/// Enables or disables tracing for a specific CUDA runtime callback ID.
///
/// The broad [`crate::types::ActivityKind::Runtime`] setting can override this finer-grained setting.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn enable_runtime_api(callback_id: CallbackId, enable: bool) -> Result<()> {
    // SAFETY: `cuptiActivityEnableRuntimeApi` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableRuntimeApi(
            callback_id.as_raw(),
            u8::from(enable)
        ))?;
    }
    Ok(())
}

/// Enables hardware-event-system tracing for the profiling session.
///
/// Call this before creating CUDA contexts when possible. Once enabled, hardware tracing persists for the session.
///
/// # Errors
///
/// - Returns [`crate::error::Status::CmpDeviceNotSupported`], [`crate::error::Status::ConfidentialComputingNotSupported`], [`crate::error::Status::MigDeviceNotSupported`], [`crate::error::Status::SliDeviceNotSupported`], [`crate::error::Status::VirtualizedDeviceNotSupported`], [`crate::error::Status::WslDeviceNotSupported`], or [`crate::error::Status::HesTraceNotSupportedOnMultiProcessService`] when hardware tracing is unavailable for the current environment.
/// - Returns [`crate::error::Status::NotSupported`] if hardware tracing cannot be enabled on this platform.
pub fn enable_hw_trace(enable: bool) -> Result<()> {
    // SAFETY: `cuptiActivityEnableHWTrace` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableHWTrace(u8::from(enable)))?;
    }
    Ok(())
}

/// Controls whether kernel activity records include latency timestamps.
///
/// This is not supported when hardware tracing is enabled.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn enable_latency_timestamulti_process_service(enable: bool) -> Result<()> {
    // SAFETY: `cuptiActivityEnableLatencyTimestamps` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableLatencyTimestamps(u8::from(enable)))?;
    }
    Ok(())
}

/// Controls whether kernel activity records include launch attributes.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn enable_launch_attributes(enable: bool) -> Result<()> {
    // SAFETY: `cuptiActivityEnableLaunchAttributes` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableLaunchAttributes(u8::from(enable)))?;
    }
    Ok(())
}

/// Flushes activity buffers for a target and delivers completed records to registered callbacks.
///
/// Use [`crate::types::ActivityFlushFlag::Forced`] before ending a profiling session to request delivery of buffers that may contain incomplete records.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidOperation`] if activity buffer callbacks have not been registered.
/// - Returns [`crate::error::Status::UnknownError`] if CUPTI reports an internal error.
pub fn flush(target: ActivityFlushTarget<'_>, flag: ActivityFlushFlag) -> Result<()> {
    let (context, stream_id) = target.context_and_stream_id()?;
    // SAFETY: `cuptiActivityFlush` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityFlush(context, stream_id, flag.into()))?;
    }
    Ok(())
}

/// Flushes all activity buffers and delivers completed records to registered callbacks.
///
/// Use [`crate::types::ActivityFlushFlag::Forced`] before ending a profiling session to request delivery of buffers that may contain incomplete records.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidOperation`] if activity buffer callbacks have not been registered.
/// - Returns [`crate::error::Status::UnknownError`] if CUPTI reports an internal error.
pub fn flush_all(flag: ActivityFlushFlag) -> Result<()> {
    // SAFETY: `cuptiActivityFlushAll` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityFlushAll(flag.into()))?;
    }
    Ok(())
}

/// Sets CUPTI's periodic activity flush interval.
///
/// Passing `0` disables periodic flushing and restores CUPTI's default worker-thread heuristics. Periodic flushes deliver only full buffers whose records are complete.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn flush_period(time: u32) -> Result<()> {
    // SAFETY: `cuptiActivityFlushPeriod` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityFlushPeriod(time))?;
    }
    Ok(())
}

fn read_attribute(attr: ActivityAttribute, value_size: *mut u64, value: *mut ()) -> Result<()> {
    // SAFETY: `cuptiActivityGetAttribute` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityGetAttribute(
            attr.into(),
            value_size,
            value.cast(),
        ))?;
    }
    Ok(())
}

fn read_attribute_v2(
    subscriber: &Subscriber,
    attr: ActivityAttribute,
    value_size: *mut u64,
    value: *mut (),
) -> Result<()> {
    let subscriber = subscriber.as_raw()?;
    // SAFETY: `cuptiActivityGetAttribute_v2` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivityGetAttribute_v2(
            subscriber,
            attr.into(),
            value_size,
            value.cast(),
        ))?;
    }
    Ok(())
}

/// Reads a process-wide activity attribute into its typed value.
///
/// # Errors
///
/// Returns an error if the typed accessor does not support the attribute, the native result has an unexpected byte size or enum value, a checked size conversion fails, or CUPTI rejects or cannot read the attribute.
pub fn global_attribute_value(attr: ActivityAttribute) -> Result<ActivityAttributeValue> {
    attribute_value_with(read_attribute, attr)
}

/// Reads an activity attribute for `subscriber` into its typed value.
///
/// # Errors
///
/// Returns an error if the typed accessor does not support the attribute, the native result has an unexpected byte size or enum value, a checked size conversion fails, or CUPTI rejects or cannot read the attribute. Also returns an error if the subscriber handle is absent.
pub fn attribute_value(
    subscriber: &Subscriber,
    attr: ActivityAttribute,
) -> Result<ActivityAttributeValue> {
    attribute_value_with(
        |attr, value_size, value| read_attribute_v2(subscriber, attr, value_size, value),
        attr,
    )
}

fn attribute_value_with(
    mut read: impl FnMut(ActivityAttribute, *mut u64, *mut ()) -> Result<()>,
    attr: ActivityAttribute,
) -> Result<ActivityAttributeValue> {
    #[expect(
        deprecated,
        reason = "wraps deprecated upstream API values that remain part of the supported surface"
    )]
    match attr {
        ActivityAttribute::DeviceBufferSize
        | ActivityAttribute::DeviceBufferSizeCudaDynamicParallelism
        | ActivityAttribute::ProfilingSemaphorePoolSize
        | ActivityAttribute::DeviceBufferSizeDeviceGraphs => Ok(ActivityAttributeValue::Bytes(
            read_attribute_value::<usize>(&mut read, attr)?,
        )),
        ActivityAttribute::DeviceBufferPoolLimit
        | ActivityAttribute::ProfilingSemaphorePoolLimit
        | ActivityAttribute::DeviceBufferPreAllocateValue
        | ActivityAttribute::ProfilingSemaphorePreAllocateValue => Ok(
            ActivityAttributeValue::Count(read_attribute_value::<usize>(&mut read, attr)?),
        ),
        ActivityAttribute::ZeroedOutActivityBuffer
        | ActivityAttribute::MemAllocationTypeHostPinned
        | ActivityAttribute::PerThreadActivityBuffer
        | ActivityAttribute::UserDefinedRecords
        | ActivityAttribute::EnableHes
        | ActivityAttribute::EnableAllocationSourceTracking
        | ActivityAttribute::EnableKernelLatencyTimestamps
        | ActivityAttribute::EnableAllSyncRecords
        | ActivityAttribute::EnableCudaEventDeviceTimestamps
        | ActivityAttribute::EnableKernelLaunchAttributes
        | ActivityAttribute::EnableDeviceGraphTrace
        | ActivityAttribute::EnableMultiSubscriberGraphLevelTrace
        | ActivityAttribute::CigMode => Ok(ActivityAttributeValue::Enabled(
            read_attribute_value::<u8>(&mut read, attr)? != 0,
        )),
        ActivityAttribute::MultipleSubscriberState => {
            Ok(ActivityAttributeValue::MultipleSubscriberState(
                MultipleSubscriberState::from(read_attribute_value::<u8>(&mut read, attr)?),
            ))
        }
        ActivityAttribute::ThreadIdType => {
            let raw = read_attribute_value::<u32>(&mut read, attr)?;
            let value =
                ActivityThreadIdType::try_from(raw).map_err(|_| Error::UnknownEnumValue {
                    name: "activity thread id type".into(),
                    value: raw,
                })?;
            Ok(ActivityAttributeValue::ThreadIdType(value))
        }
        ActivityAttribute::TimestampCallback | ActivityAttribute::DeviceBufferForceInt => {
            Err(Error::InvalidAttribute {
                name: format!("{attr:?}").into(),
            })
        }
    }
}

fn read_attribute_value<T: Copy + Default>(
    read: &mut impl FnMut(ActivityAttribute, *mut u64, *mut ()) -> Result<()>,
    attr: ActivityAttribute,
) -> Result<T> {
    // Only plain integers are read here, so CUPTI cannot write an invalid value.
    let mut value = T::default();
    let mut value_size = to_u64(size_of::<T>(), "activity_attribute_size")?;
    read(attr, &raw mut value_size, (&raw mut value).cast())?;
    let value_size = to_usize(value_size, "activity_attribute_size")?;
    if value_size != size_of::<T>() {
        return Err(Error::LengthMismatch {
            name: "activity_attribute_size".into(),
            expected: size_of::<T>(),
            actual: value_size,
        });
    }
    Ok(value)
}

/// Returns whether a CUPTI tracing session is currently active in the process.
///
/// This query does not initialize CUPTI on its own.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
/// - Returns [`crate::error::Status::UnknownError`] if CUPTI reports an internal error.
pub fn is_tracing_session_running() -> Result<bool> {
    let mut running = 0u8;
    // SAFETY: `cuptiIsTracingSessionRunning` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuptiIsTracingSessionRunning(&raw mut running))?;
    }
    Ok(running != 0)
}

/// Pops the current external correlation ID for an external API kind on the calling thread.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if the external correlation kind is invalid.
/// - Returns [`crate::error::Status::QueueEmpty`] if no external correlation ID is active for the kind.
pub fn pop_external_correlation_id(kind: ExternalCorrelationKind) -> Result<ExternalCorrelationId> {
    let mut last_id = 0;
    // SAFETY: `cuptiActivityPopExternalCorrelationId` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuptiActivityPopExternalCorrelationId(
            kind.into(),
            &raw mut last_id,
        ))?;
    }
    Ok(ExternalCorrelationId::from(last_id))
}

/// Pushes an external correlation ID for the calling thread.
///
/// When external-correlation activity collection is enabled, CUPTI emits correlation records for activity generated inside the region.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if the external correlation kind is invalid.
pub fn push_external_correlation_id(
    kind: ExternalCorrelationKind,
    id: ExternalCorrelationId,
) -> Result<()> {
    // SAFETY: `cuptiActivityPushExternalCorrelationId` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::cuptiActivityPushExternalCorrelationId(
            kind.into(),
            id.as_raw()
        ))?;
    }
    Ok(())
}

/// Registers a process-wide Rust callback receiving completed activity buffers.
///
/// Replaces the current Rust callback owner. Keep the returned registration alive while collecting; dropping it clears its callbacks if it still owns the registration. Buffer bytes are borrowed only during `completed`. Callback panics are caught by the native trampoline.
///
/// # Errors
///
/// Returns an error if the activity callback-state mutex is poisoned or CUPTI cannot register the native callbacks.
pub fn global_register_callbacks<F>(
    config: ActivityBufferCallbackConfig,
    completed: F,
) -> Result<ActivityCallbackRegistration>
where
    F: for<'a> FnMut(ActivityBuffer<'a>) + Send + 'static,
{
    let owner_id = NEXT_ACTIVITY_CALLBACK_OWNER_ID.fetch_add(1, Ordering::Relaxed);
    register_callbacks_with_owner(Some(owner_id), config, completed)?;
    Ok(ActivityCallbackRegistration { owner_id })
}

fn register_callbacks_with_owner<F>(
    owner_id: Option<u64>,
    config: ActivityBufferCallbackConfig,
    completed: F,
) -> Result<()>
where
    F: for<'a> FnMut(ActivityBuffer<'a>) + Send + 'static,
{
    *ACTIVITY_BUFFER_CALLBACKS
        .lock()
        .map_err(|_| Error::LockPoisoned {
            name: "activity buffer callbacks".into(),
        })? = Some(ActivityBufferCallbackState {
        owner_id,
        buffer_size: config.buffer_size,
        max_num_records: config.max_num_records,
        completed: Box::new(completed),
    });
    // SAFETY: `cuptiActivityRegisterCallbacks` receives only plain values.
    let result = unsafe {
        try_ffi!(sys::cuptiActivityRegisterCallbacks(
            Some(activity_buffer_requested_trampoline),
            Some(activity_buffer_completed_trampoline),
        ))
    };
    if let Err(error) = result {
        match owner_id {
            Some(owner_id) => clear_activity_callbacks_if_owner(owner_id),
            None => {
                *ACTIVITY_BUFFER_CALLBACKS
                    .lock()
                    .map_err(|_| Error::LockPoisoned {
                        name: "activity buffer callbacks".into(),
                    })? = None;
            }
        }
        return Err(error);
    }
    Ok(())
}

/// Registers subscriber-scoped activity buffer callbacks.
///
/// Register callbacks before enabling activity kinds if you want CUPTI to deliver activity records. Callback-based collection cannot be mixed with CUPTI's blocking enqueue/dequeue activity APIs.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects one of the activity buffer callbacks.
pub fn register_callbacks<F>(
    subscriber: &Subscriber,
    config: ActivityBufferCallbackConfig,
    completed: F,
) -> Result<ActivityCallbackRegistration>
where
    F: for<'a> FnMut(ActivityBuffer<'a>) + Send + 'static,
{
    let owner_id = NEXT_ACTIVITY_CALLBACK_OWNER_ID.fetch_add(1, Ordering::Relaxed);
    *ACTIVITY_BUFFER_CALLBACKS
        .lock()
        .map_err(|_| Error::LockPoisoned {
            name: "activity buffer callbacks".into(),
        })? = Some(ActivityBufferCallbackState {
        owner_id: Some(owner_id),
        buffer_size: config.buffer_size,
        max_num_records: config.max_num_records,
        completed: Box::new(completed),
    });
    let subscriber = subscriber.as_raw()?;
    // SAFETY: `cuptiActivityRegisterCallbacks_v2` receives only plain values.
    let result = unsafe {
        try_ffi!(sys::cuptiActivityRegisterCallbacks_v2(
            subscriber,
            Some(activity_buffer_requested_trampoline_v2),
            Some(activity_buffer_completed_trampoline_v2),
        ))
    };
    if let Err(error) = result {
        clear_activity_callbacks_if_owner(owner_id);
        return Err(error);
    }
    Ok(ActivityCallbackRegistration { owner_id })
}

fn clear_activity_callbacks_if_owner(owner_id: u64) {
    let Ok(mut callbacks) = ACTIVITY_BUFFER_CALLBACKS.lock() else {
        return;
    };
    if callbacks
        .as_ref()
        .is_some_and(|callbacks| callbacks.owner_id == Some(owner_id))
    {
        *callbacks = None;
    }
}

unsafe extern "C" fn activity_buffer_requested_trampoline(
    buffer: *mut *mut u8,
    size: *mut u64,
    max_num_records: *mut u64,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        request_activity_buffer(buffer, size, max_num_records);
    }));
}

unsafe extern "C" fn activity_buffer_completed_trampoline(
    _context: sys::CUcontext,
    _stream_id: u32,
    buffer: *mut u8,
    size: u64,
    valid_size: u64,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        complete_activity_buffer(buffer, size, valid_size);
    }));
}

unsafe extern "C" fn activity_buffer_requested_trampoline_v2(
    buffer: *mut *mut u8,
    size: *mut u64,
    max_num_records: *mut u64,
    _request_info: *mut sys::CUpti_BufferCallbackRequestInfo,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        request_activity_buffer(buffer, size, max_num_records);
    }));
}

unsafe extern "C" fn activity_buffer_completed_trampoline_v2(
    buffer: *mut u8,
    size: u64,
    valid_size: u64,
    _complete_info: *mut sys::CUpti_BufferCallbackCompleteInfo,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        complete_activity_buffer(buffer, size, valid_size);
    }));
}

fn request_activity_buffer(buffer: *mut *mut u8, size: *mut u64, max_num_records: *mut u64) {
    if buffer.is_null() || size.is_null() || max_num_records.is_null() {
        return;
    }

    let Some((buffer_size, max_records)) = ({
        let Ok(callbacks) = ACTIVITY_BUFFER_CALLBACKS.lock() else {
            return;
        };
        callbacks
            .as_ref()
            .map(|callbacks| (callbacks.buffer_size, callbacks.max_num_records))
    }) else {
        // SAFETY: CUPTI passes writable output locations to the buffer-request callback.
        unsafe {
            *buffer = ptr::null_mut();
            *size = 0;
            *max_num_records = 0;
        }
        return;
    };

    if buffer_size == 0 {
        // SAFETY: CUPTI passes writable output locations to the buffer-request callback.
        unsafe {
            *buffer = ptr::null_mut();
            *size = 0;
            *max_num_records = 0;
        }
        return;
    }

    let Ok(layout) = activity_buffer_layout(buffer_size) else {
        // SAFETY: CUPTI passes writable output locations to the buffer-request callback.
        unsafe {
            *buffer = ptr::null_mut();
            *size = 0;
            *max_num_records = 0;
        }
        return;
    };
    // SAFETY: `activity_buffer_layout` rejects zero sizes, so `layout` is non-zero-sized.
    let allocation_ptr = unsafe { alloc_zeroed(layout) };
    if allocation_ptr.is_null() {
        handle_alloc_error(layout);
    }

    // SAFETY: CUPTI passes writable output locations to the buffer-request callback.
    unsafe {
        *buffer = allocation_ptr;
        *size = buffer_size as u64;
        *max_num_records = max_records as u64;
    }
}

fn complete_activity_buffer(buffer: *mut u8, size: u64, valid_size: u64) {
    if buffer.is_null() {
        return;
    }

    // Reclaim the buffer allocation after the callback has copied or consumed
    // the valid bytes reported by CUPTI.
    let allocation = ActivityBufferAllocation { buffer, size };
    let Ok(size) = usize::try_from(size) else {
        return;
    };
    let Ok(valid_size) = usize::try_from(valid_size) else {
        return;
    };
    if valid_size > size {
        return;
    }

    // SAFETY: CUPTI returns the `size`-byte buffer allocated by the request callback, and `valid_size <= size`.
    let bytes = unsafe { slice::from_raw_parts(buffer.cast_const(), valid_size) };

    let Some(mut callbacks) = ACTIVITY_BUFFER_CALLBACKS
        .lock()
        .ok()
        .and_then(|mut callbacks| callbacks.take())
    else {
        return;
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        (callbacks.completed)(ActivityBuffer { bytes });
    }));

    if let Ok(mut state) = ACTIVITY_BUFFER_CALLBACKS.lock() {
        *state = Some(callbacks);
    }

    drop(allocation);

    if let Err(payload) = result {
        resume_unwind(payload);
    }
}

struct ActivityBufferAllocation {
    buffer: *mut u8,
    size: u64,
}

impl Drop for ActivityBufferAllocation {
    fn drop(&mut self) {
        if self.buffer.is_null() {
            return;
        }
        let Ok(size) = usize::try_from(self.size) else {
            return;
        };
        if size == 0 {
            return;
        }
        let Ok(layout) = activity_buffer_layout(size) else {
            return;
        };
        // SAFETY: `self.buffer` came from `alloc_zeroed` with the layout rebuilt here from the same size.
        unsafe {
            dealloc(self.buffer, layout);
        }
    }
}

fn activity_buffer_layout(size: usize) -> StdResult<Layout, LayoutError> {
    Layout::from_size_align(size, ACTIVITY_BUFFER_ALIGNMENT)
}

/// Registers a process-wide timestamp callback for activity records.
///
/// Register the callback before enabling activity kinds. Changing it during a profiling session can make earlier and later records use different timestamp sources.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the timestamp callback.
pub fn register_timestamp_callback<F>(callback: F) -> Result<TimestampCallbackRegistration>
where
    F: FnMut() -> u64 + Send + 'static,
{
    let owner_id = NEXT_ACTIVITY_CALLBACK_OWNER_ID.fetch_add(1, Ordering::Relaxed);
    *TIMESTAMP_CALLBACK.lock().map_err(|_| Error::LockPoisoned {
        name: "timestamp callback".into(),
    })? = Some(TimestampCallbackState {
        owner_id,
        callback: Box::new(callback),
    });
    // SAFETY: `cuptiActivityRegisterTimestampCallback` receives only plain values.
    let result = unsafe {
        try_ffi!(sys::cuptiActivityRegisterTimestampCallback(Some(
            timestamp_callback_trampoline
        )))
    };
    if let Err(error) = result {
        let mut callback = TIMESTAMP_CALLBACK.lock().map_err(|_| Error::LockPoisoned {
            name: "timestamp callback".into(),
        })?;
        if callback
            .as_ref()
            .is_some_and(|callback| callback.owner_id == owner_id)
        {
            *callback = None;
        }
        drop(callback);
        return Err(error);
    }
    Ok(TimestampCallbackRegistration { owner_id })
}

unsafe extern "C" fn timestamp_callback_trampoline() -> u64 {
    catch_unwind(AssertUnwindSafe(|| {
        TIMESTAMP_CALLBACK
            .lock()
            .ok()
            .and_then(|mut callback| callback.as_mut().map(|state| (state.callback)()))
            .unwrap_or(0)
    }))
    .unwrap_or(0)
}

fn set_attribute(attr: ActivityAttribute, value_size: *mut u64, value: *mut ()) -> Result<()> {
    // SAFETY: `cuptiActivitySetAttribute` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivitySetAttribute(
            attr.into(),
            value_size,
            value.cast(),
        ))?;
    }
    Ok(())
}

fn set_attribute_v2(
    subscriber: &Subscriber,
    attr: ActivityAttribute,
    value_size: *mut u64,
    value: *mut (),
) -> Result<()> {
    let subscriber = subscriber.as_raw()?;
    // SAFETY: `cuptiActivitySetAttribute_v2` receives only plain values.
    unsafe {
        try_ffi!(sys::cuptiActivitySetAttribute_v2(
            subscriber,
            attr.into(),
            value_size,
            value.cast(),
        ))?;
    }
    Ok(())
}

/// Sets a process-wide activity attribute using a typed setting.
///
/// # Errors
///
/// Returns an error if the setting variant does not match a supported attribute, a checked byte-size conversion fails, or CUPTI rejects the attribute or its value.
pub fn set_global_attribute_value(
    attr: ActivityAttribute,
    value: ActivityAttributeSetting,
) -> Result<()> {
    set_attribute_value_with(set_attribute, attr, value)
}

/// Sets an activity attribute for `subscriber` using a typed setting.
///
/// # Errors
///
/// Returns an error if the setting variant does not match a supported attribute, a checked byte-size conversion fails, or CUPTI rejects the attribute or its value. Also returns an error if the subscriber handle is absent.
pub fn set_attribute_value(
    subscriber: &Subscriber,
    attr: ActivityAttribute,
    value: ActivityAttributeSetting,
) -> Result<()> {
    set_attribute_value_with(
        |attr, value_size, value| set_attribute_v2(subscriber, attr, value_size, value),
        attr,
        value,
    )
}

fn set_attribute_value_with(
    mut write: impl FnMut(ActivityAttribute, *mut u64, *mut ()) -> Result<()>,
    attr: ActivityAttribute,
    value: ActivityAttributeSetting,
) -> Result<()> {
    #[expect(
        deprecated,
        reason = "wraps deprecated upstream API values that remain part of the supported surface"
    )]
    match (attr, value) {
        (
            ActivityAttribute::DeviceBufferSize
            | ActivityAttribute::DeviceBufferSizeCudaDynamicParallelism
            | ActivityAttribute::ProfilingSemaphorePoolSize
            | ActivityAttribute::DeviceBufferSizeDeviceGraphs,
            ActivityAttributeSetting::Bytes(mut value),
        )
        | (
            ActivityAttribute::DeviceBufferPoolLimit
            | ActivityAttribute::ProfilingSemaphorePoolLimit
            | ActivityAttribute::DeviceBufferPreAllocateValue
            | ActivityAttribute::ProfilingSemaphorePreAllocateValue,
            ActivityAttributeSetting::Count(mut value),
        ) => {
            let mut value_size = to_u64(size_of::<usize>(), "activity_attribute_size")?;
            write(attr, &raw mut value_size, (&raw mut value).cast())
        }
        (
            ActivityAttribute::ZeroedOutActivityBuffer
            | ActivityAttribute::MemAllocationTypeHostPinned
            | ActivityAttribute::PerThreadActivityBuffer
            | ActivityAttribute::UserDefinedRecords
            | ActivityAttribute::EnableHes
            | ActivityAttribute::EnableAllocationSourceTracking
            | ActivityAttribute::EnableKernelLatencyTimestamps
            | ActivityAttribute::EnableAllSyncRecords
            | ActivityAttribute::EnableCudaEventDeviceTimestamps
            | ActivityAttribute::EnableKernelLaunchAttributes
            | ActivityAttribute::EnableDeviceGraphTrace
            | ActivityAttribute::EnableMultiSubscriberGraphLevelTrace
            | ActivityAttribute::CigMode,
            ActivityAttributeSetting::Enabled(value),
        ) => {
            let mut value = u8::from(value);
            let mut value_size = to_u64(size_of::<u8>(), "activity_attribute_size")?;
            write(attr, &raw mut value_size, (&raw mut value).cast())
        }
        (ActivityAttribute::ThreadIdType, ActivityAttributeSetting::ThreadIdType(value)) => {
            let mut value = sys::CUpti_ActivityThreadIdType::from(value);
            let mut value_size = to_u64(
                size_of::<sys::CUpti_ActivityThreadIdType>(),
                "activity_attribute_size",
            )?;
            write(attr, &raw mut value_size, (&raw mut value).cast())
        }
        _ => Err(Error::InvalidAttribute {
            name: format!("{attr:?}").into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        ffi::CString,
        sync::{
            Mutex,
            atomic::{AtomicBool, Ordering},
        },
    };

    use super::*;

    static ACTIVITY_BUFFER_CALLBACK_TEST_LOCK: Mutex<()> = Mutex::new(());

    struct ActivityBufferCallbackReset;

    impl Drop for ActivityBufferCallbackReset {
        fn drop(&mut self) {
            *ACTIVITY_BUFFER_CALLBACKS
                .lock()
                .expect("activity buffer callbacks poisoned") = None;
        }
    }

    #[test]
    fn test_activity_buffer_request_returns_eight_byte_aligned_buffer() {
        let _lock = ACTIVITY_BUFFER_CALLBACK_TEST_LOCK
            .lock()
            .expect("activity buffer callback test lock poisoned");
        let _reset = ActivityBufferCallbackReset;
        *ACTIVITY_BUFFER_CALLBACKS
            .lock()
            .expect("activity buffer callbacks poisoned") = Some(ActivityBufferCallbackState {
            owner_id: None,
            buffer_size: 17,
            max_num_records: 3,
            completed: Box::new(|_| {}),
        });

        let mut buffer = ptr::null_mut();
        let mut size = 0;
        let mut max_num_records = 0;

        request_activity_buffer(&mut buffer, &mut size, &mut max_num_records);

        assert!(!buffer.is_null());
        assert_eq!(buffer.addr() % ACTIVITY_BUFFER_ALIGNMENT, 0);
        assert_eq!(size, 17);
        assert_eq!(max_num_records, 3);

        drop(ActivityBufferAllocation { buffer, size });
    }

    #[test]
    fn test_activity_buffer_rejects_valid_length_larger_than_allocation() {
        let _lock = ACTIVITY_BUFFER_CALLBACK_TEST_LOCK
            .lock()
            .expect("activity buffer callback test lock poisoned");
        let _reset = ActivityBufferCallbackReset;
        *ACTIVITY_BUFFER_CALLBACKS
            .lock()
            .expect("activity buffer callbacks poisoned") = Some(ActivityBufferCallbackState {
            owner_id: None,
            buffer_size: 8,
            max_num_records: 0,
            completed: Box::new(|_| panic!("invalid activity buffer reached user callback")),
        });
        let mut buffer = ptr::null_mut();
        let mut size = 0;
        let mut max_num_records = 0;
        request_activity_buffer(&mut buffer, &mut size, &mut max_num_records);
        assert!(!buffer.is_null());
        complete_activity_buffer(buffer, size, size + 1);
        assert!(ACTIVITY_BUFFER_CALLBACKS.lock().unwrap().is_some());
    }

    #[test]
    fn test_activity_buffer_complete_restores_state_and_frees_after_callback_panic() {
        let _lock = ACTIVITY_BUFFER_CALLBACK_TEST_LOCK
            .lock()
            .expect("activity buffer callback test lock poisoned");
        let _reset = ActivityBufferCallbackReset;
        static CALLED: AtomicBool = AtomicBool::new(false);
        CALLED.store(false, Ordering::SeqCst);
        *ACTIVITY_BUFFER_CALLBACKS
            .lock()
            .expect("activity buffer callbacks poisoned") = Some(ActivityBufferCallbackState {
            owner_id: None,
            buffer_size: 32,
            max_num_records: 0,
            completed: Box::new(|buffer| {
                assert_eq!(buffer.len(), 5);
                CALLED.store(true, Ordering::SeqCst);
                panic!("activity callback panic");
            }),
        });

        let mut buffer = ptr::null_mut();
        let mut size = 0;
        let mut max_num_records = 0;
        request_activity_buffer(&mut buffer, &mut size, &mut max_num_records);

        let result = catch_unwind(AssertUnwindSafe(|| {
            complete_activity_buffer(buffer, size, 5);
        }));

        assert!(result.is_err());
        assert!(CALLED.load(Ordering::SeqCst));
        assert!(
            ACTIVITY_BUFFER_CALLBACKS
                .lock()
                .expect("activity buffer callbacks poisoned")
                .is_some()
        );
    }

    #[test]
    fn test_activity_collector_copies_completed_buffers() -> Result<()> {
        let _lock = ACTIVITY_BUFFER_CALLBACK_TEST_LOCK
            .lock()
            .expect("activity buffer callback test lock poisoned");
        let _reset = ActivityBufferCallbackReset;
        let collector = ActivityCollector::create(ActivityBufferCallbackConfig::new(32))?;

        let mut buffer = ptr::null_mut();
        let mut size = 0;
        let mut max_num_records = 0;
        request_activity_buffer(&mut buffer, &mut size, &mut max_num_records);
        assert!(!buffer.is_null());
        assert_eq!(size, 32);

        // SAFETY: the request callback allocated `size` writable bytes at `buffer`, checked non-null above.
        unsafe {
            let bytes = slice::from_raw_parts_mut(buffer, size as usize);
            bytes[..4].copy_from_slice(&[1, 2, 3, 4]);
        }

        complete_activity_buffer(buffer, size, 4);

        assert_eq!(collector.len()?, 1);
        let buffers = collector.take_buffers()?;
        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].bytes(), &[1, 2, 3, 4]);
        assert!(collector.is_empty()?);

        Ok(())
    }

    #[test]
    fn test_activity_collector_drop_only_clears_owned_registration() -> Result<()> {
        let _lock = ACTIVITY_BUFFER_CALLBACK_TEST_LOCK
            .lock()
            .expect("activity buffer callback test lock poisoned");
        let _reset = ActivityBufferCallbackReset;
        let collector = ActivityCollector::create(ActivityBufferCallbackConfig::new(32))?;
        let _registration =
            global_register_callbacks(ActivityBufferCallbackConfig::new(64), |_| {})?;

        drop(collector);

        let callbacks = ACTIVITY_BUFFER_CALLBACKS
            .lock()
            .expect("activity buffer callbacks poisoned");
        assert!(callbacks.is_some());
        assert_eq!(callbacks.as_ref().unwrap().buffer_size, 64);

        Ok(())
    }

    #[test]
    fn test_activity_record_decodes_memory_decompress_record() {
        let raw = sys::CUpti_ActivityMemDecompress {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS,
            deviceId: 1,
            contextId: 2,
            streamId: 3,
            channelID: 4,
            channelType: sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_DECOMP,
            correlationId: 5,
            numberOfOperations: 6,
            sourceBytes: 7,
            reserved0: ptr::null_mut(),
            start: 8,
            end: 9,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMemDecompress).cast(),
            _marker: PhantomData,
        };

        assert_eq!(record.kind(), Some(ActivityKind::MemoryDecompress));
        assert_eq!(
            record.memory_decompress(),
            Some(ActivityMemoryDecompress {
                device_id: DeviceId::from(1),
                context_id: ContextId::from(2),
                stream_id: StreamId::from(3u64),
                channel_id: ChannelId::from(4),
                channel_type: ChannelType::Decompress,
                correlation_id: CorrelationId::from(5),
                number_of_operations: 6,
                source_bytes: 7,
                start: 8,
                end: 9,
            })
        );
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::MemoryDecompress(ActivityMemoryDecompress {
                device_id: DeviceId::from(1),
                context_id: ContextId::from(2),
                stream_id: StreamId::from(3u64),
                channel_id: ChannelId::from(4),
                channel_type: ChannelType::Decompress,
                correlation_id: CorrelationId::from(5),
                number_of_operations: 6,
                source_bytes: 7,
                start: 8,
                end: 9,
            })
        );
    }

    #[test]
    fn test_activity_record_rejects_memory_decompress_for_other_kinds() {
        let raw = sys::CUpti_Activity {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INVALID,
        };
        let record = ActivityRecord {
            ptr: &raw,
            _marker: PhantomData,
        };

        assert_eq!(record.memory_decompress(), None);
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::Unsupported {
                raw_kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INVALID as u32
            }
        );
    }

    #[test]
    fn test_activity_record_decodes_kernel_record() {
        let name = CString::new("kernel").unwrap();
        let mut raw = sys::CUpti_ActivityKernel13 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
            deviceId: 1,
            contextId: 2,
            streamId: 3,
            gridX: 4,
            gridY: 5,
            gridZ: 6,
            blockX: 7,
            blockY: 8,
            blockZ: 9,
            staticSharedMemory: 10,
            dynamicSharedMemory: 11,
            localMemoryPerThread: 12,
            correlationId: 13,
            gridId: 14,
            name: name.as_ptr(),
            queued: 15,
            submitted: 16,
            launchType:
                sys::CUpti_ActivityLaunchType::CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_SINGLE_DEVICE
                    as u8,
            isSharedMemoryCarveoutRequested: 1,
            sharedMemoryCarveoutRequested: 50,
            sharedMemoryExecuted: 17,
            graphNodeId: 18,
            shmemLimitConfig: sys::CUpti_FuncShmemLimitConfig::CUPTI_FUNC_SHMEM_LIMIT_OPTIN,
            graphId: 19,
            channelID: 20,
            channelType: sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_COMPUTE,
            clusterX: 21,
            clusterY: 22,
            clusterZ: 23,
            localMemoryTotal_v2: 24,
            maxPotentialClusterSize: 25,
            maxActiveClusters: 26,
            isDeviceLaunched: 1,
            priority: 27,
            ..Default::default()
        };
        raw.partitionedGlobalCacheRequested =
            sys::CUpti_ActivityPartitionedGlobalCacheConfig::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON;
        raw.partitionedGlobalCacheExecuted =
            sys::CUpti_ActivityPartitionedGlobalCacheConfig::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF;
        raw.start = 28;
        raw.end = 29;
        raw.completed = 30;
        raw.registersPerThread = 31;
        raw.sharedMemoryConfig = 32;

        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityKernel13).cast(),
            _marker: PhantomData,
        };

        let kernel = record.kernel().unwrap();
        assert_eq!(record.kind(), Some(ActivityKind::ConcurrentKernel));
        assert_eq!(kernel.name.unwrap().to_str().unwrap(), "kernel");
        assert_eq!(kernel.device_id, DeviceId::from(1));
        assert_eq!(kernel.context_id, ContextId::from(2));
        assert_eq!(kernel.stream_id, StreamId::from(3u64));
        assert_eq!(kernel.grid, Dim3 { x: 4, y: 5, z: 6 });
        assert_eq!(kernel.block, Dim3 { x: 7, y: 8, z: 9 });
        assert_eq!(
            kernel.cluster,
            Dim3 {
                x: 21,
                y: 22,
                z: 23
            }
        );
        assert_eq!(kernel.correlation_id, CorrelationId::from(13));
        assert_eq!(kernel.graph_node_id, GraphNodeId::from(18));
        assert_eq!(kernel.graph_id, GraphId::from(19));
        assert_eq!(kernel.channel_id, ChannelId::from(20));
        assert_eq!(kernel.channel_type, ChannelType::Compute);
        assert_eq!(
            kernel.launch_type,
            ActivityLaunchType::CooperativeSingleDevice
        );
        assert_eq!(
            kernel.shared_memory_limit_config,
            FunctionSharedMemoryLimitConfig::OptIn
        );
        assert_eq!(kernel.shared_memory_carveout_requested, Some(50));
        assert_eq!(kernel.local_memory_total, 24);
        assert!(kernel.is_device_launched);
        assert!(matches!(record.decode(), ActivityRecordPayload::Kernel(_)));
    }

    #[test]
    fn test_activity_record_decodes_legacy_activity_records() {
        let name = CString::new("legacy").unwrap();
        let memory = sys::CUpti_ActivityMemory {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY,
            memoryKind: sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE,
            address: 1,
            bytes: 2,
            start: 3,
            end: 4,
            allocPC: 5,
            freePC: 6,
            processId: 7,
            deviceId: 8,
            contextId: 9,
            pad: 0,
            name: name.as_ptr(),
        };
        let memory_record = ActivityRecord {
            ptr: (&memory as *const sys::CUpti_ActivityMemory).cast(),
            _marker: PhantomData,
        };
        let decoded_memory = memory_record.legacy_memory().unwrap();

        assert_eq!(decoded_memory.memory_kind, ActivityMemoryKind::Device);
        assert_eq!(decoded_memory.address, 1);
        assert_eq!(decoded_memory.device_id, DeviceId::from(8));
        assert_eq!(decoded_memory.name.unwrap().to_str().unwrap(), "legacy");
        assert!(matches!(
            memory_record.decode(),
            ActivityRecordPayload::LegacyMemory(_)
        ));

        let memory_copy = sys::CUpti_ActivityMemcpyPtoP4 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY2,
            copyKind: sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_PTOP as u8,
            srcKind: sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE as u8,
            dstKind: sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_MANAGED as u8,
            flags: 0,
            bytes: 10,
            start: 11,
            end: 12,
            deviceId: 13,
            contextId: 14,
            streamId: 15,
            srcDeviceId: 16,
            srcContextId: 17,
            dstDeviceId: 18,
            dstContextId: 19,
            correlationId: 20,
            reserved0: ptr::null_mut(),
            graphNodeId: 21,
            graphId: 22,
            channelID: 23,
            channelType: sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY,
        };
        let memory_copy_record = ActivityRecord {
            ptr: (&memory_copy as *const sys::CUpti_ActivityMemcpyPtoP4).cast(),
            _marker: PhantomData,
        };
        let decoded_memory_copy = memory_copy_record.peer_memory_copy().unwrap();

        assert_eq!(
            decoded_memory_copy.copy_kind,
            ActivityMemoryCopyKind::PeerToPeer
        );
        assert_eq!(decoded_memory_copy.source_device_id, DeviceId::from(16));
        assert_eq!(
            decoded_memory_copy.destination_context_id,
            ContextId::from(19)
        );
        assert_eq!(
            decoded_memory_copy.channel_type,
            ChannelType::AsynchronousMemoryCopy
        );
        assert_eq!(
            memory_copy_record.decode(),
            ActivityRecordPayload::PeerMemoryCopy(decoded_memory_copy)
        );

        let kernel_name = CString::new("child").unwrap();
        let cdp = sys::CUpti_ActivityCdpKernel {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CDP_KERNEL,
            cacheConfig: sys::CUpti_ActivityCdpKernel__bindgen_ty_1 { both: 0 },
            sharedMemoryConfig: 1,
            registersPerThread: 2,
            start: 3,
            end: 4,
            deviceId: 5,
            contextId: 6,
            streamId: 7,
            gridX: 8,
            gridY: 9,
            gridZ: 10,
            blockX: 11,
            blockY: 12,
            blockZ: 13,
            staticSharedMemory: 14,
            dynamicSharedMemory: 15,
            localMemoryPerThread: 16,
            localMemoryTotal: 17,
            correlationId: 18,
            gridId: 19,
            parentGridId: 20,
            queued: 21,
            submitted: 22,
            completed: 23,
            parentBlockX: 24,
            parentBlockY: 25,
            parentBlockZ: 26,
            pad: 0,
            name: kernel_name.as_ptr(),
        };
        let cdp_record = ActivityRecord {
            ptr: (&cdp as *const sys::CUpti_ActivityCdpKernel).cast(),
            _marker: PhantomData,
        };
        let decoded_cdp = cdp_record.cuda_dynamic_parallelism_kernel().unwrap();

        assert_eq!(decoded_cdp.name.unwrap().to_str().unwrap(), "child");
        assert_eq!(decoded_cdp.grid_id, GridId::from(19));
        assert_eq!(decoded_cdp.parent_grid_id, GridId::from(20));
        assert_eq!(
            decoded_cdp.parent_block,
            Dim3 {
                x: 24,
                y: 25,
                z: 26
            }
        );
        assert!(matches!(
            cdp_record.decode(),
            ActivityRecordPayload::CudaDynamicParallelismKernel(_)
        ));
    }

    #[test]
    fn test_activity_record_decodes_api_record() {
        let raw = sys::CUpti_ActivityAPI {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_RUNTIME,
            cbid: 211,
            start: 1,
            end: 2,
            processId: 3,
            threadId: 4,
            correlationId: 5,
            returnValue: 6,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityAPI).cast(),
            _marker: PhantomData,
        };

        let api = ActivityApi {
            kind: ActivityKind::Runtime,
            callback_id: CallbackId::from(211),
            start: 1,
            end: 2,
            process_id: ProcessId::from(3u64),
            thread_id: ThreadId::from(4),
            correlation_id: CorrelationId::from(5),
            return_value: 6,
        };
        assert_eq!(record.api(), Some(api));
        assert_eq!(record.decode(), ActivityRecordPayload::Api(api));
    }

    #[test]
    fn test_activity_record_decodes_stream_record() {
        let raw = sys::CUpti_ActivityStream {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_STREAM,
            contextId: 1,
            streamId: 2,
            priority: 3,
            flag: sys::CUpti_ActivityStreamFlag::CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING,
            correlationId: 4,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityStream).cast(),
            _marker: PhantomData,
        };
        let stream = ActivityStream {
            context_id: ContextId::from(1),
            stream_id: StreamId::from(2u64),
            priority: 3,
            flag: ActivityStreamFlag::NonBlocking,
            correlation_id: CorrelationId::from(4),
        };

        assert_eq!(record.stream(), Some(stream));
        assert_eq!(record.decode(), ActivityRecordPayload::Stream(stream));
    }

    #[test]
    fn test_activity_record_decodes_synchronization_record() {
        let raw = sys::CUpti_ActivitySynchronization2 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
            type_: sys::CUpti_ActivitySynchronizationType::CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE,
            start: 1,
            end: 2,
            correlationId: 3,
            contextId: 4,
            streamId: 5,
            cudaEventId: 6,
            cudaEventSyncId: 7,
            returnValue: 8,
            pad: 0,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivitySynchronization2).cast(),
            _marker: PhantomData,
        };
        let synchronization = ActivitySynchronization {
            synchronization_type: ActivitySynchronizationType::StreamSynchronize,
            start: 1,
            end: 2,
            correlation_id: CorrelationId::from(3),
            context_id: ContextId::from(4),
            stream_id: StreamId::from(5u64),
            cuda_event_id: CudaEventId::from(6),
            cuda_event_sync_id: CudaEventSyncId::from(7),
            return_value: 8,
        };

        assert_eq!(record.synchronization(), Some(synchronization));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::Synchronization(synchronization)
        );
    }

    #[test]
    fn test_activity_record_decodes_memory_record() {
        let name = CString::new("allocation").unwrap();
        let source = CString::new("libcuda.so").unwrap();
        let raw = sys::CUpti_ActivityMemory4 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY2,
            memoryOperationType: sys::CUpti_ActivityMemoryOperationType::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION,
            memoryKind: sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE,
            correlationId: 1,
            address: 2,
            bytes: 3,
            timestamp: 4,
            PC: 5,
            processId: 6,
            deviceId: 7,
            contextId: 8,
            streamId: 9,
            name: name.as_ptr(),
            isAsync: 1,
            pad1: 0,
            memoryPoolConfig: sys::CUpti_ActivityMemory4__bindgen_ty_1 {
                memoryPoolType: sys::CUpti_ActivityMemoryPoolType::CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL,
                pad2: 0,
                address: 10,
                releaseThreshold: 11,
                pool: sys::CUpti_ActivityMemory4__bindgen_ty_1__bindgen_ty_1 { size: 12 },
                utilizedSize: 13,
            },
            source: source.as_ptr(),
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMemory4).cast(),
            _marker: PhantomData,
        };

        let memory = record.memory().unwrap();
        assert_eq!(
            memory.operation_type,
            ActivityMemoryOperationType::Allocation
        );
        assert_eq!(memory.memory_kind, ActivityMemoryKind::Device);
        assert_eq!(memory.correlation_id, CorrelationId::from(1));
        assert_eq!(memory.address, 2);
        assert_eq!(memory.bytes, 3);
        assert_eq!(memory.timestamp, 4);
        assert_eq!(memory.pc, 5);
        assert_eq!(memory.process_id, ProcessId::from(6u64));
        assert_eq!(memory.device_id, DeviceId::from(7));
        assert_eq!(memory.context_id, ContextId::from(8));
        assert_eq!(memory.stream_id, StreamId::from(9u64));
        assert_eq!(memory.name.unwrap().to_str().unwrap(), "allocation");
        assert!(memory.is_asynchronous);
        assert_eq!(
            memory.memory_pool_config.pool_type,
            ActivityMemoryPoolType::Local
        );
        assert_eq!(memory.memory_pool_config.address, 10);
        assert_eq!(
            memory.memory_pool_config.data,
            ActivityMemoryPoolProperties::Local {
                size: 12,
                release_threshold: 11,
                utilized_size: 13,
            }
        );
        assert_eq!(memory.source.unwrap().to_str().unwrap(), "libcuda.so");
        assert!(matches!(record.decode(), ActivityRecordPayload::Memory(_)));
    }

    #[test]
    fn test_activity_record_decodes_imported_memory_pool_config() {
        let raw = sys::CUpti_ActivityMemory4 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY2,
            memoryOperationType: sys::CUpti_ActivityMemoryOperationType::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION,
            memoryKind: sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE,
            correlationId: 1,
            address: 2,
            bytes: 3,
            timestamp: 4,
            PC: 5,
            processId: 6,
            deviceId: 7,
            contextId: 8,
            streamId: 9,
            name: ptr::null(),
            isAsync: 0,
            pad1: 0,
            memoryPoolConfig: sys::CUpti_ActivityMemory4__bindgen_ty_1 {
                memoryPoolType: sys::CUpti_ActivityMemoryPoolType::CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED,
                pad2: 0,
                address: 10,
                releaseThreshold: 0,
                pool: sys::CUpti_ActivityMemory4__bindgen_ty_1__bindgen_ty_1 {
                    processId: 12,
                },
                utilizedSize: 0,
            },
            source: ptr::null(),
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMemory4).cast(),
            _marker: PhantomData,
        };

        let memory = record.memory().unwrap();
        assert_eq!(
            memory.memory_pool_config.data,
            ActivityMemoryPoolProperties::Imported {
                process_id: ProcessId::from(12u64),
            }
        );
    }

    #[test]
    fn test_activity_record_decodes_memory_pool_record() {
        let raw = sys::CUpti_ActivityMemoryPool3 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY_POOL,
            memoryPoolOperationType: sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED,
            memoryPoolType: sys::CUpti_ActivityMemoryPoolType::CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED,
            correlationId: 1,
            processId: 2,
            deviceId: 3,
            minBytesToKeep: 4,
            address: 5,
            size: 6,
            releaseThreshold: 7,
            timestamp: 8,
            utilizedSize: 9,
            isManagedPool: 1,
            pad2: [0; 7],
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMemoryPool3).cast(),
            _marker: PhantomData,
        };

        let memory_pool = ActivityMemoryPool {
            operation_type: ActivityMemoryPoolOperationType::Trimmed,
            pool_type: ActivityMemoryPoolType::Imported,
            correlation_id: CorrelationId::from(1),
            process_id: ProcessId::from(2u64),
            device_id: DeviceId::from(3),
            min_bytes_to_keep: 4,
            address: 5,
            size: 6,
            release_threshold: 7,
            timestamp: 8,
            utilized_size: 9,
            is_managed_pool: true,
        };
        assert_eq!(record.memory_pool(), Some(memory_pool));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::MemoryPool(memory_pool)
        );
    }

    #[test]
    fn test_activity_record_decodes_graph_trace_record() {
        let raw = sys::CUpti_ActivityGraphTrace2 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_GRAPH_TRACE,
            correlationId: 1,
            start: 2,
            end: 3,
            deviceId: 4,
            graphId: 5,
            contextId: 6,
            streamId: 7,
            reserved: ptr::null_mut(),
            endDeviceId: 8,
            endContextId: 9,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityGraphTrace2).cast(),
            _marker: PhantomData,
        };
        let graph_trace = ActivityGraphTrace {
            correlation_id: CorrelationId::from(1),
            start: 2,
            end: 3,
            device_id: DeviceId::from(4),
            graph_id: GraphId::from(5),
            context_id: ContextId::from(6),
            stream_id: StreamId::from(7u64),
            end_device_id: DeviceId::from(8),
            end_context_id: ContextId::from(9),
        };

        assert_eq!(record.graph_trace(), Some(graph_trace));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::GraphTrace(graph_trace)
        );
    }

    #[test]
    fn test_activity_record_decodes_device_graph_trace_record() {
        let raw = sys::CUpti_ActivityDeviceGraphTrace {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE,
            deviceId: 1,
            start: 2,
            end: 3,
            graphId: 4,
            launcherGraphId: 5,
            deviceLaunchMode: sys::CUpti_DeviceGraphLaunchMode::CUPTI_DEVICE_GRAPH_LAUNCH_MODE_TAIL
                as u32,
            contextId: 6,
            streamId: 7,
            reserved: ptr::null_mut(),
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityDeviceGraphTrace).cast(),
            _marker: PhantomData,
        };
        let device_graph_trace = ActivityDeviceGraphTrace {
            device_id: DeviceId::from(1),
            start: 2,
            end: 3,
            graph_id: GraphId::from(4),
            launcher_graph_id: GraphId::from(5),
            device_launch_mode: DeviceGraphLaunchMode::Tail,
            context_id: ContextId::from(6),
            stream_id: StreamId::from(7u64),
        };

        assert_eq!(record.device_graph_trace(), Some(device_graph_trace));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::DeviceGraphTrace(device_graph_trace)
        );
    }

    #[test]
    fn test_activity_record_decodes_graph_host_node_record() {
        let raw = sys::CUpti_ActivityGraphHostNode {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_GRAPH_HOST_NODE,
            streamId: 1,
            contextId: 2,
            deviceId: 3,
            correlationId: 4,
            graphId: 5,
            graphNodeId: 6,
            processId: 7,
            threadId: 8,
            start: 9,
            end: 10,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityGraphHostNode).cast(),
            _marker: PhantomData,
        };
        let graph_host_node = ActivityGraphHostNode {
            stream_id: StreamId::from(1u64),
            context_id: ContextId::from(2),
            device_id: DeviceId::from(3),
            correlation_id: CorrelationId::from(4),
            graph_id: GraphId::from(5),
            graph_node_id: GraphNodeId::from(6),
            process_id: ProcessId::from(7u64),
            thread_id: ThreadId::from(8),
            start: 9,
            end: 10,
        };

        assert_eq!(record.graph_host_node(), Some(graph_host_node));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::GraphHostNode(graph_host_node)
        );
    }

    #[test]
    fn test_activity_record_decodes_host_launch_record() {
        let raw = sys::CUpti_ActivityHostLaunch {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_HOST_LAUNCH,
            streamId: 1,
            contextId: 2,
            deviceId: 3,
            correlationId: 4,
            processId: 5,
            threadId: 6,
            padding: 0,
            start: 7,
            end: 8,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityHostLaunch).cast(),
            _marker: PhantomData,
        };
        let host_launch = ActivityHostLaunch {
            stream_id: StreamId::from(1u64),
            context_id: ContextId::from(2),
            device_id: DeviceId::from(3),
            correlation_id: CorrelationId::from(4),
            process_id: ProcessId::from(5u64),
            thread_id: ThreadId::from(6),
            start: 7,
            end: 8,
        };

        assert_eq!(record.host_launch(), Some(host_launch));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::HostLaunch(host_launch)
        );
    }

    #[test]
    fn test_activity_record_decodes_compute_engine_context_switch_record() {
        let raw = sys::CUpti_ActivityComputeEngineCtxSwitch {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_COMPUTE_ENGINE_CTX_SWITCH,
            contextId: 1,
            timestamp: 2,
            operationType: sys::CUpti_ComputeEngineCtxSwitchOperationType::CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_END,
            padding: 0,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityComputeEngineCtxSwitch).cast(),
            _marker: PhantomData,
        };
        let context_switch = ActivityComputeEngineContextSwitch {
            context_id: ContextId::from(1),
            timestamp: 2,
            operation_type: ComputeEngineContextSwitchOperationType::End,
        };

        assert_eq!(record.compute_engine_context_switch(), Some(context_switch));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::ComputeEngineContextSwitch(context_switch)
        );
    }

    #[test]
    fn test_activity_record_decodes_green_context_record() {
        let mut logical_tpc_mask = [0u32; 32];
        logical_tpc_mask[0] = 0b10;
        logical_tpc_mask[1] = 0b1000;
        let raw = sys::CUpti_ActivityGreenContext3 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_GREEN_CONTEXT,
            contextId: 1,
            parentContextId: 2,
            deviceId: 3,
            numTpcs: 4,
            numMultiprocessors: 5,
            logicalTpcMaskSize: 2,
            padding: 0,
            logicalTpcMask: logical_tpc_mask,
            ..Default::default()
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityGreenContext3).cast(),
            _marker: PhantomData,
        };
        let green_context = ActivityGreenContext {
            context_id: ContextId::from(1),
            parent_context_id: ContextId::from(2),
            device_id: DeviceId::from(3),
            num_tpcs: 4,
            num_multiprocessors: 5,
            logical_tpc_mask,
            logical_tpc_mask_size: 2,
            workqueue_resource_id: 0,
            workqueue_concurrency_limit: 0,
            workqueue_sharing_scope: 0,
            workqueue_channel_count: 0,
            workqueue_channel_ids: [0; 32],
        };

        assert_eq!(record.green_context(), Some(green_context));
        assert_eq!(
            green_context.logical_tpc_mask_words(),
            &[0b10u32, 0b1000u32]
        );
        assert!(green_context.has_logical_tpc(1));
        assert!(green_context.has_logical_tpc(35));
        assert!(!green_context.has_logical_tpc(2));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::GreenContext(green_context)
        );
    }

    #[test]
    fn test_activity_record_decodes_preemption_function_module_and_shared_records() {
        let preemption = sys::CUpti_ActivityPreemption {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PREEMPTION,
            preemptionKind:
                sys::CUpti_ActivityPreemptionKind::CUPTI_ACTIVITY_PREEMPTION_KIND_RESTORE,
            timestamp: 1,
            gridId: 2,
            blockX: 3,
            blockY: 4,
            blockZ: 5,
            pad: 0,
        };
        let preemption_record = ActivityRecord {
            ptr: (&preemption as *const sys::CUpti_ActivityPreemption).cast(),
            _marker: PhantomData,
        };
        let decoded_preemption = ActivityPreemption {
            preemption_kind: ActivityPreemptionKind::Restore,
            timestamp: 1,
            grid_id: GridId::from(2),
            block: Dim3 { x: 3, y: 4, z: 5 },
        };

        assert_eq!(preemption_record.preemption(), Some(decoded_preemption));
        assert_eq!(
            preemption_record.decode(),
            ActivityRecordPayload::Preemption(decoded_preemption)
        );

        let name = CString::new("kernel").unwrap();
        let function = sys::CUpti_ActivityFunction {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_FUNCTION,
            id: 6,
            contextId: 7,
            moduleId: 8,
            functionIndex: 9,
            pad: 0,
            name: name.as_ptr(),
        };
        let function_record = ActivityRecord {
            ptr: (&function as *const sys::CUpti_ActivityFunction).cast(),
            _marker: PhantomData,
        };
        let decoded_function = function_record.function().unwrap();

        assert_eq!(decoded_function.id, FunctionId::from(6));
        assert_eq!(decoded_function.context_id, ContextId::from(7));
        assert_eq!(decoded_function.module_id, ModuleId::from(8));
        assert_eq!(decoded_function.function_index, 9);
        assert_eq!(decoded_function.name.unwrap().to_str().unwrap(), "kernel");
        assert!(matches!(
            function_record.decode(),
            ActivityRecordPayload::Function(_)
        ));

        let cubin = [0xca, 0xfe, 0xba, 0xbe];
        let module = sys::CUpti_ActivityModule {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MODULE,
            contextId: 10,
            id: 11,
            cubinSize: cubin.len() as u32,
            cubin: cubin.as_ptr().cast(),
        };
        let module_record = ActivityRecord {
            ptr: (&module as *const sys::CUpti_ActivityModule).cast(),
            _marker: PhantomData,
        };
        let decoded_module = module_record.module().unwrap();

        assert_eq!(decoded_module.context_id, ContextId::from(10));
        assert_eq!(decoded_module.id, ModuleId::from(11));
        assert_eq!(decoded_module.cubin, Some(cubin.as_slice()));
        assert!(matches!(
            module_record.decode(),
            ActivityRecordPayload::Module(_)
        ));

        let shared = sys::CUpti_ActivitySharedAccess {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_SHARED_ACCESS,
            flags: sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_CACHED,
            sourceLocatorId: 12,
            correlationId: 13,
            functionId: 14,
            pcOffset: 15,
            threadsExecuted: 16,
            sharedTransactions: 17,
            theoreticalSharedTransactions: 18,
            executed: 19,
            pad: 0,
        };
        let shared_record = ActivityRecord {
            ptr: (&shared as *const sys::CUpti_ActivitySharedAccess).cast(),
            _marker: PhantomData,
        };
        let decoded_shared = ActivitySharedAccess {
            flags: ActivityFlags::GLOBAL_ACCESS_KIND_CACHED,
            source_locator_id: SourceLocatorId::from(12),
            correlation_id: CorrelationId::from(13),
            function_id: FunctionId::from(14),
            pc_offset: 15,
            threads_executed: 16,
            shared_transactions: 17,
            theoretical_shared_transactions: 18,
            executed: 19,
        };

        assert_eq!(shared_record.shared_access(), Some(decoded_shared));
        assert_eq!(
            shared_record.decode(),
            ActivityRecordPayload::SharedAccess(decoded_shared)
        );
    }

    #[test]
    fn test_activity_record_decodes_confidential_compute_rotation_record() {
        let raw = sys::CUpti_ActivityConfidentialComputeRotation {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONFIDENTIAL_COMPUTE_ROTATION,
            eventType: sys::CUpti_ConfidentialComputeRotationEventType::CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_DRAINED,
            deviceId: 1,
            contextId: 2,
            channelId: 3,
            channelType: sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_DECOMP,
            timestamp: 4,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityConfidentialComputeRotation).cast(),
            _marker: PhantomData,
        };
        let decoded = ActivityConfidentialComputeRotation {
            event_type: ConfidentialComputeRotationEventType::ChannelDrained,
            device_id: DeviceId::from(1),
            context_id: ContextId::from(2),
            channel_id: ChannelId::from(3),
            channel_type: ChannelType::Decompress,
            timestamp: 4,
        };

        assert_eq!(record.confidential_compute_rotation(), Some(decoded));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::ConfidentialComputeRotation(decoded)
        );
    }

    #[test]
    fn test_activity_record_decodes_name_record() {
        let name = CString::new("stream-name").unwrap();
        let raw = sys::CUpti_ActivityName {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_NAME,
            objectKind: sys::CUpti_ActivityObjectKind::CUPTI_ACTIVITY_OBJECT_STREAM,
            objectId: sys::CUpti_ActivityObjectKindId {
                dcs: sys::CUpti_ActivityObjectKindId__bindgen_ty_2 {
                    deviceId: 1,
                    contextId: 2,
                    streamId: 3,
                },
            },
            pad: 0,
            name: name.as_ptr(),
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityName).cast(),
            _marker: PhantomData,
        };
        let decoded = record.name().unwrap();

        assert_eq!(decoded.object_kind, ActivityObjectKind::Stream);
        assert_eq!(
            decoded.object_id,
            ActivityObjectId::Stream {
                device_id: DeviceId::from(1),
                context_id: ContextId::from(2),
                stream_id: StreamId::from(3u64),
            }
        );
        assert_eq!(decoded.name.unwrap().to_str().unwrap(), "stream-name");
        assert!(matches!(record.decode(), ActivityRecordPayload::Name(_)));
    }

    #[test]
    fn test_activity_record_decodes_marker_record() {
        let name = CString::new("range").unwrap();
        let domain = CString::new("domain").unwrap();
        let raw = sys::CUpti_ActivityMarker2 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MARKER,
            flags: sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_START,
            timestamp: 1,
            id: 2,
            objectKind: sys::CUpti_ActivityObjectKind::CUPTI_ACTIVITY_OBJECT_THREAD,
            objectId: sys::CUpti_ActivityObjectKindId {
                pt: sys::CUpti_ActivityObjectKindId__bindgen_ty_1 {
                    processId: 3,
                    threadId: 4,
                },
            },
            pad: 0,
            name: name.as_ptr(),
            domain: domain.as_ptr(),
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMarker2).cast(),
            _marker: PhantomData,
        };
        let decoded = record.marker().unwrap();

        assert_eq!(decoded.flags, ActivityFlags::MARKER_START);
        assert_eq!(decoded.timestamp, 1);
        assert_eq!(decoded.id, ActivityMarkerId::from(2));
        assert_eq!(decoded.object_kind, ActivityObjectKind::Thread);
        assert_eq!(
            decoded.object_id,
            ActivityObjectId::Thread {
                process_id: ProcessId::from(3u64),
                thread_id: ThreadId::from(4),
            }
        );
        assert_eq!(decoded.name.unwrap().to_str().unwrap(), "range");
        assert_eq!(decoded.domain.unwrap().to_str().unwrap(), "domain");
        assert!(matches!(record.decode(), ActivityRecordPayload::Marker(_)));
    }

    #[test]
    fn test_activity_record_decodes_external_correlation_record() {
        let raw = sys::CUpti_ActivityExternalCorrelation {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
            externalKind:
                sys::CUpti_ExternalCorrelationKind::CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,
            externalId: 1,
            correlationId: 2,
            reserved: 0,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityExternalCorrelation).cast(),
            _marker: PhantomData,
        };
        let decoded = ActivityExternalCorrelation {
            external_kind: ExternalCorrelationKind::Custom0,
            external_id: ExternalCorrelationId::from(1),
            correlation_id: CorrelationId::from(2),
        };

        assert_eq!(record.external_correlation(), Some(decoded));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::ExternalCorrelation(decoded)
        );
    }

    #[test]
    fn test_activity_record_decodes_marker_data_record() {
        let raw = sys::CUpti_ActivityMarkerData2 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MARKER_DATA,
            flags: sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_END,
            id: 1,
            payloadKind: sys::CUpti_MetricValueKind::CUPTI_METRIC_VALUE_KIND_UINT64,
            payload: sys::CUpti_MetricValue {
                metricValueUint64: 2,
            },
            color: 3,
            category: 4,
            cuptiDomainId: 5,
            padding: 0,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMarkerData2).cast(),
            _marker: PhantomData,
        };
        let decoded = ActivityMarkerPayload {
            flags: ActivityFlags::MARKER_END,
            id: ActivityMarkerId::from(1),
            payload_kind: MetricValueKind::Uint64,
            payload: MetricValue::Uint64(2),
            color: 3,
            category: 4,
            cupti_domain_id: CuptiDomainId::from(5),
        };

        assert_eq!(record.marker_data(), Some(decoded));
        assert_eq!(record.decode(), ActivityRecordPayload::MarkerData(decoded));
    }

    #[test]
    fn test_activity_record_decodes_cuda_event_record() {
        let raw = sys::CUpti_ActivityCudaEvent2 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CUDA_EVENT,
            correlationId: 1,
            contextId: 2,
            streamId: 3,
            eventId: 4,
            pad: 0,
            deviceId: 5,
            pad2: 0,
            reserved0: ptr::null_mut(),
            deviceTimestamp: 6,
            cudaEventSyncId: 7,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityCudaEvent2).cast(),
            _marker: PhantomData,
        };
        let decoded = ActivityCudaEvent {
            correlation_id: CorrelationId::from(1),
            context_id: ContextId::from(2),
            stream_id: StreamId::from(3u64),
            event_id: CudaEventId::from(4),
            device_id: DeviceId::from(5),
            device_timestamp: 6,
            cuda_event_sync_id: CudaEventSyncId::from(7),
        };

        assert_eq!(record.cuda_event(), Some(decoded));
        assert_eq!(record.decode(), ActivityRecordPayload::CudaEvent(decoded));
    }

    #[test]
    fn test_activity_record_decodes_context_record() {
        let raw = sys::CUpti_ActivityContext4 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONTEXT,
            contextId: 1,
            deviceId: 2,
            computeApiKind: sys::CUpti_ActivityComputeApiKind::CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS
                as u16,
            nullStreamId: 3,
            parentContextId: 4,
            isGreenContext: 1,
            padding: 0,
            numMultiprocessors: 5,
            cigMode: sys::CUpti_ContextCigMode::CUPTI_CONTEXT_CIG_MODE_CIG_FALLBACK,
            padding2: 0,
            processId: 6,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityContext4).cast(),
            _marker: PhantomData,
        };
        let decoded = ActivityContext {
            context_id: ContextId::from(1),
            device_id: DeviceId::from(2),
            compute_api_kind: ActivityComputeApiKind::CudaMultiProcessService,
            null_stream_id: StreamId::from(3u64),
            parent_context_id: ContextId::from(4),
            is_green_context: true,
            num_multiprocessors: 5,
            cig_mode: ContextCigMode::CigFallback,
            process_id: ProcessId::from(6u64),
        };

        assert_eq!(record.context(), Some(decoded));
        assert_eq!(record.decode(), ActivityRecordPayload::Context(decoded));
    }

    #[test]
    fn test_activity_record_decodes_event_record() {
        let raw = sys::CUpti_ActivityEvent {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_EVENT,
            id: 1,
            value: 2,
            domain: 3,
            correlationId: 4,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityEvent).cast(),
            _marker: PhantomData,
        };
        let decoded = ActivityEvent {
            id: EventId::from(1),
            value: 2,
            domain_id: EventDomainId::from(3),
            correlation_id: CorrelationId::from(4),
        };

        assert_eq!(record.event(), Some(decoded));
        assert_eq!(record.decode(), ActivityRecordPayload::Event(decoded));
    }

    #[test]
    fn test_activity_record_decodes_event_instance_record() {
        let raw = sys::CUpti_ActivityEventInstance {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_EVENT_INSTANCE,
            id: 1,
            domain: 2,
            instance: 3,
            value: 4,
            correlationId: 5,
            pad: 0,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityEventInstance).cast(),
            _marker: PhantomData,
        };
        let decoded = ActivityEventInstance {
            id: EventId::from(1),
            domain_id: EventDomainId::from(2),
            instance: 3,
            value: 4,
            correlation_id: CorrelationId::from(5),
        };

        assert_eq!(record.event_instance(), Some(decoded));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::EventInstance(decoded)
        );
    }

    #[test]
    fn test_activity_record_decodes_metric_record() {
        let raw = sys::CUpti_ActivityMetric {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_METRIC,
            id: 1,
            value: sys::CUpti_MetricValue {
                metricValueUint64: 2,
            },
            correlationId: 3,
            flags: sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_START as u8,
            pad: [0; 3],
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMetric).cast(),
            _marker: PhantomData,
        };
        let decoded = ActivityMetric {
            id: MetricId::from(1),
            value: MetricValueRaw { bits: 2 },
            correlation_id: CorrelationId::from(3),
            flags: ActivityFlags::MARKER_START,
        };

        assert_eq!(
            decoded.value.value_as(MetricValueKind::Uint64),
            MetricValue::Uint64(2)
        );
        assert_eq!(record.metric(), Some(decoded));
        assert_eq!(record.decode(), ActivityRecordPayload::Metric(decoded));
    }

    #[test]
    fn test_activity_record_decodes_metric_instance_record() {
        let raw = sys::CUpti_ActivityMetricInstance {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_METRIC_INSTANCE,
            id: 1,
            value: sys::CUpti_MetricValue {
                metricValueInt64: -2,
            },
            instance: 3,
            correlationId: 4,
            flags: 0,
            pad: [0; 7],
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMetricInstance).cast(),
            _marker: PhantomData,
        };
        let decoded = ActivityMetricInstance {
            id: MetricId::from(1),
            value: MetricValueRaw {
                bits: (-2i64) as u64,
            },
            instance: 3,
            correlation_id: CorrelationId::from(4),
            flags: ActivityFlags::empty(),
        };

        assert_eq!(
            decoded.value.value_as(MetricValueKind::Int64),
            MetricValue::Int64(-2)
        );
        assert_eq!(record.metric_instance(), Some(decoded));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::MetricInstance(decoded)
        );
    }

    #[test]
    fn test_activity_record_decodes_instantaneous_records() {
        let event = sys::CUpti_ActivityInstantaneousEvent {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT,
            id: 1,
            value: 2,
            timestamp: 3,
            deviceId: 4,
            reserved: 0,
        };
        let event_record = ActivityRecord {
            ptr: (&event as *const sys::CUpti_ActivityInstantaneousEvent).cast(),
            _marker: PhantomData,
        };
        let decoded_event = ActivityInstantaneousEvent {
            id: EventId::from(1),
            value: 2,
            timestamp: 3,
            device_id: DeviceId::from(4),
        };

        assert_eq!(event_record.instantaneous_event(), Some(decoded_event));
        assert_eq!(
            event_record.decode(),
            ActivityRecordPayload::InstantaneousEvent(decoded_event)
        );

        let metric = sys::CUpti_ActivityInstantaneousMetricInstance {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE,
            id: 5,
            value: sys::CUpti_MetricValue {
                metricValueThroughput: 6,
            },
            timestamp: 7,
            deviceId: 8,
            flags: 0,
            instance: 9,
            pad: [0; 2],
        };
        let metric_record = ActivityRecord {
            ptr: (&metric as *const sys::CUpti_ActivityInstantaneousMetricInstance).cast(),
            _marker: PhantomData,
        };
        let decoded_metric = ActivityInstantaneousMetricInstance {
            id: MetricId::from(5),
            value: MetricValueRaw { bits: 6 },
            timestamp: 7,
            device_id: DeviceId::from(8),
            flags: ActivityFlags::empty(),
            instance: 9,
        };

        assert_eq!(
            decoded_metric.value.value_as(MetricValueKind::Throughput),
            MetricValue::Throughput(6)
        );
        assert_eq!(
            metric_record.instantaneous_metric_instance(),
            Some(decoded_metric)
        );
        assert_eq!(
            metric_record.decode(),
            ActivityRecordPayload::InstantaneousMetricInstance(decoded_metric)
        );
    }

    #[test]
    fn test_activity_record_decodes_source_level_records() {
        let global = sys::CUpti_ActivityGlobalAccess3 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS,
            flags: sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_LOAD,
            sourceLocatorId: 1,
            correlationId: 2,
            functionId: 3,
            executed: 4,
            pcOffset: 5,
            threadsExecuted: 6,
            l2_transactions: 7,
            theoreticalL2Transactions: 8,
        };
        let global_record = ActivityRecord {
            ptr: (&global as *const sys::CUpti_ActivityGlobalAccess3).cast(),
            _marker: PhantomData,
        };
        let decoded_global = ActivityGlobalAccess {
            flags: ActivityFlags::GLOBAL_ACCESS_KIND_LOAD,
            source_locator_id: SourceLocatorId::from(1),
            correlation_id: CorrelationId::from(2),
            function_id: FunctionId::from(3),
            executed: 4,
            pc_offset: 5,
            threads_executed: 6,
            l2_transactions: 7,
            theoretical_l2_transactions: 8,
        };

        assert_eq!(global_record.global_access(), Some(decoded_global));
        assert_eq!(
            global_record.decode(),
            ActivityRecordPayload::GlobalAccess(decoded_global)
        );

        let branch = sys::CUpti_ActivityBranch2 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_BRANCH,
            sourceLocatorId: 9,
            correlationId: 10,
            functionId: 11,
            pcOffset: 12,
            diverged: 13,
            threadsExecuted: 14,
            executed: 15,
            pad: 0,
        };
        let branch_record = ActivityRecord {
            ptr: (&branch as *const sys::CUpti_ActivityBranch2).cast(),
            _marker: PhantomData,
        };
        let decoded_branch = ActivityBranch {
            source_locator_id: SourceLocatorId::from(9),
            correlation_id: CorrelationId::from(10),
            function_id: FunctionId::from(11),
            pc_offset: 12,
            diverged: 13,
            threads_executed: 14,
            executed: 15,
        };

        assert_eq!(branch_record.branch(), Some(decoded_branch));
        assert_eq!(
            branch_record.decode(),
            ActivityRecordPayload::Branch(decoded_branch)
        );
    }

    #[test]
    fn test_activity_record_decodes_sampling_and_unified_memory_records() {
        let instruction = sys::CUpti_ActivityInstructionExecution {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION,
            flags: sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_INSTRUCTION_CLASS_MASK,
            sourceLocatorId: 1,
            correlationId: 2,
            functionId: 3,
            pcOffset: 4,
            threadsExecuted: 5,
            notPredOffThreadsExecuted: 6,
            executed: 7,
            pad: 0,
        };
        let instruction_record = ActivityRecord {
            ptr: (&instruction as *const sys::CUpti_ActivityInstructionExecution).cast(),
            _marker: PhantomData,
        };
        let decoded_instruction = ActivityInstructionExecution {
            flags: ActivityFlags::INSTRUCTION_CLASS_MASK,
            source_locator_id: SourceLocatorId::from(1),
            correlation_id: CorrelationId::from(2),
            function_id: FunctionId::from(3),
            pc_offset: 4,
            threads_executed: 5,
            not_predicated_off_threads_executed: 6,
            executed: 7,
        };

        assert_eq!(
            instruction_record.instruction_execution(),
            Some(decoded_instruction)
        );
        assert_eq!(
            instruction_record.decode(),
            ActivityRecordPayload::InstructionExecution(decoded_instruction)
        );

        let sampling = sys::CUpti_ActivityPCSampling3 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PC_SAMPLING,
            flags: sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_NONE,
            sourceLocatorId: 8,
            correlationId: 9,
            functionId: 10,
            latencySamples: 11,
            samples: 12,
            stallReason: sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY,
            pcOffset: 13,
        };
        let sampling_record = ActivityRecord {
            ptr: (&sampling as *const sys::CUpti_ActivityPCSampling3).cast(),
            _marker: PhantomData,
        };
        let decoded_sampling = ActivityPcSampling {
            flags: ActivityFlags::empty(),
            source_locator_id: SourceLocatorId::from(8),
            correlation_id: CorrelationId::from(9),
            function_id: FunctionId::from(10),
            latency_samples: 11,
            samples: 12,
            stall_reason: ActivityPcSamplingStallReason::MemoryDependency,
            pc_offset: 13,
        };

        assert_eq!(sampling_record.pc_sampling(), Some(decoded_sampling));
        assert_eq!(
            sampling_record.decode(),
            ActivityRecordPayload::PcSampling(decoded_sampling)
        );

        let info = sys::CUpti_ActivityPCSamplingRecordInfo {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO,
            correlationId: 14,
            totalSamples: 15,
            droppedSamples: 16,
            samplingPeriodInCycles: 17,
        };
        let info_record = ActivityRecord {
            ptr: (&info as *const sys::CUpti_ActivityPCSamplingRecordInfo).cast(),
            _marker: PhantomData,
        };
        let decoded_info = ActivityPcSamplingRecordInfo {
            correlation_id: CorrelationId::from(14),
            total_samples: 15,
            dropped_samples: 16,
            sampling_period_in_cycles: 17,
        };

        assert_eq!(info_record.pc_sampling_record_info(), Some(decoded_info));
        assert_eq!(
            info_record.decode(),
            ActivityRecordPayload::PcSamplingRecordInfo(decoded_info)
        );

        let unified = sys::CUpti_ActivityUnifiedMemoryCounter3 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER,
            counterKind: sys::CUpti_ActivityUnifiedMemoryCounterKind::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD,
            value: 18,
            start: 19,
            end: 20,
            address: 21,
            srcId: 22,
            dstId: 23,
            streamId: 24,
            processId: 25,
            flags: sys::CUpti_ActivityUnifiedMemoryMigrationCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_USER as u32,
            pad: 0,
            processors: [27, 28, 29, 30, 31],
        };
        let unified_record = ActivityRecord {
            ptr: (&unified as *const sys::CUpti_ActivityUnifiedMemoryCounter3).cast(),
            _marker: PhantomData,
        };
        let decoded_unified = ActivityUnifiedMemoryCounter {
            counter_kind: ActivityUnifiedMemoryCounterKind::BytesTransferHostToDevice,
            value: 18,
            start: 19,
            end: 20,
            address: 21,
            source_id: UnifiedMemoryProcessorId::from(22),
            destination_id: UnifiedMemoryProcessorId::from(23),
            stream_id: StreamId::from(24u64),
            process_id: ProcessId::from(25u64),
            flags: ActivityUnifiedMemoryCounterFlags::MigrationCause(
                ActivityUnifiedMemoryMigrationCause::User,
            ),
            processors: [27, 28, 29, 30, 31],
        };

        assert_eq!(
            unified_record.unified_memory_counter(),
            Some(decoded_unified)
        );
        assert_eq!(
            unified_record.decode(),
            ActivityRecordPayload::UnifiedMemoryCounter(decoded_unified)
        );
    }

    #[test]
    fn test_activity_record_decodes_device_attribute_records() {
        let cupti = sys::CUpti_ActivityDeviceAttribute {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE,
            flags: sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_NONE,
            deviceId: 1,
            attribute: sys::CUpti_ActivityDeviceAttribute__bindgen_ty_1 {
                cupti: sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_MAX_EVENT_ID,
            },
            value: sys::CUpti_ActivityDeviceAttribute__bindgen_ty_2 { vUint64: 2 },
        };
        let cupti_record = ActivityRecord {
            ptr: (&cupti as *const sys::CUpti_ActivityDeviceAttribute).cast(),
            _marker: PhantomData,
        };
        let decoded_cupti = ActivityDeviceAttribute {
            flags: ActivityFlags::empty(),
            device_id: DeviceId::from(1),
            attribute: ActivityDeviceAttributeKind::Cupti(DeviceAttribute::MaxEventId),
            value: ActivityDeviceAttributeValue { bits: 2 },
        };

        assert_eq!(decoded_cupti.value.as_u64(), 2);
        assert_eq!(cupti_record.device_attribute(), Some(decoded_cupti));
        assert_eq!(
            cupti_record.decode(),
            ActivityRecordPayload::DeviceAttribute(decoded_cupti)
        );

        let cuda = sys::CUpti_ActivityDeviceAttribute {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE,
            flags: sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_DEVICE_ATTRIBUTE_CUDEVICE,
            deviceId: 3,
            attribute: sys::CUpti_ActivityDeviceAttribute__bindgen_ty_1 {
                cu: sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            },
            value: sys::CUpti_ActivityDeviceAttribute__bindgen_ty_2 { vInt64: -4 },
        };
        let cuda_record = ActivityRecord {
            ptr: (&cuda as *const sys::CUpti_ActivityDeviceAttribute).cast(),
            _marker: PhantomData,
        };
        let decoded_cuda = cuda_record.device_attribute().unwrap();

        assert_eq!(decoded_cuda.flags, ActivityFlags::DEVICE_ATTRIBUTE_CUDEVICE);
        assert_eq!(decoded_cuda.device_id, DeviceId::from(3));
        assert_eq!(
            decoded_cuda.attribute,
            ActivityDeviceAttributeKind::Cuda(CudaDeviceAttribute::from(
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
            ))
        );
        assert_eq!(decoded_cuda.value.as_i64(), -4);
        assert!(matches!(
            cuda_record.decode(),
            ActivityRecordPayload::DeviceAttribute(_)
        ));
    }

    #[test]
    fn test_activity_record_decodes_environment_records() {
        let speed = sys::CUpti_ActivityEnvironment {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_ENVIRONMENT,
            deviceId: 1,
            timestamp: 2,
            environmentKind:
                sys::CUpti_ActivityEnvironmentKind::CUPTI_ACTIVITY_ENVIRONMENT_SPEED,
            data: sys::CUpti_ActivityEnvironment__bindgen_ty_1 {
                speed: sys::CUpti_ActivityEnvironmentSpeed {
                    smClock: 3,
                    memoryClock: 4,
                    pcieLinkGen: 5,
                    pcieLinkWidth: 6,
                    clocksThrottleReasons: sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_SW_POWER_CAP,
                },
            },
        };
        let speed_record = ActivityRecord {
            ptr: (&speed as *const sys::CUpti_ActivityEnvironment).cast(),
            _marker: PhantomData,
        };
        let decoded_speed = ActivityEnvironment {
            device_id: DeviceId::from(1),
            timestamp: 2,
            environment_kind: ActivityEnvironmentKind::Speed,
            data: ActivityEnvironmentReading::Speed(ActivityEnvironmentSpeed {
                sm_clock: 3,
                memory_clock: 4,
                pcie_link_gen: 5,
                pcie_link_width: 6,
                clocks_throttle_reasons: EnvironmentClocksThrottleReasons::SOFTWARE_POWER_CAP,
            }),
        };

        assert_eq!(speed_record.environment(), Some(decoded_speed));
        assert_eq!(
            speed_record.decode(),
            ActivityRecordPayload::Environment(decoded_speed)
        );

        let power = sys::CUpti_ActivityEnvironment {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_ENVIRONMENT,
            deviceId: 7,
            timestamp: 8,
            environmentKind: sys::CUpti_ActivityEnvironmentKind::CUPTI_ACTIVITY_ENVIRONMENT_POWER,
            data: sys::CUpti_ActivityEnvironment__bindgen_ty_1 {
                power: sys::CUpti_ActivityEnvironmentPower {
                    power: 9,
                    powerLimit: 10,
                },
            },
        };
        let power_record = ActivityRecord {
            ptr: (&power as *const sys::CUpti_ActivityEnvironment).cast(),
            _marker: PhantomData,
        };
        let decoded_power = ActivityEnvironment {
            device_id: DeviceId::from(7),
            timestamp: 8,
            environment_kind: ActivityEnvironmentKind::Power,
            data: ActivityEnvironmentReading::Power(ActivityEnvironmentPower {
                power: 9,
                power_limit: 10,
            }),
        };

        assert_eq!(power_record.environment(), Some(decoded_power));
        assert_eq!(
            power_record.decode(),
            ActivityRecordPayload::Environment(decoded_power)
        );
    }

    #[test]
    fn test_activity_record_decodes_device_record() {
        let name = CString::new("gpu").unwrap();
        let raw = sys::CUpti_ActivityDevice6 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE,
            flags: sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_NONE,
            globalMemoryBandwidth: 1,
            globalMemorySize: 2,
            constantMemorySize: 3,
            l2CacheSize: 4,
            numThreadsPerWarp: 5,
            coreClockRate: 6,
            numMemcpyEngines: 7,
            numMultiprocessors: 8,
            maxIPC: 9,
            maxWarpsPerMultiprocessor: 10,
            maxBlocksPerMultiprocessor: 11,
            maxSharedMemoryPerMultiprocessor: 12,
            maxRegistersPerMultiprocessor: 13,
            maxRegistersPerBlock: 14,
            maxSharedMemoryPerBlock: 15,
            maxThreadsPerBlock: 16,
            maxBlockDimX: 17,
            maxBlockDimY: 18,
            maxBlockDimZ: 19,
            maxGridDimX: 20,
            maxGridDimY: 21,
            maxGridDimZ: 22,
            computeCapabilityMajor: 23,
            computeCapabilityMinor: 24,
            id: 25,
            eccEnabled: 1,
            uuid: sys::CUuuid { bytes: [1; 16] },
            name: name.as_ptr(),
            isCudaVisible: 1,
            isMigEnabled: 1,
            reserved: [0; 6],
            gpuInstanceId: 26,
            computeInstanceId: 27,
            migUuid: sys::CUuuid { bytes: [2; 16] },
            isNumaNode: 1,
            numaId: 28,
            numTpcs: 29,
            reserved0: 0,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityDevice6).cast(),
            _marker: PhantomData,
        };
        let decoded = record.device().unwrap();

        assert_eq!(decoded.global_memory_bandwidth, 1);
        assert_eq!(decoded.device_id, DeviceId::from(25));
        assert!(decoded.ecc_enabled);
        assert_eq!(decoded.uuid, [1u8; 16]);
        assert_eq!(decoded.name.unwrap().to_str().unwrap(), "gpu");
        assert!(decoded.is_cuda_visible);
        assert!(decoded.is_mig_enabled);
        assert_eq!(decoded.mig_uuid, [2u8; 16]);
        assert_eq!(
            decoded.max_block_dim,
            Dim3 {
                x: 17,
                y: 18,
                z: 19
            }
        );
        assert_eq!(
            decoded.max_grid_dim,
            Dim3 {
                x: 20,
                y: 21,
                z: 22
            }
        );
        assert!(matches!(record.decode(), ActivityRecordPayload::Device(_)));
    }

    #[test]
    fn test_activity_record_decodes_jit_record() {
        let cache_path = CString::new("/tmp/cache").unwrap();
        let raw = sys::CUpti_ActivityJit2 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_JIT,
            jitEntryType: sys::CUpti_ActivityJitEntryType::CUPTI_ACTIVITY_JIT_ENTRY_PTX_TO_CUBIN,
            jitOperationType:
                sys::CUpti_ActivityJitOperationType::CUPTI_ACTIVITY_JIT_OPERATION_COMPILE,
            deviceId: 1,
            start: 2,
            end: 3,
            correlationId: 4,
            padding: 0,
            jitOperationCorrelationId: 5,
            cacheSize: 6,
            cachePath: cache_path.as_ptr(),
            processId: 7,
            threadId: 8,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityJit2).cast(),
            _marker: PhantomData,
        };
        let decoded = ActivityJit {
            entry_type: ActivityJitEntryType::ParallelThreadExecutionToCudaBinary,
            operation_type: ActivityJitOperationType::Compile,
            device_id: DeviceId::from(1),
            start: 2,
            end: 3,
            correlation_id: CorrelationId::from(4),
            operation_correlation_id: JitOperationCorrelationId::from(5),
            cache_size: 6,
            cache_path: Some(cache_path.as_c_str()),
            process_id: ProcessId::from(7u64),
            thread_id: ThreadId::from(8),
        };

        assert_eq!(record.jit(), Some(decoded));
        assert_eq!(record.decode(), ActivityRecordPayload::Jit(decoded));
    }

    #[test]
    fn test_activity_record_decodes_nvlink_record() {
        let mut ports0 = [1u32, 2];
        let mut ports1 = [3u32, 4];
        let raw = sys::CUpti_ActivityNvLink5 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_NVLINK,
            nvlinkVersion: 5,
            typeDev0: sys::CUpti_DevType::CUPTI_DEV_TYPE_GPU,
            typeDev1: sys::CUpti_DevType::CUPTI_DEV_TYPE_NPU,
            idDev0: sys::CUpti_ActivityNvLink5__bindgen_ty_1 {
                uuidDev: sys::CUuuid { bytes: [6; 16] },
            },
            idDev1: sys::CUpti_ActivityNvLink5__bindgen_ty_2 {
                npu: sys::CUpti_ActivityNvLinkNpu {
                    index: 7,
                    domainId: 8,
                },
            },
            flag: (sys::CUpti_LinkFlag::CUPTI_LINK_FLAG_PEER_ACCESS as u32)
                | (sys::CUpti_LinkFlag::CUPTI_LINK_FLAG_SYSMEM_ACCESS as u32),
            physicalNvLinkCount: 2,
            portDev0: ports0.as_mut_ptr(),
            portDev1: ports1.as_mut_ptr(),
            bandwidth: 9,
            nvswitchConnected: 1,
            pad: [0; 7],
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityNvLink5).cast(),
            _marker: PhantomData,
        };
        let decoded = record.nvlink().unwrap();

        assert_eq!(decoded.version, 5);
        assert_eq!(decoded.device0_type, DeviceType::Gpu);
        assert_eq!(decoded.device1_type, DeviceType::Npu);
        assert_eq!(
            decoded.device0_id,
            ActivityTopologyDeviceId::Gpu { uuid: [6; 16] }
        );
        assert_eq!(
            decoded.device1_id,
            ActivityTopologyDeviceId::Npu(ActivityNpuId {
                index: 7,
                domain_id: NpuDomainId::from(8),
            })
        );
        assert_eq!(
            decoded.flags,
            LinkFlags::PEER_ACCESS | LinkFlags::SYSTEM_MEMORY_ACCESS
        );
        assert_eq!(decoded.device0_ports, Some(ports0.as_slice()));
        assert_eq!(decoded.device1_ports, Some(ports1.as_slice()));
        assert_eq!(decoded.bandwidth, 9);
        assert!(decoded.nvswitch_connected);
        assert!(matches!(record.decode(), ActivityRecordPayload::NvLink(_)));
    }

    #[test]
    fn test_activity_record_decodes_pcie_records() {
        let gpu = sys::CUpti_ActivityPcie {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PCIE,
            type_: sys::CUpti_PcieDeviceType::CUPTI_PCIE_DEVICE_TYPE_GPU,
            id: sys::CUpti_ActivityPcie__bindgen_ty_1 { devId: 1 },
            domain: 2,
            pcieGeneration: 3,
            linkRate: 4,
            linkWidth: 5,
            upstreamBus: 6,
            attr: sys::CUpti_ActivityPcie__bindgen_ty_2 {
                gpuAttr: sys::CUpti_ActivityPcieGpuAttr {
                    uuidDev: sys::CUuuid { bytes: [7; 16] },
                    peerDev: [8; 32],
                },
            },
        };
        let gpu_record = ActivityRecord {
            ptr: (&gpu as *const sys::CUpti_ActivityPcie).cast(),
            _marker: PhantomData,
        };
        let gpu_decoded = ActivityPcie {
            device_type: PcieDeviceType::Gpu,
            id: ActivityPcieDeviceId::Gpu(CudaDevice::from(1)),
            domain: PcieDomainId::from(2),
            pcie_generationeration: 3,
            link_rate: 4,
            link_width: 5,
            upstream_bus: 6,
            attributes: ActivityPcieAttributes::Gpu(ActivityPcieGpuAttributes {
                uuid: [7; 16],
                peer_devices: [CudaDevice::from(8); 32],
            }),
        };

        assert_eq!(gpu_record.pcie(), Some(gpu_decoded));
        assert_eq!(
            gpu_record.decode(),
            ActivityRecordPayload::Pcie(gpu_decoded)
        );

        let bridge = sys::CUpti_ActivityPcie {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PCIE,
            type_: sys::CUpti_PcieDeviceType::CUPTI_PCIE_DEVICE_TYPE_BRIDGE,
            id: sys::CUpti_ActivityPcie__bindgen_ty_1 { bridgeId: 9 },
            domain: 10,
            pcieGeneration: 11,
            linkRate: 12,
            linkWidth: 13,
            upstreamBus: 14,
            attr: sys::CUpti_ActivityPcie__bindgen_ty_2 {
                bridgeAttr: sys::CUpti_ActivityPcieBridgeAttr {
                    secondaryBus: 15,
                    deviceId: 16,
                    vendorId: 17,
                    pad0: 0,
                },
            },
        };
        let bridge_record = ActivityRecord {
            ptr: (&bridge as *const sys::CUpti_ActivityPcie).cast(),
            _marker: PhantomData,
        };
        let bridge_decoded = ActivityPcie {
            device_type: PcieDeviceType::Bridge,
            id: ActivityPcieDeviceId::Bridge(PcieBridgeId::from(9)),
            domain: PcieDomainId::from(10),
            pcie_generationeration: 11,
            link_rate: 12,
            link_width: 13,
            upstream_bus: 14,
            attributes: ActivityPcieAttributes::Bridge(ActivityPcieBridgeAttributes {
                secondary_bus: PcieBusId::from(15),
                device_id: PcieHardwareDeviceId::from(16),
                vendor_id: PcieVendorId::from(17),
            }),
        };

        assert_eq!(bridge_record.pcie(), Some(bridge_decoded));
        assert_eq!(
            bridge_record.decode(),
            ActivityRecordPayload::Pcie(bridge_decoded)
        );
    }

    #[test]
    fn test_activity_record_decodes_instruction_correlation_record() {
        let raw = sys::CUpti_ActivityInstructionCorrelation {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION,
            flags: sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_INSTRUCTION_CLASS_MASK,
            sourceLocatorId: 1,
            functionId: 2,
            pcOffset: 3,
            pad: 0,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityInstructionCorrelation).cast(),
            _marker: PhantomData,
        };
        let decoded = ActivityInstructionCorrelation {
            flags: ActivityFlags::INSTRUCTION_CLASS_MASK,
            source_locator_id: SourceLocatorId::from(1),
            function_id: FunctionId::from(2),
            pc_offset: 3,
        };

        assert_eq!(record.instruction_correlation(), Some(decoded));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::InstructionCorrelation(decoded)
        );
    }

    #[test]
    fn test_activity_record_decodes_open_multi_processing_record() {
        let raw = sys::CUpti_ActivityOpenMp {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OPENMP,
            eventKind: sys::CUpti_OpenMpEventKind::CUPTI_OPENMP_EVENT_KIND_WAIT_BARRIER,
            version: 1,
            threadId: 2,
            start: 3,
            end: 4,
            cuProcessId: 5,
            cuThreadId: 6,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityOpenMp).cast(),
            _marker: PhantomData,
        };
        let decoded = ActivityOpenMultiProcessing {
            event_kind: OpenMultiProcessingEventKind::WaitBarrier,
            version: 1,
            thread_id: ThreadId::from(2),
            start: 3,
            end: 4,
            process_id: ProcessId::from(5u64),
            cu_thread_id: ThreadId::from(6),
        };

        assert_eq!(record.open_multi_processing(), Some(decoded));
        assert_eq!(
            record.decode(),
            ActivityRecordPayload::OpenMultiProcessing(decoded)
        );
    }

    #[test]
    fn test_activity_record_decodes_open_acc_records() {
        let source_file = CString::new("main.c").unwrap();
        let function_name = CString::new("step").unwrap();
        let variable_name = CString::new("x").unwrap();
        let data = sys::CUpti_ActivityOpenAccData {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OPENACC_DATA,
            eventKind: sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_ENQUEUE_UPLOAD,
            parentConstruct: sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_DATA,
            version: 1,
            implicit: 1,
            deviceType: 2,
            deviceNumber: 3,
            threadId: 4,
            async_: 5,
            asyncMap: 6,
            lineNo: 7,
            endLineNo: 8,
            funcLineNo: 9,
            funcEndLineNo: 10,
            start: 11,
            end: 12,
            cuDeviceId: 13,
            cuContextId: 14,
            cuStreamId: 15,
            cuProcessId: 16,
            cuThreadId: 17,
            externalId: 18,
            srcFile: source_file.as_ptr(),
            funcName: function_name.as_ptr(),
            bytes: 19,
            hostPtr: 20,
            devicePtr: 21,
            varName: variable_name.as_ptr(),
        };
        let data_record = ActivityRecord {
            ptr: (&data as *const sys::CUpti_ActivityOpenAccData).cast(),
            _marker: PhantomData,
        };
        let decoded_data = data_record.open_acc_data().unwrap();

        assert_eq!(
            decoded_data.base.event_kind,
            OpenAccEventKind::EnqueueUpload
        );
        assert_eq!(
            decoded_data.base.parent_construct,
            OpenAccConstructKind::Data
        );
        assert!(decoded_data.base.implicit);
        assert_eq!(decoded_data.base.cuda_device_id, DeviceId::from(13));
        assert_eq!(decoded_data.base.cuda_context_id, ContextId::from(14));
        assert_eq!(decoded_data.base.cuda_stream_id, StreamId::from(15u64));
        assert_eq!(
            decoded_data.base.external_id,
            ExternalCorrelationId::from(18)
        );
        assert_eq!(
            decoded_data.base.source_file.unwrap().to_str().unwrap(),
            "main.c"
        );
        assert_eq!(
            decoded_data.base.function_name.unwrap().to_str().unwrap(),
            "step"
        );
        assert_eq!(decoded_data.bytes, 19);
        assert_eq!(decoded_data.host_ptr, 20);
        assert_eq!(decoded_data.device_ptr, 21);
        assert_eq!(decoded_data.variable_name.unwrap().to_str().unwrap(), "x");
        assert!(matches!(
            data_record.decode(),
            ActivityRecordPayload::OpenAccData(_)
        ));

        let kernel_name = CString::new("kernel").unwrap();
        let launch = sys::CUpti_ActivityOpenAccLaunch {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH,
            eventKind: sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_ENQUEUE_LAUNCH,
            parentConstruct: sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_PARALLEL,
            version: 1,
            implicit: 0,
            deviceType: 2,
            deviceNumber: 3,
            threadId: 4,
            async_: 5,
            asyncMap: 6,
            lineNo: 7,
            endLineNo: 8,
            funcLineNo: 9,
            funcEndLineNo: 10,
            start: 11,
            end: 12,
            cuDeviceId: 13,
            cuContextId: 14,
            cuStreamId: 15,
            cuProcessId: 16,
            cuThreadId: 17,
            externalId: 18,
            srcFile: source_file.as_ptr(),
            funcName: function_name.as_ptr(),
            numGangs: 19,
            numWorkers: 20,
            vectorLength: 21,
            kernelName: kernel_name.as_ptr(),
        };
        let launch_record = ActivityRecord {
            ptr: (&launch as *const sys::CUpti_ActivityOpenAccLaunch).cast(),
            _marker: PhantomData,
        };
        let decoded_launch = launch_record.open_acc_launch().unwrap();

        assert_eq!(
            decoded_launch.base.event_kind,
            OpenAccEventKind::EnqueueLaunch
        );
        assert_eq!(
            decoded_launch.base.parent_construct,
            OpenAccConstructKind::Parallel
        );
        assert!(!decoded_launch.base.implicit);
        assert_eq!(decoded_launch.num_gangs, 19);
        assert_eq!(decoded_launch.num_workers, 20);
        assert_eq!(decoded_launch.vector_length, 21);
        assert_eq!(
            decoded_launch.kernel_name.unwrap().to_str().unwrap(),
            "kernel"
        );
        assert!(matches!(
            launch_record.decode(),
            ActivityRecordPayload::OpenAccLaunch(_)
        ));

        let other = sys::CUpti_ActivityOpenAccOther {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OPENACC_OTHER,
            eventKind: sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_WAIT,
            parentConstruct: sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_WAIT,
            version: 1,
            implicit: 1,
            deviceType: 2,
            deviceNumber: 3,
            threadId: 4,
            async_: 5,
            asyncMap: 6,
            lineNo: 7,
            endLineNo: 8,
            funcLineNo: 9,
            funcEndLineNo: 10,
            start: 11,
            end: 12,
            cuDeviceId: 13,
            cuContextId: 14,
            cuStreamId: 15,
            cuProcessId: 16,
            cuThreadId: 17,
            externalId: 18,
            srcFile: source_file.as_ptr(),
            funcName: function_name.as_ptr(),
        };
        let other_record = ActivityRecord {
            ptr: (&other as *const sys::CUpti_ActivityOpenAccOther).cast(),
            _marker: PhantomData,
        };
        let decoded_other = other_record.open_acc_other().unwrap();

        assert_eq!(decoded_other.base.event_kind, OpenAccEventKind::Wait);
        assert_eq!(
            decoded_other.base.parent_construct,
            OpenAccConstructKind::Wait
        );
        assert!(matches!(
            other_record.decode(),
            ActivityRecordPayload::OpenAccOther(_)
        ));
    }

    #[test]
    fn test_activity_record_decodes_source_locator_record() {
        let file_name = CString::new("kernel.cu").unwrap();
        let raw = sys::CUpti_ActivitySourceLocator {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR,
            id: 1,
            lineNumber: 2,
            pad: 0,
            fileName: file_name.as_ptr(),
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivitySourceLocator).cast(),
            _marker: PhantomData,
        };
        let decoded = record.source_locator().unwrap();

        assert_eq!(decoded.id, SourceLocatorId::from(1));
        assert_eq!(decoded.line_number, 2);
        assert_eq!(decoded.file_name.unwrap().to_str().unwrap(), "kernel.cu");
        assert!(matches!(
            record.decode(),
            ActivityRecordPayload::SourceLocator(_)
        ));
    }

    #[test]
    fn test_activity_record_decodes_overhead_record() {
        let raw = sys::CUpti_ActivityOverhead3 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OVERHEAD,
            overheadKind: sys::CUpti_ActivityOverheadKind::CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE,
            objectKind: sys::CUpti_ActivityObjectKind::CUPTI_ACTIVITY_OBJECT_DEVICE,
            objectId: sys::CUpti_ActivityObjectKindId {
                dcs: sys::CUpti_ActivityObjectKindId__bindgen_ty_2 {
                    deviceId: 1,
                    contextId: 0,
                    streamId: 0,
                },
            },
            start: 2,
            end: 3,
            correlationId: 4,
            reserved0: 0,
            overheadData: ptr::null_mut(),
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityOverhead3).cast(),
            _marker: PhantomData,
        };
        let decoded = ActivityOverhead {
            overhead_kind: ActivityOverheadKind::Resource,
            object_kind: ActivityObjectKind::Device,
            object_id: ActivityObjectId::Device {
                device_id: DeviceId::from(1),
            },
            start: 2,
            end: 3,
            correlation_id: CorrelationId::from(4),
            has_overhead_data: false,
        };

        assert_eq!(record.overhead(), Some(decoded));
        assert_eq!(record.decode(), ActivityRecordPayload::Overhead(decoded));
    }

    #[test]
    fn test_activity_record_decodes_memory_copy_record() {
        let raw = sys::CUpti_ActivityMemcpy7 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY,
            copyKind: sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_HTOD as u8,
            srcKind: sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE as u8,
            dstKind: sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE as u8,
            flags: sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_DEVICE_CONCURRENT_KERNELS as u8,
            bytes: 1024,
            start: 10,
            end: 20,
            deviceId: 1,
            contextId: 2,
            streamId: 3,
            correlationId: 4,
            runtimeCorrelationId: 5,
            pad: 0,
            reserved0: ptr::null_mut(),
            graphNodeId: 6,
            graphId: 7,
            channelID: 8,
            channelType: sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY,
            isDeviceLaunched: 1,
            pad2: [0; 3],
            copyCount: 2,
            ..Default::default()
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMemcpy7).cast(),
            _marker: PhantomData,
        };

        assert_eq!(record.memory_decompress(), None);
        assert_eq!(
            record.memory_copy(),
            Some(ActivityMemoryCopy {
                source_graph_id: 0.into(),
                source_graph_node_id: 0.into(),
                copy_kind: ActivityMemoryCopyKind::HostToDevice,
                source_kind: ActivityMemoryKind::Pageable,
                destination_kind: ActivityMemoryKind::Device,
                flags: ActivityFlags::DEVICE_CONCURRENT_KERNELS,
                bytes: 1024,
                start: 10,
                end: 20,
                device_id: DeviceId::from(1),
                context_id: ContextId::from(2),
                stream_id: StreamId::from(3u64),
                correlation_id: CorrelationId::from(4),
                runtime_correlation_id: CorrelationId::from(5),
                graph_node_id: GraphNodeId::from(6),
                graph_id: GraphId::from(7),
                channel_id: ChannelId::from(8),
                channel_type: ChannelType::AsynchronousMemoryCopy,
                is_device_launched: true,
                copy_count: 2,
            })
        );
        assert!(matches!(
            record.decode(),
            ActivityRecordPayload::MemoryCopy(ActivityMemoryCopy {
                copy_kind: ActivityMemoryCopyKind::HostToDevice,
                bytes: 1024,
                ..
            })
        ));
    }

    #[test]
    fn test_activity_record_decodes_memory_set_record() {
        let raw = sys::CUpti_ActivityMemset5 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMSET,
            value: 255,
            bytes: 2048,
            start: 30,
            end: 40,
            deviceId: 1,
            contextId: 2,
            streamId: 3,
            correlationId: 4,
            flags: sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_DEVICE_CONCURRENT_KERNELS as u16,
            memoryKind: sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE as u16,
            pad: 0,
            reserved0: ptr::null_mut(),
            graphNodeId: 5,
            graphId: 6,
            channelID: 7,
            channelType: sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_COMPUTE,
            isDeviceLaunched: 0,
            pad2: [0; 3],
            ..Default::default()
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMemset5).cast(),
            _marker: PhantomData,
        };

        assert_eq!(record.memory_copy(), None);
        assert_eq!(
            record.memory_set(),
            Some(ActivityMemorySet {
                source_graph_id: 0.into(),
                source_graph_node_id: 0.into(),
                value: 255,
                bytes: 2048,
                start: 30,
                end: 40,
                device_id: DeviceId::from(1),
                context_id: ContextId::from(2),
                stream_id: StreamId::from(3u64),
                correlation_id: CorrelationId::from(4),
                flags: ActivityFlags::DEVICE_CONCURRENT_KERNELS,
                memory_kind: ActivityMemoryKind::Device,
                graph_node_id: GraphNodeId::from(5),
                graph_id: GraphId::from(6),
                channel_id: ChannelId::from(7),
                channel_type: ChannelType::Compute,
                is_device_launched: false,
            })
        );
        assert!(matches!(
            record.decode(),
            ActivityRecordPayload::MemorySet(ActivityMemorySet {
                value: 255,
                bytes: 2048,
                ..
            })
        ));
    }

    #[test]
    fn test_typed_attribute_setter_rejects_mismatched_value_kind_before_ffi() {
        let mut called = false;
        let error = set_attribute_value_with(
            |_, _, _| {
                called = true;
                Ok(())
            },
            ActivityAttribute::DeviceBufferSize,
            ActivityAttributeSetting::Enabled(true),
        )
        .unwrap_err();

        assert!(!called);
        assert!(matches!(error, Error::InvalidAttribute { .. }));
    }

    #[test]
    fn test_typed_attribute_setter_writes_size_t_for_byte_attributes() -> Result<()> {
        let mut observed = None;
        set_attribute_value_with(
            |attr, value_size, value| {
                // SAFETY: the caller passed valid size and value locations for this attribute.
                let value_size = unsafe { *value_size };
                // SAFETY: the caller passed valid size and value locations for this attribute.
                let value = unsafe { *value.cast::<usize>() };
                observed = Some((attr, value_size, value));
                Ok(())
            },
            ActivityAttribute::DeviceBufferSize,
            ActivityAttributeSetting::Bytes(4096),
        )?;

        assert_eq!(
            observed,
            Some((
                ActivityAttribute::DeviceBufferSize,
                size_of::<usize>() as u64,
                4096,
            ))
        );
        Ok(())
    }

    #[test]
    fn test_typed_attribute_setter_writes_u8_for_bool_attributes() -> Result<()> {
        let mut observed = None;
        set_attribute_value_with(
            |attr, value_size, value| {
                // SAFETY: the caller passed valid size and value locations for this attribute.
                let value_size = unsafe { *value_size };
                // SAFETY: the caller passed valid size and value locations for this attribute.
                let value = unsafe { *value.cast::<u8>() };
                observed = Some((attr, value_size, value));
                Ok(())
            },
            ActivityAttribute::EnableAllSyncRecords,
            ActivityAttributeSetting::Enabled(true),
        )?;

        assert_eq!(
            observed,
            Some((
                ActivityAttribute::EnableAllSyncRecords,
                size_of::<u8>() as u64,
                1,
            ))
        );
        Ok(())
    }
}
