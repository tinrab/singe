use std::{
    alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error},
    ffi::CStr,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
    mem::{self, MaybeUninit, size_of},
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    ptr, slice,
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
    types::*,
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

#[derive(Debug, Clone, Copy)]
pub struct ActivityBuffer<'a> {
    bytes: &'a [u8],
}

impl<'a> ActivityBuffer<'a> {
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    pub const fn len(self) -> usize {
        self.bytes.len()
    }

    pub const fn is_empty(self) -> bool {
        self.bytes.is_empty()
    }

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

#[derive(Debug, Clone)]
pub struct ActivityBufferOwned {
    bytes: Vec<u8>,
}

impl ActivityBufferOwned {
    /// Returns the raw CUPTI activity bytes captured from a completed buffer.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

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

#[derive(Debug)]
pub struct ActivityCallbackRegistration {
    owner_id: u64,
}

impl Drop for ActivityCallbackRegistration {
    fn drop(&mut self) {
        clear_activity_callbacks_if_owner(self.owner_id);
    }
}

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

#[derive(Debug, Clone, Copy)]
pub struct ActivityRecord<'a> {
    ptr: *const sys::CUpti_Activity,
    _marker: PhantomData<&'a sys::CUpti_Activity>,
}

impl<'a> ActivityRecord<'a> {
    pub fn kind(self) -> ActivityKind {
        let record = unsafe { &*self.ptr };
        ActivityKind::from(record.kind)
    }

    pub fn memory_decompress(self) -> Option<ActivityMemoryDecompress> {
        if self.kind() != ActivityKind::MemoryDecompress {
            return None;
        }
        let record = unsafe { &*self.ptr.cast::<sys::CUpti_ActivityMemDecompress>() };
        Some(ActivityMemoryDecompress::from_raw(*record))
    }

    pub fn memory_copy(self) -> Option<ActivityMemoryCopy> {
        if self.kind() != ActivityKind::MemoryCopy {
            return None;
        }
        let record = unsafe { &*self.ptr.cast::<sys::CUpti_ActivityMemcpy6>() };
        Some(ActivityMemoryCopy::from_raw(*record))
    }

    pub fn peer_memory_copy(self) -> Option<ActivityPeerMemoryCopy> {
        if self.kind() != ActivityKind::PeerMemoryCopy {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMemcpyPtoP4>()) };
        Some(ActivityPeerMemoryCopy::from_raw(record))
    }

    pub fn memory_set(self) -> Option<ActivityMemorySet> {
        if self.kind() != ActivityKind::MemorySet {
            return None;
        }
        let record = unsafe { &*self.ptr.cast::<sys::CUpti_ActivityMemset4>() };
        Some(ActivityMemorySet::from_raw(*record))
    }

    pub fn kernel(self) -> Option<ActivityKernel<'a>> {
        if !matches!(
            self.kind(),
            ActivityKind::Kernel | ActivityKind::ConcurrentKernel
        ) {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityKernel11>()) };
        Some(ActivityKernel::from_raw(record))
    }

    pub fn cuda_dynamic_parallelism_kernel(
        self,
    ) -> Option<ActivityCudaDynamicParallelismKernel<'a>> {
        if self.kind() != ActivityKind::CudaDynamicParallelismKernel {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityCdpKernel>()) };
        Some(ActivityCudaDynamicParallelismKernel::from_raw(record))
    }

    pub fn api(self) -> Option<ActivityApi> {
        if !matches!(
            self.kind(),
            ActivityKind::Driver | ActivityKind::Runtime | ActivityKind::InternalLaunchApi
        ) {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityAPI>()) };
        Some(ActivityApi::from_raw(record))
    }

    pub fn stream(self) -> Option<ActivityStream> {
        if self.kind() != ActivityKind::Stream {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityStream>()) };
        Some(ActivityStream::from_raw(record))
    }

    pub fn synchronization(self) -> Option<ActivitySynchronization> {
        if self.kind() != ActivityKind::Synchronization {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivitySynchronization2>()) };
        Some(ActivitySynchronization::from_raw(record))
    }

    pub fn memory(self) -> Option<ActivityMemory<'a>> {
        if self.kind() != ActivityKind::Memory2 {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMemory4>()) };
        Some(ActivityMemory::from_raw(record))
    }

    pub fn legacy_memory(self) -> Option<ActivityLegacyMemory<'a>> {
        if self.kind() != ActivityKind::Memory {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMemory>()) };
        Some(ActivityLegacyMemory::from_raw(record))
    }

    pub fn memory_pool(self) -> Option<ActivityMemoryPool> {
        if self.kind() != ActivityKind::MemoryPool {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMemoryPool3>()) };
        Some(ActivityMemoryPool::from_raw(record))
    }

    pub fn graph_trace(self) -> Option<ActivityGraphTrace> {
        if self.kind() != ActivityKind::GraphTrace {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityGraphTrace2>()) };
        Some(ActivityGraphTrace::from_raw(record))
    }

    pub fn device_graph_trace(self) -> Option<ActivityDeviceGraphTrace> {
        if self.kind() != ActivityKind::DeviceGraphTrace {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityDeviceGraphTrace>()) };
        Some(ActivityDeviceGraphTrace::from_raw(record))
    }

    pub fn graph_host_node(self) -> Option<ActivityGraphHostNode> {
        if self.kind() != ActivityKind::GraphHostNode {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityGraphHostNode>()) };
        Some(ActivityGraphHostNode::from_raw(record))
    }

    pub fn host_launch(self) -> Option<ActivityHostLaunch> {
        if self.kind() != ActivityKind::HostLaunch {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityHostLaunch>()) };
        Some(ActivityHostLaunch::from_raw(record))
    }

    pub fn compute_engine_context_switch(self) -> Option<ActivityComputeEngineContextSwitch> {
        if self.kind() != ActivityKind::ComputeEngineCtxSwitch {
            return None;
        }
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityComputeEngineCtxSwitch>())
        };
        Some(ActivityComputeEngineContextSwitch::from_raw(record))
    }

    pub fn green_context(self) -> Option<ActivityGreenContext> {
        if self.kind() != ActivityKind::GreenContext {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityGreenContext>()) };
        Some(ActivityGreenContext::from_raw(record))
    }

    pub fn name(self) -> Option<ActivityName<'a>> {
        if self.kind() != ActivityKind::Name {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityName>()) };
        Some(ActivityName::from_raw(record))
    }

    pub fn marker(self) -> Option<ActivityMarker<'a>> {
        if self.kind() != ActivityKind::Marker {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMarker2>()) };
        Some(ActivityMarker::from_raw(record))
    }

    pub fn external_correlation(self) -> Option<ActivityExternalCorrelation> {
        if self.kind() != ActivityKind::ExternalCorrelation {
            return None;
        }
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityExternalCorrelation>())
        };
        Some(ActivityExternalCorrelation::from_raw(record))
    }

    pub fn marker_data(self) -> Option<ActivityMarkerData> {
        if self.kind() != ActivityKind::MarkerData {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMarkerData2>()) };
        Some(ActivityMarkerData::from_raw(record))
    }

    pub fn cuda_event(self) -> Option<ActivityCudaEvent> {
        if self.kind() != ActivityKind::CudaEvent {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityCudaEvent2>()) };
        Some(ActivityCudaEvent::from_raw(record))
    }

    pub fn context(self) -> Option<ActivityContext> {
        if self.kind() != ActivityKind::Context {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityContext4>()) };
        Some(ActivityContext::from_raw(record))
    }

    pub fn event(self) -> Option<ActivityEvent> {
        if self.kind() != ActivityKind::Event {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityEvent>()) };
        Some(ActivityEvent::from_raw(record))
    }

    pub fn event_instance(self) -> Option<ActivityEventInstance> {
        if self.kind() != ActivityKind::EventInstance {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityEventInstance>()) };
        Some(ActivityEventInstance::from_raw(record))
    }

    pub fn metric(self) -> Option<ActivityMetric> {
        if self.kind() != ActivityKind::Metric {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMetric>()) };
        Some(ActivityMetric::from_raw(record))
    }

    pub fn metric_instance(self) -> Option<ActivityMetricInstance> {
        if self.kind() != ActivityKind::MetricInstance {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityMetricInstance>()) };
        Some(ActivityMetricInstance::from_raw(record))
    }

    pub fn instantaneous_event(self) -> Option<ActivityInstantaneousEvent> {
        if self.kind() != ActivityKind::InstantaneousEvent {
            return None;
        }
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityInstantaneousEvent>())
        };
        Some(ActivityInstantaneousEvent::from_raw(record))
    }

    pub fn instantaneous_event_instance(self) -> Option<ActivityInstantaneousEventInstance> {
        if self.kind() != ActivityKind::InstantaneousEventInstance {
            return None;
        }
        let record = unsafe {
            ptr::read_unaligned(
                self.ptr
                    .cast::<sys::CUpti_ActivityInstantaneousEventInstance>(),
            )
        };
        Some(ActivityInstantaneousEventInstance::from_raw(record))
    }

    pub fn instantaneous_metric(self) -> Option<ActivityInstantaneousMetric> {
        if self.kind() != ActivityKind::InstantaneousMetric {
            return None;
        }
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityInstantaneousMetric>())
        };
        Some(ActivityInstantaneousMetric::from_raw(record))
    }

    pub fn instantaneous_metric_instance(self) -> Option<ActivityInstantaneousMetricInstance> {
        if self.kind() != ActivityKind::InstantaneousMetricInstance {
            return None;
        }
        let record = unsafe {
            ptr::read_unaligned(
                self.ptr
                    .cast::<sys::CUpti_ActivityInstantaneousMetricInstance>(),
            )
        };
        Some(ActivityInstantaneousMetricInstance::from_raw(record))
    }

    pub fn global_access(self) -> Option<ActivityGlobalAccess> {
        if self.kind() != ActivityKind::GlobalAccess {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityGlobalAccess3>()) };
        Some(ActivityGlobalAccess::from_raw(record))
    }

    pub fn branch(self) -> Option<ActivityBranch> {
        if self.kind() != ActivityKind::Branch {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityBranch2>()) };
        Some(ActivityBranch::from_raw(record))
    }

    pub fn instruction_execution(self) -> Option<ActivityInstructionExecution> {
        if self.kind() != ActivityKind::InstructionExecution {
            return None;
        }
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityInstructionExecution>())
        };
        Some(ActivityInstructionExecution::from_raw(record))
    }

    pub fn pc_sampling(self) -> Option<ActivityPcSampling> {
        if self.kind() != ActivityKind::PcSampling {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityPCSampling3>()) };
        Some(ActivityPcSampling::from_raw(record))
    }

    pub fn pc_sampling_record_info(self) -> Option<ActivityPcSamplingRecordInfo> {
        if self.kind() != ActivityKind::PcSamplingRecordInfo {
            return None;
        }
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityPCSamplingRecordInfo>())
        };
        Some(ActivityPcSamplingRecordInfo::from_raw(record))
    }

    pub fn unified_memory_counter(self) -> Option<ActivityUnifiedMemoryCounter> {
        if self.kind() != ActivityKind::UnifiedMemoryCounter {
            return None;
        }
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityUnifiedMemoryCounter3>())
        };
        Some(ActivityUnifiedMemoryCounter::from_raw(record))
    }

    pub fn device_attribute(self) -> Option<ActivityDeviceAttribute> {
        if self.kind() != ActivityKind::DeviceAttribute {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityDeviceAttribute>()) };
        Some(ActivityDeviceAttribute::from_raw(record))
    }

    pub fn environment(self) -> Option<ActivityEnvironment> {
        if self.kind() != ActivityKind::Environment {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityEnvironment>()) };
        Some(ActivityEnvironment::from_raw(record))
    }

    pub fn device(self) -> Option<ActivityDevice<'a>> {
        if self.kind() != ActivityKind::Device {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityDevice6>()) };
        Some(ActivityDevice::from_raw(record))
    }

    pub fn jit(self) -> Option<ActivityJit<'a>> {
        if self.kind() != ActivityKind::Jit {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityJit2>()) };
        Some(ActivityJit::from_raw(record))
    }

    pub fn nvlink(self) -> Option<ActivityNvLink<'a>> {
        if self.kind() != ActivityKind::NvLink {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityNvLink5>()) };
        Some(ActivityNvLink::from_raw(record))
    }

    pub fn pcie(self) -> Option<ActivityPcie> {
        if self.kind() != ActivityKind::Pcie {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityPcie>()) };
        Some(ActivityPcie::from_raw(record))
    }

    pub fn instruction_correlation(self) -> Option<ActivityInstructionCorrelation> {
        if self.kind() != ActivityKind::InstructionCorrelation {
            return None;
        }
        let record = unsafe {
            ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityInstructionCorrelation>())
        };
        Some(ActivityInstructionCorrelation::from_raw(record))
    }

    pub fn open_multi_processing(self) -> Option<ActivityOpenMultiProcessing> {
        if self.kind() != ActivityKind::OpenMultiProcessing {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityOpenMp>()) };
        Some(ActivityOpenMultiProcessing::from_raw(record))
    }

    pub fn open_acc_data(self) -> Option<ActivityOpenAccData<'a>> {
        if self.kind() != ActivityKind::OpenAccData {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityOpenAccData>()) };
        Some(ActivityOpenAccData::from_raw(record))
    }

    pub fn open_acc_launch(self) -> Option<ActivityOpenAccLaunch<'a>> {
        if self.kind() != ActivityKind::OpenAccLaunch {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityOpenAccLaunch>()) };
        Some(ActivityOpenAccLaunch::from_raw(record))
    }

    pub fn open_acc_other(self) -> Option<ActivityOpenAccOther<'a>> {
        if self.kind() != ActivityKind::OpenAccOther {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityOpenAccOther>()) };
        Some(ActivityOpenAccOther::from_raw(record))
    }

    pub fn preemption(self) -> Option<ActivityPreemption> {
        if self.kind() != ActivityKind::Preemption {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityPreemption>()) };
        Some(ActivityPreemption::from_raw(record))
    }

    pub fn function(self) -> Option<ActivityFunction<'a>> {
        if self.kind() != ActivityKind::Function {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityFunction>()) };
        Some(ActivityFunction::from_raw(record))
    }

    pub fn module(self) -> Option<ActivityModule<'a>> {
        if self.kind() != ActivityKind::Module {
            return None;
        }
        let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityModule>()) };
        Some(ActivityModule::from_raw(record))
    }

    pub fn shared_access(self) -> Option<ActivitySharedAccess> {
        if self.kind() != ActivityKind::SharedAccess {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivitySharedAccess>()) };
        Some(ActivitySharedAccess::from_raw(record))
    }

    pub fn confidential_compute_rotation(self) -> Option<ActivityConfidentialComputeRotation> {
        if self.kind() != ActivityKind::ConfidentialComputeRotation {
            return None;
        }
        let record = unsafe {
            ptr::read_unaligned(
                self.ptr
                    .cast::<sys::CUpti_ActivityConfidentialComputeRotation>(),
            )
        };
        Some(ActivityConfidentialComputeRotation::from_raw(record))
    }

    pub fn source_locator(self) -> Option<ActivitySourceLocator<'a>> {
        if self.kind() != ActivityKind::SourceLocator {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivitySourceLocator>()) };
        Some(ActivitySourceLocator::from_raw(record))
    }

    pub fn overhead(self) -> Option<ActivityOverhead> {
        if self.kind() != ActivityKind::Overhead {
            return None;
        }
        let record =
            unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityOverhead3>()) };
        Some(ActivityOverhead::from_raw(record))
    }

    pub fn decode(self) -> ActivityRecordData<'a> {
        let kind = self.kind();

        macro_rules! decode_record {
            ($method:ident, $variant:ident) => {
                self.$method()
                    .map(ActivityRecordData::$variant)
                    .unwrap_or(ActivityRecordData::Unsupported { kind })
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
            kind => ActivityRecordData::Unsupported { kind },
        }
    }
}

fn memory_copy_kind_from_raw(value: u8) -> ActivityMemoryCopyKind {
    ActivityMemoryCopyKind::try_from(value as u32).unwrap_or(ActivityMemoryCopyKind::Unknown)
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
    ActivityLaunchType::try_from(value as u32).unwrap_or(ActivityLaunchType::Regular)
}

fn function_shared_memory_limit_config_from_raw(
    value: sys::CUpti_FuncShmemLimitConfig,
) -> FunctionSharedMemoryLimitConfig {
    FunctionSharedMemoryLimitConfig::try_from(value as u32)
        .unwrap_or(FunctionSharedMemoryLimitConfig::Default)
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ActivityRecordData<'a> {
    Api(ActivityApi),
    Kernel(ActivityKernel<'a>),
    CudaDynamicParallelismKernel(ActivityCudaDynamicParallelismKernel<'a>),
    MemoryCopy(ActivityMemoryCopy),
    PeerMemoryCopy(ActivityPeerMemoryCopy),
    MemorySet(ActivityMemorySet),
    MemoryDecompress(ActivityMemoryDecompress),
    Stream(ActivityStream),
    Synchronization(ActivitySynchronization),
    Memory(ActivityMemory<'a>),
    LegacyMemory(ActivityLegacyMemory<'a>),
    MemoryPool(ActivityMemoryPool),
    GraphTrace(ActivityGraphTrace),
    DeviceGraphTrace(ActivityDeviceGraphTrace),
    GraphHostNode(ActivityGraphHostNode),
    HostLaunch(ActivityHostLaunch),
    ComputeEngineContextSwitch(ActivityComputeEngineContextSwitch),
    GreenContext(ActivityGreenContext),
    Name(ActivityName<'a>),
    Marker(ActivityMarker<'a>),
    ExternalCorrelation(ActivityExternalCorrelation),
    MarkerData(ActivityMarkerData),
    CudaEvent(ActivityCudaEvent),
    Context(ActivityContext),
    Event(ActivityEvent),
    EventInstance(ActivityEventInstance),
    Metric(ActivityMetric),
    MetricInstance(ActivityMetricInstance),
    InstantaneousEvent(ActivityInstantaneousEvent),
    InstantaneousEventInstance(ActivityInstantaneousEventInstance),
    InstantaneousMetric(ActivityInstantaneousMetric),
    InstantaneousMetricInstance(ActivityInstantaneousMetricInstance),
    GlobalAccess(ActivityGlobalAccess),
    Branch(ActivityBranch),
    InstructionExecution(ActivityInstructionExecution),
    PcSampling(ActivityPcSampling),
    PcSamplingRecordInfo(ActivityPcSamplingRecordInfo),
    UnifiedMemoryCounter(ActivityUnifiedMemoryCounter),
    DeviceAttribute(ActivityDeviceAttribute),
    Environment(ActivityEnvironment),
    Device(ActivityDevice<'a>),
    Jit(ActivityJit<'a>),
    NvLink(ActivityNvLink<'a>),
    Pcie(ActivityPcie),
    InstructionCorrelation(ActivityInstructionCorrelation),
    OpenMultiProcessing(ActivityOpenMultiProcessing),
    OpenAccData(ActivityOpenAccData<'a>),
    OpenAccLaunch(ActivityOpenAccLaunch<'a>),
    OpenAccOther(ActivityOpenAccOther<'a>),
    Preemption(ActivityPreemption),
    Function(ActivityFunction<'a>),
    Module(ActivityModule<'a>),
    SharedAccess(ActivitySharedAccess),
    ConfidentialComputeRotation(ActivityConfidentialComputeRotation),
    SourceLocator(ActivitySourceLocator<'a>),
    Overhead(ActivityOverhead),
    Unsupported { kind: ActivityKind },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityApi {
    pub kind: ActivityKind,
    pub callback_id: CallbackId,
    pub start: u64,
    pub end: u64,
    pub process_id: ProcessId,
    pub thread_id: ThreadId,
    pub correlation_id: CorrelationId,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityStream {
    pub context_id: ContextId,
    pub stream_id: StreamId,
    pub priority: u32,
    pub flag: ActivityStreamFlag,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivitySynchronization {
    pub synchronization_type: ActivitySynchronizationType,
    pub start: u64,
    pub end: u64,
    pub correlation_id: CorrelationId,
    pub context_id: ContextId,
    pub stream_id: StreamId,
    pub cuda_event_id: CudaEventId,
    pub cuda_event_sync_id: CudaEventSyncId,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemoryPoolConfig {
    pub pool_type: ActivityMemoryPoolType,
    pub address: u64,
    pub data: ActivityMemoryPoolConfigData,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityMemoryPoolConfigData {
    Local {
        size: u64,
        release_threshold: u64,
        utilized_size: u64,
    },
    Imported {
        process_id: ProcessId,
    },
    Unknown {
        raw_value: u64,
        release_threshold: u64,
        utilized_size: u64,
    },
}

impl ActivityMemoryPoolConfig {
    fn from_raw(value: sys::CUpti_ActivityMemory4__bindgen_ty_1) -> Self {
        let pool_type = ActivityMemoryPoolType::from(value.memoryPoolType);
        let data = match pool_type {
            ActivityMemoryPoolType::Local => ActivityMemoryPoolConfigData::Local {
                size: unsafe { value.pool.size },
                release_threshold: value.releaseThreshold,
                utilized_size: value.utilizedSize,
            },
            ActivityMemoryPoolType::Imported => ActivityMemoryPoolConfigData::Imported {
                process_id: ProcessId::from(unsafe { value.pool.processId }),
            },
            ActivityMemoryPoolType::Invalid | ActivityMemoryPoolType::ForceInt => {
                ActivityMemoryPoolConfigData::Unknown {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemory<'a> {
    pub operation_type: ActivityMemoryOperationType,
    pub memory_kind: ActivityMemoryKind,
    pub correlation_id: CorrelationId,
    pub address: u64,
    pub bytes: u64,
    pub timestamp: u64,
    pub pc: u64,
    pub process_id: ProcessId,
    pub device_id: DeviceId,
    pub context_id: ContextId,
    pub stream_id: StreamId,
    pub name: Option<&'a CStr>,
    pub is_asynchronous: bool,
    pub memory_pool_config: ActivityMemoryPoolConfig,
    pub source: Option<&'a CStr>,
}

impl<'a> ActivityMemory<'a> {
    fn from_raw(value: sys::CUpti_ActivityMemory4) -> Self {
        let name = if value.name.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(value.name) })
        };
        let source = if value.source.is_null() {
            None
        } else {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityLegacyMemory<'a> {
    pub memory_kind: ActivityMemoryKind,
    pub address: u64,
    pub bytes: u64,
    pub start: u64,
    pub end: u64,
    pub allocation_pc: u64,
    pub free_pc: u64,
    pub process_id: ProcessId,
    pub device_id: DeviceId,
    pub context_id: ContextId,
    pub name: Option<&'a CStr>,
}

impl<'a> ActivityLegacyMemory<'a> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemoryPool {
    pub operation_type: ActivityMemoryPoolOperationType,
    pub pool_type: ActivityMemoryPoolType,
    pub correlation_id: CorrelationId,
    pub process_id: ProcessId,
    pub device_id: DeviceId,
    pub min_bytes_to_keep: u64,
    pub address: u64,
    pub size: u64,
    pub release_threshold: u64,
    pub timestamp: u64,
    pub utilized_size: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityGraphTrace {
    pub correlation_id: CorrelationId,
    pub start: u64,
    pub end: u64,
    pub device_id: DeviceId,
    pub graph_id: GraphId,
    pub context_id: ContextId,
    pub stream_id: StreamId,
    pub end_device_id: DeviceId,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityDeviceGraphTrace {
    pub device_id: DeviceId,
    pub start: u64,
    pub end: u64,
    pub graph_id: GraphId,
    pub launcher_graph_id: GraphId,
    pub device_launch_mode: DeviceGraphLaunchMode,
    pub context_id: ContextId,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityGraphHostNode {
    pub stream_id: StreamId,
    pub context_id: ContextId,
    pub device_id: DeviceId,
    pub correlation_id: CorrelationId,
    pub graph_id: GraphId,
    pub graph_node_id: GraphNodeId,
    pub process_id: ProcessId,
    pub thread_id: ThreadId,
    pub start: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityHostLaunch {
    pub stream_id: StreamId,
    pub context_id: ContextId,
    pub device_id: DeviceId,
    pub correlation_id: CorrelationId,
    pub process_id: ProcessId,
    pub thread_id: ThreadId,
    pub start: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityComputeEngineContextSwitch {
    pub context_id: ContextId,
    pub timestamp: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityGreenContext {
    pub context_id: ContextId,
    pub parent_context_id: ContextId,
    pub device_id: DeviceId,
    pub num_tpcs: u32,
    pub num_multiprocessors: u16,
    pub logical_tpc_mask: [u32; 32],
    pub logical_tpc_mask_size: u8,
}

impl ActivityGreenContext {
    fn from_raw(value: sys::CUpti_ActivityGreenContext) -> Self {
        Self {
            context_id: ContextId::from(value.contextId),
            parent_context_id: ContextId::from(value.parentContextId),
            device_id: DeviceId::from(value.deviceId),
            num_tpcs: value.numTpcs,
            num_multiprocessors: value.numMultiprocessors,
            logical_tpc_mask: value.logicalTpcMask,
            logical_tpc_mask_size: value.logicalTpcMaskSize,
        }
    }

    pub fn logical_tpc_mask_words(&self) -> &[u32] {
        let len = self
            .logical_tpc_mask_size
            .min(self.logical_tpc_mask.len() as u8) as usize;
        &self.logical_tpc_mask[..len]
    }

    pub fn has_logical_tpc(&self, logical_tpc_id: u32) -> bool {
        let word = logical_tpc_id / 32;
        let bit = logical_tpc_id % 32;
        self.logical_tpc_mask_words()
            .get(word as usize)
            .is_some_and(|mask| (mask & (1u32 << bit)) != 0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityObjectId {
    Unknown,
    Process {
        process_id: ProcessId,
    },
    Thread {
        process_id: ProcessId,
        thread_id: ThreadId,
    },
    Device {
        device_id: DeviceId,
    },
    Context {
        device_id: DeviceId,
        context_id: ContextId,
    },
    Stream {
        device_id: DeviceId,
        context_id: ContextId,
        stream_id: StreamId,
    },
}

impl ActivityObjectId {
    fn from_raw(kind: ActivityObjectKind, value: sys::CUpti_ActivityObjectKindId) -> Self {
        match kind {
            ActivityObjectKind::Process => {
                let value = unsafe { value.pt };
                Self::Process {
                    process_id: ProcessId::from(value.processId),
                }
            }
            ActivityObjectKind::Thread => {
                let value = unsafe { value.pt };
                Self::Thread {
                    process_id: ProcessId::from(value.processId),
                    thread_id: ThreadId::from(value.threadId),
                }
            }
            ActivityObjectKind::Device => {
                let value = unsafe { value.dcs };
                Self::Device {
                    device_id: DeviceId::from(value.deviceId),
                }
            }
            ActivityObjectKind::Context => {
                let value = unsafe { value.dcs };
                Self::Context {
                    device_id: DeviceId::from(value.deviceId),
                    context_id: ContextId::from(value.contextId),
                }
            }
            ActivityObjectKind::Stream => {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityName<'a> {
    pub object_kind: ActivityObjectKind,
    pub object_id: ActivityObjectId,
    pub name: Option<&'a CStr>,
}

impl<'a> ActivityName<'a> {
    fn from_raw(value: sys::CUpti_ActivityName) -> Self {
        let object_kind = ActivityObjectKind::from(value.objectKind);
        let name = if value.name.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(value.name) })
        };
        Self {
            object_kind,
            object_id: ActivityObjectId::from_raw(object_kind, value.objectId),
            name,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMarker<'a> {
    pub flags: ActivityFlags,
    pub timestamp: u64,
    pub id: ActivityMarkerId,
    pub object_kind: ActivityObjectKind,
    pub object_id: ActivityObjectId,
    pub name: Option<&'a CStr>,
    pub domain: Option<&'a CStr>,
}

impl<'a> ActivityMarker<'a> {
    fn from_raw(value: sys::CUpti_ActivityMarker2) -> Self {
        let object_kind = ActivityObjectKind::from(value.objectKind);
        let name = if value.name.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(value.name) })
        };
        let domain = if value.domain.is_null() {
            None
        } else {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityExternalCorrelation {
    pub external_kind: ExternalCorrelationKind,
    pub external_id: ExternalCorrelationId,
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
    unsafe {
        match kind {
            MetricValueKind::Double => MetricValue::Double(value.metricValueDouble),
            MetricValueKind::Uint64 => MetricValue::Uint64(value.metricValueUint64),
            MetricValueKind::Percent => MetricValue::Percent(value.metricValuePercent),
            MetricValueKind::Throughput => MetricValue::Throughput(value.metricValueThroughput),
            MetricValueKind::Int64 => MetricValue::Int64(value.metricValueInt64),
            MetricValueKind::UtilizationLevel => MetricValue::UtilizationLevel(
                MetricValueUtilizationLevel::from(value.metricValueUtilizationLevel),
            ),
            MetricValueKind::NvtxExtendedPayload => {
                MetricValue::NvtxExtendedPayload(value.metricValueNvtxExtendedPayload)
            }
            MetricValueKind::ForceInt => MetricValue::Uint64(value.metricValueUint64),
        }
    }
}

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
            bits: unsafe { value.metricValueUint64 },
        }
    }

    pub fn value_as(self, kind: MetricValueKind) -> MetricValue {
        metric_value_from_raw_kind(
            kind,
            sys::CUpti_MetricValue {
                metricValueUint64: self.bits,
            },
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivityMarkerData {
    pub flags: ActivityFlags,
    pub id: ActivityMarkerId,
    pub payload_kind: MetricValueKind,
    pub payload: MetricValue,
    pub color: u32,
    pub category: u32,
    pub cupti_domain_id: CuptiDomainId,
}

impl ActivityMarkerData {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMetric {
    pub id: MetricId,
    pub value: MetricValueRaw,
    pub correlation_id: CorrelationId,
    pub flags: ActivityFlags,
}

impl ActivityMetric {
    fn from_raw(value: sys::CUpti_ActivityMetric) -> Self {
        Self {
            id: MetricId::from(value.id),
            value: MetricValueRaw::from_raw(value.value),
            correlation_id: CorrelationId::from(value.correlationId),
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMetricInstance {
    pub id: MetricId,
    pub value: MetricValueRaw,
    pub instance: u32,
    pub correlation_id: CorrelationId,
    pub flags: ActivityFlags,
}

impl ActivityMetricInstance {
    fn from_raw(value: sys::CUpti_ActivityMetricInstance) -> Self {
        Self {
            id: MetricId::from(value.id),
            value: MetricValueRaw::from_raw(value.value),
            instance: value.instance,
            correlation_id: CorrelationId::from(value.correlationId),
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityCudaEvent {
    pub correlation_id: CorrelationId,
    pub context_id: ContextId,
    pub stream_id: StreamId,
    pub event_id: CudaEventId,
    pub device_id: DeviceId,
    pub device_timestamp: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityContext {
    pub context_id: ContextId,
    pub device_id: DeviceId,
    pub compute_api_kind: ActivityComputeApiKind,
    pub null_stream_id: StreamId,
    pub parent_context_id: ContextId,
    pub is_green_context: bool,
    pub num_multiprocessors: u16,
    pub cig_mode: ContextCigMode,
    pub process_id: ProcessId,
}

impl ActivityContext {
    fn from_raw(value: sys::CUpti_ActivityContext4) -> Self {
        Self {
            context_id: ContextId::from(value.contextId),
            device_id: DeviceId::from(value.deviceId),
            compute_api_kind: ActivityComputeApiKind::try_from(value.computeApiKind as u32)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityEvent {
    pub id: EventId,
    pub value: u64,
    pub domain_id: EventDomainId,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityEventInstance {
    pub id: EventId,
    pub domain_id: EventDomainId,
    pub instance: u32,
    pub value: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityInstantaneousEvent {
    pub id: EventId,
    pub value: u64,
    pub timestamp: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityInstantaneousEventInstance {
    pub id: EventId,
    pub value: u64,
    pub timestamp: u64,
    pub device_id: DeviceId,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityInstantaneousMetric {
    pub id: MetricId,
    pub value: MetricValueRaw,
    pub timestamp: u64,
    pub device_id: DeviceId,
    pub flags: ActivityFlags,
}

impl ActivityInstantaneousMetric {
    fn from_raw(value: sys::CUpti_ActivityInstantaneousMetric) -> Self {
        Self {
            id: MetricId::from(value.id),
            value: MetricValueRaw::from_raw(value.value),
            timestamp: value.timestamp,
            device_id: DeviceId::from(value.deviceId),
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityInstantaneousMetricInstance {
    pub id: MetricId,
    pub value: MetricValueRaw,
    pub timestamp: u64,
    pub device_id: DeviceId,
    pub flags: ActivityFlags,
    pub instance: u8,
}

impl ActivityInstantaneousMetricInstance {
    fn from_raw(value: sys::CUpti_ActivityInstantaneousMetricInstance) -> Self {
        Self {
            id: MetricId::from(value.id),
            value: MetricValueRaw::from_raw(value.value),
            timestamp: value.timestamp,
            device_id: DeviceId::from(value.deviceId),
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
            instance: value.instance,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityGlobalAccess {
    pub flags: ActivityFlags,
    pub source_locator_id: SourceLocatorId,
    pub correlation_id: CorrelationId,
    pub function_id: FunctionId,
    pub executed: u32,
    pub pc_offset: u64,
    pub threads_executed: u64,
    pub l2_transactions: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityBranch {
    pub source_locator_id: SourceLocatorId,
    pub correlation_id: CorrelationId,
    pub function_id: FunctionId,
    pub pc_offset: u32,
    pub diverged: u32,
    pub threads_executed: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityInstructionExecution {
    pub flags: ActivityFlags,
    pub source_locator_id: SourceLocatorId,
    pub correlation_id: CorrelationId,
    pub function_id: FunctionId,
    pub pc_offset: u32,
    pub threads_executed: u64,
    pub not_predicated_off_threads_executed: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPcSampling {
    pub flags: ActivityFlags,
    pub source_locator_id: SourceLocatorId,
    pub correlation_id: CorrelationId,
    pub function_id: FunctionId,
    pub latency_samples: u32,
    pub samples: u32,
    pub stall_reason: ActivityPcSamplingStallReason,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPcSamplingRecordInfo {
    pub correlation_id: CorrelationId,
    pub total_samples: u64,
    pub dropped_samples: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityUnifiedMemoryCounter {
    pub counter_kind: ActivityUnifiedMemoryCounterKind,
    pub value: u64,
    pub start: u64,
    pub end: u64,
    pub address: u64,
    pub source_id: UnifiedMemoryProcessorId,
    pub destination_id: UnifiedMemoryProcessorId,
    pub stream_id: StreamId,
    pub process_id: ProcessId,
    pub flags: ActivityUnifiedMemoryCounterFlags,
    pub processors: [u64; 5],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityUnifiedMemoryCounterFlags {
    AccessType(ActivityUnifiedMemoryAccessType),
    MigrationCause(ActivityUnifiedMemoryMigrationCause),
    RemoteMapCause(ActivityUnifiedMemoryRemoteMapCause),
    Activity(ActivityFlags),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityDeviceAttributeKind {
    Cupti(DeviceAttribute),
    Cuda(CudaDeviceAttribute),
}

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
            bits: unsafe { value.vUint64 },
        }
    }

    pub fn as_f64(self) -> f64 {
        f64::from_bits(self.bits)
    }

    pub fn as_u32(self) -> u32 {
        self.bits as u32
    }

    pub fn as_u64(self) -> u64 {
        self.bits
    }

    pub fn as_i32(self) -> i32 {
        self.bits as u32 as i32
    }

    pub fn as_i64(self) -> i64 {
        self.bits as i64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityDeviceAttribute {
    pub flags: ActivityFlags,
    pub device_id: DeviceId,
    pub attribute: ActivityDeviceAttributeKind,
    pub value: ActivityDeviceAttributeValue,
}

impl ActivityDeviceAttribute {
    fn from_raw(value: sys::CUpti_ActivityDeviceAttribute) -> Self {
        let flags = ActivityFlags::from_bits_truncate(value.flags as u32);
        let attribute = if flags.contains(ActivityFlags::DEVICE_ATTRIBUTE_CUDEVICE) {
            ActivityDeviceAttributeKind::Cuda(CudaDeviceAttribute::from(unsafe {
                value.attribute.cu
            }))
        } else {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityEnvironmentSpeed {
    pub sm_clock: u32,
    pub memory_clock: u32,
    pub pcie_link_gen: u32,
    pub pcie_link_width: u32,
    pub clocks_throttle_reasons: EnvironmentClocksThrottleReasons,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityEnvironmentPower {
    pub power: u32,
    pub power_limit: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityEnvironmentData {
    Unknown,
    Speed(ActivityEnvironmentSpeed),
    Temperature { gpu_temperature: u32 },
    Power(ActivityEnvironmentPower),
    Cooling { fan_speed: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityEnvironment {
    pub device_id: DeviceId,
    pub timestamp: u64,
    pub environment_kind: ActivityEnvironmentKind,
    pub data: ActivityEnvironmentData,
}

impl ActivityEnvironment {
    fn from_raw(value: sys::CUpti_ActivityEnvironment) -> Self {
        let environment_kind = ActivityEnvironmentKind::from(value.environmentKind);
        let data = unsafe {
            match environment_kind {
                ActivityEnvironmentKind::Speed => {
                    let speed = value.data.speed;
                    ActivityEnvironmentData::Speed(ActivityEnvironmentSpeed {
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
                ActivityEnvironmentKind::Temperature => ActivityEnvironmentData::Temperature {
                    gpu_temperature: value.data.temperature.gpuTemperature,
                },
                ActivityEnvironmentKind::Power => {
                    let power = value.data.power;
                    ActivityEnvironmentData::Power(ActivityEnvironmentPower {
                        power: power.power,
                        power_limit: power.powerLimit,
                    })
                }
                ActivityEnvironmentKind::Cooling => ActivityEnvironmentData::Cooling {
                    fan_speed: value.data.cooling.fanSpeed,
                },
                _ => ActivityEnvironmentData::Unknown,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityDevice<'a> {
    pub flags: ActivityFlags,
    pub global_memory_bandwidth: u64,
    pub global_memory_size: u64,
    pub constant_memory_size: u32,
    pub l2_cache_size: u32,
    pub num_threads_per_warp: u32,
    pub core_clock_rate: u32,
    pub num_memory_copy_engines: u32,
    pub num_multiprocessors: u32,
    pub max_instructions_per_cycle: u32,
    pub max_warps_per_multiprocessor: u32,
    pub max_blocks_per_multiprocessor: u32,
    pub max_shared_memory_per_multiprocessor: u32,
    pub max_registers_per_multiprocessor: u32,
    pub max_registers_per_block: u32,
    pub max_shared_memory_per_block: u32,
    pub max_threads_per_block: u32,
    pub max_block_dim: Dim3,
    pub max_grid_dim: Dim3,
    pub compute_capability_major: u32,
    pub compute_capability_minor: u32,
    pub device_id: DeviceId,
    pub ecc_enabled: bool,
    pub uuid: [u8; 16],
    pub name: Option<&'a CStr>,
    pub is_cuda_visible: bool,
    pub is_mig_enabled: bool,
    pub gpu_instance_id: GpuInstanceId,
    pub compute_instance_id: ComputeInstanceId,
    pub mig_uuid: [u8; 16],
    pub is_non_uniform_memory_access_node: bool,
    pub non_uniform_memory_access_id: NonUniformMemoryAccessId,
    pub num_tpcs: u32,
}

impl<'a> ActivityDevice<'a> {
    fn from_raw(value: sys::CUpti_ActivityDevice6) -> Self {
        let name = if value.name.is_null() {
            None
        } else {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityJit<'a> {
    pub entry_type: ActivityJitEntryType,
    pub operation_type: ActivityJitOperationType,
    pub device_id: DeviceId,
    pub start: u64,
    pub end: u64,
    pub correlation_id: CorrelationId,
    pub operation_correlation_id: JitOperationCorrelationId,
    pub cache_size: u64,
    pub cache_path: Option<&'a CStr>,
    pub process_id: ProcessId,
    pub thread_id: ThreadId,
}

impl<'a> ActivityJit<'a> {
    fn from_raw(value: sys::CUpti_ActivityJit2) -> Self {
        let cache_path = if value.cachePath.is_null() {
            None
        } else {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityNpuId {
    pub index: u32,
    pub domain_id: NpuDomainId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityTopologyDeviceId {
    Unknown,
    Gpu { uuid: [u8; 16] },
    Npu(ActivityNpuId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityNvLink<'a> {
    pub version: u32,
    pub device0_type: DeviceType,
    pub device1_type: DeviceType,
    pub device0_id: ActivityTopologyDeviceId,
    pub device1_id: ActivityTopologyDeviceId,
    pub flags: LinkFlags,
    pub physical_link_count: u32,
    pub device0_ports: Option<&'a [u32]>,
    pub device1_ports: Option<&'a [u32]>,
    pub bandwidth: u64,
    pub nvswitch_connected: bool,
}

impl<'a> ActivityNvLink<'a> {
    fn from_raw(value: sys::CUpti_ActivityNvLink5) -> Self {
        let device0_type = DeviceType::from(value.typeDev0);
        let device1_type = DeviceType::from(value.typeDev1);
        let port_count = value.physicalNvLinkCount as usize;
        let device0_ports = if value.portDev0.is_null() || port_count == 0 {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(value.portDev0, port_count) })
        };
        let device1_ports = if value.portDev1.is_null() || port_count == 0 {
            None
        } else {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPcieGpuAttributes {
    pub uuid: [u8; 16],
    pub peer_devices: [CudaDevice; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPcieBridgeAttributes {
    pub secondary_bus: PcieBusId,
    pub device_id: PcieHardwareDeviceId,
    pub vendor_id: PcieVendorId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityPcieDeviceId {
    Gpu(CudaDevice),
    Bridge(PcieBridgeId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityPcieAttributes {
    Gpu(ActivityPcieGpuAttributes),
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
    ActivityRecordData<'_>,
    ActivityMemoryPoolConfigData,
    ActivityObjectId,
    ActivityUnifiedMemoryCounterFlags,
    ActivityDeviceAttributeKind,
    ActivityEnvironmentData,
    ActivityTopologyDeviceId,
    ActivityPcieDeviceId,
    ActivityPcieAttributes,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPcie {
    pub device_type: PcieDeviceType,
    pub id: ActivityPcieDeviceId,
    pub domain: PcieDomainId,
    pub pcie_generationeration: u16,
    pub link_rate: u16,
    pub link_width: u16,
    pub upstream_bus: u16,
    pub attributes: ActivityPcieAttributes,
}

impl ActivityPcie {
    fn from_raw(value: sys::CUpti_ActivityPcie) -> Self {
        let device_type = PcieDeviceType::from(value.type_);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityInstructionCorrelation {
    pub flags: ActivityFlags,
    pub source_locator_id: SourceLocatorId,
    pub function_id: FunctionId,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOpenMultiProcessing {
    pub event_kind: OpenMultiProcessingEventKind,
    pub version: u32,
    pub thread_id: ThreadId,
    pub start: u64,
    pub end: u64,
    pub process_id: ProcessId,
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

fn optional_cstr<'a>(ptr: *const std::ffi::c_char) -> Option<&'a CStr> {
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(ptr) })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOpenAccBase<'a> {
    pub event_kind: OpenAccEventKind,
    pub parent_construct: OpenAccConstructKind,
    pub version: u32,
    pub implicit: bool,
    pub device_type: OpenAccDeviceTypeId,
    pub device_number: OpenAccDeviceNumber,
    pub thread_id: ThreadId,
    pub asynchronous_id: OpenAccAsynchronousId,
    pub asynchronous_map: u64,
    pub line_number: u32,
    pub end_line_number: u32,
    pub function_line_number: u32,
    pub function_end_line_number: u32,
    pub start: u64,
    pub end: u64,
    pub cuda_device_id: DeviceId,
    pub cuda_context_id: ContextId,
    pub cuda_stream_id: StreamId,
    pub process_id: ProcessId,
    pub cuda_thread_id: ThreadId,
    pub external_id: ExternalCorrelationId,
    pub source_file: Option<&'a CStr>,
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
            external_id: ExternalCorrelationId::from($value.externalId as u64),
            source_file: optional_cstr($value.srcFile),
            function_name: optional_cstr($value.funcName),
        }
    }};
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOpenAccData<'a> {
    pub base: ActivityOpenAccBase<'a>,
    pub bytes: u64,
    pub host_ptr: u64,
    pub device_ptr: u64,
    pub variable_name: Option<&'a CStr>,
}

impl<'a> ActivityOpenAccData<'a> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOpenAccLaunch<'a> {
    pub base: ActivityOpenAccBase<'a>,
    pub num_gangs: u64,
    pub num_workers: u64,
    pub vector_length: u64,
    pub kernel_name: Option<&'a CStr>,
}

impl<'a> ActivityOpenAccLaunch<'a> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOpenAccOther<'a> {
    pub base: ActivityOpenAccBase<'a>,
}

impl<'a> ActivityOpenAccOther<'a> {
    fn from_raw(value: sys::CUpti_ActivityOpenAccOther) -> Self {
        Self {
            base: open_acc_base_from_raw!(value),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPreemption {
    pub preemption_kind: ActivityPreemptionKind,
    pub timestamp: u64,
    pub grid_id: GridId,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityFunction<'a> {
    pub id: FunctionId,
    pub context_id: ContextId,
    pub module_id: ModuleId,
    pub function_index: u32,
    pub name: Option<&'a CStr>,
}

impl<'a> ActivityFunction<'a> {
    fn from_raw(value: sys::CUpti_ActivityFunction) -> Self {
        let name = if value.name.is_null() {
            None
        } else {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityModule<'a> {
    pub context_id: ContextId,
    pub id: ModuleId,
    pub cubin: Option<&'a [u8]>,
}

impl<'a> ActivityModule<'a> {
    fn from_raw(value: sys::CUpti_ActivityModule) -> Self {
        let cubin = if value.cubin.is_null() || value.cubinSize == 0 {
            None
        } else {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivitySharedAccess {
    pub flags: ActivityFlags,
    pub source_locator_id: SourceLocatorId,
    pub correlation_id: CorrelationId,
    pub function_id: FunctionId,
    pub pc_offset: u32,
    pub threads_executed: u64,
    pub shared_transactions: u64,
    pub theoretical_shared_transactions: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityConfidentialComputeRotation {
    pub event_type: ConfidentialComputeRotationEventType,
    pub device_id: DeviceId,
    pub context_id: ContextId,
    pub channel_id: ChannelId,
    pub channel_type: ChannelType,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivitySourceLocator<'a> {
    pub id: SourceLocatorId,
    pub line_number: u32,
    pub file_name: Option<&'a CStr>,
}

impl<'a> ActivitySourceLocator<'a> {
    fn from_raw(value: sys::CUpti_ActivitySourceLocator) -> Self {
        let file_name = if value.fileName.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(value.fileName) })
        };
        Self {
            id: SourceLocatorId::from(value.id),
            line_number: value.lineNumber,
            file_name,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityOverhead {
    pub overhead_kind: ActivityOverheadKind,
    pub object_kind: ActivityObjectKind,
    pub object_id: ActivityObjectId,
    pub start: u64,
    pub end: u64,
    pub correlation_id: CorrelationId,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityKernel<'a> {
    pub name: Option<&'a CStr>,
    pub device_id: DeviceId,
    pub context_id: ContextId,
    pub stream_id: StreamId,
    pub correlation_id: CorrelationId,
    pub grid_id: GridId,
    pub grid: Dim3,
    pub block: Dim3,
    pub cluster: Dim3,
    pub start: u64,
    pub end: u64,
    pub completed: u64,
    pub queued: u64,
    pub submitted: u64,
    pub static_shared_memory: u32,
    pub dynamic_shared_memory: u32,
    pub local_memory_per_thread: u32,
    pub local_memory_total: u64,
    pub registers_per_thread: u16,
    pub shared_memory_config: u8,
    pub shared_memory_executed: u32,
    pub shared_memory_carveout_requested: Option<u8>,
    pub partitioned_global_cache_requested: ActivityPartitionedGlobalCacheConfig,
    pub partitioned_global_cache_executed: ActivityPartitionedGlobalCacheConfig,
    pub launch_type: ActivityLaunchType,
    pub shared_memory_limit_config: FunctionSharedMemoryLimitConfig,
    pub graph_node_id: GraphNodeId,
    pub graph_id: GraphId,
    pub channel_id: ChannelId,
    pub channel_type: ChannelType,
    pub max_potential_cluster_size: u32,
    pub max_active_clusters: u32,
    pub is_device_launched: bool,
    pub priority: i32,
}

impl<'a> ActivityKernel<'a> {
    fn from_raw(value: sys::CUpti_ActivityKernel11) -> Self {
        let name = if value.name.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(value.name) })
        };

        Self {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityCudaDynamicParallelismKernel<'a> {
    pub name: Option<&'a CStr>,
    pub device_id: DeviceId,
    pub context_id: ContextId,
    pub stream_id: StreamId,
    pub correlation_id: CorrelationId,
    pub grid_id: GridId,
    pub parent_grid_id: GridId,
    pub parent_block: Dim3,
    pub grid: Dim3,
    pub block: Dim3,
    pub start: u64,
    pub end: u64,
    pub completed: u64,
    pub queued: u64,
    pub submitted: u64,
    pub static_shared_memory: u32,
    pub dynamic_shared_memory: u32,
    pub local_memory_per_thread: u32,
    pub local_memory_total: u32,
    pub registers_per_thread: u16,
    pub shared_memory_config: u8,
}

impl<'a> ActivityCudaDynamicParallelismKernel<'a> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemoryDecompress {
    pub device_id: DeviceId,
    pub context_id: ContextId,
    pub stream_id: StreamId,
    pub channel_id: ChannelId,
    pub channel_type: ChannelType,
    pub correlation_id: CorrelationId,
    pub number_of_operations: u32,
    pub source_bytes: u64,
    pub start: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemoryCopy {
    pub copy_kind: ActivityMemoryCopyKind,
    pub source_kind: ActivityMemoryKind,
    pub destination_kind: ActivityMemoryKind,
    pub flags: ActivityFlags,
    pub bytes: u64,
    pub start: u64,
    pub end: u64,
    pub device_id: DeviceId,
    pub context_id: ContextId,
    pub stream_id: StreamId,
    pub correlation_id: CorrelationId,
    pub runtime_correlation_id: CorrelationId,
    pub graph_node_id: GraphNodeId,
    pub graph_id: GraphId,
    pub channel_id: ChannelId,
    pub channel_type: ChannelType,
    pub is_device_launched: bool,
    pub copy_count: u64,
}

impl ActivityMemoryCopy {
    fn from_raw(value: sys::CUpti_ActivityMemcpy6) -> Self {
        Self {
            copy_kind: memory_copy_kind_from_raw(value.copyKind),
            source_kind: memory_kind_from_raw(value.srcKind as u32),
            destination_kind: memory_kind_from_raw(value.dstKind as u32),
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
            bytes: value.bytes,
            start: value.start,
            end: value.end,
            device_id: DeviceId::from(value.deviceId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            correlation_id: CorrelationId::from(value.correlationId),
            runtime_correlation_id: CorrelationId::from(value.runtimeCorrelationId),
            graph_node_id: GraphNodeId::from(value.graphNodeId),
            graph_id: GraphId::from(value.graphId),
            channel_id: ChannelId::from(value.channelID),
            channel_type: ChannelType::from(value.channelType),
            is_device_launched: value.isDeviceLaunched != 0,
            copy_count: value.copyCount,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPeerMemoryCopy {
    pub copy_kind: ActivityMemoryCopyKind,
    pub source_kind: ActivityMemoryKind,
    pub destination_kind: ActivityMemoryKind,
    pub flags: ActivityFlags,
    pub bytes: u64,
    pub start: u64,
    pub end: u64,
    pub device_id: DeviceId,
    pub context_id: ContextId,
    pub stream_id: StreamId,
    pub source_device_id: DeviceId,
    pub source_context_id: ContextId,
    pub destination_device_id: DeviceId,
    pub destination_context_id: ContextId,
    pub correlation_id: CorrelationId,
    pub graph_node_id: GraphNodeId,
    pub graph_id: GraphId,
    pub channel_id: ChannelId,
    pub channel_type: ChannelType,
}

impl ActivityPeerMemoryCopy {
    fn from_raw(value: sys::CUpti_ActivityMemcpyPtoP4) -> Self {
        Self {
            copy_kind: memory_copy_kind_from_raw(value.copyKind),
            source_kind: memory_kind_from_raw(value.srcKind as u32),
            destination_kind: memory_kind_from_raw(value.dstKind as u32),
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemorySet {
    pub value: u32,
    pub bytes: u64,
    pub start: u64,
    pub end: u64,
    pub device_id: DeviceId,
    pub context_id: ContextId,
    pub stream_id: StreamId,
    pub correlation_id: CorrelationId,
    pub flags: ActivityFlags,
    pub memory_kind: ActivityMemoryKind,
    pub graph_node_id: GraphNodeId,
    pub graph_id: GraphId,
    pub channel_id: ChannelId,
    pub channel_type: ChannelType,
    pub is_device_launched: bool,
}

impl ActivityMemorySet {
    fn from_raw(value: sys::CUpti_ActivityMemset4) -> Self {
        Self {
            value: value.value,
            bytes: value.bytes,
            start: value.start,
            end: value.end,
            device_id: DeviceId::from(value.deviceId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            correlation_id: CorrelationId::from(value.correlationId),
            flags: ActivityFlags::from_bits_truncate(value.flags as u32),
            memory_kind: memory_kind_from_raw(value.memoryKind as u32),
            graph_node_id: GraphNodeId::from(value.graphNodeId),
            graph_id: GraphId::from(value.graphId),
            channel_id: ChannelId::from(value.channelID),
            channel_type: ChannelType::from(value.channelType),
            is_device_launched: value.isDeviceLaunched != 0,
        }
    }
}

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

        let status = unsafe {
            sys::cuptiActivityGetNextRecord(self.buffer, self.valid_size_bytes, &mut self.current)
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

#[derive(Debug, Clone, Copy)]
pub struct ActivityBufferCallbackConfig {
    pub buffer_size: usize,
    pub max_num_records: usize,
}

impl ActivityBufferCallbackConfig {
    pub const fn new(buffer_size: usize) -> Self {
        Self {
            buffer_size,
            max_num_records: 0,
        }
    }

    pub fn with_max_num_records(mut self, max_num_records: usize) -> Self {
        self.max_num_records = max_num_records;
        self
    }
}

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

    pub fn enable(&self, kind: ActivityKind) -> Result<()> {
        enable_global(kind)
    }

    pub fn disable(&self, kind: ActivityKind) -> Result<()> {
        disable_global(kind)
    }

    pub fn flush_all(&self, flag: ActivityFlushFlag) -> Result<()> {
        flush_all(flag)
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self
            .buffers
            .lock()
            .map_err(|_| Error::LockPoisoned {
                name: "activity collector buffers".into(),
            })?
            .len())
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Removes all completed buffers from this collector and returns them.
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

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ActivityFlushTarget<'a> {
    Global,
    Context(&'a Context),
    Stream {
        context: &'a Context,
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
        unsafe {
            try_ffi!(sys::cuptiActivityConfigurePCSampling(
                self.as_raw(),
                &mut config,
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
        unsafe {
            try_ffi!(sys::cuptiActivityGetNumDroppedRecords(
                self.as_raw(),
                stream_id,
                &mut dropped,
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
    unsafe {
        try_ffi!(sys::cuptiActivityDisable(kind.into()))?;
    }
    Ok(())
}

/// Globally enables collection of an activity kind and asks CUPTI to emit existing resource records when supported.
///
/// Register activity buffer callbacks before calling this; otherwise CUPTI can enable the kind without delivering records. Resource dumping is mainly useful for late attach to existing devices, contexts, streams, NVLink, and PCIe records.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
/// - Returns [`crate::error::Status::NotCompatible`] if the activity kind cannot be enabled in the current configuration.
/// - Returns [`crate::error::Status::Unknown`] if CUPTI reports an internal error.
pub fn enable_global_and_dump(kind: ActivityKind) -> Result<()> {
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
    let mut config = config.as_raw_mut();
    let subscriber = subscriber.as_raw()?;
    unsafe {
        try_ffi!(sys::cuptiActivityEnable_v2(
            subscriber,
            kind.into(),
            &mut config,
        ))?;
    }
    Ok(())
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
    let mut config = config.as_raw_mut();
    let subscriber = subscriber.as_raw()?;
    unsafe {
        try_ffi!(sys::cuptiActivityDisable_v2(
            subscriber,
            kind.into(),
            &mut config,
        ))?;
    }
    Ok(())
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
/// - Returns [`crate::error::Status::Unknown`] if CUPTI reports an internal error.
pub fn enable_and_dump(
    subscriber: &Subscriber,
    kind: ActivityKind,
    config: &mut ActivityConfig,
) -> Result<()> {
    let mut config = config.as_raw_mut();
    let subscriber = subscriber.as_raw()?;
    unsafe {
        try_ffi!(sys::cuptiActivityEnableAndDump_v2(
            subscriber,
            kind.into(),
            &mut config,
        ))?;
    }
    Ok(())
}

pub fn enabled_kinds(subscriber: &Subscriber) -> Result<Vec<ActivityKind>> {
    let mut buffer_size = 0u32;
    let mut enabled_count = 0u32;
    let subscriber = subscriber.as_raw()?;
    unsafe {
        try_ffi!(sys::cuptiActivityGetEnabledKinds(
            subscriber,
            ptr::null_mut(),
            &mut buffer_size,
            &mut enabled_count,
        ))?;
    }

    if enabled_count == 0 {
        return Ok(Vec::new());
    }

    let len = to_usize(enabled_count, "enabled_activity_kinds_count")?;
    let mut kinds = vec![sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INVALID; len];
    let mut buffer_size = enabled_count;
    let mut enabled_count = 0u32;
    unsafe {
        try_ffi!(sys::cuptiActivityGetEnabledKinds(
            subscriber,
            kinds.as_mut_ptr(),
            &mut buffer_size,
            &mut enabled_count,
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
    unsafe {
        try_ffi!(sys::cuptiActivityGetStructSize(
            kind.into(),
            version,
            &mut size
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
    unsafe {
        try_ffi!(sys::cuptiActivityEnableAllSyncRecords(enable as u8))?;
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
    unsafe {
        try_ffi!(sys::cuptiActivityEnableAllocationSource(enable as u8))?;
    }
    Ok(())
}

/// Controls whether CUDA event activity records include device timestamps.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn enable_cuda_event_device_timestamulti_process_service(enable: bool) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuptiActivityEnableCudaEventDeviceTimestamps(
            enable as u8
        ))?;
    }
    Ok(())
}

/// Controls whether graph activity collection includes device graph trace records.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn enable_device_graph(enable: bool) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuptiActivityEnableDeviceGraph(enable as u8))?;
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
    unsafe {
        try_ffi!(sys::cuptiActivityEnableDriverApi(
            callback_id.as_raw(),
            enable as u8
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
    unsafe {
        try_ffi!(sys::cuptiActivityEnableRuntimeApi(
            callback_id.as_raw(),
            enable as u8
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
    unsafe {
        try_ffi!(sys::cuptiActivityEnableHWTrace(enable as u8))?;
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
    unsafe {
        try_ffi!(sys::cuptiActivityEnableLatencyTimestamps(enable as u8))?;
    }
    Ok(())
}

/// Controls whether kernel activity records include launch attributes.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn enable_launch_attributes(enable: bool) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuptiActivityEnableLaunchAttributes(enable as u8))?;
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
/// - Returns [`crate::error::Status::Unknown`] if CUPTI reports an internal error.
pub fn flush(target: ActivityFlushTarget<'_>, flag: ActivityFlushFlag) -> Result<()> {
    let (context, stream_id) = target.context_and_stream_id()?;
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
/// - Returns [`crate::error::Status::Unknown`] if CUPTI reports an internal error.
pub fn flush_all(flag: ActivityFlushFlag) -> Result<()> {
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
    unsafe {
        try_ffi!(sys::cuptiActivityFlushPeriod(time))?;
    }
    Ok(())
}

fn read_attribute(attr: ActivityAttribute, value_size: *mut u64, value: *mut ()) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuptiActivityGetAttribute(
            attr.into(),
            value_size,
            value as _,
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
    unsafe {
        try_ffi!(sys::cuptiActivityGetAttribute_v2(
            subscriber,
            attr.into(),
            value_size,
            value as _,
        ))?;
    }
    Ok(())
}

pub fn global_attribute_value(attr: ActivityAttribute) -> Result<ActivityAttributeValue> {
    attribute_value_with(read_attribute, attr)
}

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
    #[allow(deprecated)]
    match attr {
        ActivityAttribute::DeviceBufferSize
        | ActivityAttribute::DeviceBufferSizeCudaDynamicParallelism
        | ActivityAttribute::ProfilingSemaphorePoolSize
        | ActivityAttribute::DeviceBufferSizeDeviceGraphs => {
            Ok(ActivityAttributeValue::Bytes(read_usize(&mut read, attr)?))
        }
        ActivityAttribute::DeviceBufferPoolLimit
        | ActivityAttribute::ProfilingSemaphorePoolLimit
        | ActivityAttribute::DeviceBufferPreAllocateValue
        | ActivityAttribute::ProfilingSemaphorePreAllocateValue => {
            Ok(ActivityAttributeValue::Count(read_usize(&mut read, attr)?))
        }
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
            read_u8(&mut read, attr)? != 0,
        )),
        ActivityAttribute::MultipleSubscriberState => {
            Ok(ActivityAttributeValue::MultipleSubscriberState(
                MultipleSubscriberState::from(read_u8(&mut read, attr)?),
            ))
        }
        ActivityAttribute::ThreadIdType => {
            let raw = read_thread_id_type(&mut read, attr)?;
            Ok(ActivityAttributeValue::ThreadIdType(
                ActivityThreadIdType::from(raw),
            ))
        }
        ActivityAttribute::TimestampCallback | ActivityAttribute::DeviceBufferForceInt => {
            Err(Error::InvalidAttribute {
                name: format!("{attr:?}"),
            })
        }
    }
}

fn read_usize(
    read: &mut impl FnMut(ActivityAttribute, *mut u64, *mut ()) -> Result<()>,
    attr: ActivityAttribute,
) -> Result<usize> {
    let mut value = MaybeUninit::<usize>::uninit();
    let mut value_size = to_u64(size_of::<usize>(), "activity_attribute_size")?;
    read(attr, &mut value_size, value.as_mut_ptr().cast())?;
    Ok(unsafe { value.assume_init() })
}

fn read_u8(
    read: &mut impl FnMut(ActivityAttribute, *mut u64, *mut ()) -> Result<()>,
    attr: ActivityAttribute,
) -> Result<u8> {
    let mut value = MaybeUninit::<u8>::uninit();
    let mut value_size = to_u64(size_of::<u8>(), "activity_attribute_size")?;
    read(attr, &mut value_size, value.as_mut_ptr().cast())?;
    Ok(unsafe { value.assume_init() })
}

fn read_thread_id_type(
    read: &mut impl FnMut(ActivityAttribute, *mut u64, *mut ()) -> Result<()>,
    attr: ActivityAttribute,
) -> Result<sys::CUpti_ActivityThreadIdType> {
    let mut value = MaybeUninit::<sys::CUpti_ActivityThreadIdType>::uninit();
    let mut value_size = to_u64(
        size_of::<sys::CUpti_ActivityThreadIdType>(),
        "activity_attribute_size",
    )?;
    read(attr, &mut value_size, value.as_mut_ptr().cast())?;
    Ok(unsafe { value.assume_init() })
}

/// Returns whether a CUPTI tracing session is currently active in the process.
///
/// This query does not initialize CUPTI on its own.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
/// - Returns [`crate::error::Status::Unknown`] if CUPTI reports an internal error.
pub fn is_tracing_session_running() -> Result<bool> {
    let mut running = 0u8;
    unsafe {
        try_ffi!(sys::cuptiIsTracingSessionRunning(&mut running))?;
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
    unsafe {
        try_ffi!(sys::cuptiActivityPopExternalCorrelationId(
            kind.into(),
            &mut last_id,
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
    unsafe {
        try_ffi!(sys::cuptiActivityPushExternalCorrelationId(
            kind.into(),
            id.as_raw()
        ))?;
    }
    Ok(())
}

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
        unsafe {
            *buffer = ptr::null_mut();
            *size = 0;
            *max_num_records = 0;
        }
        return;
    };

    if buffer_size == 0 {
        unsafe {
            *buffer = ptr::null_mut();
            *size = 0;
            *max_num_records = 0;
        }
        return;
    }

    let Ok(layout) = activity_buffer_layout(buffer_size) else {
        unsafe {
            *buffer = ptr::null_mut();
            *size = 0;
            *max_num_records = 0;
        }
        return;
    };
    let allocation_ptr = unsafe { alloc_zeroed(layout) };
    if allocation_ptr.is_null() {
        handle_alloc_error(layout);
    }

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
    let valid_size = usize::try_from(valid_size)
        .ok()
        .filter(|valid_size| *valid_size <= size)
        .unwrap_or(size);

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
        unsafe {
            dealloc(self.buffer, layout);
        }
    }
}

fn activity_buffer_layout(size: usize) -> std::result::Result<Layout, std::alloc::LayoutError> {
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
    unsafe {
        try_ffi!(sys::cuptiActivitySetAttribute(
            attr.into(),
            value_size,
            value as _,
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
    unsafe {
        try_ffi!(sys::cuptiActivitySetAttribute_v2(
            subscriber,
            attr.into(),
            value_size,
            value as _,
        ))?;
    }
    Ok(())
}

pub fn set_global_attribute_value(
    attr: ActivityAttribute,
    value: ActivityAttributeSetting,
) -> Result<()> {
    set_attribute_value_with(set_attribute, attr, value)
}

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
    #[allow(deprecated)]
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
            write(attr, &mut value_size, (&mut value as *mut usize).cast())
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
            let mut value = value as u8;
            let mut value_size = to_u64(size_of::<u8>(), "activity_attribute_size")?;
            write(attr, &mut value_size, (&mut value as *mut u8).cast())
        }
        (ActivityAttribute::ThreadIdType, ActivityAttributeSetting::ThreadIdType(value)) => {
            let mut value = sys::CUpti_ActivityThreadIdType::from(value);
            let mut value_size = to_u64(
                size_of::<sys::CUpti_ActivityThreadIdType>(),
                "activity_attribute_size",
            )?;
            write(
                attr,
                &mut value_size,
                (&mut value as *mut sys::CUpti_ActivityThreadIdType).cast(),
            )
        }
        _ => Err(Error::InvalidAttribute {
            name: format!("{attr:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
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
    fn activity_buffer_request_returns_eight_byte_aligned_buffer() {
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
    fn activity_buffer_complete_restores_state_and_frees_after_callback_panic() {
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
    fn activity_collector_copies_completed_buffers() -> Result<()> {
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
    fn activity_collector_drop_only_clears_owned_registration() -> Result<()> {
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
    fn activity_record_decodes_memory_decompress_record() {
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
            reserved0: std::ptr::null_mut(),
            start: 8,
            end: 9,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMemDecompress).cast(),
            _marker: PhantomData,
        };

        assert_eq!(record.kind(), ActivityKind::MemoryDecompress);
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
            ActivityRecordData::MemoryDecompress(ActivityMemoryDecompress {
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
    fn activity_record_rejects_memory_decompress_for_other_kinds() {
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
            ActivityRecordData::Unsupported {
                kind: ActivityKind::Invalid
            }
        );
    }

    #[test]
    fn activity_record_decodes_kernel_record() {
        let name = std::ffi::CString::new("kernel").unwrap();
        let mut raw = sys::CUpti_ActivityKernel11 {
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
            ptr: (&raw as *const sys::CUpti_ActivityKernel11).cast(),
            _marker: PhantomData,
        };

        let kernel = record.kernel().unwrap();
        assert_eq!(record.kind(), ActivityKind::ConcurrentKernel);
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
        assert!(matches!(record.decode(), ActivityRecordData::Kernel(_)));
    }

    #[test]
    fn activity_record_decodes_legacy_activity_records() {
        let name = std::ffi::CString::new("legacy").unwrap();
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
            ActivityRecordData::LegacyMemory(_)
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
            reserved0: std::ptr::null_mut(),
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
            ActivityRecordData::PeerMemoryCopy(decoded_memory_copy)
        );

        let kernel_name = std::ffi::CString::new("child").unwrap();
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
            ActivityRecordData::CudaDynamicParallelismKernel(_)
        ));
    }

    #[test]
    fn activity_record_decodes_api_record() {
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
        assert_eq!(record.decode(), ActivityRecordData::Api(api));
    }

    #[test]
    fn activity_record_decodes_stream_record() {
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
        assert_eq!(record.decode(), ActivityRecordData::Stream(stream));
    }

    #[test]
    fn activity_record_decodes_synchronization_record() {
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
            ActivityRecordData::Synchronization(synchronization)
        );
    }

    #[test]
    fn activity_record_decodes_memory_record() {
        let name = std::ffi::CString::new("allocation").unwrap();
        let source = std::ffi::CString::new("libcuda.so").unwrap();
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
            ActivityMemoryPoolConfigData::Local {
                size: 12,
                release_threshold: 11,
                utilized_size: 13,
            }
        );
        assert_eq!(memory.source.unwrap().to_str().unwrap(), "libcuda.so");
        assert!(matches!(record.decode(), ActivityRecordData::Memory(_)));
    }

    #[test]
    fn activity_record_decodes_imported_memory_pool_config() {
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
            name: std::ptr::null(),
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
            source: std::ptr::null(),
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMemory4).cast(),
            _marker: PhantomData,
        };

        let memory = record.memory().unwrap();
        assert_eq!(
            memory.memory_pool_config.data,
            ActivityMemoryPoolConfigData::Imported {
                process_id: ProcessId::from(12u64),
            }
        );
    }

    #[test]
    fn activity_record_decodes_memory_pool_record() {
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
        assert_eq!(record.decode(), ActivityRecordData::MemoryPool(memory_pool));
    }

    #[test]
    fn activity_record_decodes_graph_trace_record() {
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
        assert_eq!(record.decode(), ActivityRecordData::GraphTrace(graph_trace));
    }

    #[test]
    fn activity_record_decodes_device_graph_trace_record() {
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
            ActivityRecordData::DeviceGraphTrace(device_graph_trace)
        );
    }

    #[test]
    fn activity_record_decodes_graph_host_node_record() {
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
            ActivityRecordData::GraphHostNode(graph_host_node)
        );
    }

    #[test]
    fn activity_record_decodes_host_launch_record() {
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
        assert_eq!(record.decode(), ActivityRecordData::HostLaunch(host_launch));
    }

    #[test]
    fn activity_record_decodes_compute_engine_context_switch_record() {
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
            ActivityRecordData::ComputeEngineContextSwitch(context_switch)
        );
    }

    #[test]
    fn activity_record_decodes_green_context_record() {
        let mut logical_tpc_mask = [0u32; 32];
        logical_tpc_mask[0] = 0b10;
        logical_tpc_mask[1] = 0b1000;
        let raw = sys::CUpti_ActivityGreenContext {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_GREEN_CONTEXT,
            contextId: 1,
            parentContextId: 2,
            deviceId: 3,
            numTpcs: 4,
            numMultiprocessors: 5,
            logicalTpcMaskSize: 2,
            padding: 0,
            logicalTpcMask: logical_tpc_mask,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityGreenContext).cast(),
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
            ActivityRecordData::GreenContext(green_context)
        );
    }

    #[test]
    fn activity_record_decodes_preemption_function_module_and_shared_records() {
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
            ActivityRecordData::Preemption(decoded_preemption)
        );

        let name = std::ffi::CString::new("kernel").unwrap();
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
            ActivityRecordData::Function(_)
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
            ActivityRecordData::Module(_)
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
            ActivityRecordData::SharedAccess(decoded_shared)
        );
    }

    #[test]
    fn activity_record_decodes_confidential_compute_rotation_record() {
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
            ActivityRecordData::ConfidentialComputeRotation(decoded)
        );
    }

    #[test]
    fn activity_record_decodes_name_record() {
        let name = std::ffi::CString::new("stream-name").unwrap();
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
        assert!(matches!(record.decode(), ActivityRecordData::Name(_)));
    }

    #[test]
    fn activity_record_decodes_marker_record() {
        let name = std::ffi::CString::new("range").unwrap();
        let domain = std::ffi::CString::new("domain").unwrap();
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
        assert!(matches!(record.decode(), ActivityRecordData::Marker(_)));
    }

    #[test]
    fn activity_record_decodes_external_correlation_record() {
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
            ActivityRecordData::ExternalCorrelation(decoded)
        );
    }

    #[test]
    fn activity_record_decodes_marker_data_record() {
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
        let decoded = ActivityMarkerData {
            flags: ActivityFlags::MARKER_END,
            id: ActivityMarkerId::from(1),
            payload_kind: MetricValueKind::Uint64,
            payload: MetricValue::Uint64(2),
            color: 3,
            category: 4,
            cupti_domain_id: CuptiDomainId::from(5),
        };

        assert_eq!(record.marker_data(), Some(decoded));
        assert_eq!(record.decode(), ActivityRecordData::MarkerData(decoded));
    }

    #[test]
    fn activity_record_decodes_cuda_event_record() {
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
        assert_eq!(record.decode(), ActivityRecordData::CudaEvent(decoded));
    }

    #[test]
    fn activity_record_decodes_context_record() {
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
        assert_eq!(record.decode(), ActivityRecordData::Context(decoded));
    }

    #[test]
    fn activity_record_decodes_event_record() {
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
        assert_eq!(record.decode(), ActivityRecordData::Event(decoded));
    }

    #[test]
    fn activity_record_decodes_event_instance_record() {
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
        assert_eq!(record.decode(), ActivityRecordData::EventInstance(decoded));
    }

    #[test]
    fn activity_record_decodes_metric_record() {
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
        assert_eq!(record.decode(), ActivityRecordData::Metric(decoded));
    }

    #[test]
    fn activity_record_decodes_metric_instance_record() {
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
        assert_eq!(record.decode(), ActivityRecordData::MetricInstance(decoded));
    }

    #[test]
    fn activity_record_decodes_instantaneous_records() {
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
            ActivityRecordData::InstantaneousEvent(decoded_event)
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
            ActivityRecordData::InstantaneousMetricInstance(decoded_metric)
        );
    }

    #[test]
    fn activity_record_decodes_source_level_records() {
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
            ActivityRecordData::GlobalAccess(decoded_global)
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
            ActivityRecordData::Branch(decoded_branch)
        );
    }

    #[test]
    fn activity_record_decodes_sampling_and_unified_memory_records() {
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
            ActivityRecordData::InstructionExecution(decoded_instruction)
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
            ActivityRecordData::PcSampling(decoded_sampling)
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
            ActivityRecordData::PcSamplingRecordInfo(decoded_info)
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
            ActivityRecordData::UnifiedMemoryCounter(decoded_unified)
        );
    }

    #[test]
    fn activity_record_decodes_device_attribute_records() {
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
            ActivityRecordData::DeviceAttribute(decoded_cupti)
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
            ActivityRecordData::DeviceAttribute(_)
        ));
    }

    #[test]
    fn activity_record_decodes_environment_records() {
        let speed = sys::CUpti_ActivityEnvironment {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_ENVIRONMENT,
            deviceId: 1,
            timestamp: 2,
            environmentKind:
                sys::CUpti_ActivityEnvironmentKind::CUPTI_ACTIVITY_ENVIRONMENT_SPEED,
            data: sys::CUpti_ActivityEnvironment__bindgen_ty_1 {
                speed: sys::CUpti_ActivityEnvironment__bindgen_ty_1__bindgen_ty_1 {
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
            data: ActivityEnvironmentData::Speed(ActivityEnvironmentSpeed {
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
            ActivityRecordData::Environment(decoded_speed)
        );

        let power = sys::CUpti_ActivityEnvironment {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_ENVIRONMENT,
            deviceId: 7,
            timestamp: 8,
            environmentKind: sys::CUpti_ActivityEnvironmentKind::CUPTI_ACTIVITY_ENVIRONMENT_POWER,
            data: sys::CUpti_ActivityEnvironment__bindgen_ty_1 {
                power: sys::CUpti_ActivityEnvironment__bindgen_ty_1__bindgen_ty_3 {
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
            data: ActivityEnvironmentData::Power(ActivityEnvironmentPower {
                power: 9,
                power_limit: 10,
            }),
        };

        assert_eq!(power_record.environment(), Some(decoded_power));
        assert_eq!(
            power_record.decode(),
            ActivityRecordData::Environment(decoded_power)
        );
    }

    #[test]
    fn activity_record_decodes_device_record() {
        let name = std::ffi::CString::new("gpu").unwrap();
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
        assert!(matches!(record.decode(), ActivityRecordData::Device(_)));
    }

    #[test]
    fn activity_record_decodes_jit_record() {
        let cache_path = std::ffi::CString::new("/tmp/cache").unwrap();
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
        assert_eq!(record.decode(), ActivityRecordData::Jit(decoded));
    }

    #[test]
    fn activity_record_decodes_nvlink_record() {
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
                npu: sys::CUpti_ActivityNvLink5__bindgen_ty_2__bindgen_ty_1 {
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
        assert!(matches!(record.decode(), ActivityRecordData::NvLink(_)));
    }

    #[test]
    fn activity_record_decodes_pcie_records() {
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
                gpuAttr: sys::CUpti_ActivityPcie__bindgen_ty_2__bindgen_ty_1 {
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
        assert_eq!(gpu_record.decode(), ActivityRecordData::Pcie(gpu_decoded));

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
                bridgeAttr: sys::CUpti_ActivityPcie__bindgen_ty_2__bindgen_ty_2 {
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
            ActivityRecordData::Pcie(bridge_decoded)
        );
    }

    #[test]
    fn activity_record_decodes_instruction_correlation_record() {
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
            ActivityRecordData::InstructionCorrelation(decoded)
        );
    }

    #[test]
    fn activity_record_decodes_open_multi_processing_record() {
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
            ActivityRecordData::OpenMultiProcessing(decoded)
        );
    }

    #[test]
    fn activity_record_decodes_open_acc_records() {
        let source_file = std::ffi::CString::new("main.c").unwrap();
        let function_name = std::ffi::CString::new("step").unwrap();
        let variable_name = std::ffi::CString::new("x").unwrap();
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
            ActivityRecordData::OpenAccData(_)
        ));

        let kernel_name = std::ffi::CString::new("kernel").unwrap();
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
            ActivityRecordData::OpenAccLaunch(_)
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
            ActivityRecordData::OpenAccOther(_)
        ));
    }

    #[test]
    fn activity_record_decodes_source_locator_record() {
        let file_name = std::ffi::CString::new("kernel.cu").unwrap();
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
            ActivityRecordData::SourceLocator(_)
        ));
    }

    #[test]
    fn activity_record_decodes_overhead_record() {
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
        assert_eq!(record.decode(), ActivityRecordData::Overhead(decoded));
    }

    #[test]
    fn activity_record_decodes_memory_copy_record() {
        let raw = sys::CUpti_ActivityMemcpy6 {
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
            reserved0: std::ptr::null_mut(),
            graphNodeId: 6,
            graphId: 7,
            channelID: 8,
            channelType: sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY,
            isDeviceLaunched: 1,
            pad2: [0; 3],
            copyCount: 2,
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMemcpy6).cast(),
            _marker: PhantomData,
        };

        assert_eq!(record.memory_decompress(), None);
        assert_eq!(
            record.memory_copy(),
            Some(ActivityMemoryCopy {
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
            ActivityRecordData::MemoryCopy(ActivityMemoryCopy {
                copy_kind: ActivityMemoryCopyKind::HostToDevice,
                bytes: 1024,
                ..
            })
        ));
    }

    #[test]
    fn activity_record_decodes_memory_set_record() {
        let raw = sys::CUpti_ActivityMemset4 {
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
            reserved0: std::ptr::null_mut(),
            graphNodeId: 5,
            graphId: 6,
            channelID: 7,
            channelType: sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_COMPUTE,
            isDeviceLaunched: 0,
            pad2: [0; 3],
        };
        let record = ActivityRecord {
            ptr: (&raw as *const sys::CUpti_ActivityMemset4).cast(),
            _marker: PhantomData,
        };

        assert_eq!(record.memory_copy(), None);
        assert_eq!(
            record.memory_set(),
            Some(ActivityMemorySet {
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
            ActivityRecordData::MemorySet(ActivityMemorySet {
                value: 255,
                bytes: 2048,
                ..
            })
        ));
    }

    #[test]
    fn typed_attribute_setter_rejects_mismatched_value_kind_before_ffi() {
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
    fn typed_attribute_setter_writes_size_t_for_byte_attributes() -> Result<()> {
        let mut observed = None;
        set_attribute_value_with(
            |attr, value_size, value| {
                let value_size = unsafe { *value_size };
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
    fn typed_attribute_setter_writes_u8_for_bool_attributes() -> Result<()> {
        let mut observed = None;
        set_attribute_value_with(
            |attr, value_size, value| {
                let value_size = unsafe { *value_size };
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
