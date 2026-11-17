use std::{ffi::c_void, marker::PhantomData, mem::size_of, ptr};

use super::{
    device::{
        ActivityDevice, ActivityDeviceAttribute, ActivityEnvironment, ActivityNvLink, ActivityPcie,
    },
    kernel::{
        ActivityBranch, ActivityCudaDynamicParallelismKernel, ActivityFunction,
        ActivityGlobalAccess, ActivityInstructionCorrelation, ActivityInstructionExecution,
        ActivityJit, ActivityKernel, ActivityModule, ActivityPcSampling,
        ActivityPcSamplingRecordInfo, ActivitySharedAccess, ActivitySourceLocator,
    },
    memory::{
        ActivityLegacyMemory, ActivityMemory, ActivityMemoryCopy, ActivityMemoryDecompress,
        ActivityMemoryPool, ActivityMemorySet, ActivityPeerMemoryCopy,
        ActivityUnifiedMemoryCounter,
    },
    trace::{
        ActivityApi, ActivityComputeEngineContextSwitch, ActivityConfidentialComputeRotation,
        ActivityContext, ActivityCudaEvent, ActivityDeviceGraphTrace, ActivityExternalCorrelation,
        ActivityGraphHostNode, ActivityGraphTrace, ActivityGreenContext, ActivityHostLaunch,
        ActivityMarker, ActivityMarkerPayload, ActivityName, ActivityOpenAccLaunch,
        ActivityOpenAccOther, ActivityOpenAccTransfer, ActivityOpenMultiProcessing,
        ActivityOverhead, ActivityPreemption, ActivityStream, ActivitySynchronization,
    },
};
use crate::{
    error::{Error, Result},
    sys,
    types::ActivityKind,
};

/// Borrowed CUPTI activity record with kind-checked payload decoders.
///
/// Copies retain the same borrow lifetime.
/// Unrecognized kinds remain available through [`Self::raw_kind`].
/// Records from a buffer that may hold user-defined records decode to no payload, since their fields follow a layout chosen at run time.
#[derive(Debug, Clone, Copy)]
pub struct ActivityRecord<'a> {
    ptr: *const sys::CUpti_Activity,
    // Bytes from `ptr` to the end of the valid buffer, which bounds every decoder's read.
    len: usize,
    // Whether the record has its fixed native layout, rather than a user-defined one.
    full_layout: bool,
    _marker: PhantomData<&'a sys::CUpti_Activity>,
}

impl ActivityRecord<'_> {
    /// Returns the record kind, or `None` for kinds newer than these bindings.
    #[must_use]
    pub fn kind(self) -> Option<ActivityKind> {
        ActivityKind::try_from(self.raw_kind()).ok()
    }

    /// Returns the raw activity-kind value of the record header.
    #[must_use]
    pub fn raw_kind(self) -> u32 {
        // SAFETY: every CUPTI record, including a user-defined one, starts with a `u32` kind header, which the constructors check lies within `self.len` bytes borrowed for `'a`; `read_unaligned` makes no alignment assumption.
        unsafe { ptr::read_unaligned(self.ptr.cast::<u32>()) }
    }

    // Whether the record matched the decoder's kinds, has the fixed native layout, and covers `T`.
    fn has_layout<T>(self, kind_matches: bool) -> bool {
        kind_matches && self.full_layout && self.len >= size_of::<T>()
    }

    // Collects the `malloc` allocations CUPTI leaves to the client in this record, which nothing else frees.
    pub(crate) fn client_allocations(self, pointers: &mut Vec<*mut c_void>) {
        match self.kind() {
            Some(ActivityKind::Device) if self.has_layout::<sys::CUpti_ActivityDevice6>(true) => {
                // SAFETY: `has_layout` checked that CUPTI wrote a full record of this type within `self.len` bytes; `read_unaligned` makes no alignment assumption.
                let record =
                    unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityDevice6>()) };
                pointers.push(record.name.cast_mut().cast());
            }
            Some(ActivityKind::NvLink) if self.has_layout::<sys::CUpti_ActivityNvLink5>(true) => {
                // SAFETY: `has_layout` checked that CUPTI wrote a full record of this type within `self.len` bytes; `read_unaligned` makes no alignment assumption.
                let record =
                    unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityNvLink5>()) };
                pointers.push(record.portDev0.cast());
                pointers.push(record.portDev1.cast());
            }
            Some(ActivityKind::Overhead)
                if self.has_layout::<sys::CUpti_ActivityOverhead3>(true) =>
            {
                // SAFETY: `has_layout` checked that CUPTI wrote a full record of this type within `self.len` bytes; `read_unaligned` makes no alignment assumption.
                let record =
                    unsafe { ptr::read_unaligned(self.ptr.cast::<sys::CUpti_ActivityOverhead3>()) };
                pointers.push(record.overheadData);
            }
            _ => {}
        }
    }
}

// Generates one kind-checked decoder per record type, and `decode`, which dispatches on the kind.
// `fallible` decoders return `Result<Option<_>>` because their records hold enum values this crate may not name.
// `borrowed` payloads borrow strings or tables from the activity buffer for `'a`.
macro_rules! record_decoders {
    ($(
        $fallible:ident $borrow:ident $name:ident -> $variant:ident($payload:ident) = $raw:ident: $first:ident $(| $kind:ident)*;
    )*) => {
        impl<'a> ActivityRecord<'a> {
            $(record_decoders!(@decoder $fallible $borrow $name $payload $raw $first $($kind)*);)*

            /// Decodes this record into a tagged payload.
            ///
            /// Kinds without a decoder, truncated records, and user-defined records produce [`ActivityRecordPayload::Unsupported`] with the original raw kind.
            /// Borrowed payload fields retain the source record lifetime.
            ///
            /// # Errors
            ///
            /// Returns an error if the record holds an enum value this crate does not name.
            pub fn decode(self) -> Result<ActivityRecordPayload<'a>> {
                let raw_kind = self.raw_kind();
                let Some(kind) = self.kind() else {
                    return Ok(ActivityRecordPayload::Unsupported { raw_kind });
                };

                Ok(match kind {
                    $(
                        ActivityKind::$first $(| ActivityKind::$kind)* => {
                            record_decoders!(@decode $fallible self.$name())
                                .map_or(ActivityRecordPayload::Unsupported { raw_kind }, ActivityRecordPayload::$variant)
                        }
                    )*
                    _ => ActivityRecordPayload::Unsupported { raw_kind },
                })
            }
        }
    };
    (@decoder fallible $borrow:ident $name:ident $payload:ident $raw:ident $first:ident $($kind:ident)*) => {
        #[doc = concat!(
            "Decodes [`", stringify!($payload), "`] from [`ActivityKind::", stringify!($first), "`]",
            $(", [`ActivityKind::", stringify!($kind), "`]",)*
            " records, or returns `None` for other kinds, truncated records, and user-defined records."
        )]
        ///
        /// # Errors
        ///
        /// Returns an error if the record holds an enum value this crate does not name.
        pub fn $name(self) -> Result<Option<record_decoders!(@payload $borrow $payload)>> {
            if !self.has_layout::<sys::$raw>(matches!(self.kind(), Some(ActivityKind::$first $(| ActivityKind::$kind)*))) {
                return Ok(None);
            }

            // SAFETY: `has_layout` checked that CUPTI wrote a full record of this type at `self.ptr` and that `self.len` bytes cover it; `read_unaligned` makes no alignment assumption.
            let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::$raw>()) };
            $payload::from_raw(&record).map(Some)
        }
    };
    (@decoder infallible $borrow:ident $name:ident $payload:ident $raw:ident $first:ident $($kind:ident)*) => {
        #[doc = concat!(
            "Decodes [`", stringify!($payload), "`] from [`ActivityKind::", stringify!($first), "`]",
            $(", [`ActivityKind::", stringify!($kind), "`]",)*
            " records, or returns `None` for other kinds, truncated records, and user-defined records."
        )]
        #[must_use]
        pub fn $name(self) -> Option<record_decoders!(@payload $borrow $payload)> {
            if !self.has_layout::<sys::$raw>(matches!(self.kind(), Some(ActivityKind::$first $(| ActivityKind::$kind)*))) {
                return None;
            }

            // SAFETY: `has_layout` checked that CUPTI wrote a full record of this type at `self.ptr` and that `self.len` bytes cover it; `read_unaligned` makes no alignment assumption.
            let record = unsafe { ptr::read_unaligned(self.ptr.cast::<sys::$raw>()) };
            Some($payload::from_raw(&record))
        }
    };
    (@payload borrowed $payload:ident) => { $payload<'a> };
    (@payload owned $payload:ident) => { $payload };
    (@decode fallible $call:expr) => { $call? };
    (@decode infallible $call:expr) => { $call };
}

record_decoders! {
    fallible owned memory_decompress -> MemoryDecompress(ActivityMemoryDecompress) = CUpti_ActivityMemDecompress: MemoryDecompress;
    fallible owned memory_copy -> MemoryCopy(ActivityMemoryCopy) = CUpti_ActivityMemcpy7: MemoryCopy;
    fallible owned peer_memory_copy -> PeerMemoryCopy(ActivityPeerMemoryCopy) = CUpti_ActivityMemcpyPtoP4: PeerMemoryCopy;
    fallible owned memory_set -> MemorySet(ActivityMemorySet) = CUpti_ActivityMemset5: MemorySet;
    fallible borrowed kernel -> Kernel(ActivityKernel) = CUpti_ActivityKernel13: Kernel | ConcurrentKernel;
    infallible borrowed cuda_dynamic_parallelism_kernel -> CudaDynamicParallelismKernel(ActivityCudaDynamicParallelismKernel) = CUpti_ActivityCdpKernel: CudaDynamicParallelismKernel;
    fallible owned api -> Api(ActivityApi) = CUpti_ActivityAPI: Driver | Runtime | InternalLaunchApi;
    fallible owned stream -> Stream(ActivityStream) = CUpti_ActivityStream: Stream;
    fallible owned synchronization -> Synchronization(ActivitySynchronization) = CUpti_ActivitySynchronization2: Synchronization;
    fallible borrowed memory -> Memory(ActivityMemory) = CUpti_ActivityMemory4: Memory2;
    fallible borrowed legacy_memory -> LegacyMemory(ActivityLegacyMemory) = CUpti_ActivityMemory: Memory;
    fallible owned memory_pool -> MemoryPool(ActivityMemoryPool) = CUpti_ActivityMemoryPool3: MemoryPool;
    infallible owned graph_trace -> GraphTrace(ActivityGraphTrace) = CUpti_ActivityGraphTrace2: GraphTrace;
    infallible owned device_graph_trace -> DeviceGraphTrace(ActivityDeviceGraphTrace) = CUpti_ActivityDeviceGraphTrace: DeviceGraphTrace;
    infallible owned graph_host_node -> GraphHostNode(ActivityGraphHostNode) = CUpti_ActivityGraphHostNode: GraphHostNode;
    infallible owned host_launch -> HostLaunch(ActivityHostLaunch) = CUpti_ActivityHostLaunch: HostLaunch;
    fallible owned compute_engine_context_switch -> ComputeEngineContextSwitch(ActivityComputeEngineContextSwitch) = CUpti_ActivityComputeEngineCtxSwitch: ComputeEngineCtxSwitch;
    infallible owned green_context -> GreenContext(ActivityGreenContext) = CUpti_ActivityGreenContext3: GreenContext;
    fallible borrowed name -> Name(ActivityName) = CUpti_ActivityName: Name;
    fallible borrowed marker -> Marker(ActivityMarker) = CUpti_ActivityMarker2: Marker;
    fallible owned external_correlation -> ExternalCorrelation(ActivityExternalCorrelation) = CUpti_ActivityExternalCorrelation: ExternalCorrelation;
    fallible owned marker_data -> MarkerData(ActivityMarkerPayload) = CUpti_ActivityMarkerData2: MarkerData;
    infallible owned cuda_event -> CudaEvent(ActivityCudaEvent) = CUpti_ActivityCudaEvent2: CudaEvent;
    fallible owned context -> Context(ActivityContext) = CUpti_ActivityContext4: Context;
    infallible owned global_access -> GlobalAccess(ActivityGlobalAccess) = CUpti_ActivityGlobalAccess3: GlobalAccess;
    infallible owned branch -> Branch(ActivityBranch) = CUpti_ActivityBranch2: Branch;
    infallible owned instruction_execution -> InstructionExecution(ActivityInstructionExecution) = CUpti_ActivityInstructionExecution: InstructionExecution;
    fallible owned pc_sampling -> PcSampling(ActivityPcSampling) = CUpti_ActivityPCSampling3: PcSampling;
    infallible owned pc_sampling_record_info -> PcSamplingRecordInfo(ActivityPcSamplingRecordInfo) = CUpti_ActivityPCSamplingRecordInfo: PcSamplingRecordInfo;
    fallible owned unified_memory_counter -> UnifiedMemoryCounter(ActivityUnifiedMemoryCounter) = CUpti_ActivityUnifiedMemoryCounter3: UnifiedMemoryCounter;
    fallible owned device_attribute -> DeviceAttribute(ActivityDeviceAttribute) = CUpti_ActivityDeviceAttribute: DeviceAttribute;
    fallible owned environment -> Environment(ActivityEnvironment) = CUpti_ActivityEnvironment: Environment;
    infallible borrowed device -> Device(ActivityDevice) = CUpti_ActivityDevice6: Device;
    fallible borrowed jit -> Jit(ActivityJit) = CUpti_ActivityJit2: Jit;
    fallible borrowed nvlink -> NvLink(ActivityNvLink) = CUpti_ActivityNvLink5: NvLink;
    fallible owned pcie -> Pcie(ActivityPcie) = CUpti_ActivityPcie: Pcie;
    infallible owned instruction_correlation -> InstructionCorrelation(ActivityInstructionCorrelation) = CUpti_ActivityInstructionCorrelation: InstructionCorrelation;
    fallible owned open_multi_processing -> OpenMultiProcessing(ActivityOpenMultiProcessing) = CUpti_ActivityOpenMp: OpenMultiProcessing;
    fallible borrowed open_acc_data -> OpenAccData(ActivityOpenAccTransfer) = CUpti_ActivityOpenAccData: OpenAccData;
    fallible borrowed open_acc_launch -> OpenAccLaunch(ActivityOpenAccLaunch) = CUpti_ActivityOpenAccLaunch: OpenAccLaunch;
    fallible borrowed open_acc_other -> OpenAccOther(ActivityOpenAccOther) = CUpti_ActivityOpenAccOther: OpenAccOther;
    fallible owned preemption -> Preemption(ActivityPreemption) = CUpti_ActivityPreemption: Preemption;
    infallible borrowed function -> Function(ActivityFunction) = CUpti_ActivityFunction: Function;
    infallible borrowed module -> Module(ActivityModule) = CUpti_ActivityModule: Module;
    infallible owned shared_access -> SharedAccess(ActivitySharedAccess) = CUpti_ActivitySharedAccess: SharedAccess;
    fallible owned confidential_compute_rotation -> ConfidentialComputeRotation(ActivityConfidentialComputeRotation) = CUpti_ActivityConfidentialComputeRotation: ConfidentialComputeRotation;
    infallible borrowed source_locator -> SourceLocator(ActivitySourceLocator) = CUpti_ActivitySourceLocator: SourceLocator;
    fallible owned overhead -> Overhead(ActivityOverhead) = CUpti_ActivityOverhead3: Overhead;
}

/// Decoded activity record payload, selected by its native activity kind.
///
/// Borrowed fields retain the source record lifetime; copying this enum does not extend that lifetime.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ActivityRecordPayload<'a> {
    /// [`ActivityApi`] decoded from an [`ActivityKind::Driver`], [`ActivityKind::Runtime`], or [`ActivityKind::InternalLaunchApi`] record.
    Api(ActivityApi),
    /// [`ActivityKernel`] decoded from an [`ActivityKind::Kernel`] or [`ActivityKind::ConcurrentKernel`] record.
    Kernel(ActivityKernel<'a>),
    /// [`ActivityCudaDynamicParallelismKernel`] decoded from an [`ActivityKind::CudaDynamicParallelismKernel`] record.
    CudaDynamicParallelismKernel(ActivityCudaDynamicParallelismKernel<'a>),
    /// [`ActivityMemoryCopy`] decoded from an [`ActivityKind::MemoryCopy`] record.
    MemoryCopy(ActivityMemoryCopy),
    /// [`ActivityPeerMemoryCopy`] decoded from an [`ActivityKind::PeerMemoryCopy`] record.
    PeerMemoryCopy(ActivityPeerMemoryCopy),
    /// [`ActivityMemorySet`] decoded from an [`ActivityKind::MemorySet`] record.
    MemorySet(ActivityMemorySet),
    /// [`ActivityMemoryDecompress`] decoded from an [`ActivityKind::MemoryDecompress`] record.
    MemoryDecompress(ActivityMemoryDecompress),
    /// [`ActivityStream`] decoded from an [`ActivityKind::Stream`] record.
    Stream(ActivityStream),
    /// [`ActivitySynchronization`] decoded from an [`ActivityKind::Synchronization`] record.
    Synchronization(ActivitySynchronization),
    /// [`ActivityMemory`] decoded from an [`ActivityKind::Memory2`] record.
    Memory(ActivityMemory<'a>),
    /// [`ActivityLegacyMemory`] decoded from an [`ActivityKind::Memory`] record.
    LegacyMemory(ActivityLegacyMemory<'a>),
    /// [`ActivityMemoryPool`] decoded from an [`ActivityKind::MemoryPool`] record.
    MemoryPool(ActivityMemoryPool),
    /// [`ActivityGraphTrace`] decoded from an [`ActivityKind::GraphTrace`] record.
    GraphTrace(ActivityGraphTrace),
    /// [`ActivityDeviceGraphTrace`] decoded from an [`ActivityKind::DeviceGraphTrace`] record.
    DeviceGraphTrace(ActivityDeviceGraphTrace),
    /// [`ActivityGraphHostNode`] decoded from an [`ActivityKind::GraphHostNode`] record.
    GraphHostNode(ActivityGraphHostNode),
    /// [`ActivityHostLaunch`] decoded from an [`ActivityKind::HostLaunch`] record.
    HostLaunch(ActivityHostLaunch),
    /// [`ActivityComputeEngineContextSwitch`] decoded from an [`ActivityKind::ComputeEngineCtxSwitch`] record.
    ComputeEngineContextSwitch(ActivityComputeEngineContextSwitch),
    /// [`ActivityGreenContext`] decoded from an [`ActivityKind::GreenContext`] record.
    GreenContext(ActivityGreenContext),
    /// [`ActivityName`] decoded from an [`ActivityKind::Name`] record.
    Name(ActivityName<'a>),
    /// [`ActivityMarker`] decoded from an [`ActivityKind::Marker`] record.
    Marker(ActivityMarker<'a>),
    /// [`ActivityExternalCorrelation`] decoded from an [`ActivityKind::ExternalCorrelation`] record.
    ExternalCorrelation(ActivityExternalCorrelation),
    /// [`ActivityMarkerPayload`] decoded from an [`ActivityKind::MarkerData`] record.
    MarkerData(ActivityMarkerPayload),
    /// [`ActivityCudaEvent`] decoded from an [`ActivityKind::CudaEvent`] record.
    CudaEvent(ActivityCudaEvent),
    /// [`ActivityContext`] decoded from an [`ActivityKind::Context`] record.
    Context(ActivityContext),
    /// [`ActivityGlobalAccess`] decoded from an [`ActivityKind::GlobalAccess`] record.
    GlobalAccess(ActivityGlobalAccess),
    /// [`ActivityBranch`] decoded from an [`ActivityKind::Branch`] record.
    Branch(ActivityBranch),
    /// [`ActivityInstructionExecution`] decoded from an [`ActivityKind::InstructionExecution`] record.
    InstructionExecution(ActivityInstructionExecution),
    /// [`ActivityPcSampling`] decoded from an [`ActivityKind::PcSampling`] record.
    PcSampling(ActivityPcSampling),
    /// [`ActivityPcSamplingRecordInfo`] decoded from an [`ActivityKind::PcSamplingRecordInfo`] record.
    PcSamplingRecordInfo(ActivityPcSamplingRecordInfo),
    /// [`ActivityUnifiedMemoryCounter`] decoded from an [`ActivityKind::UnifiedMemoryCounter`] record.
    UnifiedMemoryCounter(ActivityUnifiedMemoryCounter),
    /// [`ActivityDeviceAttribute`] decoded from an [`ActivityKind::DeviceAttribute`] record.
    DeviceAttribute(ActivityDeviceAttribute),
    /// [`ActivityEnvironment`] decoded from an [`ActivityKind::Environment`] record.
    Environment(ActivityEnvironment),
    /// [`ActivityDevice`] decoded from an [`ActivityKind::Device`] record.
    Device(ActivityDevice<'a>),
    /// [`ActivityJit`] decoded from an [`ActivityKind::Jit`] record.
    Jit(ActivityJit<'a>),
    /// [`ActivityNvLink`] decoded from an [`ActivityKind::NvLink`] record.
    NvLink(ActivityNvLink<'a>),
    /// [`ActivityPcie`] decoded from an [`ActivityKind::Pcie`] record.
    Pcie(ActivityPcie),
    /// [`ActivityInstructionCorrelation`] decoded from an [`ActivityKind::InstructionCorrelation`] record.
    InstructionCorrelation(ActivityInstructionCorrelation),
    /// [`ActivityOpenMultiProcessing`] decoded from an [`ActivityKind::OpenMultiProcessing`] record.
    OpenMultiProcessing(ActivityOpenMultiProcessing),
    /// [`ActivityOpenAccTransfer`] decoded from an [`ActivityKind::OpenAccData`] record.
    OpenAccData(ActivityOpenAccTransfer<'a>),
    /// [`ActivityOpenAccLaunch`] decoded from an [`ActivityKind::OpenAccLaunch`] record.
    OpenAccLaunch(ActivityOpenAccLaunch<'a>),
    /// [`ActivityOpenAccOther`] decoded from an [`ActivityKind::OpenAccOther`] record.
    OpenAccOther(ActivityOpenAccOther<'a>),
    /// [`ActivityPreemption`] decoded from an [`ActivityKind::Preemption`] record.
    Preemption(ActivityPreemption),
    /// [`ActivityFunction`] decoded from an [`ActivityKind::Function`] record.
    Function(ActivityFunction<'a>),
    /// [`ActivityModule`] decoded from an [`ActivityKind::Module`] record.
    Module(ActivityModule<'a>),
    /// [`ActivitySharedAccess`] decoded from an [`ActivityKind::SharedAccess`] record.
    SharedAccess(ActivitySharedAccess),
    /// [`ActivityConfidentialComputeRotation`] decoded from an [`ActivityKind::ConfidentialComputeRotation`] record.
    ConfidentialComputeRotation(ActivityConfidentialComputeRotation),
    /// [`ActivitySourceLocator`] decoded from an [`ActivityKind::SourceLocator`] record.
    SourceLocator(ActivitySourceLocator<'a>),
    /// [`ActivityOverhead`] decoded from an [`ActivityKind::Overhead`] record.
    Overhead(ActivityOverhead),
    /// Record of a kind this crate does not name or decode, a truncated record, or a user-defined record.
    Unsupported {
        /// Native activity-kind value, preserved without decoding its payload.
        raw_kind: u32,
    },
}

/// Iterator over borrowed records in a completed activity buffer.
///
/// Stops at the end of the buffer.
/// A native iteration error is yielded once as `Err`, after which iteration stops.
#[derive(Debug)]
pub struct ActivityRecords<'a> {
    bytes: &'a [u8],
    current: *mut sys::CUpti_Activity,
    full_layout: bool,
    finished: bool,
}

impl<'a> ActivityRecords<'a> {
    // `full_layout` is false when the buffer may hold user-defined records.
    pub(crate) fn new(bytes: &'a [u8], full_layout: bool) -> Self {
        Self {
            bytes,
            current: ptr::null_mut(),
            full_layout,
            finished: false,
        }
    }
}

impl<'a> Iterator for ActivityRecords<'a> {
    type Item = Result<ActivityRecord<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // SAFETY: `self.bytes` is borrowed for `'a` and CUPTI only reads it, and `self.current` is the cursor CUPTI advances, null before the first record.
        let status = unsafe {
            sys::cuptiActivityGetNextRecord(
                self.bytes.as_ptr().cast_mut(),
                self.bytes.len() as u64,
                &raw mut self.current,
            )
        };

        match status {
            sys::CUptiResult::CUPTI_SUCCESS => {
                // A record must start inside the buffer and hold at least its kind header.
                let offset = self.current.addr().wrapping_sub(self.bytes.as_ptr().addr());
                let len = self.bytes.len().saturating_sub(offset);
                if offset >= self.bytes.len() || len < size_of::<u32>() {
                    self.finished = true;
                    return Some(Err(Error::out_of_range("activity record offset")));
                }

                Some(Ok(ActivityRecord {
                    ptr: self.current.cast_const(),
                    len,
                    full_layout: self.full_layout,
                    _marker: PhantomData,
                }))
            }
            // CUPTI reports the end of the buffer as `MAX_LIMIT_REACHED`.
            sys::CUptiResult::CUPTI_ERROR_MAX_LIMIT_REACHED => {
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

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use singe_cuda::dim::Dim3;

    use super::*;
    use crate::{
        activity::{
            device::{
                ActivityDeviceAttributeKind, ActivityDeviceAttributeValue,
                ActivityEnvironmentPower, ActivityEnvironmentReading, ActivityEnvironmentSpeed,
                ActivityNpuId, ActivityPcieAttributes, ActivityPcieBridgeAttributes,
                ActivityPcieDeviceId, ActivityPcieGpuAttributes, ActivityTopologyDeviceId,
            },
            memory::{ActivityMemoryPoolProperties, ActivityUnifiedMemoryCounterFlags},
            trace::ActivityObjectId,
        },
        types::{
            ActivityComputeApiKind, ActivityEnvironmentKind, ActivityFlags, ActivityJitEntryType,
            ActivityJitOperationType, ActivityLaunchType, ActivityMarkerId, ActivityMemoryCopyKind,
            ActivityMemoryKind, ActivityMemoryOperationType, ActivityMemoryPoolOperationType,
            ActivityMemoryPoolType, ActivityObjectKind, ActivityOverheadKind,
            ActivityPcSamplingStallReason, ActivityPreemptionKind, ActivityStreamFlag,
            ActivitySynchronizationType, ActivityUnifiedMemoryCounterKind,
            ActivityUnifiedMemoryMigrationCause, CallbackId, ChannelId, ChannelType,
            ComputeEngineContextSwitchOperationType, ConfidentialComputeRotationEventType,
            ContextCigMode, ContextId, CorrelationId, CudaDevice, CudaDeviceAttribute, CudaEventId,
            CudaEventSyncId, CuptiDomainId, DeviceAttribute, DeviceGraphLaunchMode, DeviceId,
            DeviceType, EnvironmentClocksThrottleReasons, ExternalCorrelationId,
            ExternalCorrelationKind, FunctionId, FunctionSharedMemoryLimitConfig, GraphId,
            GraphNodeId, GridId, JitOperationCorrelationId, LinkFlags, MetricValue,
            MetricValueKind, ModuleId, NpuDomainId, OpenAccConstructKind, OpenAccEventKind,
            OpenMultiProcessingEventKind, PcieBridgeId, PcieBusId, PcieDeviceType, PcieDomainId,
            PcieHardwareDeviceId, PcieVendorId, ProcessId, SourceLocatorId, StreamId, ThreadId,
            UnifiedMemoryProcessorId,
        },
    };

    fn record_of<T>(value: &T) -> ActivityRecord<'_> {
        ActivityRecord {
            ptr: ptr::from_ref(value).cast(),
            len: size_of::<T>(),
            full_layout: true,
            _marker: PhantomData,
        }
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
        let record = record_of(&raw);

        assert_eq!(record.kind(), Some(ActivityKind::MemoryDecompress));
        assert_eq!(
            record.memory_decompress().unwrap(),
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
            record.decode().unwrap(),
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
        let record = record_of(&raw);

        assert_eq!(record.memory_decompress().unwrap(), None);
        assert_eq!(
            record.decode().unwrap(),
            ActivityRecordPayload::Unsupported {
                raw_kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INVALID.0
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
            launchType: u8::try_from(
                sys::CUpti_ActivityLaunchType::CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_SINGLE_DEVICE
                    .0,
            )
            .unwrap(),
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

        let record = record_of(&raw);

        let kernel = record.kernel().unwrap().unwrap();
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
        assert!(matches!(
            record.decode().unwrap(),
            ActivityRecordPayload::Kernel(_)
        ));
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
        let memory_record = record_of(&memory);
        let decoded_memory = memory_record.legacy_memory().unwrap().unwrap();

        assert_eq!(decoded_memory.memory_kind, ActivityMemoryKind::Device);
        assert_eq!(decoded_memory.address, 1);
        assert_eq!(decoded_memory.device_id, DeviceId::from(8));
        assert_eq!(decoded_memory.name.unwrap().to_str().unwrap(), "legacy");
        assert!(matches!(
            memory_record.decode().unwrap(),
            ActivityRecordPayload::LegacyMemory(_)
        ));

        let memory_copy = sys::CUpti_ActivityMemcpyPtoP4 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY2,
            copyKind: u8::try_from(
                sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_PTOP.0,
            )
            .unwrap(),
            srcKind: u8::try_from(
                sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE.0,
            )
            .unwrap(),
            dstKind: u8::try_from(
                sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_MANAGED.0,
            )
            .unwrap(),
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
        let memory_copy_record = record_of(&memory_copy);
        let decoded_memory_copy = memory_copy_record.peer_memory_copy().unwrap().unwrap();

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
            memory_copy_record.decode().unwrap(),
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
        let cdp_record = record_of(&cdp);
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
            cdp_record.decode().unwrap(),
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
        let record = record_of(&raw);

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
        assert_eq!(record.api().unwrap(), Some(api));
        assert_eq!(record.decode().unwrap(), ActivityRecordPayload::Api(api));
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
        let record = record_of(&raw);
        let stream = ActivityStream {
            context_id: ContextId::from(1),
            stream_id: StreamId::from(2u64),
            priority: 3,
            flag: ActivityStreamFlag::NonBlocking,
            correlation_id: CorrelationId::from(4),
        };

        assert_eq!(record.stream().unwrap(), Some(stream));
        assert_eq!(
            record.decode().unwrap(),
            ActivityRecordPayload::Stream(stream)
        );
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
        let record = record_of(&raw);
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

        assert_eq!(record.synchronization().unwrap(), Some(synchronization));
        assert_eq!(
            record.decode().unwrap(),
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
        let record = record_of(&raw);

        let memory = record.memory().unwrap().unwrap();
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
        assert!(matches!(
            record.decode().unwrap(),
            ActivityRecordPayload::Memory(_)
        ));
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
        let record = record_of(&raw);

        let memory = record.memory().unwrap().unwrap();
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
        let record = record_of(&raw);

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
        assert_eq!(record.memory_pool().unwrap(), Some(memory_pool));
        assert_eq!(
            record.decode().unwrap(),
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
        let record = record_of(&raw);
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
            record.decode().unwrap(),
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
                .0,
            contextId: 6,
            streamId: 7,
            reserved: ptr::null_mut(),
        };
        let record = record_of(&raw);
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
            record.decode().unwrap(),
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
        let record = record_of(&raw);
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
            record.decode().unwrap(),
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
        let record = record_of(&raw);
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
            record.decode().unwrap(),
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
        let record = record_of(&raw);
        let context_switch = ActivityComputeEngineContextSwitch {
            context_id: ContextId::from(1),
            timestamp: 2,
            operation_type: ComputeEngineContextSwitchOperationType::End,
        };

        assert_eq!(
            record.compute_engine_context_switch().unwrap(),
            Some(context_switch)
        );
        assert_eq!(
            record.decode().unwrap(),
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
        let record = record_of(&raw);
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
            record.decode().unwrap(),
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
        let preemption_record = record_of(&preemption);
        let decoded_preemption = ActivityPreemption {
            preemption_kind: ActivityPreemptionKind::Restore,
            timestamp: 1,
            grid_id: GridId::from(2),
            block: Dim3 { x: 3, y: 4, z: 5 },
        };

        assert_eq!(
            preemption_record.preemption().unwrap(),
            Some(decoded_preemption)
        );
        assert_eq!(
            preemption_record.decode().unwrap(),
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
        let function_record = record_of(&function);
        let decoded_function = function_record.function().unwrap();

        assert_eq!(decoded_function.id, FunctionId::from(6));
        assert_eq!(decoded_function.context_id, ContextId::from(7));
        assert_eq!(decoded_function.module_id, ModuleId::from(8));
        assert_eq!(decoded_function.function_index, 9);
        assert_eq!(decoded_function.name.unwrap().to_str().unwrap(), "kernel");
        assert!(matches!(
            function_record.decode().unwrap(),
            ActivityRecordPayload::Function(_)
        ));

        let cubin = [0xca, 0xfe, 0xba, 0xbe];
        let module = sys::CUpti_ActivityModule {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MODULE,
            contextId: 10,
            id: 11,
            cubinSize: u32::try_from(cubin.len()).unwrap(),
            cubin: cubin.as_ptr().cast(),
        };
        let module_record = record_of(&module);
        let decoded_module = module_record.module().unwrap();

        assert_eq!(decoded_module.context_id, ContextId::from(10));
        assert_eq!(decoded_module.id, ModuleId::from(11));
        assert_eq!(decoded_module.cubin, Some(cubin.as_slice()));
        assert!(matches!(
            module_record.decode().unwrap(),
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
        let shared_record = record_of(&shared);
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
            shared_record.decode().unwrap(),
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
        let record = record_of(&raw);
        let decoded = ActivityConfidentialComputeRotation {
            event_type: ConfidentialComputeRotationEventType::ChannelDrained,
            device_id: DeviceId::from(1),
            context_id: ContextId::from(2),
            channel_id: ChannelId::from(3),
            channel_type: ChannelType::Decompress,
            timestamp: 4,
        };

        assert_eq!(
            record.confidential_compute_rotation().unwrap(),
            Some(decoded)
        );
        assert_eq!(
            record.decode().unwrap(),
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
        let record = record_of(&raw);
        let decoded = record.name().unwrap().unwrap();

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
        assert!(matches!(
            record.decode().unwrap(),
            ActivityRecordPayload::Name(_)
        ));
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
        let record = record_of(&raw);
        let decoded = record.marker().unwrap().unwrap();

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
        assert!(matches!(
            record.decode().unwrap(),
            ActivityRecordPayload::Marker(_)
        ));
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
        let record = record_of(&raw);
        let decoded = ActivityExternalCorrelation {
            external_kind: ExternalCorrelationKind::Custom0,
            external_id: ExternalCorrelationId::from(1),
            correlation_id: CorrelationId::from(2),
        };

        assert_eq!(record.external_correlation().unwrap(), Some(decoded));
        assert_eq!(
            record.decode().unwrap(),
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
        let record = record_of(&raw);
        let decoded = ActivityMarkerPayload {
            flags: ActivityFlags::MARKER_END,
            id: ActivityMarkerId::from(1),
            payload_kind: MetricValueKind::Uint64,
            payload: MetricValue::Uint64(2),
            color: 3,
            category: 4,
            cupti_domain_id: CuptiDomainId::from(5),
        };

        assert_eq!(record.marker_data().unwrap(), Some(decoded));
        assert_eq!(
            record.decode().unwrap(),
            ActivityRecordPayload::MarkerData(decoded)
        );
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
        let record = record_of(&raw);
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
        assert_eq!(
            record.decode().unwrap(),
            ActivityRecordPayload::CudaEvent(decoded)
        );
    }

    #[test]
    fn test_activity_record_decodes_context_record() {
        let raw = sys::CUpti_ActivityContext4 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONTEXT,
            contextId: 1,
            deviceId: 2,
            computeApiKind: u16::try_from(
                sys::CUpti_ActivityComputeApiKind::CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS.0,
            )
            .unwrap(),
            nullStreamId: 3,
            parentContextId: 4,
            isGreenContext: 1,
            padding: 0,
            numMultiprocessors: 5,
            cigMode: sys::CUpti_ContextCigMode::CUPTI_CONTEXT_CIG_MODE_CIG_FALLBACK,
            padding2: 0,
            processId: 6,
        };
        let record = record_of(&raw);
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

        assert_eq!(record.context().unwrap(), Some(decoded));
        assert_eq!(
            record.decode().unwrap(),
            ActivityRecordPayload::Context(decoded)
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
        let global_record = record_of(&global);
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
            global_record.decode().unwrap(),
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
        let branch_record = record_of(&branch);
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
            branch_record.decode().unwrap(),
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
        let instruction_record = record_of(&instruction);
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
            instruction_record.decode().unwrap(),
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
        let sampling_record = record_of(&sampling);
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

        assert_eq!(
            sampling_record.pc_sampling().unwrap(),
            Some(decoded_sampling)
        );
        assert_eq!(
            sampling_record.decode().unwrap(),
            ActivityRecordPayload::PcSampling(decoded_sampling)
        );

        let info = sys::CUpti_ActivityPCSamplingRecordInfo {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO,
            correlationId: 14,
            totalSamples: 15,
            droppedSamples: 16,
            samplingPeriodInCycles: 17,
        };
        let info_record = record_of(&info);
        let decoded_info = ActivityPcSamplingRecordInfo {
            correlation_id: CorrelationId::from(14),
            total_samples: 15,
            dropped_samples: 16,
            sampling_period_in_cycles: 17,
        };

        assert_eq!(info_record.pc_sampling_record_info(), Some(decoded_info));
        assert_eq!(
            info_record.decode().unwrap(),
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
            flags: sys::CUpti_ActivityUnifiedMemoryMigrationCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_USER.0,
            pad: 0,
            processors: [27, 28, 29, 30, 31],
        };
        let unified_record = record_of(&unified);
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
            unified_record.unified_memory_counter().unwrap(),
            Some(decoded_unified)
        );
        assert_eq!(
            unified_record.decode().unwrap(),
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
        let cupti_record = record_of(&cupti);
        let decoded_cupti = ActivityDeviceAttribute {
            flags: ActivityFlags::empty(),
            device_id: DeviceId::from(1),
            attribute: ActivityDeviceAttributeKind::Cupti(DeviceAttribute::MaxEventId),
            value: ActivityDeviceAttributeValue { bits: 2 },
        };

        assert_eq!(decoded_cupti.value.as_u64(), 2);
        assert_eq!(
            cupti_record.device_attribute().unwrap(),
            Some(decoded_cupti)
        );
        assert_eq!(
            cupti_record.decode().unwrap(),
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
        let cuda_record = record_of(&cuda);
        let decoded_cuda = cuda_record.device_attribute().unwrap().unwrap();

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
            cuda_record.decode().unwrap(),
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
        let speed_record = record_of(&speed);
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

        assert_eq!(speed_record.environment().unwrap(), Some(decoded_speed));
        assert_eq!(
            speed_record.decode().unwrap(),
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
        let power_record = record_of(&power);
        let decoded_power = ActivityEnvironment {
            device_id: DeviceId::from(7),
            timestamp: 8,
            environment_kind: ActivityEnvironmentKind::Power,
            data: ActivityEnvironmentReading::Power(ActivityEnvironmentPower {
                power: 9,
                power_limit: 10,
            }),
        };

        assert_eq!(power_record.environment().unwrap(), Some(decoded_power));
        assert_eq!(
            power_record.decode().unwrap(),
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
        let record = record_of(&raw);
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
        assert!(matches!(
            record.decode().unwrap(),
            ActivityRecordPayload::Device(_)
        ));
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
        let record = record_of(&raw);
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

        assert_eq!(record.jit().unwrap(), Some(decoded));
        assert_eq!(
            record.decode().unwrap(),
            ActivityRecordPayload::Jit(decoded)
        );
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
            flag: sys::CUpti_LinkFlag::CUPTI_LINK_FLAG_PEER_ACCESS.0
                | sys::CUpti_LinkFlag::CUPTI_LINK_FLAG_SYSMEM_ACCESS.0,
            physicalNvLinkCount: 2,
            portDev0: ports0.as_mut_ptr(),
            portDev1: ports1.as_mut_ptr(),
            bandwidth: 9,
            nvswitchConnected: 1,
            pad: [0; 7],
        };
        let record = record_of(&raw);
        let decoded = record.nvlink().unwrap().unwrap();

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
        assert!(matches!(
            record.decode().unwrap(),
            ActivityRecordPayload::NvLink(_)
        ));
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
        let gpu_record = record_of(&gpu);
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

        assert_eq!(gpu_record.pcie().unwrap(), Some(gpu_decoded));
        assert_eq!(
            gpu_record.decode().unwrap(),
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
        let bridge_record = record_of(&bridge);
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

        assert_eq!(bridge_record.pcie().unwrap(), Some(bridge_decoded));
        assert_eq!(
            bridge_record.decode().unwrap(),
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
        let record = record_of(&raw);
        let decoded = ActivityInstructionCorrelation {
            flags: ActivityFlags::INSTRUCTION_CLASS_MASK,
            source_locator_id: SourceLocatorId::from(1),
            function_id: FunctionId::from(2),
            pc_offset: 3,
        };

        assert_eq!(record.instruction_correlation(), Some(decoded));
        assert_eq!(
            record.decode().unwrap(),
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
        let record = record_of(&raw);
        let decoded = ActivityOpenMultiProcessing {
            event_kind: OpenMultiProcessingEventKind::WaitBarrier,
            version: 1,
            thread_id: ThreadId::from(2),
            start: 3,
            end: 4,
            process_id: ProcessId::from(5u64),
            cu_thread_id: ThreadId::from(6),
        };

        assert_eq!(record.open_multi_processing().unwrap(), Some(decoded));
        assert_eq!(
            record.decode().unwrap(),
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
        let data_record = record_of(&data);
        let decoded_data = data_record.open_acc_data().unwrap().unwrap();

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
            data_record.decode().unwrap(),
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
        let launch_record = record_of(&launch);
        let decoded_launch = launch_record.open_acc_launch().unwrap().unwrap();

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
            launch_record.decode().unwrap(),
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
        let other_record = record_of(&other);
        let decoded_other = other_record.open_acc_other().unwrap().unwrap();

        assert_eq!(decoded_other.base.event_kind, OpenAccEventKind::Wait);
        assert_eq!(
            decoded_other.base.parent_construct,
            OpenAccConstructKind::Wait
        );
        assert!(matches!(
            other_record.decode().unwrap(),
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
        let record = record_of(&raw);
        let decoded = record.source_locator().unwrap();

        assert_eq!(decoded.id, SourceLocatorId::from(1));
        assert_eq!(decoded.line_number, 2);
        assert_eq!(decoded.file_name.unwrap().to_str().unwrap(), "kernel.cu");
        assert!(matches!(
            record.decode().unwrap(),
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
        let record = record_of(&raw);
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

        assert_eq!(record.overhead().unwrap(), Some(decoded));
        assert_eq!(
            record.decode().unwrap(),
            ActivityRecordPayload::Overhead(decoded)
        );
    }

    #[test]
    fn test_activity_record_decodes_memory_copy_record() {
        let raw = sys::CUpti_ActivityMemcpy7 {
            kind: sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY,
            copyKind: u8::try_from(
                sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_HTOD.0,
            )
            .unwrap(),
            srcKind: u8::try_from(
                sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE.0,
            )
            .unwrap(),
            dstKind: u8::try_from(
                sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE.0,
            )
            .unwrap(),
            flags: u8::try_from(
                sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_DEVICE_CONCURRENT_KERNELS.0,
            )
            .unwrap(),
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
        let record = record_of(&raw);

        assert_eq!(record.memory_decompress().unwrap(), None);
        assert_eq!(
            record.memory_copy().unwrap(),
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
            record.decode().unwrap(),
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
            flags: u16::try_from(
                sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_DEVICE_CONCURRENT_KERNELS.0,
            )
            .unwrap(),
            memoryKind: u16::try_from(
                sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE.0,
            )
            .unwrap(),
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
        let record = record_of(&raw);

        assert_eq!(record.memory_copy().unwrap(), None);
        assert_eq!(
            record.memory_set().unwrap(),
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
            record.decode().unwrap(),
            ActivityRecordPayload::MemorySet(ActivityMemorySet {
                value: 255,
                bytes: 2048,
                ..
            })
        ));
    }
}
