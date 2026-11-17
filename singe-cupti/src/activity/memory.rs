use std::ffi::CStr;

use crate::{
    error::Result,
    sys,
    types::{
        ActivityFlags, ActivityMemoryCopyKind, ActivityMemoryKind, ActivityMemoryOperationType,
        ActivityMemoryPoolOperationType, ActivityMemoryPoolType, ActivityUnifiedMemoryAccessType,
        ActivityUnifiedMemoryCounterKind, ActivityUnifiedMemoryMigrationCause,
        ActivityUnifiedMemoryRemoteMapCause, ChannelId, ChannelType, ContextId, CorrelationId,
        DeviceId, GraphId, GraphNodeId, ProcessId, StreamId, UnifiedMemoryProcessorId,
    },
    utility::optional_cstr,
};

fn memory_copy_kind_from_raw(value: u8) -> ActivityMemoryCopyKind {
    ActivityMemoryCopyKind::try_from(u32::from(value)).unwrap_or(ActivityMemoryCopyKind::Unknown)
}

fn memory_kind_from_raw(value: impl Into<u32>) -> ActivityMemoryKind {
    ActivityMemoryKind::try_from(value.into()).unwrap_or(ActivityMemoryKind::Unknown)
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
    pub(crate) fn from_raw(value: sys::CUpti_ActivityMemory4__bindgen_ty_1) -> Result<Self> {
        let pool_type = ActivityMemoryPoolType::try_from(value.memoryPoolType)?;
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

        Ok(Self {
            pool_type,
            address: value.address,
            data,
        })
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
    /// The correlation ID of the memory operation.
    /// Each memory operation is assigned a unique correlation ID that is identical to the correlation ID in the driver and runtime API activity record that launched the memory operation.
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
    /// Context identifier, or 0 when the native context is null.
    pub context_id: ContextId,
    /// The ID of the stream.
    /// It is 0 when the memory operation is not asynchronous.
    pub stream_id: StreamId,
    /// Variable name shared by records for the same symbol.
    /// `None` represents a null native pointer.
    pub name: Option<&'a CStr>,
    /// Whether the memory operation uses an asynchronous memory API.
    pub is_asynchronous: bool,
    /// The memory pool configuration used for the memory operations.
    pub memory_pool_config: ActivityMemoryPoolConfig,
    /// Shared object or binary from which the allocation request originates.
    /// `None` represents a null native pointer.
    pub source: Option<&'a CStr>,
}

impl ActivityMemory<'_> {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityMemory4) -> Result<Self> {
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

        Ok(Self {
            operation_type: ActivityMemoryOperationType::try_from(value.memoryOperationType)?,
            memory_kind: ActivityMemoryKind::try_from(value.memoryKind)?,
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
            memory_pool_config: ActivityMemoryPoolConfig::from_raw(value.memoryPoolConfig)?,
            source,
        })
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
    /// The end timestamp for the memory operation, i.e. the time when memory was freed, in nanoseconds.
    /// This will be 0 if memory is not freed in the application.
    pub end: u64,
    /// The program counter of the allocation of memory.
    pub allocation_pc: u64,
    /// The program counter of the freeing of memory.
    /// This will be 0 if memory is not freed in the application.
    pub free_pc: u64,
    /// The ID of the process to which this record belongs to.
    pub process_id: ProcessId,
    /// The ID of the device where the memory allocation is taking place.
    pub device_id: DeviceId,
    /// Context identifier, or 0 when the native context is null.
    pub context_id: ContextId,
    /// Variable name shared by records for the same symbol.
    /// `None` represents a null native pointer.
    pub name: Option<&'a CStr>,
}

impl ActivityLegacyMemory<'_> {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityMemory) -> Result<Self> {
        Ok(Self {
            memory_kind: ActivityMemoryKind::try_from(value.memoryKind)?,
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
        })
    }
}

/// Decoded memory-pool creation, destruction, or trimming activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemoryPool {
    /// The memory operation requested by the user, [`ActivityMemoryPoolOperationType`].
    pub operation_type: ActivityMemoryPoolOperationType,
    /// The type of the memory pool, [`ActivityMemoryPoolType`].
    pub pool_type: ActivityMemoryPoolType,
    /// The correlation ID of the memory pool operation.
    /// Each memory pool operation is assigned a unique correlation ID that is identical to the correlation ID in the driver and runtime API activity record that launched the memory operation.
    pub correlation_id: CorrelationId,
    /// The ID of the process to which this record belongs to.
    pub process_id: ProcessId,
    /// The ID of the device where the memory pool is created.
    pub device_id: DeviceId,
    /// The minimum bytes to keep of the memory pool.
    /// `min_bytes_to_keep` is valid for [`ActivityMemoryPoolOperationType::Trimmed`], [`ActivityMemoryPoolOperationType`].
    pub min_bytes_to_keep: u64,
    /// The virtual address of the allocation.
    pub address: u64,
    /// The size of the memory pool operation in bytes.
    /// `size` is valid for [`ActivityMemoryPoolType::Local`], [`ActivityMemoryPoolType`].
    pub size: u64,
    /// The release threshold of the memory pool.
    /// `release_threshold` is valid for [`ActivityMemoryPoolType::Local`], [`ActivityMemoryPoolType`].
    pub release_threshold: u64,
    /// The start timestamp for the memory operation, in nanoseconds.
    pub timestamp: u64,
    /// The utilized size of the memory pool.
    /// `utilized_size` is valid for [`ActivityMemoryPoolType::Local`], [`ActivityMemoryPoolType`].
    pub utilized_size: u64,
    /// Whether the pool uses managed memory (`true`) rather than pinned memory (`false`).
    pub is_managed_pool: bool,
}

impl ActivityMemoryPool {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityMemoryPool3) -> Result<Self> {
        Ok(Self {
            operation_type: ActivityMemoryPoolOperationType::try_from(
                value.memoryPoolOperationType,
            )?,
            pool_type: ActivityMemoryPoolType::try_from(value.memoryPoolType)?,
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
        })
    }
}

/// Decoded Unified Memory counter activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityUnifiedMemoryCounter {
    /// The Unified Memory counter kind.
    pub counter_kind: ActivityUnifiedMemoryCounterKind,
    /// Value of the counter For `counter_kind` [`ActivityUnifiedMemoryCounterKind::BytesTransferHostToDevice`], [`ActivityUnifiedMemoryCounterKind::BytesTransferDeviceToHost`], [`ActivityUnifiedMemoryCounterKind::Thrashing`] and [`ActivityUnifiedMemoryCounterKind::RemoteMap`], it is the size of the memory region in bytes.
    /// For `counter_kind` [`ActivityUnifiedMemoryCounterKind::GpuPageFault`], it is the number of page fault groups for the same page.
    /// For `counter_kind` [`ActivityUnifiedMemoryCounterKind::CentralProcessingUnitPageFaultCount`], it is the program counter for the instruction that caused fault.
    pub value: u64,
    /// The start timestamp of the counter, in nanoseconds.
    /// For `counter_kind` [`ActivityUnifiedMemoryCounterKind::BytesTransferHostToDevice`] and [`ActivityUnifiedMemoryCounterKind::BytesTransferDeviceToHost`], timestamp is captured when activity starts on GPU.
    /// For `counter_kind` [`ActivityUnifiedMemoryCounterKind::GpuPageFault`] and [`ActivityUnifiedMemoryCounterKind::CentralProcessingUnitPageFaultCount`], timestamp is captured when CUDA driver started processing the fault.
    /// For `counter_kind` [`ActivityUnifiedMemoryCounterKind::Thrashing`], timestamp is captured when CUDA driver detected thrashing of memory region.
    /// For `counter_kind` [`ActivityUnifiedMemoryCounterKind::Throttling`], timestamp is captured when throttling operation was started by CUDA driver.
    /// For `counter_kind` [`ActivityUnifiedMemoryCounterKind::RemoteMap`], timestamp is captured when CUDA driver has pushed all required operations to the processor specified by `destination_id`.
    pub start: u64,
    /// The end timestamp of the counter, in nanoseconds.
    /// Ignore this field if `counter_kind` is [`ActivityUnifiedMemoryCounterKind::CentralProcessingUnitPageFaultCount`] or [`ActivityUnifiedMemoryCounterKind::Thrashing`] or [`ActivityUnifiedMemoryCounterKind::RemoteMap`].
    /// For `counter_kind` [`ActivityUnifiedMemoryCounterKind::BytesTransferHostToDevice`] and [`ActivityUnifiedMemoryCounterKind::BytesTransferDeviceToHost`], timestamp is captured when activity finishes on GPU.
    /// For `counter_kind` [`ActivityUnifiedMemoryCounterKind::GpuPageFault`], timestamp is captured when CUDA driver queues the replay of faulting memory accesses on the GPU For `counter_kind` [`ActivityUnifiedMemoryCounterKind::Throttling`], timestamp is captured when throttling operation was finished by CUDA driver.
    pub end: u64,
    /// This is the virtual base address of the page/s being transferred.
    /// For cpu and gpu faults, the virtual address for the page that faulted.
    pub address: u64,
    /// The ID of the source CPU/device involved in the memory transfer, page fault, thrashing, throttling or remote map operation.
    /// For `counter_kind` [`ActivityUnifiedMemoryCounterKind::Thrashing`], it is a bitwise OR of the device IDs fighting for the memory region, ONLY if there are less than 32 devices.
    /// Ignore this field if `counter_kind` is [`ActivityUnifiedMemoryCounterKind::CentralProcessingUnitPageFaultCount`].
    pub source_id: UnifiedMemoryProcessorId,
    /// The ID of the destination CPU/device involved in the memory transfer or remote map operation.
    /// Ignore this field if `counter_kind` is [`ActivityUnifiedMemoryCounterKind::GpuPageFault`] or [`ActivityUnifiedMemoryCounterKind::CentralProcessingUnitPageFaultCount`] or [`ActivityUnifiedMemoryCounterKind::Thrashing`] or [`ActivityUnifiedMemoryCounterKind::Throttling`].
    pub destination_id: UnifiedMemoryProcessorId,
    /// The ID of the stream causing the transfer.
    /// This value of this field is invalid.
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
    pub(crate) fn from_raw(kind: ActivityUnifiedMemoryCounterKind, flags: u32) -> Self {
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
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityUnifiedMemoryCounter3) -> Result<Self> {
        let counter_kind = ActivityUnifiedMemoryCounterKind::try_from(value.counterKind)?;
        Ok(Self {
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
        })
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
    /// The correlation ID of the decompression operations.
    /// Each operation is assigned a unique correlation ID that is identical to the correlation ID in the driver API activity record that launched the operation.
    pub correlation_id: CorrelationId,
    /// The number of operations in the batch.
    pub number_of_operations: u32,
    /// The number of bytes to be read and decompressed in the batch operation.
    pub source_bytes: u64,
    /// The start timestamp.
    /// A value of 0 means the start time is unknown.
    pub start: u64,
    /// The end timestamp.
    /// A value of 0 means the end time is unknown.
    pub end: u64,
}

impl ActivityMemoryDecompress {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityMemDecompress) -> Result<Self> {
        Ok(Self {
            device_id: DeviceId::from(value.deviceId),
            context_id: ContextId::from(value.contextId),
            stream_id: StreamId::from(value.streamId),
            channel_id: ChannelId::from(value.channelID),
            channel_type: ChannelType::try_from(value.channelType)?,
            correlation_id: CorrelationId::from(value.correlationId),
            number_of_operations: value.numberOfOperations,
            source_bytes: value.sourceBytes,
            start: value.start,
            end: value.end,
        })
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
    pub flags: ActivityFlags,
    /// The number of bytes transferred by the memory copy.
    pub bytes: u64,
    /// The start timestamp for the memory copy, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the memory copy.
    pub start: u64,
    /// The end timestamp for the memory copy, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the memory copy.
    pub end: u64,
    /// The ID of the device where the memory copy is occurring.
    pub device_id: DeviceId,
    /// The ID of the context where the memory copy is occurring.
    pub context_id: ContextId,
    /// The ID of the stream where the memory copy is occurring.
    pub stream_id: StreamId,
    /// The correlation ID of the memory copy.
    /// Each memory copy is assigned a unique correlation ID that is identical to the correlation ID in the driver API activity record that launched the memory copy.
    pub correlation_id: CorrelationId,
    /// The runtime correlation ID of the memory copy.
    /// Each memory copy is assigned a unique runtime correlation ID that is identical to the correlation ID in the runtime API activity record that launched the memory copy.
    pub runtime_correlation_id: CorrelationId,
    /// Graph node ID in the source graph from which the executable graph was instantiated.
    pub source_graph_node_id: GraphNodeId,
    /// ID of the source graph from which the executable graph was instantiated.
    pub source_graph_id: GraphId,
    /// The unique ID of the graph node that executed this memcpy through graph launch.
    /// This field will be 0 if the memcpy is not done through graph launch.
    pub graph_node_id: GraphNodeId,
    /// The unique ID of the graph that executed this memcpy through graph launch.
    /// This field will be 0 if the memcpy is not done through graph launch.
    pub graph_id: GraphId,
    /// The ID of the HW channel on which the memory copy is occurring.
    pub channel_id: ChannelId,
    /// The type of the channel.
    pub channel_type: ChannelType,
    /// Whether the copy belongs to a device-launched graph.
    pub is_device_launched: bool,
    /// The total number of memcopy operations traced in this record.
    /// This field is valid for memcpy operations happening using `MemcpyBatchAsync` APIs in CUDA.
    /// In `MemcpyBatchAsync` APIs, multiple memcpy operations are batched together for optimization purposes based on certain heuristics.
    /// For other memcpy operations, this field will be 1.
    pub copy_count: u64,
}

impl ActivityMemoryCopy {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityMemcpy7) -> Result<Self> {
        Ok(Self {
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
            channel_type: ChannelType::try_from(value.channelType)?,
            is_device_launched: value.isDeviceLaunched != 0,
            copy_count: value.copyCount,
        })
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
    pub flags: ActivityFlags,
    /// The number of bytes transferred by the memory copy.
    pub bytes: u64,
    /// The start timestamp for the memory copy, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the memory copy.
    pub start: u64,
    /// The end timestamp for the memory copy, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the memory copy.
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
    /// The correlation ID of the memory copy.
    /// Each memory copy is assigned a unique correlation ID that is identical to the correlation ID in the driver and runtime API activity record that launched the memory copy.
    pub correlation_id: CorrelationId,
    /// The unique ID of the graph node that executed the memcpy through graph launch.
    /// This field will be 0 if memcpy is not done using graph launch.
    pub graph_node_id: GraphNodeId,
    /// The unique ID of the graph that executed this memcpy through graph launch.
    /// This field will be 0 if the memcpy is not done through graph launch.
    pub graph_id: GraphId,
    /// The ID of the HW channel on which the memory copy is occurring.
    pub channel_id: ChannelId,
    /// The type of the channel.
    pub channel_type: ChannelType,
}

impl ActivityPeerMemoryCopy {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityMemcpyPtoP4) -> Result<Self> {
        Ok(Self {
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
            channel_type: ChannelType::try_from(value.channelType)?,
        })
    }
}

/// Decoded device memory-set activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityMemorySet {
    /// The value being assigned to memory by the memory set.
    pub value: u32,
    /// The number of bytes being set by the memory set.
    pub bytes: u64,
    /// The start timestamp for the memory set, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the memory set.
    pub start: u64,
    /// The end timestamp for the memory set, in nanoseconds.
    /// A value of 0 for both the start and end timestamps indicates that timestamp information could not be collected for the memory set.
    pub end: u64,
    /// The ID of the device where the memory set is occurring.
    pub device_id: DeviceId,
    /// The ID of the context where the memory set is occurring.
    pub context_id: ContextId,
    /// The ID of the stream where the memory set is occurring.
    pub stream_id: StreamId,
    /// The correlation ID of the memory set.
    /// Each memory set is assigned a unique correlation ID that is identical to the correlation ID in the driver API activity record that launched the memory set.
    pub correlation_id: CorrelationId,
    /// The flags associated with the memset.
    pub flags: ActivityFlags,
    /// The memory kind of the memory set
    ///
    /// See also: [`ActivityMemoryKind`].
    pub memory_kind: ActivityMemoryKind,
    /// Graph node ID in the source graph from which the executable graph was instantiated.
    pub source_graph_node_id: GraphNodeId,
    /// ID of the source graph from which the executable graph was instantiated.
    pub source_graph_id: GraphId,
    /// The unique ID of the graph node that executed this memset through graph launch.
    /// This field will be 0 if the memset is not executed through graph launch.
    pub graph_node_id: GraphNodeId,
    /// The unique ID of the graph that executed this memset through graph launch.
    /// This field will be 0 if the memset is not executed through graph launch.
    pub graph_id: GraphId,
    /// The ID of the HW channel on which the memory set is occurring.
    pub channel_id: ChannelId,
    /// The type of the channel.
    pub channel_type: ChannelType,
    /// Whether the memory set belongs to a device-launched graph.
    pub is_device_launched: bool,
}

impl ActivityMemorySet {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityMemset5) -> Result<Self> {
        Ok(Self {
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
            channel_type: ChannelType::try_from(value.channelType)?,
            is_device_launched: value.isDeviceLaunched != 0,
        })
    }
}
