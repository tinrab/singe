//! Stream memory synchronization using CUDA-visible device addresses.
//!
//! These operations do not establish CUDA scheduler dependencies.
//! Express producer/consumer order with events as well to avoid scheduler deadlocks.
//! Managed allocations are unsupported; registered host memory needs its mapped device address.

use std::fmt;

use bitflags::parser;
use singe_cuda_sys::driver as sys;

use crate::{
    error::{Error, Result},
    graph::owned::{Recordable, Recording},
    stream::Stream,
    submission::{DeviceOperation, Execution, Operation, Resources},
    try_ffi,
    types::DevicePtr,
    utility::{driver_enum, to_u32},
};

driver_enum! {
    /// Condition used when a stream waits on a 32-bit or 64-bit memory value.
    pub enum WaitCondition: sys::CUstreamWaitValue_flags {
        /// Cyclic signed comparison: `(signed)(current - value) >= 0`.
        GreaterOrEqual = CU_STREAM_WAIT_VALUE_GEQ,
        /// Wait until \*addr == value.
        Equal = CU_STREAM_WAIT_VALUE_EQ,
        /// Wait until (\*addr & value) != 0.
        And = CU_STREAM_WAIT_VALUE_AND,
        /// Wait until ~(\*addr | value) != 0.
        ///
        /// Support for this operation can be queried with `cuDeviceGetAttribute` and `CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR`.
        Nor = CU_STREAM_WAIT_VALUE_NOR,
    }
}

bitflags::bitflags! {
    /// Flags for cuStreamWaitValue32 and cuStreamWaitValue64.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct WaitFlags: u32 {
        /// Follow the wait operation with a flush of outstanding remote writes.
        ///
        /// This means that, if a remote write operation is guaranteed to have reached the device before the wait can be satisfied, that write is guaranteed to be visible to downstream device work. The device is permitted to reorder remote writes internally. For example, this flag would be required if two remote writes arrive in a defined order, the wait is satisfied by the second write, and downstream work needs to observe the first write. Support for this operation is restricted to selected platforms and can be queried with `CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES`.
        const FLUSH_REMOTE_WRITES = sys::CUstreamWaitValue_flags::CU_STREAM_WAIT_VALUE_FLUSH as _;
    }
}

impl fmt::Display for WaitFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// Scope of a barrier in a CUDA stream memory-operation batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryBarrierScope {
    /// Uses a system-scoped memory barrier.
    System,
    /// Uses a GPU-scoped memory barrier.
    Device,
}

/// One ordered memory operation; addresses are borrowed and confer no access permission.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum MemoryOperation {
    /// Waits for a 32-bit device word to satisfy the selected comparison.
    Wait32 {
        /// Address of the device word; must be nonnull and aligned to 4 bytes.
        address: DevicePtr,
        /// 32-bit reference value used by `condition`.
        value: u32,
        /// Comparison of the current device word against `value`.
        condition: WaitCondition,
        /// Additional remote-write visibility behavior after the wait.
        flags: WaitFlags,
    },
    /// Waits for a 64-bit device word to satisfy the selected comparison.
    Wait64 {
        /// Address of the device word; must be nonnull and aligned to 8 bytes.
        address: DevicePtr,
        /// 64-bit reference value used by `condition`.
        value: u64,
        /// Comparison of the current device word against `value`.
        condition: WaitCondition,
        /// Additional remote-write visibility behavior after the wait.
        flags: WaitFlags,
    },
    /// Writes a 32-bit device word with the native stream memory fence enabled.
    Write32 {
        /// Device destination address; must be nonnull and aligned to 4 bytes.
        address: DevicePtr,
        /// 32-bit value written to the device word.
        value: u32,
    },
    /// Writes a 64-bit device word with the native stream memory fence enabled.
    Write64 {
        /// Device destination address; must be nonnull and aligned to 8 bytes.
        address: DevicePtr,
        /// 64-bit value written to the device word.
        value: u64,
    },
    /// Enqueues a memory barrier with the selected scope.
    Barrier(MemoryBarrierScope),
    /// Flushes outstanding remote writes so they are visible to downstream device work.
    FlushRemoteWrites,
}

impl MemoryOperation {
    pub(crate) fn native(self) -> Result<sys::CUstreamBatchMemOpParams> {
        use sys::CUstreamBatchMemOpType as Kind;
        let mut raw = sys::CUstreamBatchMemOpParams::default();
        match self {
            Self::Wait32 {
                address,
                value,
                condition,
                flags,
            } => {
                check_address(address, 4)?;
                raw.waitValue.operation = Kind::CU_STREAM_MEM_OP_WAIT_VALUE_32;
                raw.waitValue.address = address.as_raw() as _;
                raw.waitValue.__bindgen_anon_1.value = value;
                raw.waitValue.flags = condition as u32 | flags.bits();
            }
            Self::Wait64 {
                address,
                value,
                condition,
                flags,
            } => {
                check_address(address, 8)?;
                raw.waitValue.operation = Kind::CU_STREAM_MEM_OP_WAIT_VALUE_64;
                raw.waitValue.address = address.as_raw() as _;
                raw.waitValue.__bindgen_anon_1.value64 = value;
                raw.waitValue.flags = condition as u32 | flags.bits();
            }
            Self::Write32 { address, value } => {
                check_address(address, 4)?;
                raw.writeValue.operation = Kind::CU_STREAM_MEM_OP_WRITE_VALUE_32;
                raw.writeValue.address = address.as_raw() as _;
                raw.writeValue.__bindgen_anon_1.value = value;
            }
            Self::Write64 { address, value } => {
                check_address(address, 8)?;
                raw.writeValue.operation = Kind::CU_STREAM_MEM_OP_WRITE_VALUE_64;
                raw.writeValue.address = address.as_raw() as _;
                raw.writeValue.__bindgen_anon_1.value64 = value;
            }
            Self::Barrier(scope) => {
                raw.memoryBarrier.operation = Kind::CU_STREAM_MEM_OP_BARRIER;
                raw.memoryBarrier.flags = match scope {
                    MemoryBarrierScope::System => {
                        sys::CUstreamMemoryBarrier_flags::CU_STREAM_MEMORY_BARRIER_TYPE_SYS as _
                    }
                    MemoryBarrierScope::Device => {
                        sys::CUstreamMemoryBarrier_flags::CU_STREAM_MEMORY_BARRIER_TYPE_GPU as _
                    }
                };
            }
            Self::FlushRemoteWrites => {
                raw.flushRemoteWrites.operation = Kind::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES;
            }
        }
        Ok(raw)
    }
}

fn check_address(address: DevicePtr, alignment: usize) -> Result<()> {
    let address = address.as_raw().addr();
    if address == 0 || !address.is_multiple_of(alignment) {
        return Err(Error::InvalidMemoryAccess);
    }
    Ok(())
}

pub(crate) fn native_batch(
    operations: &[MemoryOperation],
) -> Result<Vec<sys::CUstreamBatchMemOpParams>> {
    // CUDA's batch count is strictly less than 256, including graph nodes.
    if operations.len() >= 256 {
        return Err(Error::out_of_range("stream memory operation count"));
    }
    operations
        .iter()
        .copied()
        .map(MemoryOperation::native)
        .collect()
}

impl Stream {
    /// Enqueues an ordered batch of at most 255 memory operations.
    ///
    /// # Safety
    /// Addresses must be live, naturally aligned, non-managed memory accessible in this context.
    /// Keep storage live and exclude conflicting accesses until completion or every captured replay.
    /// Waits must be satisfiable without circular dependencies, with event ordering visible to CUDA.
    ///
    /// # Errors
    ///
    /// Returns `Error::OutOfRange` for 256 or more operations, or `Error::InvalidMemoryAccess` for a null or misaligned wait or write address. Also propagates context binding and native batch errors, including unsupported operations or device capabilities. An empty batch succeeds without binding the context or calling CUDA.
    pub unsafe fn memory_operations(&self, operations: &[MemoryOperation]) -> Result<()> {
        let mut native = native_batch(operations)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.enqueue_memory_operations(&mut native) }
    }

    // The batch is validated on construction; CUDA copies the descriptors during this call.
    unsafe fn enqueue_memory_operations(
        &self,
        native: &mut [sys::CUstreamBatchMemOpParams],
    ) -> Result<()> {
        if native.is_empty() {
            return Ok(());
        }
        self.context().bind()?;
        // SAFETY: `cuStreamBatchMemOp_v2` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cuStreamBatchMemOp_v2(
                self.as_raw(),
                to_u32(native.len(), "stream memory operation count")?,
                native.as_mut_ptr(),
                0
            ))
        }
    }
}

macro_rules! word_operations {
    ($wait:ident, $write:ident, $word:ty, $native_wait:ident, $native_write:ident) => {
        impl Stream {
            /// Waits for a naturally aligned device word to satisfy the condition.
            ///
            /// # Safety
            /// Follow [`Stream::memory_operations`]'s storage, ordering and progress requirements.
            ///
            /// # Errors
            /// Returns `Error::InvalidMemoryAccess` for a null address or one not aligned
            /// to the native word size. Also propagates context binding and native wait
            /// errors for unsupported conditions, flags, or device capabilities.
            pub unsafe fn $wait(
                &self,
                address: DevicePtr,
                value: $word,
                condition: WaitCondition,
                flags: WaitFlags,
            ) -> Result<()> {
                check_address(address, size_of::<$word>())?;
                self.context().bind()?;
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    try_ffi!(sys::$native_wait(
                        self.as_raw(),
                        address.as_raw() as _,
                        value,
                        condition as u32 | flags.bits()
                    ))
                }
            }

            /// Writes a naturally aligned device word with the native stream memory fence.
            ///
            /// # Safety
            /// Follow [`Stream::memory_operations`]'s storage and ordering requirements.
            ///
            /// # Errors
            /// Returns `Error::InvalidMemoryAccess` for a null address or one not aligned
            /// to the native word size. Also propagates context binding and native write
            /// errors, including unsupported memory operations on the device.
            pub unsafe fn $write(&self, address: DevicePtr, value: $word) -> Result<()> {
                check_address(address, size_of::<$word>())?;
                self.context().bind()?;
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    try_ffi!(sys::$native_write(
                        self.as_raw(),
                        address.as_raw() as _,
                        value,
                        0
                    ))
                }
            }
        }
    };
}

word_operations!(
    wait_value32,
    write_value32,
    u32,
    cuStreamWaitValue32_v2,
    cuStreamWriteValue32_v2
);

word_operations!(
    wait_value64,
    write_value64,
    u64,
    cuStreamWaitValue64_v2,
    cuStreamWriteValue64_v2
);

/// Prepared memory operations using the common resource ledger for submission or capture.
pub struct MemoryOperations {
    operations: Vec<sys::CUstreamBatchMemOpParams>,
    resources: Resources,
}

impl fmt::Debug for MemoryOperations {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryOperations")
            .field("operation_count", &self.operations.len())
            .field("resources", &self.resources)
            .finish()
    }
}

impl MemoryOperations {
    /// Validates a batch and binds its retained owners and access declarations.
    ///
    /// # Safety
    /// Resources must retain all referenced storage and declare every access.
    /// Uphold [`Stream::memory_operations`]'s progress and native memory restrictions.
    ///
    /// # Errors
    ///
    /// Returns `Error::OutOfRange` for 256 or more operations, or `Error::InvalidMemoryAccess` for a null or misaligned wait or write address. Resource ownership, access declarations, and progress guarantees remain the caller's responsibility; no native work is enqueued during construction.
    pub unsafe fn create(operations: &[MemoryOperation], resources: Resources) -> Result<Self> {
        Ok(Self {
            operations: native_batch(operations)?,
            resources,
        })
    }
}

// SAFETY: caller-established resources are retained before the native batch is enqueued.
unsafe impl Operation for MemoryOperations {
    type Pending = ();

    type Output = ();

    fn enqueue(mut self, execution: &mut Execution) -> Result<()> {
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            execution.enqueue(self.resources, |stream| {
                stream.enqueue_memory_operations(&mut self.operations)
            })
        }
    }

    unsafe fn resolve(pending: ()) -> Result<()> {
        Ok(pending)
    }
}

// SAFETY: unit resolution is the identity and all work is enqueued eagerly.
unsafe impl DeviceOperation for MemoryOperations {}

// SAFETY: only the native graph-compatible batch is recorded, with replay resources retained.
unsafe impl Recordable for MemoryOperations {
    type Output = ();

    fn record(mut self, recording: &mut Recording) -> Result<()> {
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            recording.enqueue(self.resources, |stream| {
                stream.enqueue_memory_operations(&mut self.operations)
            })
        }
    }
}
