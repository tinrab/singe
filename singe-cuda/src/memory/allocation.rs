use std::{mem::replace, ptr, sync::Arc};

use singe_core::report_cleanup_failure;
use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    error::Result,
    memory::{DeviceMemory, MemoryPool},
    stream::Stream,
    try_ffi,
    types::DevicePtr,
    view::{AllocationId, DeviceRegion, DeviceRegionProvider},
};

#[derive(Debug)]
pub(crate) enum DeviceDeallocation {
    Immediate,
    StreamOrdered {
        stream: Stream,
        pool: Option<MemoryPool>,
    },
}

/// A queued free owns its deallocation resources without retaining its stream.
#[derive(Debug)]
pub(crate) struct DeferredFree {
    pointer: *mut u8,
    context: Arc<Context>,
    _pool: Option<MemoryPool>,
}

// SAFETY: the host never dereferences `pointer`, and `enqueue` binds the retained context on whichever thread calls it.
unsafe impl Send for DeferredFree {}

// SAFETY: `&DeferredFree` only passes the address to `cudaFreeAsync`; the stream's cleanup lock runs each free once.
unsafe impl Sync for DeferredFree {}

impl DeferredFree {
    pub(crate) fn enqueue(&self, stream: runtime::cudaStream_t) -> Result<()> {
        self.context.bind()?;
        // SAFETY: `pointer` came from a stream-ordered allocation whose final owner released it to this free, and the context is bound.
        unsafe { try_ffi!(runtime::cudaFreeAsync(self.pointer.cast(), stream)) }
    }
}

/// Owns a CUDA allocation independently of its typed buffers and views.
///
/// Retaining this owner keeps the allocation, context, and deallocation resources alive.
/// Raw CUDA work must still satisfy the lifetime and ordering contract of its launch.
#[derive(Debug)]
pub struct DeviceAllocation {
    pub(crate) id: AllocationId,
    pub(crate) context: Arc<Context>,
    pub(crate) ptr: *mut u8,
    pub(crate) bytes: usize,
    pub(crate) deallocation: DeviceDeallocation,
}

// SAFETY: the host never dereferences `ptr`, the retained context and stream are `Send`, and release binds the context on the dropping thread.
unsafe impl Send for DeviceAllocation {}

// SAFETY: `&DeviceAllocation` exposes only immutable metadata; access to the bytes goes through borrows or unsafe raw APIs.
unsafe impl Sync for DeviceAllocation {}

impl DeviceAllocation {
    pub(crate) fn from_parts(
        context: &Arc<Context>,
        ptr: *mut u8,
        bytes: usize,
        deallocation: DeviceDeallocation,
    ) -> Arc<Self> {
        Arc::new(Self {
            id: AllocationId::create(),
            context: Arc::clone(context),
            ptr,
            bytes,
            deallocation,
        })
    }

    /// Returns the identity shared by every buffer and view of this allocation.
    #[must_use]
    pub const fn id(&self) -> AllocationId {
        self.id
    }

    /// Returns the context that owns this allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Returns the allocation's base device address.
    #[must_use]
    pub fn as_device_ptr(&self) -> DevicePtr {
        // SAFETY: `ptr` is this live allocation's base or null, and `DevicePtr` only records the address.
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
    }

    /// Returns the full allocation size, independent of typed views.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.bytes
    }

    /// Returns the allocation and deallocation stream for stream-ordered memory.
    #[must_use]
    pub fn stream(&self) -> Option<&Stream> {
        match &self.deallocation {
            DeviceDeallocation::Immediate => None,
            DeviceDeallocation::StreamOrdered { stream, .. } => Some(stream),
        }
    }

    /// Returns the explicitly selected pool retained by this allocation.
    #[must_use]
    pub fn pool(&self) -> Option<&MemoryPool> {
        match &self.deallocation {
            DeviceDeallocation::Immediate => None,
            DeviceDeallocation::StreamOrdered { pool, .. } => pool.as_ref(),
        }
    }

    pub(crate) fn release(&mut self) -> Result<()> {
        if self.ptr.is_null() {
            return Ok(());
        }
        match &self.deallocation {
            DeviceDeallocation::Immediate => {
                self.context.bind()?;
                // SAFETY: `release` runs once, from `Drop` of the last `Arc` owner, so no view or submission still uses `ptr`.
                unsafe {
                    DeviceMemory::<u8>::free(&self.context, self.ptr)?;
                }

                self.ptr = ptr::null_mut();
                Ok(())
            }
            DeviceDeallocation::StreamOrdered { stream, pool } => {
                let free = DeferredFree {
                    pointer: replace(&mut self.ptr, ptr::null_mut()),
                    context: Arc::clone(&self.context),
                    _pool: pool.clone(),
                };
                // The stream keeps failed frees and captured frees until they can be enqueued.
                stream.defer_free(free)
            }
        }
    }
}

// SAFETY: the owner retains `bytes` bytes at `ptr` and its context for its lifetime, under one allocation identity.
unsafe impl DeviceRegionProvider for DeviceAllocation {
    fn region(&self) -> DeviceRegion<'_> {
        // SAFETY: this owner retains the complete allocation and its context.
        let region =
            unsafe { DeviceRegion::from_raw_parts(&self.context, self.id, self.ptr, self.bytes) }
                .expect("invalid device allocation metadata");
        self.stream().map_or(region, |stream| {
            // SAFETY: a stream-ordered allocation is allocated on this stream, so it orders the allocation.
            unsafe { region.with_stream(stream) }.expect("allocation stream context mismatch")
        })
    }
}

impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        if let Err(error) = self.release() {
            report_cleanup_failure("release", "cuda device allocation", &error);
        }
    }
}
