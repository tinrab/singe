use std::{marker::PhantomData, mem::MaybeUninit, ptr, sync::Arc};

use singe_cuda_sys::runtime;

use crate::{
    error::{Error, Result},
    memory::{DeviceAllocation, DeviceDeallocation, DeviceMemory, MemoryPool},
    stream::Stream,
    try_ffi,
    utility::checked_bytes,
    view::{DeviceRepr, ZeroableDeviceRepr},
};

impl<T> DeviceMemory<T> {
    /// Allocates stream-ordered device memory on `stream` for a raw producer to initialize.
    ///
    /// The last owner enqueues `cudaFreeAsync` on `stream` when it drops.
    ///
    /// # Safety
    ///
    /// Every element must be initialized to a valid `T` before it is read.
    /// Consumers on other streams must wait for allocation and producer work.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, the stream's context cannot be bound, CUDA cannot enqueue the allocation, or `length` is non-zero for a zero-sized `T`.
    pub unsafe fn create_stream_ordered_unchecked(length: usize, stream: &Stream) -> Result<Self> {
        let _lifecycle = stream.allocation_guard()?;
        stream.context().bind()?;
        let bytes = checked_bytes::<T>(length, "byte length")?;
        let ptr = if bytes == 0 {
            ptr::null_mut()
        } else {
            // SAFETY: the allocation is owned below and freed on `stream`, and this function's caller initializes it before reads.
            unsafe { Self::alloc_async(length, stream)? }
        };
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                stream.context(),
                ptr.cast(),
                bytes,
                DeviceDeallocation::StreamOrdered {
                    stream: stream.clone(),
                    pool: None,
                },
            ),
            ptr,
            length,
            _phantom: PhantomData,
        })
    }

    /// Allocates stream-ordered device memory from `pool` on `stream` for a raw producer to initialize.
    ///
    /// The last owner enqueues `cudaFreeAsync` on `stream` when it drops, and retains `pool` until then.
    ///
    /// # Safety
    ///
    /// Every element must be initialized to a valid `T` before it is read.
    /// Consumers on other streams must wait for allocation and producer work.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, the stream's context cannot be bound, CUDA cannot enqueue the pool allocation, CUDA returns a null pointer for a non-empty request, or `length` is non-zero for a zero-sized `T`.
    pub unsafe fn create_in_pool_unchecked(
        length: usize,
        pool: &MemoryPool,
        stream: &Stream,
    ) -> Result<Self> {
        let _lifecycle = stream.allocation_guard()?;
        stream.context().bind()?;
        let bytes = checked_bytes::<T>(length, "byte length")?;
        let mut ptr = ptr::null_mut();
        if bytes != 0 {
            // SAFETY: `ptr` is a local output, `bytes` is a checked nonzero size, and `pool` and `stream` are borrowed live in the bound context.
            unsafe {
                try_ffi!(runtime::cudaMallocFromPoolAsync(
                    &raw mut ptr,
                    bytes as _,
                    pool.as_raw(),
                    stream.as_raw()
                ))?;
            }

            if ptr.is_null() {
                return Err(Error::NullHandle);
            }
        }

        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                stream.context(),
                ptr.cast(),
                bytes,
                DeviceDeallocation::StreamOrdered {
                    stream: stream.clone(),
                    pool: Some(pool.clone()),
                },
            ),
            ptr: ptr.cast(),
            length,
            _phantom: PhantomData,
        })
    }

    /// Enqueues an uninitialized allocation on `stream` from CUDA's current pool.
    ///
    /// Access on other streams must be ordered after the allocation stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is requested, stream or context setup fails, or CUDA cannot enqueue the allocation.
    pub fn create_uninit_stream_ordered(
        length: usize,
        stream: &Stream,
    ) -> Result<DeviceMemory<MaybeUninit<T>>>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `MaybeUninit<T>` elements need no initialization.
        unsafe { DeviceMemory::<MaybeUninit<T>>::create_stream_ordered_unchecked(length, stream) }
    }

    /// Enqueues an uninitialized allocation from `pool` on `stream`.
    ///
    /// The allocation retains both the pool and the stream until it is freed.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is requested, or stream or context setup fails.
    /// Also propagates CUDA errors for the requested pool allocation and rejects a null handle for a nonempty allocation.
    pub fn create_uninit_in_pool(
        length: usize,
        pool: &MemoryPool,
        stream: &Stream,
    ) -> Result<DeviceMemory<MaybeUninit<T>>>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `MaybeUninit<T>` elements need no initialization.
        unsafe { DeviceMemory::<MaybeUninit<T>>::create_in_pool_unchecked(length, pool, stream) }
    }

    /// Returns the stream a stream-ordered allocation was made on and is freed on, or `None` for an allocation freed with `cudaFree`.
    #[must_use]
    pub fn stream(&self) -> Option<&Stream> {
        self.allocation.stream()
    }

    /// Returns the pool a stream-ordered allocation came from, if one was selected.
    #[must_use]
    pub fn pool(&self) -> Option<&MemoryPool> {
        self.allocation.pool()
    }

    /// Releases the allocation now when this is its final owner.
    ///
    /// Memory freed with `cudaFree` is freed before this returns; a stream-ordered allocation enqueues `cudaFreeAsync` on its stream.
    /// Calling this surfaces the errors that dropping the owner can only report.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OwnershipShared`] if a view, submission, graph, or interop user retains the allocation, which then stays live until its final owner drops.
    /// Otherwise returns an error if the context cannot be bound or CUDA cannot free or enqueue the free.
    pub fn release(self) -> Result<()> {
        let mut allocation = Arc::try_unwrap(self.allocation)
            .map_err(|_| Error::ownership_shared("device allocation"))?;
        allocation.release()
    }
}

impl<T: ZeroableDeviceRepr> DeviceMemory<T> {
    /// Allocates stream-ordered device memory on `stream` and enqueues a fill with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails or CUDA cannot enqueue the memset.
    pub fn zeroed_stream_ordered(length: usize, stream: &Stream) -> Result<Self> {
        let () = T::LAYOUT;
        // SAFETY: the fill below initializes every element on the allocation stream before any later consumer.
        let memory = unsafe { Self::create_stream_ordered_unchecked(length, stream)? };
        // SAFETY: the fill covers the `memory.length` owned elements on the allocation's own stream, which orders it before the free, and `T: ZeroableDeviceRepr` makes zero bytes valid.
        unsafe {
            Self::set_async(memory.ptr, 0, memory.length, stream)?;
        }
        Ok(memory)
    }

    /// Allocates stream-ordered device memory from `pool` on `stream` and enqueues a fill with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation from the pool fails or CUDA cannot enqueue the memset.
    pub fn zeroed_in_pool(length: usize, pool: &MemoryPool, stream: &Stream) -> Result<Self> {
        let () = T::LAYOUT;
        // SAFETY: the fill below initializes every element on the allocation stream before any later consumer.
        let memory = unsafe { Self::create_in_pool_unchecked(length, pool, stream)? };
        // SAFETY: the fill covers the `memory.length` owned elements on the allocation's own stream, which orders it before the free, and `T: ZeroableDeviceRepr` makes zero bytes valid.
        unsafe {
            Self::set_async(memory.ptr, 0, memory.length, stream)?;
        }
        Ok(memory)
    }
}

/// The stream that frees a stream-ordered allocation, and the pool it came from.
#[derive(Debug)]
pub struct StreamOrderedFree {
    /// Stream the allocation was made on and is freed on.
    pub stream: Stream,
    /// Pool the allocation came from, if any.
    pub pool: Option<MemoryPool>,
}
