use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{self},
    result::Result as StdResult,
    sync::Arc,
};

use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    error::{Error, Result},
    stream::Stream,
    submission::{AccessMode, Execution, Resources},
    try_ffi,
    types::DevicePtr,
    view::{
        AnyBitPatternDeviceRepr, DeviceOutputView, DeviceRegion, DeviceRegionProvider, DeviceRepr,
        DeviceView, DeviceViewMut, NoUninitDeviceRepr, ZeroableDeviceRepr, cast_length,
    },
};

use super::{
    AsyncDeviceMemory, DeviceAllocation, DeviceDeallocation, DeviceMemory, MemoryPool,
    checked_bytes,
};

impl<T> AsyncDeviceMemory<T> {
    /// Allocates stream-ordered device memory on `stream`.
    ///
    /// The allocation is freed by enqueuing `cudaFreeAsync` on the stored stream when this owner is dropped, unless [`AsyncDeviceMemory::free_async`] is called first.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, CUDA cannot enqueue the allocation, or `length` is non-zero for a zero-sized `T`.
    ///
    /// # Safety
    ///
    /// Every element must be initialized to a valid T before it is read.
    /// Consumers on other streams must wait for allocation and producer work.
    pub unsafe fn create_unchecked(length: usize, stream: &Stream) -> Result<Self> {
        let _lifecycle = stream.allocation_guard()?;
        stream.context().bind()?;
        let bytes = checked_bytes::<T>(length)?;
        let ptr = if bytes == 0 {
            ptr::null_mut()
        } else {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { DeviceMemory::<T>::alloc_async(length, stream)? }
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

    /// Allocates stream-ordered device memory from `pool` on `stream`.
    ///
    /// The allocation is freed by enqueuing `cudaFreeAsync` on the stored stream when this owner is dropped, unless [`AsyncDeviceMemory::free_async`] is called first.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, CUDA cannot enqueue the pool allocation, CUDA returns a null pointer for a non-empty request, or `length` is non-zero for a zero-sized `T`.
    ///
    /// # Safety
    ///
    /// Every element must be initialized before it is read.
    /// Consumers on other streams must wait for allocation and producer work.
    /// The returned buffer retains the pool; CUDA releases its backing resources after outstanding frees finish.
    pub unsafe fn create_in_pool_unchecked(
        length: usize,
        pool: &MemoryPool,
        stream: &Stream,
    ) -> Result<Self> {
        let _lifecycle = stream.allocation_guard()?;
        stream.context().bind()?;
        let bytes = checked_bytes::<T>(length)?;
        let mut ptr = ptr::null_mut();
        if bytes != 0 {
            // SAFETY: `cudaMallocFromPoolAsync` receives live handles owned by the wrappers and valid output locations.
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

    /// Takes ownership of a stream-ordered CUDA allocation.
    ///
    /// # Safety
    ///
    /// `ptr` must be null for an empty allocation or point to `length` live elements allocated by `cudaMallocAsync` or `cudaMallocFromPoolAsync`.
    /// The allocation must be valid to free on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is described, or a nonempty allocation has a null pointer.
    pub unsafe fn from_raw_parts(
        ptr: *mut T,
        length: usize,
        stream: &Stream,
        pool: Option<MemoryPool>,
    ) -> Result<Self> {
        let bytes = checked_bytes::<T>(length)?;
        if bytes != 0 && ptr.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                stream.context(),
                ptr.cast(),
                bytes,
                DeviceDeallocation::StreamOrdered {
                    stream: stream.clone(),
                    pool,
                },
            ),
            ptr,
            length,
            _phantom: PhantomData,
        })
    }

    /// Enqueues the asynchronous free when this buffer holds the final allocation owner.
    ///
    /// Calling this method surfaces CUDA errors that would otherwise only be logged in debug builds from `Drop`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationInUse`] if a submission, graph, or interop user retains the allocation.
    /// Otherwise returns an error if CUDA cannot enqueue `cudaFreeAsync` for the stored stream.
    /// A retained allocation remains live after `AllocationInUse` and is released by its final owner.
    pub fn free_async(self) -> Result<()> {
        let mut allocation =
            Arc::try_unwrap(self.allocation).map_err(|_| Error::AllocationInUse)?;
        allocation.release()
    }

    /// Returns the number of elements in this allocation.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether this allocation contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the logical allocation size in bytes.
    ///
    /// # Panics
    ///
    /// Panics if the byte-size calculation overflows.
    /// Safe constructors reject such allocations.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        checked_bytes::<T>(self.length).expect("async device memory byte length overflow")
    }

    /// Returns the base device pointer.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the mutable base device pointer.
    #[must_use]
    pub const fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Returns the base pointer as a CUDA device pointer wrapper.
    #[must_use]
    pub const fn as_device_ptr(&self) -> DevicePtr {
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
    }

    /// Returns the CUDA context retained by this allocation or pool.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Borrows all initialized elements as a shared device view.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_, T>
    where
        T: DeviceRepr,
    {
        DeviceView::from_buffer(self)
    }

    /// Borrows all initialized elements as an exclusive device view.
    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T>
    where
        T: DeviceRepr,
    {
        DeviceViewMut::from_buffer(self)
    }

    /// Enqueues an uninitialized allocation on `stream` using CUDA’s current pool.
    ///
    /// Access on other streams must be ordered after the allocation stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is requested, stream or context setup fails, or CUDA cannot enqueue the allocation.
    pub fn create_uninit(
        length: usize,
        stream: &Stream,
    ) -> Result<AsyncDeviceMemory<MaybeUninit<T>>>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `MaybeUninit<T>` elements need no initialization.
        unsafe { AsyncDeviceMemory::<MaybeUninit<T>>::create_unchecked(length, stream) }
    }

    /// Enqueues an uninitialized allocation from `pool` on `stream`.
    ///
    /// The allocation retains both the pool and the stream until it can be freed.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is requested, or stream or context setup fails.
    /// Also propagates CUDA errors for the requested pool allocation and rejects a null handle for a nonempty allocation.
    pub fn create_uninit_in_pool(
        length: usize,
        pool: &MemoryPool,
        stream: &Stream,
    ) -> Result<AsyncDeviceMemory<MaybeUninit<T>>>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `MaybeUninit<T>` elements need no initialization.
        unsafe {
            AsyncDeviceMemory::<MaybeUninit<T>>::create_in_pool_unchecked(length, pool, stream)
        }
    }

    /// Returns the allocation owner retained by submissions, graphs, and interop users.
    #[must_use]
    pub fn allocation(&self) -> &Arc<DeviceAllocation> {
        &self.allocation
    }

    /// Reinterprets initialized elements while preserving the stream and pool owner.
    ///
    /// # Errors
    ///
    /// Returns the original allocation if `U` is zero-sized, its size does not divide the exposed byte length, or a nonempty range is not aligned for `U`.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<AsyncDeviceMemory<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let Ok(length) = cast_length::<U>(self.region()) else {
            return Err(self);
        };
        Ok(AsyncDeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length,
            _phantom: PhantomData,
        })
    }

    /// Erases element initialization without changing allocation identity or stream ordering.
    #[must_use]
    pub fn into_uninit(self) -> AsyncDeviceMemory<MaybeUninit<T>> {
        AsyncDeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
            _phantom: PhantomData,
        }
    }

    /// Returns the explicitly selected pool, if any.
    #[must_use]
    pub fn pool(&self) -> Option<&MemoryPool> {
        self.allocation.pool()
    }

    /// Returns the stream used for allocation and freeing.
    ///
    /// # Panics
    ///
    /// Panics if the allocation has no associated stream.
    /// All constructors of this stream-ordered wrapper retain a stream, so this cannot occur for a valid instance.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        self.allocation
            .stream()
            .expect("stream-ordered allocation has no stream")
    }

    /// Enqueues a byte fill of this allocation on its stored stream.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the memset.
    pub fn set_value_async(&mut self, value: u8) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        let mut execution = Execution::create(self.stream())?;
        let mut resources = Resources::new();
        // This allocation is freed on the same retained stream, after the fill and its event.
        // Other-stream consumers retain their own allocation owners before enqueueing.
        resources.access(self.region(), AccessMode::Write);
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            execution.enqueue(resources, |stream| {
                DeviceMemory::<T>::set_async(self.ptr, value, self.length, stream)
            })?;
        }

        execution.finish(())?;
        Ok(())
    }
}

impl<T: ZeroableDeviceRepr> AsyncDeviceMemory<T> {
    /// Allocates stream-ordered device memory and fills it with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails or CUDA cannot enqueue the memset.
    pub fn zeroed(length: usize, stream: &Stream) -> Result<Self> {
        let () = T::LAYOUT;
        // SAFETY: the memory is fully initialized below before it is returned.
        let memory = unsafe { Self::create_unchecked(length, stream)? };
        // SAFETY: the allocation covers `memory.length` elements, and `T: ZeroableDeviceRepr` makes the zero fill valid.
        unsafe {
            DeviceMemory::<T>::set_async(memory.ptr, 0, memory.length, memory.stream())?;
        }
        Ok(memory)
    }

    /// Allocates stream-ordered device memory from `pool` and fills it with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation from the pool fails or CUDA cannot enqueue the memset.
    pub fn zeroed_in_pool(length: usize, pool: &MemoryPool, stream: &Stream) -> Result<Self> {
        let () = T::LAYOUT;
        // SAFETY: the memory is fully initialized below before it is returned.
        let memory = unsafe { Self::create_in_pool_unchecked(length, pool, stream)? };
        // SAFETY: the allocation covers `memory.length` elements, and `T: ZeroableDeviceRepr` makes the zero fill valid.
        unsafe {
            DeviceMemory::<T>::set_async(memory.ptr, 0, memory.length, memory.stream())?;
        }
        Ok(memory)
    }
}

unsafe impl<T> DeviceRegionProvider for AsyncDeviceMemory<T> {
    fn region(&self) -> DeviceRegion<'_> {
        let offset = self.ptr as usize - self.allocation.ptr as usize;
        self.allocation
            .region()
            .slice(offset..offset + self.byte_len())
            .expect("async buffer exceeds its allocation")
    }
}

impl<T: DeviceRepr> AsyncDeviceMemory<MaybeUninit<T>> {
    /// Borrows this stream-ordered allocation as a write-only output of `T` values.
    pub fn output(&mut self) -> DeviceOutputView<'_, T> {
        DeviceOutputView::from_output(self)
    }

    /// Exposes a producer-initialized prefix while retaining the full allocation and its stream.
    ///
    /// # Safety
    ///
    /// The prefix must contain valid initialized `T` values, with producing work ordered before all consumers.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` exceeds the currently exposed element count.
    pub unsafe fn assume_init_prefix(mut self, length: usize) -> Result<AsyncDeviceMemory<T>> {
        if length > self.length {
            return Err(Error::InvalidMemoryAccess);
        }

        self.length = length;
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks the entire allocation initialized without changing its allocation stream.
    ///
    /// # Safety
    ///
    /// Every element must contain a valid T, with producer writes ordered before every consumer.
    #[must_use]
    pub unsafe fn assume_init(self) -> AsyncDeviceMemory<T> {
        let () = T::LAYOUT;
        AsyncDeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
            _phantom: PhantomData,
        }
    }
}
