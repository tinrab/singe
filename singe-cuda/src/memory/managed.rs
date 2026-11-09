use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{self},
    result::Result as StdResult,
    slice,
    sync::Arc,
};

use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    device::DeviceAttribute,
    error::{Error, Result},
    stream::{Stream, StreamBinding},
    try_ffi,
    types::{DevicePtr, MemoryAdvise},
    view::{
        AnyBitPatternDeviceRepr, DeviceInput, DeviceOutputView, DeviceRegion, DeviceRegionProvider,
        DeviceRepr, DeviceSlice, DeviceSliceMut, DeviceView, DeviceViewMut, NoUninitDeviceRepr,
        ZeroableDeviceRepr, cast_length,
    },
};

use super::{
    DeviceAllocation, DeviceDeallocation, DeviceMemory, ManagedMemory, MemoryAttachFlags,
    MemoryLocation, checked_bytes, location_info,
};

impl<T: DeviceRepr> ManagedMemory<T> {
    /// Allocates CUDA managed memory under a caller-established initialization contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, CUDA cannot allocate managed memory, or a non-empty zero-sized allocation is requested.
    ///
    /// # Safety
    ///
    /// Every element must be initialized to a valid T before host or device reads.
    pub unsafe fn create_unchecked(
        context: &Arc<Context>,
        length: usize,
        attach_flags: MemoryAttachFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        context.bind()?;
        let bytes = checked_bytes::<T>(length)?;
        let ptr = if bytes == 0 {
            ptr::null_mut()
        } else {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { DeviceMemory::<T>::alloc_managed(length, attach_flags)? }
        };
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                context,
                ptr.cast(),
                bytes,
                DeviceDeallocation::Immediate,
            ),
            ptr,
            length,
            attach_flags,
            _phantom: PhantomData,
        })
    }

    /// Allocates CUDA managed memory and completes zero initialization.
    ///
    /// # Errors
    ///
    /// Propagates allocation, byte-size, access-tracking, context-binding, CUDA fill, and synchronization errors.
    pub fn zeroed(
        context: &Arc<Context>,
        length: usize,
        attach_flags: MemoryAttachFlags,
    ) -> Result<Self>
    where
        T: ZeroableDeviceRepr,
    {
        // SAFETY: the memory is fully initialized below before it is returned.
        let mut memory = unsafe { Self::create_unchecked(context, length, attach_flags)? };
        memory.set_zeroes()?;
        Ok(memory)
    }

    /// Takes ownership of an existing managed allocation.
    ///
    /// # Safety
    ///
    /// `ptr` must be null for an empty allocation or identify live managed storage for `length` elements in `context`, allocated with `cudaMallocManaged`.
    /// The attachment flags must describe that allocation's policy.
    /// No other owner may free it, and each element must be initialized before it is read.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte length overflows or a nonempty allocation has a null pointer.
    pub unsafe fn from_raw_parts(
        context: &Arc<Context>,
        ptr: *mut T,
        length: usize,
        attach_flags: MemoryAttachFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_bytes::<T>(length)?;
        if bytes != 0 && ptr.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                context,
                ptr.cast(),
                bytes,
                DeviceDeallocation::Immediate,
            ),
            ptr,
            length,
            attach_flags,
            _phantom: PhantomData,
        })
    }

    /// Transfers the allocation when no submission, graph, or foreign owner retains it.
    ///
    /// Returns the unchanged buffer if the allocation is shared.
    /// On success the caller must free the pointer and keep its returned context alive until then.
    ///
    /// # Errors
    ///
    /// Returns the original wrapper if another owner, submission, graph, or foreign handle retains the allocation.
    pub fn into_raw_parts(
        self,
    ) -> StdResult<(*mut T, usize, MemoryAttachFlags, Arc<Context>), Self> {
        let Self {
            allocation,
            ptr,
            length,
            attach_flags,
            ..
        } = self;
        match Arc::try_unwrap(allocation) {
            Ok(mut allocation) => {
                let context = Arc::clone(allocation.context());
                allocation.ptr = ptr::null_mut();
                Ok((ptr, length, attach_flags, context))
            }
            Err(allocation) => Err(Self {
                allocation,
                ptr,
                length,
                attach_flags,
                _phantom: PhantomData,
            }),
        }
    }

    /// Returns the CUDA context retained by this allocation or pool.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Returns the allocation owner retained by submissions, graphs, and interop users.
    #[must_use]
    pub fn allocation(&self) -> &Arc<DeviceAllocation> {
        &self.allocation
    }

    /// Reinterprets initialized elements while preserving managed allocation ownership and attachment flags.
    ///
    /// # Errors
    ///
    /// Returns the original allocation if `U` is zero-sized, its size does not divide the exposed byte length, or a nonempty range is not aligned for `U`.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<ManagedMemory<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let Ok(length) = cast_length::<U>(self.region()) else {
            return Err(self);
        };
        Ok(ManagedMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length,
            attach_flags: self.attach_flags,
            _phantom: PhantomData,
        })
    }

    /// Erases initialization while retaining the managed allocation and its attachment flags.
    #[must_use]
    pub fn into_uninit(self) -> ManagedMemory<MaybeUninit<T>> {
        ManagedMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
            attach_flags: self.attach_flags,
            _phantom: PhantomData,
        }
    }

    fn bind_stream(&self, stream: &Stream) -> Result<()> {
        if stream.context() != self.context() {
            return Err(Error::StreamContextMismatch);
        }
        self.context().bind()
    }

    /// Allocates CUDA managed memory without initializing its elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is requested, the attachment policy is unsupported, context binding fails, or CUDA cannot allocate managed memory.
    pub fn create_uninit(
        context: &Arc<Context>,
        length: usize,
        attach_flags: MemoryAttachFlags,
    ) -> Result<ManagedMemory<MaybeUninit<T>>> {
        // SAFETY: `MaybeUninit<T>` elements need no initialization.
        unsafe { ManagedMemory::<MaybeUninit<T>>::create_unchecked(context, length, attach_flags) }
    }

    /// Returns the number of elements in the logical allocation range.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the logical allocation range contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the size of the logical allocation range in bytes.
    ///
    /// # Panics
    ///
    /// Panics if the element count multiplied by the element size overflows `usize`.
    /// Allocation construction validates this size, so this condition cannot occur for a valid allocation.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.checked_byte_len()
            .expect("managed memory byte length overflow")
    }

    fn checked_byte_len(&self) -> Result<usize> {
        checked_bytes::<T>(self.length)
    }

    /// Returns the attachment policy recorded when the managed allocation was created.
    #[must_use]
    pub const fn attach_flags(&self) -> MemoryAttachFlags {
        self.attach_flags
    }

    /// Returns the allocation’s CUDA-accessible element pointer without synchronizing pending work.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the allocation’s CUDA-accessible mutable element pointer.
    ///
    /// The pointer does not establish execution ordering or permit conflicting accesses.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Borrows all initialized elements as a shared device view.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_, T> {
        DeviceView::from_buffer(self)
    }

    /// Borrows all initialized elements as an exclusive device view.
    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        DeviceViewMut::from_buffer(self)
    }

    /// Completes submitted accesses and excludes tracked GPU work while reading managed values.
    /// Requires [concurrent managed access](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#unified-memory-on-windows-wsl-and-tegra) because unrelated streams may remain active.
    ///
    /// # Errors
    ///
    /// Returns `Error::ManagedHostAccessUnsupported` if the device lacks concurrent managed access.
    /// Also propagates device-attribute query, access-tracking, and CUDA completion errors before invoking the callback.
    pub fn with_host_slice<R>(&self, read: impl FnOnce(&[T]) -> R) -> Result<R> {
        if self
            .context()
            .device()
            .attribute(DeviceAttribute::ConcurrentManagedAccess)?
            == 0
        {
            return Err(Error::ManagedHostAccessUnsupported);
        }
        self.context().with_host_access(&[self.region()], || {
            // SAFETY: the shared borrow and tracked access guard exclude conflicting access.
            Ok(read(unsafe { self.as_host_slice() }))
        })
    }

    /// Completes submitted accesses and excludes tracked GPU work while modifying managed values.
    /// Requires concurrent managed access support because unrelated streams may remain active.
    ///
    /// # Errors
    ///
    /// Returns `Error::ManagedHostAccessUnsupported` if the device lacks concurrent managed access.
    /// Also propagates device-attribute query, access-tracking, and CUDA completion errors before invoking the callback.
    pub fn with_host_slice_mut<R>(&mut self, write: impl FnOnce(&mut [T]) -> R) -> Result<R> {
        if self
            .context()
            .device()
            .attribute(DeviceAttribute::ConcurrentManagedAccess)?
            == 0
        {
            return Err(Error::ManagedHostAccessUnsupported);
        }

        let context = Arc::clone(self.context());
        let _guard = context.acquire_host_access(&[self.region()])?;
        // SAFETY: the mutable borrow and tracked access guard exclude device access.
        Ok(write(unsafe { self.as_host_slice_mut() }))
    }

    /// Returns a host slice over this managed allocation.
    ///
    /// # Safety
    ///
    /// The caller must ensure no GPU work or other CPU reference can concurrently mutate the same memory for the returned lifetime, and that the allocation is accessible from the host at the point of access.
    #[must_use]
    pub unsafe fn as_host_slice(&self) -> &[T] {
        if self.is_empty() {
            return &[];
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { slice::from_raw_parts(self.ptr, self.length) }
    }

    /// Returns a mutable host slice over this managed allocation.
    ///
    /// # Safety
    ///
    /// The caller must ensure no GPU work or other CPU reference can concurrently access the same memory for the returned lifetime, and that the allocation is accessible from the host at the point of access.
    pub unsafe fn as_host_slice_mut(&mut self) -> &mut [T] {
        if self.is_empty() {
            return &mut [];
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { slice::from_raw_parts_mut(self.ptr, self.length) }
    }

    /// Fills every element with zero bytes and waits for completion.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking, context-binding, byte-size, CUDA fill, and default-stream synchronization errors.
    pub fn set_zeroes(&mut self) -> Result<()>
    where
        T: ZeroableDeviceRepr,
    {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        let () = T::LAYOUT;
        if self.is_empty() {
            return Ok(());
        }

        // SAFETY: the allocation covers `length` elements, and `T: ZeroableDeviceRepr` makes the zero fill valid.
        unsafe {
            DeviceMemory::<T>::set(self.ptr, 0, self.length)?;
        }
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Enqueues a residency prefetch of the entire managed range to `location`.
    ///
    /// The stream orders the request; completion is asynchronous.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream belongs to another context, byte-size calculation overflows, context binding fails, or CUDA rejects the prefetch request or destination.
    pub fn prefetch_to(&self, location: MemoryLocation, stream: &Stream) -> Result<()> {
        self.bind_stream(stream)?;
        DeviceMemory::<T>::prefetch_async(
            // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
            unsafe { DevicePtr::from_raw(self.ptr.cast::<()>()) },
            self.checked_byte_len()?,
            location,
            stream,
        )
    }

    /// Applies a managed-memory advice policy to this allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the advice or location.
    pub fn advise(&self, advice: MemoryAdvise, location: MemoryLocation) -> Result<()> {
        self.context().bind()?;
        DeviceMemory::<T>::advise(
            // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
            unsafe { DevicePtr::from_raw(self.ptr.cast::<()>()) },
            self.checked_byte_len()?,
            advice,
            location,
        )
    }

    /// Advises CUDA that this managed allocation is mostly read.
    ///
    /// CUDA may create read-only copies of pages on processors that read this range.
    /// Writes remain coherent but can require additional migration or invalidation work.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the advice for this allocation.
    pub fn set_read_mostly(&self) -> Result<()> {
        self.advise(MemoryAdvise::SetReadMostly, Self::ignored_advice_location())
    }

    /// Removes explicit read-mostly advice from this managed allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the advice for this allocation.
    pub fn unset_read_mostly(&self) -> Result<()> {
        self.advise(
            MemoryAdvise::UnsetReadMostly,
            Self::ignored_advice_location(),
        )
    }

    /// Advises CUDA to prefer placing this managed allocation at `location`.
    ///
    /// This guides migration policy but does not immediately migrate pages.
    /// Use [`ManagedMemory::prefetch_to`] when an explicit migration is desired.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the location or advice for this allocation.
    pub fn set_preferred_location(&self, location: MemoryLocation) -> Result<()> {
        self.advise(MemoryAdvise::SetPreferredLocation, location)
    }

    /// Clears the preferred-location advice for this managed allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the advice for this allocation.
    pub fn unset_preferred_location(&self) -> Result<()> {
        self.advise(
            MemoryAdvise::UnsetPreferredLocation,
            Self::ignored_advice_location(),
        )
    }

    /// Advises CUDA that `location` will access this managed allocation.
    ///
    /// This can establish or maintain mappings for the target processor without migrating the data there.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the location or advice for this allocation.
    pub fn set_accessed_by(&self, location: MemoryLocation) -> Result<()> {
        self.advise(MemoryAdvise::SetAccessedBy, location)
    }

    /// Clears accessed-by advice for `location`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the location or advice for this allocation.
    pub fn unset_accessed_by(&self, location: MemoryLocation) -> Result<()> {
        self.advise(MemoryAdvise::UnsetAccessedBy, location)
    }

    const fn ignored_advice_location() -> MemoryLocation {
        MemoryLocation::Host
    }

    /// Attaches this managed allocation to `stream` with the requested attach flags.
    ///
    /// This enqueues `cudaStreamAttachMemAsync` and updates the stored attach flags after CUDA accepts the request.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot attach the allocation to `stream`, the flags are invalid for the allocation, or a previous asynchronous launch reports an error.
    pub fn attach_to_stream(&mut self, stream: &Stream, flags: MemoryAttachFlags) -> Result<()> {
        self.bind_stream(stream)?;
        // SAFETY: `cudaStreamAttachMemAsync` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaStreamAttachMemAsync(
                stream.as_raw(),
                self.ptr.cast(),
                self.checked_byte_len()? as _,
                flags.bits(),
            ))?;
        }

        self.attach_flags = flags;
        Ok(())
    }
}

unsafe impl<T: DeviceRepr> DeviceInput<T> for ManagedMemory<T> {
    fn as_device_ptr(&self) -> *const T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.length
    }
}

// SAFETY: the typed owner or view preserves initialized values for its borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for ManagedMemory<T> {}

unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for ManagedMemory<T> {}

unsafe impl<T: DeviceRepr> DeviceRegionProvider for ManagedMemory<T> {
    fn region(&self) -> DeviceRegion<'_> {
        let offset = self.ptr as usize - self.allocation.ptr as usize;
        self.allocation
            .region()
            .slice(offset..offset + self.byte_len())
            .expect("managed buffer exceeds its allocation")
    }
}

impl<T: DeviceRepr> ManagedMemory<MaybeUninit<T>> {
    /// Borrows this managed allocation as a write-only output of `T` values.
    pub fn output(&mut self) -> DeviceOutputView<'_, T> {
        DeviceOutputView::from_output(self)
    }

    /// Exposes an initialized prefix while retaining the full managed allocation.
    ///
    /// # Safety
    ///
    /// The prefix must contain valid initialized `T` values and all producing work must have completed.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` exceeds the currently exposed element count.
    pub unsafe fn assume_init_prefix(mut self, length: usize) -> Result<ManagedMemory<T>> {
        if length > self.length {
            return Err(Error::InvalidMemoryAccess);
        }

        self.length = length;
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks every element initialized after its producer has completed.
    ///
    /// # Safety
    ///
    /// Every element must contain a valid T and all producer writes must have completed.
    #[must_use]
    pub unsafe fn assume_init(self) -> ManagedMemory<T> {
        let () = T::LAYOUT;
        ManagedMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
            attach_flags: self.attach_flags,
            _phantom: PhantomData,
        }
    }
}

impl<T: DeviceRepr> ManagedMemory<T> {
    /// Gets residency information for this memory range.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid summary or sampling granularity or byte-size overflow.
    /// Also propagates context-binding, driver entry-point lookup, and CUDA location-query errors.
    pub fn location_info(
        &self,
        summary_granularity: usize,
        sampling_granularity: usize,
    ) -> Result<Vec<MemoryLocation>> {
        self.context().bind()?;
        // SAFETY: the pointer is the live managed allocation owned by `self`.
        unsafe {
            location_info(
                DevicePtr::from_raw(self.ptr.cast()),
                self.checked_byte_len()?,
                summary_granularity,
                sampling_granularity,
            )
        }
    }
}
