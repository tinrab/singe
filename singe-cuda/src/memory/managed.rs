use std::{
    marker::PhantomData, mem::MaybeUninit, ptr, result::Result as StdResult, slice, sync::Arc,
};

use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    device::DeviceAttribute,
    error::{Error, Result},
    memory::{
        DeviceAllocation, DeviceDeallocation, DeviceMemory, MemoryAttachFlags, MemoryLocation,
        location_info,
    },
    stream::{AsStream, Stream, StreamBinding},
    try_ffi,
    types::{DevicePtr, MemoryAdvise},
    utility::{checked_bytes, to_u64},
    view::{
        AnyBitPatternDeviceRepr, DeviceOutput, DeviceOutputView, DeviceRange, DeviceRegion,
        DeviceRegionProvider, DeviceRepr, DeviceSlice, DeviceSliceMut, DeviceView, DeviceViewMut,
        NoUninitDeviceRepr, ZeroableDeviceRepr, cast_length,
    },
};

impl<T: DeviceRepr> ManagedMemory<T> {
    /// Allocates CUDA managed memory and completes zero initialization.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte size overflows, the context cannot be bound, CUDA cannot allocate the memory, access tracking fails, or CUDA cannot fill or synchronize.
    pub fn zeroed(
        context: &Arc<Context>,
        length: usize,
        attach_flags: MemoryAttachFlags,
    ) -> Result<Self>
    where
        T: ZeroableDeviceRepr,
    {
        let mut memory = Self::create_uninit(context, length, attach_flags)?;
        memory.set_zeroes()?;
        // SAFETY: `set_zeroes` completed a zero fill of every element, and an all-zero `T` is valid.
        Ok(unsafe { memory.assume_init() })
    }

    /// Allocates CUDA managed memory and completes its repeated byte fill.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte size overflows, the context cannot be bound, CUDA cannot allocate the memory, access tracking fails, or CUDA cannot fill or synchronize.
    pub fn filled(
        context: &Arc<Context>,
        length: usize,
        attach_flags: MemoryAttachFlags,
        value: u8,
    ) -> Result<Self>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let mut memory = Self::create_uninit(context, length, attach_flags)?;
        memory.set_value(value)?;
        // SAFETY: `set_value` completed a fill of every element, and every bit pattern is a valid `T`.
        Ok(unsafe { memory.assume_init() })
    }

    /// Takes ownership of an existing managed allocation.
    ///
    /// # Safety
    ///
    /// `raw.ptr` must be null for an empty allocation or identify live managed storage for `raw.len` elements in `raw.context`, allocated with `cudaMallocManaged`.
    /// The attachment flags must describe that allocation's policy.
    /// No other owner may free it, and each element must be initialized before it is read.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte length overflows or a nonempty allocation has a null pointer.
    pub unsafe fn from_raw(raw: RawManagedMemory<T>) -> Result<Self> {
        let RawManagedMemory {
            ptr,
            len,
            attach_flags,
            context,
        } = raw;
        let () = T::LAYOUT;
        let bytes = checked_bytes::<T>(len, "byte length")?;
        if bytes != 0 && ptr.is_null() {
            return Err(Error::null_pointer("managed memory"));
        }
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                &context,
                ptr.cast(),
                bytes,
                DeviceDeallocation::Immediate,
            ),
            ptr,
            length: len,
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
    /// Returns the wrapper unchanged as the error if another owner, submission, graph, or foreign handle retains the allocation.
    pub fn into_raw(self) -> StdResult<RawManagedMemory<T>, Self> {
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
                Ok(RawManagedMemory {
                    ptr,
                    len: length,
                    attach_flags,
                    context,
                })
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
    /// Returns the allocation unchanged as the error if `U` is zero-sized, its size does not divide the exposed byte length, or a nonempty range is not aligned for `U`.
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
        let () = T::LAYOUT;
        context.bind()?;
        let bytes = checked_bytes::<T>(length, "byte length")?;
        let ptr = if bytes == 0 {
            ptr::null_mut()
        } else {
            // SAFETY: the `DeviceAllocation` below owns the pointer and frees it with a device free, its `MaybeUninit` elements need no initialization, and this wrapper orders host access through the access tracker.
            unsafe { DeviceMemory::<MaybeUninit<T>>::alloc_managed(context, length, attach_flags)? }
        };
        Ok(ManagedMemory {
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
        checked_bytes::<T>(self.length, "byte length").expect("managed memory byte length overflow")
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

    /// Returns the allocation’s address as an untyped CUDA device pointer.
    #[must_use]
    pub const fn as_device_ptr(&self) -> DevicePtr {
        // SAFETY: `ptr` is this live allocation's base or null, and `DevicePtr` only records the address.
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
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
    /// Returns [`Error::ManagedHostAccessUnsupported`] if the device lacks concurrent managed access.
    /// Returns an error if the device attribute query, access tracking, or waiting for submitted work fails, before the callback runs.
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
            // SAFETY: the host-access guard waited for and excludes tracked GPU work on this region, concurrent managed access was checked above, and `&self` excludes host writers.
            Ok(read(unsafe { self.as_host_slice() }))
        })
    }

    /// Completes submitted accesses and excludes tracked GPU work while modifying managed values.
    /// Requires concurrent managed access support because unrelated streams may remain active.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ManagedHostAccessUnsupported`] if the device lacks concurrent managed access.
    /// Returns an error if the device attribute query, access tracking, or waiting for submitted work fails, before the callback runs.
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
        // SAFETY: the host-access guard waited for and excludes tracked GPU work on this region, concurrent managed access was checked above, and `&mut self` excludes other host references.
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
        // SAFETY: a nonempty allocation's `ptr` names `length` initialized values, and the caller guarantees the host can read them and nothing mutates them for the borrow.
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
        // SAFETY: a nonempty allocation's `ptr` names `length` initialized values, and the caller guarantees the host can access them and nothing else does for the borrow.
        unsafe { slice::from_raw_parts_mut(self.ptr, self.length) }
    }

    /// Fills every element with zero bytes and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error if access tracking fails, the context cannot be bound, the byte size overflows, or CUDA cannot fill the memory or synchronize the default stream.
    pub fn set_zeroes(&mut self) -> Result<()>
    where
        T: ZeroableDeviceRepr,
    {
        // SAFETY: `T: ZeroableDeviceRepr` makes the all-zero fill a valid `T`.
        unsafe { self.fill_bytes(0) }
    }

    /// Fills every byte with `value` and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error if access tracking fails, the context cannot be bound, the byte size overflows, or CUDA cannot fill the memory or synchronize the default stream.
    pub fn set_value(&mut self, value: u8) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        // SAFETY: `T: AnyBitPatternDeviceRepr` makes every repeated byte pattern a valid `T`.
        unsafe { self.fill_bytes(value) }
    }

    // The caller guarantees that `value` repeated over each element is a valid `T`.
    unsafe fn fill_bytes(&mut self, value: u8) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        let () = T::LAYOUT;
        if self.is_empty() {
            return Ok(());
        }

        // SAFETY: the allocation covers `length` elements, and the caller guarantees the repeated byte is a valid `T`.
        unsafe {
            DeviceMemory::<T>::set(self.context(), self.ptr, value, self.length)?;
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
        DeviceMemory::<T>::prefetch(
            // SAFETY: `self.ptr` is the base of the managed allocation that `self` owns, and `&self` keeps it allocated while CUDA enqueues the prefetch.
            unsafe { DevicePtr::from_raw(self.ptr.cast::<()>()) },
            checked_bytes::<T>(self.length, "byte length")?,
            location,
            stream,
        )
    }

    /// Applies a managed-memory advice policy to this allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or CUDA rejects the advice or location.
    pub fn advise(&self, advice: MemoryAdvise, location: MemoryLocation) -> Result<()> {
        self.context().bind()?;
        DeviceMemory::<T>::advise(
            // SAFETY: `self.ptr` is the base of the managed allocation that `self` owns, and `&self` keeps it allocated for the call.
            unsafe { DevicePtr::from_raw(self.ptr.cast::<()>()) },
            checked_bytes::<T>(self.length, "byte length")?,
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
    /// This enqueues the attachment on `stream` and updates the stored attach flags after CUDA accepts the request.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot attach the allocation to `stream`, the flags are invalid for the allocation, or a previous asynchronous launch reports an error.
    pub fn attach_to_stream(&mut self, stream: &Stream, flags: MemoryAttachFlags) -> Result<()> {
        self.bind_stream(stream)?;
        let bytes = to_u64(
            checked_bytes::<T>(self.length, "byte length")?,
            "byte length",
        )?;
        // SAFETY: `stream` is live and belongs to this allocation's context, which is bound, and `self.ptr` with `bytes` names the managed allocation `self` owns.
        unsafe {
            try_ffi!(runtime::cudaStreamAttachMemAsync(
                stream.as_raw(),
                self.ptr.cast(),
                bytes,
                flags.bits(),
            ))?;
        }

        self.attach_flags = flags;
        Ok(())
    }
}

// SAFETY: `ptr` names `length` aligned elements of the managed allocation that `allocation` keeps live, and `region` reports that same range in the owning context.
unsafe impl<T: DeviceRepr> DeviceRange<T> for ManagedMemory<T> {
    fn as_device_ptr(&self) -> *const T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.length
    }
}

// SAFETY: a `ManagedMemory<T>` is only built from initialized values, through `zeroed`, `filled`, `from_raw`, or `assume_init`, and its methods keep them valid.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for ManagedMemory<T> {}

// SAFETY: `as_mut_ptr` returns the same `ptr` as `as_device_ptr`, and `&mut self` is the only way to reach it, so the mutable borrow is exclusive.
unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for ManagedMemory<T> {}

// SAFETY: the region is a slice of the owning allocation's region at this buffer's offset and byte length, and `allocation` retains the context and memory while `self` lives.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for ManagedMemory<T> {
    fn region(&self) -> DeviceRegion<'_> {
        let offset = self.ptr.addr() - self.allocation.ptr.addr();
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
            return Err(Error::buffer_too_small("allocation", length, self.length));
        }

        self.length = length;
        // SAFETY: the caller guarantees the first `length` elements hold valid `T` values written by completed work.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks every element initialized after its producer has completed.
    ///
    /// # Safety
    ///
    /// Every element must contain a valid `T` and all producer writes must have completed.
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
    /// Returns the residency of this memory range, one location per summary region.
    ///
    /// # Errors
    ///
    /// Returns an error if a granularity is invalid, the byte size overflows, the context cannot be bound, the driver entry point is unavailable, or CUDA rejects the location query.
    pub fn location_info(
        &self,
        summary_granularity: usize,
        sampling_granularity: usize,
    ) -> Result<Vec<MemoryLocation>> {
        self.context().bind()?;
        // SAFETY: `self.ptr` is the base of the managed allocation that `self` owns, which stays allocated for the call.
        unsafe {
            location_info(
                DevicePtr::from_raw(self.ptr.cast()),
                checked_bytes::<T>(self.length, "byte length")?,
                summary_granularity,
                sampling_granularity,
            )
        }
    }
}

/// Raw ownership of a managed allocation, its attachment policy, and its context.
///
/// Free `ptr` while `context` is alive, or re-adopt the value with [`ManagedMemory::from_raw`].
#[derive(Debug)]
pub struct RawManagedMemory<T> {
    /// Base of the allocation, or null for an empty one.
    pub ptr: *mut T,
    /// Number of elements of `T`.
    pub len: usize,
    /// Attachment policy of the allocation.
    pub attach_flags: MemoryAttachFlags,
    /// Context the allocation belongs to.
    pub context: Arc<Context>,
}

/// Owns a CUDA managed allocation and retains its context.
///
/// Host access through the guarded callback methods waits for tracked GPU work and excludes conflicting submissions for the duration of the callback.
#[derive(Debug)]
pub struct ManagedMemory<T: DeviceRepr> {
    allocation: Arc<DeviceAllocation>,
    ptr: *mut T,
    length: usize,
    // CUDA tracks the current visibility policy for managed memory separately
    // from the pointer. Store the last policy requested through this wrapper so
    // callers can reason about stream attachment without another FFI query.
    attach_flags: MemoryAttachFlags,
    _phantom: PhantomData<T>,
}

// SAFETY: the owner holds `T` values like a `Vec<T>`, and CUDA keeps managed memory coherent across host threads.
unsafe impl<T: DeviceRepr + Send> Send for ManagedMemory<T> {}

// SAFETY: shared host access goes through guarded callbacks or unsafe slices, and `T: Sync` covers shared reads.
unsafe impl<T: DeviceRepr + Sync> Sync for ManagedMemory<T> {}

// SAFETY: `&mut self` is exclusive over the `length` retained managed elements at `ptr`.
unsafe impl<T: DeviceRepr> DeviceOutput<T> for ManagedMemory<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        let () = T::LAYOUT;
        self.ptr
    }
}

// SAFETY: `MaybeUninit<T>` has `T`'s layout, so the retained range holds `length` slots for `T`, and no initialized reads are claimed.
unsafe impl<T: DeviceRepr> DeviceRange<T> for ManagedMemory<MaybeUninit<T>> {
    fn len(&self) -> usize {
        self.length
    }
}

// SAFETY: `&mut self` is exclusive over `length` retained slots with `T`'s layout.
unsafe impl<T: DeviceRepr> DeviceOutput<T> for ManagedMemory<MaybeUninit<T>> {
    fn as_mut_ptr(&mut self) -> *mut T {
        let () = T::LAYOUT;
        self.ptr.cast()
    }
}
