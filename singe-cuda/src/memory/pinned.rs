use std::{
    mem::MaybeUninit,
    ptr::{self},
    result::Result as StdResult,
    slice,
    sync::Arc,
};

use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    error::{Error, Result},
    submission::{Access, AccessMode},
    try_ffi,
    types::DevicePtr,
    view::{
        AllocationId, AnyBitPatternDeviceRepr, DeviceRepr, DeviceSlice, DeviceView, DeviceViewMut,
        NoUninitDeviceRepr, ZeroableDeviceRepr,
    },
};

use super::{
    CudaHostSlice, HostAllocationFlags, MemoryCopyKind, PinnedAllocation, PinnedHostMemory,
    alloc_host_raw, alloc_pinned_raw, checked_bytes, dangling_or_non_null, transfer,
};

impl<T: DeviceRepr> PinnedHostMemory<T> {
    /// Completes tracked device accesses and excludes submitted work while reading host values.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking and CUDA completion errors before invoking the callback.
    pub fn with_host_slice<R>(&self, read: impl FnOnce(&[T]) -> R) -> Result<R> {
        let context = self.context();
        let _guard =
            context.acquire_accesses(&[Access::host(self.host_region(), AccessMode::Read)])?;
        Ok(read(self.as_slice()))
    }

    /// Completes tracked device accesses and excludes submitted work while modifying host values.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking and CUDA completion errors before invoking the callback.
    pub fn with_host_slice_mut<R>(&mut self, write: impl FnOnce(&mut [T]) -> R) -> Result<R> {
        let context = Arc::clone(self.context());
        let _guard =
            context.acquire_accesses(&[Access::host(self.host_region(), AccessMode::ReadWrite)])?;
        Ok(write(self.as_slice_mut()))
    }

    fn allocate(context: &Arc<Context>, length: usize, flags: HostAllocationFlags) -> Result<Self> {
        let () = T::LAYOUT;
        context.bind()?;
        let bytes = checked_bytes::<T>(length)?;
        let ptr = if bytes == 0 {
            ptr::null_mut()
        } else if flags.is_empty() || flags == HostAllocationFlags::DEFAULT {
            // SAFETY: `bytes` is non-zero, and the returned allocation is owned by the new value.
            unsafe { alloc_host_raw(bytes)? }
        } else {
            // SAFETY: `bytes` is non-zero, and the returned allocation is owned by the new value.
            unsafe { alloc_pinned_raw(bytes, flags)? }
        };
        if bytes != 0 && ptr.is_null() {
            return Err(Error::NullHandle);
        }

        let allocation = Arc::new(PinnedAllocation {
            context: Arc::clone(context),
            id: AllocationId::create(),
            ptr: ptr.cast(),
            bytes,
            flags,
        });
        if bytes != 0 && !(ptr as usize).is_multiple_of(align_of::<T>()) {
            return Err(Error::InvalidMemoryAccess);
        }
        Ok(Self {
            allocation,
            ptr: dangling_or_non_null(ptr.cast(), length)?,
            length,
        })
    }

    /// Allocates pinned host memory and copies `values` into it.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation size overflows, CUDA cannot allocate page-locked host memory, or `values` is a non-empty zero-sized element slice.
    pub fn from_slice(context: &Arc<Context>, values: &[T]) -> Result<Self> {
        Self::from_slice_with_flags(context, values, HostAllocationFlags::DEFAULT)
    }

    /// Allocates pinned host memory with `flags` and copies `values` into it.
    ///
    /// `HostAllocationFlags::DEFAULT` uses the simple `cudaMallocHost` allocation path.
    /// Other flags use `cudaHostAlloc`, allowing callers to request portable, mapped, or write-combined host memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation size overflows, CUDA cannot allocate page-locked host memory with the requested flags, or `values` is a non-empty zero-sized element slice.
    pub fn from_slice_with_flags(
        context: &Arc<Context>,
        values: &[T],
        flags: HostAllocationFlags,
    ) -> Result<Self> {
        let memory = Self::allocate(context, values.len(), flags)?;
        if !values.is_empty() {
            // SAFETY: the fresh allocation holds `values.len()` elements and cannot overlap `values`.
            unsafe {
                ptr::copy_nonoverlapping(values.as_ptr(), memory.ptr.as_ptr(), values.len());
            }
        }
        Ok(memory)
    }

    /// Copies a device range into newly allocated pinned host storage.
    ///
    /// Completes the transfer before exposing initialized host values.
    ///
    /// # Errors
    ///
    /// Propagates pinned allocation, byte-size, context-binding, readiness synchronization, stream-capture, CUDA copy, and stream synchronization errors.
    /// No initialized allocation is returned unless the transfer completes.
    pub fn from_device<B: DeviceSlice<T> + ?Sized>(source: &B) -> Result<Self>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let region = source.region();
        let mut memory =
            Self::create_uninit(region.context(), source.len(), HostAllocationFlags::DEFAULT)?;
        region.context().bind()?;
        // SAFETY: the provider contains initialized values and CUDA completes the entire destination before promotion.
        unsafe {
            transfer::copy_synchronous(
                memory.as_mut_ptr().cast::<T>(),
                source.as_device_ptr(),
                source.len(),
                MemoryCopyKind::DeviceToHost,
                region.stream(),
            )?;
            Ok(memory.assume_init())
        }
    }

    /// Takes ownership of an existing CUDA pinned host allocation.
    ///
    /// # Safety
    ///
    /// `ptr` must be a live host allocation returned by `cudaMallocHost` or `cudaHostAlloc` for `length` initialized elements of `T`, and no other owner may free it while the returned value is alive.
    /// A null pointer is accepted for an empty allocation.
    /// Nonempty pointers must satisfy `T`'s alignment.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow, a nonempty zero-sized allocation, a nonempty null pointer, or a pointer not aligned for `T`.
    pub unsafe fn from_raw_parts(
        context: &Arc<Context>,
        ptr: *mut T,
        length: usize,
        flags: HostAllocationFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_bytes::<T>(length)?;
        if bytes != 0 && ptr.is_null() {
            return Err(Error::NullHandle);
        }

        if bytes != 0 && !(ptr as usize).is_multiple_of(align_of::<T>()) {
            return Err(Error::InvalidMemoryAccess);
        }
        Ok(Self {
            allocation: Arc::new(PinnedAllocation {
                context: Arc::clone(context),
                id: AllocationId::create(),
                ptr: ptr.cast(),
                bytes,
                flags,
            }),
            ptr: dangling_or_non_null(ptr, length)?,
            length,
        })
    }

    /// Transfers the raw allocation when no other owner retains it.
    ///
    /// Returns the unchanged buffer if a submission or interop user retains the allocation.
    /// A non-null returned pointer must eventually be freed with `cudaFreeHost`.
    /// Empty allocations return null rather than a typed dangling sentinel.
    ///
    /// # Errors
    ///
    /// Returns the original wrapper if another owner, submission, graph, or foreign handle retains the allocation.
    pub fn into_raw_parts(
        self,
    ) -> StdResult<(*mut T, usize, HostAllocationFlags, Arc<Context>), Self> {
        let Self {
            allocation,
            ptr,
            length,
        } = self;
        match Arc::try_unwrap(allocation) {
            Ok(mut allocation) => {
                let context = Arc::clone(allocation.context());
                let pointer = allocation.ptr.cast();
                allocation.ptr = ptr::null_mut();
                Ok((pointer, length, allocation.flags, context))
            }
            Err(allocation) => Err(Self {
                allocation,
                ptr,
                length,
            }),
        }
    }

    /// Returns the CUDA context retained by this allocation or pool.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Returns the owner retained by asynchronous transfers and interop users.
    #[must_use]
    pub fn allocation(&self) -> &Arc<PinnedAllocation> {
        &self.allocation
    }

    /// Reinterprets initialized host bytes without changing allocation ownership.
    ///
    /// # Errors
    ///
    /// Returns the original allocation if `U` is zero-sized, its size does not divide the exposed byte length, or a nonempty range is not aligned for `U`.
    ///
    /// # Panics
    ///
    /// Panics if the exposed byte length overflows `usize`.
    /// Construction and slicing preserve a validated byte length, so this cannot occur for a valid allocation.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<PinnedHostMemory<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let () = U::LAYOUT;
        let size = size_of::<U>();
        let bytes = self.byte_len().expect("pinned byte length overflow");
        if size == 0
            || !bytes.is_multiple_of(size)
            || (bytes != 0 && !self.as_ptr().addr().is_multiple_of(align_of::<U>()))
        {
            return Err(self);
        }
        Ok(PinnedHostMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: bytes / size,
        })
    }

    /// Erases initialization while preserving the pinned allocation and its flags.
    #[must_use]
    pub fn into_uninit(self) -> PinnedHostMemory<MaybeUninit<T>> {
        PinnedHostMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
        }
    }

    /// Allocates pinned storage without initializing its elements.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow or a nonempty zero-sized allocation.
    /// Also propagates context-binding and CUDA host allocation errors and rejects null or insufficiently aligned storage for a nonempty allocation.
    pub fn create_uninit(
        context: &Arc<Context>,
        length: usize,
        flags: HostAllocationFlags,
    ) -> Result<PinnedHostMemory<MaybeUninit<T>>> {
        PinnedHostMemory::<MaybeUninit<T>>::allocate(context, length, flags)
    }

    /// Allocates a new pinned buffer and copies this buffer's contents into it.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate the clone or if the byte size overflows.
    pub fn try_clone(&self) -> Result<Self> {
        Self::from_slice_with_flags(self.context(), self.as_slice(), self.flags())
    }

    /// Returns this pinned allocation as an initialized host slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        if self.is_empty() {
            return &[];
        }
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.length) }
    }

    /// Returns this pinned allocation as a mutable initialized host slice.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        if self.is_empty() {
            return &mut [];
        }
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.length) }
    }

    /// Returns the host pointer for this pinned allocation.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr().cast_const()
    }

    /// Returns the mutable host pointer for this pinned allocation.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
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

    /// Returns the allocation size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `len * size_of::<T>()` overflows or `T` is a non-empty zero-sized element type.
    pub fn byte_len(&self) -> Result<usize> {
        checked_bytes::<T>(self.length)
    }

    /// Returns the flags requested when this allocation was created.
    #[must_use]
    pub fn flags(&self) -> HostAllocationFlags {
        self.allocation.flags()
    }

    /// Queries the flags CUDA recorded for this host allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the allocation flags or reports runtime initialization diagnostics.
    pub fn host_flags(&self) -> Result<HostAllocationFlags> {
        self.context().bind()?;
        if self.is_empty() || self.flags() == HostAllocationFlags::DEFAULT {
            return Ok(self.flags());
        }

        let mut flags = 0;
        // SAFETY: `cudaHostGetFlags` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaHostGetFlags(
                &raw mut flags,
                self.ptr.as_ptr().cast(),
            ))?;
        }
        Ok(HostAllocationFlags::from_bits_retain(flags))
    }

    /// Returns the CUDA device pointer alias for this mapped host allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation is not mapped into the current CUDA context, the device cannot map host memory, or CUDA reports runtime initialization diagnostics.
    pub fn mapped_device_ptr(&self) -> Result<DevicePtr> {
        self.allocation.mapped_device_ptr()
    }

    /// Returns a device view over this allocation's mapped device alias.
    ///
    /// Host and device access through the mapped alias still require normal CUDA stream synchronization by the caller.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot provide a mapped device pointer.
    pub fn mapped_device_view(&self) -> Result<DeviceView<'_, T>> {
        let region = self
            .allocation
            .mapped_device_region()?
            .slice(..self.byte_len()?)?;
        // SAFETY: the mapped range borrows this initialized host storage.
        unsafe { DeviceView::from_raw_parts(region, self.length) }
    }

    /// Returns a mutable device view over this allocation's mapped device alias.
    ///
    /// Host and device access through the mapped alias still require normal CUDA stream synchronization by the caller.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot provide a mapped device pointer.
    pub fn mapped_device_view_mut(&mut self) -> Result<DeviceViewMut<'_, T>> {
        let region = self
            .allocation
            .mapped_device_region()?
            .slice(..self.byte_len()?)?;
        // SAFETY: the mapped range exclusively borrows this initialized host storage.
        unsafe { DeviceViewMut::from_raw_parts(region, self.length) }
    }
}

impl<T: DeviceRepr> PinnedHostMemory<MaybeUninit<T>> {
    /// Exposes a completed producer's initialized prefix while retaining the original capacity.
    ///
    /// # Safety
    ///
    /// Every prefix element must be valid and all host and device producers must have completed.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` exceeds the currently exposed element count.
    pub unsafe fn assume_init_prefix(mut self, length: usize) -> Result<PinnedHostMemory<T>> {
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
    /// All elements must be initialized to valid `T` values, with no writes in flight.
    #[must_use]
    pub unsafe fn assume_init(self) -> PinnedHostMemory<T> {
        let () = T::LAYOUT;
        PinnedHostMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
        }
    }
}

impl<T: ZeroableDeviceRepr> PinnedHostMemory<T> {
    /// Allocates pinned host memory initialized with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow or a nonempty zero-sized allocation.
    /// Also propagates context-binding and CUDA host allocation errors and rejects null or insufficiently aligned storage for a nonempty allocation.
    pub fn zeroed(
        context: &Arc<Context>,
        length: usize,
        flags: HostAllocationFlags,
    ) -> Result<Self> {
        let memory = Self::allocate(context, length, flags)?;
        if !memory.is_empty() {
            // SAFETY: the allocation is writable and the zero pattern is valid for T.
            unsafe {
                ptr::write_bytes(memory.ptr.as_ptr(), 0, memory.length);
            }
        }
        Ok(memory)
    }
}
