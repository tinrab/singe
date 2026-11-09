use std::{
    alloc::Layout,
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ptr::NonNull,
    result::Result as StdResult,
    slice,
    sync::Arc,
};

use crate::{
    context::Context,
    error::{Error, Result},
    submission::{Access, AccessMode},
    types::DevicePtr,
    view::{
        AnyBitPatternDeviceRepr, DeviceOutputView, DeviceRepr, DeviceView, DeviceViewMut,
        NoUninitDeviceRepr,
    },
};

use super::{
    CudaHostSlice, HostRegisterFlags, HostRegistration, RegisteredAllocation, RegisteredHostMemory,
    RegisteredHostSlice, RegisteredHostVec, checked_bytes, dangling_or_non_null,
};

impl<'a, T: DeviceRepr> RegisteredHostMemory<'a, T> {
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

    /// Registers a mutable host slice with CUDA for this wrapper's lifetime.
    ///
    /// Empty slices retain the context without registering a host range.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice byte length overflows, CUDA cannot register the host range, or `slice` is non-empty for a zero-sized `T`.
    pub fn create(context: &Arc<Context>, slice: &'a mut [T]) -> Result<Self> {
        Self::create_with_flags(context, slice, HostRegisterFlags::DEFAULT)
    }

    /// Registers a mutable host slice with CUDA using `flags`.
    ///
    /// Safe mutable registration rejects [`HostRegisterFlags::READ_ONLY`] and [`HostRegisterFlags::IO_MEMORY`].
    /// Use [`RegisteredHostMemory::create_io_memory`] for MMIO ranges.
    ///
    /// # Errors
    ///
    /// Returns an error if `flags` are incompatible with mutable safe registration, the slice byte length overflows, CUDA cannot register the range, or `slice` is non-empty for a zero-sized `T`.
    pub fn create_with_flags(
        context: &Arc<Context>,
        slice: &'a mut [T],
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        if flags.contains(HostRegisterFlags::READ_ONLY)
            || flags.contains(HostRegisterFlags::IO_MEMORY)
        {
            return Err(Error::InvalidValue);
        }
        // SAFETY: the registration borrows `slice` for its lifetime, so the pages stay valid until unregistration.
        unsafe { Self::register_slice(context, slice.as_mut_ptr(), slice.len(), flags) }
    }

    /// Registers memory-mapped I/O host memory.
    ///
    /// # Safety
    ///
    /// `slice` must represent an MMIO range satisfying CUDA's `cudaHostRegisterIoMemory` requirements.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow or a nonempty zero-sized range.
    /// Also propagates context-binding and CUDA I/O-memory registration errors.
    pub unsafe fn create_io_memory(context: &Arc<Context>, slice: &'a mut [T]) -> Result<Self> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Self::register_slice(
                context,
                slice.as_mut_ptr(),
                slice.len(),
                HostRegisterFlags::IO_MEMORY,
            )
        }
    }

    unsafe fn register_slice(
        context: &Arc<Context>,
        ptr: *mut T,
        length: usize,
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_bytes::<T>(length)?;
        let typed_ptr = dangling_or_non_null(ptr, length)?;
        let registration =
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { HostRegistration::create(context, typed_ptr.cast(), bytes, flags)? };
        Ok(Self {
            ptr: typed_ptr,
            length,
            registration,
            _borrow: PhantomData,
        })
    }

    /// Returns the registered memory as an immutable host slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        if self.is_empty() {
            return &[];
        }
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.length) }
    }

    /// Returns the registered memory as a mutable host slice.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        if self.is_empty() {
            return &mut [];
        }
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.length) }
    }

    /// Returns the registered host pointer.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr().cast_const()
    }

    /// Returns the mutable registered host pointer.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns the number of registered elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether this registration contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the registered range size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `len * size_of::<T>()` overflows or `T` is a non-empty zero-sized element type.
    pub fn byte_len(&self) -> Result<usize> {
        checked_bytes::<T>(self.length)
    }

    /// Returns the flags used to create this registration.
    #[must_use]
    pub const fn flags(&self) -> HostRegisterFlags {
        self.registration.flags
    }

    /// Returns the CUDA device pointer alias for this registered host range.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot map this host range into the current CUDA context or reports runtime initialization diagnostics.
    pub fn mapped_device_ptr(&self) -> Result<DevicePtr> {
        self.registration.mapped_device_ptr()
    }

    /// Returns the CUDA context retained for registration, mapping, and unregistration.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.registration.context
    }

    /// Borrows initialized registered values through their device mapping.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA mapping-query errors.
    /// Also rejects invalid mapped address ranges, byte-size overflow, or a view whose byte length or alignment does not match `T`.
    pub fn mapped_device_view(&self) -> Result<DeviceView<'_, T>> {
        // SAFETY: the registered host storage contains valid T values and retains this shared borrow.
        unsafe { DeviceView::from_raw_parts(self.registration.mapped_device_region()?, self.len()) }
    }

    /// Exclusively borrows registered values through their device mapping.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA mapping-query errors.
    /// Also rejects invalid mapped address ranges, byte-size overflow, or a view whose byte length or alignment does not match `T`.
    pub fn mapped_device_view_mut(&mut self) -> Result<DeviceViewMut<'_, T>> {
        // SAFETY: the exclusive wrapper borrow covers the complete registered host range.
        unsafe {
            DeviceViewMut::from_raw_parts(self.registration.mapped_device_region()?, self.len())
        }
    }
}

impl<'a, T: DeviceRepr> RegisteredHostSlice<'a, T> {
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

    /// Registers an immutable host slice with CUDA for this wrapper's lifetime.
    ///
    /// Empty slices retain the context without registering a host range.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice byte length overflows, CUDA cannot register the host range, or `slice` is non-empty for a zero-sized `T`.
    pub fn create(context: &Arc<Context>, slice: &'a [T]) -> Result<Self> {
        Self::create_with_flags(context, slice, HostRegisterFlags::DEFAULT)
    }

    /// Registers an immutable host slice as read-only for CUDA access.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice byte length overflows, CUDA cannot register the host range as read-only, or `slice` is non-empty for a zero-sized `T`.
    pub fn create_read_only(context: &Arc<Context>, slice: &'a [T]) -> Result<Self> {
        Self::create_with_flags(context, slice, HostRegisterFlags::READ_ONLY)
    }

    /// Registers an immutable host slice with CUDA using `flags`.
    ///
    /// Immutable registration rejects [`HostRegisterFlags::IO_MEMORY`].
    /// Use the unsafe mutable constructor for MMIO ranges.
    ///
    /// # Errors
    ///
    /// Returns an error if `flags` contain unsupported options, the byte length overflows, CUDA cannot register the range, or `slice` is non-empty for a zero-sized `T`.
    pub fn create_with_flags(
        context: &Arc<Context>,
        slice: &'a [T],
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        if flags.contains(HostRegisterFlags::IO_MEMORY) {
            return Err(Error::InvalidValue);
        }
        // SAFETY: the registration borrows `slice` for its lifetime, so the pages stay valid until unregistration.
        unsafe { Self::register_slice(context, slice.as_ptr(), slice.len(), flags) }
    }

    unsafe fn register_slice(
        context: &Arc<Context>,
        ptr: *const T,
        length: usize,
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_bytes::<T>(length)?;
        let typed_ptr = dangling_or_non_null(ptr.cast_mut(), length)?;
        let registration =
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { HostRegistration::create(context, typed_ptr.cast(), bytes, flags)? };
        Ok(Self {
            ptr: typed_ptr,
            length,
            registration,
            _borrow: PhantomData,
        })
    }

    /// Returns the registered memory as an immutable host slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        if self.is_empty() {
            return &[];
        }
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.length) }
    }

    /// Returns the registered host pointer.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr().cast_const()
    }

    /// Returns the number of registered elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether this registration contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the registered range size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `len * size_of::<T>()` overflows or `T` is a non-empty zero-sized element type.
    pub fn byte_len(&self) -> Result<usize> {
        checked_bytes::<T>(self.length)
    }

    /// Returns the flags used to create this registration.
    #[must_use]
    pub const fn flags(&self) -> HostRegisterFlags {
        self.registration.flags
    }

    /// Returns the mapped device address in the retained registration context.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA host-to-device mapping query errors, including memory that is not mapped for this context.
    pub fn mapped_device_ptr(&self) -> Result<DevicePtr> {
        self.registration.mapped_device_ptr()
    }

    /// Returns the CUDA context retained for registration, mapping, and unregistration.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.registration.context
    }

    /// Borrows initialized registered values through their device mapping.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA mapping-query errors.
    /// Also rejects invalid mapped address ranges, byte-size overflow, or a view whose byte length or alignment does not match `T`.
    pub fn mapped_device_view(&self) -> Result<DeviceView<'_, T>> {
        // SAFETY: the registered host storage contains valid T values and retains this shared borrow.
        unsafe { DeviceView::from_raw_parts(self.registration.mapped_device_region()?, self.len()) }
    }
}

impl<T: DeviceRepr> RegisteredHostVec<T> {
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

    /// Copies `values` into a new `Vec<T>` and registers its storage with CUDA.
    ///
    /// Empty slices retain the context without registering a host range.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation of the `Vec` fails, the byte length overflows, CUDA cannot register the range, or `values` is non-empty for a zero-sized `T`.
    pub fn from_slice(context: &Arc<Context>, values: &[T]) -> Result<Self> {
        Self::from_slice_with_flags(context, values, HostRegisterFlags::DEFAULT)
    }

    /// Copies `values` into a new `Vec<T>` and registers it with `flags`.
    ///
    /// This constructor rejects [`HostRegisterFlags::READ_ONLY`] because the returned vector exposes mutable access, and rejects [`HostRegisterFlags::IO_MEMORY`] because a normal `Vec<T>` cannot represent an MMIO range.
    ///
    /// # Errors
    ///
    /// Returns an error if `flags` contain unsupported options, allocation of the `Vec` fails, the byte length overflows, CUDA cannot register the range, or `values` is non-empty for a zero-sized `T`.
    pub fn from_slice_with_flags(
        context: &Arc<Context>,
        values: &[T],
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        Self::from_vec_with_flags(context, values.to_vec(), flags)
    }

    /// Registers an owned vector without copying or changing its allocation.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow, a nonempty zero-sized allocation, or an invalid allocation layout.
    /// Also propagates context-binding and CUDA host registration errors.
    pub fn from_vec(context: &Arc<Context>, values: Vec<T>) -> Result<Self> {
        Self::from_vec_with_flags(context, values, HostRegisterFlags::DEFAULT)
    }

    /// Registers an owned vector, preserving its original capacity and allocation layout.
    ///
    /// Mutable storage rejects read-only and MMIO registration flags.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow, a nonempty zero-sized allocation, an invalid allocation layout, or unsupported read-only or I/O-memory flags.
    /// Also propagates context-binding and CUDA host registration errors.
    pub fn from_vec_with_flags(
        context: &Arc<Context>,
        mut values: Vec<T>,
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        if flags.contains(HostRegisterFlags::READ_ONLY)
            || flags.contains(HostRegisterFlags::IO_MEMORY)
        {
            return Err(Error::InvalidValue);
        }

        let bytes = checked_bytes::<T>(values.len())?;
        let layout = Layout::array::<T>(values.capacity())
            .map_err(|_| Error::InvalidMemoryAllocationRequest)?;
        let ptr = NonNull::from(values.as_mut_slice()).cast();
        // SAFETY: the vector is moved into the registration, so its buffer stays valid until unregistration.
        let registration = unsafe { HostRegistration::create(context, ptr.cast(), bytes, flags)? };
        let length = values.len();
        // The shared owner unregisters and frees this original global allocation.
        let _values = ManuallyDrop::new(values);
        Ok(Self {
            allocation: Arc::new(RegisteredAllocation {
                registration,
                layout,
            }),
            ptr,
            length,
        })
    }

    /// Returns the registered vector contents as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.length) }
    }

    /// Returns the registered vector contents as a mutable slice.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.length) }
    }

    /// Returns the number of registered elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns whether this registered vector contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the registered vector size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `len * size_of::<T>()` overflows or `T` is a non-empty zero-sized element type.
    pub fn byte_len(&self) -> Result<usize> {
        checked_bytes::<T>(self.length)
    }

    /// Returns the flags used to register this vector.
    #[must_use]
    pub fn flags(&self) -> HostRegisterFlags {
        self.allocation.flags()
    }

    /// Returns the CUDA device pointer alias for this registered vector.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot map this vector into the current CUDA context or reports runtime initialization diagnostics.
    pub fn mapped_device_ptr(&self) -> Result<DevicePtr> {
        self.allocation.mapped_device_ptr()
    }

    /// Returns the CUDA context retained for registration, mapping, and unregistration.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Borrows initialized registered values through their device mapping.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA mapping-query errors.
    /// Also rejects invalid mapped address ranges, byte-size overflow, or a view whose byte length or alignment does not match `T`.
    pub fn mapped_device_view(&self) -> Result<DeviceView<'_, T>> {
        // SAFETY: the registered host storage contains valid T values and retains this shared borrow.
        unsafe {
            DeviceView::from_raw_parts(
                self.allocation
                    .mapped_device_region()?
                    .slice(..self.byte_len()?)?,
                self.len(),
            )
        }
    }

    /// Exclusively borrows registered values through their device mapping.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA mapping-query errors.
    /// Also rejects invalid mapped address ranges, byte-size overflow, or a view whose byte length or alignment does not match `T`.
    pub fn mapped_device_view_mut(&mut self) -> Result<DeviceViewMut<'_, T>> {
        // SAFETY: the exclusive wrapper borrow covers the complete registered host range.
        unsafe {
            DeviceViewMut::from_raw_parts(
                self.allocation
                    .mapped_device_region()?
                    .slice(..self.byte_len()?)?,
                self.len(),
            )
        }
    }

    /// Returns the retained owner without granting host or device access.
    #[must_use]
    pub fn allocation(&self) -> &Arc<RegisteredAllocation> {
        &self.allocation
    }

    /// Returns the host address for this typed range.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr().cast_const()
    }

    /// Returns the host address under an exclusive typed borrow.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Reinterprets initialized bytes while preserving the original registration and deallocation layout.
    ///
    /// # Errors
    ///
    /// Returns the original vector if its byte length is invalid, `U` is zero-sized, its size does not divide the exposed byte length, or a nonempty range is not aligned for `U`.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<RegisteredHostVec<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let () = U::LAYOUT;
        let size = size_of::<U>();
        let Ok(bytes) = self.byte_len() else {
            return Err(self);
        };
        if size == 0
            || !bytes.is_multiple_of(size)
            || (bytes != 0 && !self.as_ptr().addr().is_multiple_of(align_of::<U>()))
        {
            return Err(self);
        }
        Ok(RegisteredHostVec {
            allocation: self.allocation,
            ptr: if bytes == 0 {
                NonNull::dangling()
            } else {
                self.ptr.cast()
            },
            length: bytes / size,
        })
    }

    /// Erases initialization while retaining the complete allocation and registration.
    #[must_use]
    pub fn into_uninit(self) -> RegisteredHostVec<MaybeUninit<T>> {
        RegisteredHostVec {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
        }
    }

    /// Allocates and registers storage for a host or device producer without initializing values.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow, a nonempty zero-sized allocation, an invalid allocation layout, or unsupported read-only or I/O-memory flags.
    /// Also propagates context-binding and CUDA host registration errors.
    pub fn create_uninit(
        context: &Arc<Context>,
        length: usize,
        flags: HostRegisterFlags,
    ) -> Result<RegisteredHostVec<MaybeUninit<T>>> {
        let () = T::LAYOUT;
        checked_bytes::<T>(length)?;
        RegisteredHostVec::from_vec_with_flags(context, vec![MaybeUninit::uninit(); length], flags)
    }
}

impl<T: DeviceRepr> RegisteredHostVec<MaybeUninit<T>> {
    /// Marks all values initialized after all host and device production completes.
    ///
    /// # Safety
    /// Every element must be valid and all producers must have completed.
    #[must_use]
    pub unsafe fn assume_init(self) -> RegisteredHostVec<T> {
        let () = T::LAYOUT;
        RegisteredHostVec {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
        }
    }

    /// Exposes a completed initialized prefix while retaining the complete registration and capacity.
    ///
    /// # Safety
    /// Every prefix element must be valid and all producers must have completed.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` exceeds the currently exposed element count.
    pub unsafe fn assume_init_prefix(mut self, length: usize) -> Result<RegisteredHostVec<T>> {
        if length > self.length {
            return Err(Error::InvalidMemoryAccess);
        }

        self.length = length;
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { self.assume_init() })
    }

    /// Exclusively borrows mapped storage for a producer without asserting initialized values.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA mapping-query errors.
    /// Also rejects invalid mapped address ranges, byte-size overflow, or output metadata whose byte length or alignment does not match `T`.
    pub fn mapped_device_output(&mut self) -> Result<DeviceOutputView<'_, T>> {
        let region = self
            .allocation
            .mapped_device_region()?
            .slice(..self.byte_len()?)?;
        // SAFETY: this exclusive borrow retains the mapped registration and does not read its contents.
        unsafe { DeviceOutputView::from_raw_parts(region, self.length) }
    }
}
