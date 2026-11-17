use std::{marker::PhantomData, ptr, ptr::NonNull, slice, sync::Arc};

use singe_core::report_cleanup_failure;
use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    error::{Error, Result},
    memory::{
        HostRegisterFlags, PinnedHostSlice, PinnedHostSliceMut, register_host_raw,
        unregister_host_raw,
    },
    submission::{Access, AccessMode},
    try_ffi,
    types::DevicePtr,
    utility::{checked_bytes, dangling_or_non_null},
    view::{
        AllocationId, DeviceRegion, DeviceRepr, DeviceView, DeviceViewMut, HostRegion, HostSlice,
        HostSliceMut,
    },
};

impl<'a, T: DeviceRepr> RegisteredHostMemory<'a, T> {
    /// Completes tracked device accesses and excludes submitted work while reading host values.
    ///
    /// # Errors
    ///
    /// Returns an error if access tracking fails or waiting for tracked device work fails, before the callback runs.
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
    /// Returns an error if access tracking fails or waiting for tracked device work fails, before the callback runs.
    pub fn with_host_slice_mut<R>(&mut self, write: impl FnOnce(&mut [T]) -> R) -> Result<R> {
        let context = Arc::clone(self.context());
        let _guard =
            context.acquire_accesses(&[Access::host(self.host_region(), AccessMode::ReadWrite)])?;
        Ok(write(self.as_slice_mut()))
    }

    /// Registers a mutable host slice with CUDA using `flags` for this wrapper's lifetime.
    ///
    /// Empty slices retain the context without registering a host range.
    /// Safe mutable registration rejects [`HostRegisterFlags::READ_ONLY`] and [`HostRegisterFlags::IO_MEMORY`].
    /// Use [`RegisteredHostMemory::create_io_memory`] for MMIO ranges.
    ///
    /// # Errors
    ///
    /// Returns an error if `flags` are incompatible with mutable safe registration, the slice byte length overflows, CUDA cannot register the range, or `slice` is non-empty for a zero-sized `T`.
    pub fn create(
        context: &Arc<Context>,
        slice: &'a mut [T],
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        if flags.contains(HostRegisterFlags::READ_ONLY)
            || flags.contains(HostRegisterFlags::IO_MEMORY)
        {
            return Err(Error::out_of_range("host register flags"));
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
    /// Returns an error if the byte size overflows, `slice` is non-empty for a zero-sized `T`, the context cannot be bound, or CUDA cannot register the range as I/O memory.
    pub unsafe fn create_io_memory(context: &Arc<Context>, slice: &'a mut [T]) -> Result<Self> {
        // SAFETY: the registration borrows `slice` for `'a`, so the range stays mapped until unregistration, and the caller guarantees it is an MMIO range CUDA accepts.
        unsafe {
            Self::register_slice(
                context,
                slice.as_mut_ptr(),
                slice.len(),
                HostRegisterFlags::IO_MEMORY,
            )
        }
    }

    // Requires `ptr` to name `length` host values that stay allocated for `'a`, and an MMIO range when `flags` contains `IO_MEMORY`.
    unsafe fn register_slice(
        context: &Arc<Context>,
        ptr: *mut T,
        length: usize,
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_bytes::<T>(length, "byte length")?;
        let typed_ptr = dangling_or_non_null(ptr, length)?;
        let registration =
            // SAFETY: the caller guarantees `typed_ptr` names `bytes` host bytes that stay allocated for `'a`, which the `PhantomData` borrow ties this registration to.
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
        // SAFETY: `ptr` names `length` initialized values borrowed or owned for `self`'s lifetime, and `&self` excludes host mutation through this wrapper.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.length) }
    }

    /// Returns the registered memory as a mutable host slice.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        if self.is_empty() {
            return &mut [];
        }
        // SAFETY: `ptr` names `length` initialized values borrowed or owned for `self`'s lifetime, and `&mut self` makes this the only reference to them.
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
        checked_bytes::<T>(self.length, "byte length")
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
    /// Returns an error if the context cannot be bound, CUDA cannot report the mapping, the mapped range or byte size is invalid, or the view's byte length or alignment does not match `T`.
    pub fn mapped_device_view(&self) -> Result<DeviceView<'_, T>> {
        // SAFETY: the mapped region aliases the registered host values, which are initialized `T` values that `&self` keeps alive and unmodified.
        unsafe { DeviceView::from_raw_parts(self.registration.mapped_device_region()?, self.len()) }
    }

    /// Exclusively borrows registered values through their device mapping.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot report the mapping, the mapped range or byte size is invalid, or the view's byte length or alignment does not match `T`.
    pub fn mapped_device_view_mut(&mut self) -> Result<DeviceViewMut<'_, T>> {
        // SAFETY: the mapped region aliases the registered host values, which are initialized `T` values that `&mut self` keeps alive and unaliased.
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
    /// Returns an error if access tracking fails or waiting for tracked device work fails, before the callback runs.
    pub fn with_host_slice<R>(&self, read: impl FnOnce(&[T]) -> R) -> Result<R> {
        let context = self.context();
        let _guard =
            context.acquire_accesses(&[Access::host(self.host_region(), AccessMode::Read)])?;
        Ok(read(self.as_slice()))
    }

    /// Registers an immutable host slice with CUDA using `flags` for this wrapper's lifetime.
    ///
    /// Empty slices retain the context without registering a host range.
    /// Pass [`HostRegisterFlags::READ_ONLY`] to register the range as read-only for CUDA access.
    /// Immutable registration rejects [`HostRegisterFlags::IO_MEMORY`].
    /// Use [`RegisteredHostMemory::create_io_memory`] for MMIO ranges.
    ///
    /// # Errors
    ///
    /// Returns an error if `flags` contain unsupported options, the byte length overflows, CUDA cannot register the range, or `slice` is non-empty for a zero-sized `T`.
    pub fn create(
        context: &Arc<Context>,
        slice: &'a [T],
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        if flags.contains(HostRegisterFlags::IO_MEMORY) {
            return Err(Error::out_of_range("host register flags"));
        }
        // SAFETY: the registration borrows `slice` for its lifetime, so the pages stay valid until unregistration.
        unsafe { Self::register_slice(context, slice.as_ptr(), slice.len(), flags) }
    }

    // Requires `ptr` to name `length` host values that stay allocated for `'a`.
    unsafe fn register_slice(
        context: &Arc<Context>,
        ptr: *const T,
        length: usize,
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_bytes::<T>(length, "byte length")?;
        let typed_ptr = dangling_or_non_null(ptr.cast_mut(), length)?;
        let registration =
            // SAFETY: the caller guarantees `typed_ptr` names `bytes` host bytes that stay allocated for `'a`, which the `PhantomData` borrow ties this registration to.
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
        // SAFETY: `ptr` names `length` initialized values borrowed or owned for `self`'s lifetime, and `&self` excludes host mutation through this wrapper.
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
        checked_bytes::<T>(self.length, "byte length")
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
    /// Returns an error if the context cannot be bound or CUDA cannot report the mapping, including memory that is not mapped for this context.
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
    /// Returns an error if the context cannot be bound, CUDA cannot report the mapping, the mapped range or byte size is invalid, or the view's byte length or alignment does not match `T`.
    pub fn mapped_device_view(&self) -> Result<DeviceView<'_, T>> {
        // SAFETY: the mapped region aliases the registered host values, which are initialized `T` values that `&self` keeps alive and unmodified.
        unsafe { DeviceView::from_raw_parts(self.registration.mapped_device_region()?, self.len()) }
    }
}

#[derive(Debug)]
pub(crate) struct HostRegistration {
    flags: HostRegisterFlags,
    context: Arc<Context>,
    id: AllocationId,
    ptr: NonNull<()>,
    bytes: usize,
}

impl HostRegistration {
    unsafe fn create(
        context: &Arc<Context>,
        ptr: NonNull<()>,
        bytes: usize,
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        context.bind()?;
        if bytes != 0 {
            // SAFETY: the typed wrapper retains the host storage for this registration's lifetime.
            unsafe {
                register_host_raw(ptr.as_ptr(), bytes, flags)?;
            }
        }
        Ok(Self {
            flags,
            context: Arc::clone(context),
            id: AllocationId::create(),
            ptr,
            bytes,
        })
    }

    fn host_region(&self) -> HostRegion<'_> {
        // SAFETY: the registration retains this range and its original flags.
        unsafe {
            HostRegion::from_raw_parts(
                &self.context,
                self.id,
                self.bytes,
                self.flags.contains(HostRegisterFlags::PORTABLE),
            )
        }
    }

    fn mapped_device_ptr(&self) -> Result<DevicePtr> {
        self.context.bind()?;
        if self.bytes == 0 {
            return Ok(DevicePtr::null());
        }
        mapped_host_device_ptr(self.ptr.as_ptr())
    }

    fn mapped_device_region(&self) -> Result<DeviceRegion<'_>> {
        let pointer = self.mapped_device_ptr()?;
        // SAFETY: the registration and its typed wrapper retain the host borrow or owned vector.
        unsafe { DeviceRegion::from_raw_parts(&self.context, self.id, pointer.cast(), self.bytes) }
    }

    fn unregister(&mut self) -> Result<()> {
        if self.bytes != 0 {
            self.context.bind()?;
            // SAFETY: the registration still retains its live host range.
            unsafe { unregister_host_raw(self.ptr.as_ptr())? };
            self.bytes = 0;
        }
        Ok(())
    }
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        if let Err(error) = self.unregister() {
            report_cleanup_failure("unregister", "cuda host memory", &error);
        }
    }
}

/// Mutable borrowed host memory registered with CUDA for the registration lifetime.
#[derive(Debug)]
pub struct RegisteredHostMemory<'a, T: DeviceRepr> {
    ptr: NonNull<T>,
    length: usize,
    registration: HostRegistration,
    _borrow: PhantomData<&'a mut [T]>,
}

/// Immutable borrowed host memory registered with CUDA for the registration lifetime.
#[derive(Debug)]
pub struct RegisteredHostSlice<'a, T: DeviceRepr> {
    ptr: NonNull<T>,
    length: usize,
    registration: HostRegistration,
    _borrow: PhantomData<&'a [T]>,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "raw host pointers follow the owned or borrowed storage bounds explained above"
)]
// SAFETY: the wrapper is a `&mut [T]` borrow plus a registration that `cudaHostUnregister` can release from any thread.
unsafe impl<T: DeviceRepr + Send> Send for RegisteredHostMemory<'_, T> {}

// SAFETY: `&RegisteredHostMemory<T>` grants only shared reads of the borrowed `[T]`.
unsafe impl<T: DeviceRepr + Sync> Sync for RegisteredHostMemory<'_, T> {}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "raw host pointers follow the owned or borrowed storage bounds explained above"
)]
// SAFETY: the wrapper is a `&[T]` borrow, which is `Send` when `T: Sync`, plus a registration releasable from any thread.
unsafe impl<T: DeviceRepr + Sync> Send for RegisteredHostSlice<'_, T> {}

// SAFETY: `&RegisteredHostSlice<T>` grants only shared reads of the borrowed `[T]`.
unsafe impl<T: DeviceRepr + Sync> Sync for RegisteredHostSlice<'_, T> {}

// SAFETY: the wrapper borrows `length` live `T` values at `ptr` for `'a`, and `&self` grants only reads.
unsafe impl<T: DeviceRepr> HostSlice<T> for RegisteredHostMemory<'_, T> {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

// SAFETY: the wrapper holds the `&mut [T]` borrow, so `&mut self` is exclusive over the same range.
unsafe impl<T: DeviceRepr> HostSliceMut<T> for RegisteredHostMemory<'_, T> {
    fn as_host_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

// SAFETY: the wrapper borrows `length` live `T` values at `ptr` for `'a`, and nothing mutates them during the borrow.
unsafe impl<T: DeviceRepr> HostSlice<T> for RegisteredHostSlice<'_, T> {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

// SAFETY: the registration page-locks the borrowed range until it drops, and the region keeps its identity and flags.
unsafe impl<T: DeviceRepr> PinnedHostSlice<T> for RegisteredHostMemory<'_, T> {
    fn host_region(&self) -> HostRegion<'_> {
        self.registration
            .host_region()
            .slice(..self.byte_len().expect("valid host slice length"))
            .expect("host slice fits its allocation")
    }
}

// SAFETY: the type implements both `PinnedHostSlice` and `HostSliceMut` over the same range.
unsafe impl<T: DeviceRepr> PinnedHostSliceMut<T> for RegisteredHostMemory<'_, T> {}

// SAFETY: the registration page-locks the borrowed range until it drops, and the region keeps its identity and flags.
unsafe impl<T: DeviceRepr> PinnedHostSlice<T> for RegisteredHostSlice<'_, T> {
    fn host_region(&self) -> HostRegion<'_> {
        self.registration
            .host_region()
            .slice(..self.byte_len().expect("valid host slice length"))
            .expect("host slice fits its allocation")
    }
}

pub(crate) fn mapped_host_device_ptr(ptr: *mut ()) -> Result<DevicePtr> {
    if ptr.is_null() {
        return Ok(DevicePtr::null());
    }

    let mut device_ptr = ptr::null_mut();
    // SAFETY: `device_ptr` is a local output, and the query only reads CUDA's mapping of `ptr`.
    unsafe {
        try_ffi!(runtime::cudaHostGetDevicePointer(
            &raw mut device_ptr,
            ptr.cast(),
            0,
        ))?;
    }
    // SAFETY: `DevicePtr` only records the CUDA-reported alias; callers tie its use to the registration's lifetime.
    Ok(unsafe { DevicePtr::from_raw(device_ptr.cast()) })
}
