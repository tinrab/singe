use std::{marker::PhantomData, mem::MaybeUninit, ptr, result::Result as StdResult, sync::Arc};

use singe_cuda_sys::runtime;

use super::{
    DeviceAllocation, DeviceDeallocation, DeviceMemory, MemoryCopyKind, PitchedDeviceMemory,
    PitchedDeviceVolume, checked_bytes, checked_len, checked_volume,
};
use crate::{
    context::Context,
    error::{Error, Result},
    stream::{StreamBinding, StreamScope},
    submission::{Access, AccessMode},
    types::DevicePtr,
    utility::to_usize,
    view::{
        AllocationId, AnyBitPatternDeviceRepr, DeviceRegion, DeviceRegionProvider, DeviceRepr,
        DeviceView, DeviceViewMut, NoUninitDeviceRepr, ZeroableDeviceRepr,
    },
};

fn checked_pitched_bytes<T>(
    ptr: *mut T,
    width: usize,
    height: usize,
    depth: usize,
    pitch: usize,
) -> Result<usize> {
    let row_bytes = checked_bytes::<T>(width)?;
    checked_volume(row_bytes, height, depth)?;
    if width == 0 || height == 0 || depth == 0 {
        return Ok(0);
    }

    let bytes = checked_volume(pitch, height, depth)?;
    if ptr.is_null()
        || pitch < row_bytes
        || !(ptr as usize).is_multiple_of(align_of::<T>())
        || !pitch.is_multiple_of(align_of::<T>())
        || (ptr as usize).checked_add(bytes).is_none()
    {
        return Err(Error::InvalidMemoryAccess);
    }
    Ok(bytes)
}

// Adopt a newly allocated pointer before checking CUDA's returned pitch, so a
// rejected layout still frees the allocation in its owning context.
fn own_pitched_allocation<T>(
    context: &Arc<Context>,
    ptr: *mut T,
    width: usize,
    height: usize,
    depth: usize,
    pitch: usize,
) -> Result<Arc<DeviceAllocation>> {
    let mut allocation = DeviceAllocation {
        id: AllocationId::create(),
        context: Arc::clone(context),
        ptr: ptr.cast(),
        bytes: 0,
        deallocation: DeviceDeallocation::Immediate,
    };
    allocation.bytes = checked_pitched_bytes(ptr, width, height, depth, pitch)?;
    Ok(Arc::new(allocation))
}

impl<T: DeviceRepr> PitchedDeviceMemory<T> {
    fn allocate(context: &Arc<Context>, width: usize, height: usize) -> Result<Self> {
        let () = T::LAYOUT;
        context.bind()?;
        // SAFETY: the context is bound above, and the returned allocation is owned by the new value.
        let (ptr, pitch) = unsafe { DeviceMemory::<T>::alloc_pitched(width, height)? };
        let allocation = own_pitched_allocation(context, ptr, width, height, 1, pitch)?;
        Ok(Self {
            allocation,
            ptr,
            width,
            height,
            pitch,
            _phantom: PhantomData,
        })
    }

    /// Allocates pitched storage whose logical elements are uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error if size calculations overflow, a nonempty zero-sized element range is requested, context binding fails, or CUDA cannot allocate pitched storage.
    /// Also rejects a returned pointer or pitch that violates the requested row size, alignment, or address-range bounds.
    pub fn create_uninit(
        context: &Arc<Context>,
        width: usize,
        height: usize,
    ) -> Result<PitchedDeviceMemory<MaybeUninit<T>>> {
        PitchedDeviceMemory::<MaybeUninit<T>>::allocate(context, width, height)
    }

    /// Allocates pitched storage with caller-established initialization.
    ///
    /// # Safety
    ///
    /// Initialize every logical element before reading it.
    /// CUDA row padding remains uninitialized and must not be exposed as initialized bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if size calculations overflow, a nonempty zero-sized element range is requested, context binding fails, or CUDA cannot allocate pitched storage.
    /// Also rejects a returned pointer or pitch that violates the requested row size, alignment, or address-range bounds.
    pub unsafe fn create_unchecked(
        context: &Arc<Context>,
        width: usize,
        height: usize,
    ) -> Result<Self> {
        Self::allocate(context, width, height)
    }

    /// Allocates pitched storage and completes zero initialization of its logical elements.
    ///
    /// # Errors
    ///
    /// Propagates pitched allocation, size-validation, access-tracking, context-binding, CUDA fill, and synchronization errors.
    pub fn zeroed(context: &Arc<Context>, width: usize, height: usize) -> Result<Self>
    where
        T: ZeroableDeviceRepr,
    {
        let mut memory = Self::create_uninit(context, width, height)?;
        memory.set_zeroes()?;
        // SAFETY: the completed fill initializes every logical element with a valid zero value.
        Ok(unsafe { memory.assume_init() })
    }

    /// Allocates pitched storage and completes a copy of contiguous row-major host values.
    ///
    /// # Errors
    ///
    /// Returns an error if the source length differs from the logical element count.
    /// Also propagates pitched allocation, size-validation, context-binding, CUDA copy, and synchronization errors.
    pub fn from_slice(
        context: &Arc<Context>,
        width: usize,
        height: usize,
        values: &[T],
    ) -> Result<Self> {
        Self::create_uninit(context, width, height)?.initialize_from_host(values)
    }

    /// Takes ownership of a pitched allocation in `context`.
    ///
    /// # Safety
    ///
    /// `ptr` must be the base of a live CUDA allocation compatible with `cudaFree`, with no other destruction owner.
    /// It must contain initialized, valid `T` values at every logical element.
    /// The pitch and dimensions must describe its storage, and previous accesses must be ordered before future consumers.
    /// A null pointer is accepted for an empty shape.
    ///
    /// # Errors
    ///
    /// Returns an error if row or volume size calculations overflow or a nonempty zero-sized element range is requested.
    /// For nonempty storage, also rejects a null or misaligned pointer, a pitch smaller than the row width or not aligned for `T`, or an overflowing ending address.
    pub unsafe fn from_raw_parts(
        context: &Arc<Context>,
        ptr: *mut T,
        width: usize,
        height: usize,
        pitch: usize,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_pitched_bytes(ptr, width, height, 1, pitch)?;
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                context,
                ptr.cast(),
                bytes,
                DeviceDeallocation::Immediate,
            ),
            ptr,
            width,
            height,
            pitch,
            _phantom: PhantomData,
        })
    }

    /// Transfers raw ownership, or returns this buffer unchanged if another owner retains it.
    ///
    /// The returned pointer must eventually be freed in the returned context.
    ///
    /// # Errors
    ///
    /// Returns the original wrapper if another owner, submission, graph, or foreign handle retains the allocation.
    pub fn into_raw_parts(self) -> StdResult<(*mut T, usize, usize, usize, Arc<Context>), Self> {
        let Self {
            allocation,
            ptr,
            width,
            height,
            pitch,
            _phantom: PhantomData,
        } = self;
        match Arc::try_unwrap(allocation) {
            Ok(mut allocation) => {
                let context = Arc::clone(allocation.context());
                allocation.ptr = ptr::null_mut();
                Ok((ptr, width, height, pitch, context))
            }
            Err(allocation) => Err(Self {
                allocation,
                ptr,
                width,
                height,
                pitch,
                _phantom: PhantomData,
            }),
        }
    }

    /// Returns the allocation's retained CUDA context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Returns the shared owner of the physical allocation, including row padding.
    #[must_use]
    pub fn allocation(&self) -> &Arc<DeviceAllocation> {
        &self.allocation
    }

    /// Reinterprets logical row bytes while preserving pitch, capacity, and allocation identity.
    ///
    /// Row padding is excluded from the initialized representation.
    ///
    /// # Errors
    ///
    /// Returns the original allocation if `U` is zero-sized, its size does not divide the logical row byte width, or nonempty storage has a base pointer or pitch not aligned for `U`.
    ///
    /// # Panics
    ///
    /// Panics if the logical row byte width overflows `usize`.
    /// Construction validates this width, so this cannot occur for a valid pitched allocation.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<PitchedDeviceMemory<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let () = U::LAYOUT;
        let size = size_of::<U>();
        let row_bytes = self
            .row_byte_width()
            .expect("pitched row byte length overflow");
        if size == 0
            || !row_bytes.is_multiple_of(size)
            || (!self.is_empty()
                && (!(self.ptr as usize).is_multiple_of(align_of::<U>())
                    || !self.pitch.is_multiple_of(align_of::<U>())))
        {
            return Err(self);
        }
        Ok(PitchedDeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            width: row_bytes / size,
            height: self.height,
            pitch: self.pitch,
            _phantom: PhantomData,
        })
    }

    /// Erases logical element initialization without changing allocation ownership.
    #[must_use]
    pub fn into_uninit(self) -> PitchedDeviceMemory<MaybeUninit<T>> {
        PitchedDeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            width: self.width,
            height: self.height,
            pitch: self.pitch,
            _phantom: PhantomData,
        }
    }

    /// Returns the logical row width in elements.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns the logical number of rows.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Returns the row stride in bytes.
    #[must_use]
    pub const fn pitch(&self) -> usize {
        self.pitch
    }

    /// Returns the base device pointer.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the mutable base device pointer.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns the base pointer as a CUDA device pointer wrapper.
    #[must_use]
    pub const fn as_device_ptr(&self) -> DevicePtr {
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
    }

    /// Returns the logical element count.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * height` overflows.
    pub fn len(&self) -> Result<usize> {
        checked_len(self.width, self.height)
    }

    /// Returns the logical row width in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * size_of::<T>()` overflows or `T` is a non-empty zero-sized element type.
    pub fn row_byte_width(&self) -> Result<usize> {
        checked_bytes::<T>(self.width)
    }

    /// Returns the physical allocation span described by `pitch * height`.
    ///
    /// # Errors
    ///
    /// Returns an error if `pitch * height` overflows.
    pub fn byte_len(&self) -> Result<usize> {
        Ok(self.allocation.byte_len())
    }

    /// Returns whether any dimension is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Copies a contiguous host slice into this pitched device allocation.
    ///
    /// `src` must contain exactly `width * height` elements in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error if `src` has the wrong length, byte-size arithmetic overflows, or CUDA rejects the 2D copy.
    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        if src.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }

        let row_bytes = self.row_byte_width()?;
        // SAFETY: the shape and lengths were checked above, so the copy stays inside both allocations.
        unsafe {
            DeviceMemory::<T>::copy_2d(
                self.ptr,
                self.pitch,
                src.as_ptr(),
                row_bytes,
                self.width,
                self.height,
                MemoryCopyKind::HostToDevice,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Copies this pitched device allocation into a contiguous host slice.
    ///
    /// `dst` must contain exactly `width * height` elements and is filled in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` has the wrong length, byte-size arithmetic overflows, or CUDA rejects the 2D copy.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `T: AnyBitPatternDeviceRepr`, so any device bytes are valid `T` values.
        unsafe { self.copy_to_host_unchecked(dst) }
    }

    /// Copies device values whose validity is established by the producing kernel.
    ///
    /// # Safety
    ///
    /// Every copied element must be initialized and valid for `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination length differs from the logical element count or transfer size calculations overflow.
    /// Also propagates access-tracking, context-binding, CUDA copy, and default-stream synchronization errors.
    pub unsafe fn copy_to_host_unchecked(&self, dst: &mut [T]) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        if dst.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }

        let row_bytes = self.row_byte_width()?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            DeviceMemory::<T>::copy_2d(
                dst.as_mut_ptr(),
                row_bytes,
                self.ptr,
                self.pitch,
                self.width,
                self.height,
                MemoryCopyKind::DeviceToHost,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Copies another pitched device allocation with the same shape into this one.
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes differ, byte-size arithmetic overflows, or CUDA rejects the device-to-device 2D copy.
    pub fn copy_from_pitched(&mut self, src: &Self) -> Result<()> {
        self.context().bind()?;
        if self.context() != src.context() {
            return Err(Error::InvalidValue);
        }

        if self.width != src.width || self.height != src.height {
            return Err(Error::InvalidMemoryAccess);
        }

        // SAFETY: the shape and lengths were checked above, so the copy stays inside both allocations.
        unsafe {
            DeviceMemory::<T>::copy_2d(
                self.ptr,
                self.pitch,
                src.ptr,
                src.pitch,
                self.width,
                self.height,
                MemoryCopyKind::DeviceToDevice,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Enqueues a host-to-device 2D copy from a contiguous host slice.
    ///
    /// `src` must remain valid until `stream` reaches the copy.
    /// The [`StreamScope`] lifetime enforces that requirement for safe callers.
    ///
    /// # Errors
    ///
    /// Returns an error if `src` has the wrong length, byte-size arithmetic overflows, or CUDA cannot enqueue the copy.
    pub fn copy_from_host_async<'env>(
        &'env mut self,
        src: &'env [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        if self.context() != stream.stream().context() {
            return Err(Error::InvalidValue);
        }

        self.context().bind()?;
        if src.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }

        let row_bytes = self.row_byte_width()?;
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                DeviceMemory::<T>::copy_2d_async(
                    self.ptr,
                    self.pitch,
                    src.as_ptr(),
                    row_bytes,
                    self.width,
                    self.height,
                    MemoryCopyKind::HostToDevice,
                    stream.stream(),
                )
            })
        }
    }

    /// Enqueues a device-to-host 2D copy into a contiguous host slice.
    ///
    /// `dst` must remain valid until `stream` reaches the copy.
    /// The [`StreamScope`] lifetime enforces that requirement for safe callers.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` has the wrong length, byte-size arithmetic overflows, or CUDA cannot enqueue the copy.
    pub fn copy_to_host_async<'env>(
        &'env self,
        dst: &'env mut [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let accesses = [Access::new(self.region(), AccessMode::Read)];
        if self.context() != stream.stream().context() {
            return Err(Error::InvalidValue);
        }

        self.context().bind()?;
        let () = T::LAYOUT;
        if dst.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }

        let row_bytes = self.row_byte_width()?;
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                DeviceMemory::<T>::copy_2d_async(
                    dst.as_mut_ptr(),
                    row_bytes,
                    self.ptr,
                    self.pitch,
                    self.width,
                    self.height,
                    MemoryCopyKind::DeviceToHost,
                    stream.stream(),
                )
            })
        }
    }

    /// Fills this pitched allocation with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the 2D memset.
    pub fn set_zeroes(&mut self) -> Result<()>
    where
        T: ZeroableDeviceRepr,
    {
        // SAFETY: `T: ZeroableDeviceRepr`, so all-zero bytes are a valid `T`.
        unsafe { self.set_value_unchecked(0) }
    }

    /// Fills this pitched allocation with `value` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the 2D memset.
    pub fn set_value(&mut self, value: u8) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `T: AnyBitPatternDeviceRepr`, so repeating `value` produces valid `T` values.
        unsafe { self.set_value_unchecked(value) }
    }

    /// Fills storage with a byte pattern known to be valid for `T`.
    ///
    /// # Safety
    ///
    /// The repeated byte pattern must represent a valid value of `T`.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking, context-binding, transfer-size validation, CUDA fill, and default-stream synchronization errors.
    pub unsafe fn set_value_unchecked(&mut self, value: u8) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { DeviceMemory::<T>::set_2d(self.ptr, self.pitch, value, self.width, self.height) }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Enqueues a byte fill of this pitched allocation on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the 2D memset.
    pub fn set_value_async<'env>(
        &'env mut self,
        value: u8,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        if self.context() != stream.stream().context() {
            return Err(Error::InvalidValue);
        }

        self.context().bind()?;
        let () = T::LAYOUT;
        checked_len(self.row_byte_width()?, self.height)?;
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                DeviceMemory::<T>::set_2d_async(
                    self.ptr,
                    self.pitch,
                    value,
                    self.width,
                    self.height,
                    stream.stream(),
                )
            })
        }
    }

    fn row_region(&self, row: usize) -> Result<DeviceRegion<'_>> {
        if row >= self.height {
            return Err(Error::InvalidMemoryAccess);
        }

        if self.width == 0 {
            return self.region().slice(0..0);
        }

        let start = row
            .checked_mul(self.pitch)
            .ok_or(Error::InvalidMemoryAccess)?;
        let end = start
            .checked_add(self.row_byte_width()?)
            .ok_or(Error::InvalidMemoryAccess)?;
        self.region().slice(start..end)
    }

    /// Borrows one initialized logical row, excluding CUDA's row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the row index is out of bounds, row offset calculations overflow, or the resulting range fails byte-length or alignment validation.
    pub fn row(&self, row: usize) -> Result<DeviceView<'_, T>> {
        // SAFETY: construction or promotion establishes initialized logical elements.
        unsafe { DeviceView::from_raw_parts(self.row_region(row)?, self.width) }
    }

    /// Exclusively borrows one logical row, excluding CUDA's row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the row index is out of bounds, row offset calculations overflow, or the resulting range fails byte-length or alignment validation.
    pub fn row_mut(&mut self, row: usize) -> Result<DeviceViewMut<'_, T>> {
        // SAFETY: the exclusive buffer borrow covers this exact row and its initialized element type.
        unsafe { DeviceViewMut::from_raw_parts(self.row_region(row)?, self.width) }
    }
}

impl<T: DeviceRepr> PitchedDeviceMemory<MaybeUninit<T>> {
    /// Copies every logical element from host values and completes before promotion.
    ///
    /// # Errors
    ///
    /// Returns an error if the source length differs from the logical element count or transfer size calculations overflow.
    /// Also propagates context-binding, CUDA copy, and default-stream synchronization errors.
    pub fn initialize_from_host(self, values: &[T]) -> Result<PitchedDeviceMemory<T>> {
        if values.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }

        self.context().bind()?;
        // SAFETY: the allocation has writable storage for the complete logical rectangle.
        unsafe {
            DeviceMemory::<T>::copy_2d(
                self.ptr.cast(),
                self.pitch,
                values.as_ptr(),
                self.row_byte_width()?,
                self.width,
                self.height,
                MemoryCopyKind::HostToDevice,
            )?;
        }

        StreamBinding::Default(Arc::clone(self.context())).synchronize()?;
        // SAFETY: the completed full copy initializes all logical elements to valid T values.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks all logical elements initialized without changing the owner or pitch.
    ///
    /// # Safety
    ///
    /// Every logical element must contain an initialized valid T value.
    /// Producers must be ordered before subsequent consumers and complete before host readback.
    /// Row padding remains uninitialized.
    #[must_use]
    pub unsafe fn assume_init(self) -> PitchedDeviceMemory<T> {
        let () = T::LAYOUT;
        PitchedDeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            width: self.width,
            height: self.height,
            pitch: self.pitch,
            _phantom: PhantomData,
        }
    }
}

unsafe impl<T: DeviceRepr> DeviceRegionProvider for PitchedDeviceMemory<T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.allocation.region()
    }
}

// SAFETY: device pointers are not dereferenced on the host; all operations bind the retained context.
unsafe impl<T: DeviceRepr + Send> Send for PitchedDeviceMemory<T> {}

unsafe impl<T: DeviceRepr + Sync> Sync for PitchedDeviceMemory<T> {}

impl<T: DeviceRepr> PitchedDeviceVolume<T> {
    fn allocate(context: &Arc<Context>, width: usize, height: usize, depth: usize) -> Result<Self> {
        let () = T::LAYOUT;
        context.bind()?;
        // SAFETY: the context is bound above, and the returned allocation is owned by the new value.
        let pitched = unsafe { DeviceMemory::<T>::alloc_pitched_3d(width, height, depth)? };
        let ptr = pitched.ptr.cast();
        let pitch = to_usize(pitched.pitch, "allocation pitch")?;
        let allocation = own_pitched_allocation(context, ptr, width, height, depth, pitch)?;
        Ok(Self {
            allocation,
            ptr,
            width,
            height,
            depth,
            pitch,
            _phantom: PhantomData,
        })
    }

    /// Allocates pitched storage whose logical elements are uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error if size calculations overflow, a nonempty zero-sized element range is requested, context binding fails, or CUDA cannot allocate pitched storage.
    /// Also rejects a returned pointer or pitch that violates the requested row size, alignment, or address-range bounds.
    pub fn create_uninit(
        context: &Arc<Context>,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<PitchedDeviceVolume<MaybeUninit<T>>> {
        PitchedDeviceVolume::<MaybeUninit<T>>::allocate(context, width, height, depth)
    }

    /// Allocates pitched storage with caller-established initialization.
    ///
    /// # Safety
    ///
    /// Initialize every logical element before reading it.
    /// CUDA row padding remains uninitialized and must not be exposed as initialized bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if size calculations overflow, a nonempty zero-sized element range is requested, context binding fails, or CUDA cannot allocate pitched storage.
    /// Also rejects a returned pointer or pitch that violates the requested row size, alignment, or address-range bounds.
    pub unsafe fn create_unchecked(
        context: &Arc<Context>,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<Self> {
        Self::allocate(context, width, height, depth)
    }

    /// Allocates pitched storage and completes zero initialization of its logical elements.
    ///
    /// # Errors
    ///
    /// Propagates pitched allocation, size-validation, access-tracking, context-binding, CUDA fill, and synchronization errors.
    pub fn zeroed(context: &Arc<Context>, width: usize, height: usize, depth: usize) -> Result<Self>
    where
        T: ZeroableDeviceRepr,
    {
        let mut memory = Self::create_uninit(context, width, height, depth)?;
        memory.set_zeroes()?;
        // SAFETY: the completed fill initializes every logical element with a valid zero value.
        Ok(unsafe { memory.assume_init() })
    }

    /// Allocates pitched storage and completes a copy of contiguous row-major host values.
    ///
    /// # Errors
    ///
    /// Returns an error if the source length differs from the logical element count.
    /// Also propagates pitched allocation, size-validation, context-binding, CUDA copy, and synchronization errors.
    pub fn from_slice(
        context: &Arc<Context>,
        width: usize,
        height: usize,
        depth: usize,
        values: &[T],
    ) -> Result<Self> {
        Self::create_uninit(context, width, height, depth)?.initialize_from_host(values)
    }

    /// Takes ownership of a pitched allocation in `context`.
    ///
    /// # Safety
    ///
    /// `ptr` must be the base of a live CUDA allocation compatible with `cudaFree`, with no other destruction owner.
    /// It must contain initialized, valid `T` values at every logical element.
    /// The pitch and dimensions must describe its storage, and previous accesses must be ordered before future consumers.
    /// A null pointer is accepted for an empty shape.
    ///
    /// # Errors
    ///
    /// Returns an error if row or volume size calculations overflow or a nonempty zero-sized element range is requested.
    /// For nonempty storage, also rejects a null or misaligned pointer, a pitch smaller than the row width or not aligned for `T`, or an overflowing ending address.
    pub unsafe fn from_raw_parts(
        context: &Arc<Context>,
        ptr: *mut T,
        width: usize,
        height: usize,
        depth: usize,
        pitch: usize,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_pitched_bytes(ptr, width, height, depth, pitch)?;
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                context,
                ptr.cast(),
                bytes,
                DeviceDeallocation::Immediate,
            ),
            ptr,
            width,
            height,
            depth,
            pitch,
            _phantom: PhantomData,
        })
    }

    /// Transfers raw ownership, or returns this buffer unchanged if another owner retains it.
    ///
    /// The returned pointer must eventually be freed in the returned context.
    ///
    /// # Errors
    ///
    /// Returns the original wrapper if another owner, submission, graph, or foreign handle retains the allocation.
    #[expect(
        clippy::type_complexity,
        reason = "raw volume ownership includes its pointer, dimensions, pitch, and CUDA context"
    )]
    pub fn into_raw_parts(
        self,
    ) -> StdResult<(*mut T, usize, usize, usize, usize, Arc<Context>), Self> {
        let Self {
            allocation,
            ptr,
            width,
            height,
            depth,
            pitch,
            _phantom: PhantomData,
        } = self;
        match Arc::try_unwrap(allocation) {
            Ok(mut allocation) => {
                let context = Arc::clone(allocation.context());
                allocation.ptr = ptr::null_mut();
                Ok((ptr, width, height, depth, pitch, context))
            }
            Err(allocation) => Err(Self {
                allocation,
                ptr,
                width,
                height,
                depth,
                pitch,
                _phantom: PhantomData,
            }),
        }
    }

    /// Returns the allocation's retained CUDA context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Returns the shared owner of the physical allocation, including row padding.
    #[must_use]
    pub fn allocation(&self) -> &Arc<DeviceAllocation> {
        &self.allocation
    }

    /// Reinterprets logical row bytes while preserving pitch, capacity, and allocation identity.
    ///
    /// Row padding is excluded from the initialized representation.
    ///
    /// # Errors
    ///
    /// Returns the original allocation if `U` is zero-sized, its size does not divide the logical row byte width, or nonempty storage has a base pointer or pitch not aligned for `U`.
    ///
    /// # Panics
    ///
    /// Panics if the logical row byte width overflows `usize`.
    /// Construction validates this width, so this cannot occur for a valid pitched allocation.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<PitchedDeviceVolume<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let () = U::LAYOUT;
        let size = size_of::<U>();
        let row_bytes = self
            .row_byte_width()
            .expect("pitched row byte length overflow");
        if size == 0
            || !row_bytes.is_multiple_of(size)
            || (!self.is_empty()
                && (!(self.ptr as usize).is_multiple_of(align_of::<U>())
                    || !self.pitch.is_multiple_of(align_of::<U>())))
        {
            return Err(self);
        }
        Ok(PitchedDeviceVolume {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            width: row_bytes / size,
            height: self.height,
            depth: self.depth,
            pitch: self.pitch,
            _phantom: PhantomData,
        })
    }

    /// Erases logical element initialization without changing allocation ownership.
    #[must_use]
    pub fn into_uninit(self) -> PitchedDeviceVolume<MaybeUninit<T>> {
        PitchedDeviceVolume {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            width: self.width,
            height: self.height,
            depth: self.depth,
            pitch: self.pitch,
            _phantom: PhantomData,
        }
    }

    /// Returns the logical row width in elements.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns the logical number of rows per slice.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Returns the logical number of slices.
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the row stride in bytes.
    #[must_use]
    pub const fn pitch(&self) -> usize {
        self.pitch
    }

    /// Returns the base device pointer.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the mutable base device pointer.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns the base pointer as a CUDA device pointer wrapper.
    #[must_use]
    pub const fn as_device_ptr(&self) -> DevicePtr {
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
    }

    /// Returns the logical element count.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * height * depth` overflows.
    pub fn len(&self) -> Result<usize> {
        checked_volume(self.width, self.height, self.depth)
    }

    /// Returns the logical row width in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * size_of::<T>()` overflows or `T` is a non-empty zero-sized element type.
    pub fn row_byte_width(&self) -> Result<usize> {
        checked_bytes::<T>(self.width)
    }

    /// Returns the physical allocation span described by `pitch * height * depth`.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte-volume calculation overflows.
    pub fn byte_len(&self) -> Result<usize> {
        Ok(self.allocation.byte_len())
    }

    /// Returns whether any dimension is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || self.depth == 0
    }

    /// Returns a CUDA pitched pointer descriptor for this volume.
    ///
    /// The descriptor is suitable for `cudaMemcpy3D` and `cudaMemset3D`.
    /// CUDA expects the `xsize` field in bytes for linear memory, so this method reports the typed row width converted to bytes.
    #[must_use]
    pub fn pitched_ptr(&self) -> runtime::cudaPitchedPtr {
        runtime::cudaPitchedPtr {
            ptr: self.ptr.cast(),
            pitch: self.pitch as _,
            xsize: self.row_byte_width().unwrap_or(0) as _,
            ysize: self.height as _,
        }
    }

    fn linear_host_ptr(
        ptr: *mut T,
        width: usize,
        height: usize,
    ) -> Result<runtime::cudaPitchedPtr> {
        let row_bytes = checked_bytes::<T>(width)?;
        Ok(runtime::cudaPitchedPtr {
            ptr: ptr.cast(),
            pitch: row_bytes as _,
            xsize: row_bytes as _,
            ysize: height as _,
        })
    }

    /// Copies a contiguous host slice into this 3D pitched allocation.
    ///
    /// `src` must contain exactly `width * height * depth` elements in row-major slice order.
    ///
    /// # Errors
    ///
    /// Returns an error if `src` has the wrong length, byte-size arithmetic overflows, or CUDA rejects the 3D copy.
    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        if src.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }

        // SAFETY: the shape and lengths were checked above, so the copy stays inside both allocations.
        unsafe {
            DeviceMemory::<T>::copy_3d(
                self.pitched_ptr(),
                Self::linear_host_ptr(src.as_ptr().cast_mut(), self.width, self.height)?,
                self.width,
                self.height,
                self.depth,
                MemoryCopyKind::HostToDevice,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Copies this 3D pitched allocation into a contiguous host slice.
    ///
    /// `dst` must contain exactly `width * height * depth` elements and is filled in row-major slice order.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` has the wrong length, byte-size arithmetic overflows, or CUDA rejects the 3D copy.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `T: AnyBitPatternDeviceRepr`, so any device bytes are valid `T` values.
        unsafe { self.copy_to_host_unchecked(dst) }
    }

    /// Copies device values whose validity is established by the producing kernel.
    ///
    /// # Safety
    ///
    /// Every copied element must be initialized and valid for `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination length differs from the logical element count or transfer size calculations overflow.
    /// Also propagates access-tracking, context-binding, CUDA copy, and default-stream synchronization errors.
    pub unsafe fn copy_to_host_unchecked(&self, dst: &mut [T]) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        if dst.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }

        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            DeviceMemory::<T>::copy_3d(
                Self::linear_host_ptr(dst.as_mut_ptr(), self.width, self.height)?,
                self.pitched_ptr(),
                self.width,
                self.height,
                self.depth,
                MemoryCopyKind::DeviceToHost,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Copies another 3D pitched allocation with the same shape into this one.
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes differ, byte-size arithmetic overflows, or CUDA rejects the device-to-device 3D copy.
    pub fn copy_from_volume(&mut self, src: &Self) -> Result<()> {
        self.context().bind()?;
        if self.context() != src.context() {
            return Err(Error::InvalidValue);
        }

        if self.width != src.width || self.height != src.height || self.depth != src.depth {
            return Err(Error::InvalidMemoryAccess);
        }

        // SAFETY: the shape and lengths were checked above, so the copy stays inside both allocations.
        unsafe {
            DeviceMemory::<T>::copy_3d(
                self.pitched_ptr(),
                src.pitched_ptr(),
                self.width,
                self.height,
                self.depth,
                MemoryCopyKind::DeviceToDevice,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Enqueues a host-to-device 3D copy from a contiguous host slice.
    ///
    /// `src` must remain valid until `stream` reaches the copy.
    /// The [`StreamScope`] lifetime enforces that requirement for safe callers.
    ///
    /// # Errors
    ///
    /// Returns an error if `src` has the wrong length, byte-size arithmetic overflows, or CUDA cannot enqueue the copy.
    pub fn copy_from_host_async<'env>(
        &'env mut self,
        src: &'env [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        if self.context() != stream.stream().context() {
            return Err(Error::InvalidValue);
        }

        self.context().bind()?;
        if src.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }

        let host = Self::linear_host_ptr(src.as_ptr().cast_mut(), self.width, self.height)?;
        checked_volume(self.row_byte_width()?, self.height, self.depth)?;
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                DeviceMemory::<T>::copy_3d_async(
                    self.pitched_ptr(),
                    host,
                    self.width,
                    self.height,
                    self.depth,
                    MemoryCopyKind::HostToDevice,
                    stream.stream(),
                )
            })
        }
    }

    /// Enqueues a device-to-host 3D copy into a contiguous host slice.
    ///
    /// `dst` must remain valid until `stream` reaches the copy.
    /// The [`StreamScope`] lifetime enforces that requirement for safe callers.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` has the wrong length, byte-size arithmetic overflows, or CUDA cannot enqueue the copy.
    pub fn copy_to_host_async<'env>(
        &'env self,
        dst: &'env mut [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let accesses = [Access::new(self.region(), AccessMode::Read)];
        if self.context() != stream.stream().context() {
            return Err(Error::InvalidValue);
        }

        self.context().bind()?;
        let () = T::LAYOUT;
        if dst.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }

        let host = Self::linear_host_ptr(dst.as_mut_ptr(), self.width, self.height)?;
        checked_volume(self.row_byte_width()?, self.height, self.depth)?;
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                DeviceMemory::<T>::copy_3d_async(
                    host,
                    self.pitched_ptr(),
                    self.width,
                    self.height,
                    self.depth,
                    MemoryCopyKind::DeviceToHost,
                    stream.stream(),
                )
            })
        }
    }

    /// Fills this 3D pitched allocation with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the 3D memset.
    pub fn set_zeroes(&mut self) -> Result<()>
    where
        T: ZeroableDeviceRepr,
    {
        // SAFETY: `T: ZeroableDeviceRepr`, so all-zero bytes are a valid `T`.
        unsafe { self.set_value_unchecked(0) }
    }

    /// Fills this 3D pitched allocation with `value` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the 3D memset.
    pub fn set_value(&mut self, value: u8) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `T: AnyBitPatternDeviceRepr`, so repeating `value` produces valid `T` values.
        unsafe { self.set_value_unchecked(value) }
    }

    /// Fills storage with a byte pattern known to be valid for `T`.
    ///
    /// # Safety
    ///
    /// The repeated byte pattern must represent a valid value of `T`.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking, context-binding, transfer-size validation, CUDA fill, and default-stream synchronization errors.
    pub unsafe fn set_value_unchecked(&mut self, value: u8) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            DeviceMemory::<T>::set_3d(
                self.pitched_ptr(),
                value,
                self.width,
                self.height,
                self.depth,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Enqueues a byte fill of this 3D pitched allocation on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the 3D memset.
    pub fn set_value_async<'env>(
        &'env mut self,
        value: u8,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        if self.context() != stream.stream().context() {
            return Err(Error::InvalidValue);
        }

        self.context().bind()?;
        let () = T::LAYOUT;
        checked_volume(self.row_byte_width()?, self.height, self.depth)?;
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                DeviceMemory::<T>::set_3d_async(
                    self.pitched_ptr(),
                    value,
                    self.width,
                    self.height,
                    self.depth,
                    stream.stream(),
                )
            })
        }
    }

    fn row_region(&self, slice: usize, row: usize) -> Result<DeviceRegion<'_>> {
        if slice >= self.depth || row >= self.height {
            return Err(Error::InvalidMemoryAccess);
        }

        if self.width == 0 {
            return self.region().slice(0..0);
        }

        let start = slice
            .checked_mul(self.height)
            .and_then(|offset| offset.checked_add(row))
            .and_then(|offset| offset.checked_mul(self.pitch))
            .ok_or(Error::InvalidMemoryAccess)?;
        let end = start
            .checked_add(self.row_byte_width()?)
            .ok_or(Error::InvalidMemoryAccess)?;
        self.region().slice(start..end)
    }

    /// Borrows one initialized logical row, excluding CUDA's row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice or row index is out of bounds, row offset calculations overflow, or the resulting range fails byte-length or alignment validation.
    pub fn row(&self, slice: usize, row: usize) -> Result<DeviceView<'_, T>> {
        // SAFETY: construction or promotion establishes initialized logical elements.
        unsafe { DeviceView::from_raw_parts(self.row_region(slice, row)?, self.width) }
    }

    /// Exclusively borrows one logical row, excluding CUDA's row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice or row index is out of bounds, row offset calculations overflow, or the resulting range fails byte-length or alignment validation.
    pub fn row_mut(&mut self, slice: usize, row: usize) -> Result<DeviceViewMut<'_, T>> {
        // SAFETY: the exclusive buffer borrow covers this exact row and its initialized element type.
        unsafe { DeviceViewMut::from_raw_parts(self.row_region(slice, row)?, self.width) }
    }
}

impl<T: DeviceRepr> PitchedDeviceVolume<MaybeUninit<T>> {
    /// Copies every logical element from host values and completes before promotion.
    ///
    /// # Errors
    ///
    /// Returns an error if the source length differs from the logical element count or transfer size calculations overflow.
    /// Also propagates context-binding, CUDA copy, and default-stream synchronization errors.
    pub fn initialize_from_host(self, values: &[T]) -> Result<PitchedDeviceVolume<T>> {
        if values.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }

        self.context().bind()?;
        let row_bytes = self.row_byte_width()?;
        let source = runtime::cudaPitchedPtr {
            ptr: values.as_ptr().cast_mut().cast(),
            pitch: row_bytes as _,
            xsize: row_bytes as _,
            ysize: self.height as _,
        };
        // SAFETY: both operands cover the complete logical volume, with matching element sizes.
        unsafe {
            DeviceMemory::<T>::copy_3d(
                self.pitched_ptr(),
                source,
                self.width,
                self.height,
                self.depth,
                MemoryCopyKind::HostToDevice,
            )?;
        }

        StreamBinding::Default(Arc::clone(self.context())).synchronize()?;
        // SAFETY: the completed full copy initializes all logical elements to valid T values.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks all logical elements initialized without changing the owner or pitch.
    ///
    /// # Safety
    ///
    /// Every logical element must contain an initialized valid T value.
    /// Producers must be ordered before subsequent consumers and complete before host readback.
    /// Row padding remains uninitialized.
    #[must_use]
    pub unsafe fn assume_init(self) -> PitchedDeviceVolume<T> {
        let () = T::LAYOUT;
        PitchedDeviceVolume {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            width: self.width,
            height: self.height,
            depth: self.depth,
            pitch: self.pitch,
            _phantom: PhantomData,
        }
    }
}
