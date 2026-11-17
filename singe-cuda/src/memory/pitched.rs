use std::{marker::PhantomData, mem::MaybeUninit, ptr, result::Result as StdResult, sync::Arc};

use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    error::{Error, Result},
    memory::{DeviceAllocation, DeviceDeallocation, DeviceMemory, MemoryCopyKind},
    stream::{AsStream, StreamBinding, StreamScope},
    submission::{Access, AccessMode},
    types::DevicePtr,
    utility::{checked_bytes, checked_element_count, to_usize},
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
    let row_bytes = checked_bytes::<T>(width, "row bytes")?;
    checked_element_count(&[row_bytes, height, depth], "logical bytes")?;
    if width == 0 || height == 0 || depth == 0 {
        return Ok(0);
    }

    let bytes = checked_element_count(&[pitch, height, depth], "pitched bytes")?;
    if ptr.is_null() {
        return Err(Error::null_pointer("pitched pointer"));
    }

    if pitch < row_bytes {
        return Err(Error::buffer_too_small("pitch", row_bytes, pitch));
    }

    if !ptr.addr().is_multiple_of(align_of::<T>()) {
        return Err(Error::misaligned("pitched pointer", align_of::<T>()));
    }

    if !pitch.is_multiple_of(align_of::<T>()) {
        return Err(Error::misaligned("pitch", align_of::<T>()));
    }

    if ptr.addr().checked_add(bytes).is_none() {
        return Err(Error::out_of_range("pitched allocation end"));
    }

    Ok(bytes)
}

// Adopt a newly allocated pointer before checking CUDA's returned pitch, so a rejected layout still frees the allocation in its owning context.
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
        // SAFETY: the context is bound above, and `own_pitched_allocation` adopts the returned pointer before anything can fail, so it is freed in this context.
        let (ptr, pitch) = unsafe { DeviceMemory::<T>::alloc_pitched(context, width, height)? };
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
    /// Returns an error if size calculations overflow, a nonempty zero-sized element range is requested, the context cannot be bound, or CUDA cannot allocate pitched storage.
    /// Returns an error if CUDA returns a pointer or pitch that violates the requested row size, alignment, or address-range bounds.
    pub fn create_uninit(
        context: &Arc<Context>,
        width: usize,
        height: usize,
    ) -> Result<PitchedDeviceMemory<MaybeUninit<T>>> {
        PitchedDeviceMemory::<MaybeUninit<T>>::allocate(context, width, height)
    }

    /// Allocates pitched storage and completes zero initialization of its logical elements.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails for the reasons `create_uninit` gives, access tracking fails, or CUDA cannot fill or synchronize.
    pub fn zeroed(context: &Arc<Context>, width: usize, height: usize) -> Result<Self>
    where
        T: ZeroableDeviceRepr,
    {
        let mut memory = Self::create_uninit(context, width, height)?;
        memory.set_zeroes()?;
        // SAFETY: `set_zeroes` completed a zero fill of every logical element, and an all-zero `T` is valid.
        Ok(unsafe { memory.assume_init() })
    }

    /// Allocates pitched storage and completes a repeated byte fill of its logical elements.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails for the reasons `create_uninit` gives, access tracking fails, or CUDA cannot fill or synchronize.
    pub fn filled(context: &Arc<Context>, width: usize, height: usize, value: u8) -> Result<Self>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let mut memory = Self::create_uninit(context, width, height)?;
        memory.set_value(value)?;
        // SAFETY: `set_value` completed a fill of every logical element, and every bit pattern is a valid `T`.
        Ok(unsafe { memory.assume_init() })
    }

    /// Allocates pitched storage and completes a copy of contiguous row-major host values.
    ///
    /// # Errors
    ///
    /// Returns an error if the source length differs from the logical element count, allocation fails for the reasons `create_uninit` gives, or CUDA cannot copy or synchronize.
    pub fn from_slice(
        context: &Arc<Context>,
        width: usize,
        height: usize,
        values: &[T],
    ) -> Result<Self> {
        Self::create_uninit(context, width, height)?.initialize_from_host(values)
    }

    /// Takes ownership of a pitched allocation in `raw.context`.
    ///
    /// # Safety
    ///
    /// `raw.ptr` must be the base of a live CUDA allocation compatible with `cudaFree`, with no other destruction owner.
    /// It must contain initialized, valid `T` values at every logical element.
    /// The pitch and dimensions must describe its storage, and previous accesses must be ordered before future consumers.
    /// A null pointer is accepted for an empty shape.
    ///
    /// # Errors
    ///
    /// Returns an error if row or volume size calculations overflow or a nonempty zero-sized element range is requested.
    /// Returns an error if nonempty storage has a null or misaligned pointer, a pitch smaller than the row width or not aligned for `T`, or an overflowing ending address.
    pub unsafe fn from_raw(raw: RawPitchedDeviceMemory<T>) -> Result<Self> {
        let RawPitchedDeviceMemory {
            ptr,
            width,
            height,
            pitch,
            context,
        } = raw;
        let () = T::LAYOUT;
        let bytes = checked_pitched_bytes(ptr, width, height, 1, pitch)?;
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                &context,
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
    /// Returns the wrapper unchanged as the error if another owner, submission, graph, or foreign handle retains the allocation.
    pub fn into_raw(self) -> StdResult<RawPitchedDeviceMemory<T>, Self> {
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
                Ok(RawPitchedDeviceMemory {
                    ptr,
                    width,
                    height,
                    pitch,
                    context,
                })
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
    /// Returns the allocation unchanged as the error if `U` is zero-sized, its size does not divide the logical row byte width, or nonempty storage has a base pointer or pitch not aligned for `U`.
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
                && (!self.ptr.addr().is_multiple_of(align_of::<U>())
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
        // SAFETY: `self.ptr` is the base of the pitched allocation that `allocation` keeps live, and `DevicePtr` only carries the address.
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
    }

    /// Returns the logical element count.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * height` overflows.
    pub fn len(&self) -> Result<usize> {
        checked_element_count(&[self.width, self.height], "element count")
    }

    /// Returns the logical row width in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * size_of::<T>()` overflows or `T` is a non-empty zero-sized element type.
    pub fn row_byte_width(&self) -> Result<usize> {
        checked_bytes::<T>(self.width, "row bytes")
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
        let len = self.len()?;
        if src.len() != len {
            return Err(Error::length_mismatch("host source", len, src.len()));
        }

        let row_bytes = self.row_byte_width()?;
        // SAFETY: `src` holds exactly `width * height` values at a `row_bytes` pitch, construction validated this allocation's pitch and rows, the host access excludes tracked GPU work, and the context is bound.
        unsafe {
            DeviceMemory::<T>::copy_2d(
                self.context(),
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
    /// Returns an error if the destination length differs from the logical element count, transfer size calculations overflow, access tracking fails, the context cannot be bound, or CUDA cannot copy or synchronize.
    pub unsafe fn copy_to_host_unchecked(&self, dst: &mut [T]) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        let len = self.len()?;
        if dst.len() != len {
            return Err(Error::length_mismatch("host destination", len, dst.len()));
        }

        let row_bytes = self.row_byte_width()?;
        // SAFETY: `dst` holds exactly `width * height` writable values at a `row_bytes` pitch, the host access waited for tracked GPU work, and the caller guarantees the copied elements are valid `T` values.
        unsafe {
            DeviceMemory::<T>::copy_2d(
                self.context(),
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
            return Err(Error::MemoryContextMismatch);
        }

        if self.width != src.width {
            return Err(Error::length_mismatch(
                "source width",
                self.width,
                src.width,
            ));
        }

        if self.height != src.height {
            return Err(Error::length_mismatch(
                "source height",
                self.height,
                src.height,
            ));
        }

        // SAFETY: both allocations are in this bound context and have the same validated width and height, so each row copy stays inside both, and `&mut self` excludes aliasing with `src`.
        unsafe {
            DeviceMemory::<T>::copy_2d(
                self.context(),
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
    pub fn copy_from_host_scoped<'env>(
        &'env mut self,
        src: &'env [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        if self.context() != stream.stream().context() {
            return Err(Error::StreamContextMismatch);
        }

        self.context().bind()?;
        let len = self.len()?;
        if src.len() != len {
            return Err(Error::length_mismatch("host source", len, src.len()));
        }

        let row_bytes = self.row_byte_width()?;
        // SAFETY: `src` holds exactly the logical element count, the `'env` borrows keep it and this allocation live until the scope drains, the declared write orders tracked work, and the contexts match.
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
    pub fn copy_to_host_scoped<'env>(
        &'env self,
        dst: &'env mut [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let accesses = [Access::new(self.region(), AccessMode::Read)];
        if self.context() != stream.stream().context() {
            return Err(Error::StreamContextMismatch);
        }

        self.context().bind()?;
        let () = T::LAYOUT;
        let len = self.len()?;
        if dst.len() != len {
            return Err(Error::length_mismatch("host destination", len, dst.len()));
        }

        let row_bytes = self.row_byte_width()?;
        // SAFETY: `dst` holds exactly the logical element count, the `'env` borrows keep it and this allocation live until the scope drains, the declared read orders tracked work, and `T: AnyBitPatternDeviceRepr` makes any copied bytes valid.
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
    /// Returns an error if access tracking fails, the context cannot be bound, the transfer size overflows, or CUDA cannot fill the memory or synchronize the default stream.
    pub unsafe fn set_value_unchecked(&mut self, value: u8) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        // SAFETY: the validated pitch and shape keep the fill inside the allocation, the host access excludes tracked GPU work, and the caller guarantees the repeated byte is a valid `T` pattern.
        unsafe {
            DeviceMemory::<T>::set_2d(
                self.context(),
                self.ptr,
                self.pitch,
                value,
                self.width,
                self.height,
            )?;
        }
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Enqueues a byte fill of this pitched allocation on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the 2D memset.
    pub fn set_value_scoped<'env>(
        &'env mut self,
        value: u8,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        if self.context() != stream.stream().context() {
            return Err(Error::StreamContextMismatch);
        }

        self.context().bind()?;
        let () = T::LAYOUT;
        checked_element_count(&[self.row_byte_width()?, self.height], "logical bytes")?;
        // SAFETY: the `'env` borrow keeps this allocation live until the scope drains, the declared write orders tracked work, the contexts match, and `T: AnyBitPatternDeviceRepr` makes any repeated byte valid.
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
            return Err(Error::out_of_range("row"));
        }

        if self.width == 0 {
            return self.region().slice(0..0);
        }

        let start = row
            .checked_mul(self.pitch)
            .ok_or_else(|| Error::out_of_range("row offset"))?;
        let end = start
            .checked_add(self.row_byte_width()?)
            .ok_or_else(|| Error::out_of_range("row end"))?;
        self.region().slice(start..end)
    }

    /// Borrows one initialized logical row, excluding CUDA's row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the row index is out of bounds, row offset calculations overflow, or the resulting range fails byte-length or alignment validation.
    pub fn row(&self, row: usize) -> Result<DeviceView<'_, T>> {
        // SAFETY: a `T`-typed pitched owner only exists once every logical element is initialized, and the row region excludes the padding.
        unsafe { DeviceView::from_raw_parts(self.row_region(row)?, self.width) }
    }

    /// Exclusively borrows one logical row, excluding CUDA's row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the row index is out of bounds, row offset calculations overflow, or the resulting range fails byte-length or alignment validation.
    pub fn row_mut(&mut self, row: usize) -> Result<DeviceViewMut<'_, T>> {
        // SAFETY: the row region excludes the padding and holds initialized `T` values, and `&mut self` makes the view the only access to them.
        unsafe { DeviceViewMut::from_raw_parts(self.row_region(row)?, self.width) }
    }
}

impl<T: DeviceRepr> PitchedDeviceMemory<MaybeUninit<T>> {
    /// Copies every logical element from host values and completes before promotion.
    ///
    /// # Errors
    ///
    /// Returns an error if the source length differs from the logical element count, transfer size calculations overflow, the context cannot be bound, or CUDA cannot copy or synchronize.
    pub fn initialize_from_host(self, values: &[T]) -> Result<PitchedDeviceMemory<T>> {
        let len = self.len()?;
        if values.len() != len {
            return Err(Error::length_mismatch("host values", len, values.len()));
        }

        self.context().bind()?;
        // SAFETY: `values` holds exactly `width * height` values, the validated pitch and rows make the allocation writable for the whole rectangle, `self` is owned so nothing else accesses it, and the context is bound.
        unsafe {
            DeviceMemory::<T>::copy_2d(
                self.context(),
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
        // SAFETY: the synchronized copy wrote every logical element from initialized host values of `T`.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks all logical elements initialized without changing the owner or pitch.
    ///
    /// # Safety
    ///
    /// Every logical element must contain an initialized valid `T` value.
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

// SAFETY: the region is the whole pitched allocation, padding included, which `allocation` retains with its context for as long as `self` lives.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for PitchedDeviceMemory<T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.allocation.region()
    }
}

// SAFETY: `ptr` is a device address that the host never dereferences, the allocation behind it is owned through the `Send` `Arc<DeviceAllocation>`, and CUDA driver calls on it are thread-safe once the retained context is bound.
unsafe impl<T: DeviceRepr + Send> Send for PitchedDeviceMemory<T> {}

// SAFETY: `&self` methods only read the address and shape and enqueue CUDA work under the context's access tracker, and the driver API is thread-safe, so shared use from several threads performs no unsynchronized host access.
unsafe impl<T: DeviceRepr + Sync> Sync for PitchedDeviceMemory<T> {}

impl<T: DeviceRepr> PitchedDeviceVolume<T> {
    fn allocate(context: &Arc<Context>, width: usize, height: usize, depth: usize) -> Result<Self> {
        let () = T::LAYOUT;
        context.bind()?;
        // SAFETY: the context is bound above, and `own_pitched_allocation` adopts the returned pointer, so it is freed in this context.
        let pitched =
            unsafe { DeviceMemory::<T>::alloc_pitched_3d(context, width, height, depth)? };
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
    /// Returns an error if size calculations overflow, a nonempty zero-sized element range is requested, the context cannot be bound, or CUDA cannot allocate pitched storage.
    /// Returns an error if CUDA returns a pointer or pitch that violates the requested row size, alignment, or address-range bounds.
    pub fn create_uninit(
        context: &Arc<Context>,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<PitchedDeviceVolume<MaybeUninit<T>>> {
        PitchedDeviceVolume::<MaybeUninit<T>>::allocate(context, width, height, depth)
    }

    /// Allocates pitched storage and completes zero initialization of its logical elements.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails for the reasons `create_uninit` gives, access tracking fails, or CUDA cannot fill or synchronize.
    pub fn zeroed(context: &Arc<Context>, width: usize, height: usize, depth: usize) -> Result<Self>
    where
        T: ZeroableDeviceRepr,
    {
        let mut memory = Self::create_uninit(context, width, height, depth)?;
        memory.set_zeroes()?;
        // SAFETY: `set_zeroes` completed a zero fill of every logical element, and an all-zero `T` is valid.
        Ok(unsafe { memory.assume_init() })
    }

    /// Allocates pitched storage and completes a repeated byte fill of its logical elements.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails for the reasons `create_uninit` gives, access tracking fails, or CUDA cannot fill or synchronize.
    pub fn filled(
        context: &Arc<Context>,
        width: usize,
        height: usize,
        depth: usize,
        value: u8,
    ) -> Result<Self>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let mut memory = Self::create_uninit(context, width, height, depth)?;
        memory.set_value(value)?;
        // SAFETY: `set_value` completed a fill of every logical element, and every bit pattern is a valid `T`.
        Ok(unsafe { memory.assume_init() })
    }

    /// Allocates pitched storage and completes a copy of contiguous row-major host values.
    ///
    /// # Errors
    ///
    /// Returns an error if the source length differs from the logical element count, allocation fails for the reasons `create_uninit` gives, or CUDA cannot copy or synchronize.
    pub fn from_slice(
        context: &Arc<Context>,
        width: usize,
        height: usize,
        depth: usize,
        values: &[T],
    ) -> Result<Self> {
        Self::create_uninit(context, width, height, depth)?.initialize_from_host(values)
    }

    /// Takes ownership of a pitched allocation in `raw.context`.
    ///
    /// # Safety
    ///
    /// `raw.ptr` must be the base of a live CUDA allocation compatible with `cudaFree`, with no other destruction owner.
    /// It must contain initialized, valid `T` values at every logical element.
    /// The pitch and dimensions must describe its storage, and previous accesses must be ordered before future consumers.
    /// A null pointer is accepted for an empty shape.
    ///
    /// # Errors
    ///
    /// Returns an error if row or volume size calculations overflow or a nonempty zero-sized element range is requested.
    /// Returns an error if nonempty storage has a null or misaligned pointer, a pitch smaller than the row width or not aligned for `T`, or an overflowing ending address.
    pub unsafe fn from_raw(raw: RawPitchedDeviceVolume<T>) -> Result<Self> {
        let RawPitchedDeviceVolume {
            ptr,
            width,
            height,
            depth,
            pitch,
            context,
        } = raw;
        let () = T::LAYOUT;
        let bytes = checked_pitched_bytes(ptr, width, height, depth, pitch)?;
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                &context,
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
    /// Returns the wrapper unchanged as the error if another owner, submission, graph, or foreign handle retains the allocation.
    pub fn into_raw(self) -> StdResult<RawPitchedDeviceVolume<T>, Self> {
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
                Ok(RawPitchedDeviceVolume {
                    ptr,
                    width,
                    height,
                    depth,
                    pitch,
                    context,
                })
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
    /// Returns the allocation unchanged as the error if `U` is zero-sized, its size does not divide the logical row byte width, or nonempty storage has a base pointer or pitch not aligned for `U`.
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
                && (!self.ptr.addr().is_multiple_of(align_of::<U>())
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
        // SAFETY: `self.ptr` is the base of the pitched allocation that `allocation` keeps live, and `DevicePtr` only carries the address.
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
    }

    /// Returns the logical element count.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * height * depth` overflows.
    pub fn len(&self) -> Result<usize> {
        checked_element_count(&[self.width, self.height, self.depth], "element count")
    }

    /// Returns the logical row width in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * size_of::<T>()` overflows or `T` is a non-empty zero-sized element type.
    pub fn row_byte_width(&self) -> Result<usize> {
        checked_bytes::<T>(self.width, "row bytes")
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
    /// The descriptor suits CUDA's 3D copies and fills.
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
        let row_bytes = checked_bytes::<T>(width, "row bytes")?;
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
        let len = self.len()?;
        if src.len() != len {
            return Err(Error::length_mismatch("host source", len, src.len()));
        }

        // SAFETY: `src` holds exactly `width * height * depth` values in a linear layout, construction validated this allocation's pitch and shape, the host access excludes tracked GPU work, and the context is bound.
        unsafe {
            DeviceMemory::<T>::copy_3d(
                self.context(),
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
    /// Returns an error if the destination length differs from the logical element count, transfer size calculations overflow, access tracking fails, the context cannot be bound, or CUDA cannot copy or synchronize.
    pub unsafe fn copy_to_host_unchecked(&self, dst: &mut [T]) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        let len = self.len()?;
        if dst.len() != len {
            return Err(Error::length_mismatch("host destination", len, dst.len()));
        }

        // SAFETY: `dst` holds exactly `width * height * depth` writable values in a linear layout, the host access waited for tracked GPU work, and the caller guarantees the copied elements are valid `T` values.
        unsafe {
            DeviceMemory::<T>::copy_3d(
                self.context(),
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
            return Err(Error::MemoryContextMismatch);
        }

        if self.width != src.width {
            return Err(Error::length_mismatch(
                "source width",
                self.width,
                src.width,
            ));
        }

        if self.height != src.height {
            return Err(Error::length_mismatch(
                "source height",
                self.height,
                src.height,
            ));
        }

        if self.depth != src.depth {
            return Err(Error::length_mismatch(
                "source depth",
                self.depth,
                src.depth,
            ));
        }

        // SAFETY: both volumes are in this bound context and have the same validated width, height, and depth, so the copy stays inside both, and `&mut self` excludes aliasing with `src`.
        unsafe {
            DeviceMemory::<T>::copy_3d(
                self.context(),
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
    pub fn copy_from_host_scoped<'env>(
        &'env mut self,
        src: &'env [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        if self.context() != stream.stream().context() {
            return Err(Error::StreamContextMismatch);
        }

        self.context().bind()?;
        let len = self.len()?;
        if src.len() != len {
            return Err(Error::length_mismatch("host source", len, src.len()));
        }

        let host = Self::linear_host_ptr(src.as_ptr().cast_mut(), self.width, self.height)?;
        checked_element_count(
            &[self.row_byte_width()?, self.height, self.depth],
            "logical bytes",
        )?;
        // SAFETY: `src` holds exactly the logical element count, the `'env` borrows keep it and this volume live until the scope drains, the declared write orders tracked work, and the contexts match.
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
    pub fn copy_to_host_scoped<'env>(
        &'env self,
        dst: &'env mut [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let accesses = [Access::new(self.region(), AccessMode::Read)];
        if self.context() != stream.stream().context() {
            return Err(Error::StreamContextMismatch);
        }

        self.context().bind()?;
        let () = T::LAYOUT;
        let len = self.len()?;
        if dst.len() != len {
            return Err(Error::length_mismatch("host destination", len, dst.len()));
        }

        let host = Self::linear_host_ptr(dst.as_mut_ptr(), self.width, self.height)?;
        checked_element_count(
            &[self.row_byte_width()?, self.height, self.depth],
            "logical bytes",
        )?;
        // SAFETY: `dst` holds exactly the logical element count, the `'env` borrows keep it and this volume live until the scope drains, the declared read orders tracked work, and `T: AnyBitPatternDeviceRepr` makes any copied bytes valid.
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
    /// Returns an error if access tracking fails, the context cannot be bound, the transfer size overflows, or CUDA cannot fill the memory or synchronize the default stream.
    pub unsafe fn set_value_unchecked(&mut self, value: u8) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        // SAFETY: the validated pitch and shape keep the fill inside the volume, the host access excludes tracked GPU work, and the caller guarantees the repeated byte is a valid `T` pattern.
        unsafe {
            DeviceMemory::<T>::set_3d(
                self.context(),
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
    pub fn set_value_scoped<'env>(
        &'env mut self,
        value: u8,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        if self.context() != stream.stream().context() {
            return Err(Error::StreamContextMismatch);
        }

        self.context().bind()?;
        let () = T::LAYOUT;
        checked_element_count(
            &[self.row_byte_width()?, self.height, self.depth],
            "logical bytes",
        )?;
        // SAFETY: the `'env` borrow keeps this volume live until the scope drains, the declared write orders tracked work, the contexts match, and `T: AnyBitPatternDeviceRepr` makes any repeated byte valid.
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
        if slice >= self.depth {
            return Err(Error::out_of_range("slice"));
        }

        if row >= self.height {
            return Err(Error::out_of_range("row"));
        }

        if self.width == 0 {
            return self.region().slice(0..0);
        }

        let start = slice
            .checked_mul(self.height)
            .and_then(|offset| offset.checked_add(row))
            .and_then(|offset| offset.checked_mul(self.pitch))
            .ok_or_else(|| Error::out_of_range("row offset"))?;
        let end = start
            .checked_add(self.row_byte_width()?)
            .ok_or_else(|| Error::out_of_range("row end"))?;
        self.region().slice(start..end)
    }

    /// Borrows one initialized logical row, excluding CUDA's row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice or row index is out of bounds, row offset calculations overflow, or the resulting range fails byte-length or alignment validation.
    pub fn row(&self, slice: usize, row: usize) -> Result<DeviceView<'_, T>> {
        // SAFETY: a `T`-typed pitched owner only exists once every logical element is initialized, and the row region excludes the padding.
        unsafe { DeviceView::from_raw_parts(self.row_region(slice, row)?, self.width) }
    }

    /// Exclusively borrows one logical row, excluding CUDA's row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice or row index is out of bounds, row offset calculations overflow, or the resulting range fails byte-length or alignment validation.
    pub fn row_mut(&mut self, slice: usize, row: usize) -> Result<DeviceViewMut<'_, T>> {
        // SAFETY: the row region excludes the padding and holds initialized `T` values, and `&mut self` makes the view the only access to them.
        unsafe { DeviceViewMut::from_raw_parts(self.row_region(slice, row)?, self.width) }
    }
}

impl<T: DeviceRepr> PitchedDeviceVolume<MaybeUninit<T>> {
    /// Copies every logical element from host values and completes before promotion.
    ///
    /// # Errors
    ///
    /// Returns an error if the source length differs from the logical element count, transfer size calculations overflow, the context cannot be bound, or CUDA cannot copy or synchronize.
    pub fn initialize_from_host(self, values: &[T]) -> Result<PitchedDeviceVolume<T>> {
        let len = self.len()?;
        if values.len() != len {
            return Err(Error::length_mismatch("host values", len, values.len()));
        }

        self.context().bind()?;
        let row_bytes = self.row_byte_width()?;
        let source = runtime::cudaPitchedPtr {
            ptr: values.as_ptr().cast_mut().cast(),
            pitch: row_bytes as _,
            xsize: row_bytes as _,
            ysize: self.height as _,
        };
        // SAFETY: `values` holds exactly `width * height * depth` values in a linear layout, the validated pitch and shape make the volume writable throughout, `self` is owned so nothing else accesses it, and the context is bound.
        unsafe {
            DeviceMemory::<T>::copy_3d(
                self.context(),
                self.pitched_ptr(),
                source,
                self.width,
                self.height,
                self.depth,
                MemoryCopyKind::HostToDevice,
            )?;
        }

        StreamBinding::Default(Arc::clone(self.context())).synchronize()?;
        // SAFETY: the synchronized copy wrote every logical element from initialized host values of `T`.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks all logical elements initialized without changing the owner or pitch.
    ///
    /// # Safety
    ///
    /// Every logical element must contain an initialized valid `T` value.
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

/// Raw ownership of a pitched two-dimensional allocation and its context.
///
/// Free `ptr` while `context` is alive, or re-adopt the value with [`PitchedDeviceMemory::from_raw`].
#[derive(Debug)]
pub struct RawPitchedDeviceMemory<T> {
    /// Base of the allocation, or null for an empty shape.
    pub ptr: *mut T,
    /// Row width in elements.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
    /// Byte stride between rows.
    pub pitch: usize,
    /// Context the allocation belongs to.
    pub context: Arc<Context>,
}

/// Raw ownership of a pitched three-dimensional allocation and its context.
///
/// Free `ptr` while `context` is alive, or re-adopt the value with [`PitchedDeviceVolume::from_raw`].
#[derive(Debug)]
pub struct RawPitchedDeviceVolume<T> {
    /// Base of the allocation, or null for an empty shape.
    pub ptr: *mut T,
    /// Row width in elements.
    pub width: usize,
    /// Number of rows per slice.
    pub height: usize,
    /// Number of slices.
    pub depth: usize,
    /// Byte stride between rows.
    pub pitch: usize,
    /// Context the allocation belongs to.
    pub context: Arc<Context>,
}

/// Owned 2D pitched CUDA device memory.
///
/// The logical dimensions are expressed in elements of `T`; the CUDA pitch is stored as a byte stride between rows.
#[derive(Debug)]
pub struct PitchedDeviceMemory<T: DeviceRepr> {
    allocation: Arc<DeviceAllocation>,
    ptr: *mut T,
    width: usize,
    height: usize,
    pitch: usize,
    _phantom: PhantomData<T>,
}

/// Owned 3D pitched CUDA device memory.
///
/// The logical dimensions are expressed in elements of `T`; the CUDA pitch is stored as a byte stride between rows inside each slice.
#[derive(Debug)]
pub struct PitchedDeviceVolume<T: DeviceRepr> {
    allocation: Arc<DeviceAllocation>,
    ptr: *mut T,
    width: usize,
    height: usize,
    depth: usize,
    pitch: usize,
    _phantom: PhantomData<T>,
}

// SAFETY: the volume owns its whole retained allocation, so the allocation's region describes its pointers.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for PitchedDeviceVolume<T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.allocation.region()
    }
}

// SAFETY: the host never dereferences `ptr`, the owner holds `T` values like a `Vec<T>`, and every operation binds the retained context.
unsafe impl<T: DeviceRepr + Send> Send for PitchedDeviceVolume<T> {}

// SAFETY: `&PitchedDeviceVolume<T>` exposes only device pointers and metadata, and device writes need `&mut` or an unsafe raw API.
unsafe impl<T: DeviceRepr + Sync> Sync for PitchedDeviceVolume<T> {}
