//! Context-owned NCCL allocations with explicit initialization state.

use std::{
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit, align_of, size_of},
    ptr,
    result::Result as StdResult,
    sync::Arc,
};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    context::Context,
    memory::transfer,
    view::{
        AllocationId, DeviceOutput, DeviceOutputView, DeviceRange, DeviceRegion,
        DeviceRegionProvider, DeviceRepr, DeviceSlice, DeviceSliceMut, DeviceView, DeviceViewMut,
    },
};

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    utility::{checked_byte_len, ensure_primary_context, to_u64},
};

/// NCCL's native allocation owner, independent of typed initialization and access.
///
/// Retaining this owner keeps every subrange live without granting read or write access.
/// NCCL may allocate additional physical capacity; only requested bytes are exposed through this owner.
#[derive(Debug)]
pub struct Allocation {
    pointer: *mut u8,
    bytes: usize,
    context: Arc<Context>,
    id: AllocationId,
}

// SAFETY: the pointer is a device address never dereferenced on the host, and `Drop` binds the retained context before `ncclMemFree`, which may run on any thread.
unsafe impl Send for Allocation {}

// SAFETY: `&self` methods only read metadata and describe regions; none calls into NCCL.
unsafe impl Sync for Allocation {}

/// A uniquely accessible typed range in an NCCL allocation.
///
/// Allocate `MaybeUninit<T>` storage through [`Self::create_uninit`], then expose only values established by successful completed production.
#[derive(Debug)]
pub struct Memory<T: DeviceRepr> {
    allocation: Arc<Allocation>,
    length: usize,
    _t: PhantomData<T>,
}

/// Native allocation ownership transferred with its bounds, identity, and context.
/// Free `pointer` with [`sys::ncclMemFree`] before releasing the context, or transfer the complete value through [`Memory::from_raw`].
#[derive(Debug)]
pub struct RawMemory<T: DeviceRepr> {
    /// The transferred native allocation pointer.
    pub pointer: *mut T,
    /// Number of elements in the typed range.
    pub length: usize,
    /// Requested allocation capacity, in bytes.
    pub allocation_bytes: usize,
    /// Allocation identity used to coordinate device accesses.
    pub allocation_id: AllocationId,
    /// The primary CUDA context that owns the allocation.
    pub context: Arc<Context>,
}

impl Allocation {
    /// Returns the CUDA context retained by the allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Returns the requested allocation size, in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.bytes
    }
}

impl<T: DeviceRepr> Memory<T> {
    /// Allocates uninitialized elements in the actual NCCL-compatible primary context.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation size overflows, the context is not the device primary CUDA context, CUDA context binding fails, or NCCL cannot allocate suitably aligned storage.
    pub fn create_uninit(context: &Arc<Context>, length: usize) -> Result<Memory<MaybeUninit<T>>> {
        let () = T::LAYOUT;
        let bytes = checked_byte_len::<T>(length, "nccl allocation size")?;
        let native_bytes = to_u64(bytes, "nccl allocation size")?;
        ensure_primary_context(context)?;
        context.bind()?;
        let mut pointer = ptr::NonNull::<T>::dangling().as_ptr().cast();
        if bytes != 0 {
            // SAFETY: `pointer` is a local output slot, and the context bound above is the device's primary context.
            unsafe {
                try_ffi!(sys::ncclMemAlloc(&raw mut pointer, native_bytes))?;
            }
            if pointer.is_null() {
                return Err(Error::NullPointer {
                    name: "nccl allocation".into(),
                });
            }
        }

        let allocation = Arc::new(Allocation {
            pointer: pointer.cast(),
            bytes,
            context: Arc::clone(context),
            id: AllocationId::create(),
        });
        if !allocation.pointer.addr().is_multiple_of(align_of::<T>()) {
            return Err(Error::out_of_range("nccl allocation alignment"));
        }
        Ok(Memory {
            allocation,
            length,
            _t: PhantomData,
        })
    }

    /// Copies valid host elements and returns their completed device allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if allocating NCCL memory or completing the host-to-device copy fails.
    pub fn from_slice(context: &Arc<Context>, values: &[T]) -> Result<Self> {
        let mut memory = Self::create_uninit(context, values.len())?;
        transfer::copy_from_host(&mut memory.output(), values)?;
        // SAFETY: the completed transfer initialized every element from valid host values.
        Ok(unsafe { memory.assume_init() })
    }

    /// Takes ownership of an NCCL allocation and its typed range.
    ///
    /// # Safety
    ///
    /// `raw.pointer` must be the base of a uniquely owned NCCL allocation in `raw.context`, covering `raw.allocation_bytes` bytes.
    /// Every exposed element must be a valid T, with pending writes ordered before consumers.
    /// Reuse the allocation identity for all aliases and retain those aliases' owners through their accesses.
    /// Empty allocations may use a null pointer.
    ///
    /// # Errors
    ///
    /// Returns an error if the context is not the device primary CUDA context, the typed size overflows or exceeds the allocation, the nonempty allocation pointer is null or misaligned, or the device-region metadata is invalid.
    pub unsafe fn from_raw(raw: RawMemory<T>) -> Result<Self> {
        let () = T::LAYOUT;
        ensure_primary_context(&raw.context)?;
        let bytes = checked_byte_len::<T>(raw.length, "nccl allocation size")?;
        if bytes > raw.allocation_bytes
            || raw.allocation_bytes != 0 && raw.pointer.is_null()
            || !raw.pointer.addr().is_multiple_of(align_of::<T>())
        {
            return Err(Error::out_of_range("nccl allocation range"));
        }

        let pointer = if raw.allocation_bytes == 0 {
            ptr::NonNull::<T>::dangling().as_ptr()
        } else {
            raw.pointer
        };
        // Check address arithmetic once at the native ownership boundary.
        // SAFETY: the caller guarantees `raw.pointer` is the base of a live NCCL allocation of `raw.allocation_bytes` bytes in `raw.context`, and that every alias shares `raw.allocation_id`.
        unsafe {
            DeviceRegion::from_raw_parts(
                &raw.context,
                raw.allocation_id,
                pointer.cast(),
                raw.allocation_bytes,
            )?;
        }
        Ok(Self {
            allocation: Arc::new(Allocation {
                pointer: pointer.cast(),
                bytes: raw.allocation_bytes,
                context: raw.context,
                id: raw.allocation_id,
            }),
            length: raw.length,
            _t: PhantomData,
        })
    }

    /// Transfers unique native allocation ownership without freeing it.
    ///
    /// # Errors
    ///
    /// Returns the original memory as `Err` if retained regions or another owner still share the allocation.
    pub fn into_raw(self) -> StdResult<RawMemory<T>, Self> {
        let Self {
            allocation, length, ..
        } = self;
        let allocation = match Arc::try_unwrap(allocation) {
            Ok(allocation) => ManuallyDrop::new(allocation),
            Err(allocation) => {
                return Err(Self {
                    allocation,
                    length,
                    _t: PhantomData,
                });
            }
        };

        // SAFETY: `allocation` is never dropped, so its only owned field, the context reference, is read out exactly once.
        let context = unsafe { ptr::read(&raw const allocation.context) };
        Ok(RawMemory {
            pointer: allocation.pointer.cast(),
            length,
            allocation_bytes: allocation.bytes,
            allocation_id: allocation.id,
            context,
        })
    }

    /// Returns the shared allocation owner without granting additional typed access.
    #[must_use]
    pub fn allocation(&self) -> &Arc<Allocation> {
        &self.allocation
    }

    /// Returns the CUDA context retained by this memory.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Returns the number of typed elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the typed range is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns a borrowed const device pointer to the typed range.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.allocation.pointer.cast()
    }

    /// Returns a borrowed mutable device pointer to the typed range.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.allocation.pointer.cast()
    }

    /// Returns the size of the typed range, in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        // Allocation and raw import checked the product; promotion only narrows it.
        self.length * size_of::<T>()
    }

    /// Borrows the initialized typed range as a device input view.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_, T> {
        DeviceView::from_buffer(self)
    }

    /// Exclusively borrows the initialized typed range as a device view.
    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        DeviceViewMut::from_buffer(self)
    }
}

impl<T: DeviceRepr> Memory<MaybeUninit<T>> {
    /// Borrows the uninitialized range as arbitrary-byte output storage.
    pub fn output(&mut self) -> DeviceOutputView<'_, T> {
        DeviceOutputView::from_output(self)
    }

    /// Exposes every element after caller-established initialization.
    ///
    /// # Safety
    ///
    /// Every element must contain a valid T, with producer writes ordered before every consumer and host readback.
    /// Enqueue success alone is insufficient.
    #[must_use]
    pub unsafe fn assume_init(self) -> Memory<T> {
        Memory {
            allocation: self.allocation,
            length: self.length,
            _t: PhantomData,
        }
    }

    /// Exposes an initialized prefix while retaining the complete allocation.
    ///
    /// # Safety
    ///
    /// The first `length` elements must contain valid T values.
    /// Producing work must be ordered before every consumer and host readback.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` exceeds the allocated element count.
    pub unsafe fn assume_init_prefix(mut self, length: usize) -> Result<Memory<T>> {
        if length > self.length {
            return Err(Error::out_of_range("initialized prefix"));
        }

        self.length = length;
        // SAFETY: the caller established initialization of the narrowed range.
        Ok(unsafe { self.assume_init() })
    }
}

// SAFETY: the owner retains exactly the checked native allocation and its context.
unsafe impl DeviceRegionProvider for Allocation {
    fn region(&self) -> DeviceRegion<'_> {
        // SAFETY: `pointer` and `bytes` come from `ncclMemAlloc` or a checked `Memory::from_raw` import in `context`, `self` keeps them allocated, and `id` names only this allocation.
        unsafe { DeviceRegion::from_raw_parts(&self.context, self.id, self.pointer, self.bytes) }
            .expect("owned nccl allocation metadata is valid")
    }
}

// SAFETY: typed ranges are checked at construction and only narrowed by promotion.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for Memory<T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.allocation
            .region()
            .slice(..self.byte_len())
            .expect("typed range fits nccl allocation")
    }
}

// SAFETY: initialized constructors and explicit promotion establish valid T values.
unsafe impl<T: DeviceRepr> DeviceRange<T> for Memory<T> {
    fn as_device_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.length
    }
}

// SAFETY: the typed owner or view preserves initialized values for its borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for Memory<T> {}

// SAFETY: Memory grants mutable typed access only through its exclusive borrow.
unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for Memory<T> {}

// SAFETY: both initialized and MaybeUninit values have aligned writable storage.
unsafe impl<T: DeviceRepr> DeviceOutput<T> for Memory<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

// SAFETY: `MaybeUninit<T>` has `T`'s layout, and the range reports the same live checked allocation without granting initialized reads.
unsafe impl<T: DeviceRepr> DeviceRange<T> for Memory<MaybeUninit<T>> {
    fn len(&self) -> usize {
        self.length
    }
}

// SAFETY: the pointer is the aligned base of a live allocation sized for `length` elements, and `&mut self` makes write access exclusive.
unsafe impl<T: DeviceRepr> DeviceOutput<T> for Memory<MaybeUninit<T>> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.allocation.pointer.cast()
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if self.bytes == 0 {
            return;
        }

        if let Err(error) = self.context.bind() {
            report_cleanup_failure("bind the cuda context to free", "nccl memory", &error);
            return;
        }

        // SAFETY: this is the last owner of a nonempty `ncclMemAlloc` base, and its context was bound above.
        unsafe {
            if let Err(error) = try_ffi!(sys::ncclMemFree(self.pointer.cast())) {
                report_cleanup_failure("free", "nccl memory", &error);
            }
        }
    }
}
