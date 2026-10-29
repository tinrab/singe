use std::{
    marker::PhantomData,
    mem::{MaybeUninit, align_of, size_of},
    sync::Arc,
};

use singe_cuda::{
    context::Context,
    memory::transfer,
    types::{Complex32, Complex64},
    view::{
        AllocationId, AnyBitPatternDeviceRepr, DeviceInput, DeviceOutput, DeviceRegion,
        DeviceRegionProvider, DeviceRepr, DeviceSlice, DeviceSliceMut,
    },
};
use singe_npp_sys as sys;

use crate::{
    error::{Error, Result},
    memory::{Allocation, AllocationKind, RawAllocation},
    signal::view::{SignalOutputView, SignalView, SignalViewMut},
    types::{ComplexI16, ComplexI32, ComplexI64, DataTypeLike},
    utility::to_u64,
};

/// An initialized nonempty signal, or explicit `MaybeUninit<T>` output storage.
/// The retained allocation owns the actual CUDA context and native NPP memory.
#[derive(Debug)]
pub struct Signal<T> {
    allocation: Arc<Allocation>,
    length: usize,
    _t: PhantomData<T>,
}

/// Supported native signal allocation types.
pub trait SupportedSignal: DataTypeLike + private::Allocate {}

impl<T: DataTypeLike + private::Allocate> SupportedSignal for T {}

mod private {
    pub trait Allocate: Sized {
        unsafe fn allocate(length: u64) -> *mut Self;
    }
}

macro_rules! impl_allocate {
    ($ty:ty, $ffi:ident) => {
        impl private::Allocate for $ty {
            unsafe fn allocate(length: u64) -> *mut Self {
                // SAFETY: forwards a checked nonzero native element count.
                unsafe { sys::$ffi(length).cast() }
            }
        }
    };
}

impl_allocate!(u8, nppsMalloc_8u);

impl_allocate!(i8, nppsMalloc_8s);

impl_allocate!(u16, nppsMalloc_16u);

impl_allocate!(i16, nppsMalloc_16s);

impl_allocate!(u32, nppsMalloc_32u);

impl_allocate!(i32, nppsMalloc_32s);

impl_allocate!(i64, nppsMalloc_64s);

impl_allocate!(f32, nppsMalloc_32f);

impl_allocate!(f64, nppsMalloc_64f);

impl_allocate!(ComplexI16, nppsMalloc_16sc);

impl_allocate!(ComplexI32, nppsMalloc_32sc);

impl_allocate!(ComplexI64, nppsMalloc_64sc);

impl_allocate!(Complex32, nppsMalloc_32fc);

impl_allocate!(Complex64, nppsMalloc_64fc);

impl<T: SupportedSignal> Signal<T> {
    /// Allocates nonempty write-only storage without claiming initialized values.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` is zero, the byte size overflows or cannot be represented by the native interface, CUDA context binding fails, or NPP returns a null allocation.
    /// Also returns an error if allocation metadata is invalid or the returned address is not aligned for `T`.
    pub fn create_uninit(context: &Arc<Context>, length: usize) -> Result<Signal<MaybeUninit<T>>> {
        let () = T::LAYOUT;
        if length == 0 {
            return Err(Error::out_of_range("signal length"));
        }
        let bytes = length
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::out_of_range("signal allocation size"))?;
        let native_length = to_u64(length, "signal length")?;
        context.bind()?;
        // SAFETY: native allocation receives a checked element count and writes no T values.
        let pointer = unsafe { T::allocate(native_length) };
        if pointer.is_null() {
            return Err(Error::NullHandle);
        }
        let parts = RawAllocation {
            pointer: pointer.cast(),
            bytes,
            context: Arc::clone(context),
            id: AllocationId::create(),
            kind: AllocationKind::Signal,
        };
        // SAFETY: native allocation establishes this complete live range in the bound context.
        let allocation = Arc::new(unsafe { Allocation::from_raw(parts)? });
        if !(pointer as usize).is_multiple_of(align_of::<T>()) {
            return Err(Error::out_of_range("signal allocation alignment"));
        }
        Ok(Signal {
            allocation,
            length,
            _t: PhantomData,
        })
    }

    /// Returns a completed initialized upload in a native NPP signal allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice is empty, allocating native signal storage fails, or completing the host-to-device copy fails.
    pub fn from_slice(context: &Arc<Context>, values: &[T]) -> Result<Self> {
        let mut signal = Self::create_uninit(context, values.len())?;
        transfer::copy_from_host(&mut signal.output(), values)?;
        // SAFETY: the completed upload wrote every element from a valid host value.
        Ok(unsafe { signal.assume_init() })
    }
}

impl<T> Signal<T> {
    /// Returns the number of elements in the signal.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the signal is empty; valid signals are nonempty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the CUDA context retained by the native allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Returns the shared native allocation owner without granting additional typed access.
    #[must_use]
    pub fn allocation(&self) -> &Arc<Allocation> {
        &self.allocation
    }

    /// Returns a borrowed const device pointer to the signal.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.allocation.as_ptr().cast()
    }

    /// Returns a borrowed mutable device pointer to the signal.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.allocation.as_ptr().cast()
    }

    /// Borrows the initialized signal as an input view.
    ///
    /// # Panics
    ///
    /// Panics if the owned allocation metadata cannot form a nonempty aligned signal view.
    #[must_use]
    pub fn view(&self) -> SignalView<'_, T> {
        // SAFETY: initialized construction or explicit promotion establishes every T value.
        unsafe { SignalView::from_raw_parts(self.region(), self.length) }
            .expect("owned signal view metadata is valid")
    }

    /// Borrows the initialized signal for exclusive read-write access.
    ///
    /// # Panics
    ///
    /// Panics if the owned allocation metadata cannot form a nonempty aligned signal view.
    pub fn view_mut(&mut self) -> SignalViewMut<'_, T> {
        // SAFETY: the typed range is initialized and borrowed exclusively.
        unsafe { SignalViewMut::from_raw_parts(self.region(), self.length) }
            .expect("owned signal view metadata is valid")
    }

    /// Borrows initialized storage as a pure destination without exposing old values.
    ///
    /// # Panics
    ///
    /// Panics if the owned allocation metadata cannot form a nonempty aligned signal view.
    pub fn output_view(&mut self) -> SignalOutputView<'_, T> {
        // SAFETY: the allocation is live and exclusively borrowed for this output.
        unsafe { SignalOutputView::from_raw_parts(self.region(), self.length) }
            .expect("owned signal view metadata is valid")
    }

    /// Transfers uniquely retained native allocation ownership and logical length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OwnershipShared`] if another owner retains the allocation.
    pub fn into_raw(self) -> Result<(RawAllocation, usize)> {
        let owner = Arc::try_unwrap(self.allocation).map_err(|_| Error::OwnershipShared)?;
        Ok((owner.into_raw(), self.length))
    }
}

impl<T: DeviceRepr> Signal<T> {
    /// # Safety
    /// The native signal allocation must contain `length` valid T values, with
    /// pending production ordered before every consumer and host readback.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation is empty, its pointer is null, or its device-region metadata is invalid.
    /// Also returns an error if the element count is zero, its byte size overflows or exceeds the allocation, the allocation kind is not a signal, or the pointer is not aligned for `T`.
    pub unsafe fn from_raw(parts: RawAllocation, length: usize) -> Result<Self> {
        // Take native ownership before validating the typed range so errors free it.
        // SAFETY: the caller transfers a live, uniquely owned native allocation.
        let allocation = Arc::new(unsafe { Allocation::from_raw(parts)? });
        let () = T::LAYOUT;
        let bytes = length
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::out_of_range("signal allocation size"))?;
        if length == 0
            || bytes > allocation.byte_len()
            || allocation.kind() != AllocationKind::Signal
            || !allocation.as_ptr().addr().is_multiple_of(align_of::<T>())
        {
            return Err(Error::out_of_range("native signal range"));
        }
        Ok(Self {
            allocation,
            length,
            _t: PhantomData,
        })
    }

    /// Copies the initialized signal to an owned host vector and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error if preparing or completing the device-to-host transfer fails.
    pub fn copy_to_host_vec(&self) -> Result<Vec<T>>
    where
        T: AnyBitPatternDeviceRepr,
    {
        Ok(transfer::copy_to_host_vec(self)?)
    }
}

impl<T> Signal<MaybeUninit<T>> {
    /// Borrows the uninitialized signal as writable output storage.
    ///
    /// # Panics
    ///
    /// Panics if the owned allocation metadata cannot form a nonempty aligned signal view.
    pub fn output(&mut self) -> SignalOutputView<'_, T> {
        // SAFETY: MaybeUninit has T's layout and grants exclusive writable storage.
        unsafe { SignalOutputView::from_raw_parts(self.region(), self.length) }
            .expect("owned signal view metadata is valid")
    }

    /// Exposes every element after completed successful production.
    /// # Safety
    /// Every element must contain a valid T; producing work must be ordered before
    /// all consumers and host readback. Enqueue success alone is insufficient.
    #[must_use]
    pub unsafe fn assume_init(self) -> Signal<T> {
        Signal {
            allocation: self.allocation,
            length: self.length,
            _t: PhantomData,
        }
    }

    /// Exposes a nonempty initialized prefix while retaining the full allocation.
    /// # Safety
    /// The prefix must contain valid T values ordered before every consumer.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` is zero or exceeds the allocated element count.
    pub unsafe fn assume_init_prefix(mut self, length: usize) -> Result<Signal<T>> {
        if length == 0 || length > self.length {
            return Err(Error::out_of_range("signal initialized prefix"));
        }
        self.length = length;
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { self.assume_init() })
    }
}

// SAFETY: construction checks the typed range and promotion only narrows it.
unsafe impl<T> DeviceRegionProvider for Signal<T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.allocation
            .region()
            .slice(..self.length * size_of::<T>())
            .expect("signal range fits its allocation")
    }
}

// SAFETY: valid typed constructors and explicit promotion establish T initialization.
unsafe impl<T: DeviceRepr> DeviceInput<T> for Signal<T> {
    fn as_device_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.length
    }
}

// SAFETY: the typed owner or view preserves initialized values for its borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for Signal<T> {}

// SAFETY: mutable access uses an exclusive borrow of the typed owner.
unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for Signal<T> {}

// SAFETY: both initialized and MaybeUninit values have aligned writable storage.
unsafe impl<T: DeviceRepr> DeviceOutput<T> for Signal<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

// SAFETY: this grants writes to T storage and no initialized T input access.
// SAFETY: output metadata describes the same live typed range without initialized reads.
unsafe impl<T: DeviceRepr> DeviceInput<T> for Signal<MaybeUninit<T>> {
    fn len(&self) -> usize {
        self.length
    }
}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for Signal<MaybeUninit<T>> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.allocation.as_ptr().cast()
    }
}
