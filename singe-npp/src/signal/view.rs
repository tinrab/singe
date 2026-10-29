//! Nonempty NPP signal views with actual CUDA allocation metadata.

use std::{
    marker::PhantomData,
    mem::{MaybeUninit, align_of, size_of},
    num::NonZeroUsize,
    ops::Range,
    sync::Arc,
};

use singe_cuda::{
    context::Context,
    memory::transfer,
    view::{
        AnyBitPatternDeviceRepr, DeviceInput, DeviceOutput, DeviceOutputView, DeviceRegion,
        DeviceRegionProvider, DeviceRepr, DeviceSlice, DeviceSliceMut, DeviceView, DeviceViewMut,
    },
};

use crate::error::{Error, Result};

fn checked_region<T>(
    region: DeviceRegion<'_>,
    length: usize,
) -> Result<(DeviceRegion<'_>, NonZeroUsize)> {
    let length = NonZeroUsize::new(length).ok_or_else(|| Error::out_of_range("signal length"))?;
    let bytes = length
        .get()
        .checked_mul(size_of::<T>())
        .ok_or_else(|| Error::out_of_range("signal byte length"))?;
    if !region.as_ptr().addr().is_multiple_of(align_of::<T>()) {
        return Err(Error::out_of_range("signal alignment"));
    }
    if bytes > region.byte_len() {
        return Err(Error::LengthMismatch {
            name: "signal memory".into(),
            expected: length.get(),
            actual: region.byte_len() / size_of::<T>().max(1),
        });
    }
    Ok((region.slice(..bytes)?, length))
}

/// Initialized nonempty signal input.
#[derive(Debug, Clone, Copy)]
pub struct SignalView<'a, T> {
    region: DeviceRegion<'a>,
    length: NonZeroUsize,
    _t: PhantomData<&'a [T]>,
}

impl<'a, T> SignalView<'a, T> {
    /// Borrows a checked CUDA range as a nonempty NPP signal.
    ///
    /// # Errors
    ///
    /// Returns an error if the length is zero, the byte-size calculation overflows, the pointer is not aligned for `T`, or the requested range exceeds the supplied device region.
    pub fn from_view(view: DeviceView<'a, T>) -> Result<Self>
    where
        T: DeviceRepr,
    {
        // SAFETY: the shared CUDA view establishes the matching access and lifetime.
        unsafe { Self::from_raw_parts(view.region(), view.len()) }
    }

    /// # Safety
    /// The region must contain valid T values and remain readable for the view lifetime.
    /// T must have a stable plain device representation. Retain its allocation
    /// through all queued accesses and captured replay using this view.
    ///
    /// # Errors
    ///
    /// Returns an error if the length is zero, the byte-size calculation overflows, the pointer is not aligned for `T`, or the requested range exceeds the supplied device region.
    pub unsafe fn from_raw_parts(region: DeviceRegion<'a>, length: usize) -> Result<Self> {
        let (region, length) = checked_region::<T>(region, length)?;
        Ok(Self {
            region,
            length,
            _t: PhantomData,
        })
    }

    /// Returns the number of elements in the view.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length.get()
    }

    /// Returns `false`; signal views are nonempty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Returns the CUDA context associated with this signal range.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.region.context()
    }

    /// Returns the device allocation metadata for this signal range.
    #[must_use]
    pub fn region(&self) -> DeviceRegion<'_> {
        self.region
    }

    /// Narrows the view while preserving its allocation identity and context.
    ///
    /// # Errors
    ///
    /// Returns an error if `range` is empty, reversed, or extends beyond this view, or if the resulting device region is invalid.
    pub fn slice(self, range: Range<usize>) -> Result<Self> {
        if range.start >= range.end || range.end > self.len() {
            return Err(Error::out_of_range("signal slice"));
        }
        // The constructor checked the full byte product, so these subranges fit.
        let region = self
            .region
            .slice(range.start * size_of::<T>()..range.end * size_of::<T>())?;
        // SAFETY: the region and geometry come from an allocation that this value owns or borrows.
        unsafe { Self::from_raw_parts(region, range.len()) }
    }

    /// Returns a borrowed const device pointer to the signal range.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.region.as_ptr().cast()
    }

    /// Copies this initialized range to an owned host vector and waits for completion.
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

// SAFETY: checked construction preserves the exact allocation range and lifetime.
unsafe impl<T> DeviceRegionProvider for SignalView<'_, T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.region
    }
}

// SAFETY: input/read-write construction establishes initialized T values.
unsafe impl<T: DeviceRepr> DeviceInput<T> for SignalView<'_, T> {
    fn as_device_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

// SAFETY: the typed owner or view preserves initialized values for its borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for SignalView<'_, T> {}

/// Initialized nonempty read-write signal storage.
#[derive(Debug)]
pub struct SignalViewMut<'a, T> {
    region: DeviceRegion<'a>,
    length: NonZeroUsize,
    _t: PhantomData<&'a mut [T]>,
}

impl<'a, T> SignalViewMut<'a, T> {
    /// Borrows a checked CUDA range as a nonempty NPP signal.
    ///
    /// # Errors
    ///
    /// Returns an error if the length is zero, the byte-size calculation overflows, the pointer is not aligned for `T`, or the requested range exceeds the supplied device region.
    pub fn from_view(view: DeviceViewMut<'a, T>) -> Result<Self>
    where
        T: DeviceRepr,
    {
        // SAFETY: the shared CUDA view establishes the matching access and lifetime.
        let length = view.len();
        unsafe { Self::from_raw_parts(view.into_region(), length) }
    }

    /// # Safety
    /// The region must contain valid T values and remain exclusively accessible for the view lifetime.
    /// T must have a stable plain device representation. Retain its allocation
    /// through all queued accesses and captured replay using this view.
    ///
    /// # Errors
    ///
    /// Returns an error if the length is zero, the byte-size calculation overflows, the pointer is not aligned for `T`, or the requested range exceeds the supplied device region.
    pub unsafe fn from_raw_parts(region: DeviceRegion<'a>, length: usize) -> Result<Self> {
        let (region, length) = checked_region::<T>(region, length)?;
        Ok(Self {
            region,
            length,
            _t: PhantomData,
        })
    }

    /// Returns the number of elements in the view.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length.get()
    }

    /// Returns `false`; signal views are nonempty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Returns the CUDA context associated with this signal range.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.region.context()
    }

    /// Returns the device allocation metadata for this signal range.
    #[must_use]
    pub fn region(&self) -> DeviceRegion<'_> {
        self.region
    }

    /// Narrows the view while preserving its allocation identity and context.
    ///
    /// # Errors
    ///
    /// Returns an error if `range` is empty, reversed, or extends beyond this view, or if the resulting device region is invalid.
    pub fn slice(self, range: Range<usize>) -> Result<Self> {
        if range.start >= range.end || range.end > self.len() {
            return Err(Error::out_of_range("signal slice"));
        }
        // The constructor checked the full byte product, so these subranges fit.
        let region = self
            .region
            .slice(range.start * size_of::<T>()..range.end * size_of::<T>())?;
        // SAFETY: the region and geometry come from an allocation that this value owns or borrows.
        unsafe { Self::from_raw_parts(region, range.len()) }
    }

    /// Returns a borrowed const device pointer to the signal range.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.region.as_ptr().cast()
    }

    /// Copies this initialized range to an owned host vector and waits for completion.
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

    /// Returns a borrowed mutable device pointer to the signal range.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.region.as_ptr().cast()
    }

    /// Reborrows this range as exclusively writable output storage.
    pub fn output(&mut self) -> SignalOutputView<'_, T> {
        SignalOutputView {
            region: self.region,
            length: self.length,
            _t: PhantomData,
        }
    }

    /// Reborrows this initialized range for read-only access.
    #[must_use]
    pub fn as_view(&self) -> SignalView<'_, T> {
        SignalView {
            region: self.region,
            length: self.length,
            _t: PhantomData,
        }
    }

    /// Reborrows this initialized range for exclusive read-write access.
    pub fn reborrow(&mut self) -> SignalViewMut<'_, T> {
        SignalViewMut {
            region: self.region,
            length: self.length,
            _t: PhantomData,
        }
    }
}

// SAFETY: checked construction preserves the exact allocation range and lifetime.
unsafe impl<T> DeviceRegionProvider for SignalViewMut<'_, T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.region
    }
}

// SAFETY: input/read-write construction establishes initialized T values.
unsafe impl<T: DeviceRepr> DeviceInput<T> for SignalViewMut<'_, T> {
    fn as_device_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

// SAFETY: the typed owner or view preserves initialized values for its borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for SignalViewMut<'_, T> {}

unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for SignalViewMut<'_, T> {}

// SAFETY: exclusive construction grants aligned writable storage of the checked length.
unsafe impl<T: DeviceRepr> DeviceOutput<T> for SignalViewMut<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

/// Nonempty write-only signal storage, including uninitialized destinations.
#[derive(Debug)]
pub struct SignalOutputView<'a, T> {
    region: DeviceRegion<'a>,
    length: NonZeroUsize,
    _t: PhantomData<&'a mut [MaybeUninit<T>]>,
}

impl<'a, T> SignalOutputView<'a, T> {
    /// Borrows a checked CUDA range as a nonempty NPP signal.
    ///
    /// # Errors
    ///
    /// Returns an error if the length is zero, the byte-size calculation overflows, the pointer is not aligned for `T`, or the requested range exceeds the supplied device region.
    pub fn from_output(view: DeviceOutputView<'a, T>) -> Result<Self>
    where
        T: DeviceRepr,
    {
        // SAFETY: the shared CUDA view establishes the matching access and lifetime.
        let length = view.len();
        unsafe { Self::from_raw_parts(view.into_region(), length) }
    }

    /// # Safety
    /// The region must remain exclusively writable for the view lifetime; initialization is not required.
    /// T must have a stable plain device representation. Retain its allocation
    /// through all queued accesses and captured replay using this view.
    ///
    /// # Errors
    ///
    /// Returns an error if the length is zero, the byte-size calculation overflows, the pointer is not aligned for `T`, or the requested range exceeds the supplied device region.
    pub unsafe fn from_raw_parts(region: DeviceRegion<'a>, length: usize) -> Result<Self> {
        let (region, length) = checked_region::<T>(region, length)?;
        Ok(Self {
            region,
            length,
            _t: PhantomData,
        })
    }

    /// Returns the number of elements in the view.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length.get()
    }

    /// Returns `false`; signal views are nonempty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Returns the CUDA context associated with this signal range.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.region.context()
    }

    /// Returns the device allocation metadata for this signal range.
    #[must_use]
    pub fn region(&self) -> DeviceRegion<'_> {
        self.region
    }

    /// Narrows the view while preserving its allocation identity and context.
    ///
    /// # Errors
    ///
    /// Returns an error if `range` is empty, reversed, or extends beyond this view, or if the resulting device region is invalid.
    pub fn slice(self, range: Range<usize>) -> Result<Self> {
        if range.start >= range.end || range.end > self.len() {
            return Err(Error::out_of_range("signal slice"));
        }
        // The constructor checked the full byte product, so these subranges fit.
        let region = self
            .region
            .slice(range.start * size_of::<T>()..range.end * size_of::<T>())?;
        // SAFETY: the region and geometry come from an allocation that this value owns or borrows.
        unsafe { Self::from_raw_parts(region, range.len()) }
    }

    /// Returns a borrowed mutable device pointer to the signal range.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.region.as_ptr().cast()
    }

    /// Reborrows this range as exclusively writable output storage.
    pub fn output(&mut self) -> SignalOutputView<'_, T> {
        SignalOutputView {
            region: self.region,
            length: self.length,
            _t: PhantomData,
        }
    }

    /// # Safety
    /// Every element must contain a valid T, and producing writes must be ordered
    /// before every consumer and host readback. Successful enqueue alone is insufficient.
    #[must_use]
    pub unsafe fn assume_init(self) -> SignalViewMut<'a, T> {
        SignalViewMut {
            region: self.region,
            length: self.length,
            _t: PhantomData,
        }
    }
}

// SAFETY: checked construction preserves the exact allocation range and lifetime.
unsafe impl<T> DeviceRegionProvider for SignalOutputView<'_, T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.region
    }
}

// SAFETY: exclusive construction grants aligned writable storage of the checked length.
// SAFETY: output metadata describes the same live typed range without initialized reads.
unsafe impl<T: DeviceRepr> DeviceInput<T> for SignalOutputView<'_, T> {
    fn len(&self) -> usize {
        self.len()
    }
}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for SignalOutputView<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

#[cfg(test)]
mod tests {
    use singe_cuda::memory::DeviceMemory;

    use super::*;

    #[test]
    fn test_signal_views_reject_zero_length_before_null_pointer() -> Result<()> {
        let context = Context::create()?;
        let mut memory = DeviceMemory::<u8>::zeroed(&context, 0)?;
        assert!(matches!(
            SignalView::from_view(memory.view()),
            Err(Error::OutOfRange { .. })
        ));
        assert!(matches!(
            SignalViewMut::from_view(memory.view_mut()),
            Err(Error::OutOfRange { .. })
        ));
        assert!(matches!(
            // SAFETY: the region covers the allocation, and the call only validates the geometry.
            unsafe { SignalView::<u8>::from_raw_parts(memory.region(), 0) },
            Err(Error::OutOfRange { .. })
        ));
        assert!(matches!(
            // SAFETY: the region covers the allocation, and the call only validates the geometry.
            unsafe { SignalViewMut::<u8>::from_raw_parts(memory.region(), 0) },
            Err(Error::OutOfRange { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_signal_views_reject_lengths_larger_than_device_memory() -> Result<()> {
        let context = Context::create()?;
        let memory = DeviceMemory::<u8>::zeroed(&context, 0)?;
        assert!(matches!(
            // SAFETY: the region covers the allocation, and the call only validates the geometry.
            unsafe { SignalView::<u8>::from_raw_parts(memory.region(), 1) },
            Err(Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            // SAFETY: the region covers the allocation, and the call only validates the geometry.
            unsafe { SignalViewMut::<u8>::from_raw_parts(memory.region(), 1) },
            Err(Error::LengthMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_signal_views_store_checked_raw_length_as_npp_size_t() -> Result<()> {
        let context = Context::create()?;
        let memory = DeviceMemory::from_slice(&context, &[0_u8; 23])?;
        {
            // SAFETY: the region covers the allocation, and the call only validates the geometry.
            let source = unsafe { SignalView::<u8>::from_raw_parts(memory.region(), 17)? };
            assert_eq!(source.len(), 17);
        }
        // SAFETY: the region covers the allocation, and the call only validates the geometry.
        let destination = unsafe { SignalViewMut::<u8>::from_raw_parts(memory.region(), 23)? };
        assert_eq!(destination.len(), 23);
        Ok(())
    }
}
