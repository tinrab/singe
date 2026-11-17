use crate::{
    error::Result,
    memory::{PinnedHostSlice, PinnedHostSliceMut, PinnedVec},
    view::{DeviceRepr, DeviceView, DeviceViewMut, HostRegion, HostSlice, HostSliceMut},
};

/// Extension methods giving device views of a [`PinnedVec`] through its mapping into the device's address space.
///
/// A kernel or copy given such a view reads and writes the host memory directly, without a copy to device memory first.
/// Every access crosses the bus, so a view suits data the device touches once, such as a few values a kernel reports back.
/// The access tracker sees the view and the host vector as the same bytes, so tracked work through the view orders against tracked host access.
pub trait PinnedVecExt<T: DeviceRepr> {
    /// Borrows the values through the vector's device mapping.
    ///
    /// # Errors
    ///
    /// Returns an error if the view's byte length or alignment does not fit `T`.
    fn mapped_device_view(&self) -> Result<DeviceView<'_, T>>;

    /// Exclusively borrows the values through the vector's device mapping.
    ///
    /// # Errors
    ///
    /// Returns an error if the view's byte length or alignment does not fit `T`.
    fn mapped_device_view_mut(&mut self) -> Result<DeviceViewMut<'_, T>>;
}

impl<T: DeviceRepr> PinnedVecExt<T> for PinnedVec<T> {
    fn mapped_device_view(&self) -> Result<DeviceView<'_, T>> {
        let region = self
            .allocator()
            .device_region(self.as_ptr().cast(), size_of_val(self.as_slice()))?;
        // SAFETY: the region aliases this vector's initialized values, which the shared borrow keeps alive and unchanged.
        unsafe { DeviceView::from_raw_parts(region, self.len()) }
    }

    fn mapped_device_view_mut(&mut self) -> Result<DeviceViewMut<'_, T>> {
        let bytes = size_of_val(self.as_slice());
        let region = self
            .allocator()
            .device_region(self.as_ptr().cast(), bytes)?;
        // SAFETY: the region aliases this vector's initialized values, which the exclusive borrow keeps alive and unaliased.
        unsafe { DeviceViewMut::from_raw_parts(region, self.len()) }
    }
}

// SAFETY: a `Vec` points at `len()` initialized, aligned values that it owns, and `&self` keeps them alive and unmodified.
unsafe impl<T: DeviceRepr> HostSlice<T> for PinnedVec<T> {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

// SAFETY: `as_mut_ptr` names the same values as `as_ptr`, and `&mut self` grants exclusive write access to them.
unsafe impl<T: DeviceRepr> HostSliceMut<T> for PinnedVec<T> {
    fn as_host_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

// SAFETY: every block of the pool is page-locked for as long as an owner holds the allocator, and the region names the pinned block's chunk, so every alias shares its identity and offsets.
unsafe impl<T: DeviceRepr> PinnedHostSlice<T> for PinnedVec<T> {
    fn host_region(&self) -> HostRegion<'_> {
        self.allocator()
            .host_region(self.as_ptr().cast(), size_of_val(self.as_slice()))
    }
}

// SAFETY: the `PinnedHostSlice` and `HostSliceMut` implementations above describe the same pinned values.
unsafe impl<T: DeviceRepr> PinnedHostSliceMut<T> for PinnedVec<T> {}
