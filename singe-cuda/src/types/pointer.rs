use std::ptr;

use singe_cuda_sys::driver;

use crate::view::{AnyBitPatternDeviceRepr, DeviceRepr, NoUninitDeviceRepr, ZeroableDeviceRepr};

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
/// CUDA host callback function pointer.
///
/// This is a borrowed function pointer value, not an owned CUDA resource.
pub struct HostFunction(driver::CUhostFn);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
/// CUDA device function handle.
///
/// CUDA owns the underlying function as part of a module, library, or runtime registration.
/// This wrapper is a copyable handle value and does not unload or destroy the function.
pub struct DeviceFunction(driver::CUfunction);

/// Unowned CUDA device address.
/// Copying this value retains no allocation and proves no bounds, lifetime, context association, alignment, or access permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct DevicePtr(*mut ());

// SAFETY: `DevicePtr` is a transparent pointer-sized address with no padding, and copying it never dereferences the address.
unsafe impl DeviceRepr for DevicePtr {}

// SAFETY: a pointer value has no padding, so every byte of a valid `DevicePtr` is initialized.
unsafe impl NoUninitDeviceRepr for DevicePtr {}

// SAFETY: the all-zero pattern is the null address, which `DevicePtr::null` also produces.
unsafe impl ZeroableDeviceRepr for DevicePtr {}

// SAFETY: `DevicePtr` is an opaque address that is never dereferenced through this type, so every bit pattern is a valid value.
unsafe impl AnyBitPatternDeviceRepr for DevicePtr {}

impl HostFunction {
    /// Wraps a raw CUDA host callback function pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a callback function pointer with the ABI and lifetime required by CUDA for every operation that uses the returned handle.
    #[must_use]
    pub const unsafe fn new(raw: driver::CUhostFn) -> Self {
        Self(raw)
    }

    /// Wraps a raw CUDA host callback function pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a callback function pointer with the ABI and lifetime required by CUDA for every operation that uses the returned handle.
    #[must_use]
    pub const unsafe fn from_raw(raw: driver::CUhostFn) -> Self {
        // SAFETY: `new` has the same requirement on `raw` as this function, which the caller guarantees.
        unsafe { Self::new(raw) }
    }

    /// Returns the stored native CUDA host callback pointer.
    #[must_use]
    pub const fn as_raw(self) -> driver::CUhostFn {
        self.0
    }
}

impl DeviceFunction {
    /// Wraps a raw CUDA device function handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid CUDA function handle whose owning module, library, or runtime registration remains loaded for every operation that uses the returned handle.
    pub const unsafe fn new(raw: driver::CUfunction) -> Self {
        Self(raw)
    }

    /// Wraps a raw CUDA device function handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid CUDA function handle whose owning module, library, or runtime registration remains loaded for every operation that uses the returned handle.
    pub const unsafe fn from_raw(raw: driver::CUfunction) -> Self {
        // SAFETY: `new` has the same requirement on `raw` as this function, which the caller guarantees.
        unsafe { Self::new(raw) }
    }

    /// Returns the stored native CUDA device function handle without transferring ownership.
    #[must_use]
    pub const fn as_raw(self) -> driver::CUfunction {
        self.0
    }

    /// Returns whether the stored native function handle is null.
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.0.is_null()
    }
}

impl DevicePtr {
    /// Creates a null device address without allocating memory.
    #[must_use]
    pub const fn null() -> Self {
        Self(ptr::null_mut())
    }

    /// Wraps a raw CUDA device pointer value.
    ///
    /// # Safety
    ///
    /// `raw` must be either null or a pointer value that is valid for the CUDA operation it is passed to.
    /// This wrapper does not prove allocation ownership, size, lifetime, context association, or access permissions.
    pub const unsafe fn new(raw: *mut ()) -> Self {
        Self(raw)
    }

    /// Wraps a raw CUDA device pointer value.
    ///
    /// # Safety
    ///
    /// `raw` must be either null or a pointer value that is valid for the CUDA operation it is passed to.
    /// This wrapper does not prove allocation ownership, size, lifetime, context association, or access permissions.
    pub const unsafe fn from_raw(raw: *mut ()) -> Self {
        // SAFETY: `new` has the same requirement on `raw` as this function, which the caller guarantees.
        unsafe { Self::new(raw.cast()) }
    }

    /// Returns the untyped device pointer without retaining or validating its allocation.
    #[must_use]
    pub const fn as_raw(self) -> *mut () {
        self.0.cast()
    }

    /// Returns the stored mutable device pointer without dereferencing it.
    #[must_use]
    pub const fn as_ptr(self) -> *mut () {
        self.0
    }

    /// Returns the stored device address as a const pointer without dereferencing it.
    #[must_use]
    pub const fn as_const_ptr(self) -> *const () {
        self.0.cast_const()
    }

    /// Returns whether the device address is null.
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.0.is_null()
    }

    /// The numeric device address.
    #[must_use]
    pub fn addr(self) -> usize {
        self.0.addr()
    }

    /// Whether the address is a multiple of `alignment`, which must be nonzero.
    #[must_use]
    pub fn is_aligned_to(self, alignment: usize) -> bool {
        self.addr().is_multiple_of(alignment)
    }

    /// Offsets the address by `bytes`, returning `None` when the address overflows.
    /// The result is an address computation only and is not checked against any allocation.
    #[must_use]
    pub fn checked_byte_add(self, bytes: usize) -> Option<Self> {
        self.addr()
            .checked_add(bytes)
            .map(|_| Self(self.0.wrapping_byte_add(bytes)))
    }

    /// Casts the device address to a pointer to `T` without checking alignment, bounds, or the allocation's element representation.
    #[must_use]
    pub const fn cast<T>(self) -> *mut T {
        self.0.cast()
    }
}

impl From<HostFunction> for driver::CUhostFn {
    fn from(value: HostFunction) -> Self {
        value.as_raw()
    }
}

impl From<DeviceFunction> for driver::CUfunction {
    fn from(value: DeviceFunction) -> Self {
        value.as_raw()
    }
}

impl From<DevicePtr> for *mut () {
    fn from(value: DevicePtr) -> Self {
        value.as_raw()
    }
}
