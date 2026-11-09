//! Custom memory allocators for nvJPEG.

use std::{
    ffi::{c_int, c_void},
    mem::{self, size_of},
};

use singe_nvjpeg_sys as sys;
#[cfg(feature = "k2")]
use singe_nvjpeg_sys::k2 as k2_sys;

/// Callback that allocates device memory for nvJPEG.
pub type DeviceMallocCallback = unsafe extern "C" fn(
    context: *mut c_void,
    ptr: *mut *mut c_void,
    size: usize,
    stream: *mut c_void,
) -> c_int;

/// Callback that frees device memory allocated by a [`DeviceMallocCallback`].
pub type DeviceFreeCallback = unsafe extern "C" fn(
    context: *mut c_void,
    ptr: *mut c_void,
    size: usize,
    stream: *mut c_void,
) -> c_int;

/// Callback that allocates pinned host memory for nvJPEG.
pub type PinnedMallocCallback = unsafe extern "C" fn(
    context: *mut c_void,
    ptr: *mut *mut c_void,
    size: usize,
    stream: *mut c_void,
) -> c_int;

/// Callback that frees pinned host memory allocated by a [`PinnedMallocCallback`].
pub type PinnedFreeCallback = unsafe extern "C" fn(
    context: *mut c_void,
    ptr: *mut c_void,
    size: usize,
    stream: *mut c_void,
) -> c_int;

/// Device memory allocation callbacks and the context pointer passed to them.
///
/// The context pointer is passed to the callbacks unchanged.
/// This type does not own what it points to; keep it valid for as long as any handle created with the allocator exists.
#[derive(Debug, Clone, Copy)]
pub struct DeviceAllocator {
    malloc: DeviceMallocCallback,
    free: DeviceFreeCallback,
    context: *mut c_void,
}

/// Pinned host memory allocation callbacks and the context pointer passed to them.
///
/// The context pointer is passed to the callbacks unchanged.
/// This type does not own what it points to; keep it valid for as long as any handle created with the allocator exists.
#[derive(Debug, Clone, Copy)]
pub struct PinnedAllocator {
    malloc: PinnedMallocCallback,
    free: PinnedFreeCallback,
    context: *mut c_void,
}

/// Device and pinned host allocators for a [`Context`](crate::context::Context).
#[derive(Debug, Clone, Copy)]
pub struct AllocatorSet {
    device: DeviceAllocator,
    pinned: PinnedAllocator,
}

impl DeviceAllocator {
    /// Creates a device allocator from its callbacks and context pointer.
    pub const fn new(
        malloc: DeviceMallocCallback,
        free: DeviceFreeCallback,
        context: *mut c_void,
    ) -> Self {
        Self {
            malloc,
            free,
            context,
        }
    }

    /// Returns the allocation callback.
    #[must_use]
    pub const fn malloc(&self) -> DeviceMallocCallback {
        self.malloc
    }

    /// Returns the free callback.
    #[must_use]
    pub const fn free(&self) -> DeviceFreeCallback {
        self.free
    }

    /// Returns the context pointer passed to the callbacks.
    #[must_use]
    pub const fn context(&self) -> *mut c_void {
        self.context
    }

    pub(crate) fn to_nvjpeg_raw(self) -> sys::nvjpegDevAllocatorV2_t {
        sys::nvjpegDevAllocatorV2_t {
            dev_malloc: Some(device_malloc_to_sys(self.malloc)),
            dev_free: Some(device_free_to_sys(self.free)),
            dev_ctx: self.context,
        }
    }

    #[cfg(feature = "k2")]
    pub(crate) fn to_nvjpeg2k_raw(self) -> k2_sys::nvjpeg2kDeviceAllocatorV2_t {
        k2_sys::nvjpeg2kDeviceAllocatorV2_t {
            device_malloc: Some(device_malloc_to_k2_sys(self.malloc)),
            device_free: Some(device_free_to_k2_sys(self.free)),
            device_ctx: self.context,
        }
    }
}

impl PinnedAllocator {
    /// Creates a pinned host allocator from its callbacks and context pointer.
    pub const fn new(
        malloc: PinnedMallocCallback,
        free: PinnedFreeCallback,
        context: *mut c_void,
    ) -> Self {
        Self {
            malloc,
            free,
            context,
        }
    }

    /// Returns the allocation callback.
    #[must_use]
    pub const fn malloc(&self) -> PinnedMallocCallback {
        self.malloc
    }

    /// Returns the free callback.
    #[must_use]
    pub const fn free(&self) -> PinnedFreeCallback {
        self.free
    }

    /// Returns the context pointer passed to the callbacks.
    #[must_use]
    pub const fn context(&self) -> *mut c_void {
        self.context
    }

    pub(crate) fn to_nvjpeg_raw(self) -> sys::nvjpegPinnedAllocatorV2_t {
        sys::nvjpegPinnedAllocatorV2_t {
            pinned_malloc: Some(pinned_malloc_to_sys(self.malloc)),
            pinned_free: Some(pinned_free_to_sys(self.free)),
            pinned_ctx: self.context,
        }
    }

    #[cfg(feature = "k2")]
    pub(crate) fn to_nvjpeg2k_raw(self) -> k2_sys::nvjpeg2kPinnedAllocatorV2_t {
        k2_sys::nvjpeg2kPinnedAllocatorV2_t {
            pinned_malloc: Some(pinned_malloc_to_k2_sys(self.malloc)),
            pinned_free: Some(pinned_free_to_k2_sys(self.free)),
            pinned_ctx: self.context,
        }
    }
}

impl AllocatorSet {
    /// Creates a set from a device allocator and a pinned host allocator.
    #[must_use]
    pub const fn new(device: DeviceAllocator, pinned: PinnedAllocator) -> Self {
        Self { device, pinned }
    }

    /// Returns the device allocator.
    #[must_use]
    pub const fn device(&self) -> DeviceAllocator {
        self.device
    }

    /// Returns the pinned host allocator.
    #[must_use]
    pub const fn pinned(&self) -> PinnedAllocator {
        self.pinned
    }
}

trait OptionFn {
    type Fn;
}

impl<F> OptionFn for Option<F> {
    type Fn = F;
}

type NvjpegDeviceMallocFn = <sys::tDevMallocV2 as OptionFn>::Fn;

type NvjpegDeviceFreeFn = <sys::tDevFreeV2 as OptionFn>::Fn;

type NvjpegPinnedMallocFn = <sys::tPinnedMallocV2 as OptionFn>::Fn;

type NvjpegPinnedFreeFn = <sys::tPinnedFreeV2 as OptionFn>::Fn;

const _: () = {
    assert!(size_of::<DeviceMallocCallback>() == size_of::<NvjpegDeviceMallocFn>());
    assert!(size_of::<DeviceFreeCallback>() == size_of::<NvjpegDeviceFreeFn>());
    assert!(size_of::<PinnedMallocCallback>() == size_of::<NvjpegPinnedMallocFn>());
    assert!(size_of::<PinnedFreeCallback>() == size_of::<NvjpegPinnedFreeFn>());
};

fn device_malloc_to_sys(callback: DeviceMallocCallback) -> NvjpegDeviceMallocFn {
    cast_callback(callback)
}

fn device_free_to_sys(callback: DeviceFreeCallback) -> NvjpegDeviceFreeFn {
    cast_callback(callback)
}

fn pinned_malloc_to_sys(callback: PinnedMallocCallback) -> NvjpegPinnedMallocFn {
    cast_callback(callback)
}

fn pinned_free_to_sys(callback: PinnedFreeCallback) -> NvjpegPinnedFreeFn {
    cast_callback(callback)
}

#[cfg(feature = "k2")]
type Nvjpeg2kDeviceMallocFn = <k2_sys::nvjpeg2kDeviceMallocV2 as OptionFn>::Fn;

#[cfg(feature = "k2")]
type Nvjpeg2kDeviceFreeFn = <k2_sys::nvjpeg2kDeviceFreeV2 as OptionFn>::Fn;

#[cfg(feature = "k2")]
type Nvjpeg2kPinnedMallocFn = <k2_sys::nvjpeg2kPinnedMallocV2 as OptionFn>::Fn;

#[cfg(feature = "k2")]
type Nvjpeg2kPinnedFreeFn = <k2_sys::nvjpeg2kPinnedFreeV2 as OptionFn>::Fn;

#[cfg(feature = "k2")]
const _: () = {
    assert!(size_of::<DeviceMallocCallback>() == size_of::<Nvjpeg2kDeviceMallocFn>());
    assert!(size_of::<DeviceFreeCallback>() == size_of::<Nvjpeg2kDeviceFreeFn>());
    assert!(size_of::<PinnedMallocCallback>() == size_of::<Nvjpeg2kPinnedMallocFn>());
    assert!(size_of::<PinnedFreeCallback>() == size_of::<Nvjpeg2kPinnedFreeFn>());
};

#[cfg(feature = "k2")]
fn device_malloc_to_k2_sys(callback: DeviceMallocCallback) -> Nvjpeg2kDeviceMallocFn {
    cast_callback(callback)
}

#[cfg(feature = "k2")]
fn device_free_to_k2_sys(callback: DeviceFreeCallback) -> Nvjpeg2kDeviceFreeFn {
    cast_callback(callback)
}

#[cfg(feature = "k2")]
fn pinned_malloc_to_k2_sys(callback: PinnedMallocCallback) -> Nvjpeg2kPinnedMallocFn {
    cast_callback(callback)
}

#[cfg(feature = "k2")]
fn pinned_free_to_k2_sys(callback: PinnedFreeCallback) -> Nvjpeg2kPinnedFreeFn {
    cast_callback(callback)
}

fn cast_callback<Wrapper, Sys>(callback: Wrapper) -> Sys
where
    Wrapper: Copy,
    Sys: Copy,
{
    debug_assert_eq!(size_of::<Wrapper>(), size_of::<Sys>());
    let callback = mem::ManuallyDrop::new(callback);
    // SAFETY: the wrapper callback aliases intentionally avoid exposing sys callback
    // aliases. The generated signatures use the same C ABI and argument
    // layout, checked above for each concrete conversion.
    unsafe { ptr_read_callback(&raw const *callback) }
}

unsafe fn ptr_read_callback<Wrapper, Sys>(callback: *const Wrapper) -> Sys {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { callback.cast::<Sys>().read() }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn device_malloc(
        _: *mut c_void,
        _: *mut *mut c_void,
        _: usize,
        _: *mut c_void,
    ) -> c_int {
        0
    }

    unsafe extern "C" fn device_free(
        _: *mut c_void,
        _: *mut c_void,
        _: usize,
        _: *mut c_void,
    ) -> c_int {
        0
    }

    unsafe extern "C" fn pinned_malloc(
        _: *mut c_void,
        _: *mut *mut c_void,
        _: usize,
        _: *mut c_void,
    ) -> c_int {
        0
    }

    unsafe extern "C" fn pinned_free(
        _: *mut c_void,
        _: *mut c_void,
        _: usize,
        _: *mut c_void,
    ) -> c_int {
        0
    }

    #[test]
    fn test_allocator_set_keeps_callbacks_and_contexts() {
        let mut device_context_value = ();
        let mut pinned_context_value = ();
        let device_context = (&raw mut device_context_value).cast::<c_void>();
        let pinned_context = (&raw mut pinned_context_value).cast::<c_void>();
        let device = DeviceAllocator::new(device_malloc, device_free, device_context);
        let pinned = PinnedAllocator::new(pinned_malloc, pinned_free, pinned_context);
        let allocators = AllocatorSet::new(device, pinned);

        assert_eq!(allocators.device().context(), device_context);
        assert_eq!(allocators.pinned().context(), pinned_context);
    }

    #[test]
    fn test_jpeg_raw_allocators_keep_contexts() {
        let mut device_context_value = ();
        let mut pinned_context_value = ();
        let device_context = (&raw mut device_context_value).cast::<c_void>();
        let pinned_context = (&raw mut pinned_context_value).cast::<c_void>();
        let device = DeviceAllocator::new(device_malloc, device_free, device_context);
        let pinned = PinnedAllocator::new(pinned_malloc, pinned_free, pinned_context);

        let raw_device = device.to_nvjpeg_raw();
        let raw_pinned = pinned.to_nvjpeg_raw();

        assert!(raw_device.dev_malloc.is_some());
        assert!(raw_device.dev_free.is_some());
        assert_eq!(raw_device.dev_ctx, device_context);
        assert!(raw_pinned.pinned_malloc.is_some());
        assert!(raw_pinned.pinned_free.is_some());
        assert_eq!(raw_pinned.pinned_ctx, pinned_context);
    }

    #[cfg(feature = "k2")]
    #[test]
    fn test_jpeg2k_raw_allocators_keep_contexts() {
        let mut device_context_value = ();
        let mut pinned_context_value = ();
        let device_context = (&raw mut device_context_value).cast::<c_void>();
        let pinned_context = (&raw mut pinned_context_value).cast::<c_void>();
        let device = DeviceAllocator::new(device_malloc, device_free, device_context);
        let pinned = PinnedAllocator::new(pinned_malloc, pinned_free, pinned_context);

        let raw_device = device.to_nvjpeg2k_raw();
        let raw_pinned = pinned.to_nvjpeg2k_raw();

        assert!(raw_device.device_malloc.is_some());
        assert!(raw_device.device_free.is_some());
        assert_eq!(raw_device.device_ctx, device_context);
        assert!(raw_pinned.pinned_malloc.is_some());
        assert!(raw_pinned.pinned_free.is_some());
        assert_eq!(raw_pinned.pinned_ctx, pinned_context);
    }
}
