//! Custom memory allocators for nvTIFF decoders and encoders.

use std::{
    ffi::{c_int, c_ulong, c_void},
    ptr,
};

use singe_cuda_sys::runtime::cudaStream_t;
use singe_nvtiff_sys as sys;

/// Callback that allocates device memory for nvTIFF.
pub type DeviceMallocCallback = unsafe extern "C" fn(
    context: *mut c_void,
    ptr: *mut *mut c_void,
    size: c_ulong,
    stream: cudaStream_t,
) -> c_int;

/// Callback that frees device memory allocated by a [`DeviceMallocCallback`].
pub type DeviceFreeCallback = unsafe extern "C" fn(
    context: *mut c_void,
    ptr: *mut c_void,
    size: c_ulong,
    stream: cudaStream_t,
) -> c_int;

/// Callback that allocates pinned host memory for nvTIFF.
pub type PinnedMallocCallback = unsafe extern "C" fn(
    context: *mut c_void,
    ptr: *mut *mut c_void,
    size: c_ulong,
    stream: cudaStream_t,
) -> c_int;

/// Callback that frees pinned host memory allocated by a [`PinnedMallocCallback`].
pub type PinnedFreeCallback = unsafe extern "C" fn(
    context: *mut c_void,
    ptr: *mut c_void,
    size: c_ulong,
    stream: cudaStream_t,
) -> c_int;

/// Device memory allocation callbacks and the context pointer passed to them.
///
/// The context pointer is passed to the callbacks unchanged.
/// This type does not own what it points to; keep it valid for as long as any decoder or encoder created with the allocator exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceAllocator {
    raw: sys::nvtiffDeviceAllocator_t,
}

/// Pinned host memory allocation callbacks and the context pointer passed to them.
///
/// The context pointer is passed to the callbacks unchanged.
/// This type does not own what it points to; keep it valid for as long as any decoder or encoder created with the allocator exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PinnedAllocator {
    raw: sys::nvtiffPinnedAllocator_t,
}

/// Device and pinned host allocators for a decoder or encoder; unset ones use nvTIFF's defaults.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct AllocatorSet {
    device: Option<DeviceAllocator>,
    pinned: Option<PinnedAllocator>,
}

impl DeviceAllocator {
    /// Creates a device allocator from its callbacks and context pointer.
    #[must_use]
    pub const fn new(
        malloc: DeviceMallocCallback,
        free: DeviceFreeCallback,
        context: *mut c_void,
    ) -> Self {
        Self {
            raw: sys::nvtiffDeviceAllocator_t {
                device_malloc: Some(malloc),
                device_free: Some(free),
                device_ctx: context,
            },
        }
    }
}

impl PinnedAllocator {
    /// Creates a pinned host allocator from its callbacks and context pointer.
    #[must_use]
    pub const fn new(
        malloc: PinnedMallocCallback,
        free: PinnedFreeCallback,
        context: *mut c_void,
    ) -> Self {
        Self {
            raw: sys::nvtiffPinnedAllocator_t {
                pinned_malloc: Some(malloc),
                pinned_free: Some(free),
                pinned_ctx: context,
            },
        }
    }
}

impl AllocatorSet {
    /// Creates a set that uses nvTIFF's default allocators.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            device: None,
            pinned: None,
        }
    }

    /// Returns the device allocator, or `None` for nvTIFF's default.
    #[must_use]
    pub const fn device(&self) -> Option<DeviceAllocator> {
        self.device
    }

    /// Uses `allocator` for device memory.
    #[must_use]
    pub const fn with_device(mut self, allocator: DeviceAllocator) -> Self {
        self.device = Some(allocator);
        self
    }

    /// Returns the pinned host allocator, or `None` for nvTIFF's default.
    #[must_use]
    pub const fn pinned(&self) -> Option<PinnedAllocator> {
        self.pinned
    }

    /// Uses `allocator` for pinned host memory.
    #[must_use]
    pub const fn with_pinned(mut self, allocator: PinnedAllocator) -> Self {
        self.pinned = Some(allocator);
        self
    }

    pub(crate) fn device_raw(&mut self) -> *mut sys::nvtiffDeviceAllocator_t {
        self.device
            .as_mut()
            .map_or(ptr::null_mut(), |allocator| &raw mut allocator.raw)
    }

    pub(crate) fn pinned_raw(&mut self) -> *mut sys::nvtiffPinnedAllocator_t {
        self.pinned
            .as_mut()
            .map_or(ptr::null_mut(), |allocator| &raw mut allocator.raw)
    }
}
