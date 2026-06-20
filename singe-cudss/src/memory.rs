//! Device memory handler support for cuDSS.

use std::{
    ffi::{c_int, c_void},
    marker::PhantomData,
};

use singe_core::{copy_string_to_c_chars, string_from_c_chars};
use singe_cuda::{memory::DeviceMemory, types::DevicePtr};
use singe_cuda_sys::runtime::cudaStream_t;
use singe_cudss_sys as sys;

use crate::error::{Error, Result};

const MAX_ALLOCATOR_NAME_BYTES: usize = sys::CUDSS_ALLOCATOR_NAME_LEN as usize - 1;

/// Callback used by cuDSS to allocate device memory on a CUDA stream.
///
/// Return `0` on success.
/// The allocation must be stream-ordered and at least 256-byte aligned.
pub type DeviceAllocFn = unsafe extern "C" fn(
    ctx: *mut c_void,
    ptr: *mut *mut c_void,
    size: sys::size_t,
    stream: cudaStream_t,
) -> c_int;

/// Callback used by cuDSS to free device memory on a CUDA stream.
///
/// The callback receives the same context pointer and allocation size that were passed to [`DeviceAllocFn`].
pub type DeviceFreeFn = unsafe extern "C" fn(
    ctx: *mut c_void,
    ptr: *mut c_void,
    size: sys::size_t,
    stream: cudaStream_t,
) -> c_int;

/// A user-provided stream-ordered device memory handler for cuDSS.
///
/// cuDSS uses this handler during [`Context::execute`](crate::context::Context::execute)
/// to allocate internal device buffers. Those allocations are owned by
/// [`Data`](crate::data::Data) objects until the corresponding data object is
/// destroyed, so a handler must not be replaced or outlive its backing memory
/// pool while any `Data` object may still contain memory allocated by it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceMemoryHandler(sys::cudssDeviceMemHandler_t);

impl DeviceMemoryHandler {
    /// Creates a device memory handler from allocator callbacks.
    ///
    /// The name is copied into cuDSS' fixed-size allocator name field and must fit with a trailing NUL byte.
    ///
    /// # Safety
    ///
    /// `ctx` must remain valid for every cuDSS call that can allocate or free
    /// through this handler. The callbacks must implement cuDSS' stream-ordered
    /// allocation contract, return `0` on success, and allocate memory that is
    /// valid for the CUDA context bound to the cuDSS handle. Memory returned by
    /// `device_alloc` must be at least 256-byte aligned, and `device_free` must
    /// accept the exact pointer and size pairs produced by `device_alloc`.
    pub unsafe fn create(
        name: &str,
        ctx: *mut c_void,
        device_alloc: DeviceAllocFn,
        device_free: DeviceFreeFn,
    ) -> Result<Self> {
        if name.len() > MAX_ALLOCATOR_NAME_BYTES {
            return Err(Error::AllocatorNameTooLong {
                max: MAX_ALLOCATOR_NAME_BYTES,
                actual: name.len(),
            });
        }

        // Reuse the crate-wide string error mapping for interior NUL bytes.
        let _ = std::ffi::CString::new(name)?;

        let mut handler = sys::cudssDeviceMemHandler_t {
            ctx,
            device_alloc: Some(device_alloc),
            device_free: Some(device_free),
            name: [0; sys::CUDSS_ALLOCATOR_NAME_LEN as usize],
        };
        copy_string_to_c_chars(&mut handler.name, name);

        Ok(Self(handler))
    }

    /// Wraps a raw cuDSS device memory handler.
    ///
    /// # Safety
    ///
    /// The raw handler must satisfy the same callback and lifetime invariants as [`Self::create`].
    pub unsafe fn from_raw(raw: sys::cudssDeviceMemHandler_t) -> Self {
        Self(raw)
    }

    /// Returns the handler name copied into the cuDSS struct.
    pub fn name(&self) -> String {
        string_from_c_chars(&self.0.name)
    }

    /// Returns the opaque user context pointer passed to the callbacks.
    pub const fn context(&self) -> *mut c_void {
        self.0.ctx
    }

    /// Returns the allocation callback.
    pub const fn device_alloc(&self) -> Option<DeviceAllocFn> {
        self.0.device_alloc
    }

    /// Returns the free callback.
    pub const fn device_free(&self) -> Option<DeviceFreeFn> {
        self.0.device_free
    }

    pub(crate) const fn as_raw(&self) -> &sys::cudssDeviceMemHandler_t {
        &self.0
    }
}

/// Device-visible pointer table for a batch of buffers with element type `T`.
///
/// cuDSS batch descriptors store pointer tables, not contiguous values.
/// The lifetime tracks the table allocation and the buffers referenced by the table.
#[derive(Debug, Clone, Copy)]
pub struct DevicePointerTable<'a, T> {
    ptr: DevicePtr,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> DevicePointerTable<'a, T> {
    /// Builds a host pointer table from typed device buffers.
    ///
    /// The returned vector can be uploaded into `DeviceMemory<DevicePtr>` and then wrapped with [`Self::from_device_memory`].
    pub fn host_pointers(buffers: &[&DeviceMemory<T>]) -> Vec<DevicePtr> {
        buffers
            .iter()
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_ptr().cast_mut().cast()) })
            .collect()
    }

    /// Creates a typed pointer table from a raw device pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must be null or point to a device-visible table of device-visible
    /// pointers. Each table entry used by cuDSS must point to live storage for
    /// elements of type `T`, and the table and all referenced storage must
    /// outlive every matrix descriptor that stores this table.
    pub const unsafe fn from_raw(ptr: DevicePtr) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Creates a typed pointer table from device memory containing pointers.
    ///
    /// # Safety
    ///
    /// Each entry in `memory` that cuDSS reads must point to live
    /// device-visible storage for elements of type `T`, and those buffers must
    /// outlive every matrix descriptor that stores this table.
    pub unsafe fn from_device_memory(memory: &'a DeviceMemory<DevicePtr>) -> Self {
        unsafe { Self::from_raw(DevicePtr::from_raw(memory.as_ptr().cast_mut().cast())) }
    }

    /// Creates a null typed pointer table.
    pub const fn null() -> Self {
        Self {
            ptr: DevicePtr::null(),
            _marker: PhantomData,
        }
    }

    /// Returns the raw pointer to the device-visible pointer table.
    pub const fn as_ptr(self) -> DevicePtr {
        self.ptr
    }
}
