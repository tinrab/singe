//! Device memory handler support for cuDSS.

use std::{
    ffi::{CString, c_int, c_void},
    marker::PhantomData,
    sync::Arc,
};

use singe_core::{copy_string_to_c_chars, string_from_c_chars};
use singe_cuda::{
    context::Context,
    types::DevicePtr,
    view::{DeviceOutputView, DeviceRegion, DeviceRepr, DeviceSlice, DeviceView},
};
use singe_cuda_sys::runtime::cudaStream_t;
use singe_cudss_sys as sys;

use crate::error::{Error, Result};

const MAX_ALLOCATOR_NAME_BYTES: usize = sys::CUDSS_ALLOCATOR_NAME_LEN as usize - 1;

/// Callback used by cuDSS to allocate device memory on a CUDA stream.
///
/// Return `0` on success.
/// The allocation must be stream-ordered and at least 256-byte aligned.
pub type DeviceAllocFn = unsafe extern "C" fn(
    context: *mut c_void,
    ptr: *mut *mut c_void,
    size: sys::size_t,
    stream: cudaStream_t,
) -> c_int;

/// Callback used by cuDSS to free device memory on a CUDA stream.
///
/// The callback receives the same context pointer and allocation size that were passed to [`DeviceAllocFn`].
pub type DeviceFreeFn = unsafe extern "C" fn(
    context: *mut c_void,
    ptr: *mut c_void,
    size: sys::size_t,
    stream: cudaStream_t,
) -> c_int;

/// A user-provided stream-ordered device memory handler for cuDSS.
///
/// cuDSS uses this handler during [`Context::execute`](crate::context::Context::execute)
/// to allocate internal device buffers. Those allocations are owned by
/// [`SolverState`](crate::data::SolverState) objects until the corresponding data object is
/// destroyed, so a handler must not be replaced or outlive its backing memory
/// pool while any `SolverState` object may still contain memory allocated by it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceMemoryHandler(sys::cudssDeviceMemHandler_t);

impl DeviceMemoryHandler {
    /// Creates a device memory handler from allocator callbacks.
    ///
    /// The name is copied into cuDSS' fixed-size allocator name field and must fit with a trailing NUL byte.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` contains an interior NUL byte or exceeds
    /// cuDSS' allocator-name capacity, including the trailing NUL byte.
    ///
    /// # Safety
    ///
    /// `context` must remain valid for every cuDSS call that can allocate or free
    /// through this handler. The callbacks must implement cuDSS' stream-ordered
    /// allocation contract, return `0` on success, and allocate memory that is
    /// valid for the CUDA context bound to the cuDSS handle. Memory returned by
    /// `device_alloc` must be at least 256-byte aligned, and `device_free` must
    /// accept the exact pointer and size pairs produced by `device_alloc`.
    pub unsafe fn create(
        name: &str,
        context: *mut c_void,
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
        let _ = CString::new(name)?;

        let mut handler = sys::cudssDeviceMemHandler_t {
            ctx: context,
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
    #[must_use]
    pub unsafe fn from_raw(raw: sys::cudssDeviceMemHandler_t) -> Self {
        Self(raw)
    }

    /// Returns the handler name copied into the cuDSS struct.
    #[must_use]
    pub fn name(&self) -> String {
        string_from_c_chars(&self.0.name)
    }

    /// Returns the opaque user context pointer passed to the callbacks.
    #[must_use]
    pub const fn context(&self) -> *mut c_void {
        self.0.ctx
    }

    /// Returns the allocation callback.
    #[must_use]
    pub const fn device_alloc(&self) -> Option<DeviceAllocFn> {
        self.0.device_alloc
    }

    /// Returns the free callback.
    #[must_use]
    pub const fn device_free(&self) -> Option<DeviceFreeFn> {
        self.0.device_free
    }

    pub(crate) const fn as_raw(&self) -> &sys::cudssDeviceMemHandler_t {
        &self.0
    }
}

/// A borrowed, initialized device pointer table with its actual CUDA context.
///
/// The table contains addresses, not initialized `T` values. Write-only solution
/// pointers may identify uninitialized storage. Raw execution determines which
/// pointees must be initialized and which may be written.
#[derive(Debug, Clone, Copy)]
pub struct DevicePointerTable<'a, T: DeviceRepr> {
    table: Option<DeviceView<'a, DevicePtr>>,
    context: &'a Arc<Context>,
    _marker: PhantomData<&'a T>,
}

impl<'a, T: DeviceRepr> DevicePointerTable<'a, T> {
    /// Collects readable buffer addresses for upload into a device pointer table.
    #[must_use]
    pub fn host_pointers(buffers: &[DeviceView<'_, T>]) -> Vec<DevicePtr> {
        buffers
            .iter()
            // SAFETY: encoding an address neither dereferences nor owns the pointee.
            .map(|buffer| unsafe { DevicePtr::from_raw(buffer.as_ptr().cast_mut().cast()) })
            .collect()
    }

    /// Collects write-only buffer addresses without claiming initialized values.
    pub fn host_output_pointers(buffers: &mut [DeviceOutputView<'_, T>]) -> Vec<DevicePtr> {
        buffers
            .iter_mut()
            // SAFETY: output views guarantee writable addresses with T's alignment.
            .map(|buffer| unsafe { DevicePtr::from_raw(buffer.as_mut_ptr().cast()) })
            .collect()
    }

    /// Borrows an externally established region containing initialized pointers.
    ///
    /// # Errors
    ///
    /// Returns a CUDA view error if the pointer-table length overflows its byte
    /// representation, exceeds the region, or the region does not satisfy the
    /// alignment and address requirements of a `DevicePtr` slice.
    ///
    /// # Safety
    ///
    /// The region must contain `length` initialized `DevicePtr` values. Every entry
    /// used by cuDSS must identify live, aligned storage for T in the region's
    /// actual context. Retain the table and all pointees through descriptor use
    /// and queued completion. Initialization and access permissions of pointees
    /// must satisfy the execution phase; constructing a table establishes neither.
    pub unsafe fn from_raw(region: DeviceRegion<'a>, length: usize) -> Result<Self> {
        let context = region.context();
        // SAFETY: the caller establishes initialized pointers and region liveness.
        let table = unsafe { DeviceView::from_raw_parts(region, length)? };
        Ok(Self {
            table: Some(table),
            context,
            _marker: PhantomData,
        })
    }

    /// Borrows a pointer allocation without inventing an allocation owner.
    ///
    /// # Errors
    ///
    /// Propagates the pointer-table view validation errors from [`Self::from_raw`].
    ///
    /// # Safety
    ///
    /// All referenced buffers must outlive the descriptor and queued work, and
    /// satisfy the context, alignment, initialization, and access contract of
    /// [`Self::from_raw`]. This borrow alone retains only the table allocation.
    pub unsafe fn from_device_memory<B>(memory: &'a B) -> Result<Self>
    where
        B: DeviceSlice<DevicePtr> + ?Sized,
    {
        // SAFETY: DeviceSlice supplies initialized pointer values; the caller
        // establishes the validity and liveness of the referenced allocations.
        unsafe { Self::from_raw(memory.region(), memory.len()) }
    }

    /// Creates an empty null table associated with an actual context.
    #[must_use]
    pub const fn null(context: &'a Arc<Context>) -> Self {
        Self {
            table: None,
            context,
            _marker: PhantomData,
        }
    }

    /// Number of pointer entries in the table.
    #[must_use]
    pub const fn len(&self) -> usize {
        match &self.table {
            Some(table) => table.len(),
            None => 0,
        }
    }

    /// Whether the table contains no pointer entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// CUDA context in which the pointer table resides.
    #[must_use]
    pub const fn context(&self) -> &'a Arc<Context> {
        self.context
    }

    /// Returns the borrowed table address without transferring ownership.
    #[must_use]
    pub fn as_ptr(self) -> DevicePtr {
        // SAFETY: the address belongs to the borrowed pointer table.
        self.table.map_or(DevicePtr::null(), |table| unsafe {
            DevicePtr::from_raw(table.as_ptr().cast_mut().cast())
        })
    }
}
