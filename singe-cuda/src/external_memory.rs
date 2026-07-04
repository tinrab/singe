//! Safe ownership wrappers for CUDA external memory imports.

use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    mem::{self, size_of},
    ptr,
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_conversion;
use singe_cuda_sys::driver;

use crate::{
    error::{Error, Result},
    module::{KernelParameters, PushKernelArg},
    try_ffi,
    view::{DeviceRepr, DeviceSlice, DeviceSliceMut},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ExternalMemoryHandleType {
    /// Handle is an opaque file descriptor.
    OpaqueFileDescriptor =
        driver::CUexternalMemoryHandleType_enum::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD as _,
    /// Handle is an opaque shared NT handle.
    OpaqueWin32 =
        driver::CUexternalMemoryHandleType_enum::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 as _,
    /// Handle is an opaque, globally shared handle.
    OpaqueWin32Kmt =
        driver::CUexternalMemoryHandleType_enum::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
            as _,
    /// Handle is a dma_buf file descriptor.
    DmaBufferFileDescriptor =
        driver::CUexternalMemoryHandleType_enum::CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD as _,
}

impl_enum_conversion!(driver::CUexternalMemoryHandleType, ExternalMemoryHandleType);

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ExternalMemoryFlags: u32 {
        const DEDICATED = driver::CUDA_EXTERNAL_MEMORY_DEDICATED;
    }
}

impl Display for ExternalMemoryFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

#[derive(Debug)]
pub struct ExternalMemory {
    handle: driver::CUexternalMemory,
    size: usize,
}

#[derive(Debug)]
pub struct MappedBuffer<'a, T: DeviceRepr> {
    ptr: *mut T,
    length: usize,
    // The mapped buffer must not outlive the imported external-memory object it
    // was derived from. CUDA still requires the mapped pointer to be freed
    // separately with cuMemFree.
    _memory: PhantomData<&'a ExternalMemory>,
}

impl ExternalMemory {
    /// Imports an opaque file descriptor as CUDA external memory.
    ///
    /// # Safety
    ///
    /// `fd` must reference a valid external memory object of `size` bytes.
    /// CUDA takes ownership of the descriptor on a successful import.
    #[cfg(unix)]
    pub unsafe fn import_opaque_file_descriptor(
        fd: std::os::fd::RawFd,
        size: usize,
        flags: ExternalMemoryFlags,
    ) -> Result<Self> {
        let mut desc = handle_desc(ExternalMemoryHandleType::OpaqueFileDescriptor, size, flags)?;
        desc.handle.fd = fd;
        unsafe { Self::import(&desc, size) }
    }

    /// Imports an opaque Win32 shared handle as CUDA external memory.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid shared handle for a memory object of `size`
    /// bytes and must remain valid according to CUDA's external-memory import
    /// rules.
    pub unsafe fn import_opaque_win32_handle(
        handle: *mut c_void,
        size: usize,
        flags: ExternalMemoryFlags,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let mut desc = handle_desc(ExternalMemoryHandleType::OpaqueWin32, size, flags)?;
        desc.handle.win32.handle = handle;
        desc.handle.win32.name = ptr::null();
        unsafe { Self::import(&desc, size) }
    }

    /// Imports an opaque named Win32 shared object as CUDA external memory.
    ///
    /// # Safety
    ///
    /// `name` must point to a valid null-terminated Win32 object name for a
    /// memory object of `size` bytes and remain valid for the import call.
    pub unsafe fn import_opaque_win32_name(
        name: *const c_void,
        size: usize,
        flags: ExternalMemoryFlags,
    ) -> Result<Self> {
        if name.is_null() {
            return Err(Error::NullHandle);
        }

        let mut desc = handle_desc(ExternalMemoryHandleType::OpaqueWin32, size, flags)?;
        desc.handle.win32.handle = ptr::null_mut();
        desc.handle.win32.name = name;
        unsafe { Self::import(&desc, size) }
    }

    unsafe fn import(desc: &driver::CUDA_EXTERNAL_MEMORY_HANDLE_DESC, size: usize) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(driver::cuImportExternalMemory(
                &raw mut handle,
                desc as *const _,
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle, size })
    }

    pub fn map_buffer<T: DeviceRepr>(
        &self,
        offset_bytes: usize,
        length: usize,
    ) -> Result<MappedBuffer<'_, T>> {
        let bytes = checked_bytes::<T>(length)?;
        if bytes == 0 {
            return Err(Error::InvalidMemoryAllocationRequest);
        }
        let end = offset_bytes
            .checked_add(bytes)
            .ok_or(Error::InvalidMemoryAllocationRequest)?;
        if end > self.size {
            return Err(Error::InvalidMemoryAccess);
        }

        // CUDA returns a device pointer for this mapped range. That pointer is
        // a distinct CUDA allocation handle and must be freed by MappedBuffer.
        let desc = driver::CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            offset: offset_bytes as _,
            size: bytes as _,
            flags: 0,
            reserved: [0; 16],
        };
        let mut ptr = 0;
        unsafe {
            try_ffi!(driver::cuExternalMemoryGetMappedBuffer(
                &raw mut ptr,
                self.handle,
                &raw const desc,
            ))?;
        }
        if ptr == 0 {
            return Err(Error::NullHandle);
        }

        Ok(MappedBuffer {
            ptr: ptr as *mut T,
            length,
            _memory: PhantomData,
        })
    }

    pub const fn byte_len(&self) -> usize {
        self.size
    }

    pub const fn as_raw(&self) -> driver::CUexternalMemory {
        self.handle
    }

    /// Takes ownership of a raw CUDA external-memory handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `CUexternalMemory` that is not owned by any
    /// other wrapper, and `size` must match the imported memory object's byte
    /// size so mapped ranges can be bounds-checked correctly.
    pub unsafe fn from_raw(handle: driver::CUexternalMemory, size: usize) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle, size })
    }

    /// Transfers ownership of the raw CUDA external-memory handle to the
    /// caller without destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cuDestroyExternalMemory`.
    pub fn into_raw(self) -> driver::CUexternalMemory {
        let handle = self.handle;
        mem::forget(self);
        handle
    }
}

impl Drop for ExternalMemory {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        unsafe {
            if let Err(error) = try_ffi!(driver::cuDestroyExternalMemory(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cuda external memory: {error}");
            }
        }
        self.handle = ptr::null_mut();
    }
}

// External memory handles identify imported allocations. Mapping operations
// require &self, and lifetime-sensitive mapped buffers borrow the owner.
unsafe impl Send for ExternalMemory {}
unsafe impl Sync for ExternalMemory {}

impl<T: DeviceRepr> MappedBuffer<'_, T> {
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub const fn len(&self) -> usize {
        self.length
    }

    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }
}

impl<T: DeviceRepr> Drop for MappedBuffer<'_, T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        unsafe {
            if let Err(error) = try_ffi!(driver::cuMemFree_v2(self.ptr as driver::CUdeviceptr)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to free mapped external memory buffer: {error}");
            }
        }
        self.ptr = ptr::null_mut();
        self.length = 0;
    }
}

unsafe impl<T: DeviceRepr + Send> Send for MappedBuffer<'_, T> {}
unsafe impl<T: DeviceRepr + Sync> Sync for MappedBuffer<'_, T> {}

impl<T: DeviceRepr> DeviceSlice<T> for MappedBuffer<'_, T> {
    fn as_device_ptr(&self) -> *const T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.length
    }
}

impl<T: DeviceRepr> DeviceSliceMut<T> for MappedBuffer<'_, T> {
    fn as_device_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T: DeviceRepr> PushKernelArg for &MappedBuffer<'_, T> {
    fn push_to<'a>(self, params: &mut KernelParameters<'a>) {
        params.device_slice(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for &mut MappedBuffer<'_, T> {
    fn push_to<'a>(self, params: &mut KernelParameters<'a>) {
        params.device_slice_mut(self);
    }
}

fn handle_desc(
    handle_type: ExternalMemoryHandleType,
    size: usize,
    flags: ExternalMemoryFlags,
) -> Result<driver::CUDA_EXTERNAL_MEMORY_HANDLE_DESC> {
    if size == 0 {
        return Err(Error::InvalidMemoryAllocationRequest);
    }

    Ok(driver::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
        type_: handle_type.into(),
        handle: Default::default(),
        size: size as _,
        flags: flags.bits(),
        reserved: [0; 16],
    })
}

fn checked_bytes<T>(length: usize) -> Result<usize> {
    length
        .checked_mul(size_of::<T>())
        .ok_or(Error::InvalidMemoryAllocationRequest)
}
