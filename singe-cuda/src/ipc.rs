use std::{mem, ptr};

use singe_cuda_sys::{driver, runtime};

use crate::{
    error::{Error, Result},
    try_ffi,
};

bitflags::bitflags! {
    /// Flags for [`IpcMemoryHandle::create_mapping`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct IpcMemoryFlags: u32 {
        const LAZY_ENABLE_PEER_ACCESS = driver::CUipcMem_flags::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS as _;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IpcEventHandle(runtime::cudaIpcEventHandle_t);

impl IpcEventHandle {
    pub const unsafe fn from_raw(handle: runtime::cudaIpcEventHandle_t) -> Self {
        Self(handle)
    }

    pub const fn zeroed() -> Self {
        unsafe { mem::zeroed() }
    }

    pub const fn as_raw(&self) -> runtime::cudaIpcEventHandle_t {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IpcMemoryHandle(runtime::cudaIpcMemHandle_t);

impl IpcMemoryHandle {
    pub const unsafe fn from_raw(handle: runtime::cudaIpcMemHandle_t) -> Self {
        Self(handle)
    }

    pub const fn zeroed() -> Self {
        unsafe { mem::zeroed() }
    }

    pub const fn as_raw(&self) -> runtime::cudaIpcMemHandle_t {
        self.0
    }

    /// Maps memory exported from another process with [`DeviceMemory::ipc_handle`](crate::memory::DeviceMemory::ipc_handle) into the current device address space.
    /// For contexts on different devices, [`IpcMemoryHandle::create_mapping`] can attempt to enable peer access between the devices as if [`Device::enable_peer_access`](crate::device::Device::enable_peer_access) had been called.
    /// This behavior is controlled by [`IpcMemoryFlags::LAZY_ENABLE_PEER_ACCESS`].
    /// [`Device::can_access_peer`](crate::device::Device::can_access_peer) can determine if a mapping is possible.
    ///
    /// [`IpcMemoryHandle::create_mapping`] can open handles to devices that may not be visible in the current process.
    ///
    /// Imported memory handles from each device in a given process may only be opened by one context per device per other process.
    ///
    /// If the memory handle has already been opened by the current context, the reference count on the handle is incremented by 1 and the existing device pointer is returned.
    ///
    /// Memory returned from [`IpcMemoryHandle::create_mapping`] must be freed with [`OpenedIpcMemory::close`].
    ///
    /// Calling [`DeviceMemory::free`](crate::memory::DeviceMemory::free) on an
    /// exported memory region before calling [`OpenedIpcMemory::close`] in the
    /// importing context results in undefined behavior.
    ///
    /// IPC is restricted to devices with support for unified addressing on Linux and Windows operating systems.
    /// IPC on Windows is supported for compatibility, but is not recommended
    /// because it has a performance cost.
    /// Check device IPC support through the device properties exposed by this crate, for example [`DeviceProperties::ipc_event_supported`](crate::device::DeviceProperties::ipc_event_supported).
    ///
    /// Additional CUDA diagnostics:
    ///
    /// * This call may also return [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`] if it initializes internal CUDA runtime state.
    /// * Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
    ///   [`crate::error::Status::NotPermitted`] may, but is not guaranteed to, be returned as a diagnostic in that case.
    /// * No guarantees are made about the address returned.
    ///   In particular, multiple processes may not receive the same address for the same handle.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA Runtime cannot open the IPC handle, if the
    /// current device cannot access the allocation, or if CUDA returns a null
    /// mapped pointer.
    pub fn create_mapping<T>(self, flags: IpcMemoryFlags) -> Result<OpenedIpcMemory<T>> {
        let mut dev_ptr = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaIpcOpenMemHandle(
                &raw mut dev_ptr,
                self.as_raw(),
                flags.bits()
            ))?;
        }
        if dev_ptr.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(OpenedIpcMemory {
            ptr: dev_ptr.cast(),
        })
    }
}

#[derive(Debug)]
pub struct OpenedIpcMemory<T> {
    ptr: *mut T,
}

impl<T> OpenedIpcMemory<T> {
    pub const fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Decrements the reference count of the memory returned by [`IpcMemoryHandle::create_mapping`] by 1.
    /// When the reference count reaches 0, this unmaps the memory.
    /// The original allocation in the exporting process and imported mappings
    /// in other processes are unaffected.
    ///
    /// Resources used to enable peer access are freed if this is the last
    /// mapping using them.
    ///
    /// IPC is restricted to devices with support for unified addressing on Linux and Windows operating systems.
    /// IPC on Windows is supported for compatibility, but is not recommended
    /// because it has a performance cost.
    /// Check device IPC support through the device properties exposed by this crate, for example [`DeviceProperties::ipc_event_supported`](crate::device::DeviceProperties::ipc_event_supported).
    ///
    /// Additional CUDA diagnostics:
    ///
    /// * This call may also return [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`] if it initializes internal CUDA runtime state.
    /// * Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
    ///   [`crate::error::Status::NotPermitted`] may, but is not guaranteed to, be returned as a diagnostic in that case.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA Runtime cannot close the imported mapping.
    pub fn close(self) -> Result<()> {
        let ptr: *mut () = self.ptr.cast();
        mem::forget(self);
        if ptr.is_null() {
            return Ok(());
        }
        unsafe { try_ffi!(runtime::cudaIpcCloseMemHandle(ptr as _)) }
    }
}

impl<T> Drop for OpenedIpcMemory<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        unsafe {
            if let Err(err) = try_ffi!(runtime::cudaIpcCloseMemHandle(self.ptr.cast())) {
                #[cfg(debug_assertions)]
                eprintln!("failed to close cuda ipc memory handle: {err}");
            }
        }
    }
}
