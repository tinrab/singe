use std::{
    fmt::{self, Display, Formatter},
    mem, ptr,
};

use bitflags::parser;
use singe_cuda_sys::{driver, runtime};

use crate::{
    error::{Error, Result},
    try_ffi,
};

bitflags::bitflags! {
    /// Flags for [`IpcMemoryHandle::create_mapping`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct IpcMemoryFlags: u32 {
        /// Automatically enable peer access between remote devices as needed.
        const LAZY_ENABLE_PEER_ACCESS = driver::CUipcMem_flags::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS as _;
    }
}

impl Display for IpcMemoryFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// An opaque CUDA IPC event export token that can be copied between processes.
///
/// This token does not own the exporting event or an imported event handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IpcEventHandle(runtime::cudaIpcEventHandle_t);

impl IpcEventHandle {
    /// Wraps an exported native IPC event token without importing the event.
    ///
    /// # Safety
    ///
    /// Any subsequent event import must use an unchanged token exported by CUDA for a live IPC-capable event. Coordinate the exporting event's lifetime with every process importing or using it.
    #[must_use]
    pub const unsafe fn from_raw(handle: runtime::cudaIpcEventHandle_t) -> Self {
        Self(handle)
    }

    /// Creates a zero-filled placeholder token.
    ///
    /// The result does not identify an exported event resource.
    #[must_use]
    pub const fn zeroed() -> Self {
        // SAFETY: this plain C struct is valid when all bytes are zero.
        unsafe { mem::zeroed() }
    }

    /// Returns a copy of the native IPC event token without importing or retaining its resource.
    #[must_use]
    pub const fn as_raw(&self) -> runtime::cudaIpcEventHandle_t {
        self.0
    }
}

/// An opaque CUDA IPC memory export token that can be copied between processes.
///
/// This token does not retain the exporting allocation or an imported mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IpcMemoryHandle(runtime::cudaIpcMemHandle_t);

impl IpcMemoryHandle {
    /// Wraps an exported native IPC memory token without opening a mapping.
    ///
    /// # Safety
    ///
    /// Any subsequent mapping must use an unchanged token exported by CUDA for a live IPC-capable allocation. Keep that allocation live until every imported mapping has been closed and prevent conflicting accesses across processes.
    #[must_use]
    pub const unsafe fn from_raw(handle: runtime::cudaIpcMemHandle_t) -> Self {
        Self(handle)
    }

    /// Creates a zero-filled placeholder token.
    ///
    /// The result does not identify an exported memory resource.
    #[must_use]
    pub const fn zeroed() -> Self {
        // SAFETY: this plain C struct is valid when all bytes are zero.
        unsafe { mem::zeroed() }
    }

    /// Returns a copy of the native IPC memory token without importing or retaining its resource.
    #[must_use]
    pub const fn as_raw(&self) -> runtime::cudaIpcMemHandle_t {
        self.0
    }

    /// Maps memory exported from another process with [`DeviceMemory::ipc_handle`](crate::memory::DeviceMemory::ipc_handle) into the current device address space.
    /// For contexts on different devices, [`IpcMemoryHandle::create_mapping`] can attempt to enable peer access between the devices as if [`Context::enable_peer_access`](crate::context::Context::enable_peer_access) had been called.
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
        // SAFETY: `cudaIpcOpenMemHandle` receives live handles owned by the wrappers and valid output locations.
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

/// An imported device mapping of another process's CUDA allocation.
///
/// Dropping this wrapper attempts to close the mapping. It does not own the allocation in the exporting process. Finish every use of this mapping before closing or dropping it; the exporting process must keep its allocation alive until all imports close.
#[derive(Debug)]
pub struct OpenedIpcMemory<T> {
    ptr: *mut T,
}

impl<T> OpenedIpcMemory<T> {
    /// Returns the imported device address.
    ///
    /// No allocation length or host accessibility is established by this pointer, and it remains valid only while this mapping and the exporting allocation remain live.
    #[must_use]
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
        // SAFETY: `cudaIpcCloseMemHandle` receives only plain values.
        unsafe { try_ffi!(runtime::cudaIpcCloseMemHandle(ptr.cast())) }
    }
}

impl<T> Drop for OpenedIpcMemory<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        // SAFETY: `cudaIpcCloseMemHandle` receives only plain values.
        unsafe {
            if let Err(error) = try_ffi!(runtime::cudaIpcCloseMemHandle(self.ptr.cast())) {
                #[cfg(debug_assertions)]
                eprintln!("failed to close cuda ipc memory handle: {error}");
            }
        }
    }
}
