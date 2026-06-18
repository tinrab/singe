use std::ptr;

use singe_cuda::memory::DeviceMemory;

use crate::{error::Result, mp::plan::Workspace, utility::to_usize};

#[derive(Debug)]
pub struct WorkspaceMemory {
    device: DeviceMemory<u8>,
    host: HostWorkspace,
}

#[derive(Debug)]
struct HostWorkspace {
    ptr: *mut (),
}

impl WorkspaceMemory {
    pub fn create(workspace: Workspace) -> Result<Self> {
        Ok(Self {
            device: DeviceMemory::create(to_usize(workspace.device_size(), "device workspace")?)?,
            host: HostWorkspace::create(to_usize(workspace.host_size(), "host workspace")?)?,
        })
    }

    pub(crate) fn device_mut_ptr(&mut self) -> *mut () {
        self.device.as_mut_ptr().cast()
    }

    pub(crate) fn host_mut_ptr(&self) -> *mut () {
        self.host.ptr
    }
}

impl HostWorkspace {
    fn create(size: usize) -> singe_cuda::error::Result<Self> {
        let ptr = if size == 0 {
            ptr::null_mut()
        } else {
            unsafe { DeviceMemory::<u8>::alloc_host(size)? }
        };
        Ok(Self { ptr })
    }
}

impl Drop for HostWorkspace {
    fn drop(&mut self) {
        if !self.ptr.is_null()
            && let Err(err) = unsafe { DeviceMemory::<u8>::free_host(self.ptr) }
        {
            #[cfg(debug_assertions)]
            eprintln!("failed to free cutensormp host workspace: {err}");
        }
    }
}
