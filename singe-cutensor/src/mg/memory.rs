use std::ptr;

use singe_cuda::{device::Device, memory::DeviceMemory};

use crate::{
    error::{Error, Result},
    mg::{context::ContextRef, tensor::TensorDevice, workspace::Workspace},
};

#[derive(Debug)]
pub struct DistributedDeviceMemory<T> {
    buffers: Vec<DeviceMemory<T>>,
    devices: Vec<TensorDevice>,
    len_per_pointer: usize,
}

#[derive(Debug)]
pub struct WorkspaceMemory {
    device: Vec<DeviceMemory<u8>>,
    host: HostWorkspace,
}

#[derive(Debug)]
struct HostWorkspace {
    ptr: *mut (),
}

impl<T> DistributedDeviceMemory<T> {
    pub fn create(devices: &[TensorDevice], len_per_pointer: usize) -> Result<Self> {
        let mut buffers = Vec::with_capacity(devices.len());
        for device in devices {
            let TensorDevice::Device(device_id) = *device else {
                return Err(Error::MgHostTensorMemoryUnsupported {
                    name: "distributed device memory".into(),
                });
            };
            Device::new(device_id).set_current()?;
            buffers.push(DeviceMemory::<T>::create(len_per_pointer)?);
        }

        Ok(Self {
            buffers,
            devices: devices.to_vec(),
            len_per_pointer,
        })
    }

    pub fn pointer_count(&self) -> usize {
        self.buffers.len()
    }

    pub fn len_per_pointer(&self) -> usize {
        self.len_per_pointer
    }

    pub fn devices(&self) -> &[TensorDevice] {
        &self.devices
    }

    pub(crate) fn const_ptrs(&self) -> Vec<*const ()> {
        self.buffers
            .iter()
            .map(|buffer| buffer.as_ptr().cast())
            .collect()
    }

    pub(crate) fn mut_ptrs(&mut self) -> Vec<*mut ()> {
        self.buffers
            .iter()
            .map(|buffer| buffer.as_mut_ptr().cast())
            .collect()
    }
}

impl WorkspaceMemory {
    pub fn create(context: &ContextRef, workspace: &Workspace) -> Result<Self> {
        workspace.validate_for_context(context)?;

        let device_sizes = workspace.device_sizes_bytes()?;
        let mut device = Vec::with_capacity(context.device_count());
        for (&device_id, &size) in context.devices().iter().zip(&device_sizes) {
            Device::new(device_id).set_current()?;
            device.push(DeviceMemory::<u8>::create(size)?);
        }

        Ok(Self {
            device,
            host: HostWorkspace::create(workspace.host_size_bytes()?)?,
        })
    }

    pub(crate) fn device_mut_ptrs(&mut self) -> Vec<*mut ()> {
        self.device
            .iter()
            .map(|buffer| buffer.as_mut_ptr().cast())
            .collect()
    }

    pub(crate) fn device_pointer_count(&self) -> usize {
        self.device.len()
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
            eprintln!("failed to free cutensormg host workspace: {err}");
        }
    }
}
