use std::{mem::ManuallyDrop, ops::Deref, ptr::read as ptr_read};

use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    gpu_instance::GpuInstance,
    try_ffi,
    types::{ComputeInstanceInfo, ComputeInstancePlacement},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ComputeInstance(sys::nvmlComputeInstance_t);

#[derive(Debug)]
pub struct OwnedComputeInstance(ComputeInstance);

impl ComputeInstance {
    /// Wraps a borrowed NVML compute instance handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `nvmlComputeInstance_t` for the duration of any
    /// calls made through the returned wrapper.
    pub const unsafe fn from_raw(handle: sys::nvmlComputeInstance_t) -> Self {
        Self(handle)
    }

    pub const fn as_raw(&self) -> sys::nvmlComputeInstance_t {
        self.0
    }

    pub const fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns compute instance information.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle or output arguments are rejected by NVML, if
    /// the current process does not have permission to query the instance, or if
    /// NVML has not been initialized.
    pub fn info(&self) -> Result<ComputeInstanceInfo> {
        let mut info = sys::nvmlComputeInstanceInfo_t::default();
        unsafe {
            try_ffi!(sys::nvmlComputeInstanceGetInfo_v2(self.0, &raw mut info))?;
        }
        Ok(info.into())
    }

    pub fn parent_device(&self) -> Result<Device> {
        Ok(self.info()?.device)
    }

    pub fn parent_gpu_instance(&self) -> Result<GpuInstance> {
        Ok(self.info()?.gpu_instance)
    }

    pub fn id(&self) -> Result<u32> {
        Ok(self.info()?.id)
    }

    pub fn profile_id(&self) -> Result<u32> {
        Ok(self.info()?.profile_id)
    }

    pub fn placement(&self) -> Result<ComputeInstancePlacement> {
        Ok(self.info()?.placement)
    }
}

impl OwnedComputeInstance {
    /// Wraps an owned NVML compute instance handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `nvmlComputeInstance_t` owned by the caller, and
    /// ownership must not be used elsewhere after constructing this wrapper.
    /// The wrapper destroys the compute instance with
    /// `nvmlComputeInstanceDestroy` unless ownership is transferred with
    /// [`OwnedComputeInstance::into_raw`].
    pub const unsafe fn from_raw(handle: sys::nvmlComputeInstance_t) -> Self {
        Self(ComputeInstance(handle))
    }

    pub const fn as_compute_instance(&self) -> &ComputeInstance {
        &self.0
    }

    pub fn into_inner(self) -> ComputeInstance {
        let this = ManuallyDrop::new(self);
        unsafe { ptr_read(&this.0) }
    }

    /// Transfers this owned compute instance handle to the caller.
    ///
    /// After this call, the wrapper will not destroy the compute instance. The
    /// caller becomes responsible for eventually passing the handle to
    /// `nvmlComputeInstanceDestroy` or otherwise transferring ownership.
    pub fn into_raw(self) -> sys::nvmlComputeInstance_t {
        self.into_inner().0
    }
}

impl Deref for OwnedComputeInstance {
    type Target = ComputeInstance;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for OwnedComputeInstance {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::nvmlComputeInstanceDestroy(self.0.0);
        }
    }
}
