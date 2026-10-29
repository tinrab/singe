use std::{mem::ManuallyDrop, ops::Deref, ptr::read as ptr_read};

use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    gpu_instance::GpuInstance,
    try_ffi,
    types::{ComputeInstanceInfo, ComputeInstancePlacement},
};

/// Borrowed NVML compute instance handle.
///
/// Clones refer to the same native instance.
/// Destruction responsibility remains with the owner, such as [`OwnedComputeInstance`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ComputeInstance(sys::nvmlComputeInstance_t);

// SAFETY: NVML is thread-safe, and a compute instance handle is an opaque identifier that the wrapper never
// dereferences.
unsafe impl Send for ComputeInstance {}

unsafe impl Sync for ComputeInstance {}

/// Owns an NVML compute instance and requests its destruction on drop.
///
/// [`Self::into_inner`] and [`Self::into_raw`] transfer destruction responsibility to the caller.
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

    /// Returns the borrowed native instance handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvmlComputeInstance_t {
        self.0
    }

    /// Returns whether the stored native handle is null.
    #[must_use]
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
        // SAFETY: `nvmlComputeInstanceGetInfo_v2` receives valid output locations.
        unsafe {
            try_ffi!(sys::nvmlComputeInstanceGetInfo_v2(self.0, &raw mut info))?;
        }
        Ok(info.into())
    }

    /// Returns the physical GPU containing this compute instance.
    ///
    /// # Errors
    ///
    /// Propagates instance-info query errors from [`Self::info`], including an invalid or unavailable instance handle.
    pub fn parent_device(&self) -> Result<Device> {
        Ok(self.info()?.device)
    }

    /// Returns the GPU instance containing this compute instance.
    ///
    /// # Errors
    ///
    /// Propagates instance-info query errors from [`Self::info`], including an invalid or unavailable instance handle.
    pub fn parent_gpu_instance(&self) -> Result<GpuInstance> {
        Ok(self.info()?.gpu_instance)
    }

    /// Returns the compute instance identifier.
    ///
    /// # Errors
    ///
    /// Propagates instance-info query errors from [`Self::info`], including an invalid or unavailable instance handle.
    pub fn id(&self) -> Result<u32> {
        Ok(self.info()?.id)
    }

    /// Returns the compute instance profile identifier.
    ///
    /// # Errors
    ///
    /// Propagates instance-info query errors from [`Self::info`], including an invalid or unavailable instance handle.
    pub fn profile_id(&self) -> Result<u32> {
        Ok(self.info()?.profile_id)
    }

    /// Returns the compute-slice placement of this instance.
    ///
    /// # Errors
    ///
    /// Propagates instance-info query errors from [`Self::info`], including an invalid or unavailable instance handle.
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

    /// Borrows the instance handle without transferring destruction responsibility.
    #[must_use]
    pub const fn as_compute_instance(&self) -> &ComputeInstance {
        &self.0
    }

    /// Transfers destruction responsibility and returns the borrowed compute instance wrapper.
    ///
    /// The caller must eventually destroy the native instance or place it in another owner.
    #[must_use]
    pub fn into_inner(self) -> ComputeInstance {
        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so the field moves out exactly once.
        unsafe { ptr_read(&raw const this.0) }
    }

    /// Transfers this owned compute instance handle to the caller.
    ///
    /// After this call, the wrapper will not destroy the compute instance. The
    /// caller becomes responsible for eventually passing the handle to
    /// `nvmlComputeInstanceDestroy` or otherwise transferring ownership.
    #[must_use]
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
        // SAFETY: `nvmlComputeInstanceDestroy` receives only plain values.
        unsafe {
            if let Err(error) = try_ffi!(sys::nvmlComputeInstanceDestroy(self.0.0)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy nvml compute instance: {error}");
            }
        }
    }
}
