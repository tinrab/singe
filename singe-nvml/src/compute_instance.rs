use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::Deref,
    ptr::read as ptr_read,
};

use singe_core::report_cleanup_failure;
use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::{Error, Result},
    gpu_instance::GpuInstance,
    library::Library,
    try_ffi,
    types::{ComputeInstanceInfo, ComputeInstancePlacement},
};

/// NVML compute instance handle that does not own the instance.
///
/// It keeps NVML initialized.
/// Queries such as [`GpuInstance::compute_instances`] return handles to instances the caller may not own, so the instance can be destroyed while a handle exists, by an [`OwnedComputeInstance`], another process, or a device reset.
/// NVML treats such a handle as invalid and rejects calls through it.
/// Equality and hashing compare the native handle.
#[derive(Debug)]
pub struct ComputeInstance {
    raw: sys::nvmlComputeInstance_t,
    library: Library,
}

// SAFETY: NVML documents its API as thread-safe, the handle is an opaque identifier the wrapper never dereferences, and the retained `Library` is `Send`.
unsafe impl Send for ComputeInstance {}

// SAFETY: NVML documents its API as thread-safe, and every `&self` method only passes the opaque handle to NVML.
unsafe impl Sync for ComputeInstance {}

impl PartialEq for ComputeInstance {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl Eq for ComputeInstance {}

impl Hash for ComputeInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

/// Owns an NVML compute instance and requests its destruction on drop.
///
/// It lends out its instance's methods as a borrowed [`ComputeInstance`].
/// It borrows the [`GpuInstance`] it was created in, so that GPU instance cannot be destroyed first.
/// [`Self::into_inner`] and [`Self::into_raw`] transfer destruction responsibility to the caller.
#[derive(Debug)]
pub struct OwnedComputeInstance<'a> {
    instance: ComputeInstance,
    parent: PhantomData<&'a GpuInstance>,
}

/// A transferred NVML compute instance handle with the initialization it was created under.
#[derive(Debug)]
pub struct RawComputeInstance {
    /// The owned native compute instance handle.
    pub handle: sys::nvmlComputeInstance_t,
    /// The NVML initialization kept alive until `handle` is destroyed.
    pub library: Library,
}

// SAFETY: the record owns its handle exclusively, NVML documents its API as thread-safe, and the retained `Library` is `Send`.
unsafe impl Send for RawComputeInstance {}

impl ComputeInstance {
    /// Wraps an NVML compute instance handle and keeps `library`'s initialization alive.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `nvmlComputeInstance_t` obtained while `library`'s initialization was alive, and it must stay valid for the duration of any calls made through the returned wrapper.
    pub unsafe fn from_raw(handle: sys::nvmlComputeInstance_t, library: &Library) -> Self {
        Self {
            raw: handle,
            library: library.clone(),
        }
    }

    /// Returns the borrowed native instance handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvmlComputeInstance_t {
        self.raw
    }

    /// Returns whether the stored native handle is null.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.raw.is_null()
    }

    /// Returns the NVML initialization that this instance keeps alive.
    #[must_use]
    pub const fn library(&self) -> &Library {
        &self.library
    }

    /// Returns compute instance information.
    ///
    /// Requires an Ampere or newer GPU on Linux.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle or output arguments are rejected by NVML, if the current process does not have permission to query the instance, or if NVML has not been initialized.
    pub fn info(&self) -> Result<ComputeInstanceInfo> {
        let mut info = sys::nvmlComputeInstanceInfo_t::default();
        // SAFETY: `self.raw` is a compute instance handle valid per `from_raw`'s contract, and `info` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlComputeInstanceGetInfo_v2(self.raw, &raw mut info))?;
        }
        // SAFETY: NVML filled `info` above, and its handles stay valid while `self.library` keeps NVML initialized.
        Ok(unsafe { ComputeInstanceInfo::from_raw(info, &self.library) })
    }

    /// Returns the physical GPU containing this compute instance.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::info`] fails.
    pub fn parent_device(&self) -> Result<Device> {
        Ok(self.info()?.device)
    }

    /// Returns the GPU instance containing this compute instance.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::info`] fails.
    pub fn parent_gpu_instance(&self) -> Result<GpuInstance> {
        Ok(self.info()?.gpu_instance)
    }

    /// Returns the compute instance identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::info`] fails.
    pub fn id(&self) -> Result<u32> {
        Ok(self.info()?.id)
    }

    /// Returns the compute instance profile identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::info`] fails.
    pub fn profile_id(&self) -> Result<u32> {
        Ok(self.info()?.profile_id)
    }

    /// Returns the compute-slice placement of this instance.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::info`] fails.
    pub fn placement(&self) -> Result<ComputeInstancePlacement> {
        Ok(self.info()?.placement)
    }
}

impl OwnedComputeInstance<'_> {
    /// Wraps an owned NVML compute instance handle and keeps `raw.library`'s initialization alive.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid `nvmlComputeInstance_t` created while `raw.library`'s initialization was alive and owned by the caller, and ownership must not be used elsewhere after constructing this wrapper.
    /// Its parent GPU instance must not be destroyed while the returned owner lives.
    /// The wrapper destroys the compute instance with `nvmlComputeInstanceDestroy` unless ownership is transferred with [`OwnedComputeInstance::into_raw`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawComputeInstance) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            instance: ComputeInstance {
                raw: raw.handle,
                library: raw.library,
            },
            parent: PhantomData,
        })
    }

    /// Borrows the instance without transferring destruction responsibility.
    #[must_use]
    pub const fn as_compute_instance(&self) -> &ComputeInstance {
        &self.instance
    }

    /// Transfers destruction responsibility and returns the compute instance wrapper.
    ///
    /// The caller must eventually destroy the native instance or place it in another owner.
    #[must_use]
    pub fn into_inner(self) -> ComputeInstance {
        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so `instance` moves out exactly once, and `parent` holds nothing to drop.
        unsafe { ptr_read(&raw const this.instance) }
    }

    /// Transfers this owned compute instance handle to the caller.
    ///
    /// After this call, the wrapper will not destroy the compute instance.
    /// The caller becomes responsible for eventually passing the handle to `nvmlComputeInstanceDestroy` or otherwise transferring ownership.
    /// The returned [`Library`] keeps NVML initialized; drop it only after the handle is destroyed.
    #[must_use]
    pub fn into_raw(self) -> RawComputeInstance {
        let ComputeInstance { raw, library } = self.into_inner();
        RawComputeInstance {
            handle: raw,
            library,
        }
    }
}

impl Deref for OwnedComputeInstance<'_> {
    type Target = ComputeInstance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl Drop for OwnedComputeInstance<'_> {
    fn drop(&mut self) {
        // SAFETY: `self` owns the non-null instance and destroys it once, while its retained library keeps NVML initialized.
        unsafe {
            if let Err(error) = try_ffi!(sys::nvmlComputeInstanceDestroy(self.instance.raw)) {
                report_cleanup_failure("destroy", "nvml compute instance", &error);
            }
        }
    }
}
