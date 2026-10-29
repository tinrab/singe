use std::{
    mem::{self, ManuallyDrop},
    ptr,
    sync::Arc,
};

use singe_cuda::{context::Context as CudaContext, device::Device, error::Result as CudaResult};

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    utility::to_u32,
};

/// Owns a shared multi-GPU cuTENSOR handle and the primary CUDA contexts of its participating devices.
#[derive(Debug, Clone)]
pub struct Context {
    handle: Arc<Handle>,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cutensorMgHandle_t,
    devices: Vec<i32>,
    cuda_contexts: Vec<Arc<CudaContext>>,
}

/// Retains a multi-GPU library handle and its ordered participating CUDA contexts.
#[derive(Debug, Clone)]
pub struct ContextRef {
    handle: Arc<Handle>,
}

// SAFETY: cuTENSORMg contexts are immutable after creation for a fixed device set. The
// Arc-backed handle can be shared, while wrappers remain freely movable.
unsafe impl Send for Handle {}

unsafe impl Sync for Handle {}

unsafe impl Send for Context {}

unsafe impl Send for ContextRef {}

impl Context {
    /// Creates a cuTENSORMg handle for the devices that may participate in later operations.
    /// Retains each device's primary CUDA context for allocations and streams.
    ///
    /// # Errors
    ///
    /// Returns an error if the device count cannot be represented by cuTENSORMg,
    /// if cuTENSORMg cannot create a handle, or if it returns a null handle.
    pub fn create(devices: &[i32]) -> Result<Self> {
        let num_devices = to_u32(devices.len(), "devices")?;
        let cuda_contexts = devices
            .iter()
            .map(|&id| CudaContext::retain_primary_for_device(Device::new(id)))
            .collect::<CudaResult<Vec<_>>>()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `cutensorMgCreate` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorMgCreate(
                &raw mut handle,
                num_devices,
                devices.as_ptr(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Arc::new(Handle {
                raw: handle,
                devices: devices.to_vec(),
                cuda_contexts,
            }),
        })
    }

    /// Wraps an existing cuTENSORMg handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMg handle for the supplied CUDA contexts, in their device order. The returned
    /// context takes ownership of `handle` and destroys it when the last clone
    /// is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
    pub unsafe fn from_raw(
        handle: sys::cutensorMgHandle_t,
        cuda_contexts: Vec<Arc<CudaContext>>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Arc::new(Handle {
                raw: handle,
                devices: cuda_contexts
                    .iter()
                    .map(|context| context.device().id())
                    .collect(),
                cuda_contexts,
            }),
        })
    }

    /// Returns participating device ordinals in native handle order.
    #[must_use]
    pub fn devices(&self) -> &[i32] {
        &self.handle.devices
    }

    /// Returns the number of participating devices.
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.handle.devices.len()
    }

    /// Returns the retained CUDA context for a participating device.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` does not participate in this multi-GPU context.
    pub fn cuda_context(&self, device: i32) -> Result<&Arc<CudaContext>> {
        self.handle.cuda_context(device)
    }

    pub(crate) fn as_context_ref(&self) -> ContextRef {
        ContextRef {
            handle: Arc::clone(&self.handle),
        }
    }

    /// Returns the borrowed native multi-GPU library handle.
    #[must_use]
    pub fn as_raw(&self) -> sys::cutensorMgHandle_t {
        self.handle.raw
    }

    /// Returns the owned raw cuTENSORMg handle and its retained CUDA contexts.
    ///
    /// Keep the contexts alive until all work is complete and the raw handle is destroyed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OwnershipShared`] if cloned contexts or context references
    /// still point at the same handle.
    pub fn into_raw(self) -> Result<(sys::cutensorMgHandle_t, Vec<Arc<CudaContext>>)> {
        let handle = Arc::try_unwrap(self.handle).map_err(|_| Error::OwnershipShared)?;
        let mut handle = ManuallyDrop::new(handle);
        let contexts = mem::take(&mut handle.cuda_contexts);
        drop(mem::take(&mut handle.devices));
        Ok((handle.raw, contexts))
    }
}

impl ContextRef {
    /// Returns the retained CUDA context for a participating device.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` does not participate in this multi-GPU context.
    pub fn cuda_context(&self, device: i32) -> Result<&Arc<CudaContext>> {
        self.handle.cuda_context(device)
    }

    /// Returns participating device ordinals in native handle order.
    #[must_use]
    pub fn devices(&self) -> &[i32] {
        &self.handle.devices
    }

    /// Returns the number of participating devices.
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.handle.devices.len()
    }

    pub(crate) fn same_handle(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.handle, &other.handle)
    }

    /// Returns the borrowed native multi-GPU library handle.
    #[must_use]
    pub fn as_raw(&self) -> sys::cutensorMgHandle_t {
        self.handle.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        for context in &self.cuda_contexts {
            if let Err(error) = context.synchronize() {
                #[cfg(debug_assertions)]
                eprintln!("failed to synchronize cutensormg context: {error}");
                return;
            }
        }
        // SAFETY: `cutensorMgDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMgDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormg context: {error}");
            }
        }
    }
}

impl Handle {
    fn cuda_context(&self, device: i32) -> Result<&Arc<CudaContext>> {
        self.devices
            .iter()
            .position(|&id| id == device)
            .map(|index| &self.cuda_contexts[index])
            .ok_or(Error::MgDeviceNotInContext { device })
    }
}

pub(crate) fn validate_same_context(
    expected: &ContextRef,
    actual: &ContextRef,
    name: &str,
) -> Result<()> {
    if !expected.same_handle(actual) {
        return Err(Error::MgContextMismatch {
            name: name.to_owned().into(),
        });
    }
    Ok(())
}
