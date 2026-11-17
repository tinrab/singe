use std::{mem::ManuallyDrop, ptr, rc::Rc, result::Result as StdResult, sync::Arc};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    context::Context as CudaContext, device::Device, error::Result as CudaResult,
    stream::StreamBinding,
};

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    utility::to_u32,
};

/// Owns a multi-GPU cuTENSOR handle and the primary CUDA contexts of its participating devices.
///
/// cuTENSORMg forbids concurrent contractions and copies on one handle.
/// Descriptors and plans share the handle, so the context and everything created from it stay on the creating thread: none of them is `Send` or `Sync`.
/// Use one context per thread.
#[derive(Debug)]
pub struct Context {
    handle: Rc<Handle>,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cutensorMgHandle_t,
    devices: Vec<Device>,
    cuda_contexts: Vec<Arc<CudaContext>>,
}

/// Retains a multi-GPU library handle and its ordered participating CUDA contexts.
#[derive(Debug)]
pub struct ContextRef {
    handle: Rc<Handle>,
}

/// A uniquely transferred cuTENSORMg handle and the CUDA contexts needed to use and destroy it.
///
/// Destroy `handle` before dropping this value, or reclaim it with [`Context::from_raw`].
#[derive(Debug)]
pub struct RawContext {
    /// Transferred native cuTENSORMg handle; destroy it before releasing the CUDA contexts.
    pub handle: sys::cutensorMgHandle_t,
    /// Primary CUDA contexts of the participating devices, in native handle order.
    pub cuda_contexts: Vec<Arc<CudaContext>>,
}

// SAFETY: the handle has no thread affinity, so a sole owner may destroy it on another thread.
// It is deliberately not `Sync`: `cutensorMgContraction` and `cutensorMgCopy` are thread-safe only on distinct handles, and leaving `Handle` `!Sync` makes every `Rc<Handle>` holder `!Send` and `!Sync`.
unsafe impl Send for Handle {}

impl Context {
    /// Creates a cuTENSORMg handle for the devices that may participate in later operations.
    /// Retains each device's primary CUDA context for allocations and streams.
    ///
    /// # Errors
    ///
    /// Returns an error if the device count cannot be represented by cuTENSORMg, if cuTENSORMg cannot create a handle, or if it returns a null handle.
    pub fn create(devices: &[Device]) -> Result<Self> {
        let num_devices = to_u32(devices.len(), "devices")?;
        let cuda_contexts = devices
            .iter()
            .map(|&device| CudaContext::create_for_device(device))
            .collect::<CudaResult<Vec<_>>>()?;
        let device_ids = devices.iter().map(|device| device.id()).collect::<Vec<_>>();
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a valid output location and `device_ids` holds `num_devices` ordinals for the call.
        unsafe {
            try_ffi!(sys::cutensorMgCreate(
                &raw mut handle,
                num_devices,
                device_ids.as_ptr(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Rc::new(Handle {
                raw: handle,
                devices: devices.to_vec(),
                cuda_contexts,
            }),
        })
    }

    /// Takes ownership of a native cuTENSORMg handle and its participating CUDA contexts.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid, uniquely owned cuTENSORMg handle for `raw.cuda_contexts`, in their device order.
    /// The returned context destroys the handle when it and every descriptor and plan created from it are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawContext) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Rc::new(Handle {
                raw: raw.handle,
                devices: raw
                    .cuda_contexts
                    .iter()
                    .map(|context| context.device())
                    .collect(),
                cuda_contexts: raw.cuda_contexts,
            }),
        })
    }

    /// Returns participating devices in native handle order.
    #[must_use]
    pub fn devices(&self) -> &[Device] {
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
    pub fn cuda_context(&self, device: Device) -> Result<&Arc<CudaContext>> {
        self.handle.cuda_context(device)
    }

    /// Returns the legacy default stream of each participating device, in native handle order.
    #[must_use]
    pub fn default_streams(&self) -> Vec<StreamBinding> {
        self.handle
            .cuda_contexts
            .iter()
            .map(|context| StreamBinding::Default(Arc::clone(context)))
            .collect()
    }

    /// Returns a shared reference to this handle, for building raw descriptor and plan records.
    #[must_use]
    pub fn as_context_ref(&self) -> ContextRef {
        ContextRef {
            handle: Rc::clone(&self.handle),
        }
    }

    /// Returns the borrowed native multi-GPU library handle.
    #[must_use]
    pub fn as_raw(&self) -> sys::cutensorMgHandle_t {
        self.handle.raw
    }

    /// Transfers the native handle and its CUDA contexts if nothing else retains the handle.
    ///
    /// Keep the contexts alive until all work is complete and the raw handle is destroyed.
    ///
    /// # Errors
    ///
    /// Returns the original context as `Err` if descriptors or plans still retain the handle.
    pub fn into_raw(self) -> StdResult<RawContext, Self> {
        let handle = match Rc::try_unwrap(self.handle) {
            Ok(handle) => ManuallyDrop::new(handle),
            Err(handle) => return Err(Self { handle }),
        };

        // SAFETY: `handle` is never dropped, so each owned field is read out exactly once.
        let cuda_contexts = unsafe { ptr::read(&raw const handle.cuda_contexts) };
        // SAFETY: as above; the device list is derived from the contexts and released here.
        drop(unsafe { ptr::read(&raw const handle.devices) });
        Ok(RawContext {
            handle: handle.raw,
            cuda_contexts,
        })
    }
}

impl ContextRef {
    /// Returns the retained CUDA context for a participating device.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` does not participate in this multi-GPU context.
    pub fn cuda_context(&self, device: Device) -> Result<&Arc<CudaContext>> {
        self.handle.cuda_context(device)
    }

    /// Returns participating devices in native handle order.
    #[must_use]
    pub fn devices(&self) -> &[Device] {
        &self.handle.devices
    }

    /// Returns the number of participating devices.
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.handle.devices.len()
    }

    pub(crate) fn same_handle(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.handle, &other.handle)
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
                report_cleanup_failure(
                    "synchronize the cuda contexts to destroy",
                    "cutensormg context",
                    &error,
                );
                return;
            }
        }

        // SAFETY: this is the last `Rc` owner of the handle, and every participating context was synchronized above.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMgDestroy(self.raw)) {
                report_cleanup_failure("destroy", "cutensormg context", &error);
            }
        }
    }
}

impl Handle {
    fn cuda_context(&self, device: Device) -> Result<&Arc<CudaContext>> {
        self.devices
            .iter()
            .position(|&candidate| candidate == device)
            .map(|index| &self.cuda_contexts[index])
            .ok_or(Error::MgDeviceNotInContext {
                device: device.id(),
            })
    }
}

pub(crate) fn validate_same_context(
    expected: &ContextRef,
    actual: &ContextRef,
    name: &str,
) -> Result<()> {
    if !expected.same_handle(actual) {
        return Err(Error::ContextMismatch {
            name: name.to_owned().into(),
        });
    }
    Ok(())
}
