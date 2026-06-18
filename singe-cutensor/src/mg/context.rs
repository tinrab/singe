use std::{mem::ManuallyDrop, ptr, sync::Arc};

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    utility::to_u32,
};

#[derive(Debug, Clone)]
pub struct Context {
    handle: Arc<Handle>,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cutensorMgHandle_t,
    devices: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct ContextRef {
    handle: Arc<Handle>,
}

// cuTENSORMg contexts are immutable after creation for a fixed device set. The
// Arc-backed handle can be shared, while wrappers remain freely movable.
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}
unsafe impl Send for Context {}
unsafe impl Send for ContextRef {}

impl Context {
    /// Creates a cuTENSORMg handle for the devices that may participate in later operations.
    ///
    /// # Errors
    ///
    /// Returns an error if the device count cannot be represented by cuTENSORMg,
    /// if cuTENSORMg cannot create a handle, or if it returns a null handle.
    pub fn create(devices: &[i32]) -> Result<Self> {
        let num_devices = to_u32(devices.len(), "devices")?;
        let mut handle = ptr::null_mut();
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
            }),
        })
    }

    /// Wraps an existing cuTENSORMg handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMg handle for `devices`. The returned
    /// context takes ownership of `handle` and destroys it when the last clone
    /// is dropped.
    pub unsafe fn from_raw(handle: sys::cutensorMgHandle_t, devices: Vec<i32>) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Arc::new(Handle {
                raw: handle,
                devices,
            }),
        })
    }

    pub fn devices(&self) -> &[i32] {
        &self.handle.devices
    }

    pub fn device_count(&self) -> usize {
        self.handle.devices.len()
    }

    pub(crate) fn as_context_ref(&self) -> ContextRef {
        ContextRef {
            handle: Arc::clone(&self.handle),
        }
    }

    pub fn as_raw(&self) -> sys::cutensorMgHandle_t {
        self.handle.raw
    }

    /// Consumes this context and returns the owned raw cuTENSORMg handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::HandleShared`] if cloned contexts or context references
    /// still point at the same handle.
    pub fn into_raw(self) -> Result<sys::cutensorMgHandle_t> {
        let handle = Arc::try_unwrap(self.handle).map_err(|_| Error::HandleShared)?;
        let handle = ManuallyDrop::new(handle);
        Ok(handle.raw)
    }
}

impl ContextRef {
    pub fn devices(&self) -> &[i32] {
        &self.handle.devices
    }

    pub fn device_count(&self) -> usize {
        self.handle.devices.len()
    }

    pub(crate) fn same_handle(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.handle, &other.handle)
    }

    pub fn as_raw(&self) -> sys::cutensorMgHandle_t {
        self.handle.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorMgDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormg context: {err}");
            }
        }
    }
}

pub(crate) fn validate_same_context(
    expected: &ContextRef,
    actual: &ContextRef,
    name: &str,
) -> Result<()> {
    if !expected.same_handle(actual) {
        return Err(Error::MgContextMismatch { name: name.into() });
    }
    Ok(())
}
