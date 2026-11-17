//! Algorithm-selection parameters for the native 64-bit interface.

use std::{mem, ptr};

use singe_core::report_cleanup_failure;

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    types::{AlgorithmMode, Function},
};

/// Owns algorithm-selection parameters for the cuSOLVER 64-bit interface.
#[derive(Debug)]
pub struct Params {
    handle: sys::cusolverDnParams_t,
}

// SAFETY: the parameter structure is host memory with no thread affinity, owned uniquely by this wrapper, so it may be configured and destroyed from any thread.
unsafe impl Send for Params {}

// SAFETY: `set_algorithm` takes `&mut self`, and solvers only read the options through a shared handle, so shared references cannot race.
unsafe impl Sync for Params {}

impl Params {
    /// Creates and initializes the parameter structure for the cuSOLVER 64-bit interface.
    ///
    /// The returned [`Params`] owns the cuSOLVER parameter handle and destroys it when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER cannot allocate the parameter structure or if it does not return a valid handle.
    pub fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: the call writes a new handle to the local `handle`.
        unsafe {
            try_ffi!(sys::cusolverDnCreateParams(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle })
    }

    /// Configures the algorithm for a 64-bit cuSOLVER operation.
    ///
    /// # Errors
    ///
    /// Returns an error if `function` and `algorithm` are not a valid combination for cuSOLVER.
    pub fn set_algorithm(&mut self, function: Function, algorithm: AlgorithmMode) -> Result<()> {
        // SAFETY: `self` owns the live handle and is borrowed mutably, so no solver reads it concurrently.
        unsafe {
            try_ffi!(sys::cusolverDnSetAdvOptions(
                self.as_raw(),
                function.into(),
                algorithm.into(),
            ))?;
        }
        Ok(())
    }

    /// Returns the borrowed native parameter handle; ownership remains with this wrapper.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusolverDnParams_t {
        self.handle
    }

    /// Adopts a native parameter handle, which the returned value destroys on drop.
    ///
    /// # Safety
    ///
    /// `handle` must be a live cuSOLVER parameter handle that no other owner destroys or uses afterwards.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
    pub unsafe fn from_raw(handle: sys::cusolverDnParams_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle })
    }

    /// Releases ownership and returns the raw cuSOLVER params handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(self) -> sys::cusolverDnParams_t {
        let handle = self.handle;
        mem::forget(self);
        handle
    }
}

impl Drop for Params {
    fn drop(&mut self) {
        // SAFETY: `self` uniquely owns the live handle, and the raw execution contract keeps it alive until queued work that reads it completes.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusolverDnDestroyParams(self.handle)) {
                report_cleanup_failure("destroy", "cusolver params", &error);
            }
        }
    }
}
