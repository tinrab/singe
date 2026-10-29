use std::{mem, ptr};

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

// SAFETY: cuSOLVER parameter handles store solver options and do not expose mutation
// through shared references, so immutable sharing is allowed.
unsafe impl Send for Params {}

unsafe impl Sync for Params {}

impl Params {
    /// Creates and initializes the parameter structure for the cuSOLVER 64-bit interface.
    ///
    /// The returned [`Params`] owns the cuSOLVER parameter handle and destroys it
    /// when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER cannot allocate the parameter structure or
    /// if it does not return a valid handle.
    pub fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: `cusolverDnCreateParams` receives valid output locations.
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
    /// Returns an error if `function` and `algorithm` are not a valid
    /// combination for cuSOLVER.
    pub fn set_adv_options(&mut self, function: Function, algorithm: AlgorithmMode) -> Result<()> {
        // SAFETY: `cusolverDnSetAdvOptions` receives live handles owned by the wrappers.
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

    /// Takes ownership of a raw cuSOLVER params handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cusolverDnParams_t` created by cuSOLVER. The
    /// returned wrapper takes ownership and will destroy it with
    /// `cusolverDnDestroyParams`; no other owner may destroy or keep using it.
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
        // SAFETY: `cusolverDnDestroyParams` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusolverDnDestroyParams(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusolver params: {error}");
            }
        }
    }
}
