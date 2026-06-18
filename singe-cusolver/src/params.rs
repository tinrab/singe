use std::ptr;

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    types::{AlgorithmMode, Function},
};

#[derive(Debug)]
pub struct Params {
    handle: sys::cusolverDnParams_t,
}

// cuSOLVER parameter handles store solver options and do not expose mutation
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
        unsafe {
            try_ffi!(sys::cusolverDnSetAdvOptions(
                self.as_raw(),
                function.into(),
                algorithm.into(),
            ))?;
        }
        Ok(())
    }

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
    pub unsafe fn from_raw(handle: sys::cusolverDnParams_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle })
    }

    /// Releases ownership and returns the raw cuSOLVER params handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    pub fn into_raw(self) -> sys::cusolverDnParams_t {
        let handle = self.handle;
        std::mem::forget(self);
        handle
    }
}

impl Drop for Params {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cusolverDnDestroyParams(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusolver params: {err}");
            }
        }
    }
}
