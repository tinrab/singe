use std::ptr;

use crate::{
    context::Context,
    error::{Error, Result},
    sys, try_ffi,
};

#[derive(Debug)]
pub struct SyevjInfo {
    handle: sys::syevjInfo_t,
}

// syevj info handles store solver options and expose mutation only through
// &mut self, so immutable sharing is allowed.
unsafe impl Send for SyevjInfo {}
unsafe impl Sync for SyevjInfo {}

impl SyevjInfo {
    /// Creates `syevj`, `syevj_batched`, and `sygvj` parameter storage with default values.
    ///
    /// The returned [`SyevjInfo`] owns the cuSOLVER parameter handle and destroys
    /// it when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER cannot allocate the parameter storage or if
    /// it does not return a valid handle.
    pub fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusolverDnCreateSyevjInfo(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle })
    }

    /// Configures the `syevj` tolerance.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the tolerance value.
    pub fn set_tolerance(&mut self, tolerance: f64) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnXsyevjSetTolerance(self.as_raw(), tolerance,))?;
        }
        Ok(())
    }

    /// Configures the maximum number of `syevj` sweeps.
    /// The default value is 100.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the sweep count.
    pub fn set_max_sweeps(&mut self, max_sweeps: i32) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnXsyevjSetMaxSweeps(self.as_raw(), max_sweeps,))?;
        }
        Ok(())
    }

    /// If `sort_eigenvalues` is false, the eigenvalues are not sorted.
    /// This setting only applies to `syevj_batched`.
    /// `syevj` and `sygvj` always sort eigenvalues in ascending order.
    /// By default, eigenvalues are always sorted in ascending order.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the sort setting.
    pub fn set_sort_eigenvalues(&mut self, sort_eigenvalues: bool) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnXsyevjSetSortEig(
                self.as_raw(),
                i32::from(sort_eigenvalues),
            ))?;
        }
        Ok(())
    }

    /// Returns the residual reported by `syevj` or `sygvj`.
    /// This accessor does not support `syevj_batched`.
    /// Calling this after `syevj_batched` returns [`crate::error::Status::NotSupported`].
    ///
    /// # Errors
    ///
    /// Returns an error if the info handle was used with `syevj_batched`,
    /// which does not report a residual.
    pub fn residual(&self, ctx: &Context) -> Result<f64> {
        ctx.bind()?;

        let mut residual = 0.0;
        unsafe {
            try_ffi!(sys::cusolverDnXsyevjGetResidual(
                ctx.as_raw(),
                self.as_raw(),
                &raw mut residual,
            ))?;
        }
        Ok(residual)
    }

    /// Returns the number of executed `syevj` or `sygvj` sweeps.
    /// This accessor does not support `syevj_batched`.
    /// Calling this after `syevj_batched` returns [`crate::error::Status::NotSupported`].
    ///
    /// # Errors
    ///
    /// Returns an error if the info handle was used with `syevj_batched`,
    /// which does not report a sweep count.
    pub fn executed_sweeps(&self, ctx: &Context) -> Result<i32> {
        ctx.bind()?;

        let mut sweeps = 0;
        unsafe {
            try_ffi!(sys::cusolverDnXsyevjGetSweeps(
                ctx.as_raw(),
                self.as_raw(),
                &raw mut sweeps,
            ))?;
        }
        Ok(sweeps)
    }

    pub fn as_raw(&self) -> sys::syevjInfo_t {
        self.handle
    }

    /// Takes ownership of a raw cuSOLVER `syevj` info handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `syevjInfo_t` created by cuSOLVER. The
    /// returned wrapper takes ownership and will destroy it with
    /// `cusolverDnDestroySyevjInfo`; no other owner may destroy or keep using it.
    pub unsafe fn from_raw(handle: sys::syevjInfo_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle })
    }

    /// Releases ownership and returns the raw cuSOLVER `syevj` info handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    pub fn into_raw(self) -> sys::syevjInfo_t {
        let handle = self.handle;
        std::mem::forget(self);
        handle
    }
}

impl Drop for SyevjInfo {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cusolverDnDestroySyevjInfo(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusolver syevj info: {err}");
            }
        }
    }
}
