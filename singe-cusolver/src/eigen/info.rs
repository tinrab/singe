use std::{mem, ptr};

use crate::{
    context::Context,
    error::{Error, Result},
    sys, try_ffi,
};

/// Owns native options and completed convergence results for symmetric or Hermitian Jacobi eigensolvers.
///
/// Complete solver work before inspecting results or changing the options. Raw execution must retain the object and exclude concurrent access until completion.
#[derive(Debug)]
pub struct SyevjInfo {
    handle: sys::syevjInfo_t,
}

// SAFETY: the owned native parameter object is movable between host threads.
// Setters and execution require &mut self; shared access only reads options or
// completed results. Raw execution must exclude concurrent parameter access.
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
        // SAFETY: `cusolverDnCreateSyevjInfo` receives valid output locations.
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
        // SAFETY: `cusolverDnXsyevjSetTolerance` receives live handles owned by the wrappers.
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
        // SAFETY: `cusolverDnXsyevjSetMaxSweeps` receives live handles owned by the wrappers.
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
        // SAFETY: `cusolverDnXsyevjSetSortEig` receives live handles owned by the wrappers.
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
    pub fn residual(&self, context: &Context) -> Result<f64> {
        context.bind()?;

        let mut residual = 0.0;
        // SAFETY: `cusolverDnXsyevjGetResidual` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusolverDnXsyevjGetResidual(
                context.as_raw(),
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
    pub fn executed_sweeps(&self, context: &Context) -> Result<i32> {
        context.bind()?;

        let mut sweeps = 0;
        // SAFETY: `cusolverDnXsyevjGetSweeps` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusolverDnXsyevjGetSweeps(
                context.as_raw(),
                self.as_raw(),
                &raw mut sweeps,
            ))?;
        }
        Ok(sweeps)
    }

    /// Returns the borrowed native parameter handle; ownership remains with this wrapper.
    #[must_use]
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
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
    pub unsafe fn from_raw(handle: sys::syevjInfo_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle })
    }

    /// Releases ownership and returns the raw cuSOLVER `syevj` info handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(self) -> sys::syevjInfo_t {
        let handle = self.handle;
        mem::forget(self);
        handle
    }
}

impl Drop for SyevjInfo {
    fn drop(&mut self) {
        // SAFETY: `cusolverDnDestroySyevjInfo` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusolverDnDestroySyevjInfo(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusolver syevj info: {error}");
            }
        }
    }
}
