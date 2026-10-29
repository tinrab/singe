use std::{mem, ptr};

use crate::{
    context::Context,
    error::{Error, Result},
    sys, try_ffi,
};

/// Owns native options and completed convergence results for Jacobi singular value decompositions.
///
/// Complete solver work before inspecting results or changing the options. Raw execution must retain the object and exclude concurrent access until completion.
#[derive(Debug)]
pub struct GesvdjInfo {
    handle: sys::gesvdjInfo_t,
}

// SAFETY: native execution and option changes require exclusive access. Completed
// getters may share an immutable borrow of the parameter storage.
unsafe impl Send for GesvdjInfo {}

unsafe impl Sync for GesvdjInfo {}

impl GesvdjInfo {
    /// Creates `gesvdj` and `gesvdjBatched` parameter storage with default values.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER cannot allocate the parameter storage or if
    /// it does not return a valid handle.
    pub fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: `cusolverDnCreateGesvdjInfo` receives valid output locations.
        unsafe {
            try_ffi!(sys::cusolverDnCreateGesvdjInfo(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle })
    }

    /// Configures the `gesvdj` tolerance.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects `tolerance`.
    pub fn set_tolerance(&mut self, tolerance: f64) -> Result<()> {
        // SAFETY: `cusolverDnXgesvdjSetTolerance` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusolverDnXgesvdjSetTolerance(self.as_raw(), tolerance,))?;
        }
        Ok(())
    }

    /// Configures the maximum number of `gesvdj` sweeps.
    /// The default value is 100.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects `max_sweeps`.
    pub fn set_max_sweeps(&mut self, max_sweeps: i32) -> Result<()> {
        // SAFETY: `cusolverDnXgesvdjSetMaxSweeps` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusolverDnXgesvdjSetMaxSweeps(
                self.as_raw(),
                max_sweeps,
            ))?;
        }
        Ok(())
    }

    /// If `sort_eigenvalues` is false, the singular values are not sorted.
    /// This setting only applies to `gesvdjBatched`.
    /// `gesvdj` always sorts singular values in descending order.
    /// By default, singular values are always sorted in descending order.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the sort setting.
    pub fn set_sort_eigenvalues(&mut self, sort_eigenvalues: bool) -> Result<()> {
        // SAFETY: `cusolverDnXgesvdjSetSortEig` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusolverDnXgesvdjSetSortEig(
                self.as_raw(),
                i32::from(sort_eigenvalues),
            ))?;
        }
        Ok(())
    }

    /// Returns the Frobenius norm of the internal residual reported by `gesvdj`.
    /// Not the Frobenius norm of the exact residual.
    ///
    /// This accessor does not support `gesvdjBatched`.
    /// Calling this after `gesvdjBatched` returns [`crate::error::Status::NotSupported`].
    ///
    /// # Errors
    ///
    /// Returns an error if the info handle was used with `gesvdjBatched`,
    /// which does not report a residual.
    pub fn residual(&self, context: &Context) -> Result<f64> {
        context.bind()?;

        let mut residual = 0.0;
        // SAFETY: `cusolverDnXgesvdjGetResidual` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusolverDnXgesvdjGetResidual(
                context.as_raw(),
                self.as_raw(),
                &raw mut residual,
            ))?;
        }
        Ok(residual)
    }

    /// Returns the number of executed `gesvdj` sweeps.
    /// This accessor does not support `gesvdjBatched`.
    /// Calling this after `gesvdjBatched` returns [`crate::error::Status::NotSupported`].
    ///
    /// # Errors
    ///
    /// Returns an error if the info handle was used with `gesvdjBatched`,
    /// which does not report a sweep count.
    pub fn executed_sweeps(&self, context: &Context) -> Result<i32> {
        context.bind()?;

        let mut sweeps = 0;
        // SAFETY: `cusolverDnXgesvdjGetSweeps` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusolverDnXgesvdjGetSweeps(
                context.as_raw(),
                self.as_raw(),
                &raw mut sweeps,
            ))?;
        }
        Ok(sweeps)
    }

    /// Returns the borrowed native parameter handle; ownership remains with this wrapper.
    #[must_use]
    pub fn as_raw(&self) -> sys::gesvdjInfo_t {
        self.handle
    }

    /// Takes ownership of a raw cuSOLVER `gesvdj` info handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `gesvdjInfo_t` created by cuSOLVER. The
    /// returned wrapper takes ownership and will destroy it with
    /// `cusolverDnDestroyGesvdjInfo`; no other owner may destroy or keep using it.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
    pub unsafe fn from_raw(handle: sys::gesvdjInfo_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle })
    }

    /// Releases ownership and returns the raw cuSOLVER `gesvdj` info handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(self) -> sys::gesvdjInfo_t {
        let handle = self.handle;
        mem::forget(self);
        handle
    }
}

impl Drop for GesvdjInfo {
    fn drop(&mut self) {
        // SAFETY: `cusolverDnDestroyGesvdjInfo` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusolverDnDestroyGesvdjInfo(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusolver gesvdj info: {error}");
            }
        }
    }
}
