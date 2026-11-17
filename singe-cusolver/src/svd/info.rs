use std::{mem, ptr};

use singe_core::report_cleanup_failure;

use crate::{
    context::Context,
    error::{Error, Result},
    sys, try_ffi,
};

/// Owns native options and completed convergence results for Jacobi singular value decompositions.
///
/// Complete solver work before inspecting results or changing the options.
/// Raw execution must retain the object and exclude concurrent access until completion.
#[derive(Debug)]
pub struct GesvdjInfo {
    handle: sys::gesvdjInfo_t,
}

// SAFETY: `gesvdjInfo_t` is host-side parameter storage with no thread affinity, and this wrapper is its only owner.
unsafe impl Send for GesvdjInfo {}

// SAFETY: `&self` methods only read options or completed results; setters and execution take `&mut self`, so shared access never overlaps a native write.
unsafe impl Sync for GesvdjInfo {}

impl GesvdjInfo {
    /// Creates parameter storage with default options for [`Gesvdj`](crate::svd::Gesvdj) and [`GesvdjBatched`](crate::svd::GesvdjBatched).
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER cannot allocate the parameter storage or if it does not return a valid handle.
    pub fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a writable local that receives the new parameter object.
        unsafe {
            try_ffi!(sys::cusolverDnCreateGesvdjInfo(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle })
    }

    /// Sets the convergence tolerance of the Jacobi SVD.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects `tolerance`.
    pub fn set_tolerance(&mut self, tolerance: f64) -> Result<()> {
        // SAFETY: `self` owns a live parameter object, and `&mut self` excludes concurrent host access; queued solves keep exclusive use under the raw execution contract.
        unsafe {
            try_ffi!(sys::cusolverDnXgesvdjSetTolerance(self.as_raw(), tolerance,))?;
        }
        Ok(())
    }

    /// Sets the maximum number of Jacobi sweeps.
    /// The default value is 100.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects `max_sweeps`.
    pub fn set_max_sweeps(&mut self, max_sweeps: i32) -> Result<()> {
        // SAFETY: `self` owns a live parameter object, and `&mut self` excludes concurrent host access; queued solves keep exclusive use under the raw execution contract.
        unsafe {
            try_ffi!(sys::cusolverDnXgesvdjSetMaxSweeps(
                self.as_raw(),
                max_sweeps,
            ))?;
        }
        Ok(())
    }

    /// Sets whether [`GesvdjBatched`](crate::svd::GesvdjBatched) sorts singular values in descending order.
    /// Sorting is on by default; [`Gesvdj`](crate::svd::Gesvdj) always sorts.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the sort setting.
    pub fn set_sort_eigenvalues(&mut self, sort_eigenvalues: bool) -> Result<()> {
        // SAFETY: `self` owns a live parameter object, and `&mut self` excludes concurrent host access; queued solves keep exclusive use under the raw execution contract.
        unsafe {
            try_ffi!(sys::cusolverDnXgesvdjSetSortEig(
                self.as_raw(),
                i32::from(sort_eigenvalues),
            ))?;
        }
        Ok(())
    }

    /// Returns the Frobenius norm of the internal residual of the last completed [`Gesvdj`](crate::svd::Gesvdj) solve.
    ///
    /// The internal residual differs from the exact residual by roundoff.
    ///
    /// # Errors
    ///
    /// Returns an error if the object was last used by [`GesvdjBatched`](crate::svd::GesvdjBatched), which reports no residual, as [`Status::NotSupported`](crate::error::Status::NotSupported).
    pub fn residual(&self, context: &Context) -> Result<f64> {
        context.bind()?;

        let mut residual = 0.0;
        // SAFETY: `context` and `self` own live handles, and `residual` is a writable local.
        unsafe {
            try_ffi!(sys::cusolverDnXgesvdjGetResidual(
                context.as_raw(),
                self.as_raw(),
                &raw mut residual,
            ))?;
        }
        Ok(residual)
    }

    /// Returns the number of sweeps the last completed [`Gesvdj`](crate::svd::Gesvdj) solve executed.
    ///
    /// # Errors
    ///
    /// Returns an error if the object was last used by [`GesvdjBatched`](crate::svd::GesvdjBatched), which reports no sweep count, as [`Status::NotSupported`](crate::error::Status::NotSupported).
    pub fn executed_sweeps(&self, context: &Context) -> Result<i32> {
        context.bind()?;

        let mut sweeps = 0;
        // SAFETY: `context` and `self` own live handles, and `sweeps` is a writable local.
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

    /// Takes ownership of a raw Jacobi SVD parameter handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a live parameter handle created by cuSOLVER.
    /// The returned wrapper destroys it on drop, so no other owner may destroy or keep using it.
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

    /// Releases ownership and returns the raw parameter handle.
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
        // SAFETY: `self` owns the handle and is being dropped, so nothing uses it after this call.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusolverDnDestroyGesvdjInfo(self.handle)) {
                report_cleanup_failure("destroy", "cusolver gesvdj info", &error);
            }
        }
    }
}
