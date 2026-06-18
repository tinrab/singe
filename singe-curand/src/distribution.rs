//! Discrete distribution descriptors.

use std::ptr;

use singe_curand_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
};

/// A cuRAND discrete distribution histogram descriptor.
///
/// cuRAND currently exposes this safe wrapper for Poisson histograms created by
/// [`DiscreteDistribution::create_poisson`]. The descriptor owns the underlying
/// cuRAND distribution and destroys it on drop.
#[derive(Debug)]
pub struct DiscreteDistribution {
    handle: sys::curandDiscreteDistribution_t,
}

// cuRAND discrete distributions are owned descriptors with no shared mutation
// through this wrapper. Moving ownership between threads is allowed.
unsafe impl Send for DiscreteDistribution {}

impl DiscreteDistribution {
    /// Creates a Poisson distribution histogram for `lambda`.
    ///
    /// cuRAND accepts positive `lambda` values up to 400,000. The returned
    /// descriptor can be passed to lower-level APIs that consume a cuRAND
    /// discrete distribution handle.
    ///
    /// # Errors
    ///
    /// Returns an error if cuRAND cannot allocate or initialize the histogram,
    /// if `lambda` is outside cuRAND's supported range, or if cuRAND reports a
    /// pending CUDA failure.
    pub fn create_poisson(lambda: f64) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::curandCreatePoissonDistribution(
                lambda,
                &raw mut handle
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle })
    }

    /// Returns the raw cuRAND distribution handle.
    ///
    /// The returned handle is owned by this wrapper and must not be destroyed by
    /// the caller. It is valid only while `self` is alive.
    pub const fn as_raw(&self) -> sys::curandDiscreteDistribution_t {
        self.handle
    }
}

impl Drop for DiscreteDistribution {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = sys::curandDestroyDistribution(self.handle);
            }
        }
    }
}
