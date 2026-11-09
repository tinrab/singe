use std::panic;

use singe_cuda::testing::{DeviceLock, device_lock, is_unavailable};

use crate::error::{Error, Result};

/// Tries to lock CUDA device 0 for serialized tests.
///
/// Returns `None` when there is no CUDA driver, a stub library or none that loads, allowing the caller to skip hardware-dependent assertions.
///
/// # Panics
///
/// Panics if device locking fails for any reason other than a missing CUDA driver.
#[must_use]
pub fn maybe_lock_device() -> Option<DeviceLock> {
    match device_lock(0) {
        Ok(lock) => Some(lock),
        Err(error) if is_unavailable(&error) => None,
        Err(error) => panic::panic_any(format!("cuda device lock failed: {error:?}")),
    }
}

/// Converts a CUDA-dependent result into an optional test value.
///
/// Returns `None` when the result failed because there is no CUDA driver: a stub library, or none that loads.
///
/// # Panics
///
/// Panics for non-stub errors.
pub fn skip_if_stub<T>(result: Result<T>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(Error::Cuda(error)) if is_unavailable(&error) => None,
        Err(error) => panic::panic_any(format!("cuda-dependent test setup failed: {error:?}")),
    }
}
