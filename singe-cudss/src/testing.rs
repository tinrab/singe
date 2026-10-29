use std::ops::Deref;

use singe_cuda::testing::{DeviceLock, bootstrap_for_device};

use crate::{context::Context, error::Result};

/// cuDSS test context that retains the lock for CUDA device zero.
#[derive(Debug)]
pub struct TestContext {
    /// Solver context associated with the locked device.
    pub context: Context,
    _lock: DeviceLock,
}

impl Deref for TestContext {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

/// Locks CUDA device zero and creates its CUDA and cuDSS contexts.
///
/// # Errors
///
/// Returns an error if CUDA test setup or cuDSS handle creation fails.
pub fn setup_context() -> Result<TestContext> {
    let device_id: i32 = 0;
    let (lock, cuda_context) = bootstrap_for_device(device_id)?;
    let context = Context::create(&cuda_context)?;

    Ok(TestContext {
        context,
        _lock: lock,
    })
}
