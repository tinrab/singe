//! Shared setup for tests that need a GPU.

use std::ops::Deref;

use singe_cuda::testing::{DeviceLock, bootstrap_for_device};

use crate::{context::Context, error::Result};

/// nvJPEG handle for a test, with a lock that keeps GPU tests from running at the same time.
#[derive(Debug)]
pub struct TestContext {
    /// nvJPEG handle on the locked device.
    pub context: Context,
    _lock: DeviceLock,
}

impl Deref for TestContext {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

/// Locks CUDA device 0 and creates an nvJPEG handle on it.
///
/// # Errors
///
/// Returns an error if device 0 cannot be initialized or locked, its CUDA context cannot be created or bound, or nvJPEG fails to create the handle.
pub fn setup_context() -> Result<TestContext> {
    let device_id: i32 = 0;
    let (lock, cuda_context) = bootstrap_for_device(device_id)?;
    let context = Context::create_simple(&cuda_context)?;

    Ok(TestContext {
        context,
        _lock: lock,
    })
}
