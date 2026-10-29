use std::ops::Deref;

use singe_cuda::testing::{DeviceLock, bootstrap_for_device};

use crate::{context::Context, error::Result};

/// cuSPARSE context and device lock retained for a serialized GPU test.
#[derive(Debug)]
pub struct TestContext {
    context: Context,
    _lock: DeviceLock,
}

impl Deref for TestContext {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

/// Acquires the device test lock and creates a cuSPARSE context on CUDA device zero.
///
/// # Errors
///
/// Propagates device-lock setup, CUDA initialization, context creation, and native cuSPARSE handle-creation errors.
pub fn setup_context() -> Result<TestContext> {
    let device_id: i32 = 0;
    let (lock, cuda_context) = bootstrap_for_device(device_id)?;
    let context = Context::create(&cuda_context)?;

    Ok(TestContext {
        context,
        _lock: lock,
    })
}
