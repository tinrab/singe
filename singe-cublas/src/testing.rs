use std::ops::Deref;

use singe_cuda::testing::{DeviceLock, bootstrap_for_device};

use crate::{context::Context, error::Result};

/// A cuBLAS context retaining the GPU test lock until it is dropped.
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

/// Acquires the test lock for CUDA device 0 and creates a cuBLAS context.
///
/// # Errors
///
/// Returns an error if CUDA device setup, context binding, or cuBLAS handle creation fails.
pub fn setup_context() -> Result<TestContext> {
    let device_id: i32 = 0;
    let (lock, cuda_context) = bootstrap_for_device(device_id)?;
    let context = Context::create(&cuda_context)?;

    Ok(TestContext {
        context,
        _lock: lock,
    })
}
