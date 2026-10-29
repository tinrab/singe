use std::ops::Deref;

use singe_cuda::testing::{DeviceLock, bootstrap_for_device};

use crate::{context::Context, error::Result};

/// A cuTENSOR context and retained lock for serialized GPU tests.
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

/// Locks CUDA device zero and creates a cuTENSOR context in its actual CUDA context.
///
/// # Errors
///
/// Returns an error if device initialization, locking, CUDA context creation or binding, or native cuTENSOR initialization fails.
pub fn setup_context() -> Result<TestContext> {
    let device_id: i32 = 0;
    let (lock, cuda_context) = bootstrap_for_device(device_id)?;
    let context = Context::create(&cuda_context)?;

    Ok(TestContext {
        context,
        _lock: lock,
    })
}
