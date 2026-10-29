use std::ops::Deref;

use singe_cuda::testing::{DeviceLock, bootstrap_for_device};

use crate::{error::Result, plan::Plan};

/// Owns an unconfigured plan and the device lock serializing GPU tests.
#[derive(Debug)]
pub struct TestContext {
    /// Plan created in the test CUDA context on device 0.
    pub plan: Plan,
    _lock: DeviceLock,
}

impl Deref for TestContext {
    type Target = Plan;

    fn deref(&self) -> &Self::Target {
        &self.plan
    }
}

/// Locks CUDA device 0, initializes its test context, and creates an unconfigured plan.
///
/// # Errors
///
/// Returns an error if device locking, CUDA initialization, context creation,
/// or native plan creation fails.
pub fn setup_context() -> Result<TestContext> {
    let device_id: i32 = 0;
    let (lock, cuda_context) = bootstrap_for_device(device_id)?;
    let plan = Plan::create(&cuda_context)?;

    Ok(TestContext { plan, _lock: lock })
}
