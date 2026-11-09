use std::ops::Deref;

use singe_cuda::testing::{DeviceLock, bootstrap_for_device};

use crate::{error::Result, instance::Instance};

/// nvImageCodec instance for a test, with a lock that keeps GPU tests from running at the same time.
#[derive(Debug)]
pub struct TestContext {
    /// The nvImageCodec instance used by the test.
    pub instance: Instance,
    _lock: DeviceLock,
}

impl Deref for TestContext {
    type Target = Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

/// Locks CUDA device zero, initializes its context, and creates a default nvImageCodec instance.
///
/// # Errors
///
/// Returns an error if device initialization, context creation, device locking, or native instance creation fails.
pub fn setup_context() -> Result<TestContext> {
    let device_id: i32 = 0;
    let (lock, _cuda_context) = bootstrap_for_device(device_id)?;
    let instance = Instance::create_default()?;

    Ok(TestContext {
        instance,
        _lock: lock,
    })
}
