use singe_cuda::testing::DeviceLock;

use crate::{context::Context, error::Result};

/// CUDA context fixture for CUPTI tests.
pub struct TestContext {
    context: Context,
    _lock: DeviceLock,
}

impl TestContext {
    /// Returns the CUDA context used by the test fixture.
    pub fn context(&self) -> &Context {
        &self.context
    }
}

/// Creates and binds a CUDA context for CUPTI tests.
pub fn setup_context() -> Result<TestContext> {
    let lock = singe_cuda::testing::device_lock(0)?;
    let context = Context::create()?;
    Ok(TestContext {
        context,
        _lock: lock,
    })
}
