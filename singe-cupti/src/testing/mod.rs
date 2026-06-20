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
    let (lock, cuda_context) = singe_cuda::testing::bootstrap()?;
    let context = Context::from_cuda_context(cuda_context);
    Ok(TestContext {
        context,
        _lock: lock,
    })
}
