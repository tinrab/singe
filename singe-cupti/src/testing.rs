use singe_cuda::testing::{DeviceLock, bootstrap};

use crate::{context::Context, error::Result};

/// CUDA context fixture for CUPTI tests.
#[derive(Debug)]
pub struct TestContext {
    context: Context,
    _lock: DeviceLock,
}

impl TestContext {
    /// Returns the CUDA context used by the test fixture.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.context
    }
}

/// Creates and binds a CUDA context for CUPTI tests.
///
/// # Errors
///
/// Returns an error if CUDA test-device initialization, locking, context creation, or binding fails.
pub fn setup_context() -> Result<TestContext> {
    let (lock, cuda_context) = bootstrap()?;
    let context = Context::from_cuda_context(cuda_context);
    Ok(TestContext {
        context,
        _lock: lock,
    })
}
