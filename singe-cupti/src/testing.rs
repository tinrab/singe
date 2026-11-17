//! Serialized GPU test contexts.

use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext,
    testing::{TestDevice, bootstrap},
};

use crate::{context::Context, error::Result};

/// A CUPTI context on a locked test device.
///
/// The CUPTI context drops before the device's CUDA context and lock.
#[derive(Debug)]
pub struct TestContext {
    context: Context,
    device: TestDevice,
}

impl TestContext {
    /// Returns the CUPTI context.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Returns the CUDA context of the locked device.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.device.context()
    }
}

/// Creates and binds a CUDA context for CUPTI tests.
///
/// # Errors
///
/// Returns an error if CUDA test-device initialization, locking, context creation, or binding fails.
pub fn setup_context() -> Result<TestContext> {
    let device = bootstrap()?;
    let context = Context::new(device.context());
    Ok(TestContext { context, device })
}
