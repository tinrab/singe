//! Serialized GPU test contexts.

use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    testing::{TestDevice, bootstrap_for_device},
};

use crate::{context::Context, error::Result};

/// A cuTENSOR context on a locked test device.
///
/// The cuTENSOR context drops before the device's CUDA context and lock.
#[derive(Debug)]
pub struct TestContext {
    context: Context,
    device: TestDevice,
}

impl TestContext {
    /// Returns the cuTENSOR context.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Returns the cuTENSOR context for its persistent setters.
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    /// Returns the CUDA context of the locked device.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.device.context()
    }
}

/// Locks CUDA device zero and creates a cuTENSOR context in its CUDA context.
///
/// # Errors
///
/// Returns an error if device initialization, locking, CUDA context creation or binding, or native cuTENSOR initialization fails.
pub fn setup_context() -> Result<TestContext> {
    let device = bootstrap_for_device(Device::new(0))?;
    let context = Context::create(device.context())?;
    Ok(TestContext { context, device })
}
