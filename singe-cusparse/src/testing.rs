//! Serialized GPU test contexts.

use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    testing::{TestDevice, bootstrap_for_device},
};

use crate::{context::Context, error::Result};

/// A cuSPARSE context on a locked test device.
///
/// The cuSPARSE context drops before the device's CUDA context and lock.
#[derive(Debug)]
pub struct TestContext {
    context: Context,
    device: TestDevice,
}

impl TestContext {
    /// Returns the cuSPARSE context.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Returns the cuSPARSE context for its persistent setters.
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    /// Returns the CUDA context of the locked device.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.device.context()
    }
}

/// Acquires the device test lock and creates a cuSPARSE context on CUDA device zero.
///
/// # Errors
///
/// Returns an error if the device lock, CUDA initialization, CUDA context creation, or cuSPARSE handle creation fails.
pub fn setup_context() -> Result<TestContext> {
    let device = bootstrap_for_device(Device::new(0))?;
    let context = Context::create(device.context())?;
    Ok(TestContext { context, device })
}
