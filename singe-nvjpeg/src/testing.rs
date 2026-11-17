//! Serialized GPU test contexts.

use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    testing::{TestDevice, bootstrap_for_device},
};

use crate::{
    context::{Context, ContextOptions},
    error::Result,
};

/// An nvJPEG handle on a locked test device.
///
/// The handle drops before the device's CUDA context and lock.
#[derive(Debug)]
pub struct TestContext {
    context: Context,
    device: TestDevice,
}

impl TestContext {
    /// Returns the nvJPEG handle.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Returns the nvJPEG handle for its persistent setters.
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    /// Returns the CUDA context of the locked device.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.device.context()
    }
}

/// Locks CUDA device 0 and creates an nvJPEG handle on it.
///
/// # Errors
///
/// Returns an error if device 0 cannot be initialized or locked, its CUDA context cannot be created or bound, or nvJPEG fails to create the handle.
pub fn setup_context() -> Result<TestContext> {
    let device = bootstrap_for_device(Device::new(0))?;
    let context = Context::create(device.context(), &ContextOptions::default())?;
    Ok(TestContext { context, device })
}
