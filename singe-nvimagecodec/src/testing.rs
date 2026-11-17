//! Serialized GPU test contexts.

use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    testing::{TestDevice, bootstrap_for_device},
};

use crate::{
    error::Result,
    instance::{Instance, InstanceOptions},
};

/// An nvImageCodec instance on a locked test device.
///
/// The instance drops before the device's CUDA context and lock.
#[derive(Debug)]
pub struct TestContext {
    instance: Instance,
    device: TestDevice,
}

impl TestContext {
    /// Returns the nvImageCodec instance.
    #[must_use]
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    /// Returns the CUDA context of the locked device.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.device.context()
    }
}

/// Locks CUDA device zero, initializes its context, and creates a default nvImageCodec instance.
///
/// # Errors
///
/// Returns an error if device initialization, context creation, device locking, or native instance creation fails.
pub fn setup_context() -> Result<TestContext> {
    let device = bootstrap_for_device(Device::new(0))?;
    let instance = Instance::create(&InstanceOptions::default())?;
    Ok(TestContext { instance, device })
}
