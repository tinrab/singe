//! Serialized GPU test contexts.

use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    testing::{TestDevice, bootstrap_for_device},
};

use crate::{error::Result, plan::Plan};

/// An unconfigured cuFFT plan on a locked test device.
///
/// The plan drops before the device's CUDA context and lock.
#[derive(Debug)]
pub struct TestContext {
    plan: Plan,
    device: TestDevice,
}

impl TestContext {
    /// Returns the plan.
    #[must_use]
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Returns the plan for its persistent setters.
    pub fn plan_mut(&mut self) -> &mut Plan {
        &mut self.plan
    }

    /// Returns the CUDA context of the locked device.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.device.context()
    }
}

/// Locks CUDA device 0, initializes its test context, and creates an unconfigured plan.
///
/// # Errors
///
/// Returns an error if device locking, CUDA initialization, context creation, or native plan creation fails.
pub fn setup_context() -> Result<TestContext> {
    let device = bootstrap_for_device(Device::new(0))?;
    let plan = Plan::create(device.context())?;
    Ok(TestContext { plan, device })
}
