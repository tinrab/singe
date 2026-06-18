use std::ops::Deref;

use singe_cuda::{context::Context as CudaContext, device::Device, testing::DeviceLock};

use crate::{error::Result, plan::Plan};

pub struct TestContext {
    #[allow(dead_code)]
    pub plan: Plan,
    _lock: DeviceLock,
}

impl Deref for TestContext {
    type Target = Plan;

    fn deref(&self) -> &Self::Target {
        &self.plan
    }
}

pub fn setup_context() -> Result<TestContext> {
    let device_id: i32 = 0;
    let lock = singe_cuda::testing::device_lock(device_id)?;

    let cuda_context = CudaContext::create_for_device(Device::new(device_id))?;
    let plan = Plan::create(&cuda_context)?;

    Ok(TestContext { plan, _lock: lock })
}
