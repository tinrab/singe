use std::ops::Deref;

use singe_cuda::testing::DeviceLock;

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
    let (lock, cuda_context) = singe_cuda::testing::bootstrap_for_device(device_id)?;
    let plan = Plan::create(&cuda_context)?;

    Ok(TestContext { plan, _lock: lock })
}
