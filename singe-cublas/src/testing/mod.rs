use std::ops::Deref;

use singe_cuda::{context::Context as CudaContext, device::Device, testing::DeviceLock};

use crate::{context::Context, error::Result};

pub struct TestContext {
    context: Context,
    _lock: DeviceLock,
}

impl Deref for TestContext {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

pub fn setup_context() -> Result<TestContext> {
    let device_id: i32 = 0;
    let lock = singe_cuda::testing::device_lock(device_id)?;

    let cuda_context = CudaContext::create_for_device(Device::new(device_id))?;
    let context = Context::create(&cuda_context)?;

    Ok(TestContext {
        context,
        _lock: lock,
    })
}
