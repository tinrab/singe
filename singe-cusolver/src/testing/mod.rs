use std::ops::Deref;

use singe_cuda::{error::Status, testing::DeviceLock};

use crate::{
    context::Context,
    error::{Error, Result},
};

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
    let (lock, cuda_context) = singe_cuda::testing::bootstrap_for_device(device_id)?;
    let context = Context::create(&cuda_context)?;

    Ok(TestContext {
        context,
        _lock: lock,
    })
}

pub fn setup_context_if_available() -> Result<Option<TestContext>> {
    match setup_context() {
        Ok(ctx) => Ok(Some(ctx)),
        Err(Error::Cuda(singe_cuda::error::Error::Cuda {
            code: Status::StubLibrary,
            ..
        })) => Ok(None),
        Err(error) => Err(error),
    }
}
