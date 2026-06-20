use std::ops::Deref;

use singe_cuda::{stream::Stream, testing::DeviceLock};

use crate::{context::Context, error::Result};

pub struct TestContext {
    context: Context,
    stream: Stream,
    _lock: DeviceLock,
}

impl Deref for TestContext {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

impl TestContext {
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

pub fn setup_context() -> Result<TestContext> {
    let device_id: i32 = 0;
    let (lock, cuda_context) = singe_cuda::testing::bootstrap_for_device(device_id)?;
    let context = Context::create(&cuda_context)?;
    let stream = cuda_context.create_stream()?;
    context.set_stream(Some(&stream))?;

    Ok(TestContext {
        context,
        stream,
        _lock: lock,
    })
}
