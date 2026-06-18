use std::ops::Deref;

use singe_cuda::{context::Context as CudaContext, stream::Stream, testing::DeviceLock};

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
    let lock = singe_cuda::testing::device_lock(device_id)?;

    let cuda_context = CudaContext::create()?;
    let context = Context::create(&cuda_context)?;
    let stream = cuda_context.create_stream()?;
    context.set_stream(Some(&stream))?;

    Ok(TestContext {
        context,
        stream,
        _lock: lock,
    })
}
