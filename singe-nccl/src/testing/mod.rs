use std::ops::Deref;

use singe_cuda::{context::Context as CudaContext, stream::Stream, testing::DeviceLock};

use crate::{communicator::Communicator, error::Result, unique_id};

pub struct TestContext {
    communicator: Communicator,
    stream: Stream,
    _lock: DeviceLock,
}

impl Deref for TestContext {
    type Target = Communicator;

    fn deref(&self) -> &Self::Target {
        &self.communicator
    }
}

impl TestContext {
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn into_communicator(self) -> Communicator {
        self.communicator
    }
}

pub fn setup_communicator() -> Result<TestContext> {
    let device_id: i32 = 0;
    let lock = singe_cuda::testing::device_lock(device_id)?;

    let cuda_context = CudaContext::create()?;
    let stream = cuda_context.create_stream()?;
    let communicator = Communicator::create_rank(&cuda_context, 1, unique_id()?, 0)?;

    Ok(TestContext {
        communicator,
        stream,
        _lock: lock,
    })
}
