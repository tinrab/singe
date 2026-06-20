use std::ops::Deref;

use singe_cuda::{stream::Stream, testing::DeviceLock};

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
    let (lock, cuda_context) = singe_cuda::testing::bootstrap_for_device(device_id)?;
    let stream = cuda_context.create_stream()?;
    let communicator = Communicator::create_rank(&cuda_context, 1, unique_id()?, 0)?;

    Ok(TestContext {
        communicator,
        stream,
        _lock: lock,
    })
}
