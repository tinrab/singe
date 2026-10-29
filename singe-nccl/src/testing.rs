use std::ops::Deref;

use singe_cuda::{
    context::Context,
    device::Device,
    stream::Stream,
    testing::{DeviceLock, device_lock},
};

use crate::{communicator::Communicator, error::Result, unique_id};

/// A communicator and CUDA stream retaining the serialized GPU test lock.
#[derive(Debug)]
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
    /// Returns the CUDA stream used by the test communicator.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Consumes the test setup and returns its communicator.
    pub fn into_communicator(self) -> Communicator {
        self.communicator
    }
}

/// Creates a single-rank communicator and CUDA stream on device 0.
///
/// # Errors
///
/// Returns an error if retaining the primary CUDA context, creating the stream, generating a unique identifier, or initializing NCCL fails.
pub fn setup_communicator() -> Result<TestContext> {
    let device_id: i32 = 0;
    let lock = device_lock(device_id)?;
    let cuda_context = Context::retain_primary_for_device(Device::new(device_id))?;
    let stream = cuda_context.create_stream()?;
    let communicator = Communicator::create_rank(&cuda_context, 1, unique_id()?, 0)?;

    Ok(TestContext {
        communicator,
        stream,
        _lock: lock,
    })
}
