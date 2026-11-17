//! Serialized GPU test contexts.

use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    stream::{Stream, StreamOptions},
    testing::{TestDevice, bootstrap_for_device},
};

use crate::{communicator::Communicator, error::Result, types::RankId, unique_id};

/// A single-rank communicator and CUDA stream on a locked test device.
///
/// The communicator drops first, then the stream, then the device's CUDA context and lock.
#[derive(Debug)]
pub struct TestContext {
    communicator: Communicator,
    stream: Stream,
    device: TestDevice,
}

impl TestContext {
    /// Returns the test communicator.
    #[must_use]
    pub fn communicator(&self) -> &Communicator {
        &self.communicator
    }

    /// Returns the CUDA stream for test operations.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Returns the CUDA context of the locked device.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.device.context()
    }

    /// Splits the setup into its communicator and the locked device, dropping the stream.
    #[must_use]
    pub fn into_parts(self) -> (Communicator, TestDevice) {
        (self.communicator, self.device)
    }
}

/// Creates a single-rank communicator and CUDA stream on device 0.
///
/// # Errors
///
/// Returns an error if locking the device, retaining its primary CUDA context, creating the stream, generating a unique identifier, or initializing NCCL fails.
pub fn setup_communicator() -> Result<TestContext> {
    let device = bootstrap_for_device(Device::new(0))?;
    let stream = device.context().create_stream(StreamOptions::default())?;
    let communicator = Communicator::create_rank(
        device.context(),
        RankId::create(0)?,
        1,
        &[unique_id()?],
        None,
    )?;
    Ok(TestContext {
        communicator,
        stream,
        device,
    })
}
