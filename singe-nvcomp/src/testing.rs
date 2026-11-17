//! Serialized GPU test contexts.

use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext,
    stream::{Stream, StreamOptions},
    testing::{TestDevice, bootstrap},
};

use crate::error::Result;

/// A CUDA stream on a locked test device.
///
/// The stream drops before the device's CUDA context and lock.
#[derive(Debug)]
pub struct TestContext {
    stream: Stream,
    device: TestDevice,
}

impl TestContext {
    /// Returns the stream.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Returns the CUDA context of the locked device.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.device.context()
    }
}

/// Locks CUDA device 0 and creates a stream on its CUDA context.
///
/// # Errors
///
/// Returns an error if locking the device, initializing CUDA, or creating the context or stream fails.
pub fn setup_context() -> Result<TestContext> {
    let device = bootstrap()?;
    let stream = device.context().create_stream(StreamOptions::default())?;
    Ok(TestContext { stream, device })
}
