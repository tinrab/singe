//! Serialized GPU test contexts.

use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    stream::{Stream, StreamOptions},
    testing::{TestDevice, bootstrap_for_device},
};

use crate::{allocator::AllocatorSet, decode::Decoder, error::Result};

/// An nvTIFF decoder and its stream on a locked test device.
///
/// The decoder drops first, then the stream, then the device's CUDA context and lock.
#[derive(Debug)]
pub struct TestContext {
    decoder: Decoder,
    stream: Stream,
    device: TestDevice,
}

impl TestContext {
    /// Returns the decoder.
    #[must_use]
    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }

    /// Returns the stream the decoder was created on.
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

/// Locks device 0 for the test and creates a CUDA stream and decoder on it.
///
/// # Errors
///
/// Returns an error if locking the device, initializing CUDA, or creating the context, stream, or decoder fails.
pub fn setup_context() -> Result<TestContext> {
    let device = bootstrap_for_device(Device::new(0))?;
    let stream = device.context().create_stream(StreamOptions::default())?;
    let decoder = Decoder::create(AllocatorSet::new(), &stream)?;
    Ok(TestContext {
        decoder,
        stream,
        device,
    })
}
