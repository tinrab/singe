//! GPU test setup.

use std::ops::Deref;

use singe_cuda::testing::{DeviceLock, bootstrap_for_device};

use crate::{decode::Decoder, error::Result};

/// Decoder and device lock held for a GPU test.
#[derive(Debug)]
pub struct TestContext {
    /// Decoder on device 0.
    pub decoder: Decoder,
    _lock: DeviceLock,
}

impl Deref for TestContext {
    type Target = Decoder;

    fn deref(&self) -> &Self::Target {
        &self.decoder
    }
}

/// Locks device 0 for the test and creates a CUDA context, stream, and decoder on it.
///
/// # Errors
///
/// Returns an error if locking the device, initializing CUDA, or creating the context, stream, or decoder fails.
pub fn setup_context() -> Result<TestContext> {
    let device_id: i32 = 0;
    let (lock, cuda_context) = bootstrap_for_device(device_id)?;
    let stream = cuda_context.create_stream()?;
    let decoder = Decoder::create(&stream)?;

    Ok(TestContext {
        decoder,
        _lock: lock,
    })
}
