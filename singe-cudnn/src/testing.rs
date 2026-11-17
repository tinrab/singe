//! Serialized GPU test fixtures with an installed cuDNN stream.

use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    stream::{Stream, StreamOptions},
    testing::{TestDevice, bootstrap_for_device},
};

use crate::{context::Context, error::Result};

/// A cuDNN handle with an installed CUDA stream on a locked test device.
///
/// The handle drops first, then the stream, then the device's CUDA context and lock.
#[derive(Debug)]
pub struct TestContext {
    context: Context,
    stream: Stream,
    device: TestDevice,
}

impl TestContext {
    /// Returns the cuDNN handle.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Returns the cuDNN handle for its persistent setters.
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    /// Returns the CUDA stream installed on the cuDNN handle.
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

/// Locks CUDA device zero and creates a cuDNN context with an installed stream.
///
/// # Errors
///
/// Returns an error if test-device initialization or locking, CUDA context creation or binding, stream creation, cuDNN initialization, or stream installation fails.
pub fn setup_context() -> Result<TestContext> {
    let device = bootstrap_for_device(Device::new(0))?;
    let mut context = Context::create(device.context())?;
    let stream = device.context().create_stream(StreamOptions::default())?;
    context.set_stream(Some(&stream))?;
    Ok(TestContext {
        context,
        stream,
        device,
    })
}
