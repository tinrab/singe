use std::ops::Deref;

use singe_cuda::{
    stream::Stream,
    testing::{DeviceLock, bootstrap_for_device},
};

use crate::{context::Context, error::Result};

/// A cuDNN handle, configured CUDA stream, and retained device lock for serialized GPU tests.
#[derive(Debug)]
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
    /// Returns the CUDA stream installed on the test cuDNN handle.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

/// Locks CUDA device zero and creates a cuDNN context with an installed stream.
///
/// # Errors
///
/// Returns an error if test-device initialization or locking, CUDA context creation or binding, stream creation, cuDNN initialization, or stream installation fails.
pub fn setup_context() -> Result<TestContext> {
    let device_id: i32 = 0;
    let (lock, cuda_context) = bootstrap_for_device(device_id)?;
    let mut context = Context::create(&cuda_context)?;
    let stream = cuda_context.create_stream()?;
    context.set_stream(Some(&stream))?;

    Ok(TestContext {
        context,
        stream,
        _lock: lock,
    })
}
