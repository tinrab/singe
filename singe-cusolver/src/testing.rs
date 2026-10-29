use std::ops::{Deref, DerefMut};

use singe_cuda::{
    error::{Error as CudaError, Status},
    testing::{DeviceLock, bootstrap_for_device},
};

use crate::{
    context::Context,
    error::{Error, Result},
};

/// A cuSOLVER handle retaining the device lock for serialized GPU tests.
#[derive(Debug)]
pub struct TestContext {
    context: Context,
    _lock: DeviceLock,
}

impl Deref for TestContext {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

impl DerefMut for TestContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.context
    }
}

/// Locks CUDA device zero and creates a cuSOLVER context in its CUDA context.
///
/// # Errors
///
/// Returns an error if device initialization, locking, CUDA context creation or binding, or cuSOLVER handle creation fails.
pub fn setup_context() -> Result<TestContext> {
    let device_id: i32 = 0;
    let (lock, cuda_context) = bootstrap_for_device(device_id)?;
    let context = Context::create(&cuda_context)?;

    Ok(TestContext {
        context,
        _lock: lock,
    })
}

/// Creates a serialized GPU test context, or returns `None` when CUDA reports a stub library.
///
/// # Errors
///
/// Returns errors from [`setup_context`] other than the CUDA stub-library status.
pub fn setup_context_if_available() -> Result<Option<TestContext>> {
    match setup_context() {
        Ok(context) => Ok(Some(context)),
        Err(Error::Cuda(CudaError::Cuda {
            code: Status::StubLibrary,
            ..
        })) => Ok(None),
        Err(error) => Err(error),
    }
}
