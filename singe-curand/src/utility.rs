use std::{process, sync::Arc};

use singe_core::checked_conversions;
use singe_cuda::{
    context::Context,
    error::{Error as CudaError, Status},
    stream::{Stream, StreamCaptureStatus},
};
use singe_cuda_sys::runtime::{cudaStream_t, cudaStreamSynchronize};
use singe_curand_sys as sys;

use crate::error::{Error, Result};

pub fn require_eager(stream: &Stream) -> Result<()> {
    match stream.capture_status()? {
        StreamCaptureStatus::None => {
            stream.context().bind()?;
            Ok(())
        }
        StreamCaptureStatus::Invalidated => {
            Err(CudaError::from(Status::StreamCaptureInvalidated).into())
        }
        _ => Err(CudaError::from(Status::StreamCaptureUnsupported).into()),
    }
}

checked_conversions!(Error {
    to_size_t -> sys::size_t,
});

/// # Safety
/// The stream must be live and belong to the retained context.
pub unsafe fn drain_stream(context: &Arc<Context>, stream: cudaStream_t) {
    if context.bind().is_err() {
        process::abort();
    }
    // SAFETY: the caller upholds this function's safety contract.
    if unsafe { singe_cuda::try_ffi!(cudaStreamSynchronize(stream)) }.is_err()
        // SAFETY: the caller upholds this function's safety contract.
        && unsafe { singe_cuda::try_ffi!(cudaStreamSynchronize(stream)) }.is_err()
    {
        process::abort();
    }
}

/// Completes native creation that may use the CUDA default stream internally.
pub fn complete_context(context: &Context) -> Result<()> {
    let result = context.synchronize();
    if result.is_err() && context.synchronize().is_err() {
        process::abort();
    }
    result.map_err(Into::into)
}
