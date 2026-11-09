use singe_cuda::error::Error;

use crate::{
    backend::cuda::{
        context::CudaExecutionContext,
        runtime::runtime_for_device,
        stream::{CURRENT_STREAMS, CudaStream, CudaStreamGuard, current_stream_for_device},
    },
    device::Device,
    error::Result,
    storage::cuda::cuda_device_id,
};

/// Creates an execution context for `device` using the thread-local current stream.
///
/// If no current stream is set for `device`, the returned context uses CUDA default-stream semantics.
/// Current streams are per-thread and per-device.
///
/// # Errors
///
/// Returns an error if `device` is not a valid CUDA device or runtime initialization fails.
pub fn execution_context(device: Device) -> Result<CudaExecutionContext> {
    CudaExecutionContext::create_with_stream(device, current_stream(device)?)
}

/// Returns the thread-local current stream for `device`, if one is set.
///
/// Current streams are tracked independently per CUDA device.
///
/// # Errors
///
/// Returns an error if `device` is not a valid CUDA device or if the CUDA runtime cannot be initialized for it.
pub fn current_stream(device: Device) -> Result<Option<CudaStream>> {
    let device_id = cuda_device_id(device)?;
    runtime_for_device(device)?;
    Ok(current_stream_for_device(device_id).map(CudaStream::new))
}

/// Sets `stream` as the thread-local current stream for its CUDA device until the returned guard is dropped.
///
/// Current streams for other CUDA devices are not changed.
///
/// # Errors
///
/// Returns an error if the CUDA runtime cannot be initialized for the stream device or if `stream` belongs to another context.
pub fn set_current_stream(stream: &CudaStream) -> Result<CudaStreamGuard> {
    let runtime = runtime_for_device(stream.device())?;
    if stream.raw().context() != runtime.context() {
        return Err(Error::StreamContextMismatch.into());
    }

    let previous = CURRENT_STREAMS.with(|streams| {
        streams
            .borrow_mut()
            .insert(stream.device_id(), stream.raw_arc())
    });
    Ok(CudaStreamGuard::new(stream.device_id(), previous))
}

/// Synchronizes the CUDA context for `device`.
///
/// # Errors
///
/// Returns an error if the CUDA runtime cannot be initialized for `device` or if CUDA context synchronization fails.
pub fn synchronize(device: Device) -> Result<()> {
    runtime_for_device(device)?.context().synchronize()?;
    Ok(())
}
