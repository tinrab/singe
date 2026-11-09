use std::sync::Arc;

use crate::{
    backend::cuda::{
        allocator::CudaCachingAllocator, context::CudaExecutionContext,
        runtime::runtime_for_device, stream::CudaStream,
    },
    device::Device,
    error::{Error, Result},
};

impl CudaExecutionContext {
    /// Creates an execution context for `device` using CUDA default-stream semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA runtime or allocator cannot be created for `device`.
    pub fn create(device: Device) -> Result<Self> {
        Self::create_with_stream(device, None)
    }

    /// Creates an execution context for `device` using `stream` for stream-ordered work.
    ///
    /// Passing `None` is explicit default-stream behavior.
    /// Passing `Some(stream)` makes eager library calls and allocator reuse event recording follow that stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` is not a CUDA device, `stream` belongs to a different device, or the CUDA runtime or allocator cannot be created.
    pub fn create_with_stream(device: Device, stream: Option<CudaStream>) -> Result<Self> {
        if !device.is_cuda() {
            return Err(Error::device_mismatch(
                "cuda execution context device",
                Device::Cuda(0),
                device,
            ));
        }

        if let Some(stream) = &stream
            && stream.device() != device
        {
            return Err(Error::device_mismatch(
                "cuda execution context stream",
                device,
                stream.device(),
            ));
        }

        let runtime = runtime_for_device(device)?;
        let allocator = Arc::new(CudaCachingAllocator::create(
            runtime.context(),
            stream.as_ref().map(CudaStream::raw_arc),
        )?);
        Ok(Self {
            device,
            stream,
            allocator,
            runtime,
        })
    }
}
