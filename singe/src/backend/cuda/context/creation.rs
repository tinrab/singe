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
    /// Creates an execution context for `device` on the runtime's eager stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA runtime or allocator cannot be created for `device`.
    pub fn create(device: Device) -> Result<Self> {
        Self::create_with_stream(device, None)
    }

    /// Creates an execution context for `device` using `stream` for stream-ordered work.
    ///
    /// Passing `None` runs on the runtime's eager stream.
    /// Passing `Some(stream)` makes library calls, kernels, and allocator reuse follow that stream, which this orders after the eager stream.
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
        let (stream, eager) = match stream {
            Some(stream) => {
                runtime.order_after_eager(stream.raw())?;
                (stream, false)
            }
            None => (runtime.eager_stream()?.clone(), true),
        };
        let allocator = Arc::new(CudaCachingAllocator::create(
            runtime.context(),
            (!eager).then(|| stream.raw_arc()),
        )?);
        Ok(Self {
            device,
            stream,
            eager,
            allocator,
            runtime,
        })
    }
}
