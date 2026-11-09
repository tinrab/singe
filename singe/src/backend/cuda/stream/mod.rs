mod cublas;
mod current;
mod graph;
mod hold;

use std::sync::Arc;

use singe_cuda::stream::{Stream, StreamFlags};

pub use self::{
    cublas::with_cublas_stream,
    current::{
        CURRENT_STREAMS, CudaStreamGuard, current_stream, current_stream_for_device,
        ensure_not_stream_capturing,
    },
    hold::StreamHold,
};
use crate::{
    backend::{cuda, cuda::runtime::runtime_for_device},
    device::Device,
    error::{Error, Result},
};

/// CUDA stream owned by a specific CUDA device.
#[derive(Debug, Clone)]
pub struct CudaStream {
    stream: Arc<Stream>,
}

impl CudaStream {
    /// Creates a CUDA stream with default flags for `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` is not a valid CUDA device or if CUDA cannot create the stream.
    pub fn create(device: Device) -> Result<Self> {
        Self::create_with_flags(device, StreamFlags::DEFAULT)
    }

    /// Creates a CUDA stream with explicit CUDA stream flags for `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` is not a valid CUDA device or if CUDA cannot create the stream with `flags`.
    pub fn create_with_flags(device: Device, flags: StreamFlags) -> Result<Self> {
        let runtime = runtime_for_device(device)?;
        let stream = runtime.context().create_stream_with_flags(flags)?;
        Ok(Self::new(Arc::new(stream)))
    }

    /// Wraps a stream and derives its device from the retained CUDA context.
    #[must_use]
    pub fn new(stream: Arc<Stream>) -> Self {
        Self { stream }
    }

    /// Returns the device ordinal from the stream's retained CUDA context.
    #[must_use]
    pub fn device_id(&self) -> i32 {
        self.stream.context().device().id()
    }

    /// Returns the CUDA device that owns this stream.
    #[must_use]
    pub fn device(&self) -> Device {
        Device::Cuda(self.device_id())
    }

    /// Returns the CUDA stream identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the stream identifier.
    pub fn id(&self) -> Result<u64> {
        self.stream.id().map_err(Error::from)
    }

    /// Blocks until all queued work on this stream completes.
    ///
    /// # Errors
    ///
    /// Returns an error if stream synchronization fails or a previously queued asynchronous operation reports an error.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(Error::from)
    }

    /// Runs `f` with this stream set as current and drains submitted work before returning.
    ///
    /// The draining scope completes on success, error, and unwinding.
    /// The previous current stream is restored after that scope exits.
    /// Graph capture is rejected before invoking `f`.
    ///
    /// # Errors
    ///
    /// Returns an error if setting the current stream fails, `f` returns an error, or synchronization fails.
    /// A synchronization failure takes precedence over the closure's result.
    pub fn scope<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        let _guard = cuda::set_current_stream(self)?;
        self.stream.sync_scope(|_| Ok(f()))?
    }

    /// Borrows the underlying CUDA stream wrapper.
    #[must_use]
    pub fn raw(&self) -> &Stream {
        &self.stream
    }

    /// Clones the shared owner of the underlying CUDA stream.
    #[must_use]
    pub fn raw_arc(&self) -> Arc<Stream> {
        Arc::clone(&self.stream)
    }
}
