mod creation;
mod libraries;
mod numerics;
mod tensor;

use std::sync::Arc;

use singe_cuda::stream::StreamBinding;

pub use self::numerics::with_cublas_numerics;
use crate::{
    backend::cuda::{
        allocator::{CudaAllocator, CudaAllocatorStats},
        runtime::CudaRuntime,
        stream::CudaStream,
    },
    device::Device,
    error::Result,
};

/// CUDA execution context bound to a device, stream policy, and allocator.
///
/// The Singe backend keeps one cuBLAS, cuDNN, and cuTENSOR handle per CUDA runtime/device behind mutexes.
/// cuBLAS and cuDNN operations bind the handle to this context's stream for the duration of the call.
/// cuDNN restores its previous stream afterward, including on errors and unwinding; cuBLAS resets its stream.
/// cuTENSOR operations pass the context stream at launch time.
///
/// A `None` stream means operations use CUDA's legacy/default stream semantics for cuBLAS and cuDNN.
/// cuTENSOR requires an explicit stream at launch time, so `with_cutensor` creates a temporary backend-owned stream when this context has no current stream and synchronizes only that temporary stream before returning.
#[derive(Debug)]
pub struct CudaExecutionContext {
    device: Device,
    stream: Option<CudaStream>,
    allocator: Arc<dyn CudaAllocator>,
    runtime: Arc<CudaRuntime>,
}

impl CudaExecutionContext {
    /// Completes borrowed work on this context's explicit or default CUDA stream.
    /// Errors and unwinding drain submitted work before releasing borrows.
    /// Stream capture is rejected before invoking the closure.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, `f` fails, or draining submitted CUDA work fails.
    /// Synchronization failure takes precedence over the closure result.
    pub fn sync_scope<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        let binding = self.stream().map_or_else(
            || StreamBinding::Default(Arc::clone(self.allocator().context())),
            |stream| StreamBinding::Borrowed(stream.raw().to_borrowed()),
        );
        binding.sync_scope(f)
    }

    /// Returns the CUDA device for this execution context.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns the explicit stream associated with this execution context, if any.
    ///
    /// `None` means this context uses CUDA default-stream semantics except for cuTENSOR launches, where Singe creates and synchronizes a temporary backend-owned stream.
    #[must_use]
    pub fn stream(&self) -> Option<&CudaStream> {
        self.stream.as_ref()
    }

    /// Returns the allocator used by this execution context.
    #[must_use]
    pub fn allocator(&self) -> &dyn CudaAllocator {
        self.allocator.as_ref()
    }

    /// Returns allocator statistics for this execution context.
    ///
    /// # Errors
    ///
    /// Returns an error if allocator statistics cannot be collected.
    pub fn allocator_stats(&self) -> Result<CudaAllocatorStats> {
        self.allocator.stats()
    }

    /// Restarts the allocator's peak-byte tracking.
    pub(crate) fn reset_peak_memory(&self) {
        self.allocator.reset_peak();
    }

    /// Caps the bytes the allocator holds, or removes the cap for `None`.
    pub(crate) fn set_memory_limit(&self, limit: Option<usize>) {
        self.allocator.set_memory_limit(limit);
    }

    /// Clears cached free allocator blocks for this execution context.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocator cache cannot be cleared.
    pub fn clear_cache(&self) -> Result<()> {
        self.allocator.clear_cache()
    }

    pub(crate) fn bind(&self) -> Result<()> {
        self.runtime.context().bind()?;
        Ok(())
    }

    pub(crate) fn runtime(&self) -> &Arc<CudaRuntime> {
        &self.runtime
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests;
