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
/// A context without a current stream runs on the runtime's eager stream and enqueues its work without waiting for it: every eager operation runs in order on that one stream, and a host read waits for it.
/// A context on a caller's stream orders that stream after the eager stream when it is created, and waits for each operation's work.
#[derive(Debug)]
pub struct CudaExecutionContext {
    device: Device,
    stream: CudaStream,
    // Whether `stream` is the runtime's eager stream, whose work nothing waits for.
    eager: bool,
    allocator: Arc<dyn CudaAllocator>,
    runtime: Arc<CudaRuntime>,
}

impl CudaExecutionContext {
    /// Completes borrowed work on this context's explicit CUDA stream, or enqueues it on the eager stream.
    /// On an explicit stream, errors and unwinding drain submitted work before releasing borrows, and stream capture is rejected before invoking the closure.
    /// Eager work needs no drain: storage returns to the cache only for reuse ordered after it.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, `f` fails, or draining submitted CUDA work fails.
    /// Synchronization failure takes precedence over the closure result.
    pub fn sync_scope<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        if self.eager {
            return f();
        }

        StreamBinding::Borrowed(self.stream.raw().to_borrowed()).sync_scope(f)
    }

    /// Returns the CUDA device for this execution context.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns the stream this context's work runs on: the current stream it was created with, or the runtime's eager stream.
    #[must_use]
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    // Whether this context runs on the eager stream, whose work no operation waits for.
    pub(crate) const fn is_eager(&self) -> bool {
        self.eager
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
