use std::{fmt::Debug, mem::MaybeUninit, sync::Arc};

use singe_cuda::context::Context as CudaContext;

use crate::{
    backend::cuda::allocator::CudaAllocatorStats,
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    shape::Shape,
    storage::cuda::CudaStorage,
    utility::checked_mul,
};

/// Allocation interface used by CUDA execution contexts.
///
/// Implementations return `CudaStorage` so eager operations, graph execution, and direct backend calls can allocate device memory without depending on a specific allocation strategy.
pub trait CudaAllocator: Debug + Send + Sync {
    /// Returns the actual CUDA context retained by allocated storage.
    fn context(&self) -> &Arc<CudaContext>;

    /// Returns the CUDA device served by this allocator.
    fn device(&self) -> Device;

    /// Allocates representation bytes without producing readable tensor values.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size cannot be represented or the allocator cannot obtain storage in its CUDA context.
    /// Implementations may also report cache or synchronization failures.
    fn allocate_uninit(
        &self,
        data_type: DataType,
        element_count: usize,
    ) -> Result<CudaStorage<MaybeUninit<u8>>>;

    /// Allocates pending representation bytes with shape context for errors.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size cannot be represented or the allocator cannot obtain storage in its CUDA context.
    /// Implementations may also report cache or synchronization failures.
    fn allocate_uninit_with_shape(
        &self,
        data_type: DataType,
        shape: &Shape,
    ) -> Result<CudaStorage<MaybeUninit<u8>>> {
        self.allocate_uninit(data_type, shape.element_count())
    }

    /// Allocates device storage initialized to zero.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size cannot be represented, allocation fails, or initialization cannot complete.
    /// Implementations may also report cache or synchronization failures.
    fn zeroed(&self, data_type: DataType, element_count: usize) -> Result<CudaStorage>;

    /// Allocates device storage initialized to zero with tensor shape context.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size cannot be represented, allocation fails, or initialization cannot complete.
    /// Implementations may also report cache or synchronization failures.
    fn zeroed_with_shape(&self, data_type: DataType, shape: &Shape) -> Result<CudaStorage> {
        self.zeroed(data_type, shape.element_count())
    }

    /// Allocates device storage and copies host bytes into it.
    ///
    /// # Errors
    ///
    /// Returns an error if the host byte length is incompatible with the data type, allocation fails, or the host-to-device copy cannot complete.
    /// Implementations may also report cache or synchronization failures.
    fn allocate_from_bytes(&self, data_type: DataType, data: &[u8]) -> Result<CudaStorage>;

    /// Allocates device storage and copies host bytes into it with tensor shape context.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape byte count overflows, `data` has a different byte length, or allocation or copying fails.
    /// Implementations may also report cache or synchronization failures.
    fn allocate_from_bytes_with_shape(
        &self,
        data_type: DataType,
        shape: &Shape,
        data: &[u8],
    ) -> Result<CudaStorage> {
        let expected = checked_mul(
            shape.element_count(),
            data_type.size_bytes(),
            "cuda allocation host byte length",
        )?;
        if data.len() != expected {
            return Err(Error::length_mismatch(
                "cuda allocation host byte length",
                expected,
                data.len(),
            ));
        }
        self.allocate_from_bytes(data_type, data)
    }

    /// Returns allocator statistics when the implementation tracks them.
    ///
    /// # Errors
    ///
    /// Implementations can return an error if allocator statistics cannot be read.
    /// The default implementation returns zeroed statistics.
    fn stats(&self) -> Result<CudaAllocatorStats> {
        Ok(CudaAllocatorStats::default())
    }

    /// Releases reusable cached blocks held by the allocator.
    ///
    /// # Errors
    ///
    /// Implementations can return an error if cached blocks cannot be released.
    /// The default implementation succeeds without work.
    fn clear_cache(&self) -> Result<()> {
        Ok(())
    }

    /// Restarts peak tracking at the bytes currently active.
    ///
    /// The default implementation keeps no statistics and does nothing.
    fn reset_peak(&self) {}

    /// Caps active plus cached bytes at `limit`, or removes the cap for `None`.
    ///
    /// The default implementation ignores the cap.
    fn set_memory_limit(&self, _limit: Option<usize>) {}
}
