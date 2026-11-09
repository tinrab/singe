use singe_cublas::lt::types::Epilogue;
use singe_cuda::stream::StreamCaptureStatus;

use crate::{
    backend::cuda::{
        self, CublasLtMatmulTuningOptions, CublasLtMatmulTuningResult,
        allocator::{CudaAllocatorStats, CudaCachingAllocator},
        compiled::plan::cublas::tune_cublaslt_matmul_algorithm_with_workspace_limit,
        runtime::{
            AutotuneCacheStats, CublasLtMatmulAlgorithmCacheStats, CudaRuntimeCacheStats,
            runtime_for_device,
        },
        stream::CudaStream,
    },
    device::Device,
    error::{Error, Result},
    tensor::Tensor,
};

/// Clears cached CUDA backend state and free allocator blocks for `device`.
///
/// # Errors
///
/// Returns an error if the CUDA runtime cannot be initialized for `device` or if clearing allocator state fails.
pub fn clear_cache(device: Device) -> Result<()> {
    let runtime = runtime_for_device(device)?;
    runtime.clear_cache()?;
    CudaCachingAllocator::clear_cache_for_device(device)
}

/// Returns cuBLASLt matmul algorithm cache entry counts for `device`.
///
/// # Errors
///
/// Returns an error if the CUDA runtime cannot be initialized for `device` or if the algorithm cache state cannot be inspected.
pub fn cublaslt_matmul_algorithm_cache_stats(
    device: Device,
) -> Result<CublasLtMatmulAlgorithmCacheStats> {
    let runtime = runtime_for_device(device)?;
    runtime.cublaslt_matmul_algorithm_cache_stats()
}

/// Returns generic CUDA autotune cache entry counts for `device`.
///
/// # Errors
///
/// Returns an error if the CUDA runtime cannot be initialized for `device` or if the autotune cache state cannot be inspected.
pub fn autotune_cache_stats(device: Device) -> Result<AutotuneCacheStats> {
    let runtime = runtime_for_device(device)?;
    runtime.autotune_cache_stats()
}

/// Returns the retained entries and cumulative activity of every bounded provider cache for `device`.
///
/// # Errors
///
/// Returns an error if the CUDA runtime cannot be initialized for `device` or a cache cannot be inspected.
pub fn provider_cache_stats(device: Device) -> Result<CudaRuntimeCacheStats> {
    runtime_for_device(device)?.provider_cache_stats()
}

// Measures `a * b` with each cuBLASLt algorithm cached for it and keeps the fastest, which later lowering of the same matmul uses.
pub fn tune_cublaslt_matmul(
    a: &Tensor,
    b: &Tensor,
    output: &Tensor,
    epilogue: Epilogue,
    bias: Option<&Tensor>,
    workspace_limit_bytes: usize,
    config: CublasLtMatmulTuningOptions,
    stream: &CudaStream,
) -> Result<Option<CublasLtMatmulTuningResult>> {
    if stream.capture_status()? == StreamCaptureStatus::Active {
        return Err(Error::unsupported_operation(
            "cuBLASLt tuned matmul warmup during stream capture",
        ));
    }

    let runtime = runtime_for_device(output.device())?;
    tune_cublaslt_matmul_algorithm_with_workspace_limit(
        &runtime,
        a,
        b,
        output,
        epilogue,
        bias,
        workspace_limit_bytes,
        config,
        stream,
    )
}

/// Returns per-device caching allocator counters for CUDA tensor allocations.
///
/// # Errors
///
/// Returns an error if a CUDA execution context cannot be created or if allocator statistics cannot be collected.
pub fn allocator_stats(device: Device) -> Result<CudaAllocatorStats> {
    let context = cuda::execution_context(device)?;
    context.allocator_stats()
}

/// Restarts the allocator's peak-byte tracking for `device` at its currently active bytes.
///
/// # Errors
///
/// Returns an error if a CUDA execution context cannot be created for `device`.
pub fn reset_peak_memory(device: Device) -> Result<()> {
    cuda::execution_context(device)?.reset_peak_memory();
    Ok(())
}

/// Caps the bytes the allocator holds for `device`, live and cached, or removes the cap for `None`.
///
/// An allocation that would exceed the cap releases cached blocks first, then fails with [`Error::MemoryLimitExceeded`](crate::error::Error::MemoryLimitExceeded).
///
/// # Errors
///
/// Returns an error if a CUDA execution context cannot be created for `device`.
pub fn set_memory_limit(device: Device, limit: Option<usize>) -> Result<()> {
    cuda::execution_context(device)?.set_memory_limit(limit);
    Ok(())
}
