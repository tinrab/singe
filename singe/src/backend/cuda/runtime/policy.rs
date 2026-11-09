use crate::{
    backend::cuda::runtime::{CudaRuntime, ProviderCacheLimits, ProviderCacheStats},
    error::Result,
    utility::Locked,
};

/// Policies apply independently to each cache on one device.
/// Active executions and pending submissions retain their own native owners and workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaRuntimeCacheLimits {
    /// Retention limits for cuDNN backend graph templates.
    pub cudnn: ProviderCacheLimits,
    /// Retention limits for cuTENSOR contraction plans.
    pub cutensor: ProviderCacheLimits,
    /// Retention limits for supported cuBLASLt matrix multiplication algorithms.
    pub cublaslt: ProviderCacheLimits,
    /// Retention limits for memoized unsupported cuBLASLt matrix multiplication requests.
    pub unsupported_cublaslt: ProviderCacheLimits,
    /// Retention limits for autotune selection records.
    pub autotune: ProviderCacheLimits,
    /// Retention limits for cuFFT plans.
    pub fft: ProviderCacheLimits,
    /// Retention limits for general CUDA kernel modules.
    pub generated_kernels: ProviderCacheLimits,
}

impl Default for CudaRuntimeCacheLimits {
    fn default() -> Self {
        Self {
            cudnn: ProviderCacheLimits::default(),
            cutensor: ProviderCacheLimits::default(),
            cublaslt: ProviderCacheLimits::default(),
            unsupported_cublaslt: ProviderCacheLimits::default(),
            autotune: ProviderCacheLimits::default(),
            fft: ProviderCacheLimits::default(),
            generated_kernels: ProviderCacheLimits {
                entries: 256,
                accounted_bytes: 8 << 20,
                workspace_bytes: 0,
            },
        }
    }
}

/// Per-provider retained ownership.
/// Native-library allocations are not observable through this API.
/// Workspace totals are requirements, not allocated device bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaRuntimeCacheStats {
    /// Retained ownership and cumulative cache activity for cuDNN backend graph templates.
    pub cudnn: ProviderCacheStats,
    /// Retained ownership and cumulative cache activity for cuTENSOR contraction plans.
    pub cutensor: ProviderCacheStats,
    /// Retained ownership and cumulative cache activity for supported cuBLASLt matrix multiplication algorithms.
    pub cublaslt: ProviderCacheStats,
    /// Retained ownership and cumulative cache activity for memoized unsupported cuBLASLt matrix multiplication requests.
    pub unsupported_cublaslt: ProviderCacheStats,
    /// Retained ownership and cumulative cache activity for autotune selection records.
    pub autotune: ProviderCacheStats,
    /// Retained ownership and cumulative cache activity for cuFFT plans.
    pub fft: ProviderCacheStats,
    /// Retained ownership and cumulative cache activity for general CUDA kernel modules.
    pub generated_kernels: ProviderCacheStats,
}

impl CudaRuntime {
    /// Applies limits under each cache's mutex, evicting immediately.
    /// The caches remain independent; this is not an atomic multi-cache transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if a lock is poisoned.
    pub fn set_provider_cache_limits(&self, limits: CudaRuntimeCacheLimits) -> Result<()> {
        self.cudnn_backend_graphs
            .locked("cuda runtime cache")?
            .set_limits(limits.cudnn);
        self.cutensor_contraction_plans
            .locked("cuda runtime cache")?
            .set_limits(limits.cutensor);
        self.cublaslt_matmul_algorithms
            .locked("cuda runtime cache")?
            .set_limits(limits.cublaslt);
        self.unsupported_cublaslt_matmul_algorithms
            .locked("cuda runtime cache")?
            .set_limits(limits.unsupported_cublaslt);
        self.autotune_cache
            .locked("cuda runtime cache")?
            .set_limits(limits.autotune);
        self.fft_plans
            .locked("cuda runtime cache")?
            .set_limits(limits.fft);
        self.generated_kernels
            .locked("cuda runtime cache")?
            .set_limits(limits.generated_kernels);

        Ok(())
    }

    /// Samples each provider cache under its own mutex.
    ///
    /// The returned statistics are not an atomic snapshot across all caches.
    ///
    /// # Errors
    ///
    /// Returns an error if a lock is poisoned.
    pub fn provider_cache_stats(&self) -> Result<CudaRuntimeCacheStats> {
        Ok(CudaRuntimeCacheStats {
            cudnn: self
                .cudnn_backend_graphs
                .locked("cuda runtime cache")?
                .stats(),
            cutensor: self
                .cutensor_contraction_plans
                .locked("cuda runtime cache")?
                .stats(),
            cublaslt: self
                .cublaslt_matmul_algorithms
                .locked("cuda runtime cache")?
                .stats(),
            unsupported_cublaslt: self
                .unsupported_cublaslt_matmul_algorithms
                .locked("cuda runtime cache")?
                .stats(),
            autotune: self.autotune_cache.locked("cuda runtime cache")?.stats(),
            fft: self.fft_plans.locked("cuda runtime cache")?.stats(),
            generated_kernels: self.generated_kernels.locked("cuda runtime cache")?.stats(),
        })
    }
}
