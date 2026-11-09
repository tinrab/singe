use std::sync::Mutex;

use crate::{
    backend::cuda::{
        cudnn_backend::{CudnnBackendGraphCacheEntry, CudnnBackendGraphKey},
        cutensor::{CachedCutensorContractionPlan, CutensorContractionPlanKey},
        runtime::{
            AutotuneCacheEntry, AutotuneCacheKey, AutotuneCacheStats, CudaRuntime, ProviderCache,
        },
    },
    error::Result,
    utility::Locked,
};

impl CudaRuntime {
    /// Returns the mutex-protected cuDNN graph template cache.
    pub fn cudnn_backend_graphs(
        &self,
    ) -> &Mutex<ProviderCache<CudnnBackendGraphKey, CudnnBackendGraphCacheEntry>> {
        &self.cudnn_backend_graphs
    }

    /// Returns the mutex-protected cuTENSOR contraction plan cache.
    pub fn cutensor_contraction_plans(
        &self,
    ) -> &Mutex<ProviderCache<CutensorContractionPlanKey, CachedCutensorContractionPlan>> {
        &self.cutensor_contraction_plans
    }

    /// Summarizes currently retained autotune entries.
    ///
    /// # Errors
    ///
    /// Returns an error if a lock is poisoned.
    pub fn autotune_cache_stats(&self) -> Result<AutotuneCacheStats> {
        let cache = self.autotune_cache.locked("cuda runtime cache")?;
        Ok(AutotuneCacheStats::from_entries(cache.values()))
    }

    #[cfg(test)]
    pub fn autotune_cache(&self) -> &Mutex<ProviderCache<AutotuneCacheKey, AutotuneCacheEntry>> {
        &self.autotune_cache
    }

    pub fn insert_autotune_cache_entry(
        &self,
        key: AutotuneCacheKey,
        entry: AutotuneCacheEntry,
    ) -> Result<Option<AutotuneCacheEntry>> {
        let mut cache = self.autotune_cache.locked("cuda runtime cache")?;
        Ok(cache.insert(key, entry))
    }

    /// Clears kernel, provider, algorithm, unsupported-algorithm, and autotune cache entries.
    /// Resources retained by compiled executables remain alive.
    ///
    /// # Errors
    ///
    /// Returns an error if a lock is poisoned.
    pub fn clear_cache(&self) -> Result<()> {
        self.generated_kernels.locked("cuda runtime cache")?.clear();

        self.cudnn_backend_graphs
            .locked("cuda runtime cache")?
            .clear();

        self.cutensor_contraction_plans
            .locked("cuda runtime cache")?
            .clear();

        self.cublaslt_matmul_algorithms
            .locked("cuda runtime cache")?
            .clear();

        self.unsupported_cublaslt_matmul_algorithms
            .locked("cuda runtime cache")?
            .clear();

        self.autotune_cache.locked("cuda runtime cache")?.clear();

        self.fft_plans.locked("cuda runtime cache")?.clear();

        Ok(())
    }
}
