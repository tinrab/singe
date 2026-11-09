use std::sync::Mutex;

use crate::{
    backend::cuda::runtime::{
        CublasLtMatmulAlgorithmEntry, CublasLtMatmulAlgorithmKey, CudaRuntime, ProviderCache,
        autotune::AutotuneCacheKey,
        cublaslt::best_cublaslt_matmul_algorithm_entry_with_workspace_limit,
    },
    error::Result,
    utility::Locked,
};

impl CudaRuntime {
    /// Returns the mutex-protected cache of selected cuBLASLt matrix multiplication algorithms.
    pub fn cublaslt_matmul_algorithms(
        &self,
    ) -> &Mutex<ProviderCache<CublasLtMatmulAlgorithmKey, CublasLtMatmulAlgorithmEntry>> {
        &self.cublaslt_matmul_algorithms
    }

    /// Clones the preferred cached algorithm whose key matches the predicate and workspace limit.
    /// Eligible keys have a recorded workspace limit no greater than `workspace_limit_bytes`.
    /// Preference uses wave count, then required workspace bytes.
    /// Returns `None` when no key qualifies.
    ///
    /// # Errors
    ///
    /// Returns an error if a lock is poisoned.
    pub fn best_cached_cublaslt_matmul_algorithm(
        &self,
        workspace_limit_bytes: usize,
        matches_key: impl Fn(&CublasLtMatmulAlgorithmKey) -> bool,
    ) -> Result<Option<CublasLtMatmulAlgorithmEntry>> {
        let cache = self
            .cublaslt_matmul_algorithms
            .locked("cuda runtime cache")?;
        Ok(best_cublaslt_matmul_algorithm_entry_with_workspace_limit(
            cache
                .iter()
                .filter_map(|(key, entry)| matches_key(key).then_some((key, entry))),
            workspace_limit_bytes,
        )
        .cloned())
    }

    /// Clones the preferred cached algorithm in the same matrix multiplication family as `key`.
    /// Candidate keys must have a recorded workspace limit no greater than `workspace_limit_bytes`.
    /// Returns `None` when no key qualifies.
    ///
    /// # Errors
    ///
    /// Returns an error if a lock is poisoned.
    pub fn best_cached_cublaslt_matmul_algorithm_for_key(
        &self,
        key: &CublasLtMatmulAlgorithmKey,
        workspace_limit_bytes: usize,
    ) -> Result<Option<CublasLtMatmulAlgorithmEntry>> {
        self.best_cached_cublaslt_matmul_algorithm(workspace_limit_bytes, |cached_key| {
            cached_key.matches_matmul_algorithm_family(key)
        })
    }

    /// Finds a cached algorithm matching the retained cuBLASLt tuning result.
    /// The algorithm must match the matrix multiplication family and the recorded workspace requirement.
    /// Recorded compute types and fingerprints further restrict the match when present.
    /// An autotune key with no workspace limit permits only algorithm keys with a zero limit.
    /// Returns `None` if the tuning or a matching algorithm is absent.
    ///
    /// # Errors
    ///
    /// Returns an error if a lock is poisoned.
    pub fn tuned_cublaslt_matmul_algorithm_for_autotune_key(
        &self,
        autotune_key: &AutotuneCacheKey,
        algorithm_key: &CublasLtMatmulAlgorithmKey,
    ) -> Result<Option<CublasLtMatmulAlgorithmEntry>> {
        let Some(autotune_entry) = self
            .autotune_cache
            .locked("cuda runtime cache")?
            .get(autotune_key)
            .cloned()
        else {
            return Ok(None);
        };
        if let Some(compute_type) = autotune_entry.compute_type()
            && algorithm_key.compute_type() != compute_type
        {
            return Ok(None);
        }

        let algorithm_cache = self
            .cublaslt_matmul_algorithms
            .locked("cuda runtime cache")?;
        Ok(algorithm_cache
            .iter()
            .find(|(cached_key, entry)| {
                cached_key.matches_matmul_algorithm_family(algorithm_key)
                    && cached_key.workspace_limit_bytes()
                        <= autotune_key.workspace_limit_bytes().unwrap_or(0)
                    && entry.workspace_bytes() == autotune_entry.workspace_bytes()
                    && autotune_entry
                        .compute_type()
                        .is_none_or(|compute_type| entry.compute_type() == compute_type)
                    && autotune_entry
                        .cublaslt_algorithm_fingerprint()
                        .is_none_or(|fingerprint| entry.fingerprint() == Some(fingerprint))
            })
            .map(|(_, entry)| entry.clone()))
    }

    /// Returns the mutex-protected cache of unsupported matrix multiplication keys.
    pub fn unsupported_cublaslt_matmul_algorithms(
        &self,
    ) -> &Mutex<ProviderCache<CublasLtMatmulAlgorithmKey, ()>> {
        &self.unsupported_cublaslt_matmul_algorithms
    }
}
