use crate::{
    backend::cuda::runtime::{
        CublasLtMatmulAlgorithmCacheStats, CublasLtMatmulAlgorithmEntry,
        CublasLtMatmulAlgorithmKey, CudaRuntime, ProviderCache,
        stats::{
            CublasLtMatmulSupportedAlgorithmCacheStats,
            CublasLtMatmulUnsupportedAlgorithmCacheStats,
            cublaslt_matmul_algorithm_batch_capability_counts,
            cublaslt_matmul_algorithm_batch_counts, cublaslt_matmul_algorithm_compute_counts,
            cublaslt_matmul_algorithm_dimension_maxima, cublaslt_matmul_algorithm_dtype_counts,
            cublaslt_matmul_algorithm_epilogue_counts,
            cublaslt_matmul_algorithm_library_version_count,
            cublaslt_matmul_algorithm_nonzero_workspace_limit_count,
            cublaslt_matmul_algorithm_transpose_counts, cublaslt_matmul_algorithm_workspace_stats,
        },
    },
    error::Result,
    utility::Locked,
};

impl CudaRuntime {
    /// Summarizes the currently retained supported and unsupported matrix multiplication algorithm caches.
    /// Each cache is read under its own mutex, so concurrent changes can occur between the two summaries.
    ///
    /// # Errors
    ///
    /// Returns an error if a lock is poisoned.
    pub fn cublaslt_matmul_algorithm_cache_stats(
        &self,
    ) -> Result<CublasLtMatmulAlgorithmCacheStats> {
        let supported_cache = self
            .cublaslt_matmul_algorithms
            .locked("cuda runtime cache")?;
        let supported = supported_algorithm_cache_stats(&supported_cache);
        drop(supported_cache);

        let unsupported_cache = self
            .unsupported_cublaslt_matmul_algorithms
            .locked("cuda runtime cache")?;
        let unsupported = unsupported_algorithm_cache_stats(&unsupported_cache);
        drop(unsupported_cache);
        Ok(CublasLtMatmulAlgorithmCacheStats::from_cache_parts(
            supported,
            unsupported,
        ))
    }
}

#[expect(
    clippy::similar_names,
    reason = "BLAS cache metrics distinguish standard precision modes and left/right matrix operands"
)]
pub fn supported_algorithm_cache_stats(
    cache: &ProviderCache<CublasLtMatmulAlgorithmKey, CublasLtMatmulAlgorithmEntry>,
) -> CublasLtMatmulSupportedAlgorithmCacheStats {
    let (f16_entries, bf16_entries, f32_entries) =
        cublaslt_matmul_algorithm_dtype_counts(cache.keys());
    let (
        f32_compute_entries,
        tf32_compute_entries,
        f32_fast_f16_compute_entries,
        f32_fast_bf16_compute_entries,
    ) = cublaslt_matmul_algorithm_compute_counts(cache.keys());
    let library_versions = cublaslt_matmul_algorithm_library_version_count(cache.keys());
    let (max_rows, max_inner, max_cols) = cublaslt_matmul_algorithm_dimension_maxima(cache.keys());
    let (
        max_batch_count,
        batched_entries,
        nonzero_batch_stride_entries,
        nonzero_lhs_batch_stride_entries,
        nonzero_rhs_batch_stride_entries,
    ) = cublaslt_matmul_algorithm_batch_counts(cache.keys());
    let (transposed_lhs_entries, transposed_rhs_entries) =
        cublaslt_matmul_algorithm_transpose_counts(cache.keys());
    let nonzero_workspace_limit_entries =
        cublaslt_matmul_algorithm_nonzero_workspace_limit_count(cache.keys());
    let (
        zero_workspace_algorithm_entries,
        nonzero_workspace_algorithm_entries,
        algorithm_workspace_bytes,
        max_algorithm_workspace_bytes,
        algorithm_waves_count,
        max_algorithm_waves_count,
    ) = cublaslt_matmul_algorithm_workspace_stats(cache.values());
    let (strided_batch_algorithm_entries, pointer_array_batch_algorithm_entries) =
        cublaslt_matmul_algorithm_batch_capability_counts(cache.values());
    let (
        default_epilogue_entries,
        bias_epilogue_entries,
        relu_epilogue_entries,
        gelu_epilogue_entries,
    ) = cublaslt_matmul_algorithm_epilogue_counts(cache.keys());
    CublasLtMatmulSupportedAlgorithmCacheStats {
        entries: cache.len(),
        f16_entries,
        bf16_entries,
        f32_entries,
        f32_compute_entries,
        tf32_compute_entries,
        f32_fast_f16_compute_entries,
        f32_fast_bf16_compute_entries,
        library_versions,
        max_rows,
        max_inner,
        max_cols,
        max_batch_count,
        batched_entries,
        nonzero_batch_stride_entries,
        nonzero_lhs_batch_stride_entries,
        nonzero_rhs_batch_stride_entries,
        transposed_lhs_entries,
        transposed_rhs_entries,
        nonzero_workspace_limit_entries,
        zero_workspace_algorithm_entries,
        nonzero_workspace_algorithm_entries,
        algorithm_workspace_bytes,
        max_algorithm_workspace_bytes,
        algorithm_waves_count,
        max_algorithm_waves_count,
        strided_batch_algorithm_entries,
        pointer_array_batch_algorithm_entries,
        default_epilogue_entries,
        bias_epilogue_entries,
        relu_epilogue_entries,
        gelu_epilogue_entries,
    }
}

#[expect(
    clippy::similar_names,
    reason = "BLAS cache metrics distinguish standard precision modes and left/right matrix operands"
)]
pub fn unsupported_algorithm_cache_stats(
    cache: &ProviderCache<CublasLtMatmulAlgorithmKey, ()>,
) -> CublasLtMatmulUnsupportedAlgorithmCacheStats {
    let (f16_entries, bf16_entries, f32_entries) =
        cublaslt_matmul_algorithm_dtype_counts(cache.keys());
    let (
        f32_compute_entries,
        tf32_compute_entries,
        f32_fast_f16_compute_entries,
        f32_fast_bf16_compute_entries,
    ) = cublaslt_matmul_algorithm_compute_counts(cache.keys());
    let library_versions = cublaslt_matmul_algorithm_library_version_count(cache.keys());
    let (max_rows, max_inner, max_cols) = cublaslt_matmul_algorithm_dimension_maxima(cache.keys());
    let (
        max_batch_count,
        batched_entries,
        nonzero_batch_stride_entries,
        nonzero_lhs_batch_stride_entries,
        nonzero_rhs_batch_stride_entries,
    ) = cublaslt_matmul_algorithm_batch_counts(cache.keys());
    let (transposed_lhs_entries, transposed_rhs_entries) =
        cublaslt_matmul_algorithm_transpose_counts(cache.keys());
    let nonzero_workspace_limit_entries =
        cublaslt_matmul_algorithm_nonzero_workspace_limit_count(cache.keys());
    let (
        default_epilogue_entries,
        bias_epilogue_entries,
        relu_epilogue_entries,
        gelu_epilogue_entries,
    ) = cublaslt_matmul_algorithm_epilogue_counts(cache.keys());
    CublasLtMatmulUnsupportedAlgorithmCacheStats {
        entries: cache.len(),
        f16_entries,
        bf16_entries,
        f32_entries,
        f32_compute_entries,
        tf32_compute_entries,
        f32_fast_f16_compute_entries,
        f32_fast_bf16_compute_entries,
        library_versions,
        max_rows,
        max_inner,
        max_cols,
        max_batch_count,
        batched_entries,
        nonzero_batch_stride_entries,
        nonzero_lhs_batch_stride_entries,
        nonzero_rhs_batch_stride_entries,
        transposed_lhs_entries,
        transposed_rhs_entries,
        nonzero_workspace_limit_entries,
        default_epilogue_entries,
        bias_epilogue_entries,
        relu_epilogue_entries,
        gelu_epilogue_entries,
    }
}
