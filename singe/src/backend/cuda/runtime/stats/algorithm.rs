use num_traits::ToPrimitive;

macro_rules! supported_algorithm_stat_accessors {
    ($($(#[doc = $doc:literal])+ $method:ident -> $field:ident: $return_type:ty),+ $(,)?) => {
        $(
            $(#[doc = $doc])+
            pub fn $method(&self) -> $return_type {
                self.supported.$field
            }
        )+
    };
}

impl CublasLtMatmulAlgorithmCacheStats {
    supported_algorithm_stat_accessors! {
        /// For supported entries, returns the number of cached entries.
        supported_entries -> entries: usize,
        /// For supported entries, returns the number of entries with `F16` tensor data.
        supported_f16_entries -> f16_entries: usize,
        /// For supported entries, returns the number of entries with `Bf16` tensor data.
        supported_bf16_entries -> bf16_entries: usize,
        /// For supported entries, returns the number of entries with `F32` tensor data.
        supported_f32_entries -> f32_entries: usize,
        /// For supported entries, returns the number of entries using the `F32` compute mode.
        supported_f32_compute_entries -> f32_compute_entries: usize,
        /// For supported entries, returns the number of entries using the `F32FastTf32` compute mode.
        supported_tf32_compute_entries -> tf32_compute_entries: usize,
        /// For supported entries, returns the number of entries using the `F32FastF16` compute mode.
        supported_f32_fast_f16_compute_entries -> f32_fast_f16_compute_entries: usize,
        /// For supported entries, returns the number of entries using the `F32FastBf16` compute mode.
        supported_f32_fast_bf16_compute_entries -> f32_fast_bf16_compute_entries: usize,
        /// For supported entries, returns the number of distinct cuBLASLt library versions in cache keys.
        supported_library_versions -> library_versions: usize,
        /// For supported entries, returns the largest matrix row count in cache keys, or zero for an empty cache.
        supported_max_rows -> max_rows: u64,
        /// For supported entries, returns the largest contracted matrix extent in cache keys, or zero for an empty cache.
        supported_max_inner -> max_inner: u64,
        /// For supported entries, returns the largest matrix column count in cache keys, or zero for an empty cache.
        supported_max_cols -> max_cols: u64,
        /// For supported entries, returns the largest batch count in cache keys, or zero for an empty cache.
        supported_max_batch_count -> max_batch_count: usize,
        /// For supported entries, returns the number of entries with batch count greater than one.
        supported_batched_entries -> batched_entries: usize,
        /// For supported entries, returns the number of entries with a nonzero batch offset for either input.
        supported_nonzero_batch_stride_entries -> nonzero_batch_stride_entries: usize,
        /// For supported entries, returns the number of entries with a nonzero left-input batch offset.
        supported_nonzero_lhs_batch_stride_entries -> nonzero_lhs_batch_stride_entries: usize,
        /// For supported entries, returns the number of entries with a nonzero right-input batch offset.
        supported_nonzero_rhs_batch_stride_entries -> nonzero_rhs_batch_stride_entries: usize,
        /// For supported entries, returns the number of entries that transpose the left input.
        supported_transposed_lhs_entries -> transposed_lhs_entries: usize,
        /// For supported entries, returns the number of entries that transpose the right input.
        supported_transposed_rhs_entries -> transposed_rhs_entries: usize,
        /// For supported entries, returns the number of entries whose cache key allows nonzero workspace.
        supported_nonzero_workspace_limit_entries -> nonzero_workspace_limit_entries: usize,
        /// For supported entries, returns the number of selected algorithms requiring no workspace.
        supported_zero_workspace_algorithm_entries -> zero_workspace_algorithm_entries: usize,
        /// For supported entries, returns the number of selected algorithms requiring nonzero workspace.
        supported_nonzero_workspace_algorithm_entries -> nonzero_workspace_algorithm_entries: usize,
        /// For supported entries, returns the saturated sum of selected algorithm workspace requirements in bytes.
        supported_algorithm_workspace_bytes -> algorithm_workspace_bytes: usize,
        /// For supported entries, returns the largest selected algorithm workspace requirement in bytes, or zero for an empty cache.
        supported_max_algorithm_workspace_bytes -> max_algorithm_workspace_bytes: usize,
        /// For supported entries, returns the sum of selected algorithms' reported waves counts.
        supported_algorithm_waves_count -> algorithm_waves_count: f32,
        /// For supported entries, returns the largest selected algorithm waves count, starting from zero for an empty cache.
        supported_max_algorithm_waves_count -> max_algorithm_waves_count: f32,
        /// For supported entries, returns the number of selected algorithms reporting strided-batch support.
        supported_strided_batch_algorithm_entries -> strided_batch_algorithm_entries: usize,
        /// For supported entries, returns the number of selected algorithms reporting pointer-array batch support.
        supported_pointer_array_batch_algorithm_entries -> pointer_array_batch_algorithm_entries: usize,
        /// For supported entries, returns the number of keys using the default epilogue.
        supported_default_epilogue_entries -> default_epilogue_entries: usize,
        /// For supported entries, returns the number of keys using `Bias`, `ReluBias`, `ReluAuxBias`, `GeluBias`, or `GeluAuxBias` epilogues.
        supported_bias_epilogue_entries -> bias_epilogue_entries: usize,
        /// For supported entries, returns the number of keys using the standalone `Relu` epilogue, excluding fused bias variants.
        supported_relu_epilogue_entries -> relu_epilogue_entries: usize,
        /// For supported entries, returns the number of keys using the standalone GELU epilogue, excluding fused bias variants.
        supported_gelu_epilogue_entries -> gelu_epilogue_entries: usize,
    }

    /// Returns the sum of supported algorithm waves counts divided by the supported entry count.
    ///
    /// Returns `None` for an empty supported cache or if converting the count to `f32` fails.
    #[must_use]
    pub fn supported_average_algorithm_waves_count(&self) -> Option<f32> {
        if self.supported.entries == 0 {
            None
        } else {
            Some(self.supported.algorithm_waves_count / self.supported.entries.to_f32()?)
        }
    }
}

/// Snapshot of supported algorithm selections and cached unsupported cuBLASLt matrix multiplication requests.
///
/// Counts describe cached requests, not execution frequency.
/// Returned workspace totals sum per-entry requirements rather than measuring simultaneous allocations.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CublasLtMatmulAlgorithmCacheStats {
    pub(crate) supported: CublasLtMatmulSupportedAlgorithmCacheStats,
    pub(crate) unsupported: CublasLtMatmulUnsupportedAlgorithmCacheStats,
}

/// Aggregated key properties and selected algorithm metadata for supported cached matrix multiplication requests.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CublasLtMatmulSupportedAlgorithmCacheStats {
    pub(crate) entries: usize,
    pub(crate) f16_entries: usize,
    pub(crate) bf16_entries: usize,
    pub(crate) f32_entries: usize,
    pub(crate) f32_compute_entries: usize,
    pub(crate) tf32_compute_entries: usize,
    pub(crate) f32_fast_f16_compute_entries: usize,
    pub(crate) f32_fast_bf16_compute_entries: usize,
    pub(crate) library_versions: usize,
    pub(crate) max_rows: u64,
    pub(crate) max_inner: u64,
    pub(crate) max_cols: u64,
    pub(crate) max_batch_count: usize,
    pub(crate) batched_entries: usize,
    pub(crate) nonzero_batch_stride_entries: usize,
    pub(crate) nonzero_lhs_batch_stride_entries: usize,
    pub(crate) nonzero_rhs_batch_stride_entries: usize,
    pub(crate) transposed_lhs_entries: usize,
    pub(crate) transposed_rhs_entries: usize,
    pub(crate) nonzero_workspace_limit_entries: usize,
    pub(crate) zero_workspace_algorithm_entries: usize,
    pub(crate) nonzero_workspace_algorithm_entries: usize,
    pub(crate) algorithm_workspace_bytes: usize,
    pub(crate) max_algorithm_workspace_bytes: usize,
    pub(crate) algorithm_waves_count: f32,
    pub(crate) max_algorithm_waves_count: f32,
    pub(crate) strided_batch_algorithm_entries: usize,
    pub(crate) pointer_array_batch_algorithm_entries: usize,
    pub(crate) default_epilogue_entries: usize,
    pub(crate) bias_epilogue_entries: usize,
    pub(crate) relu_epilogue_entries: usize,
    pub(crate) gelu_epilogue_entries: usize,
}

/// Aggregated cache-key properties for matrix multiplication requests without a supported cuBLASLt algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CublasLtMatmulUnsupportedAlgorithmCacheStats {
    pub(crate) entries: usize,
    pub(crate) f16_entries: usize,
    pub(crate) bf16_entries: usize,
    pub(crate) f32_entries: usize,
    pub(crate) f32_compute_entries: usize,
    pub(crate) tf32_compute_entries: usize,
    pub(crate) f32_fast_f16_compute_entries: usize,
    pub(crate) f32_fast_bf16_compute_entries: usize,
    pub(crate) library_versions: usize,
    pub(crate) max_rows: u64,
    pub(crate) max_inner: u64,
    pub(crate) max_cols: u64,
    pub(crate) max_batch_count: usize,
    pub(crate) batched_entries: usize,
    pub(crate) nonzero_batch_stride_entries: usize,
    pub(crate) nonzero_lhs_batch_stride_entries: usize,
    pub(crate) nonzero_rhs_batch_stride_entries: usize,
    pub(crate) transposed_lhs_entries: usize,
    pub(crate) transposed_rhs_entries: usize,
    pub(crate) nonzero_workspace_limit_entries: usize,
    pub(crate) default_epilogue_entries: usize,
    pub(crate) bias_epilogue_entries: usize,
    pub(crate) relu_epilogue_entries: usize,
    pub(crate) gelu_epilogue_entries: usize,
}

impl CublasLtMatmulAlgorithmCacheStats {
    pub(crate) fn from_cache_parts(
        supported: CublasLtMatmulSupportedAlgorithmCacheStats,
        unsupported: CublasLtMatmulUnsupportedAlgorithmCacheStats,
    ) -> Self {
        Self {
            supported,
            unsupported,
        }
    }
}

macro_rules! unsupported_algorithm_stat_accessors {
    ($($(#[doc = $doc:literal])+ $method:ident -> $field:ident: $return_type:ty),+ $(,)?) => {
        $(
            $(#[doc = $doc])+
            pub fn $method(&self) -> $return_type {
                self.unsupported.$field
            }
        )+
    };
}

impl CublasLtMatmulAlgorithmCacheStats {
    unsupported_algorithm_stat_accessors! {
        /// For unsupported entries, returns the number of cached entries.
        unsupported_entries -> entries: usize,
        /// For unsupported entries, returns the number of entries with `F16` tensor data.
        unsupported_f16_entries -> f16_entries: usize,
        /// For unsupported entries, returns the number of entries with `Bf16` tensor data.
        unsupported_bf16_entries -> bf16_entries: usize,
        /// For unsupported entries, returns the number of entries with `F32` tensor data.
        unsupported_f32_entries -> f32_entries: usize,
        /// For unsupported entries, returns the number of entries using the `F32` compute mode.
        unsupported_f32_compute_entries -> f32_compute_entries: usize,
        /// For unsupported entries, returns the number of entries using the `F32FastTf32` compute mode.
        unsupported_tf32_compute_entries -> tf32_compute_entries: usize,
        /// For unsupported entries, returns the number of entries using the `F32FastF16` compute mode.
        unsupported_f32_fast_f16_compute_entries -> f32_fast_f16_compute_entries: usize,
        /// For unsupported entries, returns the number of entries using the `F32FastBf16` compute mode.
        unsupported_f32_fast_bf16_compute_entries -> f32_fast_bf16_compute_entries: usize,
        /// For unsupported entries, returns the number of distinct cuBLASLt library versions in cache keys.
        unsupported_library_versions -> library_versions: usize,
        /// For unsupported entries, returns the largest matrix row count in cache keys, or zero for an empty cache.
        unsupported_max_rows -> max_rows: u64,
        /// For unsupported entries, returns the largest contracted matrix extent in cache keys, or zero for an empty cache.
        unsupported_max_inner -> max_inner: u64,
        /// For unsupported entries, returns the largest matrix column count in cache keys, or zero for an empty cache.
        unsupported_max_cols -> max_cols: u64,
        /// For unsupported entries, returns the largest batch count in cache keys, or zero for an empty cache.
        unsupported_max_batch_count -> max_batch_count: usize,
        /// For unsupported entries, returns the number of entries with batch count greater than one.
        unsupported_batched_entries -> batched_entries: usize,
        /// For unsupported entries, returns the number of entries with a nonzero batch offset for either input.
        unsupported_nonzero_batch_stride_entries -> nonzero_batch_stride_entries: usize,
        /// For unsupported entries, returns the number of entries with a nonzero left-input batch offset.
        unsupported_nonzero_lhs_batch_stride_entries -> nonzero_lhs_batch_stride_entries: usize,
        /// For unsupported entries, returns the number of entries with a nonzero right-input batch offset.
        unsupported_nonzero_rhs_batch_stride_entries -> nonzero_rhs_batch_stride_entries: usize,
        /// For unsupported entries, returns the number of entries that transpose the left input.
        unsupported_transposed_lhs_entries -> transposed_lhs_entries: usize,
        /// For unsupported entries, returns the number of entries that transpose the right input.
        unsupported_transposed_rhs_entries -> transposed_rhs_entries: usize,
        /// For unsupported entries, returns the number of entries whose cache key allows nonzero workspace.
        unsupported_nonzero_workspace_limit_entries -> nonzero_workspace_limit_entries: usize,
        /// For unsupported entries, returns the number of keys using the default epilogue.
        unsupported_default_epilogue_entries -> default_epilogue_entries: usize,
        /// For unsupported entries, returns the number of keys using `Bias`, `ReluBias`, `ReluAuxBias`, `GeluBias`, or `GeluAuxBias` epilogues.
        unsupported_bias_epilogue_entries -> bias_epilogue_entries: usize,
        /// For unsupported entries, returns the number of keys using the standalone `Relu` epilogue, excluding fused bias variants.
        unsupported_relu_epilogue_entries -> relu_epilogue_entries: usize,
        /// For unsupported entries, returns the number of keys using the standalone GELU epilogue, excluding fused bias variants.
        unsupported_gelu_epilogue_entries -> gelu_epilogue_entries: usize,
    }
}
