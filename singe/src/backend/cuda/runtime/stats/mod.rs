mod algorithm;
mod cublaslt;

use num_traits::ToPrimitive;

pub use self::{
    algorithm::{
        CublasLtMatmulAlgorithmCacheStats, CublasLtMatmulSupportedAlgorithmCacheStats,
        CublasLtMatmulUnsupportedAlgorithmCacheStats,
    },
    cublaslt::{
        cublaslt_matmul_algorithm_batch_capability_counts, cublaslt_matmul_algorithm_batch_counts,
        cublaslt_matmul_algorithm_compute_counts, cublaslt_matmul_algorithm_dimension_maxima,
        cublaslt_matmul_algorithm_dtype_counts, cublaslt_matmul_algorithm_epilogue_counts,
        cublaslt_matmul_algorithm_library_version_count,
        cublaslt_matmul_algorithm_nonzero_workspace_limit_count,
        cublaslt_matmul_algorithm_transpose_counts, cublaslt_matmul_algorithm_workspace_stats,
    },
};
use crate::backend::cuda::runtime::AutotuneCacheEntry;

/// Entry counts and tuning totals from a CUDA autotune cache snapshot.
///
/// Tuning metrics include only entries whose selected algorithm is `"cublaslt matmul tuning"`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AutotuneCacheStats {
    entries: usize,
    cublaslt_matmul_tuning_entries: usize,
    cublaslt_matmul_tuning_workspace_bytes: usize,
    cublaslt_matmul_tuning_elapsed_ms: f32,
}

impl AutotuneCacheStats {
    pub(crate) fn from_entries<'a>(
        entries: impl IntoIterator<Item = &'a AutotuneCacheEntry>,
    ) -> Self {
        let mut stats = Self::default();
        for entry in entries {
            stats.entries += 1;
            stats.cublaslt_matmul_tuning_entries += 1;
            stats.cublaslt_matmul_tuning_workspace_bytes = stats
                .cublaslt_matmul_tuning_workspace_bytes
                .saturating_add(entry.workspace_bytes());
            if let Some(elapsed_ms) = entry.average_elapsed_ms() {
                stats.cublaslt_matmul_tuning_elapsed_ms += elapsed_ms;
            }
        }
        stats
    }

    /// Returns the total number of entries in the autotune cache snapshot.
    #[must_use]
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// Returns the number of entries selected by cuBLASLt matrix multiplication tuninging.
    #[must_use]
    pub fn cublaslt_matmul_tuning_entries(&self) -> usize {
        self.cublaslt_matmul_tuning_entries
    }

    /// Returns the saturated sum of workspace byte requirements for tuning-selected entries.
    ///
    /// This sums per-entry requirements without measuring simultaneous workspace allocation.
    #[must_use]
    pub fn cublaslt_matmul_tuning_workspace_bytes(&self) -> usize {
        self.cublaslt_matmul_tuning_workspace_bytes
    }

    /// Returns the sum of available average elapsed times for tuning-selected entries, in milliseconds.
    ///
    /// Entries without a timing contribute zero.
    #[must_use]
    pub fn cublaslt_matmul_tuning_elapsed_ms(&self) -> f32 {
        self.cublaslt_matmul_tuning_elapsed_ms
    }

    /// Returns the summed available tuning timings divided by all tuning-selected entries, in milliseconds.
    ///
    /// Entries without a timing still contribute to the denominator.
    /// Returns `None` when there are no tuning entries or converting their count to `f32` fails.
    #[must_use]
    pub fn cublaslt_matmul_tuning_average_elapsed_ms(&self) -> Option<f32> {
        if self.cublaslt_matmul_tuning_entries == 0 {
            None
        } else {
            Some(
                self.cublaslt_matmul_tuning_elapsed_ms
                    / self.cublaslt_matmul_tuning_entries.to_f32()?,
            )
        }
    }
}
