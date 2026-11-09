use std::collections::HashSet;

use singe_cublas::{lt::types::Epilogue, types::ComputeType};

use crate::{
    backend::cuda::runtime::{CublasLtMatmulAlgorithmEntry, CublasLtMatmulAlgorithmKey},
    data_type::DataType,
};

/// Returns zero-workspace count, nonzero-workspace count, total workspace bytes, maximum workspace bytes, total waves count, and maximum waves count, in that order.
///
/// Integer counts and byte sums saturate.
/// Floating-point waves counts use ordinary addition, and both maxima start at zero.
pub fn cublaslt_matmul_algorithm_workspace_stats<'a>(
    entries: impl IntoIterator<Item = &'a CublasLtMatmulAlgorithmEntry>,
) -> (usize, usize, usize, usize, f32, f32) {
    let mut zero_workspace_entries = 0_usize;
    let mut nonzero_workspace_entries = 0_usize;
    let mut total_workspace_bytes = 0_usize;
    let mut max_workspace_bytes = 0_usize;
    let mut total_waves_count = 0.0_f32;
    let mut max_waves_count = 0.0_f32;
    for entry in entries {
        let workspace_bytes = entry.workspace_bytes();
        if workspace_bytes == 0 {
            zero_workspace_entries = zero_workspace_entries.saturating_add(1);
        } else {
            nonzero_workspace_entries = nonzero_workspace_entries.saturating_add(1);
        }

        total_workspace_bytes = total_workspace_bytes.saturating_add(workspace_bytes);
        max_workspace_bytes = max_workspace_bytes.max(workspace_bytes);
        let waves_count = entry.waves_count();
        total_waves_count += waves_count;
        max_waves_count = max_waves_count.max(waves_count);
    }
    (
        zero_workspace_entries,
        nonzero_workspace_entries,
        total_workspace_bytes,
        max_workspace_bytes,
        total_waves_count,
        max_waves_count,
    )
}

/// Returns saturated counts of algorithms reporting strided-batch support and pointer-array batch support, in that order.
///
/// An algorithm can contribute to both counts.
pub fn cublaslt_matmul_algorithm_batch_capability_counts<'a>(
    entries: impl IntoIterator<Item = &'a CublasLtMatmulAlgorithmEntry>,
) -> (usize, usize) {
    count_pair(
        entries,
        CublasLtMatmulAlgorithmEntry::supports_strided_batch,
        CublasLtMatmulAlgorithmEntry::supports_pointer_array_batch,
    )
}

/// Returns saturated counts of `F16`, `Bf16`, and `F32` cache keys, in that order.
///
/// Other element types contribute to none of these counts.
pub fn cublaslt_matmul_algorithm_dtype_counts<'a>(
    keys: impl IntoIterator<Item = &'a CublasLtMatmulAlgorithmKey>,
) -> (usize, usize, usize) {
    let mut f16_entries = 0_usize;
    let mut bf16_entries = 0_usize;
    let mut f32_entries = 0_usize;
    for key in keys {
        match key.data_type() {
            DataType::F16 => f16_entries = f16_entries.saturating_add(1),
            DataType::Bf16 => bf16_entries = bf16_entries.saturating_add(1),
            DataType::F32 => f32_entries = f32_entries.saturating_add(1),
            _ => {}
        }
    }
    (f16_entries, bf16_entries, f32_entries)
}

/// Returns saturated counts for `F32`, `F32FastTf32`, `F32FastF16`, and `F32FastBf16` compute modes, in that order.
///
/// Other compute modes contribute to none of these counts.
#[expect(
    clippy::similar_names,
    reason = "BLAS cache metrics distinguish standard precision modes and left/right matrix operands"
)]
pub fn cublaslt_matmul_algorithm_compute_counts<'a>(
    keys: impl IntoIterator<Item = &'a CublasLtMatmulAlgorithmKey>,
) -> (usize, usize, usize, usize) {
    let mut f32_entries = 0_usize;
    let mut tf32_entries = 0_usize;
    let mut f32_fast_f16_entries = 0_usize;
    let mut f32_fast_bf16_entries = 0_usize;
    for key in keys {
        match key.compute_type() {
            ComputeType::F32 => f32_entries = f32_entries.saturating_add(1),
            ComputeType::F32FastTf32 => tf32_entries = tf32_entries.saturating_add(1),
            ComputeType::F32FastF16 => {
                f32_fast_f16_entries = f32_fast_f16_entries.saturating_add(1);
            }
            ComputeType::F32FastBf16 => {
                f32_fast_bf16_entries = f32_fast_bf16_entries.saturating_add(1);
            }
            _ => {}
        }
    }
    (
        f32_entries,
        tf32_entries,
        f32_fast_f16_entries,
        f32_fast_bf16_entries,
    )
}

/// Returns saturated counts for default, bias-bearing, standalone `Relu`, and standalone GELU epilogues, in that order.
///
/// The bias count includes `Bias`, `ReluBias`, `ReluAuxBias`, `GeluBias`, and `GeluAuxBias`.
/// Fused bias variants do not also increment the standalone activation counts.
pub fn cublaslt_matmul_algorithm_epilogue_counts<'a>(
    keys: impl IntoIterator<Item = &'a CublasLtMatmulAlgorithmKey>,
) -> (usize, usize, usize, usize) {
    let mut default_entries = 0_usize;
    let mut bias_entries = 0_usize;
    let mut relu_entries = 0_usize;
    let mut gelu_entries = 0_usize;
    for key in keys {
        match key.epilogue() {
            Epilogue::Default => default_entries = default_entries.saturating_add(1),
            Epilogue::Bias
            | Epilogue::ReluBias
            | Epilogue::ReluAuxBias
            | Epilogue::GeluBias
            | Epilogue::GeluAuxBias => bias_entries = bias_entries.saturating_add(1),
            Epilogue::Relu => relu_entries = relu_entries.saturating_add(1),
            Epilogue::Gelu => gelu_entries = gelu_entries.saturating_add(1),
            _ => {}
        }
    }
    (default_entries, bias_entries, relu_entries, gelu_entries)
}

/// Returns the number of distinct cuBLASLt library versions represented by the keys.
pub fn cublaslt_matmul_algorithm_library_version_count<'a>(
    keys: impl IntoIterator<Item = &'a CublasLtMatmulAlgorithmKey>,
) -> usize {
    keys.into_iter()
        .map(CublasLtMatmulAlgorithmKey::library_version)
        .collect::<HashSet<_>>()
        .len()
}

/// Returns saturated counts of keys transposing the left and right inputs, in that order.
///
/// A key can contribute to both counts.
pub fn cublaslt_matmul_algorithm_transpose_counts<'a>(
    keys: impl IntoIterator<Item = &'a CublasLtMatmulAlgorithmKey>,
) -> (usize, usize) {
    count_pair(
        keys,
        CublasLtMatmulAlgorithmKey::transpose_a,
        CublasLtMatmulAlgorithmKey::transpose_b,
    )
}

/// Returns the number of keys whose requested workspace limit is greater than zero.
pub fn cublaslt_matmul_algorithm_nonzero_workspace_limit_count<'a>(
    keys: impl IntoIterator<Item = &'a CublasLtMatmulAlgorithmKey>,
) -> usize {
    keys.into_iter()
        .filter(|key| key.workspace_limit_bytes() > 0)
        .count()
}

/// Returns the largest row, contracted, and column extents in the keys, in that order.
///
/// Each maximum is zero for an empty input.
pub fn cublaslt_matmul_algorithm_dimension_maxima<'a>(
    keys: impl IntoIterator<Item = &'a CublasLtMatmulAlgorithmKey>,
) -> (u64, u64, u64) {
    let mut max_rows = 0_u64;
    let mut max_inner = 0_u64;
    let mut max_cols = 0_u64;
    for key in keys {
        max_rows = max_rows.max(key.rows());
        max_inner = max_inner.max(key.inner());
        max_cols = max_cols.max(key.cols());
    }
    (max_rows, max_inner, max_cols)
}

/// Returns the maximum batch count followed by saturated counts of batched keys, keys with either batch offset nonzero, keys with a nonzero left offset, and keys with a nonzero right offset.
///
/// Batched keys have batch count greater than one.
/// The maximum is zero for an empty input.
#[expect(
    clippy::similar_names,
    reason = "BLAS cache metrics distinguish standard precision modes and left/right matrix operands"
)]
pub fn cublaslt_matmul_algorithm_batch_counts<'a>(
    keys: impl IntoIterator<Item = &'a CublasLtMatmulAlgorithmKey>,
) -> (usize, usize, usize, usize, usize) {
    let mut max_batch_count = 0_usize;
    let mut batched_entries = 0_usize;
    let mut nonzero_batch_stride_entries = 0_usize;
    let mut nonzero_lhs_batch_stride_entries = 0_usize;
    let mut nonzero_rhs_batch_stride_entries = 0_usize;
    for key in keys {
        max_batch_count = max_batch_count.max(key.batch_count());
        if key.batch_count() > 1 {
            batched_entries = batched_entries.saturating_add(1);
        }

        if key.lhs_strided_batch_offset() != 0 || key.rhs_strided_batch_offset() != 0 {
            nonzero_batch_stride_entries = nonzero_batch_stride_entries.saturating_add(1);
        }

        if key.lhs_strided_batch_offset() != 0 {
            nonzero_lhs_batch_stride_entries = nonzero_lhs_batch_stride_entries.saturating_add(1);
        }
        if key.rhs_strided_batch_offset() != 0 {
            nonzero_rhs_batch_stride_entries = nonzero_rhs_batch_stride_entries.saturating_add(1);
        }
    }
    (
        max_batch_count,
        batched_entries,
        nonzero_batch_stride_entries,
        nonzero_lhs_batch_stride_entries,
        nonzero_rhs_batch_stride_entries,
    )
}

// Counts the items each predicate holds for; one item can count toward both.
fn count_pair<'a, T: 'a>(
    items: impl IntoIterator<Item = &'a T>,
    first: impl Fn(&T) -> bool,
    second: impl Fn(&T) -> bool,
) -> (usize, usize) {
    items.into_iter().fold((0, 0), |(a, b), item| {
        (a + usize::from(first(item)), b + usize::from(second(item)))
    })
}
