use singe_cublas::lt::types::Epilogue;

use crate::{
    backend::{
        cuda::{
            compiled::plan::{
                cublas::{
                    algorithm::{CublasLtMatmulDTypePlan, with_cublaslt},
                    layout::CublasLtMatrixLayout,
                    tuning::CublasLtMatmulTuningResult,
                },
                lowering::CudaGraphProvider,
            },
            graph_value::CudaGraphValue,
            runtime::{
                AutotuneCacheEntry, AutotuneCacheKey, CublasLtMatmulAlgorithmEntry,
                CublasLtMatmulAlgorithmKey, CublasLtMatmulKeyParts, CudaRuntime,
            },
        },
        registry::Operation,
    },
    error::Result,
    utility::Locked,
};

pub fn record_tuned_cublaslt_matmul_autotune_entry(
    runtime: &CudaRuntime,
    a: &CudaGraphValue,
    b: &CudaGraphValue,
    output: &CudaGraphValue,
    bias: Option<&CudaGraphValue>,
    workspace_limit_bytes: usize,
    result: &CublasLtMatmulTuningResult,
) -> Result<()> {
    let library_version = with_cublaslt(runtime, |cublaslt| Ok(cublaslt.version()?.to_string()))?;
    let mut tensors = vec![a, b, output];
    if let Some(bias) = bias {
        tensors.push(bias);
    }

    let key = AutotuneCacheKey::from_tensors(
        Operation::MatMul,
        output.data_type(),
        tensors,
        CudaGraphProvider::CublasLt,
        library_version,
        Some(workspace_limit_bytes),
    );
    let entry = AutotuneCacheEntry::new_cublaslt_matmul_tuning(
        result.entry().compute_type(),
        result.entry().workspace_bytes(),
        result.entry().fingerprint(),
        result.average_elapsed_ms(),
    );
    runtime.insert_autotune_cache_entry(key, entry)?;
    Ok(())
}

pub fn cached_cublaslt_matmul_tuning_candidates(
    runtime: &CudaRuntime,
    a_layout: &CublasLtMatrixLayout,
    b_layout: &CublasLtMatrixLayout,
    output: &CudaGraphValue,
    epilogue: Epilogue,
    workspace_limit_bytes: usize,
) -> Result<Vec<CublasLtMatmulAlgorithmEntry>> {
    let dtype_plan = CublasLtMatmulDTypePlan::create(output.data_type(), epilogue)?;
    let library_version = with_cublaslt(runtime, |cublaslt| Ok(cublaslt.version()?.to_string()))?;
    let cache = runtime
        .cublaslt_matmul_algorithms()
        .locked("cuda runtime cache")?;
    let mut candidates = Vec::new();
    for &compute_type in dtype_plan.compute_types {
        let key = CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
            data_type: dtype_plan.data_type,
            compute_type,
            transpose_a: a_layout.is_transposed(),
            transpose_b: b_layout.is_transposed(),
            epilogue,
            rows: a_layout.logical_rows,
            inner: a_layout.logical_cols,
            cols: b_layout.logical_cols,
            batch_count: a_layout.batch_count,
            a_strided_batch_offset: a_layout.strided_batch_offset,
            b_strided_batch_offset: b_layout.strided_batch_offset,
            workspace_limit_bytes,
            library_version: library_version.clone(),
        });
        candidates.extend(
            cache
                .iter()
                .filter(|(cached_key, _)| {
                    cached_key.matches_matmul_algorithm_family(&key)
                        && cached_key.workspace_limit_bytes() <= workspace_limit_bytes
                })
                .map(|(_, entry)| entry.clone()),
        );
    }

    drop(cache);
    Ok(candidates)
}
