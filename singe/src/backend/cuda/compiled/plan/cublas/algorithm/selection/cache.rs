use crate::{
    backend::cuda::runtime::{
        AutotuneCacheKey, CublasLtMatmulAlgorithmEntry, CublasLtMatmulAlgorithmKey, CudaRuntime,
    },
    error::{Error, Result},
    utility::Locked,
};

pub fn cached_cublaslt_matmul_algorithm(
    runtime: &CudaRuntime,
    autotune_key: Option<&AutotuneCacheKey>,
    key: &CublasLtMatmulAlgorithmKey,
) -> Result<Option<CublasLtMatmulAlgorithmEntry>> {
    if let Some(autotune_key) = autotune_key
        && let Some(entry) =
            runtime.tuned_cublaslt_matmul_algorithm_for_autotune_key(autotune_key, key)?
    {
        ensure_cached_algorithm_workspace("tuned", key, &entry)?;
        return Ok(Some(entry));
    }

    if let Some(entry) =
        runtime.best_cached_cublaslt_matmul_algorithm_for_key(key, key.workspace_limit_bytes())?
    {
        ensure_cached_algorithm_workspace("cached", key, &entry)?;
        return Ok(Some(entry));
    }
    Ok(None)
}

pub fn unsupported_cublaslt_matmul_algorithm(
    runtime: &CudaRuntime,
    key: &CublasLtMatmulAlgorithmKey,
) -> Result<bool> {
    let unsupported = runtime
        .unsupported_cublaslt_matmul_algorithms()
        .locked("cuda runtime cache")?;
    Ok(unsupported.contains_key(key))
}

pub fn remember_cublaslt_matmul_algorithm(
    runtime: &CudaRuntime,
    key: CublasLtMatmulAlgorithmKey,
    entry: CublasLtMatmulAlgorithmEntry,
) -> Result<()> {
    runtime
        .cublaslt_matmul_algorithms()
        .locked("cuda runtime cache")?
        .insert(key, entry);

    Ok(())
}

pub fn mark_unsupported_cublaslt_matmul_algorithm(
    runtime: &CudaRuntime,
    key: CublasLtMatmulAlgorithmKey,
) -> Result<()> {
    runtime
        .unsupported_cublaslt_matmul_algorithms()
        .locked("cuda runtime cache")?
        .insert(key, ());

    Ok(())
}

fn ensure_cached_algorithm_workspace(
    source: &'static str,
    key: &CublasLtMatmulAlgorithmKey,
    entry: &CublasLtMatmulAlgorithmEntry,
) -> Result<()> {
    if key.workspace_limit_bytes() == 0 && entry.requires_workspace() {
        return Err(Error::unsupported_operation(format!(
            "cuda graph cuBLASLt {source} matmul algorithm requires workspace"
        )));
    }
    Ok(())
}
