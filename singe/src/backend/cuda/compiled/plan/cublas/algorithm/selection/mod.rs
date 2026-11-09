mod cache;
mod problem;

use singe_cublas::lt::types::Epilogue;

use crate::{
    backend::{
        cuda::{
            compiled::plan::{
                cublas::{
                    algorithm::{
                        context::with_cublaslt,
                        heuristic::{
                            cublaslt_configured_matmul_algorithm,
                            cublaslt_heuristic_matmul_algorithm, cublaslt_listed_matmul_configs,
                        },
                        selection::{
                            cache::{
                                cached_cublaslt_matmul_algorithm,
                                mark_unsupported_cublaslt_matmul_algorithm,
                                remember_cublaslt_matmul_algorithm,
                                unsupported_cublaslt_matmul_algorithm,
                            },
                            problem::CublasLtMatmulAlgorithmProblem,
                        },
                    },
                    layout::CublasLtMatrixLayout,
                },
                lowering::CudaGraphProvider,
            },
            graph_value::CudaGraphValue,
            runtime::{
                AutotuneCacheKey, CublasLtMatmulAlgorithmEntry, CublasLtMatmulAlgorithmFingerprint,
                CudaRuntime,
            },
        },
        registry::Operation,
    },
    error::{Error, Result},
};

pub fn cublaslt_matmul_algorithm(
    runtime: &CudaRuntime,
    a_layout: &CublasLtMatrixLayout,
    b_layout: &CublasLtMatrixLayout,
    output: &CudaGraphValue,
    epilogue: Epilogue,
    bias: Option<&CudaGraphValue>,
) -> Result<CublasLtMatmulAlgorithmEntry> {
    cublaslt_matmul_algorithm_with_workspace_limit(
        runtime, a_layout, b_layout, output, epilogue, bias, 0, None,
    )?
    .ok_or_else(|| {
        Error::library_refused(
            "cuBLASLt",
            Error::unsupported_operation_with_details(
                "cuda graph cuBLASLt matmul has no zero-workspace heuristic algorithm",
                format!(
                    "a {a_layout:?}, b {b_layout:?}, output {:?} {:?}, epilogue {epilogue:?}, bias {:?}",
                    output.data_type(),
                    output.layout(),
                    bias.map(|bias| (bias.data_type(), bias.layout().clone())),
                ),
            ),
        )
    })
}

// The algorithm a record states, past every cache, since a cache keyed on the problem would hand back another.
//
// # Errors
//
// Returns an error if cuBLASLt refuses it for this matmul.
pub fn configured_cublaslt_matmul_algorithm(
    runtime: &CudaRuntime,
    a_layout: &CublasLtMatrixLayout,
    b_layout: &CublasLtMatrixLayout,
    output: &CudaGraphValue,
    epilogue: Epilogue,
    bias: Option<&CudaGraphValue>,
    config: &CublasLtMatmulAlgorithmFingerprint,
) -> Result<CublasLtMatmulAlgorithmEntry> {
    let problem =
        CublasLtMatmulAlgorithmProblem::create(a_layout, b_layout, output, epilogue, bias)?;
    with_cublaslt(runtime, |cublaslt| {
        for &compute_type in problem.compute_types() {
            if let Some(entry) = cublaslt_configured_matmul_algorithm(
                cublaslt,
                problem.dtype_plan(),
                compute_type,
                a_layout,
                b_layout,
                epilogue,
                bias,
                problem.output_layout(),
                problem.rows(),
                problem.cols(),
                config,
            )? {
                return Ok(entry);
            }
        }
        Err(Error::unsupported_operation(
            "cuBLASLt matmul with this configuration",
        ))
    })
}

// What cuBLASLt could run this matmul with, other than what it would choose first.
//
// # Errors
//
// Returns an error if cuBLASLt fails to answer.
pub fn cublaslt_matmul_configs(
    runtime: &CudaRuntime,
    a_layout: &CublasLtMatrixLayout,
    b_layout: &CublasLtMatrixLayout,
    output: &CudaGraphValue,
) -> Result<Vec<CublasLtMatmulAlgorithmFingerprint>> {
    let problem = CublasLtMatmulAlgorithmProblem::create(
        a_layout,
        b_layout,
        output,
        Epilogue::Default,
        None,
    )?;
    with_cublaslt(runtime, |cublaslt| {
        for &compute_type in problem.compute_types() {
            let configs = cublaslt_listed_matmul_configs(
                cublaslt,
                problem.dtype_plan(),
                compute_type,
                a_layout,
                b_layout,
                Epilogue::Default,
                None,
                problem.output_layout(),
                problem.rows(),
                problem.cols(),
            )?;
            if !configs.is_empty() {
                return Ok(configs);
            }
        }
        Ok(Vec::new())
    })
}

pub fn cublaslt_matmul_algorithm_with_workspace_limit(
    runtime: &CudaRuntime,
    a_layout: &CublasLtMatrixLayout,
    b_layout: &CublasLtMatrixLayout,
    output: &CudaGraphValue,
    epilogue: Epilogue,
    bias: Option<&CudaGraphValue>,
    workspace_limit_bytes: usize,
    autotune_tensors: Option<&[&CudaGraphValue]>,
) -> Result<Option<CublasLtMatmulAlgorithmEntry>> {
    let problem =
        CublasLtMatmulAlgorithmProblem::create(a_layout, b_layout, output, epilogue, bias)?;

    with_cublaslt(runtime, |cublaslt| {
        let library_version = cublaslt.version()?.to_string();
        let autotune_key = autotune_tensors.map(|tensors| {
            AutotuneCacheKey::from_tensors(
                Operation::MatMul,
                output.data_type(),
                tensors.iter().copied(),
                CudaGraphProvider::CublasLt,
                library_version.clone(),
                Some(workspace_limit_bytes),
            )
        });
        for &compute_type in problem.compute_types() {
            let key = problem.key(compute_type, workspace_limit_bytes, library_version.clone());
            if let Some(entry) =
                cached_cublaslt_matmul_algorithm(runtime, autotune_key.as_ref(), &key)?
            {
                return Ok(Some(entry));
            }

            if unsupported_cublaslt_matmul_algorithm(runtime, &key)? {
                continue;
            }

            if let Some(entry) = cublaslt_heuristic_matmul_algorithm(
                cublaslt,
                problem.dtype_plan(),
                compute_type,
                a_layout,
                b_layout,
                epilogue,
                bias,
                problem.output_layout(),
                problem.rows(),
                problem.cols(),
                workspace_limit_bytes,
            )? {
                remember_cublaslt_matmul_algorithm(runtime, key, entry.clone())?;
                return Ok(Some(entry));
            }

            mark_unsupported_cublaslt_matmul_algorithm(runtime, key)?;
        }

        Ok(None)
    })
}
