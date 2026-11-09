use singe_cublas::lt::types::Epilogue;
use singe_cuda::event::{EventFlags, EventRecordFlags};

use crate::{
    backend::cuda::{
        CudaStream,
        compiled::plan::cublas::{
            algorithm::validate_cublaslt_matmul_problem_dimensions,
            launch::{CublasLtMatmulLaunch, launch_cublaslt_matmul},
            layout::CublasLtMatrixLayout,
            tuning::{
                CublasLtMatmulTuningOptions, CublasLtMatmulTuningResult,
                autotune::cached_cublaslt_matmul_tuning_candidates,
                best_cublaslt_matmul_tuning_result, cublaslt_average_elapsed_ms,
                validation::{cublaslt_workspace, validate_cublaslt_matmul_tensors},
            },
        },
        graph_value::CudaGraphValue,
        runtime::{CublasLtMatmulAlgorithmEntry, CudaRuntime},
    },
    error::{Error, Result},
};

pub fn measure_cached_cublaslt_matmul_algorithms_with_workspace_limit(
    runtime: &CudaRuntime,
    a: &CudaGraphValue,
    b: &CudaGraphValue,
    output: &CudaGraphValue,
    epilogue: Epilogue,
    bias: Option<&CudaGraphValue>,
    workspace_limit_bytes: usize,
    config: CublasLtMatmulTuningOptions,
    stream: &CudaStream,
) -> Result<Option<CublasLtMatmulTuningResult>> {
    validate_cublaslt_matmul_tensors(a, b, output)?;
    let a_layout = CublasLtMatrixLayout::create(a, "left-hand input")?;
    let b_layout = CublasLtMatrixLayout::create(b, "right-hand input")?;
    validate_cublaslt_matmul_problem_dimensions(&a_layout, &b_layout)?;
    if a_layout.batch_count != b_layout.batch_count {
        return Err(Error::length_mismatch(
            "cublaslt tuning batch count",
            a_layout.batch_count,
            b_layout.batch_count,
        ));
    }

    let candidates = cached_cublaslt_matmul_tuning_candidates(
        runtime,
        &a_layout,
        &b_layout,
        output,
        epilogue,
        workspace_limit_bytes,
    )?;

    let mut results = Vec::with_capacity(candidates.len());
    for entry in candidates {
        let average_elapsed_ms = tuning_cublaslt_matmul_algorithm(
            runtime, a, b, output, epilogue, bias, &entry, config, stream,
        )?;
        results.push(CublasLtMatmulTuningResult::new(entry, average_elapsed_ms));
    }

    Ok(best_cublaslt_matmul_tuning_result(&results).cloned())
}

pub fn tuning_cublaslt_matmul_algorithm(
    runtime: &CudaRuntime,
    a: &CudaGraphValue,
    b: &CudaGraphValue,
    output: &CudaGraphValue,
    epilogue: Epilogue,
    bias: Option<&CudaGraphValue>,
    entry: &CublasLtMatmulAlgorithmEntry,
    config: CublasLtMatmulTuningOptions,
    stream: &CudaStream,
) -> Result<f32> {
    validate_cublaslt_matmul_tensors(a, b, output)?;
    let a_layout = CublasLtMatrixLayout::create(a, "left-hand input")?;
    let b_layout = CublasLtMatrixLayout::create(b, "right-hand input")?;
    validate_cublaslt_matmul_problem_dimensions(&a_layout, &b_layout)?;
    let mut workspace = cublaslt_workspace(runtime.context(), entry.workspace_bytes())?;

    // Keep workspace outside the draining scope so errors and unwinding cannot release it while a tuning kernel is still using it.
    stream.scope(|| {
        for _ in 0..config.warmup_iterations() {
            launch_cublaslt_matmul(CublasLtMatmulLaunch {
                runtime,
                a,
                b,
                output,
                a_layout: &a_layout,
                b_layout: &b_layout,
                algorithm: entry.algorithm(),
                compute_type: entry.compute_type(),
                epilogue,
                bias,
                addend: None,
                workspace: workspace.as_mut(),
                stream: Some(stream),
            })?;
        }

        let start = runtime
            .context()
            .create_event_with_flags(EventFlags::DEFAULT)?;
        let stop = runtime
            .context()
            .create_event_with_flags(EventFlags::DEFAULT)?;
        start.record(stream.raw(), EventRecordFlags::DEFAULT)?;
        for _ in 0..config.measurement_iterations() {
            launch_cublaslt_matmul(CublasLtMatmulLaunch {
                runtime,
                a,
                b,
                output,
                a_layout: &a_layout,
                b_layout: &b_layout,
                algorithm: entry.algorithm(),
                compute_type: entry.compute_type(),
                epilogue,
                bias,
                addend: None,
                workspace: workspace.as_mut(),
                stream: Some(stream),
            })?;
        }

        stop.record(stream.raw(), EventRecordFlags::DEFAULT)?;
        stop.synchronize()?;
        let elapsed_ms = stop.elapsed_time_since(&start)?;
        cublaslt_average_elapsed_ms(elapsed_ms, config.measurement_iterations())
    })
}
