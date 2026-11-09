mod autotune;
mod measure;
mod validation;

use num_traits::ToPrimitive;
use singe_cublas::lt::types::Epilogue;
use singe_cuda::stream::StreamCaptureStatus;

use self::{
    autotune::record_tuned_cublaslt_matmul_autotune_entry,
    measure::measure_cached_cublaslt_matmul_algorithms_with_workspace_limit,
};

#[cfg(test)]
pub use self::validation::cublaslt_workspace;
pub use self::validation::validate_cublaslt_matmul_tensors;
use crate::{
    backend::cuda::{
        CudaStream,
        allocator::{CudaAllocator, CudaCachingAllocator},
        compiled::plan::cublas::{
            algorithm::cublaslt_matmul_algorithm_with_workspace_limit, layout::CublasLtMatrixLayout,
        },
        graph_value::CudaGraphValue,
        runtime::{CublasLtMatmulAlgorithmEntry, CudaRuntime},
    },
    error::{Error, Result},
    layout::Layout,
    tensor::Tensor,
};

pub fn warm_cublaslt_matmul_algorithm_with_workspace_limit(
    runtime: &CudaRuntime,
    a: &Tensor,
    b: &Tensor,
    output: &Tensor,
    epilogue: Epilogue,
    bias: Option<&Tensor>,
    workspace_limit_bytes: usize,
) -> Result<Option<CublasLtMatmulAlgorithmEntry>> {
    validate_cublaslt_matmul_tensors(a, b, output)?;
    let a_layout = CublasLtMatrixLayout::create(a, "left-hand input")?;
    let b_layout = CublasLtMatrixLayout::create(b, "right-hand input")?;
    let a = CudaGraphValue::from_tensor(a.clone())?;
    let b = CudaGraphValue::from_tensor(b.clone())?;
    let output = CudaGraphValue::from_tensor(output.clone())?;
    let bias = bias
        .map(|bias| CudaGraphValue::from_tensor(bias.clone()))
        .transpose()?;
    let mut autotune_tensors = vec![&a, &b, &output];
    if let Some(bias) = bias.as_ref() {
        autotune_tensors.push(bias);
    }
    cublaslt_matmul_algorithm_with_workspace_limit(
        runtime,
        &a_layout,
        &b_layout,
        &output,
        epilogue,
        bias.as_ref(),
        workspace_limit_bytes,
        Some(&autotune_tensors),
    )
}

pub fn tune_cublaslt_matmul_algorithm_with_workspace_limit(
    runtime: &CudaRuntime,
    a: &Tensor,
    b: &Tensor,
    output: &Tensor,
    epilogue: Epilogue,
    bias: Option<&Tensor>,
    workspace_limit_bytes: usize,
    config: CublasLtMatmulTuningOptions,
    stream: &CudaStream,
) -> Result<Option<CublasLtMatmulTuningResult>> {
    warm_cublaslt_matmul_algorithm_with_workspace_limit(
        runtime,
        a,
        b,
        output,
        epilogue,
        bias,
        workspace_limit_bytes,
    )?;
    let a = CudaGraphValue::from_tensor(a.clone())?;
    let b = CudaGraphValue::from_tensor(b.clone())?;
    let bias = bias
        .map(|bias| CudaGraphValue::from_tensor(bias.clone()))
        .transpose()?;
    let layout = Layout::from_tensor(output);
    let elements = layout.storage_span_element_count()?;
    if stream.raw().capture_status()? != StreamCaptureStatus::None {
        return Err(Error::unsupported_operation(
            "cuBLASLt tuning during capture",
        ));
    }

    let storage = CudaCachingAllocator::create(runtime.context(), Some(stream.raw_arc()))?
        .allocate_uninit(output.data_type(), elements)?;
    let output = CudaGraphValue::create(storage, output.data_type(), layout)?;
    stream.scope(|| {
        let mut inputs = vec![a.region()?.region(), b.region()?.region()];
        if let Some(bias) = &bias {
            inputs.push(bias.region()?.region());
        }

        stream.raw().join_regions(&inputs)?;
        let result = measure_cached_cublaslt_matmul_algorithms_with_workspace_limit(
            runtime,
            &a,
            &b,
            &output,
            epilogue,
            bias.as_ref(),
            workspace_limit_bytes,
            config,
            stream,
        )?;
        if let Some(result) = &result {
            record_tuned_cublaslt_matmul_autotune_entry(
                runtime,
                &a,
                &b,
                &output,
                bias.as_ref(),
                workspace_limit_bytes,
                result,
            )?;
        }
        Ok(result)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CublasLtMatmulTuningOptions {
    warmup_iterations: usize,
    measurement_iterations: usize,
}

impl CublasLtMatmulTuningOptions {
    pub const DEFAULT: Self = Self {
        warmup_iterations: 2,
        measurement_iterations: 5,
    };

    pub fn create(warmup_iterations: usize, measurement_iterations: usize) -> Result<Self> {
        if measurement_iterations == 0 {
            return Err(Error::invalid_argument(
                "cuBLASLt matmul tuning measurement iterations",
            ));
        }
        Ok(Self {
            warmup_iterations,
            measurement_iterations,
        })
    }

    #[must_use]
    pub fn warmup_iterations(&self) -> usize {
        self.warmup_iterations
    }

    #[must_use]
    pub fn measurement_iterations(&self) -> usize {
        self.measurement_iterations
    }
}

#[derive(Debug, Clone)]
pub struct CublasLtMatmulTuningResult {
    entry: CublasLtMatmulAlgorithmEntry,
    average_elapsed_ms: f32,
}

impl CublasLtMatmulTuningResult {
    #[must_use]
    pub fn new(entry: CublasLtMatmulAlgorithmEntry, average_elapsed_ms: f32) -> Self {
        Self {
            entry,
            average_elapsed_ms,
        }
    }

    #[must_use]
    pub fn entry(&self) -> &CublasLtMatmulAlgorithmEntry {
        &self.entry
    }

    #[must_use]
    pub fn average_elapsed_ms(&self) -> f32 {
        self.average_elapsed_ms
    }

    fn is_preferred_to(&self, other: &Self) -> bool {
        self.average_elapsed_ms < other.average_elapsed_ms
            || (self.average_elapsed_ms == other.average_elapsed_ms
                && self.entry.is_preferred_to(&other.entry))
    }
}

pub fn best_cublaslt_matmul_tuning_result<'a>(
    results: impl IntoIterator<Item = &'a CublasLtMatmulTuningResult>,
) -> Option<&'a CublasLtMatmulTuningResult> {
    results.into_iter().fold(None, |best, result| match best {
        Some(best) if !result.is_preferred_to(best) => Some(best),
        _ => Some(result),
    })
}

pub fn cublaslt_average_elapsed_ms(elapsed_ms: f32, iterations: usize) -> Result<f32> {
    let iterations = iterations
        .to_f32()
        .ok_or_else(|| Error::out_of_range("cublaslt tuning iteration count"))?;
    Ok(elapsed_ms / iterations)
}
