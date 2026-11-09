use crate::{
    backend::cuda::{self, compiled::executable::CudaCompiledGraph},
    error::Result,
    graph::GraphCudaOptions,
    onnx::OnnxModel,
};

/// Compiles an imported ONNX model into a replayable CUDA graph with the providers and shape policy of `options`.
///
/// # Errors
///
/// Returns an error if graph compilation, CUDA lowering, or CUDA graph instantiation fails.
pub fn compile_onnx_model(
    model: &OnnxModel,
    options: &GraphCudaOptions,
) -> Result<CudaCompiledGraph> {
    cuda::compile(&model.compile()?, options)
}
