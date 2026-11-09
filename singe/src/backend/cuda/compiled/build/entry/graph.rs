use crate::{
    backend::cuda::{self, compiled::executable::CudaCompiledGraph},
    error::Result,
    graph::{Graph, GraphCompileOptions, GraphCudaOptions},
};

/// Compiles a construction graph into a replayable CUDA graph with the providers and shape policy of `options`.
///
/// # Errors
///
/// Returns an error if logical graph compilation, shape policy validation, CUDA provider compilation, allocation, or native graph construction fails.
pub fn compile_graph(graph: &Graph, options: &GraphCudaOptions) -> Result<CudaCompiledGraph> {
    cuda::compile(
        &graph.compile_with_options(
            GraphCompileOptions::new().with_cuda_providers(options.lowering.graph_providers()),
        )?,
        options,
    )
}
