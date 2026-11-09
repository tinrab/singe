mod activation;
mod cublas;
mod cublaslt;
mod validation;

use singe_cuda::graph::Graph as NativeCudaGraph;

pub use self::{
    activation::build_cublaslt_matmul_activation_graph,
    cublas::build_cublas_matmul_graph,
    cublaslt::{
        build_cublaslt_matmul_add_graph, build_cublaslt_matmul_graph,
        build_cublaslt_matmul_residual_graph, list_cublaslt_matmul_configs,
    },
};

/// A captured cuBLASLt matrix multiplication child graph and its workspace byte count.
/// Tensor bindings and workspace allocations are retained separately by compilation and replay resources.
#[derive(Debug)]
pub struct CublasLtMatmulGraph {
    child_graph: NativeCudaGraph,
    workspace_byte_size: usize,
}

impl CublasLtMatmulGraph {
    pub(crate) fn new(child_graph: NativeCudaGraph, workspace_byte_size: usize) -> Self {
        Self {
            child_graph,
            workspace_byte_size,
        }
    }

    /// Returns the captured native child graph.
    #[must_use]
    pub fn child_graph(&self) -> &NativeCudaGraph {
        &self.child_graph
    }

    /// Returns the workspace byte count recorded during capture.
    #[must_use]
    pub fn workspace_byte_size(&self) -> usize {
        self.workspace_byte_size
    }

    /// Transfers ownership of the captured child graph.
    /// The caller must preserve the separately retained tensor and workspace allocations through execution.
    #[must_use]
    pub fn into_native(self) -> NativeCudaGraph {
        self.child_graph
    }
}
