mod activation;
mod cublas;
mod cublaslt;
mod validation;

use std::mem::MaybeUninit;

use singe_cuda::{graph::Graph as NativeCudaGraph, memory::DeviceMemory};

pub use self::{
    activation::build_cublaslt_matmul_activation_graph,
    cublas::build_cublas_matmul_graph,
    cublaslt::{
        build_cublaslt_matmul_add_graph, build_cublaslt_matmul_graph,
        build_cublaslt_matmul_residual_graph, list_cublaslt_matmul_configs,
    },
};

/// A captured cuBLASLt matrix multiplication child graph and the workspace its captured call reads.
#[derive(Debug)]
pub struct CublasLtMatmulGraph {
    child_graph: NativeCudaGraph,
    // The workspace the captured call reads, which every replay reads again.
    workspace: Option<DeviceMemory<MaybeUninit<u8>>>,
}

impl CublasLtMatmulGraph {
    pub(crate) fn new(
        child_graph: NativeCudaGraph,
        workspace: Option<DeviceMemory<MaybeUninit<u8>>>,
    ) -> Self {
        Self {
            child_graph,
            workspace,
        }
    }

    /// Returns the captured native child graph.
    #[must_use]
    pub fn child_graph(&self) -> &NativeCudaGraph {
        &self.child_graph
    }

    /// Returns the byte count of the workspace the captured call reads.
    #[must_use]
    pub fn workspace_byte_size(&self) -> usize {
        self.workspace.as_ref().map_or(0, DeviceMemory::byte_len)
    }
}
