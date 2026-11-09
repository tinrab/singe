mod algorithm;
mod descriptor;
mod graph;
mod launch;
mod layout;
mod tuning;

pub use self::{
    graph::{
        CublasLtMatmulGraph, build_cublas_matmul_graph, build_cublaslt_matmul_activation_graph,
        build_cublaslt_matmul_add_graph, build_cublaslt_matmul_graph,
        build_cublaslt_matmul_residual_graph, list_cublaslt_matmul_configs,
    },
    tuning::{
        CublasLtMatmulTuningOptions, CublasLtMatmulTuningResult,
        tune_cublaslt_matmul_algorithm_with_workspace_limit,
    },
};
use crate::backend::{
    cuda::compiled::plan::lowering::LoweringProvider,
    registry::{CudaGraphProvider, Operation, operation_support},
};

/// CUDA graph lowering provider for supported cuBLAS operations.
#[derive(Debug, Clone, Copy)]
pub struct CublasProvider;

/// CUDA graph lowering provider for supported cuBLASLt matrix multiplication operations.
#[derive(Debug, Clone, Copy)]
pub struct CublasLtProvider;

impl LoweringProvider for CublasLtProvider {
    fn kind(&self) -> CudaGraphProvider {
        CudaGraphProvider::CublasLt
    }

    fn cost(&self) -> u32 {
        8
    }

    fn supports_operation(&self, operation: Operation) -> bool {
        operation_support(operation).supports_cuda_graph_provider(CudaGraphProvider::CublasLt)
    }
}

impl LoweringProvider for CublasProvider {
    fn kind(&self) -> CudaGraphProvider {
        CudaGraphProvider::Cublas
    }

    fn cost(&self) -> u32 {
        5
    }

    fn supports_operation(&self, operation: Operation) -> bool {
        operation_support(operation).supports_cuda_graph_provider(CudaGraphProvider::Cublas)
    }
}

#[cfg(test)]
mod tests;
