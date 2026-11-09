use crate::backend::{
    cuda::compiled::plan::lowering::LoweringProvider,
    registry::{CudaGraphProvider, Operation, operation_support},
};

// CUDA graph lowering provider for Fourier transforms through cuFFT.
#[derive(Debug, Clone, Copy)]
pub struct CufftProvider;

impl LoweringProvider for CufftProvider {
    fn kind(&self) -> CudaGraphProvider {
        CudaGraphProvider::Cufft
    }

    fn cost(&self) -> u32 {
        30
    }

    fn supports_operation(&self, operation: Operation) -> bool {
        operation_support(operation).supports_cuda_graph_provider(CudaGraphProvider::Cufft)
    }
}
