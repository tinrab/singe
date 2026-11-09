use crate::backend::registry::CudaGraphProvider;

pub const KERNEL_CUDNN_GRAPH_PROVIDERS: &[CudaGraphProvider] =
    &[CudaGraphProvider::Kernel, CudaGraphProvider::CudnnBackend];

pub const KERNEL_CUDNN_CUBLAS_GRAPH_PROVIDERS: &[CudaGraphProvider] = &[
    CudaGraphProvider::Kernel,
    CudaGraphProvider::CudnnBackend,
    CudaGraphProvider::CublasLt,
    CudaGraphProvider::Cublas,
];

pub const CUDNN_GRAPH_PROVIDER: &[CudaGraphProvider] = &[CudaGraphProvider::CudnnBackend];

pub const CUBLASLT_GRAPH_PROVIDER: &[CudaGraphProvider] = &[CudaGraphProvider::CublasLt];

pub const CUFFT_GRAPH_PROVIDER: &[CudaGraphProvider] = &[CudaGraphProvider::Cufft];

pub const CUTENSOR_GRAPH_PROVIDER: &[CudaGraphProvider] = &[CudaGraphProvider::Cutensor];

pub const KERNEL_GRAPH_PROVIDER: &[CudaGraphProvider] = &[CudaGraphProvider::Kernel];

pub const ALIAS_GRAPH_PROVIDER: &[CudaGraphProvider] = &[CudaGraphProvider::Alias];

pub const ALIAS_KERNEL_GRAPH_PROVIDERS: &[CudaGraphProvider] =
    &[CudaGraphProvider::Alias, CudaGraphProvider::Kernel];

pub const NO_CUDA_GRAPH_PROVIDER: &[CudaGraphProvider] = &[];
