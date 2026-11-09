use crate::backend::{cuda::compiled::plan::lowering::CudaGraphProvider, registry::LayoutSupport};

/// Returns the diagnostic name for a CUDA graph lowering provider.
#[must_use]
pub fn graph_provider_name(provider: CudaGraphProvider) -> &'static str {
    match provider {
        CudaGraphProvider::Alias => "alias",
        CudaGraphProvider::Kernel => "kernel",
        CudaGraphProvider::CudnnBackend => "cudnn",
        CudaGraphProvider::CublasLt => "cublaslt",
        CudaGraphProvider::Cublas => "cublas",
        CudaGraphProvider::Cutensor => "cutensor",
        CudaGraphProvider::Cufft => "cufft",
        CudaGraphProvider::Fusion => "fusion",
    }
}

/// Returns the diagnostic description of a backend layout requirement.
#[must_use]
pub fn graph_layout_name(layout: LayoutSupport) -> &'static str {
    match layout {
        LayoutSupport::AnyStrided => "any strided",
        LayoutSupport::MaterializesToContiguous => "materializes to contiguous",
        LayoutSupport::RequiresContiguous => "requires contiguous",
    }
}
