use crate::backend::{
    cuda::compiled::build::has_provider_lowering,
    registry::{BackendExecution, CudaGraphProvider, operation_support, registered_operations},
};

/// Library providers that need an operation-specific lowering, unlike generated kernels and aliases.
const LIBRARY_PROVIDERS: [CudaGraphProvider; 5] = [
    CudaGraphProvider::CudnnBackend,
    CudaGraphProvider::CublasLt,
    CudaGraphProvider::Cublas,
    CudaGraphProvider::Cutensor,
    CudaGraphProvider::Cufft,
];

#[test]
fn test_declared_library_providers_match_registered_lowerings() {
    let mismatches = registered_operations()
        .iter()
        .filter(|&&operation| {
            let support = operation_support(operation);
            let declares_library = support.supports(BackendExecution::CudaGraph)
                && support
                    .cuda_graph_providers()
                    .iter()
                    .any(|provider| LIBRARY_PROVIDERS.contains(provider));
            declares_library != has_provider_lowering(operation)
        })
        .map(|operation| operation.name())
        .collect::<Vec<_>>();
    assert!(
        mismatches.is_empty(),
        "declared library graph providers differ from registered lowerings for {mismatches:?}"
    );
}
