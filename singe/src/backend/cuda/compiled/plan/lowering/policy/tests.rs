use crate::backend::{
    cuda::{
        codegen::PlanningContext,
        compiled::plan::{
            cublas::{CublasLtProvider, CublasProvider},
            cudnn::CudnnBackendProvider,
            cutensor::CutensorProvider,
            kernel::{KernelFeasibility, KernelProvider},
            lowering::policy::provider::{AliasProvider, LoweringProvider},
        },
    },
    registry::{CudaGraphProvider, operation_support, registered_operations},
};

#[test]
fn test_cuda_lowering_providers_own_operation_support() {
    let kernels = KernelFeasibility::new(&PlanningContext::default());
    let providers: &[(CudaGraphProvider, &dyn LoweringProvider)] = &[
        (CudaGraphProvider::Alias, &AliasProvider),
        (CudaGraphProvider::Kernel, &KernelProvider(&kernels)),
        (CudaGraphProvider::CudnnBackend, &CudnnBackendProvider),
        (CudaGraphProvider::CublasLt, &CublasLtProvider),
        (CudaGraphProvider::Cublas, &CublasProvider),
        (CudaGraphProvider::Cutensor, &CutensorProvider),
    ];

    for operation in registered_operations() {
        let support = operation_support(*operation);
        for (provider_kind, provider) in providers {
            assert_eq!(
                provider.supports_operation(*operation),
                support.supports_cuda_graph_provider(*provider_kind),
                "{provider_kind:?} support mismatch for {operation:?}"
            );
        }
    }
}
