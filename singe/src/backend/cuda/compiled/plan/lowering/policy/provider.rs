use std::fmt;

use crate::{
    backend::{
        cuda::compiled::plan::{
            cublas::{CublasLtProvider, CublasProvider},
            cudnn::CudnnBackendProvider,
            cufft::CufftProvider,
            cutensor::CutensorProvider,
            kernel::{KernelFeasibility, KernelProvider},
            lowering::LoweringCandidate,
        },
        kernel_lowering::{is_fusible, is_row_opcode},
        registry::{CudaGraphProvider, Operation, operation_support},
    },
    graph::ir::{Graph, OperationNode, ValueId},
};

/// Default CUDA providers, ordered as alias, generated kernel, cuDNN, cuBLASLt, cuBLAS, cuTENSOR, cuFFT, and fusion.
///
/// The lowering plan sorts eligible candidates by cost before choosing a provider.
pub const DEFAULT_CUDA_LOWERING_PROVIDERS: [CudaGraphProvider; 8] = [
    CudaGraphProvider::Alias,
    CudaGraphProvider::Kernel,
    CudaGraphProvider::CudnnBackend,
    CudaGraphProvider::CublasLt,
    CudaGraphProvider::Cublas,
    CudaGraphProvider::Cutensor,
    CudaGraphProvider::Cufft,
    CudaGraphProvider::Fusion,
];

/// Describes CUDA operation support and creates provider candidates for lowering.
pub trait LoweringProvider: fmt::Debug + Send + Sync {
    /// Returns this provider's identity.
    fn kind(&self) -> CudaGraphProvider;

    /// Returns this provider's selection cost; lower costs take priority.
    fn cost(&self) -> u32;

    /// Returns whether this provider supports the registry operation.
    fn supports_operation(&self, operation: Operation) -> bool;

    /// Returns a candidate for a supported operation with an opcode and schema.
    ///
    /// The default implementation uses the schema's layout support and this provider's cost.
    /// It returns `None` when the operation has no opcode or schema, or this provider does not support its opcode.
    fn candidate_for_operation(
        &self,
        _ir: &Graph,
        _id: ValueId,
        operation: &OperationNode,
    ) -> Option<LoweringCandidate> {
        let opcode = operation.opcode()?;
        let schema = operation.schema()?;
        self.supports_operation(opcode)
            .then_some(LoweringCandidate::new(
                self.kind(),
                schema.support().layout(),
                self.cost(),
            ))
    }
}

/// CUDA lowering provider for metadata-only operations that reuse input storage.
#[derive(Debug, Clone, Copy)]
pub struct AliasProvider;

impl LoweringProvider for AliasProvider {
    fn kind(&self) -> CudaGraphProvider {
        CudaGraphProvider::Alias
    }

    fn cost(&self) -> u32 {
        0
    }

    fn supports_operation(&self, operation: Operation) -> bool {
        operation_support(operation).supports_cuda_graph_provider(CudaGraphProvider::Alias)
    }
}

/// CUDA graph lowering provider that fuses connected groups of element-wise operations, and a concatenation of their results or a row operation that reads them, into one generated kernel.
///
/// Its cost exceeds the generated kernel's, so a node outside a group keeps its own kernel; lowering forms a group only where the plan lists this provider.
#[derive(Debug, Clone, Copy)]
pub struct FusionProvider;

impl LoweringProvider for FusionProvider {
    fn kind(&self) -> CudaGraphProvider {
        CudaGraphProvider::Fusion
    }

    fn cost(&self) -> u32 {
        25
    }

    fn supports_operation(&self, operation: Operation) -> bool {
        (is_fusible(operation) || is_row_opcode(operation) || operation == Operation::Concat)
            && operation_support(operation).supports_cuda_graph_provider(CudaGraphProvider::Kernel)
    }
}

/// Returns eligible CUDA provider candidates for one graph operation.
///
/// Providers are queried in default provider order.
/// The result is not sorted by cost.
/// Providers that need a generated kernel plan one through `kernels` to see whether they can, which neither collects nor tunes it.
pub fn lowering_candidates_for_operation(
    ir: &Graph,
    id: ValueId,
    operation: &OperationNode,
    kernels: &KernelFeasibility,
) -> Vec<LoweringCandidate> {
    [
        AliasProvider.candidate_for_operation(ir, id, operation),
        KernelProvider(kernels).candidate_for_operation(ir, id, operation),
        CudnnBackendProvider.candidate_for_operation(ir, id, operation),
        CublasLtProvider.candidate_for_operation(ir, id, operation),
        CublasProvider.candidate_for_operation(ir, id, operation),
        CutensorProvider.candidate_for_operation(ir, id, operation),
        CufftProvider.candidate_for_operation(ir, id, operation),
        FusionProvider.candidate_for_operation(ir, id, operation),
    ]
    .into_iter()
    .flatten()
    .collect()
}
