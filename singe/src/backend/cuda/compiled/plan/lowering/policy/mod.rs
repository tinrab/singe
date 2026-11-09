mod provider;

pub use self::provider::LoweringProvider;

#[cfg(test)]
use crate::backend::cuda::codegen::PlanningContext;
use crate::{
    backend::{
        cuda::compiled::plan::{
            kernel::KernelFeasibility,
            lowering::{
                LoweringCandidate,
                policy::provider::{
                    DEFAULT_CUDA_LOWERING_PROVIDERS, lowering_candidates_for_operation,
                },
            },
        },
        registry::{CudaGraphProvider, CudaGraphProviderSet, LibraryFeature, Operation},
    },
    graph::ir::{Graph, OperationNode, ValueId},
};

// The module kernels a policy lets the recognizer choose, each a subgraph one generated kernel computes whole.
// `All` is the default, as a library feature enum's is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModuleKernel {
    All,
    // Query and key rotation with the writes of keys and values to a paged cache.
    RotaryCacheWrite,
    // A query's norm and rotation with its attention over a paged cache, when a record asks.
    RotaryAttention,
}

/// Enabled CUDA graph lowering providers used to filter operation candidates.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoweringPolicy {
    providers: CudaGraphProviderSet,
    modules: Vec<ModuleKernel>,
}

impl LoweringPolicy {
    /// Enables `providers`, each library among them for all it may run.
    /// The list controls eligibility; candidate ordering comes from the operation candidate builder.
    #[must_use]
    pub fn new(providers: Vec<CudaGraphProvider>) -> Self {
        Self {
            providers: providers
                .into_iter()
                .fold(CudaGraphProviderSet::empty(), CudaGraphProviderSet::with),
            modules: vec![ModuleKernel::All],
        }
    }

    // Alias views and generated kernels alone: no library.
    #[must_use]
    pub fn generated_kernels() -> Self {
        Self::new(vec![
            CudaGraphProvider::Alias,
            CudaGraphProvider::Kernel,
            CudaGraphProvider::Fusion,
        ])
    }

    // Allows only `libraries`, each for what it names, and leaves aliases, kernels, and fusion as they are.
    // A `Library` or a feature enum's `All` is the whole library, and repeating a library adds to what it may run.
    #[must_use]
    pub fn with_libraries<L: Into<LibraryFeature>>(
        self,
        libraries: impl IntoIterator<Item = L>,
    ) -> Self {
        Self {
            providers: libraries
                .into_iter()
                .fold(self.providers.without_libraries(), |set, library| {
                    set.with_feature(library.into())
                }),
            modules: self.modules,
        }
    }

    // Lets the recognizer choose only `modules`; none leaves every node to its generic route.
    #[must_use]
    pub fn with_modules(mut self, modules: impl IntoIterator<Item = ModuleKernel>) -> Self {
        self.modules = modules.into_iter().collect();
        self
    }

    // Whether the recognizer may choose `module`.
    #[must_use]
    pub fn allows_module(&self, module: ModuleKernel) -> bool {
        self.modules.contains(&ModuleKernel::All) || self.modules.contains(&module)
    }

    // The policy's providers as the graph passes read them.
    #[must_use]
    pub const fn graph_providers(&self) -> CudaGraphProviderSet {
        self.providers
    }

    /// Returns whether the policy contains this provider.
    #[must_use]
    pub const fn provider_enabled(&self, provider: CudaGraphProvider) -> bool {
        self.providers.contains(provider)
    }

    // Whether `provider` may run `operation` under this policy.
    #[must_use]
    pub const fn allows(&self, provider: CudaGraphProvider, operation: Operation) -> bool {
        self.providers.allows(provider, operation)
    }

    /// Returns operation candidates whose providers are enabled by this policy.
    /// Candidate ordering from the registry and layout analysis is preserved.
    #[must_use]
    pub fn candidates_for_operation(
        &self,
        ir: &Graph,
        id: ValueId,
        operation: &OperationNode,
        kernels: &KernelFeasibility,
    ) -> Vec<LoweringCandidate> {
        lowering_candidates_for_operation(ir, id, operation, kernels)
            .into_iter()
            .filter(|candidate| {
                operation.opcode().map_or_else(
                    || self.provider_enabled(candidate.provider()),
                    |opcode| self.allows(candidate.provider(), opcode),
                )
            })
            .collect()
    }

    #[cfg(test)]
    #[must_use]
    pub fn supports_operation(
        &self,
        provider: CudaGraphProvider,
        ir: &Graph,
        id: ValueId,
        operation: &OperationNode,
    ) -> bool {
        self.candidates_for_operation(
            ir,
            id,
            operation,
            &KernelFeasibility::new(&PlanningContext::default()),
        )
        .iter()
        .any(|candidate| candidate.provider() == provider)
    }
}

impl Default for LoweringPolicy {
    fn default() -> Self {
        Self::new(DEFAULT_CUDA_LOWERING_PROVIDERS.to_vec())
    }
}

#[cfg(test)]
mod tests;
