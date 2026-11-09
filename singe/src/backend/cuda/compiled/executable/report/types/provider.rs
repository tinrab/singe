use crate::backend::{
    cuda::compiled::{
        executable::report::types::{
            GraphCompileInputLayoutRequirement, input_layout_requirement_for_support,
        },
        plan::lowering::CudaGraphProvider,
    },
    registry::LayoutSupport,
};

/// CUDA provider candidate reported for one graph operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphCompileProviderCandidate {
    /// Candidate CUDA provider.
    pub(crate) provider: CudaGraphProvider,
    /// Layout support advertised by the provider candidate.
    pub(crate) layout_support: LayoutSupport,
    /// Relative lowering cost used by the planner.
    pub(crate) cost: u32,
}

impl GraphCompileProviderCandidate {
    pub(crate) fn new(
        provider: CudaGraphProvider,
        layout_support: LayoutSupport,
        cost: u32,
    ) -> Self {
        Self {
            provider,
            layout_support,
            cost,
        }
    }

    /// Returns the candidate CUDA provider.
    #[must_use]
    pub fn provider(self) -> CudaGraphProvider {
        self.provider
    }

    /// Returns layout support advertised by the provider candidate.
    #[must_use]
    pub fn layout_support(self) -> LayoutSupport {
        self.layout_support
    }

    /// Returns the input layout required by this provider candidate.
    #[must_use]
    pub fn input_layout_requirement(self) -> GraphCompileInputLayoutRequirement {
        input_layout_requirement_for_support(self.layout_support)
    }

    /// Returns the relative lowering cost used by the planner.
    #[must_use]
    pub fn cost(self) -> u32 {
        self.cost
    }
}

/// CUDA provider rejection reported for one graph operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphCompileProviderRejection {
    /// Rejected candidate.
    pub(crate) candidate: GraphCompileProviderCandidate,
    /// Human-readable rejection reason.
    pub(crate) reason: String,
}

impl GraphCompileProviderRejection {
    pub(crate) fn new(candidate: GraphCompileProviderCandidate, reason: impl Into<String>) -> Self {
        Self {
            candidate,
            reason: reason.into(),
        }
    }

    /// Returns the rejected candidate.
    #[must_use]
    pub fn candidate(&self) -> &GraphCompileProviderCandidate {
        &self.candidate
    }

    /// Returns the human-readable rejection reason.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// CUDA provider decision reported for one graph operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphCompileProviderDecision {
    /// Selected provider, or `None` when no provider supports the operation.
    pub(crate) provider: Option<CudaGraphProvider>,
    /// Operation name attached to the selected provider decision.
    pub(crate) operation: Option<&'static str>,
    /// Unsupported reason when no provider was selected.
    pub(crate) unsupported_reason: Option<String>,
    /// Lower-priority candidates skipped by the planner.
    pub(crate) rejected_candidates: Vec<GraphCompileProviderCandidate>,
    /// Candidate rejection details.
    pub(crate) rejections: Vec<GraphCompileProviderRejection>,
}

impl GraphCompileProviderDecision {
    pub(crate) fn new(
        provider: Option<CudaGraphProvider>,
        operation: Option<&'static str>,
        unsupported_reason: Option<String>,
        rejected_candidates: Vec<GraphCompileProviderCandidate>,
        rejections: Vec<GraphCompileProviderRejection>,
    ) -> Self {
        Self {
            provider,
            operation,
            unsupported_reason,
            rejected_candidates,
            rejections,
        }
    }

    /// Returns the selected provider, or `None` when no provider supports the operation.
    #[must_use]
    pub fn provider(&self) -> Option<CudaGraphProvider> {
        self.provider
    }

    /// Returns the operation name attached to the selected provider decision.
    #[must_use]
    pub fn operation(&self) -> Option<&'static str> {
        self.operation
    }

    /// Returns the unsupported reason when no provider was selected.
    #[must_use]
    pub fn unsupported_reason(&self) -> Option<&str> {
        self.unsupported_reason.as_deref()
    }

    /// Returns candidate providers rejected by the planner.
    #[must_use]
    pub fn rejected_candidates(&self) -> &[GraphCompileProviderCandidate] {
        &self.rejected_candidates
    }

    /// Returns candidate rejection details.
    #[must_use]
    pub fn rejections(&self) -> &[GraphCompileProviderRejection] {
        &self.rejections
    }
}

/// CUDA provider attempt failure reported for one graph operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphCompileProviderFailure {
    /// Provider that failed during lowering.
    pub(crate) provider: CudaGraphProvider,
    /// Human-readable failure reason.
    pub(crate) reason: String,
}

impl GraphCompileProviderFailure {
    pub(crate) fn new(provider: CudaGraphProvider, reason: impl Into<String>) -> Self {
        Self {
            provider,
            reason: reason.into(),
        }
    }

    /// Returns the provider that failed during lowering.
    #[must_use]
    pub fn provider(&self) -> CudaGraphProvider {
        self.provider
    }

    /// Returns the human-readable failure reason.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }
}
