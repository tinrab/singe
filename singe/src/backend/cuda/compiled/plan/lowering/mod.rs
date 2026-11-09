mod plan;
mod policy;
mod realized;

pub use self::{
    plan::LoweringPlan,
    policy::{LoweringPolicy, LoweringProvider, ModuleKernel},
    realized::RealizedLoweringPlan,
};
pub use crate::backend::registry::{
    CublasFeature, CublasLtFeature, CudaGraphProvider, CudnnFeature, CufftFeature, CutensorFeature,
    LibraryFeature,
};
use crate::backend::registry::{LayoutSupport, Operation};

/// A CUDA execution provider considered for an operation, with its layout support and selection cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoweringCandidate {
    provider: CudaGraphProvider,
    layout_support: LayoutSupport,
    cost: u32,
}

impl LoweringCandidate {
    /// Creates a provider candidate with the supplied layout support and cost.
    ///
    /// Lower costs take priority when the lowering plan selects among candidates.
    #[must_use]
    pub fn new(provider: CudaGraphProvider, layout_support: LayoutSupport, cost: u32) -> Self {
        Self {
            provider,
            layout_support,
            cost,
        }
    }

    /// Returns the candidate execution provider.
    #[must_use]
    pub fn provider(self) -> CudaGraphProvider {
        self.provider
    }

    /// Returns the tensor layouts supported by this candidate.
    #[must_use]
    pub fn layout_support(self) -> LayoutSupport {
        self.layout_support
    }

    /// Returns the selection cost; lower costs take priority.
    #[must_use]
    pub fn cost(self) -> u32 {
        self.cost
    }
}

/// The CUDA execution provider selected for an operation, or an unsupported-operation reason.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LoweringDecision {
    /// Reuses input storage through a metadata-only operation.
    Alias {
        /// Operation assigned to this execution provider.
        operation: Operation,
    },
    /// Executes a generated CUDA kernel.
    Kernel {
        /// Operation assigned to this execution provider.
        operation: Operation,
    },
    /// Executes through the cuDNN backend.
    CudnnBackend {
        /// Operation assigned to this execution provider.
        operation: Operation,
    },
    /// Executes through cuBLASLt.
    CublasLt {
        /// Operation assigned to this execution provider.
        operation: Operation,
    },
    /// Executes through cuBLAS.
    Cublas {
        /// Operation assigned to this execution provider.
        operation: Operation,
    },
    /// Executes through cuTENSOR.
    Cutensor {
        /// Operation assigned to this execution provider.
        operation: Operation,
    },
    /// Executes through cuFFT.
    Cufft {
        /// Operation assigned to this execution provider.
        operation: Operation,
    },
    /// Executes as part of a fused element-wise kernel.
    Fusion {
        /// Operation assigned to this execution provider.
        operation: Operation,
    },
    /// No available CUDA provider can execute the operation.
    Unsupported {
        /// Explanation of why CUDA lowering is unsupported.
        reason: String,
    },
}

impl LoweringDecision {
    /// Returns the selected provider, or `None` for an unsupported operation.
    #[must_use]
    pub fn provider(&self) -> Option<CudaGraphProvider> {
        match self {
            Self::Alias { .. } => Some(CudaGraphProvider::Alias),
            Self::Kernel { .. } => Some(CudaGraphProvider::Kernel),
            Self::CudnnBackend { .. } => Some(CudaGraphProvider::CudnnBackend),
            Self::CublasLt { .. } => Some(CudaGraphProvider::CublasLt),
            Self::Cublas { .. } => Some(CudaGraphProvider::Cublas),
            Self::Cutensor { .. } => Some(CudaGraphProvider::Cutensor),
            Self::Cufft { .. } => Some(CudaGraphProvider::Cufft),
            Self::Fusion { .. } => Some(CudaGraphProvider::Fusion),
            Self::Unsupported { .. } => None,
        }
    }

    /// Returns the assigned operation, or `None` for an unsupported operation.
    #[cfg(test)]
    #[must_use]
    pub fn operation(&self) -> Option<Operation> {
        match self {
            Self::Alias { operation }
            | Self::Kernel { operation }
            | Self::CudnnBackend { operation }
            | Self::CublasLt { operation }
            | Self::Cublas { operation }
            | Self::Cutensor { operation }
            | Self::Cufft { operation }
            | Self::Fusion { operation } => Some(*operation),
            Self::Unsupported { .. } => None,
        }
    }
}

/// A rejected CUDA provider candidate and the reason for rejecting it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoweringRejection {
    candidate: LoweringCandidate,
    reason: String,
}

impl LoweringRejection {
    /// Records a rejected candidate and a human-readable reason.
    pub fn new(candidate: LoweringCandidate, reason: impl Into<String>) -> Self {
        Self {
            candidate,
            reason: reason.into(),
        }
    }

    /// Returns the rejected provider candidate.
    #[must_use]
    pub fn candidate(&self) -> LoweringCandidate {
        self.candidate
    }

    /// Returns the reason for rejecting this candidate.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// The chosen CUDA lowering decision and records of alternative candidates.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoweringDecisionRecord {
    decision: LoweringDecision,
    rejected_candidates: Vec<LoweringCandidate>,
    rejections: Vec<LoweringRejection>,
}

impl LoweringDecisionRecord {
    /// Stores a lowering decision, its rejected candidates, and their rejection details.
    ///
    /// The supplied candidate and rejection lists are stored without cross-validation.
    #[must_use]
    pub fn new(
        decision: LoweringDecision,
        rejected_candidates: Vec<LoweringCandidate>,
        rejections: Vec<LoweringRejection>,
    ) -> Self {
        Self {
            decision,
            rejected_candidates,
            rejections,
        }
    }

    /// Returns the selected lowering decision.
    #[must_use]
    pub fn decision(&self) -> &LoweringDecision {
        &self.decision
    }

    /// Returns the alternative provider candidates rejected during selection.
    #[must_use]
    pub fn rejected_candidates(&self) -> &[LoweringCandidate] {
        &self.rejected_candidates
    }

    /// Returns the recorded reasons for rejecting alternative candidates.
    #[must_use]
    pub fn rejections(&self) -> &[LoweringRejection] {
        &self.rejections
    }
}

impl LoweringDecision {
    pub(crate) fn of(provider: CudaGraphProvider, operation: Operation) -> Self {
        match provider {
            CudaGraphProvider::Alias => Self::Alias { operation },
            CudaGraphProvider::Kernel => Self::Kernel { operation },
            CudaGraphProvider::CudnnBackend => Self::CudnnBackend { operation },
            CudaGraphProvider::CublasLt => Self::CublasLt { operation },
            CudaGraphProvider::Cublas => Self::Cublas { operation },
            CudaGraphProvider::Cutensor => Self::Cutensor { operation },
            CudaGraphProvider::Cufft => Self::Cufft { operation },
            CudaGraphProvider::Fusion => Self::Fusion { operation },
        }
    }
}
