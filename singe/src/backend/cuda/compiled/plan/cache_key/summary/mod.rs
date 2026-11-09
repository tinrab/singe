use crate::{
    backend::cuda::compiled::plan::{
        cache_key::{ConcreteShapeBucket, DynamicShapePolicy, SymbolicDimensionKey},
        lowering::{LoweringCandidate, LoweringPolicy},
        pipeline::CompileStageVersion,
    },
    graph::{ComputePrecision, Determinism, GraphFingerprint, NumericalPolicy},
};

/// Public diagnostic summary for a CUDA graph cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphCacheKeySummary {
    numerics: NumericalPolicy,
    /// Backend-neutral graph fingerprint component.
    graph: GraphFingerprint,
    /// CUDA compile stage versions included in this key.
    compile_stage_versions: Vec<CompileStageVersion>,
    /// CUDA lowering policy included in this key.
    lowering_policy: LoweringPolicy,
    /// Provider candidate material included in this key.
    lowering_candidates: Vec<(usize, Vec<LoweringCandidate>)>,
    /// Whether this key is suitable for cross-graph cache reuse.
    cacheable: bool,
    /// Dynamic-shape cache policy included in this key.
    dynamic_shape_policy: DynamicShapePolicy,
    /// Symbolic dimension metadata included in this key.
    symbolic_dimensions: Vec<SymbolicDimensionKey>,
    /// Concrete dynamic-shape bucket included in this key, if any.
    shape_bucket: Option<ConcreteShapeBucket>,
}

impl GraphCacheKeySummary {
    pub(crate) fn new(
        graph: GraphFingerprint,
        compile_stage_versions: Vec<CompileStageVersion>,
        lowering_policy: LoweringPolicy,
        lowering_candidates: Vec<(usize, Vec<LoweringCandidate>)>,
        cacheable: bool,
        dynamic_shape_policy: DynamicShapePolicy,
        symbolic_dimensions: Vec<SymbolicDimensionKey>,
        shape_bucket: Option<ConcreteShapeBucket>,
        numerics: NumericalPolicy,
    ) -> Self {
        Self {
            numerics,
            graph,
            compile_stage_versions,
            lowering_policy,
            lowering_candidates,
            cacheable,
            dynamic_shape_policy,
            symbolic_dimensions,
            shape_bucket,
        }
    }

    /// Returns the compute precision policy included in the cache identity.
    #[must_use]
    pub fn compute_precision(&self) -> ComputePrecision {
        self.numerics.precision
    }

    /// Returns the determinism requirement included in the cache identity.
    #[must_use]
    pub fn determinism(&self) -> Determinism {
        self.numerics.determinism
    }

    /// Returns the backend-neutral graph fingerprint component.
    #[must_use]
    pub fn graph(&self) -> GraphFingerprint {
        self.graph
    }

    /// Returns CUDA compile stage versions included in this key.
    #[must_use]
    pub fn compile_stage_versions(&self) -> &[CompileStageVersion] {
        &self.compile_stage_versions
    }

    /// Returns the CUDA lowering policy included in this key.
    #[must_use]
    pub fn lowering_policy(&self) -> &LoweringPolicy {
        &self.lowering_policy
    }

    /// Returns provider candidate material included in this key.
    #[must_use]
    pub fn lowering_candidates(&self) -> &[(usize, Vec<LoweringCandidate>)] {
        &self.lowering_candidates
    }

    /// Returns whether this key is suitable for cross-graph cache reuse.
    #[must_use]
    pub fn is_cacheable(&self) -> bool {
        self.cacheable
    }

    /// Returns the dynamic-shape cache policy included in this key.
    #[must_use]
    pub fn dynamic_shape_policy(&self) -> &DynamicShapePolicy {
        &self.dynamic_shape_policy
    }

    /// Returns symbolic dimension metadata included in this key.
    #[must_use]
    pub fn symbolic_dimensions(&self) -> &[SymbolicDimensionKey] {
        &self.symbolic_dimensions
    }

    /// Returns the concrete dynamic-shape bucket included in this key, if any.
    #[must_use]
    pub fn shape_bucket(&self) -> Option<&ConcreteShapeBucket> {
        self.shape_bucket.as_ref()
    }
}

#[cfg(test)]
mod tests;
