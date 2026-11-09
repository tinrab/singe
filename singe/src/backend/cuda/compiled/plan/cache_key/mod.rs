mod create;
mod dynamic;
mod static_input;
mod summary;

use std::hash::Hash;

use self::static_input::CudaStaticInputIdentityKey;

pub use self::{
    dynamic::{
        ConcreteShapeBucket, ConcreteShapeBucketDimension, DynamicShapeMode, DynamicShapePolicy,
        SymbolicDimensionKey,
    },
    summary::GraphCacheKeySummary,
};
use crate::{
    backend::cuda::compiled::plan::{
        lowering::{LoweringCandidate, LoweringPolicy},
        pipeline::{CompilePipeline, CompileStageVersion},
    },
    graph::{GraphFingerprint, NumericalPolicy},
};

const CUDA_GRAPH_CACHE_KEY_VERSION: u64 = 1;
const CUDA_BACKEND_FEATURE_VERSION: &str = "cuda_13_4";

/// CUDA-specific compiled graph cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphCacheKey {
    numerics: NumericalPolicy,
    graph: GraphFingerprint,
    pipeline: CompilePipeline,
    compile_stage_versions: Vec<CompileStageVersion>,
    lowering_policy: LoweringPolicy,
    lowering_candidates: Vec<(usize, Vec<LoweringCandidate>)>,
    dynamic_shape_policy: DynamicShapePolicy,
    static_input_identities: Vec<CudaStaticInputIdentityKey>,
    symbolic_dimensions: Vec<SymbolicDimensionKey>,
    shape_bucket: Option<ConcreteShapeBucket>,
    backend_feature_version: &'static str,
    compiler_version_salt: u64,
}

impl GraphCacheKey {
    /// Returns whether this key is suitable for cross-graph cache reuse.
    #[must_use]
    pub fn is_cacheable(&self) -> bool {
        self.graph.is_cacheable()
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

    /// Returns a public diagnostic summary of this cache key.
    #[must_use]
    pub fn summary(&self) -> GraphCacheKeySummary {
        GraphCacheKeySummary::new(
            self.graph,
            self.compile_stage_versions.clone(),
            self.lowering_policy.clone(),
            self.lowering_candidates.clone(),
            self.is_cacheable(),
            self.dynamic_shape_policy.clone(),
            self.symbolic_dimensions.clone(),
            self.shape_bucket.clone(),
            self.numerics,
        )
    }

    /// Returns whether two keys belong to the same symbolic dynamic-shape recording group.
    #[cfg(test)]
    #[must_use]
    pub fn same_dynamic_recording_group(&self, other: &Self) -> bool {
        self.graph == other.graph
            && self.numerics == other.numerics
            && self.pipeline == other.pipeline
            && self.compile_stage_versions == other.compile_stage_versions
            && self.lowering_policy == other.lowering_policy
            && self.lowering_candidates == other.lowering_candidates
            && self.dynamic_shape_policy == other.dynamic_shape_policy
            && self.static_input_identities == other.static_input_identities
            && self.symbolic_dimensions == other.symbolic_dimensions
            && self.backend_feature_version == other.backend_feature_version
            && self.compiler_version_salt == other.compiler_version_salt
    }
}
