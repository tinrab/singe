use serde::{Deserialize, Serialize};

use crate::{
    backend::execution::attention::SdpaSubgraph,
    error::Result,
    frontend::{
        graph::{DataTypePolicy, ValidatedGraph},
        lower::{LoweredOperation, LoweringContext},
        operation::{
            AttentionBackwardRequest, AttentionForwardRequest, AttentionPrimitiveOperation,
            BlockScaleOperation, ConvolutionOperation, GenStatsOperation, MatmulOperation,
            MoeGroupedMatmulOperation, NormalizationOperation, PagedCacheLoadOperation,
            PointwiseOperation, RandomNumberGeneratorOperation, ReductionOperation,
            ResampleOperation, RopeOperation, SdpaOperation, TensorOperation, TensorUses,
        },
    },
};

/// Serializable frontend operation node.
///
/// Each variant records tensor IDs and operation-specific attributes for one
/// node in the declarative cuDNN frontend graph. The graph is later lowered into
/// backend descriptors and planned with cuDNN engines.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "operation")]
#[non_exhaustive]
#[expect(
    clippy::large_enum_variant,
    reason = "graph nodes are built once per graph; boxing variants would only complicate matching"
)]
pub enum Operation {
    /// Matrix multiplication with explicit tensor identities and compute settings.
    Matmul(MatmulOperation),
    /// Forward, input-gradient, or filter-gradient convolution.
    Convolution(ConvolutionOperation),
    /// Unary, binary, ternary, or activation-gradient pointwise computation.
    Pointwise(PointwiseOperation),
    /// Ordinary or user-defined reduction.
    Reduction(ReductionOperation),
    /// Tensor reshape, slice, transpose, or concatenation.
    Tensor(TensorOperation),
    /// Forward or backward rotary position embedding.
    Rope(RopeOperation),
    /// Pooling or interpolation and their backward operations.
    Resample(ResampleOperation),
    /// Random generation with a distribution and seed/offset request.
    RandomNumberGenerator(RandomNumberGeneratorOperation),
    /// Per-channel sums and sums of squares.
    GenStats(GenStatsOperation),
    /// Block quantization or dequantization.
    BlockScale(BlockScaleOperation),
    /// Normalization forward, backward, or statistics-finalization stage.
    Normalization(NormalizationOperation),
    /// Loads a logical tensor from a paged cache.
    PagedCacheLoad(PagedCacheLoadOperation),
    /// An explicit attention mask or softmax primitive.
    AttentionPrimitive(AttentionPrimitiveOperation),
    /// Direct native scaled dot product attention forward or backward.
    Sdpa(SdpaOperation),
    /// Deferred attention-forward request expanded for the compilation target.
    AttentionForward(AttentionForwardRequest),
    /// Deferred attention-backward request expanded for the compilation target.
    AttentionBackward(AttentionBackwardRequest),
    /// Expert-grouped matmul or grouped weight gradients.
    MoeGroupedMatmul(MoeGroupedMatmulOperation),
}

impl Operation {
    /// Resolves access requirements against the operation's graph precision policy.
    pub(crate) fn tensor_uses(&self, policy: DataTypePolicy) -> TensorUses {
        let mut uses = self.direct_tensor_uses();
        if let Self::Convolution(operation) = self {
            uses.read(operation.reads_destination(policy));
        }
        for subgraph in self.score_subgraphs() {
            uses.read(subgraph.captured_tensor_ids());
        }
        uses
    }

    /// Lowers this node into a native descriptor.
    /// SDPA receives its prepared score subgraph and custom reductions receive their validated body.
    pub(crate) fn lower(
        &self,
        context: &LoweringContext<'_>,
        subgraph: Option<SdpaSubgraph<'_>>,
        reducer: Option<ValidatedGraph>,
    ) -> Result<LoweredOperation> {
        match self {
            Self::Matmul(operation) => operation.lower(context),
            Self::Convolution(operation) => operation.lower(context),
            Self::Pointwise(operation) => operation.lower(context),
            Self::Reduction(operation) => operation.lower(context, reducer),
            Self::Tensor(operation) => operation.lower(context),
            Self::Rope(operation) => operation.lower(context),
            Self::Resample(operation) => operation.lower(context),
            Self::RandomNumberGenerator(operation) => operation.lower(context),
            Self::GenStats(operation) => operation.lower(context),
            Self::BlockScale(operation) => operation.lower(context),
            Self::Normalization(operation) => operation.lower(context),
            Self::PagedCacheLoad(operation) => operation.lower(context),
            Self::AttentionPrimitive(operation) => operation.lower(context),
            Self::Sdpa(operation) => operation.lower(context, subgraph),
            Self::AttentionForward(operation) => operation.lower(context),
            Self::AttentionBackward(operation) => operation.lower(context),
            Self::MoeGroupedMatmul(operation) => operation.lower(context),
        }
    }
}
