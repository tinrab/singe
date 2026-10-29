//! Graph operations and compilation attributes.
//!
//! Human-readable floating-point attributes preserve infinities and NaN payloads as
//! hexadecimal IEEE-754 bit strings. Finite values remain numeric.

// Imported operations retain defaults from the graph that owned each optional native attribute.
macro_rules! impl_inherited_compute_type {
    ($($config:ty),+ $(,)?) => {$(
        impl $config {
            pub(crate) fn inherit_compute_type(
                &mut self,
                inherited: Option<$crate::data_type::DataType>,
            ) {
                self.compute_type = self.compute_type.or(inherited);
            }
        }
    )+};
}

mod attention;
mod attention_request;
mod attention_workspace;
mod backward_request;
mod block_scale;
mod convolution;
mod matmul;
mod moe;
mod node;
mod normalization;
mod paged_cache;
mod pointwise;
mod policy;
mod reduction;
mod remap;
mod resample;
mod rng;
mod rope;
mod softmax;
mod stats;
mod tensor_ops;
mod visit;

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::backend::tensor::TensorId;
pub use crate::frontend::operation::{
    attention::{
        AlibiBias, AttentionBackwardAuxGradients, AttentionBackwardConfig, AttentionConfig,
        AttentionDiagonalAlignment, AttentionDiagonalBand, AttentionDropoutConfig,
        AttentionDropoutMask, AttentionDropoutMode, AttentionImplementation, AttentionMaskMode,
        AttentionMaxTotalSequenceLengths, AttentionPagedCache, AttentionPagedKvCache,
        AttentionPrimitiveOperation, AttentionScoreConfig, AttentionSequenceInputs,
        DiagonalBandMaskBound, DiagonalBandMaskConfig, DirectSdpaBackwardConfig, DirectSdpaDropout,
        DirectSdpaForwardConfig, SdpaBackwardSink, SdpaBackwardTensors, SdpaConfig,
        SdpaForwardQuantization, SdpaForwardTensors, SdpaFusedMaskMode, SdpaOperation,
        SdpaScoreModifiers, SdpaScoreSubgraph, SequenceLength, SoftmaxStatsBase,
    },
    attention_request::{AttentionForwardInputs, AttentionForwardRequest},
    attention_workspace::AttentionBackwardWorkspaceLimit,
    backward_request::AttentionBackwardRequest,
    block_scale::{BlockScaleDequantizeConfig, BlockScaleOperation, BlockScaleQuantizeConfig},
    convolution::{ConvolutionConfig, ConvolutionOperation},
    matmul::{MatmulConfig, MatmulOperation},
    moe::{
        MoeGroupedMatmulBackwardConfig, MoeGroupedMatmulConfig, MoeGroupedMatmulOperation,
        MoeGroupedMatmulRouting,
    },
    node::Operation,
    normalization::{
        AdaptiveLayerNormalizationBackwardInputs, AdaptiveLayerNormalizationBackwardOutputs,
        AdaptiveLayerNormalizationConfig, AdaptiveLayerNormalizationInputs,
        AdaptiveLayerNormalizationOutputs, BatchNormalizationBackwardConfig,
        BatchNormalizationBackwardInputs, BatchNormalizationBackwardOutputs,
        BatchNormalizationBackwardWeightsConfig, BatchNormalizationBackwardWeightsInputs,
        BatchNormalizationBackwardWeightsOutputs, BatchNormalizationConfig,
        BatchNormalizationFinalizeConfig, BatchNormalizationFinalizeInputs,
        BatchNormalizationFinalizeOutputs, BatchNormalizationInferenceInputs,
        BatchNormalizationInputs, BatchNormalizationOutputs, BatchNormalizationRunningStatistics,
        BatchNormalizationRunningStats, InstanceNormalizationBackwardInputs,
        InstanceNormalizationBackwardOutputs, InstanceNormalizationConfig,
        InstanceNormalizationInputs, InstanceNormalizationOutputs, LayerNormalizationAffine,
        LayerNormalizationAffineGradients, LayerNormalizationBackwardConfig,
        LayerNormalizationBackwardInputs, LayerNormalizationBackwardOutputs,
        LayerNormalizationConfig, LayerNormalizationInputs, LayerNormalizationOutputs,
        NormalizationConfig, NormalizationOperation, NormalizationSavedStatistics,
        NormalizationStatistics, RmsNormalizationAffineGradients, RmsNormalizationBackwardConfig,
        RmsNormalizationBackwardInputs, RmsNormalizationBackwardOutputs, RmsNormalizationConfig,
        RmsNormalizationInputs, RmsNormalizationOutputs, RmsNormalizationStatistics,
    },
    paged_cache::{
        PagedCacheLayout, PagedCacheLoadConfig, PagedCacheLoadOperation, PagedCacheLoadTensors,
    },
    pointwise::{Activation, PointwiseOperation},
    reduction::{ReductionAxes, ReductionConfig, ReductionOperation, ReductionOutput},
    resample::{ResampleBackwardInputs, ResampleConfig, ResampleOperation},
    rng::{
        RandomNumberDistribution, RandomNumberGeneratorConfig, RandomNumberGeneratorOperation,
        RandomNumberSeedSource,
    },
    rope::{RopeBackwardInputs, RopeConfig, RopeInputs, RopeOperation},
    softmax::SoftmaxConfig,
    stats::{GenStatsConfig, GenStatsOperation},
    tensor_ops::{
        ConcatConfig, ConcatInPlaceMode, ReshapeConfig, ReshapeMode, SliceConfig, TensorOperation,
        TransposeConfig,
    },
};
pub(crate) use crate::frontend::operation::{
    attention::{SdpaAuxOutputRequest, SdpaQuantizedAuxOutputRequest, SdpaSoftmaxAuxOutputRequest},
    backward_request::AttentionBackwardHeuristicOverride,
    visit::{TensorRole, TensorVisitor},
};

/// Access performed by native operations, independent of storage ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorAccess {
    /// Reads tensor values without writing them.
    Read,
    /// Writes tensor values without reading prior contents.
    Write,
    /// Reads and writes tensor values.
    ReadWrite,
}

impl TensorAccess {
    /// Returns whether this access includes reading tensor values.
    #[must_use]
    pub fn reads(self) -> bool {
        self != Self::Write
    }

    /// Returns whether this access includes writing tensor values.
    #[must_use]
    pub fn writes(self) -> bool {
        self != Self::Read
    }

    pub(crate) fn merge(self, other: Self) -> Self {
        if self == other { self } else { Self::ReadWrite }
    }
}

/// One operation's read/write dependencies and addressing metadata.
#[derive(Debug, Default, Clone)]
pub(crate) struct TensorUses {
    pub reads: BTreeSet<TensorId>,
    pub writes: BTreeSet<TensorId>,
    pub indexing: BTreeSet<TensorId>,
}

impl TensorUses {
    pub fn read(&mut self, ids: impl IntoIterator<Item = TensorId>) {
        self.reads.extend(ids);
    }

    pub fn write(&mut self, ids: impl IntoIterator<Item = TensorId>) {
        self.writes.extend(ids);
    }

    /// Marks index, offset and extent tensors as reads that control memory addressing.
    pub fn read_indices(&mut self, ids: impl IntoIterator<Item = TensorId>) {
        for id in ids {
            self.reads.insert(id);
            self.indexing.insert(id);
        }
    }
}
