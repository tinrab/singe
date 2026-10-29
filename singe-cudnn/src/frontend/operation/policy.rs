use crate::frontend::{
    graph::DataTypePolicy,
    operation::{
        BlockScaleOperation, ConvolutionOperation, MoeGroupedMatmulOperation,
        NormalizationOperation, Operation, PointwiseOperation, ReductionOperation,
        ResampleOperation, SdpaOperation,
    },
};

impl Operation {
    /// Preserves the defaults of an imported graph before its operations join another graph.
    /// Independent custom reducer bodies retain their own graph contexts.
    pub(crate) fn inherit_data_type_policy(&mut self, policy: DataTypePolicy) {
        let policy = match self {
            Self::Sdpa(SdpaOperation::Forward { config, .. }) => {
                config.data_type_policy().with_defaults(policy)
            }
            Self::Sdpa(SdpaOperation::Backward { config, .. }) => {
                config.data_type_policy().with_defaults(policy)
            }
            _ => policy,
        };
        for subgraph in self.score_subgraphs_mut() {
            subgraph.inherit_data_type_policy(policy);
        }
        let compute = policy.compute();
        match self {
            Self::Matmul(operation) => operation.config.inherit_compute_type(compute),
            Self::Convolution(
                ConvolutionOperation::Forward { config, .. }
                | ConvolutionOperation::BackwardData { config, .. }
                | ConvolutionOperation::BackwardFilter { config, .. },
            ) => config.inherit_compute_type(compute),
            Self::Pointwise(operation) => {
                let (PointwiseOperation::Unary { compute_type, .. }
                | PointwiseOperation::ActivationForward { compute_type, .. }
                | PointwiseOperation::ActivationBackward { compute_type, .. }
                | PointwiseOperation::Binary { compute_type, .. }
                | PointwiseOperation::Ternary { compute_type, .. }) = operation;
                *compute_type = compute_type.or(compute);
            }
            Self::Reduction(ReductionOperation::Reduce { compute_type, .. }) => {
                *compute_type = compute_type.or(compute);
            }
            Self::Resample(
                ResampleOperation::Forward { config, .. }
                | ResampleOperation::Backward { config, .. },
            ) => config.inherit_compute_type(compute),
            Self::BlockScale(BlockScaleOperation::Quantize { config, .. }) => {
                config.inherit_compute_type(compute);
            }
            Self::BlockScale(BlockScaleOperation::Dequantize { config, .. }) => {
                config.inherit_compute_type(compute);
            }
            Self::MoeGroupedMatmul(MoeGroupedMatmulOperation::Forward { config, .. }) => {
                config.inherit_compute_type(compute);
            }
            Self::MoeGroupedMatmul(MoeGroupedMatmulOperation::Backward { config, .. }) => {
                config.inherit_compute_type(compute);
            }
            Self::GenStats(operation) => operation.config.inherit_compute_type(compute),
            Self::Normalization(NormalizationOperation::BatchFinalize { config, .. }) => {
                config.inherit_compute_type(compute);
            }
            Self::Normalization(NormalizationOperation::BatchNormalizationBackwardWeights {
                config,
                ..
            }) => {
                config.inherit_compute_type(compute);
            }
            Self::AttentionForward(request) => request.config.inherit_compute_type(compute),
            Self::AttentionBackward(request) => request.config_mut().inherit_compute_type(compute),
            Self::Sdpa(operation) => operation.inherit_data_type_policy(policy),
            Self::Reduction(ReductionOperation::Custom { .. })
            | Self::Tensor(_)
            | Self::Rope(_)
            | Self::RandomNumberGenerator(_)
            | Self::Normalization(_)
            | Self::PagedCacheLoad(_)
            | Self::AttentionPrimitive(_) => {}
        }
    }
}
