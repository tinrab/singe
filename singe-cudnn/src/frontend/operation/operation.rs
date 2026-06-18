use serde::{Deserialize, Serialize};

use crate::{
    frontend::operation::{
        AdaptiveLayerNormalizationBackwardConfig, AdaptiveLayerNormalizationConfig,
        BatchNormalizationBackwardConfig, BatchNormalizationConfig,
        BatchNormalizationFinalizeConfig, BatchNormalizationInferenceConfig,
        BatchNormalizationRunningStats, BlockScaleDequantizeConfig, BlockScaleQuantizeConfig,
        ConvolutionConfig, DbnWeightConfig, DirectSdpaBackwardConfig, DirectSdpaForwardConfig,
        GenStatsConfig, InstanceNormalizationBackwardConfig, InstanceNormalizationConfig,
        LayerNormalizationBackwardConfig, LayerNormalizationConfig, MatmulConfig, MatmulFp8Config,
        MoeGroupedMatmulBackwardConfig, MoeGroupedMatmulConfig, PointwiseOperation,
        RandomNumberGeneratorConfig, ReductionOperation, ResampleConfig,
        RmsNormalizationBackwardConfig, RmsNormalizationConfig, SdpaScoreModifiers,
        SdpaScoreSubgraph,
    },
    pointwise::PointwiseMode,
    tensor::TensorId,
};

/// Concat in-place aliasing contract.
///
/// `None` means the output is independent from the inputs. `Input(index)`
/// requests cuDNN concat in-place mode using the given input as the output
/// storage source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ConcatInPlaceMode {
    #[default]
    None,
    Input(i64),
}

impl ConcatInPlaceMode {
    pub fn from_index(index: Option<i64>) -> Self {
        match index {
            Some(index) => Self::Input(index),
            None => Self::None,
        }
    }

    pub fn index(self) -> Option<i64> {
        match self {
            Self::None => None,
            Self::Input(index) => Some(index),
        }
    }
}

/// Serializable frontend operation node.
///
/// Each variant records tensor IDs and operation-specific attributes for one
/// node in the declarative cuDNN frontend graph. The graph is later lowered into
/// backend descriptors and planned with cuDNN engines.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "operation")]
#[non_exhaustive]
pub enum Operation {
    Matmul {
        a: TensorId,
        b: TensorId,
        c: TensorId,
        config: MatmulConfig,
    },
    MatmulFp8 {
        a: TensorId,
        b: TensorId,
        descale_a: TensorId,
        descale_b: TensorId,
        scale_c: TensorId,
        c: TensorId,
        absolute_max_c: TensorId,
        config: MatmulFp8Config,
    },
    ConvolutionForward {
        x: TensorId,
        w: TensorId,
        y: TensorId,
        config: ConvolutionConfig,
    },
    ConvolutionBackwardData {
        w: TensorId,
        dy: TensorId,
        dx: TensorId,
        config: ConvolutionConfig,
    },
    ConvolutionBackwardFilter {
        x: TensorId,
        dy: TensorId,
        dw: TensorId,
        config: ConvolutionConfig,
    },
    Pointwise(PointwiseOperation),
    Reduction(ReductionOperation),
    Reshape {
        input: TensorId,
        output: TensorId,
    },
    Slice {
        input: TensorId,
        output: TensorId,
        starts: Vec<i64>,
        limits: Vec<i64>,
        strides: Vec<i64>,
        byte_offset: i64,
    },
    Transpose {
        input: TensorId,
        output: TensorId,
        permutation: Vec<i64>,
    },
    Concat {
        inputs: Vec<TensorId>,
        output: TensorId,
        axis: i64,
        in_place: ConcatInPlaceMode,
    },
    Resample {
        input: TensorId,
        output: TensorId,
        indices: Option<TensorId>,
        config: ResampleConfig,
    },
    ResampleBackward {
        input: TensorId,
        output: TensorId,
        output_gradient: TensorId,
        input_gradient: TensorId,
        indices: Option<TensorId>,
        config: ResampleConfig,
    },
    RandomNumberGenerator {
        output: TensorId,
        config: RandomNumberGeneratorConfig,
    },
    GenStats {
        input: TensorId,
        sum: TensorId,
        sq_sum: TensorId,
        config: GenStatsConfig,
    },
    BlockScaleQuantize {
        input: TensorId,
        output: TensorId,
        scale: TensorId,
        config: BlockScaleQuantizeConfig,
    },
    BlockScaleDequantize {
        input: TensorId,
        scale: TensorId,
        output: TensorId,
        config: BlockScaleDequantizeConfig,
    },
    LayerNormalization {
        input: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        config: LayerNormalizationConfig,
    },
    RmsNormalization {
        input: TensorId,
        scale: TensorId,
        bias: Option<TensorId>,
        output: TensorId,
        inv_variance: TensorId,
        config: RmsNormalizationConfig,
    },
    LayerNormalizationBackward {
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: TensorId,
        dx: TensorId,
        config: LayerNormalizationBackwardConfig,
    },
    RmsNormalizationBackward {
        input: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: Option<TensorId>,
        dx: TensorId,
        config: RmsNormalizationBackwardConfig,
    },
    InstanceNormalization {
        input: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        config: InstanceNormalizationConfig,
    },
    InstanceNormalizationBackward {
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: TensorId,
        dx: TensorId,
        config: InstanceNormalizationBackwardConfig,
    },
    BatchNormalization {
        input: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        config: BatchNormalizationConfig,
    },
    BatchNormalizationInference {
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        config: BatchNormalizationInferenceConfig,
    },
    BatchNormalizationFinalize {
        sum: TensorId,
        sq_sum: TensorId,
        scale: TensorId,
        bias: TensorId,
        next_running_mean: Option<TensorId>,
        next_running_var: Option<TensorId>,
        saved_mean: TensorId,
        saved_inv_variance: TensorId,
        eq_scale: TensorId,
        eq_bias: TensorId,
        accum_count: TensorId,
        config: BatchNormalizationFinalizeConfig,
    },
    BatchNormalizationBackward {
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: TensorId,
        dx: TensorId,
        config: BatchNormalizationBackwardConfig,
    },
    DbnWeight {
        dy: TensorId,
        input: TensorId,
        scale: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dscale: TensorId,
        bias_gradient: TensorId,
        eq_bias: TensorId,
        eq_scale_dy: TensorId,
        eq_scale_x: TensorId,
        config: DbnWeightConfig,
    },
    AdaptiveLayerNormalization {
        input: TensorId,
        scale: TensorId,
        bias: Option<TensorId>,
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        config: AdaptiveLayerNormalizationConfig,
    },
    AdaptiveLayerNormalizationBackward {
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: Option<TensorId>,
        dx: TensorId,
        config: AdaptiveLayerNormalizationBackwardConfig,
    },
    PagedCacheLoad {
        container: TensorId,
        output: TensorId,
        sequence: TensorId,
        page_table: TensorId,
    },
    DiagonalBandMask {
        x: TensorId,
        b: TensorId,
        y: TensorId,
        comparison_mode: PointwiseMode,
        sequence_length_query: Option<TensorId>,
        sequence_length_key_value: Option<TensorId>,
        left_bound: Option<TensorId>,
        shift_right_bound: Option<TensorId>,
    },
    Softmax {
        x: TensorId,
        y: TensorId,
        stats: Option<TensorId>,
        max: Option<TensorId>,
        sum_exp: Option<TensorId>,
        sink: Option<TensorId>,
    },
    SdpaForward {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        o: TensorId,
        scale: TensorId,
        stats: Option<TensorId>,
        logit_max: Option<TensorId>,
        score_sum_exp: Option<TensorId>,
        softmax_p: Option<TensorId>,
        softmax_s: Option<TensorId>,
        sink: Option<TensorId>,
        score_modifiers: SdpaScoreModifiers,
        score_subgraph: Box<Option<SdpaScoreSubgraph>>,
        config: Box<DirectSdpaForwardConfig>,
    },
    SdpaBackward {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        o: TensorId,
        stats: TensorId,
        scale: TensorId,
        d_o: TensorId,
        d_q: TensorId,
        d_k: TensorId,
        d_v: TensorId,
        sink_gradient: Option<TensorId>,
        config: Box<DirectSdpaBackwardConfig>,
    },
    MoeGroupedMatmul {
        token: TensorId,
        weight: TensorId,
        first_token_offset: TensorId,
        output: TensorId,
        config: MoeGroupedMatmulConfig,
    },
    MoeGroupedMatmulBackward {
        output_gradient: TensorId,
        token: TensorId,
        first_token_offset: TensorId,
        weight_gradient: TensorId,
        config: MoeGroupedMatmulBackwardConfig,
    },
}

impl Operation {
    pub(crate) fn tensor_ids(&self) -> Vec<TensorId> {
        match self {
            Operation::Matmul { a, b, c, config } => {
                let mut tensors = vec![*a, *b, *c];
                tensors.extend(config.m_override());
                tensors.extend(config.k_override());
                tensors
            }
            Operation::MatmulFp8 {
                a,
                b,
                descale_a,
                descale_b,
                scale_c,
                c,
                absolute_max_c,
                config,
            } => {
                let mut tensors = vec![
                    *a,
                    *b,
                    *descale_a,
                    *descale_b,
                    *scale_c,
                    *c,
                    *absolute_max_c,
                ];
                tensors.extend(config.m_override());
                tensors.extend(config.k_override());
                tensors
            }
            Operation::ConvolutionForward { x, w, y, .. }
            | Operation::ConvolutionBackwardData {
                w: x, dy: w, dx: y, ..
            }
            | Operation::ConvolutionBackwardFilter {
                x, dy: w, dw: y, ..
            } => vec![*x, *w, *y],
            Operation::Pointwise(op) => match op {
                PointwiseOperation::Unary { input, output, .. } => vec![*input, *output],
                PointwiseOperation::ReluForward { input, output, .. } => vec![*input, *output],
                PointwiseOperation::Binary {
                    lhs, rhs, output, ..
                } => vec![*lhs, *rhs, *output],
                PointwiseOperation::Ternary {
                    x, b, t, output, ..
                } => vec![*x, *b, *t, *output],
            },
            Operation::Reduction(op) => match op {
                ReductionOperation::Reduce { input, output, .. } => vec![*input, *output],
            },
            Operation::Reshape { input, output } => vec![*input, *output],
            Operation::Slice { input, output, .. } => vec![*input, *output],
            Operation::Transpose { input, output, .. } => vec![*input, *output],
            Operation::Concat { inputs, output, .. } => {
                let mut tensors = inputs.clone();
                tensors.push(*output);
                tensors
            }
            Operation::Resample {
                input,
                output,
                indices,
                ..
            } => {
                let mut tensors = vec![*input, *output];
                tensors.extend(*indices);
                tensors
            }
            Operation::ResampleBackward {
                input,
                output,
                output_gradient,
                input_gradient,
                indices,
                ..
            } => {
                let mut tensors = vec![*input, *output, *output_gradient, *input_gradient];
                tensors.extend(*indices);
                tensors
            }
            Operation::RandomNumberGenerator { output, config } => {
                let mut tensors = vec![*output];
                tensors.extend(config.seed_tensor());
                tensors.extend(config.offset());
                tensors
            }
            Operation::GenStats {
                input, sum, sq_sum, ..
            } => vec![*input, *sum, *sq_sum],
            Operation::BlockScaleQuantize {
                input,
                output,
                scale,
                ..
            }
            | Operation::BlockScaleDequantize {
                input,
                scale,
                output,
                ..
            } => vec![*input, *scale, *output],
            Operation::LayerNormalization {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            } => {
                let mut tensors = vec![*input, *scale, *bias, *output, *mean, *inv_variance];
                tensors.push(config.epsilon());
                tensors
            }
            Operation::InstanceNormalization {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            } => {
                let mut tensors = vec![*input, *scale, *bias, *output, *mean, *inv_variance];
                tensors.push(config.epsilon());
                tensors
            }
            Operation::BatchNormalization {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            } => {
                let mut tensors = vec![*input, *scale, *bias, *output, *mean, *inv_variance];
                tensors.push(config.epsilon());
                if let Some(BatchNormalizationRunningStats {
                    momentum,
                    prev_mean,
                    prev_var,
                    next_mean,
                    next_var,
                }) = config.running()
                {
                    tensors.extend([momentum, prev_mean, prev_var, next_mean, next_var]);
                }
                tensors.extend(config.peer_stats().iter().copied());
                tensors
            }
            Operation::RmsNormalization {
                input,
                scale,
                bias,
                output,
                inv_variance,
                config,
            } => {
                let mut tensors = vec![*input, *scale, *output, *inv_variance];
                tensors.extend(*bias);
                tensors.push(config.epsilon());
                tensors
            }
            Operation::LayerNormalizationBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            } => {
                let mut tensors = vec![
                    *input,
                    *mean,
                    *inv_variance,
                    *dy,
                    *scale,
                    *dscale,
                    *bias_gradient,
                    *dx,
                ];
                tensors.push(config.epsilon());
                tensors
            }
            Operation::InstanceNormalizationBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            } => {
                let mut tensors = vec![
                    *input,
                    *mean,
                    *inv_variance,
                    *dy,
                    *scale,
                    *dscale,
                    *bias_gradient,
                    *dx,
                ];
                tensors.push(config.epsilon());
                tensors
            }
            Operation::BatchNormalizationBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            } => {
                let mut tensors = vec![
                    *input,
                    *mean,
                    *inv_variance,
                    *dy,
                    *scale,
                    *dscale,
                    *bias_gradient,
                    *dx,
                ];
                tensors.push(config.epsilon());
                tensors.extend(config.peer_stats().iter().copied());
                tensors
            }
            Operation::RmsNormalizationBackward {
                input,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config: _,
            } => {
                let mut tensors = vec![*input, *inv_variance, *dy, *scale, *dscale, *dx];
                tensors.extend(*bias_gradient);
                tensors
            }
            Operation::BatchNormalizationInference {
                input,
                mean,
                inv_variance,
                scale,
                bias,
                output,
                config,
            } => {
                let mut tensors = vec![*input, *mean, *inv_variance, *scale, *bias, *output];
                tensors.push(config.epsilon());
                tensors
            }
            Operation::BatchNormalizationFinalize {
                sum,
                sq_sum,
                scale,
                bias,
                next_running_mean,
                next_running_var,
                saved_mean,
                saved_inv_variance,
                eq_scale,
                eq_bias,
                accum_count,
                config,
            } => {
                let mut tensors = vec![
                    *sum,
                    *sq_sum,
                    *scale,
                    *bias,
                    *saved_mean,
                    *saved_inv_variance,
                    *eq_scale,
                    *eq_bias,
                    *accum_count,
                    config.epsilon(),
                ];
                tensors.extend(config.prev_running_mean());
                tensors.extend(config.prev_running_var());
                tensors.extend(*next_running_mean);
                tensors.extend(*next_running_var);
                tensors.extend(config.momentum());
                tensors
            }
            Operation::DbnWeight {
                dy,
                input,
                scale,
                mean,
                inv_variance,
                dscale,
                bias_gradient,
                eq_bias,
                eq_scale_dy,
                eq_scale_x,
                ..
            } => vec![
                *dy,
                *input,
                *scale,
                *mean,
                *inv_variance,
                *dscale,
                *bias_gradient,
                *eq_bias,
                *eq_scale_dy,
                *eq_scale_x,
            ],
            Operation::AdaptiveLayerNormalization {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            } => {
                let mut tensors = vec![*input, *scale, *output, *mean, *inv_variance];
                tensors.extend(*bias);
                tensors.push(config.epsilon());
                tensors
            }
            Operation::AdaptiveLayerNormalizationBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config: _,
            } => {
                let mut tensors = vec![*input, *mean, *inv_variance, *dy, *scale, *dscale, *dx];
                tensors.extend(*bias_gradient);
                tensors
            }
            Operation::PagedCacheLoad {
                container,
                output,
                sequence,
                page_table,
            } => vec![*container, *output, *sequence, *page_table],
            Operation::DiagonalBandMask {
                x,
                b,
                y,
                sequence_length_query,
                sequence_length_key_value,
                left_bound,
                shift_right_bound,
                ..
            } => {
                let mut tensors = vec![*x, *b, *y];
                tensors.extend(*sequence_length_query);
                tensors.extend(*sequence_length_key_value);
                tensors.extend(*left_bound);
                tensors.extend(*shift_right_bound);
                tensors
            }
            Operation::Softmax {
                x,
                y,
                stats,
                max,
                sum_exp,
                sink,
            } => {
                let mut tensors = vec![*x, *y];
                tensors.extend(*stats);
                tensors.extend(*max);
                tensors.extend(*sum_exp);
                tensors.extend(*sink);
                tensors
            }
            Operation::SdpaForward {
                q,
                k,
                v,
                o,
                scale,
                stats,
                logit_max,
                score_sum_exp,
                softmax_p,
                softmax_s,
                sink,
                score_modifiers,
                score_subgraph: _,
                config,
            } => {
                let mut tensors = vec![*q, *k, *v, *o, *scale];
                tensors.extend(*stats);
                tensors.extend(*logit_max);
                tensors.extend(*score_sum_exp);
                tensors.extend(*softmax_p);
                tensors.extend(*softmax_s);
                tensors.extend(*sink);
                tensors.extend(score_modifiers.bias);
                tensors.extend(score_modifiers.alibi_slopes);
                tensors.extend(score_modifiers.additive_mask);
                tensors.extend(score_modifiers.dropout_mask);
                tensors.extend(score_modifiers.dropout_scale);
                tensors.extend(config.sequence_length_query());
                tensors.extend(config.sequence_length_key_value());
                tensors.extend(config.page_table_k());
                tensors.extend(config.page_table_v());
                tensors.extend(config.block_mask());
                tensors.extend(config.dropout_seed());
                tensors.extend(config.dropout().and_then(|dropout| dropout.offset()));
                tensors.extend(config.random_number_generator_dump());
                tensors
            }
            Operation::SdpaBackward {
                q,
                k,
                v,
                o,
                stats,
                scale,
                d_o,
                d_q,
                d_k,
                d_v,
                sink_gradient,
                config,
            } => {
                let mut tensors = vec![*q, *k, *v, *o, *stats, *scale, *d_o, *d_q, *d_k, *d_v];
                tensors.extend(config.sequence_length_query());
                tensors.extend(config.sequence_length_key_value());
                tensors.extend(config.sink());
                tensors.extend(*sink_gradient);
                tensors
            }
            Operation::MoeGroupedMatmul {
                token,
                weight,
                first_token_offset,
                output,
                config,
            } => {
                let mut tensors = vec![*token, *weight, *first_token_offset, *output];
                tensors.extend(config.token_index());
                tensors.extend(config.token_ks());
                tensors
            }
            Operation::MoeGroupedMatmulBackward {
                output_gradient,
                token,
                first_token_offset,
                weight_gradient,
                ..
            } => vec![
                *output_gradient,
                *token,
                *first_token_offset,
                *weight_gradient,
            ],
        }
    }
}
