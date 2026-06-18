use std::collections::{BTreeMap, HashSet};

use crate::{
    data_type::DataType,
    descriptor::BackendDescriptor,
    error::{Error, Result},
    execution::{
        OperationGraph,
        advanced::{
            BlockScaleDequantizeOperation, BlockScaleQuantizeOperation, DiagonalBandMaskOperation,
            MatmulDescriptor, MatmulOperation, MoeGroupedMatmulOperation, PagedCacheLoadOperation,
            SdpaBackwardConfig, SdpaBackwardOperation, SdpaForwardConfig, SdpaForwardOperation,
            SoftmaxOperation,
        },
        convolution::{
            ConvolutionBackwardDataOperation, ConvolutionBackwardFilterOperation,
            ConvolutionDescriptor, ConvolutionForwardOperation,
        },
        normalization::{
            BatchNormalizationBackwardWeightsOperation, BatchNormalizationFinalizeStatsOperation,
            GenStatsMode, GenStatsOperation, NormalizationBackwardConfig,
            NormalizationBackwardOperation, NormalizationForwardConfig,
            NormalizationForwardOperation,
        },
        pointwise::{PointwiseDescriptor, PointwiseOperation as BackendPointwiseOperation},
        reduction::{ReductionDescriptor, ReductionOperation as BackendReductionOperation},
        resample::{ResampleBackwardOperation, ResampleDescriptor, ResampleForwardOperation},
        rng::{RngDescriptor, RngOperation, RngSeed},
        tensor_ops::{ConcatOperation, ReshapeOperation},
    },
    frontend::{
        graph::{Graph, TensorRecord},
        operation::{
            BatchNormalizationFinalizeConfig, BatchNormalizationRunningStats,
            BlockScaleDequantizeConfig, BlockScaleQuantizeConfig, ConvolutionConfig, Operation,
            PointwiseOperation, RandomNumberDistribution, RandomNumberGeneratorConfig,
            RandomNumberSeedSource, ReductionOperation, ResampleConfig,
        },
        plan::BindingReplacement,
    },
    normalization::{BackendNormalizationForwardPhase, BackendNormalizationMode},
    pointwise::PointwiseMode,
    tensor::{Tensor, TensorId, TensorSpec},
    utility::to_i32,
};

#[derive(Debug)]
pub struct LoweredGraph {
    pub tensors: BTreeMap<TensorId, TensorRecord>,
    pub backend_tensors: BTreeMap<TensorId, Tensor>,
    pub binding_template_ids: Vec<TensorId>,
    pub binding_replacements: Vec<BindingReplacement>,
    pub operations: Vec<LoweredOperation>,
}

impl LoweredGraph {
    pub fn lower(graph: &Graph) -> Result<Self> {
        let mut seen_ids = HashSet::new();
        let tensors = graph
            .tensors
            .iter()
            .map(|(&tensor_ref, tensor)| {
                let id = tensor.id.ok_or(Error::DescriptorMismatch {
                    name: "frontend tensor id".into(),
                })?;
                if !seen_ids.insert(id) {
                    return Err(Error::FrontendTensorIdConflict(id));
                }
                Ok((
                    tensor_ref,
                    TensorRecord {
                        tensor: tensor.clone(),
                        id,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;

        let mut backend_tensors = BTreeMap::new();
        let mut active_tensors = HashSet::new();
        for &tensor_ref in tensors.keys() {
            lower_tensor_at(
                tensor_ref,
                &tensors,
                &mut backend_tensors,
                &mut active_tensors,
            )?;
        }

        let mut used_tensors = graph
            .operations
            .iter()
            .flat_map(OperationSpec::tensor_ids)
            .chain(graph.alias_bindings.iter().map(|binding| binding.source))
            .collect::<HashSet<_>>();
        let seed_tensors = used_tensors.iter().copied().collect::<Vec<_>>();
        for tensor_ref in seed_tensors {
            collect_used_tensors(graph, tensor_ref, &mut used_tensors)?;
        }

        let binding_template_ids = tensors
            .iter()
            .filter(|(tensor_ref, record)| {
                !record.tensor.is_virtual && used_tensors.contains(tensor_ref)
            })
            .map(|(_, record)| record.id)
            .collect();

        let context = LoweringContext {
            tensor_records: &tensors,
            backend_tensors: &backend_tensors,
        };
        let mut binding_replacements = Vec::new();
        let mut operations = Vec::with_capacity(graph.operations.len());
        for operation in &graph.operations {
            match operation.lower(&context)? {
                LoweringOutput::Operation(operation) => operations.push(operation),
                LoweringOutput::BindingReplacement(replacement) => {
                    binding_replacements.push(replacement);
                }
                LoweringOutput::Expanded => {}
            }
        }

        Ok(Self {
            tensors,
            backend_tensors,
            binding_template_ids,
            binding_replacements,
            operations,
        })
    }

    pub(crate) fn tensor_record(&self, tensor: TensorId) -> Result<&TensorRecord> {
        self.tensors
            .get(&tensor)
            .ok_or(Error::FrontendTensorNotFound(tensor))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LoweringContext<'a> {
    tensor_records: &'a BTreeMap<TensorId, TensorRecord>,
    backend_tensors: &'a BTreeMap<TensorId, Tensor>,
}

impl LoweringContext<'_> {
    fn backend_id(&self, tensor: TensorId) -> Result<TensorId> {
        Ok(self
            .tensor_records
            .get(&tensor)
            .ok_or(Error::FrontendTensorNotFound(tensor))?
            .id)
    }
}

#[derive(Debug)]
pub(crate) enum LoweringOutput {
    Operation(LoweredOperation),
    BindingReplacement(BindingReplacement),
    Expanded,
}

pub(crate) trait OperationSpec {
    fn tensor_ids(&self) -> Vec<TensorId>;

    fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweringOutput>;
}

#[derive(Debug)]
pub(crate) enum LoweredOperation {
    Matmul(MatmulOperation),
    ConvolutionForward(ConvolutionForwardOperation),
    ConvolutionBackwardData(ConvolutionBackwardDataOperation),
    ConvolutionBackwardFilter(ConvolutionBackwardFilterOperation),
    Pointwise(BackendPointwiseOperation),
    Reduction(BackendReductionOperation),
    Reshape(ReshapeOperation),
    Concat(ConcatOperation),
    Resample(ResampleForwardOperation),
    ResampleBackward(ResampleBackwardOperation),
    Rng(RngOperation),
    GenStats(GenStatsOperation),
    BlockScaleQuantize(BlockScaleQuantizeOperation),
    BlockScaleDequantize(BlockScaleDequantizeOperation),
    LayerNormalization(NormalizationForwardOperation),
    RmsNormalization(NormalizationForwardOperation),
    LayerNormalizationBackward(NormalizationBackwardOperation),
    RmsNormalizationBackward(NormalizationBackwardOperation),
    InstanceNormalization(NormalizationForwardOperation),
    InstanceNormalizationBackward(NormalizationBackwardOperation),
    BatchNormalization(NormalizationForwardOperation),
    BatchNormalizationInference(NormalizationForwardOperation),
    BatchNormalizationFinalize(BatchNormalizationFinalizeStatsOperation),
    BatchNormalizationBackward(NormalizationBackwardOperation),
    DbnWeight(BatchNormalizationBackwardWeightsOperation),
    AdaptiveLayerNormalization(NormalizationForwardOperation),
    AdaptiveLayerNormalizationBackward(NormalizationBackwardOperation),
    PagedCacheLoad(PagedCacheLoadOperation),
    DiagonalBandMask(DiagonalBandMaskOperation),
    Softmax(SoftmaxOperation),
    SdpaForward(SdpaForwardOperation),
    SdpaBackward(SdpaBackwardOperation),
    MoeGroupedMatmul(MoeGroupedMatmulOperation),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LoweredSdpaSubgraph<'a> {
    pub(crate) operation_graph: &'a OperationGraph,
    pub(crate) input_id: TensorId,
    pub(crate) output_id: TensorId,
}

impl OperationSpec for Operation {
    fn tensor_ids(&self) -> Vec<TensorId> {
        Operation::tensor_ids(self)
    }

    fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweringOutput> {
        let tensors = context.backend_tensors;
        Ok(LoweringOutput::Operation(match self {
            Operation::Matmul { a, b, c, config } => {
                let descriptor = MatmulDescriptor::create(config.compute_type())?;
                LoweredOperation::Matmul(MatmulOperation::create_with_overrides(
                    &descriptor,
                    tensor_at(tensors, *a)?,
                    tensor_at(tensors, *b)?,
                    tensor_at(tensors, *c)?,
                    config
                        .m_override()
                        .map(|tensor| tensor_at(tensors, tensor))
                        .transpose()?,
                    None,
                    config
                        .k_override()
                        .map(|tensor| tensor_at(tensors, tensor))
                        .transpose()?,
                )?)
            }
            Operation::MatmulFp8 { .. } => return Ok(LoweringOutput::Expanded),
            Operation::ConvolutionForward { x, w, y, config } => {
                lower_convolution_forward(tensors, *x, *w, *y, config)?
            }
            Operation::ConvolutionBackwardData { w, dy, dx, config } => {
                lower_convolution_backward_data(tensors, *w, *dy, *dx, config)?
            }
            Operation::ConvolutionBackwardFilter { x, dy, dw, config } => {
                lower_convolution_backward_filter(tensors, *x, *dy, *dw, config)?
            }
            Operation::Pointwise(op) => lower_pointwise_operation(tensors, op)?,
            Operation::Reduction(op) => lower_reduction_operation(tensors, op)?,
            Operation::Reshape { input, output } => {
                lower_reshape_operation(tensors, *input, *output)?
            }
            Operation::Slice {
                input,
                output,
                byte_offset,
                ..
            } => {
                return Ok(LoweringOutput::BindingReplacement(BindingReplacement {
                    source_id: context.backend_id(*input)?,
                    target_id: context.backend_id(*output)?,
                    byte_offset: *byte_offset,
                }));
            }
            Operation::Transpose { input, output, .. } => {
                return Ok(LoweringOutput::BindingReplacement(BindingReplacement {
                    source_id: context.backend_id(*input)?,
                    target_id: context.backend_id(*output)?,
                    byte_offset: 0,
                }));
            }
            Operation::Softmax {
                x,
                y,
                stats,
                max,
                sum_exp,
                sink,
            } => lower_softmax_operation(tensors, *x, *y, *stats, *max, *sum_exp, *sink)?,
            Operation::Concat {
                inputs,
                output,
                axis,
                in_place,
            } => lower_concat_operation(tensors, inputs, *output, *axis, in_place.index())?,
            Operation::Resample {
                input,
                output,
                indices,
                config,
            } => lower_resample_operation(tensors, *input, *output, *indices, config)?,
            Operation::ResampleBackward {
                input,
                output,
                output_gradient,
                input_gradient,
                indices,
                config,
            } => lower_resample_backward_operation(
                tensors,
                *input,
                *output,
                *output_gradient,
                *input_gradient,
                *indices,
                config,
            )?,
            Operation::RandomNumberGenerator { output, config } => {
                lower_rng_operation(tensors, *output, config)?
            }
            Operation::GenStats {
                input,
                sum,
                sq_sum,
                config: _,
            } => lower_gen_stats_operation(tensors, *input, *sum, *sq_sum)?,
            Operation::BlockScaleQuantize {
                input,
                output,
                scale,
                config,
            } => lower_block_scale_quantize_operation(tensors, *input, *output, *scale, config)?,
            Operation::BlockScaleDequantize {
                input,
                scale,
                output,
                config,
            } => lower_block_scale_dequantize_operation(tensors, *input, *scale, *output, config)?,
            Operation::LayerNormalization { .. }
            | Operation::RmsNormalization { .. }
            | Operation::LayerNormalizationBackward { .. }
            | Operation::RmsNormalizationBackward { .. }
            | Operation::InstanceNormalization { .. }
            | Operation::InstanceNormalizationBackward { .. }
            | Operation::BatchNormalization { .. }
            | Operation::BatchNormalizationInference { .. }
            | Operation::BatchNormalizationFinalize { .. }
            | Operation::BatchNormalizationBackward { .. }
            | Operation::DbnWeight { .. }
            | Operation::AdaptiveLayerNormalization { .. }
            | Operation::AdaptiveLayerNormalizationBackward { .. } => {
                lower_normalization_operation(self, tensors)?
            }
            Operation::PagedCacheLoad {
                container,
                output,
                sequence,
                page_table,
            } => lower_paged_cache_load_operation(
                tensors,
                *container,
                *output,
                *sequence,
                *page_table,
            )?,
            Operation::DiagonalBandMask {
                x,
                b,
                y,
                comparison_mode,
                sequence_length_query,
                sequence_length_key_value,
                left_bound,
                shift_right_bound,
            } => lower_diagonal_band_mask_operation(
                tensors,
                *x,
                *b,
                *y,
                *comparison_mode,
                *sequence_length_query,
                *sequence_length_key_value,
                *left_bound,
                *shift_right_bound,
            )?,
            Operation::SdpaForward { .. } => {
                LoweredOperation::lower_sdpa_forward_operation(self, tensors, None)?
            }
            Operation::SdpaBackward { .. } => {
                LoweredOperation::lower_sdpa_backward_operation(self, tensors, None)?
            }
            Operation::MoeGroupedMatmul {
                token,
                weight,
                first_token_offset,
                output,
                config,
            } => LoweredOperation::MoeGroupedMatmul(MoeGroupedMatmulOperation::create(
                config.mode(),
                config.compute_type(),
                tensor_at(tensors, *token)?,
                tensor_at(tensors, *weight)?,
                tensor_at(tensors, *first_token_offset)?,
                tensor_at(tensors, *output)?,
                config
                    .token_index()
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                config
                    .token_ks()
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                config.top_k(),
            )?),
            Operation::MoeGroupedMatmulBackward { .. } => {
                return Err(Error::FrontendFeatureUnavailable {
                    feature: "moe grouped matmul backward".into(),
                    reason: "singe-cudnn is built against cuDNN 9.21 bindings, but the native backend descriptor requires cuDNN 9.22 bindings".into(),
                });
            }
        }))
    }
}

impl LoweredOperation {
    pub fn lower_sdpa_forward_operation(
        operation: &Operation,
        tensors: &BTreeMap<TensorId, Tensor>,
        subgraph: Option<LoweredSdpaSubgraph<'_>>,
    ) -> Result<Self> {
        let Operation::SdpaForward {
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
            score_modifiers: _,
            score_subgraph: _,
            config,
        } = operation
        else {
            return Err(Error::DescriptorMismatch {
                name: "sdpa forward operation".into(),
            });
        };

        let softmax = match (*softmax_p, *softmax_s) {
            (Some(softmax_p), Some(softmax_s)) => Some(SoftmaxOperation::create(
                tensor_at(tensors, softmax_p)?,
                tensor_at(tensors, softmax_s)?,
                stats.map(|stats| tensor_at(tensors, stats)).transpose()?,
                logit_max
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                score_sum_exp
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                sink.map(|tensor| tensor_at(tensors, tensor)).transpose()?,
            )?),
            (None, None) => None,
            _ => {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa unified softmax descriptor".into(),
                });
            }
        };

        let (subgraph, subgraph_input_id, subgraph_output_id) = if let Some(subgraph) = subgraph {
            (
                Some(subgraph.operation_graph),
                Some(subgraph.input_id),
                Some(subgraph.output_id),
            )
        } else {
            (None, None, None)
        };

        Ok(Self::SdpaForward(SdpaForwardOperation::create(
            tensor_at(tensors, *q)?,
            tensor_at(tensors, *k)?,
            tensor_at(tensors, *v)?,
            tensor_at(tensors, *o)?,
            tensor_at(tensors, *scale)?,
            stats.map(|stats| tensor_at(tensors, stats)).transpose()?,
            SdpaForwardConfig {
                sequence_length_query: config
                    .sequence_length_query()
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                sequence_length_key_value: config
                    .sequence_length_key_value()
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                page_table_k: config
                    .page_table_k()
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                page_table_v: config
                    .page_table_v()
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                block_mask: config
                    .block_mask()
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                softmax: softmax.as_ref(),
                subgraph,
                subgraph_input_id,
                subgraph_output_id,
                dropout_seed: config
                    .dropout_seed()
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                dropout_offset: config
                    .dropout()
                    .and_then(|dropout| dropout.offset())
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                dropout_random_number_generator_dump: config
                    .random_number_generator_dump()
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                dropout_probability: config.dropout().map(|dropout| dropout.probability()),
                unfuse_fma: config.unfuse_fma(),
            },
        )?))
    }

    pub fn lower_sdpa_backward_operation(
        operation: &Operation,
        tensors: &BTreeMap<TensorId, Tensor>,
        subgraph: Option<LoweredSdpaSubgraph<'_>>,
    ) -> Result<Self> {
        let Operation::SdpaBackward {
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
        } = operation
        else {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward operation".into(),
            });
        };

        let (subgraph, subgraph_input_id, subgraph_output_id) = if let Some(subgraph) = subgraph {
            (
                Some(subgraph.operation_graph),
                Some(subgraph.input_id),
                Some(subgraph.output_id),
            )
        } else {
            (None, None, None)
        };

        Ok(Self::SdpaBackward(SdpaBackwardOperation::create(
            tensor_at(tensors, *q)?,
            tensor_at(tensors, *k)?,
            tensor_at(tensors, *v)?,
            tensor_at(tensors, *o)?,
            tensor_at(tensors, *stats)?,
            tensor_at(tensors, *scale)?,
            tensor_at(tensors, *d_o)?,
            tensor_at(tensors, *d_q)?,
            tensor_at(tensors, *d_k)?,
            tensor_at(tensors, *d_v)?,
            SdpaBackwardConfig {
                sequence_length_query: config
                    .sequence_length_query()
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                sequence_length_key_value: config
                    .sequence_length_key_value()
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                subgraph,
                subgraph_input_id,
                subgraph_output_id,
                sink: config
                    .sink()
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                sink_gradient: sink_gradient
                    .map(|tensor| tensor_at(tensors, tensor))
                    .transpose()?,
                max_total_sequence_length_query: config.max_total_sequence_length_query(),
                max_total_sequence_length_key_value: config.max_total_sequence_length_key_value(),
            },
        )?))
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        match self {
            Self::Matmul(op) => op.descriptor(),
            Self::ConvolutionForward(op) => op.descriptor(),
            Self::ConvolutionBackwardData(op) => op.descriptor(),
            Self::ConvolutionBackwardFilter(op) => op.descriptor(),
            Self::Pointwise(op) => op.descriptor(),
            Self::Reduction(op) => op.descriptor(),
            Self::Reshape(op) => op.descriptor(),
            Self::Concat(op) => op.descriptor(),
            Self::Resample(op) => op.descriptor(),
            Self::ResampleBackward(op) => op.descriptor(),
            Self::Rng(op) => op.descriptor(),
            Self::GenStats(op) => op.descriptor(),
            Self::BlockScaleQuantize(op) => op.descriptor(),
            Self::BlockScaleDequantize(op) => op.descriptor(),
            Self::LayerNormalization(op) => op.descriptor(),
            Self::RmsNormalization(op) => op.descriptor(),
            Self::LayerNormalizationBackward(op) => op.descriptor(),
            Self::RmsNormalizationBackward(op) => op.descriptor(),
            Self::InstanceNormalization(op) => op.descriptor(),
            Self::InstanceNormalizationBackward(op) => op.descriptor(),
            Self::BatchNormalization(op) => op.descriptor(),
            Self::BatchNormalizationInference(op) => op.descriptor(),
            Self::BatchNormalizationFinalize(op) => op.descriptor(),
            Self::BatchNormalizationBackward(op) => op.descriptor(),
            Self::DbnWeight(op) => op.descriptor(),
            Self::AdaptiveLayerNormalization(op) => op.descriptor(),
            Self::AdaptiveLayerNormalizationBackward(op) => op.descriptor(),
            Self::PagedCacheLoad(op) => op.descriptor(),
            Self::DiagonalBandMask(op) => op.descriptor(),
            Self::Softmax(op) => op.descriptor(),
            Self::SdpaForward(op) => op.descriptor(),
            Self::SdpaBackward(op) => op.descriptor(),
            Self::MoeGroupedMatmul(op) => op.descriptor(),
        }
    }
}

fn convolution_descriptor(config: &ConvolutionConfig) -> Result<ConvolutionDescriptor> {
    ConvolutionDescriptor::create(
        config.compute_type(),
        config.mode(),
        config.pre_paddings(),
        config.post_paddings(),
        config.dilations(),
        config.strides(),
    )
}

fn resample_descriptor(input: &Tensor, config: &ResampleConfig) -> Result<ResampleDescriptor> {
    let spatial_dims = input.rank() as usize - 2;
    let pre_paddings = config.effective_pre_paddings(spatial_dims)?;
    let post_paddings = config.effective_post_paddings(spatial_dims)?;
    let strides = config.effective_strides(spatial_dims)?;
    let window_dims = config.effective_window_dims(spatial_dims)?;

    ResampleDescriptor::create(
        config.mode(),
        config.compute_type(),
        config.nan_propagation(),
        config.padding_mode(),
        &pre_paddings,
        &post_paddings,
        &strides,
        &window_dims,
    )
}

fn lower_reshape_operation(
    tensors: &BTreeMap<TensorId, Tensor>,
    input: TensorId,
    output: TensorId,
) -> Result<LoweredOperation> {
    Ok(LoweredOperation::Reshape(ReshapeOperation::create(
        tensor_at(tensors, input)?,
        tensor_at(tensors, output)?,
    )?))
}

#[allow(clippy::too_many_arguments)]
fn lower_softmax_operation(
    tensors: &BTreeMap<TensorId, Tensor>,
    x: TensorId,
    y: TensorId,
    stats: Option<TensorId>,
    max: Option<TensorId>,
    sum_exp: Option<TensorId>,
    sink: Option<TensorId>,
) -> Result<LoweredOperation> {
    Ok(LoweredOperation::Softmax(SoftmaxOperation::create(
        tensor_at(tensors, x)?,
        tensor_at(tensors, y)?,
        optional_tensor_at(tensors, stats)?,
        optional_tensor_at(tensors, max)?,
        optional_tensor_at(tensors, sum_exp)?,
        optional_tensor_at(tensors, sink)?,
    )?))
}

fn lower_concat_operation(
    tensors: &BTreeMap<TensorId, Tensor>,
    inputs: &[TensorId],
    output: TensorId,
    axis: i64,
    in_place_index: Option<i64>,
) -> Result<LoweredOperation> {
    let input_tensors = tensor_slice_at(tensors, inputs)?;
    Ok(LoweredOperation::Concat(ConcatOperation::create(
        &input_tensors,
        tensor_at(tensors, output)?,
        axis,
        in_place_index,
    )?))
}

fn lower_resample_operation(
    tensors: &BTreeMap<TensorId, Tensor>,
    input: TensorId,
    output: TensorId,
    indices: Option<TensorId>,
    config: &ResampleConfig,
) -> Result<LoweredOperation> {
    let descriptor = resample_descriptor(tensor_at(tensors, input)?, config)?;
    Ok(LoweredOperation::Resample(
        ResampleForwardOperation::create(
            &descriptor,
            tensor_at(tensors, input)?,
            tensor_at(tensors, output)?,
            optional_tensor_at(tensors, indices)?,
        )?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn lower_resample_backward_operation(
    tensors: &BTreeMap<TensorId, Tensor>,
    input: TensorId,
    output: TensorId,
    output_gradient: TensorId,
    input_gradient: TensorId,
    indices: Option<TensorId>,
    config: &ResampleConfig,
) -> Result<LoweredOperation> {
    let descriptor = resample_descriptor(tensor_at(tensors, input)?, config)?;
    Ok(LoweredOperation::ResampleBackward(
        ResampleBackwardOperation::create(
            &descriptor,
            tensor_at(tensors, input)?,
            tensor_at(tensors, output)?,
            tensor_at(tensors, output_gradient)?,
            tensor_at(tensors, input_gradient)?,
            optional_tensor_at(tensors, indices)?,
        )?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn normalization_forward_op(
    tensors: &BTreeMap<TensorId, Tensor>,
    mode: BackendNormalizationMode,
    phase: BackendNormalizationForwardPhase,
    input: TensorId,
    mean: Option<TensorId>,
    inv_variance: Option<TensorId>,
    scale: TensorId,
    bias: Option<TensorId>,
    epsilon: TensorId,
    running: Option<BatchNormalizationRunningStats>,
    peer_stats: &[TensorId],
    output: TensorId,
) -> Result<NormalizationForwardOperation> {
    let descriptor = NormalizationForwardConfig::new(mode, phase);
    let running_tensors = batch_running_tensors(tensors, running)?;
    NormalizationForwardOperation::create(
        &descriptor,
        tensor_at(tensors, input)?,
        optional_tensor_at(tensors, mean)?,
        optional_tensor_at(tensors, inv_variance)?,
        tensor_at(tensors, scale)?,
        optional_tensor_at(tensors, bias)?,
        Some(tensor_at(tensors, epsilon)?),
        running_tensors.0,
        running_tensors.1,
        running_tensors.2,
        running_tensors.3,
        running_tensors.4,
        &tensor_slice_at(tensors, peer_stats)?,
        tensor_at(tensors, output)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn normalization_backward_op(
    tensors: &BTreeMap<TensorId, Tensor>,
    mode: BackendNormalizationMode,
    input: TensorId,
    mean: Option<TensorId>,
    inv_variance: TensorId,
    dy: TensorId,
    scale: TensorId,
    epsilon: Option<TensorId>,
    dscale: TensorId,
    bias_gradient: Option<TensorId>,
    peer_stats: &[TensorId],
    dx: TensorId,
) -> Result<NormalizationBackwardOperation> {
    let descriptor = NormalizationBackwardConfig::new(mode);
    NormalizationBackwardOperation::create(
        &descriptor,
        tensor_at(tensors, input)?,
        optional_tensor_at(tensors, mean)?,
        tensor_at(tensors, inv_variance)?,
        tensor_at(tensors, dy)?,
        tensor_at(tensors, scale)?,
        optional_tensor_at(tensors, epsilon)?,
        tensor_at(tensors, dscale)?,
        optional_tensor_at(tensors, bias_gradient)?,
        &tensor_slice_at(tensors, peer_stats)?,
        tensor_at(tensors, dx)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn batch_normalization_finalize_op(
    tensors: &BTreeMap<TensorId, Tensor>,
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
    config: &BatchNormalizationFinalizeConfig,
) -> Result<BatchNormalizationFinalizeStatsOperation> {
    BatchNormalizationFinalizeStatsOperation::create(
        config.mode(),
        DataType::F32,
        tensor_at(tensors, sum)?,
        tensor_at(tensors, sq_sum)?,
        tensor_at(tensors, scale)?,
        tensor_at(tensors, bias)?,
        optional_tensor_at(tensors, config.prev_running_mean())?,
        optional_tensor_at(tensors, config.prev_running_var())?,
        optional_tensor_at(tensors, next_running_mean)?,
        optional_tensor_at(tensors, next_running_var)?,
        tensor_at(tensors, saved_mean)?,
        tensor_at(tensors, saved_inv_variance)?,
        tensor_at(tensors, eq_scale)?,
        tensor_at(tensors, eq_bias)?,
        tensor_at(tensors, accum_count)?,
        tensor_at(tensors, config.epsilon())?,
        optional_tensor_at(tensors, config.momentum())?,
    )
}

#[allow(clippy::too_many_arguments)]
fn batch_normalization_backward_weights_op(
    tensors: &BTreeMap<TensorId, Tensor>,
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
) -> Result<BatchNormalizationBackwardWeightsOperation> {
    BatchNormalizationBackwardWeightsOperation::create(
        DataType::F32,
        tensor_at(tensors, mean)?,
        tensor_at(tensors, inv_variance)?,
        tensor_at(tensors, scale)?,
        tensor_at(tensors, input)?,
        tensor_at(tensors, dy)?,
        tensor_at(tensors, dscale)?,
        tensor_at(tensors, bias_gradient)?,
        tensor_at(tensors, eq_scale_dy)?,
        tensor_at(tensors, eq_scale_x)?,
        tensor_at(tensors, eq_bias)?,
    )
}

fn lower_normalization_operation(
    operation: &Operation,
    tensors: &BTreeMap<TensorId, Tensor>,
) -> Result<LoweredOperation> {
    Ok(match operation {
        Operation::LayerNormalization {
            input,
            scale,
            bias,
            output,
            mean,
            inv_variance,
            config,
        } => LoweredOperation::LayerNormalization(normalization_forward_op(
            tensors,
            BackendNormalizationMode::Layer,
            config.phase(),
            *input,
            Some(*mean),
            Some(*inv_variance),
            *scale,
            Some(*bias),
            config.epsilon(),
            None,
            &[],
            *output,
        )?),
        Operation::RmsNormalization {
            input,
            scale,
            bias,
            output,
            inv_variance,
            config,
        } => LoweredOperation::RmsNormalization(normalization_forward_op(
            tensors,
            BackendNormalizationMode::Rms,
            config.phase(),
            *input,
            None,
            Some(*inv_variance),
            *scale,
            *bias,
            config.epsilon(),
            None,
            &[],
            *output,
        )?),
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
        } => LoweredOperation::LayerNormalizationBackward(normalization_backward_op(
            tensors,
            BackendNormalizationMode::Layer,
            *input,
            Some(*mean),
            *inv_variance,
            *dy,
            *scale,
            Some(config.epsilon()),
            *dscale,
            Some(*bias_gradient),
            &[],
            *dx,
        )?),
        Operation::RmsNormalizationBackward {
            input,
            inv_variance,
            dy,
            scale,
            dscale,
            bias_gradient,
            dx,
            config: _,
        } => LoweredOperation::RmsNormalizationBackward(normalization_backward_op(
            tensors,
            BackendNormalizationMode::Rms,
            *input,
            None,
            *inv_variance,
            *dy,
            *scale,
            None,
            *dscale,
            *bias_gradient,
            &[],
            *dx,
        )?),
        Operation::InstanceNormalization {
            input,
            scale,
            bias,
            output,
            mean,
            inv_variance,
            config,
        } => LoweredOperation::InstanceNormalization(normalization_forward_op(
            tensors,
            BackendNormalizationMode::Instance,
            config.phase(),
            *input,
            Some(*mean),
            Some(*inv_variance),
            *scale,
            Some(*bias),
            config.epsilon(),
            None,
            &[],
            *output,
        )?),
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
        } => LoweredOperation::InstanceNormalizationBackward(normalization_backward_op(
            tensors,
            BackendNormalizationMode::Instance,
            *input,
            Some(*mean),
            *inv_variance,
            *dy,
            *scale,
            Some(config.epsilon()),
            *dscale,
            Some(*bias_gradient),
            &[],
            *dx,
        )?),
        Operation::BatchNormalization {
            input,
            scale,
            bias,
            output,
            mean,
            inv_variance,
            config,
        } => LoweredOperation::BatchNormalization(normalization_forward_op(
            tensors,
            BackendNormalizationMode::Batch,
            BackendNormalizationForwardPhase::Training,
            *input,
            Some(*mean),
            Some(*inv_variance),
            *scale,
            Some(*bias),
            config.epsilon(),
            config.running(),
            config.peer_stats(),
            *output,
        )?),
        Operation::BatchNormalizationInference {
            input,
            mean,
            inv_variance,
            scale,
            bias,
            output,
            config,
        } => LoweredOperation::BatchNormalizationInference(normalization_forward_op(
            tensors,
            BackendNormalizationMode::Batch,
            BackendNormalizationForwardPhase::Inference,
            *input,
            Some(*mean),
            Some(*inv_variance),
            *scale,
            Some(*bias),
            config.epsilon(),
            None,
            &[],
            *output,
        )?),
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
        } => LoweredOperation::BatchNormalizationFinalize(batch_normalization_finalize_op(
            tensors,
            *sum,
            *sq_sum,
            *scale,
            *bias,
            *next_running_mean,
            *next_running_var,
            *saved_mean,
            *saved_inv_variance,
            *eq_scale,
            *eq_bias,
            *accum_count,
            config,
        )?),
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
        } => LoweredOperation::BatchNormalizationBackward(normalization_backward_op(
            tensors,
            BackendNormalizationMode::Batch,
            *input,
            Some(*mean),
            *inv_variance,
            *dy,
            *scale,
            None,
            *dscale,
            Some(*bias_gradient),
            config.peer_stats(),
            *dx,
        )?),
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
            config: _,
        } => LoweredOperation::DbnWeight(batch_normalization_backward_weights_op(
            tensors,
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
        )?),
        Operation::AdaptiveLayerNormalization {
            input,
            scale,
            bias,
            output,
            mean,
            inv_variance,
            config,
        } => LoweredOperation::AdaptiveLayerNormalization(normalization_forward_op(
            tensors,
            BackendNormalizationMode::AdaLayerNorm,
            config.phase(),
            *input,
            Some(*mean),
            Some(*inv_variance),
            *scale,
            *bias,
            config.epsilon(),
            None,
            &[],
            *output,
        )?),
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
        } => LoweredOperation::AdaptiveLayerNormalizationBackward(normalization_backward_op(
            tensors,
            BackendNormalizationMode::AdaLayerNorm,
            *input,
            Some(*mean),
            *inv_variance,
            *dy,
            *scale,
            None,
            *dscale,
            *bias_gradient,
            &[],
            *dx,
        )?),
        _ => {
            return Err(Error::DescriptorMismatch {
                name: "normalization operation".into(),
            });
        }
    })
}

fn lower_pointwise_operation(
    tensors: &BTreeMap<TensorId, Tensor>,
    operation: &PointwiseOperation,
) -> Result<LoweredOperation> {
    Ok(LoweredOperation::Pointwise(match operation {
        PointwiseOperation::Unary {
            mode,
            input,
            output,
            compute_type,
            nan_propagation,
            alpha1,
            axis,
        } => {
            let descriptor = if let Some(axis) = axis {
                PointwiseDescriptor::create_with_axis(
                    *mode,
                    *compute_type,
                    *nan_propagation,
                    *axis,
                )?
            } else {
                PointwiseDescriptor::create(*mode, *compute_type, *nan_propagation)?
            };
            BackendPointwiseOperation::unary(
                &descriptor,
                tensor_at(tensors, *input)?,
                tensor_at(tensors, *output)?,
                *alpha1,
            )?
        }
        PointwiseOperation::ReluForward {
            input,
            output,
            compute_type,
            nan_propagation,
            lower_clip,
            upper_clip,
            lower_clip_slope,
            axis,
        } => {
            let relu_clips = Some((*lower_clip, *upper_clip, *lower_clip_slope));
            let descriptor = if let Some(axis) = axis {
                PointwiseDescriptor::create_with_axis_and_relu_clips(
                    PointwiseMode::ReluFwd,
                    *compute_type,
                    *nan_propagation,
                    *axis,
                    relu_clips,
                )?
            } else {
                PointwiseDescriptor::create_with_relu_clips(
                    PointwiseMode::ReluFwd,
                    *compute_type,
                    *nan_propagation,
                    relu_clips,
                )?
            };
            BackendPointwiseOperation::unary(
                &descriptor,
                tensor_at(tensors, *input)?,
                tensor_at(tensors, *output)?,
                1.0,
            )?
        }
        PointwiseOperation::Binary {
            mode,
            lhs,
            rhs,
            output,
            compute_type,
            nan_propagation,
            alpha1,
            alpha2,
        } => {
            let descriptor = PointwiseDescriptor::create(*mode, *compute_type, *nan_propagation)?;
            if is_activation_backward_mode(*mode) {
                BackendPointwiseOperation::activation_backward(
                    &descriptor,
                    tensor_at(tensors, *lhs)?,
                    tensor_at(tensors, *rhs)?,
                    tensor_at(tensors, *output)?,
                    *alpha1,
                    *alpha2,
                )?
            } else {
                BackendPointwiseOperation::binary(
                    &descriptor,
                    tensor_at(tensors, *lhs)?,
                    tensor_at(tensors, *rhs)?,
                    tensor_at(tensors, *output)?,
                    *alpha1,
                    *alpha2,
                )?
            }
        }
        PointwiseOperation::Ternary {
            mode,
            x,
            b,
            t,
            output,
            compute_type,
            nan_propagation,
            alpha1,
            alpha2,
        } => {
            let descriptor = PointwiseDescriptor::create(*mode, *compute_type, *nan_propagation)?;
            BackendPointwiseOperation::ternary(
                &descriptor,
                tensor_at(tensors, *x)?,
                tensor_at(tensors, *b)?,
                tensor_at(tensors, *t)?,
                tensor_at(tensors, *output)?,
                *alpha1,
                *alpha2,
            )?
        }
    }))
}

fn is_activation_backward_mode(mode: PointwiseMode) -> bool {
    matches!(
        mode,
        PointwiseMode::ReluBwd
            | PointwiseMode::TanhBwd
            | PointwiseMode::SigmoidBwd
            | PointwiseMode::EluBwd
            | PointwiseMode::GeluBwd
            | PointwiseMode::GeluApproxTanhBwd
            | PointwiseMode::SoftplusBwd
            | PointwiseMode::SwishBwd
    )
}

fn lower_reduction_operation(
    tensors: &BTreeMap<TensorId, Tensor>,
    operation: &ReductionOperation,
) -> Result<LoweredOperation> {
    match operation {
        ReductionOperation::Reduce {
            op,
            input,
            output,
            compute_type,
            is_deterministic,
        } => {
            let descriptor = ReductionDescriptor::create(*op, *compute_type, *is_deterministic)?;
            Ok(LoweredOperation::Reduction(
                BackendReductionOperation::create(
                    &descriptor,
                    tensor_at(tensors, *input)?,
                    tensor_at(tensors, *output)?,
                )?,
            ))
        }
    }
}

fn lower_rng_operation(
    tensors: &BTreeMap<TensorId, Tensor>,
    output: TensorId,
    config: &RandomNumberGeneratorConfig,
) -> Result<LoweredOperation> {
    let descriptor = match config.distribution() {
        RandomNumberDistribution::Bernoulli { probability } => {
            RngDescriptor::create_bernoulli(*probability)?
        }
        RandomNumberDistribution::Uniform { minimum, maximum } => {
            RngDescriptor::create_uniform(*minimum, *maximum)?
        }
        RandomNumberDistribution::Normal {
            mean,
            standard_deviation,
        } => RngDescriptor::create_normal(*mean, *standard_deviation)?,
    };
    let seed = match config.seed_source() {
        RandomNumberSeedSource::Host(seed) => RngSeed::Host(*seed),
        RandomNumberSeedSource::Device(seed) => RngSeed::Device(tensor_at(tensors, *seed)?),
    };
    Ok(LoweredOperation::Rng(RngOperation::create(
        &descriptor,
        tensor_at(tensors, output)?,
        seed,
        config
            .offset()
            .map(|offset| tensor_at(tensors, offset))
            .transpose()?,
    )?))
}

fn lower_gen_stats_operation(
    tensors: &BTreeMap<TensorId, Tensor>,
    input: TensorId,
    sum: TensorId,
    sq_sum: TensorId,
) -> Result<LoweredOperation> {
    Ok(LoweredOperation::GenStats(GenStatsOperation::create(
        GenStatsMode::SumSqsum,
        tensor_at(tensors, input)?,
        tensor_at(tensors, sum)?,
        tensor_at(tensors, sq_sum)?,
    )?))
}

fn lower_block_scale_quantize_operation(
    tensors: &BTreeMap<TensorId, Tensor>,
    input: TensorId,
    output: TensorId,
    scale: TensorId,
    config: &BlockScaleQuantizeConfig,
) -> Result<LoweredOperation> {
    Ok(LoweredOperation::BlockScaleQuantize(
        BlockScaleQuantizeOperation::create(
            tensor_at(tensors, input)?,
            tensor_at(tensors, output)?,
            tensor_at(tensors, scale)?,
            config.compute_type(),
            to_i32(config.block_size(), "block size")?,
        )?,
    ))
}

fn lower_block_scale_dequantize_operation(
    tensors: &BTreeMap<TensorId, Tensor>,
    input: TensorId,
    scale: TensorId,
    output: TensorId,
    config: &BlockScaleDequantizeConfig,
) -> Result<LoweredOperation> {
    Ok(LoweredOperation::BlockScaleDequantize(
        BlockScaleDequantizeOperation::create(
            tensor_at(tensors, input)?,
            tensor_at(tensors, scale)?,
            tensor_at(tensors, output)?,
            config.compute_type(),
            config.block_sizes(),
            config.is_negative_scale(),
        )?,
    ))
}

fn lower_paged_cache_load_operation(
    tensors: &BTreeMap<TensorId, Tensor>,
    container: TensorId,
    output: TensorId,
    sequence: TensorId,
    page_table: TensorId,
) -> Result<LoweredOperation> {
    Ok(LoweredOperation::PagedCacheLoad(
        PagedCacheLoadOperation::create(
            tensor_at(tensors, container)?,
            tensor_at(tensors, output)?,
            tensor_at(tensors, sequence)?,
            tensor_at(tensors, page_table)?,
        )?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn lower_diagonal_band_mask_operation(
    tensors: &BTreeMap<TensorId, Tensor>,
    x: TensorId,
    b: TensorId,
    y: TensorId,
    comparison_mode: PointwiseMode,
    sequence_length_query: Option<TensorId>,
    sequence_length_key_value: Option<TensorId>,
    left_bound: Option<TensorId>,
    shift_right_bound: Option<TensorId>,
) -> Result<LoweredOperation> {
    Ok(LoweredOperation::DiagonalBandMask(
        DiagonalBandMaskOperation::create(
            tensor_at(tensors, x)?,
            tensor_at(tensors, b)?,
            tensor_at(tensors, y)?,
            comparison_mode,
            optional_tensor_at(tensors, sequence_length_query)?,
            optional_tensor_at(tensors, sequence_length_key_value)?,
            optional_tensor_at(tensors, left_bound)?,
            optional_tensor_at(tensors, shift_right_bound)?,
        )?,
    ))
}

fn lower_convolution_forward(
    tensors: &BTreeMap<TensorId, Tensor>,
    x: TensorId,
    w: TensorId,
    y: TensorId,
    config: &ConvolutionConfig,
) -> Result<LoweredOperation> {
    let descriptor = convolution_descriptor(config)?;
    Ok(LoweredOperation::ConvolutionForward(
        ConvolutionForwardOperation::create(
            &descriptor,
            tensor_at(tensors, x)?,
            tensor_at(tensors, w)?,
            tensor_at(tensors, y)?,
            1.0,
            0.0,
        )?,
    ))
}

fn lower_convolution_backward_data(
    tensors: &BTreeMap<TensorId, Tensor>,
    w: TensorId,
    dy: TensorId,
    dx: TensorId,
    config: &ConvolutionConfig,
) -> Result<LoweredOperation> {
    let descriptor = convolution_descriptor(config)?;
    Ok(LoweredOperation::ConvolutionBackwardData(
        ConvolutionBackwardDataOperation::create(
            &descriptor,
            tensor_at(tensors, w)?,
            tensor_at(tensors, dy)?,
            tensor_at(tensors, dx)?,
            1.0,
            0.0,
        )?,
    ))
}

fn lower_convolution_backward_filter(
    tensors: &BTreeMap<TensorId, Tensor>,
    x: TensorId,
    dy: TensorId,
    dw: TensorId,
    config: &ConvolutionConfig,
) -> Result<LoweredOperation> {
    let descriptor = convolution_descriptor(config)?;
    Ok(LoweredOperation::ConvolutionBackwardFilter(
        ConvolutionBackwardFilterOperation::create(
            &descriptor,
            tensor_at(tensors, x)?,
            tensor_at(tensors, dy)?,
            tensor_at(tensors, dw)?,
            1.0,
            0.0,
        )?,
    ))
}

fn lower_tensor_at(
    tensor_ref: TensorId,
    tensors: &BTreeMap<TensorId, TensorRecord>,
    lowered_tensors: &mut BTreeMap<TensorId, Tensor>,
    active_tensors: &mut HashSet<TensorId>,
) -> Result<()> {
    if lowered_tensors.contains_key(&tensor_ref) {
        return Ok(());
    }
    if !active_tensors.insert(tensor_ref) {
        return Err(Error::DescriptorMismatch {
            name: "tensor ragged offset cycle".into(),
        });
    }

    let record = tensors
        .get(&tensor_ref)
        .ok_or(Error::FrontendTensorNotFound(tensor_ref))?;
    let tensor = &record.tensor;
    if let Some(ragged_offset) = tensor.ragged_offset {
        lower_tensor_at(ragged_offset, tensors, lowered_tensors, active_tensors)?;
    }

    let lowered = lower_tensor(record.id, tensor)?;
    lowered_tensors.insert(tensor_ref, lowered);
    active_tensors.remove(&tensor_ref);
    Ok(())
}

fn collect_used_tensors(
    graph: &Graph,
    tensor_ref: TensorId,
    used_tensors: &mut HashSet<TensorId>,
) -> Result<()> {
    let Some(tensor) = graph.tensors.get(&tensor_ref) else {
        return Err(Error::FrontendTensorNotFound(tensor_ref));
    };
    if let Some(ragged_offset) = tensor.ragged_offset
        && used_tensors.insert(ragged_offset)
    {
        collect_used_tensors(graph, ragged_offset, used_tensors)?;
    }
    Ok(())
}

fn lower_tensor(id: TensorId, tensor: &TensorSpec) -> Result<Tensor> {
    if tensor.vector_count.is_some() || tensor.vectorized_dimension.is_some() {
        let vectorized_dimension =
            tensor
                .vectorized_dimension
                .ok_or(Error::DescriptorMismatch {
                    name: "frontend tensor vectorization".into(),
                })?;
        let vectorized_dimension = if vectorized_dimension < 0 {
            tensor.shape.rank() + vectorized_dimension
        } else {
            vectorized_dimension
        };
        let vector_count = tensor.vector_count.ok_or(Error::DescriptorMismatch {
            name: "frontend tensor vectorization".into(),
        })?;
        Tensor::create_with_resolved_spec(
            id,
            tensor,
            Some(vectorized_dimension),
            Some(vector_count),
        )
    } else {
        Tensor::create_with_resolved_spec(id, tensor, None, None)
    }
}

fn tensor_at(tensors: &BTreeMap<TensorId, Tensor>, tensor: TensorId) -> Result<&Tensor> {
    tensors
        .get(&tensor)
        .ok_or(Error::FrontendTensorNotFound(tensor))
}

fn optional_tensor_at(
    tensors: &BTreeMap<TensorId, Tensor>,
    tensor: Option<TensorId>,
) -> Result<Option<&Tensor>> {
    tensor.map(|tensor| tensor_at(tensors, tensor)).transpose()
}

fn tensor_slice_at<'a>(
    tensors: &'a BTreeMap<TensorId, Tensor>,
    tensor_ids: &[TensorId],
) -> Result<Vec<&'a Tensor>> {
    tensor_ids
        .iter()
        .map(|tensor_id| tensor_at(tensors, *tensor_id))
        .collect()
}

type BatchRunningTensors<'a> = (
    Option<&'a Tensor>,
    Option<&'a Tensor>,
    Option<&'a Tensor>,
    Option<&'a Tensor>,
    Option<&'a Tensor>,
);

fn batch_running_tensors(
    tensors: &BTreeMap<TensorId, Tensor>,
    running: Option<BatchNormalizationRunningStats>,
) -> Result<BatchRunningTensors<'_>> {
    let Some(BatchNormalizationRunningStats {
        momentum,
        prev_mean,
        prev_var,
        next_mean,
        next_var,
    }) = running
    else {
        return Ok((None, None, None, None, None));
    };
    Ok((
        Some(tensor_at(tensors, momentum)?),
        Some(tensor_at(tensors, prev_mean)?),
        Some(tensor_at(tensors, prev_var)?),
        Some(tensor_at(tensors, next_mean)?),
        Some(tensor_at(tensors, next_var)?),
    ))
}
