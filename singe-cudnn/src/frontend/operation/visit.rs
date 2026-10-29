//! One traversal of the tensor IDs stored in each operation.
//!
//! Dependency analysis and tensor remapping both derive from these visitors, so
//! an operation's tensors are listed in exactly one place. Score subgraphs are
//! separate graphs whose captured tensors are handled by the callers.

use crate::{
    backend::{execution::attention::AttentionSequenceInputs, tensor::TensorId},
    error::{Error, Result},
    frontend::operation::{
        AttentionBackwardRequest, AttentionForwardInputs, AttentionForwardRequest,
        AttentionPrimitiveOperation, BlockScaleOperation, ConvolutionOperation, GenStatsOperation,
        MatmulOperation, MoeGroupedMatmulOperation, NormalizationOperation, Operation,
        PagedCacheLoadOperation, PointwiseOperation, RandomNumberGeneratorOperation,
        ReductionOperation, ResampleOperation, RopeOperation, SoftmaxConfig, TensorOperation,
        TensorUses,
        attention::{
            AlibiBias, AttentionBackwardConfig, AttentionConfig, AttentionDropoutConfig,
            AttentionMaskMode, AttentionScoreConfig, DirectSdpaBackwardConfig, DirectSdpaDropout,
            DirectSdpaForwardConfig, SdpaFusedMaskMode, SdpaOperation, SdpaScoreModifiers,
        },
        normalization::{
            LayerNormalizationAffine, NormalizationStatistics, RmsNormalizationStatistics,
        },
    },
};

/// How an operation uses one of its tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    Read,
    Write,
    /// Read and overwritten, such as a blended destination.
    ReadWrite,
    /// Read to compute memory addresses, such as offsets, extents and page tables.
    Index,
}

/// Receives each tensor ID stored in an operation with its role, and may replace it.
pub type TensorVisitor<'a> = dyn FnMut(&mut TensorId, TensorRole) -> Result<()> + 'a;

impl TensorUses {
    pub(crate) fn record(&mut self, id: TensorId, role: TensorRole) {
        match role {
            TensorRole::Read => self.read([id]),
            TensorRole::Write => self.write([id]),
            TensorRole::ReadWrite => {
                self.read([id]);
                self.write([id]);
            }
            TensorRole::Index => self.read_indices([id]),
        }
    }
}

fn visit_optional(
    id: &mut Option<TensorId>,
    role: TensorRole,
    visit: &mut TensorVisitor<'_>,
) -> Result<()> {
    id.as_mut().map_or_else(|| Ok(()), |id| visit(id, role))
}

/// Collects the uses reported by a visitor run.
pub fn collect_tensor_uses(
    visit_all: impl FnOnce(&mut TensorVisitor<'_>) -> Result<()>,
) -> TensorUses {
    let mut uses = TensorUses::default();
    // Recording never fails, so the visitor cannot return an error.
    let _ = visit_all(&mut |id, role| {
        uses.record(*id, role);
        Ok(())
    });
    uses
}

/// Maps a tensor through the visitor for `try_map` helpers that rebuild a config.
fn map_with(
    visit: &mut TensorVisitor<'_>,
    role: TensorRole,
) -> impl FnMut(TensorId) -> Result<TensorId> {
    move |mut id| {
        visit(&mut id, role)?;
        Ok(id)
    }
}

impl Operation {
    /// Visits every tensor stored in this node.
    ///
    /// Score subgraphs are separate graphs; their captures are handled by the callers.
    pub(crate) fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        match self {
            Self::Matmul(operation) => operation.visit_tensor_ids(visit)?,
            Self::Convolution(operation) => operation.visit_tensor_ids(visit)?,
            Self::Reduction(operation) => operation.visit_tensor_ids(visit)?,
            Self::Tensor(operation) => operation.visit_tensor_ids(visit)?,
            Self::Rope(operation) => operation.visit_tensor_ids(visit)?,
            Self::Resample(operation) => operation.visit_tensor_ids(visit)?,
            Self::RandomNumberGenerator(operation) => operation.visit_tensor_ids(visit)?,
            Self::GenStats(operation) => operation.visit_tensor_ids(visit)?,
            Self::BlockScale(operation) => operation.visit_tensor_ids(visit)?,
            Self::PagedCacheLoad(operation) => operation.visit_tensor_ids(visit)?,
            Self::AttentionPrimitive(operation) => operation.visit_tensor_ids(visit)?,
            Self::MoeGroupedMatmul(operation) => operation.visit_tensor_ids(visit)?,
            Self::Pointwise(operation) => operation.visit_tensor_ids(visit)?,
            Self::Normalization(operation) => operation.visit_tensor_ids(visit)?,
            Self::Sdpa(operation) => operation.visit_tensor_ids(visit)?,
            Self::AttentionForward(request) => request.visit_tensor_ids(visit)?,
            Self::AttentionBackward(request) => request.visit_tensor_ids(visit)?,
        }
        Ok(())
    }

    /// Uses of the tensors stored in this node, excluding tensors captured by score subgraphs.
    pub(crate) fn direct_tensor_uses(&self) -> TensorUses {
        collect_tensor_uses(|visit| self.clone().visit_tensor_ids(visit))
    }
}

impl MatmulOperation {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        visit(&mut self.a, TensorRole::Read)?;
        visit(&mut self.b, TensorRole::Read)?;
        visit(&mut self.c, TensorRole::Write)?;
        let mut config = self.config.clone();
        if let Some(mut id) = config.m_override() {
            visit(&mut id, TensorRole::Index)?;
            config = config.with_m_override(id);
        }
        if let Some(mut id) = config.n_override() {
            visit(&mut id, TensorRole::Index)?;
            config = config.with_n_override(id);
        }
        if let Some(mut id) = config.k_override() {
            visit(&mut id, TensorRole::Index)?;
            config = config.with_k_override(id);
        }
        self.config = config;
        Ok(())
    }
}

impl ConvolutionOperation {
    /// Visits the destination as a write; [`Self::reads_destination`] adds the policy-dependent read.
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        let (inputs, destination) = match self {
            Self::Forward { x, w, y, .. } => ([x, w], y),
            Self::BackwardData { w, dy, dx, .. } => ([w, dy], dx),
            Self::BackwardFilter { x, dy, dw, .. } => ([x, dy], dw),
        };
        for input in inputs {
            visit(input, TensorRole::Read)?;
        }
        visit(destination, TensorRole::Write)
    }
}

impl ReductionOperation {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        match self {
            Self::Custom {
                inputs, outputs, ..
            } => {
                for input in inputs {
                    visit(input, TensorRole::Read)?;
                }
                for output in outputs {
                    visit(output, TensorRole::Write)?;
                }
                Ok(())
            }
            Self::Reduce {
                input,
                output,
                group_offset,
                ..
            } => {
                visit(input, TensorRole::Read)?;
                visit(output, TensorRole::Write)?;
                visit_optional(group_offset, TensorRole::Index, visit)
            }
        }
    }
}

impl TensorOperation {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        match self {
            Self::Reshape { input, output, .. }
            | Self::Slice { input, output, .. }
            | Self::Transpose { input, output, .. } => {
                visit(input, TensorRole::Read)?;
                visit(output, TensorRole::Write)
            }
            Self::Concat { inputs, output, .. } => {
                for input in inputs {
                    visit(input, TensorRole::Read)?;
                }
                visit(output, TensorRole::Write)
            }
        }
    }
}

impl RopeOperation {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        match self {
            Self::Forward { inputs, output, .. } => {
                visit(&mut inputs.input, TensorRole::Read)?;
                visit(&mut inputs.frequencies, TensorRole::Read)?;
                visit(output, TensorRole::Write)
            }
            Self::Backward {
                inputs,
                input_gradient,
                ..
            } => {
                visit(&mut inputs.output_gradient, TensorRole::Read)?;
                visit(&mut inputs.frequencies, TensorRole::Read)?;
                visit(input_gradient, TensorRole::Write)
            }
        }
    }
}

impl ResampleOperation {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        match self {
            Self::Forward {
                input,
                outputs,
                config,
            } => {
                visit(input, TensorRole::Read)?;
                let output_role = if config.beta() == 0.0 {
                    TensorRole::Write
                } else {
                    TensorRole::ReadWrite
                };
                visit(&mut outputs.output, output_role)?;
                visit_optional(&mut outputs.indices, TensorRole::Write, visit)
            }
            Self::Backward {
                inputs,
                input_gradient,
                config,
            } => {
                visit(&mut inputs.output_gradient, TensorRole::Read)?;
                visit_optional(&mut inputs.input, TensorRole::Read, visit)?;
                visit_optional(&mut inputs.output, TensorRole::Read, visit)?;
                visit_optional(&mut inputs.indices, TensorRole::Index, visit)?;
                let gradient_role = if config.beta() == 0.0 {
                    TensorRole::Write
                } else {
                    TensorRole::ReadWrite
                };
                visit(input_gradient, gradient_role)
            }
        }
    }
}

impl RandomNumberGeneratorOperation {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        let seed = self
            .config
            .seed_source()
            .try_map(map_with(visit, TensorRole::Read))?;
        self.config = self.config.clone().with_seed_source(seed);
        visit(&mut self.output, TensorRole::Write)
    }
}

impl GenStatsOperation {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        visit(&mut self.input, TensorRole::Read)?;
        visit(&mut self.outputs.sum, TensorRole::Write)?;
        visit(&mut self.outputs.square_sum, TensorRole::Write)
    }
}

impl BlockScaleOperation {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        match self {
            Self::Quantize {
                input,
                output,
                scale,
                ..
            } => {
                visit(input, TensorRole::Read)?;
                visit(output, TensorRole::Write)?;
                visit(scale, TensorRole::Write)
            }
            Self::Dequantize {
                input,
                scale,
                output,
                ..
            } => {
                visit(input, TensorRole::Read)?;
                visit(scale, TensorRole::Read)?;
                visit(output, TensorRole::Write)
            }
        }
    }
}

impl PagedCacheLoadOperation {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        visit(&mut self.tensors.container, TensorRole::Read)?;
        visit(&mut self.tensors.sequence, TensorRole::Index)?;
        visit(&mut self.tensors.page_table, TensorRole::Index)?;
        visit(&mut self.tensors.output, TensorRole::Write)
    }
}

impl AttentionPrimitiveOperation {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        match self {
            Self::DiagonalBandMask { x, b, y, config } => {
                visit(x, TensorRole::Read)?;
                visit(b, TensorRole::Read)?;
                let bound = config.bound().tensor();
                *config = config.clone().try_map(|mut id| {
                    let role = if Some(id) == bound {
                        TensorRole::Read
                    } else {
                        TensorRole::Index
                    };
                    visit(&mut id, role)?;
                    Ok::<_, Error>(id)
                })?;
                visit(y, TensorRole::Write)
            }
            Self::Softmax {
                x,
                y,
                stats,
                max,
                sum_exp,
                sink,
                ..
            } => {
                visit(x, TensorRole::Read)?;
                visit_optional(sink, TensorRole::Read, visit)?;
                visit(y, TensorRole::Write)?;
                visit_optional(stats, TensorRole::Write, visit)?;
                visit_optional(max, TensorRole::Write, visit)?;
                visit_optional(sum_exp, TensorRole::Write, visit)
            }
        }
    }
}

impl MoeGroupedMatmulOperation {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        match self {
            Self::Forward { tensors, config } => {
                visit(&mut tensors.inputs.token, TensorRole::Read)?;
                visit(&mut tensors.inputs.weight, TensorRole::Read)?;
                visit(&mut tensors.inputs.first_token_offset, TensorRole::Index)?;
                let routing = config
                    .routing()
                    .try_map(map_with(visit, TensorRole::Index))?;
                *config = config.clone().with_routing(routing);
                visit(&mut tensors.output, TensorRole::Write)
            }
            Self::Backward { tensors, .. } => {
                visit(&mut tensors.inputs.output_gradient, TensorRole::Read)?;
                visit(&mut tensors.inputs.token, TensorRole::Read)?;
                visit(&mut tensors.inputs.first_token_offset, TensorRole::Index)?;
                visit(&mut tensors.weight_gradient, TensorRole::Write)
            }
        }
    }
}

impl PointwiseOperation {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        let (inputs, output): (Vec<&mut TensorId>, &mut TensorId) = match self {
            Self::Unary { input, output, .. } | Self::ActivationForward { input, output, .. } => {
                (vec![input], output)
            }
            Self::ActivationBackward {
                output_gradient,
                input,
                input_gradient,
                ..
            } => (vec![output_gradient, input], input_gradient),
            Self::Binary {
                lhs, rhs, output, ..
            } => (vec![lhs, rhs], output),
            Self::Ternary {
                x, b, t, output, ..
            } => (vec![x, b, t], output),
        };
        for input in inputs {
            visit(input, TensorRole::Read)?;
        }
        visit(output, TensorRole::Write)
    }
}

fn visit_saved_statistics(
    statistics: &mut NormalizationStatistics,
    role: TensorRole,
    visit: &mut TensorVisitor<'_>,
) -> Result<()> {
    if let NormalizationStatistics::Training(statistics) = statistics {
        visit(&mut statistics.mean, role)?;
        visit(&mut statistics.inv_variance, role)?;
    }
    Ok(())
}

impl NormalizationOperation {
    #[expect(
        clippy::too_many_lines,
        reason = "the operation table enumerates every supported variant and its tensor contract"
    )]
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        use TensorRole::{Read, ReadWrite, Write};
        match self {
            Self::Layer {
                inputs,
                outputs,
                config,
            } => {
                visit(&mut inputs.input, Read)?;
                visit(config.epsilon_mut(), Read)?;
                match &mut inputs.affine {
                    LayerNormalizationAffine::None => {}
                    LayerNormalizationAffine::Scale(scale) => visit(scale, Read)?,
                    LayerNormalizationAffine::ScaleBias { scale, bias } => {
                        visit(scale, Read)?;
                        visit(bias, Read)?;
                    }
                }
                visit(&mut outputs.output, Write)?;
                visit_saved_statistics(&mut outputs.statistics, Write, visit)
            }
            Self::Rms {
                inputs,
                outputs,
                config,
            } => {
                visit(&mut inputs.input, Read)?;
                visit(config.epsilon_mut(), Read)?;
                visit_optional(&mut inputs.scale, Read, visit)?;
                visit_optional(&mut inputs.bias, Read, visit)?;
                visit(&mut outputs.output, Write)?;
                if let RmsNormalizationStatistics::Training { inv_variance } =
                    &mut outputs.statistics
                {
                    visit(inv_variance, Write)?;
                }
                Ok(())
            }
            Self::Instance {
                inputs,
                outputs,
                config,
            } => {
                visit(&mut inputs.input, Read)?;
                visit(&mut inputs.scale, Read)?;
                visit(&mut inputs.bias, Read)?;
                visit(&mut config.epsilon, Read)?;
                visit(&mut outputs.output, Write)?;
                visit_saved_statistics(&mut outputs.statistics, Write, visit)
            }
            Self::AdaptiveLayer {
                inputs,
                outputs,
                config,
            } => {
                visit(&mut inputs.input, Read)?;
                visit(&mut inputs.scale, Read)?;
                visit_optional(&mut inputs.bias, Read, visit)?;
                visit(&mut config.epsilon, Read)?;
                visit(&mut outputs.output, Write)?;
                visit_saved_statistics(&mut outputs.statistics, Write, visit)
            }
            Self::Batch {
                inputs,
                outputs,
                config,
            } => {
                visit(&mut inputs.input, Read)?;
                visit(&mut inputs.scale, Read)?;
                visit(&mut inputs.bias, Read)?;
                visit(&mut config.epsilon, Read)?;
                if let Some(running) = &mut config.running {
                    visit(&mut running.momentum, Read)?;
                    visit(&mut running.previous.mean, Read)?;
                    visit(&mut running.previous.variance, Read)?;
                }
                for peer in &mut config.peer_stats {
                    visit(peer, ReadWrite)?;
                }
                visit(&mut outputs.output, Write)?;
                visit(&mut outputs.statistics.mean, Write)?;
                visit(&mut outputs.statistics.inv_variance, Write)?;
                if let Some(running) = &mut outputs.running {
                    visit(&mut running.mean, Write)?;
                    visit(&mut running.variance, Write)?;
                }
                Ok(())
            }
            Self::BatchInference { inputs, output, .. } => {
                visit(&mut inputs.input, Read)?;
                visit(&mut inputs.statistics.mean, Read)?;
                visit(&mut inputs.statistics.inv_variance, Read)?;
                visit(&mut inputs.scale, Read)?;
                visit(&mut inputs.bias, Read)?;
                visit(output, Write)
            }
            Self::BatchFinalize {
                inputs,
                outputs,
                config,
            } => {
                for input in [
                    &mut inputs.sum,
                    &mut inputs.squared_sum,
                    &mut inputs.scale,
                    &mut inputs.bias,
                    &mut inputs.accumulation_count,
                    &mut config.epsilon,
                ] {
                    visit(input, Read)?;
                }
                if let Some(running) = &mut config.running {
                    visit(&mut running.momentum, Read)?;
                    visit(&mut running.previous.mean, Read)?;
                    visit(&mut running.previous.variance, Read)?;
                }
                for output in [
                    &mut outputs.equivalent_scale,
                    &mut outputs.equivalent_bias,
                    &mut outputs.statistics.mean,
                    &mut outputs.statistics.inv_variance,
                ] {
                    visit(output, Write)?;
                }
                if let Some(running) = &mut outputs.running {
                    visit(&mut running.mean, Write)?;
                    visit(&mut running.variance, Write)?;
                }
                Ok(())
            }
            Self::LayerBackward {
                inputs, outputs, ..
            } => {
                visit(&mut inputs.input, Read)?;
                visit(&mut inputs.statistics.mean, Read)?;
                visit(&mut inputs.statistics.inv_variance, Read)?;
                visit(&mut inputs.output_gradient, Read)?;
                visit_optional(&mut inputs.scale, Read, visit)?;
                visit(&mut outputs.input_gradient, Write)?;
                if let Some(affine) = &mut outputs.affine {
                    visit(&mut affine.scale_gradient, Write)?;
                    visit(&mut affine.bias_gradient, Write)?;
                }
                Ok(())
            }
            Self::RmsBackward {
                inputs, outputs, ..
            } => {
                visit(&mut inputs.input, Read)?;
                visit(&mut inputs.inv_variance, Read)?;
                visit(&mut inputs.output_gradient, Read)?;
                visit_optional(&mut inputs.scale, Read, visit)?;
                visit(&mut outputs.input_gradient, Write)?;
                if let Some(affine) = &mut outputs.affine {
                    visit(&mut affine.scale_gradient, Write)?;
                    visit_optional(&mut affine.bias_gradient, Write, visit)?;
                }
                Ok(())
            }
            Self::InstanceBackward {
                inputs, outputs, ..
            } => {
                visit(&mut inputs.input, Read)?;
                visit(&mut inputs.statistics.mean, Read)?;
                visit(&mut inputs.statistics.inv_variance, Read)?;
                visit(&mut inputs.output_gradient, Read)?;
                visit(&mut inputs.scale, Read)?;
                visit(&mut outputs.input_gradient, Write)?;
                visit(&mut outputs.scale_gradient, Write)?;
                visit(&mut outputs.bias_gradient, Write)
            }
            Self::AdaptiveLayerBackward {
                inputs, outputs, ..
            } => {
                visit(&mut inputs.input, Read)?;
                visit(&mut inputs.statistics.mean, Read)?;
                visit(&mut inputs.statistics.inv_variance, Read)?;
                visit(&mut inputs.output_gradient, Read)?;
                visit(&mut inputs.scale, Read)?;
                visit(&mut outputs.input_gradient, Write)?;
                visit(&mut outputs.scale_gradient, Write)?;
                visit_optional(&mut outputs.bias_gradient, Write, visit)
            }
            Self::BatchBackward {
                inputs,
                outputs,
                config,
            } => {
                visit(&mut inputs.input, Read)?;
                visit(&mut inputs.statistics.mean, Read)?;
                visit(&mut inputs.statistics.inv_variance, Read)?;
                visit(&mut inputs.output_gradient, Read)?;
                visit(&mut inputs.scale, Read)?;
                for peer in &mut config.peer_stats {
                    visit(peer, ReadWrite)?;
                }
                visit(&mut outputs.input_gradient, Write)?;
                visit(&mut outputs.scale_gradient, Write)?;
                visit(&mut outputs.bias_gradient, Write)
            }
            Self::BatchNormalizationBackwardWeights {
                inputs, outputs, ..
            } => {
                visit(&mut inputs.input, Read)?;
                visit(&mut inputs.output_gradient, Read)?;
                visit(&mut inputs.scale, Read)?;
                visit(&mut inputs.statistics.mean, Read)?;
                visit(&mut inputs.statistics.inv_variance, Read)?;
                for output in [
                    &mut outputs.scale_gradient,
                    &mut outputs.bias_gradient,
                    &mut outputs.equivalent_bias,
                    &mut outputs.equivalent_output_gradient_scale,
                    &mut outputs.equivalent_input_scale,
                ] {
                    visit(output, Write)?;
                }
                Ok(())
            }
        }
    }
}

fn visit_sequence(
    sequence: &mut AttentionSequenceInputs<TensorId>,
    visit: &mut TensorVisitor<'_>,
) -> Result<()> {
    *sequence = sequence.try_map(map_with(visit, TensorRole::Index))?;
    Ok(())
}

impl SoftmaxConfig {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        visit_optional(self.sink_mut(), TensorRole::Read, visit)
    }
}

impl SdpaScoreModifiers {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        visit_optional(&mut self.bias, TensorRole::Read, visit)?;
        visit_optional(&mut self.sink_token, TensorRole::Read, visit)?;
        visit_optional(&mut self.additive_mask, TensorRole::Read, visit)?;
        if let Some(AlibiBias::Slopes(slopes)) = &mut self.alibi {
            visit(slopes, TensorRole::Read)?;
        }
        if let Some(dropout) = &mut self.custom_dropout {
            visit(&mut dropout.mask, TensorRole::Read)?;
            visit(&mut dropout.scale, TensorRole::Read)?;
        }
        if let AttentionMaskMode::Block(block_mask) = &mut self.mask_mode {
            visit(block_mask, TensorRole::Index)?;
        }
        visit_sequence(&mut self.sequence, visit)
    }
}

impl AttentionDropoutConfig {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        self.seed = self.seed.try_map(map_with(visit, TensorRole::Read))?;
        Ok(())
    }
}

impl AttentionScoreConfig {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        self.modifiers.visit_tensor_ids(visit)?;
        if let Some(dropout) = &mut self.dropout {
            dropout.visit_tensor_ids(visit)?;
        }
        Ok(())
    }
}

impl AttentionConfig {
    /// Drops zero-probability dropout first, so neither uses nor remapping see its seed.
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        self.remove_zero_probability_dropout();
        self.softmax.visit_tensor_ids(visit)?;
        self.score.visit_tensor_ids(visit)?;
        if let SdpaFusedMaskMode::CausalBottomRight {
            sequence_length_query,
            sequence_length_key_value,
        } = &mut self.fused.mask
        {
            visit(sequence_length_query, TensorRole::Index)?;
            visit(sequence_length_key_value, TensorRole::Index)?;
        }
        for cache in [&mut self.paged.k, &mut self.paged.v].into_iter().flatten() {
            visit(&mut cache.page_table, TensorRole::Index)?;
        }
        Ok(())
    }
}

impl AttentionBackwardConfig {
    /// Drops an inactive dropout-scale inverse first, so neither uses nor remapping see it.
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        self.remove_inactive_dropout_scale_inverse();
        self.softmax.visit_tensor_ids(visit)?;
        self.score.visit_tensor_ids(visit)?;
        visit_optional(&mut self.dropout_scale_inverse, TensorRole::Read, visit)
    }
}

impl DirectSdpaDropout {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        visit(&mut self.seed, TensorRole::Read)?;
        visit(&mut self.offset, TensorRole::Read)?;
        visit_optional(
            &mut self.random_number_generator_dump,
            TensorRole::Write,
            visit,
        )
    }
}

impl DirectSdpaForwardConfig {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        visit_sequence(&mut self.sequence, visit)?;
        visit_optional(&mut self.key_page_table, TensorRole::Index, visit)?;
        visit_optional(&mut self.value_page_table, TensorRole::Index, visit)?;
        visit_optional(&mut self.block_mask, TensorRole::Index, visit)?;
        if let Some(dropout) = &mut self.dropout {
            dropout.visit_tensor_ids(visit)?;
        }
        if let Some(quantization) = &mut self.quantization {
            for scale in [
                &mut quantization.descale_query,
                &mut quantization.descale_key,
                &mut quantization.descale_value,
                &mut quantization.descale_scores,
                &mut quantization.scale_scores,
                &mut quantization.scale_output,
            ] {
                visit_optional(scale, TensorRole::Read, visit)?;
            }
            visit_optional(
                &mut quantization.absolute_max_probabilities,
                TensorRole::Write,
                visit,
            )?;
            visit_optional(
                &mut quantization.absolute_max_output,
                TensorRole::Write,
                visit,
            )?;
        }
        Ok(())
    }
}

impl DirectSdpaBackwardConfig {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        visit_sequence(&mut self.sequence, visit)?;
        if let Some(sink) = &mut self.sink {
            visit(&mut sink.token, TensorRole::Read)?;
            visit_optional(&mut sink.gradient, TensorRole::Write, visit)?;
        }
        Ok(())
    }
}

impl SdpaOperation {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        match self {
            Self::Forward {
                tensors,
                stats,
                logit_max,
                score_sum_exp,
                softmax_p,
                softmax_s,
                sink,
                score_modifiers,
                config,
                ..
            } => {
                for input in [
                    &mut tensors.query,
                    &mut tensors.key,
                    &mut tensors.value,
                    &mut tensors.scale,
                ] {
                    visit(input, TensorRole::Read)?;
                }
                visit_optional(sink, TensorRole::Read, visit)?;
                visit(&mut tensors.output, TensorRole::Write)?;
                for output in [stats, logit_max, score_sum_exp, softmax_p, softmax_s] {
                    visit_optional(output, TensorRole::Write, visit)?;
                }
                score_modifiers.visit_tensor_ids(visit)?;
                config.visit_tensor_ids(visit)
            }
            Self::Backward { tensors, config } => {
                for input in [
                    &mut tensors.query,
                    &mut tensors.key,
                    &mut tensors.value,
                    &mut tensors.output,
                    &mut tensors.stats,
                    &mut tensors.scale,
                    &mut tensors.output_gradient,
                ] {
                    visit(input, TensorRole::Read)?;
                }
                for output in [
                    &mut tensors.query_gradient,
                    &mut tensors.key_gradient,
                    &mut tensors.value_gradient,
                ] {
                    visit(output, TensorRole::Write)?;
                }
                config.visit_tensor_ids(visit)
            }
        }
    }

    /// Uses of the native SDPA node, without the score modifiers that its score graph consumes.
    pub(crate) fn native_tensor_uses(&self) -> TensorUses {
        let mut operation = self.clone();
        if let Self::Forward {
            score_modifiers, ..
        } = &mut operation
        {
            *score_modifiers = SdpaScoreModifiers::new();
        }
        collect_tensor_uses(|visit| operation.visit_tensor_ids(visit))
    }
}

impl AttentionForwardRequest {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        match &mut self.inputs {
            AttentionForwardInputs::Floating(inputs) => {
                inputs.visit_tensor_ids(TensorRole::Read, visit)?;
            }
            AttentionForwardInputs::Fp8(inputs) => {
                inputs.visit_tensor_ids(TensorRole::Read, visit)?;
            }
            AttentionForwardInputs::Mxfp8(inputs) => {
                inputs.visit_tensor_ids(TensorRole::Read, visit)?;
            }
        }
        let outputs = &mut self.outputs;
        visit(&mut outputs.output, TensorRole::Write)?;
        for output in [
            &mut outputs.stats,
            &mut outputs.logit_max,
            &mut outputs.score_sum_exp,
            &mut outputs.rng_dump,
            &mut outputs.absolute_max_probabilities,
            &mut outputs.absolute_max_output,
        ] {
            visit_optional(output, TensorRole::Write, visit)?;
        }
        self.config.visit_tensor_ids(visit)
    }

    /// Uses of the request's own tensors, excluding tensors captured by its score subgraph.
    pub(crate) fn direct_tensor_uses(&self) -> TensorUses {
        collect_tensor_uses(|visit| self.clone().visit_tensor_ids(visit))
    }
}

impl AttentionBackwardRequest {
    fn visit_tensor_ids(&mut self, visit: &mut TensorVisitor<'_>) -> Result<()> {
        match self {
            Self::Floating {
                inputs,
                outputs,
                config,
            } => {
                inputs.visit_tensor_ids(TensorRole::Read, visit)?;
                outputs.visit_tensor_ids(TensorRole::Write, visit)?;
                config.visit_tensor_ids(visit)
            }
            Self::Fp8 {
                inputs,
                outputs,
                config,
            } => {
                inputs.visit_tensor_ids(TensorRole::Read, visit)?;
                outputs.visit_tensor_ids(TensorRole::Write, visit)?;
                config.visit_tensor_ids(visit)
            }
            Self::Mxfp8 {
                inputs,
                outputs,
                config,
            } => {
                inputs.visit_tensor_ids(TensorRole::Read, visit)?;
                outputs.visit_tensor_ids(TensorRole::Write, visit)?;
                config.visit_tensor_ids(visit)
            }
        }
    }

    /// Uses of the request's own tensors, excluding tensors captured by its score subgraphs.
    pub(crate) fn direct_tensor_uses(&self) -> TensorUses {
        collect_tensor_uses(|visit| self.clone().visit_tensor_ids(visit))
    }
}
