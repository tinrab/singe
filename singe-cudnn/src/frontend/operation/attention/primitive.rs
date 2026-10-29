use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        execution::attention::{
            DiagonalBandMaskOperation, SoftmaxOperation, SoftmaxOperationConfig,
        },
        tensor::TensorId,
    },
    error::{Error, Result},
    frontend::{
        graph::{Graph, SoftmaxOutputs},
        lower::{LoweredOperation, LoweringContext, optional_tensor_at, tensor_at},
        operation::{DiagonalBandMaskConfig, SoftmaxStatsBase},
    },
};

/// Frontend attention primitive operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AttentionPrimitiveOperation {
    /// Selects values using a diagonal-band comparison and optional sequence-length operands.
    DiagonalBandMask {
        /// Input activation or softmax-logit tensor identity.
        x: TensorId,
        /// Scalar replacement tensor identity for positions excluded by the mask.
        b: TensorId,
        /// Masked activation or probability output tensor identity.
        y: TensorId,
        /// Comparison, sequence-length operands, and diagonal-bound settings.
        config: DiagonalBandMaskConfig,
    },
    /// Native attention softmax with independently optional row summaries and sink logits.
    Softmax {
        /// Encoding base for saved softmax log-sum-exp statistics.
        stats_base: SoftmaxStatsBase,
        /// Input activation or softmax-logit tensor identity.
        x: TensorId,
        /// Masked activation or probability output tensor identity.
        y: TensorId,
        /// Optional saved softmax log-sum-exp destination identity.
        stats: Option<TensorId>,
        /// Optional per-row maximum destination identity.
        max: Option<TensorId>,
        /// Optional per-row shifted-exponential-sum destination identity.
        sum_exp: Option<TensorId>,
        /// Optional sink-logit tensor identity broadcastable to row summaries.
        sink: Option<TensorId>,
    },
}

impl AttentionPrimitiveOperation {
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        match self {
            Self::DiagonalBandMask { x, b, y, config } => {
                let input = graph.tensor_spec(*x)?;
                let batch = input
                    .shape
                    .dimensions()
                    .first()
                    .copied()
                    .ok_or(Error::InvalidDataShape)?;
                graph.validate_tensor_data_type(*b, input.data_type, "diagonal band mask b")?;
                graph.validate_tensor_dimensions(
                    *y,
                    input.shape.dimensions(),
                    "diagonal band mask output shape",
                )?;
                graph.validate_tensor_scalar_shape(*b, "diagonal band mask b shape")?;
                graph.validate_attention_sequence_inputs(config.sequence_inputs(), batch)?;
                if let Some(bound) = config.bound().tensor() {
                    graph.validate_tensor_scalar_shape(bound, "diagonal band mask bound shape")?;
                }
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
                graph.validate_softmax_tensors(
                    *x,
                    SoftmaxOutputs {
                        output: *y,
                        stats: *stats,
                        max: *max,
                        sum_exp: *sum_exp,
                    },
                    *sink,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        match self {
            Self::DiagonalBandMask { x, b, y, config } => Ok(LoweredOperation::DiagonalBandMask(
                DiagonalBandMaskOperation::create(
                    tensor_at(tensors, *x)?,
                    tensor_at(tensors, *b)?,
                    tensor_at(tensors, *y)?,
                    config.clone().try_map(|id| tensor_at(tensors, id))?,
                )?,
            )),
            Self::Softmax {
                stats_base,
                x,
                y,
                stats,
                max,
                sum_exp,
                sink,
            } => Ok(LoweredOperation::Softmax(SoftmaxOperation::create(
                tensor_at(tensors, *x)?,
                SoftmaxOutputs {
                    output: *y,
                    stats: *stats,
                    max: *max,
                    sum_exp: *sum_exp,
                }
                .try_map(|id| tensor_at(tensors, id))?,
                SoftmaxOperationConfig {
                    stats_base: *stats_base,
                    sink: optional_tensor_at(tensors, *sink)?,
                },
            )?)),
        }
    }
}
