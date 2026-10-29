use serde::{Deserialize, Serialize};

pub use crate::backend::execution::tensor_ops::{
    ConcatConfig, ConcatInPlaceMode, ReshapeConfig, ReshapeMode, SliceConfig, TransposeConfig,
};
use crate::{
    backend::{
        execution::tensor_ops::{
            ConcatOperation, ReshapeOperation, SliceOperation, TransposeOperation,
        },
        tensor::TensorId,
    },
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::Graph,
        lower::{LoweredOperation, LoweringContext, tensor_at, tensor_slice_at},
    },
};

/// Frontend tensor shape/view operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TensorOperation {
    /// Reshapes input values into the declared output geometry.
    Reshape {
        /// Input tensor identity.
        input: TensorId,
        /// Explicit output tensor identity.
        output: TensorId,
        /// Operation-specific geometry, mode, and diagnostic-name settings.
        config: ReshapeConfig,
    },
    /// Selects an input slice described by the configuration.
    Slice {
        /// Input tensor identity.
        input: TensorId,
        /// Explicit output tensor identity.
        output: TensorId,
        /// Operation-specific geometry, mode, and diagnostic-name settings.
        config: SliceConfig,
    },
    /// Permutes input axes into the declared output.
    Transpose {
        /// Input tensor identity.
        input: TensorId,
        /// Explicit output tensor identity.
        output: TensorId,
        /// Operation-specific geometry, mode, and diagnostic-name settings.
        config: TransposeConfig,
    },
    /// Concatenates inputs along the configured axis.
    Concat {
        /// Ordered input tensor identities.
        inputs: Vec<TensorId>,
        /// Explicit output tensor identity.
        output: TensorId,
        /// Operation-specific geometry, mode, and diagnostic-name settings.
        config: ConcatConfig,
    },
}

impl TensorOperation {
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        match self {
            Self::Reshape {
                input,
                output,
                config,
            } => {
                let input_spec = graph.tensor_spec(*input)?;
                if config.mode() == ReshapeMode::Logical {
                    graph.validate_tensor_data_type(
                        *output,
                        input_spec.data_type,
                        "reshape output",
                    )?;
                }
                if !input_spec.has_same_logical_element_count(graph.tensor_spec(*output)?)? {
                    return Err(FrontendError::ReshapeElementCountMismatch {
                        input_id: *input,
                        output_id: *output,
                    }
                    .into());
                }
            }
            Self::Slice {
                input,
                output,
                config,
            } => {
                let input = graph.tensor_spec(*input)?;
                let shape = input.shape_with_scalar_rank(config.slices().len())?;
                let expected = config.infer_output_dimensions(&shape)?;
                graph.validate_tensor_data_type_and_dimensions(
                    *output,
                    input.data_type,
                    &expected,
                    "slice output",
                )?;
            }
            Self::Transpose {
                input,
                output,
                config,
            } => {
                let input = graph.tensor_spec(*input)?;
                let shape = input.shape_with_scalar_rank(config.permutation().len())?;
                let expected = config.infer_output_dimensions(&shape)?;
                graph.validate_tensor_data_type_and_dimensions(
                    *output,
                    input.data_type,
                    &expected,
                    "transpose output",
                )?;
            }
            Self::Concat {
                inputs,
                output,
                config,
            } => {
                graph.validate_concat(inputs, *output, config)?;
            }
        }
        Ok(())
    }

    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        Ok(match self {
            Self::Reshape {
                input,
                output,
                config,
            } => LoweredOperation::Reshape(ReshapeOperation::create(
                tensor_at(tensors, *input)?,
                tensor_at(tensors, *output)?,
                config.clone(),
            )?),
            Self::Slice {
                input,
                output,
                config,
            } => LoweredOperation::Slice(SliceOperation::create(
                tensor_at(tensors, *input)?,
                tensor_at(tensors, *output)?,
                config.clone(),
            )?),
            Self::Transpose {
                input,
                output,
                config,
            } => LoweredOperation::Transpose(TransposeOperation::create(
                tensor_at(tensors, *input)?,
                tensor_at(tensors, *output)?,
                config.clone(),
            )?),
            Self::Concat {
                inputs,
                output,
                config,
            } => {
                let input_tensors = tensor_slice_at(tensors, inputs)?;
                LoweredOperation::Concat(ConcatOperation::create(
                    &input_tensors,
                    tensor_at(tensors, *output)?,
                    config.clone(),
                )?)
            }
        })
    }
}
