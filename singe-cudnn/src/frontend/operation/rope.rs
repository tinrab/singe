use serde::{Deserialize, Serialize};

pub use crate::backend::execution::rope::{RopeBackwardInputs, RopeConfig, RopeInputs};
use crate::{
    backend::{
        execution::rope::{RopeBackwardOperation, RopeForwardOperation},
        tensor::TensorId,
    },
    error::Result,
    frontend::{
        graph::Graph,
        lower::{LoweredOperation, LoweringContext, tensor_at},
    },
};

/// Frontend forward or backward rotary position embedding operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RopeOperation {
    /// Applies the configured forward rotation to activation values.
    Forward {
        /// Activation and raw-angle frequency tensor identities.
        inputs: RopeInputs<TensorId>,
        /// Rotated activation destination identity.
        output: TensorId,
        /// Rotation dimension, output scale, and optional diagnostic name.
        config: RopeConfig,
    },
    /// Applies the transpose rotation to produce input gradients.
    Backward {
        /// Output-gradient and raw-angle frequency tensor identities.
        inputs: RopeBackwardInputs<TensorId>,
        /// Input-gradient destination identity.
        input_gradient: TensorId,
        /// Rotation dimension and output scale matching the corresponding forward rotation.
        config: RopeConfig,
    },
}

impl RopeOperation {
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        let (input, frequencies, output, config, name) = match self {
            Self::Forward {
                inputs,
                output,
                config,
            } => (
                inputs.input,
                inputs.frequencies,
                *output,
                config,
                "rope output",
            ),
            Self::Backward {
                inputs,
                input_gradient,
                config,
            } => (
                inputs.output_gradient,
                inputs.frequencies,
                *input_gradient,
                config,
                "rope input gradient",
            ),
        };
        config.validate()?;
        graph.validate_tensor_dimensions(output, graph.shape(input)?.dimensions(), name)?;
        graph.tensor_spec(frequencies)?;
        Ok(())
    }

    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        Ok(match self {
            Self::Forward {
                inputs,
                output,
                config,
            } => LoweredOperation::RopeForward(RopeForwardOperation::create(
                (*inputs).try_map(|id| tensor_at(tensors, id))?,
                tensor_at(tensors, *output)?,
                config,
            )?),
            Self::Backward {
                inputs,
                input_gradient,
                config,
            } => LoweredOperation::RopeBackward(RopeBackwardOperation::create(
                (*inputs).try_map(|id| tensor_at(tensors, id))?,
                tensor_at(tensors, *input_gradient)?,
                config,
            )?),
        })
    }
}
