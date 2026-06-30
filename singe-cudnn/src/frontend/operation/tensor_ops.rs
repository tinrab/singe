use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    execution::tensor_ops::{ConcatOperation, ReshapeOperation},
    frontend::{
        lower::{LoweredOperation, LoweringContext, LoweringOutput, tensor_at, tensor_slice_at},
        operation::FrontendOperationTensors,
        plan::BindingReplacement,
    },
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

/// Frontend tensor shape/view operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TensorOperation {
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
}

impl FrontendOperationTensors for TensorOperation {
    fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>) {
        match self {
            Self::Reshape { input, output }
            | Self::Slice { input, output, .. }
            | Self::Transpose { input, output, .. } => {
                tensors.extend([*input, *output]);
            }
            Self::Concat { inputs, output, .. } => {
                tensors.extend(inputs.iter().copied());
                tensors.push(*output);
            }
        }
    }
}

impl TensorOperation {
    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweringOutput> {
        let tensors = context.backend_tensors();
        Ok(match self {
            Self::Reshape { input, output } => {
                LoweringOutput::Operation(LoweredOperation::Reshape(ReshapeOperation::create(
                    tensor_at(tensors, *input)?,
                    tensor_at(tensors, *output)?,
                )?))
            }
            Self::Slice {
                input,
                output,
                byte_offset,
                ..
            } => LoweringOutput::BindingReplacement(BindingReplacement {
                source_id: context.backend_id(*input)?,
                target_id: context.backend_id(*output)?,
                byte_offset: *byte_offset,
            }),
            Self::Transpose { input, output, .. } => {
                LoweringOutput::BindingReplacement(BindingReplacement {
                    source_id: context.backend_id(*input)?,
                    target_id: context.backend_id(*output)?,
                    byte_offset: 0,
                })
            }
            Self::Concat {
                inputs,
                output,
                axis,
                in_place,
            } => {
                let input_tensors = tensor_slice_at(tensors, inputs)?;
                LoweringOutput::Operation(LoweredOperation::Concat(ConcatOperation::create(
                    &input_tensors,
                    tensor_at(tensors, *output)?,
                    *axis,
                    in_place.index(),
                )?))
            }
        })
    }
}
