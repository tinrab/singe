use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType,
    error::Result,
    execution::pointwise::{PointwiseDescriptor, PointwiseOperation as BackendPointwiseOperation},
    frontend::{
        lower::{LoweredOperation, LoweringContext, tensor_at},
        operation::FrontendOperationTensors,
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    tensor::TensorId,
};

/// Frontend pointwise operation variants.
///
/// cuDNN pointwise operations apply unary, binary, or ternary elementwise modes
/// selected by [`PointwiseMode`]. Some modes use additional attributes such as
/// ReLU clipping parameters, axis, scaling factors, compute type, and NaN
/// propagation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PointwiseOperation {
    Unary {
        mode: PointwiseMode,
        input: TensorId,
        output: TensorId,
        compute_type: DataType,
        nan_propagation: NanPropagation,
        alpha1: f64,
        axis: Option<i64>,
    },
    ReluForward {
        input: TensorId,
        output: TensorId,
        compute_type: DataType,
        nan_propagation: NanPropagation,
        lower_clip: f64,
        upper_clip: f64,
        lower_clip_slope: f64,
        axis: Option<i64>,
    },
    Binary {
        mode: PointwiseMode,
        lhs: TensorId,
        rhs: TensorId,
        output: TensorId,
        compute_type: DataType,
        nan_propagation: NanPropagation,
        alpha1: f64,
        alpha2: f64,
    },
    Ternary {
        mode: PointwiseMode,
        x: TensorId,
        b: TensorId,
        t: TensorId,
        output: TensorId,
        compute_type: DataType,
        nan_propagation: NanPropagation,
        alpha1: f64,
        alpha2: f64,
    },
}

impl FrontendOperationTensors for PointwiseOperation {
    fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>) {
        match self {
            Self::Unary { input, output, .. } | Self::ReluForward { input, output, .. } => {
                tensors.extend([*input, *output]);
            }
            Self::Binary {
                lhs, rhs, output, ..
            } => {
                tensors.extend([*lhs, *rhs, *output]);
            }
            Self::Ternary {
                x, b, t, output, ..
            } => {
                tensors.extend([*x, *b, *t, *output]);
            }
        }
    }
}

impl PointwiseOperation {
    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        Ok(LoweredOperation::Pointwise(match self {
            Self::Unary {
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
            Self::ReluForward {
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
            Self::Binary {
                mode,
                lhs,
                rhs,
                output,
                compute_type,
                nan_propagation,
                alpha1,
                alpha2,
            } => {
                let descriptor =
                    PointwiseDescriptor::create(*mode, *compute_type, *nan_propagation)?;
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
            Self::Ternary {
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
                let descriptor =
                    PointwiseDescriptor::create(*mode, *compute_type, *nan_propagation)?;
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
