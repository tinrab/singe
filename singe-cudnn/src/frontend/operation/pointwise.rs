use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType, math::NanPropagation, pointwise::PointwiseMode, tensor::TensorId,
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
