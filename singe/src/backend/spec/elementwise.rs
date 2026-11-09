use crate::{
    error::{Error, Result},
    shape::Shape,
};

/// Common output shape and aligned input dimensions for binary broadcasting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryBroadcastProblem {
    /// Shape shared by both inputs after broadcasting.
    pub output_shape: Shape,
    /// Left input dimensions padded on the left with ones to the output rank.
    pub lhs_padded_dimensions: Vec<i64>,
    /// Right input dimensions padded on the left with ones to the output rank.
    pub rhs_padded_dimensions: Vec<i64>,
}

/// Output shape and aligned dimensions for elementwise tensor operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementwiseProblem {
    /// Common shape after broadcasting the operands.
    pub output_shape: Shape,
    /// Each input shape padded on the left with ones to the output rank, in operand order.
    pub input_padded_dimensions: Vec<Vec<i64>>,
}

/// Infers a common shape for condition, true-value, and false-value operands.
///
/// Operand dimensions appear in that order.
/// This checks shapes without validating element types.
///
/// # Errors
///
/// Returns an error if the three shapes cannot broadcast or constructing an intermediate or final broadcast shape fails.
pub fn where_elementwise_problem(
    condition: &Shape,
    lhs: &Shape,
    rhs: &Shape,
) -> Result<ElementwiseProblem> {
    let condition_lhs = binary_broadcast_problem(condition, lhs)?;
    let output = binary_broadcast_problem(&condition_lhs.output_shape, rhs)?;
    let rank = output.output_shape.dimensions().len();
    Ok(ElementwiseProblem {
        output_shape: output.output_shape,
        input_padded_dimensions: vec![
            Shape::pad_left(condition.dimensions(), rank),
            Shape::pad_left(lhs.dimensions(), rank),
            Shape::pad_left(rhs.dimensions(), rank),
        ],
    })
}

/// Broadcast dimensions and scalar weight for linear interpolation.
#[derive(Debug, Clone, PartialEq)]
pub struct LerpProblem {
    /// Common shape of the interpolated output.
    pub output_shape: Shape,
    /// Start-value dimensions padded on the left with ones to the output rank.
    pub lhs_padded_dimensions: Vec<i64>,
    /// End-value dimensions padded on the left with ones to the output rank.
    pub rhs_padded_dimensions: Vec<i64>,
    /// Interpolation weight in `lhs + t * (rhs - lhs)`, stored without a range or finiteness check.
    pub t: f32,
}

/// Infers broadcast dimensions and stores the interpolation weight `t`.
///
/// The weight is not restricted to `[0, 1]` and is not checked for finiteness.
///
/// # Errors
///
/// Returns an error if input shapes cannot broadcast or constructing the output shape fails.
pub fn lerp_problem(lhs: &Shape, rhs: &Shape, t: f32) -> Result<LerpProblem> {
    let broadcast = binary_broadcast_problem(lhs, rhs)?;
    Ok(LerpProblem {
        output_shape: broadcast.output_shape,
        lhs_padded_dimensions: broadcast.lhs_padded_dimensions,
        rhs_padded_dimensions: broadcast.rhs_padded_dimensions,
        t,
    })
}

/// Output shape and bounds for elementwise clamping.
#[derive(Debug, Clone, PartialEq)]
pub struct ClampProblem {
    /// Shape of the clamped result, identical to the input shape.
    pub output_shape: Shape,
    /// Inclusive lower bound.
    pub min: f32,
    /// Inclusive upper bound.
    pub max: f32,
}

/// Preserves the input shape and validates the clamp interval.
///
/// Only `min > max` is rejected.
/// NaN bounds are retained for the executor's NaN propagation.
///
/// # Errors
///
/// Returns an error if `min` is greater than `max`.
pub fn clamp_problem(input: &Shape, min: f32, max: f32) -> Result<ClampProblem> {
    if min > max {
        return Err(Error::invalid_argument("clamp bounds"));
    }
    Ok(ClampProblem {
        output_shape: input.clone(),
        min,
        max,
    })
}

/// Broadcasts logical dimensions without constructing unused intermediate strides.
///
/// Shorter inputs receive leading singleton axes.
/// This function checks compatibility only; it does not validate negative extents or dimension products.
///
/// # Errors
///
/// Returns an error if aligned extents differ and neither is one.
/// The supplied `name` identifies the operation in the error.
pub fn broadcast_dimensions(lhs: &[i64], rhs: &[i64], name: &str) -> Result<Vec<i64>> {
    Shape::broadcast_dimensions(lhs, rhs, name)
}

/// Infers a common shape and left-padded dimensions for `lhs` and `rhs`.
///
/// Aligned extents must agree or one must be one.
///
/// # Errors
///
/// Returns an error if the shapes cannot broadcast or constructing the output shape fails validation or size arithmetic.
pub fn binary_broadcast_problem(lhs: &Shape, rhs: &Shape) -> Result<BinaryBroadcastProblem> {
    let max_rank = lhs.dimensions().len().max(rhs.dimensions().len());
    let lhs_padded_dims = Shape::pad_left(lhs.dimensions(), max_rank);
    let rhs_padded_dims = Shape::pad_left(rhs.dimensions(), max_rank);

    let output_dims = broadcast_dimensions(&lhs_padded_dims, &rhs_padded_dims, "binary broadcast")?;

    Ok(BinaryBroadcastProblem {
        output_shape: Shape::create(output_dims)?,
        lhs_padded_dimensions: lhs_padded_dims,
        rhs_padded_dimensions: rhs_padded_dims,
    })
}
