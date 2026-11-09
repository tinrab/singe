mod backward_weights;
mod batch;
mod batch_backward;
mod finalize;
mod layer_rms;
mod other_backward;
mod other_forward;

use crate::{
    backend::tensor::Shape,
    error::{Error, Result},
    frontend::error::Error as FrontendError,
};

pub fn validate_normalization_broadcast_shape(input: &Shape, operand: &Shape) -> Result<()> {
    if input.rank() < 2
        || operand.rank() != input.rank()
        || operand
            .dimensions()
            .iter()
            .zip(input.dimensions())
            .any(|(&operand, &input)| operand != 1 && operand != input)
    {
        return Err(FrontendError::NormalizationAxes {
            reason: "normalization operands must broadcast against the input at the same rank"
                .into(),
        }
        .into());
    }
    Ok(())
}

pub fn infer_adaptive_layer_normalization_stats_dimensions(
    input: &Shape,
    scale: &Shape,
) -> Result<Vec<i64>> {
    validate_normalization_broadcast_shape(input, scale)?;
    let mut dimensions = input.dimensions().to_vec();
    for (dimension, &scale) in dimensions.iter_mut().zip(scale.dimensions()).skip(1) {
        if scale != 1 {
            *dimension = 1;
        }
    }
    Ok(dimensions)
}

pub fn infer_normalization_channel_dimensions(input: &Shape) -> Result<Vec<i64>> {
    let mut dimensions = input.dimensions().to_vec();
    if dimensions.len() < 2 {
        return Err(Error::InvalidDataShape);
    }

    for (index, dimension) in dimensions.iter_mut().enumerate() {
        if index != 1 {
            *dimension = 1;
        }
    }

    Ok(dimensions)
}

pub fn infer_instance_normalization_stats_dimensions(input: &Shape) -> Result<Vec<i64>> {
    let mut dimensions = input.dimensions().to_vec();
    if dimensions.len() < 2 {
        return Err(Error::InvalidDataShape);
    }

    for dimension in dimensions.iter_mut().skip(2) {
        *dimension = 1;
    }

    Ok(dimensions)
}
