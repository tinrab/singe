use crate::{
    backend::spec::dimension_product_usize,
    error::{Error, Result},
    shape::Shape,
    utility::to_usize,
};

/// Validated channel-wise normalization dimensions and shape-preserving output metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchNormProblem {
    /// Output shape, with the same dimensions as the input.
    pub output_shape: Shape,
    /// Number of batch elements; zero is allowed.
    pub batch_size: usize,
    /// Positive channel count from input axis one.
    pub feature_count: usize,
    /// Product of dimensions after the channel axis, or one for a rank-two input.
    pub spatial_len: usize,
}

/// Checks batch-normalization shape and optional parameter element counts.
///
/// Input rank must be at least two.
/// Every optional parameter must contain one element per channel; its exact rank is not constrained.
/// `diagnostic_prefix` identifies the operation in errors.
///
/// # Errors
///
/// Returns an error if rank is too small, batch size is negative, channels or spatial extents are nonpositive, a parameter element count differs, dimension products or conversions overflow, or constructing the output shape fails.
pub fn batch_norm_problem(
    input: &Shape,
    weight: Option<&Shape>,
    bias: Option<&Shape>,
    running_mean: Option<&Shape>,
    running_var: Option<&Shape>,
    diagnostic_prefix: &'static str,
) -> Result<BatchNormProblem> {
    let dimensions = input.dimensions();
    if dimensions.len() < 2 {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} minimum rank"),
            2,
            dimensions.len(),
        ));
    }

    let batch_size = dimensions[0];
    let feature_count = dimensions[1];
    if batch_size < 0 {
        return Err(Error::invalid_argument(format!(
            "{diagnostic_prefix} batch size"
        )));
    }

    if feature_count <= 0 {
        return Err(Error::invalid_argument(format!(
            "{diagnostic_prefix} feature count"
        )));
    }

    let spatial_len = dimension_product_usize(
        format!("{diagnostic_prefix} spatial length"),
        &dimensions[2..],
        false,
    )?;
    let feature_count = to_usize(feature_count, "backend spec dimension")?;
    for (name, parameter) in [
        ("weight", weight),
        ("bias", bias),
        ("running mean", running_mean),
        ("running var", running_var),
    ] {
        if let Some(parameter) = parameter {
            let actual = parameter.element_count();
            if actual != feature_count {
                return Err(Error::length_mismatch(
                    format!("{diagnostic_prefix} {name} elements"),
                    feature_count,
                    actual,
                ));
            }
        }
    }

    Ok(BatchNormProblem {
        output_shape: Shape::create(dimensions.to_vec())?,
        batch_size: to_usize(batch_size, "backend spec dimension")?,
        feature_count,
        spatial_len,
    })
}

/// Trailing normalized dimensions and row counts for layer or root-mean-square normalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizationProblem {
    /// Output shape, identical to the input shape.
    pub output_shape: Shape,
    /// Nonempty trailing input dimensions reduced for each normalization row.
    pub normalized_shape: Vec<i64>,
    /// Zero-based first normalized axis.
    pub split_axis: usize,
    /// Product of the positive normalized dimensions.
    pub feature_size: usize,
    /// Product of dimensions before the normalized axes; zero is allowed.
    pub batch_count: usize,
}

/// Checks optional affine parameters against the normalized element count.
///
/// Only element counts are checked, not exact ranks or dimensions.
/// `diagnostic_prefix` identifies the operation in errors.
///
/// # Errors
///
/// Returns an error if a supplied weight or bias does not contain `problem.feature_size` elements.
pub fn validate_normalization_parameters(
    problem: &NormalizationProblem,
    weight: Option<&Shape>,
    bias: Option<&Shape>,
    diagnostic_prefix: &'static str,
) -> Result<()> {
    for (name, shape) in [("weight", weight), ("bias", bias)] {
        if let Some(shape) = shape {
            let actual = shape.element_count();
            if actual != problem.feature_size {
                return Err(Error::length_mismatch(
                    format!("{diagnostic_prefix} {name} elements"),
                    problem.feature_size,
                    actual,
                ));
            }
        }
    }
    Ok(())
}

/// Validates a nonempty normalized shape matching the trailing input dimensions.
///
/// Normalized extents must be positive.
/// Leading dimensions describe independent rows and may include zero.
/// `diagnostic_prefix` identifies the operation in errors.
///
/// # Errors
///
/// Returns an error if normalized rank is zero or exceeds input rank, trailing dimensions differ, a normalized extent is nonpositive, or a dimension conversion or product overflows the host size.
pub fn normalization_problem(
    input: &Shape,
    normalized_shape: &[i64],
    diagnostic_prefix: &'static str,
) -> Result<NormalizationProblem> {
    let input_dimensions = input.dimensions();
    let rank = input_dimensions.len();
    let normalized_rank = normalized_shape.len();
    if normalized_rank == 0 || normalized_rank > rank {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} normalized rank"),
            rank,
            normalized_rank,
        ));
    }

    let split_axis = rank - normalized_rank;
    if input_dimensions[split_axis..] != *normalized_shape {
        return Err(Error::shape_mismatch(
            format!("{diagnostic_prefix} normalized shape"),
            normalized_shape.to_vec(),
            input_dimensions[split_axis..].to_vec(),
        ));
    }

    let feature_size = dimension_product_usize(
        format!("{diagnostic_prefix} feature size"),
        normalized_shape,
        false,
    )?;
    let batch_count = dimension_product_usize(
        format!("{diagnostic_prefix} batch count"),
        &input_dimensions[..split_axis],
        true,
    )?;

    Ok(NormalizationProblem {
        output_shape: input.clone(),
        normalized_shape: normalized_shape.to_vec(),
        split_axis,
        feature_size,
        batch_count,
    })
}
