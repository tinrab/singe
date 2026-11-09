use crate::{
    backend::spec::{dimension_product_usize, validate_rank},
    error::{Error, Result},
    shape::{Shape, normalize_dim_usize},
    utility::{checked_add_i64, checked_mul_i64, to_usize},
};

/// Output shape and normalized per-axis widths for nonnegative tensor padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PadProblem {
    /// Input shape with each axis extended by its before and after padding.
    pub output_shape: Shape,
    /// Before and after widths for every input axis, including zero widths for omitted axes.
    pub padding: Vec<(i64, i64)>,
    /// Whether any normalized padding width is nonzero.
    pub has_padding: bool,
}

/// Validates padding and computes the padded output shape.
///
/// Entries correspond to input axes in order.
/// Omitted axes receive zero padding.
/// `diagnostic_prefix` identifies the operation in errors.
///
/// # Errors
///
/// Returns an error if padding has more entries than input axes, any width is negative, a padded extent overflows `i64`, or constructing the output shape fails.
pub fn pad_problem(
    input: &Shape,
    padding: &[(i64, i64)],
    diagnostic_prefix: &'static str,
) -> Result<PadProblem> {
    let input_dimensions = input.dimensions();
    if padding.len() > input_dimensions.len() {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} rank"),
            input_dimensions.len(),
            padding.len(),
        ));
    }

    let mut output_dimensions = Vec::with_capacity(input_dimensions.len());
    let mut normalized_padding = Vec::with_capacity(input_dimensions.len());
    let mut has_padding = false;
    for (axis, &dimension) in input_dimensions.iter().enumerate() {
        let (before, after) = padding.get(axis).copied().unwrap_or((0, 0));
        // Negative padding crops, as in PyTorch, but never below an empty axis.
        if dimension + before.min(0) + after.min(0) < 0 {
            return Err(Error::invalid_argument(format!(
                "{diagnostic_prefix} axis {axis}"
            )));
        }

        let padded = checked_add_i64(
            checked_add_i64(
                dimension,
                before,
                &format!("{diagnostic_prefix} axis {axis} size"),
            )?,
            after,
            &format!("{diagnostic_prefix} axis {axis} size"),
        )?;
        output_dimensions.push(padded);
        normalized_padding.push((before, after));
        has_padding |= before != 0 || after != 0;
    }

    Ok(PadProblem {
        output_shape: Shape::create(output_dimensions)?,
        padding: normalized_padding,
        has_padding,
    })
}

/// Validated axis sizes and contiguous copy dimensions for tensor concatenation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcatProblem {
    /// Shape formed by summing input extents along the concatenation axis.
    pub output_shape: Shape,
    /// Normalized zero-based concatenation axis.
    pub axis: usize,
    /// Extent of each input along the concatenation axis, in input order.
    pub axis_sizes: Vec<usize>,
    /// Sum of all input extents along the concatenation axis.
    pub output_axis_size: usize,
    /// Product of extents before the concatenation axis.
    pub outer_size: usize,
    /// Product of extents after the concatenation axis.
    pub inner_size: usize,
}

/// Validates concatenation shapes and computes the output shape and copy dimensions.
///
/// Negative `dim` values count from the end.
/// All input ranks and non-concatenated extents must agree.
/// `diagnostic_prefix` identifies the operation in errors.
///
/// # Errors
///
/// Returns an error if inputs are empty, the dimension is out of range, ranks or non-concatenated extents differ, a sum or product overflows, a dimension cannot fit `usize`, or constructing the output shape fails.
pub fn concat_problem(
    inputs: &[&Shape],
    dim: i64,
    diagnostic_prefix: &'static str,
) -> Result<ConcatProblem> {
    let first = inputs
        .first()
        .ok_or_else(|| Error::invalid_argument(format!("{diagnostic_prefix} inputs")))?;
    let axis = normalize_dim_usize(dim, first.rank())?;
    let first_dimensions = first.dimensions();
    let mut output_dimensions = first_dimensions.to_vec();
    let mut output_axis_size = 0_i64;
    let mut axis_sizes = Vec::with_capacity(inputs.len());

    for (input_index, input) in inputs.iter().enumerate() {
        let dimensions = input.dimensions();
        if dimensions.len() != first_dimensions.len() {
            return Err(Error::length_mismatch(
                format!("{diagnostic_prefix} input {input_index} rank"),
                first_dimensions.len(),
                dimensions.len(),
            ));
        }

        for (candidate_axis, (&expected, &actual)) in
            first_dimensions.iter().zip(dimensions).enumerate()
        {
            if candidate_axis == axis {
                output_axis_size = checked_add_i64(
                    output_axis_size,
                    actual,
                    &format!("{diagnostic_prefix} axis {axis} size"),
                )?;
            } else if expected != actual {
                return Err(Error::length_mismatch(
                    format!("{diagnostic_prefix} dimension {candidate_axis}"),
                    to_usize(expected, "backend spec dimension")?,
                    to_usize(actual, "backend spec dimension")?,
                ));
            }
        }

        axis_sizes.push(to_usize(dimensions[axis], "backend spec dimension")?);
    }

    output_dimensions[axis] = output_axis_size;
    let outer_size = dimension_product_usize(
        format!("{diagnostic_prefix} outer size"),
        &first_dimensions[..axis],
        true,
    )?;
    let inner_size = dimension_product_usize(
        format!("{diagnostic_prefix} inner size"),
        &first_dimensions[axis + 1..],
        true,
    )?;

    Ok(ConcatProblem {
        output_shape: Shape::create(output_dimensions)?,
        axis,
        axis_sizes,
        output_axis_size: to_usize(output_axis_size, "backend spec dimension")?,
        outer_size,
        inner_size,
    })
}

/// Shape metadata for selecting rows from a two-dimensional embedding table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingProblem {
    /// Index dimensions followed by the embedding width.
    pub output_shape: Shape,
    /// Number of selectable rows in the weight table.
    pub embedding_count: usize,
    /// Number of elements in each embedding row.
    pub embedding_dimension: usize,
    /// Total number of index elements.
    pub index_count: usize,
}

/// Infers an embedding output from a rank-two weight shape and an arbitrary index shape.
///
/// `diagnostic_prefix` identifies the operation in errors.
/// This checks shapes without reading index values.
///
/// # Errors
///
/// Returns an error if the weight is not rank two, a dimension or product cannot fit the host size, or constructing the output shape fails.
pub fn embedding_problem(
    weight: &Shape,
    indices: &Shape,
    diagnostic_prefix: &'static str,
) -> Result<EmbeddingProblem> {
    validate_rank(diagnostic_prefix, "weight", weight.dimensions(), 2, "NC")?;
    let index_select = gather_problem(weight, indices, 0, diagnostic_prefix)?;
    Ok(EmbeddingProblem {
        output_shape: index_select.output_shape,
        embedding_count: index_select.selected_dimension,
        embedding_dimension: index_select.inner_size,
        index_count: index_select.index_count,
    })
}

/// Shape and contiguous copy dimensions for replacing an input axis with index dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexingProblem {
    /// Input dimensions before the selected axis, then index dimensions, then remaining input dimensions.
    pub output_shape: Shape,
    /// Normalized zero-based input axis selected by indices.
    pub axis: usize,
    /// Input extent that bounds index values.
    pub selected_dimension: usize,
    /// Total number of index elements.
    pub index_count: usize,
    /// Product of input extents before the selected axis.
    pub outer_size: usize,
    /// Product of input extents after the selected axis.
    pub inner_size: usize,
}

/// Infers selection by a one-dimensional index tensor along `dim`.
///
/// Negative dimensions count from the end.
/// `diagnostic_prefix` identifies the operation in errors.
/// Index values are checked separately.
///
/// # Errors
///
/// Returns an error if indices are not rank one, `dim` is out of range, a dimension or product cannot fit the host size, or constructing the output shape fails.
pub fn index_select_problem(
    input: &Shape,
    indices: &Shape,
    dim: i64,
    diagnostic_prefix: &'static str,
) -> Result<IndexingProblem> {
    validate_rank(diagnostic_prefix, "indices", indices.dimensions(), 1, "I")?;
    gather_problem(input, indices, dim, diagnostic_prefix)
}

/// Replaces one input axis with all index dimensions.
/// Scalar indices remove it.
///
/// Negative `dim` values count from the end.
/// `diagnostic_prefix` identifies the operation in errors.
/// This checks shape metadata without validating index values.
///
/// # Errors
///
/// Returns an error if `dim` is out of range, a selected extent or dimension product cannot fit `usize`, or constructing the output shape fails.
pub fn gather_problem(
    input: &Shape,
    indices: &Shape,
    dim: i64,
    diagnostic_prefix: &'static str,
) -> Result<IndexingProblem> {
    let axis = normalize_dim_usize(dim, input.rank())?;
    let input_dimensions = input.dimensions();
    let index_count = indices.element_count();
    let selected_dimension = to_usize(input_dimensions[axis], "backend spec dimension")?;
    let outer_size = dimension_product_usize(
        format!("{diagnostic_prefix} outer size"),
        &input_dimensions[..axis],
        true,
    )?;
    let inner_size = dimension_product_usize(
        format!("{diagnostic_prefix} inner size"),
        &input_dimensions[axis + 1..],
        true,
    )?;

    let output_dimensions = input_dimensions[..axis]
        .iter()
        .chain(indices.dimensions())
        .chain(&input_dimensions[axis + 1..])
        .copied()
        .collect::<Vec<_>>();
    Ok(IndexingProblem {
        output_shape: Shape::create(output_dimensions)?,
        axis,
        selected_dimension,
        index_count,
        outer_size,
        inner_size,
    })
}

/// Shape arithmetic for indexing consecutive dimensions with several broadcast index tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiIndexProblem {
    /// Input dimensions before `axis`, then the broadcast index shape, then the input dimensions after the indexed ones.
    pub output_shape: Shape,
    /// The shape the index tensors broadcast to.
    pub index_shape: Shape,
    /// Normalized first indexed axis.
    pub axis: usize,
    /// Extents of the indexed axes, which bound each index tensor's values in order.
    pub indexed_dimensions: Vec<i64>,
    /// Product of input extents before `axis`.
    pub outer_size: usize,
    /// Product of input extents after the indexed axes.
    pub inner_size: usize,
}

/// Infers indexing of the consecutive axes starting at `dim` with one index tensor per axis.
///
/// The index tensors broadcast together, and their broadcast shape replaces the indexed axes.
/// Negative `dim` values count from the end.
/// This checks shape metadata without validating index values.
///
/// # Errors
///
/// Returns an error if there are no index tensors, the indexed axes do not fit in the input, the index shapes do not broadcast, or a dimension product cannot fit `usize`.
pub fn multi_index_problem(
    input: &Shape,
    indices: &[&Shape],
    dim: i64,
) -> Result<MultiIndexProblem> {
    if indices.is_empty() {
        return Err(Error::missing_input("multi index problem"));
    }

    let axis = normalize_dim_usize(dim, input.rank())?;
    let end = axis + indices.len();
    let input_dimensions = input.dimensions();
    if end > input_dimensions.len() {
        return Err(Error::DimensionOutOfBounds);
    }

    let mut index_shape = Shape::scalar();
    for shape in indices {
        index_shape = index_shape.broadcast_with(shape)?;
    }

    let outer_size = dimension_product_usize("index outer size", &input_dimensions[..axis], true)?;
    let inner_size = dimension_product_usize("index inner size", &input_dimensions[end..], true)?;
    let output_dimensions = input_dimensions[..axis]
        .iter()
        .chain(index_shape.dimensions())
        .chain(&input_dimensions[end..])
        .copied()
        .collect::<Vec<_>>();
    Ok(MultiIndexProblem {
        output_shape: Shape::create(output_dimensions)?,
        index_shape,
        axis,
        indexed_dimensions: input_dimensions[axis..end].to_vec(),
        outer_size,
        inner_size,
    })
}

/// Normalized axis interval and inferred output shape for flattening tensor dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlattenProblem {
    /// Shape with the selected consecutive axes replaced by their product.
    pub output_shape: Shape,
    /// First normalized zero-based axis included in the flattened interval.
    pub start_axis: usize,
    /// Last normalized zero-based axis included in the flattened interval.
    pub end_axis: usize,
    /// Product of extents in the selected interval, or one for a scalar input.
    pub flattened_dimension: i64,
}

/// Output shape and left-padded input dimensions for tensor broadcasting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandProblem {
    /// Requested broadcast output shape.
    pub output_shape: Shape,
    /// Input dimensions padded on the left with ones to match the output rank.
    pub input_padded_dimensions: Vec<i64>,
}

/// Checks that `input` can broadcast to `output` by adding leading axes and expanding singleton axes.
///
/// `diagnostic_prefix` identifies the operation in errors.
/// This infers dimensions without constructing strides or allocating tensor values.
///
/// # Errors
///
/// Returns an error if input rank exceeds output rank or an aligned input extent is neither one nor equal to the output extent.
pub fn expand_problem(
    input: &Shape,
    output: &Shape,
    diagnostic_prefix: &'static str,
) -> Result<ExpandProblem> {
    let input_dimensions = input.dimensions();
    let output_dimensions = output.dimensions();
    if input_dimensions.len() > output_dimensions.len() {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} rank"),
            output_dimensions.len(),
            input_dimensions.len(),
        ));
    }

    let rank_delta = output_dimensions.len() - input_dimensions.len();
    let mut input_padded_dimensions = vec![1_i64; output_dimensions.len()];
    for (axis, &output_dimension) in output_dimensions.iter().enumerate() {
        if axis < rank_delta {
            continue;
        }

        let input_axis = axis - rank_delta;
        let input_dimension = input_dimensions[input_axis];
        input_padded_dimensions[axis] = input_dimension;
        if input_dimension != output_dimension && input_dimension != 1 {
            return Err(Error::broadcast_mismatch(
                diagnostic_prefix,
                input_padded_dimensions.clone(),
                output_dimensions.to_vec(),
            ));
        }
    }

    Ok(ExpandProblem {
        output_shape: output.clone(),
        input_padded_dimensions,
    })
}

/// Combines the inclusive axis interval `start_dim..=end_dim` into one dimension.
///
/// Negative axes count from the end.
/// A scalar accepts axis zero or minus one and produces shape `[1]`.
/// `diagnostic_prefix` identifies the operation in errors.
///
/// # Errors
///
/// Returns an error if either axis is out of range, normalized start exceeds end, multiplying selected extents overflows `i64`, or constructing the output shape fails.
pub fn flatten_problem(
    input: &Shape,
    start_dim: i64,
    end_dim: i64,
    diagnostic_prefix: &'static str,
) -> Result<FlattenProblem> {
    let start_axis = normalize_dim_usize(start_dim, input.rank().max(1))?;
    let end_axis = normalize_dim_usize(end_dim, input.rank().max(1))?;
    if input.rank() == 0 {
        return Ok(FlattenProblem {
            output_shape: Shape::create([1])?,
            start_axis,
            end_axis,
            flattened_dimension: 1,
        });
    }

    if start_axis > end_axis {
        return Err(Error::invalid_argument(format!(
            "{diagnostic_prefix} dimension range"
        )));
    }

    let dimensions = input.dimensions();
    let flattened_dimension =
        dimensions[start_axis..=end_axis]
            .iter()
            .try_fold(1_i64, |acc, &dimension| {
                checked_mul_i64(
                    acc,
                    dimension,
                    &format!("{diagnostic_prefix} dimension product"),
                )
            })?;
    let mut output_dimensions = Vec::with_capacity(dimensions.len() - (end_axis - start_axis));
    output_dimensions.extend_from_slice(&dimensions[..start_axis]);
    output_dimensions.push(flattened_dimension);
    output_dimensions.extend_from_slice(&dimensions[end_axis + 1..]);

    Ok(FlattenProblem {
        output_shape: Shape::create(output_dimensions)?,
        start_axis,
        end_axis,
        flattened_dimension,
    })
}

#[cfg(any(test, feature = "onnx"))]
/// Computes a checked concatenation output shape.
///
/// # Errors
///
/// Returns an error if no inputs are supplied, the axis is invalid, ranks or non-concatenated dimensions differ, or the output extent or shape overflows.
pub fn concat_output_shape(inputs: &[Shape], axis: i64) -> Result<Shape> {
    let first = inputs.first().ok_or(Error::MissingConcatInput)?;
    let axis = normalize_dim_usize(axis, first.rank())?;
    let mut dimensions = first.dimensions().to_vec();
    for (input_index, input) in inputs[1..].iter().enumerate() {
        if input.rank() != first.rank() {
            return Err(Error::length_mismatch(
                format!("concat input {} rank", input_index + 1),
                first.dimensions().len(),
                input.dimensions().len(),
            ));
        }
        for (candidate_axis, (&actual, &expected)) in input
            .dimensions()
            .iter()
            .zip(first.dimensions())
            .enumerate()
        {
            if candidate_axis == axis {
                dimensions[axis] = dimensions[axis]
                    .checked_add(actual)
                    .ok_or_else(|| Error::out_of_range("concat axis size"))?;
            } else if actual != expected {
                return Err(Error::length_mismatch(
                    format!("concat axis {candidate_axis}"),
                    to_usize(expected, "backend spec dimension")?,
                    to_usize(actual, "backend spec dimension")?,
                ));
            }
        }
    }
    Shape::create(dimensions)
}
