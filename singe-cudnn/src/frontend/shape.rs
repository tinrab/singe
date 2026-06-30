use crate::{
    error::{Error, Result},
    tensor::Shape,
    utility::{check_range, to_i64, to_usize},
};

pub(crate) fn contiguous_strides(dimensions: &[i64], error_name: &str) -> Result<Vec<i64>> {
    let mut strides = vec![0_i64; dimensions.len()];
    let mut stride = 1_i64;
    for (index, dimension) in dimensions.iter().enumerate().rev() {
        strides[index] = stride;
        stride = stride
            .checked_mul(*dimension)
            .ok_or_else(|| Error::OutOfRange {
                name: error_name.into(),
            })?;
    }
    Ok(strides)
}

pub(crate) fn shape_with_nhwc_strides(dimensions: impl Into<Vec<i64>>) -> Result<Shape> {
    let dimensions = dimensions.into();
    if dimensions.len() < 2 {
        return Shape::contiguous(dimensions);
    }

    let mut ordered_indices = Vec::with_capacity(dimensions.len());
    ordered_indices.push(1);
    ordered_indices.extend((2..dimensions.len()).rev());
    ordered_indices.push(0);

    let strides = strides_from_ordered_indices(&dimensions, ordered_indices, "tensor strides")?;
    shape_with_strides(dimensions, strides)
}

pub(crate) fn shape_with_strides(
    dimensions: impl Into<Vec<i64>>,
    strides: impl Into<Vec<i64>>,
) -> Result<Shape> {
    Shape::contiguous(dimensions.into())?.with_strides(strides.into())
}

pub(crate) fn shape_preserving_stride_order(
    input: &Shape,
    dimensions: impl Into<Vec<i64>>,
) -> Result<Shape> {
    let dimensions = dimensions.into();
    let strides = strides_preserving_stride_order(input, &dimensions, "tensor strides")?;
    shape_with_strides(dimensions, strides)
}

pub(crate) fn strides_preserving_stride_order(
    input: &Shape,
    dimensions: &[i64],
    error_name: &str,
) -> Result<Vec<i64>> {
    if input.strides().len() != dimensions.len() {
        return Err(Error::LengthMismatch {
            name: error_name.into(),
            expected: dimensions.len(),
            actual: input.strides().len(),
        });
    }
    if dimensions.len() < 2 {
        return Ok(vec![1_i64; dimensions.len()]);
    }

    strides_from_ordered_indices(dimensions, stride_order_indices(input), error_name)
}

pub(crate) fn strides_preserving_packed_axis(
    input: &Shape,
    dimensions: &[i64],
    axis: i64,
    error_name: &str,
) -> Result<Vec<i64>> {
    let rank = to_i64(input.dimensions().len(), "rank")?;
    let axis = if axis < 0 { rank + axis } else { axis };
    check_range!("axis", axis >= 0 && axis < rank)?;
    let axis = to_usize(axis, "axis")?;

    let mut indices = stride_order_indices(input);
    let rotation = indices
        .iter()
        .position(|&index| index == axis)
        .ok_or(Error::InvalidDataShape)?;
    indices.rotate_left(rotation);

    strides_from_ordered_indices(dimensions, indices, error_name)
}

fn strides_from_ordered_indices(
    dimensions: &[i64],
    ordered_indices: impl IntoIterator<Item = usize>,
    error_name: &str,
) -> Result<Vec<i64>> {
    let mut strides = vec![0_i64; dimensions.len()];
    let mut stride = 1_i64;
    for index in ordered_indices {
        strides[index] = stride;
        stride = stride
            .checked_mul(dimensions[index])
            .ok_or_else(|| Error::OutOfRange {
                name: error_name.into(),
            })?;
    }
    Ok(strides)
}

pub(crate) fn stride_order_indices(input: &Shape) -> Vec<usize> {
    let mut indices = (0..input.strides().len()).collect::<Vec<_>>();
    indices.sort_by(|&left, &right| {
        let left_stride = input.strides()[left];
        let right_stride = input.strides()[right];
        if left_stride == right_stride {
            let left_dim = input.dimensions()[left];
            let right_dim = input.dimensions()[right];
            return if left_dim == 1 || right_dim != 1 {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            };
        }
        left_stride.cmp(&right_stride)
    });
    indices
}

pub(crate) fn transpose_last_two_shape(input: &Shape) -> Result<Shape> {
    let mut dimensions = input.dimensions().to_vec();
    let mut strides = input.strides().to_vec();
    let rank = dimensions.len();
    if rank < 2 {
        return Err(Error::InvalidDataShape);
    }
    dimensions.swap(rank - 2, rank - 1);
    strides.swap(rank - 2, rank - 1);
    shape_with_strides(dimensions, strides)
}

pub(crate) fn unit_shape_like(input: &Shape) -> Result<Shape> {
    shape_with_strides(
        vec![1_i64; input.dimensions().len()],
        vec![1_i64; input.strides().len()],
    )
}
