use std::mem;

use crate::{
    convolution::ConvolutionMode,
    error::{Error, Result},
    execution::resample::Fraction,
    frontend::{
        operation::{ConvolutionConfig, ReductionAxes, ResampleConfig},
        shape::{
            contiguous_strides, shape_with_nhwc_strides, shape_with_strides,
            strides_preserving_packed_axis, strides_preserving_stride_order,
        },
        support,
    },
    tensor::Shape,
    utility::{check_range, to_i64, to_usize},
};

fn checked_value<T>(value: Option<T>, name: &str) -> Result<T> {
    value.ok_or_else(|| Error::OutOfRange { name: name.into() })
}

fn shape_rank_mismatch(name: &str, expected: usize, actual: usize) -> Error {
    Error::LengthMismatch {
        name: name.into(),
        expected,
        actual,
    }
}

pub(crate) fn infer_binary_pointwise_output(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    let rank = lhs.dimensions().len().max(rhs.dimensions().len());
    let lhs_offset = rank - lhs.dimensions().len();
    let rhs_offset = rank - rhs.dimensions().len();

    let mut dimensions = Vec::with_capacity(rank);
    for index in 0..rank {
        let lhs_dim = if index < lhs_offset {
            1
        } else {
            lhs.dimensions()[index - lhs_offset]
        };
        let rhs_dim = if index < rhs_offset {
            1
        } else {
            rhs.dimensions()[index - rhs_offset]
        };

        let dimension = match (lhs_dim, rhs_dim) {
            (left, right) if left == right => left,
            (1, right) => right,
            (left, 1) => left,
            _ => {
                return Err(Error::FrontendPointwiseBroadcastShapeMismatch {
                    lhs: lhs.dimensions().to_vec(),
                    rhs: rhs.dimensions().to_vec(),
                });
            }
        };
        dimensions.push(dimension);
    }

    let lhs_dims = lhs.dimensions();
    let rhs_dims = rhs.dimensions();
    let lhs_matches = lhs_dims.len() == rank && lhs_dims == dimensions;
    let rhs_matches = rhs_dims.len() == rank && rhs_dims == dimensions;
    let strides = if lhs_matches {
        lhs.strides().to_vec()
    } else if rhs_matches {
        rhs.strides().to_vec()
    } else {
        contiguous_strides(&dimensions, "tensor strides")?
    };

    shape_with_strides(dimensions, strides)
}

pub(crate) fn reduce_last_axis(shape: &Shape) -> Result<Shape> {
    let mut dimensions = shape.dimensions().to_vec();

    let Some(last_dimension) = dimensions.last_mut() else {
        return Err(Error::InvalidDataShape);
    };
    *last_dimension = 1;

    Shape::contiguous(dimensions)
}

pub(crate) fn infer_softmax_shapes(input: &Shape) -> Result<(Shape, Shape)> {
    let reduction = reduce_last_axis(input)?;
    Ok((reduction.clone(), reduction))
}

pub(crate) fn infer_reduction_output(input: &Shape, axes: ReductionAxes) -> Result<Shape> {
    match axes {
        ReductionAxes::Last => reduce_last_axis(input),
        ReductionAxes::All => {
            let mut dimensions = vec![1; input.dimensions().len()];
            let mut strides = vec![1; input.strides().len()];
            if dimensions.is_empty() || strides.is_empty() {
                return Err(Error::InvalidDataShape);
            }
            shape_with_strides(mem::take(&mut dimensions), mem::take(&mut strides))
        }
    }
}

pub(crate) fn infer_concat_output(inputs: &[&Shape], axis: i64) -> Result<Shape> {
    let first = inputs.first().ok_or(Error::EmptyList {
        name: "concat inputs".into(),
    })?;
    let rank = to_i64(first.dimensions().len(), "rank")?;
    check_range!("axis", axis >= 0 && axis < rank)?;
    let axis = to_usize(axis, "axis")?;
    let mut dimensions = first.dimensions().to_vec();

    let mut axis_sum = 0_i64;
    for input in inputs {
        if input.dimensions().len() != first.dimensions().len() {
            return Err(shape_rank_mismatch(
                support::CONCAT_RANKS,
                first.dimensions().len(),
                input.dimensions().len(),
            ));
        }
        for (index, (&lhs, &rhs)) in first
            .dimensions()
            .iter()
            .zip(input.dimensions())
            .enumerate()
        {
            if index != axis && lhs != rhs {
                return Err(Error::ShapeMismatch {
                    name: support::CONCAT_SHAPE.into(),
                    expected: first.dimensions().to_vec(),
                    actual: input.dimensions().to_vec(),
                });
            }
        }
        axis_sum = checked_value(
            axis_sum.checked_add(input.dimensions()[axis]),
            "concat axis dimension",
        )?;
    }
    dimensions[axis] = axis_sum;

    Shape::contiguous(dimensions)
}

pub(crate) fn infer_slice_output(input: &Shape, slices: &[(i64, i64)]) -> Result<Shape> {
    infer_strided_slice_output(input, slices, &[])
}

pub(crate) fn infer_strided_slice_output(
    input: &Shape,
    slices: &[(i64, i64)],
    slice_strides: &[i64],
) -> Result<Shape> {
    if input.dimensions().len() != slices.len() {
        return Err(Error::LengthMismatch {
            name: "slice rank".into(),
            expected: input.dimensions().len(),
            actual: slices.len(),
        });
    }

    let mut dimensions = Vec::with_capacity(slices.len());
    let mut strides = Vec::with_capacity(slices.len());
    for (index, ((start, end), input_dim)) in
        slices.iter().copied().zip(input.dimensions()).enumerate()
    {
        let slice_stride = slice_strides.get(index).copied().unwrap_or(1);
        check_range!("slice stride", slice_stride > 0)?;
        check_range!(
            "slice range",
            start >= 0 && end >= start && end <= *input_dim
        )?;
        let extent = end - start;
        let rounded_extent = checked_value(
            extent.checked_add(slice_stride - 1),
            "slice output dimension",
        )?;
        dimensions.push(rounded_extent / slice_stride);
        strides.push(checked_value(
            input.strides()[index].checked_mul(slice_stride),
            "slice output strides",
        )?);
    }

    shape_with_strides(dimensions, strides)
}

pub(crate) fn infer_transpose_output(input: &Shape, permutation: &[i64]) -> Result<Shape> {
    if input.dimensions().len() != permutation.len() {
        return Err(Error::LengthMismatch {
            name: "transpose permutation rank".into(),
            expected: input.dimensions().len(),
            actual: permutation.len(),
        });
    }

    let rank = permutation.len();
    let mut seen = vec![false; rank];
    let mut dimensions = Vec::with_capacity(rank);
    let mut strides = Vec::with_capacity(rank);
    for &dimension_index in permutation {
        check_range!(
            "transpose permutation",
            dimension_index >= 0 && (dimension_index as usize) < rank
        )?;
        let index = to_usize(dimension_index, "transpose permutation")?;
        if seen[index] {
            return Err(Error::OutOfRange {
                name: support::TRANSPOSE_PERMUTATION.into(),
            });
        }
        seen[index] = true;
        dimensions.push(input.dimensions()[index]);
        strides.push(input.strides()[index]);
    }

    shape_with_strides(dimensions, strides)
}

pub(crate) fn slice_byte_offset(
    input: &Shape,
    slices: &[(i64, i64)],
    element_size: i64,
) -> Result<i64> {
    check_range!("slice element_size", element_size > 0)?;

    let mut elements = 0_i64;
    for ((start, _), stride) in slices.iter().copied().zip(input.strides()) {
        let offset = checked_value(start.checked_mul(*stride), "slice offset")?;
        elements = checked_value(elements.checked_add(offset), "slice offset")?;
    }
    checked_value(elements.checked_mul(element_size), "slice offset")
}

pub(crate) fn infer_resample_output(input: &Shape, config: &ResampleConfig) -> Result<Shape> {
    if input.dimensions().len() < 3 {
        return Err(shape_rank_mismatch(
            support::RESAMPLE_RANK,
            3,
            input.dimensions().len(),
        ));
    }

    let dims = input.dimensions();
    let spatial_dims = dims.len() - 2;
    let pre = config.effective_pre_paddings(spatial_dims)?;
    let post = config.effective_post_paddings(spatial_dims)?;
    let stride = config.effective_strides(spatial_dims)?;
    let window = config.effective_window_dims(spatial_dims)?;

    let mut out = dims[..2].to_vec();
    for index in 0..spatial_dims {
        stride[index].validate("resample strides")?;
        window[index].validate("resample window dims")?;
        check_range!(
            "resample shape",
            stride[index].numerator() > 0 && window[index].numerator() > 0
        )?;
        out.push(infer_resample_spatial_dimension(
            dims[index + 2],
            pre[index],
            post[index],
            stride[index],
            window[index],
        )?);
    }

    Shape::contiguous(out)
}

fn gcd_i128(mut left: i128, mut right: i128) -> i128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.abs()
}

fn lcm_i128(left: i128, right: i128) -> Result<i128> {
    let gcd = gcd_i128(left, right);
    let quotient = checked_value(left.checked_div(gcd), "resample shape")?;
    checked_value(quotient.checked_mul(right), "resample shape")
}

fn infer_resample_spatial_dimension(
    input: i64,
    pre_padding: Fraction,
    post_padding: Fraction,
    stride: Fraction,
    window: Fraction,
) -> Result<i64> {
    let denominators = [
        i128::from(pre_padding.denominator()),
        i128::from(post_padding.denominator()),
        i128::from(stride.denominator()),
        i128::from(window.denominator()),
    ];
    let common_denominator = denominators.into_iter().try_fold(1_i128, lcm_i128)?;

    let scaled = |value: Fraction| -> Result<i128> {
        let factor = checked_value(
            common_denominator.checked_div(i128::from(value.denominator())),
            "resample shape",
        )?;
        checked_value(
            i128::from(value.numerator()).checked_mul(factor),
            "resample shape",
        )
    };

    let input_scaled = checked_value(
        i128::from(input).checked_mul(common_denominator),
        "resample shape",
    )?;
    let pre_padding_scaled = scaled(pre_padding)?;
    let post_padding_scaled = scaled(post_padding)?;
    let window_scaled = scaled(window)?;
    let numerator = input_scaled
        .checked_add(pre_padding_scaled)
        .and_then(|value| value.checked_add(post_padding_scaled))
        .and_then(|value| value.checked_sub(window_scaled));
    let numerator = checked_value(numerator, "resample shape")?;
    let stride_scaled = scaled(stride)?;
    check_range!("resample shape", numerator >= 0 && stride_scaled > 0)?;
    let output = checked_value(numerator.checked_div(stride_scaled), "resample shape")?;
    let output = checked_value(output.checked_add(1), "resample shape")?;
    check_range!("resample shape", output <= i128::from(i64::MAX))?;
    Ok(output as i64)
}

fn validate_convolution_spatial_dims(
    input: &Shape,
    filter: &Shape,
    config: &ConvolutionConfig,
) -> Result<usize> {
    if input.dimensions().len() < 3 {
        return Err(shape_rank_mismatch(
            support::CONVOLUTION_RANK,
            3,
            input.dimensions().len(),
        ));
    }
    if input.dimensions().len() != filter.dimensions().len() {
        return Err(shape_rank_mismatch(
            support::CONVOLUTION_RANK,
            input.dimensions().len(),
            filter.dimensions().len(),
        ));
    }

    let spatial_dims = input.dimensions().len() - 2;
    for values in [
        config.pre_paddings(),
        config.post_paddings(),
        config.strides(),
        config.dilations(),
    ] {
        if values.len() != spatial_dims {
            return Err(Error::LengthMismatch {
                name: "convolution spatial dims".into(),
                expected: spatial_dims,
                actual: values.len(),
            });
        }
    }

    check_range!(
        "convolution mode",
        config.mode() == ConvolutionMode::CrossCorrelation
            || config.mode() == ConvolutionMode::Convolution
    )?;
    check_range!("convolution group_count", config.group_count() != 0)?;

    Ok(spatial_dims)
}

pub(crate) fn infer_convolution_forward_output(
    input: &Shape,
    filter: &Shape,
    config: &ConvolutionConfig,
) -> Result<Shape> {
    let spatial_dims = validate_convolution_spatial_dims(input, filter, config)?;
    let input_dims = input.dimensions();
    let filter_dims = filter.dimensions();

    let grouped_input_channels = checked_value(
        filter_dims[1].checked_mul(to_i64(config.group_count(), "convolution group_count")?),
        "convolution channels",
    )?;
    if input_dims[1] != grouped_input_channels {
        return Err(Error::ShapeMismatch {
            name: support::CONVOLUTION_CHANNELS.into(),
            expected: vec![grouped_input_channels],
            actual: vec![input_dims[1]],
        });
    }

    let mut output = Vec::with_capacity(input_dims.len());
    output.push(input_dims[0]);
    output.push(filter_dims[0]);
    for index in 0..spatial_dims {
        let stride = config.strides()[index];
        let dilation = config.dilations()[index];
        let kernel = filter_dims[index + 2];
        check_range!(
            "convolution shape",
            stride > 0 && dilation > 0 && kernel > 0
        )?;
        let numerator =
            input_dims[index + 2] + config.pre_paddings()[index] + config.post_paddings()[index]
                - ((kernel - 1) * dilation + 1);
        output.push(numerator / stride + 1);
    }

    Shape::contiguous(output)
}

pub(crate) fn infer_convolution_backward_data_output(
    filter: &Shape,
    output_gradient: &Shape,
    config: &ConvolutionConfig,
) -> Result<Shape> {
    let spatial_dims = validate_convolution_spatial_dims(output_gradient, filter, config)?;
    let filter_dims = filter.dimensions();
    let dy_dims = output_gradient.dimensions();

    if dy_dims[1] != filter_dims[0] {
        return Err(Error::ShapeMismatch {
            name: "convolution backward data channels".into(),
            expected: vec![filter_dims[0]],
            actual: vec![dy_dims[1]],
        });
    }

    let group_count = to_i64(config.group_count(), "convolution group_count")?;
    let channels = checked_value(
        filter_dims[1].checked_mul(group_count),
        "convolution backward data channels",
    )?;

    let mut output = Vec::with_capacity(dy_dims.len());
    output.push(dy_dims[0]);
    output.push(channels);
    for index in 0..spatial_dims {
        let stride = config.strides()[index];
        let dilation = config.dilations()[index];
        let kernel = filter_dims[index + 2];
        check_range!(
            "convolution backward data shape",
            stride > 0 && dilation > 0 && kernel > 0
        )?;
        let size = (dy_dims[index + 2] - 1) * stride
            - config.pre_paddings()[index]
            - config.post_paddings()[index]
            + dilation * (kernel - 1)
            + 1;
        output.push(size);
    }

    Shape::contiguous(output)
}

pub(crate) fn infer_convolution_backward_filter_output(
    input: &Shape,
    output_gradient: &Shape,
    config: &ConvolutionConfig,
) -> Result<Shape> {
    if input.dimensions().len() != output_gradient.dimensions().len()
        || input.dimensions().len() < 3
    {
        return Err(shape_rank_mismatch(
            "convolution backward filter rank",
            input.dimensions().len().max(3),
            output_gradient.dimensions().len(),
        ));
    }

    let spatial_dims = input.dimensions().len() - 2;
    for values in [
        config.pre_paddings(),
        config.post_paddings(),
        config.strides(),
        config.dilations(),
    ] {
        if values.len() != spatial_dims {
            return Err(Error::LengthMismatch {
                name: "convolution spatial dims".into(),
                expected: spatial_dims,
                actual: values.len(),
            });
        }
    }

    let input_dims = input.dimensions();
    let dy_dims = output_gradient.dimensions();
    if input_dims[0] != dy_dims[0] {
        return Err(Error::ShapeMismatch {
            name: "convolution backward filter batch".into(),
            expected: vec![input_dims[0]],
            actual: vec![dy_dims[0]],
        });
    }
    if input_dims[1] % to_i64(config.group_count(), "convolution group_count")? != 0 {
        return Err(Error::ShapeMismatch {
            name: "convolution backward filter channels".into(),
            expected: vec![to_i64(config.group_count(), "convolution group_count")?],
            actual: vec![input_dims[1]],
        });
    }

    let mut output = Vec::with_capacity(input_dims.len());
    output.push(dy_dims[1]);
    output.push(input_dims[1] / to_i64(config.group_count(), "convolution group_count")?);
    for index in 0..spatial_dims {
        let stride = config.strides()[index];
        let dilation = config.dilations()[index];
        let numerator =
            input_dims[index + 2] + config.pre_paddings()[index] + config.post_paddings()[index]
                - ((dy_dims[index + 2] - 1) * stride + 1);
        if dilation <= 0 || numerator < 0 || numerator % dilation != 0 {
            return Err(Error::ShapeMismatch {
                name: "convolution backward filter shape".into(),
                expected: input_dims.to_vec(),
                actual: dy_dims.to_vec(),
            });
        }
        output.push(numerator / dilation + 1);
    }

    Shape::contiguous(output)
}

pub(crate) fn infer_paged_cache_output(
    container: &Shape,
    sequence: &Shape,
    page_table: &Shape,
    transposed: bool,
) -> Result<Shape> {
    if container.dimensions().len() != 4
        || sequence.dimensions().len() != 4
        || page_table.dimensions().len() != 4
    {
        return Err(shape_rank_mismatch(
            support::PAGED_CACHE_LOAD_RANK,
            4,
            container
                .dimensions()
                .len()
                .min(sequence.dimensions().len())
                .min(page_table.dimensions().len()),
        ));
    }

    let container_dims = container.dimensions();
    let sequence_dims = sequence.dimensions();
    let page_table_dims = page_table.dimensions();

    if sequence_dims[0] != page_table_dims[0] || page_table_dims[1] != 1 || page_table_dims[3] != 1
    {
        return Err(Error::ShapeMismatch {
            name: support::PAGED_CACHE_LOAD_SHAPE.into(),
            expected: sequence_dims.to_vec(),
            actual: page_table_dims.to_vec(),
        });
    }
    if sequence_dims[1] != 1 || sequence_dims[2] != 1 || sequence_dims[3] != 1 {
        return Err(Error::ShapeMismatch {
            name: "paged cache load sequence shape".into(),
            expected: vec![sequence_dims[0], 1, 1, 1],
            actual: sequence_dims.to_vec(),
        });
    }

    let batch = sequence_dims[0];
    let heads = container_dims[1];
    let block_size = container_dims[2];
    let hidden = container_dims[3];
    let blocks = page_table_dims[2];
    let sequence_length = blocks * block_size;

    if transposed {
        shape_with_strides(
            [batch, heads, hidden, sequence_length],
            vec![
                heads * hidden * sequence_length,
                hidden * sequence_length,
                1,
                hidden,
            ],
        )
    } else {
        Shape::contiguous([batch, heads, sequence_length, hidden])
    }
}

pub(crate) fn infer_block_scale_quantize_output_shape(
    input: &Shape,
    axis: i64,
    transposed: bool,
) -> Result<Shape> {
    let dimensions = input.dimensions().to_vec();
    let strides = if transposed {
        strides_preserving_packed_axis(input, &dimensions, axis, "block scale shape")?
    } else {
        input.strides().to_vec()
    };
    shape_with_strides(dimensions, strides)
}

pub(crate) fn infer_block_scale_shape_with_layout(
    input: &Shape,
    block_size: i64,
    axis: i64,
    transposed: bool,
) -> Result<Shape> {
    check_range!("block_size", block_size > 0)?;

    let mut dimensions = input.dimensions().to_vec();
    let rank = to_i64(dimensions.len(), "rank")?;
    let axis = if axis < 0 { rank + axis } else { axis };
    check_range!("axis", axis >= 0 && axis < rank)?;
    let dimension = dimensions
        .get_mut(to_usize(axis, "axis")?)
        .ok_or(Error::InvalidDataShape)?;
    *dimension = (*dimension + block_size - 1) / block_size;

    let strides = if transposed {
        strides_preserving_packed_axis(input, &dimensions, axis, "block scale shape")?
    } else {
        strides_preserving_stride_order(input, &dimensions, "block scale shape")?
    };

    shape_with_strides(dimensions, strides)
}

pub(crate) fn infer_block_scale_dequantize_shape(
    input: &Shape,
    block_sizes: &[i32],
) -> Result<Shape> {
    if block_sizes.is_empty() {
        return Err(Error::EmptyList {
            name: "block_sizes".into(),
        });
    }

    let mut dimensions = input.dimensions().to_vec();
    let rank = dimensions.len();
    let block_rank = block_sizes.len();
    if block_rank > rank {
        return Err(shape_rank_mismatch(
            support::BLOCK_SCALE_RANK,
            rank,
            block_rank,
        ));
    }
    let start = rank - block_rank;
    for (offset, block_size) in block_sizes.iter().copied().enumerate() {
        let block = i64::from(block_size);
        check_range!("block_size", block > 0)?;
        let index = start + offset;
        dimensions[index] = (dimensions[index] + block - 1) / block;
    }

    let strides = strides_preserving_stride_order(input, &dimensions, "block scale shape")?;

    shape_with_strides(dimensions, strides)
}

pub(crate) fn infer_layer_norm_scale_bias_shape(input: &Shape) -> Result<Shape> {
    let mut dimensions = input.dimensions().to_vec();
    if dimensions.is_empty() {
        return Err(Error::InvalidDataShape);
    }

    let normalized_axis = input
        .dimensions()
        .iter()
        .enumerate()
        .filter(|(_, dimension)| **dimension > 1)
        .min_by_key(|(index, _)| input.strides()[*index].abs())
        .map(|(index, _)| index)
        .ok_or(Error::InvalidDataShape)?;

    for (index, dimension) in dimensions.iter_mut().enumerate() {
        if index != normalized_axis {
            *dimension = 1;
        }
    }

    Shape::contiguous(dimensions)
}

pub(crate) fn infer_adaptive_layer_norm_scale_bias_shape(input: &Shape) -> Result<Shape> {
    let mut dimensions = input.dimensions().to_vec();
    if dimensions.len() < 2 {
        return Err(Error::InvalidDataShape);
    }

    let reduced_rank = dimensions.len() - 2;
    for dimension in dimensions.iter_mut().skip(1).take(reduced_rank) {
        *dimension = 1;
    }

    Shape::contiguous(dimensions)
}

pub(crate) fn infer_layer_norm_stats_shape(input: &Shape, scale: &Shape) -> Result<Shape> {
    if input.dimensions().len() != scale.dimensions().len() {
        return Err(shape_rank_mismatch(
            support::LAYER_NORM_RANK,
            input.dimensions().len(),
            scale.dimensions().len(),
        ));
    }

    let mut dimensions = input.dimensions().to_vec();
    for (dimension, scale_dimension) in dimensions.iter_mut().zip(scale.dimensions()) {
        if *scale_dimension != 1 {
            *dimension = 1;
        }
    }

    Shape::contiguous(dimensions)
}

pub(crate) fn infer_adaptive_layer_norm_stats_shape(input: &Shape) -> Result<Shape> {
    let mut dimensions = input.dimensions().to_vec();
    if dimensions.len() < 2 {
        return Err(Error::InvalidDataShape);
    }

    let last = dimensions.last_mut().ok_or(Error::InvalidDataShape)?;
    *last = 1;

    Shape::contiguous(dimensions)
}

pub(crate) fn infer_instance_norm_scale_bias_shape(input: &Shape) -> Result<Shape> {
    let mut dimensions = input.dimensions().to_vec();
    if dimensions.len() < 2 {
        return Err(Error::InvalidDataShape);
    }

    for (index, dimension) in dimensions.iter_mut().enumerate() {
        if index != 1 {
            *dimension = 1;
        }
    }

    shape_with_nhwc_strides(dimensions)
}

pub(crate) fn infer_instance_norm_stats_shape(input: &Shape) -> Result<Shape> {
    let mut dimensions = input.dimensions().to_vec();
    if dimensions.len() < 2 {
        return Err(Error::InvalidDataShape);
    }

    for dimension in dimensions.iter_mut().skip(2) {
        *dimension = 1;
    }

    shape_with_nhwc_strides(dimensions)
}

pub(crate) fn infer_batch_norm_channel_shape(input: &Shape) -> Result<Shape> {
    infer_instance_norm_scale_bias_shape(input)
}

pub(crate) fn infer_sdpa_scores_shape(q: &Shape, k: &Shape) -> Result<Shape> {
    if q.dimensions().len() != 4 || k.dimensions().len() != 4 {
        return Err(shape_rank_mismatch(
            support::SDPA_RANK,
            4,
            q.dimensions().len().min(k.dimensions().len()),
        ));
    }

    let q_dims = q.dimensions();
    let k_dims = k.dimensions();
    if q_dims[0] != k_dims[0] || q_dims[1] % k_dims[1] != 0 {
        return Err(Error::FrontendSdpaQKShapeMismatch {
            q: q_dims.to_vec(),
            k: k_dims.to_vec(),
        });
    }

    let k_sequence = if q_dims[3] == k_dims[3] {
        k_dims[2]
    } else if q_dims[3] == k_dims[2] {
        k_dims[3]
    } else {
        return Err(Error::FrontendSdpaQKShapeMismatch {
            q: q_dims.to_vec(),
            k: k_dims.to_vec(),
        });
    };

    Shape::contiguous([q_dims[0], q_dims[1], q_dims[2], k_sequence])
}

pub(crate) fn infer_sdpa_output_shape(scores: &Shape, v: &Shape) -> Result<Shape> {
    if scores.dimensions().len() != 4 || v.dimensions().len() != 4 {
        return Err(shape_rank_mismatch(
            support::SDPA_RANK,
            4,
            scores.dimensions().len().min(v.dimensions().len()),
        ));
    }

    let scores_dims = scores.dimensions();
    let v_dims = v.dimensions();
    if scores_dims[0] != v_dims[0] || scores_dims[1] % v_dims[1] != 0 || scores_dims[3] != v_dims[2]
    {
        return Err(Error::FrontendSdpaScoresVShapeMismatch {
            scores: scores_dims.to_vec(),
            v: v_dims.to_vec(),
        });
    }

    Shape::contiguous(vec![
        scores_dims[0],
        scores_dims[1],
        scores_dims[2],
        v_dims[3],
    ])
}
