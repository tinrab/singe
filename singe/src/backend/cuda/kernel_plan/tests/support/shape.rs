pub fn expected_pad_f32(
    values: &[f32],
    input_dimensions: &[usize],
    padding: &[(i64, i64)],
    pad_value: f32,
) -> Vec<f32> {
    let output_dimensions = input_dimensions
        .iter()
        .enumerate()
        .map(|(axis, &dimension)| {
            let (before, after) = padding.get(axis).copied().unwrap_or((0, 0));
            dimension + usize::try_from(before).unwrap() + usize::try_from(after).unwrap()
        })
        .collect::<Vec<_>>();
    let output_len = output_dimensions.iter().product::<usize>();
    let input_strides = contiguous_strides_usize(input_dimensions);
    let output_strides = contiguous_strides_usize(&output_dimensions);
    let mut output = vec![pad_value; output_len];
    for (output_index, output_value) in output.iter_mut().enumerate() {
        let mut source_index = 0usize;
        let mut inside = true;
        for axis in 0..output_dimensions.len() {
            let coord = output_index / output_strides[axis] % output_dimensions[axis];
            let before = padding
                .get(axis)
                .map_or(0usize, |padding| usize::try_from(padding.0).unwrap());
            if coord < before || coord >= before + input_dimensions[axis] {
                inside = false;
                break;
            }
            source_index += (coord - before) * input_strides[axis];
        }
        if inside {
            *output_value = values[source_index];
        }
    }
    output
}

pub fn expected_concat_f32(
    lhs: &[f32],
    rhs: &[f32],
    lhs_dimensions: &[usize],
    rhs_dimensions: &[usize],
    axis: usize,
) -> Vec<f32> {
    let mut output_dimensions = lhs_dimensions.to_vec();
    output_dimensions[axis] += rhs_dimensions[axis];
    let output_len = output_dimensions.iter().product::<usize>();
    let lhs_strides = contiguous_strides_usize(lhs_dimensions);
    let rhs_strides = contiguous_strides_usize(rhs_dimensions);
    let output_strides = contiguous_strides_usize(&output_dimensions);
    let mut output = Vec::with_capacity(output_len);
    for output_index in 0..output_len {
        let mut coords = Vec::with_capacity(output_dimensions.len());
        for candidate_axis in 0..output_dimensions.len() {
            coords.push(
                output_index / output_strides[candidate_axis] % output_dimensions[candidate_axis],
            );
        }
        let from_lhs = coords[axis] < lhs_dimensions[axis];
        if !from_lhs {
            coords[axis] -= lhs_dimensions[axis];
        }
        let strides = if from_lhs { &lhs_strides } else { &rhs_strides };
        let mut source_index = 0usize;
        for candidate_axis in 0..output_dimensions.len() {
            source_index += coords[candidate_axis] * strides[candidate_axis];
        }
        output.push(if from_lhs {
            lhs[source_index]
        } else {
            rhs[source_index]
        });
    }
    output
}

pub fn expected_expand_f32(
    values: &[f32],
    input_dimensions: &[usize],
    output_dimensions: &[usize],
) -> Vec<f32> {
    let input_strides = contiguous_strides_usize(input_dimensions);
    let output_strides = contiguous_strides_usize(output_dimensions);
    let rank_delta = output_dimensions.len() - input_dimensions.len();
    let output_len = output_dimensions.iter().product::<usize>();
    let mut output = Vec::with_capacity(output_len);
    for output_index in 0..output_len {
        let mut source_index = 0usize;
        for output_axis in 0..output_dimensions.len() {
            if output_axis < rank_delta {
                continue;
            }
            let input_axis = output_axis - rank_delta;
            let coord = output_index / output_strides[output_axis] % output_dimensions[output_axis];
            let source_coord = if input_dimensions[input_axis] == 1 {
                0
            } else {
                coord
            };
            source_index += source_coord * input_strides[input_axis];
        }
        output.push(values[source_index]);
    }
    output
}

pub fn expected_repeat_kv_f32(
    values: &[f32],
    batch: usize,
    heads: usize,
    sequence: usize,
    head_dim: usize,
    repeats: usize,
) -> Vec<f32> {
    let output_heads = heads * repeats;
    let mut output = Vec::with_capacity(batch * output_heads * sequence * head_dim);
    for batch_index in 0..batch {
        for output_head in 0..output_heads {
            let input_head = output_head / repeats;
            for sequence_index in 0..sequence {
                for dim in 0..head_dim {
                    let source_index =
                        ((batch_index * heads + input_head) * sequence + sequence_index) * head_dim
                            + dim;
                    output.push(values[source_index]);
                }
            }
        }
    }
    output
}

pub fn contiguous_strides_usize(dimensions: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dimensions.len()];
    let mut stride = 1usize;
    for (index, &dimension) in dimensions.iter().enumerate().rev() {
        strides[index] = stride;
        stride *= dimension;
    }
    strides
}
