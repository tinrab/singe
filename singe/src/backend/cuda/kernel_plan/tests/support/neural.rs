use super::*;

pub fn expected_batch_norm_f32(
    values: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    running_mean: Option<&[f32]>,
    running_var: Option<&[f32]>,
    feature_count: usize,
    spatial_len: usize,
    eps: f32,
) -> Vec<f32> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let channel = index / spatial_len % feature_count;
            let mean = running_mean.map_or(0.0, |values| values[channel]);
            let var = running_var.map_or(1.0, |values| values[channel]);
            let normalized = (*value - mean) / (var + eps).sqrt();
            let scaled = weight.map_or(normalized, |values| normalized * values[channel]);
            bias.map_or(scaled, |values| scaled + values[channel])
        })
        .collect()
}

pub fn normalization_input_values(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| ((index % 37) as f32 - 18.0) * 0.03125)
        .collect()
}

pub fn expected_normalization_f32(
    operation: Operation,
    values: &[f32],
    rows: usize,
    cols: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let row_offset = row * cols;
        let row_values = &values[row_offset..row_offset + cols];
        let inv_scale = match operation {
            Operation::LayerNorm => {
                let mean = row_values.iter().sum::<f32>() / cols as f32;
                let variance = row_values
                    .iter()
                    .map(|value| {
                        let diff = *value - mean;
                        diff * diff
                    })
                    .sum::<f32>()
                    / cols as f32;
                1.0 / (variance + eps).sqrt()
            }
            Operation::RmsNorm => {
                let mean_square =
                    row_values.iter().map(|value| value * value).sum::<f32>() / cols as f32;
                1.0 / (mean_square + eps).sqrt()
            }
            _ => panic!("unsupported normalization operation"),
        };
        let mean = if operation == Operation::LayerNorm {
            row_values.iter().sum::<f32>() / cols as f32
        } else {
            0.0
        };
        for col in 0..cols {
            let normalized = (row_values[col] - mean) * inv_scale;
            let scaled = weight.map_or(normalized, |values| normalized * values[col]);
            output[row_offset + col] = bias.map_or(scaled, |values| scaled + values[col]);
        }
    }
    output
}

pub fn rotary_input_values(
    batch: usize,
    heads: usize,
    sequence: usize,
    head_dim: usize,
) -> Vec<f32> {
    (0..batch * heads * sequence * head_dim)
        .map(|value| ((value % 29) as f32 - 14.0) * 0.03125)
        .collect()
}

pub fn rotary_trig_values(sequence: usize, rotary_pairs: usize, is_cos: bool) -> Vec<f32> {
    (0..sequence * rotary_pairs)
        .map(|index| {
            let position = index / rotary_pairs;
            let pair = index % rotary_pairs;
            let angle = (pair as f32).mul_add(0.25, position as f32 * 0.125);
            if is_cos { angle.cos() } else { angle.sin() }
        })
        .collect()
}

pub fn expected_rotary_embedding_f32(
    input: &[f32],
    cos: &[f32],
    sin: &[f32],
    batch: usize,
    heads: usize,
    sequence: usize,
    head_dim: usize,
    rotary_pairs: usize,
) -> Vec<f32> {
    let mut output = input.to_vec();
    for batch_index in 0..batch {
        for head_index in 0..heads {
            for sequence_index in 0..sequence {
                let token_base =
                    ((batch_index * heads + head_index) * sequence + sequence_index) * head_dim;
                let position_base = sequence_index * rotary_pairs;
                for pair in 0..rotary_pairs {
                    let even_index = token_base + pair * 2;
                    let odd_index = even_index + 1;
                    let cos_value = cos[position_base + pair];
                    let sin_value = sin[position_base + pair];
                    let even = input[even_index];
                    let odd = input[odd_index];
                    output[even_index] = odd.mul_add(-sin_value, even * cos_value);
                    output[odd_index] = even.mul_add(sin_value, odd * cos_value);
                }
            }
        }
    }
    output
}
