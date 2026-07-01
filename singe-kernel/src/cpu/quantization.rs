//! Quantize, dequantize, int4 unpacking, fp8 helpers, and quantized matmul.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantizedMatmulConfig {
    pub rows: usize,
    pub columns: usize,
    pub reduction: usize,
    pub activation_row_stride: usize,
    pub weight_row_stride: usize,
    pub output_row_stride: usize,
    pub activation_zero_point: i32,
    pub weight_zero_point: i32,
    pub output_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DequantizedWeightMatmulConfig {
    pub rows: usize,
    pub columns: usize,
    pub reduction: usize,
    pub activation_row_stride: usize,
    pub weight_row_stride: usize,
    pub output_row_stride: usize,
}

pub fn dequantize_u8_grouped(
    input: &[u8],
    scales: &[f32],
    zero_points: &[f32],
    group_size: usize,
) -> Vec<f32> {
    input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let group = index / group_size;
            (*value as f32 - zero_points[group]) * scales[group]
        })
        .collect()
}

pub fn dequantize_i8_grouped(
    input: &[i8],
    scales: &[f32],
    zero_points: &[f32],
    group_size: usize,
) -> Vec<f32> {
    input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let group = index / group_size;
            (*value as f32 - zero_points[group]) * scales[group]
        })
        .collect()
}

pub fn matmul_i8_i8_f32(
    activations: &[i8],
    weights: &[i8],
    config: QuantizedMatmulConfig,
) -> Vec<f32> {
    let mut out = vec![0.0; config.rows * config.columns];
    for row in 0..config.rows {
        for column in 0..config.columns {
            let mut sum = 0i32;
            for reduction in 0..config.reduction {
                let activation = activations[row * config.activation_row_stride + reduction];
                let weight = weights[column * config.weight_row_stride + reduction];
                sum += (activation as i32 - config.activation_zero_point)
                    * (weight as i32 - config.weight_zero_point);
            }
            out[row * config.columns + column] = sum as f32 * config.output_scale;
        }
    }
    out
}

pub fn matmul_f32_i8_dequantize_f32(
    activations: &[f32],
    weights: &[i8],
    scales: &[f32],
    zero_points: &[f32],
    config: DequantizedWeightMatmulConfig,
) -> Vec<f32> {
    let mut out = vec![0.0; config.rows * config.columns];
    for row in 0..config.rows {
        for column in 0..config.columns {
            let mut sum = 0.0f32;
            for reduction in 0..config.reduction {
                let activation = activations[row * config.activation_row_stride + reduction];
                let weight = weights[column * config.weight_row_stride + reduction];
                let weight = (weight as f32 - zero_points[column]) * scales[column];
                sum += activation * weight;
            }
            out[row * config.columns + column] = sum;
        }
    }
    out
}

pub fn scatter_strided_i8(
    values: &[i8],
    rows: usize,
    columns: usize,
    row_stride: usize,
    fill: i8,
) -> Vec<i8> {
    let mut out = vec![fill; (rows - 1) * row_stride + columns];
    for row in 0..rows {
        for column in 0..columns {
            out[row * row_stride + column] = values[row * columns + column];
        }
    }
    out
}

pub fn scatter_strided_f32(
    values: &[f32],
    rows: usize,
    columns: usize,
    row_stride: usize,
    fill: f32,
) -> Vec<f32> {
    let mut out = vec![fill; (rows - 1) * row_stride + columns];
    for row in 0..rows {
        for column in 0..columns {
            out[row * row_stride + column] = values[row * columns + column];
        }
    }
    out
}

pub fn gather_strided_f32(
    values: &[f32],
    rows: usize,
    columns: usize,
    row_stride: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        for column in 0..columns {
            out.push(values[row * row_stride + column]);
        }
    }
    out
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
/// Per-token group int8 quantization using round-to-nearest-even.
///
/// Each group uses `scale = max(max(abs(group)), eps) / 127`.
/// Values are divided by that scale, clamped to `[-128, 127]`, and rounded to nearest even before conversion to `i8`.
pub fn per_token_group_quant_i8_f32(
    input: &[f32],
    group_size: usize,
    eps: f32,
) -> (Vec<i8>, Vec<f32>) {
    let mut out = vec![0i8; input.len()];
    let mut scales = Vec::with_capacity(input.len() / group_size);
    for (group_index, group) in input.chunks_exact(group_size).enumerate() {
        let max_abs = group
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(eps);
        let scale = max_abs / 127.0;
        scales.push(scale);
        for (index, value) in group.iter().copied().enumerate() {
            out[group_index * group_size + index] = quantize_i8_round_nearest_even(value / scale);
        }
    }
    (out, scales)
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
fn quantize_i8_round_nearest_even(value: f32) -> i8 {
    let clamped = value.clamp(-128.0, 127.0);
    let floored = clamped.floor();
    let fraction = clamped - floored;
    let rounded = if fraction > 0.5 || (fraction == 0.5 && floored.rem_euclid(2.0) != 0.0) {
        floored + 1.0
    } else {
        floored
    };
    rounded as i8
}

pub fn matmul_f8e4m3_block_scaled_f32(
    activations: &[u8],
    weights: &[u8],
    activation_scales: &[f32],
    weight_scales: &[f32],
    rows: usize,
    columns: usize,
    reduction: usize,
    group_n: usize,
    group_k: usize,
) -> Vec<f32> {
    let k_groups = reduction.div_ceil(group_k);
    let mut out = vec![0.0f32; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let mut sum = 0.0f32;
            for k in 0..reduction {
                let k_group = k / group_k;
                let activation_scale = activation_scales[row * k_groups + k_group];
                let weight_scale = weight_scales[(column / group_n) * k_groups + k_group];
                sum += f8e4m3_value(activations[row * reduction + k])
                    * f8e4m3_value(weights[column * reduction + k])
                    * activation_scale
                    * weight_scale;
            }
            out[row * columns + column] = sum;
        }
    }
    out
}

pub fn dequantize_f8e4m3_block_f32(
    input: &[u8],
    scales: &[f32],
    rows: usize,
    columns: usize,
    block_size: usize,
) -> Vec<f32> {
    let scale_columns = columns.div_ceil(block_size);
    let mut out = vec![0.0f32; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let offset = row * columns + column;
            let scale_offset = (row / block_size) * scale_columns + column / block_size;
            out[offset] = f8e4m3_value(input[offset]) * scales[scale_offset];
        }
    }
    out
}

pub fn f8e4m3_value(value: u8) -> f32 {
    let sign = if value & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (value >> 3) & 0x0f;
    let mantissa = value & 0x07;
    if exponent == 0x0f && mantissa == 0x07 {
        f32::NAN
    } else if exponent == 0 {
        if mantissa == 0 {
            sign * 0.0
        } else {
            sign * (mantissa as f32 / 8.0) * 2f32.powi(-6)
        }
    } else {
        sign * (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
    #[test]
    fn per_token_group_quant_i8_rounds_ties_to_even() {
        let input = vec![127.0f32, 1.5, 2.5, -1.5, -2.5];
        let (actual, scales) = per_token_group_quant_i8_f32(&input, 5, 1e-10);

        assert_eq!(actual, vec![127, 2, 2, -2, -2]);
        assert_eq!(scales, vec![1.0]);
    }

    #[test]
    fn f8e4m3_value_decodes_full_representative_range() {
        let cases = [
            (0x00, 0.0),
            (0x01, 0.001953125),
            (0x07, 0.013671875),
            (0x08, 0.015625),
            (0x30, 0.5),
            (0x38, 1.0),
            (0x40, 2.0),
            (0x76, 224.0),
            (0x7e, 448.0),
            (0x81, -0.001953125),
            (0xb8, -1.0),
            (0xfe, -448.0),
        ];
        for (byte, expected) in cases {
            assert_eq!(f8e4m3_value(byte), expected, "byte {byte:#04x}");
        }
        assert!(f8e4m3_value(0x7f).is_nan());
        assert!(f8e4m3_value(0xff).is_nan());
    }
}
