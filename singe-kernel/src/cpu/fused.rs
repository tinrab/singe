//! Inference-oriented fused operations that combine common model steps.

#[cfg(feature = "dtype-bf16")]
use half::bf16;
#[cfg(feature = "dtype-f16")]
use half::f16;

pub fn rms_norm_gated_silu(
    input: &[f32],
    gate: &[f32],
    weight: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
    weight_offset: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let row_start = row * cols;
        let variance = input[row_start..row_start + cols]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / cols as f32;
        let inv_rms = 1.0 / (variance + eps).sqrt();
        for (column, weight_value) in weight.iter().copied().enumerate().take(cols) {
            let offset = row_start + column;
            let gate_value = gate[offset];
            let silu_gate = gate_value / (1.0 + (-gate_value).exp());
            out[offset] = input[offset] * inv_rms * (weight_value + weight_offset) * silu_gate;
        }
    }
    out
}

pub fn rms_norm(input: &[f32], weight: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let row_start = row * cols;
        let variance = input[row_start..row_start + cols]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / cols as f32;
        let inv_rms = 1.0 / (variance + eps).sqrt();
        for (column, weight_value) in weight.iter().copied().enumerate().take(cols) {
            let offset = row_start + column;
            out[offset] = input[offset] * inv_rms * weight_value;
        }
    }
    out
}

pub fn rms_norm_weight_offset(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
    weight_offset: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let row_start = row * cols;
        let variance = input[row_start..row_start + cols]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / cols as f32;
        let inv_rms = 1.0 / (variance + eps).sqrt();
        for (column, weight_value) in weight.iter().copied().enumerate().take(cols) {
            let offset = row_start + column;
            out[offset] = input[offset] * inv_rms * (weight_value + weight_offset);
        }
    }
    out
}

pub fn silu_and_mul_packed(input: &[f32], rows: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * hidden];
    for row in 0..rows {
        let input_row = row * hidden * 2;
        let output_row = row * hidden;
        for column in 0..hidden {
            let gate = input[input_row + column];
            let up = input[input_row + hidden + column];
            out[output_row + column] = gate / (1.0 + (-gate).exp()) * up;
        }
    }
    out
}

pub fn mhc_apply_residual_f32(
    x: &[f32],
    f_out: &[f32],
    y: &[f32],
    batch: usize,
    n: usize,
    channels: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * n * channels];
    let y_row = n * (n + 2);
    for batch_index in 0..batch {
        for token in 0..n {
            for channel in 0..channels {
                let mut sum =
                    y[batch_index * y_row + n + token] * f_out[batch_index * channels + channel];
                for source_token in 0..n {
                    let y_res = y[batch_index * y_row + 2 * n + token * n + source_token];
                    let x_value = x[batch_index * n * channels + source_token * channels + channel];
                    sum += y_res * x_value;
                }
                out[batch_index * n * channels + token * channels + channel] = sum;
            }
        }
    }
    out
}

pub fn mhc_sinkhorn_f32(y: &[f32], batch: usize, n: usize) -> Vec<f32> {
    let mut out = y.to_vec();
    let y_row = n * (n + 2);
    for batch_index in 0..batch {
        let base = batch_index * y_row + 2 * n;
        let mut matrix = vec![0.0f32; n * n];
        for index in 0..n * n {
            matrix[index] = out[base + index].exp();
        }
        for _ in 0..20 {
            for row in 0..n {
                let row_start = row * n;
                let row_sum = matrix[row_start..row_start + n].iter().sum::<f32>();
                for column in 0..n {
                    matrix[row_start + column] /= row_sum;
                }
            }
            for column in 0..n {
                let mut column_sum = 0.0f32;
                for row in 0..n {
                    column_sum += matrix[row * n + column];
                }
                for row in 0..n {
                    matrix[row * n + column] /= column_sum;
                }
            }
        }
        out[base..base + n * n].copy_from_slice(&matrix);
    }
    out
}

pub fn mhc_gemm_rms_scale_f32(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    rows: usize,
    columns: usize,
    reduction: usize,
    n: usize,
    alpha_pre: f32,
    alpha_post: f32,
    alpha_res: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; rows * columns];
    let mut r = vec![0.0f32; rows];
    for row in 0..rows {
        let mut rms_sum = 0.0f32;
        for k in 0..reduction {
            let value = x[row * reduction + k];
            rms_sum += value * value;
        }
        let rms = (rms_sum / reduction as f32).sqrt();
        r[row] = rms;
        for column in 0..columns {
            let mut dot = 0.0f32;
            for k in 0..reduction {
                dot += x[row * reduction + k] * w[k * columns + column];
            }
            let scale = if column < n {
                alpha_pre
            } else if column < 2 * n {
                alpha_post
            } else {
                alpha_res
            };
            let linear = dot * scale / rms + bias[column];
            y[row * columns + column] = if column < n {
                1.0 / (1.0 + (-linear).exp())
            } else if column < 2 * n {
                2.0 / (1.0 + (-linear).exp())
            } else {
                linear
            };
        }
    }
    (y, r)
}

#[cfg(feature = "dtype-f16")]
pub fn half_vec(values: &[f32]) -> Vec<f16> {
    values.iter().copied().map(f16::from_f32).collect()
}

#[cfg(feature = "dtype-f16")]
pub fn half_to_f32(values: &[f16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

#[cfg(feature = "dtype-bf16")]
pub fn bfloat_vec(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

#[cfg(feature = "dtype-bf16")]
pub fn bfloat_to_f32(values: &[bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

#[cfg(feature = "dtype-bf16")]
pub fn round_bfloat_vec(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .copied()
        .map(bf16::from_f32)
        .map(|value| value.to_f32())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_weight_offset_changes_outputs() {
        let input = vec![0.5f32, -1.0, 2.0, -0.25, 1.5, -0.75];
        let weight = vec![0.25f32, -0.5, 0.75];
        let rows = 2usize;
        let cols = 3usize;
        let eps = 1e-5f32;

        let standard = rms_norm_weight_offset(&input, &weight, rows, cols, eps, 0.0);
        let offset_output = rms_norm_weight_offset(&input, &weight, rows, cols, eps, 1.0);

        assert_ne!(standard, offset_output);
        for row in 0..rows {
            let row_start = row * cols;
            let variance = input[row_start..row_start + cols]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / cols as f32;
            let inv_rms = 1.0 / (variance + eps).sqrt();
            for column in 0..cols {
                let offset = row_start + column;
                let expected_offset_contribution = input[offset] * inv_rms;
                singe_core::assert_close!(
                    &[offset_output[offset] - standard[offset]],
                    &[expected_offset_contribution],
                    1e-6,
                );
            }
        }
    }

    #[test]
    fn gated_rms_norm_zero_offset_is_distinct_from_offset_one() {
        let input = vec![0.5f32, -1.0, 2.0];
        let gate = vec![1.0f32, -0.5, 0.25];
        let weight = vec![0.25f32, -0.5, 0.75];
        let rows = 1usize;
        let cols = 3usize;
        let eps = 1e-5f32;

        let gated_zero = rms_norm_gated_silu(&input, &gate, &weight, rows, cols, eps, 0.0);
        let gated_offset = rms_norm_gated_silu(&input, &gate, &weight, rows, cols, eps, 1.0);

        assert_ne!(gated_zero, gated_offset);
        let variance = input.iter().map(|value| value * value).sum::<f32>() / cols as f32;
        let inv_rms = 1.0 / (variance + eps).sqrt();
        for column in 0..cols {
            let gate_value = gate[column];
            let silu_gate = gate_value / (1.0 + (-gate_value).exp());
            let expected_offset_contribution = input[column] * inv_rms * silu_gate;
            singe_core::assert_close!(
                &[gated_offset[column] - gated_zero[column]],
                &[expected_offset_contribution],
                1e-6,
            );
        }
    }
}
