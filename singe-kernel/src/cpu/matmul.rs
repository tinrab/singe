//! GEMM, batched GEMM, ragged batched GEMM, and matvec variants.

pub fn matmul_f32(
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let mut sum = 0.0f32;
            for k in 0..reduction {
                let lhs_index = if transpose_lhs {
                    k * lhs_row_stride + row
                } else {
                    row * lhs_row_stride + k
                };
                let rhs_index = if transpose_rhs {
                    column * rhs_row_stride + k
                } else {
                    k * rhs_row_stride + column
                };
                sum += lhs[lhs_index] * rhs[rhs_index];
            }
            out[row * columns + column] = sum;
        }
    }
    out
}

pub fn matmul_f64(
    lhs: &[f64],
    rhs: &[f64],
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let mut sum = 0.0f64;
            for k in 0..reduction {
                let lhs_index = if transpose_lhs {
                    k * lhs_row_stride + row
                } else {
                    row * lhs_row_stride + k
                };
                let rhs_index = if transpose_rhs {
                    column * rhs_row_stride + k
                } else {
                    k * rhs_row_stride + column
                };
                sum += lhs[lhs_index] * rhs[rhs_index];
            }
            out[row * columns + column] = sum;
        }
    }
    out
}

pub fn bmm_f32(
    lhs: &[f32],
    rhs: &[f32],
    batch: usize,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_batch_stride: usize,
    lhs_row_stride: usize,
    rhs_batch_stride: usize,
    rhs_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * rows * columns];
    for batch_index in 0..batch {
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = 0.0f32;
                for k in 0..reduction {
                    let lhs_index = if transpose_lhs {
                        batch_index * lhs_batch_stride + k * lhs_row_stride + row
                    } else {
                        batch_index * lhs_batch_stride + row * lhs_row_stride + k
                    };
                    let rhs_index = if transpose_rhs {
                        batch_index * rhs_batch_stride + column * rhs_row_stride + k
                    } else {
                        batch_index * rhs_batch_stride + k * rhs_row_stride + column
                    };
                    sum += lhs[lhs_index] * rhs[rhs_index];
                }
                out[batch_index * rows * columns + row * columns + column] = sum;
            }
        }
    }
    out
}

pub fn masked_bmm_f32(
    lhs: &[f32],
    rhs: &[f32],
    initial_out: &[f32],
    masked_rows: &[i32],
    batch: usize,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_batch_stride: usize,
    lhs_row_stride: usize,
    rhs_batch_stride: usize,
    rhs_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Vec<f32> {
    let mut out = initial_out.to_vec();
    for batch_index in 0..batch {
        let valid_rows = masked_rows[batch_index].clamp(0, rows as i32) as usize;
        for row in 0..valid_rows {
            for column in 0..columns {
                let mut sum = 0.0f32;
                for k in 0..reduction {
                    let lhs_index = if transpose_lhs {
                        batch_index * lhs_batch_stride + k * lhs_row_stride + row
                    } else {
                        batch_index * lhs_batch_stride + row * lhs_row_stride + k
                    };
                    let rhs_index = if transpose_rhs {
                        batch_index * rhs_batch_stride + column * rhs_row_stride + k
                    } else {
                        batch_index * rhs_batch_stride + k * rhs_row_stride + column
                    };
                    sum += lhs[lhs_index] * rhs[rhs_index];
                }
                out[batch_index * rows * columns + row * columns + column] = sum;
            }
        }
    }
    out
}

pub fn ragged_bmm_f32(
    lhs: &[f32],
    rhs: &[f32],
    row_indptr: &[i32],
    batch: usize,
    total_rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_batch_stride: usize,
    rhs_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; total_rows * columns];
    for batch_index in 0..batch {
        let row_start = row_indptr[batch_index] as usize;
        let row_end = row_indptr[batch_index + 1] as usize;
        for global_row in row_start..row_end {
            for column in 0..columns {
                let mut sum = 0.0f32;
                for k in 0..reduction {
                    let lhs_index = if transpose_lhs {
                        k * lhs_row_stride + global_row
                    } else {
                        global_row * lhs_row_stride + k
                    };
                    let rhs_index = if transpose_rhs {
                        batch_index * rhs_batch_stride + column * rhs_row_stride + k
                    } else {
                        batch_index * rhs_batch_stride + k * rhs_row_stride + column
                    };
                    sum += lhs[lhs_index] * rhs[rhs_index];
                }
                out[global_row * columns + column] = sum;
            }
        }
    }
    out
}

pub fn group_gemm_f32(
    lhs: &[f32],
    rhs: &[f32],
    rows: &[i32],
    columns: &[i32],
    reductions: &[i32],
    lhs_offsets: &[i32],
    rhs_offsets: &[i32],
    output_offsets: &[i32],
    output_len: usize,
    transpose_rhs: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; output_len];
    for group in 0..rows.len() {
        let group_rows = rows[group] as usize;
        let group_columns = columns[group] as usize;
        let group_reduction = reductions[group] as usize;
        let lhs_base = lhs_offsets[group] as usize;
        let rhs_base = rhs_offsets[group] as usize;
        let output_base = output_offsets[group] as usize;
        for row in 0..group_rows {
            for column in 0..group_columns {
                let mut sum = 0.0f32;
                for k in 0..group_reduction {
                    let lhs_index = lhs_base + row * group_reduction + k;
                    let rhs_index = if transpose_rhs {
                        rhs_base + column * group_reduction + k
                    } else {
                        rhs_base + k * group_columns + column
                    };
                    sum += lhs[lhs_index] * rhs[rhs_index];
                }
                out[output_base + row * group_columns + column] = sum;
            }
        }
    }
    out
}

/// Grouped expert GEMM with optional input/output permutation metadata.
///
/// `m_sizes` gives the token count for each expert. For each expert group, rows
/// are multiplied by that expert's `[columns, reduction]` weight tile.
/// When `permute_input` or `permute_output` is set, `gather_indices` maps sorted expert rows back to the original token/top-k routing order.
pub fn grouped_gemm_f32(
    input: &[f32],
    weights: &[f32],
    m_sizes: &[i32],
    gather_indices: &[i32],
    total_tokens: usize,
    columns: usize,
    reduction: usize,
    input_row_stride: usize,
    weight_expert_stride: usize,
    weight_row_stride: usize,
    output_row_stride: usize,
    permute_input: bool,
    permute_output: bool,
    top_k: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; total_tokens * output_row_stride];
    let mut token_start = 0usize;
    for (expert, m_size) in m_sizes.iter().copied().enumerate() {
        let expert_tokens = m_size as usize;
        for local_row in 0..expert_tokens {
            let sorted_row = token_start + local_row;
            let gathered_row = gather_indices[sorted_row] as usize;
            let input_row = if permute_input {
                gathered_row / top_k
            } else {
                sorted_row
            };
            let output_row = if permute_output {
                gathered_row
            } else {
                sorted_row
            };
            for column in 0..columns {
                let mut sum = 0.0f32;
                for k in 0..reduction {
                    let input_index = input_row * input_row_stride + k;
                    let weight_index =
                        expert * weight_expert_stride + column * weight_row_stride + k;
                    sum += input[input_index] * weights[weight_index];
                }
                out[output_row * output_row_stride + column] = sum;
            }
        }
        token_start += expert_tokens;
    }
    out
}
#[cfg(feature = "dtype-f8")]

pub fn ragged_block_scaled_bmm_f32(
    lhs: &[u8],
    rhs: &[u8],
    lhs_scale: &[f32],
    rhs_scale: &[f32],
    row_indptr: &[i32],
    batch: usize,
    total_rows: usize,
    columns: usize,
    reduction: usize,
    scale_block: usize,
    lhs_scale_row_stride: usize,
    rhs_scale_batch_stride: usize,
    rhs_scale_row_stride: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; total_rows * columns];
    for batch_index in 0..batch {
        let row_start = row_indptr[batch_index] as usize;
        let row_end = row_indptr[batch_index + 1] as usize;
        for global_row in row_start..row_end {
            for column in 0..columns {
                let mut sum = 0.0f32;
                for k in 0..reduction {
                    let scale_k = k / scale_block;
                    let lhs_index = global_row * reduction + k;
                    let rhs_index = batch_index * columns * reduction + column * reduction + k;
                    let lhs_scale_index = global_row * lhs_scale_row_stride + scale_k;
                    let rhs_scale_index = batch_index * rhs_scale_batch_stride
                        + (column / scale_block) * rhs_scale_row_stride
                        + scale_k;
                    sum += f8e4m3_value(lhs[lhs_index])
                        * f8e4m3_value(rhs[rhs_index])
                        * lhs_scale[lhs_scale_index]
                        * rhs_scale[rhs_scale_index];
                }
                out[global_row * columns + column] = sum;
            }
        }
    }
    out
}

#[cfg(feature = "dtype-f8")]
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

pub fn bmm_f64(
    lhs: &[f64],
    rhs: &[f64],
    batch: usize,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_batch_stride: usize,
    lhs_row_stride: usize,
    rhs_batch_stride: usize,
    rhs_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Vec<f64> {
    let mut out = vec![0.0f64; batch * rows * columns];
    for batch_index in 0..batch {
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = 0.0f64;
                for k in 0..reduction {
                    let lhs_index = if transpose_lhs {
                        batch_index * lhs_batch_stride + k * lhs_row_stride + row
                    } else {
                        batch_index * lhs_batch_stride + row * lhs_row_stride + k
                    };
                    let rhs_index = if transpose_rhs {
                        batch_index * rhs_batch_stride + column * rhs_row_stride + k
                    } else {
                        batch_index * rhs_batch_stride + k * rhs_row_stride + column
                    };
                    sum += lhs[lhs_index] * rhs[rhs_index];
                }
                out[batch_index * rows * columns + row * columns + column] = sum;
            }
        }
    }
    out
}
