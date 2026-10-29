//! RMS norm, layer norm, group norm, sparsemax, and related reductions.

#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "the f32 algorithm converts integer dimensions and accumulators to its floating-point working precision"
)]
pub fn sparsemax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let values = &input[row * cols..(row + 1) * cols];
        let mut sorted = values.to_vec();
        sorted.sort_by(|lhs, rhs| rhs.total_cmp(lhs));
        let mut prefix_sum = 0.0f32;
        let mut support = 0usize;
        for (index, &value) in sorted.iter().enumerate() {
            prefix_sum += value;
            let rank = index + 1;
            if (rank as f32).mul_add(value, 1.0) > prefix_sum {
                support = rank;
            }
        }
        let tau = (sorted[..support].iter().sum::<f32>() - 1.0) / support as f32;
        for col in 0..cols {
            out[row * cols + col] = (values[col] - tau).max(0.0);
        }
    }
    out
}

#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "the f32 algorithm converts integer dimensions and accumulators to its floating-point working precision"
)]
pub fn group_norm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    channels: usize,
    groups: usize,
    spatial_len: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * channels * spatial_len];
    let channels_per_group = channels / groups;
    for batch_index in 0..batch {
        for group in 0..groups {
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            for channel_in_group in 0..channels_per_group {
                let channel = group * channels_per_group + channel_in_group;
                for spatial in 0..spatial_len {
                    let index = (batch_index * channels + channel) * spatial_len + spatial;
                    let value = input[index];
                    sum += value;
                    sum_sq = value.mul_add(value, sum_sq);
                }
            }
            let group_len = (channels_per_group * spatial_len) as f32;
            let mean = sum / group_len;
            let variance = mean.mul_add(-mean, sum_sq / group_len);
            let rstd = 1.0 / (variance + eps).sqrt();
            for channel_in_group in 0..channels_per_group {
                let channel = group * channels_per_group + channel_in_group;
                for spatial in 0..spatial_len {
                    let index = (batch_index * channels + channel) * spatial_len + spatial;
                    out[index] =
                        ((input[index] - mean) * rstd).mul_add(weight[channel], bias[channel]);
                }
            }
        }
    }
    out
}
