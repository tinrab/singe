//! RoPE and positional embedding variants.

#[cfg(feature = "dtype-bf16")]
use half::bf16;
#[cfg(feature = "dtype-f16")]
use half::f16;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Three-axis half-split RoPE config for Q/K tensors.
///
/// The cosine and sine tables are split into temporal, height, and width sections.
/// Each section rotates dimensions from the first half of the head with the corresponding dimensions in the second half.
pub struct MultiaxisRopeConfig {
    pub batch: usize,
    pub seq_len: usize,
    pub query_heads: usize,
    pub key_heads: usize,
    pub head_dim: usize,
    pub section_t: usize,
    pub section_h: usize,
    pub section_w: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Interleaved complex-frequency RoPE config for Q/K tensors.
///
/// Adjacent dimensions are treated as real/imaginary pairs and multiplied by a precomputed complex frequency table laid out as `[seq_len, head_dim]`.
pub struct InterleavedComplexRopeConfig {
    pub batch: usize,
    pub seq_len: usize,
    pub query_heads: usize,
    pub key_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct BnsdQkRopeConfig {
    pub batch: usize,
    pub seq_len: usize,
    pub query_heads: usize,
    pub key_heads: usize,
    pub head_dim: usize,
    pub trig_batch: usize,
    pub rotary_pairs: usize,
}

/// Applies three-axis half-split RoPE to query and key tensors.
///
/// Query and key use `[batch, seq_len, heads, head_dim]` layout.
/// The cosine and sine tables contain temporal, height, and width sections.
/// Each section rotates first-half head dimensions with their matching second-half dimensions.
pub fn multiaxis_rope_qk(
    query: &mut [f32],
    key: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    config: MultiaxisRopeConfig,
) {
    rotate_multiaxis_rope_tensor(query, cos, sin, config, config.query_heads);
    rotate_multiaxis_rope_tensor(key, cos, sin, config, config.key_heads);
}

/// Applies interleaved complex-frequency RoPE to query and key tensors.
///
/// Query and key use `[batch, seq_len, heads, head_dim]` layout.
/// Adjacent dimensions `(0, 1)`, `(2, 3)`, and so on are treated as real/imaginary pairs
/// and multiplied by the `[seq_len, head_dim]` complex frequency table.
pub fn interleaved_complex_rope_qk(
    query: &mut [f32],
    key: &mut [f32],
    freqs: &[f32],
    config: InterleavedComplexRopeConfig,
) {
    rotate_interleaved_complex_rope_tensor(query, freqs, config, config.query_heads);
    rotate_interleaved_complex_rope_tensor(key, freqs, config, config.key_heads);
}

fn rotate_interleaved_complex_rope_tensor(
    values: &mut [f32],
    freqs: &[f32],
    config: InterleavedComplexRopeConfig,
    heads: usize,
) {
    let head_dim_half = config.head_dim / 2;
    for batch in 0..config.batch {
        for seq in 0..config.seq_len {
            for head in 0..heads {
                for pair in 0..head_dim_half {
                    let real_dim = pair * 2;
                    let imag_dim = real_dim + 1;
                    let base = ((batch * config.seq_len + seq) * heads + head) * config.head_dim;
                    let real_index = base + real_dim;
                    let imag_index = base + imag_dim;
                    let freq_base = seq * config.head_dim;
                    let freq_real = freqs[freq_base + real_dim];
                    let freq_imag = freqs[freq_base + imag_dim];
                    let real = values[real_index];
                    let imag = values[imag_index];
                    values[real_index] = real * freq_real - imag * freq_imag;
                    values[imag_index] = real * freq_imag + imag * freq_real;
                }
            }
        }
    }
}

fn rotate_multiaxis_rope_tensor(
    values: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    config: MultiaxisRopeConfig,
    heads: usize,
) {
    let head_dim_half = config.head_dim / 2;
    for batch in 0..config.batch {
        for seq in 0..config.seq_len {
            for head in 0..heads {
                for dim in 0..head_dim_half {
                    let section = if dim < config.section_t {
                        0
                    } else if dim < config.section_t + config.section_h {
                        1
                    } else {
                        2
                    };
                    let trig_index = ((section * config.batch + batch) * config.seq_len + seq)
                        * head_dim_half
                        + dim;
                    let base =
                        ((batch * config.seq_len + seq) * heads + head) * config.head_dim + dim;
                    let real = values[base];
                    let imag_index = base + head_dim_half;
                    let imag = values[imag_index];
                    values[base] = real * cos[trig_index] - imag * sin[trig_index];
                    values[imag_index] = imag * cos[trig_index] + real * sin[trig_index];
                }
            }
        }
    }
}

pub fn rotate_bnsd_qk_rope_reference(
    query: &mut [f32],
    key: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    config: BnsdQkRopeConfig,
) {
    rotate_bnsd_rope_tensor(query, cos, sin, config, config.query_heads);
    rotate_bnsd_rope_tensor(key, cos, sin, config, config.key_heads);
}

fn rotate_bnsd_rope_tensor(
    values: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    config: BnsdQkRopeConfig,
    heads: usize,
) {
    for batch in 0..config.batch {
        let trig_batch = if config.trig_batch == 1 { 0 } else { batch };
        for head in 0..heads {
            for seq in 0..config.seq_len {
                for pair in 0..config.rotary_pairs {
                    let trig_index =
                        (trig_batch * config.seq_len + seq) * config.rotary_pairs + pair;
                    let base = ((batch * heads + head) * config.seq_len + seq) * config.head_dim;
                    let real_index = base + pair;
                    let imag_index = base + config.rotary_pairs + pair;
                    let real = values[real_index];
                    let imag = values[imag_index];
                    values[real_index] = real * cos[trig_index] - imag * sin[trig_index];
                    values[imag_index] = imag * cos[trig_index] + real * sin[trig_index];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_conventions_match_reference_values() {
        let mut interleaved = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut half_split = interleaved.clone();
        let mut unused_key = Vec::new();
        let cos = [0.5f32, 0.25];
        let sin = [0.75f32, -0.5];

        rotate_interleaved_complex_rope_tensor(
            &mut interleaved,
            &[cos[0], sin[0], cos[1], sin[1]],
            InterleavedComplexRopeConfig {
                batch: 1,
                seq_len: 1,
                query_heads: 1,
                key_heads: 0,
                head_dim: 4,
            },
            1,
        );
        rotate_bnsd_qk_rope_reference(
            &mut half_split,
            &mut unused_key,
            &cos,
            &sin,
            BnsdQkRopeConfig {
                batch: 1,
                seq_len: 1,
                query_heads: 1,
                key_heads: 0,
                head_dim: 4,
                trig_batch: 1,
                rotary_pairs: 2,
            },
        );

        singe_core::assert_close!(&interleaved, &[-1.0, 1.75, 2.75, -0.5], 1e-6);
        singe_core::assert_close!(&half_split, &[-1.75, 2.5, 2.25, 0.0], 1e-6);
    }

    #[test]
    fn rms_norm_offset_then_rope_sequence_matches_reference_values() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let weight = [0.5f32, 1.5, 2.0, 0.25];
        let cos = [0.5f32, 0.25];
        let sin = [0.75f32, -0.5];
        let eps = 1e-5f32;
        let mut query = crate::cpu::fused::rms_norm_weight_offset(&input, &weight, 1, 4, eps, 1.0);
        let mut unused_key = Vec::new();
        let rms = ((input.iter().map(|value| value * value).sum::<f32>() / 4.0) + eps).sqrt();
        let normalized = [
            input[0] / rms * (weight[0] + 1.0),
            input[1] / rms * (weight[1] + 1.0),
            input[2] / rms * (weight[2] + 1.0),
            input[3] / rms * (weight[3] + 1.0),
        ];
        let expected = [
            normalized[0] * cos[0] - normalized[2] * sin[0],
            normalized[1] * cos[1] - normalized[3] * sin[1],
            normalized[2] * cos[0] + normalized[0] * sin[0],
            normalized[3] * cos[1] + normalized[1] * sin[1],
        ];

        rotate_bnsd_qk_rope_reference(
            &mut query,
            &mut unused_key,
            &cos,
            &sin,
            BnsdQkRopeConfig {
                batch: 1,
                seq_len: 1,
                query_heads: 1,
                key_heads: 0,
                head_dim: 4,
                trig_batch: 1,
                rotary_pairs: 2,
            },
        );

        singe_core::assert_close!(&query, &expected, 1e-6);
    }

    #[test]
    fn rope_interleaved_and_half_split_conventions_differ() {
        let mut interleaved = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut half_split = interleaved.clone();
        let mut unused_key = Vec::new();

        rotate_interleaved_complex_rope_tensor(
            &mut interleaved,
            &[0.0, 1.0, 0.0, 1.0],
            InterleavedComplexRopeConfig {
                batch: 1,
                seq_len: 1,
                query_heads: 1,
                key_heads: 0,
                head_dim: 4,
            },
            1,
        );
        rotate_bnsd_qk_rope_reference(
            &mut half_split,
            &mut unused_key,
            &[0.0, 0.0],
            &[1.0, 1.0],
            BnsdQkRopeConfig {
                batch: 1,
                seq_len: 1,
                query_heads: 1,
                key_heads: 0,
                head_dim: 4,
                trig_batch: 1,
                rotary_pairs: 2,
            },
        );

        assert_eq!(interleaved, vec![-2.0, 1.0, -4.0, 3.0]);
        assert_eq!(half_split, vec![-3.0, -4.0, 1.0, 2.0]);
        assert_ne!(interleaved, half_split);
    }
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
