//! Recurrent/state-space preprocessing helpers.

#[cfg(feature = "dtype-bf16")]
use half::bf16;
#[cfg(feature = "dtype-f16")]
use half::f16;

/// Prepares Gated Delta Rule recurrent inputs after projection.
///
/// `b` and `a` are `[rows, hidden]`; `a_log` and `dt_bias` are `[hidden]`.
/// The returned beta is `sigmoid(b)`.
/// The returned gate is `-exp(a_log) * softplus(a + dt_bias)`, using the same stable softplus threshold as the Triton reference.
pub fn gated_delta_rule_preprocess(
    b: &[f32],
    a: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    rows: usize,
    hidden: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut beta = vec![0.0f32; rows * hidden];
    let mut g = vec![0.0f32; rows * hidden];
    for row in 0..rows {
        for column in 0..hidden {
            let offset = row * hidden + column;
            beta[offset] = 1.0 / (1.0 + (-b[offset]).exp());
            g[offset] = -a_log[column].exp() * stable_softplus(a[offset] + dt_bias[column]);
        }
    }
    (beta, g)
}

fn stable_softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
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

#[cfg(feature = "dtype-f16")]
pub fn round_half_vec(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .copied()
        .map(f16::from_f32)
        .map(|value| value.to_f32())
        .collect()
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

pub fn recurrent_gated_delta_rule(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    gate: &[f32],
    beta: &[f32],
    initial_state: Option<&[f32]>,
    batch: usize,
    time: usize,
    query_heads: usize,
    value_heads: usize,
    qk_dim: usize,
    value_dim: usize,
    use_qk_l2norm: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; batch * time * value_heads * value_dim];
    let mut final_state = vec![0.0f32; batch * value_heads * qk_dim * value_dim];
    let heads_per_group = value_heads / query_heads;
    for b in 0..batch {
        for hv in 0..value_heads {
            let h = hv / heads_per_group;
            for v in 0..value_dim {
                let mut state = vec![0.0f32; qk_dim];
                if let Some(initial_state) = initial_state {
                    for k in 0..qk_dim {
                        state[k] =
                            initial_state[((b * value_heads + hv) * qk_dim + k) * value_dim + v];
                    }
                }
                for t in 0..time {
                    let mut query_t = vec![0.0f32; qk_dim];
                    let mut key_t = vec![0.0f32; qk_dim];
                    for k in 0..qk_dim {
                        let offset = ((b * time + t) * query_heads + h) * qk_dim + k;
                        query_t[k] = query[offset];
                        key_t[k] = key[offset];
                    }
                    if use_qk_l2norm {
                        normalize(&mut query_t);
                        normalize(&mut key_t);
                    }
                    let scale = 1.0 / (qk_dim as f32).sqrt();
                    for value in &mut query_t {
                        *value *= scale;
                    }
                    let gate_offset = (b * time + t) * value_heads + hv;
                    let gamma = gate[gate_offset].exp();
                    let beta_t = beta[gate_offset];
                    for value in &mut state {
                        *value *= gamma;
                    }
                    let kv_memory = dot(&state, &key_t);
                    let value_t = value[((b * time + t) * value_heads + hv) * value_dim + v];
                    let delta = (value_t - kv_memory) * beta_t;
                    for k in 0..qk_dim {
                        state[k] += key_t[k] * delta;
                    }
                    out[((b * time + t) * value_heads + hv) * value_dim + v] =
                        dot(&state, &query_t);
                }
                for k in 0..qk_dim {
                    final_state[((b * value_heads + hv) * qk_dim + k) * value_dim + v] = state[k];
                }
            }
        }
    }
    (out, final_state)
}

fn normalize(values: &mut [f32]) {
    let norm = values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-6);
    for value in values {
        *value /= norm;
    }
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdr_preprocess_uses_stable_softplus_for_large_values() {
        let b = vec![0.0; 4];
        let a = vec![-50.0, 0.0, 25.0, 100.0];
        let a_log = vec![0.0, 0.25, -0.5, 0.5];
        let dt_bias = vec![0.0; 4];

        let (_, g) = gated_delta_rule_preprocess(&b, &a, &a_log, &dt_bias, 1, 4);

        let expected = a
            .iter()
            .zip(&a_log)
            .map(|(a, a_log)| -a_log.exp() * stable_softplus(*a))
            .collect::<Vec<_>>();
        singe_core::assert_close!(&g, &expected, 1e-5);
        assert!(g.iter().all(|value| value.is_finite()));
    }
}
