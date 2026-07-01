//! Mixture-of-experts routing and alignment helpers.

pub fn moe_align_block_size_i32(
    topk_ids: &[i32],
    expert_count: usize,
    block_size: usize,
) -> (Vec<i32>, Vec<i32>, i32, Vec<i32>, i32) {
    let sentinel = topk_ids.len() as i32;
    let max_num_tokens_padded = topk_ids.len() + expert_count * (block_size - 1);
    let mut sorted_token_ids = vec![sentinel; max_num_tokens_padded];
    let mut cumsum = vec![0i32; expert_count + 1];
    let mut total = 0usize;
    let mut max_count = 0usize;
    for (expert, cumsum_slot) in cumsum.iter_mut().enumerate().take(expert_count) {
        *cumsum_slot = total as i32;
        let mut count = 0usize;
        for (token, &id) in topk_ids.iter().enumerate() {
            if id == expert as i32 {
                sorted_token_ids[total + count] = token as i32;
                count += 1;
            }
        }
        max_count = max_count.max(count);
        total += ceil_div(count, block_size) * block_size;
    }
    cumsum[expert_count] = total as i32;
    let expert_ids = (0..total / block_size)
        .map(|block| {
            let token_offset = (block * block_size) as i32;
            cumsum
                .windows(2)
                .position(|range| token_offset >= range[0] && token_offset < range[1])
                .unwrap_or(expert_count - 1) as i32
        })
        .collect::<Vec<_>>();
    (
        sorted_token_ids,
        expert_ids,
        total as i32,
        cumsum,
        max_count as i32,
    )
}

pub fn fused_moe_f32(
    input: &[f32],
    weight: &[f32],
    routed_weight: &[f32],
    topk_ids: &[i32],
    tokens: usize,
    top_k: usize,
    columns: usize,
    reduction: usize,
    mul_routed_weight: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens * top_k * columns];
    for slot in 0..tokens * top_k {
        let token = slot / top_k;
        let expert = topk_ids[slot] as usize;
        for column in 0..columns {
            let mut sum = 0.0f32;
            for reduction_index in 0..reduction {
                let input_offset = token * reduction + reduction_index;
                let weight_offset =
                    expert * columns * reduction + column * reduction + reduction_index;
                sum += input[input_offset] * weight[weight_offset];
            }
            if mul_routed_weight {
                sum *= routed_weight[slot];
            }
            out[slot * columns + column] = sum;
        }
    }
    out
}

pub fn fused_moe_f8e4m3_block_scaled_f32(
    input: &[u8],
    weight: &[u8],
    input_scales: &[f32],
    weight_scales: &[f32],
    routed_weight: &[f32],
    topk_ids: &[i32],
    tokens: usize,
    top_k: usize,
    _experts: usize,
    columns: usize,
    reduction: usize,
    group_n: usize,
    group_k: usize,
    mul_routed_weight: bool,
) -> Vec<f32> {
    let k_groups = ceil_div(reduction, group_k);
    let n_groups = ceil_div(columns, group_n);
    let mut out = vec![0.0f32; tokens * top_k * columns];
    for slot in 0..tokens * top_k {
        let token = slot / top_k;
        let expert = topk_ids[slot] as usize;
        for column in 0..columns {
            let mut sum = 0.0f32;
            for reduction_index in 0..reduction {
                let k_group = reduction_index / group_k;
                let input_offset = token * reduction + reduction_index;
                let weight_offset =
                    expert * columns * reduction + column * reduction + reduction_index;
                let input_scale = input_scales[token * k_groups + k_group];
                let weight_scale = weight_scales
                    [expert * n_groups * k_groups + (column / group_n) * k_groups + k_group];
                sum += f8e4m3_value(input[input_offset])
                    * f8e4m3_value(weight[weight_offset])
                    * input_scale
                    * weight_scale;
            }
            if mul_routed_weight {
                sum *= routed_weight[slot];
            }
            out[slot * columns + column] = sum;
        }
    }
    out
}

pub fn ceil_div(lhs: usize, rhs: usize) -> usize {
    lhs.div_ceil(rhs)
}

fn f8e4m3_value(value: u8) -> f32 {
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
