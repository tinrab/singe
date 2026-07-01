//! Prefill/decode attention kernels and related attention variants.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlidingWindowAttentionConfig {
    pub batch: usize,
    pub query_len: usize,
    pub key_len: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub query_start: usize,
    pub key_start: usize,
    pub window: usize,
    pub query_batch_stride: usize,
    pub key_batch_stride: usize,
    pub value_batch_stride: usize,
    pub output_batch_stride: usize,
    pub query_sequence_stride: usize,
    pub key_sequence_stride: usize,
    pub value_sequence_stride: usize,
    pub output_sequence_stride: usize,
    pub query_head_stride: usize,
    pub key_head_stride: usize,
    pub value_head_stride: usize,
    pub output_head_stride: usize,
    pub scale: f32,
    pub output_scale: f32,
}

/// Causal left-window attention with contiguous grouped-query head mapping.
///
/// For a query at absolute position `q_abs`, this attends only to keys with
/// `key_abs <= q_abs` and `key_abs + window > q_abs`.
/// This is not the symmetric local attention mask used by transformer-lab's sliding-window components.
pub fn sliding_window_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: SlidingWindowAttentionConfig,
) -> Vec<f32> {
    let q_features = config.heads * config.head_dim;
    let kv_features = config.kv_heads * config.head_dim;
    let query_item_len = config.query_len * q_features;
    let key_item_len = config.key_len * kv_features;
    let output_item_len = config.query_len * q_features;
    let query_group_size = config.heads / config.kv_heads;
    let mut out = vec![0.0; config.batch * output_item_len];
    for batch in 0..config.batch {
        let query_base = batch * query_item_len;
        let key_base = batch * key_item_len;
        let output_base = batch * output_item_len;
        for q_index in 0..config.query_len {
            let q_abs = config.query_start + q_index;
            for head in 0..config.heads {
                let kv_head = head / query_group_size;
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![f32::NEG_INFINITY; config.key_len];
                for (key_index, score_slot) in scores.iter_mut().enumerate() {
                    let key_abs = config.key_start + key_index;
                    if key_abs > q_abs || key_abs + config.window <= q_abs {
                        continue;
                    }
                    let mut score = 0.0f32;
                    for dim in 0..config.head_dim {
                        let q_offset =
                            query_base + q_index * q_features + head * config.head_dim + dim;
                        let k_offset =
                            key_base + key_index * kv_features + kv_head * config.head_dim + dim;
                        score += query[q_offset] * key[k_offset];
                    }
                    score *= config.scale;
                    *score_slot = score;
                    max_score = max_score.max(score);
                }
                if !max_score.is_finite() {
                    continue;
                }
                let mut denom = 0.0f32;
                for score in &scores {
                    if score.is_finite() {
                        denom += (*score - max_score).exp();
                    }
                }
                for dim in 0..config.head_dim {
                    let mut sum = 0.0f32;
                    for (key_index, score) in scores.iter().copied().enumerate() {
                        if score.is_finite() {
                            let weight = (score - max_score).exp() / denom;
                            let v_offset = key_base
                                + key_index * kv_features
                                + kv_head * config.head_dim
                                + dim;
                            sum += weight * value[v_offset];
                        }
                    }
                    out[output_base + q_index * q_features + head * config.head_dim + dim] =
                        sum * config.output_scale;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_window_attention_f32_uses_contiguous_gqa_groups() {
        let config = SlidingWindowAttentionConfig {
            batch: 1,
            query_len: 1,
            key_len: 1,
            heads: 4,
            kv_heads: 2,
            head_dim: 1,
            query_start: 0,
            key_start: 0,
            window: 1,
            query_batch_stride: 4,
            key_batch_stride: 2,
            value_batch_stride: 2,
            output_batch_stride: 4,
            query_sequence_stride: 4,
            key_sequence_stride: 2,
            value_sequence_stride: 2,
            output_sequence_stride: 4,
            query_head_stride: 1,
            key_head_stride: 1,
            value_head_stride: 1,
            output_head_stride: 1,
            scale: 1.0,
            output_scale: 1.0,
        };
        let query = vec![1.0f32; 4];
        let key = vec![1.0f32, 1.0];
        let value = vec![10.0f32, 20.0];

        let out = sliding_window_attention(&query, &key, &value, config);

        assert_eq!(out, vec![10.0, 10.0, 20.0, 20.0]);
    }

    #[test]
    fn sliding_window_attention_uses_contiguous_gqa_groups() {
        let query = vec![1.0f32; 4];
        let key = vec![1.0f32, 1.0];
        let value = vec![10.0f32, 20.0];

        let out =
            sliding_window_attention_f32(&query, &key, &value, 1, 1, 1, 4, 2, 1, 0, 0, 1, 1.0, 1.0);

        assert_eq!(out, vec![10.0, 10.0, 20.0, 20.0]);
    }

    #[test]
    fn sliding_window_attention_is_causal_left_not_symmetric() {
        let config = SlidingWindowAttentionConfig {
            batch: 1,
            query_len: 1,
            key_len: 3,
            heads: 1,
            kv_heads: 1,
            head_dim: 1,
            query_start: 1,
            key_start: 0,
            window: 1,
            query_batch_stride: 1,
            key_batch_stride: 3,
            value_batch_stride: 3,
            output_batch_stride: 1,
            query_sequence_stride: 1,
            key_sequence_stride: 1,
            value_sequence_stride: 1,
            output_sequence_stride: 1,
            query_head_stride: 1,
            key_head_stride: 1,
            value_head_stride: 1,
            output_head_stride: 1,
            scale: 1.0,
            output_scale: 1.0,
        };
        let query = vec![1.0f32];
        let key = vec![1.0f32, 1.0, 1.0];
        let value = vec![10.0f32, 20.0, 30.0];

        let out = sliding_window_attention(&query, &key, &value, config);

        assert_eq!(out, vec![20.0]);
    }
}

pub fn fused_neighborhood_attention_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    kernel_size: usize,
    dilation: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * heads * seq_len * head_dim];
    let half_kernel = kernel_size / 2;
    let radius = half_kernel * dilation;
    for batch_index in 0..batch {
        for head in 0..heads {
            let batch_head_base = ((batch_index * heads + head) * seq_len) * head_dim;
            for query_index in 0..seq_len {
                let lower = query_index.saturating_sub(radius);
                let upper = (query_index + radius).min(seq_len - 1);
                let mut scores = Vec::new();
                for key_index in lower..=upper {
                    if key_index.abs_diff(query_index) % dilation != 0 {
                        continue;
                    }
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        let query_offset = batch_head_base + query_index * head_dim + dim;
                        let key_offset = batch_head_base + key_index * head_dim + dim;
                        score += query[query_offset] * key[key_offset];
                    }
                    scores.push((key_index, score * scale));
                }
                let max_score = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denom = scores
                    .iter()
                    .map(|(_, score)| (*score - max_score).exp())
                    .sum::<f32>();
                for dim in 0..head_dim {
                    let mut sum = 0.0f32;
                    for (key_index, score) in scores.iter().copied() {
                        let weight = (score - max_score).exp() / denom;
                        let value_offset = batch_head_base + key_index * head_dim + dim;
                        sum += weight * value[value_offset];
                    }
                    out[batch_head_base + query_index * head_dim + dim] = sum;
                }
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HcaConfig {
    pub batch: usize,
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub compression_block: usize,
    pub groups: usize,
    pub group_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CsaCompressConfig {
    pub batch: usize,
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub head_dim: usize,
    pub compression_block: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CsaLightningIndexerConfig {
    pub batch: usize,
    pub query_len: usize,
    pub blocks: usize,
    pub index_heads: usize,
    pub index_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CsaTopkSelectorConfig {
    pub batch: usize,
    pub query_len: usize,
    pub blocks: usize,
    pub head_dim: usize,
    pub top_k: usize,
    pub compression_block: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CsaSharedMqaConfig {
    pub batch: usize,
    pub query_len: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub kv_len: usize,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MsaBlockMaxConfig {
    pub batch: usize,
    pub rows: usize,
    pub key_len: usize,
    pub block_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MsaSelectedTokenPositionsConfig {
    pub batch: usize,
    pub rows: usize,
    pub selected_blocks: usize,
    pub block_size: usize,
    pub seq_len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MsaSelectTopkBlocksConfig {
    pub batch: usize,
    pub rows: usize,
    pub blocks: usize,
    pub top_k: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MsaGatheredGqaConfig {
    pub batch: usize,
    pub query_len: usize,
    pub key_len: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub selected_keys: usize,
    pub scale: f32,
}

pub fn minimax_sparse_attention_block_max_f32(
    scores: &[f32],
    config: MsaBlockMaxConfig,
) -> Vec<f32> {
    let blocks = config.key_len.div_ceil(config.block_size);
    let mut out = vec![f32::NEG_INFINITY; config.batch * config.rows * blocks];
    for batch_index in 0..config.batch {
        for row in 0..config.rows {
            for block in 0..blocks {
                let start = block * config.block_size;
                let end = (start + config.block_size).min(config.key_len);
                let mut max_score = f32::NEG_INFINITY;
                for key_index in start..end {
                    let offset = (batch_index * config.rows + row) * config.key_len + key_index;
                    max_score = max_score.max(scores[offset]);
                }
                out[(batch_index * config.rows + row) * blocks + block] = max_score;
            }
        }
    }
    out
}

pub fn minimax_sparse_attention_gathered_gqa_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    positions: &[i32],
    valid_mask: &[i32],
    config: MsaGatheredGqaConfig,
) -> Vec<f32> {
    let mut out = vec![0.0f32; config.batch * config.query_len * config.heads * config.head_dim];
    let group_size = config.heads / config.kv_heads;
    for batch_index in 0..config.batch {
        for query_index in 0..config.query_len {
            for head in 0..config.heads {
                let kv_head = head / group_size;
                let sparse_base = ((batch_index * config.kv_heads + kv_head) * config.query_len
                    + query_index)
                    * config.selected_keys;
                let mut max_score = f32::NEG_INFINITY;
                let mut any_valid = false;
                for selected in 0..config.selected_keys {
                    if valid_mask[sparse_base + selected] == 0 {
                        continue;
                    }
                    any_valid = true;
                    let key_index = positions[sparse_base + selected] as usize;
                    let mut score = 0.0f32;
                    for dim in 0..config.head_dim {
                        let query_offset =
                            ((batch_index * config.query_len + query_index) * config.heads + head)
                                * config.head_dim
                                + dim;
                        let key_offset = ((batch_index * config.kv_heads + kv_head)
                            * config.key_len
                            + key_index)
                            * config.head_dim
                            + dim;
                        score += query[query_offset] * key[key_offset];
                    }
                    score *= config.scale;
                    max_score = max_score.max(score);
                }
                if !any_valid {
                    continue;
                }
                let mut denom = 0.0f32;
                for selected in 0..config.selected_keys {
                    if valid_mask[sparse_base + selected] == 0 {
                        continue;
                    }
                    let key_index = positions[sparse_base + selected] as usize;
                    let mut score = 0.0f32;
                    for dim in 0..config.head_dim {
                        let query_offset =
                            ((batch_index * config.query_len + query_index) * config.heads + head)
                                * config.head_dim
                                + dim;
                        let key_offset = ((batch_index * config.kv_heads + kv_head)
                            * config.key_len
                            + key_index)
                            * config.head_dim
                            + dim;
                        score += query[query_offset] * key[key_offset];
                    }
                    denom += (score * config.scale - max_score).exp();
                }
                for dim in 0..config.head_dim {
                    let mut sum = 0.0f32;
                    for selected in 0..config.selected_keys {
                        if valid_mask[sparse_base + selected] == 0 {
                            continue;
                        }
                        let key_index = positions[sparse_base + selected] as usize;
                        let mut score = 0.0f32;
                        for score_dim in 0..config.head_dim {
                            let query_offset = ((batch_index * config.query_len + query_index)
                                * config.heads
                                + head)
                                * config.head_dim
                                + score_dim;
                            let key_offset = ((batch_index * config.kv_heads + kv_head)
                                * config.key_len
                                + key_index)
                                * config.head_dim
                                + score_dim;
                            score += query[query_offset] * key[key_offset];
                        }
                        let weight = (score * config.scale - max_score).exp() / denom;
                        let value_offset = ((batch_index * config.kv_heads + kv_head)
                            * config.key_len
                            + key_index)
                            * config.head_dim
                            + dim;
                        sum += weight * value[value_offset];
                    }
                    let output_offset =
                        ((batch_index * config.query_len + query_index) * config.heads + head)
                            * config.head_dim
                            + dim;
                    out[output_offset] = sum;
                }
            }
        }
    }
    out
}

pub fn minimax_sparse_attention_select_topk_blocks_f32(
    block_scores: &[f32],
    local_blocks: &[i32],
    config: MsaSelectTopkBlocksConfig,
) -> Vec<i32> {
    let mut out = vec![-1i32; config.batch * config.rows * config.top_k];
    let k_eff = config.top_k.min(config.blocks);
    for batch_index in 0..config.batch {
        for row in 0..config.rows {
            let local = local_blocks[batch_index * config.rows + row]
                .clamp(0, config.blocks as i32 - 1) as usize;
            let output_base = (batch_index * config.rows + row) * config.top_k;
            if k_eff == 0 {
                continue;
            }
            out[output_base] = local as i32;
            for slot in 1..k_eff {
                let rank_target = slot - 1;
                let mut selected = -1i32;
                for candidate in 0..config.blocks {
                    if candidate == local {
                        continue;
                    }
                    let candidate_score =
                        block_scores[(batch_index * config.rows + row) * config.blocks + candidate];
                    let mut rank = 0usize;
                    for other in 0..config.blocks {
                        if other == local || other == candidate {
                            continue;
                        }
                        let other_score =
                            block_scores[(batch_index * config.rows + row) * config.blocks + other];
                        if other_score > candidate_score
                            || (other_score == candidate_score && other < candidate)
                        {
                            rank += 1;
                        }
                    }
                    if rank == rank_target {
                        selected = candidate as i32;
                        break;
                    }
                }
                out[output_base + slot] = selected;
            }
        }
    }
    out
}

pub fn minimax_sparse_attention_selected_token_positions_i32(
    selected: &[i32],
    query_positions: &[i32],
    config: MsaSelectedTokenPositionsConfig,
) -> (Vec<i32>, Vec<i32>) {
    let expanded = config.selected_blocks * config.block_size;
    let mut positions = vec![0i32; config.batch * config.rows * expanded];
    let mut valid = vec![0i32; config.batch * config.rows * expanded];
    for batch_index in 0..config.batch {
        for row in 0..config.rows {
            let query_position = query_positions[batch_index * config.rows + row];
            for selected_block in 0..config.selected_blocks {
                let block = selected
                    [(batch_index * config.rows + row) * config.selected_blocks + selected_block];
                for local_key in 0..config.block_size {
                    let output_offset = (batch_index * config.rows + row) * expanded
                        + selected_block * config.block_size
                        + local_key;
                    let raw_position = block * config.block_size as i32 + local_key as i32;
                    let clamped_position = raw_position.clamp(0, config.seq_len as i32 - 1);
                    positions[output_offset] = clamped_position;
                    valid[output_offset] = i32::from(
                        raw_position >= 0
                            && raw_position < config.seq_len as i32
                            && raw_position <= query_position,
                    );
                }
            }
        }
    }
    (positions, valid)
}

pub fn compressed_sparse_attention_compress_f32(
    hidden: &[f32],
    weight_a_kv: &[f32],
    weight_b_kv: &[f32],
    weight_a_z: &[f32],
    weight_b_z: &[f32],
    bias_a: &[f32],
    bias_b: &[f32],
    config: CsaCompressConfig,
) -> Vec<f32> {
    let blocks = config.seq_len / config.compression_block;
    let mut out = vec![0.0f32; config.batch * blocks * config.head_dim];
    for batch_index in 0..config.batch {
        for block in 0..blocks {
            for dim in 0..config.head_dim {
                let mut logits = vec![0.0f32; config.compression_block * 2];
                let mut values = vec![0.0f32; config.compression_block * 2];
                let mut max_logit = f32::NEG_INFINITY;
                for row in 0..config.compression_block {
                    let token = block * config.compression_block + row;
                    let hidden_base = (batch_index * config.seq_len + token) * config.hidden_dim;
                    let mut value = 0.0f32;
                    let mut logit = bias_a[row * config.head_dim + dim];
                    for hidden_dim in 0..config.hidden_dim {
                        let hidden_value = hidden[hidden_base + hidden_dim];
                        value += hidden_value * weight_a_kv[hidden_dim * config.head_dim + dim];
                        logit += hidden_value * weight_a_z[hidden_dim * config.head_dim + dim];
                    }
                    logits[row] = logit;
                    values[row] = value;
                    max_logit = max_logit.max(logit);
                }
                for row in 0..config.compression_block {
                    let slot = config.compression_block + row;
                    if block == 0 {
                        logits[slot] = f32::NEG_INFINITY;
                        values[slot] = 0.0;
                        continue;
                    }
                    let token = (block - 1) * config.compression_block + row;
                    let hidden_base = (batch_index * config.seq_len + token) * config.hidden_dim;
                    let mut value = 0.0f32;
                    let mut logit = bias_b[row * config.head_dim + dim];
                    for hidden_dim in 0..config.hidden_dim {
                        let hidden_value = hidden[hidden_base + hidden_dim];
                        value += hidden_value * weight_b_kv[hidden_dim * config.head_dim + dim];
                        logit += hidden_value * weight_b_z[hidden_dim * config.head_dim + dim];
                    }
                    logits[slot] = logit;
                    values[slot] = value;
                    max_logit = max_logit.max(logit);
                }

                let mut denom = 0.0f32;
                for logit in &logits {
                    denom += (*logit - max_logit).exp();
                }
                let mut sum = 0.0f32;
                for row in 0..config.compression_block * 2 {
                    sum += ((logits[row] - max_logit).exp() / denom) * values[row];
                }
                out[(batch_index * blocks + block) * config.head_dim + dim] = sum;
            }
        }
    }
    out
}

pub fn compressed_sparse_attention_lightning_indexer_f32(
    indexer_query: &[f32],
    indexer_key: &[f32],
    indexer_weight: &[f32],
    config: CsaLightningIndexerConfig,
) -> Vec<f32> {
    let mut out = vec![0.0f32; config.batch * config.query_len * config.blocks];
    for batch_index in 0..config.batch {
        for query_index in 0..config.query_len {
            for block in 0..config.blocks {
                let mut score = 0.0f32;
                for index_head in 0..config.index_heads {
                    let mut inner = 0.0f32;
                    for dim in 0..config.index_dim {
                        let query_offset = (((batch_index * config.query_len + query_index)
                            * config.index_heads
                            + index_head)
                            * config.index_dim)
                            + dim;
                        let key_offset =
                            (batch_index * config.blocks + block) * config.index_dim + dim;
                        inner += indexer_query[query_offset] * indexer_key[key_offset];
                    }
                    let weight_offset = (batch_index * config.query_len + query_index)
                        * config.index_heads
                        + index_head;
                    score += indexer_weight[weight_offset] * inner.max(0.0);
                }
                out[(batch_index * config.query_len + query_index) * config.blocks + block] = score;
            }
        }
    }
    out
}

pub fn compressed_sparse_attention_topk_selector_f32(
    scores: &[f32],
    compressed_kv: &[f32],
    query_positions: &[i32],
    config: CsaTopkSelectorConfig,
) -> (Vec<f32>, Vec<i32>) {
    let mut out = vec![0.0f32; config.batch * config.query_len * config.top_k * config.head_dim];
    let mut selected = vec![-1i32; config.batch * config.query_len * config.top_k];
    for batch_index in 0..config.batch {
        for query_index in 0..config.query_len {
            let query_position = query_positions[batch_index * config.query_len + query_index];
            let n_valid =
                ((query_position.max(0) as usize) / config.compression_block).min(config.blocks);
            let k_actual = config.top_k.min(n_valid);
            for slot in 0..k_actual {
                let mut selected_block = -1i32;
                for candidate in 0..n_valid {
                    let candidate_score = scores[(batch_index * config.query_len + query_index)
                        * config.blocks
                        + candidate];
                    let mut rank = 0usize;
                    for other in 0..n_valid {
                        if other == candidate {
                            continue;
                        }
                        let other_score = scores[(batch_index * config.query_len + query_index)
                            * config.blocks
                            + other];
                        if other_score > candidate_score
                            || (other_score == candidate_score && other < candidate)
                        {
                            rank += 1;
                        }
                    }
                    if rank == slot {
                        selected_block = candidate as i32;
                        break;
                    }
                }
                selected[(batch_index * config.query_len + query_index) * config.top_k + slot] =
                    selected_block;
                for dim in 0..config.head_dim {
                    let output_offset =
                        (((batch_index * config.query_len + query_index) * config.top_k + slot)
                            * config.head_dim)
                            + dim;
                    let source_offset = (batch_index * config.blocks + selected_block as usize)
                        * config.head_dim
                        + dim;
                    out[output_offset] = compressed_kv[source_offset];
                }
            }
        }
    }
    (out, selected)
}

pub fn compressed_sparse_attention_shared_mqa_f32(
    query: &[f32],
    kv_entries: &[f32],
    valid_mask: &[i32],
    sink: &[f32],
    config: CsaSharedMqaConfig,
) -> Vec<f32> {
    let mut out = vec![0.0f32; config.batch * config.query_len * config.heads * config.head_dim];
    for batch_index in 0..config.batch {
        for query_index in 0..config.query_len {
            let mask_base = (batch_index * config.query_len + query_index) * config.kv_len;
            for (head, sink_value) in sink.iter().copied().enumerate().take(config.heads) {
                let mut max_score = sink_value;
                for key_index in 0..config.kv_len {
                    if valid_mask[mask_base + key_index] == 0 {
                        continue;
                    }
                    let mut score = 0.0f32;
                    for dim in 0..config.head_dim {
                        let query_offset =
                            ((batch_index * config.query_len + query_index) * config.heads + head)
                                * config.head_dim
                                + dim;
                        let key_offset = ((batch_index * config.query_len + query_index)
                            * config.kv_len
                            + key_index)
                            * config.head_dim
                            + dim;
                        score += query[query_offset] * kv_entries[key_offset];
                    }
                    max_score = max_score.max(score * config.scale);
                }
                let mut denom = (sink_value - max_score).exp();
                for key_index in 0..config.kv_len {
                    if valid_mask[mask_base + key_index] == 0 {
                        continue;
                    }
                    let mut score = 0.0f32;
                    for dim in 0..config.head_dim {
                        let query_offset =
                            ((batch_index * config.query_len + query_index) * config.heads + head)
                                * config.head_dim
                                + dim;
                        let key_offset = ((batch_index * config.query_len + query_index)
                            * config.kv_len
                            + key_index)
                            * config.head_dim
                            + dim;
                        score += query[query_offset] * kv_entries[key_offset];
                    }
                    denom += (score * config.scale - max_score).exp();
                }
                for dim in 0..config.head_dim {
                    let mut sum = 0.0f32;
                    for key_index in 0..config.kv_len {
                        if valid_mask[mask_base + key_index] == 0 {
                            continue;
                        }
                        let mut score = 0.0f32;
                        for score_dim in 0..config.head_dim {
                            let query_offset = ((batch_index * config.query_len + query_index)
                                * config.heads
                                + head)
                                * config.head_dim
                                + score_dim;
                            let key_offset = ((batch_index * config.query_len + query_index)
                                * config.kv_len
                                + key_index)
                                * config.head_dim
                                + score_dim;
                            score += query[query_offset] * kv_entries[key_offset];
                        }
                        let weight = (score * config.scale - max_score).exp() / denom;
                        let value_offset = ((batch_index * config.query_len + query_index)
                            * config.kv_len
                            + key_index)
                            * config.head_dim
                            + dim;
                        sum += weight * kv_entries[value_offset];
                    }
                    let output_offset =
                        ((batch_index * config.query_len + query_index) * config.heads + head)
                            * config.head_dim
                            + dim;
                    out[output_offset] = sum;
                }
            }
        }
    }
    out
}

pub fn heavily_compressed_attention_compress_f32(
    hidden: &[f32],
    weight_kv: &[f32],
    weight_z: &[f32],
    bias: &[f32],
    config: HcaConfig,
) -> Vec<f32> {
    let blocks = config.seq_len / config.compression_block;
    let mut out = vec![0.0f32; config.batch * blocks * config.head_dim];
    for batch_index in 0..config.batch {
        for block in 0..blocks {
            for dim in 0..config.head_dim {
                let mut logits = vec![0.0f32; config.compression_block];
                let mut values = vec![0.0f32; config.compression_block];
                let mut max_logit = f32::NEG_INFINITY;
                for row in 0..config.compression_block {
                    let token = block * config.compression_block + row;
                    let hidden_base = (batch_index * config.seq_len + token) * config.hidden_dim;
                    let mut value = 0.0f32;
                    let mut logit = bias[row * config.head_dim + dim];
                    for hidden_dim in 0..config.hidden_dim {
                        let hidden_value = hidden[hidden_base + hidden_dim];
                        value += hidden_value * weight_kv[hidden_dim * config.head_dim + dim];
                        logit += hidden_value * weight_z[hidden_dim * config.head_dim + dim];
                    }
                    logits[row] = logit;
                    values[row] = value;
                    max_logit = max_logit.max(logit);
                }

                let mut denom = 0.0f32;
                for logit in &logits {
                    denom += (*logit - max_logit).exp();
                }
                let mut sum = 0.0f32;
                for row in 0..config.compression_block {
                    sum += ((logits[row] - max_logit).exp() / denom) * values[row];
                }
                out[(batch_index * blocks + block) * config.head_dim + dim] = sum;
            }
        }
    }
    out
}

pub fn heavily_compressed_attention_f32(
    query: &[f32],
    compressed_kv: &[f32],
    weight_group: &[f32],
    weight_final: &[f32],
    config: HcaConfig,
) -> Vec<f32> {
    let blocks = config.seq_len / config.compression_block;
    let heads_per_group = config.heads / config.groups;
    let mut out = vec![0.0f32; config.batch * config.seq_len * config.hidden_dim];

    for batch_index in 0..config.batch {
        for query_index in 0..config.seq_len {
            let visible_blocks = (query_index / config.compression_block).min(blocks);
            let mut heads = vec![0.0f32; config.heads * config.head_dim];
            if visible_blocks > 0 {
                for head in 0..config.heads {
                    let mut scores = vec![0.0f32; visible_blocks];
                    let mut max_score = f32::NEG_INFINITY;
                    for (block, score_slot) in scores.iter_mut().enumerate().take(visible_blocks) {
                        let mut score = 0.0f32;
                        for dim in 0..config.head_dim {
                            let query_offset = ((batch_index * config.seq_len + query_index)
                                * config.heads
                                + head)
                                * config.head_dim
                                + dim;
                            let kv_offset = (batch_index * blocks + block) * config.head_dim + dim;
                            score += query[query_offset] * compressed_kv[kv_offset];
                        }
                        score /= (config.head_dim as f32).sqrt();
                        *score_slot = score;
                        max_score = max_score.max(score);
                    }
                    let denom = scores
                        .iter()
                        .map(|score| (*score - max_score).exp())
                        .sum::<f32>();
                    for dim in 0..config.head_dim {
                        let mut sum = 0.0f32;
                        for (block, score) in scores.iter().copied().enumerate() {
                            let weight = (score - max_score).exp() / denom;
                            let kv_offset = (batch_index * blocks + block) * config.head_dim + dim;
                            sum += weight * compressed_kv[kv_offset];
                        }
                        heads[head * config.head_dim + dim] = sum;
                    }
                }
            }

            let mut inter = vec![0.0f32; config.groups * config.group_dim];
            for group in 0..config.groups {
                for group_out in 0..config.group_dim {
                    let mut sum = 0.0f32;
                    for local_head in 0..heads_per_group {
                        for dim in 0..config.head_dim {
                            let flat = local_head * config.head_dim + dim;
                            let head = group * heads_per_group + local_head;
                            let weight_offset = (group * heads_per_group * config.head_dim + flat)
                                * config.group_dim
                                + group_out;
                            sum +=
                                heads[head * config.head_dim + dim] * weight_group[weight_offset];
                        }
                    }
                    inter[group * config.group_dim + group_out] = sum;
                }
            }

            for output_dim in 0..config.hidden_dim {
                let mut sum = 0.0f32;
                for inter_dim in 0..config.groups * config.group_dim {
                    sum +=
                        inter[inter_dim] * weight_final[inter_dim * config.hidden_dim + output_dim];
                }
                out[(batch_index * config.seq_len + query_index) * config.hidden_dim
                    + output_dim] = sum;
            }
        }
    }
    out
}

pub fn multi_token_attention_f32(
    scores: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    channels_in: usize,
    channels_out: usize,
    seq_len: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    sparse: bool,
) -> Vec<f32> {
    let output_h = conv_output_len(seq_len, kernel_h, stride_h, padding_h, dilation_h);
    let output_w = conv_output_len(seq_len, kernel_w, stride_w, padding_w, dilation_w);
    assert_eq!(output_h, output_w);
    let mut probabilities = vec![0.0f32; batch * channels_in * seq_len * seq_len];
    for batch_index in 0..batch {
        for channel in 0..channels_in {
            for row in 0..seq_len {
                let base = ((batch_index * channels_in + channel) * seq_len + row) * seq_len;
                if sparse {
                    let row_scores = &scores[base..base + row + 1];
                    let tau = sparsemax_tau(row_scores);
                    for col in 0..=row {
                        probabilities[base + col] = (scores[base + col] - tau).max(0.0);
                    }
                } else {
                    let max_score = (0..=row)
                        .map(|col| scores[base + col])
                        .fold(f32::NEG_INFINITY, f32::max);
                    let denom = (0..=row)
                        .map(|col| (scores[base + col] - max_score).exp())
                        .sum::<f32>();
                    for col in 0..=row {
                        probabilities[base + col] = (scores[base + col] - max_score).exp() / denom;
                    }
                }
            }
        }
    }

    let channels_in_per_group = channels_in / groups;
    let channels_out_per_group = channels_out / groups;
    let mut out = vec![0.0f32; batch * channels_out * output_h * output_w];
    for batch_index in 0..batch {
        for output_channel in 0..channels_out {
            let group = output_channel / channels_out_per_group;
            for output_row in 0..output_h {
                for output_col in 0..output_w {
                    let out_offset = ((batch_index * channels_out + output_channel) * output_h
                        + output_row)
                        * output_w
                        + output_col;
                    if output_col > output_row {
                        continue;
                    }
                    let mut sum = bias.map_or(0.0f32, |bias| bias[output_channel]);
                    for local_input_channel in 0..channels_in_per_group {
                        let input_channel = group * channels_in_per_group + local_input_channel;
                        for kernel_row in 0..kernel_h {
                            let input_row = output_row * stride_h + kernel_row * dilation_h;
                            let Some(input_row) = input_row.checked_sub(padding_h) else {
                                continue;
                            };
                            if input_row >= seq_len {
                                continue;
                            }
                            for kernel_col in 0..kernel_w {
                                let input_col = output_col * stride_w + kernel_col * dilation_w;
                                let Some(input_col) = input_col.checked_sub(padding_w) else {
                                    continue;
                                };
                                if input_col >= seq_len || input_col > input_row {
                                    continue;
                                }
                                let prob_offset = ((batch_index * channels_in + input_channel)
                                    * seq_len
                                    + input_row)
                                    * seq_len
                                    + input_col;
                                let weight_offset = ((output_channel * channels_in_per_group
                                    + local_input_channel)
                                    * kernel_h
                                    + kernel_row)
                                    * kernel_w
                                    + kernel_col;
                                sum += probabilities[prob_offset] * weight[weight_offset];
                            }
                        }
                    }
                    out[out_offset] = sum;
                }
            }
        }
    }
    out
}

pub fn sparsemax_tau(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.partial_cmp(left).unwrap());
    let mut sum = 0.0f32;
    let mut support = 0usize;
    for (index, value) in sorted.iter().copied().enumerate() {
        sum += value;
        let threshold = (sum - 1.0) / (index + 1) as f32;
        if value > threshold {
            support = index + 1;
        }
    }
    let support_sum = sorted.iter().take(support).sum::<f32>();
    (support_sum - 1.0) / support as f32
}

pub fn conv_output_len(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let effective_kernel = dilation * (kernel - 1) + 1;
    let padded = input + padding * 2;
    if padded < effective_kernel {
        return 0;
    }
    (padded - effective_kernel) / stride + 1
}

pub fn sliding_window_attention_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    query_start: usize,
    key_start: usize,
    window: usize,
    scale: f32,
    output_scale: f32,
) -> Vec<f32> {
    let q_features = heads * head_dim;
    let kv_features = kv_heads * head_dim;
    let query_group_size = heads / kv_heads;
    let mut out = vec![0.0f32; batch * query_len * q_features];
    for batch_index in 0..batch {
        for query_index in 0..query_len {
            let query_abs = query_start + query_index;
            for head in 0..heads {
                let kv_head = head / query_group_size;
                let mut scores = Vec::new();
                for key_index in 0..key_len {
                    let key_abs = key_start + key_index;
                    if key_abs > query_abs || key_abs + window <= query_abs {
                        continue;
                    }
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        let query_offset = batch_index * query_len * q_features
                            + query_index * q_features
                            + head * head_dim
                            + dim;
                        let key_offset = batch_index * key_len * kv_features
                            + key_index * kv_features
                            + kv_head * head_dim
                            + dim;
                        score += query[query_offset] * key[key_offset];
                    }
                    scores.push((key_index, score * scale));
                }
                if scores.is_empty() {
                    continue;
                }
                let max_score = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denom = scores
                    .iter()
                    .map(|(_, score)| (*score - max_score).exp())
                    .sum::<f32>();
                for dim in 0..head_dim {
                    let mut sum = 0.0f32;
                    for &(key_index, score) in &scores {
                        let weight = (score - max_score).exp() / denom;
                        let value_offset = batch_index * key_len * kv_features
                            + key_index * kv_features
                            + kv_head * head_dim
                            + dim;
                        sum += weight * value[value_offset];
                    }
                    let output_offset = batch_index * query_len * q_features
                        + query_index * q_features
                        + head * head_dim
                        + dim;
                    out[output_offset] = sum * output_scale;
                }
            }
        }
    }
    out
}

pub fn paged_kv_decode_attention_f32(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    actual_seq_lens: &[i32],
    block_table: &[u32],
    batch: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    block_table_batch_stride: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * heads * head_dim];
    let query_group_size = heads / kv_heads;
    let cache_token_stride = kv_heads * head_dim;
    let cache_block_stride = block_size * cache_token_stride;
    for batch_index in 0..batch {
        let seq_len = actual_seq_lens[batch_index] as usize;
        for head in 0..heads {
            let kv_head = head / query_group_size;
            let mut scores = Vec::with_capacity(seq_len);
            for key_index in 0..seq_len {
                let block = block_table
                    [batch_index * block_table_batch_stride + key_index / block_size]
                    as usize;
                let block_token = key_index % block_size;
                let mut score = 0.0f32;
                for dim in 0..head_dim {
                    let query_offset = (batch_index * heads + head) * head_dim + dim;
                    let key_offset = block * cache_block_stride
                        + block_token * cache_token_stride
                        + kv_head * head_dim
                        + dim;
                    score += query[query_offset] * key_cache[key_offset];
                }
                scores.push(score * scale);
            }
            if scores.is_empty() {
                continue;
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom = scores
                .iter()
                .copied()
                .map(|score| (score - max_score).exp())
                .sum::<f32>();
            for dim in 0..head_dim {
                let mut sum = 0.0f32;
                for (key_index, score) in scores.iter().copied().enumerate() {
                    let block = block_table
                        [batch_index * block_table_batch_stride + key_index / block_size]
                        as usize;
                    let block_token = key_index % block_size;
                    let value_offset = block * cache_block_stride
                        + block_token * cache_token_stride
                        + kv_head * head_dim
                        + dim;
                    sum += (score - max_score).exp() / denom * value_cache[value_offset];
                }
                out[(batch_index * heads + head) * head_dim + dim] = sum;
            }
        }
    }
    out
}

pub fn paged_kv_prefill_attention_f32(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    actual_seq_lens_q: &[i32],
    actual_seq_lens_kv: &[i32],
    actual_seq_offsets: &[i32],
    block_table: &[u32],
    batch: usize,
    total_query_tokens: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    block_table_batch_stride: usize,
    causal: bool,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; total_query_tokens * heads * head_dim];
    let mut lse = vec![0.0f32; total_query_tokens * heads];
    let query_group_size = heads / kv_heads;
    let features = heads * head_dim;
    let cache_token_stride = kv_heads * head_dim;
    let cache_block_stride = block_size * cache_token_stride;
    for batch_index in 0..batch {
        let seq_len_q = actual_seq_lens_q[batch_index] as usize;
        let seq_len_kv = actual_seq_lens_kv[batch_index] as usize;
        let seq_offset = actual_seq_offsets[batch_index] as usize;
        for query_index in 0..seq_len_q {
            let global_token = seq_offset + query_index;
            for head in 0..heads {
                let kv_head = head / query_group_size;
                let mut scores = Vec::new();
                for key_index in 0..seq_len_kv {
                    if causal && key_index > query_index {
                        continue;
                    }
                    let block = block_table
                        [batch_index * block_table_batch_stride + key_index / block_size]
                        as usize;
                    let block_token = key_index % block_size;
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        let query_offset = global_token * features + head * head_dim + dim;
                        let key_offset = block * cache_block_stride
                            + block_token * cache_token_stride
                            + kv_head * head_dim
                            + dim;
                        score += query[query_offset] * key_cache[key_offset];
                    }
                    scores.push((key_index, score * scale));
                }
                if scores.is_empty() {
                    lse[global_token * heads + head] = -1.0e20;
                    continue;
                }
                let max_score = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denom = scores
                    .iter()
                    .map(|(_, score)| (*score - max_score).exp())
                    .sum::<f32>();
                lse[global_token * heads + head] = max_score + denom.ln();
                for dim in 0..head_dim {
                    let mut sum = 0.0f32;
                    for &(key_index, score) in &scores {
                        let block = block_table
                            [batch_index * block_table_batch_stride + key_index / block_size]
                            as usize;
                        let block_token = key_index % block_size;
                        let value_offset = block * cache_block_stride
                            + block_token * cache_token_stride
                            + kv_head * head_dim
                            + dim;
                        sum += (score - max_score).exp() / denom * value_cache[value_offset];
                    }
                    out[global_token * features + head * head_dim + dim] = sum;
                }
            }
        }
    }
    (out, lse)
}

pub fn ragged_kv_prefill_attention_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    actual_seq_lens_q: &[i32],
    actual_seq_lens_kv: &[i32],
    actual_seq_offsets: &[i32],
    batch: usize,
    total_query_tokens: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; total_query_tokens * heads * head_dim];
    let mut lse = vec![0.0f32; total_query_tokens * heads];
    let query_group_size = heads / kv_heads;
    let features = heads * head_dim;
    let kv_features = kv_heads * head_dim;
    for batch_index in 0..batch {
        let seq_len_q = actual_seq_lens_q[batch_index] as usize;
        let seq_len_kv = actual_seq_lens_kv[batch_index] as usize;
        let seq_offset = actual_seq_offsets[batch_index] as usize;
        for query_index in 0..seq_len_q {
            let global_token = seq_offset + query_index;
            for head in 0..heads {
                let kv_head = head / query_group_size;
                let mut scores = Vec::new();
                for key_index in 0..seq_len_kv {
                    if causal && key_index > query_index {
                        continue;
                    }
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        let query_offset = global_token * features + head * head_dim + dim;
                        let key_offset =
                            (seq_offset + key_index) * kv_features + kv_head * head_dim + dim;
                        score += query[query_offset] * key[key_offset];
                    }
                    scores.push((key_index, score * scale));
                }
                if scores.is_empty() {
                    lse[global_token * heads + head] = -1.0e20;
                    continue;
                }
                let max_score = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denom = scores
                    .iter()
                    .map(|(_, score)| (*score - max_score).exp())
                    .sum::<f32>();
                lse[global_token * heads + head] = max_score + denom.ln();
                for dim in 0..head_dim {
                    let mut sum = 0.0f32;
                    for &(key_index, score) in &scores {
                        let value_offset =
                            (seq_offset + key_index) * kv_features + kv_head * head_dim + dim;
                        sum += (score - max_score).exp() / denom * value[value_offset];
                    }
                    out[global_token * features + head * head_dim + dim] = sum;
                }
            }
        }
    }
    (out, lse)
}

pub fn mla_decode_lse_f32(
    query: &[f32],
    query_pe: &[f32],
    key_value: &[f32],
    key_pe: &[f32],
    batch: usize,
    key_len: usize,
    heads: usize,
    head_dim: usize,
    pe_dim: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let q_features = heads * head_dim;
    let qpe_features = heads * pe_dim;
    let mut out = vec![0.0f32; batch * q_features];
    let mut lse = vec![0.0f32; batch * heads];
    for batch_index in 0..batch {
        for head in 0..heads {
            let mut scores = Vec::new();
            for key_index in 0..key_len {
                let mut score = 0.0f32;
                for dim in 0..head_dim {
                    let query_offset = batch_index * q_features + head * head_dim + dim;
                    let key_offset = batch_index * key_len * head_dim + key_index * head_dim + dim;
                    score += query[query_offset] * key_value[key_offset];
                }
                for dim in 0..pe_dim {
                    let query_offset = batch_index * qpe_features + head * pe_dim + dim;
                    let key_offset = batch_index * key_len * pe_dim + key_index * pe_dim + dim;
                    score += query_pe[query_offset] * key_pe[key_offset];
                }
                scores.push((key_index, score * scale));
            }
            let max_score = scores
                .iter()
                .map(|(_, score)| *score)
                .fold(f32::NEG_INFINITY, f32::max);
            let denom = scores
                .iter()
                .map(|(_, score)| (*score - max_score).exp())
                .sum::<f32>();
            lse[batch_index * heads + head] = max_score + denom.ln();
            for dim in 0..head_dim {
                let mut sum = 0.0f32;
                for &(key_index, score) in &scores {
                    let weight = (score - max_score).exp() / denom;
                    let value_offset =
                        batch_index * key_len * head_dim + key_index * head_dim + dim;
                    sum += weight * key_value[value_offset];
                }
                out[batch_index * q_features + head * head_dim + dim] = sum;
            }
        }
    }
    (out, lse)
}

pub fn paged_mla_decode_attention_f32(
    query: &[f32],
    query_pe: &[f32],
    key_value_cache: &[f32],
    key_pe_cache: &[f32],
    actual_seq_lens: &[i32],
    block_table: &[u32],
    batch: usize,
    heads: usize,
    head_dim: usize,
    pe_dim: usize,
    block_size: usize,
    block_table_batch_stride: usize,
    scale: f32,
    output_scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let q_features = heads * head_dim;
    let qpe_features = heads * pe_dim;
    let cache_block_stride = block_size * head_dim;
    let pe_cache_block_stride = block_size * pe_dim;
    let mut out = vec![0.0f32; batch * q_features];
    let mut lse = vec![0.0f32; batch * heads];
    for batch_index in 0..batch {
        let seq_len = actual_seq_lens[batch_index] as usize;
        for head in 0..heads {
            let mut scores = Vec::with_capacity(seq_len);
            for key_index in 0..seq_len {
                let block = block_table
                    [batch_index * block_table_batch_stride + key_index / block_size]
                    as usize;
                let block_token = key_index % block_size;
                let mut score = 0.0f32;
                for dim in 0..head_dim {
                    let query_offset = batch_index * q_features + head * head_dim + dim;
                    let key_offset = block * cache_block_stride + block_token * head_dim + dim;
                    score += query[query_offset] * key_value_cache[key_offset];
                }
                for dim in 0..pe_dim {
                    let query_offset = batch_index * qpe_features + head * pe_dim + dim;
                    let key_offset = block * pe_cache_block_stride + block_token * pe_dim + dim;
                    score += query_pe[query_offset] * key_pe_cache[key_offset];
                }
                scores.push((key_index, score * scale));
            }
            if scores.is_empty() {
                lse[batch_index * heads + head] = -1.0e20;
                continue;
            }
            let max_score = scores
                .iter()
                .map(|(_, score)| *score)
                .fold(f32::NEG_INFINITY, f32::max);
            let denom = scores
                .iter()
                .map(|(_, score)| (*score - max_score).exp())
                .sum::<f32>();
            lse[batch_index * heads + head] = max_score + denom.ln();
            for dim in 0..head_dim {
                let mut sum = 0.0f32;
                for &(key_index, score) in &scores {
                    let block = block_table
                        [batch_index * block_table_batch_stride + key_index / block_size]
                        as usize;
                    let block_token = key_index % block_size;
                    let value_offset = block * cache_block_stride + block_token * head_dim + dim;
                    sum += (score - max_score).exp() / denom * key_value_cache[value_offset];
                }
                out[batch_index * q_features + head * head_dim + dim] = sum * output_scale;
            }
        }
    }
    (out, lse)
}

pub fn mla_prefill_f32(
    query: &[f32],
    query_pe: &[f32],
    key: &[f32],
    key_pe: &[f32],
    value: &[f32],
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    pe_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let q_features = heads * head_dim;
    let qpe_features = heads * pe_dim;
    let kv_features = kv_heads * head_dim;
    let kpe_features = kv_heads * pe_dim;
    let query_group_size = heads / kv_heads;
    let mut out = vec![0.0f32; batch * query_len * q_features];
    for batch_index in 0..batch {
        for head in 0..heads {
            let kv_head = head / query_group_size;
            for query_index in 0..query_len {
                let mut scores = Vec::new();
                for key_index in 0..key_len {
                    if key_index > query_index {
                        continue;
                    }
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        let query_offset = batch_index * query_len * q_features
                            + query_index * q_features
                            + head * head_dim
                            + dim;
                        let key_offset = batch_index * key_len * kv_features
                            + key_index * kv_features
                            + kv_head * head_dim
                            + dim;
                        score += query[query_offset] * key[key_offset];
                    }
                    for dim in 0..pe_dim {
                        let query_offset = batch_index * query_len * qpe_features
                            + query_index * qpe_features
                            + head * pe_dim
                            + dim;
                        let key_offset = batch_index * key_len * kpe_features
                            + key_index * kpe_features
                            + kv_head * pe_dim
                            + dim;
                        score += query_pe[query_offset] * key_pe[key_offset];
                    }
                    scores.push((key_index, score * scale));
                }
                let max_score = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denom = scores
                    .iter()
                    .map(|(_, score)| (*score - max_score).exp())
                    .sum::<f32>();
                for dim in 0..head_dim {
                    let mut sum = 0.0f32;
                    for &(key_index, score) in &scores {
                        let weight = (score - max_score).exp() / denom;
                        let value_offset = batch_index * key_len * kv_features
                            + key_index * kv_features
                            + kv_head * head_dim
                            + dim;
                        sum += weight * value[value_offset];
                    }
                    let output_offset = batch_index * query_len * q_features
                        + query_index * q_features
                        + head * head_dim
                        + dim;
                    out[output_offset] = sum;
                }
            }
        }
    }
    out
}

pub fn sparse_mla_prefill_f32(
    query: &[f32],
    query_pe: &[f32],
    key: &[f32],
    key_pe: &[f32],
    value: &[f32],
    indices: &[i32],
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    pe_dim: usize,
    topk: usize,
    scale: f32,
) -> Vec<f32> {
    let q_features = heads * head_dim;
    let qpe_features = heads * pe_dim;
    let kv_features = kv_heads * head_dim;
    let query_group_size = heads / kv_heads;
    let mut out = vec![0.0f32; batch * query_len * q_features];
    for batch_index in 0..batch {
        for head in 0..heads {
            let kv_head = head / query_group_size;
            for query_index in 0..query_len {
                let mut scores = Vec::new();
                for topk_index in 0..topk {
                    let index_offset = batch_index * query_len * kv_heads * topk
                        + query_index * kv_heads * topk
                        + kv_head * topk
                        + topk_index;
                    let key_index = indices[index_offset];
                    if key_index < 0 {
                        continue;
                    }
                    let key_index = key_index as usize;
                    if key_index >= key_len || key_index > query_index {
                        continue;
                    }
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        let query_offset = batch_index * query_len * q_features
                            + query_index * q_features
                            + head * head_dim
                            + dim;
                        let key_offset = batch_index * key_len * kv_features
                            + key_index * kv_features
                            + kv_head * head_dim
                            + dim;
                        score += query[query_offset] * key[key_offset];
                    }
                    for dim in 0..pe_dim {
                        let query_offset = batch_index * query_len * qpe_features
                            + query_index * qpe_features
                            + head * pe_dim
                            + dim;
                        let key_offset = batch_index * key_len * pe_dim + key_index * pe_dim + dim;
                        score += query_pe[query_offset] * key_pe[key_offset];
                    }
                    scores.push((key_index, score * scale));
                }
                if scores.is_empty() {
                    continue;
                }
                let max_score = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denom = scores
                    .iter()
                    .map(|(_, score)| (*score - max_score).exp())
                    .sum::<f32>();
                for dim in 0..head_dim {
                    let mut sum = 0.0f32;
                    for &(key_index, score) in &scores {
                        let weight = (score - max_score).exp() / denom;
                        let value_offset = batch_index * key_len * kv_features
                            + key_index * kv_features
                            + kv_head * head_dim
                            + dim;
                        sum += weight * value[value_offset];
                    }
                    let output_offset = batch_index * query_len * q_features
                        + query_index * q_features
                        + head * head_dim
                        + dim;
                    out[output_offset] = sum;
                }
            }
        }
    }
    out
}

pub fn fmha_prefill_lse_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    let q_features = heads * head_dim;
    let kv_features = kv_heads * head_dim;
    let query_group_size = heads / kv_heads;
    let mut out = vec![0.0f32; batch * query_len * q_features];
    let mut lse = vec![0.0f32; batch * heads * query_len];
    for batch_index in 0..batch {
        for head in 0..heads {
            let kv_head = head / query_group_size;
            for query_index in 0..query_len {
                let mut scores = Vec::new();
                for key_index in 0..key_len {
                    if causal && key_index > query_index {
                        continue;
                    }
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        let query_offset = batch_index * query_len * q_features
                            + query_index * q_features
                            + head * head_dim
                            + dim;
                        let key_offset = batch_index * key_len * kv_features
                            + key_index * kv_features
                            + kv_head * head_dim
                            + dim;
                        score += query[query_offset] * key[key_offset];
                    }
                    scores.push((key_index, score * scale));
                }
                let max_score = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denom = scores
                    .iter()
                    .map(|(_, score)| (*score - max_score).exp())
                    .sum::<f32>();
                lse[batch_index * heads * query_len + head * query_len + query_index] =
                    max_score + denom.ln();
                for dim in 0..head_dim {
                    let mut sum = 0.0f32;
                    for &(key_index, score) in &scores {
                        let weight = (score - max_score).exp() / denom;
                        let value_offset = batch_index * key_len * kv_features
                            + key_index * kv_features
                            + kv_head * head_dim
                            + dim;
                        sum += weight * value[value_offset];
                    }
                    let output_offset = batch_index * query_len * q_features
                        + query_index * q_features
                        + head * head_dim
                        + dim;
                    out[output_offset] = sum;
                }
            }
        }
    }
    (out, lse)
}

pub fn softcapped_window_attention_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
    window_size: usize,
    soft_cap: Option<f32>,
) -> Vec<f32> {
    let query_head_stride = query_len * head_dim;
    let key_head_stride = key_len * head_dim;
    let query_group_size = heads / kv_heads;
    let mut out = vec![0.0f32; batch * heads * query_len * head_dim];
    for batch_index in 0..batch {
        for head in 0..heads {
            let kv_head = head / query_group_size;
            for query_index in 0..query_len {
                let mut scores = Vec::new();
                for key_index in 0..key_len {
                    if causal && key_index > query_index {
                        continue;
                    }
                    if window_size > 0
                        && (key_index + window_size < query_index
                            || key_index > query_index + window_size)
                    {
                        continue;
                    }
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        let query_offset = batch_index * heads * query_head_stride
                            + head * query_head_stride
                            + query_index * head_dim
                            + dim;
                        let key_offset = batch_index * kv_heads * key_head_stride
                            + kv_head * key_head_stride
                            + key_index * head_dim
                            + dim;
                        score += query[query_offset] * key[key_offset];
                    }
                    let mut score = score * scale;
                    if let Some(soft_cap) = soft_cap {
                        score = (score / soft_cap).tanh() * soft_cap;
                    }
                    scores.push((key_index, score));
                }
                if scores.is_empty() {
                    continue;
                }
                let max_score = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denom = scores
                    .iter()
                    .map(|(_, score)| (*score - max_score).exp())
                    .sum::<f32>();
                for dim in 0..head_dim {
                    let mut sum = 0.0f32;
                    for &(key_index, score) in &scores {
                        let weight = (score - max_score).exp() / denom;
                        let value_offset = batch_index * kv_heads * key_head_stride
                            + kv_head * key_head_stride
                            + key_index * head_dim
                            + dim;
                        sum += weight * value[value_offset];
                    }
                    let output_offset = batch_index * heads * query_head_stride
                        + head * query_head_stride
                        + query_index * head_dim
                        + dim;
                    out[output_offset] = sum;
                }
            }
        }
    }
    out
}

pub fn softcapped_window_decode_lse_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    window_size: usize,
    soft_cap: Option<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let key_head_stride = key_len * head_dim;
    let query_group_size = heads / kv_heads;
    let mut out = vec![0.0f32; batch * heads * head_dim];
    let mut lse = vec![0.0f32; batch * heads];
    let query_position = key_len - 1;
    for batch_index in 0..batch {
        for head in 0..heads {
            let kv_head = head / query_group_size;
            let mut scores = Vec::new();
            for key_index in 0..key_len {
                if window_size > 0 && key_index + window_size < query_position {
                    continue;
                }
                let mut score = 0.0f32;
                for dim in 0..head_dim {
                    let query_offset = batch_index * heads * head_dim + head * head_dim + dim;
                    let key_offset = batch_index * kv_heads * key_head_stride
                        + kv_head * key_head_stride
                        + key_index * head_dim
                        + dim;
                    score += query[query_offset] * key[key_offset];
                }
                let mut score = score * scale;
                if let Some(soft_cap) = soft_cap {
                    score = (score / soft_cap).tanh() * soft_cap;
                }
                scores.push((key_index, score));
            }
            let max_score = scores
                .iter()
                .map(|(_, score)| *score)
                .fold(f32::NEG_INFINITY, f32::max);
            let denom = scores
                .iter()
                .map(|(_, score)| (*score - max_score).exp())
                .sum::<f32>();
            lse[batch_index * heads + head] = max_score + denom.ln();
            for dim in 0..head_dim {
                let mut sum = 0.0f32;
                for &(key_index, score) in &scores {
                    let weight = (score - max_score).exp() / denom;
                    let value_offset = batch_index * kv_heads * key_head_stride
                        + kv_head * key_head_stride
                        + key_index * head_dim
                        + dim;
                    sum += weight * value[value_offset];
                }
                let output_offset = batch_index * heads * head_dim + head * head_dim + dim;
                out[output_offset] = sum;
            }
        }
    }
    (out, lse)
}

pub fn attention_sink_decode_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sinks: &[f32],
    batch: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    start_q: usize,
    window: usize,
    scale: f32,
) -> Vec<f32> {
    let q_features = heads * head_dim;
    let kv_features = kv_heads * head_dim;
    let query_group_size = heads / kv_heads;
    let mut out = vec![0.0f32; batch * q_features];
    for batch_index in 0..batch {
        for head in 0..heads {
            let kv_head = head / query_group_size;
            let window_start = if window == 0 {
                0
            } else {
                (start_q + 1).saturating_sub(window)
            };
            let mut scores = vec![sinks[head]];
            let mut key_indices = Vec::new();
            for key_index in 0..key_len {
                if key_index > start_q || key_index < window_start {
                    continue;
                }
                let mut score = 0.0f32;
                for dim in 0..head_dim {
                    let query_offset = batch_index * q_features + head * head_dim + dim;
                    let key_offset = batch_index * key_len * kv_features
                        + key_index * kv_features
                        + kv_head * head_dim
                        + dim;
                    score += query[query_offset] * key[key_offset];
                }
                scores.push(score * scale);
                key_indices.push(key_index);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores
                .iter()
                .copied()
                .map(|score| (score - max_score).exp())
                .collect::<Vec<_>>();
            let denom = weights.iter().sum::<f32>();
            for (index, key_index) in key_indices.iter().copied().enumerate() {
                let weight = weights[index + 1] / denom;
                for dim in 0..head_dim {
                    let value_offset = batch_index * key_len * kv_features
                        + key_index * kv_features
                        + kv_head * head_dim
                        + dim;
                    out[batch_index * q_features + head * head_dim + dim] +=
                        weight * value[value_offset];
                }
            }
        }
    }
    out
}

pub fn attention_sink_prefill_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sinks: &[f32],
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    start_q: usize,
    window: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let q_features = heads * head_dim;
    let kv_features = kv_heads * head_dim;
    let query_group_size = heads / kv_heads;
    let mut out = vec![0.0f32; batch * query_len * q_features];
    for batch_index in 0..batch {
        for query_index in 0..query_len {
            let query_pos = start_q + query_index;
            let window_start = if window == 0 {
                0
            } else {
                query_pos.saturating_sub(window - 1)
            };
            for (head, sink_value) in sinks.iter().copied().enumerate().take(heads) {
                let kv_head = head / query_group_size;
                let mut scores = vec![sink_value];
                let mut key_indices = Vec::new();
                for key_index in 0..key_len {
                    if (causal && key_index > query_pos)
                        || (window != 0 && key_index < window_start)
                    {
                        continue;
                    }
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        let query_offset = batch_index * query_len * q_features
                            + query_index * q_features
                            + head * head_dim
                            + dim;
                        let key_offset = batch_index * key_len * kv_features
                            + key_index * kv_features
                            + kv_head * head_dim
                            + dim;
                        score += query[query_offset] * key[key_offset];
                    }
                    scores.push(score * scale);
                    key_indices.push(key_index);
                }
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .copied()
                    .map(|score| (score - max_score).exp())
                    .collect::<Vec<_>>();
                let denom = weights.iter().sum::<f32>();
                for (index, key_index) in key_indices.iter().copied().enumerate() {
                    let weight = weights[index + 1] / denom;
                    for dim in 0..head_dim {
                        let value_offset = batch_index * key_len * kv_features
                            + key_index * kv_features
                            + kv_head * head_dim
                            + dim;
                        let output_offset = batch_index * query_len * q_features
                            + query_index * q_features
                            + head * head_dim
                            + dim;
                        out[output_offset] += weight * value[value_offset];
                    }
                }
            }
        }
    }
    out
}

pub fn splitk_reduce_f32(
    attn: &[f32],
    lse: &[f32],
    batch: usize,
    heads: usize,
    splits: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * heads * head_dim];
    for batch_index in 0..batch {
        for head in 0..heads {
            let max_lse = (0..splits)
                .map(|split| lse[batch_index * heads * splits + head * splits + split])
                .fold(f32::NEG_INFINITY, f32::max);
            let denom = (0..splits)
                .map(|split| {
                    (lse[batch_index * heads * splits + head * splits + split] - max_lse).exp()
                })
                .sum::<f32>();
            for dim in 0..head_dim {
                let mut sum = 0.0f32;
                for split in 0..splits {
                    let weight =
                        (lse[batch_index * heads * splits + head * splits + split] - max_lse).exp();
                    let attn_offset = batch_index * heads * splits * head_dim
                        + head * splits * head_dim
                        + split * head_dim
                        + dim;
                    sum += weight * attn[attn_offset];
                }
                out[batch_index * heads * head_dim + head * head_dim + dim] = sum / denom;
            }
        }
    }
    out
}
