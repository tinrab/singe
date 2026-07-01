#[cutile::module]
mod kernels {
    use crate::cuda::cutile::kernel::utility::*;
    use cutile::core::*;

    const VECTOR_TILE_SIZE: i32 = 128;

    #[cutile::entry()]
    pub unsafe fn minimax_sparse_attention_gathered_gqa_f32(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        positions: *mut i32,
        valid_mask: *mut i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        selected_keys: i32,
        scale: f32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];

        // Gathered sparse attention owns one output scalar per lane.
        // The `positions` and `valid_mask` tensors provide selected KV indices.
        // They are indexed by batch, KV head, and query position.
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(valid, offsets, zero_offsets);
        let head_dim_tile = broadcast_scalar(head_dim, tile_shape);
        let heads_tile = broadcast_scalar(heads, tile_shape);
        let query_len_tile = broadcast_scalar(query_len, tile_shape);
        let dim = safe_offsets - (safe_offsets / head_dim_tile) * head_dim_tile;
        let head_offsets = safe_offsets / head_dim_tile;
        let head = head_offsets - (head_offsets / heads_tile) * heads_tile;
        let query_offsets = head_offsets / heads_tile;
        let query_index = query_offsets - (query_offsets / query_len_tile) * query_len_tile;
        let batch_index = query_offsets / query_len_tile;
        let group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(group_size, tile_shape);
        let sparse_base = ((batch_index * broadcast_scalar(kv_heads, tile_shape) + kv_head)
            * query_len_tile
            + query_index)
            * broadcast_scalar(selected_keys, tile_shape);
        let query_base =
            ((batch_index * query_len_tile + query_index) * heads_tile + head) * head_dim_tile;
        let kv_batch_base = (batch_index * broadcast_scalar(kv_heads, tile_shape) + kv_head)
            * broadcast_scalar(key_len * head_dim, tile_shape);

        // Pass 1 computes the maximum over selected sparse keys.
        // It tracks whether any selected key is valid so fully empty rows can store zero.
        let mut max_score = constant(-1.0e30f32, tile_shape);
        let mut any_valid_i32 = constant(0i32, tile_shape);
        for selected in 0i32..selected_keys {
            let selected_tile = broadcast_scalar(selected, tile_shape);
            let sparse_offset = sparse_base + selected_tile;
            let valid_value = load_i32_vector(valid_mask, sparse_offset, valid, 0i32);
            let key_index = load_i32_vector(positions, sparse_offset, valid, 0i32);
            let selected_valid = valid & cmpi(valid_value, zero_offsets, predicate::NotEqual);
            let mut score = constant(0.0f32, tile_shape);
            for score_dim in 0i32..head_dim {
                let score_dim_tile = broadcast_scalar(score_dim, tile_shape);
                let q = load_vector(query, query_base + score_dim_tile, selected_valid, 0.0f32);
                let k = load_vector(
                    key,
                    kv_batch_base + key_index * head_dim_tile + score_dim_tile,
                    selected_valid,
                    0.0f32,
                );
                score = score + q * k;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                selected_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
            any_valid_i32 = select(
                selected_valid,
                broadcast_scalar(1i32, tile_shape),
                any_valid_i32,
            );
        }

        let has_any = cmpi(any_valid_i32, zero_offsets, predicate::NotEqual);
        let mut denom = constant(0.0f32, tile_shape);
        let zero_f32: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);

        // Pass 2 recomputes scores to build the sparse softmax denominator.
        for selected in 0i32..selected_keys {
            let selected_tile = broadcast_scalar(selected, tile_shape);
            let sparse_offset = sparse_base + selected_tile;
            let valid_value = load_i32_vector(valid_mask, sparse_offset, valid, 0i32);
            let key_index = load_i32_vector(positions, sparse_offset, valid, 0i32);
            let selected_valid = valid & cmpi(valid_value, zero_offsets, predicate::NotEqual);
            let mut score = constant(0.0f32, tile_shape);
            for score_dim in 0i32..head_dim {
                let score_dim_tile = broadcast_scalar(score_dim, tile_shape);
                let q = load_vector(query, query_base + score_dim_tile, selected_valid, 0.0f32);
                let k = load_vector(
                    key,
                    kv_batch_base + key_index * head_dim_tile + score_dim_tile,
                    selected_valid,
                    0.0f32,
                );
                score = score + q * k;
            }
            let weight = exp(score * broadcast_scalar(scale, tile_shape) - max_score);
            denom = denom + select(selected_valid, weight, zero_f32);
        }

        let mut sum = constant(0.0f32, tile_shape);
        // Pass 3 applies normalized probabilities to the selected V entries.
        for selected in 0i32..selected_keys {
            let selected_tile = broadcast_scalar(selected, tile_shape);
            let sparse_offset = sparse_base + selected_tile;
            let valid_value = load_i32_vector(valid_mask, sparse_offset, valid, 0i32);
            let key_index = load_i32_vector(positions, sparse_offset, valid, 0i32);
            let selected_valid = valid & cmpi(valid_value, zero_offsets, predicate::NotEqual);
            let mut score = constant(0.0f32, tile_shape);
            for score_dim in 0i32..head_dim {
                let score_dim_tile = broadcast_scalar(score_dim, tile_shape);
                let q = load_vector(query, query_base + score_dim_tile, selected_valid, 0.0f32);
                let k = load_vector(
                    key,
                    kv_batch_base + key_index * head_dim_tile + score_dim_tile,
                    selected_valid,
                    0.0f32,
                );
                score = score + q * k;
            }
            let weight = exp(score * broadcast_scalar(scale, tile_shape) - max_score) / denom;
            let v = load_vector(
                value,
                kv_batch_base + key_index * head_dim_tile + dim,
                selected_valid,
                0.0f32,
            );
            sum = sum + select(selected_valid, weight * v, zero_f32);
        }

        store_vector(out, offsets, select(has_any, sum, zero_f32), valid);
    }

    #[cutile::entry()]
    pub unsafe fn minimax_sparse_attention_select_topk_blocks_f32(
        out: *mut i32,
        block_scores: *mut f32,
        local_blocks: *mut i32,
        rows: i32,
        blocks: i32,
        top_k: i32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(valid, offsets, zero_offsets);
        let top_k_tile = broadcast_scalar(top_k, tile_shape);
        let rows_tile = broadcast_scalar(rows, tile_shape);
        let slot = safe_offsets - (safe_offsets / top_k_tile) * top_k_tile;
        let row_offsets = safe_offsets / top_k_tile;
        let row = row_offsets - (row_offsets / rows_tile) * rows_tile;
        let batch_index = row_offsets / rows_tile;
        let row_base = (batch_index * rows_tile + row) * broadcast_scalar(blocks, tile_shape);
        let local_offset = batch_index * rows_tile + row;
        let local_raw = load_i32_vector(local_blocks, local_offset, valid, 0i32);
        let max_block = broadcast_scalar(blocks - 1, tile_shape);
        let local = min_tile(max_tile(local_raw, zero_offsets), max_block);
        let k_eff = min_tile(top_k_tile, broadcast_scalar(blocks, tile_shape));
        let slot_valid = valid & cmpi(slot, k_eff, predicate::LessThan);

        let mut selected = constant(-1i32, tile_shape);
        let is_local_slot = cmpi(slot, zero_offsets, predicate::Equal);
        selected = select(slot_valid & is_local_slot, local, selected);

        let target_rank = slot - broadcast_scalar(1i32, tile_shape);
        for candidate in 0i32..blocks {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let candidate_score =
                load_vector(block_scores, row_base + candidate_tile, slot_valid, 0.0f32);
            let mut rank = constant(0i32, tile_shape);
            for other in 0i32..blocks {
                let other_tile = broadcast_scalar(other, tile_shape);
                let other_not_local = cmpi(other_tile, local, predicate::NotEqual);
                let other_not_candidate = cmpi(other_tile, candidate_tile, predicate::NotEqual);
                let other_score =
                    load_vector(block_scores, row_base + other_tile, slot_valid, 0.0f32);
                let score_greater = cmpf(
                    other_score,
                    candidate_score,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                );
                let score_equal = cmpf(
                    other_score,
                    candidate_score,
                    predicate::Equal,
                    cmp_ordering::Ordered,
                );
                let tie_before =
                    score_equal & cmpi(other_tile, candidate_tile, predicate::LessThan);
                let before = other_not_local & other_not_candidate & (score_greater | tie_before);
                rank = rank + select(before, broadcast_scalar(1i32, tile_shape), zero_offsets);
            }
            let non_local_slot = cmpi(slot, zero_offsets, predicate::NotEqual);
            let candidate_not_local = cmpi(candidate_tile, local, predicate::NotEqual);
            let candidate_selected = slot_valid
                & non_local_slot
                & candidate_not_local
                & cmpi(rank, target_rank, predicate::Equal);
            selected = select(candidate_selected, candidate_tile, selected);
        }

        store_i32_vector(out, offsets, selected, valid);
    }

    #[cutile::entry()]
    pub unsafe fn minimax_sparse_attention_selected_token_positions_i32(
        positions: *mut i32,
        valid_mask: *mut i32,
        selected_blocks: *mut i32,
        query_positions: *mut i32,
        rows: i32,
        selected_block_count: i32,
        block_size: i32,
        seq_len: i32,
        expanded_keys: i32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(valid, offsets, zero_offsets);
        let expanded_tile = broadcast_scalar(expanded_keys, tile_shape);
        let rows_tile = broadcast_scalar(rows, tile_shape);
        let key_slot = safe_offsets - (safe_offsets / expanded_tile) * expanded_tile;
        let row_offsets = safe_offsets / expanded_tile;
        let row = row_offsets - (row_offsets / rows_tile) * rows_tile;
        let batch_index = row_offsets / rows_tile;
        let selected_block_slot = key_slot / broadcast_scalar(block_size, tile_shape);
        let local_key = key_slot - selected_block_slot * broadcast_scalar(block_size, tile_shape);
        let selected_offset = (batch_index * rows_tile + row)
            * broadcast_scalar(selected_block_count, tile_shape)
            + selected_block_slot;
        let selected_block = load_i32_vector(selected_blocks, selected_offset, valid, 0i32);
        let query_position_offset = batch_index * rows_tile + row;
        let query_position = load_i32_vector(query_positions, query_position_offset, valid, 0i32);
        let raw_position = selected_block * broadcast_scalar(block_size, tile_shape) + local_key;
        let seq_len_tile = broadcast_scalar(seq_len, tile_shape);
        let max_position = broadcast_scalar(seq_len - 1, tile_shape);
        let clamped_position = min_tile(max_tile(raw_position, zero_offsets), max_position);
        let raw_non_negative = cmpi(raw_position, zero_offsets, predicate::GreaterThanOrEqual);
        let raw_in_sequence = cmpi(raw_position, seq_len_tile, predicate::LessThan);
        let raw_causal = cmpi(raw_position, query_position, predicate::LessThanOrEqual);
        let token_valid = valid & raw_non_negative & raw_in_sequence & raw_causal;
        let one_i32: Tile<i32, { [128] }> = constant(1i32, tile_shape);
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let valid_values = select(token_valid, one_i32, zero_i32);
        store_i32_vector(positions, offsets, clamped_position, valid);
        store_i32_vector(valid_mask, offsets, valid_values, valid);
    }

    #[cutile::entry()]
    pub unsafe fn minimax_sparse_attention_block_max_f32(
        out: *mut f32,
        scores: *mut f32,
        rows: i32,
        key_len: i32,
        block_size: i32,
        blocks: i32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(valid, offsets, zero_offsets);
        let blocks_tile = broadcast_scalar(blocks, tile_shape);
        let rows_tile = broadcast_scalar(rows, tile_shape);
        let block = safe_offsets - (safe_offsets / blocks_tile) * blocks_tile;
        let row_offsets = safe_offsets / blocks_tile;
        let row = row_offsets - (row_offsets / rows_tile) * rows_tile;
        let batch_index = row_offsets / rows_tile;
        let source_base = (batch_index * rows_tile + row) * broadcast_scalar(key_len, tile_shape);
        let key_start = block * broadcast_scalar(block_size, tile_shape);
        let key_len_tile = broadcast_scalar(key_len, tile_shape);

        let mut max_score = constant(-1.0e30f32, tile_shape);
        for local_key in 0i32..block_size {
            let key_index = key_start + broadcast_scalar(local_key, tile_shape);
            let key_valid = valid & cmpi(key_index, key_len_tile, predicate::LessThan);
            let value = load_vector(scores, source_base + key_index, key_valid, -1.0e30f32);
            max_score = max_tile(max_score, value);
        }
        store_vector(out, offsets, max_score, valid);
    }

    #[cutile::entry()]
    pub unsafe fn compressed_sparse_attention_lightning_indexer_f32(
        out: *mut f32,
        indexer_query: *mut f32,
        indexer_key: *mut f32,
        indexer_weight: *mut f32,
        query_len: i32,
        blocks: i32,
        index_heads: i32,
        index_dim: i32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(valid, offsets, zero_offsets);
        let blocks_tile = broadcast_scalar(blocks, tile_shape);
        let query_len_tile = broadcast_scalar(query_len, tile_shape);
        let block = safe_offsets - (safe_offsets / blocks_tile) * blocks_tile;
        let query_offsets = safe_offsets / blocks_tile;
        let query_index = query_offsets - (query_offsets / query_len_tile) * query_len_tile;
        let batch_index = query_offsets / query_len_tile;
        let query_batch_base = (batch_index * query_len_tile + query_index)
            * broadcast_scalar(index_heads * index_dim, tile_shape);
        let weight_batch_base = (batch_index * query_len_tile + query_index)
            * broadcast_scalar(index_heads, tile_shape);
        let key_batch_base =
            (batch_index * blocks_tile + block) * broadcast_scalar(index_dim, tile_shape);

        let mut score = constant(0.0f32, tile_shape);
        for index_head in 0i32..index_heads {
            let mut inner = constant(0.0f32, tile_shape);
            for dim in 0i32..index_dim {
                let query = load_vector(
                    indexer_query,
                    query_batch_base + broadcast_scalar(index_head * index_dim + dim, tile_shape),
                    valid,
                    0.0f32,
                );
                let key = load_vector(
                    indexer_key,
                    key_batch_base + broadcast_scalar(dim, tile_shape),
                    valid,
                    0.0f32,
                );
                inner = inner + query * key;
            }
            let zero_inner: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let inner = max_tile(inner, zero_inner);
            let weight = load_vector(
                indexer_weight,
                weight_batch_base + broadcast_scalar(index_head, tile_shape),
                valid,
                0.0f32,
            );
            score = score + weight * inner;
        }
        store_vector(out, offsets, score, valid);
    }

    #[cutile::entry()]
    pub unsafe fn compressed_sparse_attention_topk_selector_f32(
        out: *mut f32,
        selected_indices: *mut i32,
        scores: *mut f32,
        compressed_kv: *mut f32,
        query_positions: *mut i32,
        query_len: i32,
        blocks: i32,
        head_dim: i32,
        top_k: i32,
        compression_block: i32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(valid, offsets, zero_offsets);
        let head_dim_tile = broadcast_scalar(head_dim, tile_shape);
        let top_k_tile = broadcast_scalar(top_k, tile_shape);
        let blocks_tile = broadcast_scalar(blocks, tile_shape);
        let query_len_tile = broadcast_scalar(query_len, tile_shape);
        let dim = safe_offsets - (safe_offsets / head_dim_tile) * head_dim_tile;
        let slot_offsets = safe_offsets / head_dim_tile;
        let slot = slot_offsets - (slot_offsets / top_k_tile) * top_k_tile;
        let query_offsets = slot_offsets / top_k_tile;
        let query_index = query_offsets - (query_offsets / query_len_tile) * query_len_tile;
        let batch_index = query_offsets / query_len_tile;
        let selected_offset = (batch_index * query_len_tile + query_index) * top_k_tile + slot;
        let query_position = load_i32_vector(
            query_positions,
            batch_index * query_len_tile + query_index,
            valid,
            0i32,
        );
        let zero_tile: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let n_valid_raw =
            max_tile(query_position, zero_tile) / broadcast_scalar(compression_block, tile_shape);
        let n_valid = min_tile(n_valid_raw, blocks_tile);
        let slot_valid = valid & cmpi(slot, n_valid, predicate::LessThan);
        let score_base = (batch_index * query_len_tile + query_index) * blocks_tile;

        let mut selected = constant(-1i32, tile_shape);
        for candidate in 0i32..blocks {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let candidate_valid = cmpi(candidate_tile, n_valid, predicate::LessThan);
            let candidate_score =
                load_vector(scores, score_base + candidate_tile, slot_valid, 0.0f32);
            let mut rank = constant(0i32, tile_shape);
            for other in 0i32..blocks {
                let other_tile = broadcast_scalar(other, tile_shape);
                let other_valid = cmpi(other_tile, n_valid, predicate::LessThan);
                let other_not_candidate = cmpi(other_tile, candidate_tile, predicate::NotEqual);
                let other_score = load_vector(scores, score_base + other_tile, slot_valid, 0.0f32);
                let score_greater = cmpf(
                    other_score,
                    candidate_score,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                );
                let score_equal = cmpf(
                    other_score,
                    candidate_score,
                    predicate::Equal,
                    cmp_ordering::Ordered,
                );
                let tie_before =
                    score_equal & cmpi(other_tile, candidate_tile, predicate::LessThan);
                let before = other_valid & other_not_candidate & (score_greater | tie_before);
                rank = rank + select(before, broadcast_scalar(1i32, tile_shape), zero_tile);
            }
            let candidate_selected =
                slot_valid & candidate_valid & cmpi(rank, slot, predicate::Equal);
            selected = select(candidate_selected, candidate_tile, selected);
        }

        let source_offset = (batch_index * blocks_tile + selected) * head_dim_tile + dim;
        let gathered = load_vector(compressed_kv, source_offset, slot_valid, 0.0f32);
        let zero_f32: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(slot_valid, gathered, zero_f32);
        store_vector(out, offsets, output, valid);
        let first_dim = cmpi(dim, zero_offsets, predicate::Equal);
        store_i32_vector(
            selected_indices,
            selected_offset,
            selected,
            valid & first_dim,
        );
    }

    #[cutile::entry()]
    pub unsafe fn compressed_sparse_attention_shared_mqa_f32(
        out: *mut f32,
        query: *mut f32,
        kv_entries: *mut f32,
        valid_mask: *mut i32,
        sink: *mut f32,
        query_len: i32,
        heads: i32,
        head_dim: i32,
        kv_len: i32,
        scale: f32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(valid, offsets, zero_offsets);
        let head_dim_tile = broadcast_scalar(head_dim, tile_shape);
        let heads_tile = broadcast_scalar(heads, tile_shape);
        let query_len_tile = broadcast_scalar(query_len, tile_shape);
        let kv_len_tile = broadcast_scalar(kv_len, tile_shape);
        let dim = safe_offsets - (safe_offsets / head_dim_tile) * head_dim_tile;
        let head_offsets = safe_offsets / head_dim_tile;
        let head = head_offsets - (head_offsets / heads_tile) * heads_tile;
        let query_offsets = head_offsets / heads_tile;
        let query_index = query_offsets - (query_offsets / query_len_tile) * query_len_tile;
        let batch_index = query_offsets / query_len_tile;
        let query_base =
            ((batch_index * query_len_tile + query_index) * heads_tile + head) * head_dim_tile;
        let kv_base = (batch_index * query_len_tile + query_index) * kv_len_tile * head_dim_tile;
        let mask_base = (batch_index * query_len_tile + query_index) * kv_len_tile;
        let sink_score = load_vector(sink, head, valid, 0.0f32);
        let zero_f32: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);

        let mut max_score = sink_score;
        for key_index in 0i32..kv_len {
            let key_tile = broadcast_scalar(key_index, tile_shape);
            let mask_value = load_i32_vector(valid_mask, mask_base + key_tile, valid, 0i32);
            let key_valid = valid & cmpi(mask_value, zero_offsets, predicate::NotEqual);
            let mut score = constant(0.0f32, tile_shape);
            for score_dim in 0i32..head_dim {
                let query_value = load_vector(
                    query,
                    query_base + broadcast_scalar(score_dim, tile_shape),
                    key_valid,
                    0.0f32,
                );
                let key_value = load_vector(
                    kv_entries,
                    kv_base + key_tile * head_dim_tile + broadcast_scalar(score_dim, tile_shape),
                    key_valid,
                    0.0f32,
                );
                score = score + query_value * key_value;
            }
            let scaled = score * broadcast_scalar(scale, tile_shape);
            max_score = select(key_valid, max_tile(max_score, scaled), max_score);
        }

        let mut denom = exp(sink_score - max_score);
        for key_index in 0i32..kv_len {
            let key_tile = broadcast_scalar(key_index, tile_shape);
            let mask_value = load_i32_vector(valid_mask, mask_base + key_tile, valid, 0i32);
            let key_valid = valid & cmpi(mask_value, zero_offsets, predicate::NotEqual);
            let mut score = constant(0.0f32, tile_shape);
            for score_dim in 0i32..head_dim {
                let query_value = load_vector(
                    query,
                    query_base + broadcast_scalar(score_dim, tile_shape),
                    key_valid,
                    0.0f32,
                );
                let key_value = load_vector(
                    kv_entries,
                    kv_base + key_tile * head_dim_tile + broadcast_scalar(score_dim, tile_shape),
                    key_valid,
                    0.0f32,
                );
                score = score + query_value * key_value;
            }
            let weight = select(
                key_valid,
                exp(score * broadcast_scalar(scale, tile_shape) - max_score),
                zero_f32,
            );
            denom = denom + weight;
        }

        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..kv_len {
            let key_tile = broadcast_scalar(key_index, tile_shape);
            let mask_value = load_i32_vector(valid_mask, mask_base + key_tile, valid, 0i32);
            let key_valid = valid & cmpi(mask_value, zero_offsets, predicate::NotEqual);
            let mut score = constant(0.0f32, tile_shape);
            for score_dim in 0i32..head_dim {
                let query_value = load_vector(
                    query,
                    query_base + broadcast_scalar(score_dim, tile_shape),
                    key_valid,
                    0.0f32,
                );
                let key_value = load_vector(
                    kv_entries,
                    kv_base + key_tile * head_dim_tile + broadcast_scalar(score_dim, tile_shape),
                    key_valid,
                    0.0f32,
                );
                score = score + query_value * key_value;
            }
            let weight = select(
                key_valid,
                exp(score * broadcast_scalar(scale, tile_shape) - max_score) / denom,
                zero_f32,
            );
            let value = load_vector(
                kv_entries,
                kv_base + key_tile * head_dim_tile + dim,
                key_valid,
                0.0f32,
            );
            sum = sum + weight * value;
        }
        store_vector(out, offsets, sum, valid);
    }

    #[cutile::entry()]
    pub unsafe fn compressed_sparse_attention_compress_f32(
        out: *mut f32,
        hidden: *mut f32,
        weight_a_kv: *mut f32,
        weight_b_kv: *mut f32,
        weight_a_z: *mut f32,
        weight_b_z: *mut f32,
        bias_a: *mut f32,
        bias_b: *mut f32,
        seq_len: i32,
        hidden_dim: i32,
        head_dim: i32,
        compression_block: i32,
        blocks: i32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(valid, offsets, zero_offsets);
        let head_dim_tile = broadcast_scalar(head_dim, tile_shape);
        let blocks_tile = broadcast_scalar(blocks, tile_shape);
        let dim = safe_offsets - (safe_offsets / head_dim_tile) * head_dim_tile;
        let block_offsets = safe_offsets / head_dim_tile;
        let block = block_offsets - (block_offsets / blocks_tile) * blocks_tile;
        let batch_index = block_offsets / blocks_tile;
        let has_prev = cmpi(block, zero_offsets, predicate::GreaterThan);

        let mut max_logit = constant(f32::NEG_INFINITY, tile_shape);
        for row in 0i32..compression_block {
            let token = block * broadcast_scalar(compression_block, tile_shape)
                + broadcast_scalar(row, tile_shape);
            let hidden_base = (batch_index * broadcast_scalar(seq_len, tile_shape) + token)
                * broadcast_scalar(hidden_dim, tile_shape);
            let mut logit = load_vector(
                bias_a,
                broadcast_scalar(row, tile_shape) * head_dim_tile + dim,
                valid,
                0.0f32,
            );
            for hidden_index in 0i32..hidden_dim {
                let hidden_value = load_vector(
                    hidden,
                    hidden_base + broadcast_scalar(hidden_index, tile_shape),
                    valid,
                    0.0f32,
                );
                let weight = load_vector(
                    weight_a_z,
                    broadcast_scalar(hidden_index, tile_shape) * head_dim_tile + dim,
                    valid,
                    0.0f32,
                );
                logit = logit + hidden_value * weight;
            }
            max_logit = max_tile(max_logit, logit);
        }
        for row in 0i32..compression_block {
            let token = (block - broadcast_scalar(1i32, tile_shape))
                * broadcast_scalar(compression_block, tile_shape)
                + broadcast_scalar(row, tile_shape);
            let hidden_base = (batch_index * broadcast_scalar(seq_len, tile_shape) + token)
                * broadcast_scalar(hidden_dim, tile_shape);
            let stream_valid = valid & has_prev;
            let mut logit = load_vector(
                bias_b,
                broadcast_scalar(row, tile_shape) * head_dim_tile + dim,
                stream_valid,
                0.0f32,
            );
            for hidden_index in 0i32..hidden_dim {
                let hidden_value = load_vector(
                    hidden,
                    hidden_base + broadcast_scalar(hidden_index, tile_shape),
                    stream_valid,
                    0.0f32,
                );
                let weight = load_vector(
                    weight_b_z,
                    broadcast_scalar(hidden_index, tile_shape) * head_dim_tile + dim,
                    stream_valid,
                    0.0f32,
                );
                logit = logit + hidden_value * weight;
            }
            let next_max_logit: Tile<f32, { [128] }> =
                select(has_prev, max_tile(max_logit, logit), max_logit);
            max_logit = next_max_logit;
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for row in 0i32..compression_block {
            let token = block * broadcast_scalar(compression_block, tile_shape)
                + broadcast_scalar(row, tile_shape);
            let hidden_base = (batch_index * broadcast_scalar(seq_len, tile_shape) + token)
                * broadcast_scalar(hidden_dim, tile_shape);
            let mut value = constant(0.0f32, tile_shape);
            let mut logit = load_vector(
                bias_a,
                broadcast_scalar(row, tile_shape) * head_dim_tile + dim,
                valid,
                0.0f32,
            );
            for hidden_index in 0i32..hidden_dim {
                let hidden_value = load_vector(
                    hidden,
                    hidden_base + broadcast_scalar(hidden_index, tile_shape),
                    valid,
                    0.0f32,
                );
                let kv_weight = load_vector(
                    weight_a_kv,
                    broadcast_scalar(hidden_index, tile_shape) * head_dim_tile + dim,
                    valid,
                    0.0f32,
                );
                let z_weight = load_vector(
                    weight_a_z,
                    broadcast_scalar(hidden_index, tile_shape) * head_dim_tile + dim,
                    valid,
                    0.0f32,
                );
                value = value + hidden_value * kv_weight;
                logit = logit + hidden_value * z_weight;
            }
            let weight = exp(logit - max_logit);
            denom = denom + weight;
            sum = sum + weight * value;
        }
        for row in 0i32..compression_block {
            let token = (block - broadcast_scalar(1i32, tile_shape))
                * broadcast_scalar(compression_block, tile_shape)
                + broadcast_scalar(row, tile_shape);
            let hidden_base = (batch_index * broadcast_scalar(seq_len, tile_shape) + token)
                * broadcast_scalar(hidden_dim, tile_shape);
            let stream_valid = valid & has_prev;
            let mut value = constant(0.0f32, tile_shape);
            let mut logit = load_vector(
                bias_b,
                broadcast_scalar(row, tile_shape) * head_dim_tile + dim,
                stream_valid,
                0.0f32,
            );
            for hidden_index in 0i32..hidden_dim {
                let hidden_value = load_vector(
                    hidden,
                    hidden_base + broadcast_scalar(hidden_index, tile_shape),
                    stream_valid,
                    0.0f32,
                );
                let kv_weight = load_vector(
                    weight_b_kv,
                    broadcast_scalar(hidden_index, tile_shape) * head_dim_tile + dim,
                    stream_valid,
                    0.0f32,
                );
                let z_weight = load_vector(
                    weight_b_z,
                    broadcast_scalar(hidden_index, tile_shape) * head_dim_tile + dim,
                    stream_valid,
                    0.0f32,
                );
                value = value + hidden_value * kv_weight;
                logit = logit + hidden_value * z_weight;
            }
            let zero_weight: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(stream_valid, exp(logit - max_logit), zero_weight);
            denom = denom + weight;
            sum = sum + weight * value;
        }
        store_vector(out, offsets, sum / denom, valid);
    }

    #[cutile::entry()]
    pub unsafe fn heavily_compressed_attention_compress_f32(
        out: *mut f32,
        hidden: *mut f32,
        weight_kv: *mut f32,
        weight_z: *mut f32,
        bias: *mut f32,
        seq_len: i32,
        hidden_dim: i32,
        head_dim: i32,
        compression_block: i32,
        blocks: i32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(valid, offsets, zero_offsets);
        let head_dim_tile = broadcast_scalar(head_dim, tile_shape);
        let blocks_tile = broadcast_scalar(blocks, tile_shape);
        let dim = safe_offsets - (safe_offsets / head_dim_tile) * head_dim_tile;
        let block_offsets = safe_offsets / head_dim_tile;
        let block = block_offsets - (block_offsets / blocks_tile) * blocks_tile;
        let batch_index = block_offsets / blocks_tile;

        let mut max_logit = constant(f32::NEG_INFINITY, tile_shape);
        for row in 0i32..compression_block {
            let token = block * broadcast_scalar(compression_block, tile_shape)
                + broadcast_scalar(row, tile_shape);
            let hidden_base = (batch_index * broadcast_scalar(seq_len, tile_shape) + token)
                * broadcast_scalar(hidden_dim, tile_shape);
            let mut logit = load_vector(
                bias,
                broadcast_scalar(row, tile_shape) * head_dim_tile + dim,
                valid,
                0.0f32,
            );
            for hidden_index in 0i32..hidden_dim {
                let hidden_value = load_vector(
                    hidden,
                    hidden_base + broadcast_scalar(hidden_index, tile_shape),
                    valid,
                    0.0f32,
                );
                let weight = load_vector(
                    weight_z,
                    broadcast_scalar(hidden_index, tile_shape) * head_dim_tile + dim,
                    valid,
                    0.0f32,
                );
                logit = logit + hidden_value * weight;
            }
            max_logit = max_tile(max_logit, logit);
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for row in 0i32..compression_block {
            let token = block * broadcast_scalar(compression_block, tile_shape)
                + broadcast_scalar(row, tile_shape);
            let hidden_base = (batch_index * broadcast_scalar(seq_len, tile_shape) + token)
                * broadcast_scalar(hidden_dim, tile_shape);
            let mut value = constant(0.0f32, tile_shape);
            let mut logit = load_vector(
                bias,
                broadcast_scalar(row, tile_shape) * head_dim_tile + dim,
                valid,
                0.0f32,
            );
            for hidden_index in 0i32..hidden_dim {
                let hidden_value = load_vector(
                    hidden,
                    hidden_base + broadcast_scalar(hidden_index, tile_shape),
                    valid,
                    0.0f32,
                );
                let kv_weight = load_vector(
                    weight_kv,
                    broadcast_scalar(hidden_index, tile_shape) * head_dim_tile + dim,
                    valid,
                    0.0f32,
                );
                let z_weight = load_vector(
                    weight_z,
                    broadcast_scalar(hidden_index, tile_shape) * head_dim_tile + dim,
                    valid,
                    0.0f32,
                );
                value = value + hidden_value * kv_weight;
                logit = logit + hidden_value * z_weight;
            }
            let weight = exp(logit - max_logit);
            denom = denom + weight;
            sum = sum + weight * value;
        }
        store_vector(out, offsets, sum / denom, valid);
    }

    #[cutile::entry()]
    pub unsafe fn heavily_compressed_attention_f32(
        out: *mut f32,
        query: *mut f32,
        compressed_kv: *mut f32,
        weight_group: *mut f32,
        weight_final: *mut f32,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        compression_block: i32,
        blocks: i32,
        groups: i32,
        group_dim: i32,
        hidden_dim: i32,
        scale: f32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(valid, offsets, zero_offsets);
        let hidden_dim_tile = broadcast_scalar(hidden_dim, tile_shape);
        let seq_len_tile = broadcast_scalar(seq_len, tile_shape);
        let output_dim = safe_offsets - (safe_offsets / hidden_dim_tile) * hidden_dim_tile;
        let token_offsets = safe_offsets / hidden_dim_tile;
        let query_index = token_offsets - (token_offsets / seq_len_tile) * seq_len_tile;
        let batch_index = token_offsets / seq_len_tile;
        let visible_blocks = min_tile(
            query_index / broadcast_scalar(compression_block, tile_shape),
            broadcast_scalar(blocks, tile_shape),
        );
        let heads_per_group = heads / groups;
        let query_token_base = (batch_index * seq_len_tile + query_index)
            * broadcast_scalar(heads * head_dim, tile_shape);
        let kv_batch_base = batch_index * broadcast_scalar(blocks * head_dim, tile_shape);
        let mut output = constant(0.0f32, tile_shape);

        for group in 0i32..groups {
            for group_out in 0i32..group_dim {
                let mut inter = constant(0.0f32, tile_shape);
                for local_head in 0i32..heads_per_group {
                    let head = group * heads_per_group + local_head;
                    for dim in 0i32..head_dim {
                        let mut max_score = constant(f32::NEG_INFINITY, tile_shape);
                        for block in 0i32..blocks {
                            let block_valid = valid
                                & cmpi(
                                    broadcast_scalar(block, tile_shape),
                                    visible_blocks,
                                    predicate::LessThan,
                                );
                            let mut score = constant(0.0f32, tile_shape);
                            for score_dim in 0i32..head_dim {
                                let q = load_vector(
                                    query,
                                    query_token_base
                                        + broadcast_scalar(head * head_dim + score_dim, tile_shape),
                                    block_valid,
                                    0.0f32,
                                );
                                let kv = load_vector(
                                    compressed_kv,
                                    kv_batch_base
                                        + broadcast_scalar(
                                            block * head_dim + score_dim,
                                            tile_shape,
                                        ),
                                    block_valid,
                                    0.0f32,
                                );
                                score = score + q * kv;
                            }
                            score = score * broadcast_scalar(scale, tile_shape);
                            let next_max_score: Tile<f32, { [128] }> = select(
                                block_valid
                                    & cmpf(
                                        score,
                                        max_score,
                                        predicate::GreaterThan,
                                        cmp_ordering::Ordered,
                                    ),
                                score,
                                max_score,
                            );
                            max_score = next_max_score;
                        }

                        let mut denom = constant(0.0f32, tile_shape);
                        let mut head_value = constant(0.0f32, tile_shape);
                        for block in 0i32..blocks {
                            let block_valid = valid
                                & cmpi(
                                    broadcast_scalar(block, tile_shape),
                                    visible_blocks,
                                    predicate::LessThan,
                                );
                            let mut score = constant(0.0f32, tile_shape);
                            for score_dim in 0i32..head_dim {
                                let q = load_vector(
                                    query,
                                    query_token_base
                                        + broadcast_scalar(head * head_dim + score_dim, tile_shape),
                                    block_valid,
                                    0.0f32,
                                );
                                let kv = load_vector(
                                    compressed_kv,
                                    kv_batch_base
                                        + broadcast_scalar(
                                            block * head_dim + score_dim,
                                            tile_shape,
                                        ),
                                    block_valid,
                                    0.0f32,
                                );
                                score = score + q * kv;
                            }
                            let weight =
                                exp(score * broadcast_scalar(scale, tile_shape) - max_score);
                            let zero_weight: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
                            let weight = select(block_valid, weight, zero_weight);
                            let value = load_vector(
                                compressed_kv,
                                kv_batch_base
                                    + broadcast_scalar(block * head_dim + dim, tile_shape),
                                block_valid,
                                0.0f32,
                            );
                            denom = denom + weight;
                            head_value = head_value + weight * value;
                        }
                        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
                        let has_blocks = cmpi(visible_blocks, zero_i32, predicate::GreaterThan);
                        let zero_head_value: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
                        head_value = select(has_blocks, head_value / denom, zero_head_value);
                        let flat = local_head * head_dim + dim;
                        let group_weight = load_vector(
                            weight_group,
                            broadcast_scalar(
                                (group * heads_per_group * head_dim + flat) * group_dim + group_out,
                                tile_shape,
                            ),
                            valid,
                            0.0f32,
                        );
                        inter = inter + head_value * group_weight;
                    }
                }
                let final_weight = load_vector(
                    weight_final,
                    broadcast_scalar((group * group_dim + group_out) * hidden_dim, tile_shape)
                        + output_dim,
                    valid,
                    0.0f32,
                );
                output = output + inter * final_weight;
            }
        }

        store_vector(out, offsets, output, valid);
    }

    fn sliding_window_attention_f32<const HEAD_DIM: i32, const WINDOW: i32>(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_start: i32,
        key_start: i32,
        window: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        output_scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_abs = query_index + broadcast_scalar(query_start, tile_shape);
        let effective_window = if WINDOW == 0 { window } else { WINDOW };
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + effective_window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + effective_window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }
        let output = (sum / denom) * broadcast_scalar(output_scale, tile_shape);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);
    }

    fn fmha_prefill_lse_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let lse_offsets = lse_offsets * broadcast_scalar(_query_len, tile_shape) + query_index;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(lse, lse_offsets, max_score + log(denom), valid & first_dim);
    }

    fn mla_score_f32(
        query: *mut f32,
        query_pe: *mut f32,
        key: *mut f32,
        key_pe: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        key_pe_batch_base: Tile<i32, { [128] }>,
        query_index: Tile<i32, { [128] }>,
        key_index: i32,
        head: Tile<i32, { [128] }>,
        kv_head: Tile<i32, { [128] }>,
        q_features: i32,
        kv_features: i32,
        qpe_features: i32,
        kpe_features: i32,
        head_dim: i32,
        pe_dim: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(q_features, tile_shape)
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = key_batch_base
                + broadcast_scalar(key_index * kv_features, tile_shape)
                + kv_head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + query_index * broadcast_scalar(qpe_features, tile_shape)
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = key_pe_batch_base
                + broadcast_scalar(key_index * kpe_features, tile_shape)
                + kv_head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query_pe, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_pe, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn sparse_mla_score_f32(
        query: *mut f32,
        query_pe: *mut f32,
        key: *mut f32,
        key_pe: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        key_pe_batch_base: Tile<i32, { [128] }>,
        query_index: Tile<i32, { [128] }>,
        key_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        kv_head: Tile<i32, { [128] }>,
        q_features: i32,
        kv_features: i32,
        qpe_features: i32,
        head_dim: i32,
        pe_dim: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(q_features, tile_shape)
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = key_batch_base
                + key_index * broadcast_scalar(kv_features, tile_shape)
                + kv_head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + query_index * broadcast_scalar(qpe_features, tile_shape)
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = key_pe_batch_base
                + key_index * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query_pe, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_pe, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn mla_decode_split_score_f32(
        query: *mut f32,
        query_pe: *mut f32,
        key_value: *mut f32,
        key_pe: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        key_pe_batch_base: Tile<i32, { [128] }>,
        key_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        head_dim: i32,
        pe_dim: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = key_batch_base
                + key_index * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_value, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = key_pe_batch_base
                + key_index * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query_pe, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_pe, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn mla_decode_score_f32(
        query: *mut f32,
        query_pe: *mut f32,
        key_value: *mut f32,
        key_pe: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        key_pe_batch_base: Tile<i32, { [128] }>,
        key_index: i32,
        head: Tile<i32, { [128] }>,
        q_features: i32,
        qpe_features: i32,
        head_dim: i32,
        pe_dim: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets =
                key_batch_base + broadcast_scalar(key_index * head_dim + dim_index, tile_shape);
            let query_values = load_vector(query, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_value, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets =
                key_pe_batch_base + broadcast_scalar(key_index * pe_dim + dim_index, tile_shape);
            let query_values = load_vector(query_pe, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_pe, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        let _ = q_features;
        let _ = qpe_features;
        score * broadcast_scalar(scale, tile_shape)
    }

    fn mla_decode_score_paged_f32(
        query: *mut f32,
        query_pe: *mut f32,
        key_value_cache: *mut f32,
        key_pe_cache: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        physical_block: Tile<i32, { [128] }>,
        block_token: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        q_features: i32,
        qpe_features: i32,
        head_dim: i32,
        pe_dim: i32,
        key_value_cache_block_stride: i32,
        key_pe_cache_block_stride: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = physical_block
                * broadcast_scalar(key_value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_value_cache, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = physical_block
                * broadcast_scalar(key_pe_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query_pe, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_pe_cache, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        let _ = q_features;
        let _ = qpe_features;
        score * broadcast_scalar(scale, tile_shape)
    }

    fn mla_decode_score_paged_f16(
        query: *mut f16,
        query_pe: *mut f16,
        key_value_cache: *mut f16,
        key_pe_cache: *mut f16,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        physical_block: Tile<i32, { [128] }>,
        block_token: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        q_features: i32,
        qpe_features: i32,
        head_dim: i32,
        pe_dim: i32,
        key_value_cache_block_stride: i32,
        key_pe_cache_block_stride: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = physical_block
                * broadcast_scalar(key_value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector_f16_as_f32(query, query_offsets, valid);
            let key_values = load_vector_f16_as_f32(key_value_cache, key_offsets, valid);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = physical_block
                * broadcast_scalar(key_pe_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector_f16_as_f32(query_pe, query_offsets, valid);
            let key_values = load_vector_f16_as_f32(key_pe_cache, key_offsets, valid);
            score = score + query_values * key_values;
        }
        let _ = q_features;
        let _ = qpe_features;
        score * broadcast_scalar(scale, tile_shape)
    }

    fn mla_decode_score_paged_bf16(
        query: *mut bf16,
        query_pe: *mut bf16,
        key_value_cache: *mut bf16,
        key_pe_cache: *mut bf16,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        physical_block: Tile<i32, { [128] }>,
        block_token: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        q_features: i32,
        qpe_features: i32,
        head_dim: i32,
        pe_dim: i32,
        key_value_cache_block_stride: i32,
        key_pe_cache_block_stride: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = physical_block
                * broadcast_scalar(key_value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector_bf16_as_f32(query, query_offsets, valid);
            let key_values = load_vector_bf16_as_f32(key_value_cache, key_offsets, valid);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = physical_block
                * broadcast_scalar(key_pe_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector_bf16_as_f32(query_pe, query_offsets, valid);
            let key_values = load_vector_bf16_as_f32(key_pe_cache, key_offsets, valid);
            score = score + query_values * key_values;
        }
        let _ = q_features;
        let _ = qpe_features;
        score * broadcast_scalar(scale, tile_shape)
    }

    fn fmha_prefill_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);
    }

    fn attention_sink_prefill_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        sinks: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        start_q: i32,
        window: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let query_pos = broadcast_scalar(start_q, tile_shape) + query_index;
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let window_start = if window == 0 {
            zero_i32
        } else {
            let raw_window_start = query_pos - broadcast_scalar(window - 1, tile_shape);
            select(
                cmpi(raw_window_start, zero_i32, predicate::GreaterThan),
                raw_window_start,
                zero_i32,
            )
        };
        let sink = load_vector(sinks, head, valid, 0.0f32);

        let mut max_score = sink;
        for key_index in 0i32..key_len {
            let mut key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_pos,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            if window != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        window_start,
                        predicate::GreaterThanOrEqual,
                    );
            }
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = exp(sink - max_score);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_pos,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            if window != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        window_start,
                        predicate::GreaterThanOrEqual,
                    );
            }
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);
    }

    fn softcapped_window_attention_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_key_valid(valid, query_index, key_index, causal, window_size);
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let score = softcapped_attention_score(score, soft_cap, has_soft_cap);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_key_valid(valid, query_index, key_index, causal, window_size);
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let score = softcapped_attention_score(score, soft_cap, has_soft_cap);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_values = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_values, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);
    }

    fn softcapped_window_attention_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_key_valid(valid, query_index, key_index, causal, window_size);
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let score = softcapped_attention_score(score, soft_cap, has_soft_cap);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_key_valid(valid, query_index, key_index, causal, window_size);
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let score = softcapped_attention_score(score, soft_cap, has_soft_cap);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_values = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_values, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);
    }

    fn softcapped_window_attention_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_key_valid(valid, query_index, key_index, causal, window_size);
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let score = softcapped_attention_score(score, soft_cap, has_soft_cap);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_key_valid(valid, query_index, key_index, causal, window_size);
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let score = softcapped_attention_score(score, soft_cap, has_soft_cap);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_values = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_values, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);
    }

    fn softcapped_window_key_valid(
        valid: Tile<bool, { [128] }>,
        query_index: Tile<i32, { [128] }>,
        key_index: i32,
        causal: i32,
        window_size: i32,
    ) -> Tile<bool, { [128] }> {
        let tile_shape = const_shape![128];
        let key_tile = broadcast_scalar(key_index, tile_shape);
        let mut key_valid = valid
            & if causal != 0 {
                cmpi(key_tile, query_index, predicate::LessThanOrEqual)
            } else {
                constant(true, tile_shape)
            };
        if window_size > 0 {
            key_valid = key_valid
                & cmpi(
                    key_tile,
                    query_index - broadcast_scalar(window_size, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    key_tile,
                    query_index + broadcast_scalar(window_size, tile_shape),
                    predicate::LessThanOrEqual,
                );
        }
        key_valid
    }

    fn softcapped_attention_score(
        score: Tile<f32, { [128] }>,
        soft_cap: f32,
        has_soft_cap: i32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        if has_soft_cap != 0 {
            let cap = broadcast_scalar(soft_cap, tile_shape);
            tanh(score / cap) * cap
        } else {
            score
        }
    }

    fn fmha_decode_lse_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(lse, lse_offsets, max_score + log(denom), valid & first_dim);
    }

    fn softcapped_window_decode_lse_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let query_position = key_len - 1;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_decode_key_valid(valid, key_index, query_position, window_size);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_decode_key_valid(valid, key_index, query_position, window_size);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_values = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_values, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(
            lse,
            lse_offsets,
            max_score + log(denom),
            valid & has_values & first_dim,
        );
    }

    fn softcapped_window_decode_lse_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let query_position = key_len - 1;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_decode_key_valid(valid, key_index, query_position, window_size);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_decode_key_valid(valid, key_index, query_position, window_size);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_values = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_values, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(
            lse,
            lse_offsets,
            max_score + log(denom),
            valid & has_values & first_dim,
        );
    }

    fn softcapped_window_decode_lse_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let query_position = key_len - 1;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_decode_key_valid(valid, key_index, query_position, window_size);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_decode_key_valid(valid, key_index, query_position, window_size);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_values = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_values, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(
            lse,
            lse_offsets,
            max_score + log(denom),
            valid & has_values & first_dim,
        );
    }

    fn softcapped_window_decode_key_valid(
        valid: Tile<bool, { [128] }>,
        key_index: i32,
        query_position: i32,
        window_size: i32,
    ) -> Tile<bool, { [128] }> {
        let tile_shape = const_shape![128];
        let mut key_valid = valid;
        if window_size > 0 {
            key_valid = key_valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    broadcast_scalar(query_position - window_size, tile_shape),
                    predicate::GreaterThanOrEqual,
                );
        }
        key_valid
    }

    fn fmha_decode_splitk_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_keys = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = sum / denom;
        let output = select(valid & has_keys, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_keys, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn softcapped_window_decode_splitk_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);
        let query_position = key_len - 1;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = softcapped_window_decode_split_key_valid(
                valid,
                key_index,
                key_len,
                query_position,
                window_size,
            );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = softcapped_window_decode_split_key_valid(
                valid,
                key_index,
                key_len,
                query_position,
                window_size,
            );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_keys = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_keys, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_keys, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn softcapped_window_decode_splitk_f16<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);
        let query_position = key_len - 1;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = softcapped_window_decode_split_key_valid(
                valid,
                key_index,
                key_len,
                query_position,
                window_size,
            );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = softcapped_window_decode_split_key_valid(
                valid,
                key_index,
                key_len,
                query_position,
                window_size,
            );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_keys = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_keys, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_keys, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn softcapped_window_decode_splitk_bf16<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);
        let query_position = key_len - 1;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = softcapped_window_decode_split_key_valid(
                valid,
                key_index,
                key_len,
                query_position,
                window_size,
            );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = softcapped_window_decode_split_key_valid(
                valid,
                key_index,
                key_len,
                query_position,
                window_size,
            );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_keys = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_keys, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_keys, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn softcapped_window_decode_split_key_valid(
        valid: Tile<bool, { [128] }>,
        key_index: Tile<i32, { [128] }>,
        key_len: i32,
        query_position: i32,
        window_size: i32,
    ) -> Tile<bool, { [128] }> {
        let tile_shape = const_shape![128];
        let mut key_valid = valid
            & cmpi(
                key_index,
                broadcast_scalar(key_len, tile_shape),
                predicate::LessThan,
            );
        if window_size > 0 {
            key_valid = key_valid
                & cmpi(
                    key_index,
                    broadcast_scalar(query_position - window_size, tile_shape),
                    predicate::GreaterThanOrEqual,
                );
        }
        key_valid
    }

    fn attention_sink_decode_splitk_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        sinks: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        start_q: i32,
        window: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);
        let start_q_tile = broadcast_scalar(start_q, tile_shape);
        let window_start = if window > 0 {
            broadcast_scalar(start_q + 1 - window, tile_shape)
        } else {
            broadcast_scalar(0i32, tile_shape)
        };

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                )
                & cmpi(key_index, start_q_tile, predicate::LessThanOrEqual)
                & cmpi(key_index, window_start, predicate::GreaterThanOrEqual);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let first_split = cmpi(split, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let sink = load_vector(sinks, head, valid & first_split, -1.0e20f32);
        max_score = select(
            valid
                & first_split
                & cmpf(
                    sink,
                    max_score,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                ),
            sink,
            max_score,
        );

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                )
                & cmpi(key_index, start_q_tile, predicate::LessThanOrEqual)
                & cmpi(key_index, window_start, predicate::GreaterThanOrEqual);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let sink_weight = exp(sink - max_score);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let sink_weight = select(valid & first_split, sink_weight, zero);
        denom = denom + sink_weight;

        let has_terms = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = sum / denom;
        let output = select(valid & has_terms, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_terms, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn attention_sink_decode_splitk_f16<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        sinks: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        start_q: i32,
        window: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);
        let start_q_tile = broadcast_scalar(start_q, tile_shape);
        let window_start = if window > 0 {
            broadcast_scalar(start_q + 1 - window, tile_shape)
        } else {
            broadcast_scalar(0i32, tile_shape)
        };

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                )
                & cmpi(key_index, start_q_tile, predicate::LessThanOrEqual)
                & cmpi(key_index, window_start, predicate::GreaterThanOrEqual);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let first_split = cmpi(split, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let sink = load_vector(sinks, head, valid & first_split, -1.0e20f32);
        max_score = select(
            valid
                & first_split
                & cmpf(
                    sink,
                    max_score,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                ),
            sink,
            max_score,
        );

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                )
                & cmpi(key_index, start_q_tile, predicate::LessThanOrEqual)
                & cmpi(key_index, window_start, predicate::GreaterThanOrEqual);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let sink_weight = exp(sink - max_score);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let sink_weight = select(valid & first_split, sink_weight, zero);
        denom = denom + sink_weight;

        let has_terms = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = sum / denom;
        let output = select(valid & has_terms, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_terms, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn attention_sink_decode_splitk_bf16<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        sinks: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        start_q: i32,
        window: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);
        let start_q_tile = broadcast_scalar(start_q, tile_shape);
        let window_start = if window > 0 {
            broadcast_scalar(start_q + 1 - window, tile_shape)
        } else {
            broadcast_scalar(0i32, tile_shape)
        };

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                )
                & cmpi(key_index, start_q_tile, predicate::LessThanOrEqual)
                & cmpi(key_index, window_start, predicate::GreaterThanOrEqual);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let first_split = cmpi(split, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let sink = load_vector(sinks, head, valid & first_split, -1.0e20f32);
        max_score = select(
            valid
                & first_split
                & cmpf(
                    sink,
                    max_score,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                ),
            sink,
            max_score,
        );

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                )
                & cmpi(key_index, start_q_tile, predicate::LessThanOrEqual)
                & cmpi(key_index, window_start, predicate::GreaterThanOrEqual);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let sink_weight = exp(sink - max_score);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let sink_weight = select(valid & first_split, sink_weight, zero);
        denom = denom + sink_weight;

        let has_terms = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = sum / denom;
        let output = select(valid & has_terms, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_terms, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn fmha_prefill_lse_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let lse_offsets = lse_offsets * broadcast_scalar(_query_len, tile_shape) + query_index;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(lse, lse_offsets, max_score + log(denom), valid & first_dim);
    }

    fn fmha_prefill_lse_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let lse_offsets = lse_offsets * broadcast_scalar(_query_len, tile_shape) + query_index;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(lse, lse_offsets, max_score + log(denom), valid & first_dim);
    }

    fn fmha_prefill_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);
    }

    fn fmha_prefill_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);
    }

    fn attention_sink_prefill_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        sinks: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        start_q: i32,
        window: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let query_pos = broadcast_scalar(start_q, tile_shape) + query_index;
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let window_start = if window == 0 {
            zero_i32
        } else {
            let raw_window_start = query_pos - broadcast_scalar(window - 1, tile_shape);
            select(
                cmpi(raw_window_start, zero_i32, predicate::GreaterThan),
                raw_window_start,
                zero_i32,
            )
        };
        let sink = load_vector(sinks, head, valid, 0.0f32);

        let mut max_score = sink;
        for key_index in 0i32..key_len {
            let mut key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_pos,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            if window != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        window_start,
                        predicate::GreaterThanOrEqual,
                    );
            }
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = exp(sink - max_score);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_pos,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            if window != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        window_start,
                        predicate::GreaterThanOrEqual,
                    );
            }
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);
    }

    fn attention_sink_prefill_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        sinks: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        start_q: i32,
        window: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let query_pos = broadcast_scalar(start_q, tile_shape) + query_index;
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let window_start = if window == 0 {
            zero_i32
        } else {
            let raw_window_start = query_pos - broadcast_scalar(window - 1, tile_shape);
            select(
                cmpi(raw_window_start, zero_i32, predicate::GreaterThan),
                raw_window_start,
                zero_i32,
            )
        };
        let sink = load_vector(sinks, head, valid, 0.0f32);

        let mut max_score = sink;
        for key_index in 0i32..key_len {
            let mut key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_pos,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            if window != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        window_start,
                        predicate::GreaterThanOrEqual,
                    );
            }
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = exp(sink - max_score);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_pos,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            if window != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        window_start,
                        predicate::GreaterThanOrEqual,
                    );
            }
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);
    }

    fn fmha_decode_lse_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(lse, lse_offsets, max_score + log(denom), valid & first_dim);
    }

    fn fmha_decode_lse_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(lse, lse_offsets, max_score + log(denom), valid & first_dim);
    }

    fn fmha_decode_splitk_f16<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_keys = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = sum / denom;
        let output = select(valid & has_keys, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_keys, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn fmha_decode_splitk_bf16<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_keys = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = sum / denom;
        let output = select(valid & has_keys, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_keys, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn sliding_window_attention_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_start: i32,
        key_start: i32,
        window: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        output_scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_abs = query_index + broadcast_scalar(query_start, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }
        let output = (sum / denom) * broadcast_scalar(output_scale, tile_shape);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);
    }

    fn sliding_window_attention_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_start: i32,
        key_start: i32,
        window: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        output_scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_abs = query_index + broadcast_scalar(query_start, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }
        let output = (sum / denom) * broadcast_scalar(output_scale, tile_shape);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);
    }

    fn attention_score_f16<const HEAD_DIM: i32>(
        query: *mut f16,
        key: *mut f16,
        query_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        query_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        key_index: i32,
        kv_head: Tile<i32, { [128] }>,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim in 0i32..HEAD_DIM {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(query_sequence_stride, tile_shape)
                + head * broadcast_scalar(query_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let key_offsets = key_batch_base
                + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let q = load_vector_f16_as_f32(query, query_offsets, valid);
            let k = load_vector_f16_as_f32(key, key_offsets, valid);
            score = score + q * k;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn attention_score_bf16<const HEAD_DIM: i32>(
        query: *mut bf16,
        key: *mut bf16,
        query_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        query_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        key_index: i32,
        kv_head: Tile<i32, { [128] }>,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim in 0i32..HEAD_DIM {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(query_sequence_stride, tile_shape)
                + head * broadcast_scalar(query_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let key_offsets = key_batch_base
                + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let q = load_vector_bf16_as_f32(query, query_offsets, valid);
            let k = load_vector_bf16_as_f32(key, key_offsets, valid);
            score = score + q * k;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn sliding_window_attention_paged_kv_f32<const HEAD_DIM: i32, const WINDOW: i32>(
        out: *mut f32,
        query: *mut f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        block_table: *mut u32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_start: i32,
        key_start: i32,
        window: i32,
        query_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        key_cache_block_stride: i32,
        value_cache_block_stride: i32,
        scale: f32,
        output_scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_abs = query_index + broadcast_scalar(query_start, tile_shape);
        let effective_window = if WINDOW == 0 { window } else { WINDOW };
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + effective_window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                key_cache_block_stride,
                query_index,
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + effective_window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                key_cache_block_stride,
                query_index,
                head,
                kv_head,
                block_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(block_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value_cache, value_offsets, block_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }
        let output = (sum / denom) * broadcast_scalar(output_scale, tile_shape);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);
    }

    fn paged_kv_decode_attention_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        query: *mut f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        actual_seq_lens: *mut i32,
        block_table: *mut u32,
        _batch: i32,
        heads: i32,
        kv_heads: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(features, tile_shape);
        let batch_output_offset = offsets - batch_index * broadcast_scalar(features, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let seq_len = load_i32_vector(actual_seq_lens, batch_index, valid, 0i32);
        let key_sequence_stride = kv_heads * HEAD_DIM;
        let key_head_stride = HEAD_DIM;
        let key_cache_block_stride = block_size * key_sequence_stride;
        let value_cache_block_stride = key_cache_block_stride;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len,
                    predicate::LessThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged::<HEAD_DIM>(
                query,
                key_cache,
                batch_index * broadcast_scalar(features, tile_shape),
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len,
                    predicate::LessThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged::<HEAD_DIM>(
                query,
                key_cache,
                batch_index * broadcast_scalar(features, tile_shape),
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(block_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value_cache, value_offsets, block_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector(out, offsets, output, valid);
    }

    fn paged_kv_decode_attention_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        query: *mut f16,
        key_cache: *mut f16,
        value_cache: *mut f16,
        actual_seq_lens: *mut i32,
        block_table: *mut u32,
        _batch: i32,
        heads: i32,
        kv_heads: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(features, tile_shape);
        let batch_output_offset = offsets - batch_index * broadcast_scalar(features, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let seq_len = load_i32_vector(actual_seq_lens, batch_index, valid, 0i32);
        let key_sequence_stride = kv_heads * HEAD_DIM;
        let key_head_stride = HEAD_DIM;
        let key_cache_block_stride = block_size * key_sequence_stride;
        let value_cache_block_stride = key_cache_block_stride;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len,
                    predicate::LessThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_f16::<HEAD_DIM>(
                query,
                key_cache,
                batch_index * broadcast_scalar(features, tile_shape),
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len,
                    predicate::LessThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_f16::<HEAD_DIM>(
                query,
                key_cache,
                batch_index * broadcast_scalar(features, tile_shape),
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(block_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value_cache, value_offsets, block_valid);
            sum = sum + weight * value_tile;
        }
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector_f16_from_f32(out, offsets, output, valid);
    }

    fn paged_kv_decode_attention_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        query: *mut bf16,
        key_cache: *mut bf16,
        value_cache: *mut bf16,
        actual_seq_lens: *mut i32,
        block_table: *mut u32,
        _batch: i32,
        heads: i32,
        kv_heads: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(features, tile_shape);
        let batch_output_offset = offsets - batch_index * broadcast_scalar(features, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let seq_len = load_i32_vector(actual_seq_lens, batch_index, valid, 0i32);
        let key_sequence_stride = kv_heads * HEAD_DIM;
        let key_head_stride = HEAD_DIM;
        let key_cache_block_stride = block_size * key_sequence_stride;
        let value_cache_block_stride = key_cache_block_stride;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len,
                    predicate::LessThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_bf16::<HEAD_DIM>(
                query,
                key_cache,
                batch_index * broadcast_scalar(features, tile_shape),
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len,
                    predicate::LessThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_bf16::<HEAD_DIM>(
                query,
                key_cache,
                batch_index * broadcast_scalar(features, tile_shape),
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(block_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value_cache, value_offsets, block_valid);
            sum = sum + weight * value_tile;
        }
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector_bf16_from_f32(out, offsets, output, valid);
    }

    fn paged_kv_prefill_attention_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        actual_seq_lens_q: *mut i32,
        actual_seq_lens_kv: *mut i32,
        actual_seq_offsets: *mut i32,
        block_table: *mut u32,
        batch: i32,
        _total_query_tokens: i32,
        heads: i32,
        kv_heads: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        causal: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let global_token = offsets / broadcast_scalar(features, tile_shape);
        let feature = offsets - global_token * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);

        let mut batch_index = broadcast_scalar(-1i32, tile_shape);
        let mut query_index = broadcast_scalar(0i32, tile_shape);
        let mut seq_len_q = broadcast_scalar(0i32, tile_shape);
        for candidate in 0i32..batch {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let seq_offset = load_i32_vector(actual_seq_offsets, candidate_tile, valid, 0i32);
            let candidate_seq_len = load_i32_vector(actual_seq_lens_q, candidate_tile, valid, 0i32);
            let in_batch = valid
                & cmpi(global_token, seq_offset, predicate::GreaterThanOrEqual)
                & cmpi(
                    global_token,
                    seq_offset + candidate_seq_len,
                    predicate::LessThan,
                );
            batch_index = select(in_batch, candidate_tile, batch_index);
            query_index = select(in_batch, global_token - seq_offset, query_index);
            seq_len_q = select(in_batch, candidate_seq_len, seq_len_q);
        }
        let valid_query = valid
            & cmpi(
                batch_index,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            )
            & cmpi(query_index, seq_len_q, predicate::LessThan);
        let seq_len_kv = load_i32_vector(actual_seq_lens_kv, batch_index, valid_query, 0i32);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let key_sequence_stride = kv_heads * HEAD_DIM;
        let key_head_stride = HEAD_DIM;
        let key_cache_block_stride = block_size * key_sequence_stride;
        let value_cache_block_stride = key_cache_block_stride;
        let query_batch_base = global_token * broadcast_scalar(features, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut denom = zero;
        let mut sum = zero;
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            let weight = select(block_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value_cache, value_offsets, block_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid_query & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector(out, offsets, output, valid);

        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let lse_offsets = global_token * broadcast_scalar(heads, tile_shape) + head;
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_denominator, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid_query & first_dim);
    }

    fn paged_kv_prefill_attention_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key_cache: *mut f16,
        value_cache: *mut f16,
        actual_seq_lens_q: *mut i32,
        actual_seq_lens_kv: *mut i32,
        actual_seq_offsets: *mut i32,
        block_table: *mut u32,
        batch: i32,
        _total_query_tokens: i32,
        heads: i32,
        kv_heads: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        causal: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let global_token = offsets / broadcast_scalar(features, tile_shape);
        let feature = offsets - global_token * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);

        let mut batch_index = broadcast_scalar(-1i32, tile_shape);
        let mut query_index = broadcast_scalar(0i32, tile_shape);
        let mut seq_len_q = broadcast_scalar(0i32, tile_shape);
        for candidate in 0i32..batch {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let seq_offset = load_i32_vector(actual_seq_offsets, candidate_tile, valid, 0i32);
            let candidate_seq_len = load_i32_vector(actual_seq_lens_q, candidate_tile, valid, 0i32);
            let in_batch = valid
                & cmpi(global_token, seq_offset, predicate::GreaterThanOrEqual)
                & cmpi(
                    global_token,
                    seq_offset + candidate_seq_len,
                    predicate::LessThan,
                );
            batch_index = select(in_batch, candidate_tile, batch_index);
            query_index = select(in_batch, global_token - seq_offset, query_index);
            seq_len_q = select(in_batch, candidate_seq_len, seq_len_q);
        }
        let valid_query = valid
            & cmpi(
                batch_index,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            )
            & cmpi(query_index, seq_len_q, predicate::LessThan);
        let seq_len_kv = load_i32_vector(actual_seq_lens_kv, batch_index, valid_query, 0i32);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let key_sequence_stride = kv_heads * HEAD_DIM;
        let key_head_stride = HEAD_DIM;
        let key_cache_block_stride = block_size * key_sequence_stride;
        let value_cache_block_stride = key_cache_block_stride;
        let query_batch_base = global_token * broadcast_scalar(features, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_f16::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut denom = zero;
        let mut sum = zero;
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_f16::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            let weight = select(block_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value_cache, value_offsets, block_valid);
            sum = sum + weight * value_tile;
        }

        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid_query & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector_f16_from_f32(out, offsets, output, valid);

        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let lse_offsets = global_token * broadcast_scalar(heads, tile_shape) + head;
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_denominator, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid_query & first_dim);
    }

    fn paged_kv_prefill_attention_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key_cache: *mut bf16,
        value_cache: *mut bf16,
        actual_seq_lens_q: *mut i32,
        actual_seq_lens_kv: *mut i32,
        actual_seq_offsets: *mut i32,
        block_table: *mut u32,
        batch: i32,
        _total_query_tokens: i32,
        heads: i32,
        kv_heads: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        causal: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let global_token = offsets / broadcast_scalar(features, tile_shape);
        let feature = offsets - global_token * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);

        let mut batch_index = broadcast_scalar(-1i32, tile_shape);
        let mut query_index = broadcast_scalar(0i32, tile_shape);
        let mut seq_len_q = broadcast_scalar(0i32, tile_shape);
        for candidate in 0i32..batch {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let seq_offset = load_i32_vector(actual_seq_offsets, candidate_tile, valid, 0i32);
            let candidate_seq_len = load_i32_vector(actual_seq_lens_q, candidate_tile, valid, 0i32);
            let in_batch = valid
                & cmpi(global_token, seq_offset, predicate::GreaterThanOrEqual)
                & cmpi(
                    global_token,
                    seq_offset + candidate_seq_len,
                    predicate::LessThan,
                );
            batch_index = select(in_batch, candidate_tile, batch_index);
            query_index = select(in_batch, global_token - seq_offset, query_index);
            seq_len_q = select(in_batch, candidate_seq_len, seq_len_q);
        }
        let valid_query = valid
            & cmpi(
                batch_index,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            )
            & cmpi(query_index, seq_len_q, predicate::LessThan);
        let seq_len_kv = load_i32_vector(actual_seq_lens_kv, batch_index, valid_query, 0i32);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let key_sequence_stride = kv_heads * HEAD_DIM;
        let key_head_stride = HEAD_DIM;
        let key_cache_block_stride = block_size * key_sequence_stride;
        let value_cache_block_stride = key_cache_block_stride;
        let query_batch_base = global_token * broadcast_scalar(features, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_bf16::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut denom = zero;
        let mut sum = zero;
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_bf16::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            let weight = select(block_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value_cache, value_offsets, block_valid);
            sum = sum + weight * value_tile;
        }

        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid_query & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector_bf16_from_f32(out, offsets, output, valid);

        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let lse_offsets = global_token * broadcast_scalar(heads, tile_shape) + head;
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_denominator, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid_query & first_dim);
    }

    fn ragged_kv_prefill_attention_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        actual_seq_lens_q: *mut i32,
        actual_seq_lens_kv: *mut i32,
        actual_seq_offsets: *mut i32,
        batch: i32,
        _total_query_tokens: i32,
        total_kv_tokens: i32,
        heads: i32,
        kv_heads: i32,
        causal: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let kv_features = kv_heads * HEAD_DIM;
        let global_token = offsets / broadcast_scalar(features, tile_shape);
        let feature = offsets - global_token * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);

        let mut batch_index = broadcast_scalar(-1i32, tile_shape);
        let mut query_index = broadcast_scalar(0i32, tile_shape);
        let mut seq_offset = broadcast_scalar(0i32, tile_shape);
        let mut seq_len_q = broadcast_scalar(0i32, tile_shape);
        for candidate in 0i32..batch {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let candidate_offset = load_i32_vector(actual_seq_offsets, candidate_tile, valid, 0i32);
            let candidate_seq_len = load_i32_vector(actual_seq_lens_q, candidate_tile, valid, 0i32);
            let in_batch = valid
                & cmpi(
                    global_token,
                    candidate_offset,
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    global_token,
                    candidate_offset + candidate_seq_len,
                    predicate::LessThan,
                );
            batch_index = select(in_batch, candidate_tile, batch_index);
            query_index = select(in_batch, global_token - candidate_offset, query_index);
            seq_offset = select(in_batch, candidate_offset, seq_offset);
            seq_len_q = select(in_batch, candidate_seq_len, seq_len_q);
        }
        let valid_query = valid
            & cmpi(
                batch_index,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            )
            & cmpi(query_index, seq_len_q, predicate::LessThan);
        let seq_len_kv = load_i32_vector(actual_seq_lens_kv, batch_index, valid_query, 0i32);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = global_token * broadcast_scalar(features, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..total_kv_tokens {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                seq_offset * broadcast_scalar(kv_features, tile_shape),
                0,
                kv_features,
                HEAD_DIM,
                HEAD_DIM,
                broadcast_scalar(0i32, tile_shape),
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut denom = zero;
        let mut sum = zero;
        for key_index in 0i32..total_kv_tokens {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                seq_offset * broadcast_scalar(kv_features, tile_shape),
                0,
                kv_features,
                HEAD_DIM,
                HEAD_DIM,
                broadcast_scalar(0i32, tile_shape),
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let weight = select(key_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = (seq_offset + broadcast_scalar(key_index, tile_shape))
                * broadcast_scalar(kv_features, tile_shape)
                + kv_head * broadcast_scalar(HEAD_DIM, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid_query & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector(out, offsets, output, valid);

        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let lse_offsets = global_token * broadcast_scalar(heads, tile_shape) + head;
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_denominator, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid_query & first_dim);
    }

    fn ragged_kv_prefill_attention_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        actual_seq_lens_q: *mut i32,
        actual_seq_lens_kv: *mut i32,
        actual_seq_offsets: *mut i32,
        batch: i32,
        _total_query_tokens: i32,
        total_kv_tokens: i32,
        heads: i32,
        kv_heads: i32,
        causal: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let kv_features = kv_heads * HEAD_DIM;
        let global_token = offsets / broadcast_scalar(features, tile_shape);
        let feature = offsets - global_token * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);

        let mut batch_index = broadcast_scalar(-1i32, tile_shape);
        let mut query_index = broadcast_scalar(0i32, tile_shape);
        let mut seq_offset = broadcast_scalar(0i32, tile_shape);
        let mut seq_len_q = broadcast_scalar(0i32, tile_shape);
        for candidate in 0i32..batch {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let candidate_offset = load_i32_vector(actual_seq_offsets, candidate_tile, valid, 0i32);
            let candidate_seq_len = load_i32_vector(actual_seq_lens_q, candidate_tile, valid, 0i32);
            let in_batch = valid
                & cmpi(
                    global_token,
                    candidate_offset,
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    global_token,
                    candidate_offset + candidate_seq_len,
                    predicate::LessThan,
                );
            batch_index = select(in_batch, candidate_tile, batch_index);
            query_index = select(in_batch, global_token - candidate_offset, query_index);
            seq_offset = select(in_batch, candidate_offset, seq_offset);
            seq_len_q = select(in_batch, candidate_seq_len, seq_len_q);
        }
        let valid_query = valid
            & cmpi(
                batch_index,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            )
            & cmpi(query_index, seq_len_q, predicate::LessThan);
        let seq_len_kv = load_i32_vector(actual_seq_lens_kv, batch_index, valid_query, 0i32);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = global_token * broadcast_scalar(features, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..total_kv_tokens {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                seq_offset * broadcast_scalar(kv_features, tile_shape),
                0,
                kv_features,
                HEAD_DIM,
                HEAD_DIM,
                broadcast_scalar(0i32, tile_shape),
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut denom = zero;
        let mut sum = zero;
        for key_index in 0i32..total_kv_tokens {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                seq_offset * broadcast_scalar(kv_features, tile_shape),
                0,
                kv_features,
                HEAD_DIM,
                HEAD_DIM,
                broadcast_scalar(0i32, tile_shape),
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = select(key_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = (seq_offset + broadcast_scalar(key_index, tile_shape))
                * broadcast_scalar(kv_features, tile_shape)
                + kv_head * broadcast_scalar(HEAD_DIM, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid_query & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector_f16_from_f32(out, offsets, output, valid);

        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let lse_offsets = global_token * broadcast_scalar(heads, tile_shape) + head;
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_denominator, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid_query & first_dim);
    }

    fn ragged_kv_prefill_attention_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        actual_seq_lens_q: *mut i32,
        actual_seq_lens_kv: *mut i32,
        actual_seq_offsets: *mut i32,
        batch: i32,
        _total_query_tokens: i32,
        total_kv_tokens: i32,
        heads: i32,
        kv_heads: i32,
        causal: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let kv_features = kv_heads * HEAD_DIM;
        let global_token = offsets / broadcast_scalar(features, tile_shape);
        let feature = offsets - global_token * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);

        let mut batch_index = broadcast_scalar(-1i32, tile_shape);
        let mut query_index = broadcast_scalar(0i32, tile_shape);
        let mut seq_offset = broadcast_scalar(0i32, tile_shape);
        let mut seq_len_q = broadcast_scalar(0i32, tile_shape);
        for candidate in 0i32..batch {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let candidate_offset = load_i32_vector(actual_seq_offsets, candidate_tile, valid, 0i32);
            let candidate_seq_len = load_i32_vector(actual_seq_lens_q, candidate_tile, valid, 0i32);
            let in_batch = valid
                & cmpi(
                    global_token,
                    candidate_offset,
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    global_token,
                    candidate_offset + candidate_seq_len,
                    predicate::LessThan,
                );
            batch_index = select(in_batch, candidate_tile, batch_index);
            query_index = select(in_batch, global_token - candidate_offset, query_index);
            seq_offset = select(in_batch, candidate_offset, seq_offset);
            seq_len_q = select(in_batch, candidate_seq_len, seq_len_q);
        }
        let valid_query = valid
            & cmpi(
                batch_index,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            )
            & cmpi(query_index, seq_len_q, predicate::LessThan);
        let seq_len_kv = load_i32_vector(actual_seq_lens_kv, batch_index, valid_query, 0i32);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = global_token * broadcast_scalar(features, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..total_kv_tokens {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                seq_offset * broadcast_scalar(kv_features, tile_shape),
                0,
                kv_features,
                HEAD_DIM,
                HEAD_DIM,
                broadcast_scalar(0i32, tile_shape),
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut denom = zero;
        let mut sum = zero;
        for key_index in 0i32..total_kv_tokens {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                seq_offset * broadcast_scalar(kv_features, tile_shape),
                0,
                kv_features,
                HEAD_DIM,
                HEAD_DIM,
                broadcast_scalar(0i32, tile_shape),
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = select(key_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = (seq_offset + broadcast_scalar(key_index, tile_shape))
                * broadcast_scalar(kv_features, tile_shape)
                + kv_head * broadcast_scalar(HEAD_DIM, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid_query & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector_bf16_from_f32(out, offsets, output, valid);

        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let lse_offsets = global_token * broadcast_scalar(heads, tile_shape) + head;
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_denominator, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid_query & first_dim);
    }

    fn attention_score<const HEAD_DIM: i32>(
        query: *mut f32,
        key: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        query_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        key_index: i32,
        kv_head: Tile<i32, { [128] }>,
        _heads: i32,
        _kv_heads: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim in 0i32..HEAD_DIM {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(query_sequence_stride, tile_shape)
                + head * broadcast_scalar(query_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let key_offsets = key_batch_base
                + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let q = load_vector(query, query_offsets, valid, 0.0f32);
            let k = load_vector(key, key_offsets, valid, 0.0f32);
            score = score + q * k;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn attention_score_paged<const HEAD_DIM: i32>(
        query: *mut f32,
        key_cache: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        physical_block: Tile<i32, { [128] }>,
        block_token: Tile<i32, { [128] }>,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        key_cache_block_stride: i32,
        query_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        kv_head: Tile<i32, { [128] }>,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim in 0i32..HEAD_DIM {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(query_sequence_stride, tile_shape)
                + head * broadcast_scalar(query_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let key_offsets = physical_block * broadcast_scalar(key_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let q = load_vector(query, query_offsets, valid, 0.0f32);
            let k = load_vector(key_cache, key_offsets, valid, 0.0f32);
            score = score + q * k;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn attention_score_paged_f16<const HEAD_DIM: i32>(
        query: *mut f16,
        key_cache: *mut f16,
        query_batch_base: Tile<i32, { [128] }>,
        physical_block: Tile<i32, { [128] }>,
        block_token: Tile<i32, { [128] }>,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        key_cache_block_stride: i32,
        query_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        kv_head: Tile<i32, { [128] }>,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim in 0i32..HEAD_DIM {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(query_sequence_stride, tile_shape)
                + head * broadcast_scalar(query_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let key_offsets = physical_block * broadcast_scalar(key_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let q = load_vector_f16_as_f32(query, query_offsets, valid);
            let k = load_vector_f16_as_f32(key_cache, key_offsets, valid);
            score = score + q * k;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn attention_score_paged_bf16<const HEAD_DIM: i32>(
        query: *mut bf16,
        key_cache: *mut bf16,
        query_batch_base: Tile<i32, { [128] }>,
        physical_block: Tile<i32, { [128] }>,
        block_token: Tile<i32, { [128] }>,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        key_cache_block_stride: i32,
        query_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        kv_head: Tile<i32, { [128] }>,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim in 0i32..HEAD_DIM {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(query_sequence_stride, tile_shape)
                + head * broadcast_scalar(query_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let key_offsets = physical_block * broadcast_scalar(key_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let q = load_vector_bf16_as_f32(query, query_offsets, valid);
            let k = load_vector_bf16_as_f32(key_cache, key_offsets, valid);
            score = score + q * k;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn paged_block_lookup(
        block_table: *mut u32,
        batch_index: Tile<i32, { [128] }>,
        key_index: i32,
        block_size: i32,
        block_table_batch_stride: i32,
        valid: Tile<bool, { [128] }>,
    ) -> (Tile<i32, { [128] }>, Tile<i32, { [128] }>) {
        let tile_shape = const_shape![128];
        let logical_block = key_index / block_size;
        let block_token = key_index - logical_block * block_size;
        let block_table_offsets = batch_index
            * broadcast_scalar(block_table_batch_stride, tile_shape)
            + broadcast_scalar(logical_block, tile_shape);
        let physical_block_u32 = load_u32_vector(block_table, block_table_offsets, valid, 0u32);
        let physical_block: Tile<i32, { [128] }> = bitcast(physical_block_u32);
        (physical_block, broadcast_scalar(block_token, tile_shape))
    }

    fn softmax_probability_f32(
        scores: *mut f32,
        score_base: Tile<i32, { [128] }>,
        score_value: Tile<f32, { [128] }>,
        input_row: Tile<i32, { [128] }>,
        seq_len: i32,
        valid_row: Tile<bool, { [128] }>,
        valid_input: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut max_score = constant(-1.0e30f32, tile_shape);
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector(scores, score_base + score_col_tile, valid_score, 0.0f32);
            max_score = select(
                valid_score
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = zero_values;
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector(scores, score_base + score_col_tile, valid_score, 0.0f32);
            denom = denom + select(valid_score, exp(score - max_score), zero_values);
        }
        select(
            valid_input,
            exp(score_value - max_score) / denom,
            zero_values,
        )
    }

    fn softmax_probability_f16(
        scores: *mut f16,
        score_base: Tile<i32, { [128] }>,
        score_value: Tile<f32, { [128] }>,
        input_row: Tile<i32, { [128] }>,
        seq_len: i32,
        valid_row: Tile<bool, { [128] }>,
        valid_input: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut max_score = constant(-1.0e30f32, tile_shape);
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector_f16_as_f32(scores, score_base + score_col_tile, valid_score);
            max_score = select(
                valid_score
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = zero_values;
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector_f16_as_f32(scores, score_base + score_col_tile, valid_score);
            denom = denom + select(valid_score, exp(score - max_score), zero_values);
        }
        select(
            valid_input,
            exp(score_value - max_score) / denom,
            zero_values,
        )
    }

    fn softmax_probability_bf16(
        scores: *mut bf16,
        score_base: Tile<i32, { [128] }>,
        score_value: Tile<f32, { [128] }>,
        input_row: Tile<i32, { [128] }>,
        seq_len: i32,
        valid_row: Tile<bool, { [128] }>,
        valid_input: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut max_score = constant(-1.0e30f32, tile_shape);
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector_bf16_as_f32(scores, score_base + score_col_tile, valid_score);
            max_score = select(
                valid_score
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = zero_values;
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector_bf16_as_f32(scores, score_base + score_col_tile, valid_score);
            denom = denom + select(valid_score, exp(score - max_score), zero_values);
        }
        select(
            valid_input,
            exp(score_value - max_score) / denom,
            zero_values,
        )
    }

    fn sparsemax_probability_f32(
        scores: *mut f32,
        score_base: Tile<i32, { [128] }>,
        score_value: Tile<f32, { [128] }>,
        input_row: Tile<i32, { [128] }>,
        seq_len: i32,
        valid_row: Tile<bool, { [128] }>,
        valid_input: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one_values: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let half_values: Tile<f32, { [128] }> = constant(0.5f32, tile_shape);
        let mut x_max = constant(-1.0e30f32, tile_shape);
        let mut x_sum = zero_values;
        let mut support_count = zero_values;
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector(scores, score_base + score_col_tile, valid_score, 0.0f32);
            x_sum = x_sum + select(valid_score, score, zero_values);
            support_count = support_count + select(valid_score, one_values, zero_values);
            x_max = select(
                valid_score & cmpf(score, x_max, predicate::GreaterThan, cmp_ordering::Ordered),
                score,
                x_max,
            );
        }

        let mut tau_lo = (x_sum - one_values) / support_count;
        let mut tau_hi = x_max;
        for _ in 0i32..20i32 {
            let tau_mid = half_values * (tau_lo + tau_hi);
            let mut sum_active = zero_values;
            for score_col in 0i32..seq_len {
                let score_col_tile = broadcast_scalar(score_col, tile_shape);
                let valid_score =
                    valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
                let score = load_vector(scores, score_base + score_col_tile, valid_score, 0.0f32);
                let active = valid_score
                    & cmpf(
                        score,
                        tau_mid,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    );
                sum_active = sum_active + select(active, score - tau_mid, zero_values);
            }
            let above_one = cmpf(
                sum_active,
                one_values,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            tau_lo = select(above_one, tau_mid, tau_lo);
            tau_hi = select(above_one, tau_hi, tau_mid);
        }

        let tau = half_values * (tau_lo + tau_hi);
        let value = score_value - tau;
        select(
            valid_input
                & cmpf(
                    value,
                    zero_values,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                ),
            value,
            zero_values,
        )
    }

    fn sparsemax_probability_f16(
        scores: *mut f16,
        score_base: Tile<i32, { [128] }>,
        score_value: Tile<f32, { [128] }>,
        input_row: Tile<i32, { [128] }>,
        seq_len: i32,
        valid_row: Tile<bool, { [128] }>,
        valid_input: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one_values: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let half_values: Tile<f32, { [128] }> = constant(0.5f32, tile_shape);
        let mut x_max = constant(-1.0e30f32, tile_shape);
        let mut x_sum = zero_values;
        let mut support_count = zero_values;
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector_f16_as_f32(scores, score_base + score_col_tile, valid_score);
            x_sum = x_sum + select(valid_score, score, zero_values);
            support_count = support_count + select(valid_score, one_values, zero_values);
            x_max = select(
                valid_score & cmpf(score, x_max, predicate::GreaterThan, cmp_ordering::Ordered),
                score,
                x_max,
            );
        }

        let mut tau_lo = (x_sum - one_values) / support_count;
        let mut tau_hi = x_max;
        for _ in 0i32..20i32 {
            let tau_mid = half_values * (tau_lo + tau_hi);
            let mut sum_active = zero_values;
            for score_col in 0i32..seq_len {
                let score_col_tile = broadcast_scalar(score_col, tile_shape);
                let valid_score =
                    valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
                let score =
                    load_vector_f16_as_f32(scores, score_base + score_col_tile, valid_score);
                let active = valid_score
                    & cmpf(
                        score,
                        tau_mid,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    );
                sum_active = sum_active + select(active, score - tau_mid, zero_values);
            }
            let above_one = cmpf(
                sum_active,
                one_values,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            tau_lo = select(above_one, tau_mid, tau_lo);
            tau_hi = select(above_one, tau_hi, tau_mid);
        }

        let tau = half_values * (tau_lo + tau_hi);
        let value = score_value - tau;
        select(
            valid_input
                & cmpf(
                    value,
                    zero_values,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                ),
            value,
            zero_values,
        )
    }

    fn sparsemax_probability_bf16(
        scores: *mut bf16,
        score_base: Tile<i32, { [128] }>,
        score_value: Tile<f32, { [128] }>,
        input_row: Tile<i32, { [128] }>,
        seq_len: i32,
        valid_row: Tile<bool, { [128] }>,
        valid_input: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one_values: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let half_values: Tile<f32, { [128] }> = constant(0.5f32, tile_shape);
        let mut x_max = constant(-1.0e30f32, tile_shape);
        let mut x_sum = zero_values;
        let mut support_count = zero_values;
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector_bf16_as_f32(scores, score_base + score_col_tile, valid_score);
            x_sum = x_sum + select(valid_score, score, zero_values);
            support_count = support_count + select(valid_score, one_values, zero_values);
            x_max = select(
                valid_score & cmpf(score, x_max, predicate::GreaterThan, cmp_ordering::Ordered),
                score,
                x_max,
            );
        }

        let mut tau_lo = (x_sum - one_values) / support_count;
        let mut tau_hi = x_max;
        for _ in 0i32..20i32 {
            let tau_mid = half_values * (tau_lo + tau_hi);
            let mut sum_active = zero_values;
            for score_col in 0i32..seq_len {
                let score_col_tile = broadcast_scalar(score_col, tile_shape);
                let valid_score =
                    valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
                let score =
                    load_vector_bf16_as_f32(scores, score_base + score_col_tile, valid_score);
                let active = valid_score
                    & cmpf(
                        score,
                        tau_mid,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    );
                sum_active = sum_active + select(active, score - tau_mid, zero_values);
            }
            let above_one = cmpf(
                sum_active,
                one_values,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            tau_lo = select(above_one, tau_mid, tau_lo);
            tau_hi = select(above_one, tau_hi, tau_mid);
        }

        let tau = half_values * (tau_lo + tau_hi);
        let value = score_value - tau;
        select(
            valid_input
                & cmpf(
                    value,
                    zero_values,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                ),
            value,
            zero_values,
        )
    }

    fn store_i32_vector(
        out: *mut i32,
        offsets: Tile<i32, { [128] }>,
        values: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let out_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut i32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut i32, { [128] }> = out_base.broadcast(const_shape![128]);
        let out_ptrs: PointerTile<*mut i32, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }
}

pub use kernels::*;
