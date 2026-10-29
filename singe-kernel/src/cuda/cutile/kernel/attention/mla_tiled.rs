//! MLA with shared scores and online softmax. MODE selects dense prefill,
//! contiguous decode, sparse prefill, paged decode, or split decode.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    use crate::cuda::cutile::{
        kernel::tiled::kernels::{load_1d, load_2d, store_1d, store_2d},
        runtime as cutile,
    };

    unsafe fn add_score_term<T: ElementType, const BM: i32, const R: i32>(
        scores: Tile<f32, { [BM, 32] }>,
        resident: Tile<T, { [BM, R] }>,
        qptr: *mut T,
        kptr: *mut T,
        base: Tile<i32, { [BM] }>,
        key_row: Tile<i32, { [32] }>,
        qvalid: Tile<bool, { [BM] }>,
        valid_key: Tile<bool, { [32] }>,
        width: i32,
    ) -> Tile<f32, { [BM, 32] }> {
        let lane: Tile<i32, { [32] }> = iota(const_shape![32]);
        let mut result = scores;
        for chunk in 0i32..ceil_div(width, 32i32) {
            let channel_start: Tile<i32, { [32] }> =
                broadcast_scalar(chunk * 32i32, const_shape![32]);
            let channel = lane + channel_start;
            let channel_limit: Tile<i32, { [32] }> = broadcast_scalar(width, const_shape![32]);
            let channel_valid = lt_tile(channel, channel_limit);
            let qoffset = base
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, 32])
                + channel
                    .reshape(const_shape![1, 32])
                    .broadcast(const_shape![BM, 32]);
            let qvalid_tile = qvalid
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, 32])
                & channel_valid
                    .reshape(const_shape![1, 32])
                    .broadcast(const_shape![BM, 32]);
            let q: Tile<T, { [BM, 32] }> = if width <= R {
                let zero: Tile<i32, { [] }> = constant(0i32, const_shape![]);
                let slice_index: Tile<i32, { [] }> = scalar_to_tile(chunk);
                extract(resident, [zero, slice_index])
            } else {
                unsafe { load_2d(qptr, qoffset, qvalid_tile) }
            };
            let zero_query_float: Tile<f32, { [BM, 32] }> = constant(0.0f32, const_shape![BM, 32]);
            let zero_query: Tile<T, { [BM, 32] }> = convert_tile(zero_query_float);
            let q = select(qvalid_tile, q, zero_query);
            let key_stride: Tile<i32, { [32] }> = broadcast_scalar(width, const_shape![32]);
            let koffset = (key_row * key_stride)
                .reshape(const_shape![32, 1])
                .broadcast(const_shape![32, 32])
                + channel
                    .reshape(const_shape![1, 32])
                    .broadcast(const_shape![32, 32]);
            let kmask = valid_key
                .reshape(const_shape![32, 1])
                .broadcast(const_shape![32, 32])
                & channel_valid
                    .reshape(const_shape![1, 32])
                    .broadcast(const_shape![32, 32]);
            let k: Tile<T, { [32, 32] }> = unsafe { load_2d(kptr, koffset, kmask) };
            let zero_key_float: Tile<f32, { [32, 32] }> = constant(0.0f32, const_shape![32, 32]);
            let zero_key: Tile<T, { [32, 32] }> = convert_tile(zero_key_float);
            let k = select(kmask, k, zero_key);
            if BM == 16 {
                let update: Tile<f32, { [BM, 32] }> = mma(q, k.transpose(), result);
                result = update.reshape(const_shape![BM, 32]);
            } else {
                let query_float: Tile<f32, { [BM, 32] }> = convert_tile(q);
                let query_broadcast: Tile<f32, { [32, 32] }> = query_float
                    .reshape(const_shape![1, 32])
                    .broadcast(const_shape![32, 32]);
                let key_float: Tile<f32, { [32, 32] }> = convert_tile(k);
                let dot: Tile<f32, { [32] }> = reduce_sum(query_broadcast * key_float, 1i32);
                result = result
                    + dot
                        .reshape(const_shape![1, 32])
                        .broadcast(const_shape![BM, 32]);
            }
        }
        result
    }

    #[cutile::entry()]
    pub unsafe fn mla<
        T: ElementType,
        O: ElementType,
        const BM: i32,
        const D: i32,
        const P: i32,
        const MODE: i32,
    >(
        out: *mut O,
        lse: *mut f32,
        query: *mut T,
        query_pe: *mut T,
        key: *mut T,
        key_pe: *mut T,
        value: *mut T,
        indices: *mut i32,
        lengths: *mut i32,
        table: *mut u32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        pe_dim: i32,
        topk: i32,
        block_size: i32,
        physical_blocks: i32,
        table_stride: i32,
        splits: i32,
        per_split: i32,
        scale: f32,
        output_scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let batch = pid.0 / heads;
        let head = pid.0 - batch * heads;
        let kv_head = head / (heads / kv_heads);
        let value_tiles = ceil_div(head_dim, D);
        let split = pid.2 / value_tiles;
        let value_tile = pid.2 - split * value_tiles;
        let query_start: Tile<i32, { [BM] }> = broadcast_scalar(pid.1 * BM, const_shape![BM]);
        let qrow: Tile<i32, { [BM] }> = iota(const_shape![BM]) + query_start;
        let query_limit: Tile<i32, { [BM] }> = broadcast_scalar(query_len, const_shape![BM]);
        let qvalid = lt_tile(qrow, query_limit);
        let query_batch: Tile<i32, { [BM] }> =
            broadcast_scalar(batch * query_len, const_shape![BM]);
        let query_stride: Tile<i32, { [BM] }> =
            broadcast_scalar(heads * head_dim, const_shape![BM]);
        let query_head: Tile<i32, { [BM] }> = broadcast_scalar(head * head_dim, const_shape![BM]);
        let content_base = (query_batch + qrow) * query_stride + query_head;
        let position_batch: Tile<i32, { [BM] }> =
            broadcast_scalar(batch * query_len, const_shape![BM]);
        let position_stride: Tile<i32, { [BM] }> =
            broadcast_scalar(heads * pe_dim, const_shape![BM]);
        let position_head: Tile<i32, { [BM] }> = broadcast_scalar(head * pe_dim, const_shape![BM]);
        let position_base = (position_batch + qrow) * position_stride + position_head;
        let d: Tile<i32, { [D] }> = iota(const_shape![D]);
        let p: Tile<i32, { [P] }> = iota(const_shape![P]);
        let qoffset = content_base
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, D])
            + d.reshape(const_shape![1, D]).broadcast(const_shape![BM, D]);
        let content_width: Tile<i32, { [D] }> = broadcast_scalar(head_dim, const_shape![D]);
        let qmask = qvalid
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, D])
            & lt_tile(d, content_width)
                .reshape(const_shape![1, D])
                .broadcast(const_shape![BM, D]);
        let qresident: Tile<T, { [BM, D] }> = unsafe { load_2d(query, qoffset, qmask) };
        let zero_content_float: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);
        let zero_content: Tile<T, { [BM, D] }> = convert_tile(zero_content_float);
        let qresident = select(qmask, qresident, zero_content);
        let poffset = position_base
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, P])
            + p.reshape(const_shape![1, P]).broadcast(const_shape![BM, P]);
        let position_width: Tile<i32, { [P] }> = broadcast_scalar(pe_dim, const_shape![P]);
        let pmask = qvalid
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, P])
            & lt_tile(p, position_width)
                .reshape(const_shape![1, P])
                .broadcast(const_shape![BM, P]);
        let president: Tile<T, { [BM, P] }> = unsafe { load_2d(query_pe, poffset, pmask) };
        let zero_position_float: Tile<f32, { [BM, P] }> = constant(0.0f32, const_shape![BM, P]);
        let zero_position: Tile<T, { [BM, P] }> = convert_tile(zero_position_float);
        let president = select(pmask, president, zero_position);
        let mut limit = key_len;
        let mut key_start = 0i32;
        if MODE == 0 {
            limit = min(key_len, (pid.1 + 1i32) * BM);
        }
        if MODE == 2 {
            limit = topk;
        }
        if MODE == 3 {
            let batch_index: Tile<i32, { [1] }> = broadcast_scalar(batch, const_shape![1]);
            let length_active: Tile<bool, { [1] }> = constant(true, const_shape![1]);
            let length = unsafe { load_1d(lengths, batch_index, length_active) };
            let length: i32 = tile_to_scalar(length.reshape(const_shape![]));
            limit = max(0i32, min(length, table_stride * block_size));
        }
        if MODE == 4 {
            key_start = split * per_split;
            limit = max(0i32, min(per_split, key_len - key_start));
        }
        let mut row_max: Tile<f32, { [BM] }> = constant(f32::NEG_INFINITY, const_shape![BM]);
        let mut row_sum: Tile<f32, { [BM] }> = constant(0.0f32, const_shape![BM]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);
        for tile in 0i32..ceil_div(limit, 32i32) {
            let lane: Tile<i32, { [32] }> = iota(const_shape![32]);
            let key_tile_start: Tile<i32, { [32] }> =
                broadcast_scalar(tile * 32i32, const_shape![32]);
            let index = lane + key_tile_start;
            let key_limit: Tile<i32, { [32] }> = broadcast_scalar(limit, const_shape![32]);
            let mut valid_key = lt_tile(index, key_limit);
            let key_split_start: Tile<i32, { [32] }> =
                broadcast_scalar(key_start, const_shape![32]);
            let mut logical = index + key_split_start;
            if MODE == 2 {
                let sparse_index_base: Tile<i32, { [32] }> = broadcast_scalar(
                    ((batch * query_len + pid.1) * kv_heads + kv_head) * topk,
                    const_shape![32],
                );
                let offset = sparse_index_base + index;
                logical = unsafe { load_1d(indices, offset, valid_key) };
                let zero_key_index: Tile<i32, { [32] }> = constant(0i32, const_shape![32]);
                let context_length: Tile<i32, { [32] }> =
                    broadcast_scalar(key_len, const_shape![32]);
                let causal_query: Tile<i32, { [32] }> = broadcast_scalar(pid.1, const_shape![32]);
                valid_key = valid_key
                    & ge_tile(logical, zero_key_index)
                    & lt_tile(logical, context_length)
                    & le_tile(logical, causal_query);
            }
            let key_batch: Tile<i32, { [32] }> =
                broadcast_scalar(batch * key_len, const_shape![32]);
            let kv_head_count: Tile<i32, { [32] }> = broadcast_scalar(kv_heads, const_shape![32]);
            let kv_head_index: Tile<i32, { [32] }> = broadcast_scalar(kv_head, const_shape![32]);
            let mut key_row = (key_batch + logical) * kv_head_count + kv_head_index;
            if MODE == 3 {
                let page_size: Tile<i32, { [32] }> = broadcast_scalar(block_size, const_shape![32]);
                let block = logical / page_size;
                let page_table_base: Tile<i32, { [32] }> =
                    broadcast_scalar(batch * table_stride, const_shape![32]);
                let physical: Tile<u32, { [32] }> =
                    unsafe { load_1d(table, page_table_base + block, valid_key) };
                let physical: Tile<i32, { [32] }> = bitcast(physical);
                let zero_page: Tile<i32, { [32] }> = constant(0i32, const_shape![32]);
                let page_count: Tile<i32, { [32] }> =
                    broadcast_scalar(physical_blocks, const_shape![32]);
                valid_key =
                    valid_key & ge_tile(physical, zero_page) & lt_tile(physical, page_count);
                let page_token_stride: Tile<i32, { [32] }> =
                    broadcast_scalar(block_size, const_shape![32]);
                let page_token_length: Tile<i32, { [32] }> =
                    broadcast_scalar(block_size, const_shape![32]);
                key_row = physical * page_token_stride + logical - block * page_token_length;
            }
            let mut scores: Tile<f32, { [BM, 32] }> = constant(0.0f32, const_shape![BM, 32]);
            // Content and positional channels contribute to one shared score tile.
            let content: Tile<f32, { [BM, 32] }> = unsafe {
                add_score_term(
                    scores,
                    qresident,
                    query,
                    key,
                    content_base,
                    key_row,
                    qvalid,
                    valid_key,
                    head_dim,
                )
            };
            scores = content.reshape(const_shape![BM, 32]);
            let positional: Tile<f32, { [BM, 32] }> = unsafe {
                add_score_term(
                    scores,
                    president,
                    query_pe,
                    key_pe,
                    position_base,
                    key_row,
                    qvalid,
                    valid_key,
                    pe_dim,
                )
            };
            scores = positional.reshape(const_shape![BM, 32]);
            let mut valid = qvalid
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, 32])
                & valid_key
                    .reshape(const_shape![1, 32])
                    .broadcast(const_shape![BM, 32]);
            if MODE == 0 {
                valid = valid
                    & ge_tile(
                        qrow.reshape(const_shape![BM, 1])
                            .broadcast(const_shape![BM, 32]),
                        logical
                            .reshape(const_shape![1, 32])
                            .broadcast(const_shape![BM, 32]),
                    );
            }
            let score_scale: Tile<f32, { [BM, 32] }> =
                broadcast_scalar(scale, const_shape![BM, 32]);
            let masked_score: Tile<f32, { [BM, 32] }> =
                constant(f32::NEG_INFINITY, const_shape![BM, 32]);
            scores = select(valid, scores * score_scale, masked_score);
            let maximum: Tile<f32, { [BM] }> = reduce_max(scores, 1i32);
            let maximum = maximum.reshape(const_shape![BM]);
            let next = max_tile(row_max, maximum);
            let empty_row_max: Tile<f32, { [BM] }> = constant(f32::NEG_INFINITY, const_shape![BM]);
            let empty_row_zero: Tile<f32, { [BM] }> = constant(0.0f32, const_shape![BM]);
            let safe = select(eq_tile(next, empty_row_max), empty_row_zero, next);
            let alpha = exp(row_max - safe);
            let probability = exp(scores
                - safe
                    .reshape(const_shape![BM, 1])
                    .broadcast(const_shape![BM, 32]));
            let sum: Tile<f32, { [BM] }> = reduce_sum(probability, 1i32);
            let sum = sum.reshape(const_shape![BM]);
            row_sum = row_sum * alpha + sum;
            acc = acc
                * alpha
                    .reshape(const_shape![BM, 1])
                    .broadcast(const_shape![BM, D]);
            let value_tile_start: Tile<i32, { [D] }> =
                broadcast_scalar(value_tile * D, const_shape![D]);
            let output_channel = d + value_tile_start;
            let value_row_stride: Tile<i32, { [32] }> =
                broadcast_scalar(head_dim, const_shape![32]);
            let voffset = (key_row * value_row_stride)
                .reshape(const_shape![32, 1])
                .broadcast(const_shape![32, D])
                + output_channel
                    .reshape(const_shape![1, D])
                    .broadcast(const_shape![32, D]);
            let value_width: Tile<i32, { [D] }> = broadcast_scalar(head_dim, const_shape![D]);
            let vmask = valid_key
                .reshape(const_shape![32, 1])
                .broadcast(const_shape![32, D])
                & lt_tile(output_channel, value_width)
                    .reshape(const_shape![1, D])
                    .broadcast(const_shape![32, D]);
            let v: Tile<T, { [32, D] }> = unsafe { load_2d(value, voffset, vmask) };
            let zero_value_float: Tile<f32, { [32, D] }> = constant(0.0f32, const_shape![32, D]);
            let zero_value: Tile<T, { [32, D] }> = convert_tile(zero_value_float);
            let v = select(vmask, v, zero_value);
            if BM == 16 {
                let probability_storage: Tile<T, { [BM, 32] }> = convert_tile(probability);
                let update: Tile<f32, { [BM, D] }> = mma(probability_storage, v, acc);
                acc = update.reshape(const_shape![BM, D]);
            } else {
                let value_float: Tile<f32, { [32, D] }> = convert_tile(v);
                let weighted = probability
                    .reshape(const_shape![32, 1])
                    .broadcast(const_shape![32, D])
                    * value_float;
                let contribution: Tile<f32, { [D] }> = reduce_sum(weighted, 0i32);
                acc = acc
                    + contribution
                        .reshape(const_shape![1, D])
                        .broadcast(const_shape![BM, D]);
            }
            row_max = next;
        }
        let denominator_zero: Tile<f32, { [BM] }> = constant(0.0f32, const_shape![BM]);
        let denominator_one: Tile<f32, { [BM] }> = constant(1.0f32, const_shape![BM]);
        let denominator = select(eq_tile(row_sum, denominator_zero), denominator_one, row_sum);
        let output_scale_tile: Tile<f32, { [BM, D] }> =
            broadcast_scalar(output_scale, const_shape![BM, D]);
        let normalized = acc
            / denominator
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, D])
            * output_scale_tile;
        let output_batch: Tile<i32, { [BM] }> =
            broadcast_scalar(batch * query_len, const_shape![BM]);
        let output_heads: Tile<i32, { [BM] }> = broadcast_scalar(heads, const_shape![BM]);
        let output_head: Tile<i32, { [BM] }> = broadcast_scalar(head, const_shape![BM]);
        let mut output_row = (output_batch + qrow) * output_heads + output_head;
        let split_output_row: Tile<i32, { [BM] }> =
            broadcast_scalar((batch * heads + head) * splits + split, const_shape![BM]);
        if MODE == 4 {
            output_row = split_output_row;
        }
        let output_channel_start: Tile<i32, { [D] }> =
            broadcast_scalar(value_tile * D, const_shape![D]);
        let output_channel = d + output_channel_start;
        let output_stride: Tile<i32, { [BM] }> = broadcast_scalar(head_dim, const_shape![BM]);
        let offset = (output_row * output_stride)
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, D])
            + output_channel
                .reshape(const_shape![1, D])
                .broadcast(const_shape![BM, D]);
        let output_width: Tile<i32, { [D] }> = broadcast_scalar(head_dim, const_shape![D]);
        let valid = qvalid
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, D])
            & lt_tile(output_channel, output_width)
                .reshape(const_shape![1, D])
                .broadcast(const_shape![BM, D]);
        let result: Tile<O, { [BM, D] }> = convert_tile(normalized);
        unsafe { store_2d(out, offset, result, valid) };
        if MODE != 0 && MODE != 2 && value_tile == 0i32 {
            let lse_value = row_max + log(denominator);
            unsafe { store_1d(lse, output_row, lse_value, qvalid) };
        }
    }
}

pub use kernels::*;
