#[cutile::module]
mod kernels {
    use cutile::core::*;

    // Forward-only Unsloth-style single-tensor RoPE for compact tensors shaped
    // (batch, seq, heads, head_dim) with half-split real/imag layout.
    #[cutile::entry()]
    pub unsafe fn rope_bshd_f32(
        input: *mut f32,
        out: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        _batch: i32,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        pair_count: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(pair_count, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let head_dim_half = head_dim / 2i32;
        let dim = safe_offsets
            - (safe_offsets / broadcast_scalar(head_dim_half, tile_shape))
                * broadcast_scalar(head_dim_half, tile_shape);
        let head_linear = safe_offsets / broadcast_scalar(head_dim_half, tile_shape);
        let head = head_linear
            - (head_linear / broadcast_scalar(heads, tile_shape))
                * broadcast_scalar(heads, tile_shape);
        let seq_head = head_linear / broadcast_scalar(heads, tile_shape);
        let seq = seq_head
            - (seq_head / broadcast_scalar(seq_len, tile_shape))
                * broadcast_scalar(seq_len, tile_shape);
        let batch_index = seq_head / broadcast_scalar(seq_len, tile_shape);

        let trig_offsets = seq * broadcast_scalar(head_dim_half, tile_shape) + dim;
        let cos_values = load_f32_vector(cos, trig_offsets, mask, 1.0f32);
        let sin_values = load_f32_vector(sin, trig_offsets, mask, 0.0f32);

        let element_base = ((batch_index * broadcast_scalar(seq_len, tile_shape) + seq)
            * broadcast_scalar(heads, tile_shape)
            + head)
            * broadcast_scalar(head_dim, tile_shape);
        let real_offsets = element_base + dim;
        let imag_offsets = element_base + broadcast_scalar(head_dim_half, tile_shape) + dim;
        let real_values = load_f32_vector(input, real_offsets, mask, 0.0f32);
        let imag_values = load_f32_vector(input, imag_offsets, mask, 0.0f32);
        let rotated_real = real_values * cos_values - imag_values * sin_values;
        let rotated_imag = imag_values * cos_values + real_values * sin_values;
        store_f32_vector(out, real_offsets, rotated_real, mask);
        store_f32_vector(out, imag_offsets, rotated_imag, mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn rope_bshd_f16(
        input: *mut f16,
        out: *mut f16,
        cos: *mut f32,
        sin: *mut f32,
        _batch: i32,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        pair_count: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(pair_count, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let head_dim_half = head_dim / 2i32;
        let dim = safe_offsets
            - (safe_offsets / broadcast_scalar(head_dim_half, tile_shape))
                * broadcast_scalar(head_dim_half, tile_shape);
        let head_linear = safe_offsets / broadcast_scalar(head_dim_half, tile_shape);
        let head = head_linear
            - (head_linear / broadcast_scalar(heads, tile_shape))
                * broadcast_scalar(heads, tile_shape);
        let seq_head = head_linear / broadcast_scalar(heads, tile_shape);
        let seq = seq_head
            - (seq_head / broadcast_scalar(seq_len, tile_shape))
                * broadcast_scalar(seq_len, tile_shape);
        let batch_index = seq_head / broadcast_scalar(seq_len, tile_shape);

        let trig_offsets = seq * broadcast_scalar(head_dim_half, tile_shape) + dim;
        let cos_values = load_f32_vector(cos, trig_offsets, mask, 1.0f32);
        let sin_values = load_f32_vector(sin, trig_offsets, mask, 0.0f32);

        let element_base = ((batch_index * broadcast_scalar(seq_len, tile_shape) + seq)
            * broadcast_scalar(heads, tile_shape)
            + head)
            * broadcast_scalar(head_dim, tile_shape);
        let real_offsets = element_base + dim;
        let imag_offsets = element_base + broadcast_scalar(head_dim_half, tile_shape) + dim;
        let real_values = load_f16_vector_as_f32(input, real_offsets, mask);
        let imag_values = load_f16_vector_as_f32(input, imag_offsets, mask);
        let rotated_real = real_values * cos_values - imag_values * sin_values;
        let rotated_imag = imag_values * cos_values + real_values * sin_values;
        store_f16_vector_from_f32(out, real_offsets, rotated_real, mask);
        store_f16_vector_from_f32(out, imag_offsets, rotated_imag, mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn rope_bshd_dynpos_f16(
        input: *mut f16,
        out: *mut f16,
        cos: *mut f32,
        sin: *mut f32,
        position_start: *mut u32,
        _batch: i32,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        pair_count: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(pair_count, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let head_dim_half = head_dim / 2i32;
        let dim = safe_offsets
            - (safe_offsets / broadcast_scalar(head_dim_half, tile_shape))
                * broadcast_scalar(head_dim_half, tile_shape);
        let head_linear = safe_offsets / broadcast_scalar(head_dim_half, tile_shape);
        let head = head_linear
            - (head_linear / broadcast_scalar(heads, tile_shape))
                * broadcast_scalar(heads, tile_shape);
        let seq_head = head_linear / broadcast_scalar(heads, tile_shape);
        let seq = seq_head
            - (seq_head / broadcast_scalar(seq_len, tile_shape))
                * broadcast_scalar(seq_len, tile_shape);
        let batch_index = seq_head / broadcast_scalar(seq_len, tile_shape);
        let position_start = load_position_start_tile(position_start);

        let trig_offsets =
            (position_start + seq) * broadcast_scalar(head_dim_half, tile_shape) + dim;
        let cos_values = load_f32_vector(cos, trig_offsets, mask, 1.0f32);
        let sin_values = load_f32_vector(sin, trig_offsets, mask, 0.0f32);

        let element_base = ((batch_index * broadcast_scalar(seq_len, tile_shape) + seq)
            * broadcast_scalar(heads, tile_shape)
            + head)
            * broadcast_scalar(head_dim, tile_shape);
        let real_offsets = element_base + dim;
        let imag_offsets = element_base + broadcast_scalar(head_dim_half, tile_shape) + dim;
        let real_values = load_f16_vector_as_f32(input, real_offsets, mask);
        let imag_values = load_f16_vector_as_f32(input, imag_offsets, mask);
        let rotated_real = real_values * cos_values - imag_values * sin_values;
        let rotated_imag = imag_values * cos_values + real_values * sin_values;
        store_f16_vector_from_f32(out, real_offsets, rotated_real, mask);
        store_f16_vector_from_f32(out, imag_offsets, rotated_imag, mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn rope_bshd_bf16(
        input: *mut bf16,
        out: *mut bf16,
        cos: *mut f32,
        sin: *mut f32,
        _batch: i32,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        pair_count: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(pair_count, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let head_dim_half = head_dim / 2i32;
        let dim = safe_offsets
            - (safe_offsets / broadcast_scalar(head_dim_half, tile_shape))
                * broadcast_scalar(head_dim_half, tile_shape);
        let head_linear = safe_offsets / broadcast_scalar(head_dim_half, tile_shape);
        let head = head_linear
            - (head_linear / broadcast_scalar(heads, tile_shape))
                * broadcast_scalar(heads, tile_shape);
        let seq_head = head_linear / broadcast_scalar(heads, tile_shape);
        let seq = seq_head
            - (seq_head / broadcast_scalar(seq_len, tile_shape))
                * broadcast_scalar(seq_len, tile_shape);
        let batch_index = seq_head / broadcast_scalar(seq_len, tile_shape);

        let trig_offsets = seq * broadcast_scalar(head_dim_half, tile_shape) + dim;
        let cos_values = load_f32_vector(cos, trig_offsets, mask, 1.0f32);
        let sin_values = load_f32_vector(sin, trig_offsets, mask, 0.0f32);

        let element_base = ((batch_index * broadcast_scalar(seq_len, tile_shape) + seq)
            * broadcast_scalar(heads, tile_shape)
            + head)
            * broadcast_scalar(head_dim, tile_shape);
        let real_offsets = element_base + dim;
        let imag_offsets = element_base + broadcast_scalar(head_dim_half, tile_shape) + dim;
        let real_values = load_bf16_vector_as_f32(input, real_offsets, mask);
        let imag_values = load_bf16_vector_as_f32(input, imag_offsets, mask);
        let rotated_real = real_values * cos_values - imag_values * sin_values;
        let rotated_imag = imag_values * cos_values + real_values * sin_values;
        store_bf16_vector_from_f32(out, real_offsets, rotated_real, mask);
        store_bf16_vector_from_f32(out, imag_offsets, rotated_imag, mask);
    }

    #[cfg(feature = "dtype-f16")]
    fn load_position_start_tile(position_start: *mut u32) -> Tile<i32, { [128] }> {
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mask: Tile<bool, { [128] }> = constant(true, tile_shape);
        let base: PointerTile<*mut u32, { [] }> = pointer_to_tile(position_start);
        let base: PointerTile<*mut u32, { [1] }> = base.reshape(const_shape![1]);
        let ptrs: PointerTile<*mut u32, { [128] }> = base.broadcast(tile_shape);
        let ptrs: PointerTile<*mut u32, { [128] }> = ptrs.offset_tile(offsets);
        let result: (Tile<u32, { [128] }>, Token) = load_ptr_tko(
            ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u32),
            None,
            Latency::<0>,
        );
        bitcast(result.0)
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn rotary_embedding_qk_bnsd_bf16(
        q_out: *mut bf16,
        k_out: *mut bf16,
        q_input: *mut bf16,
        k_input: *mut bf16,
        cos: *mut f32,
        sin: *mut f32,
        _batch: i32,
        seq_len: i32,
        query_heads: i32,
        key_heads: i32,
        head_dim: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        q_len: i32,
        _k_len: i32,
        total_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(total_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let is_query = cmpi(
            safe_offsets,
            broadcast_scalar(q_len, tile_shape),
            predicate::LessThan,
        );
        let q_len_tile = broadcast_scalar(q_len, tile_shape);
        let local_offsets = select(is_query, safe_offsets, safe_offsets - q_len_tile);
        let heads = select(
            is_query,
            broadcast_scalar(query_heads, tile_shape),
            broadcast_scalar(key_heads, tile_shape),
        );

        let dim = local_offsets
            - (local_offsets / broadcast_scalar(head_dim, tile_shape))
                * broadcast_scalar(head_dim, tile_shape);
        let seq_linear = local_offsets / broadcast_scalar(head_dim, tile_shape);
        let seq = seq_linear
            - (seq_linear / broadcast_scalar(seq_len, tile_shape))
                * broadcast_scalar(seq_len, tile_shape);
        let head_batch = seq_linear / broadcast_scalar(seq_len, tile_shape);
        let head = head_batch - (head_batch / heads) * heads;
        let batch_index = head_batch / heads;

        let in_rotary = mask
            & cmpi(
                dim,
                broadcast_scalar(rotary_pairs * 2i32, tile_shape),
                predicate::LessThan,
            );
        let is_second_half = cmpi(
            dim,
            broadcast_scalar(rotary_pairs, tile_shape),
            predicate::GreaterThanOrEqual,
        );
        let pair = select(
            is_second_half,
            dim - broadcast_scalar(rotary_pairs, tile_shape),
            dim,
        );
        let paired_dim = select(
            is_second_half,
            dim - broadcast_scalar(rotary_pairs, tile_shape),
            dim + broadcast_scalar(rotary_pairs, tile_shape),
        );
        let trig_batch_index = select(
            cmpi(
                broadcast_scalar(trig_batch, tile_shape),
                broadcast_scalar(1i32, tile_shape),
                predicate::Equal,
            ),
            zero_offsets,
            batch_index,
        );
        let trig_offsets = (trig_batch_index * broadcast_scalar(seq_len, tile_shape) + seq)
            * broadcast_scalar(rotary_pairs, tile_shape)
            + pair;
        let cos_values = load_f32_vector(cos, trig_offsets, in_rotary, 1.0f32);
        let sin_values = load_f32_vector(sin, trig_offsets, in_rotary, 0.0f32);

        let element_base = ((batch_index * heads + head) * broadcast_scalar(seq_len, tile_shape)
            + seq)
            * broadcast_scalar(head_dim, tile_shape);
        let current_offsets = element_base + dim;
        let paired_offsets = element_base + paired_dim;
        let query_mask = mask & is_query;
        let key_mask = mask & cmpi(safe_offsets, q_len_tile, predicate::GreaterThanOrEqual);
        let current_values = select(
            is_query,
            load_bf16_vector_as_f32(q_input, current_offsets, query_mask),
            load_bf16_vector_as_f32(k_input, current_offsets, key_mask),
        );
        let paired_values = select(
            is_query,
            load_bf16_vector_as_f32(q_input, paired_offsets, query_mask & in_rotary),
            load_bf16_vector_as_f32(k_input, paired_offsets, key_mask & in_rotary),
        );
        let rotated_first = current_values * cos_values - paired_values * sin_values;
        let rotated_second = current_values * cos_values + paired_values * sin_values;
        let rotated = select(is_second_half, rotated_second, rotated_first);
        let values = select(in_rotary, rotated, current_values);
        store_bf16_vector_from_f32(q_out, current_offsets, values, query_mask);
        store_bf16_vector_from_f32(k_out, current_offsets, values, key_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_qk_bnsd_f32(
        q_out: *mut f32,
        k_out: *mut f32,
        q_input: *mut f32,
        k_input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        _batch: i32,
        seq_len: i32,
        query_heads: i32,
        key_heads: i32,
        head_dim: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        q_len: i32,
        _k_len: i32,
        total_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(total_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let is_query = cmpi(
            safe_offsets,
            broadcast_scalar(q_len, tile_shape),
            predicate::LessThan,
        );
        let q_len_tile = broadcast_scalar(q_len, tile_shape);
        let local_offsets = select(is_query, safe_offsets, safe_offsets - q_len_tile);
        let heads = select(
            is_query,
            broadcast_scalar(query_heads, tile_shape),
            broadcast_scalar(key_heads, tile_shape),
        );

        let dim = local_offsets
            - (local_offsets / broadcast_scalar(head_dim, tile_shape))
                * broadcast_scalar(head_dim, tile_shape);
        let seq_linear = local_offsets / broadcast_scalar(head_dim, tile_shape);
        let seq = seq_linear
            - (seq_linear / broadcast_scalar(seq_len, tile_shape))
                * broadcast_scalar(seq_len, tile_shape);
        let head_batch = seq_linear / broadcast_scalar(seq_len, tile_shape);
        let head = head_batch - (head_batch / heads) * heads;
        let batch_index = head_batch / heads;

        let in_rotary = mask
            & cmpi(
                dim,
                broadcast_scalar(rotary_pairs * 2i32, tile_shape),
                predicate::LessThan,
            );
        let is_second_half = cmpi(
            dim,
            broadcast_scalar(rotary_pairs, tile_shape),
            predicate::GreaterThanOrEqual,
        );
        let pair = select(
            is_second_half,
            dim - broadcast_scalar(rotary_pairs, tile_shape),
            dim,
        );
        let paired_dim = select(
            is_second_half,
            dim - broadcast_scalar(rotary_pairs, tile_shape),
            dim + broadcast_scalar(rotary_pairs, tile_shape),
        );
        let trig_batch_index = select(
            cmpi(
                broadcast_scalar(trig_batch, tile_shape),
                broadcast_scalar(1i32, tile_shape),
                predicate::Equal,
            ),
            zero_offsets,
            batch_index,
        );
        let trig_offsets = (trig_batch_index * broadcast_scalar(seq_len, tile_shape) + seq)
            * broadcast_scalar(rotary_pairs, tile_shape)
            + pair;
        let cos_values = load_f32_vector(cos, trig_offsets, in_rotary, 1.0f32);
        let sin_values = load_f32_vector(sin, trig_offsets, in_rotary, 0.0f32);

        let element_base = ((batch_index * heads + head) * broadcast_scalar(seq_len, tile_shape)
            + seq)
            * broadcast_scalar(head_dim, tile_shape);
        let current_offsets = element_base + dim;
        let paired_offsets = element_base + paired_dim;
        let query_mask = mask & is_query;
        let key_mask = mask & cmpi(safe_offsets, q_len_tile, predicate::GreaterThanOrEqual);
        let current_values = select(
            is_query,
            load_f32_vector(q_input, current_offsets, query_mask, 0.0f32),
            load_f32_vector(k_input, current_offsets, key_mask, 0.0f32),
        );
        let paired_values = select(
            is_query,
            load_f32_vector(q_input, paired_offsets, query_mask & in_rotary, 0.0f32),
            load_f32_vector(k_input, paired_offsets, key_mask & in_rotary, 0.0f32),
        );
        let rotated_first = current_values * cos_values - paired_values * sin_values;
        let rotated_second = current_values * cos_values + paired_values * sin_values;
        let rotated = select(is_second_half, rotated_second, rotated_first);
        let values = select(in_rotary, rotated, current_values);
        store_f32_vector(q_out, current_offsets, values, query_mask);
        store_f32_vector(k_out, current_offsets, values, key_mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn rotary_embedding_qk_bnsd_f16(
        q_out: *mut f16,
        k_out: *mut f16,
        q_input: *mut f16,
        k_input: *mut f16,
        cos: *mut f32,
        sin: *mut f32,
        _batch: i32,
        seq_len: i32,
        query_heads: i32,
        key_heads: i32,
        head_dim: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        q_len: i32,
        _k_len: i32,
        total_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(total_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let is_query = cmpi(
            safe_offsets,
            broadcast_scalar(q_len, tile_shape),
            predicate::LessThan,
        );
        let q_len_tile = broadcast_scalar(q_len, tile_shape);
        let local_offsets = select(is_query, safe_offsets, safe_offsets - q_len_tile);
        let heads = select(
            is_query,
            broadcast_scalar(query_heads, tile_shape),
            broadcast_scalar(key_heads, tile_shape),
        );

        let dim = local_offsets
            - (local_offsets / broadcast_scalar(head_dim, tile_shape))
                * broadcast_scalar(head_dim, tile_shape);
        let seq_linear = local_offsets / broadcast_scalar(head_dim, tile_shape);
        let seq = seq_linear
            - (seq_linear / broadcast_scalar(seq_len, tile_shape))
                * broadcast_scalar(seq_len, tile_shape);
        let head_batch = seq_linear / broadcast_scalar(seq_len, tile_shape);
        let head = head_batch - (head_batch / heads) * heads;
        let batch_index = head_batch / heads;

        let in_rotary = mask
            & cmpi(
                dim,
                broadcast_scalar(rotary_pairs * 2i32, tile_shape),
                predicate::LessThan,
            );
        let is_second_half = cmpi(
            dim,
            broadcast_scalar(rotary_pairs, tile_shape),
            predicate::GreaterThanOrEqual,
        );
        let pair = select(
            is_second_half,
            dim - broadcast_scalar(rotary_pairs, tile_shape),
            dim,
        );
        let paired_dim = select(
            is_second_half,
            dim - broadcast_scalar(rotary_pairs, tile_shape),
            dim + broadcast_scalar(rotary_pairs, tile_shape),
        );
        let trig_batch_index = select(
            cmpi(
                broadcast_scalar(trig_batch, tile_shape),
                broadcast_scalar(1i32, tile_shape),
                predicate::Equal,
            ),
            zero_offsets,
            batch_index,
        );
        let trig_offsets = (trig_batch_index * broadcast_scalar(seq_len, tile_shape) + seq)
            * broadcast_scalar(rotary_pairs, tile_shape)
            + pair;
        let cos_values = load_f32_vector(cos, trig_offsets, in_rotary, 1.0f32);
        let sin_values = load_f32_vector(sin, trig_offsets, in_rotary, 0.0f32);

        let element_base = ((batch_index * heads + head) * broadcast_scalar(seq_len, tile_shape)
            + seq)
            * broadcast_scalar(head_dim, tile_shape);
        let current_offsets = element_base + dim;
        let paired_offsets = element_base + paired_dim;
        let query_mask = mask & is_query;
        let key_mask = mask & cmpi(safe_offsets, q_len_tile, predicate::GreaterThanOrEqual);
        let current_values = select(
            is_query,
            load_f16_vector_as_f32(q_input, current_offsets, query_mask),
            load_f16_vector_as_f32(k_input, current_offsets, key_mask),
        );
        let paired_values = select(
            is_query,
            load_f16_vector_as_f32(q_input, paired_offsets, query_mask & in_rotary),
            load_f16_vector_as_f32(k_input, paired_offsets, key_mask & in_rotary),
        );
        let rotated_first = current_values * cos_values - paired_values * sin_values;
        let rotated_second = current_values * cos_values + paired_values * sin_values;
        let rotated = select(is_second_half, rotated_second, rotated_first);
        let values = select(in_rotary, rotated, current_values);
        store_f16_vector_from_f32(q_out, current_offsets, values, query_mask);
        store_f16_vector_from_f32(k_out, current_offsets, values, key_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_qk_bnsd_in_place_f32(
        query: *mut f32,
        key: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        _batch: i32,
        seq_len: i32,
        query_heads: i32,
        key_heads: i32,
        head_dim: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        q_pair_count: i32,
        total_pair_count: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(total_pair_count, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let is_query = cmpi(
            safe_offsets,
            broadcast_scalar(q_pair_count, tile_shape),
            predicate::LessThan,
        );
        let q_pair_count_tile = broadcast_scalar(q_pair_count, tile_shape);
        let local_offsets = select(is_query, safe_offsets, safe_offsets - q_pair_count_tile);
        let heads = select(
            is_query,
            broadcast_scalar(query_heads, tile_shape),
            broadcast_scalar(key_heads, tile_shape),
        );
        let pair = local_offsets
            - (local_offsets / broadcast_scalar(rotary_pairs, tile_shape))
                * broadcast_scalar(rotary_pairs, tile_shape);
        let seq_linear = local_offsets / broadcast_scalar(rotary_pairs, tile_shape);
        let seq = seq_linear
            - (seq_linear / broadcast_scalar(seq_len, tile_shape))
                * broadcast_scalar(seq_len, tile_shape);
        let head_batch = seq_linear / broadcast_scalar(seq_len, tile_shape);
        let head = head_batch - (head_batch / heads) * heads;
        let batch_index = head_batch / heads;

        let trig_batch_index = select(
            cmpi(
                broadcast_scalar(trig_batch, tile_shape),
                broadcast_scalar(1i32, tile_shape),
                predicate::Equal,
            ),
            zero_offsets,
            batch_index,
        );
        let trig_offsets = (trig_batch_index * broadcast_scalar(seq_len, tile_shape) + seq)
            * broadcast_scalar(rotary_pairs, tile_shape)
            + pair;
        let cos_values = load_f32_vector(cos, trig_offsets, mask, 1.0f32);
        let sin_values = load_f32_vector(sin, trig_offsets, mask, 0.0f32);

        let element_base = ((batch_index * heads + head) * broadcast_scalar(seq_len, tile_shape)
            + seq)
            * broadcast_scalar(head_dim, tile_shape);
        let real_offsets = element_base + pair;
        let imag_offsets = element_base + broadcast_scalar(rotary_pairs, tile_shape) + pair;
        let query_mask = mask & is_query;
        let key_mask = mask
            & cmpi(
                safe_offsets,
                q_pair_count_tile,
                predicate::GreaterThanOrEqual,
            );
        let real_values = select(
            is_query,
            load_f32_vector(query, real_offsets, query_mask, 0.0f32),
            load_f32_vector(key, real_offsets, key_mask, 0.0f32),
        );
        let imag_values = select(
            is_query,
            load_f32_vector(query, imag_offsets, query_mask, 0.0f32),
            load_f32_vector(key, imag_offsets, key_mask, 0.0f32),
        );
        let rotated_real = real_values * cos_values - imag_values * sin_values;
        let rotated_imag = imag_values * cos_values + real_values * sin_values;
        store_f32_vector(query, real_offsets, rotated_real, query_mask);
        store_f32_vector(query, imag_offsets, rotated_imag, query_mask);
        store_f32_vector(key, real_offsets, rotated_real, key_mask);
        store_f32_vector(key, imag_offsets, rotated_imag, key_mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn rotary_embedding_qk_bnsd_in_place_f16(
        query: *mut f16,
        key: *mut f16,
        cos: *mut f32,
        sin: *mut f32,
        _batch: i32,
        seq_len: i32,
        query_heads: i32,
        key_heads: i32,
        head_dim: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        q_pair_count: i32,
        total_pair_count: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(total_pair_count, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let is_query = cmpi(
            safe_offsets,
            broadcast_scalar(q_pair_count, tile_shape),
            predicate::LessThan,
        );
        let q_pair_count_tile = broadcast_scalar(q_pair_count, tile_shape);
        let local_offsets = select(is_query, safe_offsets, safe_offsets - q_pair_count_tile);
        let heads = select(
            is_query,
            broadcast_scalar(query_heads, tile_shape),
            broadcast_scalar(key_heads, tile_shape),
        );
        let pair = local_offsets
            - (local_offsets / broadcast_scalar(rotary_pairs, tile_shape))
                * broadcast_scalar(rotary_pairs, tile_shape);
        let seq_linear = local_offsets / broadcast_scalar(rotary_pairs, tile_shape);
        let seq = seq_linear
            - (seq_linear / broadcast_scalar(seq_len, tile_shape))
                * broadcast_scalar(seq_len, tile_shape);
        let head_batch = seq_linear / broadcast_scalar(seq_len, tile_shape);
        let head = head_batch - (head_batch / heads) * heads;
        let batch_index = head_batch / heads;

        let trig_batch_index = select(
            cmpi(
                broadcast_scalar(trig_batch, tile_shape),
                broadcast_scalar(1i32, tile_shape),
                predicate::Equal,
            ),
            zero_offsets,
            batch_index,
        );
        let trig_offsets = (trig_batch_index * broadcast_scalar(seq_len, tile_shape) + seq)
            * broadcast_scalar(rotary_pairs, tile_shape)
            + pair;
        let cos_values = load_f32_vector(cos, trig_offsets, mask, 1.0f32);
        let sin_values = load_f32_vector(sin, trig_offsets, mask, 0.0f32);

        let element_base = ((batch_index * heads + head) * broadcast_scalar(seq_len, tile_shape)
            + seq)
            * broadcast_scalar(head_dim, tile_shape);
        let real_offsets = element_base + pair;
        let imag_offsets = element_base + broadcast_scalar(rotary_pairs, tile_shape) + pair;
        let query_mask = mask & is_query;
        let key_mask = mask
            & cmpi(
                safe_offsets,
                q_pair_count_tile,
                predicate::GreaterThanOrEqual,
            );
        let real_values = select(
            is_query,
            load_f16_vector_as_f32(query, real_offsets, query_mask),
            load_f16_vector_as_f32(key, real_offsets, key_mask),
        );
        let imag_values = select(
            is_query,
            load_f16_vector_as_f32(query, imag_offsets, query_mask),
            load_f16_vector_as_f32(key, imag_offsets, key_mask),
        );
        let rotated_real = real_values * cos_values - imag_values * sin_values;
        let rotated_imag = imag_values * cos_values + real_values * sin_values;
        store_f16_vector_from_f32(query, real_offsets, rotated_real, query_mask);
        store_f16_vector_from_f32(query, imag_offsets, rotated_imag, query_mask);
        store_f16_vector_from_f32(key, real_offsets, rotated_real, key_mask);
        store_f16_vector_from_f32(key, imag_offsets, rotated_imag, key_mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn rotary_embedding_qk_bnsd_in_place_bf16(
        query: *mut bf16,
        key: *mut bf16,
        cos: *mut f32,
        sin: *mut f32,
        _batch: i32,
        seq_len: i32,
        query_heads: i32,
        key_heads: i32,
        head_dim: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        q_pair_count: i32,
        total_pair_count: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(total_pair_count, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let is_query = cmpi(
            safe_offsets,
            broadcast_scalar(q_pair_count, tile_shape),
            predicate::LessThan,
        );
        let q_pair_count_tile = broadcast_scalar(q_pair_count, tile_shape);
        let local_offsets = select(is_query, safe_offsets, safe_offsets - q_pair_count_tile);
        let heads = select(
            is_query,
            broadcast_scalar(query_heads, tile_shape),
            broadcast_scalar(key_heads, tile_shape),
        );
        let pair = local_offsets
            - (local_offsets / broadcast_scalar(rotary_pairs, tile_shape))
                * broadcast_scalar(rotary_pairs, tile_shape);
        let seq_linear = local_offsets / broadcast_scalar(rotary_pairs, tile_shape);
        let seq = seq_linear
            - (seq_linear / broadcast_scalar(seq_len, tile_shape))
                * broadcast_scalar(seq_len, tile_shape);
        let head_batch = seq_linear / broadcast_scalar(seq_len, tile_shape);
        let head = head_batch - (head_batch / heads) * heads;
        let batch_index = head_batch / heads;

        let trig_batch_index = select(
            cmpi(
                broadcast_scalar(trig_batch, tile_shape),
                broadcast_scalar(1i32, tile_shape),
                predicate::Equal,
            ),
            zero_offsets,
            batch_index,
        );
        let trig_offsets = (trig_batch_index * broadcast_scalar(seq_len, tile_shape) + seq)
            * broadcast_scalar(rotary_pairs, tile_shape)
            + pair;
        let cos_values = load_f32_vector(cos, trig_offsets, mask, 1.0f32);
        let sin_values = load_f32_vector(sin, trig_offsets, mask, 0.0f32);

        let element_base = ((batch_index * heads + head) * broadcast_scalar(seq_len, tile_shape)
            + seq)
            * broadcast_scalar(head_dim, tile_shape);
        let real_offsets = element_base + pair;
        let imag_offsets = element_base + broadcast_scalar(rotary_pairs, tile_shape) + pair;
        let query_mask = mask & is_query;
        let key_mask = mask
            & cmpi(
                safe_offsets,
                q_pair_count_tile,
                predicate::GreaterThanOrEqual,
            );
        let real_values = select(
            is_query,
            load_bf16_vector_as_f32(query, real_offsets, query_mask),
            load_bf16_vector_as_f32(key, real_offsets, key_mask),
        );
        let imag_values = select(
            is_query,
            load_bf16_vector_as_f32(query, imag_offsets, query_mask),
            load_bf16_vector_as_f32(key, imag_offsets, key_mask),
        );
        let rotated_real = real_values * cos_values - imag_values * sin_values;
        let rotated_imag = imag_values * cos_values + real_values * sin_values;
        store_bf16_vector_from_f32(query, real_offsets, rotated_real, query_mask);
        store_bf16_vector_from_f32(query, imag_offsets, rotated_imag, query_mask);
        store_bf16_vector_from_f32(key, real_offsets, rotated_real, key_mask);
        store_bf16_vector_from_f32(key, imag_offsets, rotated_imag, key_mask);
    }

    // Forward-only multiaxis RoPE for contiguous Q/K tensors shaped
    // (batch, seq, heads, head_dim) with half-split real/imag layout.
    #[cutile::entry()]
    pub unsafe fn multiaxis_rope_qk_f32(
        query: *mut f32,
        key: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        batch: i32,
        seq_len: i32,
        query_heads: i32,
        key_heads: i32,
        head_dim: i32,
        section_t: i32,
        section_h: i32,
        pair_count: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(pair_count, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let head_dim_half = head_dim / 2i32;
        let query_pairs = batch * seq_len * query_heads * head_dim_half;
        let is_query = cmpi(
            safe_offsets,
            broadcast_scalar(query_pairs, tile_shape),
            predicate::LessThan,
        );
        let query_pairs_tile = broadcast_scalar(query_pairs, tile_shape);
        let local_offsets = select(is_query, safe_offsets, safe_offsets - query_pairs_tile);
        let heads = select(
            is_query,
            broadcast_scalar(query_heads, tile_shape),
            broadcast_scalar(key_heads, tile_shape),
        );
        let dim = local_offsets
            - (local_offsets / broadcast_scalar(head_dim_half, tile_shape))
                * broadcast_scalar(head_dim_half, tile_shape);
        let head_linear = local_offsets / broadcast_scalar(head_dim_half, tile_shape);
        let seq = head_linear / heads
            - (head_linear / (heads * broadcast_scalar(seq_len, tile_shape)))
                * broadcast_scalar(seq_len, tile_shape);
        let batch_index = head_linear / (heads * broadcast_scalar(seq_len, tile_shape));
        let head =
            head_linear - (batch_index * broadcast_scalar(seq_len, tile_shape) + seq) * heads;

        let section_t_end = broadcast_scalar(section_t, tile_shape);
        let section_h_end = broadcast_scalar(section_t + section_h, tile_shape);
        let section_zero: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let section_one: Tile<i32, { [128] }> = constant(1i32, tile_shape);
        let section_two: Tile<i32, { [128] }> = constant(2i32, tile_shape);
        let section: Tile<i32, { [128] }> = select(
            cmpi(dim, section_t_end, predicate::LessThan),
            section_zero,
            select(
                cmpi(dim, section_h_end, predicate::LessThan),
                section_one,
                section_two,
            ),
        );
        let trig_offsets = ((section * broadcast_scalar(batch, tile_shape) + batch_index)
            * broadcast_scalar(seq_len, tile_shape)
            + seq)
            * broadcast_scalar(head_dim_half, tile_shape)
            + dim;
        let cos_values = load_f32_vector(cos, trig_offsets, mask, 1.0f32);
        let sin_values = load_f32_vector(sin, trig_offsets, mask, 0.0f32);

        let element_offsets = ((batch_index * broadcast_scalar(seq_len, tile_shape) + seq) * heads
            + head)
            * broadcast_scalar(head_dim, tile_shape)
            + dim;
        let real_offsets = element_offsets;
        let imag_offsets = element_offsets + broadcast_scalar(head_dim_half, tile_shape);
        let real_values = select(
            is_query,
            load_f32_vector(query, real_offsets, mask, 0.0f32),
            load_f32_vector(key, real_offsets, mask, 0.0f32),
        );
        let imag_values = select(
            is_query,
            load_f32_vector(query, imag_offsets, mask, 0.0f32),
            load_f32_vector(key, imag_offsets, mask, 0.0f32),
        );
        let rotated_real = real_values * cos_values - imag_values * sin_values;
        let rotated_imag = imag_values * cos_values + real_values * sin_values;
        let key_mask = mask
            & cmpi(
                safe_offsets,
                query_pairs_tile,
                predicate::GreaterThanOrEqual,
            );
        store_f32_vector(query, real_offsets, rotated_real, mask & is_query);
        store_f32_vector(query, imag_offsets, rotated_imag, mask & is_query);
        store_f32_vector(key, real_offsets, rotated_real, key_mask);
        store_f32_vector(key, imag_offsets, rotated_imag, key_mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn multiaxis_rope_qk_f16(
        query: *mut f16,
        key: *mut f16,
        cos: *mut f32,
        sin: *mut f32,
        batch: i32,
        seq_len: i32,
        query_heads: i32,
        key_heads: i32,
        head_dim: i32,
        section_t: i32,
        section_h: i32,
        pair_count: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(pair_count, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let head_dim_half = head_dim / 2i32;
        let query_pairs = batch * seq_len * query_heads * head_dim_half;
        let is_query = cmpi(
            safe_offsets,
            broadcast_scalar(query_pairs, tile_shape),
            predicate::LessThan,
        );
        let query_pairs_tile = broadcast_scalar(query_pairs, tile_shape);
        let local_offsets = select(is_query, safe_offsets, safe_offsets - query_pairs_tile);
        let heads = select(
            is_query,
            broadcast_scalar(query_heads, tile_shape),
            broadcast_scalar(key_heads, tile_shape),
        );
        let dim = local_offsets
            - (local_offsets / broadcast_scalar(head_dim_half, tile_shape))
                * broadcast_scalar(head_dim_half, tile_shape);
        let head_linear = local_offsets / broadcast_scalar(head_dim_half, tile_shape);
        let seq = head_linear / heads
            - (head_linear / (heads * broadcast_scalar(seq_len, tile_shape)))
                * broadcast_scalar(seq_len, tile_shape);
        let batch_index = head_linear / (heads * broadcast_scalar(seq_len, tile_shape));
        let head =
            head_linear - (batch_index * broadcast_scalar(seq_len, tile_shape) + seq) * heads;

        let section_t_end = broadcast_scalar(section_t, tile_shape);
        let section_h_end = broadcast_scalar(section_t + section_h, tile_shape);
        let section_zero: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let section_one: Tile<i32, { [128] }> = constant(1i32, tile_shape);
        let section_two: Tile<i32, { [128] }> = constant(2i32, tile_shape);
        let section: Tile<i32, { [128] }> = select(
            cmpi(dim, section_t_end, predicate::LessThan),
            section_zero,
            select(
                cmpi(dim, section_h_end, predicate::LessThan),
                section_one,
                section_two,
            ),
        );
        let trig_offsets = ((section * broadcast_scalar(batch, tile_shape) + batch_index)
            * broadcast_scalar(seq_len, tile_shape)
            + seq)
            * broadcast_scalar(head_dim_half, tile_shape)
            + dim;
        let cos_values = load_f32_vector(cos, trig_offsets, mask, 1.0f32);
        let sin_values = load_f32_vector(sin, trig_offsets, mask, 0.0f32);

        let element_offsets = ((batch_index * broadcast_scalar(seq_len, tile_shape) + seq) * heads
            + head)
            * broadcast_scalar(head_dim, tile_shape)
            + dim;
        let real_offsets = element_offsets;
        let imag_offsets = element_offsets + broadcast_scalar(head_dim_half, tile_shape);
        let real_values = select(
            is_query,
            load_f16_vector_as_f32(query, real_offsets, mask),
            load_f16_vector_as_f32(key, real_offsets, mask),
        );
        let imag_values = select(
            is_query,
            load_f16_vector_as_f32(query, imag_offsets, mask),
            load_f16_vector_as_f32(key, imag_offsets, mask),
        );
        let rotated_real = real_values * cos_values - imag_values * sin_values;
        let rotated_imag = imag_values * cos_values + real_values * sin_values;
        let key_mask = mask
            & cmpi(
                safe_offsets,
                query_pairs_tile,
                predicate::GreaterThanOrEqual,
            );
        store_f16_vector_from_f32(query, real_offsets, rotated_real, mask & is_query);
        store_f16_vector_from_f32(query, imag_offsets, rotated_imag, mask & is_query);
        store_f16_vector_from_f32(key, real_offsets, rotated_real, key_mask);
        store_f16_vector_from_f32(key, imag_offsets, rotated_imag, key_mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn multiaxis_rope_qk_bf16(
        query: *mut bf16,
        key: *mut bf16,
        cos: *mut f32,
        sin: *mut f32,
        batch: i32,
        seq_len: i32,
        query_heads: i32,
        key_heads: i32,
        head_dim: i32,
        section_t: i32,
        section_h: i32,
        pair_count: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(pair_count, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let head_dim_half = head_dim / 2i32;
        let query_pairs = batch * seq_len * query_heads * head_dim_half;
        let is_query = cmpi(
            safe_offsets,
            broadcast_scalar(query_pairs, tile_shape),
            predicate::LessThan,
        );
        let query_pairs_tile = broadcast_scalar(query_pairs, tile_shape);
        let local_offsets = select(is_query, safe_offsets, safe_offsets - query_pairs_tile);
        let heads = select(
            is_query,
            broadcast_scalar(query_heads, tile_shape),
            broadcast_scalar(key_heads, tile_shape),
        );
        let dim = local_offsets
            - (local_offsets / broadcast_scalar(head_dim_half, tile_shape))
                * broadcast_scalar(head_dim_half, tile_shape);
        let head_linear = local_offsets / broadcast_scalar(head_dim_half, tile_shape);
        let seq = head_linear / heads
            - (head_linear / (heads * broadcast_scalar(seq_len, tile_shape)))
                * broadcast_scalar(seq_len, tile_shape);
        let batch_index = head_linear / (heads * broadcast_scalar(seq_len, tile_shape));
        let head =
            head_linear - (batch_index * broadcast_scalar(seq_len, tile_shape) + seq) * heads;

        let section_t_end = broadcast_scalar(section_t, tile_shape);
        let section_h_end = broadcast_scalar(section_t + section_h, tile_shape);
        let section_zero: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let section_one: Tile<i32, { [128] }> = constant(1i32, tile_shape);
        let section_two: Tile<i32, { [128] }> = constant(2i32, tile_shape);
        let section: Tile<i32, { [128] }> = select(
            cmpi(dim, section_t_end, predicate::LessThan),
            section_zero,
            select(
                cmpi(dim, section_h_end, predicate::LessThan),
                section_one,
                section_two,
            ),
        );
        let trig_offsets = ((section * broadcast_scalar(batch, tile_shape) + batch_index)
            * broadcast_scalar(seq_len, tile_shape)
            + seq)
            * broadcast_scalar(head_dim_half, tile_shape)
            + dim;
        let cos_values = load_f32_vector(cos, trig_offsets, mask, 1.0f32);
        let sin_values = load_f32_vector(sin, trig_offsets, mask, 0.0f32);

        let element_offsets = ((batch_index * broadcast_scalar(seq_len, tile_shape) + seq) * heads
            + head)
            * broadcast_scalar(head_dim, tile_shape)
            + dim;
        let real_offsets = element_offsets;
        let imag_offsets = element_offsets + broadcast_scalar(head_dim_half, tile_shape);
        let real_values = select(
            is_query,
            load_bf16_vector_as_f32(query, real_offsets, mask),
            load_bf16_vector_as_f32(key, real_offsets, mask),
        );
        let imag_values = select(
            is_query,
            load_bf16_vector_as_f32(query, imag_offsets, mask),
            load_bf16_vector_as_f32(key, imag_offsets, mask),
        );
        let rotated_real = real_values * cos_values - imag_values * sin_values;
        let rotated_imag = imag_values * cos_values + real_values * sin_values;
        let key_mask = mask
            & cmpi(
                safe_offsets,
                query_pairs_tile,
                predicate::GreaterThanOrEqual,
            );
        store_bf16_vector_from_f32(query, real_offsets, rotated_real, mask & is_query);
        store_bf16_vector_from_f32(query, imag_offsets, rotated_imag, mask & is_query);
        store_bf16_vector_from_f32(key, real_offsets, rotated_real, key_mask);
        store_bf16_vector_from_f32(key, imag_offsets, rotated_imag, key_mask);
    }

    // Forward-only interleaved complex RoPE for contiguous Q/K tensors shaped
    // (batch, seq, heads, head_dim) with interleaved real/imag pairs.
    #[cutile::entry()]
    pub unsafe fn interleaved_complex_rope_qk_f32(
        query: *mut f32,
        key: *mut f32,
        freqs: *mut f32,
        batch: i32,
        seq_len: i32,
        query_heads: i32,
        key_heads: i32,
        head_dim: i32,
        pair_count: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(pair_count, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let head_dim_half = head_dim / 2i32;
        let query_pairs = batch * seq_len * query_heads * head_dim_half;
        let is_query = cmpi(
            safe_offsets,
            broadcast_scalar(query_pairs, tile_shape),
            predicate::LessThan,
        );
        let query_pairs_tile = broadcast_scalar(query_pairs, tile_shape);
        let local_offsets = select(is_query, safe_offsets, safe_offsets - query_pairs_tile);
        let heads = select(
            is_query,
            broadcast_scalar(query_heads, tile_shape),
            broadcast_scalar(key_heads, tile_shape),
        );
        let dim = local_offsets
            - (local_offsets / broadcast_scalar(head_dim_half, tile_shape))
                * broadcast_scalar(head_dim_half, tile_shape);
        let head_linear = local_offsets / broadcast_scalar(head_dim_half, tile_shape);
        let seq = head_linear / heads
            - (head_linear / (heads * broadcast_scalar(seq_len, tile_shape)))
                * broadcast_scalar(seq_len, tile_shape);
        let batch_index = head_linear / (heads * broadcast_scalar(seq_len, tile_shape));
        let head =
            head_linear - (batch_index * broadcast_scalar(seq_len, tile_shape) + seq) * heads;

        let real_dim = dim * broadcast_scalar(2i32, tile_shape);
        let imag_dim = real_dim + broadcast_scalar(1i32, tile_shape);
        let freq_row = seq * broadcast_scalar(head_dim, tile_shape);
        let freq_real = load_f32_vector(freqs, freq_row + real_dim, mask, 1.0f32);
        let freq_imag = load_f32_vector(freqs, freq_row + imag_dim, mask, 0.0f32);

        let element_base = ((batch_index * broadcast_scalar(seq_len, tile_shape) + seq) * heads
            + head)
            * broadcast_scalar(head_dim, tile_shape);
        let real_offsets = element_base + real_dim;
        let imag_offsets = element_base + imag_dim;
        let real_values = select(
            is_query,
            load_f32_vector(query, real_offsets, mask, 0.0f32),
            load_f32_vector(key, real_offsets, mask, 0.0f32),
        );
        let imag_values = select(
            is_query,
            load_f32_vector(query, imag_offsets, mask, 0.0f32),
            load_f32_vector(key, imag_offsets, mask, 0.0f32),
        );
        let rotated_real = real_values * freq_real - imag_values * freq_imag;
        let rotated_imag = real_values * freq_imag + imag_values * freq_real;
        let key_mask = mask
            & cmpi(
                safe_offsets,
                query_pairs_tile,
                predicate::GreaterThanOrEqual,
            );
        store_f32_vector(query, real_offsets, rotated_real, mask & is_query);
        store_f32_vector(query, imag_offsets, rotated_imag, mask & is_query);
        store_f32_vector(key, real_offsets, rotated_real, key_mask);
        store_f32_vector(key, imag_offsets, rotated_imag, key_mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn interleaved_complex_rope_qk_f16(
        query: *mut f16,
        key: *mut f16,
        freqs: *mut f32,
        batch: i32,
        seq_len: i32,
        query_heads: i32,
        key_heads: i32,
        head_dim: i32,
        pair_count: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(pair_count, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let head_dim_half = head_dim / 2i32;
        let query_pairs = batch * seq_len * query_heads * head_dim_half;
        let is_query = cmpi(
            safe_offsets,
            broadcast_scalar(query_pairs, tile_shape),
            predicate::LessThan,
        );
        let query_pairs_tile = broadcast_scalar(query_pairs, tile_shape);
        let local_offsets = select(is_query, safe_offsets, safe_offsets - query_pairs_tile);
        let heads = select(
            is_query,
            broadcast_scalar(query_heads, tile_shape),
            broadcast_scalar(key_heads, tile_shape),
        );
        let dim = local_offsets
            - (local_offsets / broadcast_scalar(head_dim_half, tile_shape))
                * broadcast_scalar(head_dim_half, tile_shape);
        let head_linear = local_offsets / broadcast_scalar(head_dim_half, tile_shape);
        let seq = head_linear / heads
            - (head_linear / (heads * broadcast_scalar(seq_len, tile_shape)))
                * broadcast_scalar(seq_len, tile_shape);
        let batch_index = head_linear / (heads * broadcast_scalar(seq_len, tile_shape));
        let head =
            head_linear - (batch_index * broadcast_scalar(seq_len, tile_shape) + seq) * heads;

        let real_dim = dim * broadcast_scalar(2i32, tile_shape);
        let imag_dim = real_dim + broadcast_scalar(1i32, tile_shape);
        let freq_row = seq * broadcast_scalar(head_dim, tile_shape);
        let freq_real = load_f32_vector(freqs, freq_row + real_dim, mask, 1.0f32);
        let freq_imag = load_f32_vector(freqs, freq_row + imag_dim, mask, 0.0f32);

        let element_base = ((batch_index * broadcast_scalar(seq_len, tile_shape) + seq) * heads
            + head)
            * broadcast_scalar(head_dim, tile_shape);
        let real_offsets = element_base + real_dim;
        let imag_offsets = element_base + imag_dim;
        let real_values = select(
            is_query,
            load_f16_vector_as_f32(query, real_offsets, mask),
            load_f16_vector_as_f32(key, real_offsets, mask),
        );
        let imag_values = select(
            is_query,
            load_f16_vector_as_f32(query, imag_offsets, mask),
            load_f16_vector_as_f32(key, imag_offsets, mask),
        );
        let rotated_real = real_values * freq_real - imag_values * freq_imag;
        let rotated_imag = real_values * freq_imag + imag_values * freq_real;
        let key_mask = mask
            & cmpi(
                safe_offsets,
                query_pairs_tile,
                predicate::GreaterThanOrEqual,
            );
        store_f16_vector_from_f32(query, real_offsets, rotated_real, mask & is_query);
        store_f16_vector_from_f32(query, imag_offsets, rotated_imag, mask & is_query);
        store_f16_vector_from_f32(key, real_offsets, rotated_real, key_mask);
        store_f16_vector_from_f32(key, imag_offsets, rotated_imag, key_mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn interleaved_complex_rope_qk_bf16(
        query: *mut bf16,
        key: *mut bf16,
        freqs: *mut f32,
        batch: i32,
        seq_len: i32,
        query_heads: i32,
        key_heads: i32,
        head_dim: i32,
        pair_count: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(pair_count, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let head_dim_half = head_dim / 2i32;
        let query_pairs = batch * seq_len * query_heads * head_dim_half;
        let is_query = cmpi(
            safe_offsets,
            broadcast_scalar(query_pairs, tile_shape),
            predicate::LessThan,
        );
        let query_pairs_tile = broadcast_scalar(query_pairs, tile_shape);
        let local_offsets = select(is_query, safe_offsets, safe_offsets - query_pairs_tile);
        let heads = select(
            is_query,
            broadcast_scalar(query_heads, tile_shape),
            broadcast_scalar(key_heads, tile_shape),
        );
        let dim = local_offsets
            - (local_offsets / broadcast_scalar(head_dim_half, tile_shape))
                * broadcast_scalar(head_dim_half, tile_shape);
        let head_linear = local_offsets / broadcast_scalar(head_dim_half, tile_shape);
        let seq = head_linear / heads
            - (head_linear / (heads * broadcast_scalar(seq_len, tile_shape)))
                * broadcast_scalar(seq_len, tile_shape);
        let batch_index = head_linear / (heads * broadcast_scalar(seq_len, tile_shape));
        let head =
            head_linear - (batch_index * broadcast_scalar(seq_len, tile_shape) + seq) * heads;

        let real_dim = dim * broadcast_scalar(2i32, tile_shape);
        let imag_dim = real_dim + broadcast_scalar(1i32, tile_shape);
        let freq_row = seq * broadcast_scalar(head_dim, tile_shape);
        let freq_real = load_f32_vector(freqs, freq_row + real_dim, mask, 1.0f32);
        let freq_imag = load_f32_vector(freqs, freq_row + imag_dim, mask, 0.0f32);

        let element_base = ((batch_index * broadcast_scalar(seq_len, tile_shape) + seq) * heads
            + head)
            * broadcast_scalar(head_dim, tile_shape);
        let real_offsets = element_base + real_dim;
        let imag_offsets = element_base + imag_dim;
        let real_values = select(
            is_query,
            load_bf16_vector_as_f32(query, real_offsets, mask),
            load_bf16_vector_as_f32(key, real_offsets, mask),
        );
        let imag_values = select(
            is_query,
            load_bf16_vector_as_f32(query, imag_offsets, mask),
            load_bf16_vector_as_f32(key, imag_offsets, mask),
        );
        let rotated_real = real_values * freq_real - imag_values * freq_imag;
        let rotated_imag = real_values * freq_imag + imag_values * freq_real;
        let key_mask = mask
            & cmpi(
                safe_offsets,
                query_pairs_tile,
                predicate::GreaterThanOrEqual,
            );
        store_bf16_vector_from_f32(query, real_offsets, rotated_real, mask & is_query);
        store_bf16_vector_from_f32(query, imag_offsets, rotated_imag, mask & is_query);
        store_bf16_vector_from_f32(key, real_offsets, rotated_real, key_mask);
        store_bf16_vector_from_f32(key, imag_offsets, rotated_imag, key_mask);
    }

    fn load_f32_vector(
        ptr: *mut f32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: f32,
    ) -> Tile<f32, { [128] }> {
        load_vector(ptr, offsets, mask, fill)
    }

    fn store_f32_vector(
        ptr: *mut f32,
        offsets: Tile<i32, { [128] }>,
        value: Tile<f32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        store_vector(ptr, offsets, value, mask);
    }

    #[cfg(feature = "dtype-f16")]
    fn load_f16_vector_as_f32(
        ptr: *mut f16,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let input_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(ptr);
        let input_base: PointerTile<*mut f16, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f16, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut f16, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<f16, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        let values: Tile<f32, { [128] }> = convert_tile(result.0);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
        select(mask, values, zero)
    }

    #[cfg(feature = "dtype-f16")]
    fn store_f16_vector_from_f32(
        ptr: *mut f16,
        offsets: Tile<i32, { [128] }>,
        value: Tile<f32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let value: Tile<f16, { [128] }> = convert_tile(value);
        store_vector(ptr, offsets, value, mask);
    }

    #[cfg(feature = "dtype-bf16")]
    fn load_bf16_vector_as_f32(
        ptr: *mut bf16,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let input_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(ptr);
        let input_base: PointerTile<*mut bf16, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut bf16, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut bf16, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<bf16, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        let values: Tile<f32, { [128] }> = convert_tile(result.0);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
        select(mask, values, zero)
    }

    #[cfg(feature = "dtype-bf16")]
    fn store_bf16_vector_from_f32(
        ptr: *mut bf16,
        offsets: Tile<i32, { [128] }>,
        value: Tile<f32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let value: Tile<bf16, { [128] }> = convert_tile(value);
        store_vector(ptr, offsets, value, mask);
    }

    fn load_vector<T: ElementType>(
        input: *mut T,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: T,
    ) -> Tile<T, { [128] }> {
        let input_base: PointerTile<*mut T, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut T, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut T, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut T, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<T, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(fill),
            None,
            Latency::<0>,
        );
        result.0
    }

    fn store_vector<T: ElementType>(
        out: *mut T,
        offsets: Tile<i32, { [128] }>,
        values: Tile<T, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let out_base: PointerTile<*mut T, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut T, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut T, { [128] }> = out_base.broadcast(const_shape![128]);
        let out_ptrs: PointerTile<*mut T, { [128] }> = out_ptrs.offset_tile(offsets);
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
