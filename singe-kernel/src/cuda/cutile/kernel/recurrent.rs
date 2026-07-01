#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn gated_delta_rule_preprocess_f32(
        beta_out: *mut f32,
        g_out: *mut f32,
        b_in: *mut f32,
        a_in: *mut f32,
        a_log: *mut f32,
        dt_bias: *mut f32,
        hidden: i32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];

        // Preprocess one flattened (token, hidden) element per lane.
        // The hidden column selects the per-channel A-log and dt-bias parameters.
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let column = safe_offsets
            - (safe_offsets / broadcast_scalar(hidden, tile_shape))
                * broadcast_scalar(hidden, tile_shape);

        let b = load_f32_vector(b_in, safe_offsets, mask);
        let a = load_f32_vector(a_in, safe_offsets, mask);
        let a_log_values = load_f32_vector(a_log, column, mask);
        let dt_bias_values = load_f32_vector(dt_bias, column, mask);

        let zero = constant(0.0f32, tile_shape);
        let one = constant(1.0f32, tile_shape);

        // Beta is sigmoid(b).
        // G is the negative decay term -exp(a_log) * softplus(a + dt_bias).
        // This matches the recurrent gated-delta-rule reference kernel.
        let beta = one / (one + exp(zero - b));
        let softplus_input = a + dt_bias_values;
        let softplus = stable_softplus_f32(softplus_input);
        let g = zero - exp(a_log_values) * softplus;

        store_f32_vector(beta_out, offsets, beta, mask);
        store_f32_vector(g_out, offsets, g, mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn gated_delta_rule_preprocess_f16(
        beta_out: *mut f16,
        g_out: *mut f16,
        b_in: *mut f16,
        a_in: *mut f16,
        a_log: *mut f16,
        dt_bias: *mut f16,
        hidden: i32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let column = safe_offsets
            - (safe_offsets / broadcast_scalar(hidden, tile_shape))
                * broadcast_scalar(hidden, tile_shape);

        let b = load_f16_vector_as_f32(b_in, safe_offsets, mask);
        let a = load_f16_vector_as_f32(a_in, safe_offsets, mask);
        let a_log_values = load_f16_vector_as_f32(a_log, column, mask);
        let dt_bias_values = load_f16_vector_as_f32(dt_bias, column, mask);

        let zero = constant(0.0f32, tile_shape);
        let one = constant(1.0f32, tile_shape);
        let beta = one / (one + exp(zero - b));
        let softplus_input = a + dt_bias_values;
        let softplus = stable_softplus_f32(softplus_input);
        let g = zero - exp(a_log_values) * softplus;

        store_f16_vector_from_f32(beta_out, offsets, beta, mask);
        store_f16_vector_from_f32(g_out, offsets, g, mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn gated_delta_rule_preprocess_f16_g_f32(
        beta_out: *mut f16,
        g_out: *mut f32,
        b_in: *mut f16,
        a_in: *mut f16,
        a_log: *mut f16,
        dt_bias: *mut f16,
        hidden: i32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let column = safe_offsets
            - (safe_offsets / broadcast_scalar(hidden, tile_shape))
                * broadcast_scalar(hidden, tile_shape);

        let b = load_f16_vector_as_f32(b_in, safe_offsets, mask);
        let a = load_f16_vector_as_f32(a_in, safe_offsets, mask);
        let a_log_values = load_f16_vector_as_f32(a_log, column, mask);
        let dt_bias_values = load_f16_vector_as_f32(dt_bias, column, mask);

        let zero = constant(0.0f32, tile_shape);
        let one = constant(1.0f32, tile_shape);
        let beta = one / (one + exp(zero - b));
        let softplus_input = a + dt_bias_values;
        let softplus = stable_softplus_f32(softplus_input);
        let g = zero - exp(a_log_values) * softplus;

        store_f16_vector_from_f32(beta_out, offsets, beta, mask);
        store_f32_vector(g_out, offsets, g, mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn gated_delta_rule_preprocess_bf16(
        beta_out: *mut bf16,
        g_out: *mut bf16,
        b_in: *mut bf16,
        a_in: *mut bf16,
        a_log: *mut bf16,
        dt_bias: *mut bf16,
        hidden: i32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let column = safe_offsets
            - (safe_offsets / broadcast_scalar(hidden, tile_shape))
                * broadcast_scalar(hidden, tile_shape);

        let b = load_bf16_vector_as_f32(b_in, safe_offsets, mask);
        let a = load_bf16_vector_as_f32(a_in, safe_offsets, mask);
        let a_log_values = load_bf16_vector_as_f32(a_log, column, mask);
        let dt_bias_values = load_bf16_vector_as_f32(dt_bias, column, mask);

        let zero = constant(0.0f32, tile_shape);
        let one = constant(1.0f32, tile_shape);
        let beta = one / (one + exp(zero - b));
        let softplus_input = a + dt_bias_values;
        let softplus = stable_softplus_f32(softplus_input);
        let g = zero - exp(a_log_values) * softplus;

        store_bf16_vector_from_f32(beta_out, offsets, beta, mask);
        store_bf16_vector_from_f32(g_out, offsets, g, mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn gated_delta_rule_preprocess_bf16_g_f32(
        beta_out: *mut bf16,
        g_out: *mut f32,
        b_in: *mut bf16,
        a_in: *mut bf16,
        a_log: *mut bf16,
        dt_bias: *mut bf16,
        hidden: i32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let column = safe_offsets
            - (safe_offsets / broadcast_scalar(hidden, tile_shape))
                * broadcast_scalar(hidden, tile_shape);

        let b = load_bf16_vector_as_f32(b_in, safe_offsets, mask);
        let a = load_bf16_vector_as_f32(a_in, safe_offsets, mask);
        let a_log_values = load_bf16_vector_as_f32(a_log, column, mask);
        let dt_bias_values = load_bf16_vector_as_f32(dt_bias, column, mask);

        let zero = constant(0.0f32, tile_shape);
        let one = constant(1.0f32, tile_shape);
        let beta = one / (one + exp(zero - b));
        let softplus_input = a + dt_bias_values;
        let softplus = stable_softplus_f32(softplus_input);
        let g = zero - exp(a_log_values) * softplus;

        store_bf16_vector_from_f32(beta_out, offsets, beta, mask);
        store_f32_vector(g_out, offsets, g, mask);
    }

    fn stable_softplus_f32(input: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let one = constant(1.0f32, tile_shape);
        let threshold = constant(20.0f32, tile_shape);
        let is_large = cmpf(
            input,
            threshold,
            predicate::GreaterThan,
            cmp_ordering::Ordered,
        );

        // Avoid overflow from exp(x) for large positive x.
        // Return x once softplus(x) is numerically indistinguishable from x.
        select(is_large, input, log(one + exp(input)))
    }

    #[cutile::entry()]
    pub unsafe fn recurrent_gated_delta_rule_f32(
        out: *mut f32,
        final_state: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        gate: *mut f32,
        beta: *mut f32,
        initial_state: *mut f32,
        batch: i32,
        time: i32,
        query_heads: i32,
        value_heads: i32,
        qk_dim: i32,
        value_dim: i32,
        scale: f32,
        has_initial_state: i32,
        output_final_state: i32,
        use_qk_l2norm: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];

        // A CTA owns one (batch, value-head, value-dim) stream.
        // Lanes span the qk dimension of that stream's recurrent state vector.
        let k_offsets: Tile<i32, { [128] }> = iota(tile_shape);
        let k_mask = cmpi(
            k_offsets,
            broadcast_scalar(qk_dim, tile_shape),
            predicate::LessThan,
        );
        let idx_bhv = pid.0;
        let idx_v = pid.1;
        let idx_b = idx_bhv / value_heads;
        let idx_hv = idx_bhv - idx_b * value_heads;
        let heads_per_group = value_heads / query_heads;
        let idx_h = idx_hv / heads_per_group;

        let state_offsets = (((broadcast_scalar(idx_b, tile_shape)
            * broadcast_scalar(value_heads, tile_shape)
            + broadcast_scalar(idx_hv, tile_shape))
            * broadcast_scalar(qk_dim, tile_shape)
            + k_offsets)
            * broadcast_scalar(value_dim, tile_shape))
            + broadcast_scalar(idx_v, tile_shape);
        let mut state = if has_initial_state != 0 {
            load_f32_vector(initial_state, state_offsets, k_mask)
        } else {
            constant(0.0f32, tile_shape)
        };

        let one_scalar: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
        for idx_t in 0i32..time {
            // Query/key can optionally be L2-normalized before applying the attention scale.
            // This matches the TileGym recurrent kernel option.
            let qk_offsets = (((broadcast_scalar(idx_b, tile_shape)
                * broadcast_scalar(time, tile_shape)
                + broadcast_scalar(idx_t, tile_shape))
                * broadcast_scalar(query_heads, tile_shape)
                + broadcast_scalar(idx_h, tile_shape))
                * broadcast_scalar(qk_dim, tile_shape))
                + k_offsets;
            let mut query_values = load_f32_vector(query, qk_offsets, k_mask);
            let mut key_values = load_f32_vector(key, qk_offsets, k_mask);
            if use_qk_l2norm != 0 {
                let query_norm_scalar: Tile<f32, { [] }> =
                    reduce_sum(query_values * query_values, 0i32);
                let key_norm_scalar: Tile<f32, { [] }> = reduce_sum(key_values * key_values, 0i32);
                let query_norm: Tile<f32, { [1] }> = query_norm_scalar.reshape(const_shape![1]);
                let key_norm: Tile<f32, { [1] }> = key_norm_scalar.reshape(const_shape![1]);
                let query_scale = l2norm_scale(query_norm).broadcast(tile_shape);
                let key_scale = l2norm_scale(key_norm).broadcast(tile_shape);
                query_values = query_values * query_scale;
                key_values = key_values * key_scale;
            }
            query_values = query_values * broadcast_scalar(scale, tile_shape);

            let value_offset = ((idx_b * time + idx_t) * value_heads + idx_hv) * value_dim + idx_v;
            let gate_offset = (idx_b * time + idx_t) * value_heads + idx_hv;
            let value_t = load_f32_scalar(value, value_offset);
            let gate_t = load_f32_scalar(gate, gate_offset);
            let beta_t = load_f32_scalar(beta, gate_offset);
            let gate_decay = exp(gate_t).reshape(const_shape![1]).broadcast(tile_shape);
            state = state * gate_decay;

            // Delta-rule update reads the current memory K^T state.
            // It corrects toward V by beta and writes the rank-1 key * delta update back to state.
            let kv_memory_scalar: Tile<f32, { [] }> = reduce_sum(state * key_values, 0i32);
            let kv_memory: Tile<f32, { [1] }> = kv_memory_scalar.reshape(const_shape![1]);
            let delta: Tile<f32, { [1] }> = (value_t - kv_memory) * beta_t;
            state = state + key_values * delta.reshape(const_shape![1]).broadcast(tile_shape);
            let out_t_scalar: Tile<f32, { [] }> = reduce_sum(state * query_values, 0i32);
            let out_t: Tile<f32, { [1] }> = out_t_scalar.reshape(const_shape![1]);
            store_f32_scalar(out, value_offset, out_t);
        }

        if output_final_state != 0 {
            store_f32_vector(final_state, state_offsets, state, k_mask);
        }

        let _ = batch;
        let _ = one_scalar;
    }

    #[cfg(feature = "dtype-f32")]
    #[cutile::entry()]
    pub unsafe fn recurrent_gated_delta_rule_decode_f32(
        out: *mut f32,
        final_state: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        gate: *mut f32,
        beta: *mut f32,
        initial_state: *mut f32,
        batch: i32,
        query_heads: i32,
        value_heads: i32,
        qk_dim: i32,
        value_dim: i32,
        scale: f32,
        output_final_state: i32,
        use_qk_l2norm: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let k_offsets: Tile<i32, { [128] }> = iota(tile_shape);
        let k_mask = cmpi(
            k_offsets,
            broadcast_scalar(qk_dim, tile_shape),
            predicate::LessThan,
        );
        let idx_bhv = pid.0;
        let idx_v = pid.1;
        let idx_b = idx_bhv / value_heads;
        let idx_hv = idx_bhv - idx_b * value_heads;
        let heads_per_group = value_heads / query_heads;
        let idx_h = idx_hv / heads_per_group;

        let state_offsets = (((broadcast_scalar(idx_b, tile_shape)
            * broadcast_scalar(value_heads, tile_shape)
            + broadcast_scalar(idx_hv, tile_shape))
            * broadcast_scalar(qk_dim, tile_shape)
            + k_offsets)
            * broadcast_scalar(value_dim, tile_shape))
            + broadcast_scalar(idx_v, tile_shape);
        let mut state = load_f32_vector(initial_state, state_offsets, k_mask);

        let qk_offsets = ((broadcast_scalar(idx_b, tile_shape)
            * broadcast_scalar(query_heads, tile_shape)
            + broadcast_scalar(idx_h, tile_shape))
            * broadcast_scalar(qk_dim, tile_shape))
            + k_offsets;
        let mut query_values = load_f32_vector(query, qk_offsets, k_mask);
        let mut key_values = load_f32_vector(key, qk_offsets, k_mask);
        if use_qk_l2norm != 0 {
            let query_norm_scalar: Tile<f32, { [] }> =
                reduce_sum(query_values * query_values, 0i32);
            let key_norm_scalar: Tile<f32, { [] }> = reduce_sum(key_values * key_values, 0i32);
            let query_norm: Tile<f32, { [1] }> = query_norm_scalar.reshape(const_shape![1]);
            let key_norm: Tile<f32, { [1] }> = key_norm_scalar.reshape(const_shape![1]);
            let query_scale = l2norm_scale(query_norm).broadcast(tile_shape);
            let key_scale = l2norm_scale(key_norm).broadcast(tile_shape);
            query_values = query_values * query_scale;
            key_values = key_values * key_scale;
        }
        query_values = query_values * broadcast_scalar(scale, tile_shape);

        let value_offset = (idx_b * value_heads + idx_hv) * value_dim + idx_v;
        let gate_offset = idx_b * value_heads + idx_hv;
        let value_t = load_f32_scalar(value, value_offset);
        let gate_t = load_f32_scalar(gate, gate_offset);
        let beta_t = load_f32_scalar(beta, gate_offset);
        let gate_decay = exp(gate_t).reshape(const_shape![1]).broadcast(tile_shape);
        state = state * gate_decay;

        let kv_memory_scalar: Tile<f32, { [] }> = reduce_sum(state * key_values, 0i32);
        let out_base_scalar: Tile<f32, { [] }> = reduce_sum(state * query_values, 0i32);
        let key_query_scalar: Tile<f32, { [] }> = reduce_sum(key_values * query_values, 0i32);
        let kv_memory: Tile<f32, { [1] }> = kv_memory_scalar.reshape(const_shape![1]);
        let out_base: Tile<f32, { [1] }> = out_base_scalar.reshape(const_shape![1]);
        let key_query: Tile<f32, { [1] }> = key_query_scalar.reshape(const_shape![1]);
        let delta: Tile<f32, { [1] }> = (value_t - kv_memory) * beta_t;
        let out_t = out_base + delta * key_query;
        state = state + key_values * delta.reshape(const_shape![1]).broadcast(tile_shape);
        store_f32_scalar(out, value_offset, out_t);

        if output_final_state != 0 {
            store_f32_vector(final_state, state_offsets, state, k_mask);
        }

        let _ = batch;
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn recurrent_gated_delta_rule_decode_f16(
        out: *mut f16,
        final_state: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        gate: *mut f16,
        beta: *mut f16,
        initial_state: *mut f16,
        batch: i32,
        query_heads: i32,
        value_heads: i32,
        qk_dim: i32,
        value_dim: i32,
        scale: f32,
        output_final_state: i32,
        use_qk_l2norm: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let k_offsets: Tile<i32, { [128] }> = iota(tile_shape);
        let k_mask = cmpi(
            k_offsets,
            broadcast_scalar(qk_dim, tile_shape),
            predicate::LessThan,
        );
        let idx_bhv = pid.0;
        let idx_v = pid.1;
        let idx_b = idx_bhv / value_heads;
        let idx_hv = idx_bhv - idx_b * value_heads;
        let heads_per_group = value_heads / query_heads;
        let idx_h = idx_hv / heads_per_group;

        let state_offsets = (((broadcast_scalar(idx_b, tile_shape)
            * broadcast_scalar(value_heads, tile_shape)
            + broadcast_scalar(idx_hv, tile_shape))
            * broadcast_scalar(qk_dim, tile_shape)
            + k_offsets)
            * broadcast_scalar(value_dim, tile_shape))
            + broadcast_scalar(idx_v, tile_shape);
        let mut state = load_f16_vector_as_f32(initial_state, state_offsets, k_mask);

        let qk_offsets = ((broadcast_scalar(idx_b, tile_shape)
            * broadcast_scalar(query_heads, tile_shape)
            + broadcast_scalar(idx_h, tile_shape))
            * broadcast_scalar(qk_dim, tile_shape))
            + k_offsets;
        let mut query_values = load_f16_vector_as_f32(query, qk_offsets, k_mask);
        let mut key_values = load_f16_vector_as_f32(key, qk_offsets, k_mask);
        if use_qk_l2norm != 0 {
            let query_norm_scalar: Tile<f32, { [] }> =
                reduce_sum(query_values * query_values, 0i32);
            let key_norm_scalar: Tile<f32, { [] }> = reduce_sum(key_values * key_values, 0i32);
            let query_norm: Tile<f32, { [1] }> = query_norm_scalar.reshape(const_shape![1]);
            let key_norm: Tile<f32, { [1] }> = key_norm_scalar.reshape(const_shape![1]);
            let query_scale = l2norm_scale(query_norm).broadcast(tile_shape);
            let key_scale = l2norm_scale(key_norm).broadcast(tile_shape);
            query_values = query_values * query_scale;
            key_values = key_values * key_scale;
        }
        query_values = query_values * broadcast_scalar(scale, tile_shape);

        let value_offset = (idx_b * value_heads + idx_hv) * value_dim + idx_v;
        let gate_offset = idx_b * value_heads + idx_hv;
        let value_t = load_f16_scalar_as_f32(value, value_offset);
        let gate_t = load_f16_scalar_as_f32(gate, gate_offset);
        let beta_t = load_f16_scalar_as_f32(beta, gate_offset);
        let gate_decay = exp(gate_t).reshape(const_shape![1]).broadcast(tile_shape);
        state = state * gate_decay;

        let kv_memory_scalar: Tile<f32, { [] }> = reduce_sum(state * key_values, 0i32);
        let out_base_scalar: Tile<f32, { [] }> = reduce_sum(state * query_values, 0i32);
        let key_query_scalar: Tile<f32, { [] }> = reduce_sum(key_values * query_values, 0i32);
        let kv_memory: Tile<f32, { [1] }> = kv_memory_scalar.reshape(const_shape![1]);
        let out_base: Tile<f32, { [1] }> = out_base_scalar.reshape(const_shape![1]);
        let key_query: Tile<f32, { [1] }> = key_query_scalar.reshape(const_shape![1]);
        let delta: Tile<f32, { [1] }> = (value_t - kv_memory) * beta_t;
        let out_t = out_base + delta * key_query;
        state = state + key_values * delta.reshape(const_shape![1]).broadcast(tile_shape);
        store_f16_scalar_from_f32(out, value_offset, out_t);

        if output_final_state != 0 {
            store_f16_vector_from_f32(final_state, state_offsets, state, k_mask);
        }

        let _ = batch;
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn recurrent_gated_delta_rule_decode_bf16(
        out: *mut bf16,
        final_state: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        gate: *mut bf16,
        beta: *mut bf16,
        initial_state: *mut bf16,
        batch: i32,
        query_heads: i32,
        value_heads: i32,
        qk_dim: i32,
        value_dim: i32,
        scale: f32,
        output_final_state: i32,
        use_qk_l2norm: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let k_offsets: Tile<i32, { [128] }> = iota(tile_shape);
        let k_mask = cmpi(
            k_offsets,
            broadcast_scalar(qk_dim, tile_shape),
            predicate::LessThan,
        );
        let idx_bhv = pid.0;
        let idx_v = pid.1;
        let idx_b = idx_bhv / value_heads;
        let idx_hv = idx_bhv - idx_b * value_heads;
        let heads_per_group = value_heads / query_heads;
        let idx_h = idx_hv / heads_per_group;

        let state_offsets = (((broadcast_scalar(idx_b, tile_shape)
            * broadcast_scalar(value_heads, tile_shape)
            + broadcast_scalar(idx_hv, tile_shape))
            * broadcast_scalar(qk_dim, tile_shape)
            + k_offsets)
            * broadcast_scalar(value_dim, tile_shape))
            + broadcast_scalar(idx_v, tile_shape);
        let mut state = load_bf16_vector_as_f32(initial_state, state_offsets, k_mask);

        let qk_offsets = ((broadcast_scalar(idx_b, tile_shape)
            * broadcast_scalar(query_heads, tile_shape)
            + broadcast_scalar(idx_h, tile_shape))
            * broadcast_scalar(qk_dim, tile_shape))
            + k_offsets;
        let mut query_values = load_bf16_vector_as_f32(query, qk_offsets, k_mask);
        let mut key_values = load_bf16_vector_as_f32(key, qk_offsets, k_mask);
        if use_qk_l2norm != 0 {
            let query_norm_scalar: Tile<f32, { [] }> =
                reduce_sum(query_values * query_values, 0i32);
            let key_norm_scalar: Tile<f32, { [] }> = reduce_sum(key_values * key_values, 0i32);
            let query_norm: Tile<f32, { [1] }> = query_norm_scalar.reshape(const_shape![1]);
            let key_norm: Tile<f32, { [1] }> = key_norm_scalar.reshape(const_shape![1]);
            let query_scale = l2norm_scale(query_norm).broadcast(tile_shape);
            let key_scale = l2norm_scale(key_norm).broadcast(tile_shape);
            query_values = query_values * query_scale;
            key_values = key_values * key_scale;
        }
        query_values = query_values * broadcast_scalar(scale, tile_shape);

        let value_offset = (idx_b * value_heads + idx_hv) * value_dim + idx_v;
        let gate_offset = idx_b * value_heads + idx_hv;
        let value_t = load_bf16_scalar_as_f32(value, value_offset);
        let gate_t = load_bf16_scalar_as_f32(gate, gate_offset);
        let beta_t = load_bf16_scalar_as_f32(beta, gate_offset);
        let gate_decay = exp(gate_t).reshape(const_shape![1]).broadcast(tile_shape);
        state = state * gate_decay;

        let kv_memory_scalar: Tile<f32, { [] }> = reduce_sum(state * key_values, 0i32);
        let out_base_scalar: Tile<f32, { [] }> = reduce_sum(state * query_values, 0i32);
        let key_query_scalar: Tile<f32, { [] }> = reduce_sum(key_values * query_values, 0i32);
        let kv_memory: Tile<f32, { [1] }> = kv_memory_scalar.reshape(const_shape![1]);
        let out_base: Tile<f32, { [1] }> = out_base_scalar.reshape(const_shape![1]);
        let key_query: Tile<f32, { [1] }> = key_query_scalar.reshape(const_shape![1]);
        let delta: Tile<f32, { [1] }> = (value_t - kv_memory) * beta_t;
        let out_t = out_base + delta * key_query;
        state = state + key_values * delta.reshape(const_shape![1]).broadcast(tile_shape);
        store_bf16_scalar_from_f32(out, value_offset, out_t);

        if output_final_state != 0 {
            store_bf16_vector_from_f32(final_state, state_offsets, state, k_mask);
        }

        let _ = batch;
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn recurrent_gated_delta_rule_f16(
        out: *mut f16,
        final_state: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        gate: *mut f16,
        beta: *mut f16,
        initial_state: *mut f16,
        batch: i32,
        time: i32,
        query_heads: i32,
        value_heads: i32,
        qk_dim: i32,
        value_dim: i32,
        scale: f32,
        has_initial_state: i32,
        output_final_state: i32,
        use_qk_l2norm: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let k_offsets: Tile<i32, { [128] }> = iota(tile_shape);
        let k_mask = cmpi(
            k_offsets,
            broadcast_scalar(qk_dim, tile_shape),
            predicate::LessThan,
        );
        let idx_bhv = pid.0;
        let idx_v = pid.1;
        let idx_b = idx_bhv / value_heads;
        let idx_hv = idx_bhv - idx_b * value_heads;
        let heads_per_group = value_heads / query_heads;
        let idx_h = idx_hv / heads_per_group;

        let state_offsets = (((broadcast_scalar(idx_b, tile_shape)
            * broadcast_scalar(value_heads, tile_shape)
            + broadcast_scalar(idx_hv, tile_shape))
            * broadcast_scalar(qk_dim, tile_shape)
            + k_offsets)
            * broadcast_scalar(value_dim, tile_shape))
            + broadcast_scalar(idx_v, tile_shape);
        let mut state = if has_initial_state != 0 {
            load_f16_vector_as_f32(initial_state, state_offsets, k_mask)
        } else {
            constant(0.0f32, tile_shape)
        };

        for idx_t in 0i32..time {
            let qk_offsets = (((broadcast_scalar(idx_b, tile_shape)
                * broadcast_scalar(time, tile_shape)
                + broadcast_scalar(idx_t, tile_shape))
                * broadcast_scalar(query_heads, tile_shape)
                + broadcast_scalar(idx_h, tile_shape))
                * broadcast_scalar(qk_dim, tile_shape))
                + k_offsets;
            let mut query_values = load_f16_vector_as_f32(query, qk_offsets, k_mask);
            let mut key_values = load_f16_vector_as_f32(key, qk_offsets, k_mask);
            if use_qk_l2norm != 0 {
                let query_norm_scalar: Tile<f32, { [] }> =
                    reduce_sum(query_values * query_values, 0i32);
                let key_norm_scalar: Tile<f32, { [] }> = reduce_sum(key_values * key_values, 0i32);
                let query_norm: Tile<f32, { [1] }> = query_norm_scalar.reshape(const_shape![1]);
                let key_norm: Tile<f32, { [1] }> = key_norm_scalar.reshape(const_shape![1]);
                let query_scale = l2norm_scale(query_norm).broadcast(tile_shape);
                let key_scale = l2norm_scale(key_norm).broadcast(tile_shape);
                query_values = query_values * query_scale;
                key_values = key_values * key_scale;
            }
            query_values = query_values * broadcast_scalar(scale, tile_shape);

            let value_offset = ((idx_b * time + idx_t) * value_heads + idx_hv) * value_dim + idx_v;
            let gate_offset = (idx_b * time + idx_t) * value_heads + idx_hv;
            let value_t = load_f16_scalar_as_f32(value, value_offset);
            let gate_t = load_f16_scalar_as_f32(gate, gate_offset);
            let beta_t = load_f16_scalar_as_f32(beta, gate_offset);
            let gate_decay = exp(gate_t).reshape(const_shape![1]).broadcast(tile_shape);
            state = state * gate_decay;

            let kv_memory_scalar: Tile<f32, { [] }> = reduce_sum(state * key_values, 0i32);
            let kv_memory: Tile<f32, { [1] }> = kv_memory_scalar.reshape(const_shape![1]);
            let delta: Tile<f32, { [1] }> = (value_t - kv_memory) * beta_t;
            state = state + key_values * delta.reshape(const_shape![1]).broadcast(tile_shape);
            let out_t_scalar: Tile<f32, { [] }> = reduce_sum(state * query_values, 0i32);
            let out_t: Tile<f32, { [1] }> = out_t_scalar.reshape(const_shape![1]);
            store_f16_scalar_from_f32(out, value_offset, out_t);
        }

        if output_final_state != 0 {
            store_f16_vector_from_f32(final_state, state_offsets, state, k_mask);
        }

        let _ = batch;
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn recurrent_gated_delta_rule_bf16(
        out: *mut bf16,
        final_state: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        gate: *mut bf16,
        beta: *mut bf16,
        initial_state: *mut bf16,
        batch: i32,
        time: i32,
        query_heads: i32,
        value_heads: i32,
        qk_dim: i32,
        value_dim: i32,
        scale: f32,
        has_initial_state: i32,
        output_final_state: i32,
        use_qk_l2norm: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let k_offsets: Tile<i32, { [128] }> = iota(tile_shape);
        let k_mask = cmpi(
            k_offsets,
            broadcast_scalar(qk_dim, tile_shape),
            predicate::LessThan,
        );
        let idx_bhv = pid.0;
        let idx_v = pid.1;
        let idx_b = idx_bhv / value_heads;
        let idx_hv = idx_bhv - idx_b * value_heads;
        let heads_per_group = value_heads / query_heads;
        let idx_h = idx_hv / heads_per_group;

        let state_offsets = (((broadcast_scalar(idx_b, tile_shape)
            * broadcast_scalar(value_heads, tile_shape)
            + broadcast_scalar(idx_hv, tile_shape))
            * broadcast_scalar(qk_dim, tile_shape)
            + k_offsets)
            * broadcast_scalar(value_dim, tile_shape))
            + broadcast_scalar(idx_v, tile_shape);
        let mut state = if has_initial_state != 0 {
            load_bf16_vector_as_f32(initial_state, state_offsets, k_mask)
        } else {
            constant(0.0f32, tile_shape)
        };

        for idx_t in 0i32..time {
            let qk_offsets = (((broadcast_scalar(idx_b, tile_shape)
                * broadcast_scalar(time, tile_shape)
                + broadcast_scalar(idx_t, tile_shape))
                * broadcast_scalar(query_heads, tile_shape)
                + broadcast_scalar(idx_h, tile_shape))
                * broadcast_scalar(qk_dim, tile_shape))
                + k_offsets;
            let mut query_values = load_bf16_vector_as_f32(query, qk_offsets, k_mask);
            let mut key_values = load_bf16_vector_as_f32(key, qk_offsets, k_mask);
            if use_qk_l2norm != 0 {
                let query_norm_scalar: Tile<f32, { [] }> =
                    reduce_sum(query_values * query_values, 0i32);
                let key_norm_scalar: Tile<f32, { [] }> = reduce_sum(key_values * key_values, 0i32);
                let query_norm: Tile<f32, { [1] }> = query_norm_scalar.reshape(const_shape![1]);
                let key_norm: Tile<f32, { [1] }> = key_norm_scalar.reshape(const_shape![1]);
                let query_scale = l2norm_scale(query_norm).broadcast(tile_shape);
                let key_scale = l2norm_scale(key_norm).broadcast(tile_shape);
                query_values = query_values * query_scale;
                key_values = key_values * key_scale;
            }
            query_values = query_values * broadcast_scalar(scale, tile_shape);

            let value_offset = ((idx_b * time + idx_t) * value_heads + idx_hv) * value_dim + idx_v;
            let gate_offset = (idx_b * time + idx_t) * value_heads + idx_hv;
            let value_t = load_bf16_scalar_as_f32(value, value_offset);
            let gate_t = load_bf16_scalar_as_f32(gate, gate_offset);
            let beta_t = load_bf16_scalar_as_f32(beta, gate_offset);
            let gate_decay = exp(gate_t).reshape(const_shape![1]).broadcast(tile_shape);
            state = state * gate_decay;

            let kv_memory_scalar: Tile<f32, { [] }> = reduce_sum(state * key_values, 0i32);
            let kv_memory: Tile<f32, { [1] }> = kv_memory_scalar.reshape(const_shape![1]);
            let delta: Tile<f32, { [1] }> = (value_t - kv_memory) * beta_t;
            state = state + key_values * delta.reshape(const_shape![1]).broadcast(tile_shape);
            let out_t_scalar: Tile<f32, { [] }> = reduce_sum(state * query_values, 0i32);
            let out_t: Tile<f32, { [1] }> = out_t_scalar.reshape(const_shape![1]);
            store_bf16_scalar_from_f32(out, value_offset, out_t);
        }

        if output_final_state != 0 {
            store_bf16_vector_from_f32(final_state, state_offsets, state, k_mask);
        }

        let _ = batch;
    }

    fn l2norm_scale(norm_squared: Tile<f32, { [1] }>) -> Tile<f32, { [1] }> {
        let min_norm_squared: Tile<f32, { [1] }> = constant(1e-12f32, const_shape![1]);
        let norm_squared = select(
            cmpf(
                norm_squared,
                min_norm_squared,
                predicate::GreaterThan,
                cmp_ordering::Ordered,
            ),
            norm_squared,
            min_norm_squared,
        );
        rsqrt(norm_squared, ftz::Disabled)
    }

    fn load_f32_vector(
        input: *mut f32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let input_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f32, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        let zero: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
        select(mask, result.0, zero)
    }

    fn load_f32_scalar(input: *mut f32, offset: i32) -> Tile<f32, { [1] }> {
        let input_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f32, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [1] }> =
            input_base.offset_tile(broadcast_scalar(offset, const_shape![1]));
        let mask: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        let result: (Tile<f32, { [1] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        result.0
    }

    #[cfg(feature = "dtype-f16")]
    fn load_f16_scalar_as_f32(input: *mut f16, offset: i32) -> Tile<f32, { [1] }> {
        let input_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f16, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f16, { [1] }> =
            input_base.offset_tile(broadcast_scalar(offset, const_shape![1]));
        let mask: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        let result: (Tile<f16, { [1] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        convert_tile(result.0)
    }

    #[cfg(feature = "dtype-bf16")]
    fn load_bf16_scalar_as_f32(input: *mut bf16, offset: i32) -> Tile<f32, { [1] }> {
        let input_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut bf16, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut bf16, { [1] }> =
            input_base.offset_tile(broadcast_scalar(offset, const_shape![1]));
        let mask: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        let result: (Tile<bf16, { [1] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        convert_tile(result.0)
    }

    #[cfg(feature = "dtype-f16")]
    fn load_f16_vector_as_f32(
        input: *mut f16,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let input_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(input);
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

    #[cfg(feature = "dtype-bf16")]
    fn load_bf16_vector_as_f32(
        input: *mut bf16,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let input_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(input);
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

    fn store_f32_vector(
        out: *mut f32,
        offsets: Tile<i32, { [128] }>,
        values: Tile<f32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(const_shape![128]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(offsets);
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

    fn store_f32_scalar(out: *mut f32, offset: i32, value: Tile<f32, { [1] }>) {
        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [1] }> =
            out_base.offset_tile(broadcast_scalar(offset, const_shape![1]));
        let mask: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        store_ptr_tko(
            out_ptrs,
            value,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cfg(feature = "dtype-f16")]
    fn store_f16_scalar_from_f32(out: *mut f16, offset: i32, value: Tile<f32, { [1] }>) {
        let out_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f16, { [1] }> =
            out_base.offset_tile(broadcast_scalar(offset, const_shape![1]));
        let mask: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        let value: Tile<f16, { [1] }> = convert_tile(value);
        store_ptr_tko(
            out_ptrs,
            value,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cfg(feature = "dtype-bf16")]
    fn store_bf16_scalar_from_f32(out: *mut bf16, offset: i32, value: Tile<f32, { [1] }>) {
        let out_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut bf16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut bf16, { [1] }> =
            out_base.offset_tile(broadcast_scalar(offset, const_shape![1]));
        let mask: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        let value: Tile<bf16, { [1] }> = convert_tile(value);
        store_ptr_tko(
            out_ptrs,
            value,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cfg(feature = "dtype-f16")]
    fn store_f16_vector_from_f32(
        out: *mut f16,
        offsets: Tile<i32, { [128] }>,
        values: Tile<f32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let out_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_base.broadcast(const_shape![128]);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_ptrs.offset_tile(offsets);
        let values: Tile<f16, { [128] }> = convert_tile(values);
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

    #[cfg(feature = "dtype-bf16")]
    fn store_bf16_vector_from_f32(
        out: *mut bf16,
        offsets: Tile<i32, { [128] }>,
        values: Tile<f32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let out_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut bf16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut bf16, { [128] }> = out_base.broadcast(const_shape![128]);
        let out_ptrs: PointerTile<*mut bf16, { [128] }> = out_ptrs.offset_tile(offsets);
        let values: Tile<bf16, { [128] }> = convert_tile(values);
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
