#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn moe_align_block_size_i32(
        topk_ids: *mut i32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        num_tokens_post_pad: *mut i32,
        cumsum: *mut i32,
        max_expert_count: *mut i32,
        element_count: i32,
        expert_count: i32,
        block_size: i32,
        sorted_token_ids_len: i32,
        expert_ids_len: i32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();

        // This combines the TileGym multi-stage align-block flow into one pass over experts.
        // Each lane owns one position in the padded sorted token buffer.
        // It discovers which expert block that position belongs to.
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let one: Tile<i32, { [128] }> = constant(1i32, tile_shape);
        let zero: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let all_lanes: Tile<bool, { [128] }> = constant(true, tile_shape);
        let sentinel = broadcast_scalar(element_count, tile_shape);

        let mut running = zero;
        let mut max_count = zero;
        let mut selected_expert = constant(-1i32, tile_shape);
        let mut selected_start = zero;
        let mut selected_count = zero;
        for expert in 0i32..expert_count {
            let expert_tile = broadcast_scalar(expert, tile_shape);

            // Count how many flattened top-k token assignments route to this expert.
            // Round the count up to `block_size` for block-M GEMM.
            let mut count = zero;
            for token in 0i32..element_count {
                let token_id = load_i32_vector(
                    topk_ids,
                    broadcast_scalar(token, tile_shape),
                    all_lanes,
                    0i32,
                );
                let matches = cmpi(token_id, expert_tile, predicate::Equal);
                count = count + select(matches, one, zero);
            }

            let padded_count = ((count + broadcast_scalar(block_size - 1, tile_shape))
                / broadcast_scalar(block_size, tile_shape))
                * broadcast_scalar(block_size, tile_shape);

            // `cumsum[expert]` is the starting offset of this expert's padded token block in `sorted_token_ids`.
            let cumsum_mask = valid & cmpi(offsets, expert_tile, predicate::Equal);
            store_i32_vector(cumsum, offsets, running, cumsum_mask);

            max_count = select(
                cmpi(count, max_count, predicate::GreaterThan),
                count,
                max_count,
            );

            let in_sorted_range = valid
                & cmpi(offsets, running, predicate::GreaterThanOrEqual)
                & cmpi(offsets, running + padded_count, predicate::LessThan);
            selected_expert = select(in_sorted_range, expert_tile, selected_expert);
            selected_start = select(in_sorted_range, running, selected_start);
            selected_count = select(in_sorted_range, count, selected_count);

            // `expert_ids` is indexed by padded block, not by token.
            // Each block stores the expert whose weight matrix should be used by fused_moe.
            let block_start = running / broadcast_scalar(block_size, tile_shape);
            let block_count = padded_count / broadcast_scalar(block_size, tile_shape);
            let expert_block_mask = valid
                & cmpi(
                    offsets,
                    broadcast_scalar(expert_ids_len, tile_shape),
                    predicate::LessThan,
                )
                & cmpi(offsets, block_start, predicate::GreaterThanOrEqual)
                & cmpi(offsets, block_start + block_count, predicate::LessThan);
            store_i32_vector(expert_ids, offsets, expert_tile, expert_block_mask);

            running = running + padded_count;
        }

        let final_cumsum_mask = valid
            & cmpi(
                offsets,
                broadcast_scalar(expert_count, tile_shape),
                predicate::Equal,
            );
        store_i32_vector(cumsum, offsets, running, final_cumsum_mask);

        let first_lane = valid & cmpi(offsets, zero, predicate::Equal);
        store_i32_vector(num_tokens_post_pad, zero, running, first_lane);
        store_i32_vector(max_expert_count, zero, max_count, first_lane);

        let sorted_mask = valid
            & cmpi(
                offsets,
                broadcast_scalar(sorted_token_ids_len, tile_shape),
                predicate::LessThan,
            );
        let local_index = offsets - selected_start;
        let mut seen = zero;
        let mut token_value = sentinel;

        // Reconstruct the token id at this expert-local rank by scanning the original top-k assignments.
        // Padded slots receive the sentinel.
        for token in 0i32..element_count {
            let token_id = load_i32_vector(
                topk_ids,
                broadcast_scalar(token, tile_shape),
                all_lanes,
                0i32,
            );
            let matches = cmpi(token_id, selected_expert, predicate::Equal);
            let target = matches & cmpi(seen, local_index, predicate::Equal);
            token_value = select(target, broadcast_scalar(token, tile_shape), token_value);
            seen = seen + select(matches, one, zero);
        }
        let real_token = cmpi(local_index, selected_count, predicate::LessThan);
        let token_value = select(real_token, token_value, sentinel);
        store_i32_vector(sorted_token_ids, offsets, token_value, sorted_mask);
    }

    #[cutile::entry()]
    pub unsafe fn fused_moe_f32(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        routed_weight: *mut f32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        num_tokens_post_pad: *mut i32,
        element_count: i32,
        columns: i32,
        reduction: i32,
        top_k: i32,
        block_size: i32,
        input_row_stride: i32,
        expert_stride: i32,
        weight_row_stride: i32,
        output_row_stride: i32,
        mul_routed_weight: i32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();

        // Each lane computes one output column for one sorted token/expert assignment.
        // The sorted position determines both the token row and the expert weight matrix.
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let sorted_position = offsets / broadcast_scalar(columns, tile_shape);
        let column = offsets - sorted_position * broadcast_scalar(columns, tile_shape);
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let padded_total = load_i32_vector(num_tokens_post_pad, zero_offsets, valid, 0i32);
        let sorted_valid = valid & cmpi(sorted_position, padded_total, predicate::LessThan);
        let token_id = load_i32_vector(
            sorted_token_ids,
            sorted_position,
            sorted_valid,
            element_count,
        );
        let token_valid = sorted_valid
            & cmpi(
                token_id,
                broadcast_scalar(element_count, tile_shape),
                predicate::LessThan,
            );
        let expert_offsets = sorted_position / broadcast_scalar(block_size, tile_shape);
        let expert = load_i32_vector(expert_ids, expert_offsets, token_valid, 0i32);
        let token_row = token_id / broadcast_scalar(top_k, tile_shape);

        // Dot the routed token row with the selected expert's output column.
        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let input_offsets = token_row * broadcast_scalar(input_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let weight_offsets = expert * broadcast_scalar(expert_stride, tile_shape)
                + column * broadcast_scalar(weight_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let input_values = load_f32_vector(input, input_offsets, token_valid, 0.0f32);
            let weight_values = load_f32_vector(weight, weight_offsets, token_valid, 0.0f32);
            sum = sum + input_values * weight_values;
        }

        if mul_routed_weight != 0 {
            // Optional top-k routing weight scales the expert contribution.
            // The contribution is then scattered back to the unsorted token slot.
            let route = load_f32_vector(routed_weight, token_id, token_valid, 0.0f32);
            sum = sum * route;
        }
        let output_offsets = token_id * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f32_vector(out, output_offsets, sum, token_valid);
    }

    #[cutile::entry()]
    pub unsafe fn fused_moe_f16(
        out: *mut f16,
        input: *mut f16,
        weight: *mut f16,
        routed_weight: *mut f32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        num_tokens_post_pad: *mut i32,
        element_count: i32,
        columns: i32,
        reduction: i32,
        top_k: i32,
        block_size: i32,
        input_row_stride: i32,
        expert_stride: i32,
        weight_row_stride: i32,
        output_row_stride: i32,
        mul_routed_weight: i32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let sorted_position = offsets / broadcast_scalar(columns, tile_shape);
        let column = offsets - sorted_position * broadcast_scalar(columns, tile_shape);
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let padded_total = load_i32_vector(num_tokens_post_pad, zero_offsets, valid, 0i32);
        let sorted_valid = valid & cmpi(sorted_position, padded_total, predicate::LessThan);
        let token_id = load_i32_vector(
            sorted_token_ids,
            sorted_position,
            sorted_valid,
            element_count,
        );
        let token_valid = sorted_valid
            & cmpi(
                token_id,
                broadcast_scalar(element_count, tile_shape),
                predicate::LessThan,
            );
        let expert_offsets = sorted_position / broadcast_scalar(block_size, tile_shape);
        let expert = load_i32_vector(expert_ids, expert_offsets, token_valid, 0i32);
        let token_row = token_id / broadcast_scalar(top_k, tile_shape);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let input_offsets = token_row * broadcast_scalar(input_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let weight_offsets = expert * broadcast_scalar(expert_stride, tile_shape)
                + column * broadcast_scalar(weight_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let input_values = load_f16_vector_as_f32(input, input_offsets, token_valid);
            let weight_values = load_f16_vector_as_f32(weight, weight_offsets, token_valid);
            sum = sum + input_values * weight_values;
        }

        if mul_routed_weight != 0 {
            let route = load_f32_vector(routed_weight, token_id, token_valid, 0.0f32);
            sum = sum * route;
        }
        let output_offsets = token_id * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f16_vector_from_f32(out, output_offsets, sum, token_valid);
    }

    #[cutile::entry()]
    pub unsafe fn fused_moe_bf16(
        out: *mut bf16,
        input: *mut bf16,
        weight: *mut bf16,
        routed_weight: *mut f32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        num_tokens_post_pad: *mut i32,
        element_count: i32,
        columns: i32,
        reduction: i32,
        top_k: i32,
        block_size: i32,
        input_row_stride: i32,
        expert_stride: i32,
        weight_row_stride: i32,
        output_row_stride: i32,
        mul_routed_weight: i32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let sorted_position = offsets / broadcast_scalar(columns, tile_shape);
        let column = offsets - sorted_position * broadcast_scalar(columns, tile_shape);
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let padded_total = load_i32_vector(num_tokens_post_pad, zero_offsets, valid, 0i32);
        let sorted_valid = valid & cmpi(sorted_position, padded_total, predicate::LessThan);
        let token_id = load_i32_vector(
            sorted_token_ids,
            sorted_position,
            sorted_valid,
            element_count,
        );
        let token_valid = sorted_valid
            & cmpi(
                token_id,
                broadcast_scalar(element_count, tile_shape),
                predicate::LessThan,
            );
        let expert_offsets = sorted_position / broadcast_scalar(block_size, tile_shape);
        let expert = load_i32_vector(expert_ids, expert_offsets, token_valid, 0i32);
        let token_row = token_id / broadcast_scalar(top_k, tile_shape);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let input_offsets = token_row * broadcast_scalar(input_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let weight_offsets = expert * broadcast_scalar(expert_stride, tile_shape)
                + column * broadcast_scalar(weight_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let input_values = load_bf16_vector_as_f32(input, input_offsets, token_valid);
            let weight_values = load_bf16_vector_as_f32(weight, weight_offsets, token_valid);
            sum = sum + input_values * weight_values;
        }

        if mul_routed_weight != 0 {
            let route = load_f32_vector(routed_weight, token_id, token_valid, 0.0f32);
            sum = sum * route;
        }
        let output_offsets = token_id * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_bf16_vector_from_f32(out, output_offsets, sum, token_valid);
    }

    #[cutile::entry()]
    pub unsafe fn fused_moe_f8e4m3_block_scaled_f32(
        out: *mut f32,
        input: *mut f8e4m3fn,
        weight: *mut f8e4m3fn,
        input_scales: *mut f32,
        weight_scales: *mut f32,
        routed_weight: *mut f32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        num_tokens_post_pad: *mut i32,
        element_count: i32,
        columns: i32,
        reduction: i32,
        top_k: i32,
        block_size: i32,
        group_n: i32,
        group_k: i32,
        k_groups: i32,
        input_row_stride: i32,
        expert_stride: i32,
        weight_row_stride: i32,
        output_row_stride: i32,
        mul_routed_weight: i32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let sorted_position = offsets / broadcast_scalar(columns, tile_shape);
        let column = offsets - sorted_position * broadcast_scalar(columns, tile_shape);
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let padded_total = load_i32_vector(num_tokens_post_pad, zero_offsets, valid, 0i32);
        let sorted_valid = valid & cmpi(sorted_position, padded_total, predicate::LessThan);
        let token_id = load_i32_vector(
            sorted_token_ids,
            sorted_position,
            sorted_valid,
            element_count,
        );
        let token_valid = sorted_valid
            & cmpi(
                token_id,
                broadcast_scalar(element_count, tile_shape),
                predicate::LessThan,
            );
        let expert_offsets = sorted_position / broadcast_scalar(block_size, tile_shape);
        let expert = load_i32_vector(expert_ids, expert_offsets, token_valid, 0i32);
        let token_row = token_id / broadcast_scalar(top_k, tile_shape);
        let weight_scale_row = column / broadcast_scalar(group_n, tile_shape);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let k_group = broadcast_scalar(reduction_index / group_k, tile_shape);
            let input_offsets = token_row * broadcast_scalar(input_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let weight_offsets = expert * broadcast_scalar(expert_stride, tile_shape)
                + column * broadcast_scalar(weight_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let input_scale_offsets = token_row * broadcast_scalar(k_groups, tile_shape) + k_group;
            let weight_scale_offsets = expert
                * broadcast_scalar((columns + group_n - 1) / group_n * k_groups, tile_shape)
                + weight_scale_row * broadcast_scalar(k_groups, tile_shape)
                + k_group;
            let input_values = load_f8e4m3_vector_as_f32(input, input_offsets, token_valid);
            let weight_values = load_f8e4m3_vector_as_f32(weight, weight_offsets, token_valid);
            let input_scale_values =
                load_f32_vector(input_scales, input_scale_offsets, token_valid, 0.0f32);
            let weight_scale_values =
                load_f32_vector(weight_scales, weight_scale_offsets, token_valid, 0.0f32);
            sum = sum + input_values * weight_values * input_scale_values * weight_scale_values;
        }

        if mul_routed_weight != 0 {
            let route = load_f32_vector(routed_weight, token_id, token_valid, 0.0f32);
            sum = sum * route;
        }
        let output_offsets = token_id * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f32_vector(out, output_offsets, sum, token_valid);
    }

    fn load_i32_vector(
        input: *mut i32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: i32,
    ) -> Tile<i32, { [128] }> {
        let input_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut i32, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut i32, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut i32, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<i32, { [128] }>, Token) = load_ptr_tko(
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

    fn store_i32_vector(
        out: *mut i32,
        offsets: Tile<i32, { [128] }>,
        values: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
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

    fn load_f32_vector(
        input: *mut f32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: f32,
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
            Some(fill),
            None,
            Latency::<0>,
        );
        result.0
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
        let output: Tile<f16, { [128] }> = convert_tile(values);
        store_ptr_tko(
            out_ptrs,
            output,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

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
        let output: Tile<bf16, { [128] }> = convert_tile(values);
        store_ptr_tko(
            out_ptrs,
            output,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    fn load_f8e4m3_vector_as_f32(
        input: *mut f8e4m3fn,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let input_base: PointerTile<*mut f8e4m3fn, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f8e4m3fn, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f8e4m3fn, { [128] }> =
            input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut f8e4m3fn, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<f8e4m3fn, { [128] }>, Token) = load_ptr_tko(
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
}

pub use kernels::*;
