#[cutile::module]
mod kernels {
    use cutile::core::*;

    pub fn activation_unary_store<const OP: i32, const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        let x: Tile<f32, { [B] }> = load_tile_like(input, out);
        let zero = constant(0.0f32, out.shape());
        let one = constant(1.0f32, out.shape());
        let half = constant(0.5f32, out.shape());
        let third = constant(0.33333334f32, out.shape());
        let three = constant(3.0f32, out.shape());
        let six = constant(6.0f32, out.shape());
        let pi = constant(std::f32::consts::PI, out.shape());
        let one_eighty = constant(180.0f32, out.shape());
        let selu_alpha = constant(1.6732632f32, out.shape());
        let selu_scale = constant(1.050701f32, out.shape());
        let value = if OP == 0 {
            zero - x
        } else if OP == 1 {
            absf(x)
        } else if OP == 2 {
            sqrt(x, rounding::NearestEven, ftz::Disabled)
        } else if OP == 3 {
            exp(x)
        } else if OP == 4 {
            log(x)
        } else if OP == 5 {
            sin(x)
        } else if OP == 6 {
            cos(x)
        } else if OP == 7 {
            tan(x)
        } else if OP == 8 {
            sinh(x)
        } else if OP == 9 {
            cosh(x)
        } else if OP == 10 {
            tanh(x)
        } else if OP == 11 {
            atan2(x, sqrt(one - x * x, rounding::NearestEven, ftz::Disabled))
        } else if OP == 12 {
            atan2(sqrt(one - x * x, rounding::NearestEven, ftz::Disabled), x)
        } else if OP == 13 {
            atan2(x, one)
        } else if OP == 14 {
            ceil(x)
        } else if OP == 15 {
            floor(x)
        } else if OP == 16 {
            let positive = cmpf(
                x,
                zero,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            select(positive, floor(x), ceil(x))
        } else if OP == 17 {
            one / x
        } else if OP == 18 {
            x * x
        } else if OP == 19 {
            rsqrt(x, ftz::Disabled)
        } else if OP == 20 {
            log(x + sqrt(x * x + one, rounding::NearestEven, ftz::Disabled))
        } else if OP == 21 {
            log(x + sqrt(x * x - one, rounding::NearestEven, ftz::Disabled))
        } else if OP == 22 {
            half * log((one + x) / (one - x))
        } else if OP == 23 {
            max_tile(x, zero)
        } else if OP == 24 {
            one / (one + exp(zero - x))
        } else if OP == 25 {
            x / (one + exp(zero - x))
        } else if OP == 26 {
            let sqrt_2_over_pi = constant(0.7978846f32, out.shape());
            let cubic_coeff = constant(0.044715f32, out.shape());
            half * x * (one + tanh(sqrt_2_over_pi * (x + cubic_coeff * x * x * x)))
        } else if OP == 27 {
            let negative = cmpf(x, zero, predicate::LessThan, cmp_ordering::Ordered);
            let positive = cmpf(x, zero, predicate::GreaterThan, cmp_ordering::Ordered);
            select(negative, zero - one, select(positive, one, zero))
        } else if OP == 28 {
            log(one + x)
        } else if OP == 29 {
            exp(x) - one
        } else if OP == 30 {
            exp2(x, ftz::Disabled)
        } else if OP == 31 {
            log2(x)
        } else if OP == 32 {
            let ln_10 = constant(std::f32::consts::LN_10, out.shape());
            log(x) / ln_10
        } else if OP == 33 {
            let root = pow(absf(x), third);
            let negative = cmpf(x, zero, predicate::LessThan, cmp_ordering::Ordered);
            select(negative, zero - root, root)
        } else if OP == 34 {
            max_tile(x, zero) + log(one + exp(zero - absf(x)))
        } else if OP == 35 {
            min_tile(max_tile((x + three) / six, zero), one)
        } else if OP == 36 {
            x * min_tile(max_tile((x + three) / six, zero), one)
        } else if OP == 37 {
            x * tanh(max_tile(x, zero) + log(one + exp(zero - absf(x))))
        } else if OP == 38 {
            let positive = cmpf(
                x,
                zero,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            selu_scale * select(positive, x, selu_alpha * (exp(x) - one))
        } else if OP == 39 {
            x / (one + absf(x))
        } else if OP == 40 {
            x * pi / one_eighty
        } else if OP == 41 {
            x * one_eighty / pi
        } else if OP == 42 {
            let positive = cmpf(
                x,
                zero,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            let truncated = select(positive, floor(x), ceil(x));
            x - truncated
        } else if OP == 43 {
            min_tile(max_tile(x, zero), six)
        } else if OP == 44 {
            x - tanh(x)
        } else if OP == 45 {
            zero - (max_tile(zero - x, zero) + log(one + exp(zero - absf(zero - x))))
        } else {
            x
        };
        out.store(value);
    }

    pub fn binary_store<const OP: i32, const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        lhs: &Tensor<f32, { [-1] }>,
        rhs: &Tensor<f32, { [-1] }>,
    ) {
        let lhs_tile: Tile<f32, { [B] }> = load_tile_like(lhs, out);
        let rhs_tile: Tile<f32, { [B] }> = load_tile_like(rhs, out);
        let value = if OP == 0 {
            lhs_tile + rhs_tile
        } else if OP == 1 {
            lhs_tile - rhs_tile
        } else if OP == 2 {
            lhs_tile * rhs_tile
        } else if OP == 3 {
            lhs_tile / rhs_tile
        } else if OP == 4 {
            maxf(lhs_tile, rhs_tile, nan::Enabled, ftz::Disabled)
        } else if OP == 5 {
            minf(lhs_tile, rhs_tile, nan::Enabled, ftz::Disabled)
        } else if OP == 6 {
            atan2(lhs_tile, rhs_tile)
        } else if OP == 7 {
            lhs_tile - floor(lhs_tile / rhs_tile) * rhs_tile
        } else if OP == 8 {
            pow(lhs_tile, rhs_tile)
        } else if OP == 9 {
            sqrt(
                lhs_tile * lhs_tile + rhs_tile * rhs_tile,
                rounding::NearestEven,
                ftz::Disabled,
            )
        } else if OP == 10 {
            let diff = lhs_tile - rhs_tile;
            diff * diff
        } else if OP == 11 {
            let max = maxf(lhs_tile, rhs_tile, nan::Enabled, ftz::Disabled);
            let neg_inf = constant(f32::NEG_INFINITY, out.shape());
            let both_neg_inf = cmpf(lhs_tile, neg_inf, predicate::Equal, cmp_ordering::Ordered)
                & cmpf(rhs_tile, neg_inf, predicate::Equal, cmp_ordering::Ordered);
            let value = max + log(exp(lhs_tile - max) + exp(rhs_tile - max));
            select(both_neg_inf, neg_inf, value)
        } else if OP == 12 {
            let zero = constant(0.0f32, out.shape());
            let zero_lhs = cmpf(lhs_tile, zero, predicate::Equal, cmp_ordering::Ordered);
            select(zero_lhs, zero, lhs_tile * log(rhs_tile))
        } else {
            lhs_tile
        };
        out.store(value);
    }

    pub fn cmp_store<const OP: i32, const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        lhs: &Tensor<f32, { [-1] }>,
        rhs: &Tensor<f32, { [-1] }>,
    ) {
        let lhs_tile: Tile<f32, { [B] }> = load_tile_like(lhs, out);
        let rhs_tile: Tile<f32, { [B] }> = load_tile_like(rhs, out);
        let one = constant(1.0f32, out.shape());
        let zero = constant(0.0f32, out.shape());
        let mask = if OP == 0 {
            cmpf(lhs_tile, rhs_tile, predicate::Equal, cmp_ordering::Ordered)
        } else if OP == 1 {
            cmpf(
                lhs_tile,
                rhs_tile,
                predicate::NotEqual,
                cmp_ordering::Unordered,
            )
        } else if OP == 2 {
            cmpf(
                lhs_tile,
                rhs_tile,
                predicate::LessThan,
                cmp_ordering::Ordered,
            )
        } else if OP == 3 {
            cmpf(
                lhs_tile,
                rhs_tile,
                predicate::LessThanOrEqual,
                cmp_ordering::Ordered,
            )
        } else if OP == 4 {
            cmpf(
                lhs_tile,
                rhs_tile,
                predicate::GreaterThan,
                cmp_ordering::Ordered,
            )
        } else {
            cmpf(
                lhs_tile,
                rhs_tile,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            )
        };
        out.store(select(mask, one, zero));
    }

    pub fn cmp_bool_store<const OP: i32, const B: i32>(
        out: &mut Tensor<u8, { [B] }>,
        lhs: &Tensor<f32, { [-1] }>,
        rhs: &Tensor<f32, { [-1] }>,
    ) {
        let lhs_tile: Tile<f32, { [B] }> = load_tile_like(lhs, out);
        let rhs_tile: Tile<f32, { [B] }> = load_tile_like(rhs, out);
        let one = constant(1u8, out.shape());
        let zero = constant(0u8, out.shape());
        let mask = if OP == 0 {
            cmpf(lhs_tile, rhs_tile, predicate::Equal, cmp_ordering::Ordered)
        } else if OP == 1 {
            cmpf(
                lhs_tile,
                rhs_tile,
                predicate::NotEqual,
                cmp_ordering::Unordered,
            )
        } else if OP == 2 {
            cmpf(
                lhs_tile,
                rhs_tile,
                predicate::LessThan,
                cmp_ordering::Ordered,
            )
        } else if OP == 3 {
            cmpf(
                lhs_tile,
                rhs_tile,
                predicate::LessThanOrEqual,
                cmp_ordering::Ordered,
            )
        } else if OP == 4 {
            cmpf(
                lhs_tile,
                rhs_tile,
                predicate::GreaterThan,
                cmp_ordering::Ordered,
            )
        } else {
            cmpf(
                lhs_tile,
                rhs_tile,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            )
        };
        out.store(select(mask, one, zero));
    }

    #[cutile::entry()]
    pub unsafe fn dequantize_u8_f32(
        out: *mut f32,
        input: *mut u8,
        scale: f32,
        zero_point: f32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let len_tile = broadcast_scalar(len, tile_shape);
        let mask = cmpi(offsets, len_tile, predicate::LessThan);
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let input_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut u8, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut u8, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut u8, { [128] }> = input_ptrs.offset_tile(safe_offsets);
        let input_result: (Tile<u8, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u8),
            None,
            Latency::<0>,
        );
        let input_f32: Tile<f32, { [128] }> = convert_tile(input_result.0);
        let values = (input_f32 - broadcast_scalar(zero_point, tile_shape))
            * broadcast_scalar(scale, tile_shape);

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(tile_shape);
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

    #[cutile::entry()]
    pub unsafe fn dequantize_i8_f32(
        out: *mut f32,
        input: *mut i8,
        scale: f32,
        zero_point: f32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let len_tile = broadcast_scalar(len, tile_shape);
        let mask = cmpi(offsets, len_tile, predicate::LessThan);
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);

        let input_base: PointerTile<*mut i8, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut i8, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut i8, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut i8, { [128] }> = input_ptrs.offset_tile(safe_offsets);
        let input_result: (Tile<i8, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0i8),
            None,
            Latency::<0>,
        );
        let input_f32: Tile<f32, { [128] }> = convert_tile(input_result.0);
        let values = (input_f32 - broadcast_scalar(zero_point, tile_shape))
            * broadcast_scalar(scale, tile_shape);

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(tile_shape);
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

    #[cutile::entry()]
    pub unsafe fn dequantize_u8_f32_grouped(
        out: *mut f32,
        input: *mut u8,
        scales: *mut f32,
        zero_points: *mut f32,
        len: i32,
        group_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let len_tile = broadcast_scalar(len, tile_shape);
        let mask = cmpi(offsets, len_tile, predicate::LessThan);
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let scale_offsets = safe_offsets / broadcast_scalar(group_size, tile_shape);

        let input_values = load_u8_vector(input, safe_offsets, mask);
        let input_f32: Tile<f32, { [128] }> = convert_tile(input_values);
        let scale_values = load_f32_vector(scales, scale_offsets, mask, 0.0f32);
        let zero_point_values = load_f32_vector(zero_points, scale_offsets, mask, 0.0f32);
        let values = (input_f32 - zero_point_values) * scale_values;
        store_f32_vector(out, offsets, values, mask);
    }

    #[cutile::entry()]
    pub unsafe fn dequantize_i8_f32_grouped(
        out: *mut f32,
        input: *mut i8,
        scales: *mut f32,
        zero_points: *mut f32,
        len: i32,
        group_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let len_tile = broadcast_scalar(len, tile_shape);
        let mask = cmpi(offsets, len_tile, predicate::LessThan);
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let scale_offsets = safe_offsets / broadcast_scalar(group_size, tile_shape);

        let input_values = load_i8_vector(input, safe_offsets, mask);
        let input_f32: Tile<f32, { [128] }> = convert_tile(input_values);
        let scale_values = load_f32_vector(scales, scale_offsets, mask, 0.0f32);
        let zero_point_values = load_f32_vector(zero_points, scale_offsets, mask, 0.0f32);
        let values = (input_f32 - zero_point_values) * scale_values;
        store_f32_vector(out, offsets, values, mask);
    }

    #[cutile::entry()]
    pub unsafe fn per_token_group_quant_i8_f32(
        out: *mut i8,
        scales: *mut f32,
        input: *mut f32,
        groups: i32,
        group_size: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let lanes: Tile<i32, { [128] }> = iota(tile_shape);
        let group_mask = cmpi(
            broadcast_scalar(pid.0, tile_shape),
            broadcast_scalar(groups, tile_shape),
            predicate::LessThan,
        );
        let lane_mask = cmpi(
            lanes,
            broadcast_scalar(group_size, tile_shape),
            predicate::LessThan,
        );
        let mask = group_mask & lane_mask;
        let offsets = broadcast_scalar(pid.0 * group_size, tile_shape) + lanes;

        let values = load_f32_vector(input, offsets, mask, 0.0f32);
        let abs_values = absf(values);
        let max_abs: Tile<f32, { [1] }> = reduce_max(abs_values, 0i32);
        let max_abs_tile = max_abs.reshape(const_shape![1]).broadcast(tile_shape);
        let eps_tile = broadcast_scalar(eps, tile_shape);
        let scale = max_tile(max_abs_tile, eps_tile) / constant(127.0f32, tile_shape);
        let low = constant(-128.0f32, tile_shape);
        let high = constant(127.0f32, tile_shape);
        let quantized_f32 =
            round_nearest_even_i8_f32(min_tile(max_tile(values / scale, low), high));
        let quantized: Tile<i8, { [128] }> = convert_tile(quantized_f32);
        store_i8_vector(out, offsets, quantized, mask);

        let lane_zero = constant(0i32, tile_shape);
        store_f32_vector(
            scales,
            broadcast_scalar(pid.0, tile_shape),
            scale,
            group_mask & cmpi(lanes, lane_zero, predicate::Equal),
        );
    }

    #[cutile::entry()]
    pub unsafe fn per_token_group_quant_i8_f32_column_major_scales(
        out: *mut i8,
        scales: *mut f32,
        input: *mut f32,
        rows: i32,
        columns: i32,
        group_size: i32,
        scale_column_stride: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let lanes: Tile<i32, { [128] }> = iota(tile_shape);
        let groups_per_row = columns / group_size;
        let row = pid.0 / groups_per_row;
        let group = pid.0 - row * groups_per_row;
        let group_mask = cmpi(
            broadcast_scalar(row, tile_shape),
            broadcast_scalar(rows, tile_shape),
            predicate::LessThan,
        );
        let lane_mask = cmpi(
            lanes,
            broadcast_scalar(group_size, tile_shape),
            predicate::LessThan,
        );
        let mask = group_mask & lane_mask;
        let offsets = broadcast_scalar(row * columns + group * group_size, tile_shape) + lanes;

        let values = load_f32_vector(input, offsets, mask, 0.0f32);
        let abs_values = absf(values);
        let max_abs: Tile<f32, { [1] }> = reduce_max(abs_values, 0i32);
        let max_abs_tile = max_abs.reshape(const_shape![1]).broadcast(tile_shape);
        let eps_tile = broadcast_scalar(eps, tile_shape);
        let scale = max_tile(max_abs_tile, eps_tile) / constant(127.0f32, tile_shape);
        let low = constant(-128.0f32, tile_shape);
        let high = constant(127.0f32, tile_shape);
        let quantized_f32 =
            round_nearest_even_i8_f32(min_tile(max_tile(values / scale, low), high));
        let quantized: Tile<i8, { [128] }> = convert_tile(quantized_f32);
        store_i8_vector(out, offsets, quantized, mask);

        let lane_zero = constant(0i32, tile_shape);
        let scale_offset = group * scale_column_stride + row;
        store_f32_vector(
            scales,
            broadcast_scalar(scale_offset, tile_shape),
            scale,
            group_mask & cmpi(lanes, lane_zero, predicate::Equal),
        );
    }

    #[cutile::entry()]
    pub unsafe fn per_token_group_quant_i8_f32_column_major_ue8m0_scales(
        out: *mut i8,
        scales: *mut f32,
        input: *mut f32,
        rows: i32,
        columns: i32,
        group_size: i32,
        scale_column_stride: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let lanes: Tile<i32, { [128] }> = iota(tile_shape);
        let groups_per_row = columns / group_size;
        let row = pid.0 / groups_per_row;
        let group = pid.0 - row * groups_per_row;
        let group_mask = cmpi(
            broadcast_scalar(row, tile_shape),
            broadcast_scalar(rows, tile_shape),
            predicate::LessThan,
        );
        let lane_mask = cmpi(
            lanes,
            broadcast_scalar(group_size, tile_shape),
            predicate::LessThan,
        );
        let mask = group_mask & lane_mask;
        let offsets = broadcast_scalar(row * columns + group * group_size, tile_shape) + lanes;

        let values = load_f32_vector(input, offsets, mask, 0.0f32);
        let abs_values = absf(values);
        let max_abs: Tile<f32, { [1] }> = reduce_max(abs_values, 0i32);
        let max_abs_tile = max_abs.reshape(const_shape![1]).broadcast(tile_shape);
        let eps_tile = broadcast_scalar(eps, tile_shape);
        let raw_scale = max_tile(max_abs_tile, eps_tile) / constant(127.0f32, tile_shape);
        let scale = ue8m0_round_scale_f32(raw_scale);
        let low = constant(-128.0f32, tile_shape);
        let high = constant(127.0f32, tile_shape);
        let quantized_f32 =
            round_nearest_even_i8_f32(min_tile(max_tile(values / scale, low), high));
        let quantized: Tile<i8, { [128] }> = convert_tile(quantized_f32);
        store_i8_vector(out, offsets, quantized, mask);

        let lane_zero = constant(0i32, tile_shape);
        let scale_offset = group * scale_column_stride + row;
        store_f32_vector(
            scales,
            broadcast_scalar(scale_offset, tile_shape),
            scale,
            group_mask & cmpi(lanes, lane_zero, predicate::Equal),
        );
    }

    #[cutile::entry()]
    pub unsafe fn per_token_group_quant_f8e4m3_f32(
        out: *mut f8e4m3fn,
        scales: *mut f32,
        input: *mut f32,
        groups: i32,
        group_size: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let lanes: Tile<i32, { [128] }> = iota(tile_shape);
        let group_mask = cmpi(
            broadcast_scalar(pid.0, tile_shape),
            broadcast_scalar(groups, tile_shape),
            predicate::LessThan,
        );
        let lane_mask = cmpi(
            lanes,
            broadcast_scalar(group_size, tile_shape),
            predicate::LessThan,
        );
        let mask = group_mask & lane_mask;
        let offsets = broadcast_scalar(pid.0 * group_size, tile_shape) + lanes;

        let values = load_f32_vector(input, offsets, mask, 0.0f32);
        let abs_values = absf(values);
        let max_abs: Tile<f32, { [1] }> = reduce_max(abs_values, 0i32);
        let max_abs_tile = max_abs.reshape(const_shape![1]).broadcast(tile_shape);
        let eps_tile = broadcast_scalar(eps, tile_shape);
        let scale = max_tile(max_abs_tile, eps_tile) / constant(448.0f32, tile_shape);
        let low = constant(-448.0f32, tile_shape);
        let high = constant(448.0f32, tile_shape);
        let quantized_f32 = min_tile(max_tile(values / scale, low), high);
        let quantized: Tile<f8e4m3fn, { [128] }> = convert_tile(quantized_f32);
        store_f8e4m3_vector(out, offsets, quantized, mask);

        let lane_zero = constant(0i32, tile_shape);
        store_f32_vector(
            scales,
            broadcast_scalar(pid.0, tile_shape),
            scale,
            group_mask & cmpi(lanes, lane_zero, predicate::Equal),
        );
    }

    #[cutile::entry()]
    pub unsafe fn per_token_group_quant_f8e4m3_f32_column_major_scales(
        out: *mut f8e4m3fn,
        scales: *mut f32,
        input: *mut f32,
        rows: i32,
        columns: i32,
        group_size: i32,
        scale_column_stride: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let lanes: Tile<i32, { [128] }> = iota(tile_shape);
        let groups_per_row = columns / group_size;
        let row = pid.0 / groups_per_row;
        let group = pid.0 - row * groups_per_row;
        let group_mask = cmpi(
            broadcast_scalar(row, tile_shape),
            broadcast_scalar(rows, tile_shape),
            predicate::LessThan,
        );
        let lane_mask = cmpi(
            lanes,
            broadcast_scalar(group_size, tile_shape),
            predicate::LessThan,
        );
        let mask = group_mask & lane_mask;
        let offsets = broadcast_scalar(row * columns + group * group_size, tile_shape) + lanes;

        let values = load_f32_vector(input, offsets, mask, 0.0f32);
        let abs_values = absf(values);
        let max_abs: Tile<f32, { [1] }> = reduce_max(abs_values, 0i32);
        let max_abs_tile = max_abs.reshape(const_shape![1]).broadcast(tile_shape);
        let eps_tile = broadcast_scalar(eps, tile_shape);
        let scale = max_tile(max_abs_tile, eps_tile) / constant(448.0f32, tile_shape);
        let low = constant(-448.0f32, tile_shape);
        let high = constant(448.0f32, tile_shape);
        let quantized_f32 = min_tile(max_tile(values / scale, low), high);
        let quantized: Tile<f8e4m3fn, { [128] }> = convert_tile(quantized_f32);
        store_f8e4m3_vector(out, offsets, quantized, mask);

        let lane_zero = constant(0i32, tile_shape);
        let scale_offset = group * scale_column_stride + row;
        store_f32_vector(
            scales,
            broadcast_scalar(scale_offset, tile_shape),
            scale,
            group_mask & cmpi(lanes, lane_zero, predicate::Equal),
        );
    }

    #[cutile::entry()]
    pub unsafe fn per_token_group_quant_f8e4m3_f32_column_major_ue8m0_scales(
        out: *mut f8e4m3fn,
        scales: *mut f32,
        input: *mut f32,
        rows: i32,
        columns: i32,
        group_size: i32,
        scale_column_stride: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let lanes: Tile<i32, { [128] }> = iota(tile_shape);
        let groups_per_row = columns / group_size;
        let row = pid.0 / groups_per_row;
        let group = pid.0 - row * groups_per_row;
        let group_mask = cmpi(
            broadcast_scalar(row, tile_shape),
            broadcast_scalar(rows, tile_shape),
            predicate::LessThan,
        );
        let lane_mask = cmpi(
            lanes,
            broadcast_scalar(group_size, tile_shape),
            predicate::LessThan,
        );
        let mask = group_mask & lane_mask;
        let offsets = broadcast_scalar(row * columns + group * group_size, tile_shape) + lanes;

        let values = load_f32_vector(input, offsets, mask, 0.0f32);
        let abs_values = absf(values);
        let max_abs: Tile<f32, { [1] }> = reduce_max(abs_values, 0i32);
        let max_abs_tile = max_abs.reshape(const_shape![1]).broadcast(tile_shape);
        let eps_tile = broadcast_scalar(eps, tile_shape);
        let raw_scale = max_tile(max_abs_tile, eps_tile) / constant(448.0f32, tile_shape);
        let scale = ue8m0_round_scale_f32(raw_scale);
        let low = constant(-448.0f32, tile_shape);
        let high = constant(448.0f32, tile_shape);
        let quantized_f32 = min_tile(max_tile(values / scale, low), high);
        let quantized: Tile<f8e4m3fn, { [128] }> = convert_tile(quantized_f32);
        store_f8e4m3_vector(out, offsets, quantized, mask);

        let lane_zero = constant(0i32, tile_shape);
        let scale_offset = group * scale_column_stride + row;
        store_f32_vector(
            scales,
            broadcast_scalar(scale_offset, tile_shape),
            scale,
            group_mask & cmpi(lanes, lane_zero, predicate::Equal),
        );
    }

    #[cutile::entry()]
    pub unsafe fn rope_quantize_f8e4m3_f32_interleaved(
        out: *mut f8e4m3fn,
        input: *mut f32,
        cos_sin_cache: *mut f32,
        pos_ids: *mut u32,
        heads: i32,
        rope_dim: i32,
        cos_sin_cache_stride: i32,
        quant_scale: f32,
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
        let head_dim = heads * rope_dim;
        let token = safe_offsets / broadcast_scalar(head_dim, tile_shape);
        let token_remaining = safe_offsets - token * broadcast_scalar(head_dim, tile_shape);
        let head = token_remaining / broadcast_scalar(rope_dim, tile_shape);
        let dim = token_remaining - head * broadcast_scalar(rope_dim, tile_shape);
        let pair = dim / broadcast_scalar(2i32, tile_shape);
        let even_dim = pair * broadcast_scalar(2i32, tile_shape);
        let odd_dim = even_dim + broadcast_scalar(1i32, tile_shape);
        let is_even = cmpi(dim, even_dim, predicate::Equal);

        let base = token * broadcast_scalar(head_dim, tile_shape)
            + head * broadcast_scalar(rope_dim, tile_shape);
        let even_offsets = base + even_dim;
        let odd_offsets = base + odd_dim;
        let even = load_f32_vector(input, even_offsets, mask, 0.0f32);
        let odd = load_f32_vector(input, odd_offsets, mask, 0.0f32);

        let pos_u32 = load_u32_vector(pos_ids, token, mask);
        let pos: Tile<i32, { [128] }> = bitcast(pos_u32);
        let cos_offsets = pos * broadcast_scalar(cos_sin_cache_stride, tile_shape) + pair;
        let sin_offsets = cos_offsets + broadcast_scalar(rope_dim / 2i32, tile_shape);
        let cos = load_f32_vector(cos_sin_cache, cos_offsets, mask, 1.0f32);
        let sin = load_f32_vector(cos_sin_cache, sin_offsets, mask, 0.0f32);

        let rotated_even = even * cos - odd * sin;
        let rotated_odd = odd * cos + even * sin;
        let rotated = select(is_even, rotated_even, rotated_odd);
        let scaled = rotated * broadcast_scalar(quant_scale, tile_shape);
        let low = constant(-448.0f32, tile_shape);
        let high = constant(448.0f32, tile_shape);
        let quantized_f32 = min_tile(max_tile(scaled, low), high);
        let quantized: Tile<f8e4m3fn, { [128] }> = convert_tile(quantized_f32);
        store_f8e4m3_vector(out, offsets, quantized, mask);
    }

    #[cutile::entry()]
    pub unsafe fn qk_rope_quantize_f8e4m3_f32_interleaved_single_kernel(
        q_rope_out: *mut f8e4m3fn,
        k_rope_out: *mut f8e4m3fn,
        q_nope_out: *mut f8e4m3fn,
        k_nope_out: *mut f8e4m3fn,
        q_rope: *mut f32,
        k_rope: *mut f32,
        q_nope: *mut f32,
        k_nope: *mut f32,
        cos_sin_cache: *mut f32,
        pos_ids: *mut u32,
        q_heads: i32,
        kv_heads: i32,
        rope_dim: i32,
        no_rope_dim: i32,
        cos_sin_cache_stride: i32,
        quant_scale_q: f32,
        quant_scale_kv: f32,
        q_rope_len: i32,
        k_rope_len: i32,
        q_nope_len: i32,
        total_len: i32,
    ) {
        let _ = no_rope_dim;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(total_len, tile_shape),
            predicate::LessThan,
        );
        let q_rope_region = cmpi(
            offsets,
            broadcast_scalar(q_rope_len, tile_shape),
            predicate::LessThan,
        );
        let k_rope_start = q_rope_len;
        let q_nope_start = q_rope_len + k_rope_len;
        let k_nope_start = q_nope_start + q_nope_len;
        let k_rope_region =
            mask & cmpi(
                offsets,
                broadcast_scalar(k_rope_start, tile_shape),
                predicate::GreaterThanOrEqual,
            ) & cmpi(
                offsets,
                broadcast_scalar(q_nope_start, tile_shape),
                predicate::LessThan,
            );
        let q_nope_region =
            mask & cmpi(
                offsets,
                broadcast_scalar(q_nope_start, tile_shape),
                predicate::GreaterThanOrEqual,
            ) & cmpi(
                offsets,
                broadcast_scalar(k_nope_start, tile_shape),
                predicate::LessThan,
            );
        let k_nope_region = mask
            & cmpi(
                offsets,
                broadcast_scalar(k_nope_start, tile_shape),
                predicate::GreaterThanOrEqual,
            );

        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let q_rope_local = select(q_rope_region, offsets, zero_offsets);
        let k_rope_local = select(
            k_rope_region,
            offsets - broadcast_scalar(k_rope_start, tile_shape),
            zero_offsets,
        );
        let q_nope_local = select(
            q_nope_region,
            offsets - broadcast_scalar(q_nope_start, tile_shape),
            zero_offsets,
        );
        let k_nope_local = select(
            k_nope_region,
            offsets - broadcast_scalar(k_nope_start, tile_shape),
            zero_offsets,
        );

        let q_rope_values = qk_rope_quantize_value_f32(
            q_rope,
            cos_sin_cache,
            pos_ids,
            q_rope_local,
            q_heads,
            rope_dim,
            cos_sin_cache_stride,
            quant_scale_q,
            q_rope_region & mask,
        );
        store_f8e4m3_vector(
            q_rope_out,
            q_rope_local,
            q_rope_values,
            q_rope_region & mask,
        );

        let k_rope_values = qk_rope_quantize_value_f32(
            k_rope,
            cos_sin_cache,
            pos_ids,
            k_rope_local,
            kv_heads,
            rope_dim,
            cos_sin_cache_stride,
            quant_scale_kv,
            k_rope_region,
        );
        store_f8e4m3_vector(k_rope_out, k_rope_local, k_rope_values, k_rope_region);

        let q_nope_values =
            scaled_quantize_value_f32(q_nope, q_nope_local, quant_scale_q, q_nope_region);
        store_f8e4m3_vector(q_nope_out, q_nope_local, q_nope_values, q_nope_region);

        let k_nope_values =
            scaled_quantize_value_f32(k_nope, k_nope_local, quant_scale_kv, k_nope_region);
        store_f8e4m3_vector(k_nope_out, k_nope_local, k_nope_values, k_nope_region);
    }

    #[cutile::entry()]
    pub unsafe fn qk_rope_quantize_f8e4m3_f32_interleaved_rope_only(
        q_rope_out: *mut f8e4m3fn,
        k_rope_out: *mut f8e4m3fn,
        q_rope: *mut f32,
        k_rope: *mut f32,
        cos_sin_cache: *mut f32,
        pos_ids: *mut u32,
        q_heads: i32,
        kv_heads: i32,
        rope_dim: i32,
        cos_sin_cache_stride: i32,
        quant_scale_q: f32,
        quant_scale_kv: f32,
        q_rope_len: i32,
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
        let q_rope_region = cmpi(
            offsets,
            broadcast_scalar(q_rope_len, tile_shape),
            predicate::LessThan,
        );
        let k_rope_region = mask
            & cmpi(
                offsets,
                broadcast_scalar(q_rope_len, tile_shape),
                predicate::GreaterThanOrEqual,
            );

        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let q_rope_local = select(q_rope_region, offsets, zero_offsets);
        let k_rope_local = select(
            k_rope_region,
            offsets - broadcast_scalar(q_rope_len, tile_shape),
            zero_offsets,
        );

        let q_rope_values = qk_rope_quantize_value_f32(
            q_rope,
            cos_sin_cache,
            pos_ids,
            q_rope_local,
            q_heads,
            rope_dim,
            cos_sin_cache_stride,
            quant_scale_q,
            q_rope_region & mask,
        );
        store_f8e4m3_vector(
            q_rope_out,
            q_rope_local,
            q_rope_values,
            q_rope_region & mask,
        );

        let k_rope_values = qk_rope_quantize_value_f32(
            k_rope,
            cos_sin_cache,
            pos_ids,
            k_rope_local,
            kv_heads,
            rope_dim,
            cos_sin_cache_stride,
            quant_scale_kv,
            k_rope_region,
        );
        store_f8e4m3_vector(k_rope_out, k_rope_local, k_rope_values, k_rope_region);
    }

    pub fn qk_rope_quantize_value_f32(
        input: *mut f32,
        cos_sin_cache: *mut f32,
        pos_ids: *mut u32,
        local_offsets: Tile<i32, { [128] }>,
        heads: i32,
        rope_dim: i32,
        cos_sin_cache_stride: i32,
        quant_scale: f32,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<f8e4m3fn, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, local_offsets, zero_offsets);
        let head_dim = heads * rope_dim;
        let token = safe_offsets / broadcast_scalar(head_dim, tile_shape);
        let token_remaining = safe_offsets - token * broadcast_scalar(head_dim, tile_shape);
        let head = token_remaining / broadcast_scalar(rope_dim, tile_shape);
        let dim = token_remaining - head * broadcast_scalar(rope_dim, tile_shape);
        let pair = dim / broadcast_scalar(2i32, tile_shape);
        let even_dim = pair * broadcast_scalar(2i32, tile_shape);
        let odd_dim = even_dim + broadcast_scalar(1i32, tile_shape);
        let is_even = cmpi(dim, even_dim, predicate::Equal);

        let base = token * broadcast_scalar(head_dim, tile_shape)
            + head * broadcast_scalar(rope_dim, tile_shape);
        let even_offsets = base + even_dim;
        let odd_offsets = base + odd_dim;
        let even = load_f32_vector(input, even_offsets, mask, 0.0f32);
        let odd = load_f32_vector(input, odd_offsets, mask, 0.0f32);

        let pos_u32 = load_u32_vector(pos_ids, token, mask);
        let pos: Tile<i32, { [128] }> = bitcast(pos_u32);
        let cos_offsets = pos * broadcast_scalar(cos_sin_cache_stride, tile_shape) + pair;
        let sin_offsets = cos_offsets + broadcast_scalar(rope_dim / 2i32, tile_shape);
        let cos = load_f32_vector(cos_sin_cache, cos_offsets, mask, 1.0f32);
        let sin = load_f32_vector(cos_sin_cache, sin_offsets, mask, 0.0f32);

        let rotated_even = even * cos - odd * sin;
        let rotated_odd = odd * cos + even * sin;
        let rotated = select(is_even, rotated_even, rotated_odd);
        let scaled = rotated * broadcast_scalar(quant_scale, tile_shape);
        let low = constant(-448.0f32, tile_shape);
        let high = constant(448.0f32, tile_shape);
        let quantized_f32 = min_tile(max_tile(scaled, low), high);
        convert_tile(quantized_f32)
    }

    pub fn scaled_quantize_value_f32(
        input: *mut f32,
        local_offsets: Tile<i32, { [128] }>,
        quant_scale: f32,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<f8e4m3fn, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, local_offsets, zero_offsets);
        let values = load_f32_vector(input, safe_offsets, mask, 0.0f32)
            * broadcast_scalar(quant_scale, tile_shape);
        let low = constant(-448.0f32, tile_shape);
        let high = constant(448.0f32, tile_shape);
        let quantized_f32 = min_tile(max_tile(values, low), high);
        convert_tile(quantized_f32)
    }

    #[cutile::entry()]
    pub unsafe fn quantize_f32_f8e4m3_scaled(
        out: *mut f8e4m3fn,
        input: *mut f32,
        quant_scale: f32,
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
        let values = load_f32_vector(input, safe_offsets, mask, 0.0f32)
            * broadcast_scalar(quant_scale, tile_shape);
        let low = constant(-448.0f32, tile_shape);
        let high = constant(448.0f32, tile_shape);
        let quantized_f32 = min_tile(max_tile(values, low), high);
        let quantized: Tile<f8e4m3fn, { [128] }> = convert_tile(quantized_f32);
        store_f8e4m3_vector(out, offsets, quantized, mask);
    }

    #[cutile::entry()]
    pub unsafe fn matmul_i8_i8_f32(
        out: *mut f32,
        activations: *mut i8,
        weights: *mut i8,
        _rows: i32,
        columns: i32,
        reduction: i32,
        activation_row_stride: i32,
        weight_row_stride: i32,
        output_row_stride: i32,
        activation_zero_point: f32,
        weight_zero_point: f32,
        output_scale: f32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let row = safe_offsets / broadcast_scalar(columns, tile_shape);
        let column = safe_offsets - row * broadcast_scalar(columns, tile_shape);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let activation_zero_point = broadcast_scalar(activation_zero_point, tile_shape);
        let weight_zero_point = broadcast_scalar(weight_zero_point, tile_shape);
        for reduction_index in 0i32..reduction {
            let activation_offsets = row * broadcast_scalar(activation_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let weight_offsets = column * broadcast_scalar(weight_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let activation_values: Tile<f32, { [128] }> =
                convert_tile(load_i8_vector(activations, activation_offsets, mask));
            let weight_values: Tile<f32, { [128] }> =
                convert_tile(load_i8_vector(weights, weight_offsets, mask));
            sum = sum
                + (activation_values - activation_zero_point) * (weight_values - weight_zero_point);
        }

        let output_values = sum * broadcast_scalar(output_scale, tile_shape);
        let output_offsets = row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f32_vector(out, output_offsets, output_values, mask);
    }

    #[cutile::entry()]
    pub unsafe fn matmul_f32_i8_dequantize_f32(
        out: *mut f32,
        activations: *mut f32,
        weights: *mut i8,
        scales: *mut f32,
        zero_points: *mut f32,
        _rows: i32,
        columns: i32,
        reduction: i32,
        activation_row_stride: i32,
        weight_row_stride: i32,
        output_row_stride: i32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let row = safe_offsets / broadcast_scalar(columns, tile_shape);
        let column = safe_offsets - row * broadcast_scalar(columns, tile_shape);
        let weight_scales = load_f32_vector(scales, column, mask, 0.0f32);
        let weight_zero_points = load_f32_vector(zero_points, column, mask, 0.0f32);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let activation_offsets = row * broadcast_scalar(activation_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let weight_offsets = column * broadcast_scalar(weight_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let activation_values = load_f32_vector(activations, activation_offsets, mask, 0.0f32);
            let weight_values: Tile<f32, { [128] }> =
                convert_tile(load_i8_vector(weights, weight_offsets, mask));
            let dequantized_weights = (weight_values - weight_zero_points) * weight_scales;
            sum = sum + activation_values * dequantized_weights;
        }

        let output_offsets = row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f32_vector(out, output_offsets, sum, mask);
    }

    #[cutile::entry()]
    pub unsafe fn matmul_f8e4m3_f8e4m3_block_scaled_f32(
        out: *mut f32,
        activations: *mut f8e4m3fn,
        weights: *mut f8e4m3fn,
        activation_scales: *mut f32,
        weight_scales: *mut f32,
        _rows: i32,
        columns: i32,
        reduction: i32,
        group_n: i32,
        group_k: i32,
        k_groups: i32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let row = safe_offsets / broadcast_scalar(columns, tile_shape);
        let column = safe_offsets - row * broadcast_scalar(columns, tile_shape);
        let weight_scale_row = column / broadcast_scalar(group_n, tile_shape);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let k_group = broadcast_scalar(reduction_index / group_k, tile_shape);
            let activation_offsets = row * broadcast_scalar(reduction, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let weight_offsets = column * broadcast_scalar(reduction, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let activation_scale_offsets = row * broadcast_scalar(k_groups, tile_shape) + k_group;
            let weight_scale_offsets =
                weight_scale_row * broadcast_scalar(k_groups, tile_shape) + k_group;
            let activation_values =
                load_f8e4m3_vector_as_f32(activations, activation_offsets, mask);
            let weight_values = load_f8e4m3_vector_as_f32(weights, weight_offsets, mask);
            let activation_scale_values =
                load_f32_vector(activation_scales, activation_scale_offsets, mask, 0.0f32);
            let weight_scale_values =
                load_f32_vector(weight_scales, weight_scale_offsets, mask, 0.0f32);
            sum = sum
                + activation_values * weight_values * activation_scale_values * weight_scale_values;
        }

        store_f32_vector(out, offsets, sum, mask);
    }

    #[cutile::entry()]
    pub unsafe fn matmul_f8e4m3_f8e4m3_block_scaled_f16(
        out: *mut f16,
        activations: *mut f8e4m3fn,
        weights: *mut f8e4m3fn,
        activation_scales: *mut f32,
        weight_scales: *mut f32,
        _rows: i32,
        columns: i32,
        reduction: i32,
        group_n: i32,
        group_k: i32,
        k_groups: i32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let row = safe_offsets / broadcast_scalar(columns, tile_shape);
        let column = safe_offsets - row * broadcast_scalar(columns, tile_shape);
        let weight_scale_row = column / broadcast_scalar(group_n, tile_shape);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let k_group = broadcast_scalar(reduction_index / group_k, tile_shape);
            let activation_offsets = row * broadcast_scalar(reduction, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let weight_offsets = column * broadcast_scalar(reduction, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let activation_scale_offsets = row * broadcast_scalar(k_groups, tile_shape) + k_group;
            let weight_scale_offsets =
                weight_scale_row * broadcast_scalar(k_groups, tile_shape) + k_group;
            let activation_values =
                load_f8e4m3_vector_as_f32(activations, activation_offsets, mask);
            let weight_values = load_f8e4m3_vector_as_f32(weights, weight_offsets, mask);
            let activation_scale_values =
                load_f32_vector(activation_scales, activation_scale_offsets, mask, 0.0f32);
            let weight_scale_values =
                load_f32_vector(weight_scales, weight_scale_offsets, mask, 0.0f32);
            sum = sum
                + activation_values * weight_values * activation_scale_values * weight_scale_values;
        }

        store_f16_vector_from_f32(out, offsets, sum, mask);
    }

    #[cutile::entry()]
    pub unsafe fn matmul_f8e4m3_f8e4m3_block_scaled_bf16(
        out: *mut bf16,
        activations: *mut f8e4m3fn,
        weights: *mut f8e4m3fn,
        activation_scales: *mut f32,
        weight_scales: *mut f32,
        _rows: i32,
        columns: i32,
        reduction: i32,
        group_n: i32,
        group_k: i32,
        k_groups: i32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let row = safe_offsets / broadcast_scalar(columns, tile_shape);
        let column = safe_offsets - row * broadcast_scalar(columns, tile_shape);
        let weight_scale_row = column / broadcast_scalar(group_n, tile_shape);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let k_group = broadcast_scalar(reduction_index / group_k, tile_shape);
            let activation_offsets = row * broadcast_scalar(reduction, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let weight_offsets = column * broadcast_scalar(reduction, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let activation_scale_offsets = row * broadcast_scalar(k_groups, tile_shape) + k_group;
            let weight_scale_offsets =
                weight_scale_row * broadcast_scalar(k_groups, tile_shape) + k_group;
            let activation_values =
                load_f8e4m3_vector_as_f32(activations, activation_offsets, mask);
            let weight_values = load_f8e4m3_vector_as_f32(weights, weight_offsets, mask);
            let activation_scale_values =
                load_f32_vector(activation_scales, activation_scale_offsets, mask, 0.0f32);
            let weight_scale_values =
                load_f32_vector(weight_scales, weight_scale_offsets, mask, 0.0f32);
            sum = sum
                + activation_values * weight_values * activation_scale_values * weight_scale_values;
        }

        store_bf16_vector_from_f32(out, offsets, sum, mask);
    }

    #[cutile::entry()]
    pub unsafe fn quantize_f32_f8e4m3_block(
        out: *mut f8e4m3fn,
        scales: *mut f32,
        input: *mut f32,
        len: i32,
        block_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let lanes: Tile<i32, { [128] }> = iota(tile_shape);
        let offsets = lanes + broadcast_scalar(pid.0 * block_size, tile_shape);
        let mask = cmpi(
            lanes,
            broadcast_scalar(block_size, tile_shape),
            predicate::LessThan,
        ) & cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let values = load_f32_vector(input, offsets, mask, 0.0f32);
        let abs_values = absf(values);
        let max_abs: Tile<f32, { [1] }> = reduce_max(abs_values, 0i32);
        let max_abs_tile = max_abs.reshape(const_shape![1i32]).broadcast(tile_shape);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let fp8_max: Tile<f32, { [128] }> = constant(448.0f32, tile_shape);
        let scale_tile = select(
            cmpf(max_abs_tile, zero, predicate::Equal, cmp_ordering::Ordered),
            one,
            max_abs_tile / fp8_max,
        );
        let low: Tile<f32, { [128] }> = constant(-448.0f32, tile_shape);
        let high: Tile<f32, { [128] }> = constant(448.0f32, tile_shape);
        let quantized_f32 = min_tile(max_tile(values / scale_tile, low), high);
        let quantized: Tile<f8e4m3fn, { [128] }> = convert_tile(quantized_f32);
        let lane_zero: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        store_f8e4m3_vector(out, offsets, quantized, mask);
        store_f32_vector(
            scales,
            broadcast_scalar(pid.0, tile_shape),
            scale_tile,
            cmpi(lanes, lane_zero, predicate::Equal),
        );
    }

    #[cutile::entry()]
    pub unsafe fn dequantize_f8e4m3_f32_block(
        out: *mut f32,
        input: *mut f8e4m3fn,
        scales: *mut f32,
        columns: i32,
        block_size: i32,
        scale_columns: i32,
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
        let row = safe_offsets / broadcast_scalar(columns, tile_shape);
        let column = safe_offsets - row * broadcast_scalar(columns, tile_shape);
        let scale_row = row / broadcast_scalar(block_size, tile_shape);
        let scale_column = column / broadcast_scalar(block_size, tile_shape);
        let scale_offsets = scale_row * broadcast_scalar(scale_columns, tile_shape) + scale_column;

        let input_values = load_f8e4m3_vector_as_f32(input, safe_offsets, mask);
        let scale_values = load_f32_vector(scales, scale_offsets, mask, 0.0f32);
        store_f32_vector(out, offsets, input_values * scale_values, mask);
    }

    pub fn load_f32_vector(
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

    pub fn load_u32_vector(
        input: *mut u32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<u32, { [128] }> {
        let input_base: PointerTile<*mut u32, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut u32, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut u32, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut u32, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<u32, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u32),
            None,
            Latency::<0>,
        );
        result.0
    }

    pub fn ue8m0_round_scale_f32(scale: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        let min_scale: Tile<f32, { [128] }> = constant(1e-10f32, const_shape![128]);
        let safe_scale = max_tile(absf(scale), min_scale);
        exp2(ceil(log2(safe_scale)), ftz::Disabled)
    }

    pub fn load_u8_vector(
        input: *mut u8,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<u8, { [128] }> {
        let input_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut u8, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut u8, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut u8, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<u8, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u8),
            None,
            Latency::<0>,
        );
        result.0
    }

    pub fn load_f8e4m3_vector_as_f32(
        input: *mut f8e4m3fn,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
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

    pub fn load_i8_vector(
        input: *mut i8,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<i8, { [128] }> {
        let input_base: PointerTile<*mut i8, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut i8, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut i8, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut i8, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<i8, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0i8),
            None,
            Latency::<0>,
        );
        result.0
    }

    pub fn store_f32_vector(
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

    pub fn store_f16_vector_from_f32(
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

    pub fn store_bf16_vector_from_f32(
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

    pub fn store_f8e4m3_vector(
        out: *mut f8e4m3fn,
        offsets: Tile<i32, { [128] }>,
        values: Tile<f8e4m3fn, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let out_base: PointerTile<*mut f8e4m3fn, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f8e4m3fn, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f8e4m3fn, { [128] }> = out_base.broadcast(const_shape![128]);
        let out_ptrs: PointerTile<*mut f8e4m3fn, { [128] }> = out_ptrs.offset_tile(offsets);
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

    pub fn store_i8_vector(
        out: *mut i8,
        offsets: Tile<i32, { [128] }>,
        values: Tile<i8, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let out_base: PointerTile<*mut i8, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut i8, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut i8, { [128] }> = out_base.broadcast(const_shape![128]);
        let out_ptrs: PointerTile<*mut i8, { [128] }> = out_ptrs.offset_tile(offsets);
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

    pub fn round_nearest_even_i8_f32(values: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let floored = floor(values);
        let fraction = values - floored;
        let half = constant(0.5f32, tile_shape);
        let one = constant(1.0f32, tile_shape);
        let two = constant(2.0f32, tile_shape);
        let floor_half = floor(floored / two);
        let floor_is_odd = cmpf(
            floor_half * two,
            floored,
            predicate::NotEqual,
            cmp_ordering::Ordered,
        );
        let above_half = cmpf(
            fraction,
            half,
            predicate::GreaterThan,
            cmp_ordering::Ordered,
        );
        let tie = cmpf(fraction, half, predicate::Equal, cmp_ordering::Ordered);
        select(above_half | (tie & floor_is_odd), floored + one, floored)
    }

    pub fn reduce_impl<const OP: i32, const BN: i32>(
        out: &mut Tensor<f32, { [1, 1] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        divisor: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1i32, BN];
        let input_tile_raw: Tile<f32, { [1, BN] }> =
            input.partition(tile_shape).load([pid.0, 0i32]);
        let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let col_iota = col_iota.reshape(tile_shape);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let cols_tile = broadcast_scalar(input_shape[1], tile_shape);
        let valid_cols = cmpi(col_iota, cols_tile, predicate::LessThan);
        let value: Tile<f32, { [1] }> = if OP == 0 || OP == 1 {
            let zero = constant(0.0f32, tile_shape);
            let input_tile = select(valid_cols, input_tile_raw, zero);
            let sum: Tile<f32, { [1] }> = reduce_sum(input_tile, 1i32);
            if OP == 1 {
                let divisor = broadcast_scalar(divisor, const_shape![1]);
                sum / divisor
            } else {
                sum
            }
        } else if OP == 5 || OP == 6 {
            let zero = constant(0.0f32, tile_shape);
            let input_tile = select(valid_cols, input_tile_raw, zero);
            let divisor = broadcast_scalar(divisor, const_shape![1]);
            let sum: Tile<f32, { [1] }> = reduce_sum(input_tile, 1i32);
            let mean: Tile<f32, { [1] }> = sum / divisor;
            let mean_broadcast = mean.reshape(const_shape![1i32, 1i32]).broadcast(tile_shape);
            let diff = select(valid_cols, input_tile_raw - mean_broadcast, zero);
            let variance_sum: Tile<f32, { [1] }> = reduce_sum(diff * diff, 1i32);
            let variance: Tile<f32, { [1] }> = variance_sum / divisor;
            if OP == 6 {
                sqrt(variance, rounding::NearestEven, ftz::Disabled)
            } else {
                variance
            }
        } else if OP == 2 {
            let fill = constant(f32::NEG_INFINITY, tile_shape);
            reduce_max(select(valid_cols, input_tile_raw, fill), 1i32)
        } else if OP == 3 {
            let fill = constant(f32::INFINITY, tile_shape);
            reduce_min(select(valid_cols, input_tile_raw, fill), 1i32)
        } else {
            let one = constant(1.0f32, tile_shape);
            reduce_prod(select(valid_cols, input_tile_raw, one), 1i32)
        };
        out.store(value.reshape(const_shape![1i32, 1i32]));
    }

    pub fn arg_reduce_impl<const OP: i32, const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1i32, BN];
        let input_tile_raw: Tile<f32, { [1, BN] }> =
            input.partition(tile_shape).load([pid.0, 0i32]);
        let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let col_iota = col_iota.reshape(tile_shape);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let cols_tile = broadcast_scalar(input_shape[1], tile_shape);
        let valid_cols = cmpi(col_iota, cols_tile, predicate::LessThan);
        let numeric = cmpf(
            input_tile_raw,
            input_tile_raw,
            predicate::Equal,
            cmp_ordering::Ordered,
        );
        let one_i32: Tile<i32, { [1, BN] }> = constant(1i32, tile_shape);
        let zero_i32: Tile<i32, { [1, BN] }> = constant(0i32, tile_shape);
        let numeric_i32 = select(valid_cols & numeric, one_i32, zero_i32);
        let numeric_count: Tile<i32, { [1] }> = reduce_sum(numeric_i32, 1i32);
        let zero_row: Tile<i32, { [1, 1] }> = constant(0i32, const_shape![1i32, 1i32]);
        let has_numeric = cmpi(
            numeric_count.reshape(const_shape![1i32, 1i32]),
            zero_row,
            predicate::GreaterThan,
        )
        .broadcast(tile_shape);
        let no_numeric = cmpi(
            numeric_count.reshape(const_shape![1i32, 1i32]),
            zero_row,
            predicate::Equal,
        )
        .broadcast(tile_shape);
        let candidates = if OP == 0 {
            let fill: Tile<f32, { [1, BN] }> = constant(f32::NEG_INFINITY, tile_shape);
            select(valid_cols & numeric, input_tile_raw, fill)
        } else {
            let fill: Tile<f32, { [1, BN] }> = constant(f32::INFINITY, tile_shape);
            select(valid_cols & numeric, input_tile_raw, fill)
        };
        let best_value: Tile<f32, { [1] }> = if OP == 0 {
            reduce_max(candidates, 1i32)
        } else {
            reduce_min(candidates, 1i32)
        };
        let best_value = best_value
            .reshape(const_shape![1i32, 1i32])
            .broadcast(tile_shape);
        let best_numeric = cmpf(
            input_tile_raw,
            best_value,
            predicate::Equal,
            cmp_ordering::Ordered,
        );
        let best_nan = no_numeric & valid_cols;
        let best = (has_numeric & valid_cols & numeric & best_numeric) | best_nan;
        let sentinel: Tile<i32, { [1, BN] }> = broadcast_scalar(input_shape[1], tile_shape);
        let best_indices = select(best, col_iota, sentinel);
        let best_index: Tile<i32, { [1] }> = reduce_min(best_indices, 1i32);
        let lane_iota: Tile<i32, { [2] }> = iota(const_shape![2]);
        let lane_iota = lane_iota.reshape(const_shape![1i32, 2i32]);
        let zero_pair: Tile<i32, { [1, 2] }> = constant(0i32, const_shape![1i32, 2i32]);
        let first_lane = cmpi(lane_iota, zero_pair, predicate::Equal);
        let best_index = best_index
            .reshape(const_shape![1i32, 1i32])
            .broadcast(const_shape![1i32, 2i32]);
        out.store(select(first_lane, best_index, zero_pair));
    }

    pub fn scalar_store<const OP: i32, const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        input: &Tensor<f32, { [-1] }>,
        scalar: f32,
    ) {
        let x: Tile<f32, { [B] }> = load_tile_like(input, out);
        let scalar = broadcast_scalar(scalar, out.shape());
        let value = if OP == 0 {
            x + scalar
        } else if OP == 1 {
            x - scalar
        } else if OP == 2 {
            scalar - x
        } else if OP == 3 {
            x * scalar
        } else if OP == 4 {
            x / scalar
        } else if OP == 5 {
            scalar / x
        } else if OP == 10 {
            x - floor(x / scalar) * scalar
        } else if OP == 11 {
            scalar - floor(scalar / x) * x
        } else if OP == 12 {
            atan2(x, scalar)
        } else if OP == 13 {
            atan2(scalar, x)
        } else if OP == 6 {
            pow(x, scalar)
        } else if OP == 7 {
            pow(scalar, x)
        } else if OP == 8 {
            min_tile(x, scalar)
        } else if OP == 9 {
            max_tile(x, scalar)
        } else if OP == 14 {
            sqrt(
                x * x + scalar * scalar,
                rounding::NearestEven,
                ftz::Disabled,
            )
        } else if OP == 15 {
            let diff = x - scalar;
            diff * diff
        } else if OP == 16 {
            let max = maxf(x, scalar, nan::Enabled, ftz::Disabled);
            let neg_inf = constant(f32::NEG_INFINITY, out.shape());
            let both_neg_inf = cmpf(x, neg_inf, predicate::Equal, cmp_ordering::Ordered)
                & cmpf(scalar, neg_inf, predicate::Equal, cmp_ordering::Ordered);
            let value = max + log(exp(x - max) + exp(scalar - max));
            select(both_neg_inf, neg_inf, value)
        } else if OP == 17 {
            let zero = constant(0.0f32, out.shape());
            let zero_x = cmpf(x, zero, predicate::Equal, cmp_ordering::Ordered);
            select(zero_x, zero, x * log(scalar))
        } else if OP == 18 {
            let zero = constant(0.0f32, out.shape());
            let zero_scalar = cmpf(scalar, zero, predicate::Equal, cmp_ordering::Ordered);
            select(zero_scalar, zero, scalar * log(x))
        } else {
            x
        };
        out.store(value);
    }

    pub fn scalar_cmp_bool_store<const OP: i32, const B: i32>(
        out: &mut Tensor<u8, { [B] }>,
        input: &Tensor<f32, { [-1] }>,
        scalar: f32,
    ) {
        let x: Tile<f32, { [B] }> = load_tile_like(input, out);
        let scalar = broadcast_scalar(scalar, out.shape());
        let one = constant(1u8, out.shape());
        let zero = constant(0u8, out.shape());
        let mask = if OP == 0 {
            cmpf(x, scalar, predicate::Equal, cmp_ordering::Ordered)
        } else if OP == 1 {
            cmpf(x, scalar, predicate::NotEqual, cmp_ordering::Unordered)
        } else if OP == 2 {
            cmpf(x, scalar, predicate::LessThan, cmp_ordering::Ordered)
        } else if OP == 3 {
            cmpf(x, scalar, predicate::LessThanOrEqual, cmp_ordering::Ordered)
        } else if OP == 4 {
            cmpf(x, scalar, predicate::GreaterThan, cmp_ordering::Ordered)
        } else {
            cmpf(
                x,
                scalar,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            )
        };
        out.store(select(mask, one, zero));
    }

    pub fn softmax_impl<
        const BM: i32,
        const BN: i32,
        const LOG_OUTPUT: bool,
        const NEGATE_INPUT: bool,
    >(
        out: &mut Tensor<f32, { [BM, BN] }>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let input_tile_raw: Tile<f32, { [BM, BN] }> = load_tile_like(input, out);
        let zero = constant(0.0f32, out.shape());
        let input_tile_raw = if NEGATE_INPUT {
            zero - input_tile_raw
        } else {
            input_tile_raw
        };
        let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let col_iota = col_iota
            .reshape(const_shape![1i32, BN])
            .broadcast(out.shape());
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let cols_tile = broadcast_scalar(input_shape[1], out.shape());
        let valid_cols = cmpi(col_iota, cols_tile, predicate::LessThan);
        let negative_infinity = constant(f32::NEG_INFINITY, out.shape());
        let input_tile = select(valid_cols, input_tile_raw, negative_infinity);
        let input_max: Tile<f32, { [BM] }> = reduce_max(input_tile, 1i32);
        let input_max = input_max
            .reshape(const_shape![BM, 1i32])
            .broadcast(out.shape());
        let numerator = exp(input_tile - input_max);
        let denominator: Tile<f32, { [BM] }> = reduce_sum(numerator, 1i32);
        let denominator = denominator
            .reshape(const_shape![BM, 1i32])
            .broadcast(out.shape());
        let value = if LOG_OUTPUT {
            input_tile - input_max - log(denominator)
        } else {
            numerator / denominator
        };
        out.store(value);
    }

    #[cutile::entry()]
    pub fn softmax_lse_f32<const BN: i32>(
        out: &mut Tensor<f32, { [1, 1] }>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_shape = const_shape![1i32, BN];
        let input_tile_raw: Tile<f32, { [1, BN] }> = input
            .partition(tile_shape)
            .load([get_tile_block_id().0, 0i32]);
        let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let col_iota = col_iota.reshape(tile_shape);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let cols_tile = broadcast_scalar(input_shape[1], tile_shape);
        let valid_cols = cmpi(col_iota, cols_tile, predicate::LessThan);
        let input_tile = select(
            valid_cols,
            input_tile_raw,
            constant(f32::NEG_INFINITY, tile_shape),
        );
        let input_max: Tile<f32, { [1] }> = reduce_max(input_tile, 1i32);
        let input_max_broadcast = input_max
            .reshape(const_shape![1i32, 1i32])
            .broadcast(tile_shape);
        let numerator = exp(input_tile - input_max_broadcast);
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerator, 1i32);
        let value = input_max + log(denominator);
        out.store(value.reshape(const_shape![1i32, 1i32]));
    }

    pub fn unary_store<const OP: i32, const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        let x: Tile<f32, { [B] }> = load_tile_like(input, out);
        let zero = constant(0.0f32, out.shape());
        let one = constant(1.0f32, out.shape());
        let half = constant(0.5f32, out.shape());
        let third = constant(0.33333334f32, out.shape());
        let three = constant(3.0f32, out.shape());
        let six = constant(6.0f32, out.shape());
        let pi = constant(std::f32::consts::PI, out.shape());
        let one_eighty = constant(180.0f32, out.shape());
        let selu_alpha = constant(1.6732632f32, out.shape());
        let selu_scale = constant(1.050701f32, out.shape());
        let value = if OP == 0 {
            zero - x
        } else if OP == 1 {
            absf(x)
        } else if OP == 2 {
            sqrt(x, rounding::NearestEven, ftz::Disabled)
        } else if OP == 3 {
            exp(x)
        } else if OP == 4 {
            log(x)
        } else if OP == 5 {
            sin(x)
        } else if OP == 6 {
            cos(x)
        } else if OP == 7 {
            tan(x)
        } else if OP == 8 {
            sinh(x)
        } else if OP == 9 {
            cosh(x)
        } else if OP == 10 {
            tanh(x)
        } else if OP == 11 {
            atan2(x, sqrt(one - x * x, rounding::NearestEven, ftz::Disabled))
        } else if OP == 12 {
            atan2(sqrt(one - x * x, rounding::NearestEven, ftz::Disabled), x)
        } else if OP == 13 {
            atan2(x, one)
        } else if OP == 14 {
            ceil(x)
        } else if OP == 15 {
            floor(x)
        } else if OP == 16 {
            let positive = cmpf(
                x,
                zero,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            select(positive, floor(x), ceil(x))
        } else if OP == 17 {
            one / x
        } else if OP == 18 {
            x * x
        } else if OP == 19 {
            rsqrt(x, ftz::Disabled)
        } else if OP == 20 {
            log(x + sqrt(x * x + one, rounding::NearestEven, ftz::Disabled))
        } else if OP == 21 {
            log(x + sqrt(x * x - one, rounding::NearestEven, ftz::Disabled))
        } else if OP == 22 {
            half * log((one + x) / (one - x))
        } else if OP == 23 {
            max_tile(x, zero)
        } else if OP == 24 {
            one / (one + exp(zero - x))
        } else if OP == 25 {
            x / (one + exp(zero - x))
        } else if OP == 26 {
            let sqrt_2_over_pi = constant(0.7978846f32, out.shape());
            let cubic_coeff = constant(0.044715f32, out.shape());
            half * x * (one + tanh(sqrt_2_over_pi * (x + cubic_coeff * x * x * x)))
        } else if OP == 27 {
            let negative = cmpf(x, zero, predicate::LessThan, cmp_ordering::Ordered);
            let positive = cmpf(x, zero, predicate::GreaterThan, cmp_ordering::Ordered);
            select(negative, zero - one, select(positive, one, zero))
        } else if OP == 28 {
            log(one + x)
        } else if OP == 29 {
            exp(x) - one
        } else if OP == 30 {
            exp2(x, ftz::Disabled)
        } else if OP == 31 {
            log2(x)
        } else if OP == 32 {
            let ln_10 = constant(std::f32::consts::LN_10, out.shape());
            log(x) / ln_10
        } else if OP == 33 {
            let root = pow(absf(x), third);
            let negative = cmpf(x, zero, predicate::LessThan, cmp_ordering::Ordered);
            select(negative, zero - root, root)
        } else if OP == 34 {
            max_tile(x, zero) + log(one + exp(zero - absf(x)))
        } else if OP == 35 {
            min_tile(max_tile((x + three) / six, zero), one)
        } else if OP == 36 {
            x * min_tile(max_tile((x + three) / six, zero), one)
        } else if OP == 37 {
            x * tanh(max_tile(x, zero) + log(one + exp(zero - absf(x))))
        } else if OP == 38 {
            let positive = cmpf(
                x,
                zero,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            selu_scale * select(positive, x, selu_alpha * (exp(x) - one))
        } else if OP == 39 {
            x / (one + absf(x))
        } else if OP == 40 {
            x * pi / one_eighty
        } else if OP == 41 {
            x * one_eighty / pi
        } else if OP == 42 {
            let positive = cmpf(
                x,
                zero,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            let truncated = select(positive, floor(x), ceil(x));
            x - truncated
        } else if OP == 43 {
            min_tile(max_tile(x, zero), six)
        } else if OP == 44 {
            x - tanh(x)
        } else if OP == 45 {
            zero - (max_tile(zero - x, zero) + log(one + exp(zero - absf(zero - x))))
        } else {
            x
        };
        out.store(value);
    }

    pub fn nan_to_num_store<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        input: &Tensor<f32, { [-1] }>,
        nan: f32,
        posinf: f32,
        neginf: f32,
    ) {
        let x: Tile<f32, { [B] }> = load_tile_like(input, out);
        let zero = constant(0.0f32, out.shape());
        let infinity = constant(f32::INFINITY, out.shape());
        let abs_x = absf(x);
        let numeric = cmpf(x, x, predicate::Equal, cmp_ordering::Ordered);
        let infinite = cmpf(abs_x, infinity, predicate::Equal, cmp_ordering::Ordered);
        let positive = cmpf(x, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let nan = broadcast_scalar(nan, out.shape());
        let posinf = broadcast_scalar(posinf, out.shape());
        let neginf = broadcast_scalar(neginf, out.shape());
        let inf_value = select(positive, posinf, neginf);
        out.store(select(numeric, select(infinite, inf_value, x), nan));
    }

    pub fn unary_bool_store<const OP: i32, const B: i32>(
        out: &mut Tensor<u8, { [B] }>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        let x: Tile<f32, { [B] }> = load_tile_like(input, out);
        let one = constant(1u8, out.shape());
        let zero = constant(0u8, out.shape());
        let infinity = constant(f32::INFINITY, out.shape());
        let abs_x = absf(x);
        let mask = if OP == 0 {
            cmpf(x, x, predicate::NotEqual, cmp_ordering::Unordered)
        } else if OP == 1 {
            cmpf(abs_x, infinity, predicate::Equal, cmp_ordering::Ordered)
        } else if OP == 2 {
            cmpf(x, infinity, predicate::Equal, cmp_ordering::Ordered)
        } else if OP == 3 {
            cmpf(
                x,
                constant(f32::NEG_INFINITY, out.shape()),
                predicate::Equal,
                cmp_ordering::Ordered,
            )
        } else if OP == 4 {
            let numeric = cmpf(x, x, predicate::Equal, cmp_ordering::Ordered);
            let not_infinite = cmpf(abs_x, infinity, predicate::NotEqual, cmp_ordering::Ordered);
            numeric & not_infinite
        } else if OP == 5 {
            let numeric = cmpf(x, x, predicate::Equal, cmp_ordering::Ordered);
            let not_infinite = cmpf(abs_x, infinity, predicate::NotEqual, cmp_ordering::Ordered);
            let min_normal = constant(f32::MIN_POSITIVE, out.shape());
            let normal_magnitude = cmpf(
                abs_x,
                min_normal,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            numeric & not_infinite & normal_magnitude
        } else if OP == 10 {
            let numeric = cmpf(x, x, predicate::Equal, cmp_ordering::Ordered);
            let nonzero = cmpf(
                x,
                constant(0.0f32, out.shape()),
                predicate::NotEqual,
                cmp_ordering::Ordered,
            );
            let min_normal = constant(f32::MIN_POSITIVE, out.shape());
            let subnormal_magnitude = cmpf(
                abs_x,
                min_normal,
                predicate::LessThan,
                cmp_ordering::Ordered,
            );
            numeric & nonzero & subnormal_magnitude
        } else if OP == 6 {
            cmpf(
                x,
                constant(0.0f32, out.shape()),
                predicate::Equal,
                cmp_ordering::Ordered,
            )
        } else if OP == 7 {
            cmpf(
                x,
                constant(0.0f32, out.shape()),
                predicate::GreaterThan,
                cmp_ordering::Ordered,
            )
        } else if OP == 9 {
            cmpf(
                x,
                constant(0.0f32, out.shape()),
                predicate::NotEqual,
                cmp_ordering::Unordered,
            )
        } else {
            cmpf(
                x,
                constant(0.0f32, out.shape()),
                predicate::LessThan,
                cmp_ordering::Ordered,
            )
        };
        out.store(select(mask, one, zero));
    }
}

pub use kernels::*;
