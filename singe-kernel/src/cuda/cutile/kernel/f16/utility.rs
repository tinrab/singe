#[cutile::module]
mod kernels {
    use cutile::core::*;

    pub fn activation_unary_store<const OP: i32, const B: i32>(
        out: &mut Tensor<f16, { [B] }>,
        input: &Tensor<f16, { [-1] }>,
    ) {
        let x: Tile<f16, { [B] }> = load_tile_like(input, out);
        let zero = constant(f16::from_f32(0.0), out.shape());
        let one = constant(f16::from_f32(1.0), out.shape());
        let half = constant(f16::from_f32(0.5), out.shape());
        let third = constant(f16::from_f32(0.33333334), out.shape());
        let three = constant(f16::from_f32(3.0), out.shape());
        let six = constant(f16::from_f32(6.0), out.shape());
        let pi = constant(f16::from_f32(std::f32::consts::PI), out.shape());
        let one_eighty = constant(f16::from_f32(180.0), out.shape());
        let selu_alpha = constant(f16::from_f32(1.6732632), out.shape());
        let selu_scale = constant(f16::from_f32(1.050701), out.shape());
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
            let sqrt_2_over_pi = constant(f16::from_f32(0.7978846), out.shape());
            let cubic_coeff = constant(f16::from_f32(0.044715), out.shape());
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
            let ln_10 = constant(f16::from_f32(std::f32::consts::LN_10), out.shape());
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
        out: &mut Tensor<f16, { [B] }>,
        lhs: &Tensor<f16, { [-1] }>,
        rhs: &Tensor<f16, { [-1] }>,
    ) {
        let lhs_tile: Tile<f16, { [B] }> = load_tile_like(lhs, out);
        let rhs_tile: Tile<f16, { [B] }> = load_tile_like(rhs, out);
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
            let neg_inf = constant(f16::from_f32(f32::NEG_INFINITY), out.shape());
            let both_neg_inf = cmpf(lhs_tile, neg_inf, predicate::Equal, cmp_ordering::Ordered)
                & cmpf(rhs_tile, neg_inf, predicate::Equal, cmp_ordering::Ordered);
            let value = max + log(exp(lhs_tile - max) + exp(rhs_tile - max));
            select(both_neg_inf, neg_inf, value)
        } else if OP == 12 {
            let zero = constant(f16::from_f32(0.0), out.shape());
            let zero_lhs = cmpf(lhs_tile, zero, predicate::Equal, cmp_ordering::Ordered);
            select(zero_lhs, zero, lhs_tile * log(rhs_tile))
        } else {
            lhs_tile
        };
        out.store(value);
    }

    pub fn cmp_store<const OP: i32, const B: i32>(
        out: &mut Tensor<f16, { [B] }>,
        lhs: &Tensor<f16, { [-1] }>,
        rhs: &Tensor<f16, { [-1] }>,
    ) {
        let lhs_tile: Tile<f16, { [B] }> = load_tile_like(lhs, out);
        let rhs_tile: Tile<f16, { [B] }> = load_tile_like(rhs, out);
        let one = constant(f16::from_f32(1.0), out.shape());
        let zero = constant(f16::from_f32(0.0), out.shape());
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
        lhs: &Tensor<f16, { [-1] }>,
        rhs: &Tensor<f16, { [-1] }>,
    ) {
        let lhs_tile: Tile<f16, { [B] }> = load_tile_like(lhs, out);
        let rhs_tile: Tile<f16, { [B] }> = load_tile_like(rhs, out);
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
    pub unsafe fn dequantize_u8_f16(
        out: *mut f16,
        input: *mut u8,
        scale: f16,
        zero_point: f16,
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
        let input_f16: Tile<f16, { [128] }> = convert_tile(input_result.0);
        let values = (input_f16 - broadcast_scalar(zero_point, tile_shape))
            * broadcast_scalar(scale, tile_shape);

        let out_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_ptrs.offset_tile(offsets);
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
    pub unsafe fn dequantize_i8_f16(
        out: *mut f16,
        input: *mut i8,
        scale: f16,
        zero_point: f16,
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
        let input_f16: Tile<f16, { [128] }> = convert_tile(input_result.0);
        let values = (input_f16 - broadcast_scalar(zero_point, tile_shape))
            * broadcast_scalar(scale, tile_shape);

        let out_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_ptrs.offset_tile(offsets);
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

    pub fn reduce_impl<const OP: i32, const BN: i32>(
        out: &mut Tensor<f16, { [1, 1] }>,
        input: &Tensor<f16, { [-1, -1] }>,
        divisor: f16,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1i32, BN];
        let input_tile_raw: Tile<f16, { [1, BN] }> =
            input.partition(tile_shape).load([pid.0, 0i32]);
        let input_tile_raw_f32: Tile<f32, { [1, BN] }> = convert_tile(input_tile_raw);
        let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let col_iota = col_iota.reshape(tile_shape);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let cols_tile = broadcast_scalar(input_shape[1], tile_shape);
        let valid_cols = cmpi(col_iota, cols_tile, predicate::LessThan);
        let value: Tile<f16, { [1] }> = if OP == 0 || OP == 1 {
            let zero = constant(0.0f32, tile_shape);
            let input_tile = select(valid_cols, input_tile_raw_f32, zero);
            let sum: Tile<f32, { [1] }> = reduce_sum(input_tile, 1i32);
            if OP == 1 {
                let divisor = broadcast_scalar(convert_scalar(divisor), const_shape![1]);
                convert_tile(sum / divisor)
            } else {
                convert_tile(sum)
            }
        } else if OP == 5 || OP == 6 {
            let zero = constant(0.0f32, tile_shape);
            let input_tile = select(valid_cols, input_tile_raw_f32, zero);
            let divisor: Tile<f32, { [1] }> =
                broadcast_scalar(convert_scalar(divisor), const_shape![1]);
            let sum: Tile<f32, { [1] }> = reduce_sum(input_tile, 1i32);
            let mean: Tile<f32, { [1] }> = sum / divisor;
            let mean_broadcast = mean.reshape(const_shape![1i32, 1i32]).broadcast(tile_shape);
            let diff = select(valid_cols, input_tile_raw_f32 - mean_broadcast, zero);
            let variance_sum: Tile<f32, { [1] }> = reduce_sum(diff * diff, 1i32);
            let variance: Tile<f32, { [1] }> = variance_sum / divisor;
            if OP == 6 {
                convert_tile(sqrt(variance, rounding::NearestEven, ftz::Disabled))
            } else {
                convert_tile(variance)
            }
        } else if OP == 2 {
            let fill = constant(f16::NEG_INFINITY, tile_shape);
            reduce_max(select(valid_cols, input_tile_raw, fill), 1i32)
        } else if OP == 3 {
            let fill = constant(f16::INFINITY, tile_shape);
            reduce_min(select(valid_cols, input_tile_raw, fill), 1i32)
        } else {
            let one = constant(1.0f32, tile_shape);
            let product: Tile<f32, { [1] }> =
                reduce_prod(select(valid_cols, input_tile_raw_f32, one), 1i32);
            convert_tile(product)
        };
        out.store(value.reshape(const_shape![1i32, 1i32]));
    }

    pub fn arg_reduce_impl<const OP: i32, const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<f16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1i32, BN];
        let input_tile_raw: Tile<f16, { [1, BN] }> =
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
            let fill: Tile<f16, { [1, BN] }> = constant(f16::NEG_INFINITY, tile_shape);
            select(valid_cols & numeric, input_tile_raw, fill)
        } else {
            let fill: Tile<f16, { [1, BN] }> = constant(f16::INFINITY, tile_shape);
            select(valid_cols & numeric, input_tile_raw, fill)
        };
        let best_value: Tile<f16, { [1] }> = if OP == 0 {
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
        out: &mut Tensor<f16, { [B] }>,
        input: &Tensor<f16, { [-1] }>,
        scalar: f16,
    ) {
        let x: Tile<f16, { [B] }> = load_tile_like(input, out);
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
            let neg_inf = constant(f16::from_f32(f32::NEG_INFINITY), out.shape());
            let both_neg_inf = cmpf(x, neg_inf, predicate::Equal, cmp_ordering::Ordered)
                & cmpf(scalar, neg_inf, predicate::Equal, cmp_ordering::Ordered);
            let value = max + log(exp(x - max) + exp(scalar - max));
            select(both_neg_inf, neg_inf, value)
        } else if OP == 17 {
            let zero = constant(f16::from_f32(0.0), out.shape());
            let zero_x = cmpf(x, zero, predicate::Equal, cmp_ordering::Ordered);
            select(zero_x, zero, x * log(scalar))
        } else if OP == 18 {
            let zero = constant(f16::from_f32(0.0), out.shape());
            let zero_scalar = cmpf(scalar, zero, predicate::Equal, cmp_ordering::Ordered);
            select(zero_scalar, zero, scalar * log(x))
        } else {
            x
        };
        out.store(value);
    }

    pub fn scalar_cmp_bool_store<const OP: i32, const B: i32>(
        out: &mut Tensor<u8, { [B] }>,
        input: &Tensor<f16, { [-1] }>,
        scalar: f16,
    ) {
        let x: Tile<f16, { [B] }> = load_tile_like(input, out);
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
        out: &mut Tensor<f16, { [BM, BN] }>,
        input: &Tensor<f16, { [-1, -1] }>,
    ) {
        let input_tile_raw: Tile<f16, { [BM, BN] }> = load_tile_like(input, out);
        let input_tile_raw: Tile<f32, { [BM, BN] }> = convert_tile(input_tile_raw);
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
        out.store(convert_tile(value));
    }

    #[cutile::entry()]
    pub fn softmax_lse_f16<const BN: i32>(
        out: &mut Tensor<f32, { [1, 1] }>,
        input: &Tensor<f16, { [-1, -1] }>,
    ) {
        let tile_shape = const_shape![1i32, BN];
        let input_tile_raw: Tile<f16, { [1, BN] }> = input
            .partition(tile_shape)
            .load([get_tile_block_id().0, 0i32]);
        let input_tile_raw: Tile<f32, { [1, BN] }> = convert_tile(input_tile_raw);
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
        out: &mut Tensor<f16, { [B] }>,
        input: &Tensor<f16, { [-1] }>,
    ) {
        let x: Tile<f16, { [B] }> = load_tile_like(input, out);
        let zero = constant(f16::from_f32(0.0), out.shape());
        let one = constant(f16::from_f32(1.0), out.shape());
        let half = constant(f16::from_f32(0.5), out.shape());
        let third = constant(f16::from_f32(0.33333334), out.shape());
        let three = constant(f16::from_f32(3.0), out.shape());
        let six = constant(f16::from_f32(6.0), out.shape());
        let pi = constant(f16::from_f32(std::f32::consts::PI), out.shape());
        let one_eighty = constant(f16::from_f32(180.0), out.shape());
        let selu_alpha = constant(f16::from_f32(1.6732632), out.shape());
        let selu_scale = constant(f16::from_f32(1.050701), out.shape());
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
            let sqrt_2_over_pi = constant(f16::from_f32(0.7978846), out.shape());
            let cubic_coeff = constant(f16::from_f32(0.044715), out.shape());
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
            let ln_10 = constant(f16::from_f32(std::f32::consts::LN_10), out.shape());
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
        out: &mut Tensor<f16, { [B] }>,
        input: &Tensor<f16, { [-1] }>,
        nan: f16,
        posinf: f16,
        neginf: f16,
    ) {
        let x: Tile<f16, { [B] }> = load_tile_like(input, out);
        let zero = constant(f16::from_f32(0.0), out.shape());
        let infinity = constant(f16::from_f32(f32::INFINITY), out.shape());
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
        input: &Tensor<f16, { [-1] }>,
    ) {
        let x: Tile<f16, { [B] }> = load_tile_like(input, out);
        let one = constant(1u8, out.shape());
        let zero = constant(0u8, out.shape());
        let infinity = constant(f16::from_f32(f32::INFINITY), out.shape());
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
                constant(f16::from_f32(f32::NEG_INFINITY), out.shape()),
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
            let min_normal = constant(f16::from_f32(0.000_061_035_156), out.shape());
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
                constant(f16::from_f32(0.0), out.shape()),
                predicate::NotEqual,
                cmp_ordering::Ordered,
            );
            let min_normal = constant(f16::from_f32(0.000_061_035_156), out.shape());
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
                constant(f16::from_f32(0.0), out.shape()),
                predicate::Equal,
                cmp_ordering::Ordered,
            )
        } else if OP == 7 {
            cmpf(
                x,
                constant(f16::from_f32(0.0), out.shape()),
                predicate::GreaterThan,
                cmp_ordering::Ordered,
            )
        } else if OP == 9 {
            cmpf(
                x,
                constant(f16::from_f32(0.0), out.shape()),
                predicate::NotEqual,
                cmp_ordering::Unordered,
            )
        } else {
            cmpf(
                x,
                constant(f16::from_f32(0.0), out.shape()),
                predicate::LessThan,
                cmp_ordering::Ordered,
            )
        };
        out.store(select(mask, one, zero));
    }
}

pub use kernels::*;
