#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn layer_norm_f32<const BN: i32>(
        out: &mut Tensor<f32, { [1, BN] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        weight: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        has_weight: i32,
        has_bias: i32,
        eps: f32,
        weight_offset: f32,
    ) {
        normalization_impl::<0, BN>(
            out,
            input,
            weight,
            bias,
            has_weight,
            has_bias,
            eps,
            weight_offset,
        );
    }

    #[cutile::entry()]
    pub fn rms_norm_f32<const BN: i32>(
        out: &mut Tensor<f32, { [1, BN] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        weight: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        has_weight: i32,
        has_bias: i32,
        eps: f32,
        weight_offset: f32,
    ) {
        normalization_impl::<1, BN>(
            out,
            input,
            weight,
            bias,
            has_weight,
            has_bias,
            eps,
            weight_offset,
        );
    }

    #[cutile::entry()]
    pub fn sparsemax_f32<const BN: i32>(
        out: &mut Tensor<f32, { [1, BN] }>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_shape = const_shape![1, BN];
        let input_tile_raw: Tile<f32, { [1, BN] }> = load_tile_like(input, out);
        let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let col_iota = col_iota.reshape(tile_shape);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let cols_tile = broadcast_scalar(input_shape[1], tile_shape);
        let valid_cols = cmpi(col_iota, cols_tile, predicate::LessThan);
        let zero = constant(0.0f32, tile_shape);
        let one_row: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
        let half_row: Tile<f32, { [1] }> = constant(0.5f32, const_shape![1]);
        let input_tile = select(valid_cols, input_tile_raw, zero);
        let cols: f32 = convert_scalar(input_shape[1]);
        let divisor = broadcast_scalar(cols, const_shape![1]);
        let sum: Tile<f32, { [1] }> = reduce_sum(input_tile, 1i32);
        let neg_large = zero - constant(1.0e38f32, tile_shape);
        let max_values = select(valid_cols, input_tile_raw, neg_large);
        let max_value: Tile<f32, { [1] }> = reduce_max(max_values, 1i32);
        let mut tau_lo = (sum - one_row) / divisor;
        let mut tau_hi = max_value;

        for _ in 0..20 {
            let tau_mid = half_row * (tau_lo + tau_hi);
            let tau = tau_mid.reshape(const_shape![1, 1]).broadcast(tile_shape);
            let in_support = valid_cols
                & cmpf(
                    input_tile_raw,
                    tau,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                );
            let support_values: Tile<f32, { [1, BN] }> =
                select(in_support, input_tile_raw - tau, zero);
            let support_sum: Tile<f32, { [1] }> = reduce_sum(support_values, 1i32);
            let enough_mass = cmpf(
                support_sum,
                one_row,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            tau_lo = select(enough_mass, tau_mid, tau_lo);
            tau_hi = select(enough_mass, tau_hi, tau_mid);
        }

        let tau = (half_row * (tau_lo + tau_hi))
            .reshape(const_shape![1, 1])
            .broadcast(tile_shape);
        let values: Tile<f32, { [1, BN] }> =
            select(valid_cols, max_tile(input_tile_raw - tau, zero), zero);
        out.store(values);
    }

    #[cutile::entry()]
    pub unsafe fn sparsemax_f32_wide(out: *mut f32, input: *mut f32, rows: i32, cols: i32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let tile_shape = const_shape![128];
        let chunks = (cols + 127i32) / 128i32;
        let row_valid = row < rows;
        let row_offset = row * cols;
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);

        let mut sum_tile: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut max_tile_values: Tile<f32, { [128] }> = constant(-1.0e38f32, tile_shape);
        for chunk in 0i32..chunks {
            let col_offsets = iota(tile_shape) + broadcast_scalar(chunk * 128i32, tile_shape);
            let mask = cmpi(
                col_offsets,
                broadcast_scalar(cols, tile_shape),
                predicate::LessThan,
            ) & broadcast_scalar(row_valid, tile_shape);
            let safe_col_offsets = select(mask, col_offsets, zero_offsets);
            let values = load_f32_parameter(
                input,
                broadcast_scalar(row_offset, tile_shape) + safe_col_offsets,
                mask,
                0.0f32,
            );
            let neg_large: Tile<f32, { [128] }> = constant(-1.0e38f32, tile_shape);
            sum_tile = sum_tile + select(mask, values, zero_values);
            max_tile_values = max_tile(max_tile_values, select(mask, values, neg_large));
        }

        let sum: Tile<f32, { [] }> = reduce_sum(sum_tile, 0i32);
        let max_value: Tile<f32, { [] }> = reduce_max(max_tile_values, 0i32);
        let cols_f32: f32 = convert_scalar(cols);
        let divisor = broadcast_scalar(cols_f32, const_shape![]);
        let one: Tile<f32, { [] }> = constant(1.0f32, const_shape![]);
        let half: Tile<f32, { [] }> = constant(0.5f32, const_shape![]);
        let mut tau_lo = (sum - one) / divisor;
        let mut tau_hi = max_value;

        for _ in 0..20 {
            let tau_mid = half * (tau_lo + tau_hi);
            let tau_tile = tau_mid.reshape(const_shape![1]).broadcast(tile_shape);
            let mut support_sum_tile: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            for chunk in 0i32..chunks {
                let col_offsets = iota(tile_shape) + broadcast_scalar(chunk * 128i32, tile_shape);
                let mask = cmpi(
                    col_offsets,
                    broadcast_scalar(cols, tile_shape),
                    predicate::LessThan,
                ) & broadcast_scalar(row_valid, tile_shape);
                let safe_col_offsets = select(mask, col_offsets, zero_offsets);
                let values = load_f32_parameter(
                    input,
                    broadcast_scalar(row_offset, tile_shape) + safe_col_offsets,
                    mask,
                    0.0f32,
                );
                let in_support = mask
                    & cmpf(
                        values,
                        tau_tile,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    );
                support_sum_tile =
                    support_sum_tile + select(in_support, values - tau_tile, zero_values);
            }
            let support_sum: Tile<f32, { [] }> = reduce_sum(support_sum_tile, 0i32);
            let enough_mass = cmpf(
                support_sum,
                one,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            tau_lo = select(enough_mass, tau_mid, tau_lo);
            tau_hi = select(enough_mass, tau_hi, tau_mid);
        }

        let tau = (half * (tau_lo + tau_hi))
            .reshape(const_shape![1])
            .broadcast(tile_shape);
        for chunk in 0i32..chunks {
            let col_offsets = iota(tile_shape) + broadcast_scalar(chunk * 128i32, tile_shape);
            let mask = cmpi(
                col_offsets,
                broadcast_scalar(cols, tile_shape),
                predicate::LessThan,
            ) & broadcast_scalar(row_valid, tile_shape);
            let safe_col_offsets = select(mask, col_offsets, zero_offsets);
            let offsets = broadcast_scalar(row_offset, tile_shape) + col_offsets;
            let safe_offsets = broadcast_scalar(row_offset, tile_shape) + safe_col_offsets;
            let values = load_f32_parameter(input, safe_offsets, mask, 0.0f32);
            let output_values = max_tile(values - tau, zero_values);
            store_f32_vector(out, offsets, output_values, mask);
        }
    }

    #[cutile::entry()]
    pub unsafe fn group_norm_f32(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        channels: i32,
        groups: i32,
        spatial_len: i32,
        eps: f32,
    ) {
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let batch = tile_id.0;
        let group = tile_id.1;
        let tile_shape = const_shape![128];
        let channels_per_group = channels / groups;
        let spatial_chunks = (spatial_len + 127i32) / 128i32;
        let group_values = channels_per_group * spatial_len;
        let group_values_f32: f32 = convert_scalar(group_values);
        let inv_group_values: Tile<f32, { [] }> =
            constant(1.0f32, const_shape![]) / broadcast_scalar(group_values_f32, const_shape![]);

        let mut sum_tile: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut sum_sq_tile: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for channel_in_group in 0i32..channels_per_group {
            let channel = group * channels_per_group + channel_in_group;
            let row_offset = (batch * channels + channel) * spatial_len;
            for chunk in 0i32..spatial_chunks {
                let spatial_offsets =
                    iota(tile_shape) + broadcast_scalar(chunk * 128i32, tile_shape);
                let mask = cmpi(
                    spatial_offsets,
                    broadcast_scalar(spatial_len, tile_shape),
                    predicate::LessThan,
                );
                let values = load_f32_parameter(
                    input,
                    broadcast_scalar(row_offset, tile_shape) + spatial_offsets,
                    mask,
                    0.0f32,
                );
                sum_tile = sum_tile + values;
                sum_sq_tile = sum_sq_tile + values * values;
            }
        }

        let sum: Tile<f32, { [] }> = reduce_sum(sum_tile, 0i32);
        let sum_sq: Tile<f32, { [] }> = reduce_sum(sum_sq_tile, 0i32);
        let mean = sum * inv_group_values;
        let variance = sum_sq * inv_group_values - mean * mean;
        let rstd = constant(1.0f32, const_shape![])
            / sqrt(
                variance + broadcast_scalar(eps, const_shape![]),
                rounding::NearestEven,
                ftz::Disabled,
            );

        let mean_values = mean.reshape(const_shape![1]).broadcast(tile_shape);
        let rstd_values = rstd.reshape(const_shape![1]).broadcast(tile_shape);
        let full_mask: Tile<bool, { [128] }> = constant(true, tile_shape);
        for channel_in_group in 0i32..channels_per_group {
            let channel = group * channels_per_group + channel_in_group;
            let row_offset = (batch * channels + channel) * spatial_len;
            let channel_tile = broadcast_scalar(channel, tile_shape);
            let scale = load_f32_parameter(weight, channel_tile, full_mask, 1.0f32);
            let shift = load_f32_parameter(bias, channel_tile, full_mask, 0.0f32);
            for chunk in 0i32..spatial_chunks {
                let spatial_offsets =
                    iota(tile_shape) + broadcast_scalar(chunk * 128i32, tile_shape);
                let mask = cmpi(
                    spatial_offsets,
                    broadcast_scalar(spatial_len, tile_shape),
                    predicate::LessThan,
                );
                let offsets = broadcast_scalar(row_offset, tile_shape) + spatial_offsets;
                let values = load_f32_parameter(input, offsets, mask, 0.0f32);
                let normalized = (values - mean_values) * rstd_values * scale + shift;
                store_f32_vector(out, offsets, normalized, mask);
            }
        }
    }

    fn normalization_impl<const OP: i32, const BN: i32>(
        out: &mut Tensor<f32, { [1, BN] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        weight: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        has_weight: i32,
        has_bias: i32,
        eps: f32,
        weight_offset: f32,
    ) {
        let tile_shape = const_shape![1i32, BN];
        let input_tile_raw: Tile<f32, { [1, BN] }> = load_tile_like(input, out);
        let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let col_iota = col_iota.reshape(tile_shape);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let cols_tile = broadcast_scalar(input_shape[1], tile_shape);
        let valid_cols = cmpi(col_iota, cols_tile, predicate::LessThan);
        let zero = constant(0.0f32, tile_shape);
        let one = constant(1.0f32, tile_shape);
        let one_row: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
        let input_tile = select(valid_cols, input_tile_raw, zero);
        let cols: f32 = convert_scalar(input_shape[1]);
        let divisor = broadcast_scalar(cols, const_shape![1]);
        let values = if OP == 0 {
            let mean_sum: Tile<f32, { [1] }> = reduce_sum(input_tile, 1i32);
            let mean = (mean_sum / divisor)
                .reshape(const_shape![1i32, 1i32])
                .broadcast(tile_shape);
            let diff = select(valid_cols, input_tile - mean, zero);
            let variance_sum: Tile<f32, { [1] }> = reduce_sum(diff * diff, 1i32);
            let variance = variance_sum / divisor;
            let inv_std_value = one_row
                / sqrt(
                    variance + broadcast_scalar(eps, const_shape![1]),
                    rounding::NearestEven,
                    ftz::Disabled,
                );
            let inv_std = inv_std_value
                .reshape(const_shape![1i32, 1i32])
                .broadcast(tile_shape);
            diff * inv_std
        } else {
            let mean_square_sum: Tile<f32, { [1] }> = reduce_sum(input_tile * input_tile, 1i32);
            let mean_square = mean_square_sum / divisor;
            let inv_rms_value = one_row
                / sqrt(
                    mean_square + broadcast_scalar(eps, const_shape![1]),
                    rounding::NearestEven,
                    ftz::Disabled,
                );
            let inv_rms = inv_rms_value
                .reshape(const_shape![1i32, 1i32])
                .broadcast(tile_shape);
            input_tile * inv_rms
        };
        let weight_tile: Tile<f32, { [1, BN] }> = if has_weight != 0 {
            let raw: Tile<f32, { [BN] }> = weight.partition(const_shape![BN]).load([0i32]);
            let raw = raw.reshape(tile_shape);
            if OP == 1 {
                raw + broadcast_scalar(weight_offset, tile_shape)
            } else {
                raw
            }
        } else {
            one
        };
        let bias_tile: Tile<f32, { [1, BN] }> = if has_bias != 0 {
            let raw: Tile<f32, { [BN] }> = bias.partition(const_shape![BN]).load([0i32]);
            raw.reshape(tile_shape)
        } else {
            zero
        };
        out.store(select(valid_cols, values * weight_tile + bias_tile, zero));
    }

    #[cutile::entry()]
    pub unsafe fn batch_norm_inference_f32(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        running_mean: *mut f32,
        running_var: *mut f32,
        feature_count: i32,
        spatial_len: i32,
        has_weight: i32,
        has_bias: i32,
        has_running_mean: i32,
        has_running_var: i32,
        eps: f32,
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

        let feature_period = feature_count * spatial_len;
        let feature_index: Tile<i32, { [128] }> = (safe_offsets
            - (safe_offsets / broadcast_scalar(feature_period, tile_shape))
                * broadcast_scalar(feature_period, tile_shape))
            / broadcast_scalar(spatial_len, tile_shape);

        let input_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f32, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_ptrs.offset_tile(safe_offsets);
        let input_result: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f32),
            None,
            Latency::<0>,
        );

        let zero = constant(0.0f32, tile_shape);
        let one = constant(1.0f32, tile_shape);
        let mean = if has_running_mean != 0 {
            load_f32_parameter(running_mean, feature_index, mask, 0.0f32)
        } else {
            zero
        };
        let var = if has_running_var != 0 {
            load_f32_parameter(running_var, feature_index, mask, 1.0f32)
        } else {
            one
        };
        let scale = if has_weight != 0 {
            load_f32_parameter(weight, feature_index, mask, 1.0f32)
        } else {
            one
        };
        let shift = if has_bias != 0 {
            load_f32_parameter(bias, feature_index, mask, 0.0f32)
        } else {
            zero
        };
        let inv_std = one
            / sqrt(
                var + broadcast_scalar(eps, tile_shape),
                rounding::NearestEven,
                ftz::Disabled,
            );
        let values = (input_result.0 - mean) * inv_std * scale + shift;

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

    fn load_f32_parameter(
        input: *mut f32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: f32,
    ) -> Tile<f32, { [128] }> {
        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*mut f32, { [1] }> = base.reshape(const_shape![1]);
        let ptrs: PointerTile<*mut f32, { [128] }> = base.broadcast(const_shape![128]);
        let ptrs: PointerTile<*mut f32, { [128] }> = ptrs.offset_tile(offsets);
        let result: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
            ptrs,
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
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
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
}

pub use kernels::*;
