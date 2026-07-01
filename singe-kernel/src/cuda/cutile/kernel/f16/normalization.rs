#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn layer_norm_f16<const BN: i32>(
        out: &mut Tensor<f16, { [1, BN] }>,
        input: &Tensor<f16, { [-1, -1] }>,
        weight: &Tensor<f16, { [-1] }>,
        bias: &Tensor<f16, { [-1] }>,
        has_weight: i32,
        has_bias: i32,
        eps: f16,
        weight_offset: f16,
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
    pub fn rms_norm_f16<const BN: i32>(
        out: &mut Tensor<f16, { [1, BN] }>,
        input: &Tensor<f16, { [-1, -1] }>,
        weight: &Tensor<f16, { [-1] }>,
        bias: &Tensor<f16, { [-1] }>,
        has_weight: i32,
        has_bias: i32,
        eps: f16,
        weight_offset: f16,
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

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub fn layer_norm_bf16<const BN: i32>(
        out: &mut Tensor<bf16, { [1, BN] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
        weight: &Tensor<bf16, { [-1] }>,
        bias: &Tensor<bf16, { [-1] }>,
        has_weight: i32,
        has_bias: i32,
        eps: bf16,
        weight_offset: bf16,
    ) {
        normalization_impl_bf16::<0, BN>(
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

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub fn rms_norm_bf16<const BN: i32>(
        out: &mut Tensor<bf16, { [1, BN] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
        weight: &Tensor<bf16, { [-1] }>,
        bias: &Tensor<bf16, { [-1] }>,
        has_weight: i32,
        has_bias: i32,
        eps: bf16,
        weight_offset: bf16,
    ) {
        normalization_impl_bf16::<1, BN>(
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

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub fn sparsemax_f16<const BN: i32>(
        out: &mut Tensor<f16, { [1, BN] }>,
        input: &Tensor<f16, { [-1, -1] }>,
    ) {
        sparsemax_tile_impl_f16::<BN>(out, input);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub fn sparsemax_bf16<const BN: i32>(
        out: &mut Tensor<bf16, { [1, BN] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
    ) {
        sparsemax_tile_impl_bf16::<BN>(out, input);
    }

    #[cfg(feature = "dtype-f16")]
    fn sparsemax_tile_impl_f16<const BN: i32>(
        out: &mut Tensor<f16, { [1, BN] }>,
        input: &Tensor<f16, { [-1, -1] }>,
    ) {
        let tile_shape = const_shape![1, BN];
        let input_tile_raw: Tile<f16, { [1, BN] }> = load_tile_like(input, out);
        let input_tile_raw: Tile<f32, { [1, BN] }> = convert_tile(input_tile_raw);
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
        let values: Tile<f16, { [1, BN] }> = convert_tile(values);
        out.store(values);
    }

    #[cfg(feature = "dtype-bf16")]
    fn sparsemax_tile_impl_bf16<const BN: i32>(
        out: &mut Tensor<bf16, { [1, BN] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
    ) {
        let tile_shape = const_shape![1, BN];
        let input_tile_raw: Tile<bf16, { [1, BN] }> = load_tile_like(input, out);
        let input_tile_raw: Tile<f32, { [1, BN] }> = convert_tile(input_tile_raw);
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
        let values: Tile<bf16, { [1, BN] }> = convert_tile(values);
        out.store(values);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn sparsemax_f16_wide(out: *mut f16, input: *mut f16, rows: i32, cols: i32) {
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
            let values: Tile<f32, { [128] }> = convert_tile(load_f16_parameter_no_fill(
                input,
                broadcast_scalar(row_offset, tile_shape) + safe_col_offsets,
                mask,
            ));
            let values = select(mask, values, zero_values);
            let neg_large: Tile<f32, { [128] }> = constant(-1.0e38f32, tile_shape);
            sum_tile = sum_tile + values;
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
                let values: Tile<f32, { [128] }> = convert_tile(load_f16_parameter_no_fill(
                    input,
                    broadcast_scalar(row_offset, tile_shape) + safe_col_offsets,
                    mask,
                ));
                let values = select(mask, values, zero_values);
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
            let values: Tile<f32, { [128] }> =
                convert_tile(load_f16_parameter_no_fill(input, safe_offsets, mask));
            let values = select(mask, values, zero_values);
            let output_values: Tile<f16, { [128] }> =
                convert_tile(max_tile(values - tau, zero_values));
            store_f16_vector(out, offsets, output_values, mask);
        }
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn sparsemax_bf16_wide(out: *mut bf16, input: *mut bf16, rows: i32, cols: i32) {
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
            let values: Tile<f32, { [128] }> = convert_tile(load_bf16_parameter_no_fill(
                input,
                broadcast_scalar(row_offset, tile_shape) + safe_col_offsets,
                mask,
            ));
            let values = select(mask, values, zero_values);
            let neg_large: Tile<f32, { [128] }> = constant(-1.0e38f32, tile_shape);
            sum_tile = sum_tile + values;
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
                let values: Tile<f32, { [128] }> = convert_tile(load_bf16_parameter_no_fill(
                    input,
                    broadcast_scalar(row_offset, tile_shape) + safe_col_offsets,
                    mask,
                ));
                let values = select(mask, values, zero_values);
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
            let values: Tile<f32, { [128] }> =
                convert_tile(load_bf16_parameter_no_fill(input, safe_offsets, mask));
            let values = select(mask, values, zero_values);
            let output_values: Tile<bf16, { [128] }> =
                convert_tile(max_tile(values - tau, zero_values));
            store_bf16_vector(out, offsets, output_values, mask);
        }
    }

    fn normalization_impl<const OP: i32, const BN: i32>(
        out: &mut Tensor<f16, { [1, BN] }>,
        input: &Tensor<f16, { [-1, -1] }>,
        weight: &Tensor<f16, { [-1] }>,
        bias: &Tensor<f16, { [-1] }>,
        has_weight: i32,
        has_bias: i32,
        eps: f16,
        weight_offset: f16,
    ) {
        let tile_shape = const_shape![1i32, BN];
        let input_tile_raw: Tile<f16, { [1, BN] }> = load_tile_like(input, out);
        let input_tile_raw: Tile<f32, { [1, BN] }> = convert_tile(input_tile_raw);
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
        let eps_f32: f32 = convert_scalar(eps);
        let weight_offset_f32: f32 = convert_scalar(weight_offset);
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
                    variance + broadcast_scalar(eps_f32, const_shape![1]),
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
                    mean_square + broadcast_scalar(eps_f32, const_shape![1]),
                    rounding::NearestEven,
                    ftz::Disabled,
                );
            let inv_rms = inv_rms_value
                .reshape(const_shape![1i32, 1i32])
                .broadcast(tile_shape);
            input_tile * inv_rms
        };
        let weight_tile: Tile<f32, { [1, BN] }> = if has_weight != 0 {
            let raw: Tile<f16, { [BN] }> = weight.partition(const_shape![BN]).load([0i32]);
            let raw: Tile<f32, { [BN] }> = convert_tile(raw);
            let raw = raw.reshape(tile_shape);
            if OP == 1 {
                raw + broadcast_scalar(weight_offset_f32, tile_shape)
            } else {
                raw
            }
        } else {
            one
        };
        let bias_tile: Tile<f32, { [1, BN] }> = if has_bias != 0 {
            let raw: Tile<f16, { [BN] }> = bias.partition(const_shape![BN]).load([0i32]);
            let raw: Tile<f32, { [BN] }> = convert_tile(raw);
            raw.reshape(tile_shape)
        } else {
            zero
        };
        let output_f32: Tile<f32, { [1, BN] }> =
            select(valid_cols, values * weight_tile + bias_tile, zero);
        let output: Tile<f16, { [1, BN] }> = convert_tile(output_f32);
        out.store(output);
    }

    #[cfg(feature = "dtype-bf16")]
    fn normalization_impl_bf16<const OP: i32, const BN: i32>(
        out: &mut Tensor<bf16, { [1, BN] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
        weight: &Tensor<bf16, { [-1] }>,
        bias: &Tensor<bf16, { [-1] }>,
        has_weight: i32,
        has_bias: i32,
        eps: bf16,
        weight_offset: bf16,
    ) {
        let tile_shape = const_shape![1i32, BN];
        let input_tile_raw: Tile<bf16, { [1, BN] }> = load_tile_like(input, out);
        let input_tile_raw: Tile<f32, { [1, BN] }> = convert_tile(input_tile_raw);
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
        let eps_f32: f32 = convert_scalar(eps);
        let weight_offset_f32: f32 = convert_scalar(weight_offset);
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
                    variance + broadcast_scalar(eps_f32, const_shape![1]),
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
                    mean_square + broadcast_scalar(eps_f32, const_shape![1]),
                    rounding::NearestEven,
                    ftz::Disabled,
                );
            let inv_rms = inv_rms_value
                .reshape(const_shape![1i32, 1i32])
                .broadcast(tile_shape);
            input_tile * inv_rms
        };
        let weight_tile: Tile<f32, { [1, BN] }> = if has_weight != 0 {
            let raw: Tile<bf16, { [BN] }> = weight.partition(const_shape![BN]).load([0i32]);
            let raw: Tile<f32, { [BN] }> = convert_tile(raw);
            let raw = raw.reshape(tile_shape);
            if OP == 1 {
                raw + broadcast_scalar(weight_offset_f32, tile_shape)
            } else {
                raw
            }
        } else {
            one
        };
        let bias_tile: Tile<f32, { [1, BN] }> = if has_bias != 0 {
            let raw: Tile<bf16, { [BN] }> = bias.partition(const_shape![BN]).load([0i32]);
            let raw: Tile<f32, { [BN] }> = convert_tile(raw);
            raw.reshape(tile_shape)
        } else {
            zero
        };
        let output_f32: Tile<f32, { [1, BN] }> =
            select(valid_cols, values * weight_tile + bias_tile, zero);
        let output: Tile<bf16, { [1, BN] }> = convert_tile(output_f32);
        out.store(output);
    }

    #[cutile::entry()]
    pub unsafe fn batch_norm_inference_f16(
        out: *mut f16,
        input: *mut f16,
        weight: *mut f16,
        bias: *mut f16,
        running_mean: *mut f16,
        running_var: *mut f16,
        feature_count: i32,
        spatial_len: i32,
        has_weight: i32,
        has_bias: i32,
        has_running_mean: i32,
        has_running_var: i32,
        eps: f16,
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

        let input_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f16, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f16, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f16, { [128] }> = input_ptrs.offset_tile(safe_offsets);
        let input_result: (Tile<f16, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(f16::from_f32(0.0)),
            None,
            Latency::<0>,
        );

        let zero = constant(0.0f32, tile_shape);
        let one = constant(1.0f32, tile_shape);
        let input_values: Tile<f32, { [128] }> = convert_tile(input_result.0);
        let mean = if has_running_mean != 0 {
            convert_tile(load_f16_parameter(
                running_mean,
                feature_index,
                mask,
                f16::from_f32(0.0),
            ))
        } else {
            zero
        };
        let var = if has_running_var != 0 {
            convert_tile(load_f16_parameter(
                running_var,
                feature_index,
                mask,
                f16::from_f32(1.0),
            ))
        } else {
            one
        };
        let scale = if has_weight != 0 {
            convert_tile(load_f16_parameter(
                weight,
                feature_index,
                mask,
                f16::from_f32(1.0),
            ))
        } else {
            one
        };
        let shift = if has_bias != 0 {
            convert_tile(load_f16_parameter(
                bias,
                feature_index,
                mask,
                f16::from_f32(0.0),
            ))
        } else {
            zero
        };
        let inv_std = one
            / sqrt(
                var + broadcast_scalar(convert_scalar(eps), tile_shape),
                rounding::NearestEven,
                ftz::Disabled,
            );
        let values = (input_values - mean) * inv_std * scale + shift;

        let out_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            convert_tile(values),
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn group_norm_f16(
        out: *mut f16,
        input: *mut f16,
        weight: *mut f16,
        bias: *mut f16,
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
                let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
                let safe_spatial_offsets = select(mask, spatial_offsets, zero_offsets);
                let values: Tile<f32, { [128] }> = convert_tile(load_f16_parameter_no_fill(
                    input,
                    broadcast_scalar(row_offset, tile_shape) + safe_spatial_offsets,
                    mask,
                ));
                let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
                let values = select(mask, values, zero_values);
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
            let scale: Tile<f32, { [128] }> =
                convert_tile(load_f16_parameter_no_fill(weight, channel_tile, full_mask));
            let shift: Tile<f32, { [128] }> =
                convert_tile(load_f16_parameter_no_fill(bias, channel_tile, full_mask));
            for chunk in 0i32..spatial_chunks {
                let spatial_offsets =
                    iota(tile_shape) + broadcast_scalar(chunk * 128i32, tile_shape);
                let mask = cmpi(
                    spatial_offsets,
                    broadcast_scalar(spatial_len, tile_shape),
                    predicate::LessThan,
                );
                let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
                let safe_spatial_offsets = select(mask, spatial_offsets, zero_offsets);
                let offsets = broadcast_scalar(row_offset, tile_shape) + spatial_offsets;
                let safe_offsets = broadcast_scalar(row_offset, tile_shape) + safe_spatial_offsets;
                let values: Tile<f32, { [128] }> =
                    convert_tile(load_f16_parameter_no_fill(input, safe_offsets, mask));
                let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
                let values = select(mask, values, zero_values);
                let normalized = (values - mean_values) * rstd_values * scale + shift;
                let normalized: Tile<f16, { [128] }> = convert_tile(normalized);
                store_f16_vector(out, offsets, normalized, mask);
            }
        }
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn group_norm_bf16(
        out: *mut bf16,
        input: *mut bf16,
        weight: *mut bf16,
        bias: *mut bf16,
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
                let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
                let safe_spatial_offsets = select(mask, spatial_offsets, zero_offsets);
                let values: Tile<f32, { [128] }> = convert_tile(load_bf16_parameter_no_fill(
                    input,
                    broadcast_scalar(row_offset, tile_shape) + safe_spatial_offsets,
                    mask,
                ));
                let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
                let values = select(mask, values, zero_values);
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
            let scale: Tile<f32, { [128] }> =
                convert_tile(load_bf16_parameter_no_fill(weight, channel_tile, full_mask));
            let shift: Tile<f32, { [128] }> =
                convert_tile(load_bf16_parameter_no_fill(bias, channel_tile, full_mask));
            for chunk in 0i32..spatial_chunks {
                let spatial_offsets =
                    iota(tile_shape) + broadcast_scalar(chunk * 128i32, tile_shape);
                let mask = cmpi(
                    spatial_offsets,
                    broadcast_scalar(spatial_len, tile_shape),
                    predicate::LessThan,
                );
                let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
                let safe_spatial_offsets = select(mask, spatial_offsets, zero_offsets);
                let offsets = broadcast_scalar(row_offset, tile_shape) + spatial_offsets;
                let safe_offsets = broadcast_scalar(row_offset, tile_shape) + safe_spatial_offsets;
                let values: Tile<f32, { [128] }> =
                    convert_tile(load_bf16_parameter_no_fill(input, safe_offsets, mask));
                let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
                let values = select(mask, values, zero_values);
                let normalized = (values - mean_values) * rstd_values * scale + shift;
                let normalized: Tile<bf16, { [128] }> = convert_tile(normalized);
                store_bf16_vector(out, offsets, normalized, mask);
            }
        }
    }

    fn load_f16_parameter(
        input: *mut f16,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: f16,
    ) -> Tile<f16, { [128] }> {
        let base: PointerTile<*mut f16, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*mut f16, { [1] }> = base.reshape(const_shape![1]);
        let ptrs: PointerTile<*mut f16, { [128] }> = base.broadcast(const_shape![128]);
        let ptrs: PointerTile<*mut f16, { [128] }> = ptrs.offset_tile(offsets);
        let result: (Tile<f16, { [128] }>, Token) = load_ptr_tko(
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

    fn load_f16_parameter_no_fill(
        input: *mut f16,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<f16, { [128] }> {
        let base: PointerTile<*mut f16, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*mut f16, { [1] }> = base.reshape(const_shape![1]);
        let ptrs: PointerTile<*mut f16, { [128] }> = base.broadcast(const_shape![128]);
        let ptrs: PointerTile<*mut f16, { [128] }> = ptrs.offset_tile(offsets);
        let result: (Tile<f16, { [128] }>, Token) = load_ptr_tko(
            ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        result.0
    }

    #[cfg(feature = "dtype-bf16")]
    fn load_bf16_parameter(
        input: *mut bf16,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: bf16,
    ) -> Tile<bf16, { [128] }> {
        let base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*mut bf16, { [1] }> = base.reshape(const_shape![1]);
        let ptrs: PointerTile<*mut bf16, { [128] }> = base.broadcast(const_shape![128]);
        let ptrs: PointerTile<*mut bf16, { [128] }> = ptrs.offset_tile(offsets);
        let result: (Tile<bf16, { [128] }>, Token) = load_ptr_tko(
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

    #[cfg(feature = "dtype-bf16")]
    fn load_bf16_parameter_no_fill(
        input: *mut bf16,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<bf16, { [128] }> {
        let base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*mut bf16, { [1] }> = base.reshape(const_shape![1]);
        let ptrs: PointerTile<*mut bf16, { [128] }> = base.broadcast(const_shape![128]);
        let ptrs: PointerTile<*mut bf16, { [128] }> = ptrs.offset_tile(offsets);
        let result: (Tile<bf16, { [128] }>, Token) = load_ptr_tko(
            ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        result.0
    }

    fn store_f16_vector(
        out: *mut f16,
        offsets: Tile<i32, { [128] }>,
        values: Tile<f16, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let out_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_base.broadcast(const_shape![128]);
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

    #[cfg(feature = "dtype-bf16")]
    fn store_bf16_vector(
        out: *mut bf16,
        offsets: Tile<i32, { [128] }>,
        values: Tile<bf16, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let out_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut bf16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut bf16, { [128] }> = out_base.broadcast(const_shape![128]);
        let out_ptrs: PointerTile<*mut bf16, { [128] }> = out_ptrs.offset_tile(offsets);
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
        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out);
        let base: PointerTile<*mut f32, { [1] }> = base.reshape(const_shape![1]);
        let offsets: Tile<i32, { [1] }> = broadcast_scalar(offset, const_shape![1]);
        let ptrs: PointerTile<*mut f32, { [1] }> = base.broadcast(const_shape![1]);
        let ptrs: PointerTile<*mut f32, { [1] }> = ptrs.offset_tile(offsets);
        store_ptr_tko(
            ptrs,
            value,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }
}

pub use kernels::*;
