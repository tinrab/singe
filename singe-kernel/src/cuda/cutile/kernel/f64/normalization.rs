#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn layer_norm_f64<const BN: i32>(
        out: &mut Tensor<f64, { [1, BN] }>,
        input: &Tensor<f64, { [-1, -1] }>,
        weight: &Tensor<f64, { [-1] }>,
        bias: &Tensor<f64, { [-1] }>,
        has_weight: i32,
        has_bias: i32,
        eps: f64,
        weight_offset: f64,
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
    pub fn rms_norm_f64<const BN: i32>(
        out: &mut Tensor<f64, { [1, BN] }>,
        input: &Tensor<f64, { [-1, -1] }>,
        weight: &Tensor<f64, { [-1] }>,
        bias: &Tensor<f64, { [-1] }>,
        has_weight: i32,
        has_bias: i32,
        eps: f64,
        weight_offset: f64,
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

    fn normalization_impl<const OP: i32, const BN: i32>(
        out: &mut Tensor<f64, { [1, BN] }>,
        input: &Tensor<f64, { [-1, -1] }>,
        weight: &Tensor<f64, { [-1] }>,
        bias: &Tensor<f64, { [-1] }>,
        has_weight: i32,
        has_bias: i32,
        eps: f64,
        weight_offset: f64,
    ) {
        let tile_shape = const_shape![1i32, BN];
        let input_tile_raw: Tile<f64, { [1, BN] }> = load_tile_like(input, out);
        let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let col_iota = col_iota.reshape(tile_shape);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let cols_tile = broadcast_scalar(input_shape[1], tile_shape);
        let valid_cols = cmpi(col_iota, cols_tile, predicate::LessThan);
        let zero = constant(0.0f64, tile_shape);
        let one = constant(1.0f64, tile_shape);
        let one_row: Tile<f64, { [1] }> = constant(1.0f64, const_shape![1]);
        let input_tile = select(valid_cols, input_tile_raw, zero);
        let cols: f64 = convert_scalar(input_shape[1]);
        let divisor = broadcast_scalar(cols, const_shape![1]);
        let values = if OP == 0 {
            let mean_sum: Tile<f64, { [1] }> = reduce_sum(input_tile, 1i32);
            let mean = (mean_sum / divisor)
                .reshape(const_shape![1i32, 1i32])
                .broadcast(tile_shape);
            let diff = select(valid_cols, input_tile - mean, zero);
            let variance_sum: Tile<f64, { [1] }> = reduce_sum(diff * diff, 1i32);
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
            let mean_square_sum: Tile<f64, { [1] }> = reduce_sum(input_tile * input_tile, 1i32);
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
        let weight_tile: Tile<f64, { [1, BN] }> = if has_weight != 0 {
            let raw: Tile<f64, { [BN] }> = weight.partition(const_shape![BN]).load([0i32]);
            let raw = raw.reshape(tile_shape);
            if OP == 1 {
                raw + broadcast_scalar(weight_offset, tile_shape)
            } else {
                raw
            }
        } else {
            one
        };
        let bias_tile: Tile<f64, { [1, BN] }> = if has_bias != 0 {
            let raw: Tile<f64, { [BN] }> = bias.partition(const_shape![BN]).load([0i32]);
            raw.reshape(tile_shape)
        } else {
            zero
        };
        out.store(select(valid_cols, values * weight_tile + bias_tile, zero));
    }

    #[cutile::entry()]
    pub unsafe fn batch_norm_inference_f64(
        out: *mut f64,
        input: *mut f64,
        weight: *mut f64,
        bias: *mut f64,
        running_mean: *mut f64,
        running_var: *mut f64,
        feature_count: i32,
        spatial_len: i32,
        has_weight: i32,
        has_bias: i32,
        has_running_mean: i32,
        has_running_var: i32,
        eps: f64,
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

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(safe_offsets);
        let input_result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let zero = constant(0.0f64, tile_shape);
        let one = constant(1.0f64, tile_shape);
        let mean = if has_running_mean != 0 {
            load_f64_parameter(running_mean, feature_index, mask, 0.0f64)
        } else {
            zero
        };
        let var = if has_running_var != 0 {
            load_f64_parameter(running_var, feature_index, mask, 1.0f64)
        } else {
            one
        };
        let scale = if has_weight != 0 {
            load_f64_parameter(weight, feature_index, mask, 1.0f64)
        } else {
            one
        };
        let shift = if has_bias != 0 {
            load_f64_parameter(bias, feature_index, mask, 0.0f64)
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

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
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

    fn load_f64_parameter(
        input: *mut f64,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: f64,
    ) -> Tile<f64, { [128] }> {
        let base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*mut f64, { [1] }> = base.reshape(const_shape![1]);
        let ptrs: PointerTile<*mut f64, { [128] }> = base.broadcast(const_shape![128]);
        let ptrs: PointerTile<*mut f64, { [128] }> = ptrs.offset_tile(offsets);
        let result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
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
}

pub use kernels::*;
