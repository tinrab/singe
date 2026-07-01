#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn max_pool2d_f16(
        out: *mut f16,
        input: *mut f16,
        _batch: i32,
        channels: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
        kernel_height: i32,
        kernel_width: i32,
        stride_height: i32,
        stride_width: i32,
        pad_height_start: i32,
        pad_width_start: i32,
        dilation_height: i32,
        dilation_width: i32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let len_tile = broadcast_scalar(len, tile_shape);
        let output_mask = cmpi(offsets, len_tile, predicate::LessThan);
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero_i32);

        let chw_out = channels * output_height * output_width;
        let hw_out = output_height * output_width;
        let n: Tile<i32, { [128] }> = safe_offsets / broadcast_scalar(chw_out, tile_shape);
        let rem_n: Tile<i32, { [128] }> = safe_offsets - n * broadcast_scalar(chw_out, tile_shape);
        let c: Tile<i32, { [128] }> = rem_n / broadcast_scalar(hw_out, tile_shape);
        let rem_c: Tile<i32, { [128] }> = rem_n - c * broadcast_scalar(hw_out, tile_shape);
        let oh: Tile<i32, { [128] }> = rem_c / broadcast_scalar(output_width, tile_shape);
        let ow: Tile<i32, { [128] }> = rem_c - oh * broadcast_scalar(output_width, tile_shape);

        let mut values: Tile<f16, { [128] }> = constant(f16::NEG_INFINITY, tile_shape);
        for kh in 0i32..kernel_height {
            for kw in 0i32..kernel_width {
                let ih: Tile<i32, { [128] }> = oh * broadcast_scalar(stride_height, tile_shape)
                    + broadcast_scalar(kh * dilation_height - pad_height_start, tile_shape);
                let iw: Tile<i32, { [128] }> = ow * broadcast_scalar(stride_width, tile_shape)
                    + broadcast_scalar(kw * dilation_width - pad_width_start, tile_shape);
                let valid_h = cmpi(ih, zero_i32, predicate::GreaterThanOrEqual)
                    & cmpi(
                        ih,
                        broadcast_scalar(input_height, tile_shape),
                        predicate::LessThan,
                    );
                let valid_w = cmpi(iw, zero_i32, predicate::GreaterThanOrEqual)
                    & cmpi(
                        iw,
                        broadcast_scalar(input_width, tile_shape),
                        predicate::LessThan,
                    );
                let valid = output_mask & valid_h & valid_w;
                let source_offsets: Tile<i32, { [128] }> =
                    n * broadcast_scalar(channels * input_height * input_width, tile_shape)
                        + c * broadcast_scalar(input_height * input_width, tile_shape)
                        + ih * broadcast_scalar(input_width, tile_shape)
                        + iw;
                let source_offsets = select(valid, source_offsets, zero_i32);
                let input_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(input);
                let input_base: PointerTile<*mut f16, { [1] }> =
                    input_base.reshape(const_shape![1]);
                let input_ptrs: PointerTile<*mut f16, { [128] }> = input_base.broadcast(tile_shape);
                let input_ptrs: PointerTile<*mut f16, { [128] }> =
                    input_ptrs.offset_tile(source_offsets);
                let result: (Tile<f16, { [128] }>, Token) = load_ptr_tko(
                    input_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(valid),
                    Some(f16::NEG_INFINITY),
                    None,
                    Latency::<0>,
                );
                values = maxf(values, result.0, nan::Enabled, ftz::Disabled);
            }
        }

        let out_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(output_mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn avg_pool2d_f16(
        out: *mut f16,
        input: *mut f16,
        _batch: i32,
        channels: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
        kernel_height: i32,
        kernel_width: i32,
        stride_height: i32,
        stride_width: i32,
        pad_height_start: i32,
        pad_height_end: i32,
        pad_width_start: i32,
        pad_width_end: i32,
        dilation_height: i32,
        dilation_width: i32,
        count_include_pad: i32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let len_tile = broadcast_scalar(len, tile_shape);
        let output_mask = cmpi(offsets, len_tile, predicate::LessThan);
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero_i32);

        let chw_out = channels * output_height * output_width;
        let hw_out = output_height * output_width;
        let n: Tile<i32, { [128] }> = safe_offsets / broadcast_scalar(chw_out, tile_shape);
        let rem_n: Tile<i32, { [128] }> = safe_offsets - n * broadcast_scalar(chw_out, tile_shape);
        let c: Tile<i32, { [128] }> = rem_n / broadcast_scalar(hw_out, tile_shape);
        let rem_c: Tile<i32, { [128] }> = rem_n - c * broadcast_scalar(hw_out, tile_shape);
        let oh: Tile<i32, { [128] }> = rem_c / broadcast_scalar(output_width, tile_shape);
        let ow: Tile<i32, { [128] }> = rem_c - oh * broadcast_scalar(output_width, tile_shape);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut count: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut padded_count: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let zero_f32: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one_f32: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        for kh in 0i32..kernel_height {
            for kw in 0i32..kernel_width {
                let ih: Tile<i32, { [128] }> = oh * broadcast_scalar(stride_height, tile_shape)
                    + broadcast_scalar(kh * dilation_height - pad_height_start, tile_shape);
                let iw: Tile<i32, { [128] }> = ow * broadcast_scalar(stride_width, tile_shape)
                    + broadcast_scalar(kw * dilation_width - pad_width_start, tile_shape);
                let valid_h = cmpi(ih, zero_i32, predicate::GreaterThanOrEqual)
                    & cmpi(
                        ih,
                        broadcast_scalar(input_height, tile_shape),
                        predicate::LessThan,
                    );
                let valid_w = cmpi(iw, zero_i32, predicate::GreaterThanOrEqual)
                    & cmpi(
                        iw,
                        broadcast_scalar(input_width, tile_shape),
                        predicate::LessThan,
                    );
                let valid = output_mask & valid_h & valid_w;
                let padded_valid_h = cmpi(
                    ih + broadcast_scalar(pad_height_start, tile_shape),
                    zero_i32,
                    predicate::GreaterThanOrEqual,
                ) & cmpi(
                    ih,
                    broadcast_scalar(input_height + pad_height_end, tile_shape),
                    predicate::LessThan,
                );
                let padded_valid_w = cmpi(
                    iw + broadcast_scalar(pad_width_start, tile_shape),
                    zero_i32,
                    predicate::GreaterThanOrEqual,
                ) & cmpi(
                    iw,
                    broadcast_scalar(input_width + pad_width_end, tile_shape),
                    predicate::LessThan,
                );
                let padded_valid = output_mask & padded_valid_h & padded_valid_w;
                let source_offsets: Tile<i32, { [128] }> =
                    n * broadcast_scalar(channels * input_height * input_width, tile_shape)
                        + c * broadcast_scalar(input_height * input_width, tile_shape)
                        + ih * broadcast_scalar(input_width, tile_shape)
                        + iw;
                let source_offsets = select(valid, source_offsets, zero_i32);
                let input_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(input);
                let input_base: PointerTile<*mut f16, { [1] }> =
                    input_base.reshape(const_shape![1]);
                let input_ptrs: PointerTile<*mut f16, { [128] }> = input_base.broadcast(tile_shape);
                let input_ptrs: PointerTile<*mut f16, { [128] }> =
                    input_ptrs.offset_tile(source_offsets);
                let result: (Tile<f16, { [128] }>, Token) = load_ptr_tko(
                    input_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(valid),
                    Some(f16::from_f32(0.0)),
                    None,
                    Latency::<0>,
                );
                let values: Tile<f32, { [128] }> = convert_tile(result.0);
                sum = sum + values;
                count = count + select(valid, one_f32, zero_f32);
                padded_count = padded_count + select(padded_valid, one_f32, zero_f32);
            }
        }
        let denominator = if count_include_pad != 0 {
            padded_count
        } else {
            count
        };
        let has_denominator = cmpf(
            denominator,
            zero_f32,
            predicate::GreaterThan,
            cmp_ordering::Ordered,
        );
        let values = select(has_denominator, sum / denominator, zero_f32);
        let out_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            convert_tile(values),
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(output_mask),
            None,
            Latency::<0>,
        );
    }
}

pub use kernels::*;
