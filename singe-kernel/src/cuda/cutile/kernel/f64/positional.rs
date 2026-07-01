#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_f64(
        out: *mut f64,
        input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let len_tile = broadcast_scalar(len, tile_shape);
        let output_mask = cmpi(offsets, len_tile, predicate::LessThan);
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero_offsets);

        let dim1_dim2_dim3 = output_dim1 * output_dim2 * output_dim3;
        let dim2_dim3 = output_dim2 * output_dim3;
        let coord0: Tile<i32, { [128] }> =
            safe_offsets / broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let remaining0: Tile<i32, { [128] }> =
            safe_offsets - coord0 * broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let coord1: Tile<i32, { [128] }> = remaining0 / broadcast_scalar(dim2_dim3, tile_shape);
        let remaining1: Tile<i32, { [128] }> =
            remaining0 - coord1 * broadcast_scalar(dim2_dim3, tile_shape);
        let coord2: Tile<i32, { [128] }> = remaining1 / broadcast_scalar(output_dim3, tile_shape);
        let coord3: Tile<i32, { [128] }> =
            remaining1 - coord2 * broadcast_scalar(output_dim3, tile_shape);

        let current_offsets: Tile<i32, { [128] }> = coord0
            * broadcast_scalar(input_stride0, tile_shape)
            + coord1 * broadcast_scalar(input_stride1, tile_shape)
            + coord2 * broadcast_scalar(input_stride2, tile_shape)
            + coord3 * broadcast_scalar(input_stride3, tile_shape);
        let pair: Tile<i32, { [128] }> = coord3 / broadcast_scalar(2i32, tile_shape);
        let even_coord3: Tile<i32, { [128] }> = pair * broadcast_scalar(2i32, tile_shape);
        let odd_coord3: Tile<i32, { [128] }> = even_coord3 + broadcast_scalar(1i32, tile_shape);
        let in_rotary = output_mask
            & cmpi(
                pair,
                broadcast_scalar(rotary_pairs, tile_shape),
                predicate::LessThan,
            );
        let is_even = cmpi(coord3, even_coord3, predicate::Equal);
        let even_offsets: Tile<i32, { [128] }> = coord0
            * broadcast_scalar(input_stride0, tile_shape)
            + coord1 * broadcast_scalar(input_stride1, tile_shape)
            + coord2 * broadcast_scalar(input_stride2, tile_shape)
            + even_coord3 * broadcast_scalar(input_stride3, tile_shape);
        let odd_offsets: Tile<i32, { [128] }> = coord0
            * broadcast_scalar(input_stride0, tile_shape)
            + coord1 * broadcast_scalar(input_stride1, tile_shape)
            + coord2 * broadcast_scalar(input_stride2, tile_shape)
            + odd_coord3 * broadcast_scalar(input_stride3, tile_shape);
        let trig_offsets: Tile<i32, { [128] }> = coord2 * broadcast_scalar(cos_stride0, tile_shape)
            + pair * broadcast_scalar(cos_stride1, tile_shape);
        let sin_offsets: Tile<i32, { [128] }> = coord2 * broadcast_scalar(sin_stride0, tile_shape)
            + pair * broadcast_scalar(sin_stride1, tile_shape);

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let current_ptrs: PointerTile<*mut f64, { [128] }> = input_base
            .broadcast(tile_shape)
            .offset_tile(select(output_mask, current_offsets, zero_offsets));
        let current_result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            current_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(output_mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );
        let even_ptrs: PointerTile<*mut f64, { [128] }> = input_base
            .broadcast(tile_shape)
            .offset_tile(select(in_rotary, even_offsets, zero_offsets));
        let even_result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            even_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(in_rotary),
            Some(0.0f64),
            None,
            Latency::<0>,
        );
        let odd_ptrs: PointerTile<*mut f64, { [128] }> = input_base
            .broadcast(tile_shape)
            .offset_tile(select(in_rotary, odd_offsets, zero_offsets));
        let odd_result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            odd_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(in_rotary),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let cos_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(cos);
        let cos_base: PointerTile<*mut f64, { [1] }> = cos_base.reshape(const_shape![1]);
        let cos_ptrs: PointerTile<*mut f64, { [128] }> = cos_base
            .broadcast(tile_shape)
            .offset_tile(select(in_rotary, trig_offsets, zero_offsets));
        let cos_result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            cos_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(in_rotary),
            Some(1.0f64),
            None,
            Latency::<0>,
        );
        let sin_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(sin);
        let sin_base: PointerTile<*mut f64, { [1] }> = sin_base.reshape(const_shape![1]);
        let sin_ptrs: PointerTile<*mut f64, { [128] }> = sin_base
            .broadcast(tile_shape)
            .offset_tile(select(in_rotary, sin_offsets, zero_offsets));
        let sin_result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            sin_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(in_rotary),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let even_rotated = even_result.0 * cos_result.0 - odd_result.0 * sin_result.0;
        let odd_rotated = odd_result.0 * cos_result.0 + even_result.0 * sin_result.0;
        let rotated = select(is_even, even_rotated, odd_rotated);
        let values = select(in_rotary, rotated, current_result.0);

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
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
}

pub use kernels::*;
