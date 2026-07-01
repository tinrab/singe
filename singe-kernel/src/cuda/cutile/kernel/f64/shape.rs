#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn copy_f64(out: *mut f64, input: *mut f64, len: i32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_i32);

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(safe_offsets);
        let values: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            values.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn transpose_2d_f64(out: *mut f64, input: *mut f64, rows: i32, cols: i32, len: i32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_i32);

        let rows_tile = broadcast_scalar(rows, tile_shape);
        let cols_tile = broadcast_scalar(cols, tile_shape);
        let output_rows = safe_offsets / rows_tile;
        let output_cols = safe_offsets - output_rows * rows_tile;
        let input_offsets = output_cols * cols_tile + output_rows;

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(input_offsets);
        let values: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            values.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank3_f64(
        out: *mut f64,
        input: *mut f64,
        _batch: i32,
        rows: i32,
        cols: i32,
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
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_i32);

        let rows_tile = broadcast_scalar(rows, tile_shape);
        let cols_tile = broadcast_scalar(cols, tile_shape);
        let output_matrix_len = rows_tile * cols_tile;
        let batch_index = safe_offsets / output_matrix_len;
        let matrix_offset = safe_offsets - batch_index * output_matrix_len;
        let output_rows = matrix_offset / rows_tile;
        let output_cols = matrix_offset - output_rows * rows_tile;
        let input_offsets = batch_index * output_matrix_len + output_cols * cols_tile + output_rows;

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(input_offsets);
        let values: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            values.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank4_f64(
        out: *mut f64,
        input: *mut f64,
        _dim0: i32,
        dim1: i32,
        rows: i32,
        cols: i32,
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
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_i32);

        let dim1_tile = broadcast_scalar(dim1, tile_shape);
        let rows_tile = broadcast_scalar(rows, tile_shape);
        let cols_tile = broadcast_scalar(cols, tile_shape);
        let matrix_len = rows_tile * cols_tile;
        let dim1_matrix_len = dim1_tile * matrix_len;
        let index0 = safe_offsets / dim1_matrix_len;
        let remaining0 = safe_offsets - index0 * dim1_matrix_len;
        let index1 = remaining0 / matrix_len;
        let matrix_offset = remaining0 - index1 * matrix_len;
        let output_rows = matrix_offset / rows_tile;
        let output_cols = matrix_offset - output_rows * rows_tile;
        let input_offsets =
            index0 * dim1_matrix_len + index1 * matrix_len + output_cols * cols_tile + output_rows;

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(input_offsets);
        let values: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            values.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank2_f64(
        out: *mut f64,
        input: *mut f64,
        _dim0: i32,
        dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
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
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_i32);

        let dim1_tile = broadcast_scalar(dim1, tile_shape);
        let index0: Tile<i32, { [128] }> = safe_offsets / dim1_tile;
        let index1: Tile<i32, { [128] }> = safe_offsets - index0 * dim1_tile;
        let input_offsets = index0 * broadcast_scalar(input_stride0, tile_shape)
            + index1 * broadcast_scalar(input_stride1, tile_shape);

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(input_offsets);
        let values: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            values.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank2_f64(
        out: *mut f64,
        input: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        start0: i32,
        start1: i32,
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

        let dim1_tile = broadcast_scalar(output_dim1, tile_shape);
        let coord0: Tile<i32, { [128] }> = safe_offsets / dim1_tile;
        let coord1: Tile<i32, { [128] }> = safe_offsets - coord0 * dim1_tile;
        let source_offsets: Tile<i32, { [128] }> = (coord0 + broadcast_scalar(start0, tile_shape))
            * broadcast_scalar(input_stride0, tile_shape)
            + (coord1 + broadcast_scalar(start1, tile_shape))
                * broadcast_scalar(input_stride1, tile_shape);
        let source_offsets = select(mask, source_offsets, zero_offsets);

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(source_offsets);
        let result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            result.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank2_f64(
        out: *mut f64,
        input: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_value: f64,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let output_mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero_offsets);

        let dim1_tile = broadcast_scalar(output_dim1, tile_shape);
        let coord0: Tile<i32, { [128] }> = safe_offsets / dim1_tile;
        let coord1: Tile<i32, { [128] }> = safe_offsets - coord0 * dim1_tile;

        let pad0 = broadcast_scalar(pad_before0, tile_shape);
        let pad1 = broadcast_scalar(pad_before1, tile_shape);
        let input_dim0_end = broadcast_scalar(pad_before0 + input_dim0, tile_shape);
        let input_dim1_end = broadcast_scalar(pad_before1 + input_dim1, tile_shape);
        let valid0 = cmpi(coord0, pad0, predicate::GreaterThanOrEqual)
            & cmpi(coord0, input_dim0_end, predicate::LessThan);
        let valid1 = cmpi(coord1, pad1, predicate::GreaterThanOrEqual)
            & cmpi(coord1, input_dim1_end, predicate::LessThan);
        let input_mask = output_mask & valid0 & valid1;

        let source_offsets: Tile<i32, { [128] }> = (coord0 - pad0)
            * broadcast_scalar(input_stride0, tile_shape)
            + (coord1 - pad1) * broadcast_scalar(input_stride1, tile_shape);
        let source_offsets = select(input_mask, source_offsets, zero_offsets);

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(source_offsets);
        let result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(input_mask),
            Some(pad_value),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            result.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(output_mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank3_f64(
        out: *mut f64,
        input: *mut f64,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
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
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_i32);

        let dim1_dim2 = dim1 * dim2;
        let dim1_dim2_tile = broadcast_scalar(dim1_dim2, tile_shape);
        let dim2_tile = broadcast_scalar(dim2, tile_shape);
        let index0: Tile<i32, { [128] }> = safe_offsets / dim1_dim2_tile;
        let remaining0: Tile<i32, { [128] }> = safe_offsets - index0 * dim1_dim2_tile;
        let index1: Tile<i32, { [128] }> = remaining0 / dim2_tile;
        let index2: Tile<i32, { [128] }> = remaining0 - index1 * dim2_tile;
        let input_offsets = index0 * broadcast_scalar(input_stride0, tile_shape)
            + index1 * broadcast_scalar(input_stride1, tile_shape)
            + index2 * broadcast_scalar(input_stride2, tile_shape);

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(input_offsets);
        let values: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            values.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank3_f64(
        out: *mut f64,
        input: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        start0: i32,
        start1: i32,
        start2: i32,
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

        let dim1_dim2 = output_dim1 * output_dim2;
        let dim1_dim2_tile = broadcast_scalar(dim1_dim2, tile_shape);
        let dim2_tile = broadcast_scalar(output_dim2, tile_shape);
        let coord0: Tile<i32, { [128] }> = safe_offsets / dim1_dim2_tile;
        let remaining0: Tile<i32, { [128] }> = safe_offsets - coord0 * dim1_dim2_tile;
        let coord1: Tile<i32, { [128] }> = remaining0 / dim2_tile;
        let coord2: Tile<i32, { [128] }> = remaining0 - coord1 * dim2_tile;
        let source_offsets: Tile<i32, { [128] }> = (coord0 + broadcast_scalar(start0, tile_shape))
            * broadcast_scalar(input_stride0, tile_shape)
            + (coord1 + broadcast_scalar(start1, tile_shape))
                * broadcast_scalar(input_stride1, tile_shape)
            + (coord2 + broadcast_scalar(start2, tile_shape))
                * broadcast_scalar(input_stride2, tile_shape);
        let source_offsets = select(mask, source_offsets, zero_offsets);

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(source_offsets);
        let result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            result.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank3_f64(
        out: *mut f64,
        input: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_value: f64,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let output_mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero_offsets);

        let dim1_dim2 = output_dim1 * output_dim2;
        let dim1_dim2_tile = broadcast_scalar(dim1_dim2, tile_shape);
        let dim2_tile = broadcast_scalar(output_dim2, tile_shape);
        let coord0: Tile<i32, { [128] }> = safe_offsets / dim1_dim2_tile;
        let remaining0: Tile<i32, { [128] }> = safe_offsets - coord0 * dim1_dim2_tile;
        let coord1: Tile<i32, { [128] }> = remaining0 / dim2_tile;
        let coord2: Tile<i32, { [128] }> = remaining0 - coord1 * dim2_tile;

        let pad0 = broadcast_scalar(pad_before0, tile_shape);
        let pad1 = broadcast_scalar(pad_before1, tile_shape);
        let pad2 = broadcast_scalar(pad_before2, tile_shape);
        let input_dim0_end = broadcast_scalar(pad_before0 + input_dim0, tile_shape);
        let input_dim1_end = broadcast_scalar(pad_before1 + input_dim1, tile_shape);
        let input_dim2_end = broadcast_scalar(pad_before2 + input_dim2, tile_shape);
        let valid0 = cmpi(coord0, pad0, predicate::GreaterThanOrEqual)
            & cmpi(coord0, input_dim0_end, predicate::LessThan);
        let valid1 = cmpi(coord1, pad1, predicate::GreaterThanOrEqual)
            & cmpi(coord1, input_dim1_end, predicate::LessThan);
        let valid2 = cmpi(coord2, pad2, predicate::GreaterThanOrEqual)
            & cmpi(coord2, input_dim2_end, predicate::LessThan);
        let input_mask = output_mask & valid0 & valid1 & valid2;

        let source_offsets: Tile<i32, { [128] }> = (coord0 - pad0)
            * broadcast_scalar(input_stride0, tile_shape)
            + (coord1 - pad1) * broadcast_scalar(input_stride1, tile_shape)
            + (coord2 - pad2) * broadcast_scalar(input_stride2, tile_shape);
        let source_offsets = select(input_mask, source_offsets, zero_offsets);

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(source_offsets);
        let result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(input_mask),
            Some(pad_value),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            result.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(output_mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank4_f64(
        out: *mut f64,
        input: *mut f64,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
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

        let dim1_dim2_dim3 = dim1 * dim2 * dim3;
        let dim2_dim3 = dim2 * dim3;
        let dim1_dim2_dim3_tile = broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let dim2_dim3_tile = broadcast_scalar(dim2_dim3, tile_shape);
        let dim3_tile = broadcast_scalar(dim3, tile_shape);
        let index0: Tile<i32, { [128] }> = safe_offsets / dim1_dim2_dim3_tile;
        let remaining0: Tile<i32, { [128] }> = safe_offsets - index0 * dim1_dim2_dim3_tile;
        let index1: Tile<i32, { [128] }> = remaining0 / dim2_dim3_tile;
        let remaining1: Tile<i32, { [128] }> = remaining0 - index1 * dim2_dim3_tile;
        let index2: Tile<i32, { [128] }> = remaining1 / dim3_tile;
        let index3: Tile<i32, { [128] }> = remaining1 - index2 * dim3_tile;
        let source_offsets: Tile<i32, { [128] }> = index0
            * broadcast_scalar(input_stride0, tile_shape)
            + index1 * broadcast_scalar(input_stride1, tile_shape)
            + index2 * broadcast_scalar(input_stride2, tile_shape)
            + index3 * broadcast_scalar(input_stride3, tile_shape);

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(source_offsets);
        let result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            result.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank4_f64(
        out: *mut f64,
        input: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_before3: i32,
        pad_value: f64,
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
        let dim1_dim2_dim3_tile = broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let dim2_dim3_tile = broadcast_scalar(dim2_dim3, tile_shape);
        let dim3_tile = broadcast_scalar(output_dim3, tile_shape);
        let coord0: Tile<i32, { [128] }> = safe_offsets / dim1_dim2_dim3_tile;
        let remaining0: Tile<i32, { [128] }> = safe_offsets - coord0 * dim1_dim2_dim3_tile;
        let coord1: Tile<i32, { [128] }> = remaining0 / dim2_dim3_tile;
        let remaining1: Tile<i32, { [128] }> = remaining0 - coord1 * dim2_dim3_tile;
        let coord2: Tile<i32, { [128] }> = remaining1 / dim3_tile;
        let coord3: Tile<i32, { [128] }> = remaining1 - coord2 * dim3_tile;

        let pad0 = broadcast_scalar(pad_before0, tile_shape);
        let pad1 = broadcast_scalar(pad_before1, tile_shape);
        let pad2 = broadcast_scalar(pad_before2, tile_shape);
        let pad3 = broadcast_scalar(pad_before3, tile_shape);
        let input_dim0_end = broadcast_scalar(pad_before0 + input_dim0, tile_shape);
        let input_dim1_end = broadcast_scalar(pad_before1 + input_dim1, tile_shape);
        let input_dim2_end = broadcast_scalar(pad_before2 + input_dim2, tile_shape);
        let input_dim3_end = broadcast_scalar(pad_before3 + input_dim3, tile_shape);
        let valid0 = cmpi(coord0, pad0, predicate::GreaterThanOrEqual)
            & cmpi(coord0, input_dim0_end, predicate::LessThan);
        let valid1 = cmpi(coord1, pad1, predicate::GreaterThanOrEqual)
            & cmpi(coord1, input_dim1_end, predicate::LessThan);
        let valid2 = cmpi(coord2, pad2, predicate::GreaterThanOrEqual)
            & cmpi(coord2, input_dim2_end, predicate::LessThan);
        let valid3 = cmpi(coord3, pad3, predicate::GreaterThanOrEqual)
            & cmpi(coord3, input_dim3_end, predicate::LessThan);
        let input_mask = output_mask & valid0 & valid1 & valid2 & valid3;

        let source_offsets: Tile<i32, { [128] }> = (coord0 - pad0)
            * broadcast_scalar(input_stride0, tile_shape)
            + (coord1 - pad1) * broadcast_scalar(input_stride1, tile_shape)
            + (coord2 - pad2) * broadcast_scalar(input_stride2, tile_shape)
            + (coord3 - pad3) * broadcast_scalar(input_stride3, tile_shape);
        let source_offsets = select(input_mask, source_offsets, zero_offsets);

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(source_offsets);
        let result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(input_mask),
            Some(pad_value),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            result.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(output_mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank4_f64(
        out: *mut f64,
        input: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        start3: i32,
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

        let dim1_dim2_dim3 = output_dim1 * output_dim2 * output_dim3;
        let dim2_dim3 = output_dim2 * output_dim3;
        let dim1_dim2_dim3_tile = broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let dim2_dim3_tile = broadcast_scalar(dim2_dim3, tile_shape);
        let dim3_tile = broadcast_scalar(output_dim3, tile_shape);
        let coord0: Tile<i32, { [128] }> = safe_offsets / dim1_dim2_dim3_tile;
        let remaining0: Tile<i32, { [128] }> = safe_offsets - coord0 * dim1_dim2_dim3_tile;
        let coord1: Tile<i32, { [128] }> = remaining0 / dim2_dim3_tile;
        let remaining1: Tile<i32, { [128] }> = remaining0 - coord1 * dim2_dim3_tile;
        let coord2: Tile<i32, { [128] }> = remaining1 / dim3_tile;
        let coord3: Tile<i32, { [128] }> = remaining1 - coord2 * dim3_tile;

        let source_offsets: Tile<i32, { [128] }> = (coord0 + broadcast_scalar(start0, tile_shape))
            * broadcast_scalar(input_stride0, tile_shape)
            + (coord1 + broadcast_scalar(start1, tile_shape))
                * broadcast_scalar(input_stride1, tile_shape)
            + (coord2 + broadcast_scalar(start2, tile_shape))
                * broadcast_scalar(input_stride2, tile_shape)
            + (coord3 + broadcast_scalar(start3, tile_shape))
                * broadcast_scalar(input_stride3, tile_shape);
        let source_offsets = select(mask, source_offsets, zero_offsets);

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(source_offsets);
        let result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            result.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    unsafe fn concat_rank2_f64_impl<const AXIS: i32>(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let output_mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero_offsets);

        let dim1_tile = broadcast_scalar(output_dim1, tile_shape);
        let coord0: Tile<i32, { [128] }> = safe_offsets / dim1_tile;
        let coord1: Tile<i32, { [128] }> = safe_offsets - coord0 * dim1_tile;

        let lhs_axis_size = if AXIS == 0 { lhs_dim0 } else { lhs_dim1 };
        let axis_coord: Tile<i32, { [128] }> = if AXIS == 0 { coord0 } else { coord1 };
        let lhs_mask = output_mask
            & cmpi(
                axis_coord,
                broadcast_scalar(lhs_axis_size, tile_shape),
                predicate::LessThan,
            );
        let rhs_mask = output_mask
            & cmpi(
                axis_coord,
                broadcast_scalar(lhs_axis_size, tile_shape),
                predicate::GreaterThanOrEqual,
            );

        let rhs_coord0 = if AXIS == 0 {
            coord0 - broadcast_scalar(lhs_axis_size, tile_shape)
        } else {
            coord0
        };
        let rhs_coord1 = if AXIS == 1 {
            coord1 - broadcast_scalar(lhs_axis_size, tile_shape)
        } else {
            coord1
        };

        let lhs_offsets = coord0 * broadcast_scalar(lhs_stride0, tile_shape)
            + coord1 * broadcast_scalar(lhs_stride1, tile_shape);
        let rhs_offsets = rhs_coord0 * broadcast_scalar(rhs_stride0, tile_shape)
            + rhs_coord1 * broadcast_scalar(rhs_stride1, tile_shape);
        let lhs_offsets = select(lhs_mask, lhs_offsets, zero_offsets);
        let rhs_offsets = select(rhs_mask, rhs_offsets, zero_offsets);

        let lhs_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(lhs);
        let lhs_base: PointerTile<*mut f64, { [1] }> = lhs_base.reshape(const_shape![1]);
        let lhs_ptrs: PointerTile<*mut f64, { [128] }> = lhs_base.broadcast(tile_shape);
        let lhs_ptrs: PointerTile<*mut f64, { [128] }> = lhs_ptrs.offset_tile(lhs_offsets);
        let lhs_result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            lhs_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(lhs_mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let rhs_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(rhs);
        let rhs_base: PointerTile<*mut f64, { [1] }> = rhs_base.reshape(const_shape![1]);
        let rhs_ptrs: PointerTile<*mut f64, { [128] }> = rhs_base.broadcast(tile_shape);
        let rhs_ptrs: PointerTile<*mut f64, { [128] }> = rhs_ptrs.offset_tile(rhs_offsets);
        let rhs_result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            rhs_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(rhs_mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let values = select(lhs_mask, lhs_result.0, rhs_result.0);
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

    unsafe fn concat_rank3_f64_impl<const AXIS: i32>(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let output_mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero_offsets);

        let dim1_dim2 = output_dim1 * output_dim2;
        let dim1_dim2_tile = broadcast_scalar(dim1_dim2, tile_shape);
        let dim2_tile = broadcast_scalar(output_dim2, tile_shape);
        let coord0: Tile<i32, { [128] }> = safe_offsets / dim1_dim2_tile;
        let remaining0: Tile<i32, { [128] }> = safe_offsets - coord0 * dim1_dim2_tile;
        let coord1: Tile<i32, { [128] }> = remaining0 / dim2_tile;
        let coord2: Tile<i32, { [128] }> = remaining0 - coord1 * dim2_tile;

        let lhs_axis_size = if AXIS == 0 {
            lhs_dim0
        } else if AXIS == 1 {
            lhs_dim1
        } else {
            lhs_dim2
        };
        let axis_coord: Tile<i32, { [128] }> = if AXIS == 0 {
            coord0
        } else if AXIS == 1 {
            coord1
        } else {
            coord2
        };
        let lhs_mask = output_mask
            & cmpi(
                axis_coord,
                broadcast_scalar(lhs_axis_size, tile_shape),
                predicate::LessThan,
            );
        let rhs_mask = output_mask
            & cmpi(
                axis_coord,
                broadcast_scalar(lhs_axis_size, tile_shape),
                predicate::GreaterThanOrEqual,
            );

        let rhs_coord0 = if AXIS == 0 {
            coord0 - broadcast_scalar(lhs_axis_size, tile_shape)
        } else {
            coord0
        };
        let rhs_coord1 = if AXIS == 1 {
            coord1 - broadcast_scalar(lhs_axis_size, tile_shape)
        } else {
            coord1
        };
        let rhs_coord2 = if AXIS == 2 {
            coord2 - broadcast_scalar(lhs_axis_size, tile_shape)
        } else {
            coord2
        };

        let lhs_offsets = coord0 * broadcast_scalar(lhs_stride0, tile_shape)
            + coord1 * broadcast_scalar(lhs_stride1, tile_shape)
            + coord2 * broadcast_scalar(lhs_stride2, tile_shape);
        let rhs_offsets = rhs_coord0 * broadcast_scalar(rhs_stride0, tile_shape)
            + rhs_coord1 * broadcast_scalar(rhs_stride1, tile_shape)
            + rhs_coord2 * broadcast_scalar(rhs_stride2, tile_shape);
        let lhs_offsets = select(lhs_mask, lhs_offsets, zero_offsets);
        let rhs_offsets = select(rhs_mask, rhs_offsets, zero_offsets);

        let lhs_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(lhs);
        let lhs_base: PointerTile<*mut f64, { [1] }> = lhs_base.reshape(const_shape![1]);
        let lhs_ptrs: PointerTile<*mut f64, { [128] }> = lhs_base.broadcast(tile_shape);
        let lhs_ptrs: PointerTile<*mut f64, { [128] }> = lhs_ptrs.offset_tile(lhs_offsets);
        let lhs_result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            lhs_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(lhs_mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let rhs_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(rhs);
        let rhs_base: PointerTile<*mut f64, { [1] }> = rhs_base.reshape(const_shape![1]);
        let rhs_ptrs: PointerTile<*mut f64, { [128] }> = rhs_base.broadcast(tile_shape);
        let rhs_ptrs: PointerTile<*mut f64, { [128] }> = rhs_ptrs.offset_tile(rhs_offsets);
        let rhs_result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            rhs_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(rhs_mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let values = select(lhs_mask, lhs_result.0, rhs_result.0);
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

    #[cutile::entry()]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn concat_rank2_f64_axis0(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_f64_impl::<0>(
            out,
            lhs,
            rhs,
            output_dim0,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len,
        );
    }

    #[cutile::entry()]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn concat_rank2_f64_axis1(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_f64_impl::<1>(
            out,
            lhs,
            rhs,
            output_dim0,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len,
        );
    }

    #[cutile::entry()]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn concat_rank3_f64_axis0(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_f64_impl::<0>(
            out,
            lhs,
            rhs,
            output_dim0,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len,
        );
    }

    #[cutile::entry()]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn concat_rank3_f64_axis1(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_f64_impl::<1>(
            out,
            lhs,
            rhs,
            output_dim0,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len,
        );
    }

    #[cutile::entry()]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn concat_rank3_f64_axis2(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_f64_impl::<2>(
            out,
            lhs,
            rhs,
            output_dim0,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len,
        );
    }

    unsafe fn concat_rank4_f64_impl<const AXIS: i32>(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
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
        let dim1_dim2_dim3_tile = broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let dim2_dim3_tile = broadcast_scalar(dim2_dim3, tile_shape);
        let dim3_tile = broadcast_scalar(output_dim3, tile_shape);
        let coord0: Tile<i32, { [128] }> = safe_offsets / dim1_dim2_dim3_tile;
        let remaining0: Tile<i32, { [128] }> = safe_offsets - coord0 * dim1_dim2_dim3_tile;
        let coord1: Tile<i32, { [128] }> = remaining0 / dim2_dim3_tile;
        let remaining1: Tile<i32, { [128] }> = remaining0 - coord1 * dim2_dim3_tile;
        let coord2: Tile<i32, { [128] }> = remaining1 / dim3_tile;
        let coord3: Tile<i32, { [128] }> = remaining1 - coord2 * dim3_tile;

        let lhs_axis_size = if AXIS == 0 {
            lhs_dim0
        } else if AXIS == 1 {
            lhs_dim1
        } else if AXIS == 2 {
            lhs_dim2
        } else {
            lhs_dim3
        };
        let axis_coord: Tile<i32, { [128] }> = if AXIS == 0 {
            coord0
        } else if AXIS == 1 {
            coord1
        } else if AXIS == 2 {
            coord2
        } else {
            coord3
        };
        let lhs_mask = output_mask
            & cmpi(
                axis_coord,
                broadcast_scalar(lhs_axis_size, tile_shape),
                predicate::LessThan,
            );
        let rhs_mask = output_mask
            & cmpi(
                axis_coord,
                broadcast_scalar(lhs_axis_size, tile_shape),
                predicate::GreaterThanOrEqual,
            );

        let axis_offset = broadcast_scalar(lhs_axis_size, tile_shape);
        let rhs_coord0 = if AXIS == 0 {
            coord0 - axis_offset
        } else {
            coord0
        };
        let rhs_coord1 = if AXIS == 1 {
            coord1 - axis_offset
        } else {
            coord1
        };
        let rhs_coord2 = if AXIS == 2 {
            coord2 - axis_offset
        } else {
            coord2
        };
        let rhs_coord3 = if AXIS == 3 {
            coord3 - axis_offset
        } else {
            coord3
        };

        let lhs_offsets: Tile<i32, { [128] }> = coord0 * broadcast_scalar(lhs_stride0, tile_shape)
            + coord1 * broadcast_scalar(lhs_stride1, tile_shape)
            + coord2 * broadcast_scalar(lhs_stride2, tile_shape)
            + coord3 * broadcast_scalar(lhs_stride3, tile_shape);
        let rhs_offsets: Tile<i32, { [128] }> = rhs_coord0
            * broadcast_scalar(rhs_stride0, tile_shape)
            + rhs_coord1 * broadcast_scalar(rhs_stride1, tile_shape)
            + rhs_coord2 * broadcast_scalar(rhs_stride2, tile_shape)
            + rhs_coord3 * broadcast_scalar(rhs_stride3, tile_shape);
        let lhs_offsets = select(lhs_mask, lhs_offsets, zero_offsets);
        let rhs_offsets = select(rhs_mask, rhs_offsets, zero_offsets);

        let lhs_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(lhs);
        let lhs_base: PointerTile<*mut f64, { [1] }> = lhs_base.reshape(const_shape![1]);
        let lhs_ptrs: PointerTile<*mut f64, { [128] }> = lhs_base.broadcast(tile_shape);
        let lhs_ptrs: PointerTile<*mut f64, { [128] }> = lhs_ptrs.offset_tile(lhs_offsets);
        let lhs_result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            lhs_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(lhs_mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let rhs_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(rhs);
        let rhs_base: PointerTile<*mut f64, { [1] }> = rhs_base.reshape(const_shape![1]);
        let rhs_ptrs: PointerTile<*mut f64, { [128] }> = rhs_base.broadcast(tile_shape);
        let rhs_ptrs: PointerTile<*mut f64, { [128] }> = rhs_ptrs.offset_tile(rhs_offsets);
        let rhs_result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            rhs_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(rhs_mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );
        let values = select(lhs_mask, lhs_result.0, rhs_result.0);

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

    #[cutile::entry()]
    pub unsafe fn concat_rank4_f64_axis0(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        unsafe {
            concat_rank4_f64_impl::<0>(
                out,
                lhs,
                rhs,
                output_dim0,
                output_dim1,
                output_dim2,
                output_dim3,
                lhs_dim0,
                lhs_dim1,
                lhs_dim2,
                lhs_dim3,
                lhs_stride0,
                lhs_stride1,
                lhs_stride2,
                lhs_stride3,
                rhs_stride0,
                rhs_stride1,
                rhs_stride2,
                rhs_stride3,
                len,
            )
        };
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_f64_axis1(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        unsafe {
            concat_rank4_f64_impl::<1>(
                out,
                lhs,
                rhs,
                output_dim0,
                output_dim1,
                output_dim2,
                output_dim3,
                lhs_dim0,
                lhs_dim1,
                lhs_dim2,
                lhs_dim3,
                lhs_stride0,
                lhs_stride1,
                lhs_stride2,
                lhs_stride3,
                rhs_stride0,
                rhs_stride1,
                rhs_stride2,
                rhs_stride3,
                len,
            )
        };
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_f64_axis2(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        unsafe {
            concat_rank4_f64_impl::<2>(
                out,
                lhs,
                rhs,
                output_dim0,
                output_dim1,
                output_dim2,
                output_dim3,
                lhs_dim0,
                lhs_dim1,
                lhs_dim2,
                lhs_dim3,
                lhs_stride0,
                lhs_stride1,
                lhs_stride2,
                lhs_stride3,
                rhs_stride0,
                rhs_stride1,
                rhs_stride2,
                rhs_stride3,
                len,
            )
        };
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_f64_axis3(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        unsafe {
            concat_rank4_f64_impl::<3>(
                out,
                lhs,
                rhs,
                output_dim0,
                output_dim1,
                output_dim2,
                output_dim3,
                lhs_dim0,
                lhs_dim1,
                lhs_dim2,
                lhs_dim3,
                lhs_stride0,
                lhs_stride1,
                lhs_stride2,
                lhs_stride3,
                rhs_stride0,
                rhs_stride1,
                rhs_stride2,
                rhs_stride3,
                len,
            )
        };
    }

    #[cutile::entry()]
    pub unsafe fn repeat_kv_rank4_f64(
        out: *mut f64,
        input: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        repeats: i32,
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
        let input_head: Tile<i32, { [128] }> = coord1 / broadcast_scalar(repeats, tile_shape);

        let source_offsets: Tile<i32, { [128] }> = coord0
            * broadcast_scalar(input_stride0, tile_shape)
            + input_head * broadcast_scalar(input_stride1, tile_shape)
            + coord2 * broadcast_scalar(input_stride2, tile_shape)
            + coord3 * broadcast_scalar(input_stride3, tile_shape);

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(source_offsets);
        let result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            result.0,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }
}

pub use kernels::*;
