#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn gather_rows_f64(
        out: *mut f64,
        input: *mut f64,
        row_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let len_tile = broadcast_scalar(len, tile_shape);
        let mask = cmpi(offsets, len_tile, predicate::LessThan);
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_i32);

        let cols_tile = broadcast_scalar(cols, tile_shape);
        let output_rows: Tile<i32, { [128] }> = safe_offsets / cols_tile;
        let columns: Tile<i32, { [128] }> = safe_offsets - output_rows * cols_tile;

        let indices_base: PointerTile<*mut u32, { [] }> = pointer_to_tile(row_indices);
        let indices_base: PointerTile<*mut u32, { [1] }> = indices_base.reshape(const_shape![1]);
        let index_ptrs: PointerTile<*mut u32, { [128] }> = indices_base.broadcast(tile_shape);
        let index_ptrs: PointerTile<*mut u32, { [128] }> = index_ptrs.offset_tile(output_rows);
        let index_result: (Tile<u32, { [128] }>, Token) = load_ptr_tko(
            index_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u32),
            None,
            Latency::<0>,
        );
        let source_rows: Tile<i32, { [128] }> = bitcast(index_result.0);
        let source_offsets: Tile<i32, { [128] }> = source_rows * cols_tile + columns;

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(source_offsets);
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
    pub unsafe fn gather_f64(out: *mut f64, input: *mut f64, indices: *mut u32, len: i32) {
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

        let indices_base: PointerTile<*mut u32, { [] }> = pointer_to_tile(indices);
        let indices_base: PointerTile<*mut u32, { [1] }> = indices_base.reshape(const_shape![1]);
        let index_ptrs: PointerTile<*mut u32, { [128] }> = indices_base.broadcast(tile_shape);
        let index_ptrs: PointerTile<*mut u32, { [128] }> = index_ptrs.offset_tile(safe_offsets);
        let index_result: (Tile<u32, { [128] }>, Token) = load_ptr_tko(
            index_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u32),
            None,
            Latency::<0>,
        );
        let source_offsets: Tile<i32, { [128] }> = bitcast(index_result.0);

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(source_offsets);
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
    pub unsafe fn gather_row_f64(out: *mut f64, input: *mut f64, row_index: i32, cols: i32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let cols_tile = broadcast_scalar(cols, tile_shape);
        let mask = cmpi(offsets, cols_tile, predicate::LessThan);
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_i32);
        let source_offsets = broadcast_scalar(row_index * cols, tile_shape) + safe_offsets;

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(source_offsets);
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
    pub unsafe fn scatter_rows_f64(
        out: *mut f64,
        input: *mut f64,
        row_indices: *mut u32,
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

        let cols_tile = broadcast_scalar(cols, tile_shape);
        let source_rows: Tile<i32, { [128] }> = safe_offsets / cols_tile;
        let columns: Tile<i32, { [128] }> = safe_offsets - source_rows * cols_tile;

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

        let indices_base: PointerTile<*mut u32, { [] }> = pointer_to_tile(row_indices);
        let indices_base: PointerTile<*mut u32, { [1] }> = indices_base.reshape(const_shape![1]);
        let index_ptrs: PointerTile<*mut u32, { [128] }> = indices_base.broadcast(tile_shape);
        let index_ptrs: PointerTile<*mut u32, { [128] }> = index_ptrs.offset_tile(source_rows);
        let index_result: (Tile<u32, { [128] }>, Token) = load_ptr_tko(
            index_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u32),
            None,
            Latency::<0>,
        );
        let dest_rows: Tile<i32, { [128] }> = bitcast(index_result.0);
        let dest_offsets = dest_rows * cols_tile + columns;

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(dest_offsets);
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
    pub unsafe fn scatter_f64(out: *mut f64, input: *mut f64, indices: *mut u32, len: i32) {
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

        let indices_base: PointerTile<*mut u32, { [] }> = pointer_to_tile(indices);
        let indices_base: PointerTile<*mut u32, { [1] }> = indices_base.reshape(const_shape![1]);
        let index_ptrs: PointerTile<*mut u32, { [128] }> = indices_base.broadcast(tile_shape);
        let index_ptrs: PointerTile<*mut u32, { [128] }> = index_ptrs.offset_tile(safe_offsets);
        let index_result: (Tile<u32, { [128] }>, Token) = load_ptr_tko(
            index_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u32),
            None,
            Latency::<0>,
        );
        let dest_offsets: Tile<i32, { [128] }> = bitcast(index_result.0);

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(dest_offsets);
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
    pub unsafe fn copy_indexed_rows_f64(
        out: *mut f64,
        input: *mut f64,
        source_indices: *mut u32,
        output_indices: *mut u32,
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

        let cols_tile = broadcast_scalar(cols, tile_shape);
        let logical_rows: Tile<i32, { [128] }> = safe_offsets / cols_tile;
        let columns: Tile<i32, { [128] }> = safe_offsets - logical_rows * cols_tile;

        let source_base: PointerTile<*mut u32, { [] }> = pointer_to_tile(source_indices);
        let source_base: PointerTile<*mut u32, { [1] }> = source_base.reshape(const_shape![1]);
        let source_ptrs: PointerTile<*mut u32, { [128] }> = source_base.broadcast(tile_shape);
        let source_ptrs: PointerTile<*mut u32, { [128] }> = source_ptrs.offset_tile(logical_rows);
        let source_result: (Tile<u32, { [128] }>, Token) = load_ptr_tko(
            source_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u32),
            None,
            Latency::<0>,
        );
        let source_rows: Tile<i32, { [128] }> = bitcast(source_result.0);
        let source_offsets = source_rows * cols_tile + columns;

        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(tile_shape);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(source_offsets);
        let values: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f64),
            None,
            Latency::<0>,
        );

        let output_base: PointerTile<*mut u32, { [] }> = pointer_to_tile(output_indices);
        let output_base: PointerTile<*mut u32, { [1] }> = output_base.reshape(const_shape![1]);
        let output_ptrs: PointerTile<*mut u32, { [128] }> = output_base.broadcast(tile_shape);
        let output_ptrs: PointerTile<*mut u32, { [128] }> = output_ptrs.offset_tile(logical_rows);
        let output_result: (Tile<u32, { [128] }>, Token) = load_ptr_tko(
            output_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u32),
            None,
            Latency::<0>,
        );
        let output_rows: Tile<i32, { [128] }> = bitcast(output_result.0);
        let output_offsets = output_rows * cols_tile + columns;

        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(output_offsets);
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
}

pub use kernels::*;
