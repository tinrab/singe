#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn embedding_lookup_f16(
        out: *mut f16,
        token_ids: *mut u32,
        table: *mut f16,
        width: i32,
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

        let width_tile = broadcast_scalar(width, tile_shape);
        let token_offsets: Tile<i32, { [128] }> = safe_offsets / width_tile;
        let columns: Tile<i32, { [128] }> = safe_offsets - token_offsets * width_tile;

        let ids_base: PointerTile<*mut u32, { [] }> = pointer_to_tile(token_ids);
        let ids_base: PointerTile<*mut u32, { [1] }> = ids_base.reshape(const_shape![1]);
        let id_ptrs: PointerTile<*mut u32, { [128] }> = ids_base.broadcast(tile_shape);
        let id_ptrs: PointerTile<*mut u32, { [128] }> = id_ptrs.offset_tile(token_offsets);
        let id_result: (Tile<u32, { [128] }>, Token) = load_ptr_tko(
            id_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u32),
            None,
            Latency::<0>,
        );
        let rows: Tile<i32, { [128] }> = bitcast(id_result.0);
        let source_offsets: Tile<i32, { [128] }> = rows * width_tile + columns;

        let table_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(table);
        let table_base: PointerTile<*mut f16, { [1] }> = table_base.reshape(const_shape![1]);
        let table_ptrs: PointerTile<*mut f16, { [128] }> = table_base.broadcast(tile_shape);
        let table_ptrs: PointerTile<*mut f16, { [128] }> = table_ptrs.offset_tile(source_offsets);
        let values: (Tile<f16, { [128] }>, Token) = load_ptr_tko(
            table_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );

        let out_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_base.broadcast(tile_shape);
        let out_ptrs: PointerTile<*mut f16, { [128] }> = out_ptrs.offset_tile(offsets);
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
