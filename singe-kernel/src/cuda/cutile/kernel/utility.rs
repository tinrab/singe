#[cutile::module]
pub mod kernel {
    use cutile::core::{
        Latency, PointerTile_0, PointerTile_1, Reshape, ReshapePtr, Shape_1, Tile_1, Token, bf16,
        broadcast_scalar, cmpi, constant, convert_tile, f16, iota, load_ptr_tko, ordering,
        pointer_to_tile, predicate, scope, select, store_ptr_tko,
    };

    use crate::cuda::cutile::runtime as cutile;

    pub unsafe fn load_vector(
        input: *mut f32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: f32,
    ) -> Tile<f32, { [128] }> {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let input_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f32, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<f32, { [128] }>, Token) = unsafe {
            load_ptr_tko(
                input_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(fill),
                None,
                Latency::<0>,
            )
        };
        result.0
    }

    pub unsafe fn load_vector_1024(
        input: *mut f32,
        offsets: Tile<i32, { [1024] }>,
        mask: Tile<bool, { [1024] }>,
        fill: f32,
    ) -> Tile<f32, { [1024] }> {
        let zero_offsets: Tile<i32, { [1024] }> = constant(0i32, const_shape![1024]);
        let offsets = select(mask, offsets, zero_offsets);
        let input_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f32, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [1024] }> =
            input_base.broadcast(const_shape![1024]);
        let input_ptrs: PointerTile<*mut f32, { [1024] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<f32, { [1024] }>, Token) = unsafe {
            load_ptr_tko(
                input_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(fill),
                None,
                Latency::<0>,
            )
        };
        result.0
    }

    pub unsafe fn load_u32_vector(
        input: *mut u32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: u32,
    ) -> Tile<u32, { [128] }> {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let input_base: PointerTile<*mut u32, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut u32, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut u32, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut u32, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<u32, { [128] }>, Token) = unsafe {
            load_ptr_tko(
                input_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(fill),
                None,
                Latency::<0>,
            )
        };
        result.0
    }

    pub unsafe fn load_i32_vector(
        input: *mut i32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: i32,
    ) -> Tile<i32, { [128] }> {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let input_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut i32, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut i32, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut i32, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<i32, { [128] }>, Token) = unsafe {
            load_ptr_tko(
                input_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(fill),
                None,
                Latency::<0>,
            )
        };
        result.0
    }

    pub unsafe fn load_vector_f16_as_f32(
        input: *mut f16,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let input_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f16, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f16, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut f16, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<f16, { [128] }>, Token) = unsafe {
            load_ptr_tko(
                input_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                None,
                Latency::<0>,
            )
        };
        let values: Tile<f32, { [128] }> = convert_tile(result.0);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
        select(mask, values, zero)
    }

    pub unsafe fn load_vector_bf16_as_f32(
        input: *mut bf16,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let input_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut bf16, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut bf16, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut bf16, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<bf16, { [128] }>, Token) = unsafe {
            load_ptr_tko(
                input_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                None,
                Latency::<0>,
            )
        };
        let values: Tile<f32, { [128] }> = convert_tile(result.0);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
        select(mask, values, zero)
    }

    pub unsafe fn store_vector(
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
        unsafe {
            store_ptr_tko(
                out_ptrs,
                values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            )
        };
    }

    pub unsafe fn store_vector_16(
        out: *mut f32,
        offsets: Tile<i32, { [16] }>,
        values: Tile<f32, { [16] }>,
        mask: Tile<bool, { [16] }>,
    ) {
        let zero_offsets: Tile<i32, { [16] }> = constant(0i32, const_shape![16]);
        let offsets = select(mask, offsets, zero_offsets);
        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [16] }> = out_base.broadcast(const_shape![16]);
        let out_ptrs: PointerTile<*mut f32, { [16] }> = out_ptrs.offset_tile(offsets);
        unsafe {
            store_ptr_tko(
                out_ptrs,
                values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            )
        };
    }

    pub unsafe fn store_vector_f16_from_f32(
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
        unsafe {
            store_ptr_tko(
                out_ptrs,
                output,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            )
        };
    }

    pub unsafe fn store_vector_bf16_from_f32(
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
        unsafe {
            store_ptr_tko(
                out_ptrs,
                output,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            )
        };
    }

    pub unsafe fn store_first_lane(out: *mut f32, offset: i32, value: Tile<f32, { [1] }>) {
        let tile_shape = const_shape![128];
        let lane: Tile<i32, { [128] }> = iota(tile_shape);
        let offsets = broadcast_scalar(offset, tile_shape);
        let values = value.reshape(const_shape![1]).broadcast(tile_shape);
        let zero: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mask = cmpi(lane, zero, predicate::Equal);
        unsafe { store_vector(out, offsets, values, mask) };
    }
}

pub use kernel::*;
