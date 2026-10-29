//! Pointer access shared by tiled inference kernels.

#[cutile::module]
pub mod kernels {
    use cutile::core::{
        ElementType, Latency, PointerTile_0, PointerTile_1, PointerTile_2, ReshapePtr, Shape_1,
        Shape_2, Tile_1, Tile_2, Token, absf, bitcast, constant, load_ptr_tko, max_tile, ordering,
        pointer_to_tile, scope, select, store_ptr_tko,
    };

    use crate::cuda::cutile::runtime as cutile;

    #[must_use]
    pub fn ue8m0_round_scale_f32<const N: i32>(scale: Tile<f32, { [N] }>) -> Tile<f32, { [N] }> {
        let min_scale: Tile<f32, { [N] }> = constant(1e-10f32, const_shape![N]);
        let safe_scale = max_tile(absf(scale), min_scale);
        // The clamp makes every finite input normal. Subtract before discarding
        // the mantissa so exact powers stay unchanged, including adjacent floats.
        let bits: Tile<u32, { [N] }> = bitcast(safe_scale);
        let one: Tile<u32, { [N] }> = constant(1u32, const_shape![N]);
        let shift: Tile<u32, { [N] }> = constant(23u32, const_shape![N]);
        let rounded = (((bits - one) >> shift) + one) << shift;
        bitcast(rounded)
    }

    pub unsafe fn load_1d<T: ElementType, const N: i32>(
        input: *mut T,
        offsets: Tile<i32, { [N] }>,
        valid: Tile<bool, { [N] }>,
    ) -> Tile<T, { [N] }> {
        let ptr: PointerTile<*mut T, { [] }> = pointer_to_tile(input);
        let ptr: PointerTile<*mut T, { [1] }> = ptr.reshape(const_shape![1]);
        let ptr: PointerTile<*mut T, { [N] }> = ptr.broadcast(const_shape![N]);
        let zero: Tile<i32, { [N] }> = constant(0i32, const_shape![N]);
        let offsets = select(valid, offsets, zero);
        let ptr = ptr.offset_tile(offsets);
        let result: (Tile<T, { [N] }>, Token) = unsafe {
            load_ptr_tko(
                ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid),
                None,
                None,
                Latency::<0>,
            )
        };
        result.0
    }

    pub unsafe fn store_1d<T: ElementType, const N: i32>(
        out: *mut T,
        offsets: Tile<i32, { [N] }>,
        values: Tile<T, { [N] }>,
        valid: Tile<bool, { [N] }>,
    ) {
        let ptr: PointerTile<*mut T, { [] }> = pointer_to_tile(out);
        let ptr: PointerTile<*mut T, { [1] }> = ptr.reshape(const_shape![1]);
        let ptr: PointerTile<*mut T, { [N] }> = ptr.broadcast(const_shape![N]);
        let zero: Tile<i32, { [N] }> = constant(0i32, const_shape![N]);
        let offsets = select(valid, offsets, zero);
        let ptr = ptr.offset_tile(offsets);
        unsafe {
            store_ptr_tko(
                ptr,
                values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid),
                None,
                Latency::<0>,
            )
        };
    }

    pub unsafe fn load_2d<T: ElementType, const M: i32, const N: i32>(
        input: *mut T,
        offsets: Tile<i32, { [M, N] }>,
        valid: Tile<bool, { [M, N] }>,
    ) -> Tile<T, { [M, N] }> {
        let ptr: PointerTile<*mut T, { [] }> = pointer_to_tile(input);
        let ptr: PointerTile<*mut T, { [1, 1] }> = ptr.reshape(const_shape![1, 1]);
        let ptr: PointerTile<*mut T, { [M, N] }> = ptr.broadcast(const_shape![M, N]);
        let zero: Tile<i32, { [M, N] }> = constant(0i32, const_shape![M, N]);
        let offsets = select(valid, offsets, zero);
        let ptr = ptr.offset_tile(offsets);
        let result: (Tile<T, { [M, N] }>, Token) = unsafe {
            load_ptr_tko(
                ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid),
                None,
                None,
                Latency::<0>,
            )
        };
        result.0
    }

    pub unsafe fn store_2d<T: ElementType, const M: i32, const N: i32>(
        out: *mut T,
        offsets: Tile<i32, { [M, N] }>,
        values: Tile<T, { [M, N] }>,
        valid: Tile<bool, { [M, N] }>,
    ) {
        let ptr: PointerTile<*mut T, { [] }> = pointer_to_tile(out);
        let ptr: PointerTile<*mut T, { [1, 1] }> = ptr.reshape(const_shape![1, 1]);
        let ptr: PointerTile<*mut T, { [M, N] }> = ptr.broadcast(const_shape![M, N]);
        let zero: Tile<i32, { [M, N] }> = constant(0i32, const_shape![M, N]);
        let offsets = select(valid, offsets, zero);
        let ptr = ptr.offset_tile(offsets);
        unsafe {
            store_ptr_tko(
                ptr,
                values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid),
                None,
                Latency::<0>,
            )
        };
    }
}
