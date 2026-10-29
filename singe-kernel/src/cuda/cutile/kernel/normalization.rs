#[cutile::module]
mod kernels {
    use cutile::core::{
        ElementType, Reshape, Shape_1, Shape_2, Tile_1, Tile_2, broadcast_scalar, constant,
        convert_scalar, convert_tile, ftz, get_tile_block_id, iota, lt_tile, reduce_sum, rsqrt,
        select,
    };

    use crate::cuda::cutile::{
        kernel::tiled::kernels::{load_2d, store_2d},
        runtime as cutile,
    };

    #[cutile::entry()]
    pub unsafe fn normalization<T: ElementType, const BM: i32, const BN: i32, const RMS: i32>(
        out: *mut T,
        input: *mut T,
        weight: *mut T,
        bias: *mut T,
        rows: i32,
        columns: i32,
        has_weight: i32,
        has_bias: i32,
        eps: f32,
        weight_offset: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let shape = const_shape![BM, BN];
        let row_start: Tile<i32, { [BM] }> = broadcast_scalar(pid.0 * BM, const_shape![BM]);
        let row: Tile<i32, { [BM] }> = iota(const_shape![BM]) + row_start;
        let col: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let row = row.reshape(const_shape![BM, 1]).broadcast(shape);
        let col = col.reshape(const_shape![1, BN]).broadcast(shape);
        let column_limit: Tile<i32, { [BM, BN] }> = broadcast_scalar(columns, shape);
        let valid_col = lt_tile(col, column_limit);
        let row_limit: Tile<i32, { [BM, BN] }> = broadcast_scalar(rows, shape);
        let valid = valid_col & lt_tile(row, row_limit);
        let column_stride: Tile<i32, { [BM, BN] }> = broadcast_scalar(columns, shape);
        let offsets = row * column_stride + col;
        let zero: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape);
        let x: Tile<f32, { [BM, BN] }> = convert_tile(unsafe { load_2d(input, offsets, valid) });
        let x = select(valid, x, zero);
        let n: f32 = convert_scalar(columns);
        let centered = if RMS == 0 {
            let total: Tile<f32, { [BM] }> = reduce_sum(x, 1i32);
            let total = total.reshape(const_shape![BM]);
            let mean_width: Tile<f32, { [BM] }> = broadcast_scalar(n, const_shape![BM]);
            let mean = total / mean_width;
            select(
                valid_col,
                x - mean.reshape(const_shape![BM, 1]).broadcast(shape),
                zero,
            )
        } else {
            x
        };
        let total: Tile<f32, { [BM] }> = reduce_sum(centered * centered, 1i32);
        let total = total.reshape(const_shape![BM]);
        let variance_width: Tile<f32, { [BM] }> = broadcast_scalar(n, const_shape![BM]);
        let variance = total / variance_width;
        let epsilon: Tile<f32, { [BM] }> = broadcast_scalar(eps, const_shape![BM]);
        let inv: Tile<f32, { [BM] }> = rsqrt(variance + epsilon, ftz::Disabled);
        let mut value = centered * inv.reshape(const_shape![BM, 1]).broadcast(shape);
        if has_weight != 0 {
            let w: Tile<f32, { [BM, BN] }> =
                convert_tile(unsafe { load_2d(weight, col, valid_col) });
            let offset = if RMS != 0 { weight_offset } else { 0.0f32 };
            let weight_shift: Tile<f32, { [BM, BN] }> = broadcast_scalar(offset, shape);
            value = value * (w + weight_shift);
        }
        if has_bias != 0 {
            let b: Tile<f32, { [BM, BN] }> = convert_tile(unsafe { load_2d(bias, col, valid_col) });
            value = value + b;
        }
        let value: Tile<T, { [BM, BN] }> = convert_tile(value);
        unsafe { store_2d(out, offsets, value, valid) };
    }
}

pub use kernels::*;
