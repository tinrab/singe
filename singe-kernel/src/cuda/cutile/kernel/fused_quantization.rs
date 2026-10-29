#[cutile::module]
mod kernels {
    use cutile::core::{
        ElementType, Reshape, Shape_1, Shape_2, Tile_0, Tile_1, Tile_2, absf, broadcast_scalar,
        ceil_div, constant, convert_scalar, convert_tile, exp, f8e4m3fn, ftz, get_tile_block_id,
        iota, lt_tile, max_tile, min_tile, reduce_max, reduce_sum, rsqrt, scalar_to_tile, select,
    };

    use crate::cuda::cutile::{
        kernel::tiled::kernels::{load_1d, load_2d, store_1d, store_2d, ue8m0_round_scale_f32},
        runtime as cutile,
    };

    #[cutile::entry()]
    pub unsafe fn rms_norm_add_fp8<
        T: ElementType,
        const BN: i32,
        const GROUP: i32,
        const NG: i32,
        const POW2: i32,
    >(
        out: *mut f8e4m3fn,
        scales: *mut f32,
        residual_out: *mut T,
        input: *mut T,
        residual: *mut T,
        weight: *mut T,
        columns: i32,
        scale_stride: i32,
        column_major: i32,
        norm_eps: f32,
        weight_offset: f32,
        quant_eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let shape = const_shape![BN];
        let lane: Tile<i32, { [BN] }> = iota(shape);
        let zero: Tile<f32, { [BN] }> = constant(0.0f32, shape);
        let mut squares = zero;
        let mut resident = zero;
        for chunk in 0i32..ceil_div(columns, BN) {
            let statistic_tile_start: Tile<i32, { [BN] }> = broadcast_scalar(chunk * BN, shape);
            let col = lane + statistic_tile_start;
            let statistic_column_limit: Tile<i32, { [BN] }> = broadcast_scalar(columns, shape);
            let valid = lt_tile(col, statistic_column_limit);
            let statistic_row_base: Tile<i32, { [BN] }> = broadcast_scalar(pid.0 * columns, shape);
            let offset = statistic_row_base + col;
            let x: Tile<f32, { [BN] }> = convert_tile(unsafe { load_1d(input, offset, valid) });
            let r: Tile<f32, { [BN] }> = convert_tile(unsafe { load_1d(residual, offset, valid) });
            let rounded: Tile<T, { [BN] }> = convert_tile(x + r);
            let x: Tile<f32, { [BN] }> = convert_tile(rounded);
            resident = select(valid, x, zero);
            squares = squares + resident * resident;
            unsafe { store_1d(residual_out, offset, rounded, valid) };
        }
        let sum: Tile<f32, { [] }> = reduce_sum(squares, 0i32);
        let n: f32 = convert_scalar(columns);
        let normalization_width: Tile<f32, { [] }> = scalar_to_tile(n);
        let normalization_epsilon: Tile<f32, { [] }> = scalar_to_tile(norm_eps);
        let inv = rsqrt(
            sum / normalization_width + normalization_epsilon,
            ftz::Disabled,
        );
        for chunk in 0i32..ceil_div(columns, BN) {
            let output_tile_start: Tile<i32, { [BN] }> = broadcast_scalar(chunk * BN, shape);
            let col = lane + output_tile_start;
            let output_column_limit: Tile<i32, { [BN] }> = broadcast_scalar(columns, shape);
            let valid = lt_tile(col, output_column_limit);
            let output_row_base: Tile<i32, { [BN] }> = broadcast_scalar(pid.0 * columns, shape);
            let offset = output_row_base + col;
            let x = if columns > BN {
                convert_tile(unsafe { load_1d(residual_out, offset, valid) })
            } else {
                resident
            };
            let w: Tile<f32, { [BN] }> = convert_tile(unsafe { load_1d(weight, col, valid) });
            let weight_shift: Tile<f32, { [BN] }> = broadcast_scalar(weight_offset, shape);
            let y = x * inv.reshape(const_shape![1]).broadcast(shape) * (w + weight_shift);
            let stored: Tile<T, { [BN] }> = convert_tile(y);
            let y: Tile<f32, { [BN] }> = convert_tile(stored);
            let y = select(valid, y, zero);
            let grouped = absf(y).reshape(const_shape![NG, GROUP]);
            let amax: Tile<f32, { [NG] }> = reduce_max(grouped, 1i32);
            let amax = amax.reshape(const_shape![NG]);
            let quant_epsilon: Tile<f32, { [NG] }> = broadcast_scalar(quant_eps, const_shape![NG]);
            let fp8_max_group: Tile<f32, { [NG] }> = constant(448.0f32, const_shape![NG]);
            let raw = max_tile(amax, quant_epsilon) / fp8_max_group;
            let scale = if POW2 != 0 {
                ue8m0_round_scale_f32(raw)
            } else {
                raw
            };
            let expanded = scale
                .reshape(const_shape![NG, 1])
                .broadcast(const_shape![NG, GROUP])
                .reshape(shape);
            let fp8_min_value: Tile<f32, { [BN] }> = constant(-448.0f32, shape);
            let fp8_max_value: Tile<f32, { [BN] }> = constant(448.0f32, shape);
            let quantized = min_tile(max_tile(y / expanded, fp8_min_value), fp8_max_value);
            let quantized: Tile<f8e4m3fn, { [BN] }> = convert_tile(quantized);
            unsafe { store_1d(out, offset, quantized, valid) };
            let scale_tile_start: Tile<i32, { [NG] }> =
                broadcast_scalar(chunk * (NG), const_shape![NG]);
            let group: Tile<i32, { [NG] }> = iota(const_shape![NG]) + scale_tile_start;
            let scale_column_stride: Tile<i32, { [NG] }> =
                broadcast_scalar(scale_stride, const_shape![NG]);
            let scale_row: Tile<i32, { [NG] }> = broadcast_scalar(pid.0, const_shape![NG]);
            let scale_row_base: Tile<i32, { [NG] }> =
                broadcast_scalar(pid.0 * (columns / GROUP), const_shape![NG]);
            let scale_offset = if column_major != 0 {
                group * scale_column_stride + scale_row
            } else {
                scale_row_base + group
            };
            let scale_group_limit: Tile<i32, { [NG] }> =
                broadcast_scalar(columns / GROUP, const_shape![NG]);
            unsafe {
                store_1d(
                    scales,
                    scale_offset,
                    scale,
                    lt_tile(group, scale_group_limit),
                );
            };
        }
    }

    #[cutile::entry()]
    pub unsafe fn silu_mul_fp8<T: ElementType, const GROUP: i32, const POW2: i32>(
        out: *mut f8e4m3fn,
        scales: *mut f32,
        gate: *mut T,
        up: *mut T,
        rows: i32,
        columns: i32,
        scale_stride: i32,
        column_major: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let shape = const_shape![4, GROUP];
        let group_tile_start: Tile<i32, { [4] }> = broadcast_scalar(pid.0 * 4i32, const_shape![4]);
        let group: Tile<i32, { [4] }> = iota(const_shape![4]) + group_tile_start;
        let per_row = columns / GROUP;
        let groups_per_row: Tile<i32, { [4] }> = broadcast_scalar(per_row, const_shape![4]);
        let row = group / groups_per_row;
        let group_row_stride: Tile<i32, { [4] }> = broadcast_scalar(per_row, const_shape![4]);
        let col_group = group - row * group_row_stride;
        let lane: Tile<i32, { [GROUP] }> = iota(const_shape![GROUP]);
        let group_width: Tile<i32, { [4] }> = broadcast_scalar(GROUP, const_shape![4]);
        let offsets = (group * group_width)
            .reshape(const_shape![4, 1])
            .broadcast(shape)
            + lane.reshape(const_shape![1, GROUP]).broadcast(shape);
        let row_limit: Tile<i32, { [4] }> = broadcast_scalar(rows, const_shape![4]);
        let row_valid = lt_tile(row, row_limit);
        let valid = row_valid.reshape(const_shape![4, 1]).broadcast(shape);
        let gate: Tile<f32, { [4, GROUP] }> =
            convert_tile(unsafe { load_2d(gate, offsets, valid) });
        let up: Tile<f32, { [4, GROUP] }> = convert_tile(unsafe { load_2d(up, offsets, valid) });
        let one: Tile<f32, { [4, GROUP] }> = constant(1.0f32, shape);
        let zero: Tile<f32, { [4, GROUP] }> = constant(0.0f32, shape);
        let value = gate * (one / (one + exp(zero - gate))) * up;
        let stored: Tile<T, { [4, GROUP] }> = convert_tile(value);
        let value: Tile<f32, { [4, GROUP] }> = convert_tile(stored);
        let inactive_zero: Tile<f32, { [4, GROUP] }> = constant(0.0f32, shape);
        let value = select(valid, value, inactive_zero);
        let amax: Tile<f32, { [4] }> = reduce_max(absf(value), 1i32);
        let quant_epsilon: Tile<f32, { [4] }> = broadcast_scalar(eps, const_shape![4]);
        let fp8_max_group: Tile<f32, { [4] }> = constant(448.0f32, const_shape![4]);
        let raw = max_tile(amax, quant_epsilon) / fp8_max_group;
        let scale = if POW2 != 0 {
            ue8m0_round_scale_f32(raw)
        } else {
            raw
        };
        let quantized = value / scale.reshape(const_shape![4, 1]).broadcast(shape);
        let fp8_min_value: Tile<f32, { [4, GROUP] }> = constant(-448.0f32, shape);
        let fp8_max_value: Tile<f32, { [4, GROUP] }> = constant(448.0f32, shape);
        let quantized = min_tile(max_tile(quantized, fp8_min_value), fp8_max_value);
        let quantized: Tile<f8e4m3fn, { [4, GROUP] }> = convert_tile(quantized);
        unsafe { store_2d(out, offsets, quantized, valid) };
        let scale_column_stride: Tile<i32, { [4] }> =
            broadcast_scalar(scale_stride, const_shape![4]);
        let scale_offset = if column_major != 0 {
            col_group * scale_column_stride + row
        } else {
            group
        };
        unsafe { store_1d(scales, scale_offset, scale, row_valid) };
    }
}

pub use kernels::*;
