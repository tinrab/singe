#[cutile::module]
mod kernels {
    use cutile::core::*;

    use crate::cuda::cutile::{
        kernel::tiled::kernels::{load_1d, load_2d, store_2d},
        runtime as cutile,
    };

    #[cutile::entry()]
    pub unsafe fn expert_product<T: ElementType, O: ElementType, const QUANT: i32>(
        out: *mut O,
        input: *mut T,
        weight: *mut T,
        routed_weight: *mut f32,
        sorted_ids: *mut i32,
        expert_ids: *mut i32,
        total: *mut i32,
        assignments: i32,
        columns: i32,
        reduction: i32,
        top_k: i32,
        block_size: i32,
        input_stride: i32,
        expert_stride: i32,
        weight_stride: i32,
        output_stride: i32,
        mul_route: i32,
        input_scales: *mut f32,
        weight_scales: *mut f32,
        group_n: i32,
        group_k: i32,
        k_groups: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let active: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        let first_index: Tile<i32, { [1] }> = constant(0i32, const_shape![1]);
        let count = unsafe { load_1d(total, first_index, active) };
        let count: i32 = tile_to_scalar(count.reshape(const_shape![]));
        let block_valid = pid.0 * block_size < count;
        let routing_block_index: Tile<i32, { [1] }> = broadcast_scalar(pid.0, const_shape![1]);
        let routing_block_valid: Tile<bool, { [1] }> =
            broadcast_scalar(block_valid, const_shape![1]);
        let expert = unsafe { load_1d(expert_ids, routing_block_index, routing_block_valid) };
        let expert_valid: Tile<bool, { [1] }> = broadcast_scalar(block_valid, const_shape![1]);
        let zero_expert: Tile<i32, { [1] }> = constant(0i32, const_shape![1]);
        let expert = select(expert_valid, expert, zero_expert);
        let expert: i32 = tile_to_scalar(expert.reshape(const_shape![]));
        let local_row_start: Tile<i32, { [16] }> =
            broadcast_scalar(pid.2 * 16i32, const_shape![16]);
        let local: Tile<i32, { [16] }> = iota(const_shape![16]) + local_row_start;
        let sorted_block_start: Tile<i32, { [16] }> =
            broadcast_scalar(pid.0 * block_size, const_shape![16]);
        let position = local + sorted_block_start;
        let local_row_limit: Tile<i32, { [16] }> = broadcast_scalar(block_size, const_shape![16]);
        let padded_count: Tile<i32, { [16] }> = broadcast_scalar(count, const_shape![16]);
        let valid_row = lt_tile(local, local_row_limit) & lt_tile(position, padded_count);
        let id = unsafe { load_1d(sorted_ids, position, valid_row) };
        let assignment_sentinel: Tile<i32, { [16] }> =
            broadcast_scalar(assignments, const_shape![16]);
        let id = select(valid_row, id, assignment_sentinel);
        let assignment_limit: Tile<i32, { [16] }> = broadcast_scalar(assignments, const_shape![16]);
        let valid_row = valid_row & lt_tile(id, assignment_limit);
        let assignments_per_token: Tile<i32, { [16] }> = broadcast_scalar(top_k, const_shape![16]);
        let token = id / assignments_per_token;
        let column_start: Tile<i32, { [64] }> = broadcast_scalar(pid.1 * 64i32, const_shape![64]);
        let col: Tile<i32, { [64] }> = iota(const_shape![64]) + column_start;
        let column_limit: Tile<i32, { [64] }> = broadcast_scalar(columns, const_shape![64]);
        let valid_col = lt_tile(col, column_limit);
        let mut acc: Tile<f32, { [16, 64] }> = constant(0.0f32, const_shape![16, 64]);
        let groups = if QUANT != 0 { k_groups } else { 1i32 };
        let width = if QUANT != 0 { group_k } else { reduction };
        for group in 0i32..groups {
            let mut part: Tile<f32, { [16, 64] }> = constant(0.0f32, const_shape![16, 64]);
            for tile in 0i32..ceil_div(width, 32i32) {
                let reduction_start: Tile<i32, { [32] }> =
                    broadcast_scalar(group * width + tile * 32i32, const_shape![32]);
                let k: Tile<i32, { [32] }> = iota(const_shape![32]) + reduction_start;
                let reduction_limit: Tile<i32, { [32] }> =
                    broadcast_scalar(reduction, const_shape![32]);
                let group_end: Tile<i32, { [32] }> =
                    broadcast_scalar((group + 1i32) * width, const_shape![32]);
                let valid_k = lt_tile(k, reduction_limit) & lt_tile(k, group_end);
                let input_row_stride: Tile<i32, { [16] }> =
                    broadcast_scalar(input_stride, const_shape![16]);
                let lhs_offsets = (token * input_row_stride)
                    .reshape(const_shape![16, 1])
                    .broadcast(const_shape![16, 32])
                    + k.reshape(const_shape![1, 32])
                        .broadcast(const_shape![16, 32]);
                let lhs_valid = valid_row
                    .reshape(const_shape![16, 1])
                    .broadcast(const_shape![16, 32])
                    & valid_k
                        .reshape(const_shape![1, 32])
                        .broadcast(const_shape![16, 32]);
                let lhs: Tile<T, { [16, 32] }> = unsafe { load_2d(input, lhs_offsets, lhs_valid) };
                let zero_lhs_float: Tile<f32, { [16, 32] }> =
                    constant(0.0f32, const_shape![16, 32]);
                let zero_lhs: Tile<T, { [16, 32] }> = convert_tile(zero_lhs_float);
                let lhs = select(lhs_valid, lhs, zero_lhs);
                let expert_weight_base: Tile<i32, { [32, 64] }> =
                    broadcast_scalar(expert * expert_stride, const_shape![32, 64]);
                let weight_column_stride: Tile<i32, { [64] }> =
                    broadcast_scalar(weight_stride, const_shape![64]);
                let rhs_offsets = expert_weight_base
                    + k.reshape(const_shape![32, 1])
                        .broadcast(const_shape![32, 64])
                    + (col * weight_column_stride)
                        .reshape(const_shape![1, 64])
                        .broadcast(const_shape![32, 64]);
                let weight_block_valid: Tile<bool, { [32, 64] }> =
                    broadcast_scalar(block_valid, const_shape![32, 64]);
                let rhs_valid = valid_k
                    .reshape(const_shape![32, 1])
                    .broadcast(const_shape![32, 64])
                    & valid_col
                        .reshape(const_shape![1, 64])
                        .broadcast(const_shape![32, 64])
                    & weight_block_valid;
                let rhs: Tile<T, { [32, 64] }> = unsafe { load_2d(weight, rhs_offsets, rhs_valid) };
                let zero_rhs_float: Tile<f32, { [32, 64] }> =
                    constant(0.0f32, const_shape![32, 64]);
                let zero_rhs: Tile<T, { [32, 64] }> = convert_tile(zero_rhs_float);
                let rhs = select(rhs_valid, rhs, zero_rhs);
                if QUANT < 0 {
                    let lhs_float: Tile<f32, { [16, 32] }> = convert_tile(lhs);
                    let rhs_float: Tile<f32, { [32, 64] }> = convert_tile(rhs);
                    let zero_index: Tile<i32, { [] }> = constant(0i32, const_shape![]);
                    for column in 0i32..64i32 {
                        let column_index: Tile<i32, { [] }> = scalar_to_tile(column);
                        let rhs_column: Tile<f32, { [32, 1] }> =
                            extract(rhs_float, [zero_index, column_index]);
                        let rhs_row = rhs_column
                            .reshape(const_shape![1, 32])
                            .broadcast(const_shape![16, 32]);
                        let dot: Tile<f32, { [16] }> = reduce_sum(lhs_float * rhs_row, 1i32);
                        let old_column: Tile<f32, { [16, 1] }> =
                            extract(part, [zero_index, column_index]);
                        let sum = old_column + dot.reshape(const_shape![16, 1]);
                        part = unsafe { insert(sum, part, [zero_index, column_index]) };
                    }
                } else {
                    part = mma(lhs, rhs, part);
                }
            }
            if QUANT != 0 {
                let activation_scale_stride: Tile<i32, { [16] }> =
                    broadcast_scalar(k_groups, const_shape![16]);
                let scale_group_index: Tile<i32, { [16] }> =
                    broadcast_scalar(group, const_shape![16]);
                let a_offsets = token * activation_scale_stride + scale_group_index;
                let a = unsafe { load_1d(input_scales, a_offsets, valid_row) };
                let n_groups = ceil_div(columns, group_n);
                let expert_scale_base: Tile<i32, { [64] }> =
                    broadcast_scalar(expert * n_groups * k_groups + group, const_shape![64]);
                let scale_group_width: Tile<i32, { [64] }> =
                    broadcast_scalar(group_n, const_shape![64]);
                let weight_scale_stride: Tile<i32, { [64] }> =
                    broadcast_scalar(k_groups, const_shape![64]);
                let b_offsets = expert_scale_base + (col / scale_group_width) * weight_scale_stride;
                let scale_block_valid: Tile<bool, { [64] }> =
                    broadcast_scalar(block_valid, const_shape![64]);
                let b = unsafe { load_1d(weight_scales, b_offsets, valid_col & scale_block_valid) };
                part = part
                    * a.reshape(const_shape![16, 1])
                        .broadcast(const_shape![16, 64])
                    * b.reshape(const_shape![1, 64])
                        .broadcast(const_shape![16, 64]);
            }
            acc = acc + part;
        }
        if mul_route != 0 {
            let route = unsafe { load_1d(routed_weight, id, valid_row) };
            acc = acc
                * route
                    .reshape(const_shape![16, 1])
                    .broadcast(const_shape![16, 64]);
        }
        let output_row_stride: Tile<i32, { [16] }> =
            broadcast_scalar(output_stride, const_shape![16]);
        let offsets = (id * output_row_stride)
            .reshape(const_shape![16, 1])
            .broadcast(const_shape![16, 64])
            + col
                .reshape(const_shape![1, 64])
                .broadcast(const_shape![16, 64]);
        let valid = valid_row
            .reshape(const_shape![16, 1])
            .broadcast(const_shape![16, 64])
            & valid_col
                .reshape(const_shape![1, 64])
                .broadcast(const_shape![16, 64]);
        let value: Tile<O, { [16, 64] }> = convert_tile(acc);
        unsafe { store_2d(out, offsets, value, valid) };
    }
}

pub use kernels::*;
