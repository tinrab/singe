#[cutile::module]
mod kernels {
    use cutile::core::*;

    use crate::cuda::cutile::{
        kernel::tiled::kernels::{load_1d, store_1d},
        runtime as cutile,
    };

    #[cutile::entry()]
    pub unsafe fn count(scratch: *mut i32, ids: *mut i32, assignments: i32, chunks: i32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let shape = const_shape![256];
        let lanes: Tile<i32, { [256] }> = iota(shape);
        let assignment_start: Tile<i32, { [256] }> = broadcast_scalar(pid.0 * 256i32, shape);
        let position = lanes + assignment_start;
        let assignment_limit: Tile<i32, { [256] }> = broadcast_scalar(assignments, shape);
        let valid = lt_tile(position, assignment_limit);
        let id = unsafe { load_1d(ids, position, valid) };
        let expert_index: Tile<i32, { [256] }> = broadcast_scalar(pid.1, shape);
        let matched = valid & eq_tile(id, expert_index);
        let one_flag: Tile<i32, { [256] }> = constant(1i32, shape);
        let zero_flag: Tile<i32, { [256] }> = constant(0i32, shape);
        let flags = select(matched, one_flag, zero_flag);
        let total: Tile<i32, { [] }> = reduce_sum(flags, 0i32);
        let count_index: Tile<i32, { [1] }> =
            broadcast_scalar(pid.1 * chunks + pid.0, const_shape![1]);
        let store_active: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        unsafe {
            store_1d(
                scratch,
                count_index,
                total.reshape(const_shape![1]),
                store_active,
            )
        };
    }

    #[cutile::entry()]
    pub unsafe fn prefix_chunks(scratch: *mut i32, chunks: i32, experts: i32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let shape = const_shape![256];
        let mut carry: Tile<i32, { [] }> = constant(0i32, const_shape![]);
        for tile in 0i32..ceil_div(chunks, 256i32) {
            let lanes: Tile<i32, { [256] }> = iota(shape);
            let chunk_start: Tile<i32, { [256] }> = broadcast_scalar(tile * 256i32, shape);
            let index = lanes + chunk_start;
            let chunk_limit: Tile<i32, { [256] }> = broadcast_scalar(chunks, shape);
            let valid = lt_tile(index, chunk_limit);
            let expert_chunk_base: Tile<i32, { [256] }> = broadcast_scalar(pid.0 * chunks, shape);
            let offset = expert_chunk_base + index;
            let counts = unsafe { load_1d(scratch, offset, valid) };
            let zero_count: Tile<i32, { [256] }> = constant(0i32, shape);
            let counts = select(valid, counts, zero_count);
            let sum: Tile<i32, { [] }> = reduce_sum(counts, 0i32);
            let inclusive = scan_sum(counts, 0i32, reverse::Forward, 0i32);
            let before = inclusive - counts + carry.reshape(const_shape![1]).broadcast(shape);
            unsafe { store_1d(scratch, offset, before, valid) };
            carry = carry + sum;
        }
        let total_index: Tile<i32, { [1] }> =
            broadcast_scalar(experts * chunks + pid.0, const_shape![1]);
        let store_active: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        unsafe {
            store_1d(
                scratch,
                total_index,
                carry.reshape(const_shape![1]),
                store_active,
            )
        };
    }

    #[cutile::entry()]
    pub unsafe fn prefix_experts(
        scratch: *mut i32,
        cumsum: *mut i32,
        total: *mut i32,
        maximum: *mut i32,
        chunks: i32,
        experts: i32,
        block_size: i32,
    ) {
        let shape = const_shape![256];
        let mut carry: Tile<i32, { [] }> = constant(0i32, const_shape![]);
        let mut largest: Tile<i32, { [] }> = constant(0i32, const_shape![]);
        for tile in 0i32..ceil_div(experts, 256i32) {
            let lanes: Tile<i32, { [256] }> = iota(shape);
            let expert_start: Tile<i32, { [256] }> = broadcast_scalar(tile * 256i32, shape);
            let expert = lanes + expert_start;
            let expert_limit: Tile<i32, { [256] }> = broadcast_scalar(experts, shape);
            let valid = lt_tile(expert, expert_limit);
            let totals_base: Tile<i32, { [256] }> = broadcast_scalar(experts * chunks, shape);
            let count = unsafe { load_1d(scratch, totals_base + expert, valid) };
            let zero_count: Tile<i32, { [256] }> = constant(0i32, shape);
            let count = select(valid, count, zero_count);
            let tile_max: Tile<i32, { [] }> = reduce_max(count, 0i32);
            largest = max_tile(largest, tile_max);
            let block: Tile<i32, { [256] }> = broadcast_scalar(block_size, shape);
            let block_padding: Tile<i32, { [256] }> = constant(1i32, shape);
            let padded = ((count + block - block_padding) / block) * block;
            let sum: Tile<i32, { [] }> = reduce_sum(padded, 0i32);
            let inclusive = scan_sum(padded, 0i32, reverse::Forward, 0i32);
            let offset = inclusive - padded + carry.reshape(const_shape![1]).broadcast(shape);
            unsafe { store_1d(cumsum, expert, offset, valid) };
            carry = carry + sum;
        }
        let active: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        let zero: Tile<i32, { [1] }> = constant(0i32, const_shape![1]);
        unsafe {
            let final_offset_index: Tile<i32, { [1] }> = broadcast_scalar(experts, const_shape![1]);
            store_1d(
                cumsum,
                final_offset_index,
                carry.reshape(const_shape![1]),
                active,
            );
            store_1d(total, zero, carry.reshape(const_shape![1]), active);
            store_1d(maximum, zero, largest.reshape(const_shape![1]), active);
        }
    }

    #[cutile::entry()]
    pub unsafe fn scatter(
        sorted: *mut i32,
        expert_ids: *mut i32,
        ids: *mut i32,
        scratch: *mut i32,
        cumsum: *mut i32,
        assignments: i32,
        chunks: i32,
        block_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let one = const_shape![1];
        let active: Tile<bool, { [1] }> = constant(true, one);
        let expert_index: Tile<i32, { [1] }> = broadcast_scalar(pid.1, one);
        let start = unsafe { load_1d(cumsum, expert_index, active) };
        let chunk_index: Tile<i32, { [1] }> = broadcast_scalar(pid.1 * chunks + pid.0, one);
        let chunk = unsafe { load_1d(scratch, chunk_index, active) };
        let shape = const_shape![256];
        let assignment_lanes: Tile<i32, { [256] }> = iota(shape);
        let assignment_start: Tile<i32, { [256] }> = broadcast_scalar(pid.0 * 256i32, shape);
        let index = assignment_lanes + assignment_start;
        let assignment_limit: Tile<i32, { [256] }> = broadcast_scalar(assignments, shape);
        let valid = lt_tile(index, assignment_limit);
        let id = unsafe { load_1d(ids, index, valid) };
        let selected_expert: Tile<i32, { [256] }> = broadcast_scalar(pid.1, shape);
        let matched = valid & eq_tile(id, selected_expert);
        let one_flag: Tile<i32, { [256] }> = constant(1i32, shape);
        let zero_flag: Tile<i32, { [256] }> = constant(0i32, shape);
        let flags = select(matched, one_flag, zero_flag);
        let inclusive_adjustment: Tile<i32, { [256] }> = constant(1i32, shape);
        let rank = scan_sum(flags, 0i32, reverse::Forward, 0i32) - inclusive_adjustment;
        let offset = (start + chunk).broadcast(shape) + rank;
        unsafe { store_1d(sorted, offset, index, matched) };
        if pid.0 == 0i32 {
            let next_expert_index: Tile<i32, { [1] }> = broadcast_scalar(pid.1 + 1i32, one);
            let end = unsafe { load_1d(cumsum, next_expert_index, active) };
            let first: i32 = tile_to_scalar(start.reshape(const_shape![]));
            let first = first / block_size;
            let last: i32 = tile_to_scalar(end.reshape(const_shape![]));
            let last = last / block_size;
            for tile in 0i32..ceil_div(last - first, 256i32) {
                let block_lanes: Tile<i32, { [256] }> = iota(shape);
                let block_start: Tile<i32, { [256] }> =
                    broadcast_scalar(first + tile * 256i32, shape);
                let index = block_lanes + block_start;
                let block_limit: Tile<i32, { [256] }> = broadcast_scalar(last, shape);
                let valid = lt_tile(index, block_limit);
                let block_expert: Tile<i32, { [256] }> = broadcast_scalar(pid.1, shape);
                unsafe { store_1d(expert_ids, index, block_expert, valid) };
            }
        }
    }
}

pub use kernels::*;
