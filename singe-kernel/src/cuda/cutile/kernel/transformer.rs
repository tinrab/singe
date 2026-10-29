//! Fused transformer kernels using half storage and f32 arithmetic.

#[cutile::module]
mod kernels {
    use cutile::core::*;

    use crate::cuda::cutile::runtime as cutile;

    /// Normalizes Q/K heads, applies half-split RoPE, and optionally updates head-major KV caches.
    ///
    /// # Safety
    /// Launch one block per Q/K head and sequence token.
    /// `D` must be a power of two at least `head_dim`, which must be positive and even.
    /// Inputs and outputs use sequence-major layout, and trigonometric tables have `head_dim / 2` columns.
    /// Provide a one-element position buffer when `DYNAMIC` is nonzero and cache destinations when `CACHE` is nonzero.
    /// Keep correctly sized allocations alive and prevent conflicting accesses until completion and throughout graph replay.
    #[cutile::entry()]
    pub unsafe fn qk_norm_rope_f16<const D: i32, const CACHE: i32, const DYNAMIC: i32>(
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        q_weight: *mut f16,
        k_weight: *mut f16,
        q_out: *mut f16,
        k_out: *mut f16,
        k_cache: *mut f16,
        v_cache: *mut f16,
        cos: *mut f32,
        sin: *mut f32,
        position: *mut u32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        table_rows: i32,
        position_start: i32,
        table_start: i32,
        max_seq: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let is_query = pid.0 < heads;
        let head = if is_query { pid.0 } else { pid.0 - heads };
        let head_count = if is_query { heads } else { kv_heads };
        let input = if is_query { query } else { key };
        let weight = if is_query { q_weight } else { k_weight };
        let out = if is_query { q_out } else { k_out };
        let mut start = position_start;
        let mut trig_start = table_start;
        if DYNAMIC != 0 {
            let offsets: Tile<i32, { [1] }> = constant(0i32, const_shape![1]);
            let active: Tile<bool, { [1] }> = constant(true, const_shape![1]);
            let p: Tile<u32, { [1] }> = unsafe { load(position, offsets, active) };
            let p: Tile<i32, { [1] }> = bitcast(p);
            start = tile_to_scalar(p.reshape(const_shape![]));
            trig_start = start;
        }
        let table_row = trig_start + pid.1;
        let cache_row = start + pid.1;
        let shape = const_shape![D];
        let dim: Tile<i32, { [D] }> = iota(shape);
        let valid: Tile<bool, { [D] }> = lt_tile(dim, broadcast_scalar(head_dim, shape))
            & broadcast_scalar(table_row >= 0i32 && table_row < table_rows, shape)
            & broadcast_scalar(
                CACHE == 0 || (cache_row >= 0i32 && cache_row < max_seq),
                shape,
            );
        let offsets: Tile<i32, { [D] }> =
            broadcast_scalar((pid.1 * head_count + head) * head_dim, shape) + dim;
        let x: Tile<f32, { [D] }> = convert_tile(unsafe { load(input, offsets, valid) });
        let zero: Tile<f32, { [D] }> = constant(0.0f32, shape);
        let x: Tile<f32, { [D] }> = select(valid, x, zero);
        let sum: Tile<f32, { [] }> = reduce_sum(x * x, 0i32);
        let n: f32 = convert_scalar(head_dim);
        let inv: Tile<f32, { [] }> =
            rsqrt(sum / scalar_to_tile(n) + scalar_to_tile(eps), ftz::Disabled);
        let inv: Tile<f32, { [1] }> = inv.reshape(const_shape![1]);
        let inv: Tile<f32, { [D] }> = inv.broadcast(shape);
        let half = head_dim / 2i32;
        let first: Tile<bool, { [D] }> = lt_tile(dim, broadcast_scalar(half, shape));
        let paired: Tile<i32, { [D] }> = select(
            first,
            dim + broadcast_scalar(half, shape),
            dim - broadcast_scalar(half, shape),
        );
        let paired_offsets: Tile<i32, { [D] }> =
            broadcast_scalar((pid.1 * head_count + head) * head_dim, shape) + paired;
        let other: Tile<f32, { [D] }> = convert_tile(unsafe { load(input, paired_offsets, valid) });
        let w: Tile<f32, { [D] }> = convert_tile(unsafe { load(weight, dim, valid) });
        let other_w: Tile<f32, { [D] }> = convert_tile(unsafe { load(weight, paired, valid) });
        let pair: Tile<i32, { [D] }> = select(first, dim, dim - broadcast_scalar(half, shape));
        let trig_offsets: Tile<i32, { [D] }> = broadcast_scalar(table_row * half, shape) + pair;
        let c: Tile<f32, { [D] }> = convert_tile(unsafe { load(cos, trig_offsets, valid) });
        let s: Tile<f32, { [D] }> = convert_tile(unsafe { load(sin, trig_offsets, valid) });
        let current: Tile<f32, { [D] }> = x * inv * w;
        let other: Tile<f32, { [D] }> = other * inv * other_w;
        let rotated: Tile<f32, { [D] }> = current * c + select(first, zero - other * s, other * s);
        let result: Tile<f16, { [D] }> = convert_tile(rotated);
        unsafe { store(out, offsets, result, valid) };
        if CACHE != 0 && pid.0 >= heads {
            let cache_offsets: Tile<i32, { [D] }> =
                broadcast_scalar((head * max_seq + cache_row) * head_dim, shape) + dim;
            let v: Tile<f16, { [D] }> = unsafe { load(value, offsets, valid) };
            unsafe { store(k_cache, cache_offsets, result, valid) };
            unsafe { store(v_cache, cache_offsets, v, valid) };
        }
    }

    /// Computes LM-head scores and emits one f32 maximum and i32 vocabulary index per 64 rows.
    ///
    /// # Safety
    /// Weights must have shape `[vocab_size, hidden_size]` and hidden values shape `[1, hidden_size]`.
    /// Both dimensions must be positive, with one block and scratch element per 64 vocabulary rows.
    /// Scratch destinations must not alias inputs and must remain valid until completion and throughout graph replay.
    #[cutile::entry()]
    pub unsafe fn lm_head_argmax_f16(
        weights: &Tensor<f16, { [-1, -1] }>,
        hidden: &Tensor<f16, { [1, -1] }>,
        block_max: *mut f32,
        block_idx: *mut i32,
        vocab_size: i32,
        hidden_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = weights.partition(const_shape![64, 32]);
        let h_part = hidden.partition(const_shape![1, 32]);
        let mut acc: Tile<f32, { [64] }> = constant(0.0f32, const_shape![64]);
        let chunks = (hidden_size - 1i32) / 32i32 + 1i32;
        for chunk in 0i32..chunks {
            let w: Tile<f32, { [64, 32] }> = convert_tile(w_part.load([pid.0, chunk]));
            let h: Tile<f32, { [1, 32] }> = convert_tile(h_part.load([0i32, chunk]));
            let partial: Tile<f32, { [64] }> =
                reduce_sum(w * h.broadcast(const_shape![64, 32]), 1i32);
            acc = acc + partial;
        }
        let indices: Tile<i32, { [64] }> =
            iota(const_shape![64]) + broadcast_scalar(pid.0 * 64i32, const_shape![64]);
        let valid: Tile<bool, { [64] }> =
            lt_tile(indices, broadcast_scalar(vocab_size, const_shape![64]));
        let neg_inf: Tile<f32, { [64] }> = constant(f32::NEG_INFINITY, const_shape![64]);
        let values: Tile<f32, { [64] }> = select(valid, acc, neg_inf);
        let best: Tile<f32, { [1] }> = reduce_max(values, 0i32);
        let best: Tile<f32, { [1] }> = best.reshape(const_shape![1]);
        let matches: Tile<bool, { [64] }> = valid
            & cmpf(
                values,
                best.broadcast(const_shape![64]),
                predicate::Equal,
                cmp_ordering::Ordered,
            );
        let invalid: Tile<i32, { [64] }> = constant(2_147_483_647i32, const_shape![64]);
        let candidates: Tile<i32, { [64] }> = select(matches, indices, invalid);
        let index: Tile<i32, { [1] }> = reduce_min(candidates, 0i32);
        let index: Tile<i32, { [1] }> = index.reshape(const_shape![1]);
        let offset: Tile<i32, { [1] }> = broadcast_scalar(pid.0, const_shape![1]);
        let active: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        unsafe { store(block_max, offset, best, active) };
        unsafe { store(block_idx, offset, index, active) };
    }

    unsafe fn load<T: ElementType, const N: i32>(
        input: *mut T,
        offsets: Tile<i32, { [N] }>,
        valid: Tile<bool, { [N] }>,
    ) -> Tile<T, { [N] }> {
        let zero: Tile<i32, { [N] }> = constant(0i32, const_shape![N]);
        let safe_offsets = select(valid, offsets, zero);
        let ptrs: PointerTile<*mut T, { [] }> = pointer_to_tile(input);
        let ptrs: PointerTile<*mut T, { [1] }> = ptrs.reshape(const_shape![1]);
        let ptrs: PointerTile<*mut T, { [N] }> = ptrs.broadcast(const_shape![N]);
        let ptrs: PointerTile<*mut T, { [N] }> = ptrs.offset_tile(safe_offsets);
        let result: (Tile<T, { [N] }>, Token) = unsafe {
            load_ptr_tko(
                ptrs,
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

    unsafe fn store<T: ElementType, const N: i32>(
        out: *mut T,
        offsets: Tile<i32, { [N] }>,
        values: Tile<T, { [N] }>,
        valid: Tile<bool, { [N] }>,
    ) {
        let zero: Tile<i32, { [N] }> = constant(0i32, const_shape![N]);
        let safe_offsets = select(valid, offsets, zero);
        let ptrs: PointerTile<*mut T, { [] }> = pointer_to_tile(out);
        let ptrs: PointerTile<*mut T, { [1] }> = ptrs.reshape(const_shape![1]);
        let ptrs: PointerTile<*mut T, { [N] }> = ptrs.broadcast(const_shape![N]);
        let ptrs: PointerTile<*mut T, { [N] }> = ptrs.offset_tile(safe_offsets);
        unsafe {
            store_ptr_tko(
                ptrs,
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

pub use kernels::*;
