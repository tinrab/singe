#[doc(hidden)]
#[allow(unused_imports)]
#[path = "../../../-generated/cutile/fused.rs"]
mod fused_gen;

#[allow(ambiguous_glob_reexports)]
pub use fused_gen::kernels::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn rms_norm_add_f32<const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        residual_out: &mut Tensor<f32, { [1, B] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        residual: &Tensor<f32, { [-1, -1] }>,
        weight: &Tensor<f32, { [-1] }>,
        eps: f32,
        weight_offset: f32,
    ) {
        rms_norm_add_impl_f32::<B>(
            out,
            residual_out,
            input,
            residual,
            weight,
            eps,
            weight_offset,
        );
    }

    #[cutile::entry()]
    pub fn rms_norm_silu_mul_f32<const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        weight: &Tensor<f32, { [-1] }>,
        up: &Tensor<f32, { [-1, -1] }>,
        eps: f32,
        weight_offset: f32,
    ) {
        let normalized = rms_norm_tile_f32::<B>(out, input, weight, eps, weight_offset);
        let up_tile: Tile<f32, { [1, B] }> = load_tile_like(up, out);
        let shape = const_shape![1, B];
        let one = constant(1.0f32, shape);
        let zero = constant(0.0f32, shape);

        // Fused SwiGLU-style path normalizes the gate/input tile.
        // It applies SiLU and multiplies by the paired up-projection tile.
        let silu = normalized * (one / (one + exp(zero - normalized)));
        out.store(silu * up_tile);
    }

    #[cutile::entry()]
    pub fn rms_norm_gated_silu_f32<const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        gate: &Tensor<f32, { [-1, -1] }>,
        weight: &Tensor<f32, { [-1] }>,
        eps: f32,
        weight_offset: f32,
    ) {
        let normalized = rms_norm_tile_f32::<B>(out, input, weight, eps, weight_offset);
        let gate_tile: Tile<f32, { [1, B] }> = load_tile_like(gate, out);
        let shape = const_shape![1, B];
        let one = constant(1.0f32, shape);
        let zero = constant(0.0f32, shape);

        // Keep the RMS-normalized input in registers.
        // Apply SiLU to the separate gate tile.
        let silu_gate = gate_tile * (one / (one + exp(zero - gate_tile)));
        out.store(normalized * silu_gate);
    }

    #[cutile::entry()]
    pub unsafe fn exact_gelu_mul_f32(out: *mut f32, gate: *mut f32, up: *mut f32, len: i32) {
        let (offsets, mask) = vector_offsets_1024(len);
        let gate_tile = load_f32_vector_1024(gate, offsets, mask);
        let up_tile = load_f32_vector_1024(up, offsets, mask);
        let value = exact_gelu_tile(gate_tile) * up_tile;
        store_f32_vector_1024(out, offsets, value, mask);
    }

    #[cutile::entry()]
    pub unsafe fn exact_gelu_f32(out: *mut f32, input: *mut f32, len: i32) {
        let (offsets, mask) = vector_offsets_1024(len);
        let input_tile = load_f32_vector_1024(input, offsets, mask);
        let value = exact_gelu_tile(input_tile);
        store_f32_vector_1024(out, offsets, value, mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn exact_gelu_mul_f16(out: *mut f16, gate: *mut f16, up: *mut f16, len: i32) {
        let (offsets, mask) = vector_offsets_1024(len);
        let gate_tile = load_f16_vector_1024(gate, offsets, mask);
        let up_tile = load_f16_vector_1024(up, offsets, mask);
        let value = exact_gelu_tile(gate_tile) * up_tile;
        store_f16_vector_1024(out, offsets, value, mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn exact_gelu_f16(out: *mut f16, input: *mut f16, len: i32) {
        let (offsets, mask) = vector_offsets_1024(len);
        let input_tile = load_f16_vector_1024(input, offsets, mask);
        let value = exact_gelu_tile(input_tile);
        store_f16_vector_1024(out, offsets, value, mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn exact_gelu_mul_bf16(out: *mut bf16, gate: *mut bf16, up: *mut bf16, len: i32) {
        let (offsets, mask) = vector_offsets_1024(len);
        let gate_tile = load_bf16_vector_1024(gate, offsets, mask);
        let up_tile = load_bf16_vector_1024(up, offsets, mask);
        let value = exact_gelu_tile(gate_tile) * up_tile;
        store_bf16_vector_1024(out, offsets, value, mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn exact_gelu_bf16(out: *mut bf16, input: *mut bf16, len: i32) {
        let (offsets, mask) = vector_offsets_1024(len);
        let input_tile = load_bf16_vector_1024(input, offsets, mask);
        let value = exact_gelu_tile(input_tile);
        store_bf16_vector_1024(out, offsets, value, mask);
    }

    #[cutile::entry()]
    pub unsafe fn silu_and_mul_f32(out: *mut f32, input: *mut f32, hidden: i32, len: i32) {
        // Packed MLP activations are laid out as [gate, up].
        // The helper returns offsets for gate input, up input, and compact output.
        let (offsets, mask) = packed_silu_offsets(hidden, len);
        let gate_tile = load_f32_vector_1024(input, offsets.0, mask);
        let up_tile = load_f32_vector_1024(input, offsets.1, mask);
        store_f32_vector_1024(out, offsets.2, silu_tile(gate_tile) * up_tile, mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn silu_and_mul_f16(out: *mut f16, input: *mut f16, hidden: i32, len: i32) {
        let (offsets, mask) = packed_silu_offsets(hidden, len);
        let gate_tile = load_f16_vector_1024(input, offsets.0, mask);
        let up_tile = load_f16_vector_1024(input, offsets.1, mask);
        store_f16_vector_1024(out, offsets.2, silu_tile(gate_tile) * up_tile, mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn silu_and_mul_bf16(out: *mut bf16, input: *mut bf16, hidden: i32, len: i32) {
        let (offsets, mask) = packed_silu_offsets(hidden, len);
        let gate_tile = load_bf16_vector_1024(input, offsets.0, mask);
        let up_tile = load_bf16_vector_1024(input, offsets.1, mask);
        store_bf16_vector_1024(out, offsets.2, silu_tile(gate_tile) * up_tile, mask);
    }

    #[cutile::entry()]
    pub unsafe fn rms_norm_add_wide_partial_f32(
        residual_out: *mut f32,
        partial_sum: *mut f32,
        input: *mut f32,
        residual: *mut f32,
        cols: i32,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let lanes: Tile<i32, { [1024] }> = iota(tile_shape);

        // Wide rows are split into 1024-column chunks.
        // This stage writes input + residual and records each chunk's sum of squares.
        let chunk_start = pid.0 * 1024i32;
        let columns = broadcast_scalar(chunk_start, tile_shape) + lanes;
        let mask = cmpi(
            columns,
            broadcast_scalar(cols, tile_shape),
            predicate::LessThan,
        );
        let offsets = broadcast_scalar(pid.1 * cols, tile_shape) + columns;
        let input_tile = load_f32_vector_1024(input, offsets, mask);
        let residual_tile = load_f32_vector_1024(residual, offsets, mask);
        let sum = input_tile + residual_tile;
        store_f32_vector_1024(residual_out, offsets, sum, mask);
        let sum_square: Tile<f32, { [] }> = reduce_sum(sum * sum, 0i32);
        let first_lane = cmpi(
            lanes,
            broadcast_scalar(1i32, tile_shape),
            predicate::LessThan,
        );
        store_f32_vector_1024(
            partial_sum,
            broadcast_scalar(pid.1 * chunks + pid.0, tile_shape),
            sum_square.reshape(const_shape![1]).broadcast(tile_shape),
            first_lane,
        );
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn rms_norm_add_wide_partial_f16(
        residual_out: *mut f16,
        partial_sum: *mut f32,
        input: *mut f16,
        residual: *mut f16,
        cols: i32,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let lanes: Tile<i32, { [1024] }> = iota(tile_shape);
        let chunk_start = pid.0 * 1024i32;
        let columns = broadcast_scalar(chunk_start, tile_shape) + lanes;
        let mask = cmpi(
            columns,
            broadcast_scalar(cols, tile_shape),
            predicate::LessThan,
        );
        let offsets = broadcast_scalar(pid.1 * cols, tile_shape) + columns;
        let input_tile = load_f16_vector_1024(input, offsets, mask);
        let residual_tile = load_f16_vector_1024(residual, offsets, mask);
        let sum = input_tile + residual_tile;
        store_f16_vector_1024(residual_out, offsets, sum, mask);
        let sum_square: Tile<f32, { [] }> = reduce_sum(sum * sum, 0i32);
        let first_lane = cmpi(
            lanes,
            broadcast_scalar(1i32, tile_shape),
            predicate::LessThan,
        );
        store_f32_vector_1024(
            partial_sum,
            broadcast_scalar(pid.1 * chunks + pid.0, tile_shape),
            sum_square.reshape(const_shape![1]).broadcast(tile_shape),
            first_lane,
        );
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn rms_norm_add_wide_partial_bf16(
        residual_out: *mut bf16,
        partial_sum: *mut f32,
        input: *mut bf16,
        residual: *mut bf16,
        cols: i32,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let lanes: Tile<i32, { [1024] }> = iota(tile_shape);
        let chunk_start = pid.0 * 1024i32;
        let columns = broadcast_scalar(chunk_start, tile_shape) + lanes;
        let mask = cmpi(
            columns,
            broadcast_scalar(cols, tile_shape),
            predicate::LessThan,
        );
        let offsets = broadcast_scalar(pid.1 * cols, tile_shape) + columns;
        let input_tile = load_bf16_vector_1024(input, offsets, mask);
        let residual_tile = load_bf16_vector_1024(residual, offsets, mask);
        let sum = input_tile + residual_tile;
        store_bf16_vector_1024(residual_out, offsets, sum, mask);
        let sum_square: Tile<f32, { [] }> = reduce_sum(sum * sum, 0i32);
        let first_lane = cmpi(
            lanes,
            broadcast_scalar(1i32, tile_shape),
            predicate::LessThan,
        );
        store_f32_vector_1024(
            partial_sum,
            broadcast_scalar(pid.1 * chunks + pid.0, tile_shape),
            sum_square.reshape(const_shape![1]).broadcast(tile_shape),
            first_lane,
        );
    }

    #[cutile::entry()]
    pub unsafe fn rms_norm_add_wide_reduce_f32(
        row_sum: *mut f32,
        partial_sum: *mut f32,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let lanes: Tile<i32, { [1024] }> = iota(tile_shape);
        let mask = cmpi(
            lanes,
            broadcast_scalar(chunks, tile_shape),
            predicate::LessThan,
        );
        let offsets = broadcast_scalar(pid.0 * chunks, tile_shape) + lanes;
        let partial = load_f32_vector_1024(partial_sum, offsets, mask);
        let total: Tile<f32, { [] }> = reduce_sum(partial, 0i32);
        let first_lane = cmpi(
            lanes,
            broadcast_scalar(1i32, tile_shape),
            predicate::LessThan,
        );
        store_f32_vector_1024(
            row_sum,
            broadcast_scalar(pid.0, tile_shape),
            total.reshape(const_shape![1]).broadcast(tile_shape),
            first_lane,
        );
    }

    #[cutile::entry()]
    pub unsafe fn rms_norm_add_wide_normalize_f32(
        out: *mut f32,
        residual_out: *mut f32,
        row_sum: *mut f32,
        weight: *mut f32,
        cols: i32,
        len: i32,
        eps: f32,
        weight_offset: f32,
    ) {
        let (offsets, mask) = vector_offsets_1024(len);
        let row = offsets / broadcast_scalar(cols, const_shape![1024]);
        let column = offsets - row * broadcast_scalar(cols, const_shape![1024]);

        // Final wide RMS norm stage loads the residual sum and row sum.
        // It computes rsqrt(mean(square) + eps) and applies affine weight.
        let sum = load_f32_vector_1024(residual_out, offsets, mask);
        let weight_tile = load_f32_vector_1024(weight, column, mask);
        let total = load_f32_vector_1024(row_sum, row, mask);
        let cols_f32: Tile<f32, { [1024] }> =
            convert_tile(broadcast_scalar(cols, const_shape![1024]));
        let inv_rms = rsqrt(
            total / cols_f32 + broadcast_scalar(eps, const_shape![1024]),
            ftz::Disabled,
        );
        let value =
            sum * inv_rms * (weight_tile + broadcast_scalar(weight_offset, const_shape![1024]));
        store_f32_vector_1024(out, offsets, value, mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn rms_norm_add_wide_normalize_f16(
        out: *mut f16,
        residual_out: *mut f16,
        row_sum: *mut f32,
        weight: *mut f16,
        cols: i32,
        len: i32,
        eps: f32,
        weight_offset: f32,
    ) {
        let (offsets, mask) = vector_offsets_1024(len);
        let row = offsets / broadcast_scalar(cols, const_shape![1024]);
        let column = offsets - row * broadcast_scalar(cols, const_shape![1024]);
        let sum = load_f16_vector_1024(residual_out, offsets, mask);
        let weight_tile = load_f16_vector_1024(weight, column, mask);
        let total = load_f32_vector_1024(row_sum, row, mask);
        let cols_f32: Tile<f32, { [1024] }> =
            convert_tile(broadcast_scalar(cols, const_shape![1024]));
        let inv_rms = rsqrt(
            total / cols_f32 + broadcast_scalar(eps, const_shape![1024]),
            ftz::Disabled,
        );
        let value =
            sum * inv_rms * (weight_tile + broadcast_scalar(weight_offset, const_shape![1024]));
        store_f16_vector_1024(out, offsets, value, mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn rms_norm_add_wide_normalize_bf16(
        out: *mut bf16,
        residual_out: *mut bf16,
        row_sum: *mut f32,
        weight: *mut bf16,
        cols: i32,
        len: i32,
        eps: f32,
        weight_offset: f32,
    ) {
        let (offsets, mask) = vector_offsets_1024(len);
        let row = offsets / broadcast_scalar(cols, const_shape![1024]);
        let column = offsets - row * broadcast_scalar(cols, const_shape![1024]);
        let sum = load_bf16_vector_1024(residual_out, offsets, mask);
        let weight_tile = load_bf16_vector_1024(weight, column, mask);
        let total = load_f32_vector_1024(row_sum, row, mask);
        let cols_f32: Tile<f32, { [1024] }> =
            convert_tile(broadcast_scalar(cols, const_shape![1024]));
        let inv_rms = rsqrt(
            total / cols_f32 + broadcast_scalar(eps, const_shape![1024]),
            ftz::Disabled,
        );
        let value =
            sum * inv_rms * (weight_tile + broadcast_scalar(weight_offset, const_shape![1024]));
        store_bf16_vector_1024(out, offsets, value, mask);
    }

    #[cutile::entry()]
    pub unsafe fn rms_norm_gated_silu_f32_exact(
        out: *mut f32,
        input: *mut f32,
        gate: *mut f32,
        weight: *mut f32,
        cols: i32,
        eps: f32,
        weight_offset: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let columns: Tile<i32, { [1024] }> = iota(tile_shape);
        let mask = cmpi(
            columns,
            broadcast_scalar(cols, tile_shape),
            predicate::LessThan,
        );
        let row_offsets = broadcast_scalar(pid.0 * cols, tile_shape) + columns;

        let input_tile = load_f32_vector_1024(input, row_offsets, mask);
        let gate_tile = load_f32_vector_1024(gate, row_offsets, mask);
        let weight_tile = load_f32_vector_1024(weight, columns, mask);
        let variance_sum: Tile<f32, { [] }> = reduce_sum(input_tile * input_tile, 0i32);
        let cols_f32: Tile<f32, { [1] }> = convert_tile(broadcast_scalar(cols, const_shape![1]));
        let inv_rms = rsqrt(
            variance_sum.reshape(const_shape![1]) / cols_f32
                + broadcast_scalar(eps, const_shape![1]),
            ftz::Disabled,
        )
        .reshape(const_shape![1])
        .broadcast(tile_shape);
        let one = constant(1.0f32, tile_shape);
        let zero = constant(0.0f32, tile_shape);
        let silu_gate = gate_tile * (one / (one + exp(zero - gate_tile)));
        let value = input_tile
            * inv_rms
            * (weight_tile + broadcast_scalar(weight_offset, tile_shape))
            * silu_gate;
        store_f32_vector_1024(out, row_offsets, value, mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn rms_norm_gated_silu_f16_exact(
        out: *mut f16,
        input: *mut f16,
        gate: *mut f16,
        weight: *mut f16,
        cols: i32,
        eps: f32,
        weight_offset: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let columns: Tile<i32, { [1024] }> = iota(tile_shape);
        let mask = cmpi(
            columns,
            broadcast_scalar(cols, tile_shape),
            predicate::LessThan,
        );
        let row_offsets = broadcast_scalar(pid.0 * cols, tile_shape) + columns;

        let input_tile = load_f16_vector_1024(input, row_offsets, mask);
        let gate_tile = load_f16_vector_1024(gate, row_offsets, mask);
        let weight_tile = load_f16_vector_1024(weight, columns, mask);
        let variance_sum: Tile<f32, { [] }> = reduce_sum(input_tile * input_tile, 0i32);
        let cols_f32: Tile<f32, { [1] }> = convert_tile(broadcast_scalar(cols, const_shape![1]));
        let inv_rms = rsqrt(
            variance_sum.reshape(const_shape![1]) / cols_f32
                + broadcast_scalar(eps, const_shape![1]),
            ftz::Disabled,
        )
        .reshape(const_shape![1])
        .broadcast(tile_shape);
        let one = constant(1.0f32, tile_shape);
        let zero = constant(0.0f32, tile_shape);
        let silu_gate = gate_tile * (one / (one + exp(zero - gate_tile)));
        let value = input_tile
            * inv_rms
            * (weight_tile + broadcast_scalar(weight_offset, tile_shape))
            * silu_gate;
        store_f16_vector_1024(out, row_offsets, value, mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn rms_norm_gated_silu_bf16_exact(
        out: *mut bf16,
        input: *mut bf16,
        gate: *mut bf16,
        weight: *mut bf16,
        cols: i32,
        eps: f32,
        weight_offset: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let columns: Tile<i32, { [1024] }> = iota(tile_shape);
        let mask = cmpi(
            columns,
            broadcast_scalar(cols, tile_shape),
            predicate::LessThan,
        );
        let row_offsets = broadcast_scalar(pid.0 * cols, tile_shape) + columns;

        let input_tile = load_bf16_vector_1024(input, row_offsets, mask);
        let gate_tile = load_bf16_vector_1024(gate, row_offsets, mask);
        let weight_tile = load_bf16_vector_1024(weight, columns, mask);
        let variance_sum: Tile<f32, { [] }> = reduce_sum(input_tile * input_tile, 0i32);
        let cols_f32: Tile<f32, { [1] }> = convert_tile(broadcast_scalar(cols, const_shape![1]));
        let inv_rms = rsqrt(
            variance_sum.reshape(const_shape![1]) / cols_f32
                + broadcast_scalar(eps, const_shape![1]),
            ftz::Disabled,
        )
        .reshape(const_shape![1])
        .broadcast(tile_shape);
        let one = constant(1.0f32, tile_shape);
        let zero = constant(0.0f32, tile_shape);
        let silu_gate = gate_tile * (one / (one + exp(zero - gate_tile)));
        let value = input_tile
            * inv_rms
            * (weight_tile + broadcast_scalar(weight_offset, tile_shape))
            * silu_gate;
        store_bf16_vector_1024(out, row_offsets, value, mask);
    }

    #[cutile::entry()]
    pub unsafe fn dual_rms_norm_f32_exact(
        q_out: *mut f32,
        k_out: *mut f32,
        q: *mut f32,
        k: *mut f32,
        q_weight: *mut f32,
        k_weight: *mut f32,
        cols: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let columns: Tile<i32, { [1024] }> = iota(tile_shape);
        let mask = cmpi(
            columns,
            broadcast_scalar(cols, tile_shape),
            predicate::LessThan,
        );
        let row_offsets = broadcast_scalar(pid.0 * cols, tile_shape) + columns;
        let cols_f32: Tile<f32, { [1] }> = convert_tile(broadcast_scalar(cols, const_shape![1]));
        let eps_tile = broadcast_scalar(eps, const_shape![1]);

        let q_tile = load_f32_vector_1024(q, row_offsets, mask);
        let q_weight_tile = load_f32_vector_1024(q_weight, columns, mask);
        let q_variance_sum: Tile<f32, { [] }> = reduce_sum(q_tile * q_tile, 0i32);
        let q_inv_rms = rsqrt(
            q_variance_sum.reshape(const_shape![1]) / cols_f32 + eps_tile,
            ftz::Disabled,
        )
        .reshape(const_shape![1])
        .broadcast(tile_shape);
        store_f32_vector_1024(q_out, row_offsets, q_tile * q_inv_rms * q_weight_tile, mask);

        let k_tile = load_f32_vector_1024(k, row_offsets, mask);
        let k_weight_tile = load_f32_vector_1024(k_weight, columns, mask);
        let k_variance_sum: Tile<f32, { [] }> = reduce_sum(k_tile * k_tile, 0i32);
        let k_inv_rms = rsqrt(
            k_variance_sum.reshape(const_shape![1]) / cols_f32 + eps_tile,
            ftz::Disabled,
        )
        .reshape(const_shape![1])
        .broadcast(tile_shape);
        store_f32_vector_1024(k_out, row_offsets, k_tile * k_inv_rms * k_weight_tile, mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn dual_rms_norm_f16_exact(
        q_out: *mut f16,
        k_out: *mut f16,
        q: *mut f16,
        k: *mut f16,
        q_weight: *mut f16,
        k_weight: *mut f16,
        cols: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let columns: Tile<i32, { [1024] }> = iota(tile_shape);
        let mask = cmpi(
            columns,
            broadcast_scalar(cols, tile_shape),
            predicate::LessThan,
        );
        let row_offsets = broadcast_scalar(pid.0 * cols, tile_shape) + columns;
        let cols_f32: Tile<f32, { [1] }> = convert_tile(broadcast_scalar(cols, const_shape![1]));
        let eps_tile = broadcast_scalar(eps, const_shape![1]);

        let q_tile = load_f16_vector_1024(q, row_offsets, mask);
        let q_weight_tile = load_f16_vector_1024(q_weight, columns, mask);
        let q_variance_sum: Tile<f32, { [] }> = reduce_sum(q_tile * q_tile, 0i32);
        let q_inv_rms = rsqrt(
            q_variance_sum.reshape(const_shape![1]) / cols_f32 + eps_tile,
            ftz::Disabled,
        )
        .reshape(const_shape![1])
        .broadcast(tile_shape);
        store_f16_vector_1024(q_out, row_offsets, q_tile * q_inv_rms * q_weight_tile, mask);

        let k_tile = load_f16_vector_1024(k, row_offsets, mask);
        let k_weight_tile = load_f16_vector_1024(k_weight, columns, mask);
        let k_variance_sum: Tile<f32, { [] }> = reduce_sum(k_tile * k_tile, 0i32);
        let k_inv_rms = rsqrt(
            k_variance_sum.reshape(const_shape![1]) / cols_f32 + eps_tile,
            ftz::Disabled,
        )
        .reshape(const_shape![1])
        .broadcast(tile_shape);
        store_f16_vector_1024(k_out, row_offsets, k_tile * k_inv_rms * k_weight_tile, mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn dual_rms_norm_bf16_exact(
        q_out: *mut bf16,
        k_out: *mut bf16,
        q: *mut bf16,
        k: *mut bf16,
        q_weight: *mut bf16,
        k_weight: *mut bf16,
        cols: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let columns: Tile<i32, { [1024] }> = iota(tile_shape);
        let mask = cmpi(
            columns,
            broadcast_scalar(cols, tile_shape),
            predicate::LessThan,
        );
        let row_offsets = broadcast_scalar(pid.0 * cols, tile_shape) + columns;
        let cols_f32: Tile<f32, { [1] }> = convert_tile(broadcast_scalar(cols, const_shape![1]));
        let eps_tile = broadcast_scalar(eps, const_shape![1]);

        let q_tile = load_bf16_vector_1024(q, row_offsets, mask);
        let q_weight_tile = load_bf16_vector_1024(q_weight, columns, mask);
        let q_variance_sum: Tile<f32, { [] }> = reduce_sum(q_tile * q_tile, 0i32);
        let q_inv_rms = rsqrt(
            q_variance_sum.reshape(const_shape![1]) / cols_f32 + eps_tile,
            ftz::Disabled,
        )
        .reshape(const_shape![1])
        .broadcast(tile_shape);
        store_bf16_vector_1024(q_out, row_offsets, q_tile * q_inv_rms * q_weight_tile, mask);

        let k_tile = load_bf16_vector_1024(k, row_offsets, mask);
        let k_weight_tile = load_bf16_vector_1024(k_weight, columns, mask);
        let k_variance_sum: Tile<f32, { [] }> = reduce_sum(k_tile * k_tile, 0i32);
        let k_inv_rms = rsqrt(
            k_variance_sum.reshape(const_shape![1]) / cols_f32 + eps_tile,
            ftz::Disabled,
        )
        .reshape(const_shape![1])
        .broadcast(tile_shape);
        store_bf16_vector_1024(k_out, row_offsets, k_tile * k_inv_rms * k_weight_tile, mask);
    }

    #[cutile::entry()]
    pub unsafe fn rms_norm_residual_add_f32_exact(
        out: *mut f32,
        input: *mut f32,
        residual: *mut f32,
        weight: *mut f32,
        cols: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let columns: Tile<i32, { [1024] }> = iota(tile_shape);
        let mask = cmpi(
            columns,
            broadcast_scalar(cols, tile_shape),
            predicate::LessThan,
        );
        let row_offsets = broadcast_scalar(pid.0 * cols, tile_shape) + columns;
        let cols_f32: Tile<f32, { [1] }> = convert_tile(broadcast_scalar(cols, const_shape![1]));
        let eps_tile = broadcast_scalar(eps, const_shape![1]);

        let input_tile = load_f32_vector_1024(input, row_offsets, mask);
        let residual_tile = load_f32_vector_1024(residual, row_offsets, mask);
        let weight_tile = load_f32_vector_1024(weight, columns, mask);
        let variance_sum: Tile<f32, { [] }> = reduce_sum(input_tile * input_tile, 0i32);
        let inv_rms = rsqrt(
            variance_sum.reshape(const_shape![1]) / cols_f32 + eps_tile,
            ftz::Disabled,
        )
        .reshape(const_shape![1])
        .broadcast(tile_shape);
        store_f32_vector_1024(
            out,
            row_offsets,
            residual_tile + input_tile * inv_rms * weight_tile,
            mask,
        );
    }

    #[cutile::entry()]
    pub unsafe fn mhc_apply_residual_f32_n1(
        out: *mut f32,
        x: *mut f32,
        f_out: *mut f32,
        y: *mut f32,
        channels: i32,
        len: i32,
    ) {
        unsafe { mhc_apply_residual_f32_impl::<1>(out, x, f_out, y, channels, len) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_apply_residual_f32_n2(
        out: *mut f32,
        x: *mut f32,
        f_out: *mut f32,
        y: *mut f32,
        channels: i32,
        len: i32,
    ) {
        unsafe { mhc_apply_residual_f32_impl::<2>(out, x, f_out, y, channels, len) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_apply_residual_f32_n3(
        out: *mut f32,
        x: *mut f32,
        f_out: *mut f32,
        y: *mut f32,
        channels: i32,
        len: i32,
    ) {
        unsafe { mhc_apply_residual_f32_impl::<3>(out, x, f_out, y, channels, len) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_apply_residual_f32_n4(
        out: *mut f32,
        x: *mut f32,
        f_out: *mut f32,
        y: *mut f32,
        channels: i32,
        len: i32,
    ) {
        unsafe { mhc_apply_residual_f32_impl::<4>(out, x, f_out, y, channels, len) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_apply_residual_f32_n5(
        out: *mut f32,
        x: *mut f32,
        f_out: *mut f32,
        y: *mut f32,
        channels: i32,
        len: i32,
    ) {
        unsafe { mhc_apply_residual_f32_impl::<5>(out, x, f_out, y, channels, len) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_apply_residual_f32_n6(
        out: *mut f32,
        x: *mut f32,
        f_out: *mut f32,
        y: *mut f32,
        channels: i32,
        len: i32,
    ) {
        unsafe { mhc_apply_residual_f32_impl::<6>(out, x, f_out, y, channels, len) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_apply_residual_f32_n7(
        out: *mut f32,
        x: *mut f32,
        f_out: *mut f32,
        y: *mut f32,
        channels: i32,
        len: i32,
    ) {
        unsafe { mhc_apply_residual_f32_impl::<7>(out, x, f_out, y, channels, len) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_apply_residual_f32_n8(
        out: *mut f32,
        x: *mut f32,
        f_out: *mut f32,
        y: *mut f32,
        channels: i32,
        len: i32,
    ) {
        unsafe { mhc_apply_residual_f32_impl::<8>(out, x, f_out, y, channels, len) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_apply_residual_f32_n16(
        out: *mut f32,
        x: *mut f32,
        f_out: *mut f32,
        y: *mut f32,
        channels: i32,
        len: i32,
    ) {
        unsafe { mhc_apply_residual_f32_impl::<16>(out, x, f_out, y, channels, len) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_sinkhorn_f32_n1(y: *mut f32, batch: i32) {
        unsafe { mhc_sinkhorn_f32_impl::<1, 1>(y, batch) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_sinkhorn_f32_n2(y: *mut f32, batch: i32) {
        unsafe { mhc_sinkhorn_f32_impl::<2, 4>(y, batch) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_sinkhorn_f32_n3(y: *mut f32, batch: i32) {
        unsafe { mhc_sinkhorn_f32_impl::<3, 9>(y, batch) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_sinkhorn_f32_n4(y: *mut f32, batch: i32) {
        unsafe { mhc_sinkhorn_f32_impl::<4, 16>(y, batch) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_sinkhorn_f32_n5(y: *mut f32, batch: i32) {
        unsafe { mhc_sinkhorn_f32_impl::<5, 25>(y, batch) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_sinkhorn_f32_n6(y: *mut f32, batch: i32) {
        unsafe { mhc_sinkhorn_f32_impl::<6, 36>(y, batch) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_sinkhorn_f32_n7(y: *mut f32, batch: i32) {
        unsafe { mhc_sinkhorn_f32_impl::<7, 49>(y, batch) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_sinkhorn_f32_n8(y: *mut f32, batch: i32) {
        unsafe { mhc_sinkhorn_f32_impl::<8, 64>(y, batch) };
    }

    #[cutile::entry()]
    pub unsafe fn mhc_gemm_rms_scale_f32(
        y: *mut f32,
        r: *mut f32,
        x: *mut f32,
        w: *mut f32,
        bias: *mut f32,
        columns: i32,
        reduction: i32,
        n: i32,
        alpha_pre: f32,
        alpha_post: f32,
        alpha_res: f32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let row = safe_offsets / broadcast_scalar(columns, tile_shape);
        let column = safe_offsets - row * broadcast_scalar(columns, tile_shape);

        let mut dot: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut rms_sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let k = broadcast_scalar(reduction_index, tile_shape);
            let x_offsets = row * broadcast_scalar(reduction, tile_shape) + k;
            let w_offsets = k * broadcast_scalar(columns, tile_shape) + column;
            let x_values = load_vector(x, x_offsets, mask, 0.0f32);
            let w_values = load_vector(w, w_offsets, mask, 0.0f32);
            dot = dot + x_values * w_values;
            rms_sum = rms_sum + x_values * x_values;
        }

        let reduction_f32: Tile<f32, { [128] }> =
            convert_tile(broadcast_scalar(reduction, tile_shape));
        let rms = sqrt(
            rms_sum / reduction_f32,
            rounding::NearestEven,
            ftz::Disabled,
        );
        let bias_values = load_vector(bias, column, mask, 0.0f32);
        let pre_mask = cmpi(column, broadcast_scalar(n, tile_shape), predicate::LessThan);
        let post_lower = cmpi(
            column,
            broadcast_scalar(n - 1, tile_shape),
            predicate::GreaterThan,
        );
        let post_mask = cmpi(
            column,
            broadcast_scalar(2 * n, tile_shape),
            predicate::LessThan,
        ) & post_lower;
        let scale = select(
            pre_mask,
            broadcast_scalar(alpha_pre, tile_shape),
            select(
                post_mask,
                broadcast_scalar(alpha_post, tile_shape),
                broadcast_scalar(alpha_res, tile_shape),
            ),
        );
        let linear = dot * scale / rms + bias_values;
        let one = constant(1.0f32, tile_shape);
        let two = constant(2.0f32, tile_shape);
        let zero = constant(0.0f32, tile_shape);
        let sigmoid = one / (one + exp(zero - linear));
        let value = select(pre_mask, sigmoid, select(post_mask, two * sigmoid, linear));
        store_vector(y, offsets, value, mask);

        let first_column = cmpi(column, zero_offsets, predicate::Equal) & mask;
        store_vector(r, row, rms, first_column);
    }

    #[cutile::entry()]
    pub unsafe fn mhc_split_gemm_rms_f32(
        y_acc: *mut f32,
        r_acc: *mut f32,
        x: *mut f32,
        w: *mut f32,
        columns: i32,
        reduction: i32,
        split_k: i32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let per_split_len = broadcast_scalar(output_len / split_k, tile_shape);
        let split = safe_offsets / per_split_len;
        let local_offset = safe_offsets - split * per_split_len;
        let row_in_split = local_offset / broadcast_scalar(columns, tile_shape);
        let column = local_offset - row_in_split * broadcast_scalar(columns, tile_shape);

        let k_per_split = (reduction + split_k - 1) / split_k;
        let k_start = split * broadcast_scalar(k_per_split, tile_shape);
        let k_end_raw = k_start + broadcast_scalar(k_per_split, tile_shape);
        let reduction_tile = broadcast_scalar(reduction, tile_shape);
        let k_end = select(
            cmpi(k_end_raw, reduction_tile, predicate::LessThan),
            k_end_raw,
            reduction_tile,
        );

        let mut dot: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut rms_sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let k = broadcast_scalar(reduction_index, tile_shape);
            let in_split = cmpi(k, k_start, predicate::GreaterThanOrEqual)
                & cmpi(k, k_end, predicate::LessThan);
            let active = mask & in_split;
            let x_offsets = row_in_split * broadcast_scalar(reduction, tile_shape) + k;
            let w_offsets = k * broadcast_scalar(columns, tile_shape) + column;
            let x_values = load_vector(x, x_offsets, active, 0.0f32);
            let w_values = load_vector(w, w_offsets, active, 0.0f32);
            dot = dot + x_values * w_values;
            rms_sum = rms_sum + x_values * x_values;
        }

        store_vector(y_acc, offsets, dot, mask);
        store_vector(r_acc, offsets, rms_sum, mask);
    }

    #[cutile::entry()]
    pub unsafe fn mhc_finalize_scale_bias_sigmoid_f32(
        y: *mut f32,
        r: *mut f32,
        y_acc: *mut f32,
        r_acc: *mut f32,
        bias: *mut f32,
        columns: i32,
        reduction: i32,
        n: i32,
        split_k: i32,
        alpha_pre: f32,
        alpha_post: f32,
        alpha_res: f32,
        output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let row = safe_offsets / broadcast_scalar(columns, tile_shape);
        let column = safe_offsets - row * broadcast_scalar(columns, tile_shape);

        let mut dot: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut rms_sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for split_index in 0i32..split_k {
            let split_offset = broadcast_scalar(split_index * output_len, tile_shape) + offsets;
            dot = dot + load_vector(y_acc, split_offset, mask, 0.0f32);
            rms_sum = rms_sum + load_vector(r_acc, split_offset, mask, 0.0f32);
        }

        let reduction_f32: Tile<f32, { [128] }> =
            convert_tile(broadcast_scalar(reduction, tile_shape));
        let rms = sqrt(
            rms_sum / reduction_f32,
            rounding::NearestEven,
            ftz::Disabled,
        );
        let bias_values = load_vector(bias, column, mask, 0.0f32);
        let pre_mask = cmpi(column, broadcast_scalar(n, tile_shape), predicate::LessThan);
        let post_lower = cmpi(
            column,
            broadcast_scalar(n - 1, tile_shape),
            predicate::GreaterThan,
        );
        let post_mask = cmpi(
            column,
            broadcast_scalar(2 * n, tile_shape),
            predicate::LessThan,
        ) & post_lower;
        let scale = select(
            pre_mask,
            broadcast_scalar(alpha_pre, tile_shape),
            select(
                post_mask,
                broadcast_scalar(alpha_post, tile_shape),
                broadcast_scalar(alpha_res, tile_shape),
            ),
        );
        let linear = dot * scale / rms + bias_values;
        let one = constant(1.0f32, tile_shape);
        let two = constant(2.0f32, tile_shape);
        let zero = constant(0.0f32, tile_shape);
        let sigmoid = one / (one + exp(zero - linear));
        let value = select(pre_mask, sigmoid, select(post_mask, two * sigmoid, linear));
        store_vector(y, offsets, value, mask);

        let first_column = cmpi(column, zero_offsets, predicate::Equal) & mask;
        store_vector(r, row, rms, first_column);
    }

    unsafe fn mhc_sinkhorn_f32_impl<const N: i32, const TOTAL: i32>(y: *mut f32, batch: i32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> = iota(tile_shape);
        let total_tile = broadcast_scalar(TOTAL, tile_shape);
        let mask = cmpi(offsets, total_tile, predicate::LessThan);
        let zero_offsets = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let row = safe_offsets / broadcast_scalar(N, tile_shape);
        let col = safe_offsets - row * broadcast_scalar(N, tile_shape);
        let y_row_stride = N * (N + 2);
        let base = broadcast_scalar(pid.0 * y_row_stride + 2 * N, tile_shape);
        let matrix_offsets = base + safe_offsets;
        let one = constant(1.0f32, tile_shape);
        let zero = constant(0.0f32, tile_shape);
        let mut mat = exp(load_vector(y, matrix_offsets, mask, 0.0f32));

        for _ in 0..20 {
            for row_index in 0..N {
                let row_mask = cmpi(
                    row,
                    broadcast_scalar(row_index, tile_shape),
                    predicate::Equal,
                ) & mask;
                let row_sum: Tile<f32, { [] }> = reduce_sum(select(row_mask, mat, zero), 0i32);
                let row_sum = row_sum.reshape(const_shape![1]).broadcast(tile_shape);
                mat = select(row_mask, mat / row_sum, mat);
            }
            for col_index in 0..N {
                let col_mask = cmpi(
                    col,
                    broadcast_scalar(col_index, tile_shape),
                    predicate::Equal,
                ) & mask;
                let col_sum: Tile<f32, { [] }> = reduce_sum(select(col_mask, mat, zero), 0i32);
                let col_sum = col_sum.reshape(const_shape![1]).broadcast(tile_shape);
                mat = select(col_mask, mat / col_sum, mat);
            }
        }
        store_vector(y, matrix_offsets, select(mask, mat, one), mask);

        let _ = batch;
    }

    unsafe fn mhc_apply_residual_f32_impl<const N: i32>(
        out: *mut f32,
        x: *mut f32,
        f_out: *mut f32,
        y: *mut f32,
        channels: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets_1024(len);
        let tile_shape = const_shape![1024];
        let zero_offsets = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let channels_tile = broadcast_scalar(channels, tile_shape);
        let n_tile = broadcast_scalar(N, tile_shape);
        let token_channels = n_tile * channels_tile;
        let batch_index = safe_offsets / token_channels;
        let batch_offset = batch_index * token_channels;
        let token_channel_offset = safe_offsets - batch_offset;
        let token = token_channel_offset / channels_tile;
        let channel = token_channel_offset - token * channels_tile;

        let y_row_stride = N * (N + 2);
        let y_row_stride_tile = broadcast_scalar(y_row_stride, tile_shape);
        let y_post_base = batch_index * y_row_stride_tile + broadcast_scalar(N, tile_shape);
        let y_res_base = batch_index * y_row_stride_tile + broadcast_scalar(2 * N, tile_shape);
        let f_offsets = batch_index * channels_tile + channel;
        let y_post = load_f32_vector_1024(y, y_post_base + token, mask);
        let f_tile = load_f32_vector_1024(f_out, f_offsets, mask);
        let mut sum = y_post * f_tile;

        for j in 0..N {
            let j_tile = broadcast_scalar(j, tile_shape);
            let x_offsets = batch_index * token_channels + j_tile * channels_tile + channel;
            let y_res_offsets = y_res_base + token * n_tile + j_tile;
            let x_tile = load_f32_vector_1024(x, x_offsets, mask);
            let y_res = load_f32_vector_1024(y, y_res_offsets, mask);
            sum = sum + y_res * x_tile;
        }
        store_f32_vector_1024(out, offsets, sum, mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn rms_norm_residual_add_f16_exact(
        out: *mut f16,
        input: *mut f16,
        residual: *mut f16,
        weight: *mut f16,
        cols: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let columns: Tile<i32, { [1024] }> = iota(tile_shape);
        let mask = cmpi(
            columns,
            broadcast_scalar(cols, tile_shape),
            predicate::LessThan,
        );
        let row_offsets = broadcast_scalar(pid.0 * cols, tile_shape) + columns;
        let cols_f32: Tile<f32, { [1] }> = convert_tile(broadcast_scalar(cols, const_shape![1]));
        let eps_tile = broadcast_scalar(eps, const_shape![1]);

        let input_tile = load_f16_vector_1024(input, row_offsets, mask);
        let residual_tile = load_f16_vector_1024(residual, row_offsets, mask);
        let weight_tile = load_f16_vector_1024(weight, columns, mask);
        let variance_sum: Tile<f32, { [] }> = reduce_sum(input_tile * input_tile, 0i32);
        let inv_rms = rsqrt(
            variance_sum.reshape(const_shape![1]) / cols_f32 + eps_tile,
            ftz::Disabled,
        )
        .reshape(const_shape![1])
        .broadcast(tile_shape);
        store_f16_vector_1024(
            out,
            row_offsets,
            residual_tile + input_tile * inv_rms * weight_tile,
            mask,
        );
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn rms_norm_residual_add_bf16_exact(
        out: *mut bf16,
        input: *mut bf16,
        residual: *mut bf16,
        weight: *mut bf16,
        cols: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let columns: Tile<i32, { [1024] }> = iota(tile_shape);
        let mask = cmpi(
            columns,
            broadcast_scalar(cols, tile_shape),
            predicate::LessThan,
        );
        let row_offsets = broadcast_scalar(pid.0 * cols, tile_shape) + columns;
        let cols_f32: Tile<f32, { [1] }> = convert_tile(broadcast_scalar(cols, const_shape![1]));
        let eps_tile = broadcast_scalar(eps, const_shape![1]);

        let input_tile = load_bf16_vector_1024(input, row_offsets, mask);
        let residual_tile = load_bf16_vector_1024(residual, row_offsets, mask);
        let weight_tile = load_bf16_vector_1024(weight, columns, mask);
        let variance_sum: Tile<f32, { [] }> = reduce_sum(input_tile * input_tile, 0i32);
        let inv_rms = rsqrt(
            variance_sum.reshape(const_shape![1]) / cols_f32 + eps_tile,
            ftz::Disabled,
        )
        .reshape(const_shape![1])
        .broadcast(tile_shape);
        store_bf16_vector_1024(
            out,
            row_offsets,
            residual_tile + input_tile * inv_rms * weight_tile,
            mask,
        );
    }

    #[cutile::entry()]
    pub fn rms_norm_add_f16<const B: i32>(
        out: &mut Tensor<f16, { [1, B] }>,
        residual_out: &mut Tensor<f16, { [1, B] }>,
        input: &Tensor<f16, { [-1, -1] }>,
        residual: &Tensor<f16, { [-1, -1] }>,
        weight: &Tensor<f16, { [-1] }>,
        eps: f32,
        weight_offset: f32,
    ) {
        rms_norm_add_impl_f16::<B>(
            out,
            residual_out,
            input,
            residual,
            weight,
            eps,
            weight_offset,
        );
    }

    #[cutile::entry()]
    pub fn rms_norm_silu_mul_f16<const B: i32>(
        out: &mut Tensor<f16, { [1, B] }>,
        input: &Tensor<f16, { [-1, -1] }>,
        weight: &Tensor<f16, { [-1] }>,
        up: &Tensor<f16, { [-1, -1] }>,
        eps: f32,
        weight_offset: f32,
    ) {
        let normalized = rms_norm_tile_f16::<B>(out, input, weight, eps, weight_offset);
        let up_raw: Tile<f16, { [1, B] }> = load_tile_like(up, out);
        let up_tile: Tile<f32, { [1, B] }> = convert_tile(up_raw);
        let shape = const_shape![1, B];
        let one = constant(1.0f32, shape);
        let zero = constant(0.0f32, shape);
        let silu = normalized * (one / (one + exp(zero - normalized)));
        let value: Tile<f16, { [1, B] }> = convert_tile(silu * up_tile);
        out.store(value);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub fn rms_norm_add_bf16<const B: i32>(
        out: &mut Tensor<bf16, { [1, B] }>,
        residual_out: &mut Tensor<bf16, { [1, B] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
        residual: &Tensor<bf16, { [-1, -1] }>,
        weight: &Tensor<bf16, { [-1] }>,
        eps: f32,
        weight_offset: f32,
    ) {
        rms_norm_add_impl_bf16::<B>(
            out,
            residual_out,
            input,
            residual,
            weight,
            eps,
            weight_offset,
        );
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub fn rms_norm_silu_mul_bf16<const B: i32>(
        out: &mut Tensor<bf16, { [1, B] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
        weight: &Tensor<bf16, { [-1] }>,
        up: &Tensor<bf16, { [-1, -1] }>,
        eps: f32,
        weight_offset: f32,
    ) {
        let normalized = rms_norm_tile_bf16::<B>(out, input, weight, eps, weight_offset);
        let up_raw: Tile<bf16, { [1, B] }> = load_tile_like(up, out);
        let up_tile: Tile<f32, { [1, B] }> = convert_tile(up_raw);
        let shape = const_shape![1, B];
        let one = constant(1.0f32, shape);
        let zero = constant(0.0f32, shape);
        let silu = normalized * (one / (one + exp(zero - normalized)));
        let value: Tile<bf16, { [1, B] }> = convert_tile(silu * up_tile);
        out.store(value);
    }

    #[cutile::entry()]
    pub fn rms_norm_add_f64<const B: i32>(
        out: &mut Tensor<f64, { [1, B] }>,
        residual_out: &mut Tensor<f64, { [1, B] }>,
        input: &Tensor<f64, { [-1, -1] }>,
        residual: &Tensor<f64, { [-1, -1] }>,
        weight: &Tensor<f64, { [-1] }>,
        eps: f64,
        weight_offset: f64,
    ) {
        rms_norm_add_impl_f64::<B>(
            out,
            residual_out,
            input,
            residual,
            weight,
            eps,
            weight_offset,
        );
    }

    #[cutile::entry()]
    pub fn rms_norm_silu_mul_f64<const B: i32>(
        out: &mut Tensor<f64, { [1, B] }>,
        input: &Tensor<f64, { [-1, -1] }>,
        weight: &Tensor<f64, { [-1] }>,
        up: &Tensor<f64, { [-1, -1] }>,
        eps: f64,
        weight_offset: f64,
    ) {
        let normalized = rms_norm_tile_f64::<B>(out, input, weight, eps, weight_offset);
        let up_tile: Tile<f64, { [1, B] }> = load_tile_like(up, out);
        let shape = const_shape![1, B];
        let one = constant(1.0f64, shape);
        let zero = constant(0.0f64, shape);
        let silu = normalized * (one / (one + exp(zero - normalized)));
        out.store(silu * up_tile);
    }

    #[cutile::entry()]
    pub unsafe fn kv_cache_update_f32(
        k_cache: *mut f32,
        v_cache: *mut f32,
        new_k: *mut f32,
        new_v: *mut f32,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        position_start: i32,
        len: i32,
    ) {
        kv_cache_update_impl(
            k_cache,
            v_cache,
            new_k,
            new_v,
            seq_len,
            heads,
            head_dim,
            max_seq,
            position_start,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn kv_cache_update_dynpos_f32(
        k_cache: *mut f32,
        v_cache: *mut f32,
        new_k: *mut f32,
        new_v: *mut f32,
        position_start: *mut u32,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        len: i32,
    ) {
        kv_cache_update_dynpos_impl(
            k_cache,
            v_cache,
            new_k,
            new_v,
            position_start,
            seq_len,
            heads,
            head_dim,
            max_seq,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn kv_cache_update_f16(
        k_cache: *mut f16,
        v_cache: *mut f16,
        new_k: *mut f16,
        new_v: *mut f16,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        position_start: i32,
        len: i32,
    ) {
        kv_cache_update_impl(
            k_cache,
            v_cache,
            new_k,
            new_v,
            seq_len,
            heads,
            head_dim,
            max_seq,
            position_start,
            len,
        );
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn bshd_rope_qk_cache_update_f16(
        q_out: *mut f16,
        k_out: *mut f16,
        k_cache: *mut f16,
        v_cache: *mut f16,
        q_input: *mut f16,
        k_input: *mut f16,
        v_input: *mut f16,
        cos: *mut f32,
        sin: *mut f32,
        seq_len: i32,
        q_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        max_seq: i32,
        position_start: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets(len);
        bshd_rope_q_cache_update_impl(
            q_out, q_input, cos, sin, offsets, mask, seq_len, q_heads, head_dim,
        );
        bshd_rope_k_cache_update_impl(
            k_out,
            k_cache,
            k_input,
            cos,
            sin,
            offsets,
            mask,
            seq_len,
            kv_heads,
            head_dim,
            max_seq,
            position_start,
        );
        v_cache_update_impl(
            v_cache,
            v_input,
            offsets,
            mask,
            seq_len,
            kv_heads,
            head_dim,
            max_seq,
            position_start,
        );
    }

    #[cutile::entry()]
    pub unsafe fn kv_cache_update_dynpos_f16(
        k_cache: *mut f16,
        v_cache: *mut f16,
        new_k: *mut f16,
        new_v: *mut f16,
        position_start: *mut u32,
        _seq_len: i32,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets(len);
        let position_start_tile = load_u32_position_as_i32_tile(position_start);
        kv_cache_update_with_position_impl(
            k_cache,
            v_cache,
            new_k,
            new_v,
            heads,
            head_dim,
            max_seq,
            len,
            position_start_tile,
            offsets,
            mask,
        );
    }

    #[cutile::entry()]
    pub unsafe fn kv_cache_update_f64(
        k_cache: *mut f64,
        v_cache: *mut f64,
        new_k: *mut f64,
        new_v: *mut f64,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        position_start: i32,
        len: i32,
    ) {
        kv_cache_update_impl(
            k_cache,
            v_cache,
            new_k,
            new_v,
            seq_len,
            heads,
            head_dim,
            max_seq,
            position_start,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn kv_cache_update_dynpos_f64(
        k_cache: *mut f64,
        v_cache: *mut f64,
        new_k: *mut f64,
        new_v: *mut f64,
        position_start: *mut u32,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        len: i32,
    ) {
        kv_cache_update_dynpos_impl(
            k_cache,
            v_cache,
            new_k,
            new_v,
            position_start,
            seq_len,
            heads,
            head_dim,
            max_seq,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn kv_cache_compact_f32(
        k_cache: *mut f32,
        v_cache: *mut f32,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        source_start: i32,
        token_count: i32,
        len: i32,
    ) {
        kv_cache_compact_impl(
            k_cache,
            v_cache,
            heads,
            head_dim,
            max_seq,
            source_start,
            token_count,
            len,
            0.0f32,
        );
    }

    #[cutile::entry()]
    pub unsafe fn kv_cache_compact_f16(
        k_cache: *mut f16,
        v_cache: *mut f16,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        source_start: i32,
        token_count: i32,
        len: i32,
    ) {
        kv_cache_compact_impl(
            k_cache,
            v_cache,
            heads,
            head_dim,
            max_seq,
            source_start,
            token_count,
            len,
            f16::from_f32(0.0),
        );
    }

    #[cutile::entry()]
    pub unsafe fn kv_cache_compact_f64(
        k_cache: *mut f64,
        v_cache: *mut f64,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        source_start: i32,
        token_count: i32,
        len: i32,
    ) {
        kv_cache_compact_impl(
            k_cache,
            v_cache,
            heads,
            head_dim,
            max_seq,
            source_start,
            token_count,
            len,
            0.0f64,
        );
    }

    #[cutile::entry()]
    pub unsafe fn rope_qk_cache_update_f32(
        q_out: *mut f32,
        k_out: *mut f32,
        k_cache: *mut f32,
        q_input: *mut f32,
        k_input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        _q_output_dim0: i32,
        q_output_dim1: i32,
        q_output_dim2: i32,
        q_output_dim3: i32,
        q_input_stride0: i32,
        q_input_stride1: i32,
        q_input_stride2: i32,
        q_input_stride3: i32,
        _k_output_dim0: i32,
        k_output_dim1: i32,
        k_output_dim2: i32,
        k_output_dim3: i32,
        k_input_stride0: i32,
        k_input_stride1: i32,
        k_input_stride2: i32,
        k_input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        cache_max_seq: i32,
        cache_position_start: i32,
        q_len: i32,
        k_len: i32,
    ) {
        rotary_interleaved_f32(
            q_out,
            q_input,
            cos,
            sin,
            q_output_dim1,
            q_output_dim2,
            q_output_dim3,
            q_input_stride0,
            q_input_stride1,
            q_input_stride2,
            q_input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            q_len,
        );
        rotary_interleaved_cache_f32(
            k_out,
            k_cache,
            k_input,
            cos,
            sin,
            k_output_dim1,
            k_output_dim2,
            k_output_dim3,
            k_input_stride0,
            k_input_stride1,
            k_input_stride2,
            k_input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            cache_max_seq,
            cache_position_start,
            k_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn rope_qk_cache_update_f16(
        q_out: *mut f16,
        k_out: *mut f16,
        k_cache: *mut f16,
        q_input: *mut f16,
        k_input: *mut f16,
        cos: *mut f16,
        sin: *mut f16,
        _q_output_dim0: i32,
        q_output_dim1: i32,
        q_output_dim2: i32,
        q_output_dim3: i32,
        q_input_stride0: i32,
        q_input_stride1: i32,
        q_input_stride2: i32,
        q_input_stride3: i32,
        _k_output_dim0: i32,
        k_output_dim1: i32,
        k_output_dim2: i32,
        k_output_dim3: i32,
        k_input_stride0: i32,
        k_input_stride1: i32,
        k_input_stride2: i32,
        k_input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        cache_max_seq: i32,
        cache_position_start: i32,
        q_len: i32,
        k_len: i32,
    ) {
        rotary_interleaved_f16(
            q_out,
            q_input,
            cos,
            sin,
            q_output_dim1,
            q_output_dim2,
            q_output_dim3,
            q_input_stride0,
            q_input_stride1,
            q_input_stride2,
            q_input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            q_len,
        );
        rotary_interleaved_cache_f16(
            k_out,
            k_cache,
            k_input,
            cos,
            sin,
            k_output_dim1,
            k_output_dim2,
            k_output_dim3,
            k_input_stride0,
            k_input_stride1,
            k_input_stride2,
            k_input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            cache_max_seq,
            cache_position_start,
            k_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn rope_qk_cache_update_f64(
        q_out: *mut f64,
        k_out: *mut f64,
        k_cache: *mut f64,
        q_input: *mut f64,
        k_input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        _q_output_dim0: i32,
        q_output_dim1: i32,
        q_output_dim2: i32,
        q_output_dim3: i32,
        q_input_stride0: i32,
        q_input_stride1: i32,
        q_input_stride2: i32,
        q_input_stride3: i32,
        _k_output_dim0: i32,
        k_output_dim1: i32,
        k_output_dim2: i32,
        k_output_dim3: i32,
        k_input_stride0: i32,
        k_input_stride1: i32,
        k_input_stride2: i32,
        k_input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        cache_max_seq: i32,
        cache_position_start: i32,
        q_len: i32,
        k_len: i32,
    ) {
        rotary_interleaved_f64(
            q_out,
            q_input,
            cos,
            sin,
            q_output_dim1,
            q_output_dim2,
            q_output_dim3,
            q_input_stride0,
            q_input_stride1,
            q_input_stride2,
            q_input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            q_len,
        );
        rotary_interleaved_cache_f64(
            k_out,
            k_cache,
            k_input,
            cos,
            sin,
            k_output_dim1,
            k_output_dim2,
            k_output_dim3,
            k_input_stride0,
            k_input_stride1,
            k_input_stride2,
            k_input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            cache_max_seq,
            cache_position_start,
            k_len,
        );
    }

    fn rms_norm_add_impl_f32<const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        residual_out: &mut Tensor<f32, { [1, B] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        residual: &Tensor<f32, { [-1, -1] }>,
        weight: &Tensor<f32, { [-1] }>,
        eps: f32,
        weight_offset: f32,
    ) {
        let input_tile: Tile<f32, { [1, B] }> = load_tile_like(input, out);
        let residual_tile: Tile<f32, { [1, B] }> = load_tile_like(residual, out);

        // Residual-add RMS norm stores the residual sum for the caller.
        // It reuses the same sum tile for normalization.
        let sum = input_tile + residual_tile;
        residual_out.store(sum);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let value = rms_norm_values_f32::<B>(out, sum, weight, eps, weight_offset, input_shape[1]);
        out.store(value);
    }

    fn rms_norm_add_impl_f16<const B: i32>(
        out: &mut Tensor<f16, { [1, B] }>,
        residual_out: &mut Tensor<f16, { [1, B] }>,
        input: &Tensor<f16, { [-1, -1] }>,
        residual: &Tensor<f16, { [-1, -1] }>,
        weight: &Tensor<f16, { [-1] }>,
        eps: f32,
        weight_offset: f32,
    ) {
        let input_raw: Tile<f16, { [1, B] }> = load_tile_like(input, out);
        let residual_raw: Tile<f16, { [1, B] }> = load_tile_like(residual, out);
        let input_tile: Tile<f32, { [1, B] }> = convert_tile(input_raw);
        let residual_tile: Tile<f32, { [1, B] }> = convert_tile(residual_raw);
        let sum = input_tile + residual_tile;
        let residual_value: Tile<f16, { [1, B] }> = convert_tile(sum);
        residual_out.store(residual_value);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let value: Tile<f16, { [1, B] }> = convert_tile(rms_norm_values_f16::<B>(
            out,
            sum,
            weight,
            eps,
            weight_offset,
            input_shape[1],
        ));
        out.store(value);
    }

    #[cfg(feature = "dtype-bf16")]
    fn rms_norm_add_impl_bf16<const B: i32>(
        out: &mut Tensor<bf16, { [1, B] }>,
        residual_out: &mut Tensor<bf16, { [1, B] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
        residual: &Tensor<bf16, { [-1, -1] }>,
        weight: &Tensor<bf16, { [-1] }>,
        eps: f32,
        weight_offset: f32,
    ) {
        let input_raw: Tile<bf16, { [1, B] }> = load_tile_like(input, out);
        let residual_raw: Tile<bf16, { [1, B] }> = load_tile_like(residual, out);
        let input_tile: Tile<f32, { [1, B] }> = convert_tile(input_raw);
        let residual_tile: Tile<f32, { [1, B] }> = convert_tile(residual_raw);
        let sum = input_tile + residual_tile;
        let residual_value: Tile<bf16, { [1, B] }> = convert_tile(sum);
        residual_out.store(residual_value);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let value: Tile<bf16, { [1, B] }> = convert_tile(rms_norm_values_bf16::<B>(
            out,
            sum,
            weight,
            eps,
            weight_offset,
            input_shape[1],
        ));
        out.store(value);
    }

    fn rms_norm_add_impl_f64<const B: i32>(
        out: &mut Tensor<f64, { [1, B] }>,
        residual_out: &mut Tensor<f64, { [1, B] }>,
        input: &Tensor<f64, { [-1, -1] }>,
        residual: &Tensor<f64, { [-1, -1] }>,
        weight: &Tensor<f64, { [-1] }>,
        eps: f64,
        weight_offset: f64,
    ) {
        let input_tile: Tile<f64, { [1, B] }> = load_tile_like(input, out);
        let residual_tile: Tile<f64, { [1, B] }> = load_tile_like(residual, out);
        let sum = input_tile + residual_tile;
        residual_out.store(sum);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let value = rms_norm_values_f64::<B>(out, sum, weight, eps, weight_offset, input_shape[1]);
        out.store(value);
    }

    fn rms_norm_tile_f32<const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        weight: &Tensor<f32, { [-1] }>,
        eps: f32,
        weight_offset: f32,
    ) -> Tile<f32, { [1, B] }> {
        let input_tile: Tile<f32, { [1, B] }> = load_tile_like(input, out);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        rms_norm_values_f32::<B>(out, input_tile, weight, eps, weight_offset, input_shape[1])
    }

    fn rms_norm_tile_f16<const B: i32>(
        out: &mut Tensor<f16, { [1, B] }>,
        input: &Tensor<f16, { [-1, -1] }>,
        weight: &Tensor<f16, { [-1] }>,
        eps: f32,
        weight_offset: f32,
    ) -> Tile<f32, { [1, B] }> {
        let input_raw: Tile<f16, { [1, B] }> = load_tile_like(input, out);
        let input_tile: Tile<f32, { [1, B] }> = convert_tile(input_raw);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        rms_norm_values_f16::<B>(out, input_tile, weight, eps, weight_offset, input_shape[1])
    }

    #[cfg(feature = "dtype-bf16")]
    fn rms_norm_tile_bf16<const B: i32>(
        out: &mut Tensor<bf16, { [1, B] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
        weight: &Tensor<bf16, { [-1] }>,
        eps: f32,
        weight_offset: f32,
    ) -> Tile<f32, { [1, B] }> {
        let input_raw: Tile<bf16, { [1, B] }> = load_tile_like(input, out);
        let input_tile: Tile<f32, { [1, B] }> = convert_tile(input_raw);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        rms_norm_values_bf16::<B>(out, input_tile, weight, eps, weight_offset, input_shape[1])
    }

    fn rms_norm_tile_f64<const B: i32>(
        out: &mut Tensor<f64, { [1, B] }>,
        input: &Tensor<f64, { [-1, -1] }>,
        weight: &Tensor<f64, { [-1] }>,
        eps: f64,
        weight_offset: f64,
    ) -> Tile<f64, { [1, B] }> {
        let input_tile: Tile<f64, { [1, B] }> = load_tile_like(input, out);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        rms_norm_values_f64::<B>(out, input_tile, weight, eps, weight_offset, input_shape[1])
    }

    fn rms_norm_values_f32<const B: i32>(
        _out: &mut Tensor<f32, { [1, B] }>,
        input_tile: Tile<f32, { [1, B] }>,
        weight: &Tensor<f32, { [-1] }>,
        eps: f32,
        weight_offset: f32,
        cols: i32,
    ) -> Tile<f32, { [1, B] }> {
        let tile_shape = const_shape![1, B];
        let cols_f32: f32 = convert_scalar(cols);
        let divisor = broadcast_scalar(cols_f32, const_shape![1]);

        // RMS norm normalizes by rsqrt(mean(x^2) + eps).
        // It then applies the learned weight plus optional offset.
        let mean_square_sum: Tile<f32, { [1] }> = reduce_sum(input_tile * input_tile, 1i32);
        let mean_square = mean_square_sum / divisor;
        let one_row: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
        let inv_rms = one_row
            / sqrt(
                mean_square + broadcast_scalar(eps, const_shape![1]),
                rounding::NearestEven,
                ftz::Disabled,
            );
        let inv_rms = inv_rms.reshape(const_shape![1, 1]).broadcast(tile_shape);
        let weight_raw: Tile<f32, { [B] }> = weight.partition(const_shape![B]).load([0i32]);
        let weight_tile =
            weight_raw.reshape(tile_shape) + broadcast_scalar(weight_offset, tile_shape);
        input_tile * inv_rms * weight_tile
    }

    fn rms_norm_values_f16<const B: i32>(
        _out: &mut Tensor<f16, { [1, B] }>,
        input_tile: Tile<f32, { [1, B] }>,
        weight: &Tensor<f16, { [-1] }>,
        eps: f32,
        weight_offset: f32,
        cols: i32,
    ) -> Tile<f32, { [1, B] }> {
        let tile_shape = const_shape![1, B];
        let cols_f32: f32 = convert_scalar(cols);
        let divisor = broadcast_scalar(cols_f32, const_shape![1]);
        let mean_square_sum: Tile<f32, { [1] }> = reduce_sum(input_tile * input_tile, 1i32);
        let mean_square = mean_square_sum / divisor;
        let one_row: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
        let inv_rms = one_row
            / sqrt(
                mean_square + broadcast_scalar(eps, const_shape![1]),
                rounding::NearestEven,
                ftz::Disabled,
            );
        let inv_rms = inv_rms.reshape(const_shape![1, 1]).broadcast(tile_shape);
        let weight_raw: Tile<f16, { [B] }> = weight.partition(const_shape![B]).load([0i32]);
        let weight_tile: Tile<f32, { [1, B] }> = convert_tile(weight_raw.reshape(tile_shape));
        let weight_tile = weight_tile + broadcast_scalar(weight_offset, tile_shape);
        input_tile * inv_rms * weight_tile
    }

    #[cfg(feature = "dtype-bf16")]
    fn rms_norm_values_bf16<const B: i32>(
        _out: &mut Tensor<bf16, { [1, B] }>,
        input_tile: Tile<f32, { [1, B] }>,
        weight: &Tensor<bf16, { [-1] }>,
        eps: f32,
        weight_offset: f32,
        cols: i32,
    ) -> Tile<f32, { [1, B] }> {
        let tile_shape = const_shape![1, B];
        let cols_f32: f32 = convert_scalar(cols);
        let divisor = broadcast_scalar(cols_f32, const_shape![1]);
        let mean_square_sum: Tile<f32, { [1] }> = reduce_sum(input_tile * input_tile, 1i32);
        let mean_square = mean_square_sum / divisor;
        let one_row: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
        let inv_rms = one_row
            / sqrt(
                mean_square + broadcast_scalar(eps, const_shape![1]),
                rounding::NearestEven,
                ftz::Disabled,
            );
        let inv_rms = inv_rms.reshape(const_shape![1, 1]).broadcast(tile_shape);
        let weight_raw: Tile<bf16, { [B] }> = weight.partition(const_shape![B]).load([0i32]);
        let weight_tile: Tile<f32, { [1, B] }> = convert_tile(weight_raw.reshape(tile_shape));
        let weight_tile = weight_tile + broadcast_scalar(weight_offset, tile_shape);
        input_tile * inv_rms * weight_tile
    }

    fn rms_norm_values_f64<const B: i32>(
        _out: &mut Tensor<f64, { [1, B] }>,
        input_tile: Tile<f64, { [1, B] }>,
        weight: &Tensor<f64, { [-1] }>,
        eps: f64,
        weight_offset: f64,
        cols: i32,
    ) -> Tile<f64, { [1, B] }> {
        let tile_shape = const_shape![1, B];
        let cols_f64: f64 = convert_scalar(cols);
        let divisor = broadcast_scalar(cols_f64, const_shape![1]);
        let mean_square_sum: Tile<f64, { [1] }> = reduce_sum(input_tile * input_tile, 1i32);
        let mean_square = mean_square_sum / divisor;
        let one_row: Tile<f64, { [1] }> = constant(1.0f64, const_shape![1]);
        let inv_rms = one_row
            / sqrt(
                mean_square + broadcast_scalar(eps, const_shape![1]),
                rounding::NearestEven,
                ftz::Disabled,
            );
        let inv_rms = inv_rms.reshape(const_shape![1, 1]).broadcast(tile_shape);
        let weight_raw: Tile<f64, { [B] }> = weight.partition(const_shape![B]).load([0i32]);
        let weight_tile =
            weight_raw.reshape(tile_shape) + broadcast_scalar(weight_offset, tile_shape);
        input_tile * inv_rms * weight_tile
    }

    fn rotary_interleaved_f32(
        out: *mut f32,
        input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            len,
        );
        let values = rotary_values_f32(input, cos, sin, args);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    fn rotary_interleaved_cache_f32(
        out: *mut f32,
        cache: *mut f32,
        input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        cache_max_seq: i32,
        cache_position_start: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            len,
        );
        let values = rotary_values_f32(input, cos, sin, args);
        store_vector(out, args.offsets, values, args.output_mask);
        store_vector(
            cache,
            cache_offsets(
                args,
                output_dim1,
                output_dim3,
                cache_max_seq,
                cache_position_start,
            ),
            values,
            args.output_mask,
        );
    }

    fn rotary_interleaved_f16(
        out: *mut f16,
        input: *mut f16,
        cos: *mut f16,
        sin: *mut f16,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            len,
        );
        let values = rotary_values_f16(input, cos, sin, args);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    fn rotary_interleaved_cache_f16(
        out: *mut f16,
        cache: *mut f16,
        input: *mut f16,
        cos: *mut f16,
        sin: *mut f16,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        cache_max_seq: i32,
        cache_position_start: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            len,
        );
        let values = rotary_values_f16(input, cos, sin, args);
        store_vector(out, args.offsets, values, args.output_mask);
        store_vector(
            cache,
            cache_offsets(
                args,
                output_dim1,
                output_dim3,
                cache_max_seq,
                cache_position_start,
            ),
            values,
            args.output_mask,
        );
    }

    fn rotary_interleaved_f64(
        out: *mut f64,
        input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            len,
        );
        let values = rotary_values_f64(input, cos, sin, args);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    fn rotary_interleaved_cache_f64(
        out: *mut f64,
        cache: *mut f64,
        input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        cache_max_seq: i32,
        cache_position_start: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            len,
        );
        let values = rotary_values_f64(input, cos, sin, args);
        store_vector(out, args.offsets, values, args.output_mask);
        store_vector(
            cache,
            cache_offsets(
                args,
                output_dim1,
                output_dim3,
                cache_max_seq,
                cache_position_start,
            ),
            values,
            args.output_mask,
        );
    }

    #[derive(Clone, Copy)]
    struct RotaryInterleavedOffsets {
        offsets: Tile<i32, { [128] }>,
        output_mask: Tile<bool, { [128] }>,
        in_rotary: Tile<bool, { [128] }>,
        is_even: Tile<bool, { [128] }>,
        current_offsets: Tile<i32, { [128] }>,
        even_offsets: Tile<i32, { [128] }>,
        odd_offsets: Tile<i32, { [128] }>,
        cos_offsets: Tile<i32, { [128] }>,
        sin_offsets: Tile<i32, { [128] }>,
        coord0: Tile<i32, { [128] }>,
        coord1: Tile<i32, { [128] }>,
        coord2: Tile<i32, { [128] }>,
        coord3: Tile<i32, { [128] }>,
    }

    fn rotary_values_f32(
        input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        args: RotaryInterleavedOffsets,
    ) -> Tile<f32, { [128] }> {
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f32);
        let even = load_vector(input, args.even_offsets, args.in_rotary, 0.0f32);
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, 0.0f32);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f32);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f32);
        let even_values = even * cos_values - odd * sin_values;
        let odd_values = odd * cos_values + even * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        select(args.in_rotary, rotated, current)
    }

    fn rotary_values_f16(
        input: *mut f16,
        cos: *mut f16,
        sin: *mut f16,
        args: RotaryInterleavedOffsets,
    ) -> Tile<f16, { [128] }> {
        let current = load_vector(
            input,
            args.current_offsets,
            args.output_mask,
            f16::from_f32(0.0),
        );
        let even = load_vector(input, args.even_offsets, args.in_rotary, f16::from_f32(0.0));
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, f16::from_f32(0.0));
        let current_f32: Tile<f32, { [128] }> = convert_tile(current);
        let even_f32: Tile<f32, { [128] }> = convert_tile(even);
        let odd_f32: Tile<f32, { [128] }> = convert_tile(odd);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, f16::from_f32(1.0));
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, f16::from_f32(0.0));
        let cos_values: Tile<f32, { [128] }> = convert_tile(cos_values);
        let sin_values: Tile<f32, { [128] }> = convert_tile(sin_values);
        let even_values = even_f32 * cos_values - odd_f32 * sin_values;
        let odd_values = odd_f32 * cos_values + even_f32 * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        convert_tile(select(args.in_rotary, rotated, current_f32))
    }

    fn rotary_values_f64(
        input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        args: RotaryInterleavedOffsets,
    ) -> Tile<f64, { [128] }> {
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f64);
        let even = load_vector(input, args.even_offsets, args.in_rotary, 0.0f64);
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, 0.0f64);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f64);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f64);
        let even_values = even * cos_values - odd * sin_values;
        let odd_values = odd * cos_values + even * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        select(args.in_rotary, rotated, current)
    }

    fn rotary_interleaved_offsets(
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        len: i32,
    ) -> RotaryInterleavedOffsets {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let zero = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero);

        let dim1_dim2_dim3 = output_dim1 * output_dim2 * output_dim3;
        let dim2_dim3 = output_dim2 * output_dim3;
        let coord0: Tile<i32, { [128] }> =
            safe_offsets / broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let remaining0 = safe_offsets - coord0 * broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let coord1: Tile<i32, { [128] }> = remaining0 / broadcast_scalar(dim2_dim3, tile_shape);
        let remaining1 = remaining0 - coord1 * broadcast_scalar(dim2_dim3, tile_shape);
        let coord2: Tile<i32, { [128] }> = remaining1 / broadcast_scalar(output_dim3, tile_shape);
        let coord3: Tile<i32, { [128] }> =
            remaining1 - coord2 * broadcast_scalar(output_dim3, tile_shape);

        let current_offsets = coord0 * broadcast_scalar(input_stride0, tile_shape)
            + coord1 * broadcast_scalar(input_stride1, tile_shape)
            + coord2 * broadcast_scalar(input_stride2, tile_shape)
            + coord3 * broadcast_scalar(input_stride3, tile_shape);
        let pair = coord3 / broadcast_scalar(2i32, tile_shape);
        let even_coord3 = pair * broadcast_scalar(2i32, tile_shape);
        let odd_coord3 = even_coord3 + broadcast_scalar(1i32, tile_shape);
        let in_rotary = output_mask
            & cmpi(
                pair,
                broadcast_scalar(rotary_pairs, tile_shape),
                predicate::LessThan,
            );
        let is_even = cmpi(coord3, even_coord3, predicate::Equal);
        let even_offsets = coord0 * broadcast_scalar(input_stride0, tile_shape)
            + coord1 * broadcast_scalar(input_stride1, tile_shape)
            + coord2 * broadcast_scalar(input_stride2, tile_shape)
            + even_coord3 * broadcast_scalar(input_stride3, tile_shape);
        let odd_offsets = coord0 * broadcast_scalar(input_stride0, tile_shape)
            + coord1 * broadcast_scalar(input_stride1, tile_shape)
            + coord2 * broadcast_scalar(input_stride2, tile_shape)
            + odd_coord3 * broadcast_scalar(input_stride3, tile_shape);
        let cos_offsets = coord2 * broadcast_scalar(cos_stride0, tile_shape)
            + pair * broadcast_scalar(cos_stride1, tile_shape);
        let sin_offsets = coord2 * broadcast_scalar(sin_stride0, tile_shape)
            + pair * broadcast_scalar(sin_stride1, tile_shape);

        RotaryInterleavedOffsets {
            offsets,
            output_mask,
            in_rotary,
            is_even,
            current_offsets: select(output_mask, current_offsets, zero),
            even_offsets: select(in_rotary, even_offsets, zero),
            odd_offsets: select(in_rotary, odd_offsets, zero),
            cos_offsets: select(in_rotary, cos_offsets, zero),
            sin_offsets: select(in_rotary, sin_offsets, zero),
            coord0,
            coord1,
            coord2,
            coord3,
        }
    }

    fn cache_offsets(
        args: RotaryInterleavedOffsets,
        cache_heads: i32,
        head_dim: i32,
        cache_max_seq: i32,
        cache_position_start: i32,
    ) -> Tile<i32, { [128] }> {
        let tile_shape = const_shape![128];
        let cache_seq = args.coord2 + broadcast_scalar(cache_position_start, tile_shape);
        ((args.coord0 * broadcast_scalar(cache_heads, tile_shape) + args.coord1)
            * broadcast_scalar(cache_max_seq, tile_shape)
            + cache_seq)
            * broadcast_scalar(head_dim, tile_shape)
            + args.coord3
    }

    fn kv_cache_update_impl<T: ElementType>(
        k_cache: *mut T,
        v_cache: *mut T,
        new_k: *mut T,
        new_v: *mut T,
        _seq_len: i32,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        position_start: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let position_start_tile = broadcast_scalar(position_start, tile_shape);
        kv_cache_update_with_position_impl(
            k_cache,
            v_cache,
            new_k,
            new_v,
            heads,
            head_dim,
            max_seq,
            len,
            position_start_tile,
            offsets,
            mask,
        );
    }

    fn kv_cache_update_dynpos_impl<T: ElementType>(
        k_cache: *mut T,
        v_cache: *mut T,
        new_k: *mut T,
        new_v: *mut T,
        position_start: *mut u32,
        _seq_len: i32,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let load_mask: Tile<bool, { [128] }> = constant(true, tile_shape);
        let position_start_u32 = load_vector(position_start, zero_offsets, load_mask, 0u32);
        let position_start_tile: Tile<i32, { [128] }> = convert_tile(position_start_u32);
        kv_cache_update_with_position_impl(
            k_cache,
            v_cache,
            new_k,
            new_v,
            heads,
            head_dim,
            max_seq,
            len,
            position_start_tile,
            offsets,
            mask,
        );
    }

    #[cfg(feature = "dtype-f16")]
    fn load_u32_position_as_i32_tile(position_start: *mut u32) -> Tile<i32, { [128] }> {
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mask: Tile<bool, { [128] }> = constant(true, tile_shape);
        let base: PointerTile<*mut u32, { [] }> = pointer_to_tile(position_start);
        let base: PointerTile<*mut u32, { [1] }> = base.reshape(const_shape![1]);
        let ptrs: PointerTile<*mut u32, { [128] }> = base.broadcast(tile_shape);
        let ptrs: PointerTile<*mut u32, { [128] }> = ptrs.offset_tile(offsets);
        let result: (Tile<u32, { [128] }>, Token) = load_ptr_tko(
            ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u32),
            None,
            Latency::<0>,
        );
        bitcast(result.0)
    }

    fn kv_cache_update_with_position_impl<T: ElementType>(
        k_cache: *mut T,
        v_cache: *mut T,
        new_k: *mut T,
        new_v: *mut T,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        _len: i32,
        position_start_tile: Tile<i32, { [128] }>,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let tile_shape = const_shape![128];
        let head_dim_tile = broadcast_scalar(head_dim, tile_shape);
        let heads_tile = broadcast_scalar(heads, tile_shape);
        let max_seq_tile = broadcast_scalar(max_seq, tile_shape);

        let token_dim_offsets = offsets / head_dim_tile;
        let dim = offsets - token_dim_offsets * head_dim_tile;
        let seq = token_dim_offsets / heads_tile;
        let head = token_dim_offsets - seq * heads_tile;
        let cache_seq = position_start_tile + seq;
        let cache_offsets = (head * max_seq_tile + cache_seq) * head_dim_tile + dim;

        let k_values = load_vector_safe(new_k, offsets, mask);
        let v_values = load_vector_safe(new_v, offsets, mask);
        store_vector(k_cache, cache_offsets, k_values, mask);
        store_vector(v_cache, cache_offsets, v_values, mask);
    }

    #[cfg(feature = "dtype-f16")]
    fn bshd_rope_q_cache_update_impl(
        q_out: *mut f16,
        q_input: *mut f16,
        cos: *mut f32,
        sin: *mut f32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
    ) {
        bshd_rope_store_impl(
            q_out, q_input, cos, sin, offsets, mask, seq_len, heads, head_dim,
        );
    }

    #[cfg(feature = "dtype-f16")]
    fn bshd_rope_k_cache_update_impl(
        k_out: *mut f16,
        k_cache: *mut f16,
        k_input: *mut f16,
        cos: *mut f32,
        sin: *mut f32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        position_start: i32,
    ) {
        let tile_shape = const_shape![128];
        let head_dim_half = head_dim / 2i32;
        let pair_count = seq_len * heads * head_dim_half;
        let active = mask
            & cmpi(
                offsets,
                broadcast_scalar(pair_count, tile_shape),
                predicate::LessThan,
            );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(active, offsets, zero_offsets);
        let dim = safe_offsets
            - (safe_offsets / broadcast_scalar(head_dim_half, tile_shape))
                * broadcast_scalar(head_dim_half, tile_shape);
        let head_linear = safe_offsets / broadcast_scalar(head_dim_half, tile_shape);
        let head = head_linear
            - (head_linear / broadcast_scalar(heads, tile_shape))
                * broadcast_scalar(heads, tile_shape);
        let seq = head_linear / broadcast_scalar(heads, tile_shape);
        let base = (seq * broadcast_scalar(heads, tile_shape) + head)
            * broadcast_scalar(head_dim, tile_shape);
        let real_offsets = base + dim;
        let imag_offsets = base + broadcast_scalar(head_dim_half, tile_shape) + dim;
        let trig_offsets = seq * broadcast_scalar(head_dim_half, tile_shape) + dim;
        let cos_values = load_vector(cos, trig_offsets, active, 1.0f32);
        let sin_values = load_vector(sin, trig_offsets, active, 0.0f32);
        let real_values: Tile<f32, { [128] }> =
            convert_tile(load_vector_safe(k_input, real_offsets, active));
        let imag_values: Tile<f32, { [128] }> =
            convert_tile(load_vector_safe(k_input, imag_offsets, active));
        let rotated_real: Tile<f16, { [128] }> =
            convert_tile(real_values * cos_values - imag_values * sin_values);
        let rotated_imag: Tile<f16, { [128] }> =
            convert_tile(imag_values * cos_values + real_values * sin_values);
        store_vector(k_out, real_offsets, rotated_real, active);
        store_vector(k_out, imag_offsets, rotated_imag, active);

        let cache_seq = seq + broadcast_scalar(position_start, tile_shape);
        let cache_base = (head * broadcast_scalar(max_seq, tile_shape) + cache_seq)
            * broadcast_scalar(head_dim, tile_shape);
        store_vector(k_cache, cache_base + dim, rotated_real, active);
        store_vector(
            k_cache,
            cache_base + broadcast_scalar(head_dim_half, tile_shape) + dim,
            rotated_imag,
            active,
        );
    }

    #[cfg(feature = "dtype-f16")]
    fn bshd_rope_store_impl(
        out: *mut f16,
        input: *mut f16,
        cos: *mut f32,
        sin: *mut f32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
    ) {
        let tile_shape = const_shape![128];
        let head_dim_half = head_dim / 2i32;
        let pair_count = seq_len * heads * head_dim_half;
        let active = mask
            & cmpi(
                offsets,
                broadcast_scalar(pair_count, tile_shape),
                predicate::LessThan,
            );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(active, offsets, zero_offsets);
        let dim = safe_offsets
            - (safe_offsets / broadcast_scalar(head_dim_half, tile_shape))
                * broadcast_scalar(head_dim_half, tile_shape);
        let head_linear = safe_offsets / broadcast_scalar(head_dim_half, tile_shape);
        let head = head_linear
            - (head_linear / broadcast_scalar(heads, tile_shape))
                * broadcast_scalar(heads, tile_shape);
        let seq = head_linear / broadcast_scalar(heads, tile_shape);
        let base = (seq * broadcast_scalar(heads, tile_shape) + head)
            * broadcast_scalar(head_dim, tile_shape);
        let real_offsets = base + dim;
        let imag_offsets = base + broadcast_scalar(head_dim_half, tile_shape) + dim;
        let trig_offsets = seq * broadcast_scalar(head_dim_half, tile_shape) + dim;
        let cos_values = load_vector(cos, trig_offsets, active, 1.0f32);
        let sin_values = load_vector(sin, trig_offsets, active, 0.0f32);
        let real_values: Tile<f32, { [128] }> =
            convert_tile(load_vector_safe(input, real_offsets, active));
        let imag_values: Tile<f32, { [128] }> =
            convert_tile(load_vector_safe(input, imag_offsets, active));
        let rotated_real: Tile<f16, { [128] }> =
            convert_tile(real_values * cos_values - imag_values * sin_values);
        let rotated_imag: Tile<f16, { [128] }> =
            convert_tile(imag_values * cos_values + real_values * sin_values);
        store_vector(out, real_offsets, rotated_real, active);
        store_vector(out, imag_offsets, rotated_imag, active);
    }

    #[cfg(feature = "dtype-f16")]
    fn v_cache_update_impl(
        v_cache: *mut f16,
        v_input: *mut f16,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        position_start: i32,
    ) {
        let tile_shape = const_shape![128];
        let len = seq_len * heads * head_dim;
        let active = mask
            & cmpi(
                offsets,
                broadcast_scalar(len, tile_shape),
                predicate::LessThan,
            );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(active, offsets, zero_offsets);
        let token_dim_offsets = safe_offsets / broadcast_scalar(head_dim, tile_shape);
        let dim = safe_offsets - token_dim_offsets * broadcast_scalar(head_dim, tile_shape);
        let seq = token_dim_offsets / broadcast_scalar(heads, tile_shape);
        let head = token_dim_offsets - seq * broadcast_scalar(heads, tile_shape);
        let cache_seq = seq + broadcast_scalar(position_start, tile_shape);
        let cache_offsets = (head * broadcast_scalar(max_seq, tile_shape) + cache_seq)
            * broadcast_scalar(head_dim, tile_shape)
            + dim;
        let values = load_vector_safe(v_input, safe_offsets, active);
        store_vector(v_cache, cache_offsets, values, active);
    }

    fn kv_cache_compact_impl<T: ElementType>(
        k_cache: *mut T,
        v_cache: *mut T,
        heads: i32,
        head_dim: i32,
        max_seq: i32,
        source_start: i32,
        _token_count: i32,
        len: i32,
        fill: T,
    ) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let head_dim_tile = broadcast_scalar(head_dim, tile_shape);
        let token_dim_offsets = offsets / head_dim_tile;
        let dim = offsets - token_dim_offsets * head_dim_tile;
        let seq = token_dim_offsets / broadcast_scalar(heads, tile_shape);
        let head = token_dim_offsets - seq * broadcast_scalar(heads, tile_shape);
        let src_seq = seq + broadcast_scalar(source_start, tile_shape);
        let dst_offsets =
            (head * broadcast_scalar(max_seq, tile_shape) + seq) * head_dim_tile + dim;
        let src_offsets =
            (head * broadcast_scalar(max_seq, tile_shape) + src_seq) * head_dim_tile + dim;

        let k_values = load_vector(k_cache, src_offsets, mask, fill);
        let v_values = load_vector(v_cache, src_offsets, mask, fill);
        store_vector(k_cache, dst_offsets, k_values, mask);
        store_vector(v_cache, dst_offsets, v_values, mask);
    }

    fn vector_offsets(len: i32) -> (Tile<i32, { [128] }>, Tile<bool, { [128] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let len_tile = broadcast_scalar(len, tile_shape);
        let mask = cmpi(offsets, len_tile, predicate::LessThan);
        (offsets, mask)
    }

    fn vector_offsets_1024(len: i32) -> (Tile<i32, { [1024] }>, Tile<bool, { [1024] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let offsets: Tile<i32, { [1024] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 1024i32, tile_shape);
        let len_tile = broadcast_scalar(len, tile_shape);
        let mask = cmpi(offsets, len_tile, predicate::LessThan);
        (offsets, mask)
    }

    fn exact_gelu_tile(x: Tile<f32, { [1024] }>) -> Tile<f32, { [1024] }> {
        let shape = const_shape![1024];
        let half = constant(0.5f32, shape);
        let one = constant(1.0f32, shape);
        let inv_sqrt_2 = constant(0.707_106_77f32, shape);

        // Exact GELU form used by fused gate/up projections:
        // 0.5 * x * (1 + erf(x / sqrt(2))).
        half * x * (one + erf_approx_tile(x * inv_sqrt_2))
    }

    fn silu_tile(x: Tile<f32, { [1024] }>) -> Tile<f32, { [1024] }> {
        let shape = const_shape![1024];
        let zero = constant(0.0f32, shape);
        let one = constant(1.0f32, shape);
        x * (one / (one + exp(zero - x)))
    }

    fn packed_silu_offsets(
        hidden: i32,
        len: i32,
    ) -> (
        (
            Tile<i32, { [1024] }>,
            Tile<i32, { [1024] }>,
            Tile<i32, { [1024] }>,
        ),
        Tile<bool, { [1024] }>,
    ) {
        let (output_offsets, mask) = vector_offsets_1024(len);
        let shape = const_shape![1024];
        let hidden_tile = broadcast_scalar(hidden, shape);

        // Input is packed by row as [gate(hidden), up(hidden)].
        // Output is a compact hidden-wide row.
        let rows = output_offsets / hidden_tile;
        let columns = output_offsets - rows * hidden_tile;
        let row_base = rows * (hidden_tile + hidden_tile);
        let gate_offsets = row_base + columns;
        let up_offsets = gate_offsets + hidden_tile;
        ((gate_offsets, up_offsets, output_offsets), mask)
    }

    fn erf_approx_tile(x: Tile<f32, { [1024] }>) -> Tile<f32, { [1024] }> {
        let shape = const_shape![1024];
        let zero = constant(0.0f32, shape);
        let one = constant(1.0f32, shape);
        let sign = select(
            cmpf(x, zero, predicate::LessThan, cmp_ordering::Ordered),
            zero - one,
            one,
        );
        let ax = absf(x);
        let p = constant(0.327_591_1f32, shape);
        let t = one / (one + p * ax);
        let a1 = constant(0.254_829_6f32, shape);
        let a2 = constant(-0.284_496_72f32, shape);
        let a3 = constant(1.421_413_8f32, shape);
        let a4 = constant(-1.453_152_1f32, shape);
        let a5 = constant(1.061_405_4f32, shape);
        let poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
        sign * (one - poly * exp(zero - ax * ax))
    }

    fn load_vector<T: ElementType>(
        input: *mut T,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: T,
    ) -> Tile<T, { [128] }> {
        let input_base: PointerTile<*mut T, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut T, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut T, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut T, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<T, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(fill),
            None,
            Latency::<0>,
        );
        result.0
    }

    fn load_vector_safe<T: ElementType>(
        input: *mut T,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<T, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let load_mask: Tile<bool, { [128] }> = constant(true, tile_shape);
        let input_base: PointerTile<*mut T, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut T, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut T, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut T, { [128] }> = input_ptrs.offset_tile(safe_offsets);
        let result: (Tile<T, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(load_mask),
            None,
            None,
            Latency::<0>,
        );
        result.0
    }

    fn store_vector<T: ElementType>(
        out: *mut T,
        offsets: Tile<i32, { [128] }>,
        values: Tile<T, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let out_base: PointerTile<*mut T, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut T, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut T, { [128] }> = out_base.broadcast(const_shape![128]);
        let out_ptrs: PointerTile<*mut T, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    fn load_f32_vector_1024(
        input: *mut f32,
        offsets: Tile<i32, { [1024] }>,
        mask: Tile<bool, { [1024] }>,
    ) -> Tile<f32, { [1024] }> {
        let input_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f32, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [1024] }> =
            input_base.broadcast(const_shape![1024]);
        let input_ptrs: PointerTile<*mut f32, { [1024] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<f32, { [1024] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f32),
            None,
            Latency::<0>,
        );
        result.0
    }

    #[cfg(feature = "dtype-f16")]
    fn load_f16_vector_1024(
        input: *mut f16,
        offsets: Tile<i32, { [1024] }>,
        mask: Tile<bool, { [1024] }>,
    ) -> Tile<f32, { [1024] }> {
        let input_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f16, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f16, { [1024] }> =
            input_base.broadcast(const_shape![1024]);
        let input_ptrs: PointerTile<*mut f16, { [1024] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<f16, { [1024] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        let values: Tile<f32, { [1024] }> = convert_tile(result.0);
        let zero: Tile<f32, { [1024] }> = constant(0.0f32, const_shape![1024]);
        select(mask, values, zero)
    }

    #[cfg(feature = "dtype-bf16")]
    fn load_bf16_vector_1024(
        input: *mut bf16,
        offsets: Tile<i32, { [1024] }>,
        mask: Tile<bool, { [1024] }>,
    ) -> Tile<f32, { [1024] }> {
        let input_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut bf16, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut bf16, { [1024] }> =
            input_base.broadcast(const_shape![1024]);
        let input_ptrs: PointerTile<*mut bf16, { [1024] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<bf16, { [1024] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        let values: Tile<f32, { [1024] }> = convert_tile(result.0);
        let zero: Tile<f32, { [1024] }> = constant(0.0f32, const_shape![1024]);
        select(mask, values, zero)
    }

    fn store_f32_vector_1024(
        out: *mut f32,
        offsets: Tile<i32, { [1024] }>,
        values: Tile<f32, { [1024] }>,
        mask: Tile<bool, { [1024] }>,
    ) {
        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [1024] }> = out_base.broadcast(const_shape![1024]);
        let out_ptrs: PointerTile<*mut f32, { [1024] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cfg(feature = "dtype-f16")]
    fn store_f16_vector_1024(
        out: *mut f16,
        offsets: Tile<i32, { [1024] }>,
        values: Tile<f32, { [1024] }>,
        mask: Tile<bool, { [1024] }>,
    ) {
        let out_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f16, { [1024] }> = out_base.broadcast(const_shape![1024]);
        let out_ptrs: PointerTile<*mut f16, { [1024] }> = out_ptrs.offset_tile(offsets);
        let values: Tile<f16, { [1024] }> = convert_tile(values);
        store_ptr_tko(
            out_ptrs,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cfg(feature = "dtype-bf16")]
    fn store_bf16_vector_1024(
        out: *mut bf16,
        offsets: Tile<i32, { [1024] }>,
        values: Tile<f32, { [1024] }>,
        mask: Tile<bool, { [1024] }>,
    ) {
        let out_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut bf16, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut bf16, { [1024] }> = out_base.broadcast(const_shape![1024]);
        let out_ptrs: PointerTile<*mut bf16, { [1024] }> = out_ptrs.offset_tile(offsets);
        let values: Tile<bf16, { [1024] }> = convert_tile(values);
        store_ptr_tko(
            out_ptrs,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }
}

#[allow(ambiguous_glob_reexports)]
pub use kernels::*;
