#[cutile::module]
mod kernels {
    use cutile::core::*;

    // Mirrors the tiled ct.mma path in
    // thirdparty/TileGym/src/tilegym/ops/cutile/matmul.py for compact row-major f32 inputs.
    #[cutile::entry()]
    pub fn matmul_mma_f32<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [BM, BN] }>,
        lhs: &Tensor<f32, { [-1, -1] }>,
        rhs: &Tensor<f32, { [-1, -1] }>,
    ) {
        let lhs_tiles = lhs.partition(const_shape![BM, 8i32]);
        let rhs_tiles = rhs.partition(const_shape![8i32, BN]);
        let output_tile_id: (i32, i32, i32) = get_tile_block_id();
        let rhs_shape: [i32; 2] = get_tensor_shape(rhs);
        let reduction_tiles = (rhs_shape[0] + 7i32) / 8i32;

        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for reduction_tile in 0i32..reduction_tiles {
            let lhs_tile = lhs_tiles.load([output_tile_id.0, reduction_tile]);
            let rhs_tile = rhs_tiles.load([reduction_tile, output_tile_id.1]);
            accumulator = mma(lhs_tile, rhs_tile, accumulator);
        }
        out.store(accumulator);
    }

    // Mirrors the tiled ct.mma path in
    // thirdparty/TileGym/src/tilegym/ops/cutile/matmul.py for compact row-major f16 inputs.
    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub fn matmul_mma_f16_f32<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [BM, BN] }>,
        lhs: &Tensor<f16, { [-1, -1] }>,
        rhs: &Tensor<f16, { [-1, -1] }>,
    ) {
        let lhs_tiles = lhs.partition(const_shape![BM, 8i32]);
        let rhs_tiles = rhs.partition(const_shape![8i32, BN]);
        let output_tile_id: (i32, i32, i32) = get_tile_block_id();
        let rhs_shape: [i32; 2] = get_tensor_shape(rhs);
        let reduction_tiles = (rhs_shape[0] + 7i32) / 8i32;

        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for reduction_tile in 0i32..reduction_tiles {
            let lhs_tile = lhs_tiles.load([output_tile_id.0, reduction_tile]);
            let rhs_tile = rhs_tiles.load([reduction_tile, output_tile_id.1]);
            accumulator = mma(lhs_tile, rhs_tile, accumulator);
        }
        out.store(accumulator);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub fn matmul_mma_transposed_rhs_f16_f32<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [BM, BN] }>,
        lhs: &Tensor<f16, { [-1, -1] }>,
        rhs: &Tensor<f16, { [-1, -1] }>,
    ) {
        let lhs_tiles = lhs.partition(const_shape![BM, 8i32]);
        let rhs_tiles = rhs.partition(const_shape![BN, 8i32]);
        let output_tile_id: (i32, i32, i32) = get_tile_block_id();
        let lhs_shape: [i32; 2] = get_tensor_shape(lhs);
        let reduction_tiles = (lhs_shape[1] + 7i32) / 8i32;

        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for reduction_tile in 0i32..reduction_tiles {
            let lhs_tile = lhs_tiles.load([output_tile_id.0, reduction_tile]);
            let rhs_tile = rhs_tiles.load([output_tile_id.1, reduction_tile]);
            let rhs_tile = rhs_tile.transpose();
            accumulator = mma(lhs_tile, rhs_tile, accumulator);
        }
        out.store(accumulator);
    }

    // Mirrors the tiled ct.mma path in
    // thirdparty/TileGym/src/tilegym/ops/cutile/matmul.py for compact row-major bf16 inputs.
    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub fn matmul_mma_bf16_f32<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [BM, BN] }>,
        lhs: &Tensor<bf16, { [-1, -1] }>,
        rhs: &Tensor<bf16, { [-1, -1] }>,
    ) {
        let lhs_tiles = lhs.partition(const_shape![BM, 8i32]);
        let rhs_tiles = rhs.partition(const_shape![8i32, BN]);
        let output_tile_id: (i32, i32, i32) = get_tile_block_id();
        let rhs_shape: [i32; 2] = get_tensor_shape(rhs);
        let reduction_tiles = (rhs_shape[0] + 7i32) / 8i32;

        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for reduction_tile in 0i32..reduction_tiles {
            let lhs_tile = lhs_tiles.load([output_tile_id.0, reduction_tile]);
            let rhs_tile = rhs_tiles.load([reduction_tile, output_tile_id.1]);
            accumulator = mma(lhs_tile, rhs_tile, accumulator);
        }
        out.store(accumulator);
    }

    // Mirrors the row-major A[m, k] @ B[k, n] layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/matmul.py.
    #[cutile::entry()]
    pub unsafe fn matmul_f32(
        out: *mut f32,
        lhs: *mut f32,
        rhs: *mut f32,
        _rows: i32,
        columns: i32,
        reduction: i32,
        lhs_row_stride: i32,
        rhs_row_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
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

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_f32_vector(lhs, lhs_offsets, mask, 0.0f32);
            let rhs_values = load_f32_vector(rhs, rhs_offsets, mask, 0.0f32);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f32_vector(out, output_offsets, sum, mask);
    }

    #[cfg(feature = "dtype-f64")]
    // Mirrors the row-major A[m, k] @ B[k, n] layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/matmul.py.
    #[cutile::entry()]
    pub unsafe fn matmul_f64(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        _rows: i32,
        columns: i32,
        reduction: i32,
        lhs_row_stride: i32,
        rhs_row_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
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

        let mut sum: Tile<f64, { [128] }> = constant(0.0f64, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_f64_vector(lhs, lhs_offsets, mask, 0.0f64);
            let rhs_values = load_f64_vector(rhs, rhs_offsets, mask, 0.0f64);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f64_vector(out, output_offsets, sum, mask);
    }

    // Mirrors the row-major A[m, k] @ B[k, n] layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/matmul.py, accumulating half inputs into f32.
    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn matmul_f16_f32(
        out: *mut f32,
        lhs: *mut f16,
        rhs: *mut f16,
        _rows: i32,
        columns: i32,
        reduction: i32,
        lhs_row_stride: i32,
        rhs_row_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
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

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_f16_vector_as_f32(lhs, lhs_offsets, mask);
            let rhs_values = load_f16_vector_as_f32(rhs, rhs_offsets, mask);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f32_vector(out, output_offsets, sum, mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn matvec_transposed_rhs_f16_f32_1024(
        out: *mut f32,
        lhs: *mut f16,
        rhs: *mut f16,
        columns: i32,
        rhs_row_stride: i32,
    ) {
        unsafe {
            matvec_transposed_rhs_f16_f32_const::<1024>(out, lhs, rhs, columns, rhs_row_stride);
        }
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn matvec_transposed_rhs_f16_f32_2048(
        out: *mut f32,
        lhs: *mut f16,
        rhs: *mut f16,
        columns: i32,
        rhs_row_stride: i32,
    ) {
        unsafe {
            matvec_transposed_rhs_f16_f32_const::<2048>(out, lhs, rhs, columns, rhs_row_stride);
        }
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn matvec_transposed_rhs_f16_f32_3072(
        out: *mut f32,
        lhs: *mut f16,
        rhs: *mut f16,
        columns: i32,
        rhs_row_stride: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let column = pid.0;
        let active_2048 = cmpi(
            broadcast_scalar(column, const_shape![2048]),
            broadcast_scalar(columns, const_shape![2048]),
            predicate::LessThan,
        );
        let offsets_2048: Tile<i32, { [2048] }> = iota(const_shape![2048]);
        let lhs_values_2048 = load_f16_vector_as_f32_const(lhs, offsets_2048, active_2048);
        let rhs_offsets_2048 =
            broadcast_scalar(column * rhs_row_stride, const_shape![2048]) + offsets_2048;
        let rhs_values_2048 = load_f16_vector_as_f32_const(rhs, rhs_offsets_2048, active_2048);
        let sum_2048: Tile<f32, { [] }> = reduce_sum(lhs_values_2048 * rhs_values_2048, 0i32);

        let active_1024 = cmpi(
            broadcast_scalar(column, const_shape![1024]),
            broadcast_scalar(columns, const_shape![1024]),
            predicate::LessThan,
        );
        let offsets_1024: Tile<i32, { [1024] }> =
            iota(const_shape![1024]) + broadcast_scalar(2048i32, const_shape![1024]);
        let lhs_values_1024 = load_f16_vector_as_f32_const(lhs, offsets_1024, active_1024);
        let rhs_offsets_1024 =
            broadcast_scalar(column * rhs_row_stride, const_shape![1024]) + offsets_1024;
        let rhs_values_1024 = load_f16_vector_as_f32_const(rhs, rhs_offsets_1024, active_1024);
        let sum_1024: Tile<f32, { [] }> = reduce_sum(lhs_values_1024 * rhs_values_1024, 0i32);

        store_f32_scalar(out, column, (sum_2048 + sum_1024).reshape(const_shape![1]));
    }

    #[cfg(feature = "dtype-f16")]
    unsafe fn matvec_transposed_rhs_f16_f32_const<const K: i32>(
        out: *mut f32,
        lhs: *mut f16,
        rhs: *mut f16,
        columns: i32,
        rhs_row_stride: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let column = pid.0;
        let tile_shape = const_shape![K];
        let reduction_offsets: Tile<i32, { [K] }> = iota(tile_shape);
        let active = cmpi(
            broadcast_scalar(column, tile_shape),
            broadcast_scalar(columns, tile_shape),
            predicate::LessThan,
        );

        let lhs_values = load_f16_vector_as_f32_const(lhs, reduction_offsets, active);
        let rhs_offsets = broadcast_scalar(column * rhs_row_stride, tile_shape) + reduction_offsets;
        let rhs_values = load_f16_vector_as_f32_const(rhs, rhs_offsets, active);
        let sum: Tile<f32, { [] }> = reduce_sum(lhs_values * rhs_values, 0i32);
        store_f32_scalar(out, column, sum.reshape(const_shape![1]));
    }

    // Mirrors the row-major A[m, k] @ B[k, n] layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/matmul.py, accumulating bf16 inputs into f32.
    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn matmul_bf16_f32(
        out: *mut f32,
        lhs: *mut bf16,
        rhs: *mut bf16,
        _rows: i32,
        columns: i32,
        reduction: i32,
        lhs_row_stride: i32,
        rhs_row_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
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

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_bf16_vector_as_f32(lhs, lhs_offsets, mask);
            let rhs_values = load_bf16_vector_as_f32(rhs, rhs_offsets, mask);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f32_vector(out, output_offsets, sum, mask);
    }

    // Mirrors the alpha/beta epilogue in
    // thirdparty/TileGym/src/tilegym/suites/flashinfer/cutile/gemm/gemm_alpha_beta.py.
    #[cutile::entry()]
    pub unsafe fn matmul_alpha_beta_f32(
        out: *mut f32,
        lhs: *mut f32,
        rhs: *mut f32,
        _rows: i32,
        columns: i32,
        reduction: i32,
        lhs_row_stride: i32,
        rhs_row_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
        alpha: f32,
        beta: f32,
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

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_f32_vector(lhs, lhs_offsets, mask, 0.0f32);
            let rhs_values = load_f32_vector(rhs, rhs_offsets, mask, 0.0f32);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = row * broadcast_scalar(output_row_stride, tile_shape) + column;
        let prior = load_f32_vector(out, output_offsets, mask, 0.0f32);
        let result =
            broadcast_scalar(alpha, tile_shape) * sum + broadcast_scalar(beta, tile_shape) * prior;
        store_f32_vector(out, output_offsets, result, mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn matmul_alpha_beta_f16_f32(
        out: *mut f32,
        lhs: *mut f16,
        rhs: *mut f16,
        _rows: i32,
        columns: i32,
        reduction: i32,
        lhs_row_stride: i32,
        rhs_row_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
        alpha: f32,
        beta: f32,
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

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_f16_vector_as_f32(lhs, lhs_offsets, mask);
            let rhs_values = load_f16_vector_as_f32(rhs, rhs_offsets, mask);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = row * broadcast_scalar(output_row_stride, tile_shape) + column;
        let prior = load_f32_vector(out, output_offsets, mask, 0.0f32);
        let result =
            broadcast_scalar(alpha, tile_shape) * sum + broadcast_scalar(beta, tile_shape) * prior;
        store_f32_vector(out, output_offsets, result, mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn matmul_alpha_beta_bf16_f32(
        out: *mut f32,
        lhs: *mut bf16,
        rhs: *mut bf16,
        _rows: i32,
        columns: i32,
        reduction: i32,
        lhs_row_stride: i32,
        rhs_row_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
        alpha: f32,
        beta: f32,
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

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_bf16_vector_as_f32(lhs, lhs_offsets, mask);
            let rhs_values = load_bf16_vector_as_f32(rhs, rhs_offsets, mask);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = row * broadcast_scalar(output_row_stride, tile_shape) + column;
        let prior = load_f32_vector(out, output_offsets, mask, 0.0f32);
        let result =
            broadcast_scalar(alpha, tile_shape) * sum + broadcast_scalar(beta, tile_shape) * prior;
        store_f32_vector(out, output_offsets, result, mask);
    }

    #[cfg(feature = "dtype-f64")]
    // Mirrors the alpha/beta epilogue in
    // thirdparty/TileGym/src/tilegym/suites/flashinfer/cutile/gemm/gemm_alpha_beta.py.
    #[cutile::entry()]
    pub unsafe fn matmul_alpha_beta_f64(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        _rows: i32,
        columns: i32,
        reduction: i32,
        lhs_row_stride: i32,
        rhs_row_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
        alpha: f64,
        beta: f64,
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

        let mut sum: Tile<f64, { [128] }> = constant(0.0f64, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_f64_vector(lhs, lhs_offsets, mask, 0.0f64);
            let rhs_values = load_f64_vector(rhs, rhs_offsets, mask, 0.0f64);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = row * broadcast_scalar(output_row_stride, tile_shape) + column;
        let prior = load_f64_vector(out, output_offsets, mask, 0.0f64);
        let result =
            broadcast_scalar(alpha, tile_shape) * sum + broadcast_scalar(beta, tile_shape) * prior;
        store_f64_vector(out, output_offsets, result, mask);
    }

    // Mirrors the row-major (Q, M, K) @ (Q, K, N) layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/bmm.py.
    #[cutile::entry()]
    pub unsafe fn bmm_f32(
        out: *mut f32,
        lhs: *mut f32,
        rhs: *mut f32,
        rows: i32,
        columns: i32,
        reduction: i32,
        lhs_batch_stride: i32,
        lhs_row_stride: i32,
        rhs_batch_stride: i32,
        rhs_row_stride: i32,
        output_batch_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
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
        let matrix_len = broadcast_scalar(rows * columns, tile_shape);
        let batch = safe_offsets / matrix_len;
        let matrix_offsets = safe_offsets - batch * matrix_len;
        let row = matrix_offsets / broadcast_scalar(columns, tile_shape);
        let column = matrix_offsets - row * broadcast_scalar(columns, tile_shape);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_f32_vector(lhs, lhs_offsets, mask, 0.0f32);
            let rhs_values = load_f32_vector(rhs, rhs_offsets, mask, 0.0f32);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = batch * broadcast_scalar(output_batch_stride, tile_shape)
            + row * broadcast_scalar(output_row_stride, tile_shape)
            + column;
        store_f32_vector(out, output_offsets, sum, mask);
    }

    // Mirrors the tiled ct.mma path in
    // thirdparty/TileGym/src/tilegym/ops/cutile/bmm.py for compact row-major f32 inputs.
    #[cutile::entry()]
    pub fn bmm_mma_f32<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [1, BM, BN] }>,
        lhs: &Tensor<f32, { [-1, -1, -1] }>,
        rhs: &Tensor<f32, { [-1, -1, -1] }>,
    ) {
        let lhs_tiles = lhs.partition(const_shape![1i32, BM, 8i32]);
        let rhs_tiles = rhs.partition(const_shape![1i32, 8i32, BN]);
        let output_tile_id: (i32, i32, i32) = get_tile_block_id();
        let rhs_shape: [i32; 3] = get_tensor_shape(rhs);
        let reduction_tiles = (rhs_shape[1] + 7i32) / 8i32;

        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for reduction_tile in 0i32..reduction_tiles {
            let lhs_tile_3d = lhs_tiles.load([output_tile_id.0, output_tile_id.1, reduction_tile]);
            let rhs_tile_3d = rhs_tiles.load([output_tile_id.0, reduction_tile, output_tile_id.2]);
            let lhs_tile = lhs_tile_3d.reshape(const_shape![BM, 8i32]);
            let rhs_tile = rhs_tile_3d.reshape(const_shape![8i32, BN]);
            accumulator = mma(lhs_tile, rhs_tile, accumulator);
        }
        out.store(accumulator.reshape(const_shape![1i32, BM, BN]));
    }

    // Mirrors the transposed-A layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/bmm.py for compact row-major f32 inputs.
    #[cutile::entry()]
    pub fn bmm_mma_transposed_lhs_f32<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [1, BM, BN] }>,
        lhs: &Tensor<f32, { [-1, -1, -1] }>,
        rhs: &Tensor<f32, { [-1, -1, -1] }>,
    ) {
        let lhs_tiles = lhs.partition(const_shape![1i32, 8i32, BM]);
        let rhs_tiles = rhs.partition(const_shape![1i32, 8i32, BN]);
        let output_tile_id: (i32, i32, i32) = get_tile_block_id();
        let rhs_shape: [i32; 3] = get_tensor_shape(rhs);
        let reduction_tiles = (rhs_shape[1] + 7i32) / 8i32;

        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for reduction_tile in 0i32..reduction_tiles {
            let lhs_tile_3d = lhs_tiles.load([output_tile_id.0, reduction_tile, output_tile_id.1]);
            let rhs_tile_3d = rhs_tiles.load([output_tile_id.0, reduction_tile, output_tile_id.2]);
            let lhs_tile = lhs_tile_3d.reshape(const_shape![8i32, BM]).transpose();
            let rhs_tile = rhs_tile_3d.reshape(const_shape![8i32, BN]);
            accumulator = mma(lhs_tile, rhs_tile, accumulator);
        }
        out.store(accumulator.reshape(const_shape![1i32, BM, BN]));
    }

    // Mirrors the transposed-B layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/bmm.py for compact row-major f32 inputs.
    #[cutile::entry()]
    pub fn bmm_mma_transposed_rhs_f32<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [1, BM, BN] }>,
        lhs: &Tensor<f32, { [-1, -1, -1] }>,
        rhs: &Tensor<f32, { [-1, -1, -1] }>,
    ) {
        let lhs_tiles = lhs.partition(const_shape![1i32, BM, 8i32]);
        let rhs_tiles = rhs.partition(const_shape![1i32, BN, 8i32]);
        let output_tile_id: (i32, i32, i32) = get_tile_block_id();
        let lhs_shape: [i32; 3] = get_tensor_shape(lhs);
        let reduction_tiles = (lhs_shape[2] + 7i32) / 8i32;

        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for reduction_tile in 0i32..reduction_tiles {
            let lhs_tile_3d = lhs_tiles.load([output_tile_id.0, output_tile_id.1, reduction_tile]);
            let rhs_tile_3d = rhs_tiles.load([output_tile_id.0, output_tile_id.2, reduction_tile]);
            let lhs_tile = lhs_tile_3d.reshape(const_shape![BM, 8i32]);
            let rhs_tile = rhs_tile_3d.reshape(const_shape![BN, 8i32]).transpose();
            accumulator = mma(lhs_tile, rhs_tile, accumulator);
        }
        out.store(accumulator.reshape(const_shape![1i32, BM, BN]));
    }

    // Mirrors the transposed-A and transposed-B layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/bmm.py for compact row-major f32 inputs.
    #[cutile::entry()]
    pub fn bmm_mma_transposed_inputs_f32<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [1, BM, BN] }>,
        lhs: &Tensor<f32, { [-1, -1, -1] }>,
        rhs: &Tensor<f32, { [-1, -1, -1] }>,
    ) {
        let lhs_tiles = lhs.partition(const_shape![1i32, 8i32, BM]);
        let rhs_tiles = rhs.partition(const_shape![1i32, BN, 8i32]);
        let output_tile_id: (i32, i32, i32) = get_tile_block_id();
        let lhs_shape: [i32; 3] = get_tensor_shape(lhs);
        let reduction_tiles = (lhs_shape[1] + 7i32) / 8i32;

        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for reduction_tile in 0i32..reduction_tiles {
            let lhs_tile_3d = lhs_tiles.load([output_tile_id.0, reduction_tile, output_tile_id.1]);
            let rhs_tile_3d = rhs_tiles.load([output_tile_id.0, output_tile_id.2, reduction_tile]);
            let lhs_tile = lhs_tile_3d.reshape(const_shape![8i32, BM]).transpose();
            let rhs_tile = rhs_tile_3d.reshape(const_shape![BN, 8i32]).transpose();
            accumulator = mma(lhs_tile, rhs_tile, accumulator);
        }
        out.store(accumulator.reshape(const_shape![1i32, BM, BN]));
    }

    // Mirrors the tiled ct.mma path in
    // thirdparty/TileGym/src/tilegym/ops/cutile/bmm.py for compact row-major f16 inputs.
    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub fn bmm_mma_f16_f32<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [1, BM, BN] }>,
        lhs: &Tensor<f16, { [-1, -1, -1] }>,
        rhs: &Tensor<f16, { [-1, -1, -1] }>,
    ) {
        let lhs_tiles = lhs.partition(const_shape![1i32, BM, 8i32]);
        let rhs_tiles = rhs.partition(const_shape![1i32, 8i32, BN]);
        let output_tile_id: (i32, i32, i32) = get_tile_block_id();
        let rhs_shape: [i32; 3] = get_tensor_shape(rhs);
        let reduction_tiles = (rhs_shape[1] + 7i32) / 8i32;

        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for reduction_tile in 0i32..reduction_tiles {
            let lhs_tile_3d = lhs_tiles.load([output_tile_id.0, output_tile_id.1, reduction_tile]);
            let rhs_tile_3d = rhs_tiles.load([output_tile_id.0, reduction_tile, output_tile_id.2]);
            let lhs_tile = lhs_tile_3d.reshape(const_shape![BM, 8i32]);
            let rhs_tile = rhs_tile_3d.reshape(const_shape![8i32, BN]);
            accumulator = mma(lhs_tile, rhs_tile, accumulator);
        }
        out.store(accumulator.reshape(const_shape![1i32, BM, BN]));
    }

    // Mirrors the transposed-B layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/bmm.py for compact row-major f16 inputs.
    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub fn bmm_mma_transposed_rhs_f16_f32<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [1, BM, BN] }>,
        lhs: &Tensor<f16, { [-1, -1, -1] }>,
        rhs: &Tensor<f16, { [-1, -1, -1] }>,
    ) {
        let lhs_tiles = lhs.partition(const_shape![1i32, BM, 8i32]);
        let rhs_tiles = rhs.partition(const_shape![1i32, BN, 8i32]);
        let output_tile_id: (i32, i32, i32) = get_tile_block_id();
        let lhs_shape: [i32; 3] = get_tensor_shape(lhs);
        let reduction_tiles = (lhs_shape[2] + 7i32) / 8i32;

        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for reduction_tile in 0i32..reduction_tiles {
            let lhs_tile_3d = lhs_tiles.load([output_tile_id.0, output_tile_id.1, reduction_tile]);
            let rhs_tile_3d = rhs_tiles.load([output_tile_id.0, output_tile_id.2, reduction_tile]);
            let lhs_tile = lhs_tile_3d.reshape(const_shape![BM, 8i32]);
            let rhs_tile = rhs_tile_3d.reshape(const_shape![BN, 8i32]).transpose();
            accumulator = mma(lhs_tile, rhs_tile, accumulator);
        }
        out.store(accumulator.reshape(const_shape![1i32, BM, BN]));
    }

    // Mirrors the transposed-A and transposed-B layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/bmm.py for compact row-major f16 inputs.
    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub fn bmm_mma_transposed_inputs_f16_f32<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [1, BM, BN] }>,
        lhs: &Tensor<f16, { [-1, -1, -1] }>,
        rhs: &Tensor<f16, { [-1, -1, -1] }>,
    ) {
        let lhs_tiles = lhs.partition(const_shape![1i32, 8i32, BM]);
        let rhs_tiles = rhs.partition(const_shape![1i32, BN, 8i32]);
        let output_tile_id: (i32, i32, i32) = get_tile_block_id();
        let lhs_shape: [i32; 3] = get_tensor_shape(lhs);
        let reduction_tiles = (lhs_shape[1] + 7i32) / 8i32;

        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for reduction_tile in 0i32..reduction_tiles {
            let lhs_tile_3d = lhs_tiles.load([output_tile_id.0, reduction_tile, output_tile_id.1]);
            let rhs_tile_3d = rhs_tiles.load([output_tile_id.0, output_tile_id.2, reduction_tile]);
            let lhs_tile = lhs_tile_3d.reshape(const_shape![8i32, BM]).transpose();
            let rhs_tile = rhs_tile_3d.reshape(const_shape![BN, 8i32]).transpose();
            accumulator = mma(lhs_tile, rhs_tile, accumulator);
        }
        out.store(accumulator.reshape(const_shape![1i32, BM, BN]));
    }

    // Mirrors the tiled ct.mma path in
    // thirdparty/TileGym/src/tilegym/ops/cutile/bmm.py for compact row-major bf16 inputs.
    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub fn bmm_mma_bf16_f32<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [1, BM, BN] }>,
        lhs: &Tensor<bf16, { [-1, -1, -1] }>,
        rhs: &Tensor<bf16, { [-1, -1, -1] }>,
    ) {
        let lhs_tiles = lhs.partition(const_shape![1i32, BM, 8i32]);
        let rhs_tiles = rhs.partition(const_shape![1i32, 8i32, BN]);
        let output_tile_id: (i32, i32, i32) = get_tile_block_id();
        let rhs_shape: [i32; 3] = get_tensor_shape(rhs);
        let reduction_tiles = (rhs_shape[1] + 7i32) / 8i32;

        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for reduction_tile in 0i32..reduction_tiles {
            let lhs_tile_3d = lhs_tiles.load([output_tile_id.0, output_tile_id.1, reduction_tile]);
            let rhs_tile_3d = rhs_tiles.load([output_tile_id.0, reduction_tile, output_tile_id.2]);
            let lhs_tile = lhs_tile_3d.reshape(const_shape![BM, 8i32]);
            let rhs_tile = rhs_tile_3d.reshape(const_shape![8i32, BN]);
            accumulator = mma(lhs_tile, rhs_tile, accumulator);
        }
        out.store(accumulator.reshape(const_shape![1i32, BM, BN]));
    }

    // Mirrors the transposed-A and transposed-B layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/bmm.py for compact row-major bf16 inputs.
    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub fn bmm_mma_transposed_inputs_bf16_f32<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [1, BM, BN] }>,
        lhs: &Tensor<bf16, { [-1, -1, -1] }>,
        rhs: &Tensor<bf16, { [-1, -1, -1] }>,
    ) {
        let lhs_tiles = lhs.partition(const_shape![1i32, 8i32, BM]);
        let rhs_tiles = rhs.partition(const_shape![1i32, BN, 8i32]);
        let output_tile_id: (i32, i32, i32) = get_tile_block_id();
        let lhs_shape: [i32; 3] = get_tensor_shape(lhs);
        let reduction_tiles = (lhs_shape[1] + 7i32) / 8i32;

        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for reduction_tile in 0i32..reduction_tiles {
            let lhs_tile_3d = lhs_tiles.load([output_tile_id.0, reduction_tile, output_tile_id.1]);
            let rhs_tile_3d = rhs_tiles.load([output_tile_id.0, output_tile_id.2, reduction_tile]);
            let lhs_tile = lhs_tile_3d.reshape(const_shape![8i32, BM]).transpose();
            let rhs_tile = rhs_tile_3d.reshape(const_shape![BN, 8i32]).transpose();
            accumulator = mma(lhs_tile, rhs_tile, accumulator);
        }
        out.store(accumulator.reshape(const_shape![1i32, BM, BN]));
    }

    // Mirrors the per-batch valid-M mask in
    // thirdparty/TileGym/src/tilegym/suites/flashinfer/cutile/gemm/masked_bmm.py.
    #[cutile::entry()]
    pub unsafe fn masked_bmm_f32(
        out: *mut f32,
        lhs: *mut f32,
        rhs: *mut f32,
        masked_rows: *mut i32,
        rows: i32,
        columns: i32,
        reduction: i32,
        lhs_batch_stride: i32,
        lhs_row_stride: i32,
        rhs_batch_stride: i32,
        rhs_row_stride: i32,
        output_batch_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
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
        let matrix_len = broadcast_scalar(rows * columns, tile_shape);
        let batch = safe_offsets / matrix_len;
        let matrix_offsets = safe_offsets - batch * matrix_len;
        let row = matrix_offsets / broadcast_scalar(columns, tile_shape);
        let column = matrix_offsets - row * broadcast_scalar(columns, tile_shape);
        let valid_rows = load_i32_vector(masked_rows, batch, mask, 0i32);
        let active_mask = mask & cmpi(row, valid_rows, predicate::LessThan);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_f32_vector(lhs, lhs_offsets, active_mask, 0.0f32);
            let rhs_values = load_f32_vector(rhs, rhs_offsets, active_mask, 0.0f32);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = batch * broadcast_scalar(output_batch_stride, tile_shape)
            + row * broadcast_scalar(output_row_stride, tile_shape)
            + column;
        store_f32_vector(out, output_offsets, sum, active_mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn masked_bmm_f16(
        out: *mut f16,
        lhs: *mut f16,
        rhs: *mut f16,
        masked_rows: *mut i32,
        rows: i32,
        columns: i32,
        reduction: i32,
        lhs_batch_stride: i32,
        lhs_row_stride: i32,
        rhs_batch_stride: i32,
        rhs_row_stride: i32,
        output_batch_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
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
        let matrix_len = broadcast_scalar(rows * columns, tile_shape);
        let batch = safe_offsets / matrix_len;
        let matrix_offsets = safe_offsets - batch * matrix_len;
        let row = matrix_offsets / broadcast_scalar(columns, tile_shape);
        let column = matrix_offsets - row * broadcast_scalar(columns, tile_shape);
        let valid_rows = load_i32_vector(masked_rows, batch, mask, 0i32);
        let active_mask = mask & cmpi(row, valid_rows, predicate::LessThan);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_f16_vector_as_f32(lhs, lhs_offsets, active_mask);
            let rhs_values = load_f16_vector_as_f32(rhs, rhs_offsets, active_mask);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = batch * broadcast_scalar(output_batch_stride, tile_shape)
            + row * broadcast_scalar(output_row_stride, tile_shape)
            + column;
        store_f16_vector_from_f32(out, output_offsets, sum, active_mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn masked_bmm_bf16(
        out: *mut bf16,
        lhs: *mut bf16,
        rhs: *mut bf16,
        masked_rows: *mut i32,
        rows: i32,
        columns: i32,
        reduction: i32,
        lhs_batch_stride: i32,
        lhs_row_stride: i32,
        rhs_batch_stride: i32,
        rhs_row_stride: i32,
        output_batch_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
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
        let matrix_len = broadcast_scalar(rows * columns, tile_shape);
        let batch = safe_offsets / matrix_len;
        let matrix_offsets = safe_offsets - batch * matrix_len;
        let row = matrix_offsets / broadcast_scalar(columns, tile_shape);
        let column = matrix_offsets - row * broadcast_scalar(columns, tile_shape);
        let valid_rows = load_i32_vector(masked_rows, batch, mask, 0i32);
        let active_mask = mask & cmpi(row, valid_rows, predicate::LessThan);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_bf16_vector_as_f32(lhs, lhs_offsets, active_mask);
            let rhs_values = load_bf16_vector_as_f32(rhs, rhs_offsets, active_mask);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = batch * broadcast_scalar(output_batch_stride, tile_shape)
            + row * broadcast_scalar(output_row_stride, tile_shape)
            + column;
        store_bf16_vector_from_f32(out, output_offsets, sum, active_mask);
    }

    // Mirrors flattened A/C segment offsets in
    // thirdparty/TileGym/src/tilegym/suites/flashinfer/cutile/gemm/ragged_bmm.py.
    #[cutile::entry()]
    pub unsafe fn ragged_bmm_f32(
        out: *mut f32,
        lhs: *mut f32,
        rhs: *mut f32,
        row_indptr: *mut i32,
        max_rows: i32,
        columns: i32,
        reduction: i32,
        lhs_row_stride: i32,
        rhs_batch_stride: i32,
        rhs_row_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
        virtual_output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(virtual_output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let virtual_matrix_len = broadcast_scalar(max_rows * columns, tile_shape);
        let batch = safe_offsets / virtual_matrix_len;
        let matrix_offsets = safe_offsets - batch * virtual_matrix_len;
        let local_row = matrix_offsets / broadcast_scalar(columns, tile_shape);
        let column = matrix_offsets - local_row * broadcast_scalar(columns, tile_shape);

        let segment_start = load_i32_vector(row_indptr, batch, mask, 0i32);
        let segment_end = load_i32_vector(
            row_indptr,
            batch + broadcast_scalar(1i32, tile_shape),
            mask,
            0i32,
        );
        let valid_rows = segment_end - segment_start;
        let active_mask = mask & cmpi(local_row, valid_rows, predicate::LessThan);
        let global_row = segment_start + local_row;

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(lhs_row_stride, tile_shape)
                    + global_row
            } else {
                global_row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_f32_vector(lhs, lhs_offsets, active_mask, 0.0f32);
            let rhs_values = load_f32_vector(rhs, rhs_offsets, active_mask, 0.0f32);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = global_row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f32_vector(out, output_offsets, sum, active_mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn ragged_bmm_f16(
        out: *mut f16,
        lhs: *mut f16,
        rhs: *mut f16,
        row_indptr: *mut i32,
        max_rows: i32,
        columns: i32,
        reduction: i32,
        lhs_row_stride: i32,
        rhs_batch_stride: i32,
        rhs_row_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
        virtual_output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(virtual_output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let virtual_matrix_len = broadcast_scalar(max_rows * columns, tile_shape);
        let batch = safe_offsets / virtual_matrix_len;
        let matrix_offsets = safe_offsets - batch * virtual_matrix_len;
        let local_row = matrix_offsets / broadcast_scalar(columns, tile_shape);
        let column = matrix_offsets - local_row * broadcast_scalar(columns, tile_shape);

        let segment_start = load_i32_vector(row_indptr, batch, mask, 0i32);
        let segment_end = load_i32_vector(
            row_indptr,
            batch + broadcast_scalar(1i32, tile_shape),
            mask,
            0i32,
        );
        let valid_rows = segment_end - segment_start;
        let active_mask = mask & cmpi(local_row, valid_rows, predicate::LessThan);
        let global_row = segment_start + local_row;

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(lhs_row_stride, tile_shape)
                    + global_row
            } else {
                global_row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_f16_vector_as_f32(lhs, lhs_offsets, active_mask);
            let rhs_values = load_f16_vector_as_f32(rhs, rhs_offsets, active_mask);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = global_row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f16_vector_from_f32(out, output_offsets, sum, active_mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn ragged_bmm_bf16(
        out: *mut bf16,
        lhs: *mut bf16,
        rhs: *mut bf16,
        row_indptr: *mut i32,
        max_rows: i32,
        columns: i32,
        reduction: i32,
        lhs_row_stride: i32,
        rhs_batch_stride: i32,
        rhs_row_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
        virtual_output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(virtual_output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let virtual_matrix_len = broadcast_scalar(max_rows * columns, tile_shape);
        let batch = safe_offsets / virtual_matrix_len;
        let matrix_offsets = safe_offsets - batch * virtual_matrix_len;
        let local_row = matrix_offsets / broadcast_scalar(columns, tile_shape);
        let column = matrix_offsets - local_row * broadcast_scalar(columns, tile_shape);

        let segment_start = load_i32_vector(row_indptr, batch, mask, 0i32);
        let segment_end = load_i32_vector(
            row_indptr,
            batch + broadcast_scalar(1i32, tile_shape),
            mask,
            0i32,
        );
        let valid_rows = segment_end - segment_start;
        let active_mask = mask & cmpi(local_row, valid_rows, predicate::LessThan);
        let global_row = segment_start + local_row;

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                broadcast_scalar(reduction_index, tile_shape)
                    * broadcast_scalar(lhs_row_stride, tile_shape)
                    + global_row
            } else {
                global_row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_bf16_vector_as_f32(lhs, lhs_offsets, active_mask);
            let rhs_values = load_bf16_vector_as_f32(rhs, rhs_offsets, active_mask);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = global_row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_bf16_vector_from_f32(out, output_offsets, sum, active_mask);
    }

    // Mirrors list-based group GEMM from
    // thirdparty/TileGym/src/tilegym/ops/cutile/group_gemm.py using packed matrices plus descriptors.
    #[cutile::entry()]
    pub unsafe fn group_gemm_f32(
        out: *mut f32,
        lhs: *mut f32,
        rhs: *mut f32,
        rows: *mut i32,
        columns: *mut i32,
        reductions: *mut i32,
        lhs_offsets: *mut i32,
        rhs_offsets: *mut i32,
        output_offsets_base: *mut i32,
        max_rows: i32,
        max_columns: i32,
        max_reduction: i32,
        transpose_rhs: i32,
        virtual_output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(virtual_output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let virtual_matrix_len = broadcast_scalar(max_rows * max_columns, tile_shape);
        let group = safe_offsets / virtual_matrix_len;
        let matrix_offsets = safe_offsets - group * virtual_matrix_len;
        let row = matrix_offsets / broadcast_scalar(max_columns, tile_shape);
        let column = matrix_offsets - row * broadcast_scalar(max_columns, tile_shape);

        let group_rows = load_i32_vector(rows, group, mask, 0i32);
        let group_columns = load_i32_vector(columns, group, mask, 0i32);
        let group_reduction = load_i32_vector(reductions, group, mask, 0i32);
        let lhs_base = load_i32_vector(lhs_offsets, group, mask, 0i32);
        let rhs_base = load_i32_vector(rhs_offsets, group, mask, 0i32);
        let output_base = load_i32_vector(output_offsets_base, group, mask, 0i32);
        let active_mask = mask
            & cmpi(row, group_rows, predicate::LessThan)
            & cmpi(column, group_columns, predicate::LessThan);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..max_reduction {
            let reduction_mask = active_mask
                & cmpi(
                    broadcast_scalar(reduction_index, tile_shape),
                    group_reduction,
                    predicate::LessThan,
                );
            let lhs_element_offsets =
                lhs_base + row * group_reduction + broadcast_scalar(reduction_index, tile_shape);
            let rhs_element_offsets = if transpose_rhs != 0 {
                rhs_base + column * group_reduction + broadcast_scalar(reduction_index, tile_shape)
            } else {
                rhs_base + broadcast_scalar(reduction_index, tile_shape) * group_columns + column
            };
            let lhs_values = load_f32_vector(lhs, lhs_element_offsets, reduction_mask, 0.0f32);
            let rhs_values = load_f32_vector(rhs, rhs_element_offsets, reduction_mask, 0.0f32);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = output_base + row * group_columns + column;
        store_f32_vector(out, output_offsets, sum, active_mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn group_gemm_f16(
        out: *mut f16,
        lhs: *mut f16,
        rhs: *mut f16,
        rows: *mut i32,
        columns: *mut i32,
        reductions: *mut i32,
        lhs_offsets: *mut i32,
        rhs_offsets: *mut i32,
        output_offsets_base: *mut i32,
        max_rows: i32,
        max_columns: i32,
        max_reduction: i32,
        transpose_rhs: i32,
        virtual_output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(virtual_output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let virtual_matrix_len = broadcast_scalar(max_rows * max_columns, tile_shape);
        let group = safe_offsets / virtual_matrix_len;
        let matrix_offsets = safe_offsets - group * virtual_matrix_len;
        let row = matrix_offsets / broadcast_scalar(max_columns, tile_shape);
        let column = matrix_offsets - row * broadcast_scalar(max_columns, tile_shape);

        let group_rows = load_i32_vector(rows, group, mask, 0i32);
        let group_columns = load_i32_vector(columns, group, mask, 0i32);
        let group_reduction = load_i32_vector(reductions, group, mask, 0i32);
        let lhs_base = load_i32_vector(lhs_offsets, group, mask, 0i32);
        let rhs_base = load_i32_vector(rhs_offsets, group, mask, 0i32);
        let output_base = load_i32_vector(output_offsets_base, group, mask, 0i32);
        let active_mask = mask
            & cmpi(row, group_rows, predicate::LessThan)
            & cmpi(column, group_columns, predicate::LessThan);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..max_reduction {
            let reduction_mask = active_mask
                & cmpi(
                    broadcast_scalar(reduction_index, tile_shape),
                    group_reduction,
                    predicate::LessThan,
                );
            let lhs_element_offsets =
                lhs_base + row * group_reduction + broadcast_scalar(reduction_index, tile_shape);
            let rhs_element_offsets = if transpose_rhs != 0 {
                rhs_base + column * group_reduction + broadcast_scalar(reduction_index, tile_shape)
            } else {
                rhs_base + broadcast_scalar(reduction_index, tile_shape) * group_columns + column
            };
            let lhs_values = load_f16_vector_as_f32(lhs, lhs_element_offsets, reduction_mask);
            let rhs_values = load_f16_vector_as_f32(rhs, rhs_element_offsets, reduction_mask);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = output_base + row * group_columns + column;
        store_f16_vector_from_f32(out, output_offsets, sum, active_mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn group_gemm_bf16(
        out: *mut bf16,
        lhs: *mut bf16,
        rhs: *mut bf16,
        rows: *mut i32,
        columns: *mut i32,
        reductions: *mut i32,
        lhs_offsets: *mut i32,
        rhs_offsets: *mut i32,
        output_offsets_base: *mut i32,
        max_rows: i32,
        max_columns: i32,
        max_reduction: i32,
        transpose_rhs: i32,
        virtual_output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(virtual_output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let virtual_matrix_len = broadcast_scalar(max_rows * max_columns, tile_shape);
        let group = safe_offsets / virtual_matrix_len;
        let matrix_offsets = safe_offsets - group * virtual_matrix_len;
        let row = matrix_offsets / broadcast_scalar(max_columns, tile_shape);
        let column = matrix_offsets - row * broadcast_scalar(max_columns, tile_shape);

        let group_rows = load_i32_vector(rows, group, mask, 0i32);
        let group_columns = load_i32_vector(columns, group, mask, 0i32);
        let group_reduction = load_i32_vector(reductions, group, mask, 0i32);
        let lhs_base = load_i32_vector(lhs_offsets, group, mask, 0i32);
        let rhs_base = load_i32_vector(rhs_offsets, group, mask, 0i32);
        let output_base = load_i32_vector(output_offsets_base, group, mask, 0i32);
        let active_mask = mask
            & cmpi(row, group_rows, predicate::LessThan)
            & cmpi(column, group_columns, predicate::LessThan);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..max_reduction {
            let reduction_mask = active_mask
                & cmpi(
                    broadcast_scalar(reduction_index, tile_shape),
                    group_reduction,
                    predicate::LessThan,
                );
            let lhs_element_offsets =
                lhs_base + row * group_reduction + broadcast_scalar(reduction_index, tile_shape);
            let rhs_element_offsets = if transpose_rhs != 0 {
                rhs_base + column * group_reduction + broadcast_scalar(reduction_index, tile_shape)
            } else {
                rhs_base + broadcast_scalar(reduction_index, tile_shape) * group_columns + column
            };
            let lhs_values = load_bf16_vector_as_f32(lhs, lhs_element_offsets, reduction_mask);
            let rhs_values = load_bf16_vector_as_f32(rhs, rhs_element_offsets, reduction_mask);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = output_base + row * group_columns + column;
        store_bf16_vector_from_f32(out, output_offsets, sum, active_mask);
    }

    // Mirrors the forward grouped GEMM in
    // thirdparty/TileGym/src/tilegym/suites/unsloth/cutile/grouped_gemm.py.
    #[cutile::entry()]
    pub unsafe fn grouped_gemm_f32(
        out: *mut f32,
        input: *mut f32,
        weights: *mut f32,
        m_sizes: *mut i32,
        gather_indices: *mut i32,
        columns: i32,
        reduction: i32,
        expert_count: i32,
        input_row_stride: i32,
        weight_expert_stride: i32,
        weight_row_stride: i32,
        output_row_stride: i32,
        permute_input: i32,
        permute_output: i32,
        top_k: i32,
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
        let sorted_row = safe_offsets / broadcast_scalar(columns, tile_shape);
        let column = safe_offsets - sorted_row * broadcast_scalar(columns, tile_shape);

        let mut expert_id: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mut expert_start: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mut expert_size: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mut token_cursor: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        for expert_index in 0i32..expert_count {
            let current_size = load_i32_vector(
                m_sizes,
                broadcast_scalar(expert_index, tile_shape),
                mask,
                0i32,
            );
            let next_cursor = token_cursor + current_size;
            let selected = mask
                & cmpi(expert_size, zero_offsets, predicate::Equal)
                & cmpi(sorted_row, token_cursor, predicate::GreaterThanOrEqual)
                & cmpi(sorted_row, next_cursor, predicate::LessThan);
            expert_id = select(
                selected,
                broadcast_scalar(expert_index, tile_shape),
                expert_id,
            );
            expert_start = select(selected, token_cursor, expert_start);
            expert_size = select(selected, current_size, expert_size);
            token_cursor = next_cursor;
        }

        let active_mask = mask & cmpi(expert_size, zero_offsets, predicate::GreaterThan);
        let gathered_rows = load_i32_vector(gather_indices, sorted_row, active_mask, 0i32);
        let input_row = if permute_input != 0 {
            gathered_rows / broadcast_scalar(top_k, tile_shape)
        } else {
            sorted_row
        };
        let output_row = if permute_output != 0 {
            gathered_rows
        } else {
            sorted_row
        };

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let input_offsets = input_row * broadcast_scalar(input_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let weight_offsets = expert_id * broadcast_scalar(weight_expert_stride, tile_shape)
                + column * broadcast_scalar(weight_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let input_values = load_f32_vector(input, input_offsets, active_mask, 0.0f32);
            let weight_values = load_f32_vector(weights, weight_offsets, active_mask, 0.0f32);
            sum = sum + input_values * weight_values;
        }

        let output_offsets = output_row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f32_vector(out, output_offsets, sum, active_mask);
    }

    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn grouped_gemm_f16(
        out: *mut f16,
        input: *mut f16,
        weights: *mut f16,
        m_sizes: *mut i32,
        gather_indices: *mut i32,
        columns: i32,
        reduction: i32,
        expert_count: i32,
        input_row_stride: i32,
        weight_expert_stride: i32,
        weight_row_stride: i32,
        output_row_stride: i32,
        permute_input: i32,
        permute_output: i32,
        top_k: i32,
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
        let sorted_row = safe_offsets / broadcast_scalar(columns, tile_shape);
        let column = safe_offsets - sorted_row * broadcast_scalar(columns, tile_shape);

        let mut expert_id: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mut expert_start: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mut expert_size: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mut token_cursor: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        for expert_index in 0i32..expert_count {
            let current_size = load_i32_vector(
                m_sizes,
                broadcast_scalar(expert_index, tile_shape),
                mask,
                0i32,
            );
            let next_cursor = token_cursor + current_size;
            let selected = mask
                & cmpi(expert_size, zero_offsets, predicate::Equal)
                & cmpi(sorted_row, token_cursor, predicate::GreaterThanOrEqual)
                & cmpi(sorted_row, next_cursor, predicate::LessThan);
            expert_id = select(
                selected,
                broadcast_scalar(expert_index, tile_shape),
                expert_id,
            );
            expert_start = select(selected, token_cursor, expert_start);
            expert_size = select(selected, current_size, expert_size);
            token_cursor = next_cursor;
        }

        let active_mask = mask & cmpi(expert_size, zero_offsets, predicate::GreaterThan);
        let gathered_rows = load_i32_vector(gather_indices, sorted_row, active_mask, 0i32);
        let input_row = if permute_input != 0 {
            gathered_rows / broadcast_scalar(top_k, tile_shape)
        } else {
            sorted_row
        };
        let output_row = if permute_output != 0 {
            gathered_rows
        } else {
            sorted_row
        };

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let input_offsets = input_row * broadcast_scalar(input_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let weight_offsets = expert_id * broadcast_scalar(weight_expert_stride, tile_shape)
                + column * broadcast_scalar(weight_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let input_values = load_f16_vector_as_f32(input, input_offsets, active_mask);
            let weight_values = load_f16_vector_as_f32(weights, weight_offsets, active_mask);
            sum = sum + input_values * weight_values;
        }

        let output_offsets = output_row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f16_vector_from_f32(out, output_offsets, sum, active_mask);
    }

    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn grouped_gemm_bf16(
        out: *mut bf16,
        input: *mut bf16,
        weights: *mut bf16,
        m_sizes: *mut i32,
        gather_indices: *mut i32,
        columns: i32,
        reduction: i32,
        expert_count: i32,
        input_row_stride: i32,
        weight_expert_stride: i32,
        weight_row_stride: i32,
        output_row_stride: i32,
        permute_input: i32,
        permute_output: i32,
        top_k: i32,
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
        let sorted_row = safe_offsets / broadcast_scalar(columns, tile_shape);
        let column = safe_offsets - sorted_row * broadcast_scalar(columns, tile_shape);

        let mut expert_id: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mut expert_start: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mut expert_size: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mut token_cursor: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        for expert_index in 0i32..expert_count {
            let current_size = load_i32_vector(
                m_sizes,
                broadcast_scalar(expert_index, tile_shape),
                mask,
                0i32,
            );
            let next_cursor = token_cursor + current_size;
            let selected = mask
                & cmpi(expert_size, zero_offsets, predicate::Equal)
                & cmpi(sorted_row, token_cursor, predicate::GreaterThanOrEqual)
                & cmpi(sorted_row, next_cursor, predicate::LessThan);
            expert_id = select(
                selected,
                broadcast_scalar(expert_index, tile_shape),
                expert_id,
            );
            expert_start = select(selected, token_cursor, expert_start);
            expert_size = select(selected, current_size, expert_size);
            token_cursor = next_cursor;
        }

        let active_mask = mask & cmpi(expert_size, zero_offsets, predicate::GreaterThan);
        let gathered_rows = load_i32_vector(gather_indices, sorted_row, active_mask, 0i32);
        let input_row = if permute_input != 0 {
            gathered_rows / broadcast_scalar(top_k, tile_shape)
        } else {
            sorted_row
        };
        let output_row = if permute_output != 0 {
            gathered_rows
        } else {
            sorted_row
        };

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let input_offsets = input_row * broadcast_scalar(input_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let weight_offsets = expert_id * broadcast_scalar(weight_expert_stride, tile_shape)
                + column * broadcast_scalar(weight_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let input_values = load_bf16_vector_as_f32(input, input_offsets, active_mask);
            let weight_values = load_bf16_vector_as_f32(weights, weight_offsets, active_mask);
            sum = sum + input_values * weight_values;
        }

        let output_offsets = output_row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_bf16_vector_from_f32(out, output_offsets, sum, active_mask);
    }

    // Mirrors the scale application in
    // thirdparty/TileGym/src/tilegym/suites/flashinfer/cutile/gemm/ragged_block_scaled_bmm.py.
    #[cfg(feature = "dtype-f8")]
    #[cutile::entry()]
    pub unsafe fn ragged_block_scaled_bmm_f8e4m3_f32(
        out: *mut f32,
        lhs: *mut f8e4m3fn,
        rhs: *mut f8e4m3fn,
        lhs_scale: *mut f32,
        rhs_scale: *mut f32,
        row_indptr: *mut i32,
        max_rows: i32,
        columns: i32,
        reduction: i32,
        scale_block: i32,
        rhs_batch_stride: i32,
        rhs_row_stride: i32,
        lhs_scale_row_stride: i32,
        rhs_scale_batch_stride: i32,
        rhs_scale_row_stride: i32,
        output_row_stride: i32,
        virtual_output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(virtual_output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let virtual_matrix_len = broadcast_scalar(max_rows * columns, tile_shape);
        let batch = safe_offsets / virtual_matrix_len;
        let matrix_offsets = safe_offsets - batch * virtual_matrix_len;
        let local_row = matrix_offsets / broadcast_scalar(columns, tile_shape);
        let column = matrix_offsets - local_row * broadcast_scalar(columns, tile_shape);

        let segment_start = load_i32_vector(row_indptr, batch, mask, 0i32);
        let segment_end = load_i32_vector(
            row_indptr,
            batch + broadcast_scalar(1i32, tile_shape),
            mask,
            0i32,
        );
        let valid_rows = segment_end - segment_start;
        let active_mask = mask & cmpi(local_row, valid_rows, predicate::LessThan);
        let global_row = segment_start + local_row;

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let scale_k = broadcast_scalar(reduction_index, tile_shape)
                / broadcast_scalar(scale_block, tile_shape);
            let lhs_offsets = global_row * broadcast_scalar(reduction, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let rhs_offsets = batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                + column * broadcast_scalar(rhs_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let lhs_scale_offsets =
                global_row * broadcast_scalar(lhs_scale_row_stride, tile_shape) + scale_k;
            let rhs_scale_offsets = batch * broadcast_scalar(rhs_scale_batch_stride, tile_shape)
                + (column / broadcast_scalar(scale_block, tile_shape))
                    * broadcast_scalar(rhs_scale_row_stride, tile_shape)
                + scale_k;
            let lhs_values = load_f8e4m3_vector_as_f32(lhs, lhs_offsets, active_mask);
            let rhs_values = load_f8e4m3_vector_as_f32(rhs, rhs_offsets, active_mask);
            let lhs_scales = load_f32_vector(lhs_scale, lhs_scale_offsets, active_mask, 0.0f32);
            let rhs_scales = load_f32_vector(rhs_scale, rhs_scale_offsets, active_mask, 0.0f32);
            sum = sum + lhs_values * rhs_values * lhs_scales * rhs_scales;
        }

        let output_offsets = global_row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f32_vector(out, output_offsets, sum, active_mask);
    }

    // Mirrors the scale application in
    // thirdparty/TileGym/src/tilegym/suites/flashinfer/cutile/gemm/ragged_block_scaled_bmm.py.
    #[cfg(all(feature = "dtype-f8", feature = "dtype-f16"))]
    #[cutile::entry()]
    pub unsafe fn ragged_block_scaled_bmm_f8e4m3_f16(
        out: *mut f16,
        lhs: *mut f8e4m3fn,
        rhs: *mut f8e4m3fn,
        lhs_scale: *mut f32,
        rhs_scale: *mut f32,
        row_indptr: *mut i32,
        max_rows: i32,
        columns: i32,
        reduction: i32,
        scale_block: i32,
        rhs_batch_stride: i32,
        rhs_row_stride: i32,
        lhs_scale_row_stride: i32,
        rhs_scale_batch_stride: i32,
        rhs_scale_row_stride: i32,
        output_row_stride: i32,
        virtual_output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(virtual_output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let virtual_matrix_len = broadcast_scalar(max_rows * columns, tile_shape);
        let batch = safe_offsets / virtual_matrix_len;
        let matrix_offsets = safe_offsets - batch * virtual_matrix_len;
        let local_row = matrix_offsets / broadcast_scalar(columns, tile_shape);
        let column = matrix_offsets - local_row * broadcast_scalar(columns, tile_shape);

        let segment_start = load_i32_vector(row_indptr, batch, mask, 0i32);
        let segment_end = load_i32_vector(
            row_indptr,
            batch + broadcast_scalar(1i32, tile_shape),
            mask,
            0i32,
        );
        let valid_rows = segment_end - segment_start;
        let active_mask = mask & cmpi(local_row, valid_rows, predicate::LessThan);
        let global_row = segment_start + local_row;

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let scale_k = broadcast_scalar(reduction_index, tile_shape)
                / broadcast_scalar(scale_block, tile_shape);
            let lhs_offsets = global_row * broadcast_scalar(reduction, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let rhs_offsets = batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                + column * broadcast_scalar(rhs_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let lhs_scale_offsets =
                global_row * broadcast_scalar(lhs_scale_row_stride, tile_shape) + scale_k;
            let rhs_scale_offsets = batch * broadcast_scalar(rhs_scale_batch_stride, tile_shape)
                + (column / broadcast_scalar(scale_block, tile_shape))
                    * broadcast_scalar(rhs_scale_row_stride, tile_shape)
                + scale_k;
            let lhs_values = load_f8e4m3_vector_as_f32(lhs, lhs_offsets, active_mask);
            let rhs_values = load_f8e4m3_vector_as_f32(rhs, rhs_offsets, active_mask);
            let lhs_scales = load_f32_vector(lhs_scale, lhs_scale_offsets, active_mask, 0.0f32);
            let rhs_scales = load_f32_vector(rhs_scale, rhs_scale_offsets, active_mask, 0.0f32);
            sum = sum + lhs_values * rhs_values * lhs_scales * rhs_scales;
        }

        let output_offsets = global_row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_f16_vector_from_f32(out, output_offsets, sum, active_mask);
    }

    // Mirrors the scale application in
    // thirdparty/TileGym/src/tilegym/suites/flashinfer/cutile/gemm/ragged_block_scaled_bmm.py.
    #[cfg(all(feature = "dtype-f8", feature = "dtype-bf16"))]
    #[cutile::entry()]
    pub unsafe fn ragged_block_scaled_bmm_f8e4m3_bf16(
        out: *mut bf16,
        lhs: *mut f8e4m3fn,
        rhs: *mut f8e4m3fn,
        lhs_scale: *mut f32,
        rhs_scale: *mut f32,
        row_indptr: *mut i32,
        max_rows: i32,
        columns: i32,
        reduction: i32,
        scale_block: i32,
        rhs_batch_stride: i32,
        rhs_row_stride: i32,
        lhs_scale_row_stride: i32,
        rhs_scale_batch_stride: i32,
        rhs_scale_row_stride: i32,
        output_row_stride: i32,
        virtual_output_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(virtual_output_len, tile_shape),
            predicate::LessThan,
        );
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let safe_offsets = select(mask, offsets, zero_offsets);
        let virtual_matrix_len = broadcast_scalar(max_rows * columns, tile_shape);
        let batch = safe_offsets / virtual_matrix_len;
        let matrix_offsets = safe_offsets - batch * virtual_matrix_len;
        let local_row = matrix_offsets / broadcast_scalar(columns, tile_shape);
        let column = matrix_offsets - local_row * broadcast_scalar(columns, tile_shape);

        let segment_start = load_i32_vector(row_indptr, batch, mask, 0i32);
        let segment_end = load_i32_vector(
            row_indptr,
            batch + broadcast_scalar(1i32, tile_shape),
            mask,
            0i32,
        );
        let valid_rows = segment_end - segment_start;
        let active_mask = mask & cmpi(local_row, valid_rows, predicate::LessThan);
        let global_row = segment_start + local_row;

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let scale_k = broadcast_scalar(reduction_index, tile_shape)
                / broadcast_scalar(scale_block, tile_shape);
            let lhs_offsets = global_row * broadcast_scalar(reduction, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let rhs_offsets = batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                + column * broadcast_scalar(rhs_row_stride, tile_shape)
                + broadcast_scalar(reduction_index, tile_shape);
            let lhs_scale_offsets =
                global_row * broadcast_scalar(lhs_scale_row_stride, tile_shape) + scale_k;
            let rhs_scale_offsets = batch * broadcast_scalar(rhs_scale_batch_stride, tile_shape)
                + (column / broadcast_scalar(scale_block, tile_shape))
                    * broadcast_scalar(rhs_scale_row_stride, tile_shape)
                + scale_k;
            let lhs_values = load_f8e4m3_vector_as_f32(lhs, lhs_offsets, active_mask);
            let rhs_values = load_f8e4m3_vector_as_f32(rhs, rhs_offsets, active_mask);
            let lhs_scales = load_f32_vector(lhs_scale, lhs_scale_offsets, active_mask, 0.0f32);
            let rhs_scales = load_f32_vector(rhs_scale, rhs_scale_offsets, active_mask, 0.0f32);
            sum = sum + lhs_values * rhs_values * lhs_scales * rhs_scales;
        }

        let output_offsets = global_row * broadcast_scalar(output_row_stride, tile_shape) + column;
        store_bf16_vector_from_f32(out, output_offsets, sum, active_mask);
    }

    #[cfg(feature = "dtype-f64")]
    // Mirrors the row-major (Q, M, K) @ (Q, K, N) layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/bmm.py.
    #[cutile::entry()]
    pub unsafe fn bmm_f64(
        out: *mut f64,
        lhs: *mut f64,
        rhs: *mut f64,
        rows: i32,
        columns: i32,
        reduction: i32,
        lhs_batch_stride: i32,
        lhs_row_stride: i32,
        rhs_batch_stride: i32,
        rhs_row_stride: i32,
        output_batch_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
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
        let matrix_len = broadcast_scalar(rows * columns, tile_shape);
        let batch = safe_offsets / matrix_len;
        let matrix_offsets = safe_offsets - batch * matrix_len;
        let row = matrix_offsets / broadcast_scalar(columns, tile_shape);
        let column = matrix_offsets - row * broadcast_scalar(columns, tile_shape);

        let mut sum: Tile<f64, { [128] }> = constant(0.0f64, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_f64_vector(lhs, lhs_offsets, mask, 0.0f64);
            let rhs_values = load_f64_vector(rhs, rhs_offsets, mask, 0.0f64);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = batch * broadcast_scalar(output_batch_stride, tile_shape)
            + row * broadcast_scalar(output_row_stride, tile_shape)
            + column;
        store_f64_vector(out, output_offsets, sum, mask);
    }

    // Mirrors the row-major (Q, M, K) @ (Q, K, N) layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/bmm.py, accumulating half inputs into f32.
    #[cfg(feature = "dtype-f16")]
    #[cutile::entry()]
    pub unsafe fn bmm_f16_f32(
        out: *mut f32,
        lhs: *mut f16,
        rhs: *mut f16,
        rows: i32,
        columns: i32,
        reduction: i32,
        lhs_batch_stride: i32,
        lhs_row_stride: i32,
        rhs_batch_stride: i32,
        rhs_row_stride: i32,
        output_batch_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
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
        let matrix_len = broadcast_scalar(rows * columns, tile_shape);
        let batch = safe_offsets / matrix_len;
        let matrix_offsets = safe_offsets - batch * matrix_len;
        let row = matrix_offsets / broadcast_scalar(columns, tile_shape);
        let column = matrix_offsets - row * broadcast_scalar(columns, tile_shape);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_f16_vector_as_f32(lhs, lhs_offsets, mask);
            let rhs_values = load_f16_vector_as_f32(rhs, rhs_offsets, mask);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = batch * broadcast_scalar(output_batch_stride, tile_shape)
            + row * broadcast_scalar(output_row_stride, tile_shape)
            + column;
        store_f32_vector(out, output_offsets, sum, mask);
    }

    // Mirrors the row-major (Q, M, K) @ (Q, K, N) layout in
    // thirdparty/TileGym/src/tilegym/ops/cutile/bmm.py, accumulating bf16 inputs into f32.
    #[cfg(feature = "dtype-bf16")]
    #[cutile::entry()]
    pub unsafe fn bmm_bf16_f32(
        out: *mut f32,
        lhs: *mut bf16,
        rhs: *mut bf16,
        rows: i32,
        columns: i32,
        reduction: i32,
        lhs_batch_stride: i32,
        lhs_row_stride: i32,
        rhs_batch_stride: i32,
        rhs_row_stride: i32,
        output_batch_stride: i32,
        output_row_stride: i32,
        transpose_lhs: i32,
        transpose_rhs: i32,
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
        let matrix_len = broadcast_scalar(rows * columns, tile_shape);
        let batch = safe_offsets / matrix_len;
        let matrix_offsets = safe_offsets - batch * matrix_len;
        let row = matrix_offsets / broadcast_scalar(columns, tile_shape);
        let column = matrix_offsets - row * broadcast_scalar(columns, tile_shape);

        let mut sum: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        for reduction_index in 0i32..reduction {
            let lhs_offsets = if transpose_lhs != 0 {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(lhs_row_stride, tile_shape)
                    + row
            } else {
                batch * broadcast_scalar(lhs_batch_stride, tile_shape)
                    + row * broadcast_scalar(lhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            };
            let rhs_offsets = if transpose_rhs != 0 {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + column * broadcast_scalar(rhs_row_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
            } else {
                batch * broadcast_scalar(rhs_batch_stride, tile_shape)
                    + broadcast_scalar(reduction_index, tile_shape)
                        * broadcast_scalar(rhs_row_stride, tile_shape)
                    + column
            };
            let lhs_values = load_bf16_vector_as_f32(lhs, lhs_offsets, mask);
            let rhs_values = load_bf16_vector_as_f32(rhs, rhs_offsets, mask);
            sum = sum + lhs_values * rhs_values;
        }

        let output_offsets = batch * broadcast_scalar(output_batch_stride, tile_shape)
            + row * broadcast_scalar(output_row_stride, tile_shape)
            + column;
        store_f32_vector(out, output_offsets, sum, mask);
    }

    fn load_f32_vector(
        input: *mut f32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: f32,
    ) -> Tile<f32, { [128] }> {
        let input_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f32, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
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

    fn store_f32_vector(
        out: *mut f32,
        offsets: Tile<i32, { [128] }>,
        values: Tile<f32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(const_shape![128]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(offsets);
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

    fn store_f32_scalar(out: *mut f32, offset: i32, value: Tile<f32, { [1] }>) {
        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [1] }> =
            out_base.offset_tile(broadcast_scalar(offset, const_shape![1]));
        store_ptr_tko(
            out_ptrs,
            value,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }

    fn load_i32_vector(
        input: *mut i32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: i32,
    ) -> Tile<i32, { [128] }> {
        let input_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut i32, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut i32, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut i32, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<i32, { [128] }>, Token) = load_ptr_tko(
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

    #[cfg(feature = "dtype-f64")]
    fn load_f64_vector(
        input: *mut f64,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: f64,
    ) -> Tile<f64, { [128] }> {
        let input_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f64, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut f64, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<f64, { [128] }>, Token) = load_ptr_tko(
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

    #[cfg(feature = "dtype-f64")]
    fn store_f64_vector(
        out: *mut f64,
        offsets: Tile<i32, { [128] }>,
        values: Tile<f64, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut f64, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_base.broadcast(const_shape![128]);
        let out_ptrs: PointerTile<*mut f64, { [128] }> = out_ptrs.offset_tile(offsets);
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
    fn load_f16_vector_as_f32(
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
        let result: (Tile<f16, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        let values: Tile<f32, { [128] }> = convert_tile(result.0);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
        select(mask, values, zero)
    }

    #[cfg(feature = "dtype-f16")]
    fn load_f16_vector_as_f32_const<const K: i32>(
        input: *mut f16,
        offsets: Tile<i32, { [K] }>,
        mask: Tile<bool, { [K] }>,
    ) -> Tile<f32, { [K] }> {
        let zero_offsets: Tile<i32, { [K] }> = constant(0i32, const_shape![K]);
        let offsets = select(mask, offsets, zero_offsets);
        let input_base: PointerTile<*mut f16, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f16, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f16, { [K] }> = input_base.broadcast(const_shape![K]);
        let input_ptrs: PointerTile<*mut f16, { [K] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<f16, { [K] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        let values: Tile<f32, { [K] }> = convert_tile(result.0);
        let zero: Tile<f32, { [K] }> = constant(0.0f32, const_shape![K]);
        select(mask, values, zero)
    }

    #[cfg(feature = "dtype-f8")]
    fn load_f8e4m3_vector_as_f32(
        input: *mut f8e4m3fn,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let input_base: PointerTile<*mut f8e4m3fn, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut f8e4m3fn, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f8e4m3fn, { [128] }> =
            input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut f8e4m3fn, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<f8e4m3fn, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        let values: Tile<f32, { [128] }> = convert_tile(result.0);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
        select(mask, values, zero)
    }

    #[cfg(feature = "dtype-f16")]
    fn store_f16_vector_from_f32(
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
        store_ptr_tko(
            out_ptrs,
            output,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cfg(feature = "dtype-bf16")]
    fn store_bf16_vector_from_f32(
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
        store_ptr_tko(
            out_ptrs,
            output,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cfg(feature = "dtype-bf16")]
    fn load_bf16_vector_as_f32(
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
        let result: (Tile<bf16, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            None,
            Latency::<0>,
        );
        let values: Tile<f32, { [128] }> = convert_tile(result.0);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
        select(mask, values, zero)
    }
}

pub use kernels::*;
