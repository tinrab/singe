#[cutile::module]
mod kernels {
    use cutile::core::*;

    const VECTOR_TILE_SIZE: i32 = 128;

    #[cutile::entry()]
    pub unsafe fn conv2d_bias_gelu_f32(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        batch: i32,
        channels_in: i32,
        channels_out: i32,
        input_h: i32,
        input_w: i32,
        output_h: i32,
        output_w: i32,
        kernel_h: i32,
        kernel_w: i32,
        stride_h: i32,
        stride_w: i32,
        padding_h: i32,
        padding_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        groups: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_len: i32,
    ) {
        let _ = batch;
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets = iota(tile_shape) + broadcast_scalar(tile_id.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );

        // Each lane owns one flattened output scalar.
        // It decodes the scalar into batch, output channel, and spatial coordinates.
        let output_plane = output_h * output_w;
        let output_values_per_batch = channels_out * output_plane;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let out_channel = batch_offset / broadcast_scalar(output_plane, tile_shape);
        let output_spatial =
            batch_offset - out_channel * broadcast_scalar(output_plane, tile_shape);
        let out_h = output_spatial / broadcast_scalar(output_w, tile_shape);
        let out_w = output_spatial - out_h * broadcast_scalar(output_w, tile_shape);

        let channels_in_per_group = channels_in / groups;
        let channels_out_per_group = channels_out / groups;
        let group = out_channel / broadcast_scalar(channels_out_per_group, tile_shape);
        let input_group_start = group * broadcast_scalar(channels_in_per_group, tile_shape);
        let input_batch_base = batch_index * broadcast_scalar(input_batch_stride, tile_shape);

        let mut sum = load_vector(bias, out_channel, valid, 0.0f32);
        // Accumulate a grouped 2D convolution.
        // Taps that fall into padding or beyond the input bounds are masked.
        for group_input_channel in 0i32..channels_in_per_group {
            let input_channel =
                input_group_start + broadcast_scalar(group_input_channel, tile_shape);
            for kernel_y in 0i32..kernel_h {
                let input_y = out_h * broadcast_scalar(stride_h, tile_shape)
                    + broadcast_scalar(kernel_y * dilation_h, tile_shape)
                    - broadcast_scalar(padding_h, tile_shape);
                let y_valid = valid
                    & cmpi(
                        input_y,
                        broadcast_scalar(0i32, tile_shape),
                        predicate::GreaterThanOrEqual,
                    )
                    & cmpi(
                        input_y,
                        broadcast_scalar(input_h, tile_shape),
                        predicate::LessThan,
                    );
                for kernel_x in 0i32..kernel_w {
                    let input_x = out_w * broadcast_scalar(stride_w, tile_shape)
                        + broadcast_scalar(kernel_x * dilation_w, tile_shape)
                        - broadcast_scalar(padding_w, tile_shape);
                    let input_valid = y_valid
                        & cmpi(
                            input_x,
                            broadcast_scalar(0i32, tile_shape),
                            predicate::GreaterThanOrEqual,
                        )
                        & cmpi(
                            input_x,
                            broadcast_scalar(input_w, tile_shape),
                            predicate::LessThan,
                        );
                    let input_offsets = input_batch_base
                        + input_channel * broadcast_scalar(input_h * input_w, tile_shape)
                        + input_y * broadcast_scalar(input_w, tile_shape)
                        + input_x;
                    let weight_offsets = out_channel
                        * broadcast_scalar(channels_in_per_group * kernel_h * kernel_w, tile_shape)
                        + broadcast_scalar(
                            group_input_channel * kernel_h * kernel_w
                                + kernel_y * kernel_w
                                + kernel_x,
                            tile_shape,
                        );
                    let input_values = load_vector(input, input_offsets, input_valid, 0.0f32);
                    let weight_values = load_vector(weight, weight_offsets, valid, 0.0f32);
                    sum = sum + input_values * weight_values;
                }
            }
        }

        // The convolution epilogue fuses bias and the exact GELU form.
        // This mirrors the fused conv2d-bias-GELU entry point.
        let half = constant(0.5f32, tile_shape);
        let one = constant(1.0f32, tile_shape);
        let inv_sqrt_2 = constant(0.707_106_77f32, tile_shape);
        let gelu = half * sum * (one + erf_approx_tile_128(sum * inv_sqrt_2));
        let output_offsets =
            batch_index * broadcast_scalar(output_batch_stride, tile_shape) + batch_offset;
        store_vector(out, output_offsets, gelu, valid);
    }

    #[cutile::entry()]
    pub unsafe fn conv2d_bias_gelu_f16(
        out: *mut f16,
        input: *mut f16,
        weight: *mut f16,
        bias: *mut f16,
        batch: i32,
        channels_in: i32,
        channels_out: i32,
        input_h: i32,
        input_w: i32,
        output_h: i32,
        output_w: i32,
        kernel_h: i32,
        kernel_w: i32,
        stride_h: i32,
        stride_w: i32,
        padding_h: i32,
        padding_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        groups: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_len: i32,
    ) {
        let _ = batch;
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets = iota(tile_shape) + broadcast_scalar(tile_id.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );

        let output_plane = output_h * output_w;
        let output_values_per_batch = channels_out * output_plane;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let out_channel = batch_offset / broadcast_scalar(output_plane, tile_shape);
        let output_spatial =
            batch_offset - out_channel * broadcast_scalar(output_plane, tile_shape);
        let out_h = output_spatial / broadcast_scalar(output_w, tile_shape);
        let out_w = output_spatial - out_h * broadcast_scalar(output_w, tile_shape);

        let channels_in_per_group = channels_in / groups;
        let channels_out_per_group = channels_out / groups;
        let group = out_channel / broadcast_scalar(channels_out_per_group, tile_shape);
        let input_group_start = group * broadcast_scalar(channels_in_per_group, tile_shape);
        let input_batch_base = batch_index * broadcast_scalar(input_batch_stride, tile_shape);

        let mut sum = load_vector_f16_as_f32(bias, out_channel, valid);
        for group_input_channel in 0i32..channels_in_per_group {
            let input_channel =
                input_group_start + broadcast_scalar(group_input_channel, tile_shape);
            for kernel_y in 0i32..kernel_h {
                let input_y = out_h * broadcast_scalar(stride_h, tile_shape)
                    + broadcast_scalar(kernel_y * dilation_h, tile_shape)
                    - broadcast_scalar(padding_h, tile_shape);
                let y_valid = valid
                    & cmpi(
                        input_y,
                        broadcast_scalar(0i32, tile_shape),
                        predicate::GreaterThanOrEqual,
                    )
                    & cmpi(
                        input_y,
                        broadcast_scalar(input_h, tile_shape),
                        predicate::LessThan,
                    );
                for kernel_x in 0i32..kernel_w {
                    let input_x = out_w * broadcast_scalar(stride_w, tile_shape)
                        + broadcast_scalar(kernel_x * dilation_w, tile_shape)
                        - broadcast_scalar(padding_w, tile_shape);
                    let input_valid = y_valid
                        & cmpi(
                            input_x,
                            broadcast_scalar(0i32, tile_shape),
                            predicate::GreaterThanOrEqual,
                        )
                        & cmpi(
                            input_x,
                            broadcast_scalar(input_w, tile_shape),
                            predicate::LessThan,
                        );
                    let input_offsets = input_batch_base
                        + input_channel * broadcast_scalar(input_h * input_w, tile_shape)
                        + input_y * broadcast_scalar(input_w, tile_shape)
                        + input_x;
                    let weight_offsets = out_channel
                        * broadcast_scalar(channels_in_per_group * kernel_h * kernel_w, tile_shape)
                        + broadcast_scalar(
                            group_input_channel * kernel_h * kernel_w
                                + kernel_y * kernel_w
                                + kernel_x,
                            tile_shape,
                        );
                    let input_values = load_vector_f16_as_f32(input, input_offsets, input_valid);
                    let weight_values = load_vector_f16_as_f32(weight, weight_offsets, valid);
                    sum = sum + input_values * weight_values;
                }
            }
        }

        let half = constant(0.5f32, tile_shape);
        let one = constant(1.0f32, tile_shape);
        let inv_sqrt_2 = constant(0.707_106_77f32, tile_shape);
        let gelu = half * sum * (one + erf_approx_tile_128(sum * inv_sqrt_2));
        let output_offsets =
            batch_index * broadcast_scalar(output_batch_stride, tile_shape) + batch_offset;
        store_vector_f16_from_f32(out, output_offsets, gelu, valid);
    }

    #[cutile::entry()]
    pub unsafe fn conv2d_bias_gelu_bf16(
        out: *mut bf16,
        input: *mut bf16,
        weight: *mut bf16,
        bias: *mut bf16,
        batch: i32,
        channels_in: i32,
        channels_out: i32,
        input_h: i32,
        input_w: i32,
        output_h: i32,
        output_w: i32,
        kernel_h: i32,
        kernel_w: i32,
        stride_h: i32,
        stride_w: i32,
        padding_h: i32,
        padding_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        groups: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_len: i32,
    ) {
        let _ = batch;
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets = iota(tile_shape) + broadcast_scalar(tile_id.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );

        let output_plane = output_h * output_w;
        let output_values_per_batch = channels_out * output_plane;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let out_channel = batch_offset / broadcast_scalar(output_plane, tile_shape);
        let output_spatial =
            batch_offset - out_channel * broadcast_scalar(output_plane, tile_shape);
        let out_h = output_spatial / broadcast_scalar(output_w, tile_shape);
        let out_w = output_spatial - out_h * broadcast_scalar(output_w, tile_shape);

        let channels_in_per_group = channels_in / groups;
        let channels_out_per_group = channels_out / groups;
        let group = out_channel / broadcast_scalar(channels_out_per_group, tile_shape);
        let input_group_start = group * broadcast_scalar(channels_in_per_group, tile_shape);
        let input_batch_base = batch_index * broadcast_scalar(input_batch_stride, tile_shape);

        let mut sum = load_vector_bf16_as_f32(bias, out_channel, valid);
        for group_input_channel in 0i32..channels_in_per_group {
            let input_channel =
                input_group_start + broadcast_scalar(group_input_channel, tile_shape);
            for kernel_y in 0i32..kernel_h {
                let input_y = out_h * broadcast_scalar(stride_h, tile_shape)
                    + broadcast_scalar(kernel_y * dilation_h, tile_shape)
                    - broadcast_scalar(padding_h, tile_shape);
                let y_valid = valid
                    & cmpi(
                        input_y,
                        broadcast_scalar(0i32, tile_shape),
                        predicate::GreaterThanOrEqual,
                    )
                    & cmpi(
                        input_y,
                        broadcast_scalar(input_h, tile_shape),
                        predicate::LessThan,
                    );
                for kernel_x in 0i32..kernel_w {
                    let input_x = out_w * broadcast_scalar(stride_w, tile_shape)
                        + broadcast_scalar(kernel_x * dilation_w, tile_shape)
                        - broadcast_scalar(padding_w, tile_shape);
                    let input_valid = y_valid
                        & cmpi(
                            input_x,
                            broadcast_scalar(0i32, tile_shape),
                            predicate::GreaterThanOrEqual,
                        )
                        & cmpi(
                            input_x,
                            broadcast_scalar(input_w, tile_shape),
                            predicate::LessThan,
                        );
                    let input_offsets = input_batch_base
                        + input_channel * broadcast_scalar(input_h * input_w, tile_shape)
                        + input_y * broadcast_scalar(input_w, tile_shape)
                        + input_x;
                    let weight_offsets = out_channel
                        * broadcast_scalar(channels_in_per_group * kernel_h * kernel_w, tile_shape)
                        + broadcast_scalar(
                            group_input_channel * kernel_h * kernel_w
                                + kernel_y * kernel_w
                                + kernel_x,
                            tile_shape,
                        );
                    let input_values = load_vector_bf16_as_f32(input, input_offsets, input_valid);
                    let weight_values = load_vector_bf16_as_f32(weight, weight_offsets, valid);
                    sum = sum + input_values * weight_values;
                }
            }
        }

        let half = constant(0.5f32, tile_shape);
        let one = constant(1.0f32, tile_shape);
        let inv_sqrt_2 = constant(0.707_106_77f32, tile_shape);
        let gelu = half * sum * (one + erf_approx_tile_128(sum * inv_sqrt_2));
        let output_offsets =
            batch_index * broadcast_scalar(output_batch_stride, tile_shape) + batch_offset;
        store_vector_bf16_from_f32(out, output_offsets, gelu, valid);
    }

    #[cutile::entry()]
    pub unsafe fn causal_conv1d_update_silu_f32(
        out: *mut f32,
        conv_state: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        channels: i32,
        kernel_size: i32,
        len: i32,
    ) {
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets = iota(tile_shape) + broadcast_scalar(tile_id.0 * VECTOR_TILE_SIZE, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let channel = offsets
            - (offsets / broadcast_scalar(channels, tile_shape))
                * broadcast_scalar(channels, tile_shape);

        // The flat update kernel owns one `(token, channel)` state row per lane.
        let state_offsets = offsets * broadcast_scalar(kernel_size, tile_shape);
        let weight_offsets = channel * broadcast_scalar(kernel_size, tile_shape);
        let one_offset = broadcast_scalar(1i32, tile_shape);
        let last_kernel = kernel_size - 1i32;
        let last_kernel_offset = broadcast_scalar(last_kernel, tile_shape);
        let x = load_vector(input, offsets, mask, 0.0f32);
        let mut dot = load_vector(bias, channel, mask, 0.0f32);
        // Existing state slots provide the previous causal taps.
        // The current input supplies the newest tap.
        for kernel in 0i32..last_kernel {
            let kernel_offset = broadcast_scalar(kernel, tile_shape);
            let shifted_kernel_offset = kernel_offset + one_offset;
            let value = load_vector(
                conv_state,
                state_offsets + shifted_kernel_offset,
                mask,
                0.0f32,
            );
            let weight_value = load_vector(weight, weight_offsets + kernel_offset, mask, 0.0f32);
            dot = dot + value * weight_value;
        }
        let last_weight = load_vector(weight, weight_offsets + last_kernel_offset, mask, 0.0f32);
        dot = dot + x * last_weight;

        let one = constant(1.0f32, tile_shape);
        let zero = constant(0.0f32, tile_shape);
        let silu = dot * (one / (one + exp(zero - dot)));

        store_vector(out, offsets, silu, mask);
        // Shift the rolling state left.
        // `append_state` can then place the current input in the newest slot.
        for kernel in 0i32..last_kernel {
            let kernel_offset = broadcast_scalar(kernel, tile_shape);
            let next_kernel_offset = kernel_offset + one_offset;
            let shifted = load_vector(conv_state, state_offsets + next_kernel_offset, mask, 0.0f32);
            store_vector(conv_state, state_offsets + kernel_offset, shifted, mask);
        }
    }

    #[cutile::entry()]
    pub unsafe fn causal_conv1d_append_state_f32(
        conv_state: *mut f32,
        input: *mut f32,
        kernel_size: i32,
        len: i32,
    ) {
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets = iota(tile_shape) + broadcast_scalar(tile_id.0 * VECTOR_TILE_SIZE, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        // Append stores the current input as the last causal state tap.
        // `update_silu` has already shifted older taps forward.
        let state_offsets = offsets * broadcast_scalar(kernel_size, tile_shape);
        let last_kernel_offset = broadcast_scalar(kernel_size - 1i32, tile_shape);
        let x = load_vector(input, offsets, mask, 0.0f32);
        store_vector(conv_state, state_offsets + last_kernel_offset, x, mask);
    }

    #[cutile::entry()]
    pub unsafe fn causal_conv1d_update_silu_f16(
        out: *mut f16,
        conv_state: *mut f16,
        input: *mut f16,
        weight: *mut f16,
        bias: *mut f16,
        channels: i32,
        kernel_size: i32,
        len: i32,
    ) {
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets = iota(tile_shape) + broadcast_scalar(tile_id.0 * VECTOR_TILE_SIZE, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let channel = offsets
            - (offsets / broadcast_scalar(channels, tile_shape))
                * broadcast_scalar(channels, tile_shape);

        let state_offsets = offsets * broadcast_scalar(kernel_size, tile_shape);
        let weight_offsets = channel * broadcast_scalar(kernel_size, tile_shape);
        let one_offset = broadcast_scalar(1i32, tile_shape);
        let last_kernel = kernel_size - 1i32;
        let last_kernel_offset = broadcast_scalar(last_kernel, tile_shape);
        let x = load_vector_f16_as_f32(input, offsets, mask);
        let mut dot = load_vector_f16_as_f32(bias, channel, mask);
        for kernel in 0i32..last_kernel {
            let kernel_offset = broadcast_scalar(kernel, tile_shape);
            let shifted_kernel_offset = kernel_offset + one_offset;
            let value =
                load_vector_f16_as_f32(conv_state, state_offsets + shifted_kernel_offset, mask);
            let weight_value = load_vector_f16_as_f32(weight, weight_offsets + kernel_offset, mask);
            dot = dot + value * weight_value;
        }
        let last_weight = load_vector_f16_as_f32(weight, weight_offsets + last_kernel_offset, mask);
        dot = dot + x * last_weight;

        let one = constant(1.0f32, tile_shape);
        let zero = constant(0.0f32, tile_shape);
        let silu = dot * (one / (one + exp(zero - dot)));

        store_vector_f16_from_f32(out, offsets, silu, mask);
        for kernel in 0i32..last_kernel {
            let kernel_offset = broadcast_scalar(kernel, tile_shape);
            let next_kernel_offset = kernel_offset + one_offset;
            let shifted =
                load_vector_f16_as_f32(conv_state, state_offsets + next_kernel_offset, mask);
            store_vector_f16_from_f32(conv_state, state_offsets + kernel_offset, shifted, mask);
        }
    }

    #[cutile::entry()]
    pub unsafe fn causal_conv1d_append_state_f16(
        conv_state: *mut f16,
        input: *mut f16,
        kernel_size: i32,
        len: i32,
    ) {
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets = iota(tile_shape) + broadcast_scalar(tile_id.0 * VECTOR_TILE_SIZE, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let state_offsets = offsets * broadcast_scalar(kernel_size, tile_shape);
        let last_kernel_offset = broadcast_scalar(kernel_size - 1i32, tile_shape);
        let x = load_vector_f16_as_f32(input, offsets, mask);
        store_vector_f16_from_f32(conv_state, state_offsets + last_kernel_offset, x, mask);
    }

    #[cutile::entry()]
    pub unsafe fn causal_conv1d_update_silu_bf16(
        out: *mut bf16,
        conv_state: *mut bf16,
        input: *mut bf16,
        weight: *mut bf16,
        bias: *mut bf16,
        channels: i32,
        kernel_size: i32,
        len: i32,
    ) {
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets = iota(tile_shape) + broadcast_scalar(tile_id.0 * VECTOR_TILE_SIZE, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let channel = offsets
            - (offsets / broadcast_scalar(channels, tile_shape))
                * broadcast_scalar(channels, tile_shape);

        let state_offsets = offsets * broadcast_scalar(kernel_size, tile_shape);
        let weight_offsets = channel * broadcast_scalar(kernel_size, tile_shape);
        let one_offset = broadcast_scalar(1i32, tile_shape);
        let last_kernel = kernel_size - 1i32;
        let last_kernel_offset = broadcast_scalar(last_kernel, tile_shape);
        let x = load_vector_bf16_as_f32(input, offsets, mask);
        let mut dot = load_vector_bf16_as_f32(bias, channel, mask);
        for kernel in 0i32..last_kernel {
            let kernel_offset = broadcast_scalar(kernel, tile_shape);
            let shifted_kernel_offset = kernel_offset + one_offset;
            let value =
                load_vector_bf16_as_f32(conv_state, state_offsets + shifted_kernel_offset, mask);
            let weight_value =
                load_vector_bf16_as_f32(weight, weight_offsets + kernel_offset, mask);
            dot = dot + value * weight_value;
        }
        let last_weight =
            load_vector_bf16_as_f32(weight, weight_offsets + last_kernel_offset, mask);
        dot = dot + x * last_weight;

        let one = constant(1.0f32, tile_shape);
        let zero = constant(0.0f32, tile_shape);
        let silu = dot * (one / (one + exp(zero - dot)));

        store_vector_bf16_from_f32(out, offsets, silu, mask);
        for kernel in 0i32..last_kernel {
            let kernel_offset = broadcast_scalar(kernel, tile_shape);
            let next_kernel_offset = kernel_offset + one_offset;
            let shifted =
                load_vector_bf16_as_f32(conv_state, state_offsets + next_kernel_offset, mask);
            store_vector_bf16_from_f32(conv_state, state_offsets + kernel_offset, shifted, mask);
        }
    }

    #[cutile::entry()]
    pub unsafe fn causal_conv1d_append_state_bf16(
        conv_state: *mut bf16,
        input: *mut bf16,
        kernel_size: i32,
        len: i32,
    ) {
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets = iota(tile_shape) + broadcast_scalar(tile_id.0 * VECTOR_TILE_SIZE, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let state_offsets = offsets * broadcast_scalar(kernel_size, tile_shape);
        let last_kernel_offset = broadcast_scalar(kernel_size - 1i32, tile_shape);
        let x = load_vector_bf16_as_f32(input, offsets, mask);
        store_vector_bf16_from_f32(conv_state, state_offsets + last_kernel_offset, x, mask);
    }

    #[cutile::entry()]
    pub unsafe fn causal_conv1d_prefill_silu_f32(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        channels: i32,
        kernel_size: i32,
        input_length: i32,
        output_length: i32,
    ) {
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let channel = tile_id.0;
        let batch = tile_id.2;
        let time_offsets =
            iota(tile_shape) + broadcast_scalar(tile_id.1 * VECTOR_TILE_SIZE, tile_shape);
        let output_mask = cmpi(
            time_offsets,
            broadcast_scalar(output_length, tile_shape),
            predicate::LessThan,
        );

        let input_batch_offset = batch * channels * input_length;
        let output_batch_offset = batch * channels * output_length;
        // Prefill computes a contiguous time tile for one batch and channel.
        let channel_input_offset =
            broadcast_scalar(input_batch_offset + channel * input_length, tile_shape);
        let channel_output_offset =
            broadcast_scalar(output_batch_offset + channel * output_length, tile_shape);
        let channel_weight_offset = broadcast_scalar(channel * kernel_size, tile_shape);
        let zero_i32 = broadcast_scalar(0i32, tile_shape);
        let input_bound = broadcast_scalar(input_length, tile_shape);
        let mut dot = load_vector(
            bias,
            broadcast_scalar(channel, tile_shape),
            output_mask,
            0.0f32,
        );
        // Causal indexing aligns the last kernel tap with the current output time.
        // It masks taps before the sequence start.
        for kernel in 0i32..kernel_size {
            let kernel_offset = broadcast_scalar(kernel, tile_shape);
            let input_index =
                time_offsets + kernel_offset + broadcast_scalar(1i32 - kernel_size, tile_shape);
            let input_mask = output_mask
                & cmpi(input_index, zero_i32, predicate::GreaterThanOrEqual)
                & cmpi(input_index, input_bound, predicate::LessThan);
            let value = load_vector(
                input,
                channel_input_offset + input_index,
                input_mask,
                0.0f32,
            );
            let weight_value = load_vector(
                weight,
                channel_weight_offset + kernel_offset,
                output_mask,
                0.0f32,
            );
            dot = dot + value * weight_value;
        }

        let one = constant(1.0f32, tile_shape);
        let zero = constant(0.0f32, tile_shape);
        let silu = dot * (one / (one + exp(zero - dot)));

        store_vector(out, channel_output_offset + time_offsets, silu, output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn causal_conv1d_prefill_state_f32(
        conv_state: *mut f32,
        input: *mut f32,
        kernel_size: i32,
        input_length: i32,
        len: i32,
    ) {
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets = iota(tile_shape) + broadcast_scalar(tile_id.0 * VECTOR_TILE_SIZE, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let input_token = offsets / broadcast_scalar(kernel_size, tile_shape);
        let kernel = offsets - input_token * broadcast_scalar(kernel_size, tile_shape);
        // The prefill state is the tail window of each input sequence.
        // It uses zero fill when the sequence is shorter than the convolution kernel.
        let input_time = broadcast_scalar(input_length - kernel_size, tile_shape) + kernel;
        let input_mask = mask
            & cmpi(
                input_time,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            );
        let value = load_vector(
            input,
            input_token * broadcast_scalar(input_length, tile_shape) + input_time,
            input_mask,
            0.0f32,
        );
        store_vector(conv_state, offsets, value, mask);
    }

    #[cutile::entry()]
    pub unsafe fn causal_conv1d_prefill_silu_f16(
        out: *mut f16,
        input: *mut f16,
        weight: *mut f16,
        bias: *mut f16,
        channels: i32,
        kernel_size: i32,
        input_length: i32,
        output_length: i32,
    ) {
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let channel = tile_id.0;
        let batch = tile_id.2;
        let time_offsets =
            iota(tile_shape) + broadcast_scalar(tile_id.1 * VECTOR_TILE_SIZE, tile_shape);
        let output_mask = cmpi(
            time_offsets,
            broadcast_scalar(output_length, tile_shape),
            predicate::LessThan,
        );

        let input_batch_offset = batch * channels * input_length;
        let output_batch_offset = batch * channels * output_length;
        let channel_input_offset =
            broadcast_scalar(input_batch_offset + channel * input_length, tile_shape);
        let channel_output_offset =
            broadcast_scalar(output_batch_offset + channel * output_length, tile_shape);
        let channel_weight_offset = broadcast_scalar(channel * kernel_size, tile_shape);
        let zero_i32 = broadcast_scalar(0i32, tile_shape);
        let input_bound = broadcast_scalar(input_length, tile_shape);
        let mut dot =
            load_vector_f16_as_f32(bias, broadcast_scalar(channel, tile_shape), output_mask);
        for kernel in 0i32..kernel_size {
            let kernel_offset = broadcast_scalar(kernel, tile_shape);
            let input_index =
                time_offsets + kernel_offset + broadcast_scalar(1i32 - kernel_size, tile_shape);
            let input_mask = output_mask
                & cmpi(input_index, zero_i32, predicate::GreaterThanOrEqual)
                & cmpi(input_index, input_bound, predicate::LessThan);
            let value =
                load_vector_f16_as_f32(input, channel_input_offset + input_index, input_mask);
            let weight_value =
                load_vector_f16_as_f32(weight, channel_weight_offset + kernel_offset, output_mask);
            dot = dot + value * weight_value;
        }

        let one = constant(1.0f32, tile_shape);
        let zero = constant(0.0f32, tile_shape);
        let silu = dot * (one / (one + exp(zero - dot)));

        store_vector_f16_from_f32(out, channel_output_offset + time_offsets, silu, output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn causal_conv1d_prefill_state_f16(
        conv_state: *mut f16,
        input: *mut f16,
        kernel_size: i32,
        input_length: i32,
        len: i32,
    ) {
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets = iota(tile_shape) + broadcast_scalar(tile_id.0 * VECTOR_TILE_SIZE, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let input_token = offsets / broadcast_scalar(kernel_size, tile_shape);
        let kernel = offsets - input_token * broadcast_scalar(kernel_size, tile_shape);
        let input_time = broadcast_scalar(input_length - kernel_size, tile_shape) + kernel;
        let input_mask = mask
            & cmpi(
                input_time,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            );
        let value = load_vector_f16_as_f32(
            input,
            input_token * broadcast_scalar(input_length, tile_shape) + input_time,
            input_mask,
        );
        store_vector_f16_from_f32(conv_state, offsets, value, mask);
    }

    #[cutile::entry()]
    pub unsafe fn causal_conv1d_prefill_silu_bf16(
        out: *mut bf16,
        input: *mut bf16,
        weight: *mut bf16,
        bias: *mut bf16,
        channels: i32,
        kernel_size: i32,
        input_length: i32,
        output_length: i32,
    ) {
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let channel = tile_id.0;
        let batch = tile_id.2;
        let time_offsets =
            iota(tile_shape) + broadcast_scalar(tile_id.1 * VECTOR_TILE_SIZE, tile_shape);
        let output_mask = cmpi(
            time_offsets,
            broadcast_scalar(output_length, tile_shape),
            predicate::LessThan,
        );

        let input_batch_offset = batch * channels * input_length;
        let output_batch_offset = batch * channels * output_length;
        let channel_input_offset =
            broadcast_scalar(input_batch_offset + channel * input_length, tile_shape);
        let channel_output_offset =
            broadcast_scalar(output_batch_offset + channel * output_length, tile_shape);
        let channel_weight_offset = broadcast_scalar(channel * kernel_size, tile_shape);
        let zero_i32 = broadcast_scalar(0i32, tile_shape);
        let input_bound = broadcast_scalar(input_length, tile_shape);
        let mut dot =
            load_vector_bf16_as_f32(bias, broadcast_scalar(channel, tile_shape), output_mask);
        for kernel in 0i32..kernel_size {
            let kernel_offset = broadcast_scalar(kernel, tile_shape);
            let input_index =
                time_offsets + kernel_offset + broadcast_scalar(1i32 - kernel_size, tile_shape);
            let input_mask = output_mask
                & cmpi(input_index, zero_i32, predicate::GreaterThanOrEqual)
                & cmpi(input_index, input_bound, predicate::LessThan);
            let value =
                load_vector_bf16_as_f32(input, channel_input_offset + input_index, input_mask);
            let weight_value =
                load_vector_bf16_as_f32(weight, channel_weight_offset + kernel_offset, output_mask);
            dot = dot + value * weight_value;
        }

        let one = constant(1.0f32, tile_shape);
        let zero = constant(0.0f32, tile_shape);
        let silu = dot * (one / (one + exp(zero - dot)));

        store_vector_bf16_from_f32(out, channel_output_offset + time_offsets, silu, output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn causal_conv1d_prefill_state_bf16(
        conv_state: *mut bf16,
        input: *mut bf16,
        kernel_size: i32,
        input_length: i32,
        len: i32,
    ) {
        let tile_id: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets = iota(tile_shape) + broadcast_scalar(tile_id.0 * VECTOR_TILE_SIZE, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let input_token = offsets / broadcast_scalar(kernel_size, tile_shape);
        let kernel = offsets - input_token * broadcast_scalar(kernel_size, tile_shape);
        let input_time = broadcast_scalar(input_length - kernel_size, tile_shape) + kernel;
        let input_mask = mask
            & cmpi(
                input_time,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            );
        let value = load_vector_bf16_as_f32(
            input,
            input_token * broadcast_scalar(input_length, tile_shape) + input_time,
            input_mask,
        );
        store_vector_bf16_from_f32(conv_state, offsets, value, mask);
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_f32_k1(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f32::<1, false, false>(
            out,
            input,
            weight,
            input,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_f32_k2(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f32::<2, false, false>(
            out,
            input,
            weight,
            input,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_f32_k3(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f32::<3, false, false>(
            out,
            input,
            weight,
            input,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_f32_k5(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f32::<5, false, false>(
            out,
            input,
            weight,
            input,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_f32_k1(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_f32::<1>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_f32_k2(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_f32::<2>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_f32_k3(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_f32::<3>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_f32_k5(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_f32::<5>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_im2col_f32_k1(
        out: *mut f32,
        input: *mut f32,
        channels_in: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_im2col_f32::<1>(
            out,
            input,
            channels_in,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_im2col_f32_k2(
        out: *mut f32,
        input: *mut f32,
        channels_in: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_im2col_f32::<2>(
            out,
            input,
            channels_in,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_im2col_f32_k3(
        out: *mut f32,
        input: *mut f32,
        channels_in: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_im2col_f32::<3>(
            out,
            input,
            channels_in,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_im2col_f32_k5(
        out: *mut f32,
        input: *mut f32,
        channels_in: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_im2col_f32::<5>(
            out,
            input,
            channels_in,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_f16_f32_k1(
        out: *mut f32,
        input: *mut f16,
        weight: *mut f16,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f16_f32::<1>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_f16_f32_k2(
        out: *mut f32,
        input: *mut f16,
        weight: *mut f16,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f16_f32::<2>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_f16_f32_k3(
        out: *mut f32,
        input: *mut f16,
        weight: *mut f16,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f16_f32::<3>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_f16_f32_k5(
        out: *mut f32,
        input: *mut f16,
        weight: *mut f16,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f16_f32::<5>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_f16_f32_k1(
        out: *mut f32,
        input: *mut f16,
        weight: *mut f16,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_f16_f32::<1>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_f16_f32_k2(
        out: *mut f32,
        input: *mut f16,
        weight: *mut f16,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_f16_f32::<2>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_f16_f32_k3(
        out: *mut f32,
        input: *mut f16,
        weight: *mut f16,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_f16_f32::<3>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_f16_f32_k5(
        out: *mut f32,
        input: *mut f16,
        weight: *mut f16,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_f16_f32::<5>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_bf16_f32_k1(
        out: *mut f32,
        input: *mut bf16,
        weight: *mut bf16,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_bf16_f32::<1>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_bf16_f32_k2(
        out: *mut f32,
        input: *mut bf16,
        weight: *mut bf16,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_bf16_f32::<2>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_bf16_f32_k3(
        out: *mut f32,
        input: *mut bf16,
        weight: *mut bf16,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_bf16_f32::<3>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_bf16_f32_k5(
        out: *mut f32,
        input: *mut bf16,
        weight: *mut bf16,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_bf16_f32::<5>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_bf16_f32_k1(
        out: *mut f32,
        input: *mut bf16,
        weight: *mut bf16,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_bf16_f32::<1>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_bf16_f32_k2(
        out: *mut f32,
        input: *mut bf16,
        weight: *mut bf16,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_bf16_f32::<2>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_bf16_f32_k3(
        out: *mut f32,
        input: *mut bf16,
        weight: *mut bf16,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_bf16_f32::<3>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_batched_bf16_f32_k5(
        out: *mut f32,
        input: *mut bf16,
        weight: *mut bf16,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        conv1d_batched_bf16_f32::<5>(
            out,
            input,
            weight,
            channels_in,
            _channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_bias_f32_k1(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f32::<1, true, false>(
            out,
            input,
            weight,
            bias,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_bias_f32_k2(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f32::<2, true, false>(
            out,
            input,
            weight,
            bias,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_bias_f32_k3(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f32::<3, true, false>(
            out,
            input,
            weight,
            bias,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_bias_f32_k5(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f32::<5, true, false>(
            out,
            input,
            weight,
            bias,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_bias_gelu_f32_k1(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f32::<1, true, true>(
            out,
            input,
            weight,
            bias,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_bias_gelu_f32_k2(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f32::<2, true, true>(
            out,
            input,
            weight,
            bias,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_bias_gelu_f32_k3(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f32::<3, true, true>(
            out,
            input,
            weight,
            bias,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn conv1d_causal_bias_gelu_f32_k5(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        conv1d_causal_f32::<5, true, true>(
            out,
            input,
            weight,
            bias,
            channels_in,
            _channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len,
        );
    }

    fn conv1d_causal_f32<const K: i32, const HAS_BIAS: bool, const GELU: bool>(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let out_channel = offsets / broadcast_scalar(output_length, tile_shape);
        let out_pos = offsets - out_channel * broadcast_scalar(output_length, tile_shape);
        let absolute_out_pos = out_pos + broadcast_scalar(output_start, tile_shape);
        let input_end = input_start + input_length;
        let channels_in_per_group = channels_in / groups;
        let channels_out_per_group = _channels_out / groups;
        let group = out_channel / broadcast_scalar(channels_out_per_group, tile_shape);

        let mut sum = constant(0.0f32, tile_shape);
        if HAS_BIAS {
            sum = load_vector(bias, out_channel, valid, 0.0f32);
        }
        // `input_start` and `output_start` let callers run causal convolution over sequence chunks.
        // They preserve absolute positions.
        for group_input_channel in 0i32..channels_in_per_group {
            let input_channel = group * broadcast_scalar(channels_in_per_group, tile_shape)
                + broadcast_scalar(group_input_channel, tile_shape);
            for kernel_index in 0i32..K {
                let input_pos = absolute_out_pos * broadcast_scalar(stride, tile_shape)
                    + broadcast_scalar(kernel_index * dilation, tile_shape)
                    - broadcast_scalar(left_padding, tile_shape);
                let input_valid = valid
                    & cmpi(
                        input_pos,
                        broadcast_scalar(input_start, tile_shape),
                        predicate::GreaterThanOrEqual,
                    )
                    & cmpi(
                        input_pos,
                        broadcast_scalar(input_end, tile_shape),
                        predicate::LessThan,
                    );
                let input_offsets = input_channel * broadcast_scalar(input_length, tile_shape)
                    + input_pos
                    - broadcast_scalar(input_start, tile_shape);
                let input_values = load_vector(input, input_offsets, input_valid, 0.0f32);
                let weight_offsets = out_channel
                    * broadcast_scalar(channels_in_per_group * K, tile_shape)
                    + broadcast_scalar(group_input_channel * K + kernel_index, tile_shape);
                let weight_values = load_vector(weight, weight_offsets, valid, 0.0f32);
                sum = sum + input_values * weight_values;
            }
        }
        if GELU {
            // Optional approximate GELU matches the fused causal-conv variants.
            let half = constant(0.5f32, tile_shape);
            let one = constant(1.0f32, tile_shape);
            let sqrt_2_over_pi = constant(0.7978846f32, tile_shape);
            let cubic_coeff = constant(0.044715f32, tile_shape);
            sum = half * sum * (one + tanh(sqrt_2_over_pi * (sum + cubic_coeff * sum * sum * sum)));
        }
        store_vector(out, offsets, sum, valid);
    }

    fn conv1d_batched_f32<const K: i32>(
        out: *mut f32,
        input: *mut f32,
        weight: *mut f32,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch * broadcast_scalar(output_values_per_batch, tile_shape);
        let out_channel = batch_output_offset / broadcast_scalar(output_length, tile_shape);
        let out_pos =
            batch_output_offset - out_channel * broadcast_scalar(output_length, tile_shape);
        let input_end = input_length;
        let channels_in_per_group = channels_in / groups;
        let channels_out_per_group = _channels_out / groups;
        let group = out_channel / broadcast_scalar(channels_out_per_group, tile_shape);

        let mut sum = constant(0.0f32, tile_shape);
        for group_input_channel in 0i32..channels_in_per_group {
            let input_channel = group * broadcast_scalar(channels_in_per_group, tile_shape)
                + broadcast_scalar(group_input_channel, tile_shape);
            for kernel_index in 0i32..K {
                let input_pos = out_pos * broadcast_scalar(stride, tile_shape)
                    + broadcast_scalar(kernel_index * dilation, tile_shape)
                    - broadcast_scalar(left_padding, tile_shape);
                let input_valid = valid
                    & cmpi(
                        input_pos,
                        broadcast_scalar(0i32, tile_shape),
                        predicate::GreaterThanOrEqual,
                    )
                    & cmpi(
                        input_pos,
                        broadcast_scalar(input_end, tile_shape),
                        predicate::LessThan,
                    );
                let input_offsets = batch * broadcast_scalar(input_batch_stride, tile_shape)
                    + input_channel * broadcast_scalar(input_length, tile_shape)
                    + input_pos;
                let input_values = load_vector(input, input_offsets, input_valid, 0.0f32);
                let weight_offsets = out_channel
                    * broadcast_scalar(channels_in_per_group * K, tile_shape)
                    + broadcast_scalar(group_input_channel * K + kernel_index, tile_shape);
                let weight_values = load_vector(weight, weight_offsets, valid, 0.0f32);
                sum = sum + input_values * weight_values;
            }
        }
        let output_offsets =
            batch * broadcast_scalar(output_batch_stride, tile_shape) + batch_output_offset;
        store_vector(out, output_offsets, sum, valid);
    }

    fn conv1d_batched_im2col_f32<const K: i32>(
        out: *mut f32,
        input: *mut f32,
        channels_in: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch * broadcast_scalar(output_values_per_batch, tile_shape);
        let channels_in_per_group = channels_in / groups;
        let group_values = output_length * channels_in_per_group * K;
        // Materialize grouped im2col as `[group, output_position, input_channel_within_group, kernel_index]`.
        let group = batch_output_offset / broadcast_scalar(group_values, tile_shape);
        let group_offset = batch_output_offset - group * broadcast_scalar(group_values, tile_shape);
        let output_position_values = channels_in_per_group * K;
        let out_pos = group_offset / broadcast_scalar(output_position_values, tile_shape);
        let output_position_offset =
            group_offset - out_pos * broadcast_scalar(output_position_values, tile_shape);
        let group_input_channel = output_position_offset / broadcast_scalar(K, tile_shape);
        let kernel_index =
            output_position_offset - group_input_channel * broadcast_scalar(K, tile_shape);
        let input_channel =
            group * broadcast_scalar(channels_in_per_group, tile_shape) + group_input_channel;
        let input_pos = out_pos * broadcast_scalar(stride, tile_shape)
            + kernel_index * broadcast_scalar(dilation, tile_shape)
            - broadcast_scalar(left_padding, tile_shape);
        let input_valid = valid
            & cmpi(
                input_pos,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            )
            & cmpi(
                input_pos,
                broadcast_scalar(input_length, tile_shape),
                predicate::LessThan,
            );
        let input_offsets = batch * broadcast_scalar(input_batch_stride, tile_shape)
            + input_channel * broadcast_scalar(input_length, tile_shape)
            + input_pos;
        let values = load_vector(input, input_offsets, input_valid, 0.0f32);
        store_vector(out, offsets, values, valid);
    }

    fn conv1d_causal_f16_f32<const K: i32>(
        out: *mut f32,
        input: *mut f16,
        weight: *mut f16,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let out_channel = offsets / broadcast_scalar(output_length, tile_shape);
        let out_pos = offsets - out_channel * broadcast_scalar(output_length, tile_shape);
        let absolute_out_pos = out_pos + broadcast_scalar(output_start, tile_shape);
        let input_end = input_start + input_length;
        let channels_in_per_group = channels_in / groups;
        let channels_out_per_group = _channels_out / groups;
        let group = out_channel / broadcast_scalar(channels_out_per_group, tile_shape);

        let mut sum = constant(0.0f32, tile_shape);
        for group_input_channel in 0i32..channels_in_per_group {
            let input_channel = group * broadcast_scalar(channels_in_per_group, tile_shape)
                + broadcast_scalar(group_input_channel, tile_shape);
            for kernel_index in 0i32..K {
                let input_pos = absolute_out_pos * broadcast_scalar(stride, tile_shape)
                    + broadcast_scalar(kernel_index * dilation, tile_shape)
                    - broadcast_scalar(left_padding, tile_shape);
                let input_valid = valid
                    & cmpi(
                        input_pos,
                        broadcast_scalar(input_start, tile_shape),
                        predicate::GreaterThanOrEqual,
                    )
                    & cmpi(
                        input_pos,
                        broadcast_scalar(input_end, tile_shape),
                        predicate::LessThan,
                    );
                let input_offsets = input_channel * broadcast_scalar(input_length, tile_shape)
                    + input_pos
                    - broadcast_scalar(input_start, tile_shape);
                let input_values = load_vector_f16_as_f32(input, input_offsets, input_valid);
                let weight_offsets = out_channel
                    * broadcast_scalar(channels_in_per_group * K, tile_shape)
                    + broadcast_scalar(group_input_channel * K + kernel_index, tile_shape);
                let weight_values = load_vector_f16_as_f32(weight, weight_offsets, valid);
                sum = sum + input_values * weight_values;
            }
        }
        store_vector(out, offsets, sum, valid);
    }

    fn conv1d_batched_f16_f32<const K: i32>(
        out: *mut f32,
        input: *mut f16,
        weight: *mut f16,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch * broadcast_scalar(output_values_per_batch, tile_shape);
        let out_channel = batch_output_offset / broadcast_scalar(output_length, tile_shape);
        let out_pos =
            batch_output_offset - out_channel * broadcast_scalar(output_length, tile_shape);
        let input_end = input_length;
        let channels_in_per_group = channels_in / groups;
        let channels_out_per_group = _channels_out / groups;
        let group = out_channel / broadcast_scalar(channels_out_per_group, tile_shape);

        let mut sum = constant(0.0f32, tile_shape);
        for group_input_channel in 0i32..channels_in_per_group {
            let input_channel = group * broadcast_scalar(channels_in_per_group, tile_shape)
                + broadcast_scalar(group_input_channel, tile_shape);
            for kernel_index in 0i32..K {
                let input_pos = out_pos * broadcast_scalar(stride, tile_shape)
                    + broadcast_scalar(kernel_index * dilation, tile_shape)
                    - broadcast_scalar(left_padding, tile_shape);
                let input_valid = valid
                    & cmpi(
                        input_pos,
                        broadcast_scalar(0i32, tile_shape),
                        predicate::GreaterThanOrEqual,
                    )
                    & cmpi(
                        input_pos,
                        broadcast_scalar(input_end, tile_shape),
                        predicate::LessThan,
                    );
                let input_offsets = batch * broadcast_scalar(input_batch_stride, tile_shape)
                    + input_channel * broadcast_scalar(input_length, tile_shape)
                    + input_pos;
                let input_values = load_vector_f16_as_f32(input, input_offsets, input_valid);
                let weight_offsets = out_channel
                    * broadcast_scalar(channels_in_per_group * K, tile_shape)
                    + broadcast_scalar(group_input_channel * K + kernel_index, tile_shape);
                let weight_values = load_vector_f16_as_f32(weight, weight_offsets, valid);
                sum = sum + input_values * weight_values;
            }
        }
        let output_offsets =
            batch * broadcast_scalar(output_batch_stride, tile_shape) + batch_output_offset;
        store_vector(out, output_offsets, sum, valid);
    }

    fn conv1d_causal_bf16_f32<const K: i32>(
        out: *mut f32,
        input: *mut bf16,
        weight: *mut bf16,
        channels_in: i32,
        _channels_out: i32,
        input_start: i32,
        input_length: i32,
        output_start: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let out_channel = offsets / broadcast_scalar(output_length, tile_shape);
        let out_pos = offsets - out_channel * broadcast_scalar(output_length, tile_shape);
        let absolute_out_pos = out_pos + broadcast_scalar(output_start, tile_shape);
        let input_end = input_start + input_length;
        let channels_in_per_group = channels_in / groups;
        let channels_out_per_group = _channels_out / groups;
        let group = out_channel / broadcast_scalar(channels_out_per_group, tile_shape);

        let mut sum = constant(0.0f32, tile_shape);
        for group_input_channel in 0i32..channels_in_per_group {
            let input_channel = group * broadcast_scalar(channels_in_per_group, tile_shape)
                + broadcast_scalar(group_input_channel, tile_shape);
            for kernel_index in 0i32..K {
                let input_pos = absolute_out_pos * broadcast_scalar(stride, tile_shape)
                    + broadcast_scalar(kernel_index * dilation, tile_shape)
                    - broadcast_scalar(left_padding, tile_shape);
                let input_valid = valid
                    & cmpi(
                        input_pos,
                        broadcast_scalar(input_start, tile_shape),
                        predicate::GreaterThanOrEqual,
                    )
                    & cmpi(
                        input_pos,
                        broadcast_scalar(input_end, tile_shape),
                        predicate::LessThan,
                    );
                let input_offsets = input_channel * broadcast_scalar(input_length, tile_shape)
                    + input_pos
                    - broadcast_scalar(input_start, tile_shape);
                let input_values = load_vector_bf16_as_f32(input, input_offsets, input_valid);
                let weight_offsets = out_channel
                    * broadcast_scalar(channels_in_per_group * K, tile_shape)
                    + broadcast_scalar(group_input_channel * K + kernel_index, tile_shape);
                let weight_values = load_vector_bf16_as_f32(weight, weight_offsets, valid);
                sum = sum + input_values * weight_values;
            }
        }
        store_vector(out, offsets, sum, valid);
    }

    fn conv1d_batched_bf16_f32<const K: i32>(
        out: *mut f32,
        input: *mut bf16,
        weight: *mut bf16,
        channels_in: i32,
        _channels_out: i32,
        input_length: i32,
        output_length: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        left_padding: i32,
        input_batch_stride: i32,
        output_batch_stride: i32,
        output_values_per_batch: i32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch * broadcast_scalar(output_values_per_batch, tile_shape);
        let out_channel = batch_output_offset / broadcast_scalar(output_length, tile_shape);
        let out_pos =
            batch_output_offset - out_channel * broadcast_scalar(output_length, tile_shape);
        let input_end = input_length;
        let channels_in_per_group = channels_in / groups;
        let channels_out_per_group = _channels_out / groups;
        let group = out_channel / broadcast_scalar(channels_out_per_group, tile_shape);

        let mut sum = constant(0.0f32, tile_shape);
        for group_input_channel in 0i32..channels_in_per_group {
            let input_channel = group * broadcast_scalar(channels_in_per_group, tile_shape)
                + broadcast_scalar(group_input_channel, tile_shape);
            for kernel_index in 0i32..K {
                let input_pos = out_pos * broadcast_scalar(stride, tile_shape)
                    + broadcast_scalar(kernel_index * dilation, tile_shape)
                    - broadcast_scalar(left_padding, tile_shape);
                let input_valid = valid
                    & cmpi(
                        input_pos,
                        broadcast_scalar(0i32, tile_shape),
                        predicate::GreaterThanOrEqual,
                    )
                    & cmpi(
                        input_pos,
                        broadcast_scalar(input_end, tile_shape),
                        predicate::LessThan,
                    );
                let input_offsets = batch * broadcast_scalar(input_batch_stride, tile_shape)
                    + input_channel * broadcast_scalar(input_length, tile_shape)
                    + input_pos;
                let input_values = load_vector_bf16_as_f32(input, input_offsets, input_valid);
                let weight_offsets = out_channel
                    * broadcast_scalar(channels_in_per_group * K, tile_shape)
                    + broadcast_scalar(group_input_channel * K + kernel_index, tile_shape);
                let weight_values = load_vector_bf16_as_f32(weight, weight_offsets, valid);
                sum = sum + input_values * weight_values;
            }
        }
        let output_offsets =
            batch * broadcast_scalar(output_batch_stride, tile_shape) + batch_output_offset;
        store_vector(out, output_offsets, sum, valid);
    }

    fn load_vector(
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

    fn load_vector_f16_as_f32(
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

    fn load_vector_bf16_as_f32(
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

    fn store_vector(
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

    fn store_vector_f16_from_f32(
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

    fn store_vector_bf16_from_f32(
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

    fn erf_approx_tile_128(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        let shape = const_shape![128];
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
}

pub use kernels::*;
