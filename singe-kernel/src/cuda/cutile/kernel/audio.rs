#[cutile::module]
mod kernels {
    use cutile::core::*;

    const VECTOR_TILE_SIZE: i32 = 128;

    #[cutile::entry()]
    pub unsafe fn r2c_power_f32(out: *mut f32, spectrum: *mut f32, output_len: i32) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let spectrum_offsets = offsets * broadcast_scalar(2i32, tile_shape);
        let real = load_vector(spectrum, spectrum_offsets, valid, 0.0f32);
        let imag = load_vector(
            spectrum,
            spectrum_offsets + broadcast_scalar(1i32, tile_shape),
            valid,
            0.0f32,
        );
        store_vector(out, offsets, real * real + imag * imag, valid);
    }

    #[cutile::entry()]
    pub unsafe fn stft_r2c_stage1_f32_400(
        workspace: *mut f32,
        input: *mut f32,
        window: *mut f32,
        dft_real: *mut f32,
        dft_imag: *mut f32,
        input_len: i32,
        batch: i32,
        frames: i32,
        first_frame: i32,
        hop_length: i32,
        pad: i32,
        pad_mode: i32,
    ) {
        stft_r2c_stage1_f32::<16, 25, 400>(
            workspace,
            input,
            window,
            dft_real,
            dft_imag,
            input_len,
            batch,
            frames,
            first_frame,
            hop_length,
            pad,
            pad_mode,
        );
    }

    #[cutile::entry()]
    pub unsafe fn stft_r2c_stage2_f32_400_201(
        out: *mut f32,
        workspace: *mut f32,
        dft_real: *mut f32,
        dft_imag: *mut f32,
        batch: i32,
        frames: i32,
    ) {
        stft_r2c_stage2_f32::<16, 25, 400, 201>(out, workspace, dft_real, dft_imag, batch, frames);
    }

    #[cutile::entry()]
    pub unsafe fn log_mel_f32_201_128(
        out: *mut f32,
        power: *mut f32,
        mel_filters: *mut f32,
        batch: i32,
        frames: i32,
        floor_value: f32,
        reference_max: f32,
        dynamic_range: f32,
        offset: f32,
        scale: f32,
        layout: i32,
        first_frame: i32,
        output_frame_stride: i32,
    ) {
        log_mel_f32::<201, 128>(
            out,
            power,
            mel_filters,
            batch,
            frames,
            floor_value,
            reference_max,
            dynamic_range,
            offset,
            scale,
            layout,
            first_frame,
            output_frame_stride,
        );
    }

    #[cutile::entry()]
    pub unsafe fn log_mel_partial_max_f32_201_128(
        partial: *mut f32,
        power: *mut f32,
        mel_filters: *mut f32,
        batch: i32,
        frames: i32,
        partial_per_batch: i32,
        floor_value: f32,
    ) {
        log_mel_partial_max_f32::<201, 128>(
            partial,
            power,
            mel_filters,
            batch,
            frames,
            partial_per_batch,
            floor_value,
        );
    }

    #[cutile::entry()]
    pub unsafe fn log_mel_reduce_partial_max_f32(
        reference_max: *mut f32,
        partial: *mut f32,
        partial_len: i32,
    ) {
        let tile_shape = const_shape![1024];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [1024] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * partial_len, tile_shape);
        let end = broadcast_scalar((pid.0 + 1i32) * partial_len, tile_shape);
        let valid = cmpi(offsets, end, predicate::LessThan);
        let values = load_vector_1024(partial, offsets, valid, -3.4028235e38f32);
        let value: Tile<f32, { [1] }> = reduce_max(values, 0i32);
        store_first_lane(reference_max, pid.0, value);
    }

    #[cutile::entry()]
    pub unsafe fn log_mel_dynamic_reference_f32_201_128(
        out: *mut f32,
        power: *mut f32,
        mel_filters: *mut f32,
        reference_max: *mut f32,
        batch: i32,
        frames: i32,
        floor_value: f32,
        dynamic_range: f32,
        offset: f32,
        scale: f32,
        layout: i32,
        first_frame: i32,
        output_frame_stride: i32,
    ) {
        log_mel_dynamic_reference_f32::<201, 128>(
            out,
            power,
            mel_filters,
            reference_max,
            batch,
            frames,
            floor_value,
            dynamic_range,
            offset,
            scale,
            layout,
            first_frame,
            output_frame_stride,
        );
    }

    fn stft_r2c_stage1_f32<const R: i32, const M: i32, const N_FFT: i32>(
        workspace: *mut f32,
        input: *mut f32,
        window: *mut f32,
        dft_real: *mut f32,
        dft_imag: *mut f32,
        input_len: i32,
        batch: i32,
        frames: i32,
        first_frame: i32,
        hop_length: i32,
        pad: i32,
        pad_mode: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let workspace_complex = broadcast_scalar(batch * frames * N_FFT, tile_shape);
        let valid = cmpi(offsets, workspace_complex, predicate::LessThan);

        let frame_linear = offsets / broadcast_scalar(N_FFT, tile_shape);
        let batch_index = frame_linear / broadcast_scalar(frames, tile_shape);
        let frame = frame_linear - batch_index * broadcast_scalar(frames, tile_shape);
        let absolute_frame = frame + broadcast_scalar(first_frame, tile_shape);
        let within_frame = offsets - frame_linear * broadcast_scalar(N_FFT, tile_shape);
        let residue = within_frame / broadcast_scalar(M, tile_shape);
        let q = within_frame - residue * broadcast_scalar(M, tile_shape);
        let frame_start = absolute_frame * broadcast_scalar(hop_length, tile_shape);
        let pad = broadcast_scalar(pad, tile_shape);
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let one_i32: Tile<i32, { [128] }> = constant(1i32, tile_shape);
        let true_tile: Tile<bool, { [128] }> = constant(true, tile_shape);

        let mut real = constant(0.0f32, tile_shape);
        let mut imag = constant(0.0f32, tile_shape);

        for m in 0i32..M {
            let sample = residue + broadcast_scalar(R * m, tile_shape);
            let raw_input_index = frame_start + sample - pad;
            let zero_padded = cmpi(
                broadcast_scalar(pad_mode, tile_shape),
                one_i32,
                predicate::Equal,
            );
            let in_bounds = cmpi(raw_input_index, zero_i32, predicate::GreaterThanOrEqual)
                & cmpi(
                    raw_input_index,
                    broadcast_scalar(input_len, tile_shape),
                    predicate::LessThan,
                );
            let reflected_index = reflect_index(raw_input_index, input_len);
            let zero_input_index = select(in_bounds, raw_input_index, zero_i32);
            let input_index = batch_index * broadcast_scalar(input_len, tile_shape)
                + select(zero_padded, zero_input_index, reflected_index);
            let input_mask = valid & select(zero_padded, in_bounds, true_tile);
            let input_value = load_vector(input, input_index, input_mask, 0.0f32);
            let window_value = load_vector(window, sample, valid, 0.0f32);
            let weighted = input_value * window_value;

            let twiddle_sample = broadcast_scalar(R * m, tile_shape);
            let table_offset = q * broadcast_scalar(N_FFT, tile_shape) + twiddle_sample;
            let table_real = load_vector(dft_real, table_offset, valid, 0.0f32);
            let table_imag = load_vector(dft_imag, table_offset, valid, 0.0f32);
            real = real + weighted * table_real;
            imag = imag + weighted * table_imag;
        }

        let output_offsets = offsets * broadcast_scalar(2i32, tile_shape);
        store_vector(workspace, output_offsets, real, valid);
        store_vector(
            workspace,
            output_offsets + broadcast_scalar(1i32, tile_shape),
            imag,
            valid,
        );
    }

    fn stft_r2c_stage2_f32<const R: i32, const M: i32, const N_FFT: i32, const N_FREQ: i32>(
        out: *mut f32,
        workspace: *mut f32,
        dft_real: *mut f32,
        dft_imag: *mut f32,
        batch: i32,
        frames: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let output_complex = broadcast_scalar(batch * frames * N_FREQ, tile_shape);
        let valid = cmpi(offsets, output_complex, predicate::LessThan);

        let frame_linear = offsets / broadcast_scalar(N_FREQ, tile_shape);
        let batch_index = frame_linear / broadcast_scalar(frames, tile_shape);
        let frame = frame_linear - batch_index * broadcast_scalar(frames, tile_shape);
        let frequency = offsets - frame_linear * broadcast_scalar(N_FREQ, tile_shape);
        let q = frequency
            - (frequency / broadcast_scalar(M, tile_shape)) * broadcast_scalar(M, tile_shape);
        let frame_workspace_base = (batch_index * broadcast_scalar(frames, tile_shape) + frame)
            * broadcast_scalar(N_FFT, tile_shape);

        let mut real = constant(0.0f32, tile_shape);
        let mut imag = constant(0.0f32, tile_shape);

        for residue in 0i32..R {
            let workspace_offsets =
                (frame_workspace_base + broadcast_scalar(residue * M, tile_shape) + q)
                    * broadcast_scalar(2i32, tile_shape);
            let partial_real = load_vector(workspace, workspace_offsets, valid, 0.0f32);
            let partial_imag = load_vector(
                workspace,
                workspace_offsets + broadcast_scalar(1i32, tile_shape),
                valid,
                0.0f32,
            );

            let table_offset = frequency * broadcast_scalar(N_FFT, tile_shape)
                + broadcast_scalar(residue, tile_shape);
            let twiddle_real = load_vector(dft_real, table_offset, valid, 0.0f32);
            let twiddle_imag = load_vector(dft_imag, table_offset, valid, 0.0f32);
            real = real + partial_real * twiddle_real - partial_imag * twiddle_imag;
            imag = imag + partial_real * twiddle_imag + partial_imag * twiddle_real;
        }

        let output_offsets = offsets * broadcast_scalar(2i32, tile_shape);
        store_vector(out, output_offsets, real, valid);
        store_vector(
            out,
            output_offsets + broadcast_scalar(1i32, tile_shape),
            imag,
            valid,
        );
    }

    fn log_mel_f32<const N_FREQ: i32, const N_MEL: i32>(
        out: *mut f32,
        power: *mut f32,
        mel_filters: *mut f32,
        batch: i32,
        frames: i32,
        floor_value: f32,
        reference_max: f32,
        dynamic_range: f32,
        offset: f32,
        scale: f32,
        layout: i32,
        first_frame: i32,
        output_frame_stride: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let output_len = broadcast_scalar(batch * frames * N_MEL, tile_shape);
        let valid = cmpi(offsets, output_len, predicate::LessThan);

        let mel = offsets
            - (offsets / broadcast_scalar(N_MEL, tile_shape)) * broadcast_scalar(N_MEL, tile_shape);
        let frame_linear = offsets / broadcast_scalar(N_MEL, tile_shape);
        let batch_index = frame_linear / broadcast_scalar(frames, tile_shape);
        let frame = frame_linear - batch_index * broadcast_scalar(frames, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);

        for frequency in 0i32..N_FREQ {
            let power_offsets = (batch_index * broadcast_scalar(frames, tile_shape) + frame)
                * broadcast_scalar(N_FREQ, tile_shape)
                + broadcast_scalar(frequency, tile_shape);
            let filter_offsets = mel * broadcast_scalar(N_FREQ, tile_shape)
                + broadcast_scalar(frequency, tile_shape);
            let power_values = load_vector(power, power_offsets, valid, 0.0f32);
            let filter_values = load_vector(mel_filters, filter_offsets, valid, 0.0f32);
            sum = sum + power_values * filter_values;
        }

        let floor_tile = broadcast_scalar(floor_value, tile_shape);
        let sum = max_tile(sum, floor_tile);
        let ln_10 = broadcast_scalar(2.3025851f32, tile_shape);
        let mut value = log(sum) / ln_10;
        let lower = broadcast_scalar(reference_max - dynamic_range, tile_shape);
        value = max_tile(value, lower);
        value =
            (value + broadcast_scalar(offset, tile_shape)) / broadcast_scalar(scale, tile_shape);

        let mels_first = cmpi(
            broadcast_scalar(layout, tile_shape),
            broadcast_scalar(1i32, tile_shape),
            predicate::Equal,
        );
        let absolute_frame = frame + broadcast_scalar(first_frame, tile_shape);
        let feature_len = broadcast_scalar(output_frame_stride * N_MEL, tile_shape);
        let batch_base = batch_index * feature_len;
        let mels_first_offsets =
            batch_base + mel * broadcast_scalar(output_frame_stride, tile_shape) + absolute_frame;
        let frames_first_offsets =
            batch_base + absolute_frame * broadcast_scalar(N_MEL, tile_shape) + mel;
        let output_offsets = select(mels_first, mels_first_offsets, frames_first_offsets);
        store_vector(out, output_offsets, value, valid);
    }

    fn log_mel_partial_max_f32<const N_FREQ: i32, const N_MEL: i32>(
        partial: *mut f32,
        power: *mut f32,
        mel_filters: *mut f32,
        _batch: i32,
        frames: i32,
        partial_per_batch: i32,
        floor_value: f32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let batch_index = pid.0 / partial_per_batch;
        let block_in_batch = pid.0 - batch_index * partial_per_batch;
        let local_offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(block_in_batch * VECTOR_TILE_SIZE, tile_shape);
        let output_len = broadcast_scalar(frames * N_MEL, tile_shape);
        let valid = cmpi(local_offsets, output_len, predicate::LessThan);
        let value = log_mel_unclamped_values_batched::<N_FREQ, N_MEL>(
            power,
            mel_filters,
            local_offsets,
            broadcast_scalar(batch_index, tile_shape),
            frames,
            valid,
            floor_value,
        );
        let invalid: Tile<f32, { [128] }> = constant(-3.4028235e38f32, tile_shape);
        let value = select(valid, value, invalid);
        let max_value: Tile<f32, { [1] }> = reduce_max(value, 0i32);
        store_first_lane(partial, pid.0, max_value);
    }

    fn log_mel_dynamic_reference_f32<const N_FREQ: i32, const N_MEL: i32>(
        out: *mut f32,
        power: *mut f32,
        mel_filters: *mut f32,
        reference_max: *mut f32,
        batch: i32,
        frames: i32,
        floor_value: f32,
        dynamic_range: f32,
        offset: f32,
        scale: f32,
        layout: i32,
        first_frame: i32,
        output_frame_stride: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let output_len = broadcast_scalar(batch * frames * N_MEL, tile_shape);
        let valid = cmpi(offsets, output_len, predicate::LessThan);
        let frame_linear = offsets / broadcast_scalar(N_MEL, tile_shape);
        let batch_index = frame_linear / broadcast_scalar(frames, tile_shape);
        let frame = frame_linear - batch_index * broadcast_scalar(frames, tile_shape);
        let mel = offsets - frame_linear * broadcast_scalar(N_MEL, tile_shape);
        let local_offsets = frame * broadcast_scalar(N_MEL, tile_shape) + mel;

        let mut value = log_mel_unclamped_values_batched::<N_FREQ, N_MEL>(
            power,
            mel_filters,
            local_offsets,
            batch_index,
            frames,
            valid,
            floor_value,
        );
        let reference_values = load_vector(reference_max, batch_index, valid, 0.0f32);
        let lower = reference_values - broadcast_scalar(dynamic_range, tile_shape);
        value = max_tile(value, lower);
        value =
            (value + broadcast_scalar(offset, tile_shape)) / broadcast_scalar(scale, tile_shape);

        let mels_first = cmpi(
            broadcast_scalar(layout, tile_shape),
            broadcast_scalar(1i32, tile_shape),
            predicate::Equal,
        );
        let absolute_frame = frame + broadcast_scalar(first_frame, tile_shape);
        let feature_len = broadcast_scalar(output_frame_stride * N_MEL, tile_shape);
        let batch_base = batch_index * feature_len;
        let mels_first_offsets =
            batch_base + mel * broadcast_scalar(output_frame_stride, tile_shape) + absolute_frame;
        let frames_first_offsets =
            batch_base + absolute_frame * broadcast_scalar(N_MEL, tile_shape) + mel;
        let output_offsets = select(mels_first, mels_first_offsets, frames_first_offsets);
        store_vector(out, output_offsets, value, valid);
    }

    fn log_mel_unclamped_values_batched<const N_FREQ: i32, const N_MEL: i32>(
        power: *mut f32,
        mel_filters: *mut f32,
        offsets: Tile<i32, { [128] }>,
        batch_index: Tile<i32, { [128] }>,
        frames: i32,
        valid: Tile<bool, { [128] }>,
        floor_value: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mel = offsets
            - (offsets / broadcast_scalar(N_MEL, tile_shape)) * broadcast_scalar(N_MEL, tile_shape);
        let frame = offsets / broadcast_scalar(N_MEL, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);

        for frequency in 0i32..N_FREQ {
            let power_offsets = (batch_index * broadcast_scalar(frames, tile_shape) + frame)
                * broadcast_scalar(N_FREQ, tile_shape)
                + broadcast_scalar(frequency, tile_shape);
            let filter_offsets = mel * broadcast_scalar(N_FREQ, tile_shape)
                + broadcast_scalar(frequency, tile_shape);
            let power_values = load_vector(power, power_offsets, valid, 0.0f32);
            let filter_values = load_vector(mel_filters, filter_offsets, valid, 0.0f32);
            sum = sum + power_values * filter_values;
        }

        let floor_tile = broadcast_scalar(floor_value, tile_shape);
        let sum = max_tile(sum, floor_tile);
        let ln_10 = broadcast_scalar(2.3025851f32, tile_shape);
        log(sum) / ln_10
    }

    fn log_mel_unclamped_values<const N_FREQ: i32, const N_MEL: i32>(
        power: *mut f32,
        mel_filters: *mut f32,
        offsets: Tile<i32, { [128] }>,
        valid: Tile<bool, { [128] }>,
        floor_value: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mel = offsets
            - (offsets / broadcast_scalar(N_MEL, tile_shape)) * broadcast_scalar(N_MEL, tile_shape);
        let frame = offsets / broadcast_scalar(N_MEL, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);

        for frequency in 0i32..N_FREQ {
            let power_offsets = frame * broadcast_scalar(N_FREQ, tile_shape)
                + broadcast_scalar(frequency, tile_shape);
            let filter_offsets = mel * broadcast_scalar(N_FREQ, tile_shape)
                + broadcast_scalar(frequency, tile_shape);
            let power_values = load_vector(power, power_offsets, valid, 0.0f32);
            let filter_values = load_vector(mel_filters, filter_offsets, valid, 0.0f32);
            sum = sum + power_values * filter_values;
        }

        let floor_tile = broadcast_scalar(floor_value, tile_shape);
        let sum = max_tile(sum, floor_tile);
        let ln_10 = broadcast_scalar(std::f32::consts::LN_10, tile_shape);
        log(sum) / ln_10
    }

    fn reflect_index(index: Tile<i32, { [128] }>, input_len: i32) -> Tile<i32, { [128] }> {
        let tile_shape = const_shape![128];
        let zero = constant(0i32, tile_shape);
        let input_len = broadcast_scalar(input_len, tile_shape);
        let last = input_len - broadcast_scalar(1i32, tile_shape);
        let right_reflected = broadcast_scalar(2i32, tile_shape) * last - index;
        let left_reflected = zero - index;
        let index = select(
            cmpi(index, zero, predicate::LessThan),
            left_reflected,
            index,
        );
        select(
            cmpi(index, input_len, predicate::GreaterThanOrEqual),
            right_reflected,
            index,
        )
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

    fn load_vector_1024(
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
        let result: (Tile<f32, { [1024] }>, Token) = load_ptr_tko(
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

    fn store_first_lane(out: *mut f32, offset: i32, value: Tile<f32, { [1] }>) {
        let tile_shape = const_shape![128];
        let lane: Tile<i32, { [128] }> = iota(tile_shape);
        let offsets = broadcast_scalar(offset, tile_shape);
        let values = value.reshape(const_shape![1]).broadcast(tile_shape);
        let zero: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mask = cmpi(lane, zero, predicate::Equal);
        store_vector(out, offsets, values, mask);
    }
}

pub use kernels::*;
