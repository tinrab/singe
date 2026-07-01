#[cutile::module]
mod kernels {
    use crate::cuda::cutile::kernel::utility::*;
    use cutile::core::*;

    const VECTOR_TILE_SIZE: i32 = 128;

    #[cutile::entry()]
    pub fn copy_c2c_f32_interleaved<const N: i32>(
        out: &mut Tensor<f32, { [1, N, 2] }>,
        input: &Tensor<f32, { [-1, -1, -1] }>,
    ) {
        let values: Tile<f32, { [1, N, 2] }> = load_tile_like(input, out);
        out.store(values);
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f32_interleaved_mixed_radix_8(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        mixed_radix_c2c_f32::<2, 4, 8>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f32_interleaved_mixed_radix_16(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        mixed_radix_c2c_f32::<4, 4, 16>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f16_interleaved_8(
        out: *mut f16,
        input: *mut f16,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        twiddle_imag_scale: f32,
    ) {
        dft_c2c_f16::<8>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f16_interleaved_16(
        out: *mut f16,
        input: *mut f16,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        twiddle_imag_scale: f32,
    ) {
        dft_c2c_f16::<16>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_bf16_interleaved_8(
        out: *mut bf16,
        input: *mut bf16,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        twiddle_imag_scale: f32,
    ) {
        dft_c2c_bf16::<8>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_bf16_interleaved_16(
        out: *mut bf16,
        input: *mut bf16,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        twiddle_imag_scale: f32,
    ) {
        dft_c2c_bf16::<16>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f32_interleaved_mixed_radix_12(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        mixed_radix_c2c_f32::<3, 4, 12>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f32_interleaved_mixed_radix_20(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        mixed_radix_c2c_f32::<4, 5, 20>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f32_interleaved_mixed_radix_24(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        mixed_radix_c2c_f32::<4, 6, 24>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f32_interleaved_mixed_radix_40(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        mixed_radix_c2c_f32::<5, 8, 40>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f32_interleaved_mixed_radix_48(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        mixed_radix_c2c_f32::<6, 8, 48>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f32_interleaved_mixed_radix_60(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        mixed_radix_c2c_f32::<6, 10, 60>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f32_interleaved_mixed_radix_80(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        mixed_radix_c2c_f32::<8, 10, 80>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f32_interleaved_mixed_radix_96(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        mixed_radix_c2c_f32::<8, 12, 96>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f32_interleaved_mixed_radix_120(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        mixed_radix_c2c_f32::<10, 12, 120>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f32_interleaved_mixed_radix_144(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        mixed_radix_c2c_f32::<12, 12, 144>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_c2c_f32_interleaved_32(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        dft_c2c_f32::<32>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_complex,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
            twiddle_imag_scale,
        );
    }

    #[cutile::entry()]
    pub unsafe fn ifft_c2r_f32_interleaved_8(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_c2r_f32::<8>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_real,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn ifft_c2r_f32_interleaved_16(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_c2r_f32::<16>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_real,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn ifft_c2r_f32_interleaved_12(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_c2r_f32::<12>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_real,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn ifft_c2r_f32_interleaved_20(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_c2r_f32::<20>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_real,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn ifft_c2r_f32_interleaved_24(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_c2r_f32::<24>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_real,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn ifft_c2r_f32_interleaved_40(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_c2r_f32::<40>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_real,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn ifft_c2r_f32_interleaved_48(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_c2r_f32::<48>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_real,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn ifft_c2r_f32_interleaved_60(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_c2r_f32::<60>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_real,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn ifft_c2r_f32_interleaved_80(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_c2r_f32::<80>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_real,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn ifft_c2r_f32_interleaved_96(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_c2r_f32::<96>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_real,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn ifft_c2r_f32_interleaved_120(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_c2r_f32::<120>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_real,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn ifft_c2r_f32_interleaved_144(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_c2r_f32::<144>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_real,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn ifft_c2r_f32_interleaved_32(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_c2r_f32::<32>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_real,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_r2c_f32_interleaved_8(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_r2c_f32::<8>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_frequency,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_r2c_f32_interleaved_16(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_r2c_f32::<16>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_frequency,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_r2c_f32_interleaved_12(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_r2c_f32::<12>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_frequency,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_r2c_f32_interleaved_20(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_r2c_f32::<20>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_frequency,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_r2c_f32_interleaved_24(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_r2c_f32::<24>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_frequency,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_r2c_f32_interleaved_40(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_r2c_f32::<40>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_frequency,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_r2c_f32_interleaved_48(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_r2c_f32::<48>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_frequency,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_r2c_f32_interleaved_60(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_r2c_f32::<60>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_frequency,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_r2c_f32_interleaved_80(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_r2c_f32::<80>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_frequency,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_r2c_f32_interleaved_96(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_r2c_f32::<96>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_frequency,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_r2c_f32_interleaved_120(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_r2c_f32::<120>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_frequency,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_r2c_f32_interleaved_144(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_r2c_f32::<144>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_frequency,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fft_r2c_f32_interleaved_32(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        dft_r2c_f32::<32>(
            out,
            input,
            twiddle_real,
            twiddle_imag,
            total_frequency,
            input_stride,
            output_stride,
            input_distance,
            output_distance,
        );
    }

    #[cutile::entry()]
    pub unsafe fn scale_c2c_f32_interleaved(data: *mut f32, len: i32, scale: f32) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let values = load_vector(data, offsets, valid, 0.0f32);
        store_vector(
            data,
            offsets,
            values * broadcast_scalar(scale, tile_shape),
            valid,
        );
    }

    fn mixed_radix_c2c_f32<const R: i32, const M: i32, const N: i32>(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();

        // One lane computes one output frequency for one batch item.
        // `k` is the frequency within the N-point transform.
        // `batch` selects the strided input/output record.
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(total_complex, tile_shape),
            predicate::LessThan,
        );

        let n_tile = broadcast_scalar(N, tile_shape);
        let batch = offsets / n_tile;
        let k = offsets - batch * n_tile;
        let input_batch_base = batch * broadcast_scalar(input_distance, tile_shape);
        let output_batch_base = batch * broadcast_scalar(output_distance, tile_shape);

        let mut real = constant(0.0f32, tile_shape);
        let mut imag = constant(0.0f32, tile_shape);

        // Factor the N-point DFT as M groups of R samples.
        // The inner loop contracts a contiguous radix-R group.
        // The outer loop applies the group twiddle before accumulating the final frequency bin.
        for major in 0i32..M {
            let mut inner_real = constant(0.0f32, tile_shape);
            let mut inner_imag = constant(0.0f32, tile_shape);

            for minor in 0i32..R {
                let sample = major * R + minor;
                let input_offsets =
                    input_batch_base + broadcast_scalar(sample * input_stride, tile_shape);
                let input_real = load_vector(input, input_offsets, valid, 0.0f32);
                let input_imag = load_vector(
                    input,
                    input_offsets + broadcast_scalar(1i32, tile_shape),
                    valid,
                    0.0f32,
                );

                // Twiddles are stored as one period of real/imag values.
                // The inner radix-local phase is `k * minor` reduced modulo N.
                let phase = k * broadcast_scalar(minor, tile_shape);
                let phase = phase - (phase / n_tile) * n_tile;
                let twiddle_real_values = load_vector(twiddle_real, phase, valid, 0.0f32);
                let twiddle_imag_values = load_vector(twiddle_imag, phase, valid, 0.0f32)
                    * broadcast_scalar(twiddle_imag_scale, tile_shape);

                inner_real = inner_real + input_real * twiddle_real_values
                    - input_imag * twiddle_imag_values;
                inner_imag = inner_imag
                    + input_real * twiddle_imag_values
                    + input_imag * twiddle_real_values;
            }

            let phase = k * broadcast_scalar(major * R, tile_shape);
            let phase = phase - (phase / n_tile) * n_tile;
            let twiddle_real_values = load_vector(twiddle_real, phase, valid, 0.0f32);
            let twiddle_imag_values = load_vector(twiddle_imag, phase, valid, 0.0f32)
                * broadcast_scalar(twiddle_imag_scale, tile_shape);

            real = real + inner_real * twiddle_real_values - inner_imag * twiddle_imag_values;
            imag = imag + inner_real * twiddle_imag_values + inner_imag * twiddle_real_values;
        }

        let output_offsets = output_batch_base + k * broadcast_scalar(output_stride, tile_shape);
        store_vector(out, output_offsets, real, valid);
        store_vector(
            out,
            output_offsets + broadcast_scalar(1i32, tile_shape),
            imag,
            valid,
        );
    }

    fn dft_c2c_f32<const N: i32>(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
        twiddle_imag_scale: f32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();

        // Direct complex-to-complex DFT.
        // Each lane owns one output frequency `k` in an interleaved real/imag batch record.
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(total_complex, tile_shape),
            predicate::LessThan,
        );

        let n_tile = broadcast_scalar(N, tile_shape);
        let batch = offsets / n_tile;
        let k = offsets - batch * n_tile;
        let input_batch_base = batch * broadcast_scalar(input_distance, tile_shape);
        let output_batch_base = batch * broadcast_scalar(output_distance, tile_shape);

        let mut real = constant(0.0f32, tile_shape);
        let mut imag = constant(0.0f32, tile_shape);

        // Accumulate sum_j x[j] * W_N^(k*j).
        // `twiddle_imag_scale` flips the imaginary sign for inverse transforms while sharing the same table.
        for j in 0i32..N {
            let input_offsets = input_batch_base + broadcast_scalar(j * input_stride, tile_shape);
            let input_real = load_vector(input, input_offsets, valid, 0.0f32);
            let input_imag = load_vector(
                input,
                input_offsets + broadcast_scalar(1i32, tile_shape),
                valid,
                0.0f32,
            );

            let phase = k * broadcast_scalar(j, tile_shape);
            let phase = phase - (phase / n_tile) * n_tile;
            let twiddle_real_values = load_vector(twiddle_real, phase, valid, 0.0f32);
            let twiddle_imag_values = load_vector(twiddle_imag, phase, valid, 0.0f32)
                * broadcast_scalar(twiddle_imag_scale, tile_shape);

            real = real + input_real * twiddle_real_values - input_imag * twiddle_imag_values;
            imag = imag + input_real * twiddle_imag_values + input_imag * twiddle_real_values;
        }

        let output_offsets = output_batch_base + k * broadcast_scalar(output_stride, tile_shape);
        store_vector(out, output_offsets, real, valid);
        store_vector(
            out,
            output_offsets + broadcast_scalar(1i32, tile_shape),
            imag,
            valid,
        );
    }

    fn dft_c2c_f16<const N: i32>(
        out: *mut f16,
        input: *mut f16,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        twiddle_imag_scale: f32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();

        // Same direct DFT as f32.
        // Inputs and outputs stay packed as f16 while accumulation and twiddle math remain f32.
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(total_complex, tile_shape),
            predicate::LessThan,
        );

        let n_tile = broadcast_scalar(N, tile_shape);
        let batch = offsets / n_tile;
        let k = offsets - batch * n_tile;
        let batch_base = batch * broadcast_scalar(N * 2, tile_shape);

        let mut real = constant(0.0f32, tile_shape);
        let mut imag = constant(0.0f32, tile_shape);

        for j in 0i32..N {
            let input_offsets = batch_base + broadcast_scalar(j * 2, tile_shape);
            let input_real = load_vector_f16_as_f32(input, input_offsets, valid);
            let input_imag = load_vector_f16_as_f32(
                input,
                input_offsets + broadcast_scalar(1i32, tile_shape),
                valid,
            );

            let phase = k * broadcast_scalar(j, tile_shape);
            let phase = phase - (phase / n_tile) * n_tile;
            let twiddle_real_values = load_vector(twiddle_real, phase, valid, 0.0f32);
            let twiddle_imag_values = load_vector(twiddle_imag, phase, valid, 0.0f32)
                * broadcast_scalar(twiddle_imag_scale, tile_shape);

            real = real + input_real * twiddle_real_values - input_imag * twiddle_imag_values;
            imag = imag + input_real * twiddle_imag_values + input_imag * twiddle_real_values;
        }

        let output_offsets = batch_base + k * broadcast_scalar(2i32, tile_shape);
        store_vector_f16_from_f32(out, output_offsets, real, valid);
        store_vector_f16_from_f32(
            out,
            output_offsets + broadcast_scalar(1i32, tile_shape),
            imag,
            valid,
        );
    }

    fn dft_c2c_bf16<const N: i32>(
        out: *mut bf16,
        input: *mut bf16,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_complex: i32,
        twiddle_imag_scale: f32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();

        // Same direct DFT as f16.
        // It uses bf16 loads/stores around an f32 accumulator to keep the reduction numerically stable.
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(total_complex, tile_shape),
            predicate::LessThan,
        );

        let n_tile = broadcast_scalar(N, tile_shape);
        let batch = offsets / n_tile;
        let k = offsets - batch * n_tile;
        let batch_base = batch * broadcast_scalar(N * 2, tile_shape);

        let mut real = constant(0.0f32, tile_shape);
        let mut imag = constant(0.0f32, tile_shape);

        for j in 0i32..N {
            let input_offsets = batch_base + broadcast_scalar(j * 2, tile_shape);
            let input_real = load_vector_bf16_as_f32(input, input_offsets, valid);
            let input_imag = load_vector_bf16_as_f32(
                input,
                input_offsets + broadcast_scalar(1i32, tile_shape),
                valid,
            );

            let phase = k * broadcast_scalar(j, tile_shape);
            let phase = phase - (phase / n_tile) * n_tile;
            let twiddle_real_values = load_vector(twiddle_real, phase, valid, 0.0f32);
            let twiddle_imag_values = load_vector(twiddle_imag, phase, valid, 0.0f32)
                * broadcast_scalar(twiddle_imag_scale, tile_shape);

            real = real + input_real * twiddle_real_values - input_imag * twiddle_imag_values;
            imag = imag + input_real * twiddle_imag_values + input_imag * twiddle_real_values;
        }

        let output_offsets = batch_base + k * broadcast_scalar(2i32, tile_shape);
        store_vector_bf16_from_f32(out, output_offsets, real, valid);
        store_vector_bf16_from_f32(
            out,
            output_offsets + broadcast_scalar(1i32, tile_shape),
            imag,
            valid,
        );
    }

    fn dft_c2r_f32<const N: i32>(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_real: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();

        // Inverse real FFT reconstruction.
        // Each lane writes one time-domain sample `j`.
        // The packed input contains DC, positive frequencies, and the Nyquist term for an N-point real signal.
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(total_real, tile_shape),
            predicate::LessThan,
        );

        let n_tile = broadcast_scalar(N, tile_shape);
        let batch = offsets / n_tile;
        let j = offsets - batch * n_tile;
        let input_batch_base = batch * broadcast_scalar(input_distance, tile_shape);
        let output_batch_base = batch * broadcast_scalar(output_distance, tile_shape);

        let dc_real = load_vector(input, input_batch_base, valid, 0.0f32);
        let nyquist_real = load_vector(
            input,
            input_batch_base + broadcast_scalar((N / 2) * input_stride, tile_shape),
            valid,
            0.0f32,
        );
        let nyquist_phase = j * broadcast_scalar(N / 2, tile_shape);
        let nyquist_phase = nyquist_phase - (nyquist_phase / n_tile) * n_tile;
        let nyquist_twiddle_real = load_vector(twiddle_real, nyquist_phase, valid, 0.0f32);
        let mut value = dc_real + nyquist_real * nyquist_twiddle_real;

        // Positive frequencies are mirrored by conjugate symmetry.
        // Each non-DC/non-Nyquist contribution appears twice in the real output.
        for k in 1i32..(N / 2) {
            let input_offsets = input_batch_base + broadcast_scalar(k * input_stride, tile_shape);
            let input_real = load_vector(input, input_offsets, valid, 0.0f32);
            let input_imag = load_vector(
                input,
                input_offsets + broadcast_scalar(1i32, tile_shape),
                valid,
                0.0f32,
            );

            let phase = j * broadcast_scalar(k, tile_shape);
            let phase = phase - (phase / n_tile) * n_tile;
            let twiddle_real_values = load_vector(twiddle_real, phase, valid, 0.0f32);
            let twiddle_imag_values = load_vector(twiddle_imag, phase, valid, 0.0f32);
            let contribution = input_real * twiddle_real_values + input_imag * twiddle_imag_values;
            value = value + contribution * broadcast_scalar(2.0f32, tile_shape);
        }

        let output_offsets = output_batch_base + j * broadcast_scalar(output_stride, tile_shape);
        store_vector(out, output_offsets, value, valid);
    }

    fn dft_r2c_f32<const N: i32>(
        out: *mut f32,
        input: *mut f32,
        twiddle_real: *mut f32,
        twiddle_imag: *mut f32,
        total_frequency: i32,
        input_stride: i32,
        output_stride: i32,
        input_distance: i32,
        output_distance: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();

        // Real-to-complex forward DFT.
        // Only N / 2 + 1 frequency bins are materialized because the remaining bins are the conjugate mirror.
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(total_frequency, tile_shape),
            predicate::LessThan,
        );

        let frequency_count = broadcast_scalar((N / 2) + 1, tile_shape);
        let n_tile = broadcast_scalar(N, tile_shape);
        let batch = offsets / frequency_count;
        let k = offsets - batch * frequency_count;
        let input_batch_base = batch * broadcast_scalar(input_distance, tile_shape);
        let output_batch_base = batch * broadcast_scalar(output_distance, tile_shape);

        let mut real = constant(0.0f32, tile_shape);
        let mut imag = constant(0.0f32, tile_shape);

        for j in 0i32..N {
            let input_offsets = input_batch_base + broadcast_scalar(j * input_stride, tile_shape);
            let input_values = load_vector(input, input_offsets, valid, 0.0f32);

            let phase = k * broadcast_scalar(j, tile_shape);
            let phase = phase - (phase / n_tile) * n_tile;
            let twiddle_real_values = load_vector(twiddle_real, phase, valid, 0.0f32);
            let twiddle_imag_values = load_vector(twiddle_imag, phase, valid, 0.0f32);

            real = real + input_values * twiddle_real_values;
            imag = imag + input_values * twiddle_imag_values;
        }

        let output_offsets = output_batch_base + k * broadcast_scalar(output_stride, tile_shape);
        store_vector(out, output_offsets, real, valid);
        store_vector(
            out,
            output_offsets + broadcast_scalar(1i32, tile_shape),
            imag,
            valid,
        );
    }
}

pub use kernels::*;
