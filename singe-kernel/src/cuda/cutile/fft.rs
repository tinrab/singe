use std::sync::Arc;

#[cfg(feature = "dtype-bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

use crate::{
    cuda::cutile::{
        DeviceOpExt,
        adapter::TensorAdapter,
        kernel::fft as kernel_fft,
        utility::{checked_device_pointer, raw_vector_grid},
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FftStrided {
    len: i32,
    input_stride: i32,
    output_stride: i32,
    input_distance: i32,
    output_distance: i32,
    grid: (u32, u32, u32),
}

impl FftStrided {
    fn create(
        n: usize,
        batch: usize,
        input_stride: usize,
        output_stride: usize,
        input_distance: usize,
        output_distance: usize,
    ) -> Result<Self> {
        if n == 0 || batch == 0 {
            return Err(Error::InvalidLength);
        }
        let total_complex = checked_element_count(batch, n)?;
        let lanes = checked_element_count(total_complex, 2)?;
        checked_i32_value(lanes)?;
        Ok(Self {
            len: checked_i32_value(total_complex)?,
            input_stride: checked_i32_value(input_stride)?,
            output_stride: checked_i32_value(output_stride)?,
            input_distance: checked_i32_value(input_distance)?,
            output_distance: checked_i32_value(output_distance)?,
            grid: raw_vector_grid(total_complex)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FftPackedC2c {
    len: i32,
    grid: (u32, u32, u32),
}

impl FftPackedC2c {
    fn create(n: usize, batch: usize) -> Result<Self> {
        if n == 0 || batch == 0 {
            return Err(Error::InvalidLength);
        }
        let total_complex = checked_element_count(batch, n)?;
        let lanes = checked_element_count(total_complex, 2)?;
        checked_i32_value(lanes)?;
        Ok(Self {
            len: checked_i32_value(total_complex)?,
            grid: raw_vector_grid(total_complex)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FftC2r {
    len: i32,
    input_stride: i32,
    output_stride: i32,
    input_distance: i32,
    output_distance: i32,
    grid: (u32, u32, u32),
}

impl FftC2r {
    fn create(
        n: usize,
        batch: usize,
        input_stride: usize,
        output_stride: usize,
        input_distance: usize,
        output_distance: usize,
    ) -> Result<Self> {
        if n == 0 || batch == 0 || !n.is_multiple_of(2) {
            return Err(Error::InvalidLength);
        }
        let total_real = checked_element_count(batch, n)?;
        let input_frequency_count = n / 2 + 1;
        let input_lanes =
            checked_element_count(checked_element_count(batch, input_frequency_count)?, 2)?;
        checked_i32_value(input_lanes)?;
        Ok(Self {
            len: checked_i32_value(total_real)?,
            input_stride: checked_i32_value(input_stride)?,
            output_stride: checked_i32_value(output_stride)?,
            input_distance: checked_i32_value(input_distance)?,
            output_distance: checked_i32_value(output_distance)?,
            grid: raw_vector_grid(total_real)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FftR2c {
    len: i32,
    input_stride: i32,
    output_stride: i32,
    input_distance: i32,
    output_distance: i32,
    grid: (u32, u32, u32),
}

impl FftR2c {
    fn create(
        n: usize,
        batch: usize,
        input_stride: usize,
        output_stride: usize,
        input_distance: usize,
        output_distance: usize,
    ) -> Result<Self> {
        if n == 0 || batch == 0 || !n.is_multiple_of(2) {
            return Err(Error::InvalidLength);
        }
        let frequency_count = n / 2 + 1;
        let total_frequency = checked_element_count(batch, frequency_count)?;
        let output_lanes = checked_element_count(total_frequency, 2)?;
        checked_i32_value(output_lanes)?;
        let input_values = checked_element_count(batch, n)?;
        checked_i32_value(input_values)?;
        Ok(Self {
            len: checked_i32_value(total_frequency)?,
            input_stride: checked_i32_value(input_stride)?,
            output_stride: checked_i32_value(output_stride)?,
            input_distance: checked_i32_value(input_distance)?,
            output_distance: checked_i32_value(output_distance)?,
            grid: raw_vector_grid(total_frequency)?,
        })
    }
}

#[cfg(feature = "dtype-f32")]
pub fn copy_c2c_f32_interleaved(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    n: usize,
    batch: usize,
) -> Result<()> {
    if n == 0 || batch == 0 {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;

    match n {
        8 | 12 | 16 | 20 | 24 | 32 | 40 | 48 | 60 | 80 | 96 | 120 | 144 => {
            let out = TensorAdapter::contiguous_3d(out, batch, n, 2)?.partition([1, n, 2])?;
            let input = TensorAdapter::contiguous_3d(input, batch, n, 2)?;
            #[cfg(feature = "dtype-f32")]
            kernel_fft::copy_c2c_f32_interleaved(out, input).enqueue_on(stream)?;
            Ok(())
        }
        _ => Err(Error::UnsupportedFftSize {
            transform: "copy c2c f32".into(),
            n,
        }),
    }
}

#[cfg(feature = "dtype-f32")]
pub fn fft_c2c_f32_interleaved(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    twiddle_real: DevicePointer<f32>,
    twiddle_imag: DevicePointer<f32>,
    n: usize,
    batch: usize,
    input_stride: usize,
    output_stride: usize,
    input_distance: usize,
    output_distance: usize,
    twiddle_imag_scale: f32,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(twiddle_real)?;
    checked_device_pointer(twiddle_imag)?;

    let params = FftStrided::create(
        n,
        batch,
        input_stride,
        output_stride,
        input_distance,
        output_distance,
    )?;
    let FftStrided {
        len: total_complex_i32,
        input_stride: input_stride_i32,
        output_stride: output_stride_i32,
        input_distance: input_distance_i32,
        output_distance: output_distance_i32,
        grid,
    } = params;
    match n {
        8 => unsafe {
            kernel_fft::fft_c2c_f32_interleaved_mixed_radix_8(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        16 => unsafe {
            kernel_fft::fft_c2c_f32_interleaved_mixed_radix_16(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        12 => unsafe {
            kernel_fft::fft_c2c_f32_interleaved_mixed_radix_12(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        20 => unsafe {
            kernel_fft::fft_c2c_f32_interleaved_mixed_radix_20(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        24 => unsafe {
            kernel_fft::fft_c2c_f32_interleaved_mixed_radix_24(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        40 => unsafe {
            kernel_fft::fft_c2c_f32_interleaved_mixed_radix_40(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        48 => unsafe {
            kernel_fft::fft_c2c_f32_interleaved_mixed_radix_48(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        60 => unsafe {
            kernel_fft::fft_c2c_f32_interleaved_mixed_radix_60(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        80 => unsafe {
            kernel_fft::fft_c2c_f32_interleaved_mixed_radix_80(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        96 => unsafe {
            kernel_fft::fft_c2c_f32_interleaved_mixed_radix_96(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        120 => unsafe {
            kernel_fft::fft_c2c_f32_interleaved_mixed_radix_120(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        144 => unsafe {
            kernel_fft::fft_c2c_f32_interleaved_mixed_radix_144(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        32 => unsafe {
            kernel_fft::fft_c2c_f32_interleaved_32(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedFftSize {
                transform: "c2c f32".into(),
                n,
            });
        }
    };
    Ok(())
}

#[cfg(feature = "dtype-f16")]
pub fn fft_c2c_f16_interleaved(
    stream: &Arc<Stream>,
    out: DevicePointer<f16>,
    input: DevicePointer<f16>,
    twiddle_real: DevicePointer<f32>,
    twiddle_imag: DevicePointer<f32>,
    n: usize,
    batch: usize,
    twiddle_imag_scale: f32,
) -> Result<()> {
    if !twiddle_imag_scale.is_finite() {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(twiddle_real)?;
    checked_device_pointer(twiddle_imag)?;

    let params = FftPackedC2c::create(n, batch)?;
    let total_complex_i32 = params.len;
    let grid = params.grid;
    match n {
        8 => unsafe {
            kernel_fft::fft_c2c_f16_interleaved_8(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        16 => unsafe {
            kernel_fft::fft_c2c_f16_interleaved_16(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedFftSize {
                transform: "c2c f16".into(),
                n,
            });
        }
    };
    Ok(())
}

#[cfg(feature = "dtype-bf16")]
pub fn fft_c2c_bf16_interleaved(
    stream: &Arc<Stream>,
    out: DevicePointer<bf16>,
    input: DevicePointer<bf16>,
    twiddle_real: DevicePointer<f32>,
    twiddle_imag: DevicePointer<f32>,
    n: usize,
    batch: usize,
    twiddle_imag_scale: f32,
) -> Result<()> {
    if !twiddle_imag_scale.is_finite() {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(twiddle_real)?;
    checked_device_pointer(twiddle_imag)?;

    let params = FftPackedC2c::create(n, batch)?;
    let total_complex_i32 = params.len;
    let grid = params.grid;
    match n {
        8 => unsafe {
            kernel_fft::fft_c2c_bf16_interleaved_8(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        16 => unsafe {
            kernel_fft::fft_c2c_bf16_interleaved_16(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_complex_i32,
                twiddle_imag_scale,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedFftSize {
                transform: "c2c bf16".into(),
                n,
            });
        }
    };
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn ifft_c2r_f32_interleaved(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    twiddle_real: DevicePointer<f32>,
    twiddle_imag: DevicePointer<f32>,
    n: usize,
    batch: usize,
    input_stride: usize,
    output_stride: usize,
    input_distance: usize,
    output_distance: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(twiddle_real)?;
    checked_device_pointer(twiddle_imag)?;

    let params = FftC2r::create(
        n,
        batch,
        input_stride,
        output_stride,
        input_distance,
        output_distance,
    )?;
    let FftC2r {
        len: total_real_i32,
        input_stride: input_stride_i32,
        output_stride: output_stride_i32,
        input_distance: input_distance_i32,
        output_distance: output_distance_i32,
        grid,
    } = params;
    match n {
        8 => unsafe {
            kernel_fft::ifft_c2r_f32_interleaved_8(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_real_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        16 => unsafe {
            kernel_fft::ifft_c2r_f32_interleaved_16(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_real_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        12 => unsafe {
            kernel_fft::ifft_c2r_f32_interleaved_12(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_real_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        20 => unsafe {
            kernel_fft::ifft_c2r_f32_interleaved_20(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_real_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        24 => unsafe {
            kernel_fft::ifft_c2r_f32_interleaved_24(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_real_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        40 => unsafe {
            kernel_fft::ifft_c2r_f32_interleaved_40(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_real_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        48 => unsafe {
            kernel_fft::ifft_c2r_f32_interleaved_48(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_real_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        60 => unsafe {
            kernel_fft::ifft_c2r_f32_interleaved_60(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_real_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        80 => unsafe {
            kernel_fft::ifft_c2r_f32_interleaved_80(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_real_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        96 => unsafe {
            kernel_fft::ifft_c2r_f32_interleaved_96(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_real_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        120 => unsafe {
            kernel_fft::ifft_c2r_f32_interleaved_120(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_real_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        144 => unsafe {
            kernel_fft::ifft_c2r_f32_interleaved_144(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_real_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        32 => unsafe {
            kernel_fft::ifft_c2r_f32_interleaved_32(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_real_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedFftSize {
                transform: "c2r f32".into(),
                n,
            });
        }
    };
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn fft_r2c_f32_interleaved(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    twiddle_real: DevicePointer<f32>,
    twiddle_imag: DevicePointer<f32>,
    n: usize,
    batch: usize,
    input_stride: usize,
    output_stride: usize,
    input_distance: usize,
    output_distance: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(twiddle_real)?;
    checked_device_pointer(twiddle_imag)?;

    let params = FftR2c::create(
        n,
        batch,
        input_stride,
        output_stride,
        input_distance,
        output_distance,
    )?;
    let FftR2c {
        len: total_frequency_i32,
        input_stride: input_stride_i32,
        output_stride: output_stride_i32,
        input_distance: input_distance_i32,
        output_distance: output_distance_i32,
        grid,
    } = params;
    match n {
        8 => unsafe {
            kernel_fft::fft_r2c_f32_interleaved_8(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_frequency_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        16 => unsafe {
            kernel_fft::fft_r2c_f32_interleaved_16(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_frequency_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        12 => unsafe {
            kernel_fft::fft_r2c_f32_interleaved_12(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_frequency_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        20 => unsafe {
            kernel_fft::fft_r2c_f32_interleaved_20(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_frequency_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        24 => unsafe {
            kernel_fft::fft_r2c_f32_interleaved_24(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_frequency_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        40 => unsafe {
            kernel_fft::fft_r2c_f32_interleaved_40(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_frequency_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        48 => unsafe {
            kernel_fft::fft_r2c_f32_interleaved_48(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_frequency_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        60 => unsafe {
            kernel_fft::fft_r2c_f32_interleaved_60(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_frequency_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        80 => unsafe {
            kernel_fft::fft_r2c_f32_interleaved_80(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_frequency_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        96 => unsafe {
            kernel_fft::fft_r2c_f32_interleaved_96(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_frequency_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        120 => unsafe {
            kernel_fft::fft_r2c_f32_interleaved_120(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_frequency_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        144 => unsafe {
            kernel_fft::fft_r2c_f32_interleaved_144(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_frequency_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        32 => unsafe {
            kernel_fft::fft_r2c_f32_interleaved_32(
                out,
                input,
                twiddle_real,
                twiddle_imag,
                total_frequency_i32,
                input_stride_i32,
                output_stride_i32,
                input_distance_i32,
                output_distance_i32,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedFftSize {
                transform: "r2c f32".into(),
                n,
            });
        }
    };
    Ok(())
}
