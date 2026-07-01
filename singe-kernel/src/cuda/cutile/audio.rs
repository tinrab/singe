use std::sync::Arc;

use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

use crate::{
    cuda::cutile::{
        DeviceOpExt,
        kernel::audio as kernel_audio,
        utility::{VectorLaunch, checked_device_pointer, raw_vector_grid},
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LogMel {
    batch: i32,
    frames: i32,
    first_frame: i32,
    output_frame_stride: i32,
    grid: (u32, u32, u32),
}

impl LogMel {
    fn create(
        batch: usize,
        frames: usize,
        mel_bins: usize,
        frequency_bins: usize,
        first_frame: usize,
        output_frame_stride: usize,
    ) -> Result<Self> {
        if batch == 0 || frames == 0 || mel_bins == 0 || frequency_bins == 0 {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(checked_element_count(batch, frames)?, mel_bins)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            frames: checked_i32_value(frames)?,
            first_frame: checked_i32_value(first_frame)?,
            output_frame_stride: checked_i32_value(output_frame_stride)?,
            grid: VectorLaunch::create(output_len)?.grid,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LogMelDynamicMax {
    batch: i32,
    frames: i32,
    partial_per_batch: i32,
    partial_grid: (u32, u32, u32),
    reduce_grid: (u32, u32, u32),
}

impl LogMelDynamicMax {
    fn create(
        batch: usize,
        frames: usize,
        mel_bins: usize,
        frequency_bins: usize,
        partial_per_batch: usize,
    ) -> Result<Self> {
        if batch == 0
            || frames == 0
            || mel_bins == 0
            || frequency_bins == 0
            || partial_per_batch == 0
        {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(checked_element_count(batch, frames)?, mel_bins)?;
        checked_i32_value(output_len)?;
        let expected_partial_per_batch = checked_element_count(frames, mel_bins)?.div_ceil(128);
        if partial_per_batch != expected_partial_per_batch || partial_per_batch > 1024 {
            return Err(Error::InvalidLength);
        }
        let partial_blocks = checked_element_count(batch, partial_per_batch)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            frames: checked_i32_value(frames)?,
            partial_per_batch: checked_i32_value(partial_per_batch)?,
            partial_grid: (
                u32::try_from(partial_blocks).map_err(|_| Error::SizeOverflow)?,
                1,
                1,
            ),
            reduce_grid: (u32::try_from(batch).map_err(|_| Error::SizeOverflow)?, 1, 1),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StftR2c {
    input_len: i32,
    batch: i32,
    frames: i32,
    first_frame: i32,
    hop_length: i32,
    pad: i32,
    stage1_grid: (u32, u32, u32),
    stage2_grid: (u32, u32, u32),
}

impl StftR2c {
    fn create(
        input_len: usize,
        batch: usize,
        first_frame: usize,
        frames: usize,
        n_fft: usize,
        hop_length: usize,
        center: bool,
        pad_mode: i32,
        frequency_bin_count: usize,
    ) -> Result<Self> {
        if input_len == 0
            || batch == 0
            || frames == 0
            || n_fft == 0
            || hop_length == 0
            || frequency_bin_count == 0
        {
            return Err(Error::InvalidLength);
        }
        if !(0..=1).contains(&pad_mode) {
            return Err(Error::InvalidLength);
        }
        let complex_values =
            checked_element_count(checked_element_count(batch, frames)?, frequency_bin_count)?;
        let output_len = checked_element_count(complex_values, 2)?;
        checked_i32_value(output_len)?;
        let stage1_len = checked_element_count(checked_element_count(batch, frames)?, n_fft)?;
        Ok(Self {
            input_len: checked_i32_value(input_len)?,
            batch: checked_i32_value(batch)?,
            frames: checked_i32_value(frames)?,
            first_frame: checked_i32_value(first_frame)?,
            hop_length: checked_i32_value(hop_length)?,
            pad: if center {
                checked_i32_value(n_fft / 2)?
            } else {
                0
            },
            stage1_grid: raw_vector_grid(stage1_len)?,
            stage2_grid: raw_vector_grid(complex_values)?,
        })
    }
}

#[cfg(feature = "dtype-f32")]
pub fn r2c_power_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    spectrum: DevicePointer<f32>,
    batch: usize,
    frames: usize,
    frequency_bin_count: usize,
) -> Result<()> {
    if batch == 0 || frames == 0 || frequency_bin_count == 0 {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(spectrum)?;

    let output_len =
        checked_element_count(checked_element_count(batch, frames)?, frequency_bin_count)?;
    let launch = VectorLaunch::create(output_len)?;
    unsafe { kernel_audio::r2c_power_f32(out, spectrum, launch.len_i32) }
        .grid(launch.grid)
        .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn stft_r2c_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    workspace: DevicePointer<f32>,
    input: DevicePointer<f32>,
    window: DevicePointer<f32>,
    dft_real: DevicePointer<f32>,
    dft_imag: DevicePointer<f32>,
    input_len: usize,
    batch: usize,
    first_frame: usize,
    frames: usize,
    n_fft: usize,
    hop_length: usize,
    center: bool,
    pad_mode: i32,
    frequency_bin_count: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(workspace)?;
    checked_device_pointer(input)?;
    checked_device_pointer(window)?;
    checked_device_pointer(dft_real)?;
    checked_device_pointer(dft_imag)?;

    let params = StftR2c::create(
        input_len,
        batch,
        first_frame,
        frames,
        n_fft,
        hop_length,
        center,
        pad_mode,
        frequency_bin_count,
    )?;

    match (n_fft, frequency_bin_count) {
        (400, 201) => {
            unsafe {
                kernel_audio::stft_r2c_stage1_f32_400(
                    workspace,
                    input,
                    window,
                    dft_real,
                    dft_imag,
                    params.input_len,
                    params.batch,
                    params.frames,
                    params.first_frame,
                    params.hop_length,
                    params.pad,
                    pad_mode,
                )
            }
            .grid(params.stage1_grid)
            .enqueue_on(stream)?;

            unsafe {
                kernel_audio::stft_r2c_stage2_f32_400_201(
                    out,
                    workspace,
                    dft_real,
                    dft_imag,
                    params.batch,
                    params.frames,
                )
            }
            .grid(params.stage2_grid)
            .enqueue_on(stream)?;
            Ok(())
        }
        _ => Err(Error::UnsupportedConfiguration {
            op: "stft_r2c_f32".into(),
            reason: "requires n_fft = 400 and frequency bins = 201".into(),
        }),
    }
}

#[cfg(feature = "dtype-f32")]
pub fn log_mel_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    power: DevicePointer<f32>,
    mel_filters: DevicePointer<f32>,
    batch: usize,
    frames: usize,
    mel_bins: usize,
    frequency_bins: usize,
    floor: f32,
    reference_max: f32,
    dynamic_range: f32,
    offset: f32,
    scale: f32,
    layout: i32,
    first_frame: usize,
    output_frame_stride: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(power)?;
    checked_device_pointer(mel_filters)?;

    let params = LogMel::create(
        batch,
        frames,
        mel_bins,
        frequency_bins,
        first_frame,
        output_frame_stride,
    )?;

    match (frequency_bins, mel_bins) {
        (201, 128) => unsafe {
            kernel_audio::log_mel_f32_201_128(
                out,
                power,
                mel_filters,
                params.batch,
                params.frames,
                floor,
                reference_max,
                dynamic_range,
                offset,
                scale,
                layout,
                params.first_frame,
                params.output_frame_stride,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedConfiguration {
                op: "log_mel_f32".into(),
                reason: "requires frequency bins = 201 and mel bins = 128".into(),
            });
        }
    };
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn log_mel_dynamic_max_f32(
    stream: &Arc<Stream>,
    reference_max: DevicePointer<f32>,
    partial: DevicePointer<f32>,
    power: DevicePointer<f32>,
    mel_filters: DevicePointer<f32>,
    batch: usize,
    frames: usize,
    mel_bins: usize,
    frequency_bins: usize,
    floor: f32,
    partial_per_batch: usize,
) -> Result<()> {
    checked_device_pointer(reference_max)?;
    checked_device_pointer(partial)?;
    checked_device_pointer(power)?;
    checked_device_pointer(mel_filters)?;

    let params =
        LogMelDynamicMax::create(batch, frames, mel_bins, frequency_bins, partial_per_batch)?;

    match (frequency_bins, mel_bins) {
        (201, 128) => unsafe {
            kernel_audio::log_mel_partial_max_f32_201_128(
                partial,
                power,
                mel_filters,
                params.batch,
                params.frames,
                params.partial_per_batch,
                floor,
            )
        }
        .grid(params.partial_grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedConfiguration {
                op: "log_mel_dynamic_max_f32".into(),
                reason: "requires frequency bins = 201 and mel bins = 128".into(),
            });
        }
    };

    unsafe {
        kernel_audio::log_mel_reduce_partial_max_f32(
            reference_max,
            partial,
            params.partial_per_batch,
        )
    }
    .grid(params.reduce_grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn log_mel_dynamic_reference_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    power: DevicePointer<f32>,
    mel_filters: DevicePointer<f32>,
    reference_max: DevicePointer<f32>,
    batch: usize,
    frames: usize,
    mel_bins: usize,
    frequency_bins: usize,
    floor: f32,
    dynamic_range: f32,
    offset: f32,
    scale: f32,
    layout: i32,
    first_frame: usize,
    output_frame_stride: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(power)?;
    checked_device_pointer(mel_filters)?;
    checked_device_pointer(reference_max)?;

    let params = LogMel::create(
        batch,
        frames,
        mel_bins,
        frequency_bins,
        first_frame,
        output_frame_stride,
    )?;

    match (frequency_bins, mel_bins) {
        (201, 128) => unsafe {
            kernel_audio::log_mel_dynamic_reference_f32_201_128(
                out,
                power,
                mel_filters,
                reference_max,
                params.batch,
                params.frames,
                floor,
                dynamic_range,
                offset,
                scale,
                layout,
                params.first_frame,
                params.output_frame_stride,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedConfiguration {
                op: "log_mel_dynamic_reference_f32".into(),
                reason: "requires frequency bins = 201 and mel bins = 128".into(),
            });
        }
    };
    Ok(())
}
