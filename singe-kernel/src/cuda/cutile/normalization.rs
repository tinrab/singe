use std::sync::Arc;

#[cfg(feature = "dtype-bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
use crate::cuda::cutile::kernel::f16::normalization as kernel_f16;
#[cfg(feature = "dtype-f32")]
use crate::cuda::cutile::kernel::f32::normalization as kernel_f32;
#[cfg(feature = "dtype-f64")]
use crate::cuda::cutile::kernel::f64::normalization as kernel_f64;
use crate::{
    cuda::cutile::{
        DeviceOpExt,
        adapter::TensorAdapter,
        utility::{VectorLaunch, checked_device_pointer},
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

const NORMALIZATION_MAX_TILE_WIDTH: usize = 2048;

fn normalization_tile_width(cols: usize) -> Result<usize> {
    match cols {
        1 => Ok(1),
        2..=8 => Ok(8),
        9..=32 => Ok(32),
        33..=64 => Ok(64),
        65..=128 => Ok(128),
        129..=256 => Ok(256),
        257..=512 => Ok(512),
        513..=1024 => Ok(1024),
        1025..=2048 => Ok(2048),
        _ => Err(Error::UnsupportedWidth {
            op: "normalization".into(),
            width: cols,
        }),
    }
}

macro_rules! dispatch_normalization {
    ($kernel:ident::$kernel_fn:ident, $out:expr, $input:expr, $weight:expr, $bias:expr, $has_weight:expr, $has_bias:expr, $eps:expr, $weight_offset:expr, $stream:expr, $bn:expr) => {{
        match $bn {
            1 => $kernel::$kernel_fn(
                $out,
                $input,
                $weight,
                $bias,
                $has_weight,
                $has_bias,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            8 => $kernel::$kernel_fn(
                $out,
                $input,
                $weight,
                $bias,
                $has_weight,
                $has_bias,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            32 => $kernel::$kernel_fn(
                $out,
                $input,
                $weight,
                $bias,
                $has_weight,
                $has_bias,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            64 => $kernel::$kernel_fn(
                $out,
                $input,
                $weight,
                $bias,
                $has_weight,
                $has_bias,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            128 => $kernel::$kernel_fn(
                $out,
                $input,
                $weight,
                $bias,
                $has_weight,
                $has_bias,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            256 => $kernel::$kernel_fn(
                $out,
                $input,
                $weight,
                $bias,
                $has_weight,
                $has_bias,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            512 => $kernel::$kernel_fn(
                $out,
                $input,
                $weight,
                $bias,
                $has_weight,
                $has_bias,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            1024 => $kernel::$kernel_fn(
                $out,
                $input,
                $weight,
                $bias,
                $has_weight,
                $has_bias,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            2048 => $kernel::$kernel_fn(
                $out,
                $input,
                $weight,
                $bias,
                $has_weight,
                $has_bias,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "normalization".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), crate::error::Error>(())
    }};
}

macro_rules! normalization_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            weight: DevicePointer<$ty>,
            bias: DevicePointer<$ty>,
            has_weight: bool,
            has_bias: bool,
            rows: usize,
            cols: usize,
            eps: $ty,
            weight_offset: $ty,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            if cols > NORMALIZATION_MAX_TILE_WIDTH {
                return Err(Error::UnsupportedWidth {
                    op: "normalization".into(),
                    width: cols,
                });
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            if has_weight {
                checked_device_pointer(weight)?;
            }
            if has_bias {
                checked_device_pointer(bias)?;
            }

            let bn = normalization_tile_width(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, cols)?.partition([1, bn])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            let weight = TensorAdapter::contiguous_1d(weight, cols)?;
            let bias = TensorAdapter::contiguous_1d(bias, cols)?;
            let has_weight = i32::from(has_weight);
            let has_bias = i32::from(has_bias);
            dispatch_normalization!(
                $kernel::$kernel_fn,
                out,
                input,
                weight,
                bias,
                has_weight,
                has_bias,
                eps,
                weight_offset,
                stream,
                bn
            )?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! dispatch_sparsemax {
    ($kernel:ident::$kernel_fn:ident, $out:expr, $input:expr, $stream:expr, $bn:expr) => {{
        match $bn {
            1 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            8 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            32 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            64 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            128 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            256 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            512 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            1024 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            2048 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "sparsemax".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), crate::error::Error>(())
    }};
}

macro_rules! sparsemax_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident, $wide_kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;

            if cols > NORMALIZATION_MAX_TILE_WIDTH {
                checked_i32_value(rows)?;
                checked_i32_value(cols)?;
                unsafe {
                    $kernel::$wide_kernel_fn(
                        out,
                        input,
                        checked_i32_value(rows)?,
                        checked_i32_value(cols)?,
                    )
                }
                .grid((u32::try_from(rows).map_err(|_| Error::SizeOverflow)?, 1, 1))
                .enqueue_on(stream)?;
                return Ok(());
            }

            let bn = normalization_tile_width(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, cols)?.partition([1, bn])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            dispatch_sparsemax!($kernel::$kernel_fn, out, input, stream, bn)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
sparsemax_fn!(
    sparsemax_f32,
    f32,
    kernel_f32,
    sparsemax_f32,
    sparsemax_f32_wide
);

#[cfg(feature = "dtype-f16")]
sparsemax_fn!(
    sparsemax_f16,
    f16,
    kernel_f16,
    sparsemax_f16,
    sparsemax_f16_wide
);

#[cfg(feature = "dtype-bf16")]
sparsemax_fn!(
    sparsemax_bf16,
    bf16,
    kernel_f16,
    sparsemax_bf16,
    sparsemax_bf16_wide
);

#[cfg(feature = "dtype-f32")]

pub fn group_norm_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    weight: DevicePointer<f32>,
    bias: DevicePointer<f32>,
    batch: usize,
    channels: usize,
    groups: usize,
    spatial_len: usize,
    eps: f32,
) -> Result<()> {
    if batch == 0 || channels == 0 || groups == 0 || spatial_len == 0 {
        return Err(Error::InvalidLength);
    }
    if channels % groups != 0 {
        return Err(Error::LengthMismatch);
    }
    let len = checked_element_count(checked_element_count(batch, channels)?, spatial_len)?;
    checked_i32_value(len)?;
    checked_i32_value(channels)?;
    checked_i32_value(groups)?;
    checked_i32_value(spatial_len)?;
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    checked_device_pointer(bias)?;
    let batch_blocks = u32::try_from(batch).map_err(|_| Error::SizeOverflow)?;
    let group_blocks = u32::try_from(groups).map_err(|_| Error::SizeOverflow)?;
    unsafe {
        kernel_f32::group_norm_f32(
            out,
            input,
            weight,
            bias,
            checked_i32_value(channels)?,
            checked_i32_value(groups)?,
            checked_i32_value(spatial_len)?,
            eps,
        )
    }
    .grid((batch_blocks, group_blocks, 1))
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f16")]

pub fn group_norm_f16(
    stream: &Arc<Stream>,
    out: DevicePointer<f16>,
    input: DevicePointer<f16>,
    weight: DevicePointer<f16>,
    bias: DevicePointer<f16>,
    batch: usize,
    channels: usize,
    groups: usize,
    spatial_len: usize,
    eps: f32,
) -> Result<()> {
    if batch == 0 || channels == 0 || groups == 0 || spatial_len == 0 {
        return Err(Error::InvalidLength);
    }
    if channels % groups != 0 {
        return Err(Error::LengthMismatch);
    }
    let len = checked_element_count(checked_element_count(batch, channels)?, spatial_len)?;
    checked_i32_value(len)?;
    checked_i32_value(channels)?;
    checked_i32_value(groups)?;
    checked_i32_value(spatial_len)?;
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    checked_device_pointer(bias)?;
    let batch_blocks = u32::try_from(batch).map_err(|_| Error::SizeOverflow)?;
    let group_blocks = u32::try_from(groups).map_err(|_| Error::SizeOverflow)?;
    unsafe {
        kernel_f16::group_norm_f16(
            out,
            input,
            weight,
            bias,
            checked_i32_value(channels)?,
            checked_i32_value(groups)?,
            checked_i32_value(spatial_len)?,
            eps,
        )
    }
    .grid((batch_blocks, group_blocks, 1))
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-bf16")]

pub fn group_norm_bf16(
    stream: &Arc<Stream>,
    out: DevicePointer<bf16>,
    input: DevicePointer<bf16>,
    weight: DevicePointer<bf16>,
    bias: DevicePointer<bf16>,
    batch: usize,
    channels: usize,
    groups: usize,
    spatial_len: usize,
    eps: f32,
) -> Result<()> {
    if batch == 0 || channels == 0 || groups == 0 || spatial_len == 0 {
        return Err(Error::InvalidLength);
    }
    if channels % groups != 0 {
        return Err(Error::LengthMismatch);
    }
    let len = checked_element_count(checked_element_count(batch, channels)?, spatial_len)?;
    checked_i32_value(len)?;
    checked_i32_value(channels)?;
    checked_i32_value(groups)?;
    checked_i32_value(spatial_len)?;
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    checked_device_pointer(bias)?;
    let batch_blocks = u32::try_from(batch).map_err(|_| Error::SizeOverflow)?;
    let group_blocks = u32::try_from(groups).map_err(|_| Error::SizeOverflow)?;
    unsafe {
        kernel_f16::group_norm_bf16(
            out,
            input,
            weight,
            bias,
            checked_i32_value(channels)?,
            checked_i32_value(groups)?,
            checked_i32_value(spatial_len)?,
            eps,
        )
    }
    .grid((batch_blocks, group_blocks, 1))
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
normalization_fn!(layer_norm_f32, f32, kernel_f32, layer_norm_f32);
#[cfg(feature = "dtype-f32")]
normalization_fn!(rms_norm_f32, f32, kernel_f32, rms_norm_f32);

#[cfg(feature = "dtype-f16")]
normalization_fn!(layer_norm_f16, f16, kernel_f16, layer_norm_f16);
#[cfg(feature = "dtype-f16")]
normalization_fn!(rms_norm_f16, f16, kernel_f16, rms_norm_f16);

#[cfg(feature = "dtype-bf16")]
normalization_fn!(layer_norm_bf16, bf16, kernel_f16, layer_norm_bf16);
#[cfg(feature = "dtype-bf16")]
normalization_fn!(rms_norm_bf16, bf16, kernel_f16, rms_norm_bf16);

#[cfg(feature = "dtype-f64")]
normalization_fn!(layer_norm_f64, f64, kernel_f64, layer_norm_f64);
#[cfg(feature = "dtype-f64")]
normalization_fn!(rms_norm_f64, f64, kernel_f64, rms_norm_f64);

macro_rules! batch_norm_inference_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            weight: DevicePointer<$ty>,
            bias: DevicePointer<$ty>,
            running_mean: DevicePointer<$ty>,
            running_var: DevicePointer<$ty>,
            feature_count: usize,
            spatial_len: usize,
            has_weight: bool,
            has_bias: bool,
            has_running_mean: bool,
            has_running_var: bool,
            eps: $ty,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            if feature_count == 0 || spatial_len == 0 {
                return Err(Error::InvalidLength);
            }
            let feature_period = checked_element_count(feature_count, spatial_len)?;
            if len % feature_period != 0 {
                return Err(Error::LengthMismatch);
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            if has_weight {
                checked_device_pointer(weight)?;
            }
            if has_bias {
                checked_device_pointer(bias)?;
            }
            if has_running_mean {
                checked_device_pointer(running_mean)?;
            }
            if has_running_var {
                checked_device_pointer(running_var)?;
            }
            let feature_count = checked_i32_value(feature_count)?;
            let spatial_len = checked_i32_value(spatial_len)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    feature_count,
                    spatial_len,
                    i32::from(has_weight),
                    i32::from(has_bias),
                    i32::from(has_running_mean),
                    i32::from(has_running_var),
                    eps,
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
batch_norm_inference_fn!(
    batch_norm_inference_f32,
    f32,
    kernel_f32,
    batch_norm_inference_f32
);
#[cfg(feature = "dtype-f16")]
batch_norm_inference_fn!(
    batch_norm_inference_f16,
    f16,
    kernel_f16,
    batch_norm_inference_f16
);
#[cfg(feature = "dtype-f64")]
batch_norm_inference_fn!(
    batch_norm_inference_f64,
    f64,
    kernel_f64,
    batch_norm_inference_f64
);

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cutile::prelude::*;

    use super::*;
    use crate::{
        cpu::normalization::{group_norm, sparsemax},
        error::Result,
    };

    #[cfg(feature = "dtype-f32")]
    #[test]
    fn layer_norm_affine_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows = 2usize;
        let cols = 5usize;
        let eps = 1e-5f32;
        let input_host = vec![0.25f32, -0.5, 1.0, 2.0, -1.5, 0.75, -1.25, 0.5, -0.25, 1.5];
        let weight_host = vec![1.0f32, 0.5, -1.5, 2.0, 0.25];
        let bias_host = vec![0.1f32, -0.2, 0.3, -0.4, 0.5];
        let mut expected = vec![0.0f32; rows * cols];
        for row in 0..rows {
            let values = &input_host[row * cols..(row + 1) * cols];
            let mean = values.iter().sum::<f32>() / cols as f32;
            let variance = values
                .iter()
                .map(|value| {
                    let centered = value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / cols as f32;
            let rstd = 1.0 / (variance + eps).sqrt();
            for col in 0..cols {
                expected[row * cols + col] =
                    (values[col] - mean) * rstd * weight_host[col] + bias_host[col];
            }
        }

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let bias = api::copy_host_vec_to_device(&Arc::new(bias_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * cols]).sync_on(&stream)?;

        layer_norm_f32(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            weight.device_pointer(),
            bias.device_pointer(),
            true,
            true,
            rows,
            cols,
            eps,
            0.0,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f32")]
    #[test]
    fn group_norm_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let batch = 2usize;
        let channels = 4usize;
        let groups = 2usize;
        let spatial_len = 3usize;
        let eps = 1e-5f32;
        let input_host = vec![
            0.25f32, -0.5, 1.0, -1.5, 0.75, 0.5, -0.75, 1.25, -0.25, 1.5, -1.0, 0.0, 0.5, -1.25,
            0.25, 1.75, -0.5, 0.75, -1.5, 1.0, 0.0, -0.25, 1.25, -0.75,
        ];
        let weight_host = vec![1.0f32, -0.5, 0.75, 1.5];
        let bias_host = vec![0.1f32, -0.2, 0.3, -0.4];
        let expected_out = group_norm(
            &input_host,
            &weight_host,
            &bias_host,
            batch,
            channels,
            groups,
            spatial_len,
            eps,
        );

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let bias = api::copy_host_vec_to_device(&Arc::new(bias_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * channels * spatial_len]).sync_on(&stream)?;

        group_norm_f32(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            weight.device_pointer(),
            bias.device_pointer(),
            batch,
            channels,
            groups,
            spatial_len,
            eps,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected_out, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn group_norm_f16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let batch = 2usize;
        let channels = 4usize;
        let groups = 2usize;
        let spatial_len = 3usize;
        let eps = 1e-5f32;
        let input_f32 = vec![
            0.25f32, -0.5, 1.0, -1.5, 0.75, 0.5, -0.75, 1.25, -0.25, 1.5, -1.0, 0.0, 0.5, -1.25,
            0.25, 1.75, -0.5, 0.75, -1.5, 1.0, 0.0, -0.25, 1.25, -0.75,
        ];
        let weight_f32 = vec![1.0f32, -0.5, 0.75, 1.5];
        let bias_f32 = vec![0.1f32, -0.2, 0.3, -0.4];
        let input_host = input_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let weight_host = weight_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let bias_host = bias_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let input_reference = input_host
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let weight_reference = weight_host
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let bias_reference = bias_host
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let expected_out = group_norm(
            &input_reference,
            &weight_reference,
            &bias_reference,
            batch,
            channels,
            groups,
            spatial_len,
            eps,
        );

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let bias = api::copy_host_vec_to_device(&Arc::new(bias_host)).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[batch * channels * spatial_len]).sync_on(&stream)?;

        group_norm_f16(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            weight.device_pointer(),
            bias.device_pointer(),
            batch,
            channels,
            groups,
            spatial_len,
            eps,
        )?;

        let out_host = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let expected_out = expected_out
            .iter()
            .copied()
            .map(f16::from_f32)
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        singe_core::assert_close!(&out_host, &expected_out, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn group_norm_bf16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let batch = 2usize;
        let channels = 4usize;
        let groups = 2usize;
        let spatial_len = 3usize;
        let eps = 1e-5f32;
        let input_f32 = vec![
            0.25f32, -0.5, 1.0, -1.5, 0.75, 0.5, -0.75, 1.25, -0.25, 1.5, -1.0, 0.0, 0.5, -1.25,
            0.25, 1.75, -0.5, 0.75, -1.5, 1.0, 0.0, -0.25, 1.25, -0.75,
        ];
        let weight_f32 = vec![1.0f32, -0.5, 0.75, 1.5];
        let bias_f32 = vec![0.1f32, -0.2, 0.3, -0.4];
        let input_host = input_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let weight_host = weight_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let bias_host = bias_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let input_reference = input_host
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let weight_reference = weight_host
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let bias_reference = bias_host
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let expected_out = group_norm(
            &input_reference,
            &weight_reference,
            &bias_reference,
            batch,
            channels,
            groups,
            spatial_len,
            eps,
        );

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let bias = api::copy_host_vec_to_device(&Arc::new(bias_host)).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[batch * channels * spatial_len]).sync_on(&stream)?;

        group_norm_bf16(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            weight.device_pointer(),
            bias.device_pointer(),
            batch,
            channels,
            groups,
            spatial_len,
            eps,
        )?;

        let out_host = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let expected_out = expected_out
            .iter()
            .copied()
            .map(bf16::from_f32)
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        singe_core::assert_close!(&out_host, &expected_out, 4e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-f32")]
    #[test]
    fn sparsemax_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows = 3usize;
        let cols = 5usize;
        let input_host = vec![
            1.0f32, 0.0, -1.0, 2.0, 0.5, -0.5, -0.25, -0.75, -1.0, -1.25, 3.0, 1.0, 0.0, 2.0, -4.0,
        ];
        let expected = sparsemax(&input_host, rows, cols);

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * cols]).sync_on(&stream)?;

        sparsemax_f32(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            rows,
            cols,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f32")]
    #[test]
    fn sparsemax_f32_wide() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows = 2usize;
        let cols = 1031usize;
        let input_host = (0..rows * cols)
            .map(|index| {
                let row = index / cols;
                let col = index % cols;
                ((col as f32 % 37.0) * 0.03125) - 0.5 + row as f32 * 0.125
            })
            .collect::<Vec<_>>();
        let expected = sparsemax(&input_host, rows, cols);

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * cols]).sync_on(&stream)?;

        sparsemax_f32(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            rows,
            cols,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn sparsemax_f16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows = 3usize;
        let cols = 5usize;
        let input_f32 = vec![
            1.0f32, 0.0, -1.0, 2.0, 0.5, -0.5, -0.25, -0.75, -1.0, -1.25, 3.0, 1.0, 0.0, 2.0, -4.0,
        ];
        let input_host = input_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let input_reference = input_host
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let expected = sparsemax(&input_reference, rows, cols)
            .iter()
            .copied()
            .map(f16::from_f32)
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[rows * cols]).sync_on(&stream)?;

        sparsemax_f16(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            rows,
            cols,
        )?;

        let out_host = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        singe_core::assert_close!(&out_host, &expected, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn sparsemax_f16_wide() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows = 2usize;
        let cols = 1031usize;
        let input_host = (0..rows * cols)
            .map(|index| {
                let row = index / cols;
                let col = index % cols;
                f16::from_f32(((col as f32 % 37.0) * 0.03125) - 0.5 + row as f32 * 0.125)
            })
            .collect::<Vec<_>>();
        let input_reference = input_host
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let expected = sparsemax(&input_reference, rows, cols)
            .iter()
            .copied()
            .map(f16::from_f32)
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[rows * cols]).sync_on(&stream)?;

        sparsemax_f16(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            rows,
            cols,
        )?;

        let out_host = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        singe_core::assert_close!(&out_host, &expected, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn sparsemax_bf16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows = 3usize;
        let cols = 5usize;
        let input_f32 = vec![
            1.0f32, 0.0, -1.0, 2.0, 0.5, -0.5, -0.25, -0.75, -1.0, -1.25, 3.0, 1.0, 0.0, 2.0, -4.0,
        ];
        let input_host = input_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let input_reference = input_host
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let expected = sparsemax(&input_reference, rows, cols)
            .iter()
            .copied()
            .map(bf16::from_f32)
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[rows * cols]).sync_on(&stream)?;

        sparsemax_bf16(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            rows,
            cols,
        )?;

        let out_host = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        singe_core::assert_close!(&out_host, &expected, 4e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn sparsemax_bf16_wide() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows = 2usize;
        let cols = 1031usize;
        let input_host = (0..rows * cols)
            .map(|index| {
                let row = index / cols;
                let col = index % cols;
                bf16::from_f32(((col as f32 % 37.0) * 0.03125) - 0.5 + row as f32 * 0.125)
            })
            .collect::<Vec<_>>();
        let input_reference = input_host
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let expected = sparsemax(&input_reference, rows, cols)
            .iter()
            .copied()
            .map(bf16::from_f32)
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[rows * cols]).sync_on(&stream)?;

        sparsemax_bf16(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            rows,
            cols,
        )?;

        let out_host = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        singe_core::assert_close!(&out_host, &expected, 4e-3);
        Ok(())
    }
}
