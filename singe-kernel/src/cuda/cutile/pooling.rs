use std::sync::Arc;

#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

#[cfg(feature = "dtype-f16")]
use crate::cuda::cutile::kernel::f16::pooling as kernel_f16;
#[cfg(feature = "dtype-f32")]
use crate::cuda::cutile::kernel::f32::pooling as kernel_f32;
#[cfg(feature = "dtype-f64")]
use crate::cuda::cutile::kernel::f64::pooling as kernel_f64;
use crate::{
    cuda::{
        cutile::{
            DeviceOpExt,
            utility::{VectorLaunch, checked_device_pointer},
        },
        pooling::Pool2dConfig,
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

fn checked_pool2d_config(config: Pool2dConfig) -> Result<[i32; 16]> {
    Ok([
        checked_i32_value(config.batch)?,
        checked_i32_value(config.channels)?,
        checked_i32_value(config.input_height)?,
        checked_i32_value(config.input_width)?,
        checked_i32_value(config.output_height)?,
        checked_i32_value(config.output_width)?,
        checked_i32_value(config.kernel_height)?,
        checked_i32_value(config.kernel_width)?,
        checked_i32_value(config.stride_height)?,
        checked_i32_value(config.stride_width)?,
        checked_i32_value(config.pad_height_start)?,
        checked_i32_value(config.pad_height_end)?,
        checked_i32_value(config.pad_width_start)?,
        checked_i32_value(config.pad_width_end)?,
        checked_i32_value(config.dilation_height)?,
        checked_i32_value(config.dilation_width)?,
    ])
}

fn pool2d_output_len(config: Pool2dConfig) -> Result<usize> {
    validate_pool2d_config(config)?;
    let hw = checked_element_count(config.output_height, config.output_width)?;
    let chw = checked_element_count(config.channels, hw)?;
    checked_element_count(config.batch, chw)
}

fn validate_pool2d_config(config: Pool2dConfig) -> Result<()> {
    if config.batch == 0
        || config.channels == 0
        || config.input_height == 0
        || config.input_width == 0
        || config.output_height == 0
        || config.output_width == 0
        || config.kernel_height == 0
        || config.kernel_width == 0
        || config.stride_height == 0
        || config.stride_width == 0
        || config.dilation_height == 0
        || config.dilation_width == 0
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

macro_rules! max_pool2d_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            config: Pool2dConfig,
        ) -> Result<()> {
            let len = pool2d_output_len(config)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let c = checked_pool2d_config(config)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    c[0],
                    c[1],
                    c[2],
                    c[3],
                    c[4],
                    c[5],
                    c[6],
                    c[7],
                    c[8],
                    c[9],
                    c[10],
                    c[12],
                    c[14],
                    c[15],
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! avg_pool2d_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            config: Pool2dConfig,
            count_include_pad: bool,
        ) -> Result<()> {
            let len = pool2d_output_len(config)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let c = checked_pool2d_config(config)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    c[0],
                    c[1],
                    c[2],
                    c[3],
                    c[4],
                    c[5],
                    c[6],
                    c[7],
                    c[8],
                    c[9],
                    c[10],
                    c[11],
                    c[12],
                    c[13],
                    c[14],
                    c[15],
                    i32::from(count_include_pad),
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
max_pool2d_fn!(max_pool2d_f32, f32, kernel_f32, max_pool2d_f32);
#[cfg(feature = "dtype-f32")]
avg_pool2d_fn!(avg_pool2d_f32, f32, kernel_f32, avg_pool2d_f32);
#[cfg(feature = "dtype-f16")]
max_pool2d_fn!(max_pool2d_f16, f16, kernel_f16, max_pool2d_f16);
#[cfg(feature = "dtype-f16")]
avg_pool2d_fn!(avg_pool2d_f16, f16, kernel_f16, avg_pool2d_f16);
#[cfg(feature = "dtype-f64")]
max_pool2d_fn!(max_pool2d_f64, f64, kernel_f64, max_pool2d_f64);
#[cfg(feature = "dtype-f64")]
avg_pool2d_fn!(avg_pool2d_f64, f64, kernel_f64, avg_pool2d_f64);
