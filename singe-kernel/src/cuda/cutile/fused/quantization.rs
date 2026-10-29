use std::sync::Arc;

use cutile::{
    core::{ElementType, f8e4m3fn},
    cuda_async::device_buffer::DevicePointer,
    cuda_core::{DType, Stream},
};

use crate::{
    cuda::{
        cutile::{
            DeviceOpExt, TileKernel, kernel::fused_quantization as kernels,
            utility::checked_device_pointer,
        },
        quantization::{Fp8GroupQuantConfig, Fp8ScaleLayout, Fp8ScaleRounding},
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

/// # Safety
/// Pointers must cover the configured ranges in the stream context.
/// Writable ranges must not overlap each other or any input; read-only inputs may alias.
/// Retain them through completion and replay. `dtype` must name T's cuTile storage type.
pub unsafe fn rms_norm_add_quantize_fp8<T: DType + ElementType + Send>(
    stream: &Arc<Stream>,
    out: DevicePointer<f8e4m3fn>,
    scales: DevicePointer<f32>,
    residual_out: DevicePointer<T>,
    input: DevicePointer<T>,
    residual: DevicePointer<T>,
    weight: DevicePointer<T>,
    config: Fp8GroupQuantConfig,
    norm_eps: f32,
    weight_offset: f32,
    dtype: &str,
) -> Result<()> {
    config.scale_len()?;
    if !norm_eps.is_finite() || norm_eps < 0.0 || !weight_offset.is_finite() {
        return Err(Error::InvalidLength);
    }
    if config.rows == 0 {
        return Ok(());
    }
    checked_device_pointer(&out)?;
    checked_device_pointer(&scales)?;
    for ptr in [residual_out, input, residual, weight] {
        checked_device_pointer(&ptr)?;
    }
    let bn = config
        .columns
        .checked_next_power_of_two()
        .ok_or(Error::SizeOverflow)?
        .min(8192);
    let (stride, column_major) = match config.layout {
        Fp8ScaleLayout::RowMajor => (0, 0),
        Fp8ScaleLayout::ColumnMajor { column_stride } => (checked_i32_value(column_stride)?, 1),
    };
    // SAFETY: validated scalar configuration and the caller's pointer contract cover this launch.
    unsafe {
        kernels::rms_norm_add_fp8(
            out,
            scales,
            residual_out,
            input,
            residual,
            weight,
            checked_i32_value(config.columns)?,
            stride,
            column_major,
            norm_eps,
            weight_offset,
            config.eps,
        )
    }
    .generics(vec![
        dtype.into(),
        bn.to_string(),
        config.group_size.to_string(),
        (bn / config.group_size).to_string(),
        i32::from(config.rounding == Fp8ScaleRounding::PowerOfTwo).to_string(),
    ])
    .grid((
        u32::try_from(config.rows).map_err(|_| Error::SizeOverflow)?,
        1,
        1,
    ))
    .enqueue_on(stream)?;
    Ok(())
}

/// # Safety
/// Pointers must cover the configured ranges in the stream context.
/// Writable ranges must not overlap each other or any input; read-only inputs may alias.
/// Retain them through completion and replay. `dtype` must name T's cuTile storage type.
pub unsafe fn silu_mul_quantize_fp8<T: DType + ElementType + Send>(
    stream: &Arc<Stream>,
    out: DevicePointer<f8e4m3fn>,
    scales: DevicePointer<f32>,
    gate: DevicePointer<T>,
    up: DevicePointer<T>,
    config: Fp8GroupQuantConfig,
    dtype: &str,
) -> Result<()> {
    config.scale_len()?;
    if config.rows == 0 {
        return Ok(());
    }
    checked_device_pointer(&out)?;
    checked_device_pointer(&scales)?;
    checked_device_pointer(&gate)?;
    checked_device_pointer(&up)?;
    let groups = checked_element_count(config.rows, config.columns / config.group_size)?;
    let (stride, column_major) = match config.layout {
        Fp8ScaleLayout::RowMajor => (0, 0),
        Fp8ScaleLayout::ColumnMajor { column_stride } => (checked_i32_value(column_stride)?, 1),
    };
    // SAFETY: validated scalar configuration and the caller's pointer contract cover this launch.
    unsafe {
        kernels::silu_mul_fp8(
            out,
            scales,
            gate,
            up,
            checked_i32_value(config.rows)?,
            checked_i32_value(config.columns)?,
            stride,
            column_major,
            config.eps,
        )
    }
    .generics(vec![
        dtype.into(),
        config.group_size.to_string(),
        i32::from(config.rounding == Fp8ScaleRounding::PowerOfTwo).to_string(),
    ])
    .grid((
        u32::try_from(groups.div_ceil(4)).map_err(|_| Error::SizeOverflow)?,
        1,
        1,
    ))
    .enqueue_on(stream)?;
    Ok(())
}
