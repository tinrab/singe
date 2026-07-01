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
        kernel::fused as kernel_fused,
        positional,
        utility::{checked_device_pointer, raw_vector_grid},
    },
    cuda::fused::RopeQkCacheUpdateConfig,
    cuda::positional::QkRotaryEmbeddingConfig,
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value, checked_rank4_len},
};

fn fused_tile_size(width: usize) -> usize {
    match width {
        0..=128 => 128,
        129..=256 => 256,
        257..=512 => 512,
        _ => 1024,
    }
}

fn rms_norm_tile_size(cols: usize) -> Result<usize> {
    match cols {
        1 => Ok(1),
        2..=8 => Ok(8),
        9..=32 => Ok(32),
        33..=64 => Ok(64),
        65..=128 => Ok(128),
        129..=256 => Ok(256),
        257..=512 => Ok(512),
        513..=1024 => Ok(1024),
        _ => Err(Error::UnsupportedWidth {
            op: "rms norm fused".into(),
            width: cols,
        }),
    }
}

const RMS_NORM_WIDE_TILE_WIDTH: usize = 1024;
const RMS_NORM_WIDE_MAX_CHUNKS: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RmsNormWidePlan {
    pub chunks: usize,
    pub partial_len: usize,
    pub row_sum_len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RmsNormWide {
    plan: RmsNormWidePlan,
    cols: i32,
    chunks: i32,
    len: i32,
    chunk_grid: (u32, u32, u32),
    row_grid: (u32, u32, u32),
    len_grid: (u32, u32, u32),
}

impl RmsNormWide {
    fn create(rows: usize, cols: usize) -> Result<Option<Self>> {
        if cols == 0 {
            return Err(Error::InvalidLength);
        }
        let len = checked_element_count(rows, cols)?;
        if len == 0 {
            return Ok(None);
        }
        let chunks = cols.div_ceil(RMS_NORM_WIDE_TILE_WIDTH);
        if chunks > RMS_NORM_WIDE_MAX_CHUNKS {
            return Err(Error::UnsupportedWidth {
                op: "rms norm add wide".into(),
                width: cols,
            });
        }
        let rows_u32 = u32::try_from(rows).map_err(|_| Error::SizeOverflow)?;
        let chunks_u32 = u32::try_from(chunks).map_err(|_| Error::SizeOverflow)?;
        let partial_len = checked_element_count(rows, chunks)?;
        Ok(Some(Self {
            plan: RmsNormWidePlan {
                chunks,
                partial_len,
                row_sum_len: rows,
            },
            cols: checked_i32_value(cols)?,
            chunks: checked_i32_value(chunks)?,
            len: checked_i32_value(len)?,
            chunk_grid: (chunks_u32, rows_u32, 1),
            row_grid: (rows_u32, 1, 1),
            len_grid: raw_vector_grid(len)?,
        }))
    }
}

pub fn rms_norm_wide_plan(rows: usize, cols: usize) -> Result<Option<RmsNormWidePlan>> {
    Ok(RmsNormWide::create(rows, cols)?.map(|wide| wide.plan))
}

macro_rules! dispatch_1d {
    ($kernel_fn:ident, $out:expr, $a:expr, $b:expr, $stream:expr, $tile:expr) => {{
        match $tile {
            128 => kernel_fused::$kernel_fn($out, $a, $b).enqueue_on($stream)?,
            256 => kernel_fused::$kernel_fn($out, $a, $b).enqueue_on($stream)?,
            512 => kernel_fused::$kernel_fn($out, $a, $b).enqueue_on($stream)?,
            1024 => kernel_fused::$kernel_fn($out, $a, $b).enqueue_on($stream)?,
            _ => unreachable!("unsupported fused tile size"),
        };
        Ok::<(), crate::error::Error>(())
    }};
}

macro_rules! dispatch_1d_ternary {
    ($kernel_fn:ident, $out:expr, $a:expr, $b:expr, $c:expr, $stream:expr, $tile:expr) => {{
        match $tile {
            128 => kernel_fused::$kernel_fn($out, $a, $b, $c).enqueue_on($stream)?,
            256 => kernel_fused::$kernel_fn($out, $a, $b, $c).enqueue_on($stream)?,
            512 => kernel_fused::$kernel_fn($out, $a, $b, $c).enqueue_on($stream)?,
            1024 => kernel_fused::$kernel_fn($out, $a, $b, $c).enqueue_on($stream)?,
            _ => unreachable!("unsupported fused tile size"),
        };
        Ok::<(), crate::error::Error>(())
    }};
}

macro_rules! dispatch_conditional_ternary {
    ($kernel_fn:ident, $out:expr, $condition:expr, $a:expr, $b:expr, $c:expr, $stream:expr, $tile:expr) => {{
        match $tile {
            128 => kernel_fused::$kernel_fn($out, $condition, $a, $b, $c).enqueue_on($stream)?,
            256 => kernel_fused::$kernel_fn($out, $condition, $a, $b, $c).enqueue_on($stream)?,
            512 => kernel_fused::$kernel_fn($out, $condition, $a, $b, $c).enqueue_on($stream)?,
            1024 => kernel_fused::$kernel_fn($out, $condition, $a, $b, $c).enqueue_on($stream)?,
            _ => unreachable!("unsupported fused tile size"),
        };
        Ok::<(), crate::error::Error>(())
    }};
}

macro_rules! dispatch_2d_bias {
    ($kernel_fn:ident, $out:expr, $input:expr, $bias:expr, $stream:expr, $tile:expr) => {{
        match $tile {
            128 => kernel_fused::$kernel_fn($out, $input, $bias).enqueue_on($stream)?,
            256 => kernel_fused::$kernel_fn($out, $input, $bias).enqueue_on($stream)?,
            512 => kernel_fused::$kernel_fn($out, $input, $bias).enqueue_on($stream)?,
            1024 => kernel_fused::$kernel_fn($out, $input, $bias).enqueue_on($stream)?,
            _ => unreachable!("unsupported fused tile size"),
        };
        Ok::<(), crate::error::Error>(())
    }};
}

macro_rules! dispatch_rms_norm_add {
    ($kernel_fn:ident, $out:expr, $residual_out:expr, $input:expr, $residual:expr, $weight:expr, $eps:expr, $weight_offset:expr, $stream:expr, $tile:expr) => {{
        match $tile {
            1 => kernel_fused::$kernel_fn(
                $out,
                $residual_out,
                $input,
                $residual,
                $weight,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            8 => kernel_fused::$kernel_fn(
                $out,
                $residual_out,
                $input,
                $residual,
                $weight,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            32 => kernel_fused::$kernel_fn(
                $out,
                $residual_out,
                $input,
                $residual,
                $weight,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            64 => kernel_fused::$kernel_fn(
                $out,
                $residual_out,
                $input,
                $residual,
                $weight,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            128 => kernel_fused::$kernel_fn(
                $out,
                $residual_out,
                $input,
                $residual,
                $weight,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            256 => kernel_fused::$kernel_fn(
                $out,
                $residual_out,
                $input,
                $residual,
                $weight,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            512 => kernel_fused::$kernel_fn(
                $out,
                $residual_out,
                $input,
                $residual,
                $weight,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            1024 => kernel_fused::$kernel_fn(
                $out,
                $residual_out,
                $input,
                $residual,
                $weight,
                $eps,
                $weight_offset,
            )
            .enqueue_on($stream)?,
            _ => unreachable!("unsupported rms norm fused tile size"),
        };
        Ok::<(), crate::error::Error>(())
    }};
}

macro_rules! dispatch_rms_norm_silu_mul {
    ($kernel_fn:ident, $out:expr, $input:expr, $weight:expr, $up:expr, $eps:expr, $weight_offset:expr, $stream:expr, $tile:expr) => {{
        match $tile {
            1 => kernel_fused::$kernel_fn($out, $input, $weight, $up, $eps, $weight_offset)
                .enqueue_on($stream)?,
            8 => kernel_fused::$kernel_fn($out, $input, $weight, $up, $eps, $weight_offset)
                .enqueue_on($stream)?,
            32 => kernel_fused::$kernel_fn($out, $input, $weight, $up, $eps, $weight_offset)
                .enqueue_on($stream)?,
            64 => kernel_fused::$kernel_fn($out, $input, $weight, $up, $eps, $weight_offset)
                .enqueue_on($stream)?,
            128 => kernel_fused::$kernel_fn($out, $input, $weight, $up, $eps, $weight_offset)
                .enqueue_on($stream)?,
            256 => kernel_fused::$kernel_fn($out, $input, $weight, $up, $eps, $weight_offset)
                .enqueue_on($stream)?,
            512 => kernel_fused::$kernel_fn($out, $input, $weight, $up, $eps, $weight_offset)
                .enqueue_on($stream)?,
            1024 => kernel_fused::$kernel_fn($out, $input, $weight, $up, $eps, $weight_offset)
                .enqueue_on($stream)?,
            _ => unreachable!("unsupported rms norm fused tile size"),
        };
        Ok::<(), crate::error::Error>(())
    }};
}

macro_rules! binary_fused_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lhs: DevicePointer<$ty>,
            rhs: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(lhs)?;
            checked_device_pointer(rhs)?;
            let tile = fused_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let lhs = TensorAdapter::contiguous_1d(lhs, len)?;
            let rhs = TensorAdapter::contiguous_1d(rhs, len)?;
            dispatch_1d!($kernel_fn, out, lhs, rhs, stream, tile)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! binary_fused_2d_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lhs: DevicePointer<$ty>,
            rhs: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(lhs)?;
            checked_device_pointer(rhs)?;
            let tile = fused_tile_size(cols);
            let out = TensorAdapter::contiguous_2d(out, rows, cols)?.partition([1, tile])?;
            let lhs = TensorAdapter::contiguous_2d(lhs, rows, cols)?;
            let rhs = TensorAdapter::contiguous_2d(rhs, rows, cols)?;
            dispatch_1d!($kernel_fn, out, lhs, rhs, stream, tile)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! ternary_fused_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            a: DevicePointer<$ty>,
            b: DevicePointer<$ty>,
            c: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(a)?;
            checked_device_pointer(b)?;
            checked_device_pointer(c)?;
            let tile = fused_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let a = TensorAdapter::contiguous_1d(a, len)?;
            let b = TensorAdapter::contiguous_1d(b, len)?;
            let c = TensorAdapter::contiguous_1d(c, len)?;
            dispatch_1d_ternary!($kernel_fn, out, a, b, c, stream, tile)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! ternary_fused_2d_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            a: DevicePointer<$ty>,
            b: DevicePointer<$ty>,
            c: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(a)?;
            checked_device_pointer(b)?;
            checked_device_pointer(c)?;
            let tile = fused_tile_size(cols);
            let out = TensorAdapter::contiguous_2d(out, rows, cols)?.partition([1, tile])?;
            let a = TensorAdapter::contiguous_2d(a, rows, cols)?;
            let b = TensorAdapter::contiguous_2d(b, rows, cols)?;
            let c = TensorAdapter::contiguous_2d(c, rows, cols)?;
            dispatch_1d_ternary!($kernel_fn, out, a, b, c, stream, tile)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! conditional_fused_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            condition: DevicePointer<u8>,
            a: DevicePointer<$ty>,
            b: DevicePointer<$ty>,
            c: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(condition)?;
            checked_device_pointer(a)?;
            checked_device_pointer(b)?;
            checked_device_pointer(c)?;
            let tile = fused_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let condition = TensorAdapter::contiguous_1d(condition, len)?;
            let a = TensorAdapter::contiguous_1d(a, len)?;
            let b = TensorAdapter::contiguous_1d(b, len)?;
            let c = TensorAdapter::contiguous_1d(c, len)?;
            dispatch_conditional_ternary!($kernel_fn, out, condition, a, b, c, stream, tile)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! conditional_fused_2d_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            condition: DevicePointer<u8>,
            a: DevicePointer<$ty>,
            b: DevicePointer<$ty>,
            c: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(condition)?;
            checked_device_pointer(a)?;
            checked_device_pointer(b)?;
            checked_device_pointer(c)?;
            let tile = fused_tile_size(cols);
            let out = TensorAdapter::contiguous_2d(out, rows, cols)?.partition([1, tile])?;
            let condition = TensorAdapter::contiguous_2d(condition, rows, cols)?;
            let a = TensorAdapter::contiguous_2d(a, rows, cols)?;
            let b = TensorAdapter::contiguous_2d(b, rows, cols)?;
            let c = TensorAdapter::contiguous_2d(c, rows, cols)?;
            dispatch_conditional_ternary!($kernel_fn, out, condition, a, b, c, stream, tile)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! bias_fused_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            bias: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(bias)?;
            let tile = fused_tile_size(cols);
            let out = TensorAdapter::contiguous_2d(out, rows, cols)?.partition([1, tile])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            let bias = TensorAdapter::contiguous_1d(bias, cols)?;
            dispatch_2d_bias!($kernel_fn, out, input, bias, stream, tile)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! affine_fused_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            scale: DevicePointer<$ty>,
            bias: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(scale)?;
            checked_device_pointer(bias)?;
            let tile = fused_tile_size(cols);
            let out = TensorAdapter::contiguous_2d(out, rows, cols)?.partition([1, tile])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            let scale = TensorAdapter::contiguous_1d(scale, cols)?;
            let bias = TensorAdapter::contiguous_1d(bias, cols)?;
            dispatch_1d_ternary!($kernel_fn, out, input, scale, bias, stream, tile)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! column2_fused_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            lhs: DevicePointer<$ty>,
            rhs: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(lhs)?;
            checked_device_pointer(rhs)?;
            let tile = fused_tile_size(cols);
            let out = TensorAdapter::contiguous_2d(out, rows, cols)?.partition([1, tile])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            let lhs = TensorAdapter::contiguous_1d(lhs, cols)?;
            let rhs = TensorAdapter::contiguous_1d(rhs, cols)?;
            dispatch_1d_ternary!($kernel_fn, out, input, lhs, rhs, stream, tile)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! rms_norm_add_fn {
    ($name:ident, $ty:ty, $eps_ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            residual_out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            residual: DevicePointer<$ty>,
            weight: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
            weight_offset: $eps_ty,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(residual_out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(residual)?;
            checked_device_pointer(weight)?;
            let tile = rms_norm_tile_size(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, cols)?.partition([1, tile])?;
            let residual_out =
                TensorAdapter::contiguous_2d(residual_out, rows, cols)?.partition([1, tile])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            let residual = TensorAdapter::contiguous_2d(residual, rows, cols)?;
            let weight = TensorAdapter::contiguous_1d(weight, cols)?;
            dispatch_rms_norm_add!(
                $kernel_fn,
                out,
                residual_out,
                input,
                residual,
                weight,
                eps,
                weight_offset,
                stream,
                tile
            )?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! rms_norm_add_wide_fn {
    (
        $name:ident,
        $ty:ty,
        $eps_ty:ty,
        $partial:ident,
        $normalize:ident
    ) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            residual_out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            residual: DevicePointer<$ty>,
            weight: DevicePointer<$ty>,
            partial_sum: DevicePointer<f32>,
            row_sum: DevicePointer<f32>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
            weight_offset: $eps_ty,
        ) -> Result<()> {
            let Some(wide) = RmsNormWide::create(rows, cols)? else {
                return Ok::<(), crate::error::Error>(());
            };
            checked_device_pointer(out)?;
            checked_device_pointer(residual_out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(residual)?;
            checked_device_pointer(weight)?;
            checked_device_pointer(partial_sum)?;
            checked_device_pointer(row_sum)?;

            unsafe {
                kernel_fused::$partial(
                    residual_out,
                    partial_sum,
                    input,
                    residual,
                    wide.cols,
                    wide.chunks,
                )
            }
            .grid(wide.chunk_grid)
            .enqueue_on(stream)?;

            unsafe {
                kernel_fused::rms_norm_add_wide_reduce_f32(row_sum, partial_sum, wide.chunks)
            }
            .grid(wide.row_grid)
            .enqueue_on(stream)?;

            unsafe {
                kernel_fused::$normalize(
                    out,
                    residual_out,
                    row_sum,
                    weight,
                    wide.cols,
                    wide.len,
                    eps,
                    weight_offset,
                )
            }
            .grid(wide.len_grid)
            .enqueue_on(stream)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! rms_norm_silu_mul_fn {
    ($name:ident, $ty:ty, $eps_ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            weight: DevicePointer<$ty>,
            up: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
            weight_offset: $eps_ty,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(weight)?;
            checked_device_pointer(up)?;
            let tile = rms_norm_tile_size(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, cols)?.partition([1, tile])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            let weight = TensorAdapter::contiguous_1d(weight, cols)?;
            let up = TensorAdapter::contiguous_2d(up, rows, cols)?;
            dispatch_rms_norm_silu_mul!(
                $kernel_fn,
                out,
                input,
                weight,
                up,
                eps,
                weight_offset,
                stream,
                tile
            )?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! rms_norm_gated_silu_fn {
    ($name:ident, $ty:ty, $eps_ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            gate: DevicePointer<$ty>,
            weight: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
            weight_offset: $eps_ty,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            if cols > 1024 {
                return Err(Error::UnsupportedWidth {
                    op: "rms norm gated silu".into(),
                    width: cols,
                });
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(gate)?;
            checked_device_pointer(weight)?;
            unsafe {
                kernel_fused::$kernel_fn(
                    out,
                    input,
                    gate,
                    weight,
                    checked_i32_value(cols)?,
                    eps,
                    weight_offset,
                )
            }
            .grid((u32::try_from(rows).map_err(|_| Error::SizeOverflow)?, 1, 1))
            .enqueue_on(stream)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! dual_rms_norm_fn {
    ($name:ident, $ty:ty, $eps_ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            q_out: DevicePointer<$ty>,
            k_out: DevicePointer<$ty>,
            q: DevicePointer<$ty>,
            k: DevicePointer<$ty>,
            q_weight: DevicePointer<$ty>,
            k_weight: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            if cols > 1024 {
                return Err(Error::UnsupportedWidth {
                    op: "dual rms norm".into(),
                    width: cols,
                });
            }
            checked_device_pointer(q_out)?;
            checked_device_pointer(k_out)?;
            checked_device_pointer(q)?;
            checked_device_pointer(k)?;
            checked_device_pointer(q_weight)?;
            checked_device_pointer(k_weight)?;
            unsafe {
                kernel_fused::$kernel_fn(
                    q_out,
                    k_out,
                    q,
                    k,
                    q_weight,
                    k_weight,
                    checked_i32_value(cols)?,
                    eps,
                )
            }
            .grid((u32::try_from(rows).map_err(|_| Error::SizeOverflow)?, 1, 1))
            .enqueue_on(stream)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! rms_norm_residual_add_fn {
    ($name:ident, $ty:ty, $eps_ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            residual: DevicePointer<$ty>,
            weight: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            if cols > 1024 {
                return Err(Error::UnsupportedWidth {
                    op: "rms norm residual add".into(),
                    width: cols,
                });
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(residual)?;
            checked_device_pointer(weight)?;
            unsafe {
                kernel_fused::$kernel_fn(
                    out,
                    input,
                    residual,
                    weight,
                    checked_i32_value(cols)?,
                    eps,
                )
            }
            .grid((u32::try_from(rows).map_err(|_| Error::SizeOverflow)?, 1, 1))
            .enqueue_on(stream)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
pub fn mhc_apply_residual_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    x: DevicePointer<f32>,
    f_out: DevicePointer<f32>,
    y: DevicePointer<f32>,
    batch: usize,
    n: usize,
    channels: usize,
) -> Result<()> {
    let len = checked_element_count(checked_element_count(batch, n)?, channels)?;
    if len == 0 {
        return Ok::<(), crate::error::Error>(());
    }
    checked_device_pointer(out)?;
    checked_device_pointer(x)?;
    checked_device_pointer(f_out)?;
    checked_device_pointer(y)?;
    let len_i32 = checked_i32_value(len)?;
    let channels_i32 = checked_i32_value(channels)?;
    let grid = raw_vector_grid(len)?;
    unsafe {
        match n {
            1 => kernel_fused::mhc_apply_residual_f32_n1(out, x, f_out, y, channels_i32, len_i32)
                .grid(grid)
                .enqueue_on(stream)?,
            2 => kernel_fused::mhc_apply_residual_f32_n2(out, x, f_out, y, channels_i32, len_i32)
                .grid(grid)
                .enqueue_on(stream)?,
            3 => kernel_fused::mhc_apply_residual_f32_n3(out, x, f_out, y, channels_i32, len_i32)
                .grid(grid)
                .enqueue_on(stream)?,
            4 => kernel_fused::mhc_apply_residual_f32_n4(out, x, f_out, y, channels_i32, len_i32)
                .grid(grid)
                .enqueue_on(stream)?,
            5 => kernel_fused::mhc_apply_residual_f32_n5(out, x, f_out, y, channels_i32, len_i32)
                .grid(grid)
                .enqueue_on(stream)?,
            6 => kernel_fused::mhc_apply_residual_f32_n6(out, x, f_out, y, channels_i32, len_i32)
                .grid(grid)
                .enqueue_on(stream)?,
            7 => kernel_fused::mhc_apply_residual_f32_n7(out, x, f_out, y, channels_i32, len_i32)
                .grid(grid)
                .enqueue_on(stream)?,
            8 => kernel_fused::mhc_apply_residual_f32_n8(out, x, f_out, y, channels_i32, len_i32)
                .grid(grid)
                .enqueue_on(stream)?,
            16 => kernel_fused::mhc_apply_residual_f32_n16(out, x, f_out, y, channels_i32, len_i32)
                .grid(grid)
                .enqueue_on(stream)?,
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "mhc apply residual n".into(),
                    width: n,
                });
            }
        }
    };
    Ok::<(), crate::error::Error>(())
}

#[cfg(feature = "dtype-f32")]
pub fn mhc_sinkhorn_f32(
    stream: &Arc<Stream>,
    y: DevicePointer<f32>,
    batch: usize,
    n: usize,
) -> Result<()> {
    if batch == 0 || n == 0 {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(y)?;
    let batch_i32 = checked_i32_value(batch)?;
    unsafe {
        match n {
            1 => kernel_fused::mhc_sinkhorn_f32_n1(y, batch_i32)
                .grid((u32::try_from(batch).map_err(|_| Error::SizeOverflow)?, 1, 1))
                .enqueue_on(stream)?,
            2 => kernel_fused::mhc_sinkhorn_f32_n2(y, batch_i32)
                .grid((u32::try_from(batch).map_err(|_| Error::SizeOverflow)?, 1, 1))
                .enqueue_on(stream)?,
            3 => kernel_fused::mhc_sinkhorn_f32_n3(y, batch_i32)
                .grid((u32::try_from(batch).map_err(|_| Error::SizeOverflow)?, 1, 1))
                .enqueue_on(stream)?,
            4 => kernel_fused::mhc_sinkhorn_f32_n4(y, batch_i32)
                .grid((u32::try_from(batch).map_err(|_| Error::SizeOverflow)?, 1, 1))
                .enqueue_on(stream)?,
            5 => kernel_fused::mhc_sinkhorn_f32_n5(y, batch_i32)
                .grid((u32::try_from(batch).map_err(|_| Error::SizeOverflow)?, 1, 1))
                .enqueue_on(stream)?,
            6 => kernel_fused::mhc_sinkhorn_f32_n6(y, batch_i32)
                .grid((u32::try_from(batch).map_err(|_| Error::SizeOverflow)?, 1, 1))
                .enqueue_on(stream)?,
            7 => kernel_fused::mhc_sinkhorn_f32_n7(y, batch_i32)
                .grid((u32::try_from(batch).map_err(|_| Error::SizeOverflow)?, 1, 1))
                .enqueue_on(stream)?,
            8 => kernel_fused::mhc_sinkhorn_f32_n8(y, batch_i32)
                .grid((u32::try_from(batch).map_err(|_| Error::SizeOverflow)?, 1, 1))
                .enqueue_on(stream)?,
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "mhc sinkhorn n".into(),
                    width: n,
                });
            }
        }
    };
    Ok::<(), crate::error::Error>(())
}

#[cfg(feature = "dtype-f32")]

pub fn mhc_gemm_rms_scale_f32(
    stream: &Arc<Stream>,
    y: DevicePointer<f32>,
    r: DevicePointer<f32>,
    x: DevicePointer<f32>,
    w: DevicePointer<f32>,
    bias: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    reduction: usize,
    n: usize,
    alpha_pre: f32,
    alpha_post: f32,
    alpha_res: f32,
) -> Result<()> {
    let output_len = checked_element_count(rows, columns)?;
    if output_len == 0 || reduction == 0 || n == 0 {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(y)?;
    checked_device_pointer(r)?;
    checked_device_pointer(x)?;
    checked_device_pointer(w)?;
    checked_device_pointer(bias)?;
    unsafe {
        kernel_fused::mhc_gemm_rms_scale_f32(
            y,
            r,
            x,
            w,
            bias,
            checked_i32_value(columns)?,
            checked_i32_value(reduction)?,
            checked_i32_value(n)?,
            alpha_pre,
            alpha_post,
            alpha_res,
            checked_i32_value(output_len)?,
        )
    }
    .grid(raw_vector_grid(output_len)?)
    .enqueue_on(stream)?;
    Ok::<(), crate::error::Error>(())
}

#[cfg(feature = "dtype-f32")]

pub fn mhc_split_gemm_rms_f32(
    stream: &Arc<Stream>,
    y_acc: DevicePointer<f32>,
    r_acc: DevicePointer<f32>,
    x: DevicePointer<f32>,
    w: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    reduction: usize,
    split_k: usize,
) -> Result<()> {
    let base_len = checked_element_count(rows, columns)?;
    let output_len = checked_element_count(base_len, split_k)?;
    if output_len == 0 || reduction == 0 {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(y_acc)?;
    checked_device_pointer(r_acc)?;
    checked_device_pointer(x)?;
    checked_device_pointer(w)?;
    unsafe {
        kernel_fused::mhc_split_gemm_rms_f32(
            y_acc,
            r_acc,
            x,
            w,
            checked_i32_value(columns)?,
            checked_i32_value(reduction)?,
            checked_i32_value(split_k)?,
            checked_i32_value(output_len)?,
        )
    }
    .grid(raw_vector_grid(output_len)?)
    .enqueue_on(stream)?;
    Ok::<(), crate::error::Error>(())
}

#[cfg(feature = "dtype-f32")]

pub fn mhc_finalize_scale_bias_sigmoid_f32(
    stream: &Arc<Stream>,
    y: DevicePointer<f32>,
    r: DevicePointer<f32>,
    y_acc: DevicePointer<f32>,
    r_acc: DevicePointer<f32>,
    bias: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    reduction: usize,
    n: usize,
    split_k: usize,
    alpha_pre: f32,
    alpha_post: f32,
    alpha_res: f32,
) -> Result<()> {
    let output_len = checked_element_count(rows, columns)?;
    if output_len == 0 || reduction == 0 || n == 0 || split_k == 0 {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(y)?;
    checked_device_pointer(r)?;
    checked_device_pointer(y_acc)?;
    checked_device_pointer(r_acc)?;
    checked_device_pointer(bias)?;
    unsafe {
        kernel_fused::mhc_finalize_scale_bias_sigmoid_f32(
            y,
            r,
            y_acc,
            r_acc,
            bias,
            checked_i32_value(columns)?,
            checked_i32_value(reduction)?,
            checked_i32_value(n)?,
            checked_i32_value(split_k)?,
            alpha_pre,
            alpha_post,
            alpha_res,
            checked_i32_value(output_len)?,
        )
    }
    .grid(raw_vector_grid(output_len)?)
    .enqueue_on(stream)?;
    Ok::<(), crate::error::Error>(())
}

macro_rules! kv_cache_update_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            k_cache: DevicePointer<$ty>,
            v_cache: DevicePointer<$ty>,
            new_k: DevicePointer<$ty>,
            new_v: DevicePointer<$ty>,
            seq_len: usize,
            heads: usize,
            head_dim: usize,
            max_seq: usize,
            position_start: usize,
        ) -> Result<()> {
            if seq_len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            if heads == 0 || head_dim == 0 || max_seq == 0 {
                return Err(Error::InvalidLength);
            }
            let position_end = position_start
                .checked_add(seq_len)
                .ok_or(Error::SizeOverflow)?;
            if position_end > max_seq {
                return Err(Error::LengthMismatch);
            }
            let len = checked_element_count(checked_element_count(seq_len, heads)?, head_dim)?;
            checked_device_pointer(k_cache)?;
            checked_device_pointer(v_cache)?;
            checked_device_pointer(new_k)?;
            checked_device_pointer(new_v)?;
            let seq_len = checked_i32_value(seq_len)?;
            let heads = checked_i32_value(heads)?;
            let head_dim = checked_i32_value(head_dim)?;
            let max_seq = checked_i32_value(max_seq)?;
            let position_start = checked_i32_value(position_start)?;
            let len_i32 = checked_i32_value(len)?;
            let grid = raw_vector_grid(len)?;
            unsafe {
                kernel_fused::$kernel_fn(
                    k_cache,
                    v_cache,
                    new_k,
                    new_v,
                    seq_len,
                    heads,
                    head_dim,
                    max_seq,
                    position_start,
                    len_i32,
                )
            }
            .grid(grid)
            .enqueue_on(stream)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! kv_cache_update_dynpos_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            k_cache: DevicePointer<$ty>,
            v_cache: DevicePointer<$ty>,
            new_k: DevicePointer<$ty>,
            new_v: DevicePointer<$ty>,
            position_start: DevicePointer<u32>,
            seq_len: usize,
            heads: usize,
            head_dim: usize,
            max_seq: usize,
        ) -> Result<()> {
            if seq_len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            if heads == 0 || head_dim == 0 || max_seq == 0 {
                return Err(Error::InvalidLength);
            }
            if seq_len > max_seq {
                return Err(Error::LengthMismatch);
            }
            let len = checked_element_count(checked_element_count(seq_len, heads)?, head_dim)?;
            checked_device_pointer(k_cache)?;
            checked_device_pointer(v_cache)?;
            checked_device_pointer(new_k)?;
            checked_device_pointer(new_v)?;
            checked_device_pointer(position_start)?;
            let seq_len = checked_i32_value(seq_len)?;
            let heads = checked_i32_value(heads)?;
            let head_dim = checked_i32_value(head_dim)?;
            let max_seq = checked_i32_value(max_seq)?;
            let len_i32 = checked_i32_value(len)?;
            let grid = raw_vector_grid(len)?;
            unsafe {
                kernel_fused::$kernel_fn(
                    k_cache,
                    v_cache,
                    new_k,
                    new_v,
                    position_start,
                    seq_len,
                    heads,
                    head_dim,
                    max_seq,
                    len_i32,
                )
            }
            .grid(grid)
            .enqueue_on(stream)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

#[cfg(feature = "dtype-bf16")]
binary_fused_fn!(silu_mul_bf16, bf16, silu_mul_bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_fn!(silu_mul_f16, f16, silu_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(silu_mul_f32, f32, silu_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(silu_mul_f64, f64, silu_mul_f64);
#[cfg(feature = "dtype-bf16")]
binary_fused_2d_fn!(silu_mul_2d_bf16, bf16, silu_mul_2d_bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(silu_mul_2d_f16, f16, silu_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(silu_mul_2d_f32, f32, silu_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(silu_mul_2d_f64, f64, silu_mul_2d_f64);

#[cfg(feature = "dtype-bf16")]
binary_fused_fn!(swiglu_bf16, bf16, silu_mul_bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_fn!(swiglu_f16, f16, silu_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(swiglu_f32, f32, silu_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(swiglu_f64, f64, silu_mul_f64);
#[cfg(feature = "dtype-bf16")]
binary_fused_2d_fn!(swiglu_2d_bf16, bf16, silu_mul_2d_bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(swiglu_2d_f16, f16, silu_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(swiglu_2d_f32, f32, silu_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(swiglu_2d_f64, f64, silu_mul_2d_f64);

#[cfg(feature = "dtype-bf16")]
binary_fused_fn!(gelu_mul_bf16, bf16, gelu_mul_bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_fn!(gelu_mul_f16, f16, gelu_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(gelu_mul_f32, f32, gelu_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(gelu_mul_f64, f64, gelu_mul_f64);
#[cfg(feature = "dtype-bf16")]
binary_fused_2d_fn!(gelu_mul_2d_bf16, bf16, gelu_mul_2d_bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(gelu_mul_2d_f16, f16, gelu_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(gelu_mul_2d_f32, f32, gelu_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(gelu_mul_2d_f64, f64, gelu_mul_2d_f64);

#[cfg(feature = "dtype-bf16")]
binary_fused_fn!(geglu_approx_bf16, bf16, gelu_mul_bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_fn!(geglu_approx_f16, f16, gelu_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(geglu_approx_f32, f32, gelu_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(geglu_approx_f64, f64, gelu_mul_f64);
#[cfg(feature = "dtype-bf16")]
binary_fused_2d_fn!(geglu_approx_2d_bf16, bf16, gelu_mul_2d_bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(geglu_approx_2d_f16, f16, gelu_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(geglu_approx_2d_f32, f32, gelu_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(geglu_approx_2d_f64, f64, gelu_mul_2d_f64);

macro_rules! exact_gelu_mul_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            gate: DevicePointer<$ty>,
            up: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(gate)?;
            checked_device_pointer(up)?;
            let len_i32 = checked_i32_value(len)?;
            let blocks = u32::try_from(len.div_ceil(1024)).map_err(|_| Error::SizeOverflow)?;
            unsafe { kernel_fused::$kernel_fn(out, gate, up, len_i32) }
                .grid((blocks, 1, 1))
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! exact_gelu_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let len_i32 = checked_i32_value(len)?;
            let blocks = u32::try_from(len.div_ceil(1024)).map_err(|_| Error::SizeOverflow)?;
            unsafe { kernel_fused::$kernel_fn(out, input, len_i32) }
                .grid((blocks, 1, 1))
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! silu_and_mul_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            rows: usize,
            hidden: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, hidden)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let hidden_i32 = checked_i32_value(hidden)?;
            let len_i32 = checked_i32_value(len)?;
            let blocks = u32::try_from(len.div_ceil(1024)).map_err(|_| Error::SizeOverflow)?;
            unsafe { kernel_fused::$kernel_fn(out, input, hidden_i32, len_i32) }
                .grid((blocks, 1, 1))
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
exact_gelu_mul_fn!(exact_gelu_mul_f32, f32, exact_gelu_mul_f32);
#[cfg(feature = "dtype-f32")]
exact_gelu_fn!(exact_gelu_f32, f32, exact_gelu_f32);
#[cfg(feature = "dtype-f16")]
exact_gelu_mul_fn!(exact_gelu_mul_f16, f16, exact_gelu_mul_f16);
#[cfg(feature = "dtype-f16")]
exact_gelu_fn!(exact_gelu_f16, f16, exact_gelu_f16);
#[cfg(feature = "dtype-bf16")]
exact_gelu_mul_fn!(exact_gelu_mul_bf16, bf16, exact_gelu_mul_bf16);
#[cfg(feature = "dtype-bf16")]
exact_gelu_fn!(exact_gelu_bf16, bf16, exact_gelu_bf16);
#[cfg(feature = "dtype-f32")]
exact_gelu_mul_fn!(geglu_exact_f32, f32, exact_gelu_mul_f32);
#[cfg(feature = "dtype-f16")]
exact_gelu_mul_fn!(geglu_exact_f16, f16, exact_gelu_mul_f16);
#[cfg(feature = "dtype-bf16")]
exact_gelu_mul_fn!(geglu_exact_bf16, bf16, exact_gelu_mul_bf16);

#[cfg(feature = "dtype-f32")]
silu_and_mul_fn!(silu_and_mul_f32, f32, silu_and_mul_f32);
#[cfg(feature = "dtype-f16")]
silu_and_mul_fn!(silu_and_mul_f16, f16, silu_and_mul_f16);
#[cfg(feature = "dtype-bf16")]
silu_and_mul_fn!(silu_and_mul_bf16, bf16, silu_and_mul_bf16);

#[cfg(feature = "dtype-bf16")]
binary_fused_fn!(sigmoid_mul_bf16, bf16, sigmoid_mul_bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_fn!(sigmoid_mul_f16, f16, sigmoid_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(sigmoid_mul_f32, f32, sigmoid_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(sigmoid_mul_f64, f64, sigmoid_mul_f64);
#[cfg(feature = "dtype-bf16")]
binary_fused_2d_fn!(sigmoid_mul_2d_bf16, bf16, sigmoid_mul_2d_bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(sigmoid_mul_2d_f16, f16, sigmoid_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(sigmoid_mul_2d_f32, f32, sigmoid_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(sigmoid_mul_2d_f64, f64, sigmoid_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(relu_mul_f16, f16, relu_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(relu_mul_f32, f32, relu_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(relu_mul_f64, f64, relu_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(relu_mul_2d_f16, f16, relu_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(relu_mul_2d_f32, f32, relu_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(relu_mul_2d_f64, f64, relu_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(relu6_mul_f16, f16, relu6_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(relu6_mul_f32, f32, relu6_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(relu6_mul_f64, f64, relu6_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(relu6_mul_2d_f16, f16, relu6_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(relu6_mul_2d_f32, f32, relu6_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(relu6_mul_2d_f64, f64, relu6_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(hard_sigmoid_mul_f16, f16, hard_sigmoid_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(hard_sigmoid_mul_f32, f32, hard_sigmoid_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(hard_sigmoid_mul_f64, f64, hard_sigmoid_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(hard_sigmoid_mul_2d_f16, f16, hard_sigmoid_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(hard_sigmoid_mul_2d_f32, f32, hard_sigmoid_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(hard_sigmoid_mul_2d_f64, f64, hard_sigmoid_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(hard_swish_mul_f16, f16, hard_swish_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(hard_swish_mul_f32, f32, hard_swish_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(hard_swish_mul_f64, f64, hard_swish_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(hard_swish_mul_2d_f16, f16, hard_swish_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(hard_swish_mul_2d_f32, f32, hard_swish_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(hard_swish_mul_2d_f64, f64, hard_swish_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(softsign_mul_f16, f16, softsign_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(softsign_mul_f32, f32, softsign_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(softsign_mul_f64, f64, softsign_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(softsign_mul_2d_f16, f16, softsign_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(softsign_mul_2d_f32, f32, softsign_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(softsign_mul_2d_f64, f64, softsign_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(mish_mul_f16, f16, mish_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(mish_mul_f32, f32, mish_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(mish_mul_f64, f64, mish_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(mish_mul_2d_f16, f16, mish_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(mish_mul_2d_f32, f32, mish_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(mish_mul_2d_f64, f64, mish_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(selu_mul_f16, f16, selu_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(selu_mul_f32, f32, selu_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(selu_mul_f64, f64, selu_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(selu_mul_2d_f16, f16, selu_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(selu_mul_2d_f32, f32, selu_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(selu_mul_2d_f64, f64, selu_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(tanhshrink_mul_f16, f16, tanhshrink_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(tanhshrink_mul_f32, f32, tanhshrink_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(tanhshrink_mul_f64, f64, tanhshrink_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(tanhshrink_mul_2d_f16, f16, tanhshrink_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(tanhshrink_mul_2d_f32, f32, tanhshrink_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(tanhshrink_mul_2d_f64, f64, tanhshrink_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(log_sigmoid_mul_f16, f16, log_sigmoid_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(log_sigmoid_mul_f32, f32, log_sigmoid_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(log_sigmoid_mul_f64, f64, log_sigmoid_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(log_sigmoid_mul_2d_f16, f16, log_sigmoid_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(log_sigmoid_mul_2d_f32, f32, log_sigmoid_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(log_sigmoid_mul_2d_f64, f64, log_sigmoid_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(reciprocal_mul_f16, f16, reciprocal_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(reciprocal_mul_f32, f32, reciprocal_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(reciprocal_mul_f64, f64, reciprocal_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(reciprocal_mul_2d_f16, f16, reciprocal_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(reciprocal_mul_2d_f32, f32, reciprocal_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(reciprocal_mul_2d_f64, f64, reciprocal_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(square_mul_f16, f16, square_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(square_mul_f32, f32, square_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(square_mul_f64, f64, square_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(square_mul_2d_f16, f16, square_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(square_mul_2d_f32, f32, square_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(square_mul_2d_f64, f64, square_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(ceil_mul_f16, f16, ceil_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(ceil_mul_f32, f32, ceil_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(ceil_mul_f64, f64, ceil_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(ceil_mul_2d_f16, f16, ceil_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(ceil_mul_2d_f32, f32, ceil_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(ceil_mul_2d_f64, f64, ceil_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(floor_mul_f16, f16, floor_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(floor_mul_f32, f32, floor_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(floor_mul_f64, f64, floor_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(floor_mul_2d_f16, f16, floor_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(floor_mul_2d_f32, f32, floor_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(floor_mul_2d_f64, f64, floor_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(trunc_mul_f16, f16, trunc_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(trunc_mul_f32, f32, trunc_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(trunc_mul_f64, f64, trunc_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(trunc_mul_2d_f16, f16, trunc_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(trunc_mul_2d_f32, f32, trunc_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(trunc_mul_2d_f64, f64, trunc_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(frac_mul_f16, f16, frac_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(frac_mul_f32, f32, frac_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(frac_mul_f64, f64, frac_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(frac_mul_2d_f16, f16, frac_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(frac_mul_2d_f32, f32, frac_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(frac_mul_2d_f64, f64, frac_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(deg2rad_mul_f16, f16, deg2rad_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(deg2rad_mul_f32, f32, deg2rad_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(deg2rad_mul_f64, f64, deg2rad_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(deg2rad_mul_2d_f16, f16, deg2rad_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(deg2rad_mul_2d_f32, f32, deg2rad_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(deg2rad_mul_2d_f64, f64, deg2rad_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(rad2deg_mul_f16, f16, rad2deg_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(rad2deg_mul_f32, f32, rad2deg_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(rad2deg_mul_f64, f64, rad2deg_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(rad2deg_mul_2d_f16, f16, rad2deg_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(rad2deg_mul_2d_f32, f32, rad2deg_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(rad2deg_mul_2d_f64, f64, rad2deg_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(sin_mul_f16, f16, sin_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(sin_mul_f32, f32, sin_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(sin_mul_f64, f64, sin_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(sin_mul_2d_f16, f16, sin_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(sin_mul_2d_f32, f32, sin_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(sin_mul_2d_f64, f64, sin_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(cos_mul_f16, f16, cos_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(cos_mul_f32, f32, cos_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(cos_mul_f64, f64, cos_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(cos_mul_2d_f16, f16, cos_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(cos_mul_2d_f32, f32, cos_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(cos_mul_2d_f64, f64, cos_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(tan_mul_f16, f16, tan_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(tan_mul_f32, f32, tan_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(tan_mul_f64, f64, tan_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(tan_mul_2d_f16, f16, tan_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(tan_mul_2d_f32, f32, tan_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(tan_mul_2d_f64, f64, tan_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(asin_mul_f16, f16, asin_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(asin_mul_f32, f32, asin_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(asin_mul_f64, f64, asin_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(asin_mul_2d_f16, f16, asin_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(asin_mul_2d_f32, f32, asin_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(asin_mul_2d_f64, f64, asin_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(acos_mul_f16, f16, acos_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(acos_mul_f32, f32, acos_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(acos_mul_f64, f64, acos_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(acos_mul_2d_f16, f16, acos_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(acos_mul_2d_f32, f32, acos_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(acos_mul_2d_f64, f64, acos_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(atan_mul_f16, f16, atan_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(atan_mul_f32, f32, atan_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(atan_mul_f64, f64, atan_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(atan_mul_2d_f16, f16, atan_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(atan_mul_2d_f32, f32, atan_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(atan_mul_2d_f64, f64, atan_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(sinh_mul_f16, f16, sinh_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(sinh_mul_f32, f32, sinh_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(sinh_mul_f64, f64, sinh_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(sinh_mul_2d_f16, f16, sinh_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(sinh_mul_2d_f32, f32, sinh_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(sinh_mul_2d_f64, f64, sinh_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(cosh_mul_f16, f16, cosh_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(cosh_mul_f32, f32, cosh_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(cosh_mul_f64, f64, cosh_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(cosh_mul_2d_f16, f16, cosh_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(cosh_mul_2d_f32, f32, cosh_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(cosh_mul_2d_f64, f64, cosh_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(asinh_mul_f16, f16, asinh_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(asinh_mul_f32, f32, asinh_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(asinh_mul_f64, f64, asinh_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(asinh_mul_2d_f16, f16, asinh_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(asinh_mul_2d_f32, f32, asinh_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(asinh_mul_2d_f64, f64, asinh_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(acosh_mul_f16, f16, acosh_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(acosh_mul_f32, f32, acosh_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(acosh_mul_f64, f64, acosh_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(acosh_mul_2d_f16, f16, acosh_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(acosh_mul_2d_f32, f32, acosh_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(acosh_mul_2d_f64, f64, acosh_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(atanh_mul_f16, f16, atanh_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(atanh_mul_f32, f32, atanh_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(atanh_mul_f64, f64, atanh_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(atanh_mul_2d_f16, f16, atanh_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(atanh_mul_2d_f32, f32, atanh_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(atanh_mul_2d_f64, f64, atanh_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(tanh_mul_f16, f16, tanh_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(tanh_mul_f32, f32, tanh_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(tanh_mul_f64, f64, tanh_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(tanh_mul_2d_f16, f16, tanh_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(tanh_mul_2d_f32, f32, tanh_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(tanh_mul_2d_f64, f64, tanh_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(exp_mul_f16, f16, exp_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(exp_mul_f32, f32, exp_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(exp_mul_f64, f64, exp_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(exp_mul_2d_f16, f16, exp_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(exp_mul_2d_f32, f32, exp_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(exp_mul_2d_f64, f64, exp_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(exp2_mul_f16, f16, exp2_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(exp2_mul_f32, f32, exp2_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(exp2_mul_f64, f64, exp2_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(exp2_mul_2d_f16, f16, exp2_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(exp2_mul_2d_f32, f32, exp2_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(exp2_mul_2d_f64, f64, exp2_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(expm1_mul_f16, f16, expm1_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(expm1_mul_f32, f32, expm1_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(expm1_mul_f64, f64, expm1_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(expm1_mul_2d_f16, f16, expm1_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(expm1_mul_2d_f32, f32, expm1_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(expm1_mul_2d_f64, f64, expm1_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(log_mul_f16, f16, log_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(log_mul_f32, f32, log_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(log_mul_f64, f64, log_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(log_mul_2d_f16, f16, log_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(log_mul_2d_f32, f32, log_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(log_mul_2d_f64, f64, log_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(log2_mul_f16, f16, log2_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(log2_mul_f32, f32, log2_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(log2_mul_f64, f64, log2_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(log2_mul_2d_f16, f16, log2_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(log2_mul_2d_f32, f32, log2_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(log2_mul_2d_f64, f64, log2_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(log10_mul_f16, f16, log10_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(log10_mul_f32, f32, log10_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(log10_mul_f64, f64, log10_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(log10_mul_2d_f16, f16, log10_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(log10_mul_2d_f32, f32, log10_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(log10_mul_2d_f64, f64, log10_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(log1p_mul_f16, f16, log1p_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(log1p_mul_f32, f32, log1p_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(log1p_mul_f64, f64, log1p_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(log1p_mul_2d_f16, f16, log1p_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(log1p_mul_2d_f32, f32, log1p_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(log1p_mul_2d_f64, f64, log1p_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(softplus_mul_f16, f16, softplus_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(softplus_mul_f32, f32, softplus_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(softplus_mul_f64, f64, softplus_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(softplus_mul_2d_f16, f16, softplus_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(softplus_mul_2d_f32, f32, softplus_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(softplus_mul_2d_f64, f64, softplus_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(sqrt_mul_f16, f16, sqrt_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(sqrt_mul_f32, f32, sqrt_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(sqrt_mul_f64, f64, sqrt_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(sqrt_mul_2d_f16, f16, sqrt_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(sqrt_mul_2d_f32, f32, sqrt_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(sqrt_mul_2d_f64, f64, sqrt_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(rsqrt_mul_f16, f16, rsqrt_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(rsqrt_mul_f32, f32, rsqrt_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(rsqrt_mul_f64, f64, rsqrt_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(rsqrt_mul_2d_f16, f16, rsqrt_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(rsqrt_mul_2d_f32, f32, rsqrt_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(rsqrt_mul_2d_f64, f64, rsqrt_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(cbrt_mul_f16, f16, cbrt_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(cbrt_mul_f32, f32, cbrt_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(cbrt_mul_f64, f64, cbrt_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(cbrt_mul_2d_f16, f16, cbrt_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(cbrt_mul_2d_f32, f32, cbrt_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(cbrt_mul_2d_f64, f64, cbrt_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(abs_mul_f16, f16, abs_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(abs_mul_f32, f32, abs_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(abs_mul_f64, f64, abs_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(abs_mul_2d_f16, f16, abs_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(abs_mul_2d_f32, f32, abs_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(abs_mul_2d_f64, f64, abs_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(neg_mul_f16, f16, neg_mul_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(neg_mul_f32, f32, neg_mul_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(neg_mul_f64, f64, neg_mul_f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(neg_mul_2d_f16, f16, neg_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(neg_mul_2d_f32, f32, neg_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(neg_mul_2d_f64, f64, neg_mul_2d_f64);

#[cfg(feature = "dtype-bf16")]
ternary_fused_fn!(add_silu_mul_bf16, bf16, add_silu_mul_bf16);
#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_silu_mul_f16, f16, add_silu_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_silu_mul_f32, f32, add_silu_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_silu_mul_f64, f64, add_silu_mul_f64);
#[cfg(feature = "dtype-bf16")]
ternary_fused_2d_fn!(add_silu_mul_2d_bf16, bf16, add_silu_mul_2d_bf16);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_silu_mul_2d_f16, f16, add_silu_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_silu_mul_2d_f32, f32, add_silu_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_silu_mul_2d_f64, f64, add_silu_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_add_f16, f16, bias_add_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_add_f32, f32, bias_add_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_add_f64, f64, bias_add_f64);

#[cfg(feature = "dtype-bf16")]
ternary_fused_fn!(add_gelu_mul_bf16, bf16, add_gelu_mul_bf16);
#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_gelu_mul_f16, f16, add_gelu_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_gelu_mul_f32, f32, add_gelu_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_gelu_mul_f64, f64, add_gelu_mul_f64);
#[cfg(feature = "dtype-bf16")]
ternary_fused_2d_fn!(add_gelu_mul_2d_bf16, bf16, add_gelu_mul_2d_bf16);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_gelu_mul_2d_f16, f16, add_gelu_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_gelu_mul_2d_f32, f32, add_gelu_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_gelu_mul_2d_f64, f64, add_gelu_mul_2d_f64);

#[cfg(feature = "dtype-bf16")]
ternary_fused_fn!(add_sigmoid_mul_bf16, bf16, add_sigmoid_mul_bf16);
#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_sigmoid_mul_f16, f16, add_sigmoid_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_sigmoid_mul_f32, f32, add_sigmoid_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_sigmoid_mul_f64, f64, add_sigmoid_mul_f64);
#[cfg(feature = "dtype-bf16")]
ternary_fused_2d_fn!(add_sigmoid_mul_2d_bf16, bf16, add_sigmoid_mul_2d_bf16);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_sigmoid_mul_2d_f16, f16, add_sigmoid_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_sigmoid_mul_2d_f32, f32, add_sigmoid_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_sigmoid_mul_2d_f64, f64, add_sigmoid_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_relu_mul_f16, f16, add_relu_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_relu_mul_f32, f32, add_relu_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_relu_mul_f64, f64, add_relu_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_relu_mul_2d_f16, f16, add_relu_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_relu_mul_2d_f32, f32, add_relu_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_relu_mul_2d_f64, f64, add_relu_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_relu6_mul_f16, f16, add_relu6_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_relu6_mul_f32, f32, add_relu6_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_relu6_mul_f64, f64, add_relu6_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_relu6_mul_2d_f16, f16, add_relu6_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_relu6_mul_2d_f32, f32, add_relu6_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_relu6_mul_2d_f64, f64, add_relu6_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_hard_sigmoid_mul_f16, f16, add_hard_sigmoid_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_hard_sigmoid_mul_f32, f32, add_hard_sigmoid_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_hard_sigmoid_mul_f64, f64, add_hard_sigmoid_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(
    add_hard_sigmoid_mul_2d_f16,
    f16,
    add_hard_sigmoid_mul_2d_f16
);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(
    add_hard_sigmoid_mul_2d_f32,
    f32,
    add_hard_sigmoid_mul_2d_f32
);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(
    add_hard_sigmoid_mul_2d_f64,
    f64,
    add_hard_sigmoid_mul_2d_f64
);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_hard_swish_mul_f16, f16, add_hard_swish_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_hard_swish_mul_f32, f32, add_hard_swish_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_hard_swish_mul_f64, f64, add_hard_swish_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_hard_swish_mul_2d_f16, f16, add_hard_swish_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_hard_swish_mul_2d_f32, f32, add_hard_swish_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_hard_swish_mul_2d_f64, f64, add_hard_swish_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_softsign_mul_f16, f16, add_softsign_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_softsign_mul_f32, f32, add_softsign_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_softsign_mul_f64, f64, add_softsign_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_softsign_mul_2d_f16, f16, add_softsign_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_softsign_mul_2d_f32, f32, add_softsign_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_softsign_mul_2d_f64, f64, add_softsign_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_mish_mul_f16, f16, add_mish_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_mish_mul_f32, f32, add_mish_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_mish_mul_f64, f64, add_mish_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_mish_mul_2d_f16, f16, add_mish_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_mish_mul_2d_f32, f32, add_mish_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_mish_mul_2d_f64, f64, add_mish_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_selu_mul_f16, f16, add_selu_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_selu_mul_f32, f32, add_selu_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_selu_mul_f64, f64, add_selu_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_selu_mul_2d_f16, f16, add_selu_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_selu_mul_2d_f32, f32, add_selu_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_selu_mul_2d_f64, f64, add_selu_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_tanhshrink_mul_f16, f16, add_tanhshrink_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_tanhshrink_mul_f32, f32, add_tanhshrink_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_tanhshrink_mul_f64, f64, add_tanhshrink_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_tanhshrink_mul_2d_f16, f16, add_tanhshrink_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_tanhshrink_mul_2d_f32, f32, add_tanhshrink_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_tanhshrink_mul_2d_f64, f64, add_tanhshrink_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_log_sigmoid_mul_f16, f16, add_log_sigmoid_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_log_sigmoid_mul_f32, f32, add_log_sigmoid_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_log_sigmoid_mul_f64, f64, add_log_sigmoid_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_log_sigmoid_mul_2d_f16, f16, add_log_sigmoid_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_log_sigmoid_mul_2d_f32, f32, add_log_sigmoid_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_log_sigmoid_mul_2d_f64, f64, add_log_sigmoid_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_reciprocal_mul_f16, f16, add_reciprocal_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_reciprocal_mul_f32, f32, add_reciprocal_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_reciprocal_mul_f64, f64, add_reciprocal_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_reciprocal_mul_2d_f16, f16, add_reciprocal_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_reciprocal_mul_2d_f32, f32, add_reciprocal_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_reciprocal_mul_2d_f64, f64, add_reciprocal_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_square_mul_f16, f16, add_square_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_square_mul_f32, f32, add_square_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_square_mul_f64, f64, add_square_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_square_mul_2d_f16, f16, add_square_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_square_mul_2d_f32, f32, add_square_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_square_mul_2d_f64, f64, add_square_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_ceil_mul_f16, f16, add_ceil_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_ceil_mul_f32, f32, add_ceil_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_ceil_mul_f64, f64, add_ceil_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_ceil_mul_2d_f16, f16, add_ceil_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_ceil_mul_2d_f32, f32, add_ceil_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_ceil_mul_2d_f64, f64, add_ceil_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_floor_mul_f16, f16, add_floor_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_floor_mul_f32, f32, add_floor_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_floor_mul_f64, f64, add_floor_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_floor_mul_2d_f16, f16, add_floor_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_floor_mul_2d_f32, f32, add_floor_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_floor_mul_2d_f64, f64, add_floor_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_trunc_mul_f16, f16, add_trunc_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_trunc_mul_f32, f32, add_trunc_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_trunc_mul_f64, f64, add_trunc_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_trunc_mul_2d_f16, f16, add_trunc_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_trunc_mul_2d_f32, f32, add_trunc_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_trunc_mul_2d_f64, f64, add_trunc_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_frac_mul_f16, f16, add_frac_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_frac_mul_f32, f32, add_frac_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_frac_mul_f64, f64, add_frac_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_frac_mul_2d_f16, f16, add_frac_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_frac_mul_2d_f32, f32, add_frac_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_frac_mul_2d_f64, f64, add_frac_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_deg2rad_mul_f16, f16, add_deg2rad_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_deg2rad_mul_f32, f32, add_deg2rad_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_deg2rad_mul_f64, f64, add_deg2rad_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_deg2rad_mul_2d_f16, f16, add_deg2rad_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_deg2rad_mul_2d_f32, f32, add_deg2rad_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_deg2rad_mul_2d_f64, f64, add_deg2rad_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_rad2deg_mul_f16, f16, add_rad2deg_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_rad2deg_mul_f32, f32, add_rad2deg_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_rad2deg_mul_f64, f64, add_rad2deg_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_rad2deg_mul_2d_f16, f16, add_rad2deg_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_rad2deg_mul_2d_f32, f32, add_rad2deg_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_rad2deg_mul_2d_f64, f64, add_rad2deg_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_sin_mul_f16, f16, add_sin_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_sin_mul_f32, f32, add_sin_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_sin_mul_f64, f64, add_sin_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_sin_mul_2d_f16, f16, add_sin_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_sin_mul_2d_f32, f32, add_sin_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_sin_mul_2d_f64, f64, add_sin_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_cos_mul_f16, f16, add_cos_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_cos_mul_f32, f32, add_cos_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_cos_mul_f64, f64, add_cos_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_cos_mul_2d_f16, f16, add_cos_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_cos_mul_2d_f32, f32, add_cos_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_cos_mul_2d_f64, f64, add_cos_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_tan_mul_f16, f16, add_tan_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_tan_mul_f32, f32, add_tan_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_tan_mul_f64, f64, add_tan_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_tan_mul_2d_f16, f16, add_tan_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_tan_mul_2d_f32, f32, add_tan_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_tan_mul_2d_f64, f64, add_tan_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_asin_mul_f16, f16, add_asin_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_asin_mul_f32, f32, add_asin_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_asin_mul_f64, f64, add_asin_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_asin_mul_2d_f16, f16, add_asin_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_asin_mul_2d_f32, f32, add_asin_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_asin_mul_2d_f64, f64, add_asin_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_acos_mul_f16, f16, add_acos_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_acos_mul_f32, f32, add_acos_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_acos_mul_f64, f64, add_acos_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_acos_mul_2d_f16, f16, add_acos_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_acos_mul_2d_f32, f32, add_acos_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_acos_mul_2d_f64, f64, add_acos_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_atan_mul_f16, f16, add_atan_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_atan_mul_f32, f32, add_atan_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_atan_mul_f64, f64, add_atan_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_atan_mul_2d_f16, f16, add_atan_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_atan_mul_2d_f32, f32, add_atan_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_atan_mul_2d_f64, f64, add_atan_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_sinh_mul_f16, f16, add_sinh_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_sinh_mul_f32, f32, add_sinh_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_sinh_mul_f64, f64, add_sinh_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_sinh_mul_2d_f16, f16, add_sinh_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_sinh_mul_2d_f32, f32, add_sinh_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_sinh_mul_2d_f64, f64, add_sinh_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_cosh_mul_f16, f16, add_cosh_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_cosh_mul_f32, f32, add_cosh_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_cosh_mul_f64, f64, add_cosh_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_cosh_mul_2d_f16, f16, add_cosh_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_cosh_mul_2d_f32, f32, add_cosh_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_cosh_mul_2d_f64, f64, add_cosh_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_asinh_mul_f16, f16, add_asinh_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_asinh_mul_f32, f32, add_asinh_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_asinh_mul_f64, f64, add_asinh_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_asinh_mul_2d_f16, f16, add_asinh_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_asinh_mul_2d_f32, f32, add_asinh_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_asinh_mul_2d_f64, f64, add_asinh_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_acosh_mul_f16, f16, add_acosh_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_acosh_mul_f32, f32, add_acosh_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_acosh_mul_f64, f64, add_acosh_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_acosh_mul_2d_f16, f16, add_acosh_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_acosh_mul_2d_f32, f32, add_acosh_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_acosh_mul_2d_f64, f64, add_acosh_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_atanh_mul_f16, f16, add_atanh_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_atanh_mul_f32, f32, add_atanh_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_atanh_mul_f64, f64, add_atanh_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_atanh_mul_2d_f16, f16, add_atanh_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_atanh_mul_2d_f32, f32, add_atanh_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_atanh_mul_2d_f64, f64, add_atanh_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_tanh_mul_f16, f16, add_tanh_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_tanh_mul_f32, f32, add_tanh_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_tanh_mul_f64, f64, add_tanh_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_tanh_mul_2d_f16, f16, add_tanh_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_tanh_mul_2d_f32, f32, add_tanh_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_tanh_mul_2d_f64, f64, add_tanh_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_exp_mul_f16, f16, add_exp_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_exp_mul_f32, f32, add_exp_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_exp_mul_f64, f64, add_exp_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_exp_mul_2d_f16, f16, add_exp_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_exp_mul_2d_f32, f32, add_exp_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_exp_mul_2d_f64, f64, add_exp_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_exp2_mul_f16, f16, add_exp2_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_exp2_mul_f32, f32, add_exp2_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_exp2_mul_f64, f64, add_exp2_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_exp2_mul_2d_f16, f16, add_exp2_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_exp2_mul_2d_f32, f32, add_exp2_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_exp2_mul_2d_f64, f64, add_exp2_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_expm1_mul_f16, f16, add_expm1_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_expm1_mul_f32, f32, add_expm1_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_expm1_mul_f64, f64, add_expm1_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_expm1_mul_2d_f16, f16, add_expm1_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_expm1_mul_2d_f32, f32, add_expm1_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_expm1_mul_2d_f64, f64, add_expm1_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_log_mul_f16, f16, add_log_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_log_mul_f32, f32, add_log_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_log_mul_f64, f64, add_log_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_log_mul_2d_f16, f16, add_log_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_log_mul_2d_f32, f32, add_log_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_log_mul_2d_f64, f64, add_log_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_log2_mul_f16, f16, add_log2_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_log2_mul_f32, f32, add_log2_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_log2_mul_f64, f64, add_log2_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_log2_mul_2d_f16, f16, add_log2_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_log2_mul_2d_f32, f32, add_log2_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_log2_mul_2d_f64, f64, add_log2_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_log10_mul_f16, f16, add_log10_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_log10_mul_f32, f32, add_log10_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_log10_mul_f64, f64, add_log10_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_log10_mul_2d_f16, f16, add_log10_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_log10_mul_2d_f32, f32, add_log10_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_log10_mul_2d_f64, f64, add_log10_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_log1p_mul_f16, f16, add_log1p_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_log1p_mul_f32, f32, add_log1p_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_log1p_mul_f64, f64, add_log1p_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_log1p_mul_2d_f16, f16, add_log1p_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_log1p_mul_2d_f32, f32, add_log1p_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_log1p_mul_2d_f64, f64, add_log1p_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_softplus_mul_f16, f16, add_softplus_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_softplus_mul_f32, f32, add_softplus_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_softplus_mul_f64, f64, add_softplus_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_softplus_mul_2d_f16, f16, add_softplus_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_softplus_mul_2d_f32, f32, add_softplus_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_softplus_mul_2d_f64, f64, add_softplus_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_sqrt_mul_f16, f16, add_sqrt_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_sqrt_mul_f32, f32, add_sqrt_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_sqrt_mul_f64, f64, add_sqrt_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_sqrt_mul_2d_f16, f16, add_sqrt_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_sqrt_mul_2d_f32, f32, add_sqrt_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_sqrt_mul_2d_f64, f64, add_sqrt_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_rsqrt_mul_f16, f16, add_rsqrt_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_rsqrt_mul_f32, f32, add_rsqrt_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_rsqrt_mul_f64, f64, add_rsqrt_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_rsqrt_mul_2d_f16, f16, add_rsqrt_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_rsqrt_mul_2d_f32, f32, add_rsqrt_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_rsqrt_mul_2d_f64, f64, add_rsqrt_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_cbrt_mul_f16, f16, add_cbrt_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_cbrt_mul_f32, f32, add_cbrt_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_cbrt_mul_f64, f64, add_cbrt_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_cbrt_mul_2d_f16, f16, add_cbrt_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_cbrt_mul_2d_f32, f32, add_cbrt_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_cbrt_mul_2d_f64, f64, add_cbrt_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_abs_mul_f16, f16, add_abs_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_abs_mul_f32, f32, add_abs_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_abs_mul_f64, f64, add_abs_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_abs_mul_2d_f16, f16, add_abs_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_abs_mul_2d_f32, f32, add_abs_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_abs_mul_2d_f64, f64, add_abs_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_neg_mul_f16, f16, add_neg_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_neg_mul_f32, f32, add_neg_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_neg_mul_f64, f64, add_neg_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_neg_mul_2d_f16, f16, add_neg_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_neg_mul_2d_f32, f32, add_neg_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_neg_mul_2d_f64, f64, add_neg_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(mul_add_f16, f16, mul_add_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(mul_add_f32, f32, mul_add_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(mul_add_f64, f64, mul_add_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(mul_add_2d_f16, f16, mul_add_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(mul_add_2d_f32, f32, mul_add_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(mul_add_2d_f64, f64, mul_add_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_mul_f16, f16, add_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_mul_f32, f32, add_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_mul_f64, f64, add_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_mul_2d_f16, f16, add_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_mul_2d_f32, f32, add_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_mul_2d_f64, f64, add_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(sub_mul_f16, f16, sub_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(sub_mul_f32, f32, sub_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(sub_mul_f64, f64, sub_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(sub_mul_2d_f16, f16, sub_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(sub_mul_2d_f32, f32, sub_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(sub_mul_2d_f64, f64, sub_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(rsub_mul_f16, f16, rsub_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(rsub_mul_f32, f32, rsub_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(rsub_mul_f64, f64, rsub_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(rsub_mul_2d_f16, f16, rsub_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(rsub_mul_2d_f32, f32, rsub_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(rsub_mul_2d_f64, f64, rsub_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(div_mul_f16, f16, div_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(div_mul_f32, f32, div_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(div_mul_f64, f64, div_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(div_mul_2d_f16, f16, div_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(div_mul_2d_f32, f32, div_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(div_mul_2d_f64, f64, div_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(rdiv_mul_f16, f16, rdiv_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(rdiv_mul_f32, f32, rdiv_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(rdiv_mul_f64, f64, rdiv_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(rdiv_mul_2d_f16, f16, rdiv_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(rdiv_mul_2d_f32, f32, rdiv_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(rdiv_mul_2d_f64, f64, rdiv_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(min_add_f16, f16, min_add_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(min_add_f32, f32, min_add_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(min_add_f64, f64, min_add_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(min_add_2d_f16, f16, min_add_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(min_add_2d_f32, f32, min_add_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(min_add_2d_f64, f64, min_add_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(max_add_f16, f16, max_add_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(max_add_f32, f32, max_add_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(max_add_f64, f64, max_add_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(max_add_2d_f16, f16, max_add_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(max_add_2d_f32, f32, max_add_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(max_add_2d_f64, f64, max_add_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(min_mul_f16, f16, min_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(min_mul_f32, f32, min_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(min_mul_f64, f64, min_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(min_mul_2d_f16, f16, min_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(min_mul_2d_f32, f32, min_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(min_mul_2d_f64, f64, min_mul_2d_f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(max_mul_f16, f16, max_mul_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(max_mul_f32, f32, max_mul_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(max_mul_f64, f64, max_mul_f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(max_mul_2d_f16, f16, max_mul_2d_f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(max_mul_2d_f32, f32, max_mul_2d_f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(max_mul_2d_f64, f64, max_mul_2d_f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_mul_add_f16, f16, where_mul_add_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_mul_add_f32, f32, where_mul_add_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_mul_add_f64, f64, where_mul_add_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_mul_add_2d_f16, f16, where_mul_add_2d_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_mul_add_2d_f32, f32, where_mul_add_2d_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_mul_add_2d_f64, f64, where_mul_add_2d_f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_add_f16, f16, where_add_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_add_f32, f32, where_add_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_add_f64, f64, where_add_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_add_2d_f16, f16, where_add_2d_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_add_2d_f32, f32, where_add_2d_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_add_2d_f64, f64, where_add_2d_f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_sub_f16, f16, where_sub_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_sub_f32, f32, where_sub_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_sub_f64, f64, where_sub_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_sub_2d_f16, f16, where_sub_2d_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_sub_2d_f32, f32, where_sub_2d_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_sub_2d_f64, f64, where_sub_2d_f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_mul_f16, f16, where_mul_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_mul_f32, f32, where_mul_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_mul_f64, f64, where_mul_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_mul_2d_f16, f16, where_mul_2d_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_mul_2d_f32, f32, where_mul_2d_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_mul_2d_f64, f64, where_mul_2d_f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_div_f16, f16, where_div_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_div_f32, f32, where_div_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_div_f64, f64, where_div_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_div_2d_f16, f16, where_div_2d_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_div_2d_f32, f32, where_div_2d_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_div_2d_f64, f64, where_div_2d_f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_min_f16, f16, where_min_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_min_f32, f32, where_min_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_min_f64, f64, where_min_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_min_2d_f16, f16, where_min_2d_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_min_2d_f32, f32, where_min_2d_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_min_2d_f64, f64, where_min_2d_f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_max_f16, f16, where_max_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_max_f32, f32, where_max_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_max_f64, f64, where_max_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_max_2d_f16, f16, where_max_2d_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_max_2d_f32, f32, where_max_2d_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_max_2d_f64, f64, where_max_2d_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_gelu_f16, f16, bias_gelu_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_gelu_f32, f32, bias_gelu_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_gelu_f64, f64, bias_gelu_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_sub_f16, f16, bias_sub_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_sub_f32, f32, bias_sub_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_sub_f64, f64, bias_sub_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_rsub_f16, f16, bias_rsub_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_rsub_f32, f32, bias_rsub_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_rsub_f64, f64, bias_rsub_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_mul_f16, f16, bias_mul_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_mul_f32, f32, bias_mul_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_mul_f64, f64, bias_mul_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_div_f16, f16, bias_div_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_div_f32, f32, bias_div_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_div_f64, f64, bias_div_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_rdiv_f16, f16, bias_rdiv_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_rdiv_f32, f32, bias_rdiv_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_rdiv_f64, f64, bias_rdiv_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_add_f16, f16, bias_scale_add_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_add_f32, f32, bias_scale_add_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_add_f64, f64, bias_scale_add_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_sub_f16, f16, bias_scale_sub_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_sub_f32, f32, bias_scale_sub_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_sub_f64, f64, bias_scale_sub_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_add_scale_f16, f16, bias_add_scale_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_add_scale_f32, f32, bias_add_scale_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_add_scale_f64, f64, bias_add_scale_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_sub_scale_f16, f16, bias_sub_scale_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_sub_scale_f32, f32, bias_sub_scale_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_sub_scale_f64, f64, bias_sub_scale_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_gelu_f16, f16, bias_scale_gelu_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_gelu_f32, f32, bias_scale_gelu_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_gelu_f64, f64, bias_scale_gelu_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_silu_f16, f16, bias_scale_silu_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_silu_f32, f32, bias_scale_silu_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_silu_f64, f64, bias_scale_silu_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_sigmoid_f16, f16, bias_scale_sigmoid_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_sigmoid_f32, f32, bias_scale_sigmoid_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_sigmoid_f64, f64, bias_scale_sigmoid_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_relu_f16, f16, bias_scale_relu_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_relu_f32, f32, bias_scale_relu_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_relu_f64, f64, bias_scale_relu_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_relu6_f16, f16, bias_scale_relu6_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_relu6_f32, f32, bias_scale_relu6_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_relu6_f64, f64, bias_scale_relu6_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(
    bias_scale_hard_sigmoid_f16,
    f16,
    bias_scale_hard_sigmoid_f16
);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(
    bias_scale_hard_sigmoid_f32,
    f32,
    bias_scale_hard_sigmoid_f32
);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(
    bias_scale_hard_sigmoid_f64,
    f64,
    bias_scale_hard_sigmoid_f64
);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_hard_swish_f16, f16, bias_scale_hard_swish_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_hard_swish_f32, f32, bias_scale_hard_swish_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_hard_swish_f64, f64, bias_scale_hard_swish_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_softsign_f16, f16, bias_scale_softsign_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_softsign_f32, f32, bias_scale_softsign_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_softsign_f64, f64, bias_scale_softsign_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_mish_f16, f16, bias_scale_mish_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_mish_f32, f32, bias_scale_mish_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_mish_f64, f64, bias_scale_mish_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_selu_f16, f16, bias_scale_selu_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_selu_f32, f32, bias_scale_selu_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_selu_f64, f64, bias_scale_selu_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_tanhshrink_f16, f16, bias_scale_tanhshrink_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_tanhshrink_f32, f32, bias_scale_tanhshrink_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_tanhshrink_f64, f64, bias_scale_tanhshrink_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_log_sigmoid_f16, f16, bias_scale_log_sigmoid_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_log_sigmoid_f32, f32, bias_scale_log_sigmoid_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_log_sigmoid_f64, f64, bias_scale_log_sigmoid_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_reciprocal_f16, f16, bias_scale_reciprocal_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_reciprocal_f32, f32, bias_scale_reciprocal_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_reciprocal_f64, f64, bias_scale_reciprocal_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_square_f16, f16, bias_scale_square_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_square_f32, f32, bias_scale_square_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_square_f64, f64, bias_scale_square_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_ceil_f16, f16, bias_scale_ceil_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_ceil_f32, f32, bias_scale_ceil_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_ceil_f64, f64, bias_scale_ceil_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_floor_f16, f16, bias_scale_floor_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_floor_f32, f32, bias_scale_floor_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_floor_f64, f64, bias_scale_floor_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_trunc_f16, f16, bias_scale_trunc_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_trunc_f32, f32, bias_scale_trunc_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_trunc_f64, f64, bias_scale_trunc_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_frac_f16, f16, bias_scale_frac_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_frac_f32, f32, bias_scale_frac_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_frac_f64, f64, bias_scale_frac_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_deg2rad_f16, f16, bias_scale_deg2rad_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_deg2rad_f32, f32, bias_scale_deg2rad_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_deg2rad_f64, f64, bias_scale_deg2rad_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_rad2deg_f16, f16, bias_scale_rad2deg_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_rad2deg_f32, f32, bias_scale_rad2deg_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_rad2deg_f64, f64, bias_scale_rad2deg_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_sin_f16, f16, bias_scale_sin_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_sin_f32, f32, bias_scale_sin_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_sin_f64, f64, bias_scale_sin_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_cos_f16, f16, bias_scale_cos_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_cos_f32, f32, bias_scale_cos_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_cos_f64, f64, bias_scale_cos_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_tan_f16, f16, bias_scale_tan_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_tan_f32, f32, bias_scale_tan_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_tan_f64, f64, bias_scale_tan_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_asin_f16, f16, bias_scale_asin_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_asin_f32, f32, bias_scale_asin_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_asin_f64, f64, bias_scale_asin_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_acos_f16, f16, bias_scale_acos_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_acos_f32, f32, bias_scale_acos_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_acos_f64, f64, bias_scale_acos_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_atan_f16, f16, bias_scale_atan_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_atan_f32, f32, bias_scale_atan_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_atan_f64, f64, bias_scale_atan_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_sinh_f16, f16, bias_scale_sinh_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_sinh_f32, f32, bias_scale_sinh_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_sinh_f64, f64, bias_scale_sinh_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_cosh_f16, f16, bias_scale_cosh_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_cosh_f32, f32, bias_scale_cosh_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_cosh_f64, f64, bias_scale_cosh_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_asinh_f16, f16, bias_scale_asinh_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_asinh_f32, f32, bias_scale_asinh_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_asinh_f64, f64, bias_scale_asinh_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_acosh_f16, f16, bias_scale_acosh_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_acosh_f32, f32, bias_scale_acosh_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_acosh_f64, f64, bias_scale_acosh_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_atanh_f16, f16, bias_scale_atanh_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_atanh_f32, f32, bias_scale_atanh_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_atanh_f64, f64, bias_scale_atanh_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_tanh_f16, f16, bias_scale_tanh_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_tanh_f32, f32, bias_scale_tanh_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_tanh_f64, f64, bias_scale_tanh_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_exp_f16, f16, bias_scale_exp_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_exp_f32, f32, bias_scale_exp_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_exp_f64, f64, bias_scale_exp_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_exp2_f16, f16, bias_scale_exp2_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_exp2_f32, f32, bias_scale_exp2_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_exp2_f64, f64, bias_scale_exp2_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_expm1_f16, f16, bias_scale_expm1_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_expm1_f32, f32, bias_scale_expm1_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_expm1_f64, f64, bias_scale_expm1_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_log_f16, f16, bias_scale_log_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_log_f32, f32, bias_scale_log_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_log_f64, f64, bias_scale_log_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_log2_f16, f16, bias_scale_log2_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_log2_f32, f32, bias_scale_log2_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_log2_f64, f64, bias_scale_log2_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_log10_f16, f16, bias_scale_log10_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_log10_f32, f32, bias_scale_log10_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_log10_f64, f64, bias_scale_log10_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_log1p_f16, f16, bias_scale_log1p_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_log1p_f32, f32, bias_scale_log1p_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_log1p_f64, f64, bias_scale_log1p_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_softplus_f16, f16, bias_scale_softplus_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_softplus_f32, f32, bias_scale_softplus_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_softplus_f64, f64, bias_scale_softplus_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_sqrt_f16, f16, bias_scale_sqrt_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_sqrt_f32, f32, bias_scale_sqrt_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_sqrt_f64, f64, bias_scale_sqrt_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_rsqrt_f16, f16, bias_scale_rsqrt_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_rsqrt_f32, f32, bias_scale_rsqrt_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_rsqrt_f64, f64, bias_scale_rsqrt_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_cbrt_f16, f16, bias_scale_cbrt_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_cbrt_f32, f32, bias_scale_cbrt_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_cbrt_f64, f64, bias_scale_cbrt_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_abs_f16, f16, bias_scale_abs_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_abs_f32, f32, bias_scale_abs_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_abs_f64, f64, bias_scale_abs_f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_neg_f16, f16, bias_scale_neg_f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_neg_f32, f32, bias_scale_neg_f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_neg_f64, f64, bias_scale_neg_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_silu_f16, f16, bias_silu_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_silu_f32, f32, bias_silu_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_silu_f64, f64, bias_silu_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_sigmoid_f16, f16, bias_sigmoid_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_sigmoid_f32, f32, bias_sigmoid_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_sigmoid_f64, f64, bias_sigmoid_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_relu_f16, f16, bias_relu_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_relu_f32, f32, bias_relu_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_relu_f64, f64, bias_relu_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_relu6_f16, f16, bias_relu6_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_relu6_f32, f32, bias_relu6_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_relu6_f64, f64, bias_relu6_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_hard_sigmoid_f16, f16, bias_hard_sigmoid_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_hard_sigmoid_f32, f32, bias_hard_sigmoid_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_hard_sigmoid_f64, f64, bias_hard_sigmoid_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_hard_swish_f16, f16, bias_hard_swish_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_hard_swish_f32, f32, bias_hard_swish_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_hard_swish_f64, f64, bias_hard_swish_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_softsign_f16, f16, bias_softsign_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_softsign_f32, f32, bias_softsign_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_softsign_f64, f64, bias_softsign_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_mish_f16, f16, bias_mish_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_mish_f32, f32, bias_mish_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_mish_f64, f64, bias_mish_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_selu_f16, f16, bias_selu_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_selu_f32, f32, bias_selu_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_selu_f64, f64, bias_selu_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_tanhshrink_f16, f16, bias_tanhshrink_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_tanhshrink_f32, f32, bias_tanhshrink_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_tanhshrink_f64, f64, bias_tanhshrink_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_log_sigmoid_f16, f16, bias_log_sigmoid_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_log_sigmoid_f32, f32, bias_log_sigmoid_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_log_sigmoid_f64, f64, bias_log_sigmoid_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_reciprocal_f16, f16, bias_reciprocal_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_reciprocal_f32, f32, bias_reciprocal_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_reciprocal_f64, f64, bias_reciprocal_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_square_f16, f16, bias_square_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_square_f32, f32, bias_square_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_square_f64, f64, bias_square_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_ceil_f16, f16, bias_ceil_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_ceil_f32, f32, bias_ceil_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_ceil_f64, f64, bias_ceil_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_floor_f16, f16, bias_floor_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_floor_f32, f32, bias_floor_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_floor_f64, f64, bias_floor_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_trunc_f16, f16, bias_trunc_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_trunc_f32, f32, bias_trunc_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_trunc_f64, f64, bias_trunc_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_frac_f16, f16, bias_frac_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_frac_f32, f32, bias_frac_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_frac_f64, f64, bias_frac_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_deg2rad_f16, f16, bias_deg2rad_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_deg2rad_f32, f32, bias_deg2rad_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_deg2rad_f64, f64, bias_deg2rad_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_rad2deg_f16, f16, bias_rad2deg_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_rad2deg_f32, f32, bias_rad2deg_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_rad2deg_f64, f64, bias_rad2deg_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_sin_f16, f16, bias_sin_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_sin_f32, f32, bias_sin_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_sin_f64, f64, bias_sin_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_cos_f16, f16, bias_cos_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_cos_f32, f32, bias_cos_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_cos_f64, f64, bias_cos_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_tan_f16, f16, bias_tan_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_tan_f32, f32, bias_tan_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_tan_f64, f64, bias_tan_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_asin_f16, f16, bias_asin_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_asin_f32, f32, bias_asin_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_asin_f64, f64, bias_asin_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_acos_f16, f16, bias_acos_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_acos_f32, f32, bias_acos_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_acos_f64, f64, bias_acos_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_atan_f16, f16, bias_atan_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_atan_f32, f32, bias_atan_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_atan_f64, f64, bias_atan_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_sinh_f16, f16, bias_sinh_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_sinh_f32, f32, bias_sinh_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_sinh_f64, f64, bias_sinh_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_cosh_f16, f16, bias_cosh_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_cosh_f32, f32, bias_cosh_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_cosh_f64, f64, bias_cosh_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_asinh_f16, f16, bias_asinh_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_asinh_f32, f32, bias_asinh_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_asinh_f64, f64, bias_asinh_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_acosh_f16, f16, bias_acosh_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_acosh_f32, f32, bias_acosh_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_acosh_f64, f64, bias_acosh_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_atanh_f16, f16, bias_atanh_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_atanh_f32, f32, bias_atanh_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_atanh_f64, f64, bias_atanh_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_tanh_f16, f16, bias_tanh_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_tanh_f32, f32, bias_tanh_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_tanh_f64, f64, bias_tanh_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_exp_f16, f16, bias_exp_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_exp_f32, f32, bias_exp_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_exp_f64, f64, bias_exp_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_exp2_f16, f16, bias_exp2_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_exp2_f32, f32, bias_exp2_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_exp2_f64, f64, bias_exp2_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_expm1_f16, f16, bias_expm1_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_expm1_f32, f32, bias_expm1_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_expm1_f64, f64, bias_expm1_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_log_f16, f16, bias_log_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_log_f32, f32, bias_log_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_log_f64, f64, bias_log_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_log2_f16, f16, bias_log2_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_log2_f32, f32, bias_log2_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_log2_f64, f64, bias_log2_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_log10_f16, f16, bias_log10_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_log10_f32, f32, bias_log10_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_log10_f64, f64, bias_log10_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_log1p_f16, f16, bias_log1p_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_log1p_f32, f32, bias_log1p_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_log1p_f64, f64, bias_log1p_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_softplus_f16, f16, bias_softplus_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_softplus_f32, f32, bias_softplus_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_softplus_f64, f64, bias_softplus_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_sqrt_f16, f16, bias_sqrt_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_sqrt_f32, f32, bias_sqrt_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_sqrt_f64, f64, bias_sqrt_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_rsqrt_f16, f16, bias_rsqrt_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_rsqrt_f32, f32, bias_rsqrt_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_rsqrt_f64, f64, bias_rsqrt_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_cbrt_f16, f16, bias_cbrt_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_cbrt_f32, f32, bias_cbrt_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_cbrt_f64, f64, bias_cbrt_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_abs_f16, f16, bias_abs_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_abs_f32, f32, bias_abs_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_abs_f64, f64, bias_abs_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_neg_f16, f16, bias_neg_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_neg_f32, f32, bias_neg_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_neg_f64, f64, bias_neg_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_min_f16, f16, bias_min_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_min_f32, f32, bias_min_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_min_f64, f64, bias_min_f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_max_f16, f16, bias_max_f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_max_f32, f32, bias_max_f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_max_f64, f64, bias_max_f64);

#[cfg(feature = "dtype-f16")]
column2_fused_fn!(column_clamp_f16, f16, column_clamp_f16);
#[cfg(feature = "dtype-f32")]
column2_fused_fn!(column_clamp_f32, f32, column_clamp_f32);
#[cfg(feature = "dtype-f64")]
column2_fused_fn!(column_clamp_f64, f64, column_clamp_f64);

#[cfg(feature = "dtype-f16")]
column2_fused_fn!(column_lerp_f16, f16, column_lerp_f16);
#[cfg(feature = "dtype-f32")]
column2_fused_fn!(column_lerp_f32, f32, column_lerp_f32);
#[cfg(feature = "dtype-f64")]
column2_fused_fn!(column_lerp_f64, f64, column_lerp_f64);

#[cfg(feature = "dtype-f16")]
rms_norm_add_fn!(rms_norm_add_f16, f16, f32, rms_norm_add_f16);
#[cfg(feature = "dtype-bf16")]
rms_norm_add_fn!(rms_norm_add_bf16, bf16, f32, rms_norm_add_bf16);
#[cfg(feature = "dtype-f32")]
rms_norm_add_fn!(rms_norm_add_f32, f32, f32, rms_norm_add_f32);
#[cfg(feature = "dtype-f64")]
rms_norm_add_fn!(rms_norm_add_f64, f64, f64, rms_norm_add_f64);

#[cfg(feature = "dtype-f16")]
rms_norm_add_wide_fn!(
    rms_norm_add_wide_f16,
    f16,
    f32,
    rms_norm_add_wide_partial_f16,
    rms_norm_add_wide_normalize_f16
);
#[cfg(feature = "dtype-bf16")]
rms_norm_add_wide_fn!(
    rms_norm_add_wide_bf16,
    bf16,
    f32,
    rms_norm_add_wide_partial_bf16,
    rms_norm_add_wide_normalize_bf16
);
#[cfg(feature = "dtype-f32")]
rms_norm_add_wide_fn!(
    rms_norm_add_wide_f32,
    f32,
    f32,
    rms_norm_add_wide_partial_f32,
    rms_norm_add_wide_normalize_f32
);

#[cfg(feature = "dtype-f16")]
rms_norm_silu_mul_fn!(rms_norm_silu_mul_f16, f16, f32, rms_norm_silu_mul_f16);
#[cfg(feature = "dtype-bf16")]
rms_norm_silu_mul_fn!(rms_norm_silu_mul_bf16, bf16, f32, rms_norm_silu_mul_bf16);
#[cfg(feature = "dtype-f32")]
rms_norm_silu_mul_fn!(rms_norm_silu_mul_f32, f32, f32, rms_norm_silu_mul_f32);
#[cfg(feature = "dtype-f64")]
rms_norm_silu_mul_fn!(rms_norm_silu_mul_f64, f64, f64, rms_norm_silu_mul_f64);

#[cfg(feature = "dtype-f32")]
rms_norm_gated_silu_fn!(
    rms_norm_gated_silu_f32,
    f32,
    f32,
    rms_norm_gated_silu_f32_exact
);
#[cfg(feature = "dtype-f16")]
rms_norm_gated_silu_fn!(
    rms_norm_gated_silu_f16,
    f16,
    f32,
    rms_norm_gated_silu_f16_exact
);
#[cfg(feature = "dtype-bf16")]
rms_norm_gated_silu_fn!(
    rms_norm_gated_silu_bf16,
    bf16,
    f32,
    rms_norm_gated_silu_bf16_exact
);

#[cfg(feature = "dtype-f32")]
dual_rms_norm_fn!(dual_rms_norm_f32, f32, f32, dual_rms_norm_f32_exact);
#[cfg(feature = "dtype-f16")]
dual_rms_norm_fn!(dual_rms_norm_f16, f16, f32, dual_rms_norm_f16_exact);
#[cfg(feature = "dtype-bf16")]
dual_rms_norm_fn!(dual_rms_norm_bf16, bf16, f32, dual_rms_norm_bf16_exact);

#[cfg(feature = "dtype-f32")]
rms_norm_residual_add_fn!(
    rms_norm_residual_add_f32,
    f32,
    f32,
    rms_norm_residual_add_f32_exact
);
#[cfg(feature = "dtype-f16")]
rms_norm_residual_add_fn!(
    rms_norm_residual_add_f16,
    f16,
    f32,
    rms_norm_residual_add_f16_exact
);
#[cfg(feature = "dtype-bf16")]
rms_norm_residual_add_fn!(
    rms_norm_residual_add_bf16,
    bf16,
    f32,
    rms_norm_residual_add_bf16_exact
);

#[cfg(feature = "dtype-f16")]
kv_cache_update_fn!(kv_cache_update_f16, f16, kv_cache_update_f16);

#[cfg(feature = "dtype-f16")]

pub fn bshd_rope_qk_cache_update_f16(
    stream: &Arc<Stream>,
    q_out: DevicePointer<f16>,
    k_out: DevicePointer<f16>,
    k_cache: DevicePointer<f16>,
    v_cache: DevicePointer<f16>,
    q_input: DevicePointer<f16>,
    k_input: DevicePointer<f16>,
    v_input: DevicePointer<f16>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    seq_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    position_start: usize,
) -> Result<()> {
    if seq_len == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || max_seq == 0 {
        return Err(Error::InvalidLength);
    }
    if head_dim % 2 != 0 {
        return Err(Error::InvalidLength);
    }
    let q_len = checked_element_count(checked_element_count(seq_len, q_heads)?, head_dim)?;
    let kv_len = checked_element_count(checked_element_count(seq_len, kv_heads)?, head_dim)?;
    let total_len = q_len.max(kv_len);
    let position_end = position_start
        .checked_add(seq_len)
        .ok_or(Error::SizeOverflow)?;
    if position_end > max_seq {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(q_out)?;
    checked_device_pointer(k_out)?;
    checked_device_pointer(k_cache)?;
    checked_device_pointer(v_cache)?;
    checked_device_pointer(q_input)?;
    checked_device_pointer(k_input)?;
    checked_device_pointer(v_input)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    unsafe {
        kernel_fused::bshd_rope_qk_cache_update_f16(
            q_out,
            k_out,
            k_cache,
            v_cache,
            q_input,
            k_input,
            v_input,
            cos,
            sin,
            checked_i32_value(seq_len)?,
            checked_i32_value(q_heads)?,
            checked_i32_value(kv_heads)?,
            checked_i32_value(head_dim)?,
            checked_i32_value(max_seq)?,
            checked_i32_value(position_start)?,
            checked_i32_value(total_len)?,
        )
    }
    .grid((
        u32::try_from(total_len.div_ceil(128)).map_err(|_| Error::SizeOverflow)?,
        1,
        1,
    ))
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
kv_cache_update_fn!(kv_cache_update_f32, f32, kv_cache_update_f32);
#[cfg(feature = "dtype-f64")]
kv_cache_update_fn!(kv_cache_update_f64, f64, kv_cache_update_f64);
#[cfg(feature = "dtype-f16")]
kv_cache_update_fn!(kv_cache_update_seq_f16, f16, kv_cache_update_f16);
#[cfg(feature = "dtype-f32")]
kv_cache_update_fn!(kv_cache_update_seq_f32, f32, kv_cache_update_f32);
#[cfg(feature = "dtype-f64")]
kv_cache_update_fn!(kv_cache_update_seq_f64, f64, kv_cache_update_f64);

macro_rules! kv_cache_compact_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            k_cache: DevicePointer<$ty>,
            v_cache: DevicePointer<$ty>,
            heads: usize,
            head_dim: usize,
            max_seq: usize,
            source_start: usize,
            token_count: usize,
        ) -> Result<()> {
            if heads == 0 || head_dim == 0 || max_seq == 0 || token_count == 0 {
                return Err(Error::InvalidLength);
            }
            let source_end = source_start
                .checked_add(token_count)
                .ok_or(Error::SizeOverflow)?;
            if source_end > max_seq {
                return Err(Error::InvalidLength);
            }
            checked_device_pointer(k_cache)?;
            checked_device_pointer(v_cache)?;
            let len = checked_element_count(checked_element_count(heads, token_count)?, head_dim)?;
            let heads = checked_i32_value(heads)?;
            let head_dim = checked_i32_value(head_dim)?;
            let max_seq = checked_i32_value(max_seq)?;
            let source_start = checked_i32_value(source_start)?;
            let token_count = checked_i32_value(token_count)?;
            let len_i32 = checked_i32_value(len)?;
            let grid = raw_vector_grid(len)?;
            unsafe {
                kernel_fused::$kernel_fn(
                    k_cache,
                    v_cache,
                    heads,
                    head_dim,
                    max_seq,
                    source_start,
                    token_count,
                    len_i32,
                )
            }
            .grid(grid)
            .enqueue_on(stream)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
kv_cache_compact_fn!(kv_cache_compact_f16, f16, kv_cache_compact_f16);
#[cfg(feature = "dtype-f32")]
kv_cache_compact_fn!(kv_cache_compact_f32, f32, kv_cache_compact_f32);
#[cfg(feature = "dtype-f64")]
kv_cache_compact_fn!(kv_cache_compact_f64, f64, kv_cache_compact_f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_dynpos_f16, f16, kv_cache_update_dynpos_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_dynpos_f32, f32, kv_cache_update_dynpos_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_dynpos_f64, f64, kv_cache_update_dynpos_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u32"))]
kv_cache_update_dynpos_fn!(
    kv_cache_update_seq_dynpos_f16,
    f16,
    kv_cache_update_dynpos_f16
);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u32"))]
kv_cache_update_dynpos_fn!(
    kv_cache_update_seq_dynpos_f32,
    f32,
    kv_cache_update_dynpos_f32
);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u32"))]
kv_cache_update_dynpos_fn!(
    kv_cache_update_seq_dynpos_f64,
    f64,
    kv_cache_update_dynpos_f64
);

macro_rules! rope_qk_fn {
    ($name:ident, $target:ident, $ty:ty) => {
        pub fn $name(
            stream: &Arc<Stream>,
            q_out: DevicePointer<$ty>,
            k_out: DevicePointer<$ty>,
            q_input: DevicePointer<$ty>,
            k_input: DevicePointer<$ty>,
            cos: DevicePointer<$ty>,
            sin: DevicePointer<$ty>,
            config: QkRotaryEmbeddingConfig,
        ) -> Result<()> {
            positional::$target(stream, q_out, k_out, q_input, k_input, cos, sin, config)
        }
    };
}

#[cfg(feature = "dtype-f16")]
rope_qk_fn!(rope_qk_f16, rotary_embedding_qk_f16, f16);
#[cfg(feature = "dtype-f32")]
rope_qk_fn!(rope_qk_f32, rotary_embedding_qk_f32, f32);
#[cfg(feature = "dtype-f64")]
rope_qk_fn!(rope_qk_f64, rotary_embedding_qk_f64, f64);

macro_rules! rope_qk_cache_update_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            q_out: DevicePointer<$ty>,
            k_out: DevicePointer<$ty>,
            k_cache: DevicePointer<$ty>,
            q_input: DevicePointer<$ty>,
            k_input: DevicePointer<$ty>,
            cos: DevicePointer<$ty>,
            sin: DevicePointer<$ty>,
            config: RopeQkCacheUpdateConfig,
        ) -> Result<()> {
            let rotary = config.rotary;
            if rotary.q_output_dimensions[0] != rotary.k_output_dimensions[0]
                || rotary.q_output_dimensions[2] != rotary.k_output_dimensions[2]
                || rotary.q_output_dimensions[3] != rotary.k_output_dimensions[3]
            {
                return Err(Error::LengthMismatch);
            }
            if rotary
                .rotary_pairs
                .checked_mul(2)
                .ok_or(Error::SizeOverflow)?
                > rotary.q_output_dimensions[3]
            {
                return Err(Error::LengthMismatch);
            }
            let cache_end = config
                .cache_position_start
                .checked_add(rotary.k_output_dimensions[2])
                .ok_or(Error::SizeOverflow)?;
            if cache_end > config.cache_max_seq {
                return Err(Error::LengthMismatch);
            }
            let q_len = checked_rank4_len(rotary.q_output_dimensions)?;
            let k_len = checked_rank4_len(rotary.k_output_dimensions)?;
            let len = q_len.max(k_len);
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            checked_device_pointer(q_out)?;
            checked_device_pointer(k_out)?;
            checked_device_pointer(k_cache)?;
            checked_device_pointer(q_input)?;
            checked_device_pointer(k_input)?;
            if rotary.rotary_pairs > 0 {
                checked_device_pointer(cos)?;
                checked_device_pointer(sin)?;
            }

            let q_output_dimensions = [
                checked_i32_value(rotary.q_output_dimensions[0])?,
                checked_i32_value(rotary.q_output_dimensions[1])?,
                checked_i32_value(rotary.q_output_dimensions[2])?,
                checked_i32_value(rotary.q_output_dimensions[3])?,
            ];
            let k_output_dimensions = [
                checked_i32_value(rotary.k_output_dimensions[0])?,
                checked_i32_value(rotary.k_output_dimensions[1])?,
                checked_i32_value(rotary.k_output_dimensions[2])?,
                checked_i32_value(rotary.k_output_dimensions[3])?,
            ];
            let q_input_strides = [
                checked_i32_value(rotary.q_input_strides[0])?,
                checked_i32_value(rotary.q_input_strides[1])?,
                checked_i32_value(rotary.q_input_strides[2])?,
                checked_i32_value(rotary.q_input_strides[3])?,
            ];
            let k_input_strides = [
                checked_i32_value(rotary.k_input_strides[0])?,
                checked_i32_value(rotary.k_input_strides[1])?,
                checked_i32_value(rotary.k_input_strides[2])?,
                checked_i32_value(rotary.k_input_strides[3])?,
            ];
            let cos_strides = [
                checked_i32_value(rotary.cos_strides[0])?,
                checked_i32_value(rotary.cos_strides[1])?,
            ];
            let sin_strides = [
                checked_i32_value(rotary.sin_strides[0])?,
                checked_i32_value(rotary.sin_strides[1])?,
            ];
            let rotary_pairs = checked_i32_value(rotary.rotary_pairs)?;
            let cache_max_seq = checked_i32_value(config.cache_max_seq)?;
            let cache_position_start = checked_i32_value(config.cache_position_start)?;
            let q_len = checked_i32_value(q_len)?;
            let k_len = checked_i32_value(k_len)?;
            let grid = raw_vector_grid(len)?;
            unsafe {
                kernel_fused::$kernel_fn(
                    q_out,
                    k_out,
                    k_cache,
                    q_input,
                    k_input,
                    cos,
                    sin,
                    q_output_dimensions[0],
                    q_output_dimensions[1],
                    q_output_dimensions[2],
                    q_output_dimensions[3],
                    q_input_strides[0],
                    q_input_strides[1],
                    q_input_strides[2],
                    q_input_strides[3],
                    k_output_dimensions[0],
                    k_output_dimensions[1],
                    k_output_dimensions[2],
                    k_output_dimensions[3],
                    k_input_strides[0],
                    k_input_strides[1],
                    k_input_strides[2],
                    k_input_strides[3],
                    cos_strides[0],
                    cos_strides[1],
                    sin_strides[0],
                    sin_strides[1],
                    rotary_pairs,
                    cache_max_seq,
                    cache_position_start,
                    q_len,
                    k_len,
                )
            }
            .grid(grid)
            .enqueue_on(stream)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
rope_qk_cache_update_fn!(rope_qk_cache_update_f16, f16, rope_qk_cache_update_f16);
#[cfg(feature = "dtype-f32")]
rope_qk_cache_update_fn!(rope_qk_cache_update_f32, f32, rope_qk_cache_update_f32);
#[cfg(feature = "dtype-f64")]
rope_qk_cache_update_fn!(rope_qk_cache_update_f64, f64, rope_qk_cache_update_f64);
