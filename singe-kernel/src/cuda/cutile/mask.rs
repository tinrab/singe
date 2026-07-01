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
        kernel::common as kernel_common,
        utility::{VectorLaunch, checked_device_pointer},
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

const MASK_REDUCTION_MAX_TILE_WIDTH: usize = 1024;

fn mask_reduction_tile_width(cols: usize) -> Result<usize> {
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
            op: "mask reduction".into(),
            width: cols,
        }),
    }
}

macro_rules! unary_mask_fn {
    ($name:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u8>,
            input: DevicePointer<u8>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok::<(), Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { kernel_common::$kernel_fn(out, input, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok::<(), Error>(())
        }
    };
}

macro_rules! binary_mask_fn {
    ($name:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u8>,
            lhs: DevicePointer<u8>,
            rhs: DevicePointer<u8>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok::<(), Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(lhs)?;
            checked_device_pointer(rhs)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { kernel_common::$kernel_fn(out, lhs, rhs, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok::<(), Error>(())
        }
    };
}

unary_mask_fn!(mask_not_u8, mask_not_u8);
binary_mask_fn!(mask_and_u8, mask_and_u8);
binary_mask_fn!(mask_or_u8, mask_or_u8);
binary_mask_fn!(mask_xor_u8, mask_xor_u8);

pub fn logical_not_u8(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    input: DevicePointer<u8>,
    len: usize,
) -> Result<()> {
    mask_not_u8(stream, out, input, len)
}

pub fn logical_and_u8(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    lhs: DevicePointer<u8>,
    rhs: DevicePointer<u8>,
    len: usize,
) -> Result<()> {
    mask_and_u8(stream, out, lhs, rhs, len)
}

pub fn logical_or_u8(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    lhs: DevicePointer<u8>,
    rhs: DevicePointer<u8>,
    len: usize,
) -> Result<()> {
    mask_or_u8(stream, out, lhs, rhs, len)
}

pub fn logical_xor_u8(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    lhs: DevicePointer<u8>,
    rhs: DevicePointer<u8>,
    len: usize,
) -> Result<()> {
    mask_xor_u8(stream, out, lhs, rhs, len)
}

macro_rules! triangular_mask_fn {
    ($name:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u8>,
            rows: usize,
            cols: usize,
            diagonal: isize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), Error>(());
            }
            checked_device_pointer(out)?;
            let rows_i32 = checked_i32_value(rows)?;
            let cols_i32 = checked_i32_value(cols)?;
            let diagonal_i32 = i32::try_from(diagonal).map_err(|_| Error::LengthExceedsI32)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                kernel_common::$kernel_fn(out, rows_i32, cols_i32, diagonal_i32, launch.len_i32)
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok::<(), Error>(())
        }
    };
}

triangular_mask_fn!(tril_mask_u8, tril_mask_u8);
triangular_mask_fn!(triu_mask_u8, triu_mask_u8);

macro_rules! causal_mask_fill_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            batch: usize,
            seq_len: usize,
            fill_value: $ty,
        ) -> Result<()> {
            let len = causal_mask_len(batch, seq_len)?;
            if len == 0 {
                return Ok::<(), Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let seq_len_i32 = checked_i32_value(seq_len)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                kernel_common::$kernel_fn(out, input, fill_value, seq_len_i32, launch.len_i32)
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok::<(), Error>(())
        }
    };
}

macro_rules! causal_mask_zero_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            batch: usize,
            seq_len: usize,
        ) -> Result<()> {
            let len = causal_mask_len(batch, seq_len)?;
            if len == 0 {
                return Ok::<(), Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let seq_len_i32 = checked_i32_value(seq_len)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { kernel_common::$kernel_fn(out, input, seq_len_i32, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok::<(), Error>(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
causal_mask_fill_fn!(causal_mask_fill_f32, f32, causal_mask_fill_f32);
#[cfg(feature = "dtype-f32")]
causal_mask_zero_fn!(causal_mask_zero_f32, f32, causal_mask_zero_f32);
#[cfg(feature = "dtype-f16")]
causal_mask_fill_fn!(causal_mask_fill_f16, f16, causal_mask_fill_f16);
#[cfg(feature = "dtype-f16")]
causal_mask_zero_fn!(causal_mask_zero_f16, f16, causal_mask_zero_f16);
#[cfg(feature = "dtype-bf16")]
causal_mask_fill_fn!(causal_mask_fill_bf16, bf16, causal_mask_fill_bf16);
#[cfg(feature = "dtype-bf16")]
causal_mask_zero_fn!(causal_mask_zero_bf16, bf16, causal_mask_zero_bf16);

pub fn lower_triangular_mask_u8(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    tril_mask_u8(stream, out, rows, cols, diagonal)
}

pub fn upper_triangular_mask_u8(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    triu_mask_u8(stream, out, rows, cols, diagonal)
}

pub fn causal_mask_u8(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    tril_mask_u8(stream, out, rows, cols, 0)
}

pub fn sequence_mask_u8(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    lengths: DevicePointer<u32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let len = checked_element_count(rows, cols)?;
    if len == 0 {
        return Ok::<(), Error>(());
    }
    checked_device_pointer(out)?;
    checked_device_pointer(lengths)?;
    let cols_i32 = checked_i32_value(cols)?;
    let launch = VectorLaunch::create(len)?;
    unsafe { kernel_common::sequence_mask_u8(out, lengths, cols_i32, launch.len_i32) }
        .grid(launch.grid)
        .enqueue_on(stream)?;
    Ok::<(), Error>(())
}

macro_rules! triangular_sequence_mask_fn {
    ($name:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u8>,
            lengths: DevicePointer<u32>,
            rows: usize,
            cols: usize,
            diagonal: isize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), Error>(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(lengths)?;
            let rows_i32 = checked_i32_value(rows)?;
            let cols_i32 = checked_i32_value(cols)?;
            let diagonal_i32 = i32::try_from(diagonal).map_err(|_| Error::LengthExceedsI32)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                kernel_common::$kernel_fn(
                    out,
                    lengths,
                    rows_i32,
                    cols_i32,
                    diagonal_i32,
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok::<(), Error>(())
        }
    };
}

triangular_sequence_mask_fn!(tril_sequence_mask_u8, tril_sequence_mask_u8);
triangular_sequence_mask_fn!(triu_sequence_mask_u8, triu_sequence_mask_u8);

pub fn lower_triangular_sequence_mask_u8(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    lengths: DevicePointer<u32>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    tril_sequence_mask_u8(stream, out, lengths, rows, cols, diagonal)
}

pub fn upper_triangular_sequence_mask_u8(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    lengths: DevicePointer<u32>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    triu_sequence_mask_u8(stream, out, lengths, rows, cols, diagonal)
}

pub fn causal_sequence_mask_u8(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    lengths: DevicePointer<u32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    tril_sequence_mask_u8(stream, out, lengths, rows, cols, 0)
}

fn causal_mask_len(batch: usize, seq_len: usize) -> Result<usize> {
    let matrix_len = checked_element_count(seq_len, seq_len)?;
    checked_element_count(batch, matrix_len)
}

macro_rules! dispatch_mask_reduction {
    ($out:expr, $input:expr, $stream:expr, $bn:expr, [$k1:ident, $k8:ident, $k32:ident, $k64:ident, $k128:ident, $k256:ident, $k512:ident, $k1024:ident]) => {{
        match $bn {
            1 => kernel_common::$k1($out, $input).enqueue_on($stream)?,
            8 => kernel_common::$k8($out, $input).enqueue_on($stream)?,
            32 => kernel_common::$k32($out, $input).enqueue_on($stream)?,
            64 => kernel_common::$k64($out, $input).enqueue_on($stream)?,
            128 => kernel_common::$k128($out, $input).enqueue_on($stream)?,
            256 => kernel_common::$k256($out, $input).enqueue_on($stream)?,
            512 => kernel_common::$k512($out, $input).enqueue_on($stream)?,
            1024 => kernel_common::$k1024($out, $input).enqueue_on($stream)?,
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "mask reduction".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), Error>(())
    }};
}

macro_rules! mask_reduction_fn {
    ($name:ident, [$($kernel_fn:ident),+]) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u8>,
            input: DevicePointer<u8>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), Error>(());
            }
            if cols > MASK_REDUCTION_MAX_TILE_WIDTH {
                return Err(Error::UnsupportedWidth {
                    op: "mask reduction".into(),
                    width: cols,
                });
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let bn = mask_reduction_tile_width(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 1)?.partition([1, 1])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            dispatch_mask_reduction!(out, input, stream, bn, [$($kernel_fn),+])
        }
    };
}

mask_reduction_fn!(
    mask_any_u8,
    [
        mask_any_u8_1,
        mask_any_u8_8,
        mask_any_u8_32,
        mask_any_u8_64,
        mask_any_u8_128,
        mask_any_u8_256,
        mask_any_u8_512,
        mask_any_u8_1024
    ]
);
mask_reduction_fn!(
    mask_all_u8,
    [
        mask_all_u8_1,
        mask_all_u8_8,
        mask_all_u8_32,
        mask_all_u8_64,
        mask_all_u8_128,
        mask_all_u8_256,
        mask_all_u8_512,
        mask_all_u8_1024
    ]
);

macro_rules! mask_count_nonzero_fn {
    ($name:ident, [$($kernel_fn:ident),+]) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u32>,
            input: DevicePointer<u8>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), Error>(());
            }
            if cols > MASK_REDUCTION_MAX_TILE_WIDTH {
                return Err(Error::UnsupportedWidth {
                    op: "mask reduction".into(),
                    width: cols,
                });
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let bn = mask_reduction_tile_width(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 1)?.partition([1, 1])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            dispatch_mask_reduction!(out, input, stream, bn, [$($kernel_fn),+])
        }
    };
}

mask_count_nonzero_fn!(
    mask_count_nonzero_u8,
    [
        mask_count_nonzero_u8_1,
        mask_count_nonzero_u8_8,
        mask_count_nonzero_u8_32,
        mask_count_nonzero_u8_64,
        mask_count_nonzero_u8_128,
        mask_count_nonzero_u8_256,
        mask_count_nonzero_u8_512,
        mask_count_nonzero_u8_1024
    ]
);
