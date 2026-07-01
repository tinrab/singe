use std::sync::Arc;

#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

#[cfg(feature = "dtype-f16")]
use crate::cuda::cutile::kernel::f16::softmax as kernel_f16;
#[cfg(feature = "dtype-f32")]
use crate::cuda::cutile::kernel::f32::softmax as kernel_f32;
#[cfg(feature = "dtype-f64")]
use crate::cuda::cutile::kernel::f64::softmax as kernel_f64;
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

const SOFTMAX_MAX_TILE_WIDTH: usize = 1024;
const SOFTMAX_WIDE_TILE_WIDTH: usize = 1024;
const SOFTMAX_WIDE_MAX_CHUNKS: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SoftmaxWide {
    cols: i32,
    rows: i32,
    chunks: i32,
    len: i32,
    chunk_grid: (u32, u32, u32),
    row_grid: (u32, u32, u32),
    len_grid: (u32, u32, u32),
    row_vector_grid: (u32, u32, u32),
}

impl SoftmaxWide {
    fn create(rows: usize, cols: usize) -> Result<Option<Self>> {
        if cols == 0 {
            return Err(Error::InvalidLength);
        }
        let len = checked_element_count(rows, cols)?;
        if len == 0 {
            return Ok(None);
        }
        let chunks = cols.div_ceil(SOFTMAX_WIDE_TILE_WIDTH);
        if chunks > SOFTMAX_WIDE_MAX_CHUNKS {
            return Err(Error::UnsupportedWidth {
                op: "softmax".into(),
                width: cols,
            });
        }
        let rows_u32 = u32::try_from(rows).map_err(|_| Error::SizeOverflow)?;
        let chunks_u32 = u32::try_from(chunks).map_err(|_| Error::SizeOverflow)?;
        Ok(Some(Self {
            cols: checked_i32_value(cols)?,
            rows: checked_i32_value(rows)?,
            chunks: checked_i32_value(chunks)?,
            len: checked_i32_value(len)?,
            chunk_grid: (chunks_u32, rows_u32, 1),
            row_grid: (rows_u32, 1, 1),
            len_grid: VectorLaunch::create(len)?.grid,
            row_vector_grid: VectorLaunch::create(rows)?.grid,
        }))
    }
}

fn softmax_tile_shape(cols: usize) -> Result<(usize, usize)> {
    match cols {
        1..=8 => Ok((4, 8)),
        9..=16 => Ok((2, 16)),
        17..=32 => Ok((1, 32)),
        33..=64 => Ok((1, 64)),
        65..=128 => Ok((1, 128)),
        129..=256 => Ok((1, 256)),
        257..=512 => Ok((1, 512)),
        513..=1024 => Ok((1, 1024)),
        _ => Err(Error::UnsupportedWidth {
            op: "softmax".into(),
            width: cols,
        }),
    }
}

macro_rules! dispatch_softmax {
    ($kernel:ident::$kernel_fn:ident, $out:expr, $input:expr, $stream:expr, $bm:expr, $bn:expr) => {{
        match ($bm, $bn) {
            (4, 8) => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            (2, 16) => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            (1, 32) => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            (1, 64) => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            (1, 128) => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            (1, 256) => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            (1, 512) => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            (1, 1024) => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "softmax".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), crate::error::Error>(())
    }};
}

macro_rules! dispatch_softmax_lse {
    ($kernel:ident::$kernel_fn:ident, $out:expr, $input:expr, $stream:expr, $bn:expr) => {{
        match $bn {
            8 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            16 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            32 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            64 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            128 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            256 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            512 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            1024 => $kernel::$kernel_fn($out, $input).enqueue_on($stream)?,
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "softmax".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), crate::error::Error>(())
    }};
}

macro_rules! softmax_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
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
                return Ok::<(), crate::error::Error>(());
            }
            if cols > SOFTMAX_MAX_TILE_WIDTH {
                return Err(Error::UnsupportedWidth {
                    op: "softmax".into(),
                    width: cols,
                });
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let (bm, bn) = softmax_tile_shape(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, cols)?.partition([bm, bn])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            dispatch_softmax!($kernel::$kernel_fn, out, input, stream, bm, bn)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
softmax_fn!(softmax_f32, f32, kernel_f32, softmax_f32);
#[cfg(feature = "dtype-f32")]
softmax_fn!(log_softmax_f32, f32, kernel_f32, log_softmax_f32);
#[cfg(feature = "dtype-f32")]
softmax_fn!(softmin_f32, f32, kernel_f32, softmin_f32);
#[cfg(feature = "dtype-f16")]
softmax_fn!(softmax_f16, f16, kernel_f16, softmax_f16);
#[cfg(feature = "dtype-f16")]
softmax_fn!(log_softmax_f16, f16, kernel_f16, log_softmax_f16);
#[cfg(feature = "dtype-f16")]
softmax_fn!(softmin_f16, f16, kernel_f16, softmin_f16);
#[cfg(feature = "dtype-f64")]
softmax_fn!(softmax_f64, f64, kernel_f64, softmax_f64);
#[cfg(feature = "dtype-f64")]
softmax_fn!(log_softmax_f64, f64, kernel_f64, log_softmax_f64);
#[cfg(feature = "dtype-f64")]
softmax_fn!(softmin_f64, f64, kernel_f64, softmin_f64);

macro_rules! softmax_lse_fn {
    ($name:ident, $input_ty:ty, $output_ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$output_ty>,
            input: DevicePointer<$input_ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), crate::error::Error>(());
            }
            if cols > SOFTMAX_MAX_TILE_WIDTH {
                return Err(Error::UnsupportedWidth {
                    op: "softmax".into(),
                    width: cols,
                });
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let (_, bn) = softmax_tile_shape(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 1)?.partition([1, 1])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            dispatch_softmax_lse!($kernel::$kernel_fn, out, input, stream, bn)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
softmax_lse_fn!(softmax_lse_f32, f32, f32, kernel_f32, softmax_lse_f32);
#[cfg(feature = "dtype-f16")]
softmax_lse_fn!(softmax_lse_f16, f16, f32, kernel_f16, softmax_lse_f16);
#[cfg(feature = "dtype-f64")]
softmax_lse_fn!(softmax_lse_f64, f64, f64, kernel_f64, softmax_lse_f64);

macro_rules! softmax_wide_fn {
    (
        $name:ident,
        $input_ty:ty,
        $scratch_ty:ty,
        $partial_max:ident,
        $reduce_max:ident,
        $partial_sum:ident,
        $reduce_sum:ident,
        $normalize:ident
    ) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$input_ty>,
            input: DevicePointer<$input_ty>,
            partial: DevicePointer<$scratch_ty>,
            row_max: DevicePointer<$scratch_ty>,
            row_sum: DevicePointer<$scratch_ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let Some(params) = SoftmaxWide::create(rows, cols)? else {
                return Ok::<(), crate::error::Error>(());
            };
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(partial)?;
            checked_device_pointer(row_max)?;
            checked_device_pointer(row_sum)?;

            unsafe { kernel_common::$partial_max(partial, input, params.cols, params.chunks) }
                .grid(params.chunk_grid)
                .enqueue_on(stream)?;

            unsafe { kernel_common::$reduce_max(row_max, partial, params.chunks) }
                .grid(params.row_grid)
                .enqueue_on(stream)?;

            unsafe {
                kernel_common::$partial_sum(partial, input, row_max, params.cols, params.chunks)
            }
            .grid(params.chunk_grid)
            .enqueue_on(stream)?;

            unsafe { kernel_common::$reduce_sum(row_sum, partial, params.chunks) }
                .grid(params.row_grid)
                .enqueue_on(stream)?;

            unsafe {
                kernel_common::$normalize(out, input, row_max, row_sum, params.cols, params.len)
            }
            .grid(params.len_grid)
            .enqueue_on(stream)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

macro_rules! softmax_lse_wide_fn {
    (
        $name:ident,
        $input_ty:ty,
        $output_ty:ty,
        $scratch_ty:ty,
        $partial_max:ident,
        $reduce_max:ident,
        $partial_sum:ident,
        $reduce_sum:ident,
        $finalize:ident
    ) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$output_ty>,
            input: DevicePointer<$input_ty>,
            partial: DevicePointer<$scratch_ty>,
            row_max: DevicePointer<$scratch_ty>,
            row_sum: DevicePointer<$scratch_ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let Some(params) = SoftmaxWide::create(rows, cols)? else {
                return Ok::<(), crate::error::Error>(());
            };
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(partial)?;
            checked_device_pointer(row_max)?;
            checked_device_pointer(row_sum)?;

            unsafe { kernel_common::$partial_max(partial, input, params.cols, params.chunks) }
                .grid(params.chunk_grid)
                .enqueue_on(stream)?;

            unsafe { kernel_common::$reduce_max(row_max, partial, params.chunks) }
                .grid(params.row_grid)
                .enqueue_on(stream)?;

            unsafe {
                kernel_common::$partial_sum(partial, input, row_max, params.cols, params.chunks)
            }
            .grid(params.chunk_grid)
            .enqueue_on(stream)?;

            unsafe { kernel_common::$reduce_sum(row_sum, partial, params.chunks) }
                .grid(params.row_grid)
                .enqueue_on(stream)?;

            unsafe { kernel_common::$finalize(out, row_max, row_sum, params.rows) }
                .grid(params.row_vector_grid)
                .enqueue_on(stream)?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
softmax_wide_fn!(
    softmax_wide_f32,
    f32,
    f32,
    softmax_partial_max_f32,
    softmax_reduce_max_f32,
    softmax_partial_sum_f32,
    softmax_reduce_sum_f32,
    softmax_normalize_f32
);
#[cfg(feature = "dtype-f16")]
softmax_wide_fn!(
    softmax_wide_f16,
    f16,
    f32,
    softmax_partial_max_f16,
    softmax_reduce_max_f32,
    softmax_partial_sum_f16,
    softmax_reduce_sum_f32,
    softmax_normalize_f16
);
#[cfg(feature = "dtype-f64")]
softmax_wide_fn!(
    softmax_wide_f64,
    f64,
    f64,
    softmax_partial_max_f64,
    softmax_reduce_max_f64,
    softmax_partial_sum_f64,
    softmax_reduce_sum_f64,
    softmax_normalize_f64
);

#[cfg(feature = "dtype-f32")]
softmax_wide_fn!(
    log_softmax_wide_f32,
    f32,
    f32,
    softmax_partial_max_f32,
    softmax_reduce_max_f32,
    softmax_partial_sum_f32,
    softmax_reduce_sum_f32,
    log_softmax_normalize_f32
);
#[cfg(feature = "dtype-f16")]
softmax_wide_fn!(
    log_softmax_wide_f16,
    f16,
    f32,
    softmax_partial_max_f16,
    softmax_reduce_max_f32,
    softmax_partial_sum_f16,
    softmax_reduce_sum_f32,
    log_softmax_normalize_f16
);
#[cfg(feature = "dtype-f64")]
softmax_wide_fn!(
    log_softmax_wide_f64,
    f64,
    f64,
    softmax_partial_max_f64,
    softmax_reduce_max_f64,
    softmax_partial_sum_f64,
    softmax_reduce_sum_f64,
    log_softmax_normalize_f64
);

#[cfg(feature = "dtype-f32")]
softmax_wide_fn!(
    softmin_wide_f32,
    f32,
    f32,
    softmin_partial_min_f32,
    softmin_reduce_min_f32,
    softmin_partial_sum_f32,
    softmax_reduce_sum_f32,
    softmin_normalize_f32
);
#[cfg(feature = "dtype-f16")]
softmax_wide_fn!(
    softmin_wide_f16,
    f16,
    f32,
    softmin_partial_min_f16,
    softmin_reduce_min_f32,
    softmin_partial_sum_f16,
    softmax_reduce_sum_f32,
    softmin_normalize_f16
);
#[cfg(feature = "dtype-f64")]
softmax_wide_fn!(
    softmin_wide_f64,
    f64,
    f64,
    softmin_partial_min_f64,
    softmin_reduce_min_f64,
    softmin_partial_sum_f64,
    softmax_reduce_sum_f64,
    softmin_normalize_f64
);

#[cfg(feature = "dtype-f32")]
softmax_lse_wide_fn!(
    softmax_lse_wide_f32,
    f32,
    f32,
    f32,
    softmax_partial_max_f32,
    softmax_reduce_max_f32,
    softmax_partial_sum_f32,
    softmax_reduce_sum_f32,
    softmax_lse_from_parts_f32
);
#[cfg(feature = "dtype-f16")]
softmax_lse_wide_fn!(
    softmax_lse_wide_f16,
    f16,
    f32,
    f32,
    softmax_partial_max_f16,
    softmax_reduce_max_f32,
    softmax_partial_sum_f16,
    softmax_reduce_sum_f32,
    softmax_lse_from_parts_f32
);
#[cfg(feature = "dtype-f64")]
softmax_lse_wide_fn!(
    softmax_lse_wide_f64,
    f64,
    f64,
    f64,
    softmax_partial_max_f64,
    softmax_reduce_max_f64,
    softmax_partial_sum_f64,
    softmax_reduce_sum_f64,
    softmax_lse_from_parts_f64
);
