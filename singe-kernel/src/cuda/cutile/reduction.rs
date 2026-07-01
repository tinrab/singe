use std::sync::Arc;

#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

#[cfg(feature = "dtype-f16")]
use crate::cuda::cutile::kernel::f16::reduction as kernel_f16;
#[cfg(feature = "dtype-f32")]
use crate::cuda::cutile::kernel::f32::reduction as kernel_f32;
#[cfg(feature = "dtype-f64")]
use crate::cuda::cutile::kernel::f64::reduction as kernel_f64;
use crate::{
    cuda::cutile::{
        DeviceOpExt, adapter::TensorAdapter, kernel::common as kernel_common,
        utility::checked_device_pointer,
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

const REDUCTION_MAX_TILE_WIDTH: usize = 1024;
const BLOCK_ARGMAX_TILE_WIDTH: usize = 256;
const BLOCK_ARGMAX_REDUCE_MAX_BLOCKS: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BlockArgmax {
    len: i32,
    blocks: usize,
    grid: (u32, u32, u32),
}

impl BlockArgmax {
    fn create(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::InvalidLength);
        }
        let blocks = len.div_ceil(BLOCK_ARGMAX_TILE_WIDTH);
        if blocks > BLOCK_ARGMAX_REDUCE_MAX_BLOCKS {
            return Err(Error::UnsupportedBlockCount {
                op: "block argmax".into(),
                blocks,
            });
        }
        Ok(Self {
            len: checked_i32_value(len)?,
            blocks,
            grid: (
                u32::try_from(blocks).map_err(|_| Error::SizeOverflow)?,
                1,
                1,
            ),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BlockArgmaxReduce {
    blocks: i32,
    grid: (u32, u32, u32),
}

impl BlockArgmaxReduce {
    fn create(blocks: usize) -> Result<Self> {
        if blocks == 0 {
            return Err(Error::InvalidLength);
        }
        if blocks > BLOCK_ARGMAX_REDUCE_MAX_BLOCKS {
            return Err(Error::UnsupportedBlockCount {
                op: "block argmax".into(),
                blocks,
            });
        }
        Ok(Self {
            blocks: checked_i32_value(blocks)?,
            grid: (1, 1, 1),
        })
    }
}

fn reduction_tile_width(cols: usize) -> Result<usize> {
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
            op: "reduction".into(),
            width: cols,
        }),
    }
}

fn arg_reduction_tile_width(cols: usize) -> Result<usize> {
    match cols {
        1 => Ok(1),
        2..=8 => Ok(8),
        9..=128 => Ok(128),
        129..=256 => Ok(256),
        257..=512 => Ok(512),
        513..=1024 => Ok(1024),
        _ => Err(Error::UnsupportedWidth {
            op: "arg reduction".into(),
            width: cols,
        }),
    }
}

trait ReductionDenominator {
    fn from_usize(value: usize) -> Self;
}

impl ReductionDenominator for f32 {
    fn from_usize(value: usize) -> Self {
        value as f32
    }
}

#[cfg(feature = "dtype-f16")]
impl ReductionDenominator for f16 {
    fn from_usize(value: usize) -> Self {
        f16::from_f32(value as f32)
    }
}

impl ReductionDenominator for f64 {
    fn from_usize(value: usize) -> Self {
        value as f64
    }
}

macro_rules! dispatch_reduction {
    ($kernel:ident, $out:expr, $input:expr, $stream:expr, $bn:expr, [$k1:ident, $k8:ident, $k32:ident, $k64:ident, $k128:ident, $k256:ident, $k512:ident, $k1024:ident]) => {{
        match $bn {
            1 => $kernel::$k1($out, $input).enqueue_on($stream)?,
            8 => $kernel::$k8($out, $input).enqueue_on($stream)?,
            32 => $kernel::$k32($out, $input).enqueue_on($stream)?,
            64 => $kernel::$k64($out, $input).enqueue_on($stream)?,
            128 => $kernel::$k128($out, $input).enqueue_on($stream)?,
            256 => $kernel::$k256($out, $input).enqueue_on($stream)?,
            512 => $kernel::$k512($out, $input).enqueue_on($stream)?,
            1024 => $kernel::$k1024($out, $input).enqueue_on($stream)?,
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), crate::error::Error>(())
    }};
}

macro_rules! dispatch_mean {
    ($kernel:ident, $out:expr, $input:expr, $stream:expr, $cols:expr, $bn:expr, [$k1:ident, $k8:ident, $k32:ident, $k64:ident, $k128:ident, $k256:ident, $k512:ident, $k1024:ident]) => {{
        match $bn {
            1 => $kernel::$k1($out, $input, $cols).enqueue_on($stream)?,
            8 => $kernel::$k8($out, $input, $cols).enqueue_on($stream)?,
            32 => $kernel::$k32($out, $input, $cols).enqueue_on($stream)?,
            64 => $kernel::$k64($out, $input, $cols).enqueue_on($stream)?,
            128 => $kernel::$k128($out, $input, $cols).enqueue_on($stream)?,
            256 => $kernel::$k256($out, $input, $cols).enqueue_on($stream)?,
            512 => $kernel::$k512($out, $input, $cols).enqueue_on($stream)?,
            1024 => $kernel::$k1024($out, $input, $cols).enqueue_on($stream)?,
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), crate::error::Error>(())
    }};
}

macro_rules! reduction_fn {
    ($name:ident, $ty:ty, $kernel:ident, [$($kernel_fn:ident),+]) => {
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
            if cols > REDUCTION_MAX_TILE_WIDTH {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: cols,
                });
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let bn = reduction_tile_width(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 1)?.partition([1, 1])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            dispatch_reduction!($kernel, out, input, stream, bn, [$($kernel_fn),+])
        }
    };
}

macro_rules! mean_fn {
    ($name:ident, $ty:ty, $kernel:ident, [$($kernel_fn:ident),+]) => {
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
            if cols > REDUCTION_MAX_TILE_WIDTH {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: cols,
                });
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let bn = reduction_tile_width(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 1)?.partition([1, 1])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            let denominator = <$ty as ReductionDenominator>::from_usize(cols);
            dispatch_mean!($kernel, out, input, stream, denominator, bn, [$($kernel_fn),+])
        }
    };
}

macro_rules! integer_reduction_fns {
    ($ty:ty, $sum:ident, $max:ident, $min:ident, $sum_kernel:ident, $max_kernel:ident, $min_kernel:ident) => {
        reduction_fn!(
            $sum,
            $ty,
            kernel_common,
            [
                $sum_kernel,
                $sum_kernel,
                $sum_kernel,
                $sum_kernel,
                $sum_kernel,
                $sum_kernel,
                $sum_kernel,
                $sum_kernel
            ]
        );
        reduction_fn!(
            $max,
            $ty,
            kernel_common,
            [
                $max_kernel,
                $max_kernel,
                $max_kernel,
                $max_kernel,
                $max_kernel,
                $max_kernel,
                $max_kernel,
                $max_kernel
            ]
        );
        reduction_fn!(
            $min,
            $ty,
            kernel_common,
            [
                $min_kernel,
                $min_kernel,
                $min_kernel,
                $min_kernel,
                $min_kernel,
                $min_kernel,
                $min_kernel,
                $min_kernel
            ]
        );
    };
}

#[cfg(feature = "dtype-f32")]
reduction_fn!(
    sum_f32,
    f32,
    kernel_f32,
    [
        reduce_sum_f32_1,
        reduce_sum_f32_8,
        reduce_sum_f32_32,
        reduce_sum_f32_64,
        reduce_sum_f32_128,
        reduce_sum_f32_256,
        reduce_sum_f32_512,
        reduce_sum_f32_1024
    ]
);
#[cfg(feature = "dtype-f32")]
mean_fn!(
    mean_f32,
    f32,
    kernel_f32,
    [
        reduce_mean_f32_1,
        reduce_mean_f32_8,
        reduce_mean_f32_32,
        reduce_mean_f32_64,
        reduce_mean_f32_128,
        reduce_mean_f32_256,
        reduce_mean_f32_512,
        reduce_mean_f32_1024
    ]
);
#[cfg(feature = "dtype-f32")]
mean_fn!(
    variance_f32,
    f32,
    kernel_f32,
    [
        reduce_variance_f32,
        reduce_variance_f32,
        reduce_variance_f32,
        reduce_variance_f32,
        reduce_variance_f32,
        reduce_variance_f32,
        reduce_variance_f32,
        reduce_variance_f32
    ]
);
#[cfg(feature = "dtype-f32")]
mean_fn!(
    std_f32,
    f32,
    kernel_f32,
    [
        reduce_std_f32,
        reduce_std_f32,
        reduce_std_f32,
        reduce_std_f32,
        reduce_std_f32,
        reduce_std_f32,
        reduce_std_f32,
        reduce_std_f32
    ]
);
#[cfg(feature = "dtype-f32")]
reduction_fn!(
    max_f32,
    f32,
    kernel_f32,
    [
        reduce_max_f32_1,
        reduce_max_f32_8,
        reduce_max_f32_32,
        reduce_max_f32_64,
        reduce_max_f32_128,
        reduce_max_f32_256,
        reduce_max_f32_512,
        reduce_max_f32_1024
    ]
);
#[cfg(feature = "dtype-f32")]
reduction_fn!(
    min_f32,
    f32,
    kernel_f32,
    [
        reduce_min_f32_1,
        reduce_min_f32_8,
        reduce_min_f32_32,
        reduce_min_f32_64,
        reduce_min_f32_128,
        reduce_min_f32_256,
        reduce_min_f32_512,
        reduce_min_f32_1024
    ]
);
#[cfg(feature = "dtype-f32")]
reduction_fn!(
    product_f32,
    f32,
    kernel_f32,
    [
        reduce_product_f32_1,
        reduce_product_f32_8,
        reduce_product_f32_32,
        reduce_product_f32_64,
        reduce_product_f32_128,
        reduce_product_f32_256,
        reduce_product_f32_512,
        reduce_product_f32_1024
    ]
);

#[cfg(feature = "dtype-f16")]
reduction_fn!(
    sum_f16,
    f16,
    kernel_f16,
    [
        reduce_sum_f16_1,
        reduce_sum_f16_8,
        reduce_sum_f16_32,
        reduce_sum_f16_64,
        reduce_sum_f16_128,
        reduce_sum_f16_256,
        reduce_sum_f16_512,
        reduce_sum_f16_1024
    ]
);
#[cfg(feature = "dtype-f16")]
mean_fn!(
    mean_f16,
    f16,
    kernel_f16,
    [
        reduce_mean_f16_1,
        reduce_mean_f16_8,
        reduce_mean_f16_32,
        reduce_mean_f16_64,
        reduce_mean_f16_128,
        reduce_mean_f16_256,
        reduce_mean_f16_512,
        reduce_mean_f16_1024
    ]
);
#[cfg(feature = "dtype-f16")]
mean_fn!(
    variance_f16,
    f16,
    kernel_f16,
    [
        reduce_variance_f16,
        reduce_variance_f16,
        reduce_variance_f16,
        reduce_variance_f16,
        reduce_variance_f16,
        reduce_variance_f16,
        reduce_variance_f16,
        reduce_variance_f16
    ]
);
#[cfg(feature = "dtype-f16")]
mean_fn!(
    std_f16,
    f16,
    kernel_f16,
    [
        reduce_std_f16,
        reduce_std_f16,
        reduce_std_f16,
        reduce_std_f16,
        reduce_std_f16,
        reduce_std_f16,
        reduce_std_f16,
        reduce_std_f16
    ]
);
#[cfg(feature = "dtype-f16")]
reduction_fn!(
    max_f16,
    f16,
    kernel_f16,
    [
        reduce_max_f16_1,
        reduce_max_f16_8,
        reduce_max_f16_32,
        reduce_max_f16_64,
        reduce_max_f16_128,
        reduce_max_f16_256,
        reduce_max_f16_512,
        reduce_max_f16_1024
    ]
);
#[cfg(feature = "dtype-f16")]
reduction_fn!(
    min_f16,
    f16,
    kernel_f16,
    [
        reduce_min_f16_1,
        reduce_min_f16_8,
        reduce_min_f16_32,
        reduce_min_f16_64,
        reduce_min_f16_128,
        reduce_min_f16_256,
        reduce_min_f16_512,
        reduce_min_f16_1024
    ]
);
#[cfg(feature = "dtype-f16")]
reduction_fn!(
    product_f16,
    f16,
    kernel_f16,
    [
        reduce_product_f16_1,
        reduce_product_f16_8,
        reduce_product_f16_32,
        reduce_product_f16_64,
        reduce_product_f16_128,
        reduce_product_f16_256,
        reduce_product_f16_512,
        reduce_product_f16_1024
    ]
);

#[cfg(feature = "dtype-f64")]
reduction_fn!(
    sum_f64,
    f64,
    kernel_f64,
    [
        reduce_sum_f64_1,
        reduce_sum_f64_8,
        reduce_sum_f64_32,
        reduce_sum_f64_64,
        reduce_sum_f64_128,
        reduce_sum_f64_256,
        reduce_sum_f64_512,
        reduce_sum_f64_1024
    ]
);
#[cfg(feature = "dtype-f64")]
mean_fn!(
    mean_f64,
    f64,
    kernel_f64,
    [
        reduce_mean_f64_1,
        reduce_mean_f64_8,
        reduce_mean_f64_32,
        reduce_mean_f64_64,
        reduce_mean_f64_128,
        reduce_mean_f64_256,
        reduce_mean_f64_512,
        reduce_mean_f64_1024
    ]
);
#[cfg(feature = "dtype-f64")]
mean_fn!(
    variance_f64,
    f64,
    kernel_f64,
    [
        reduce_variance_f64,
        reduce_variance_f64,
        reduce_variance_f64,
        reduce_variance_f64,
        reduce_variance_f64,
        reduce_variance_f64,
        reduce_variance_f64,
        reduce_variance_f64
    ]
);
#[cfg(feature = "dtype-f64")]
mean_fn!(
    std_f64,
    f64,
    kernel_f64,
    [
        reduce_std_f64,
        reduce_std_f64,
        reduce_std_f64,
        reduce_std_f64,
        reduce_std_f64,
        reduce_std_f64,
        reduce_std_f64,
        reduce_std_f64
    ]
);
#[cfg(feature = "dtype-f64")]
reduction_fn!(
    max_f64,
    f64,
    kernel_f64,
    [
        reduce_max_f64_1,
        reduce_max_f64_8,
        reduce_max_f64_32,
        reduce_max_f64_64,
        reduce_max_f64_128,
        reduce_max_f64_256,
        reduce_max_f64_512,
        reduce_max_f64_1024
    ]
);
#[cfg(feature = "dtype-f64")]
reduction_fn!(
    min_f64,
    f64,
    kernel_f64,
    [
        reduce_min_f64_1,
        reduce_min_f64_8,
        reduce_min_f64_32,
        reduce_min_f64_64,
        reduce_min_f64_128,
        reduce_min_f64_256,
        reduce_min_f64_512,
        reduce_min_f64_1024
    ]
);
#[cfg(feature = "dtype-f64")]
reduction_fn!(
    product_f64,
    f64,
    kernel_f64,
    [
        reduce_product_f64_1,
        reduce_product_f64_8,
        reduce_product_f64_32,
        reduce_product_f64_64,
        reduce_product_f64_128,
        reduce_product_f64_256,
        reduce_product_f64_512,
        reduce_product_f64_1024
    ]
);

#[cfg(feature = "dtype-u8")]
integer_reduction_fns!(
    u8,
    sum_u8,
    max_u8,
    min_u8,
    reduce_sum_u8,
    reduce_max_u8,
    reduce_min_u8
);
#[cfg(feature = "dtype-i8")]
integer_reduction_fns!(
    i8,
    sum_i8,
    max_i8,
    min_i8,
    reduce_sum_i8,
    reduce_max_i8,
    reduce_min_i8
);
#[cfg(feature = "dtype-u32")]
integer_reduction_fns!(
    u32,
    sum_u32,
    max_u32,
    min_u32,
    reduce_sum_u32,
    reduce_max_u32,
    reduce_min_u32
);
#[cfg(feature = "dtype-i32")]
integer_reduction_fns!(
    i32,
    sum_i32,
    max_i32,
    min_i32,
    reduce_sum_i32,
    reduce_max_i32,
    reduce_min_i32
);
#[cfg(feature = "dtype-u64")]
integer_reduction_fns!(
    u64,
    sum_u64,
    max_u64,
    min_u64,
    reduce_sum_u64,
    reduce_max_u64,
    reduce_min_u64
);
#[cfg(feature = "dtype-i64")]
integer_reduction_fns!(
    i64,
    sum_i64,
    max_i64,
    min_i64,
    reduce_sum_i64,
    reduce_max_i64,
    reduce_min_i64
);

macro_rules! dispatch_arg_reduction {
    ($kernel:ident, $out:expr, $input:expr, $stream:expr, $bn:expr, [$k1:ident, $k8:ident, $k128:ident, $k256:ident, $k512:ident, $k1024:ident]) => {{
        match $bn {
            1 => $kernel::$k1($out, $input).enqueue_on($stream)?,
            8 => $kernel::$k8($out, $input).enqueue_on($stream)?,
            128 => $kernel::$k128($out, $input).enqueue_on($stream)?,
            256 => $kernel::$k256($out, $input).enqueue_on($stream)?,
            512 => $kernel::$k512($out, $input).enqueue_on($stream)?,
            1024 => $kernel::$k1024($out, $input).enqueue_on($stream)?,
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "arg reduction".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), crate::error::Error>(())
    }};
}

macro_rules! arg_reduction_fn {
    ($name:ident, $ty:ty, $kernel:ident, [$($kernel_fn:ident),+]) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<i32>,
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
            checked_element_count(rows, 2)?;
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let bn = arg_reduction_tile_width(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 2)?.partition([1, 2])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            dispatch_arg_reduction!($kernel, out, input, stream, bn, [$($kernel_fn),+])
        }
    };
}

macro_rules! integer_arg_reduction_fns {
    ($ty:ty, $argmax:ident, $argmin:ident, $argmax_kernel:ident, $argmin_kernel:ident) => {
        arg_reduction_fn!(
            $argmax,
            $ty,
            kernel_common,
            [
                $argmax_kernel,
                $argmax_kernel,
                $argmax_kernel,
                $argmax_kernel,
                $argmax_kernel,
                $argmax_kernel
            ]
        );
        arg_reduction_fn!(
            $argmin,
            $ty,
            kernel_common,
            [
                $argmin_kernel,
                $argmin_kernel,
                $argmin_kernel,
                $argmin_kernel,
                $argmin_kernel,
                $argmin_kernel
            ]
        );
    };
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
arg_reduction_fn!(
    argmax_f32,
    f32,
    kernel_f32,
    [
        reduce_argmax_f32_1,
        reduce_argmax_f32_8,
        reduce_argmax_f32_128,
        reduce_argmax_f32_256,
        reduce_argmax_f32_512,
        reduce_argmax_f32_1024
    ]
);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
arg_reduction_fn!(
    argmin_f32,
    f32,
    kernel_f32,
    [
        reduce_argmin_f32_1,
        reduce_argmin_f32_8,
        reduce_argmin_f32_128,
        reduce_argmin_f32_256,
        reduce_argmin_f32_512,
        reduce_argmin_f32_1024
    ]
);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i32"))]
arg_reduction_fn!(
    argmax_f16,
    f16,
    kernel_f16,
    [
        reduce_argmax_f16_1,
        reduce_argmax_f16_8,
        reduce_argmax_f16_128,
        reduce_argmax_f16_256,
        reduce_argmax_f16_512,
        reduce_argmax_f16_1024
    ]
);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i32"))]
arg_reduction_fn!(
    argmin_f16,
    f16,
    kernel_f16,
    [
        reduce_argmin_f16_1,
        reduce_argmin_f16_8,
        reduce_argmin_f16_128,
        reduce_argmin_f16_256,
        reduce_argmin_f16_512,
        reduce_argmin_f16_1024
    ]
);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
arg_reduction_fn!(
    argmax_f64,
    f64,
    kernel_f64,
    [
        reduce_argmax_f64_1,
        reduce_argmax_f64_8,
        reduce_argmax_f64_128,
        reduce_argmax_f64_256,
        reduce_argmax_f64_512,
        reduce_argmax_f64_1024
    ]
);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
arg_reduction_fn!(
    argmin_f64,
    f64,
    kernel_f64,
    [
        reduce_argmin_f64_1,
        reduce_argmin_f64_8,
        reduce_argmin_f64_128,
        reduce_argmin_f64_256,
        reduce_argmin_f64_512,
        reduce_argmin_f64_1024
    ]
);

#[cfg(all(feature = "dtype-u8", feature = "dtype-i32"))]
integer_arg_reduction_fns!(u8, argmax_u8, argmin_u8, reduce_argmax_u8, reduce_argmin_u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-i32"))]
integer_arg_reduction_fns!(i8, argmax_i8, argmin_i8, reduce_argmax_i8, reduce_argmin_i8);
#[cfg(all(feature = "dtype-u32", feature = "dtype-i32"))]
integer_arg_reduction_fns!(
    u32,
    argmax_u32,
    argmin_u32,
    reduce_argmax_u32,
    reduce_argmin_u32
);
#[cfg(feature = "dtype-i32")]
integer_arg_reduction_fns!(
    i32,
    argmax_i32,
    argmin_i32,
    reduce_argmax_i32,
    reduce_argmin_i32
);
#[cfg(all(feature = "dtype-u64", feature = "dtype-i32"))]
integer_arg_reduction_fns!(
    u64,
    argmax_u64,
    argmin_u64,
    reduce_argmax_u64,
    reduce_argmin_u64
);
#[cfg(all(feature = "dtype-i64", feature = "dtype-i32"))]
integer_arg_reduction_fns!(
    i64,
    argmax_i64,
    argmin_i64,
    reduce_argmax_i64,
    reduce_argmin_i64
);

macro_rules! block_argmax_fn {
    ($name:ident, $input_ty:ty, $max_ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            block_max: DevicePointer<$max_ty>,
            block_idx: DevicePointer<i32>,
            input: DevicePointer<$input_ty>,
            len: usize,
        ) -> Result<usize> {
            let params = BlockArgmax::create(len)?;
            checked_device_pointer(block_max)?;
            checked_device_pointer(block_idx)?;
            checked_device_pointer(input)?;
            unsafe { kernel_common::$kernel_fn(block_max, block_idx, input, params.len) }
                .grid(params.grid)
                .enqueue_on(stream)?;
            Ok(params.blocks)
        }
    };
}

#[cfg(all(feature = "dtype-f16", feature = "dtype-f32", feature = "dtype-i32"))]
block_argmax_fn!(block_argmax_f16, f16, f32, block_argmax_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
block_argmax_fn!(block_argmax_f32, f32, f32, block_argmax_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
block_argmax_fn!(block_argmax_f64, f64, f64, block_argmax_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-f32", feature = "dtype-i32"))]
block_argmax_fn!(block_argmin_f16, f16, f32, block_argmin_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
block_argmax_fn!(block_argmin_f32, f32, f32, block_argmin_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
block_argmax_fn!(block_argmin_f64, f64, f64, block_argmin_f64);

macro_rules! reduce_block_argmax_fn {
    ($name:ident, $max_ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<i32>,
            block_max: DevicePointer<$max_ty>,
            block_idx: DevicePointer<i32>,
            num_blocks: usize,
        ) -> Result<()> {
            let params = BlockArgmaxReduce::create(num_blocks)?;
            checked_device_pointer(out)?;
            checked_device_pointer(block_max)?;
            checked_device_pointer(block_idx)?;
            unsafe {
                kernel_common::$kernel_fn(out, block_max, block_idx, params.blocks)
                    .grid(params.grid)
                    .enqueue_on(stream)
            }?;
            Ok::<(), crate::error::Error>(())
        }
    };
}

#[cfg(all(feature = "dtype-f16", feature = "dtype-f32", feature = "dtype-i32"))]
reduce_block_argmax_fn!(reduce_block_argmax_f16, f32, reduce_block_argmax_f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
reduce_block_argmax_fn!(reduce_block_argmax_f32, f32, reduce_block_argmax_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
reduce_block_argmax_fn!(reduce_block_argmax_f64, f64, reduce_block_argmax_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-f32", feature = "dtype-i32"))]
reduce_block_argmax_fn!(reduce_block_argmin_f16, f32, reduce_block_argmin_f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
reduce_block_argmax_fn!(reduce_block_argmin_f32, f32, reduce_block_argmin_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
reduce_block_argmax_fn!(reduce_block_argmin_f64, f64, reduce_block_argmin_f64);
