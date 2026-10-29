use std::sync::Arc;

#[cfg(feature = "dtype_f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::CompileOptions,
};

#[cfg(feature = "dtype_f16")]
use crate::cuda::cutile::kernel::f16::reduction as kernel_f16;
#[cfg(feature = "dtype_f32")]
use crate::cuda::cutile::kernel::f32::reduction as kernel_f32;
#[cfg(feature = "dtype_f64")]
use crate::cuda::cutile::kernel::f64::reduction as kernel_f64;
use crate::{
    cuda::{
        autotune::KernelCompileConfig,
        cutile::{
            DeviceOpExt, TileKernel, adapter::TensorAdapter, kernel::common as kernel_common,
            utility::checked_device_pointer,
        },
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

const REDUCTION_MAX_TILE_WIDTH: usize = 1024;
const BLOCK_ARGMAX_TILE_WIDTH: usize = 256;

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

fn reduction_compile_options(compile: KernelCompileConfig) -> CompileOptions {
    let mut options = CompileOptions::default();
    if let Some(occupancy) = compile.occupancy {
        options = options.occupancy(occupancy);
    }
    if let Some(num_cta_in_cga) = compile.num_cta_in_cga {
        options = options.num_cta_in_cga(num_cta_in_cga);
    }
    if let Some(max_divisibility) = compile.max_divisibility {
        options = options.max_divisibility(max_divisibility);
    }
    options
}

trait ReductionDenominator {
    fn from_usize(value: usize) -> Self;
}

impl ReductionDenominator for f32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "the f32 algorithm converts integer dimensions and accumulators to its floating-point working precision"
    )]
    fn from_usize(value: usize) -> Self {
        value as Self
    }
}

#[cfg(feature = "dtype_f16")]
impl ReductionDenominator for f16 {
    fn from_usize(value: usize) -> Self {
        f16::from_f32(value as f32)
    }
}

impl ReductionDenominator for f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "floating-point reduction denominators use the output dtype precision"
    )]
    fn from_usize(value: usize) -> Self {
        value as Self
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
        Ok::<(), $crate::error::Error>(())
    }};
}

macro_rules! enqueue_reduction_op {
    ($op:expr, $stream:expr, $compile:expr) => {{
        if $compile.is_default() {
            $op.enqueue_on($stream)?;
        } else {
            $op.compile_options(reduction_compile_options($compile))
                .enqueue_on($stream)?;
        }
    }};
}

macro_rules! dispatch_reduction_tuned {
    ($kernel:ident, $out:expr, $input:expr, $stream:expr, $bn:expr, $compile:expr, [$k1:ident, $k8:ident, $k32:ident, $k64:ident, $k128:ident, $k256:ident, $k512:ident, $k1024:ident]) => {{
        match $bn {
            1 => enqueue_reduction_op!($kernel::$k1($out, $input), $stream, $compile),
            8 => enqueue_reduction_op!($kernel::$k8($out, $input), $stream, $compile),
            32 => enqueue_reduction_op!($kernel::$k32($out, $input), $stream, $compile),
            64 => enqueue_reduction_op!($kernel::$k64($out, $input), $stream, $compile),
            128 => enqueue_reduction_op!($kernel::$k128($out, $input), $stream, $compile),
            256 => enqueue_reduction_op!($kernel::$k256($out, $input), $stream, $compile),
            512 => enqueue_reduction_op!($kernel::$k512($out, $input), $stream, $compile),
            1024 => enqueue_reduction_op!($kernel::$k1024($out, $input), $stream, $compile),
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), $crate::error::Error>(())
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
        Ok::<(), $crate::error::Error>(())
    }};
}

macro_rules! dispatch_mean_tuned {
    ($kernel:ident, $out:expr, $input:expr, $stream:expr, $cols:expr, $bn:expr, $compile:expr, [$k1:ident, $k8:ident, $k32:ident, $k64:ident, $k128:ident, $k256:ident, $k512:ident, $k1024:ident]) => {{
        match $bn {
            1 => enqueue_reduction_op!($kernel::$k1($out, $input, $cols), $stream, $compile),
            8 => enqueue_reduction_op!($kernel::$k8($out, $input, $cols), $stream, $compile),
            32 => enqueue_reduction_op!($kernel::$k32($out, $input, $cols), $stream, $compile),
            64 => enqueue_reduction_op!($kernel::$k64($out, $input, $cols), $stream, $compile),
            128 => enqueue_reduction_op!($kernel::$k128($out, $input, $cols), $stream, $compile),
            256 => enqueue_reduction_op!($kernel::$k256($out, $input, $cols), $stream, $compile),
            512 => enqueue_reduction_op!($kernel::$k512($out, $input, $cols), $stream, $compile),
            1024 => enqueue_reduction_op!($kernel::$k1024($out, $input, $cols), $stream, $compile),
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), $crate::error::Error>(())
    }};
}

macro_rules! reduction_fn {
    ($name:ident, $ty:ty, $kernel:ident, [$($kernel_fn:ident),+]) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
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
                return Ok::<(), $crate::error::Error>(());
            }
            if cols > REDUCTION_MAX_TILE_WIDTH {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: cols,
                });
            }
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let bn = reduction_tile_width(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 1)?.partition([1, 1])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            dispatch_reduction!($kernel, out, input, stream, bn, [$($kernel_fn),+])
        }
    };
}

macro_rules! mean_fn {
    ($name:ident, $ty:ty, $kernel:ident, [$($kernel_fn:ident),+]) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
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
                return Ok::<(), $crate::error::Error>(());
            }
            if cols > REDUCTION_MAX_TILE_WIDTH {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: cols,
                });
            }
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let bn = reduction_tile_width(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 1)?.partition([1, 1])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            let denominator = <$ty as ReductionDenominator>::from_usize(cols);
            dispatch_mean!($kernel, out, input, stream, denominator, bn, [$($kernel_fn),+])
        }
    };
}

macro_rules! reduction_tuned_fn {
    ($name:ident, $ty:ty, $kernel:ident, [$($kernel_fn:ident),+]) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
            tile_width: usize,
            compile: KernelCompileConfig,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), $crate::error::Error>(());
            }
            if cols > REDUCTION_MAX_TILE_WIDTH || tile_width < cols {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: cols,
                });
            }
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 1)?.partition([1, 1])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            dispatch_reduction_tuned!(
                $kernel,
                out,
                input,
                stream,
                tile_width,
                compile,
                [$($kernel_fn),+]
            )
        }
    };
}

macro_rules! mean_tuned_fn {
    ($name:ident, $ty:ty, $kernel:ident, [$($kernel_fn:ident),+]) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
            tile_width: usize,
            compile: KernelCompileConfig,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), $crate::error::Error>(());
            }
            if cols > REDUCTION_MAX_TILE_WIDTH || tile_width < cols {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: cols,
                });
            }
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 1)?.partition([1, 1])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            let denominator = <$ty as ReductionDenominator>::from_usize(cols);
            dispatch_mean_tuned!(
                $kernel,
                out,
                input,
                stream,
                denominator,
                tile_width,
                compile,
                [$($kernel_fn),+]
            )
        }
    };
}

macro_rules! dispatch_integer_reduction {
    ($kernel:ident::$kernel_fn:ident, $out:expr, $input:expr, $stream:expr, $bn:expr) => {{
        match $bn {
            1 => $kernel::$kernel_fn($out, $input)
                .generics(vec!["1".into()])
                .enqueue_on($stream)?,
            8 => $kernel::$kernel_fn($out, $input)
                .generics(vec!["8".into()])
                .enqueue_on($stream)?,
            32 => $kernel::$kernel_fn($out, $input)
                .generics(vec!["32".into()])
                .enqueue_on($stream)?,
            64 => $kernel::$kernel_fn($out, $input)
                .generics(vec!["64".into()])
                .enqueue_on($stream)?,
            128 => $kernel::$kernel_fn($out, $input)
                .generics(vec!["128".into()])
                .enqueue_on($stream)?,
            256 => $kernel::$kernel_fn($out, $input)
                .generics(vec!["256".into()])
                .enqueue_on($stream)?,
            512 => $kernel::$kernel_fn($out, $input)
                .generics(vec!["512".into()])
                .enqueue_on($stream)?,
            1024 => $kernel::$kernel_fn($out, $input)
                .generics(vec!["1024".into()])
                .enqueue_on($stream)?,
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), $crate::error::Error>(())
    }};
}

macro_rules! dispatch_integer_reduction_tuned {
    ($kernel:ident::$kernel_fn:ident, $out:expr, $input:expr, $stream:expr, $bn:expr, $compile:expr) => {{
        match $bn {
            1 => enqueue_reduction_op!(
                $kernel::$kernel_fn($out, $input).generics(vec!["1".into()]),
                $stream,
                $compile
            ),
            8 => enqueue_reduction_op!(
                $kernel::$kernel_fn($out, $input).generics(vec!["8".into()]),
                $stream,
                $compile
            ),
            32 => enqueue_reduction_op!(
                $kernel::$kernel_fn($out, $input).generics(vec!["32".into()]),
                $stream,
                $compile
            ),
            64 => enqueue_reduction_op!(
                $kernel::$kernel_fn($out, $input).generics(vec!["64".into()]),
                $stream,
                $compile
            ),
            128 => enqueue_reduction_op!(
                $kernel::$kernel_fn($out, $input).generics(vec!["128".into()]),
                $stream,
                $compile
            ),
            256 => enqueue_reduction_op!(
                $kernel::$kernel_fn($out, $input).generics(vec!["256".into()]),
                $stream,
                $compile
            ),
            512 => enqueue_reduction_op!(
                $kernel::$kernel_fn($out, $input).generics(vec!["512".into()]),
                $stream,
                $compile
            ),
            1024 => enqueue_reduction_op!(
                $kernel::$kernel_fn($out, $input).generics(vec!["1024".into()]),
                $stream,
                $compile
            ),
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), $crate::error::Error>(())
    }};
}

macro_rules! integer_reduction_fn {
    ($name:ident, $ty:ty, $kernel:ident::$kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
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
                return Ok::<(), $crate::error::Error>(());
            }
            if cols > REDUCTION_MAX_TILE_WIDTH {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: cols,
                });
            }
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let bn = reduction_tile_width(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 1)?.partition([1, 1])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            dispatch_integer_reduction!($kernel::$kernel_fn, out, input, stream, bn)
        }
    };
}

macro_rules! integer_reduction_tuned_fn {
    ($name:ident, $ty:ty, $kernel:ident::$kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
            tile_width: usize,
            compile: KernelCompileConfig,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), $crate::error::Error>(());
            }
            if cols > REDUCTION_MAX_TILE_WIDTH || tile_width < cols {
                return Err(Error::UnsupportedWidth {
                    op: "reduction".into(),
                    width: cols,
                });
            }
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 1)?.partition([1, 1])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            dispatch_integer_reduction_tuned!(
                $kernel::$kernel_fn,
                out,
                input,
                stream,
                tile_width,
                compile
            )
        }
    };
}

macro_rules! integer_reduction_fns {
    (
        $ty:ty,
        $sum:ident,
        $sum_tuned:ident,
        $max:ident,
        $max_tuned:ident,
        $min:ident,
        $min_tuned:ident,
        $sum_kernel:ident,
        $max_kernel:ident,
        $min_kernel:ident
    ) => {
        integer_reduction_fn!($sum, $ty, kernel_common::$sum_kernel);
        integer_reduction_tuned_fn!($sum_tuned, $ty, kernel_common::$sum_kernel);
        integer_reduction_fn!($max, $ty, kernel_common::$max_kernel);
        integer_reduction_tuned_fn!($max_tuned, $ty, kernel_common::$max_kernel);
        integer_reduction_fn!($min, $ty, kernel_common::$min_kernel);
        integer_reduction_tuned_fn!($min_tuned, $ty, kernel_common::$min_kernel);
    };
}

#[cfg(feature = "dtype_f32")]
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

#[cfg(feature = "dtype_f32")]
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

#[cfg(feature = "dtype_f32")]
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

#[cfg(feature = "dtype_f32")]
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

#[cfg(feature = "dtype_f32")]
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

#[cfg(feature = "dtype_f32")]
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

#[cfg(feature = "dtype_f32")]
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

#[cfg(feature = "dtype_f32")]
reduction_tuned_fn!(
    sum_f32_tuned,
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

#[cfg(feature = "dtype_f32")]
mean_tuned_fn!(
    mean_f32_tuned,
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

#[cfg(feature = "dtype_f32")]
mean_tuned_fn!(
    variance_f32_tuned,
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

#[cfg(feature = "dtype_f32")]
mean_tuned_fn!(
    std_f32_tuned,
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

#[cfg(feature = "dtype_f32")]
reduction_tuned_fn!(
    max_f32_tuned,
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

#[cfg(feature = "dtype_f32")]
reduction_tuned_fn!(
    min_f32_tuned,
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

#[cfg(feature = "dtype_f32")]
reduction_tuned_fn!(
    product_f32_tuned,
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

#[cfg(feature = "dtype_f16")]
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

#[cfg(feature = "dtype_f16")]
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

#[cfg(feature = "dtype_f16")]
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

#[cfg(feature = "dtype_f16")]
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

#[cfg(feature = "dtype_f16")]
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

#[cfg(feature = "dtype_f16")]
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

#[cfg(feature = "dtype_f16")]
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

#[cfg(feature = "dtype_f16")]
reduction_tuned_fn!(
    sum_f16_tuned,
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

#[cfg(feature = "dtype_f16")]
mean_tuned_fn!(
    mean_f16_tuned,
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

#[cfg(feature = "dtype_f16")]
mean_tuned_fn!(
    variance_f16_tuned,
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

#[cfg(feature = "dtype_f16")]
mean_tuned_fn!(
    std_f16_tuned,
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

#[cfg(feature = "dtype_f16")]
reduction_tuned_fn!(
    max_f16_tuned,
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

#[cfg(feature = "dtype_f16")]
reduction_tuned_fn!(
    min_f16_tuned,
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

#[cfg(feature = "dtype_f16")]
reduction_tuned_fn!(
    product_f16_tuned,
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

#[cfg(feature = "dtype_f64")]
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

#[cfg(feature = "dtype_f64")]
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

#[cfg(feature = "dtype_f64")]
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

#[cfg(feature = "dtype_f64")]
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

#[cfg(feature = "dtype_f64")]
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

#[cfg(feature = "dtype_f64")]
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

#[cfg(feature = "dtype_f64")]
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

#[cfg(feature = "dtype_f64")]
reduction_tuned_fn!(
    sum_f64_tuned,
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

#[cfg(feature = "dtype_f64")]
mean_tuned_fn!(
    mean_f64_tuned,
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

#[cfg(feature = "dtype_f64")]
mean_tuned_fn!(
    variance_f64_tuned,
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

#[cfg(feature = "dtype_f64")]
mean_tuned_fn!(
    std_f64_tuned,
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

#[cfg(feature = "dtype_f64")]
reduction_tuned_fn!(
    max_f64_tuned,
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

#[cfg(feature = "dtype_f64")]
reduction_tuned_fn!(
    min_f64_tuned,
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

#[cfg(feature = "dtype_f64")]
reduction_tuned_fn!(
    product_f64_tuned,
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

#[cfg(feature = "dtype_u8")]
integer_reduction_fns!(
    u8,
    sum_u8,
    sum_u8_tuned,
    max_u8,
    max_u8_tuned,
    min_u8,
    min_u8_tuned,
    reduce_sum_u8,
    reduce_max_u8,
    reduce_min_u8
);

#[cfg(feature = "dtype_i8")]
integer_reduction_fns!(
    i8,
    sum_i8,
    sum_i8_tuned,
    max_i8,
    max_i8_tuned,
    min_i8,
    min_i8_tuned,
    reduce_sum_i8,
    reduce_max_i8,
    reduce_min_i8
);

#[cfg(feature = "dtype_u32")]
integer_reduction_fns!(
    u32,
    sum_u32,
    sum_u32_tuned,
    max_u32,
    max_u32_tuned,
    min_u32,
    min_u32_tuned,
    reduce_sum_u32,
    reduce_max_u32,
    reduce_min_u32
);

#[cfg(feature = "dtype_i32")]
integer_reduction_fns!(
    i32,
    sum_i32,
    sum_i32_tuned,
    max_i32,
    max_i32_tuned,
    min_i32,
    min_i32_tuned,
    reduce_sum_i32,
    reduce_max_i32,
    reduce_min_i32
);

#[cfg(feature = "dtype_u64")]
integer_reduction_fns!(
    u64,
    sum_u64,
    sum_u64_tuned,
    max_u64,
    max_u64_tuned,
    min_u64,
    min_u64_tuned,
    reduce_sum_u64,
    reduce_max_u64,
    reduce_min_u64
);

#[cfg(feature = "dtype_i64")]
integer_reduction_fns!(
    i64,
    sum_i64,
    sum_i64_tuned,
    max_i64,
    max_i64_tuned,
    min_i64,
    min_i64_tuned,
    reduce_sum_i64,
    reduce_max_i64,
    reduce_min_i64
);

macro_rules! arg_reduction_op {
    ($operation:expr, $width:literal) => {
        $operation
    };
    ($operation:expr, $width:literal, generic) => {
        $operation.generics(vec![$width.to_string()])
    };
}

macro_rules! dispatch_arg_reduction {
    ($kernel:ident, $out:expr, $input:expr, $stream:expr, $bn:expr, [$k1:ident, $k8:ident, $k128:ident, $k256:ident, $k512:ident, $k1024:ident] $(, $generic:ident)?) => {{
        match $bn {
            1 => arg_reduction_op!($kernel::$k1($out, $input), 1 $(, $generic)?).enqueue_on($stream)?,
            8 => arg_reduction_op!($kernel::$k8($out, $input), 8 $(, $generic)?).enqueue_on($stream)?,
            128 => arg_reduction_op!($kernel::$k128($out, $input), 128 $(, $generic)?).enqueue_on($stream)?,
            256 => arg_reduction_op!($kernel::$k256($out, $input), 256 $(, $generic)?).enqueue_on($stream)?,
            512 => arg_reduction_op!($kernel::$k512($out, $input), 512 $(, $generic)?).enqueue_on($stream)?,
            1024 => arg_reduction_op!($kernel::$k1024($out, $input), 1024 $(, $generic)?).enqueue_on($stream)?,
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "arg reduction".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), $crate::error::Error>(())
    }};
}

macro_rules! dispatch_arg_reduction_tuned {
    ($kernel:ident, $out:expr, $input:expr, $stream:expr, $bn:expr, $compile:expr, [$k1:ident, $k8:ident, $k128:ident, $k256:ident, $k512:ident, $k1024:ident] $(, $generic:ident)?) => {{
        match $bn {
            1 => enqueue_reduction_op!(arg_reduction_op!($kernel::$k1($out, $input), 1 $(, $generic)?), $stream, $compile),
            8 => enqueue_reduction_op!(arg_reduction_op!($kernel::$k8($out, $input), 8 $(, $generic)?), $stream, $compile),
            128 => enqueue_reduction_op!(arg_reduction_op!($kernel::$k128($out, $input), 128 $(, $generic)?), $stream, $compile),
            256 => enqueue_reduction_op!(arg_reduction_op!($kernel::$k256($out, $input), 256 $(, $generic)?), $stream, $compile),
            512 => enqueue_reduction_op!(arg_reduction_op!($kernel::$k512($out, $input), 512 $(, $generic)?), $stream, $compile),
            1024 => enqueue_reduction_op!(arg_reduction_op!($kernel::$k1024($out, $input), 1024 $(, $generic)?), $stream, $compile),
            _ => {
                return Err(Error::UnsupportedWidth {
                    op: "arg reduction".into(),
                    width: $bn,
                });
            }
        };
        Ok::<(), $crate::error::Error>(())
    }};
}

macro_rules! arg_reduction_fn {
    ($name:ident, $ty:ty, $kernel:ident, [$($kernel_fn:ident),+] $(, $generic:ident)?) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
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
                return Ok::<(), $crate::error::Error>(());
            }
            checked_element_count(rows, 2)?;
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let bn = arg_reduction_tile_width(cols)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 2)?.partition([1, 2])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            dispatch_arg_reduction!($kernel, out, input, stream, bn, [$($kernel_fn),+] $(, $generic)?)
        }
    };
}

macro_rules! arg_reduction_tuned_fn {
    ($name:ident, $ty:ty, $kernel:ident, [$($kernel_fn:ident),+] $(, $generic:ident)?) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<i32>,
            input: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
            tile_width: usize,
            compile: KernelCompileConfig,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok::<(), $crate::error::Error>(());
            }
            if cols > REDUCTION_MAX_TILE_WIDTH || tile_width < cols {
                return Err(Error::UnsupportedWidth {
                    op: "arg reduction".into(),
                    width: cols,
                });
            }
            checked_element_count(rows, 2)?;
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let out = TensorAdapter::contiguous_2d(out, rows, 2)?.partition([1, 2])?;
            let input = TensorAdapter::contiguous_2d(input, rows, cols)?;
            dispatch_arg_reduction_tuned!(
                $kernel,
                out,
                input,
                stream,
                tile_width,
                compile,
                [$($kernel_fn),+] $(, $generic)?
            )
        }
    };
}

macro_rules! integer_arg_reduction_fns {
    (
        $ty:ty,
        $argmax:ident,
        $argmax_tuned:ident,
        $argmin:ident,
        $argmin_tuned:ident,
        $argmax_kernel:ident,
        $argmin_kernel:ident
    ) => {
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
            ],
            generic
        );
        arg_reduction_tuned_fn!(
            $argmax_tuned,
            $ty,
            kernel_common,
            [
                $argmax_kernel,
                $argmax_kernel,
                $argmax_kernel,
                $argmax_kernel,
                $argmax_kernel,
                $argmax_kernel
            ],
            generic
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
            ],
            generic
        );
        arg_reduction_tuned_fn!(
            $argmin_tuned,
            $ty,
            kernel_common,
            [
                $argmin_kernel,
                $argmin_kernel,
                $argmin_kernel,
                $argmin_kernel,
                $argmin_kernel,
                $argmin_kernel
            ],
            generic
        );
    };
}

#[cfg(feature = "dtype_f32")]
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

#[cfg(feature = "dtype_f32")]
arg_reduction_tuned_fn!(
    argmax_f32_tuned,
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

#[cfg(feature = "dtype_f32")]
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

#[cfg(feature = "dtype_f32")]
arg_reduction_tuned_fn!(
    argmin_f32_tuned,
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

#[cfg(feature = "dtype_f16")]
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

#[cfg(feature = "dtype_f16")]
arg_reduction_tuned_fn!(
    argmax_f16_tuned,
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

#[cfg(feature = "dtype_f16")]
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

#[cfg(feature = "dtype_f16")]
arg_reduction_tuned_fn!(
    argmin_f16_tuned,
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

#[cfg(feature = "dtype_f64")]
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

#[cfg(feature = "dtype_f64")]
arg_reduction_tuned_fn!(
    argmax_f64_tuned,
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

#[cfg(feature = "dtype_f64")]
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

#[cfg(feature = "dtype_f64")]
arg_reduction_tuned_fn!(
    argmin_f64_tuned,
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

#[cfg(feature = "dtype_u8")]
integer_arg_reduction_fns!(
    u8,
    argmax_u8,
    argmax_u8_tuned,
    argmin_u8,
    argmin_u8_tuned,
    reduce_argmax_u8,
    reduce_argmin_u8
);

#[cfg(feature = "dtype_i8")]
integer_arg_reduction_fns!(
    i8,
    argmax_i8,
    argmax_i8_tuned,
    argmin_i8,
    argmin_i8_tuned,
    reduce_argmax_i8,
    reduce_argmin_i8
);

#[cfg(feature = "dtype_u32")]
integer_arg_reduction_fns!(
    u32,
    argmax_u32,
    argmax_u32_tuned,
    argmin_u32,
    argmin_u32_tuned,
    reduce_argmax_u32,
    reduce_argmin_u32
);

#[cfg(feature = "dtype_i32")]
integer_arg_reduction_fns!(
    i32,
    argmax_i32,
    argmax_i32_tuned,
    argmin_i32,
    argmin_i32_tuned,
    reduce_argmax_i32,
    reduce_argmin_i32
);

#[cfg(feature = "dtype_u64")]
integer_arg_reduction_fns!(
    u64,
    argmax_u64,
    argmax_u64_tuned,
    argmin_u64,
    argmin_u64_tuned,
    reduce_argmax_u64,
    reduce_argmin_u64
);

#[cfg(feature = "dtype_i64")]
integer_arg_reduction_fns!(
    i64,
    argmax_i64,
    argmax_i64_tuned,
    argmin_i64,
    argmin_i64_tuned,
    reduce_argmax_i64,
    reduce_argmin_i64
);

macro_rules! block_argmax_fn {
    ($name:ident, $input_ty:ty, $max_ty:ty, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            block_max: DevicePointer<$max_ty>,
            block_idx: DevicePointer<i32>,
            input: DevicePointer<$input_ty>,
            len: usize,
        ) -> Result<usize> {
            let params = BlockArgmax::create(len)?;
            checked_device_pointer(&block_max)?;
            checked_device_pointer(&block_idx)?;
            checked_device_pointer(&input)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { kernel_common::$kernel_fn(block_max, block_idx, input, params.len) }
                .grid(params.grid)
                .enqueue_on(stream)?;
            Ok(params.blocks)
        }
    };
}

#[cfg(all(feature = "dtype_f16", feature = "dtype_f32"))]
block_argmax_fn!(block_argmax_f16, f16, f32, block_argmax_f16);

#[cfg(feature = "dtype_f32")]
block_argmax_fn!(block_argmax_f32, f32, f32, block_argmax_f32);

#[cfg(feature = "dtype_f64")]
block_argmax_fn!(block_argmax_f64, f64, f64, block_argmax_f64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_f32"))]
block_argmax_fn!(block_argmin_f16, f16, f32, block_argmin_f16);

#[cfg(feature = "dtype_f32")]
block_argmax_fn!(block_argmin_f32, f32, f32, block_argmin_f32);

#[cfg(feature = "dtype_f64")]
block_argmax_fn!(block_argmin_f64, f64, f64, block_argmin_f64);

macro_rules! reduce_block_argmax_fn {
    ($name:ident, $max_ty:ty, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<i32>,
            block_max: DevicePointer<$max_ty>,
            block_idx: DevicePointer<i32>,
            num_blocks: usize,
        ) -> Result<()> {
            let params = BlockArgmaxReduce::create(num_blocks)?;
            checked_device_pointer(&out)?;
            checked_device_pointer(&block_max)?;
            checked_device_pointer(&block_idx)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                kernel_common::$kernel_fn(out, block_max, block_idx, params.blocks)
                    .grid(params.grid)
                    .enqueue_on(stream)
            }?;
            Ok::<(), $crate::error::Error>(())
        }
    };
}

#[cfg(all(feature = "dtype_f16", feature = "dtype_f32"))]
reduce_block_argmax_fn!(reduce_block_argmax_f16, f32, reduce_block_argmax_f32);

#[cfg(feature = "dtype_f32")]
reduce_block_argmax_fn!(reduce_block_argmax_f32, f32, reduce_block_argmax_f32);

#[cfg(feature = "dtype_f64")]
reduce_block_argmax_fn!(reduce_block_argmax_f64, f64, reduce_block_argmax_f64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_f32"))]
reduce_block_argmax_fn!(reduce_block_argmin_f16, f32, reduce_block_argmin_f32);

#[cfg(feature = "dtype_f32")]
reduce_block_argmax_fn!(reduce_block_argmin_f32, f32, reduce_block_argmin_f32);

#[cfg(feature = "dtype_f64")]
reduce_block_argmax_fn!(reduce_block_argmin_f64, f64, reduce_block_argmin_f64);
