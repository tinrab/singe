use std::sync::Arc;

#[cfg(feature = "dtype_bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype_f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::CompileOptions,
};

#[cfg(any(feature = "dtype_f16", feature = "dtype_bf16"))]
use crate::cuda::cutile::kernel::f16::cast as kernel_f16;
#[cfg(feature = "dtype_f32")]
use crate::cuda::cutile::kernel::f32::cast as kernel_f32;
#[cfg(feature = "dtype_f64")]
use crate::cuda::cutile::kernel::f64::cast as kernel_f64;
#[cfg(any(
    feature = "dtype_u8",
    feature = "dtype_i8",
    feature = "dtype_u32",
    feature = "dtype_i32",
    feature = "dtype_u64",
    feature = "dtype_i64",
))]
use crate::cuda::cutile::{kernel::common as kernel_common, utility::VectorLaunch};
use crate::{
    cuda::{
        autotune::KernelCompileConfig,
        cutile::{
            DeviceOpExt, TileKernel,
            adapter::TensorAdapter,
            utility::{checked_device_pointer, vector_tile_size},
        },
    },
    error::Result,
};

fn vector_compile_options(compile: KernelCompileConfig) -> CompileOptions {
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

macro_rules! enqueue_vector_op {
    ($op:expr, $stream:expr, $compile:expr) => {{
        if $compile.is_default() {
            $op.enqueue_on($stream)?;
        } else {
            $op.compile_options(vector_compile_options($compile))
                .enqueue_on($stream)?;
        }
    }};
}

macro_rules! cast_fn {
    ($name:ident, $out_ty:ty, $in_ty:ty, $kernel:ident, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$out_ty>,
            input: DevicePointer<$in_ty>,
            len: usize,
            tile_width: Option<usize>,
            compile: KernelCompileConfig,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let tile = tile_width.unwrap_or_else(|| vector_tile_size(len));
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let input = TensorAdapter::contiguous_1d(input, len)?;
            enqueue_vector_op!($kernel::$kernel_fn(out, input), stream, compile);
            Ok(())
        }
    };
}

#[cfg(feature = "dtype_u8")]
macro_rules! bool_cast_fn {
    ($name:ident, $in_ty:ty, $kernel:ident, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u8>,
            input: DevicePointer<$in_ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let launch = VectorLaunch::create(len)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { $kernel::$kernel_fn(out, input, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(any(
    feature = "dtype_u8",
    feature = "dtype_i8",
    feature = "dtype_u32",
    feature = "dtype_i32",
    feature = "dtype_u64",
    feature = "dtype_i64",
))]
macro_rules! raw_cast_fn {
    ($name:ident, $out_ty:ty, $in_ty:ty, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$out_ty>,
            input: DevicePointer<$in_ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let launch = VectorLaunch::create(len)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { kernel_common::$kernel_fn(out, input, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype_f16")]
#[cfg(feature = "dtype_f32")]
cast_fn!(f16_to_f32, f32, f16, kernel_f16, cast_f16_f32);

#[cfg(feature = "dtype_bf16")]
#[cfg(feature = "dtype_f32")]
cast_fn!(bf16_to_f32, f32, bf16, kernel_f16, cast_bf16_f32);

#[cfg(feature = "dtype_f16")]
#[cfg(feature = "dtype_f64")]
cast_fn!(f16_to_f64, f64, f16, kernel_f16, cast_f16_f64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
cast_fn!(f16_to_u8, u8, f16, kernel_f16, cast_f16_u8);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i8"))]
cast_fn!(f16_to_i8, i8, f16, kernel_f16, cast_f16_i8);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u32"))]
cast_fn!(f16_to_u32, u32, f16, kernel_f16, cast_f16_u32);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i32"))]
cast_fn!(f16_to_i32, i32, f16, kernel_f16, cast_f16_i32);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u64"))]
cast_fn!(f16_to_u64, u64, f16, kernel_f16, cast_f16_u64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i64"))]
cast_fn!(f16_to_i64, i64, f16, kernel_f16, cast_f16_i64);

#[cfg(feature = "dtype_f16")]
#[cfg(feature = "dtype_f32")]
cast_fn!(f32_to_f16, f16, f32, kernel_f32, cast_f32_f16);

#[cfg(feature = "dtype_bf16")]
#[cfg(feature = "dtype_f32")]
cast_fn!(f32_to_bf16, bf16, f32, kernel_f32, cast_f32_bf16);

#[cfg(feature = "dtype_f64")]
#[cfg(feature = "dtype_f32")]
cast_fn!(f32_to_f64, f64, f32, kernel_f32, cast_f32_f64);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
cast_fn!(f32_to_u8, u8, f32, kernel_f32, cast_f32_u8);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i8"))]
cast_fn!(f32_to_i8, i8, f32, kernel_f32, cast_f32_i8);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u32"))]
cast_fn!(f32_to_u32, u32, f32, kernel_f32, cast_f32_u32);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i32"))]
cast_fn!(f32_to_i32, i32, f32, kernel_f32, cast_f32_i32);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u64"))]
cast_fn!(f32_to_u64, u64, f32, kernel_f32, cast_f32_u64);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i64"))]
cast_fn!(f32_to_i64, i64, f32, kernel_f32, cast_f32_i64);

#[cfg(feature = "dtype_f16")]
#[cfg(feature = "dtype_f64")]
cast_fn!(f64_to_f16, f16, f64, kernel_f64, cast_f64_f16);

#[cfg(feature = "dtype_f64")]
#[cfg(feature = "dtype_f32")]
cast_fn!(f64_to_f32, f32, f64, kernel_f64, cast_f64_f32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
cast_fn!(f64_to_u8, u8, f64, kernel_f64, cast_f64_u8);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i8"))]
cast_fn!(f64_to_i8, i8, f64, kernel_f64, cast_f64_i8);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u32"))]
cast_fn!(f64_to_u32, u32, f64, kernel_f64, cast_f64_u32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i32"))]
cast_fn!(f64_to_i32, i32, f64, kernel_f64, cast_f64_i32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u64"))]
cast_fn!(f64_to_u64, u64, f64, kernel_f64, cast_f64_u64);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i64"))]
cast_fn!(f64_to_i64, i64, f64, kernel_f64, cast_f64_i64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
cast_fn!(bool_to_f16, f16, u8, kernel_f16, cast_bool_f16);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
cast_fn!(bool_to_f32, f32, u8, kernel_f32, cast_bool_f32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
cast_fn!(bool_to_f64, f64, u8, kernel_f64, cast_bool_f64);

#[cfg(feature = "dtype_u8")]
raw_cast_fn!(bool_to_u8, u8, u8, cast_bool_u8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_i8"))]
raw_cast_fn!(bool_to_i8, i8, u8, cast_bool_i8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_u32"))]
raw_cast_fn!(bool_to_u32, u32, u8, cast_bool_u32);

#[cfg(all(feature = "dtype_u8", feature = "dtype_i32"))]
raw_cast_fn!(bool_to_i32, i32, u8, cast_bool_i32);

#[cfg(all(feature = "dtype_u8", feature = "dtype_u64"))]
raw_cast_fn!(bool_to_u64, u64, u8, cast_bool_u64);

#[cfg(all(feature = "dtype_u8", feature = "dtype_i64"))]
raw_cast_fn!(bool_to_i64, i64, u8, cast_bool_i64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
cast_fn!(u8_to_f16, f16, u8, kernel_f16, cast_u8_f16);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
cast_fn!(u8_to_f32, f32, u8, kernel_f32, cast_u8_f32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
cast_fn!(u8_to_f64, f64, u8, kernel_f64, cast_u8_f64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i8"))]
cast_fn!(i8_to_f16, f16, i8, kernel_f16, cast_i8_f16);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i8"))]
cast_fn!(i8_to_f32, f32, i8, kernel_f32, cast_i8_f32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i8"))]
cast_fn!(i8_to_f64, f64, i8, kernel_f64, cast_i8_f64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u32"))]
cast_fn!(u32_to_f16, f16, u32, kernel_f16, cast_u32_f16);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u32"))]
cast_fn!(u32_to_f32, f32, u32, kernel_f32, cast_u32_f32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u32"))]
cast_fn!(u32_to_f64, f64, u32, kernel_f64, cast_u32_f64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u64"))]
cast_fn!(u64_to_f16, f16, u64, kernel_f16, cast_u64_f16);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u64"))]
cast_fn!(u64_to_f32, f32, u64, kernel_f32, cast_u64_f32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u64"))]
cast_fn!(u64_to_f64, f64, u64, kernel_f64, cast_u64_f64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i32"))]
cast_fn!(i32_to_f16, f16, i32, kernel_f16, cast_i32_f16);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i32"))]
cast_fn!(i32_to_f32, f32, i32, kernel_f32, cast_i32_f32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i32"))]
cast_fn!(i32_to_f64, f64, i32, kernel_f64, cast_i32_f64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i64"))]
cast_fn!(i64_to_f16, f16, i64, kernel_f16, cast_i64_f16);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i64"))]
cast_fn!(i64_to_f32, f32, i64, kernel_f32, cast_i64_f32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i64"))]
cast_fn!(i64_to_f64, f64, i64, kernel_f64, cast_i64_f64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
bool_cast_fn!(f16_to_bool, f16, kernel_f16, cast_f16_bool);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
bool_cast_fn!(f32_to_bool, f32, kernel_f32, cast_f32_bool);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
bool_cast_fn!(f64_to_bool, f64, kernel_f64, cast_f64_bool);

#[cfg(feature = "dtype_u8")]
bool_cast_fn!(u8_to_bool, u8, kernel_common, is_nonzero_bool_u8);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
bool_cast_fn!(i8_to_bool, i8, kernel_common, is_nonzero_bool_i8);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
bool_cast_fn!(u32_to_bool, u32, kernel_common, is_nonzero_bool_u32);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
bool_cast_fn!(i32_to_bool, i32, kernel_common, is_nonzero_bool_i32);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
bool_cast_fn!(u64_to_bool, u64, kernel_common, is_nonzero_bool_u64);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
bool_cast_fn!(i64_to_bool, i64, kernel_common, is_nonzero_bool_i64);

#[cfg(feature = "dtype_u8")]
raw_cast_fn!(u8_to_u8, u8, u8, cast_u8_u8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_i8"))]
raw_cast_fn!(u8_to_i8, i8, u8, cast_u8_i8);

#[cfg(all(feature = "dtype_u8", feature = "dtype_u32"))]
raw_cast_fn!(u8_to_u32, u32, u8, cast_u8_u32);

#[cfg(all(feature = "dtype_u8", feature = "dtype_i32"))]
raw_cast_fn!(u8_to_i32, i32, u8, cast_u8_i32);

#[cfg(all(feature = "dtype_u8", feature = "dtype_u64"))]
raw_cast_fn!(u8_to_u64, u64, u8, cast_u8_u64);

#[cfg(all(feature = "dtype_u8", feature = "dtype_i64"))]
raw_cast_fn!(u8_to_i64, i64, u8, cast_u8_i64);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
raw_cast_fn!(i8_to_u8, u8, i8, cast_i8_u8);

#[cfg(feature = "dtype_i8")]
raw_cast_fn!(i8_to_i8, i8, i8, cast_i8_i8);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u32"))]
raw_cast_fn!(i8_to_u32, u32, i8, cast_i8_u32);

#[cfg(all(feature = "dtype_i8", feature = "dtype_i32"))]
raw_cast_fn!(i8_to_i32, i32, i8, cast_i8_i32);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u64"))]
raw_cast_fn!(i8_to_u64, u64, i8, cast_i8_u64);

#[cfg(all(feature = "dtype_i8", feature = "dtype_i64"))]
raw_cast_fn!(i8_to_i64, i64, i8, cast_i8_i64);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
raw_cast_fn!(u32_to_u8, u8, u32, cast_u32_u8);

#[cfg(all(feature = "dtype_u32", feature = "dtype_i8"))]
raw_cast_fn!(u32_to_i8, i8, u32, cast_u32_i8);

#[cfg(feature = "dtype_u32")]
raw_cast_fn!(u32_to_u32, u32, u32, cast_u32_u32);

#[cfg(all(feature = "dtype_u32", feature = "dtype_i32"))]
raw_cast_fn!(u32_to_i32, i32, u32, cast_u32_i32);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u64"))]
raw_cast_fn!(u32_to_u64, u64, u32, cast_u32_u64);

#[cfg(all(feature = "dtype_u32", feature = "dtype_i64"))]
raw_cast_fn!(u32_to_i64, i64, u32, cast_u32_i64);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
raw_cast_fn!(i32_to_u8, u8, i32, cast_i32_u8);

#[cfg(all(feature = "dtype_i32", feature = "dtype_i8"))]
raw_cast_fn!(i32_to_i8, i8, i32, cast_i32_i8);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u32"))]
raw_cast_fn!(i32_to_u32, u32, i32, cast_i32_u32);

#[cfg(feature = "dtype_i32")]
raw_cast_fn!(i32_to_i32, i32, i32, cast_i32_i32);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u64"))]
raw_cast_fn!(i32_to_u64, u64, i32, cast_i32_u64);

#[cfg(all(feature = "dtype_i32", feature = "dtype_i64"))]
raw_cast_fn!(i32_to_i64, i64, i32, cast_i32_i64);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
raw_cast_fn!(u64_to_u8, u8, u64, cast_u64_u8);

#[cfg(all(feature = "dtype_u64", feature = "dtype_i8"))]
raw_cast_fn!(u64_to_i8, i8, u64, cast_u64_i8);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u32"))]
raw_cast_fn!(u64_to_u32, u32, u64, cast_u64_u32);

#[cfg(all(feature = "dtype_u64", feature = "dtype_i32"))]
raw_cast_fn!(u64_to_i32, i32, u64, cast_u64_i32);

#[cfg(feature = "dtype_u64")]
raw_cast_fn!(u64_to_u64, u64, u64, cast_u64_u64);

#[cfg(all(feature = "dtype_u64", feature = "dtype_i64"))]
raw_cast_fn!(u64_to_i64, i64, u64, cast_u64_i64);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
raw_cast_fn!(i64_to_u8, u8, i64, cast_i64_u8);

#[cfg(all(feature = "dtype_i64", feature = "dtype_i8"))]
raw_cast_fn!(i64_to_i8, i8, i64, cast_i64_i8);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u32"))]
raw_cast_fn!(i64_to_u32, u32, i64, cast_i64_u32);

#[cfg(all(feature = "dtype_i64", feature = "dtype_i32"))]
raw_cast_fn!(i64_to_i32, i32, i64, cast_i64_i32);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u64"))]
raw_cast_fn!(i64_to_u64, u64, i64, cast_i64_u64);

#[cfg(feature = "dtype_i64")]
raw_cast_fn!(i64_to_i64, i64, i64, cast_i64_i64);
