use std::sync::Arc;

#[cfg(feature = "dtype-bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
use crate::cuda::cutile::kernel::f16::cast as kernel_f16;
#[cfg(feature = "dtype-f32")]
use crate::cuda::cutile::kernel::f32::cast as kernel_f32;
#[cfg(feature = "dtype-f64")]
use crate::cuda::cutile::kernel::f64::cast as kernel_f64;
use crate::{
    cuda::cutile::{
        DeviceOpExt,
        adapter::TensorAdapter,
        kernel::common as kernel_common,
        utility::{VectorLaunch, checked_device_pointer, vector_tile_size},
    },
    error::Result,
};

macro_rules! cast_fn {
    ($name:ident, $out_ty:ty, $in_ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$out_ty>,
            input: DevicePointer<$in_ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let tile = vector_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let input = TensorAdapter::contiguous_1d(input, len)?;
            $kernel::$kernel_fn(out, input).enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! bool_cast_fn {
    ($name:ident, $in_ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u8>,
            input: DevicePointer<$in_ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { $kernel::$kernel_fn(out, input, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! raw_cast_fn {
    ($name:ident, $out_ty:ty, $in_ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$out_ty>,
            input: DevicePointer<$in_ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { kernel_common::$kernel_fn(out, input, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
#[cfg(feature = "dtype-f32")]
cast_fn!(f16_to_f32, f32, f16, kernel_f16, cast_f16_f32);
#[cfg(feature = "dtype-bf16")]
#[cfg(feature = "dtype-f32")]
cast_fn!(bf16_to_f32, f32, bf16, kernel_f16, cast_bf16_f32);
#[cfg(feature = "dtype-f16")]
#[cfg(feature = "dtype-f64")]
cast_fn!(f16_to_f64, f64, f16, kernel_f16, cast_f16_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
cast_fn!(f16_to_u8, u8, f16, kernel_f16, cast_f16_u8);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i8"))]
cast_fn!(f16_to_i8, i8, f16, kernel_f16, cast_f16_i8);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u32"))]
cast_fn!(f16_to_u32, u32, f16, kernel_f16, cast_f16_u32);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i32"))]
cast_fn!(f16_to_i32, i32, f16, kernel_f16, cast_f16_i32);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u64"))]
cast_fn!(f16_to_u64, u64, f16, kernel_f16, cast_f16_u64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i64"))]
cast_fn!(f16_to_i64, i64, f16, kernel_f16, cast_f16_i64);
#[cfg(feature = "dtype-f16")]
#[cfg(feature = "dtype-f32")]
cast_fn!(f32_to_f16, f16, f32, kernel_f32, cast_f32_f16);
#[cfg(feature = "dtype-bf16")]
#[cfg(feature = "dtype-f32")]
cast_fn!(f32_to_bf16, bf16, f32, kernel_f32, cast_f32_bf16);
#[cfg(feature = "dtype-f64")]
#[cfg(feature = "dtype-f32")]
cast_fn!(f32_to_f64, f64, f32, kernel_f32, cast_f32_f64);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
cast_fn!(f32_to_u8, u8, f32, kernel_f32, cast_f32_u8);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
cast_fn!(f32_to_i8, i8, f32, kernel_f32, cast_f32_i8);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u32"))]
cast_fn!(f32_to_u32, u32, f32, kernel_f32, cast_f32_u32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
cast_fn!(f32_to_i32, i32, f32, kernel_f32, cast_f32_i32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u64"))]
cast_fn!(f32_to_u64, u64, f32, kernel_f32, cast_f32_u64);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i64"))]
cast_fn!(f32_to_i64, i64, f32, kernel_f32, cast_f32_i64);
#[cfg(feature = "dtype-f16")]
#[cfg(feature = "dtype-f64")]
cast_fn!(f64_to_f16, f16, f64, kernel_f64, cast_f64_f16);
#[cfg(feature = "dtype-f64")]
#[cfg(feature = "dtype-f32")]
cast_fn!(f64_to_f32, f32, f64, kernel_f64, cast_f64_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
cast_fn!(f64_to_u8, u8, f64, kernel_f64, cast_f64_u8);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i8"))]
cast_fn!(f64_to_i8, i8, f64, kernel_f64, cast_f64_i8);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u32"))]
cast_fn!(f64_to_u32, u32, f64, kernel_f64, cast_f64_u32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
cast_fn!(f64_to_i32, i32, f64, kernel_f64, cast_f64_i32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u64"))]
cast_fn!(f64_to_u64, u64, f64, kernel_f64, cast_f64_u64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i64"))]
cast_fn!(f64_to_i64, i64, f64, kernel_f64, cast_f64_i64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
cast_fn!(bool_to_f16, f16, u8, kernel_f16, cast_bool_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
cast_fn!(bool_to_f32, f32, u8, kernel_f32, cast_bool_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
cast_fn!(bool_to_f64, f64, u8, kernel_f64, cast_bool_f64);
#[cfg(feature = "dtype-u8")]
raw_cast_fn!(bool_to_u8, u8, u8, cast_bool_u8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-i8"))]
raw_cast_fn!(bool_to_i8, i8, u8, cast_bool_i8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-u32"))]
raw_cast_fn!(bool_to_u32, u32, u8, cast_bool_u32);
#[cfg(all(feature = "dtype-u8", feature = "dtype-i32"))]
raw_cast_fn!(bool_to_i32, i32, u8, cast_bool_i32);
#[cfg(all(feature = "dtype-u8", feature = "dtype-u64"))]
raw_cast_fn!(bool_to_u64, u64, u8, cast_bool_u64);
#[cfg(all(feature = "dtype-u8", feature = "dtype-i64"))]
raw_cast_fn!(bool_to_i64, i64, u8, cast_bool_i64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
cast_fn!(u8_to_f16, f16, u8, kernel_f16, cast_u8_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
cast_fn!(u8_to_f32, f32, u8, kernel_f32, cast_u8_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
cast_fn!(u8_to_f64, f64, u8, kernel_f64, cast_u8_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i8"))]
cast_fn!(i8_to_f16, f16, i8, kernel_f16, cast_i8_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
cast_fn!(i8_to_f32, f32, i8, kernel_f32, cast_i8_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i8"))]
cast_fn!(i8_to_f64, f64, i8, kernel_f64, cast_i8_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u32"))]
cast_fn!(u32_to_f16, f16, u32, kernel_f16, cast_u32_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u32"))]
cast_fn!(u32_to_f32, f32, u32, kernel_f32, cast_u32_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u32"))]
cast_fn!(u32_to_f64, f64, u32, kernel_f64, cast_u32_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u64"))]
cast_fn!(u64_to_f16, f16, u64, kernel_f16, cast_u64_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u64"))]
cast_fn!(u64_to_f32, f32, u64, kernel_f32, cast_u64_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u64"))]
cast_fn!(u64_to_f64, f64, u64, kernel_f64, cast_u64_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i32"))]
cast_fn!(i32_to_f16, f16, i32, kernel_f16, cast_i32_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
cast_fn!(i32_to_f32, f32, i32, kernel_f32, cast_i32_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
cast_fn!(i32_to_f64, f64, i32, kernel_f64, cast_i32_f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i64"))]
cast_fn!(i64_to_f16, f16, i64, kernel_f16, cast_i64_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i64"))]
cast_fn!(i64_to_f32, f32, i64, kernel_f32, cast_i64_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i64"))]
cast_fn!(i64_to_f64, f64, i64, kernel_f64, cast_i64_f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
bool_cast_fn!(f16_to_bool, f16, kernel_f16, cast_f16_bool);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
bool_cast_fn!(f32_to_bool, f32, kernel_f32, cast_f32_bool);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
bool_cast_fn!(f64_to_bool, f64, kernel_f64, cast_f64_bool);
#[cfg(feature = "dtype-u8")]
bool_cast_fn!(u8_to_bool, u8, kernel_common, is_nonzero_bool_u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
bool_cast_fn!(i8_to_bool, i8, kernel_common, is_nonzero_bool_i8);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
bool_cast_fn!(u32_to_bool, u32, kernel_common, is_nonzero_bool_u32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
bool_cast_fn!(i32_to_bool, i32, kernel_common, is_nonzero_bool_i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
bool_cast_fn!(u64_to_bool, u64, kernel_common, is_nonzero_bool_u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
bool_cast_fn!(i64_to_bool, i64, kernel_common, is_nonzero_bool_i64);

#[cfg(feature = "dtype-u8")]
raw_cast_fn!(u8_to_u8, u8, u8, cast_u8_u8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-i8"))]
raw_cast_fn!(u8_to_i8, i8, u8, cast_u8_i8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-u32"))]
raw_cast_fn!(u8_to_u32, u32, u8, cast_u8_u32);
#[cfg(all(feature = "dtype-u8", feature = "dtype-i32"))]
raw_cast_fn!(u8_to_i32, i32, u8, cast_u8_i32);
#[cfg(all(feature = "dtype-u8", feature = "dtype-u64"))]
raw_cast_fn!(u8_to_u64, u64, u8, cast_u8_u64);
#[cfg(all(feature = "dtype-u8", feature = "dtype-i64"))]
raw_cast_fn!(u8_to_i64, i64, u8, cast_u8_i64);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
raw_cast_fn!(i8_to_u8, u8, i8, cast_i8_u8);
#[cfg(feature = "dtype-i8")]
raw_cast_fn!(i8_to_i8, i8, i8, cast_i8_i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u32"))]
raw_cast_fn!(i8_to_u32, u32, i8, cast_i8_u32);
#[cfg(all(feature = "dtype-i8", feature = "dtype-i32"))]
raw_cast_fn!(i8_to_i32, i32, i8, cast_i8_i32);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u64"))]
raw_cast_fn!(i8_to_u64, u64, i8, cast_i8_u64);
#[cfg(all(feature = "dtype-i8", feature = "dtype-i64"))]
raw_cast_fn!(i8_to_i64, i64, i8, cast_i8_i64);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
raw_cast_fn!(u32_to_u8, u8, u32, cast_u32_u8);
#[cfg(all(feature = "dtype-u32", feature = "dtype-i8"))]
raw_cast_fn!(u32_to_i8, i8, u32, cast_u32_i8);
#[cfg(feature = "dtype-u32")]
raw_cast_fn!(u32_to_u32, u32, u32, cast_u32_u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-i32"))]
raw_cast_fn!(u32_to_i32, i32, u32, cast_u32_i32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u64"))]
raw_cast_fn!(u32_to_u64, u64, u32, cast_u32_u64);
#[cfg(all(feature = "dtype-u32", feature = "dtype-i64"))]
raw_cast_fn!(u32_to_i64, i64, u32, cast_u32_i64);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
raw_cast_fn!(i32_to_u8, u8, i32, cast_i32_u8);
#[cfg(all(feature = "dtype-i32", feature = "dtype-i8"))]
raw_cast_fn!(i32_to_i8, i8, i32, cast_i32_i8);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u32"))]
raw_cast_fn!(i32_to_u32, u32, i32, cast_i32_u32);
#[cfg(feature = "dtype-i32")]
raw_cast_fn!(i32_to_i32, i32, i32, cast_i32_i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u64"))]
raw_cast_fn!(i32_to_u64, u64, i32, cast_i32_u64);
#[cfg(all(feature = "dtype-i32", feature = "dtype-i64"))]
raw_cast_fn!(i32_to_i64, i64, i32, cast_i32_i64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
raw_cast_fn!(u64_to_u8, u8, u64, cast_u64_u8);
#[cfg(all(feature = "dtype-u64", feature = "dtype-i8"))]
raw_cast_fn!(u64_to_i8, i8, u64, cast_u64_i8);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u32"))]
raw_cast_fn!(u64_to_u32, u32, u64, cast_u64_u32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-i32"))]
raw_cast_fn!(u64_to_i32, i32, u64, cast_u64_i32);
#[cfg(feature = "dtype-u64")]
raw_cast_fn!(u64_to_u64, u64, u64, cast_u64_u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-i64"))]
raw_cast_fn!(u64_to_i64, i64, u64, cast_u64_i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
raw_cast_fn!(i64_to_u8, u8, i64, cast_i64_u8);
#[cfg(all(feature = "dtype-i64", feature = "dtype-i8"))]
raw_cast_fn!(i64_to_i8, i8, i64, cast_i64_i8);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u32"))]
raw_cast_fn!(i64_to_u32, u32, i64, cast_i64_u32);
#[cfg(all(feature = "dtype-i64", feature = "dtype-i32"))]
raw_cast_fn!(i64_to_i32, i32, i64, cast_i64_i32);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u64"))]
raw_cast_fn!(i64_to_u64, u64, i64, cast_i64_u64);
#[cfg(feature = "dtype-i64")]
raw_cast_fn!(i64_to_i64, i64, i64, cast_i64_i64);
