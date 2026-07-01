use std::sync::Arc;

#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

use crate::{
    cuda::cutile::{
        DeviceOpExt,
        kernel::common as kernel_common,
        utility::{VectorLaunch, checked_device_pointer},
    },
    error::Result,
};

macro_rules! fill_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            value: $ty,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { kernel_common::$kernel_fn(out, value, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! arange_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            start: $ty,
            step: $ty,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { kernel_common::$kernel_fn(out, start, step, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! constant_fill_fn {
    ($name:ident, $target:ident, $ty:ty, $value:expr) => {
        pub fn $name(stream: &Arc<Stream>, out: DevicePointer<$ty>, len: usize) -> Result<()> {
            $target(stream, out, $value, len)
        }
    };
}

#[cfg(feature = "dtype-f16")]
fill_fn!(fill_f16, f16, fill_f16);
#[cfg(feature = "dtype-f32")]
fill_fn!(fill_f32, f32, fill_f32);
#[cfg(feature = "dtype-f64")]
fill_fn!(fill_f64, f64, fill_f64);
#[cfg(feature = "dtype-u8")]
fill_fn!(fill_u8, u8, fill_u8);
#[cfg(feature = "dtype-i8")]
fill_fn!(fill_i8, i8, fill_i8);
#[cfg(feature = "dtype-u32")]
fill_fn!(fill_u32, u32, fill_u32);
#[cfg(feature = "dtype-u64")]
fill_fn!(fill_u64, u64, fill_u64);
#[cfg(feature = "dtype-i32")]
fill_fn!(fill_i32, i32, fill_i32);
#[cfg(feature = "dtype-i64")]
fill_fn!(fill_i64, i64, fill_i64);

#[cfg(feature = "dtype-f16")]
#[cfg(feature = "dtype-f32")]
constant_fill_fn!(zeros_f16, fill_f16, f16, f16::from_f32(0.0));
#[cfg(feature = "dtype-f16")]
#[cfg(feature = "dtype-f32")]
constant_fill_fn!(ones_f16, fill_f16, f16, f16::from_f32(1.0));
#[cfg(feature = "dtype-f32")]
constant_fill_fn!(zeros_f32, fill_f32, f32, 0.0f32);
#[cfg(feature = "dtype-f32")]
constant_fill_fn!(ones_f32, fill_f32, f32, 1.0f32);
#[cfg(feature = "dtype-f64")]
constant_fill_fn!(zeros_f64, fill_f64, f64, 0.0f64);
#[cfg(feature = "dtype-f64")]
constant_fill_fn!(ones_f64, fill_f64, f64, 1.0f64);
#[cfg(feature = "dtype-u8")]
constant_fill_fn!(zeros_u8, fill_u8, u8, 0u8);
#[cfg(feature = "dtype-u8")]
constant_fill_fn!(ones_u8, fill_u8, u8, 1u8);
#[cfg(feature = "dtype-i8")]
constant_fill_fn!(zeros_i8, fill_i8, i8, 0i8);
#[cfg(feature = "dtype-i8")]
constant_fill_fn!(ones_i8, fill_i8, i8, 1i8);
#[cfg(feature = "dtype-u32")]
constant_fill_fn!(zeros_u32, fill_u32, u32, 0u32);
#[cfg(feature = "dtype-u32")]
constant_fill_fn!(ones_u32, fill_u32, u32, 1u32);
#[cfg(feature = "dtype-i32")]
constant_fill_fn!(zeros_i32, fill_i32, i32, 0i32);
#[cfg(feature = "dtype-i32")]
constant_fill_fn!(ones_i32, fill_i32, i32, 1i32);
#[cfg(feature = "dtype-u64")]
constant_fill_fn!(zeros_u64, fill_u64, u64, 0u64);
#[cfg(feature = "dtype-u64")]
constant_fill_fn!(ones_u64, fill_u64, u64, 1u64);
#[cfg(feature = "dtype-i64")]
constant_fill_fn!(zeros_i64, fill_i64, i64, 0i64);
#[cfg(feature = "dtype-i64")]
constant_fill_fn!(ones_i64, fill_i64, i64, 1i64);

#[cfg(feature = "dtype-f32")]
arange_fn!(arange_f32, f32, arange_f32);
#[cfg(feature = "dtype-f64")]
arange_fn!(arange_f64, f64, arange_f64);
#[cfg(feature = "dtype-f16")]
arange_fn!(arange_f16, f16, arange_f16);
#[cfg(feature = "dtype-u8")]
arange_fn!(arange_u8, u8, arange_u8);
#[cfg(feature = "dtype-i8")]
arange_fn!(arange_i8, i8, arange_i8);
#[cfg(feature = "dtype-i32")]
arange_fn!(arange_i32, i32, arange_i32);
#[cfg(feature = "dtype-i64")]
arange_fn!(arange_i64, i64, arange_i64);
#[cfg(feature = "dtype-u32")]
arange_fn!(arange_u32, u32, arange_u32);
#[cfg(feature = "dtype-u64")]
arange_fn!(arange_u64, u64, arange_u64);

#[cfg(feature = "dtype-f16")]
arange_fn!(linspace_f16, f16, linspace_f16);
#[cfg(feature = "dtype-f32")]
arange_fn!(linspace_f32, f32, linspace_f32);
#[cfg(feature = "dtype-f64")]
arange_fn!(linspace_f64, f64, linspace_f64);
