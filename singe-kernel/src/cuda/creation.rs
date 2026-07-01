//! Fill, zeros, ones, arange, and other device initialization kernels.

use std::sync::Arc;

#[cfg(feature = "cutile")]
use ::cutile::cuda_async::device_buffer::DevicePointer;
#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceRepr, DeviceSliceMut},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{borrowed_stream, output_pointer},
    error::Result,
};

macro_rules! fill_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            value: $ty,
        ) -> Result<()> {
            fill(stream, out, value, cutile::creation::$name)
        }
    };
}

macro_rules! arange_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            start: $ty,
            step: $ty,
        ) -> Result<()> {
            arange(stream, out, start, step, cutile::creation::$name)
        }
    };
}

macro_rules! constant_fill_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(stream: &Stream, out: &mut impl DeviceSliceMut<$ty>) -> Result<()> {
            let stream = borrowed_stream(stream)?;
            cutile::creation::$name(&stream, output_pointer(out), out.len())
        }
    };
}

#[cfg(feature = "dtype-f16")]
fill_fn!(fill_f16, f16);
#[cfg(feature = "dtype-f32")]
fill_fn!(fill_f32, f32);
#[cfg(feature = "dtype-f64")]
fill_fn!(fill_f64, f64);
#[cfg(feature = "dtype-u8")]
fill_fn!(fill_u8, u8);
#[cfg(feature = "dtype-i8")]
fill_fn!(fill_i8, i8);
#[cfg(feature = "dtype-u32")]
fill_fn!(fill_u32, u32);
#[cfg(feature = "dtype-u64")]
fill_fn!(fill_u64, u64);
#[cfg(feature = "dtype-i32")]
fill_fn!(fill_i32, i32);
#[cfg(feature = "dtype-i64")]
fill_fn!(fill_i64, i64);

#[cfg(feature = "dtype-f16")]
constant_fill_fn!(zeros_f16, f16);
#[cfg(feature = "dtype-f16")]
constant_fill_fn!(ones_f16, f16);
#[cfg(feature = "dtype-f32")]
constant_fill_fn!(zeros_f32, f32);
#[cfg(feature = "dtype-f32")]
constant_fill_fn!(ones_f32, f32);
#[cfg(feature = "dtype-f64")]
constant_fill_fn!(zeros_f64, f64);
#[cfg(feature = "dtype-f64")]
constant_fill_fn!(ones_f64, f64);
#[cfg(feature = "dtype-u8")]
constant_fill_fn!(zeros_u8, u8);
#[cfg(feature = "dtype-u8")]
constant_fill_fn!(ones_u8, u8);
#[cfg(feature = "dtype-i8")]
constant_fill_fn!(zeros_i8, i8);
#[cfg(feature = "dtype-i8")]
constant_fill_fn!(ones_i8, i8);
#[cfg(feature = "dtype-u32")]
constant_fill_fn!(zeros_u32, u32);
#[cfg(feature = "dtype-u32")]
constant_fill_fn!(ones_u32, u32);
#[cfg(feature = "dtype-i32")]
constant_fill_fn!(zeros_i32, i32);
#[cfg(feature = "dtype-i32")]
constant_fill_fn!(ones_i32, i32);
#[cfg(feature = "dtype-u64")]
constant_fill_fn!(zeros_u64, u64);
#[cfg(feature = "dtype-u64")]
constant_fill_fn!(ones_u64, u64);
#[cfg(feature = "dtype-i64")]
constant_fill_fn!(zeros_i64, i64);
#[cfg(feature = "dtype-i64")]
constant_fill_fn!(ones_i64, i64);

#[cfg(feature = "dtype-f32")]
arange_fn!(arange_f32, f32);
#[cfg(feature = "dtype-f64")]
arange_fn!(arange_f64, f64);
#[cfg(feature = "dtype-f16")]
arange_fn!(arange_f16, f16);
#[cfg(feature = "dtype-u8")]
arange_fn!(arange_u8, u8);
#[cfg(feature = "dtype-i8")]
arange_fn!(arange_i8, i8);
#[cfg(feature = "dtype-i32")]
arange_fn!(arange_i32, i32);
#[cfg(feature = "dtype-i64")]
arange_fn!(arange_i64, i64);
#[cfg(feature = "dtype-u32")]
arange_fn!(arange_u32, u32);
#[cfg(feature = "dtype-u64")]
arange_fn!(arange_u64, u64);

#[cfg(feature = "dtype-f16")]
arange_fn!(linspace_f16, f16);
#[cfg(feature = "dtype-f32")]
arange_fn!(linspace_f32, f32);
#[cfg(feature = "dtype-f64")]
arange_fn!(linspace_f64, f64);

fn fill<T, F>(stream: &Stream, out: &mut impl DeviceSliceMut<T>, value: T, launch: F) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(&Arc<::cutile::cuda_core::Stream>, DevicePointer<T>, T, usize) -> Result<()>,
{
    let stream = borrowed_stream(stream)?;
    launch(&stream, output_pointer(out), value, out.len())
}

fn arange<T, F>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<T>,
    start: T,
    step: T,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(&Arc<::cutile::cuda_core::Stream>, DevicePointer<T>, T, T, usize) -> Result<()>,
{
    let stream = borrowed_stream(stream)?;
    launch(&stream, output_pointer(out), start, step, out.len())
}
