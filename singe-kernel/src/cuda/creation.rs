//! Fill, zeros, ones, arange, and other device initialization kernels.

use std::sync::Arc;

#[cfg(feature = "cutile")]
use ::cutile::cuda_async::device_buffer::DevicePointer;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceOutput, DeviceRepr},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::error::Result;

macro_rules! fill_fn {
    ($operation:literal, $name:ident, $ty:ty) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Retain exclusive access to the destination in the stream's context through
        /// completion on every path and throughout graph replay.
        /// Order prior accesses before launch. Every destination element is written.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            value: $ty,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            fill(stream, out, value, |arg0, arg1, arg2, arg3| unsafe {
                cutile::creation::$name(arg0, arg1, arg2, arg3)
            })
        }
    };
}

macro_rules! arange_fn {
    ($operation:literal, $name:ident, $ty:ty) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Retain exclusive access to the destination in the stream's context through
        /// completion on every path and throughout graph replay.
        /// Order prior accesses before launch. Every destination element is written.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            start: $ty,
            step: $ty,
        ) -> Result<()> {
            arange(
                stream,
                out,
                start,
                step,
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4| unsafe {
                    cutile::creation::$name(arg0, arg1, arg2, arg3, arg4)
                },
            )
        }
    };
}

macro_rules! constant_fill_fn {
    ($operation:literal, $name:ident, $ty:ty) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Retain exclusive access to the destination in the stream's context through
        /// completion on every path and throughout graph replay.
        /// Order prior accesses before launch. Every destination element is written.
        pub unsafe fn $name(stream: &Stream, out: &mut impl DeviceOutput<$ty>) -> Result<()> {
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::creation::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    out.len(),
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f16")]
fill_fn!("Fills every output element with `value`.", fill_f16, f16);

#[cfg(feature = "dtype_f32")]
fill_fn!("Fills every output element with `value`.", fill_f32, f32);

#[cfg(feature = "dtype_f64")]
fill_fn!("Fills every output element with `value`.", fill_f64, f64);

#[cfg(feature = "dtype_u8")]
fill_fn!("Fills every output element with `value`.", fill_u8, u8);

#[cfg(feature = "dtype_i8")]
fill_fn!("Fills every output element with `value`.", fill_i8, i8);

#[cfg(feature = "dtype_u32")]
fill_fn!("Fills every output element with `value`.", fill_u32, u32);

#[cfg(feature = "dtype_u64")]
fill_fn!("Fills every output element with `value`.", fill_u64, u64);

#[cfg(feature = "dtype_i32")]
fill_fn!("Fills every output element with `value`.", fill_i32, i32);

#[cfg(feature = "dtype_i64")]
fill_fn!("Fills every output element with `value`.", fill_i64, i64);

#[cfg(feature = "dtype_f16")]
constant_fill_fn!("Fills every output element with zero.", zeros_f16, f16);

#[cfg(feature = "dtype_f16")]
constant_fill_fn!("Fills every output element with one.", ones_f16, f16);

#[cfg(feature = "dtype_f32")]
constant_fill_fn!("Fills every output element with zero.", zeros_f32, f32);

#[cfg(feature = "dtype_f32")]
constant_fill_fn!("Fills every output element with one.", ones_f32, f32);

#[cfg(feature = "dtype_f64")]
constant_fill_fn!("Fills every output element with zero.", zeros_f64, f64);

#[cfg(feature = "dtype_f64")]
constant_fill_fn!("Fills every output element with one.", ones_f64, f64);

#[cfg(feature = "dtype_u8")]
constant_fill_fn!("Fills every output element with zero.", zeros_u8, u8);

#[cfg(feature = "dtype_u8")]
constant_fill_fn!("Fills every output element with one.", ones_u8, u8);

#[cfg(feature = "dtype_i8")]
constant_fill_fn!("Fills every output element with zero.", zeros_i8, i8);

#[cfg(feature = "dtype_i8")]
constant_fill_fn!("Fills every output element with one.", ones_i8, i8);

#[cfg(feature = "dtype_u32")]
constant_fill_fn!("Fills every output element with zero.", zeros_u32, u32);

#[cfg(feature = "dtype_u32")]
constant_fill_fn!("Fills every output element with one.", ones_u32, u32);

#[cfg(feature = "dtype_i32")]
constant_fill_fn!("Fills every output element with zero.", zeros_i32, i32);

#[cfg(feature = "dtype_i32")]
constant_fill_fn!("Fills every output element with one.", ones_i32, i32);

#[cfg(feature = "dtype_u64")]
constant_fill_fn!("Fills every output element with zero.", zeros_u64, u64);

#[cfg(feature = "dtype_u64")]
constant_fill_fn!("Fills every output element with one.", ones_u64, u64);

#[cfg(feature = "dtype_i64")]
constant_fill_fn!("Fills every output element with zero.", zeros_i64, i64);

#[cfg(feature = "dtype_i64")]
constant_fill_fn!("Fills every output element with one.", ones_i64, i64);

#[cfg(feature = "dtype_f32")]
arange_fn!(
    "Fills `start + index * step` over the output range.",
    arange_f32,
    f32
);

#[cfg(feature = "dtype_f64")]
arange_fn!(
    "Fills `start + index * step` over the output range.",
    arange_f64,
    f64
);

#[cfg(feature = "dtype_f16")]
arange_fn!(
    "Fills `start + index * step` over the output range.",
    arange_f16,
    f16
);

#[cfg(feature = "dtype_u8")]
arange_fn!(
    "Fills `start + index * step` over the output range.",
    arange_u8,
    u8
);

#[cfg(feature = "dtype_i8")]
arange_fn!(
    "Fills `start + index * step` over the output range.",
    arange_i8,
    i8
);

#[cfg(feature = "dtype_i32")]
arange_fn!(
    "Fills `start + index * step` over the output range.",
    arange_i32,
    i32
);

#[cfg(feature = "dtype_i64")]
arange_fn!(
    "Fills `start + index * step` over the output range.",
    arange_i64,
    i64
);

#[cfg(feature = "dtype_u32")]
arange_fn!(
    "Fills `start + index * step` over the output range.",
    arange_u32,
    u32
);

#[cfg(feature = "dtype_u64")]
arange_fn!(
    "Fills `start + index * step` over the output range.",
    arange_u64,
    u64
);

#[cfg(feature = "dtype_f16")]
arange_fn!(
    "Fills `start + index * step` over the output range.",
    linspace_f16,
    f16
);

#[cfg(feature = "dtype_f32")]
arange_fn!(
    "Fills `start + index * step` over the output range.",
    linspace_f32,
    f32
);

#[cfg(feature = "dtype_f64")]
arange_fn!(
    "Fills `start + index * step` over the output range.",
    linspace_f64,
    f64
);

fn fill<T, F>(stream: &Stream, out: &mut impl DeviceOutput<T>, value: T, launch: F) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(&Arc<::cutile::cuda_core::Stream>, DevicePointer<T>, T, usize) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        value,
        out.len(),
    )
}

fn arange<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    start: T,
    step: T,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(&Arc<::cutile::cuda_core::Stream>, DevicePointer<T>, T, T, usize) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        start,
        step,
        out.len(),
    )
}
