use std::sync::Arc;

#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

#[cfg(feature = "dtype-f16")]
use crate::cuda::cutile::kernel::f16::scalar as kernel_f16;
#[cfg(feature = "dtype-f32")]
use crate::cuda::cutile::kernel::f32::scalar as kernel_f32;
#[cfg(feature = "dtype-f64")]
use crate::cuda::cutile::kernel::f64::scalar as kernel_f64;
use crate::{
    cuda::cutile::{
        DeviceOpExt,
        adapter::TensorAdapter,
        kernel::common as kernel_common,
        utility::{VectorLaunch, checked_device_pointer, vector_tile_size},
    },
    error::Result,
};

macro_rules! scalar_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            scalar: $ty,
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
            $kernel::$kernel_fn(out, input, scalar).enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! scalar_bool_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u8>,
            input: DevicePointer<$ty>,
            scalar: $ty,
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
            $kernel::$kernel_fn(out, input, scalar).enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! scalar_fns_for_type {
    ($ty:ty, $kernel:ident, $(
        $name:ident => $kernel_fn:ident
    ),* $(,)?) => {
        $(scalar_fn!($name, $ty, $kernel, $kernel_fn);)*
    };
}

macro_rules! scalar_bool_fns_for_type {
    ($ty:ty, $kernel:ident, $(
        $name:ident => $kernel_fn:ident
    ),* $(,)?) => {
        $(scalar_bool_fn!($name, $ty, $kernel, $kernel_fn);)*
    };
}

#[cfg(feature = "dtype-f32")]
scalar_fns_for_type!(f32, kernel_f32,
    add_scalar_f32 => add_scalar_f32,
    sub_scalar_f32 => sub_scalar_f32,
    rsub_scalar_f32 => rsub_scalar_f32,
    mul_scalar_f32 => scale_f32,
    scale_f32 => scale_f32,
    div_scalar_f32 => div_scalar_f32,
    rdiv_scalar_f32 => rdiv_scalar_f32,
    modulo_scalar_f32 => modulo_scalar_f32,
    rmodulo_scalar_f32 => rmodulo_scalar_f32,
    atan2_scalar_f32 => atan2_scalar_f32,
    ratan2_scalar_f32 => ratan2_scalar_f32,
    pow_scalar_f32 => pow_scalar_f32,
    rpow_scalar_f32 => rpow_scalar_f32,
    hypot_scalar_f32 => hypot_scalar_f32,
    squared_difference_scalar_f32 => squared_difference_scalar_f32,
    logaddexp_scalar_f32 => logaddexp_scalar_f32,
    xlogy_scalar_f32 => xlogy_scalar_f32,
    rxlogy_scalar_f32 => rxlogy_scalar_f32,
    min_scalar_f32 => min_scalar_f32,
    max_scalar_f32 => max_scalar_f32,
);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
scalar_bool_fns_for_type!(f32, kernel_f32,
    equal_scalar_bool_f32 => equal_scalar_bool_f32,
    not_equal_scalar_bool_f32 => not_equal_scalar_bool_f32,
    less_scalar_bool_f32 => less_scalar_bool_f32,
    less_equal_scalar_bool_f32 => less_equal_scalar_bool_f32,
    greater_scalar_bool_f32 => greater_scalar_bool_f32,
    greater_equal_scalar_bool_f32 => greater_equal_scalar_bool_f32,
);

#[cfg(feature = "dtype-f16")]
scalar_fns_for_type!(f16, kernel_f16,
    add_scalar_f16 => add_scalar_f16,
    sub_scalar_f16 => sub_scalar_f16,
    rsub_scalar_f16 => rsub_scalar_f16,
    mul_scalar_f16 => scale_f16,
    scale_f16 => scale_f16,
    div_scalar_f16 => div_scalar_f16,
    rdiv_scalar_f16 => rdiv_scalar_f16,
    modulo_scalar_f16 => modulo_scalar_f16,
    rmodulo_scalar_f16 => rmodulo_scalar_f16,
    atan2_scalar_f16 => atan2_scalar_f16,
    ratan2_scalar_f16 => ratan2_scalar_f16,
    pow_scalar_f16 => pow_scalar_f16,
    rpow_scalar_f16 => rpow_scalar_f16,
    hypot_scalar_f16 => hypot_scalar_f16,
    squared_difference_scalar_f16 => squared_difference_scalar_f16,
    logaddexp_scalar_f16 => logaddexp_scalar_f16,
    xlogy_scalar_f16 => xlogy_scalar_f16,
    rxlogy_scalar_f16 => rxlogy_scalar_f16,
    min_scalar_f16 => min_scalar_f16,
    max_scalar_f16 => max_scalar_f16,
);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
scalar_bool_fns_for_type!(f16, kernel_f16,
    equal_scalar_bool_f16 => equal_scalar_bool_f16,
    not_equal_scalar_bool_f16 => not_equal_scalar_bool_f16,
    less_scalar_bool_f16 => less_scalar_bool_f16,
    less_equal_scalar_bool_f16 => less_equal_scalar_bool_f16,
    greater_scalar_bool_f16 => greater_scalar_bool_f16,
    greater_equal_scalar_bool_f16 => greater_equal_scalar_bool_f16,
);

#[cfg(feature = "dtype-f64")]
scalar_fns_for_type!(f64, kernel_f64,
    add_scalar_f64 => add_scalar_f64,
    sub_scalar_f64 => sub_scalar_f64,
    rsub_scalar_f64 => rsub_scalar_f64,
    mul_scalar_f64 => scale_f64,
    scale_f64 => scale_f64,
    div_scalar_f64 => div_scalar_f64,
    rdiv_scalar_f64 => rdiv_scalar_f64,
    modulo_scalar_f64 => modulo_scalar_f64,
    rmodulo_scalar_f64 => rmodulo_scalar_f64,
    atan2_scalar_f64 => atan2_scalar_f64,
    ratan2_scalar_f64 => ratan2_scalar_f64,
    pow_scalar_f64 => pow_scalar_f64,
    rpow_scalar_f64 => rpow_scalar_f64,
    hypot_scalar_f64 => hypot_scalar_f64,
    squared_difference_scalar_f64 => squared_difference_scalar_f64,
    logaddexp_scalar_f64 => logaddexp_scalar_f64,
    xlogy_scalar_f64 => xlogy_scalar_f64,
    rxlogy_scalar_f64 => rxlogy_scalar_f64,
    min_scalar_f64 => min_scalar_f64,
    max_scalar_f64 => max_scalar_f64,
);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
scalar_bool_fns_for_type!(f64, kernel_f64,
    equal_scalar_bool_f64 => equal_scalar_bool_f64,
    not_equal_scalar_bool_f64 => not_equal_scalar_bool_f64,
    less_scalar_bool_f64 => less_scalar_bool_f64,
    less_equal_scalar_bool_f64 => less_equal_scalar_bool_f64,
    greater_scalar_bool_f64 => greater_scalar_bool_f64,
    greater_equal_scalar_bool_f64 => greater_equal_scalar_bool_f64,
);

macro_rules! scalar_bool_common_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u8>,
            input: DevicePointer<$ty>,
            scalar: $ty,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { kernel_common::$kernel_fn(out, input, scalar, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-u8")]
scalar_bool_common_fn!(equal_scalar_bool_u8, u8, equal_scalar_bool_u8);
#[cfg(feature = "dtype-u8")]
scalar_bool_common_fn!(not_equal_scalar_bool_u8, u8, not_equal_scalar_bool_u8);
#[cfg(feature = "dtype-u8")]
scalar_bool_common_fn!(less_scalar_bool_u8, u8, less_scalar_bool_u8);
#[cfg(feature = "dtype-u8")]
scalar_bool_common_fn!(less_equal_scalar_bool_u8, u8, less_equal_scalar_bool_u8);
#[cfg(feature = "dtype-u8")]
scalar_bool_common_fn!(greater_scalar_bool_u8, u8, greater_scalar_bool_u8);
#[cfg(feature = "dtype-u8")]
scalar_bool_common_fn!(
    greater_equal_scalar_bool_u8,
    u8,
    greater_equal_scalar_bool_u8
);

#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
scalar_bool_common_fn!(equal_scalar_bool_i8, i8, equal_scalar_bool_i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
scalar_bool_common_fn!(not_equal_scalar_bool_i8, i8, not_equal_scalar_bool_i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
scalar_bool_common_fn!(less_scalar_bool_i8, i8, less_scalar_bool_i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
scalar_bool_common_fn!(less_equal_scalar_bool_i8, i8, less_equal_scalar_bool_i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
scalar_bool_common_fn!(greater_scalar_bool_i8, i8, greater_scalar_bool_i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
scalar_bool_common_fn!(
    greater_equal_scalar_bool_i8,
    i8,
    greater_equal_scalar_bool_i8
);

#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
scalar_bool_common_fn!(equal_scalar_bool_u32, u32, equal_scalar_bool_u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
scalar_bool_common_fn!(not_equal_scalar_bool_u32, u32, not_equal_scalar_bool_u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
scalar_bool_common_fn!(less_scalar_bool_u32, u32, less_scalar_bool_u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
scalar_bool_common_fn!(less_equal_scalar_bool_u32, u32, less_equal_scalar_bool_u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
scalar_bool_common_fn!(greater_scalar_bool_u32, u32, greater_scalar_bool_u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
scalar_bool_common_fn!(
    greater_equal_scalar_bool_u32,
    u32,
    greater_equal_scalar_bool_u32
);

#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
scalar_bool_common_fn!(equal_scalar_bool_i32, i32, equal_scalar_bool_i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
scalar_bool_common_fn!(not_equal_scalar_bool_i32, i32, not_equal_scalar_bool_i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
scalar_bool_common_fn!(less_scalar_bool_i32, i32, less_scalar_bool_i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
scalar_bool_common_fn!(less_equal_scalar_bool_i32, i32, less_equal_scalar_bool_i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
scalar_bool_common_fn!(greater_scalar_bool_i32, i32, greater_scalar_bool_i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
scalar_bool_common_fn!(
    greater_equal_scalar_bool_i32,
    i32,
    greater_equal_scalar_bool_i32
);

#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
scalar_bool_common_fn!(equal_scalar_bool_u64, u64, equal_scalar_bool_u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
scalar_bool_common_fn!(not_equal_scalar_bool_u64, u64, not_equal_scalar_bool_u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
scalar_bool_common_fn!(less_scalar_bool_u64, u64, less_scalar_bool_u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
scalar_bool_common_fn!(less_equal_scalar_bool_u64, u64, less_equal_scalar_bool_u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
scalar_bool_common_fn!(greater_scalar_bool_u64, u64, greater_scalar_bool_u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
scalar_bool_common_fn!(
    greater_equal_scalar_bool_u64,
    u64,
    greater_equal_scalar_bool_u64
);

#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
scalar_bool_common_fn!(equal_scalar_bool_i64, i64, equal_scalar_bool_i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
scalar_bool_common_fn!(not_equal_scalar_bool_i64, i64, not_equal_scalar_bool_i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
scalar_bool_common_fn!(less_scalar_bool_i64, i64, less_scalar_bool_i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
scalar_bool_common_fn!(less_equal_scalar_bool_i64, i64, less_equal_scalar_bool_i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
scalar_bool_common_fn!(greater_scalar_bool_i64, i64, greater_scalar_bool_i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
scalar_bool_common_fn!(
    greater_equal_scalar_bool_i64,
    i64,
    greater_equal_scalar_bool_i64
);

macro_rules! scalar_common_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            scalar: $ty,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { kernel_common::$kernel_fn(out, input, scalar, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-u32")]
scalar_common_fn!(min_scalar_u32, u32, min_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(max_scalar_u32, u32, max_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(add_scalar_u32, u32, add_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(sub_scalar_u32, u32, sub_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(rsub_scalar_u32, u32, rsub_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(mul_scalar_u32, u32, mul_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(div_scalar_u32, u32, div_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(rdiv_scalar_u32, u32, rdiv_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(modulo_scalar_u32, u32, modulo_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(rmodulo_scalar_u32, u32, rmodulo_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(bitwise_and_scalar_u32, u32, bitwise_and_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(bitwise_or_scalar_u32, u32, bitwise_or_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(bitwise_xor_scalar_u32, u32, bitwise_xor_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(shift_left_scalar_u32, u32, shift_left_scalar_u32);
#[cfg(feature = "dtype-u32")]
scalar_common_fn!(shift_right_scalar_u32, u32, shift_right_scalar_u32);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(min_scalar_u8, u8, min_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(max_scalar_u8, u8, max_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(add_scalar_u8, u8, add_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(sub_scalar_u8, u8, sub_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(rsub_scalar_u8, u8, rsub_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(mul_scalar_u8, u8, mul_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(div_scalar_u8, u8, div_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(rdiv_scalar_u8, u8, rdiv_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(modulo_scalar_u8, u8, modulo_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(rmodulo_scalar_u8, u8, rmodulo_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(bitwise_and_scalar_u8, u8, bitwise_and_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(bitwise_or_scalar_u8, u8, bitwise_or_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(bitwise_xor_scalar_u8, u8, bitwise_xor_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(shift_left_scalar_u8, u8, shift_left_scalar_u8);
#[cfg(feature = "dtype-u8")]
scalar_common_fn!(shift_right_scalar_u8, u8, shift_right_scalar_u8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(min_scalar_i8, i8, min_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(max_scalar_i8, i8, max_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(add_scalar_i8, i8, add_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(sub_scalar_i8, i8, sub_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(rsub_scalar_i8, i8, rsub_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(mul_scalar_i8, i8, mul_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(div_scalar_i8, i8, div_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(rdiv_scalar_i8, i8, rdiv_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(modulo_scalar_i8, i8, modulo_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(rmodulo_scalar_i8, i8, rmodulo_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(bitwise_and_scalar_i8, i8, bitwise_and_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(bitwise_or_scalar_i8, i8, bitwise_or_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(bitwise_xor_scalar_i8, i8, bitwise_xor_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(shift_left_scalar_i8, i8, shift_left_scalar_i8);
#[cfg(feature = "dtype-i8")]
scalar_common_fn!(shift_right_scalar_i8, i8, shift_right_scalar_i8);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(min_scalar_i32, i32, min_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(max_scalar_i32, i32, max_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(add_scalar_i32, i32, add_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(sub_scalar_i32, i32, sub_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(rsub_scalar_i32, i32, rsub_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(mul_scalar_i32, i32, mul_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(div_scalar_i32, i32, div_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(rdiv_scalar_i32, i32, rdiv_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(modulo_scalar_i32, i32, modulo_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(rmodulo_scalar_i32, i32, rmodulo_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(bitwise_and_scalar_i32, i32, bitwise_and_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(bitwise_or_scalar_i32, i32, bitwise_or_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(bitwise_xor_scalar_i32, i32, bitwise_xor_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(shift_left_scalar_i32, i32, shift_left_scalar_i32);
#[cfg(feature = "dtype-i32")]
scalar_common_fn!(shift_right_scalar_i32, i32, shift_right_scalar_i32);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(min_scalar_u64, u64, min_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(max_scalar_u64, u64, max_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(add_scalar_u64, u64, add_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(sub_scalar_u64, u64, sub_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(rsub_scalar_u64, u64, rsub_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(mul_scalar_u64, u64, mul_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(div_scalar_u64, u64, div_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(rdiv_scalar_u64, u64, rdiv_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(modulo_scalar_u64, u64, modulo_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(rmodulo_scalar_u64, u64, rmodulo_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(bitwise_and_scalar_u64, u64, bitwise_and_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(bitwise_or_scalar_u64, u64, bitwise_or_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(bitwise_xor_scalar_u64, u64, bitwise_xor_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(shift_left_scalar_u64, u64, shift_left_scalar_u64);
#[cfg(feature = "dtype-u64")]
scalar_common_fn!(shift_right_scalar_u64, u64, shift_right_scalar_u64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(min_scalar_i64, i64, min_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(max_scalar_i64, i64, max_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(add_scalar_i64, i64, add_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(sub_scalar_i64, i64, sub_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(rsub_scalar_i64, i64, rsub_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(mul_scalar_i64, i64, mul_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(div_scalar_i64, i64, div_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(rdiv_scalar_i64, i64, rdiv_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(modulo_scalar_i64, i64, modulo_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(rmodulo_scalar_i64, i64, rmodulo_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(bitwise_and_scalar_i64, i64, bitwise_and_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(bitwise_or_scalar_i64, i64, bitwise_or_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(bitwise_xor_scalar_i64, i64, bitwise_xor_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(shift_left_scalar_i64, i64, shift_left_scalar_i64);
#[cfg(feature = "dtype-i64")]
scalar_common_fn!(shift_right_scalar_i64, i64, shift_right_scalar_i64);
