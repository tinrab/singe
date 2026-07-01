use std::sync::Arc;

#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

#[cfg(feature = "dtype-f16")]
use crate::cuda::cutile::kernel::f16::elementwise as kernel_f16;
#[cfg(feature = "dtype-f32")]
use crate::cuda::cutile::kernel::f32::elementwise as kernel_f32;
#[cfg(feature = "dtype-f64")]
use crate::cuda::cutile::kernel::f64::elementwise as kernel_f64;
use crate::{
    cuda::cutile::{
        DeviceOpExt,
        adapter::TensorAdapter,
        kernel::common as kernel_common,
        utility::{VectorLaunch, checked_device_pointer, vector_tile_size},
    },
    error::Result,
};

macro_rules! binary_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lhs: DevicePointer<$ty>,
            rhs: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(lhs)?;
            checked_device_pointer(rhs)?;
            let tile = vector_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let lhs = TensorAdapter::contiguous_1d(lhs, len)?;
            let rhs = TensorAdapter::contiguous_1d(rhs, len)?;
            $kernel::$kernel_fn(out, lhs, rhs).enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! binary_bool_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u8>,
            lhs: DevicePointer<$ty>,
            rhs: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(lhs)?;
            checked_device_pointer(rhs)?;
            let tile = vector_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let lhs = TensorAdapter::contiguous_1d(lhs, len)?;
            let rhs = TensorAdapter::contiguous_1d(rhs, len)?;
            $kernel::$kernel_fn(out, lhs, rhs).enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! ternary_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lhs: DevicePointer<$ty>,
            rhs: DevicePointer<$ty>,
            acc: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(lhs)?;
            checked_device_pointer(rhs)?;
            checked_device_pointer(acc)?;
            let tile = vector_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let lhs = TensorAdapter::contiguous_1d(lhs, len)?;
            let rhs = TensorAdapter::contiguous_1d(rhs, len)?;
            let acc = TensorAdapter::contiguous_1d(acc, len)?;
            $kernel::$kernel_fn(out, lhs, rhs, acc).enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
binary_fn!(add_f32, f32, kernel_f32, add_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(sub_f32, f32, kernel_f32, sub_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(mul_f32, f32, kernel_f32, mul_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(div_f32, f32, kernel_f32, div_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(pow_f32, f32, kernel_f32, pow_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(hypot_f32, f32, kernel_f32, hypot_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(
    squared_difference_f32,
    f32,
    kernel_f32,
    squared_difference_f32
);
#[cfg(feature = "dtype-f32")]
binary_fn!(logaddexp_f32, f32, kernel_f32, logaddexp_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(xlogy_f32, f32, kernel_f32, xlogy_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(max_f32, f32, kernel_f32, maximum_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(min_f32, f32, kernel_f32, minimum_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(modulo_f32, f32, kernel_f32, modulo_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(remainder_f32, f32, kernel_f32, modulo_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(atan2_f32, f32, kernel_f32, atan2_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(equal_f32, f32, kernel_f32, equal_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(not_equal_f32, f32, kernel_f32, not_equal_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(less_f32, f32, kernel_f32, less_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(less_equal_f32, f32, kernel_f32, less_equal_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(greater_f32, f32, kernel_f32, greater_f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(greater_equal_f32, f32, kernel_f32, greater_equal_f32);
#[cfg(feature = "dtype-f32")]
ternary_fn!(fma_f32, f32, kernel_f32, fma_f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
binary_bool_fn!(equal_bool_f32, f32, kernel_f32, equal_bool_f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
binary_bool_fn!(not_equal_bool_f32, f32, kernel_f32, not_equal_bool_f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
binary_bool_fn!(less_bool_f32, f32, kernel_f32, less_bool_f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
binary_bool_fn!(less_equal_bool_f32, f32, kernel_f32, less_equal_bool_f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
binary_bool_fn!(greater_bool_f32, f32, kernel_f32, greater_bool_f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
binary_bool_fn!(
    greater_equal_bool_f32,
    f32,
    kernel_f32,
    greater_equal_bool_f32
);

#[cfg(feature = "dtype-f16")]
binary_fn!(add_f16, f16, kernel_f16, add_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(sub_f16, f16, kernel_f16, sub_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(mul_f16, f16, kernel_f16, mul_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(div_f16, f16, kernel_f16, div_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(pow_f16, f16, kernel_f16, pow_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(hypot_f16, f16, kernel_f16, hypot_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(
    squared_difference_f16,
    f16,
    kernel_f16,
    squared_difference_f16
);
#[cfg(feature = "dtype-f16")]
binary_fn!(logaddexp_f16, f16, kernel_f16, logaddexp_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(xlogy_f16, f16, kernel_f16, xlogy_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(max_f16, f16, kernel_f16, maximum_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(min_f16, f16, kernel_f16, minimum_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(modulo_f16, f16, kernel_f16, modulo_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(remainder_f16, f16, kernel_f16, modulo_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(atan2_f16, f16, kernel_f16, atan2_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(equal_f16, f16, kernel_f16, equal_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(not_equal_f16, f16, kernel_f16, not_equal_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(less_f16, f16, kernel_f16, less_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(less_equal_f16, f16, kernel_f16, less_equal_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(greater_f16, f16, kernel_f16, greater_f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(greater_equal_f16, f16, kernel_f16, greater_equal_f16);
#[cfg(feature = "dtype-f16")]
ternary_fn!(fma_f16, f16, kernel_f16, fma_f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
binary_bool_fn!(equal_bool_f16, f16, kernel_f16, equal_bool_f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
binary_bool_fn!(not_equal_bool_f16, f16, kernel_f16, not_equal_bool_f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
binary_bool_fn!(less_bool_f16, f16, kernel_f16, less_bool_f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
binary_bool_fn!(less_equal_bool_f16, f16, kernel_f16, less_equal_bool_f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
binary_bool_fn!(greater_bool_f16, f16, kernel_f16, greater_bool_f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
binary_bool_fn!(
    greater_equal_bool_f16,
    f16,
    kernel_f16,
    greater_equal_bool_f16
);

#[cfg(feature = "dtype-f64")]
binary_fn!(add_f64, f64, kernel_f64, add_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(sub_f64, f64, kernel_f64, sub_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(mul_f64, f64, kernel_f64, mul_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(div_f64, f64, kernel_f64, div_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(pow_f64, f64, kernel_f64, pow_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(hypot_f64, f64, kernel_f64, hypot_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(
    squared_difference_f64,
    f64,
    kernel_f64,
    squared_difference_f64
);
#[cfg(feature = "dtype-f64")]
binary_fn!(logaddexp_f64, f64, kernel_f64, logaddexp_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(xlogy_f64, f64, kernel_f64, xlogy_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(max_f64, f64, kernel_f64, maximum_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(min_f64, f64, kernel_f64, minimum_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(modulo_f64, f64, kernel_f64, modulo_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(remainder_f64, f64, kernel_f64, modulo_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(atan2_f64, f64, kernel_f64, atan2_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(equal_f64, f64, kernel_f64, equal_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(not_equal_f64, f64, kernel_f64, not_equal_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(less_f64, f64, kernel_f64, less_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(less_equal_f64, f64, kernel_f64, less_equal_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(greater_f64, f64, kernel_f64, greater_f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(greater_equal_f64, f64, kernel_f64, greater_equal_f64);
#[cfg(feature = "dtype-f64")]
ternary_fn!(fma_f64, f64, kernel_f64, fma_f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
binary_bool_fn!(equal_bool_f64, f64, kernel_f64, equal_bool_f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
binary_bool_fn!(not_equal_bool_f64, f64, kernel_f64, not_equal_bool_f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
binary_bool_fn!(less_bool_f64, f64, kernel_f64, less_bool_f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
binary_bool_fn!(less_equal_bool_f64, f64, kernel_f64, less_equal_bool_f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
binary_bool_fn!(greater_bool_f64, f64, kernel_f64, greater_bool_f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
binary_bool_fn!(
    greater_equal_bool_f64,
    f64,
    kernel_f64,
    greater_equal_bool_f64
);

macro_rules! binary_bool_common_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u8>,
            lhs: DevicePointer<$ty>,
            rhs: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(lhs)?;
            checked_device_pointer(rhs)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { kernel_common::$kernel_fn(out, lhs, rhs, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
binary_bool_common_fn!(equal_bool_u32, u32, equal_bool_u32);
#[cfg(feature = "dtype-u8")]
binary_bool_common_fn!(equal_bool_u8, u8, equal_bool_u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
binary_bool_common_fn!(equal_bool_i8, i8, equal_bool_i8);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
binary_bool_common_fn!(equal_bool_i32, i32, equal_bool_i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
binary_bool_common_fn!(equal_bool_u64, u64, equal_bool_u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
binary_bool_common_fn!(equal_bool_i64, i64, equal_bool_i64);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
binary_bool_common_fn!(not_equal_bool_u32, u32, not_equal_bool_u32);
#[cfg(feature = "dtype-u8")]
binary_bool_common_fn!(not_equal_bool_u8, u8, not_equal_bool_u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
binary_bool_common_fn!(not_equal_bool_i8, i8, not_equal_bool_i8);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
binary_bool_common_fn!(not_equal_bool_i32, i32, not_equal_bool_i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
binary_bool_common_fn!(not_equal_bool_u64, u64, not_equal_bool_u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
binary_bool_common_fn!(not_equal_bool_i64, i64, not_equal_bool_i64);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
binary_bool_common_fn!(less_bool_u32, u32, less_bool_u32);
#[cfg(feature = "dtype-u8")]
binary_bool_common_fn!(less_bool_u8, u8, less_bool_u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
binary_bool_common_fn!(less_bool_i8, i8, less_bool_i8);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
binary_bool_common_fn!(less_bool_i32, i32, less_bool_i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
binary_bool_common_fn!(less_bool_u64, u64, less_bool_u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
binary_bool_common_fn!(less_bool_i64, i64, less_bool_i64);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
binary_bool_common_fn!(less_equal_bool_u32, u32, less_equal_bool_u32);
#[cfg(feature = "dtype-u8")]
binary_bool_common_fn!(less_equal_bool_u8, u8, less_equal_bool_u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
binary_bool_common_fn!(less_equal_bool_i8, i8, less_equal_bool_i8);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
binary_bool_common_fn!(less_equal_bool_i32, i32, less_equal_bool_i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
binary_bool_common_fn!(less_equal_bool_u64, u64, less_equal_bool_u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
binary_bool_common_fn!(less_equal_bool_i64, i64, less_equal_bool_i64);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
binary_bool_common_fn!(greater_bool_u32, u32, greater_bool_u32);
#[cfg(feature = "dtype-u8")]
binary_bool_common_fn!(greater_bool_u8, u8, greater_bool_u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
binary_bool_common_fn!(greater_bool_i8, i8, greater_bool_i8);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
binary_bool_common_fn!(greater_bool_i32, i32, greater_bool_i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
binary_bool_common_fn!(greater_bool_u64, u64, greater_bool_u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
binary_bool_common_fn!(greater_bool_i64, i64, greater_bool_i64);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
binary_bool_common_fn!(greater_equal_bool_u32, u32, greater_equal_bool_u32);
#[cfg(feature = "dtype-u8")]
binary_bool_common_fn!(greater_equal_bool_u8, u8, greater_equal_bool_u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
binary_bool_common_fn!(greater_equal_bool_i8, i8, greater_equal_bool_i8);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
binary_bool_common_fn!(greater_equal_bool_i32, i32, greater_equal_bool_i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
binary_bool_common_fn!(greater_equal_bool_u64, u64, greater_equal_bool_u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
binary_bool_common_fn!(greater_equal_bool_i64, i64, greater_equal_bool_i64);

macro_rules! binary_common_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lhs: DevicePointer<$ty>,
            rhs: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(lhs)?;
            checked_device_pointer(rhs)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { kernel_common::$kernel_fn(out, lhs, rhs, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-u32")]
binary_common_fn!(min_u32, u32, min_u32);
#[cfg(feature = "dtype-u32")]
binary_common_fn!(max_u32, u32, max_u32);
#[cfg(feature = "dtype-u8")]
binary_common_fn!(min_u8, u8, min_u8);
#[cfg(feature = "dtype-u8")]
binary_common_fn!(max_u8, u8, max_u8);
#[cfg(feature = "dtype-i8")]
binary_common_fn!(min_i8, i8, min_i8);
#[cfg(feature = "dtype-i8")]
binary_common_fn!(max_i8, i8, max_i8);
#[cfg(feature = "dtype-i32")]
binary_common_fn!(min_i32, i32, min_i32);
#[cfg(feature = "dtype-i32")]
binary_common_fn!(max_i32, i32, max_i32);
#[cfg(feature = "dtype-u64")]
binary_common_fn!(min_u64, u64, min_u64);
#[cfg(feature = "dtype-u64")]
binary_common_fn!(max_u64, u64, max_u64);
#[cfg(feature = "dtype-i64")]
binary_common_fn!(min_i64, i64, min_i64);
#[cfg(feature = "dtype-i64")]
binary_common_fn!(max_i64, i64, max_i64);

#[cfg(feature = "dtype-u8")]
binary_common_fn!(add_u8, u8, add_u8);
#[cfg(feature = "dtype-u8")]
binary_common_fn!(sub_u8, u8, sub_u8);
#[cfg(feature = "dtype-u8")]
binary_common_fn!(mul_u8, u8, mul_u8);
#[cfg(feature = "dtype-u8")]
binary_common_fn!(div_u8, u8, div_u8);
#[cfg(feature = "dtype-u8")]
binary_common_fn!(modulo_u8, u8, modulo_u8);
#[cfg(feature = "dtype-u8")]
binary_common_fn!(remainder_u8, u8, modulo_u8);
#[cfg(feature = "dtype-u8")]
binary_common_fn!(bitwise_and_u8, u8, bitwise_and_u8);
#[cfg(feature = "dtype-u8")]
binary_common_fn!(bitwise_or_u8, u8, bitwise_or_u8);
#[cfg(feature = "dtype-u8")]
binary_common_fn!(bitwise_xor_u8, u8, bitwise_xor_u8);
#[cfg(feature = "dtype-u8")]
binary_common_fn!(shift_left_u8, u8, shift_left_u8);
#[cfg(feature = "dtype-u8")]
binary_common_fn!(shift_right_u8, u8, shift_right_u8);
#[cfg(feature = "dtype-i8")]
binary_common_fn!(add_i8, i8, add_i8);
#[cfg(feature = "dtype-i8")]
binary_common_fn!(sub_i8, i8, sub_i8);
#[cfg(feature = "dtype-i8")]
binary_common_fn!(mul_i8, i8, mul_i8);
#[cfg(feature = "dtype-i8")]
binary_common_fn!(div_i8, i8, div_i8);
#[cfg(feature = "dtype-i8")]
binary_common_fn!(modulo_i8, i8, modulo_i8);
#[cfg(feature = "dtype-i8")]
binary_common_fn!(remainder_i8, i8, modulo_i8);
#[cfg(feature = "dtype-i8")]
binary_common_fn!(bitwise_and_i8, i8, bitwise_and_i8);
#[cfg(feature = "dtype-i8")]
binary_common_fn!(bitwise_or_i8, i8, bitwise_or_i8);
#[cfg(feature = "dtype-i8")]
binary_common_fn!(bitwise_xor_i8, i8, bitwise_xor_i8);
#[cfg(feature = "dtype-i8")]
binary_common_fn!(shift_left_i8, i8, shift_left_i8);
#[cfg(feature = "dtype-i8")]
binary_common_fn!(shift_right_i8, i8, shift_right_i8);
#[cfg(feature = "dtype-u32")]
binary_common_fn!(add_u32, u32, add_u32);
#[cfg(feature = "dtype-u32")]
binary_common_fn!(sub_u32, u32, sub_u32);
#[cfg(feature = "dtype-u32")]
binary_common_fn!(mul_u32, u32, mul_u32);
#[cfg(feature = "dtype-u32")]
binary_common_fn!(div_u32, u32, div_u32);
#[cfg(feature = "dtype-u32")]
binary_common_fn!(modulo_u32, u32, modulo_u32);
#[cfg(feature = "dtype-u32")]
binary_common_fn!(remainder_u32, u32, modulo_u32);
#[cfg(feature = "dtype-u32")]
binary_common_fn!(bitwise_and_u32, u32, bitwise_and_u32);
#[cfg(feature = "dtype-u32")]
binary_common_fn!(bitwise_or_u32, u32, bitwise_or_u32);
#[cfg(feature = "dtype-u32")]
binary_common_fn!(bitwise_xor_u32, u32, bitwise_xor_u32);
#[cfg(feature = "dtype-u32")]
binary_common_fn!(shift_left_u32, u32, shift_left_u32);
#[cfg(feature = "dtype-u32")]
binary_common_fn!(shift_right_u32, u32, shift_right_u32);
#[cfg(feature = "dtype-i32")]
binary_common_fn!(add_i32, i32, add_i32);
#[cfg(feature = "dtype-i32")]
binary_common_fn!(sub_i32, i32, sub_i32);
#[cfg(feature = "dtype-i32")]
binary_common_fn!(mul_i32, i32, mul_i32);
#[cfg(feature = "dtype-i32")]
binary_common_fn!(div_i32, i32, div_i32);
#[cfg(feature = "dtype-i32")]
binary_common_fn!(modulo_i32, i32, modulo_i32);
#[cfg(feature = "dtype-i32")]
binary_common_fn!(remainder_i32, i32, modulo_i32);
#[cfg(feature = "dtype-i32")]
binary_common_fn!(bitwise_and_i32, i32, bitwise_and_i32);
#[cfg(feature = "dtype-i32")]
binary_common_fn!(bitwise_or_i32, i32, bitwise_or_i32);
#[cfg(feature = "dtype-i32")]
binary_common_fn!(bitwise_xor_i32, i32, bitwise_xor_i32);
#[cfg(feature = "dtype-i32")]
binary_common_fn!(shift_left_i32, i32, shift_left_i32);
#[cfg(feature = "dtype-i32")]
binary_common_fn!(shift_right_i32, i32, shift_right_i32);
#[cfg(feature = "dtype-u64")]
binary_common_fn!(add_u64, u64, add_u64);
#[cfg(feature = "dtype-u64")]
binary_common_fn!(sub_u64, u64, sub_u64);
#[cfg(feature = "dtype-u64")]
binary_common_fn!(mul_u64, u64, mul_u64);
#[cfg(feature = "dtype-u64")]
binary_common_fn!(div_u64, u64, div_u64);
#[cfg(feature = "dtype-u64")]
binary_common_fn!(modulo_u64, u64, modulo_u64);
#[cfg(feature = "dtype-u64")]
binary_common_fn!(remainder_u64, u64, modulo_u64);
#[cfg(feature = "dtype-u64")]
binary_common_fn!(bitwise_and_u64, u64, bitwise_and_u64);
#[cfg(feature = "dtype-u64")]
binary_common_fn!(bitwise_or_u64, u64, bitwise_or_u64);
#[cfg(feature = "dtype-u64")]
binary_common_fn!(bitwise_xor_u64, u64, bitwise_xor_u64);
#[cfg(feature = "dtype-u64")]
binary_common_fn!(shift_left_u64, u64, shift_left_u64);
#[cfg(feature = "dtype-u64")]
binary_common_fn!(shift_right_u64, u64, shift_right_u64);
#[cfg(feature = "dtype-i64")]
binary_common_fn!(add_i64, i64, add_i64);
#[cfg(feature = "dtype-i64")]
binary_common_fn!(sub_i64, i64, sub_i64);
#[cfg(feature = "dtype-i64")]
binary_common_fn!(mul_i64, i64, mul_i64);
#[cfg(feature = "dtype-i64")]
binary_common_fn!(div_i64, i64, div_i64);
#[cfg(feature = "dtype-i64")]
binary_common_fn!(modulo_i64, i64, modulo_i64);
#[cfg(feature = "dtype-i64")]
binary_common_fn!(remainder_i64, i64, modulo_i64);
#[cfg(feature = "dtype-i64")]
binary_common_fn!(bitwise_and_i64, i64, bitwise_and_i64);
#[cfg(feature = "dtype-i64")]
binary_common_fn!(bitwise_or_i64, i64, bitwise_or_i64);
#[cfg(feature = "dtype-i64")]
binary_common_fn!(bitwise_xor_i64, i64, bitwise_xor_i64);
#[cfg(feature = "dtype-i64")]
binary_common_fn!(shift_left_i64, i64, shift_left_i64);
#[cfg(feature = "dtype-i64")]
binary_common_fn!(shift_right_i64, i64, shift_right_i64);

macro_rules! clamp_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            min: $ty,
            max: $ty,
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
            $kernel::$kernel_fn(out, input, min, max).enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
clamp_fn!(clamp_f32, f32, kernel_f32, clamp_f32);
#[cfg(feature = "dtype-f16")]
clamp_fn!(clamp_f16, f16, kernel_f16, clamp_f16);
#[cfg(feature = "dtype-f64")]
clamp_fn!(clamp_f64, f64, kernel_f64, clamp_f64);

macro_rules! clamp_common_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            min: $ty,
            max: $ty,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { kernel_common::$kernel_fn(out, input, min, max, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-u8")]
clamp_common_fn!(clamp_u8, u8, clamp_u8);
#[cfg(feature = "dtype-i8")]
clamp_common_fn!(clamp_i8, i8, clamp_i8);
#[cfg(feature = "dtype-u32")]
clamp_common_fn!(clamp_u32, u32, clamp_u32);
#[cfg(feature = "dtype-i32")]
clamp_common_fn!(clamp_i32, i32, clamp_i32);
#[cfg(feature = "dtype-u64")]
clamp_common_fn!(clamp_u64, u64, clamp_u64);
#[cfg(feature = "dtype-i64")]
clamp_common_fn!(clamp_i64, i64, clamp_i64);

macro_rules! lerp_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lhs: DevicePointer<$ty>,
            rhs: DevicePointer<$ty>,
            scalar: $ty,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(lhs)?;
            checked_device_pointer(rhs)?;
            let tile = vector_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let lhs = TensorAdapter::contiguous_1d(lhs, len)?;
            let rhs = TensorAdapter::contiguous_1d(rhs, len)?;
            $kernel::$kernel_fn(out, lhs, rhs, scalar).enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
lerp_fn!(lerp_f32, f32, kernel_f32, lerp_f32);
#[cfg(feature = "dtype-f16")]
lerp_fn!(lerp_f16, f16, kernel_f16, lerp_f16);
#[cfg(feature = "dtype-f64")]
lerp_fn!(lerp_f64, f64, kernel_f64, lerp_f64);

macro_rules! where_fn {
    ($name:ident, $ty:ty, $condition:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            condition: DevicePointer<$condition>,
            x: DevicePointer<$ty>,
            y: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(condition)?;
            checked_device_pointer(x)?;
            checked_device_pointer(y)?;
            let tile = vector_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let condition = TensorAdapter::contiguous_1d(condition, len)?;
            let x = TensorAdapter::contiguous_1d(x, len)?;
            let y = TensorAdapter::contiguous_1d(y, len)?;
            $kernel::$kernel_fn(out, condition, x, y).enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
where_fn!(where_f32, f32, f32, kernel_f32, where_f32);
#[cfg(feature = "dtype-f16")]
where_fn!(where_f16, f16, f16, kernel_f16, where_f16);
#[cfg(feature = "dtype-f64")]
where_fn!(where_f64, f64, f64, kernel_f64, where_f64);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
where_fn!(where_bool_f32, f32, u8, kernel_f32, where_bool_f32);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
where_fn!(where_bool_f16, f16, u8, kernel_f16, where_bool_f16);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
where_fn!(where_bool_f64, f64, u8, kernel_f64, where_bool_f64);

macro_rules! where_bool_common_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            condition: DevicePointer<u8>,
            x: DevicePointer<$ty>,
            y: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(condition)?;
            checked_device_pointer(x)?;
            checked_device_pointer(y)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { kernel_common::$kernel_fn(out, condition, x, y, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-u8")]
where_bool_common_fn!(where_bool_u8, u8, where_bool_u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
where_bool_common_fn!(where_bool_i8, i8, where_bool_i8);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
where_bool_common_fn!(where_bool_u32, u32, where_bool_u32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
where_bool_common_fn!(where_bool_i32, i32, where_bool_i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
where_bool_common_fn!(where_bool_u64, u64, where_bool_u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
where_bool_common_fn!(where_bool_i64, i64, where_bool_i64);

macro_rules! masked_fill_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            mask: DevicePointer<u8>,
            value: $ty,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(mask)?;
            let tile = vector_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let input = TensorAdapter::contiguous_1d(input, len)?;
            let mask = TensorAdapter::contiguous_1d(mask, len)?;
            $kernel::$kernel_fn(out, input, mask, value).enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
masked_fill_fn!(masked_fill_f32, f32, kernel_f32, masked_fill_f32);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
masked_fill_fn!(masked_fill_f16, f16, kernel_f16, masked_fill_f16);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
masked_fill_fn!(masked_fill_f64, f64, kernel_f64, masked_fill_f64);

macro_rules! masked_fill_common_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            mask: DevicePointer<u8>,
            value: $ty,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(mask)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { kernel_common::$kernel_fn(out, input, mask, value, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-u8")]
masked_fill_common_fn!(masked_fill_u8, u8, masked_fill_u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
masked_fill_common_fn!(masked_fill_i8, i8, masked_fill_i8);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
masked_fill_common_fn!(masked_fill_u32, u32, masked_fill_u32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
masked_fill_common_fn!(masked_fill_i32, i32, masked_fill_i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
masked_fill_common_fn!(masked_fill_u64, u64, masked_fill_u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
masked_fill_common_fn!(masked_fill_i64, i64, masked_fill_i64);
