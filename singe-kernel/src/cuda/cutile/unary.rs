use std::sync::Arc;

#[cfg(feature = "dtype_f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::CompileOptions,
};

#[cfg(feature = "dtype_f16")]
use crate::cuda::cutile::kernel::f16::unary as kernel_f16;
#[cfg(feature = "dtype_f32")]
use crate::cuda::cutile::kernel::f32::unary as kernel_f32;
#[cfg(feature = "dtype_f64")]
use crate::cuda::cutile::kernel::f64::unary as kernel_f64;
#[cfg(any(
    feature = "dtype_i8",
    feature = "dtype_i32",
    feature = "dtype_i64",
    feature = "dtype_u8",
    feature = "dtype_u32",
    feature = "dtype_u64",
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

macro_rules! unary_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
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

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f32", feature = "dtype_f16", feature = "dtype_f64")
))]
macro_rules! unary_bool_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u8>,
            input: DevicePointer<$ty>,
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

macro_rules! nan_to_num_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            nan: $ty,
            posinf: $ty,
            neginf: $ty,
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
            enqueue_vector_op!(
                $kernel::$kernel_fn(out, input, nan, posinf, neginf),
                stream,
                compile
            );
            Ok(())
        }
    };
}

macro_rules! unary_fns_for_type {
    ($ty:ty, $kernel:ident, $(
        $name:ident => $kernel_fn:ident
    ),* $(,)?) => {
        $(unary_fn!($name, $ty, $kernel, $kernel_fn);)*
    };
}

#[cfg(any(
    feature = "dtype_i8",
    feature = "dtype_i32",
    feature = "dtype_i64",
    feature = "dtype_u8",
    feature = "dtype_u32",
    feature = "dtype_u64",
))]
macro_rules! unary_common_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
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

#[cfg(feature = "dtype_u8")]
macro_rules! unary_bool_common_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<u8>,
            input: DevicePointer<$ty>,
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

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f32", feature = "dtype_f16", feature = "dtype_f64")
))]
macro_rules! unary_bool_fns_for_type {
    ($ty:ty, $kernel:ident, $(
        $name:ident => $kernel_fn:ident
    ),* $(,)?) => {
        $(unary_bool_fn!($name, $ty, $kernel, $kernel_fn);)*
    };
}

macro_rules! nan_to_num_fns_for_type {
    ($ty:ty, $kernel:ident, $(
        $name:ident => $kernel_fn:ident
    ),* $(,)?) => {
        $(nan_to_num_fn!($name, $ty, $kernel, $kernel_fn);)*
    };
}

#[cfg(feature = "dtype_f32")]
unary_fns_for_type!(f32, kernel_f32,
    neg_f32 => neg_f32,
    abs_f32 => abs_f32,
    sqrt_f32 => sqrt_f32,
    exp_f32 => exp_f32,
    exp2_f32 => exp2_f32,
    expm1_f32 => expm1_f32,
    log_f32 => log_f32,
    log2_f32 => log2_f32,
    log10_f32 => log10_f32,
    log1p_f32 => log1p_f32,
    deg2rad_f32 => deg2rad_f32,
    rad2deg_f32 => rad2deg_f32,
    sin_f32 => sin_f32,
    cos_f32 => cos_f32,
    tan_f32 => tan_f32,
    sinh_f32 => sinh_f32,
    cosh_f32 => cosh_f32,
    tanh_f32 => tanh_f32,
    asin_f32 => asin_f32,
    acos_f32 => acos_f32,
    atan_f32 => atan_f32,
    ceil_f32 => ceil_f32,
    floor_f32 => floor_f32,
    trunc_f32 => trunc_f32,
    frac_f32 => frac_f32,
    reciprocal_f32 => reciprocal_f32,
    square_f32 => square_f32,
    rsqrt_f32 => rsqrt_f32,
    cbrt_f32 => cbrt_f32,
    asinh_f32 => asinh_f32,
    acosh_f32 => acosh_f32,
    atanh_f32 => atanh_f32,
    sign_f32 => sign_f32,
);

#[cfg(feature = "dtype_f32")]
nan_to_num_fns_for_type!(f32, kernel_f32,
    nan_to_num_f32 => nan_to_num_f32,
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
unary_bool_fns_for_type!(f32, kernel_f32,
    is_nan_bool_f32 => is_nan_bool_f32,
    is_inf_bool_f32 => is_inf_bool_f32,
    is_pos_inf_bool_f32 => is_pos_inf_bool_f32,
    is_neg_inf_bool_f32 => is_neg_inf_bool_f32,
    is_finite_bool_f32 => is_finite_bool_f32,
    is_normal_bool_f32 => is_normal_bool_f32,
    is_subnormal_bool_f32 => is_subnormal_bool_f32,
    is_zero_bool_f32 => is_zero_bool_f32,
    is_nonzero_bool_f32 => is_nonzero_bool_f32,
    is_positive_bool_f32 => is_positive_bool_f32,
    is_negative_bool_f32 => is_negative_bool_f32,
);

#[cfg(feature = "dtype_f16")]
unary_fns_for_type!(f16, kernel_f16,
    neg_f16 => neg_f16,
    abs_f16 => abs_f16,
    sqrt_f16 => sqrt_f16,
    exp_f16 => exp_f16,
    exp2_f16 => exp2_f16,
    expm1_f16 => expm1_f16,
    log_f16 => log_f16,
    log2_f16 => log2_f16,
    log10_f16 => log10_f16,
    log1p_f16 => log1p_f16,
    deg2rad_f16 => deg2rad_f16,
    rad2deg_f16 => rad2deg_f16,
    sin_f16 => sin_f16,
    cos_f16 => cos_f16,
    tan_f16 => tan_f16,
    sinh_f16 => sinh_f16,
    cosh_f16 => cosh_f16,
    tanh_f16 => tanh_f16,
    asin_f16 => asin_f16,
    acos_f16 => acos_f16,
    atan_f16 => atan_f16,
    ceil_f16 => ceil_f16,
    floor_f16 => floor_f16,
    trunc_f16 => trunc_f16,
    frac_f16 => frac_f16,
    reciprocal_f16 => reciprocal_f16,
    square_f16 => square_f16,
    rsqrt_f16 => rsqrt_f16,
    cbrt_f16 => cbrt_f16,
    asinh_f16 => asinh_f16,
    acosh_f16 => acosh_f16,
    atanh_f16 => atanh_f16,
    sign_f16 => sign_f16,
);

#[cfg(feature = "dtype_f16")]
nan_to_num_fns_for_type!(f16, kernel_f16,
    nan_to_num_f16 => nan_to_num_f16,
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
unary_bool_fns_for_type!(f16, kernel_f16,
    is_nan_bool_f16 => is_nan_bool_f16,
    is_inf_bool_f16 => is_inf_bool_f16,
    is_pos_inf_bool_f16 => is_pos_inf_bool_f16,
    is_neg_inf_bool_f16 => is_neg_inf_bool_f16,
    is_finite_bool_f16 => is_finite_bool_f16,
    is_normal_bool_f16 => is_normal_bool_f16,
    is_subnormal_bool_f16 => is_subnormal_bool_f16,
    is_zero_bool_f16 => is_zero_bool_f16,
    is_nonzero_bool_f16 => is_nonzero_bool_f16,
    is_positive_bool_f16 => is_positive_bool_f16,
    is_negative_bool_f16 => is_negative_bool_f16,
);

#[cfg(feature = "dtype_f64")]
unary_fns_for_type!(f64, kernel_f64,
    neg_f64 => neg_f64,
    abs_f64 => abs_f64,
    sqrt_f64 => sqrt_f64,
    exp_f64 => exp_f64,
    exp2_f64 => exp2_f64,
    expm1_f64 => expm1_f64,
    log_f64 => log_f64,
    log2_f64 => log2_f64,
    log10_f64 => log10_f64,
    log1p_f64 => log1p_f64,
    deg2rad_f64 => deg2rad_f64,
    rad2deg_f64 => rad2deg_f64,
    sin_f64 => sin_f64,
    cos_f64 => cos_f64,
    tan_f64 => tan_f64,
    sinh_f64 => sinh_f64,
    cosh_f64 => cosh_f64,
    tanh_f64 => tanh_f64,
    asin_f64 => asin_f64,
    acos_f64 => acos_f64,
    atan_f64 => atan_f64,
    ceil_f64 => ceil_f64,
    floor_f64 => floor_f64,
    trunc_f64 => trunc_f64,
    frac_f64 => frac_f64,
    reciprocal_f64 => reciprocal_f64,
    square_f64 => square_f64,
    rsqrt_f64 => rsqrt_f64,
    cbrt_f64 => cbrt_f64,
    asinh_f64 => asinh_f64,
    acosh_f64 => acosh_f64,
    atanh_f64 => atanh_f64,
    sign_f64 => sign_f64,
);

#[cfg(feature = "dtype_f64")]
nan_to_num_fns_for_type!(f64, kernel_f64,
    nan_to_num_f64 => nan_to_num_f64,
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
unary_bool_fns_for_type!(f64, kernel_f64,
    is_nan_bool_f64 => is_nan_bool_f64,
    is_inf_bool_f64 => is_inf_bool_f64,
    is_pos_inf_bool_f64 => is_pos_inf_bool_f64,
    is_neg_inf_bool_f64 => is_neg_inf_bool_f64,
    is_finite_bool_f64 => is_finite_bool_f64,
    is_normal_bool_f64 => is_normal_bool_f64,
    is_subnormal_bool_f64 => is_subnormal_bool_f64,
    is_zero_bool_f64 => is_zero_bool_f64,
    is_nonzero_bool_f64 => is_nonzero_bool_f64,
    is_positive_bool_f64 => is_positive_bool_f64,
    is_negative_bool_f64 => is_negative_bool_f64,
);

#[cfg(feature = "dtype_i8")]
unary_common_fn!(neg_i8, i8, neg_i8);

#[cfg(feature = "dtype_i32")]
unary_common_fn!(neg_i32, i32, neg_i32);

#[cfg(feature = "dtype_i64")]
unary_common_fn!(neg_i64, i64, neg_i64);

#[cfg(feature = "dtype_i8")]
unary_common_fn!(abs_i8, i8, abs_i8);

#[cfg(feature = "dtype_i32")]
unary_common_fn!(abs_i32, i32, abs_i32);

#[cfg(feature = "dtype_i64")]
unary_common_fn!(abs_i64, i64, abs_i64);

#[cfg(feature = "dtype_i8")]
unary_common_fn!(sign_i8, i8, sign_i8);

#[cfg(feature = "dtype_i32")]
unary_common_fn!(sign_i32, i32, sign_i32);

#[cfg(feature = "dtype_i64")]
unary_common_fn!(sign_i64, i64, sign_i64);

#[cfg(feature = "dtype_u8")]
unary_common_fn!(sign_u8, u8, sign_u8);

#[cfg(feature = "dtype_u32")]
unary_common_fn!(sign_u32, u32, sign_u32);

#[cfg(feature = "dtype_u64")]
unary_common_fn!(sign_u64, u64, sign_u64);

#[cfg(feature = "dtype_u8")]
unary_common_fn!(bitwise_not_u8, u8, bitwise_not_u8);

#[cfg(feature = "dtype_i8")]
unary_common_fn!(bitwise_not_i8, i8, bitwise_not_i8);

#[cfg(feature = "dtype_u32")]
unary_common_fn!(bitwise_not_u32, u32, bitwise_not_u32);

#[cfg(feature = "dtype_i32")]
unary_common_fn!(bitwise_not_i32, i32, bitwise_not_i32);

#[cfg(feature = "dtype_u64")]
unary_common_fn!(bitwise_not_u64, u64, bitwise_not_u64);

#[cfg(feature = "dtype_i64")]
unary_common_fn!(bitwise_not_i64, i64, bitwise_not_i64);

#[cfg(feature = "dtype_u8")]
unary_bool_common_fn!(is_zero_bool_u8, u8, is_zero_bool_u8);

#[cfg(feature = "dtype_u8")]
unary_bool_common_fn!(is_nonzero_bool_u8, u8, is_nonzero_bool_u8);

#[cfg(feature = "dtype_u8")]
unary_bool_common_fn!(is_positive_bool_u8, u8, is_positive_bool_u8);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
unary_bool_common_fn!(is_zero_bool_i8, i8, is_zero_bool_i8);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
unary_bool_common_fn!(is_nonzero_bool_i8, i8, is_nonzero_bool_i8);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
unary_bool_common_fn!(is_positive_bool_i8, i8, is_positive_bool_i8);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
unary_bool_common_fn!(is_negative_bool_i8, i8, is_negative_bool_i8);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
unary_bool_common_fn!(is_zero_bool_u32, u32, is_zero_bool_u32);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
unary_bool_common_fn!(is_nonzero_bool_u32, u32, is_nonzero_bool_u32);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
unary_bool_common_fn!(is_positive_bool_u32, u32, is_positive_bool_u32);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
unary_bool_common_fn!(is_zero_bool_i32, i32, is_zero_bool_i32);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
unary_bool_common_fn!(is_nonzero_bool_i32, i32, is_nonzero_bool_i32);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
unary_bool_common_fn!(is_positive_bool_i32, i32, is_positive_bool_i32);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
unary_bool_common_fn!(is_negative_bool_i32, i32, is_negative_bool_i32);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
unary_bool_common_fn!(is_zero_bool_u64, u64, is_zero_bool_u64);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
unary_bool_common_fn!(is_nonzero_bool_u64, u64, is_nonzero_bool_u64);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
unary_bool_common_fn!(is_positive_bool_u64, u64, is_positive_bool_u64);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
unary_bool_common_fn!(is_zero_bool_i64, i64, is_zero_bool_i64);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
unary_bool_common_fn!(is_nonzero_bool_i64, i64, is_nonzero_bool_i64);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
unary_bool_common_fn!(is_positive_bool_i64, i64, is_positive_bool_i64);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
unary_bool_common_fn!(is_negative_bool_i64, i64, is_negative_bool_i64);
