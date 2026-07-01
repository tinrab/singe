//! Single-input math operations and predicates.

#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceSlice, DeviceSliceMut},
};

use crate::{
    cuda::{
        cutile,
        interop::{borrowed_stream, input_pointer, output_pointer},
        utility::ensure_unary_lengths,
    },
    error::Result,
};

macro_rules! unary_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
        ) -> Result<()> {
            let len = ensure_unary_lengths(out, input)?;
            let stream = borrowed_stream(stream)?;
            cutile::unary::$name(&stream, output_pointer(out), input_pointer(input), len)
        }
    };
}

macro_rules! unary_bool_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<u8>,
            input: &impl DeviceSlice<$ty>,
        ) -> Result<()> {
            let len = out.len();
            crate::utility::ensure_len(input.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::unary::$name(&stream, output_pointer(out), input_pointer(input), len)
        }
    };
}

macro_rules! nan_to_num_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            nan: $ty,
            posinf: $ty,
            neginf: $ty,
        ) -> Result<()> {
            let len = ensure_unary_lengths(out, input)?;
            let stream = borrowed_stream(stream)?;
            cutile::unary::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                nan,
                posinf,
                neginf,
                len,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
unary_fn!(neg_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(abs_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(sqrt_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(exp_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(exp2_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(expm1_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(log_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(log2_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(log10_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(log1p_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(deg2rad_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(rad2deg_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(sin_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(cos_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(tan_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(sinh_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(cosh_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(tanh_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(asin_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(acos_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(atan_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(ceil_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(floor_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(trunc_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(frac_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(reciprocal_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(square_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(rsqrt_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(cbrt_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(asinh_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(acosh_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(atanh_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fn!(sign_f32, f32);
#[cfg(feature = "dtype-f32")]
nan_to_num_fn!(nan_to_num_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
unary_bool_fn!(is_nan_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
unary_bool_fn!(is_inf_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
unary_bool_fn!(is_pos_inf_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
unary_bool_fn!(is_neg_inf_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
unary_bool_fn!(is_finite_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
unary_bool_fn!(is_normal_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
unary_bool_fn!(is_subnormal_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
unary_bool_fn!(is_zero_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
unary_bool_fn!(is_nonzero_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
unary_bool_fn!(is_positive_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
unary_bool_fn!(is_negative_bool_f32, f32);

#[cfg(feature = "dtype-f16")]
unary_fn!(neg_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(abs_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(sqrt_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(exp_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(exp2_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(expm1_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(log_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(log2_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(log10_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(log1p_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(deg2rad_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(rad2deg_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(sin_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(cos_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(tan_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(sinh_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(cosh_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(tanh_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(asin_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(acos_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(atan_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(ceil_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(floor_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(trunc_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(frac_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(reciprocal_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(square_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(rsqrt_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(cbrt_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(asinh_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(acosh_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(atanh_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fn!(sign_f16, f16);
#[cfg(feature = "dtype-f16")]
nan_to_num_fn!(nan_to_num_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
unary_bool_fn!(is_nan_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
unary_bool_fn!(is_inf_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
unary_bool_fn!(is_pos_inf_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
unary_bool_fn!(is_neg_inf_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
unary_bool_fn!(is_finite_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
unary_bool_fn!(is_normal_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
unary_bool_fn!(is_subnormal_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
unary_bool_fn!(is_zero_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
unary_bool_fn!(is_nonzero_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
unary_bool_fn!(is_positive_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
unary_bool_fn!(is_negative_bool_f16, f16);

#[cfg(feature = "dtype-f64")]
unary_fn!(neg_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(abs_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(sqrt_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(exp_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(exp2_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(expm1_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(log_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(log2_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(log10_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(log1p_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(deg2rad_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(rad2deg_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(sin_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(cos_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(tan_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(sinh_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(cosh_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(tanh_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(asin_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(acos_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(atan_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(ceil_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(floor_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(trunc_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(frac_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(reciprocal_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(square_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(rsqrt_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(cbrt_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(asinh_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(acosh_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(atanh_f64, f64);
#[cfg(feature = "dtype-f64")]
unary_fn!(sign_f64, f64);
#[cfg(feature = "dtype-f64")]
nan_to_num_fn!(nan_to_num_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
unary_bool_fn!(is_nan_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
unary_bool_fn!(is_inf_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
unary_bool_fn!(is_pos_inf_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
unary_bool_fn!(is_neg_inf_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
unary_bool_fn!(is_finite_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
unary_bool_fn!(is_normal_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
unary_bool_fn!(is_subnormal_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
unary_bool_fn!(is_zero_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
unary_bool_fn!(is_nonzero_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
unary_bool_fn!(is_positive_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
unary_bool_fn!(is_negative_bool_f64, f64);

#[cfg(feature = "dtype-i8")]
unary_fn!(neg_i8, i8);
#[cfg(feature = "dtype-i32")]
unary_fn!(neg_i32, i32);
#[cfg(feature = "dtype-i64")]
unary_fn!(neg_i64, i64);
#[cfg(feature = "dtype-i8")]
unary_fn!(abs_i8, i8);
#[cfg(feature = "dtype-i32")]
unary_fn!(abs_i32, i32);
#[cfg(feature = "dtype-i64")]
unary_fn!(abs_i64, i64);
#[cfg(feature = "dtype-i8")]
unary_fn!(sign_i8, i8);
#[cfg(feature = "dtype-i32")]
unary_fn!(sign_i32, i32);
#[cfg(feature = "dtype-i64")]
unary_fn!(sign_i64, i64);
#[cfg(feature = "dtype-u8")]
unary_fn!(sign_u8, u8);
#[cfg(feature = "dtype-u32")]
unary_fn!(sign_u32, u32);
#[cfg(feature = "dtype-u64")]
unary_fn!(sign_u64, u64);
#[cfg(feature = "dtype-u8")]
unary_fn!(bitwise_not_u8, u8);
#[cfg(feature = "dtype-i8")]
unary_fn!(bitwise_not_i8, i8);
#[cfg(feature = "dtype-u32")]
unary_fn!(bitwise_not_u32, u32);
#[cfg(feature = "dtype-i32")]
unary_fn!(bitwise_not_i32, i32);
#[cfg(feature = "dtype-u64")]
unary_fn!(bitwise_not_u64, u64);
#[cfg(feature = "dtype-i64")]
unary_fn!(bitwise_not_i64, i64);
#[cfg(feature = "dtype-u8")]
unary_bool_fn!(is_zero_bool_u8, u8);
#[cfg(feature = "dtype-u8")]
unary_bool_fn!(is_nonzero_bool_u8, u8);
#[cfg(feature = "dtype-u8")]
unary_bool_fn!(is_positive_bool_u8, u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
unary_bool_fn!(is_zero_bool_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
unary_bool_fn!(is_nonzero_bool_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
unary_bool_fn!(is_positive_bool_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
unary_bool_fn!(is_negative_bool_i8, i8);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
unary_bool_fn!(is_zero_bool_u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
unary_bool_fn!(is_nonzero_bool_u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
unary_bool_fn!(is_positive_bool_u32, u32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
unary_bool_fn!(is_zero_bool_i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
unary_bool_fn!(is_nonzero_bool_i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
unary_bool_fn!(is_positive_bool_i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
unary_bool_fn!(is_negative_bool_i32, i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
unary_bool_fn!(is_zero_bool_u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
unary_bool_fn!(is_nonzero_bool_u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
unary_bool_fn!(is_positive_bool_u64, u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
unary_bool_fn!(is_zero_bool_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
unary_bool_fn!(is_nonzero_bool_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
unary_bool_fn!(is_positive_bool_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
unary_bool_fn!(is_negative_bool_i64, i64);
