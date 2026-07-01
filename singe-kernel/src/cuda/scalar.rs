//! Scalar arithmetic and scalar comparisons on device slices.

#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceSlice, DeviceSliceMut},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::{
        interop::{borrowed_stream, input_pointer, output_pointer},
        utility::ensure_unary_lengths,
    },
    error::Result,
};

macro_rules! scalar_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            scalar: $ty,
        ) -> Result<()> {
            let len = ensure_unary_lengths(out, input)?;
            let stream = borrowed_stream(stream)?;
            cutile::scalar::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                scalar,
                len,
            )
        }
    };
}

macro_rules! scalar_bool_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<u8>,
            input: &impl DeviceSlice<$ty>,
            scalar: $ty,
        ) -> Result<()> {
            let len = out.len();
            crate::utility::ensure_len(input.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::scalar::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                scalar,
                len,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
scalar_fn!(add_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(sub_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(rsub_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(mul_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(scale_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(div_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(rdiv_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(modulo_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(rmodulo_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(atan2_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(ratan2_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(pow_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(rpow_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(hypot_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(squared_difference_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(logaddexp_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(xlogy_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(rxlogy_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(min_scalar_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_fn!(max_scalar_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
scalar_bool_fn!(equal_scalar_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
scalar_bool_fn!(not_equal_scalar_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
scalar_bool_fn!(less_scalar_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
scalar_bool_fn!(less_equal_scalar_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
scalar_bool_fn!(greater_scalar_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
scalar_bool_fn!(greater_equal_scalar_bool_f32, f32);

#[cfg(feature = "dtype-f16")]
scalar_fn!(add_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(sub_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(rsub_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(mul_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(scale_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(div_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(rdiv_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(modulo_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(rmodulo_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(atan2_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(ratan2_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(pow_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(rpow_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(hypot_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(squared_difference_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(logaddexp_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(xlogy_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(rxlogy_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(min_scalar_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_fn!(max_scalar_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
scalar_bool_fn!(equal_scalar_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
scalar_bool_fn!(not_equal_scalar_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
scalar_bool_fn!(less_scalar_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
scalar_bool_fn!(less_equal_scalar_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
scalar_bool_fn!(greater_scalar_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
scalar_bool_fn!(greater_equal_scalar_bool_f16, f16);

#[cfg(feature = "dtype-f64")]
scalar_fn!(add_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(sub_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(rsub_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(mul_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(scale_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(div_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(rdiv_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(modulo_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(rmodulo_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(atan2_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(ratan2_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(pow_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(rpow_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(hypot_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(squared_difference_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(logaddexp_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(xlogy_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(rxlogy_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(min_scalar_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_fn!(max_scalar_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
scalar_bool_fn!(equal_scalar_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
scalar_bool_fn!(not_equal_scalar_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
scalar_bool_fn!(less_scalar_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
scalar_bool_fn!(less_equal_scalar_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
scalar_bool_fn!(greater_scalar_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
scalar_bool_fn!(greater_equal_scalar_bool_f64, f64);

#[cfg(feature = "dtype-u8")]
scalar_bool_fn!(equal_scalar_bool_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_bool_fn!(not_equal_scalar_bool_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_bool_fn!(less_scalar_bool_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_bool_fn!(less_equal_scalar_bool_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_bool_fn!(greater_scalar_bool_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_bool_fn!(greater_equal_scalar_bool_u8, u8);

#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
scalar_bool_fn!(equal_scalar_bool_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
scalar_bool_fn!(not_equal_scalar_bool_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
scalar_bool_fn!(less_scalar_bool_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
scalar_bool_fn!(less_equal_scalar_bool_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
scalar_bool_fn!(greater_scalar_bool_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
scalar_bool_fn!(greater_equal_scalar_bool_i8, i8);

#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
scalar_bool_fn!(equal_scalar_bool_u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
scalar_bool_fn!(not_equal_scalar_bool_u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
scalar_bool_fn!(less_scalar_bool_u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
scalar_bool_fn!(less_equal_scalar_bool_u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
scalar_bool_fn!(greater_scalar_bool_u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
scalar_bool_fn!(greater_equal_scalar_bool_u32, u32);

#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
scalar_bool_fn!(equal_scalar_bool_i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
scalar_bool_fn!(not_equal_scalar_bool_i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
scalar_bool_fn!(less_scalar_bool_i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
scalar_bool_fn!(less_equal_scalar_bool_i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
scalar_bool_fn!(greater_scalar_bool_i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
scalar_bool_fn!(greater_equal_scalar_bool_i32, i32);

#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
scalar_bool_fn!(equal_scalar_bool_u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
scalar_bool_fn!(not_equal_scalar_bool_u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
scalar_bool_fn!(less_scalar_bool_u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
scalar_bool_fn!(less_equal_scalar_bool_u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
scalar_bool_fn!(greater_scalar_bool_u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
scalar_bool_fn!(greater_equal_scalar_bool_u64, u64);

#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
scalar_bool_fn!(equal_scalar_bool_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
scalar_bool_fn!(not_equal_scalar_bool_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
scalar_bool_fn!(less_scalar_bool_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
scalar_bool_fn!(less_equal_scalar_bool_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
scalar_bool_fn!(greater_scalar_bool_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
scalar_bool_fn!(greater_equal_scalar_bool_i64, i64);

#[cfg(feature = "dtype-u32")]
scalar_fn!(min_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(max_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(add_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(sub_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(rsub_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(mul_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(div_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(rdiv_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(modulo_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(rmodulo_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(bitwise_and_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(bitwise_or_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(bitwise_xor_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(shift_left_scalar_u32, u32);
#[cfg(feature = "dtype-u32")]
scalar_fn!(shift_right_scalar_u32, u32);
#[cfg(feature = "dtype-u8")]
scalar_fn!(min_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(max_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(add_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(sub_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(rsub_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(mul_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(div_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(rdiv_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(modulo_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(rmodulo_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(bitwise_and_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(bitwise_or_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(bitwise_xor_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(shift_left_scalar_u8, u8);
#[cfg(feature = "dtype-u8")]
scalar_fn!(shift_right_scalar_u8, u8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(min_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(max_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(add_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(sub_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(rsub_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(mul_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(div_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(rdiv_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(modulo_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(rmodulo_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(bitwise_and_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(bitwise_or_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(bitwise_xor_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(shift_left_scalar_i8, i8);
#[cfg(feature = "dtype-i8")]
scalar_fn!(shift_right_scalar_i8, i8);
#[cfg(feature = "dtype-i32")]
scalar_fn!(min_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(max_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(add_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(sub_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(rsub_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(mul_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(div_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(rdiv_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(modulo_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(rmodulo_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(bitwise_and_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(bitwise_or_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(bitwise_xor_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(shift_left_scalar_i32, i32);
#[cfg(feature = "dtype-i32")]
scalar_fn!(shift_right_scalar_i32, i32);
#[cfg(feature = "dtype-u64")]
scalar_fn!(min_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(max_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(add_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(sub_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(rsub_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(mul_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(div_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(rdiv_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(modulo_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(rmodulo_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(bitwise_and_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(bitwise_or_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(bitwise_xor_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(shift_left_scalar_u64, u64);
#[cfg(feature = "dtype-u64")]
scalar_fn!(shift_right_scalar_u64, u64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(min_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(max_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(add_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(sub_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(rsub_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(mul_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(div_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(rdiv_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(modulo_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(rmodulo_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(bitwise_and_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(bitwise_or_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(bitwise_xor_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(shift_left_scalar_i64, i64);
#[cfg(feature = "dtype-i64")]
scalar_fn!(shift_right_scalar_i64, i64);
