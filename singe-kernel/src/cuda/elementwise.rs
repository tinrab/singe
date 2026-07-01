//! Binary/ternary elementwise operations, clamps, masks, and selects.

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
        utility::ensure_binary_lengths,
    },
    error::Result,
    utility::ensure_len,
};

macro_rules! binary_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            lhs: &impl DeviceSlice<$ty>,
            rhs: &impl DeviceSlice<$ty>,
        ) -> Result<()> {
            let len = ensure_binary_lengths(out, lhs, rhs)?;
            let stream = borrowed_stream(stream)?;
            cutile::elementwise::$name(
                &stream,
                output_pointer(out),
                input_pointer(lhs),
                input_pointer(rhs),
                len,
            )
        }
    };
}

macro_rules! binary_bool_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<u8>,
            lhs: &impl DeviceSlice<$ty>,
            rhs: &impl DeviceSlice<$ty>,
        ) -> Result<()> {
            let len = out.len();
            ensure_len(lhs.len(), len)?;
            ensure_len(rhs.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::elementwise::$name(
                &stream,
                output_pointer(out),
                input_pointer(lhs),
                input_pointer(rhs),
                len,
            )
        }
    };
}

macro_rules! ternary_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            lhs: &impl DeviceSlice<$ty>,
            rhs: &impl DeviceSlice<$ty>,
            acc: &impl DeviceSlice<$ty>,
        ) -> Result<()> {
            let len = ensure_binary_lengths(out, lhs, rhs)?;
            ensure_len(acc.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::elementwise::$name(
                &stream,
                output_pointer(out),
                input_pointer(lhs),
                input_pointer(rhs),
                input_pointer(acc),
                len,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
binary_fn!(add_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(sub_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(mul_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(div_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(pow_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(hypot_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(squared_difference_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(logaddexp_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(xlogy_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(max_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(min_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(modulo_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(remainder_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(atan2_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(equal_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(not_equal_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(less_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(less_equal_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(greater_f32, f32);
#[cfg(feature = "dtype-f32")]
binary_fn!(greater_equal_f32, f32);
#[cfg(feature = "dtype-f32")]
ternary_fn!(fma_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
binary_bool_fn!(equal_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
binary_bool_fn!(not_equal_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
binary_bool_fn!(less_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
binary_bool_fn!(less_equal_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
binary_bool_fn!(greater_bool_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
binary_bool_fn!(greater_equal_bool_f32, f32);

#[cfg(feature = "dtype-f16")]
binary_fn!(add_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(sub_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(mul_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(div_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(pow_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(hypot_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(squared_difference_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(logaddexp_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(xlogy_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(max_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(min_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(modulo_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(remainder_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(atan2_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(equal_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(not_equal_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(less_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(less_equal_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(greater_f16, f16);
#[cfg(feature = "dtype-f16")]
binary_fn!(greater_equal_f16, f16);
#[cfg(feature = "dtype-f16")]
ternary_fn!(fma_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
binary_bool_fn!(equal_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
binary_bool_fn!(not_equal_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
binary_bool_fn!(less_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
binary_bool_fn!(less_equal_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
binary_bool_fn!(greater_bool_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
binary_bool_fn!(greater_equal_bool_f16, f16);

#[cfg(feature = "dtype-f64")]
binary_fn!(add_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(sub_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(mul_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(div_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(pow_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(hypot_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(squared_difference_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(logaddexp_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(xlogy_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(max_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(min_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(modulo_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(remainder_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(atan2_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(equal_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(not_equal_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(less_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(less_equal_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(greater_f64, f64);
#[cfg(feature = "dtype-f64")]
binary_fn!(greater_equal_f64, f64);
#[cfg(feature = "dtype-f64")]
ternary_fn!(fma_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
binary_bool_fn!(equal_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
binary_bool_fn!(not_equal_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
binary_bool_fn!(less_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
binary_bool_fn!(less_equal_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
binary_bool_fn!(greater_bool_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
binary_bool_fn!(greater_equal_bool_f64, f64);

#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
binary_bool_fn!(equal_bool_u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
binary_bool_fn!(not_equal_bool_u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
binary_bool_fn!(less_bool_u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
binary_bool_fn!(less_equal_bool_u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
binary_bool_fn!(greater_bool_u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
binary_bool_fn!(greater_equal_bool_u32, u32);

#[cfg(feature = "dtype-u8")]
binary_bool_fn!(equal_bool_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_bool_fn!(not_equal_bool_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_bool_fn!(less_bool_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_bool_fn!(less_equal_bool_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_bool_fn!(greater_bool_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_bool_fn!(greater_equal_bool_u8, u8);

#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
binary_bool_fn!(equal_bool_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
binary_bool_fn!(not_equal_bool_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
binary_bool_fn!(less_bool_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
binary_bool_fn!(less_equal_bool_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
binary_bool_fn!(greater_bool_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
binary_bool_fn!(greater_equal_bool_i8, i8);

#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
binary_bool_fn!(equal_bool_i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
binary_bool_fn!(not_equal_bool_i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
binary_bool_fn!(less_bool_i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
binary_bool_fn!(less_equal_bool_i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
binary_bool_fn!(greater_bool_i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
binary_bool_fn!(greater_equal_bool_i32, i32);

#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
binary_bool_fn!(equal_bool_u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
binary_bool_fn!(not_equal_bool_u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
binary_bool_fn!(less_bool_u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
binary_bool_fn!(less_equal_bool_u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
binary_bool_fn!(greater_bool_u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
binary_bool_fn!(greater_equal_bool_u64, u64);

#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
binary_bool_fn!(equal_bool_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
binary_bool_fn!(not_equal_bool_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
binary_bool_fn!(less_bool_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
binary_bool_fn!(less_equal_bool_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
binary_bool_fn!(greater_bool_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
binary_bool_fn!(greater_equal_bool_i64, i64);

#[cfg(feature = "dtype-u32")]
binary_fn!(min_u32, u32);
#[cfg(feature = "dtype-u32")]
binary_fn!(max_u32, u32);
#[cfg(feature = "dtype-u32")]
binary_fn!(add_u32, u32);
#[cfg(feature = "dtype-u32")]
binary_fn!(sub_u32, u32);
#[cfg(feature = "dtype-u32")]
binary_fn!(mul_u32, u32);
#[cfg(feature = "dtype-u32")]
binary_fn!(div_u32, u32);
#[cfg(feature = "dtype-u32")]
binary_fn!(modulo_u32, u32);
#[cfg(feature = "dtype-u32")]
binary_fn!(remainder_u32, u32);
#[cfg(feature = "dtype-u32")]
binary_fn!(bitwise_and_u32, u32);
#[cfg(feature = "dtype-u32")]
binary_fn!(bitwise_or_u32, u32);
#[cfg(feature = "dtype-u32")]
binary_fn!(bitwise_xor_u32, u32);
#[cfg(feature = "dtype-u32")]
binary_fn!(shift_left_u32, u32);
#[cfg(feature = "dtype-u32")]
binary_fn!(shift_right_u32, u32);
#[cfg(feature = "dtype-u8")]
binary_fn!(min_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_fn!(max_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_fn!(add_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_fn!(sub_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_fn!(mul_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_fn!(div_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_fn!(modulo_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_fn!(remainder_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_fn!(bitwise_and_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_fn!(bitwise_or_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_fn!(bitwise_xor_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_fn!(shift_left_u8, u8);
#[cfg(feature = "dtype-u8")]
binary_fn!(shift_right_u8, u8);
#[cfg(feature = "dtype-i8")]
binary_fn!(min_i8, i8);
#[cfg(feature = "dtype-i8")]
binary_fn!(max_i8, i8);
#[cfg(feature = "dtype-i8")]
binary_fn!(add_i8, i8);
#[cfg(feature = "dtype-i8")]
binary_fn!(sub_i8, i8);
#[cfg(feature = "dtype-i8")]
binary_fn!(mul_i8, i8);
#[cfg(feature = "dtype-i8")]
binary_fn!(div_i8, i8);
#[cfg(feature = "dtype-i8")]
binary_fn!(modulo_i8, i8);
#[cfg(feature = "dtype-i8")]
binary_fn!(remainder_i8, i8);
#[cfg(feature = "dtype-i8")]
binary_fn!(bitwise_and_i8, i8);
#[cfg(feature = "dtype-i8")]
binary_fn!(bitwise_or_i8, i8);
#[cfg(feature = "dtype-i8")]
binary_fn!(bitwise_xor_i8, i8);
#[cfg(feature = "dtype-i8")]
binary_fn!(shift_left_i8, i8);
#[cfg(feature = "dtype-i8")]
binary_fn!(shift_right_i8, i8);
#[cfg(feature = "dtype-i32")]
binary_fn!(min_i32, i32);
#[cfg(feature = "dtype-i32")]
binary_fn!(max_i32, i32);
#[cfg(feature = "dtype-i32")]
binary_fn!(add_i32, i32);
#[cfg(feature = "dtype-i32")]
binary_fn!(sub_i32, i32);
#[cfg(feature = "dtype-i32")]
binary_fn!(mul_i32, i32);
#[cfg(feature = "dtype-i32")]
binary_fn!(div_i32, i32);
#[cfg(feature = "dtype-i32")]
binary_fn!(modulo_i32, i32);
#[cfg(feature = "dtype-i32")]
binary_fn!(remainder_i32, i32);
#[cfg(feature = "dtype-i32")]
binary_fn!(bitwise_and_i32, i32);
#[cfg(feature = "dtype-i32")]
binary_fn!(bitwise_or_i32, i32);
#[cfg(feature = "dtype-i32")]
binary_fn!(bitwise_xor_i32, i32);
#[cfg(feature = "dtype-i32")]
binary_fn!(shift_left_i32, i32);
#[cfg(feature = "dtype-i32")]
binary_fn!(shift_right_i32, i32);
#[cfg(feature = "dtype-u64")]
binary_fn!(min_u64, u64);
#[cfg(feature = "dtype-u64")]
binary_fn!(max_u64, u64);
#[cfg(feature = "dtype-u64")]
binary_fn!(add_u64, u64);
#[cfg(feature = "dtype-u64")]
binary_fn!(sub_u64, u64);
#[cfg(feature = "dtype-u64")]
binary_fn!(mul_u64, u64);
#[cfg(feature = "dtype-u64")]
binary_fn!(div_u64, u64);
#[cfg(feature = "dtype-u64")]
binary_fn!(modulo_u64, u64);
#[cfg(feature = "dtype-u64")]
binary_fn!(remainder_u64, u64);
#[cfg(feature = "dtype-u64")]
binary_fn!(bitwise_and_u64, u64);
#[cfg(feature = "dtype-u64")]
binary_fn!(bitwise_or_u64, u64);
#[cfg(feature = "dtype-u64")]
binary_fn!(bitwise_xor_u64, u64);
#[cfg(feature = "dtype-u64")]
binary_fn!(shift_left_u64, u64);
#[cfg(feature = "dtype-u64")]
binary_fn!(shift_right_u64, u64);
#[cfg(feature = "dtype-i64")]
binary_fn!(min_i64, i64);
#[cfg(feature = "dtype-i64")]
binary_fn!(max_i64, i64);
#[cfg(feature = "dtype-i64")]
binary_fn!(add_i64, i64);
#[cfg(feature = "dtype-i64")]
binary_fn!(sub_i64, i64);
#[cfg(feature = "dtype-i64")]
binary_fn!(mul_i64, i64);
#[cfg(feature = "dtype-i64")]
binary_fn!(div_i64, i64);
#[cfg(feature = "dtype-i64")]
binary_fn!(modulo_i64, i64);
#[cfg(feature = "dtype-i64")]
binary_fn!(remainder_i64, i64);
#[cfg(feature = "dtype-i64")]
binary_fn!(bitwise_and_i64, i64);
#[cfg(feature = "dtype-i64")]
binary_fn!(bitwise_or_i64, i64);
#[cfg(feature = "dtype-i64")]
binary_fn!(bitwise_xor_i64, i64);
#[cfg(feature = "dtype-i64")]
binary_fn!(shift_left_i64, i64);
#[cfg(feature = "dtype-i64")]
binary_fn!(shift_right_i64, i64);

#[cfg(feature = "dtype-f32")]
pub fn clamp_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    min: f32,
    max: f32,
) -> Result<()> {
    let len = out.len();
    ensure_len(input.len(), len)?;
    let stream = borrowed_stream(stream)?;
    cutile::elementwise::clamp_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        min,
        max,
        len,
    )
}

#[cfg(feature = "dtype-f64")]
pub fn clamp_f64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f64>,
    input: &impl DeviceSlice<f64>,
    min: f64,
    max: f64,
) -> Result<()> {
    let len = out.len();
    ensure_len(input.len(), len)?;
    let stream = borrowed_stream(stream)?;
    cutile::elementwise::clamp_f64(
        &stream,
        output_pointer(out),
        input_pointer(input),
        min,
        max,
        len,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn clamp_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    input: &impl DeviceSlice<f16>,
    min: f16,
    max: f16,
) -> Result<()> {
    let len = out.len();
    ensure_len(input.len(), len)?;
    let stream = borrowed_stream(stream)?;
    cutile::elementwise::clamp_f16(
        &stream,
        output_pointer(out),
        input_pointer(input),
        min,
        max,
        len,
    )
}

macro_rules! clamp_helper_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            min: $ty,
            max: $ty,
        ) -> Result<()> {
            let len = out.len();
            ensure_len(input.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::elementwise::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                min,
                max,
                len,
            )
        }
    };
}

#[cfg(feature = "dtype-u8")]
clamp_helper_fn!(clamp_u8, u8);
#[cfg(feature = "dtype-i8")]
clamp_helper_fn!(clamp_i8, i8);
#[cfg(feature = "dtype-u32")]
clamp_helper_fn!(clamp_u32, u32);
#[cfg(feature = "dtype-i32")]
clamp_helper_fn!(clamp_i32, i32);
#[cfg(feature = "dtype-u64")]
clamp_helper_fn!(clamp_u64, u64);
#[cfg(feature = "dtype-i64")]
clamp_helper_fn!(clamp_i64, i64);

#[cfg(feature = "dtype-f32")]
pub fn lerp_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    scalar: f32,
) -> Result<()> {
    let len = ensure_binary_lengths(out, lhs, rhs)?;
    let stream = borrowed_stream(stream)?;
    cutile::elementwise::lerp_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        scalar,
        len,
    )
}

#[cfg(feature = "dtype-f64")]
pub fn lerp_f64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f64>,
    lhs: &impl DeviceSlice<f64>,
    rhs: &impl DeviceSlice<f64>,
    scalar: f64,
) -> Result<()> {
    let len = ensure_binary_lengths(out, lhs, rhs)?;
    let stream = borrowed_stream(stream)?;
    cutile::elementwise::lerp_f64(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        scalar,
        len,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn lerp_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    scalar: f16,
) -> Result<()> {
    let len = ensure_binary_lengths(out, lhs, rhs)?;
    let stream = borrowed_stream(stream)?;
    cutile::elementwise::lerp_f16(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        scalar,
        len,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn where_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    condition: &impl DeviceSlice<f32>,
    x: &impl DeviceSlice<f32>,
    y: &impl DeviceSlice<f32>,
) -> Result<()> {
    let len = ensure_binary_lengths(out, x, y)?;
    ensure_len(condition.len(), len)?;
    let stream = borrowed_stream(stream)?;
    cutile::elementwise::where_f32(
        &stream,
        output_pointer(out),
        input_pointer(condition),
        input_pointer(x),
        input_pointer(y),
        len,
    )
}

#[cfg(feature = "dtype-f64")]
pub fn where_f64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f64>,
    condition: &impl DeviceSlice<f64>,
    x: &impl DeviceSlice<f64>,
    y: &impl DeviceSlice<f64>,
) -> Result<()> {
    let len = ensure_binary_lengths(out, x, y)?;
    ensure_len(condition.len(), len)?;
    let stream = borrowed_stream(stream)?;
    cutile::elementwise::where_f64(
        &stream,
        output_pointer(out),
        input_pointer(condition),
        input_pointer(x),
        input_pointer(y),
        len,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn where_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    condition: &impl DeviceSlice<f16>,
    x: &impl DeviceSlice<f16>,
    y: &impl DeviceSlice<f16>,
) -> Result<()> {
    let len = ensure_binary_lengths(out, x, y)?;
    ensure_len(condition.len(), len)?;
    let stream = borrowed_stream(stream)?;
    cutile::elementwise::where_f16(
        &stream,
        output_pointer(out),
        input_pointer(condition),
        input_pointer(x),
        input_pointer(y),
        len,
    )
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
pub fn where_bool_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    condition: &impl DeviceSlice<u8>,
    x: &impl DeviceSlice<f32>,
    y: &impl DeviceSlice<f32>,
) -> Result<()> {
    let len = ensure_binary_lengths(out, x, y)?;
    ensure_len(condition.len(), len)?;
    let stream = borrowed_stream(stream)?;
    cutile::elementwise::where_bool_f32(
        &stream,
        output_pointer(out),
        input_pointer(condition),
        input_pointer(x),
        input_pointer(y),
        len,
    )
}

#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
pub fn where_bool_f64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f64>,
    condition: &impl DeviceSlice<u8>,
    x: &impl DeviceSlice<f64>,
    y: &impl DeviceSlice<f64>,
) -> Result<()> {
    let len = ensure_binary_lengths(out, x, y)?;
    ensure_len(condition.len(), len)?;
    let stream = borrowed_stream(stream)?;
    cutile::elementwise::where_bool_f64(
        &stream,
        output_pointer(out),
        input_pointer(condition),
        input_pointer(x),
        input_pointer(y),
        len,
    )
}

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
pub fn where_bool_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    condition: &impl DeviceSlice<u8>,
    x: &impl DeviceSlice<f16>,
    y: &impl DeviceSlice<f16>,
) -> Result<()> {
    let len = ensure_binary_lengths(out, x, y)?;
    ensure_len(condition.len(), len)?;
    let stream = borrowed_stream(stream)?;
    cutile::elementwise::where_bool_f16(
        &stream,
        output_pointer(out),
        input_pointer(condition),
        input_pointer(x),
        input_pointer(y),
        len,
    )
}

macro_rules! where_bool_helper_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            condition: &impl DeviceSlice<u8>,
            x: &impl DeviceSlice<$ty>,
            y: &impl DeviceSlice<$ty>,
        ) -> Result<()> {
            let len = ensure_binary_lengths(out, x, y)?;
            ensure_len(condition.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::elementwise::$name(
                &stream,
                output_pointer(out),
                input_pointer(condition),
                input_pointer(x),
                input_pointer(y),
                len,
            )
        }
    };
}

#[cfg(feature = "dtype-u8")]
where_bool_helper_fn!(where_bool_u8, u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
where_bool_helper_fn!(where_bool_i8, i8);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
where_bool_helper_fn!(where_bool_u32, u32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
where_bool_helper_fn!(where_bool_i32, i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
where_bool_helper_fn!(where_bool_u64, u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
where_bool_helper_fn!(where_bool_i64, i64);

macro_rules! masked_fill_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            mask: &impl DeviceSlice<u8>,
            value: $ty,
        ) -> Result<()> {
            let len = out.len();
            ensure_len(input.len(), len)?;
            ensure_len(mask.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::elementwise::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(mask),
                value,
                len,
            )
        }
    };
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
masked_fill_fn!(masked_fill_f32, f32);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
masked_fill_fn!(masked_fill_f16, f16);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
masked_fill_fn!(masked_fill_f64, f64);

macro_rules! masked_fill_helper_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            mask: &impl DeviceSlice<u8>,
            value: $ty,
        ) -> Result<()> {
            let len = out.len();
            ensure_len(input.len(), len)?;
            ensure_len(mask.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::elementwise::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(mask),
                value,
                len,
            )
        }
    };
}

#[cfg(feature = "dtype-u8")]
masked_fill_helper_fn!(masked_fill_u8, u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
masked_fill_helper_fn!(masked_fill_i8, i8);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
masked_fill_helper_fn!(masked_fill_u32, u32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
masked_fill_helper_fn!(masked_fill_i32, i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
masked_fill_helper_fn!(masked_fill_u64, u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
masked_fill_helper_fn!(masked_fill_i64, i64);
