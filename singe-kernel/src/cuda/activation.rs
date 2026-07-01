//! Elementwise nonlinearities and gated activations.

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
        utility::{ensure_binary_lengths, ensure_unary_lengths},
    },
    error::Result,
};

macro_rules! activation_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
        ) -> Result<()> {
            let len = ensure_unary_lengths(out, input)?;
            let stream = borrowed_stream(stream)?;
            cutile::activation::$name(&stream, output_pointer(out), input_pointer(input), len)
        }
    };
}

macro_rules! scalar_activation_fn {
    ($name:ident, $ty:ty, $param:ident) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            $param: $ty,
        ) -> Result<()> {
            let len = ensure_unary_lengths(out, input)?;
            let stream = borrowed_stream(stream)?;
            cutile::activation::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                $param,
                len,
            )
        }
    };
}

macro_rules! binary_scalar_activation_fn {
    ($name:ident, $ty:ty, $first:ident, $second:ident) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            $first: $ty,
            $second: $ty,
        ) -> Result<()> {
            let len = ensure_unary_lengths(out, input)?;
            let stream = borrowed_stream(stream)?;
            cutile::activation::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                $first,
                $second,
                len,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
activation_fn!(relu_f32, f32);
#[cfg(feature = "dtype-f32")]
activation_fn!(relu6_f32, f32);
#[cfg(feature = "dtype-f32")]
activation_fn!(sigmoid_f32, f32);
#[cfg(feature = "dtype-f32")]
activation_fn!(silu_f32, f32);
#[cfg(feature = "dtype-f32")]
activation_fn!(gelu_f32, f32);
#[cfg(feature = "dtype-f32")]
activation_fn!(softplus_f32, f32);
#[cfg(feature = "dtype-f32")]
activation_fn!(hard_sigmoid_f32, f32);
#[cfg(feature = "dtype-f32")]
activation_fn!(hard_swish_f32, f32);
#[cfg(feature = "dtype-f32")]
activation_fn!(mish_f32, f32);
#[cfg(feature = "dtype-f32")]
activation_fn!(selu_f32, f32);
#[cfg(feature = "dtype-f32")]
activation_fn!(softsign_f32, f32);
#[cfg(feature = "dtype-f32")]
activation_fn!(tanhshrink_f32, f32);
#[cfg(feature = "dtype-f32")]
activation_fn!(log_sigmoid_f32, f32);
#[cfg(feature = "dtype-f32")]
scalar_activation_fn!(leaky_relu_f32, f32, negative_slope);
#[cfg(feature = "dtype-f32")]
scalar_activation_fn!(elu_f32, f32, alpha);
#[cfg(feature = "dtype-f32")]
scalar_activation_fn!(celu_f32, f32, alpha);
#[cfg(feature = "dtype-f32")]
scalar_activation_fn!(hardshrink_f32, f32, lambda);
#[cfg(feature = "dtype-f32")]
scalar_activation_fn!(softshrink_f32, f32, lambda);
#[cfg(feature = "dtype-f32")]
binary_scalar_activation_fn!(threshold_f32, f32, threshold, value);
#[cfg(feature = "dtype-f32")]
binary_scalar_activation_fn!(hardtanh_f32, f32, min_value, max_value);

#[cfg(feature = "dtype-f16")]
activation_fn!(relu_f16, f16);
#[cfg(feature = "dtype-f16")]
activation_fn!(relu6_f16, f16);
#[cfg(feature = "dtype-f16")]
activation_fn!(sigmoid_f16, f16);
#[cfg(feature = "dtype-f16")]
activation_fn!(silu_f16, f16);
#[cfg(feature = "dtype-f16")]
activation_fn!(gelu_f16, f16);
#[cfg(feature = "dtype-f16")]
activation_fn!(softplus_f16, f16);
#[cfg(feature = "dtype-f16")]
activation_fn!(hard_sigmoid_f16, f16);
#[cfg(feature = "dtype-f16")]
activation_fn!(hard_swish_f16, f16);
#[cfg(feature = "dtype-f16")]
activation_fn!(mish_f16, f16);
#[cfg(feature = "dtype-f16")]
activation_fn!(selu_f16, f16);
#[cfg(feature = "dtype-f16")]
activation_fn!(softsign_f16, f16);
#[cfg(feature = "dtype-f16")]
activation_fn!(tanhshrink_f16, f16);
#[cfg(feature = "dtype-f16")]
activation_fn!(log_sigmoid_f16, f16);
#[cfg(feature = "dtype-f16")]
scalar_activation_fn!(leaky_relu_f16, f16, negative_slope);
#[cfg(feature = "dtype-f16")]
scalar_activation_fn!(elu_f16, f16, alpha);
#[cfg(feature = "dtype-f16")]
scalar_activation_fn!(celu_f16, f16, alpha);
#[cfg(feature = "dtype-f16")]
scalar_activation_fn!(hardshrink_f16, f16, lambda);
#[cfg(feature = "dtype-f16")]
scalar_activation_fn!(softshrink_f16, f16, lambda);
#[cfg(feature = "dtype-f16")]
binary_scalar_activation_fn!(threshold_f16, f16, threshold, value);
#[cfg(feature = "dtype-f16")]
binary_scalar_activation_fn!(hardtanh_f16, f16, min_value, max_value);

#[cfg(feature = "dtype-f64")]
activation_fn!(relu_f64, f64);
#[cfg(feature = "dtype-f64")]
activation_fn!(relu6_f64, f64);
#[cfg(feature = "dtype-f64")]
activation_fn!(sigmoid_f64, f64);
#[cfg(feature = "dtype-f64")]
activation_fn!(silu_f64, f64);
#[cfg(feature = "dtype-f64")]
activation_fn!(gelu_f64, f64);
#[cfg(feature = "dtype-f64")]
activation_fn!(softplus_f64, f64);
#[cfg(feature = "dtype-f64")]
activation_fn!(hard_sigmoid_f64, f64);
#[cfg(feature = "dtype-f64")]
activation_fn!(hard_swish_f64, f64);
#[cfg(feature = "dtype-f64")]
activation_fn!(mish_f64, f64);
#[cfg(feature = "dtype-f64")]
activation_fn!(selu_f64, f64);
#[cfg(feature = "dtype-f64")]
activation_fn!(softsign_f64, f64);
#[cfg(feature = "dtype-f64")]
activation_fn!(tanhshrink_f64, f64);
#[cfg(feature = "dtype-f64")]
activation_fn!(log_sigmoid_f64, f64);
#[cfg(feature = "dtype-f64")]
scalar_activation_fn!(leaky_relu_f64, f64, negative_slope);
#[cfg(feature = "dtype-f64")]
scalar_activation_fn!(elu_f64, f64, alpha);
#[cfg(feature = "dtype-f64")]
scalar_activation_fn!(celu_f64, f64, alpha);
#[cfg(feature = "dtype-f64")]
scalar_activation_fn!(hardshrink_f64, f64, lambda);
#[cfg(feature = "dtype-f64")]
scalar_activation_fn!(softshrink_f64, f64, lambda);
#[cfg(feature = "dtype-f64")]
binary_scalar_activation_fn!(threshold_f64, f64, threshold, value);
#[cfg(feature = "dtype-f64")]
binary_scalar_activation_fn!(hardtanh_f64, f64, min_value, max_value);

#[cfg(feature = "dtype-f32")]
pub fn swiglu_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    gate: &impl DeviceSlice<f32>,
) -> Result<()> {
    let len = ensure_binary_lengths(out, input, gate)?;
    let stream = borrowed_stream(stream)?;
    cutile::activation::swiglu_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(gate),
        len,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn swiglu_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    input: &impl DeviceSlice<f16>,
    gate: &impl DeviceSlice<f16>,
) -> Result<()> {
    let len = ensure_binary_lengths(out, input, gate)?;
    let stream = borrowed_stream(stream)?;
    cutile::activation::swiglu_f16(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(gate),
        len,
    )
}

#[cfg(feature = "dtype-f64")]
pub fn swiglu_f64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f64>,
    input: &impl DeviceSlice<f64>,
    gate: &impl DeviceSlice<f64>,
) -> Result<()> {
    let len = ensure_binary_lengths(out, input, gate)?;
    let stream = borrowed_stream(stream)?;
    cutile::activation::swiglu_f64(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(gate),
        len,
    )
}
