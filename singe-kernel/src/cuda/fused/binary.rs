#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use ::cutile::cuda_async::device_buffer::DevicePointer;
#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
#[cfg(any(
    feature = "dtype_bf16",
    feature = "dtype_f16",
    feature = "dtype_f32",
    feature = "dtype_f64"
))]
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use crate::cuda::{
    fused::{validate_binary_fused_lengths, validate_silu_and_mul_packed_lengths},
    interop::{input_pointer, write_only_pointer},
    utility::ensure_unary_lengths,
};
#[cfg(any(
    feature = "dtype_bf16",
    feature = "dtype_f16",
    feature = "dtype_f32",
    feature = "dtype_f64"
))]
use crate::{
    cuda::{
        autotune::{TuneOptions, TuneReport},
        cutile,
        fused::{
            Fused1dConfig, Fused2dConfig, FusedTileTuning, fused_1d, fused_2d, tune_fused_1d,
            tune_fused_2d,
        },
    },
    error::Result,
};

#[cfg(feature = "dtype_bf16")]
autotuned_binary_fused_fn!(
    "Computes `silu(lhs) * rhs` elementwise.",
    silu_mul_bf16,
    tune_silu_mul_bf16,
    bf16,
    "bf16",
    silu_mul_bf16,
    silu_mul_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `silu(lhs) * rhs` elementwise.",
    silu_mul_f16,
    tune_silu_mul_f16,
    f16,
    "f16",
    silu_mul_f16,
    silu_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `silu(lhs) * rhs` elementwise.",
    silu_mul_f32,
    tune_silu_mul_f32,
    f32,
    "f32",
    silu_mul_f32,
    silu_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `silu(lhs) * rhs` elementwise.",
    silu_mul_f64,
    tune_silu_mul_f64,
    f64,
    "f64",
    silu_mul_f64,
    silu_mul_f64_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_binary_fused_2d_fn!(
    silu_mul_2d_bf16,
    tune_silu_mul_2d_bf16,
    bf16,
    "bf16",
    silu_mul_2d_bf16,
    silu_mul_2d_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    silu_mul_2d_f16,
    tune_silu_mul_2d_f16,
    f16,
    "f16",
    silu_mul_2d_f16,
    silu_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    silu_mul_2d_f32,
    tune_silu_mul_2d_f32,
    f32,
    "f32",
    silu_mul_2d_f32,
    silu_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    silu_mul_2d_f64,
    tune_silu_mul_2d_f64,
    f64,
    "f64",
    silu_mul_2d_f64,
    silu_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_binary_fused_fn!(
    "Computes `silu(lhs) * rhs` elementwise.",
    swiglu_bf16,
    tune_swiglu_bf16,
    bf16,
    "bf16",
    swiglu_bf16,
    swiglu_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `silu(lhs) * rhs` elementwise.",
    swiglu_f16,
    tune_swiglu_f16,
    f16,
    "f16",
    swiglu_f16,
    swiglu_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `silu(lhs) * rhs` elementwise.",
    swiglu_f32,
    tune_swiglu_f32,
    f32,
    "f32",
    swiglu_f32,
    swiglu_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `silu(lhs) * rhs` elementwise.",
    swiglu_f64,
    tune_swiglu_f64,
    f64,
    "f64",
    swiglu_f64,
    swiglu_f64_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_binary_fused_2d_fn!(
    swiglu_2d_bf16,
    tune_swiglu_2d_bf16,
    bf16,
    "bf16",
    swiglu_2d_bf16,
    swiglu_2d_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    swiglu_2d_f16,
    tune_swiglu_2d_f16,
    f16,
    "f16",
    swiglu_2d_f16,
    swiglu_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    swiglu_2d_f32,
    tune_swiglu_2d_f32,
    f32,
    "f32",
    swiglu_2d_f32,
    swiglu_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    swiglu_2d_f64,
    tune_swiglu_2d_f64,
    f64,
    "f64",
    swiglu_2d_f64,
    swiglu_2d_f64_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_binary_fused_fn!(
    "Computes `gelu(lhs) * rhs` elementwise using the tanh approximation.",
    gelu_mul_bf16,
    tune_gelu_mul_bf16,
    bf16,
    "bf16",
    gelu_mul_bf16,
    gelu_mul_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `gelu(lhs) * rhs` elementwise using the tanh approximation.",
    gelu_mul_f16,
    tune_gelu_mul_f16,
    f16,
    "f16",
    gelu_mul_f16,
    gelu_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `gelu(lhs) * rhs` elementwise using the tanh approximation.",
    gelu_mul_f32,
    tune_gelu_mul_f32,
    f32,
    "f32",
    gelu_mul_f32,
    gelu_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `gelu(lhs) * rhs` elementwise using the tanh approximation.",
    gelu_mul_f64,
    tune_gelu_mul_f64,
    f64,
    "f64",
    gelu_mul_f64,
    gelu_mul_f64_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_binary_fused_2d_fn!(
    gelu_mul_2d_bf16,
    tune_gelu_mul_2d_bf16,
    bf16,
    "bf16",
    gelu_mul_2d_bf16,
    gelu_mul_2d_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    gelu_mul_2d_f16,
    tune_gelu_mul_2d_f16,
    f16,
    "f16",
    gelu_mul_2d_f16,
    gelu_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    gelu_mul_2d_f32,
    tune_gelu_mul_2d_f32,
    f32,
    "f32",
    gelu_mul_2d_f32,
    gelu_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    gelu_mul_2d_f64,
    tune_gelu_mul_2d_f64,
    f64,
    "f64",
    gelu_mul_2d_f64,
    gelu_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_binary_fused_fn!(
    "Computes `gelu(lhs) * rhs` elementwise using the tanh approximation.",
    geglu_approx_bf16,
    tune_geglu_approx_bf16,
    bf16,
    "bf16",
    geglu_approx_bf16,
    geglu_approx_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `gelu(lhs) * rhs` elementwise using the tanh approximation.",
    geglu_approx_f16,
    tune_geglu_approx_f16,
    f16,
    "f16",
    geglu_approx_f16,
    geglu_approx_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `gelu(lhs) * rhs` elementwise using the tanh approximation.",
    geglu_approx_f32,
    tune_geglu_approx_f32,
    f32,
    "f32",
    geglu_approx_f32,
    geglu_approx_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `gelu(lhs) * rhs` elementwise using the tanh approximation.",
    geglu_approx_f64,
    tune_geglu_approx_f64,
    f64,
    "f64",
    geglu_approx_f64,
    geglu_approx_f64_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_binary_fused_2d_fn!(
    geglu_approx_2d_bf16,
    tune_geglu_approx_2d_bf16,
    bf16,
    "bf16",
    geglu_approx_2d_bf16,
    geglu_approx_2d_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    geglu_approx_2d_f16,
    tune_geglu_approx_2d_f16,
    f16,
    "f16",
    geglu_approx_2d_f16,
    geglu_approx_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    geglu_approx_2d_f32,
    tune_geglu_approx_2d_f32,
    f32,
    "f32",
    geglu_approx_2d_f32,
    geglu_approx_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    geglu_approx_2d_f64,
    tune_geglu_approx_2d_f64,
    f64,
    "f64",
    geglu_approx_2d_f64,
    geglu_approx_2d_f64_tuned
);

#[cfg(feature = "dtype_f32")]
binary_fused_fn!(
    "Computes `exact_gelu(lhs) * rhs` elementwise.",
    exact_gelu_mul_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_fused_fn!(
    "Applies exact GELU using the error function elementwise.",
    exact_gelu_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
binary_fused_fn!(
    "Computes `exact_gelu(lhs) * rhs` elementwise.",
    exact_gelu_mul_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_fused_fn!(
    "Applies exact GELU using the error function elementwise.",
    exact_gelu_f16,
    f16
);

#[cfg(feature = "dtype_bf16")]
binary_fused_fn!(
    "Computes `exact_gelu(lhs) * rhs` elementwise.",
    exact_gelu_mul_bf16,
    bf16
);

#[cfg(feature = "dtype_bf16")]
unary_fused_fn!(
    "Applies exact GELU using the error function elementwise.",
    exact_gelu_bf16,
    bf16
);

#[cfg(feature = "dtype_f32")]
binary_fused_fn!(
    "Computes `exact_gelu(lhs) * rhs` elementwise.",
    geglu_exact_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
binary_fused_fn!(
    "Computes `exact_gelu(lhs) * rhs` elementwise.",
    geglu_exact_f16,
    f16
);

#[cfg(feature = "dtype_bf16")]
binary_fused_fn!(
    "Computes `exact_gelu(lhs) * rhs` elementwise.",
    geglu_exact_bf16,
    bf16
);

#[cfg(feature = "dtype_f32")]
silu_and_mul_fn!(silu_and_mul_f32, f32);

#[cfg(feature = "dtype_f16")]
silu_and_mul_fn!(silu_and_mul_f16, f16);

#[cfg(feature = "dtype_bf16")]
silu_and_mul_fn!(silu_and_mul_bf16, bf16);

#[cfg(feature = "dtype_bf16")]
autotuned_binary_fused_fn!(
    "Computes `sigmoid(lhs) * rhs` elementwise.",
    sigmoid_mul_bf16,
    tune_sigmoid_mul_bf16,
    bf16,
    "bf16",
    sigmoid_mul_bf16,
    sigmoid_mul_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `sigmoid(lhs) * rhs` elementwise.",
    sigmoid_mul_f16,
    tune_sigmoid_mul_f16,
    f16,
    "f16",
    sigmoid_mul_f16,
    sigmoid_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `sigmoid(lhs) * rhs` elementwise.",
    sigmoid_mul_f32,
    tune_sigmoid_mul_f32,
    f32,
    "f32",
    sigmoid_mul_f32,
    sigmoid_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `sigmoid(lhs) * rhs` elementwise.",
    sigmoid_mul_f64,
    tune_sigmoid_mul_f64,
    f64,
    "f64",
    sigmoid_mul_f64,
    sigmoid_mul_f64_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_binary_fused_2d_fn!(
    sigmoid_mul_2d_bf16,
    tune_sigmoid_mul_2d_bf16,
    bf16,
    "bf16",
    sigmoid_mul_2d_bf16,
    sigmoid_mul_2d_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    sigmoid_mul_2d_f16,
    tune_sigmoid_mul_2d_f16,
    f16,
    "f16",
    sigmoid_mul_2d_f16,
    sigmoid_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    sigmoid_mul_2d_f32,
    tune_sigmoid_mul_2d_f32,
    f32,
    "f32",
    sigmoid_mul_2d_f32,
    sigmoid_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    sigmoid_mul_2d_f64,
    tune_sigmoid_mul_2d_f64,
    f64,
    "f64",
    sigmoid_mul_2d_f64,
    sigmoid_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `relu(lhs) * rhs` elementwise.",
    relu_mul_f16,
    tune_relu_mul_f16,
    f16,
    "f16",
    relu_mul_f16,
    relu_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `relu(lhs) * rhs` elementwise.",
    relu_mul_f32,
    tune_relu_mul_f32,
    f32,
    "f32",
    relu_mul_f32,
    relu_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `relu(lhs) * rhs` elementwise.",
    relu_mul_f64,
    tune_relu_mul_f64,
    f64,
    "f64",
    relu_mul_f64,
    relu_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    relu_mul_2d_f16,
    tune_relu_mul_2d_f16,
    f16,
    "f16",
    relu_mul_2d_f16,
    relu_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    relu_mul_2d_f32,
    tune_relu_mul_2d_f32,
    f32,
    "f32",
    relu_mul_2d_f32,
    relu_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    relu_mul_2d_f64,
    tune_relu_mul_2d_f64,
    f64,
    "f64",
    relu_mul_2d_f64,
    relu_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `relu6(lhs) * rhs` elementwise.",
    relu6_mul_f16,
    tune_relu6_mul_f16,
    f16,
    "f16",
    relu6_mul_f16,
    relu6_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `relu6(lhs) * rhs` elementwise.",
    relu6_mul_f32,
    tune_relu6_mul_f32,
    f32,
    "f32",
    relu6_mul_f32,
    relu6_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `relu6(lhs) * rhs` elementwise.",
    relu6_mul_f64,
    tune_relu6_mul_f64,
    f64,
    "f64",
    relu6_mul_f64,
    relu6_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    relu6_mul_2d_f16,
    tune_relu6_mul_2d_f16,
    f16,
    "f16",
    relu6_mul_2d_f16,
    relu6_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    relu6_mul_2d_f32,
    tune_relu6_mul_2d_f32,
    f32,
    "f32",
    relu6_mul_2d_f32,
    relu6_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    relu6_mul_2d_f64,
    tune_relu6_mul_2d_f64,
    f64,
    "f64",
    relu6_mul_2d_f64,
    relu6_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `hard_sigmoid(lhs) * rhs` elementwise.",
    hard_sigmoid_mul_f16,
    tune_hard_sigmoid_mul_f16,
    f16,
    "f16",
    hard_sigmoid_mul_f16,
    hard_sigmoid_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `hard_sigmoid(lhs) * rhs` elementwise.",
    hard_sigmoid_mul_f32,
    tune_hard_sigmoid_mul_f32,
    f32,
    "f32",
    hard_sigmoid_mul_f32,
    hard_sigmoid_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `hard_sigmoid(lhs) * rhs` elementwise.",
    hard_sigmoid_mul_f64,
    tune_hard_sigmoid_mul_f64,
    f64,
    "f64",
    hard_sigmoid_mul_f64,
    hard_sigmoid_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    hard_sigmoid_mul_2d_f16,
    tune_hard_sigmoid_mul_2d_f16,
    f16,
    "f16",
    hard_sigmoid_mul_2d_f16,
    hard_sigmoid_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    hard_sigmoid_mul_2d_f32,
    tune_hard_sigmoid_mul_2d_f32,
    f32,
    "f32",
    hard_sigmoid_mul_2d_f32,
    hard_sigmoid_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    hard_sigmoid_mul_2d_f64,
    tune_hard_sigmoid_mul_2d_f64,
    f64,
    "f64",
    hard_sigmoid_mul_2d_f64,
    hard_sigmoid_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `hard_swish(lhs) * rhs` elementwise.",
    hard_swish_mul_f16,
    tune_hard_swish_mul_f16,
    f16,
    "f16",
    hard_swish_mul_f16,
    hard_swish_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `hard_swish(lhs) * rhs` elementwise.",
    hard_swish_mul_f32,
    tune_hard_swish_mul_f32,
    f32,
    "f32",
    hard_swish_mul_f32,
    hard_swish_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `hard_swish(lhs) * rhs` elementwise.",
    hard_swish_mul_f64,
    tune_hard_swish_mul_f64,
    f64,
    "f64",
    hard_swish_mul_f64,
    hard_swish_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    hard_swish_mul_2d_f16,
    tune_hard_swish_mul_2d_f16,
    f16,
    "f16",
    hard_swish_mul_2d_f16,
    hard_swish_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    hard_swish_mul_2d_f32,
    tune_hard_swish_mul_2d_f32,
    f32,
    "f32",
    hard_swish_mul_2d_f32,
    hard_swish_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    hard_swish_mul_2d_f64,
    tune_hard_swish_mul_2d_f64,
    f64,
    "f64",
    hard_swish_mul_2d_f64,
    hard_swish_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `softsign(lhs) * rhs` elementwise.",
    softsign_mul_f16,
    tune_softsign_mul_f16,
    f16,
    "f16",
    softsign_mul_f16,
    softsign_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `softsign(lhs) * rhs` elementwise.",
    softsign_mul_f32,
    tune_softsign_mul_f32,
    f32,
    "f32",
    softsign_mul_f32,
    softsign_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `softsign(lhs) * rhs` elementwise.",
    softsign_mul_f64,
    tune_softsign_mul_f64,
    f64,
    "f64",
    softsign_mul_f64,
    softsign_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    softsign_mul_2d_f16,
    tune_softsign_mul_2d_f16,
    f16,
    "f16",
    softsign_mul_2d_f16,
    softsign_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    softsign_mul_2d_f32,
    tune_softsign_mul_2d_f32,
    f32,
    "f32",
    softsign_mul_2d_f32,
    softsign_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    softsign_mul_2d_f64,
    tune_softsign_mul_2d_f64,
    f64,
    "f64",
    softsign_mul_2d_f64,
    softsign_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `mish(lhs) * rhs` elementwise.",
    mish_mul_f16,
    tune_mish_mul_f16,
    f16,
    "f16",
    mish_mul_f16,
    mish_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `mish(lhs) * rhs` elementwise.",
    mish_mul_f32,
    tune_mish_mul_f32,
    f32,
    "f32",
    mish_mul_f32,
    mish_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `mish(lhs) * rhs` elementwise.",
    mish_mul_f64,
    tune_mish_mul_f64,
    f64,
    "f64",
    mish_mul_f64,
    mish_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    mish_mul_2d_f16,
    tune_mish_mul_2d_f16,
    f16,
    "f16",
    mish_mul_2d_f16,
    mish_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    mish_mul_2d_f32,
    tune_mish_mul_2d_f32,
    f32,
    "f32",
    mish_mul_2d_f32,
    mish_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    mish_mul_2d_f64,
    tune_mish_mul_2d_f64,
    f64,
    "f64",
    mish_mul_2d_f64,
    mish_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `selu(lhs) * rhs` elementwise.",
    selu_mul_f16,
    tune_selu_mul_f16,
    f16,
    "f16",
    selu_mul_f16,
    selu_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `selu(lhs) * rhs` elementwise.",
    selu_mul_f32,
    tune_selu_mul_f32,
    f32,
    "f32",
    selu_mul_f32,
    selu_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `selu(lhs) * rhs` elementwise.",
    selu_mul_f64,
    tune_selu_mul_f64,
    f64,
    "f64",
    selu_mul_f64,
    selu_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    selu_mul_2d_f16,
    tune_selu_mul_2d_f16,
    f16,
    "f16",
    selu_mul_2d_f16,
    selu_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    selu_mul_2d_f32,
    tune_selu_mul_2d_f32,
    f32,
    "f32",
    selu_mul_2d_f32,
    selu_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    selu_mul_2d_f64,
    tune_selu_mul_2d_f64,
    f64,
    "f64",
    selu_mul_2d_f64,
    selu_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `tanhshrink(lhs) * rhs` elementwise.",
    tanhshrink_mul_f16,
    tune_tanhshrink_mul_f16,
    f16,
    "f16",
    tanhshrink_mul_f16,
    tanhshrink_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `tanhshrink(lhs) * rhs` elementwise.",
    tanhshrink_mul_f32,
    tune_tanhshrink_mul_f32,
    f32,
    "f32",
    tanhshrink_mul_f32,
    tanhshrink_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `tanhshrink(lhs) * rhs` elementwise.",
    tanhshrink_mul_f64,
    tune_tanhshrink_mul_f64,
    f64,
    "f64",
    tanhshrink_mul_f64,
    tanhshrink_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    tanhshrink_mul_2d_f16,
    tune_tanhshrink_mul_2d_f16,
    f16,
    "f16",
    tanhshrink_mul_2d_f16,
    tanhshrink_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    tanhshrink_mul_2d_f32,
    tune_tanhshrink_mul_2d_f32,
    f32,
    "f32",
    tanhshrink_mul_2d_f32,
    tanhshrink_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    tanhshrink_mul_2d_f64,
    tune_tanhshrink_mul_2d_f64,
    f64,
    "f64",
    tanhshrink_mul_2d_f64,
    tanhshrink_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `log_sigmoid(lhs) * rhs` elementwise.",
    log_sigmoid_mul_f16,
    tune_log_sigmoid_mul_f16,
    f16,
    "f16",
    log_sigmoid_mul_f16,
    log_sigmoid_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `log_sigmoid(lhs) * rhs` elementwise.",
    log_sigmoid_mul_f32,
    tune_log_sigmoid_mul_f32,
    f32,
    "f32",
    log_sigmoid_mul_f32,
    log_sigmoid_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `log_sigmoid(lhs) * rhs` elementwise.",
    log_sigmoid_mul_f64,
    tune_log_sigmoid_mul_f64,
    f64,
    "f64",
    log_sigmoid_mul_f64,
    log_sigmoid_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    log_sigmoid_mul_2d_f16,
    tune_log_sigmoid_mul_2d_f16,
    f16,
    "f16",
    log_sigmoid_mul_2d_f16,
    log_sigmoid_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    log_sigmoid_mul_2d_f32,
    tune_log_sigmoid_mul_2d_f32,
    f32,
    "f32",
    log_sigmoid_mul_2d_f32,
    log_sigmoid_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    log_sigmoid_mul_2d_f64,
    tune_log_sigmoid_mul_2d_f64,
    f64,
    "f64",
    log_sigmoid_mul_2d_f64,
    log_sigmoid_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `reciprocal(lhs) * rhs` elementwise.",
    reciprocal_mul_f16,
    tune_reciprocal_mul_f16,
    f16,
    "f16",
    reciprocal_mul_f16,
    reciprocal_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `reciprocal(lhs) * rhs` elementwise.",
    reciprocal_mul_f32,
    tune_reciprocal_mul_f32,
    f32,
    "f32",
    reciprocal_mul_f32,
    reciprocal_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `reciprocal(lhs) * rhs` elementwise.",
    reciprocal_mul_f64,
    tune_reciprocal_mul_f64,
    f64,
    "f64",
    reciprocal_mul_f64,
    reciprocal_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    reciprocal_mul_2d_f16,
    tune_reciprocal_mul_2d_f16,
    f16,
    "f16",
    reciprocal_mul_2d_f16,
    reciprocal_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    reciprocal_mul_2d_f32,
    tune_reciprocal_mul_2d_f32,
    f32,
    "f32",
    reciprocal_mul_2d_f32,
    reciprocal_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    reciprocal_mul_2d_f64,
    tune_reciprocal_mul_2d_f64,
    f64,
    "f64",
    reciprocal_mul_2d_f64,
    reciprocal_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `square(lhs) * rhs` elementwise.",
    square_mul_f16,
    tune_square_mul_f16,
    f16,
    "f16",
    square_mul_f16,
    square_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `square(lhs) * rhs` elementwise.",
    square_mul_f32,
    tune_square_mul_f32,
    f32,
    "f32",
    square_mul_f32,
    square_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `square(lhs) * rhs` elementwise.",
    square_mul_f64,
    tune_square_mul_f64,
    f64,
    "f64",
    square_mul_f64,
    square_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    square_mul_2d_f16,
    tune_square_mul_2d_f16,
    f16,
    "f16",
    square_mul_2d_f16,
    square_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    square_mul_2d_f32,
    tune_square_mul_2d_f32,
    f32,
    "f32",
    square_mul_2d_f32,
    square_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    square_mul_2d_f64,
    tune_square_mul_2d_f64,
    f64,
    "f64",
    square_mul_2d_f64,
    square_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `ceil(lhs) * rhs` elementwise.",
    ceil_mul_f16,
    tune_ceil_mul_f16,
    f16,
    "f16",
    ceil_mul_f16,
    ceil_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `ceil(lhs) * rhs` elementwise.",
    ceil_mul_f32,
    tune_ceil_mul_f32,
    f32,
    "f32",
    ceil_mul_f32,
    ceil_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `ceil(lhs) * rhs` elementwise.",
    ceil_mul_f64,
    tune_ceil_mul_f64,
    f64,
    "f64",
    ceil_mul_f64,
    ceil_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    ceil_mul_2d_f16,
    tune_ceil_mul_2d_f16,
    f16,
    "f16",
    ceil_mul_2d_f16,
    ceil_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    ceil_mul_2d_f32,
    tune_ceil_mul_2d_f32,
    f32,
    "f32",
    ceil_mul_2d_f32,
    ceil_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    ceil_mul_2d_f64,
    tune_ceil_mul_2d_f64,
    f64,
    "f64",
    ceil_mul_2d_f64,
    ceil_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `floor(lhs) * rhs` elementwise.",
    floor_mul_f16,
    tune_floor_mul_f16,
    f16,
    "f16",
    floor_mul_f16,
    floor_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `floor(lhs) * rhs` elementwise.",
    floor_mul_f32,
    tune_floor_mul_f32,
    f32,
    "f32",
    floor_mul_f32,
    floor_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `floor(lhs) * rhs` elementwise.",
    floor_mul_f64,
    tune_floor_mul_f64,
    f64,
    "f64",
    floor_mul_f64,
    floor_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    floor_mul_2d_f16,
    tune_floor_mul_2d_f16,
    f16,
    "f16",
    floor_mul_2d_f16,
    floor_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    floor_mul_2d_f32,
    tune_floor_mul_2d_f32,
    f32,
    "f32",
    floor_mul_2d_f32,
    floor_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    floor_mul_2d_f64,
    tune_floor_mul_2d_f64,
    f64,
    "f64",
    floor_mul_2d_f64,
    floor_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `trunc(lhs) * rhs` elementwise.",
    trunc_mul_f16,
    tune_trunc_mul_f16,
    f16,
    "f16",
    trunc_mul_f16,
    trunc_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `trunc(lhs) * rhs` elementwise.",
    trunc_mul_f32,
    tune_trunc_mul_f32,
    f32,
    "f32",
    trunc_mul_f32,
    trunc_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `trunc(lhs) * rhs` elementwise.",
    trunc_mul_f64,
    tune_trunc_mul_f64,
    f64,
    "f64",
    trunc_mul_f64,
    trunc_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    trunc_mul_2d_f16,
    tune_trunc_mul_2d_f16,
    f16,
    "f16",
    trunc_mul_2d_f16,
    trunc_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    trunc_mul_2d_f32,
    tune_trunc_mul_2d_f32,
    f32,
    "f32",
    trunc_mul_2d_f32,
    trunc_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    trunc_mul_2d_f64,
    tune_trunc_mul_2d_f64,
    f64,
    "f64",
    trunc_mul_2d_f64,
    trunc_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `frac(lhs) * rhs` elementwise.",
    frac_mul_f16,
    tune_frac_mul_f16,
    f16,
    "f16",
    frac_mul_f16,
    frac_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `frac(lhs) * rhs` elementwise.",
    frac_mul_f32,
    tune_frac_mul_f32,
    f32,
    "f32",
    frac_mul_f32,
    frac_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `frac(lhs) * rhs` elementwise.",
    frac_mul_f64,
    tune_frac_mul_f64,
    f64,
    "f64",
    frac_mul_f64,
    frac_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    frac_mul_2d_f16,
    tune_frac_mul_2d_f16,
    f16,
    "f16",
    frac_mul_2d_f16,
    frac_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    frac_mul_2d_f32,
    tune_frac_mul_2d_f32,
    f32,
    "f32",
    frac_mul_2d_f32,
    frac_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    frac_mul_2d_f64,
    tune_frac_mul_2d_f64,
    f64,
    "f64",
    frac_mul_2d_f64,
    frac_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `deg2rad(lhs) * rhs` elementwise.",
    deg2rad_mul_f16,
    tune_deg2rad_mul_f16,
    f16,
    "f16",
    deg2rad_mul_f16,
    deg2rad_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `deg2rad(lhs) * rhs` elementwise.",
    deg2rad_mul_f32,
    tune_deg2rad_mul_f32,
    f32,
    "f32",
    deg2rad_mul_f32,
    deg2rad_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `deg2rad(lhs) * rhs` elementwise.",
    deg2rad_mul_f64,
    tune_deg2rad_mul_f64,
    f64,
    "f64",
    deg2rad_mul_f64,
    deg2rad_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    deg2rad_mul_2d_f16,
    tune_deg2rad_mul_2d_f16,
    f16,
    "f16",
    deg2rad_mul_2d_f16,
    deg2rad_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    deg2rad_mul_2d_f32,
    tune_deg2rad_mul_2d_f32,
    f32,
    "f32",
    deg2rad_mul_2d_f32,
    deg2rad_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    deg2rad_mul_2d_f64,
    tune_deg2rad_mul_2d_f64,
    f64,
    "f64",
    deg2rad_mul_2d_f64,
    deg2rad_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `rad2deg(lhs) * rhs` elementwise.",
    rad2deg_mul_f16,
    tune_rad2deg_mul_f16,
    f16,
    "f16",
    rad2deg_mul_f16,
    rad2deg_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `rad2deg(lhs) * rhs` elementwise.",
    rad2deg_mul_f32,
    tune_rad2deg_mul_f32,
    f32,
    "f32",
    rad2deg_mul_f32,
    rad2deg_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `rad2deg(lhs) * rhs` elementwise.",
    rad2deg_mul_f64,
    tune_rad2deg_mul_f64,
    f64,
    "f64",
    rad2deg_mul_f64,
    rad2deg_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    rad2deg_mul_2d_f16,
    tune_rad2deg_mul_2d_f16,
    f16,
    "f16",
    rad2deg_mul_2d_f16,
    rad2deg_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    rad2deg_mul_2d_f32,
    tune_rad2deg_mul_2d_f32,
    f32,
    "f32",
    rad2deg_mul_2d_f32,
    rad2deg_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    rad2deg_mul_2d_f64,
    tune_rad2deg_mul_2d_f64,
    f64,
    "f64",
    rad2deg_mul_2d_f64,
    rad2deg_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `sin(lhs) * rhs` elementwise.",
    sin_mul_f16,
    tune_sin_mul_f16,
    f16,
    "f16",
    sin_mul_f16,
    sin_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `sin(lhs) * rhs` elementwise.",
    sin_mul_f32,
    tune_sin_mul_f32,
    f32,
    "f32",
    sin_mul_f32,
    sin_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `sin(lhs) * rhs` elementwise.",
    sin_mul_f64,
    tune_sin_mul_f64,
    f64,
    "f64",
    sin_mul_f64,
    sin_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    sin_mul_2d_f16,
    tune_sin_mul_2d_f16,
    f16,
    "f16",
    sin_mul_2d_f16,
    sin_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    sin_mul_2d_f32,
    tune_sin_mul_2d_f32,
    f32,
    "f32",
    sin_mul_2d_f32,
    sin_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    sin_mul_2d_f64,
    tune_sin_mul_2d_f64,
    f64,
    "f64",
    sin_mul_2d_f64,
    sin_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `cos(lhs) * rhs` elementwise.",
    cos_mul_f16,
    tune_cos_mul_f16,
    f16,
    "f16",
    cos_mul_f16,
    cos_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `cos(lhs) * rhs` elementwise.",
    cos_mul_f32,
    tune_cos_mul_f32,
    f32,
    "f32",
    cos_mul_f32,
    cos_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `cos(lhs) * rhs` elementwise.",
    cos_mul_f64,
    tune_cos_mul_f64,
    f64,
    "f64",
    cos_mul_f64,
    cos_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    cos_mul_2d_f16,
    tune_cos_mul_2d_f16,
    f16,
    "f16",
    cos_mul_2d_f16,
    cos_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    cos_mul_2d_f32,
    tune_cos_mul_2d_f32,
    f32,
    "f32",
    cos_mul_2d_f32,
    cos_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    cos_mul_2d_f64,
    tune_cos_mul_2d_f64,
    f64,
    "f64",
    cos_mul_2d_f64,
    cos_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `tan(lhs) * rhs` elementwise.",
    tan_mul_f16,
    tune_tan_mul_f16,
    f16,
    "f16",
    tan_mul_f16,
    tan_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `tan(lhs) * rhs` elementwise.",
    tan_mul_f32,
    tune_tan_mul_f32,
    f32,
    "f32",
    tan_mul_f32,
    tan_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `tan(lhs) * rhs` elementwise.",
    tan_mul_f64,
    tune_tan_mul_f64,
    f64,
    "f64",
    tan_mul_f64,
    tan_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    tan_mul_2d_f16,
    tune_tan_mul_2d_f16,
    f16,
    "f16",
    tan_mul_2d_f16,
    tan_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    tan_mul_2d_f32,
    tune_tan_mul_2d_f32,
    f32,
    "f32",
    tan_mul_2d_f32,
    tan_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    tan_mul_2d_f64,
    tune_tan_mul_2d_f64,
    f64,
    "f64",
    tan_mul_2d_f64,
    tan_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `asin(lhs) * rhs` elementwise.",
    asin_mul_f16,
    tune_asin_mul_f16,
    f16,
    "f16",
    asin_mul_f16,
    asin_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `asin(lhs) * rhs` elementwise.",
    asin_mul_f32,
    tune_asin_mul_f32,
    f32,
    "f32",
    asin_mul_f32,
    asin_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `asin(lhs) * rhs` elementwise.",
    asin_mul_f64,
    tune_asin_mul_f64,
    f64,
    "f64",
    asin_mul_f64,
    asin_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    asin_mul_2d_f16,
    tune_asin_mul_2d_f16,
    f16,
    "f16",
    asin_mul_2d_f16,
    asin_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    asin_mul_2d_f32,
    tune_asin_mul_2d_f32,
    f32,
    "f32",
    asin_mul_2d_f32,
    asin_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    asin_mul_2d_f64,
    tune_asin_mul_2d_f64,
    f64,
    "f64",
    asin_mul_2d_f64,
    asin_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `acos(lhs) * rhs` elementwise.",
    acos_mul_f16,
    tune_acos_mul_f16,
    f16,
    "f16",
    acos_mul_f16,
    acos_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `acos(lhs) * rhs` elementwise.",
    acos_mul_f32,
    tune_acos_mul_f32,
    f32,
    "f32",
    acos_mul_f32,
    acos_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `acos(lhs) * rhs` elementwise.",
    acos_mul_f64,
    tune_acos_mul_f64,
    f64,
    "f64",
    acos_mul_f64,
    acos_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    acos_mul_2d_f16,
    tune_acos_mul_2d_f16,
    f16,
    "f16",
    acos_mul_2d_f16,
    acos_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    acos_mul_2d_f32,
    tune_acos_mul_2d_f32,
    f32,
    "f32",
    acos_mul_2d_f32,
    acos_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    acos_mul_2d_f64,
    tune_acos_mul_2d_f64,
    f64,
    "f64",
    acos_mul_2d_f64,
    acos_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `atan(lhs) * rhs` elementwise.",
    atan_mul_f16,
    tune_atan_mul_f16,
    f16,
    "f16",
    atan_mul_f16,
    atan_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `atan(lhs) * rhs` elementwise.",
    atan_mul_f32,
    tune_atan_mul_f32,
    f32,
    "f32",
    atan_mul_f32,
    atan_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `atan(lhs) * rhs` elementwise.",
    atan_mul_f64,
    tune_atan_mul_f64,
    f64,
    "f64",
    atan_mul_f64,
    atan_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    atan_mul_2d_f16,
    tune_atan_mul_2d_f16,
    f16,
    "f16",
    atan_mul_2d_f16,
    atan_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    atan_mul_2d_f32,
    tune_atan_mul_2d_f32,
    f32,
    "f32",
    atan_mul_2d_f32,
    atan_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    atan_mul_2d_f64,
    tune_atan_mul_2d_f64,
    f64,
    "f64",
    atan_mul_2d_f64,
    atan_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `sinh(lhs) * rhs` elementwise.",
    sinh_mul_f16,
    tune_sinh_mul_f16,
    f16,
    "f16",
    sinh_mul_f16,
    sinh_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `sinh(lhs) * rhs` elementwise.",
    sinh_mul_f32,
    tune_sinh_mul_f32,
    f32,
    "f32",
    sinh_mul_f32,
    sinh_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `sinh(lhs) * rhs` elementwise.",
    sinh_mul_f64,
    tune_sinh_mul_f64,
    f64,
    "f64",
    sinh_mul_f64,
    sinh_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    sinh_mul_2d_f16,
    tune_sinh_mul_2d_f16,
    f16,
    "f16",
    sinh_mul_2d_f16,
    sinh_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    sinh_mul_2d_f32,
    tune_sinh_mul_2d_f32,
    f32,
    "f32",
    sinh_mul_2d_f32,
    sinh_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    sinh_mul_2d_f64,
    tune_sinh_mul_2d_f64,
    f64,
    "f64",
    sinh_mul_2d_f64,
    sinh_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `cosh(lhs) * rhs` elementwise.",
    cosh_mul_f16,
    tune_cosh_mul_f16,
    f16,
    "f16",
    cosh_mul_f16,
    cosh_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `cosh(lhs) * rhs` elementwise.",
    cosh_mul_f32,
    tune_cosh_mul_f32,
    f32,
    "f32",
    cosh_mul_f32,
    cosh_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `cosh(lhs) * rhs` elementwise.",
    cosh_mul_f64,
    tune_cosh_mul_f64,
    f64,
    "f64",
    cosh_mul_f64,
    cosh_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    cosh_mul_2d_f16,
    tune_cosh_mul_2d_f16,
    f16,
    "f16",
    cosh_mul_2d_f16,
    cosh_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    cosh_mul_2d_f32,
    tune_cosh_mul_2d_f32,
    f32,
    "f32",
    cosh_mul_2d_f32,
    cosh_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    cosh_mul_2d_f64,
    tune_cosh_mul_2d_f64,
    f64,
    "f64",
    cosh_mul_2d_f64,
    cosh_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `asinh(lhs) * rhs` elementwise.",
    asinh_mul_f16,
    tune_asinh_mul_f16,
    f16,
    "f16",
    asinh_mul_f16,
    asinh_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `asinh(lhs) * rhs` elementwise.",
    asinh_mul_f32,
    tune_asinh_mul_f32,
    f32,
    "f32",
    asinh_mul_f32,
    asinh_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `asinh(lhs) * rhs` elementwise.",
    asinh_mul_f64,
    tune_asinh_mul_f64,
    f64,
    "f64",
    asinh_mul_f64,
    asinh_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    asinh_mul_2d_f16,
    tune_asinh_mul_2d_f16,
    f16,
    "f16",
    asinh_mul_2d_f16,
    asinh_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    asinh_mul_2d_f32,
    tune_asinh_mul_2d_f32,
    f32,
    "f32",
    asinh_mul_2d_f32,
    asinh_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    asinh_mul_2d_f64,
    tune_asinh_mul_2d_f64,
    f64,
    "f64",
    asinh_mul_2d_f64,
    asinh_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `acosh(lhs) * rhs` elementwise.",
    acosh_mul_f16,
    tune_acosh_mul_f16,
    f16,
    "f16",
    acosh_mul_f16,
    acosh_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `acosh(lhs) * rhs` elementwise.",
    acosh_mul_f32,
    tune_acosh_mul_f32,
    f32,
    "f32",
    acosh_mul_f32,
    acosh_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `acosh(lhs) * rhs` elementwise.",
    acosh_mul_f64,
    tune_acosh_mul_f64,
    f64,
    "f64",
    acosh_mul_f64,
    acosh_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    acosh_mul_2d_f16,
    tune_acosh_mul_2d_f16,
    f16,
    "f16",
    acosh_mul_2d_f16,
    acosh_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    acosh_mul_2d_f32,
    tune_acosh_mul_2d_f32,
    f32,
    "f32",
    acosh_mul_2d_f32,
    acosh_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    acosh_mul_2d_f64,
    tune_acosh_mul_2d_f64,
    f64,
    "f64",
    acosh_mul_2d_f64,
    acosh_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `atanh(lhs) * rhs` elementwise.",
    atanh_mul_f16,
    tune_atanh_mul_f16,
    f16,
    "f16",
    atanh_mul_f16,
    atanh_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `atanh(lhs) * rhs` elementwise.",
    atanh_mul_f32,
    tune_atanh_mul_f32,
    f32,
    "f32",
    atanh_mul_f32,
    atanh_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `atanh(lhs) * rhs` elementwise.",
    atanh_mul_f64,
    tune_atanh_mul_f64,
    f64,
    "f64",
    atanh_mul_f64,
    atanh_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    atanh_mul_2d_f16,
    tune_atanh_mul_2d_f16,
    f16,
    "f16",
    atanh_mul_2d_f16,
    atanh_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    atanh_mul_2d_f32,
    tune_atanh_mul_2d_f32,
    f32,
    "f32",
    atanh_mul_2d_f32,
    atanh_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    atanh_mul_2d_f64,
    tune_atanh_mul_2d_f64,
    f64,
    "f64",
    atanh_mul_2d_f64,
    atanh_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `tanh(lhs) * rhs` elementwise.",
    tanh_mul_f16,
    tune_tanh_mul_f16,
    f16,
    "f16",
    tanh_mul_f16,
    tanh_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `tanh(lhs) * rhs` elementwise.",
    tanh_mul_f32,
    tune_tanh_mul_f32,
    f32,
    "f32",
    tanh_mul_f32,
    tanh_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `tanh(lhs) * rhs` elementwise.",
    tanh_mul_f64,
    tune_tanh_mul_f64,
    f64,
    "f64",
    tanh_mul_f64,
    tanh_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    tanh_mul_2d_f16,
    tune_tanh_mul_2d_f16,
    f16,
    "f16",
    tanh_mul_2d_f16,
    tanh_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    tanh_mul_2d_f32,
    tune_tanh_mul_2d_f32,
    f32,
    "f32",
    tanh_mul_2d_f32,
    tanh_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    tanh_mul_2d_f64,
    tune_tanh_mul_2d_f64,
    f64,
    "f64",
    tanh_mul_2d_f64,
    tanh_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `exp(lhs) * rhs` elementwise.",
    exp_mul_f16,
    tune_exp_mul_f16,
    f16,
    "f16",
    exp_mul_f16,
    exp_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `exp(lhs) * rhs` elementwise.",
    exp_mul_f32,
    tune_exp_mul_f32,
    f32,
    "f32",
    exp_mul_f32,
    exp_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `exp(lhs) * rhs` elementwise.",
    exp_mul_f64,
    tune_exp_mul_f64,
    f64,
    "f64",
    exp_mul_f64,
    exp_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    exp_mul_2d_f16,
    tune_exp_mul_2d_f16,
    f16,
    "f16",
    exp_mul_2d_f16,
    exp_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    exp_mul_2d_f32,
    tune_exp_mul_2d_f32,
    f32,
    "f32",
    exp_mul_2d_f32,
    exp_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    exp_mul_2d_f64,
    tune_exp_mul_2d_f64,
    f64,
    "f64",
    exp_mul_2d_f64,
    exp_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `exp2(lhs) * rhs` elementwise.",
    exp2_mul_f16,
    tune_exp2_mul_f16,
    f16,
    "f16",
    exp2_mul_f16,
    exp2_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `exp2(lhs) * rhs` elementwise.",
    exp2_mul_f32,
    tune_exp2_mul_f32,
    f32,
    "f32",
    exp2_mul_f32,
    exp2_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `exp2(lhs) * rhs` elementwise.",
    exp2_mul_f64,
    tune_exp2_mul_f64,
    f64,
    "f64",
    exp2_mul_f64,
    exp2_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    exp2_mul_2d_f16,
    tune_exp2_mul_2d_f16,
    f16,
    "f16",
    exp2_mul_2d_f16,
    exp2_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    exp2_mul_2d_f32,
    tune_exp2_mul_2d_f32,
    f32,
    "f32",
    exp2_mul_2d_f32,
    exp2_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    exp2_mul_2d_f64,
    tune_exp2_mul_2d_f64,
    f64,
    "f64",
    exp2_mul_2d_f64,
    exp2_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `expm1(lhs) * rhs` elementwise.",
    expm1_mul_f16,
    tune_expm1_mul_f16,
    f16,
    "f16",
    expm1_mul_f16,
    expm1_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `expm1(lhs) * rhs` elementwise.",
    expm1_mul_f32,
    tune_expm1_mul_f32,
    f32,
    "f32",
    expm1_mul_f32,
    expm1_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `expm1(lhs) * rhs` elementwise.",
    expm1_mul_f64,
    tune_expm1_mul_f64,
    f64,
    "f64",
    expm1_mul_f64,
    expm1_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    expm1_mul_2d_f16,
    tune_expm1_mul_2d_f16,
    f16,
    "f16",
    expm1_mul_2d_f16,
    expm1_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    expm1_mul_2d_f32,
    tune_expm1_mul_2d_f32,
    f32,
    "f32",
    expm1_mul_2d_f32,
    expm1_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    expm1_mul_2d_f64,
    tune_expm1_mul_2d_f64,
    f64,
    "f64",
    expm1_mul_2d_f64,
    expm1_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `log(lhs) * rhs` elementwise.",
    log_mul_f16,
    tune_log_mul_f16,
    f16,
    "f16",
    log_mul_f16,
    log_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `log(lhs) * rhs` elementwise.",
    log_mul_f32,
    tune_log_mul_f32,
    f32,
    "f32",
    log_mul_f32,
    log_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `log(lhs) * rhs` elementwise.",
    log_mul_f64,
    tune_log_mul_f64,
    f64,
    "f64",
    log_mul_f64,
    log_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    log_mul_2d_f16,
    tune_log_mul_2d_f16,
    f16,
    "f16",
    log_mul_2d_f16,
    log_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    log_mul_2d_f32,
    tune_log_mul_2d_f32,
    f32,
    "f32",
    log_mul_2d_f32,
    log_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    log_mul_2d_f64,
    tune_log_mul_2d_f64,
    f64,
    "f64",
    log_mul_2d_f64,
    log_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `log2(lhs) * rhs` elementwise.",
    log2_mul_f16,
    tune_log2_mul_f16,
    f16,
    "f16",
    log2_mul_f16,
    log2_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `log2(lhs) * rhs` elementwise.",
    log2_mul_f32,
    tune_log2_mul_f32,
    f32,
    "f32",
    log2_mul_f32,
    log2_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `log2(lhs) * rhs` elementwise.",
    log2_mul_f64,
    tune_log2_mul_f64,
    f64,
    "f64",
    log2_mul_f64,
    log2_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    log2_mul_2d_f16,
    tune_log2_mul_2d_f16,
    f16,
    "f16",
    log2_mul_2d_f16,
    log2_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    log2_mul_2d_f32,
    tune_log2_mul_2d_f32,
    f32,
    "f32",
    log2_mul_2d_f32,
    log2_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    log2_mul_2d_f64,
    tune_log2_mul_2d_f64,
    f64,
    "f64",
    log2_mul_2d_f64,
    log2_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `log10(lhs) * rhs` elementwise.",
    log10_mul_f16,
    tune_log10_mul_f16,
    f16,
    "f16",
    log10_mul_f16,
    log10_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `log10(lhs) * rhs` elementwise.",
    log10_mul_f32,
    tune_log10_mul_f32,
    f32,
    "f32",
    log10_mul_f32,
    log10_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `log10(lhs) * rhs` elementwise.",
    log10_mul_f64,
    tune_log10_mul_f64,
    f64,
    "f64",
    log10_mul_f64,
    log10_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    log10_mul_2d_f16,
    tune_log10_mul_2d_f16,
    f16,
    "f16",
    log10_mul_2d_f16,
    log10_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    log10_mul_2d_f32,
    tune_log10_mul_2d_f32,
    f32,
    "f32",
    log10_mul_2d_f32,
    log10_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    log10_mul_2d_f64,
    tune_log10_mul_2d_f64,
    f64,
    "f64",
    log10_mul_2d_f64,
    log10_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `log1p(lhs) * rhs` elementwise.",
    log1p_mul_f16,
    tune_log1p_mul_f16,
    f16,
    "f16",
    log1p_mul_f16,
    log1p_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `log1p(lhs) * rhs` elementwise.",
    log1p_mul_f32,
    tune_log1p_mul_f32,
    f32,
    "f32",
    log1p_mul_f32,
    log1p_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `log1p(lhs) * rhs` elementwise.",
    log1p_mul_f64,
    tune_log1p_mul_f64,
    f64,
    "f64",
    log1p_mul_f64,
    log1p_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    log1p_mul_2d_f16,
    tune_log1p_mul_2d_f16,
    f16,
    "f16",
    log1p_mul_2d_f16,
    log1p_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    log1p_mul_2d_f32,
    tune_log1p_mul_2d_f32,
    f32,
    "f32",
    log1p_mul_2d_f32,
    log1p_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    log1p_mul_2d_f64,
    tune_log1p_mul_2d_f64,
    f64,
    "f64",
    log1p_mul_2d_f64,
    log1p_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `softplus(lhs) * rhs` elementwise.",
    softplus_mul_f16,
    tune_softplus_mul_f16,
    f16,
    "f16",
    softplus_mul_f16,
    softplus_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `softplus(lhs) * rhs` elementwise.",
    softplus_mul_f32,
    tune_softplus_mul_f32,
    f32,
    "f32",
    softplus_mul_f32,
    softplus_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `softplus(lhs) * rhs` elementwise.",
    softplus_mul_f64,
    tune_softplus_mul_f64,
    f64,
    "f64",
    softplus_mul_f64,
    softplus_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    softplus_mul_2d_f16,
    tune_softplus_mul_2d_f16,
    f16,
    "f16",
    softplus_mul_2d_f16,
    softplus_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    softplus_mul_2d_f32,
    tune_softplus_mul_2d_f32,
    f32,
    "f32",
    softplus_mul_2d_f32,
    softplus_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    softplus_mul_2d_f64,
    tune_softplus_mul_2d_f64,
    f64,
    "f64",
    softplus_mul_2d_f64,
    softplus_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `sqrt(lhs) * rhs` elementwise.",
    sqrt_mul_f16,
    tune_sqrt_mul_f16,
    f16,
    "f16",
    sqrt_mul_f16,
    sqrt_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `sqrt(lhs) * rhs` elementwise.",
    sqrt_mul_f32,
    tune_sqrt_mul_f32,
    f32,
    "f32",
    sqrt_mul_f32,
    sqrt_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `sqrt(lhs) * rhs` elementwise.",
    sqrt_mul_f64,
    tune_sqrt_mul_f64,
    f64,
    "f64",
    sqrt_mul_f64,
    sqrt_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    sqrt_mul_2d_f16,
    tune_sqrt_mul_2d_f16,
    f16,
    "f16",
    sqrt_mul_2d_f16,
    sqrt_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    sqrt_mul_2d_f32,
    tune_sqrt_mul_2d_f32,
    f32,
    "f32",
    sqrt_mul_2d_f32,
    sqrt_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    sqrt_mul_2d_f64,
    tune_sqrt_mul_2d_f64,
    f64,
    "f64",
    sqrt_mul_2d_f64,
    sqrt_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `rsqrt(lhs) * rhs` elementwise.",
    rsqrt_mul_f16,
    tune_rsqrt_mul_f16,
    f16,
    "f16",
    rsqrt_mul_f16,
    rsqrt_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `rsqrt(lhs) * rhs` elementwise.",
    rsqrt_mul_f32,
    tune_rsqrt_mul_f32,
    f32,
    "f32",
    rsqrt_mul_f32,
    rsqrt_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `rsqrt(lhs) * rhs` elementwise.",
    rsqrt_mul_f64,
    tune_rsqrt_mul_f64,
    f64,
    "f64",
    rsqrt_mul_f64,
    rsqrt_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    rsqrt_mul_2d_f16,
    tune_rsqrt_mul_2d_f16,
    f16,
    "f16",
    rsqrt_mul_2d_f16,
    rsqrt_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    rsqrt_mul_2d_f32,
    tune_rsqrt_mul_2d_f32,
    f32,
    "f32",
    rsqrt_mul_2d_f32,
    rsqrt_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    rsqrt_mul_2d_f64,
    tune_rsqrt_mul_2d_f64,
    f64,
    "f64",
    rsqrt_mul_2d_f64,
    rsqrt_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `cbrt(lhs) * rhs` elementwise.",
    cbrt_mul_f16,
    tune_cbrt_mul_f16,
    f16,
    "f16",
    cbrt_mul_f16,
    cbrt_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `cbrt(lhs) * rhs` elementwise.",
    cbrt_mul_f32,
    tune_cbrt_mul_f32,
    f32,
    "f32",
    cbrt_mul_f32,
    cbrt_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `cbrt(lhs) * rhs` elementwise.",
    cbrt_mul_f64,
    tune_cbrt_mul_f64,
    f64,
    "f64",
    cbrt_mul_f64,
    cbrt_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    cbrt_mul_2d_f16,
    tune_cbrt_mul_2d_f16,
    f16,
    "f16",
    cbrt_mul_2d_f16,
    cbrt_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    cbrt_mul_2d_f32,
    tune_cbrt_mul_2d_f32,
    f32,
    "f32",
    cbrt_mul_2d_f32,
    cbrt_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    cbrt_mul_2d_f64,
    tune_cbrt_mul_2d_f64,
    f64,
    "f64",
    cbrt_mul_2d_f64,
    cbrt_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `abs(lhs) * rhs` elementwise.",
    abs_mul_f16,
    tune_abs_mul_f16,
    f16,
    "f16",
    abs_mul_f16,
    abs_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `abs(lhs) * rhs` elementwise.",
    abs_mul_f32,
    tune_abs_mul_f32,
    f32,
    "f32",
    abs_mul_f32,
    abs_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `abs(lhs) * rhs` elementwise.",
    abs_mul_f64,
    tune_abs_mul_f64,
    f64,
    "f64",
    abs_mul_f64,
    abs_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    abs_mul_2d_f16,
    tune_abs_mul_2d_f16,
    f16,
    "f16",
    abs_mul_2d_f16,
    abs_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    abs_mul_2d_f32,
    tune_abs_mul_2d_f32,
    f32,
    "f32",
    abs_mul_2d_f32,
    abs_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    abs_mul_2d_f64,
    tune_abs_mul_2d_f64,
    f64,
    "f64",
    abs_mul_2d_f64,
    abs_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_fn!(
    "Computes `neg(lhs) * rhs` elementwise.",
    neg_mul_f16,
    tune_neg_mul_f16,
    f16,
    "f16",
    neg_mul_f16,
    neg_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_fn!(
    "Computes `neg(lhs) * rhs` elementwise.",
    neg_mul_f32,
    tune_neg_mul_f32,
    f32,
    "f32",
    neg_mul_f32,
    neg_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_fn!(
    "Computes `neg(lhs) * rhs` elementwise.",
    neg_mul_f64,
    tune_neg_mul_f64,
    f64,
    "f64",
    neg_mul_f64,
    neg_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_binary_fused_2d_fn!(
    neg_mul_2d_f16,
    tune_neg_mul_2d_f16,
    f16,
    "f16",
    neg_mul_2d_f16,
    neg_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_binary_fused_2d_fn!(
    neg_mul_2d_f32,
    tune_neg_mul_2d_f32,
    f32,
    "f32",
    neg_mul_2d_f32,
    neg_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_binary_fused_2d_fn!(
    neg_mul_2d_f64,
    tune_neg_mul_2d_f64,
    f64,
    "f64",
    neg_mul_2d_f64,
    neg_mul_2d_f64_tuned
);
