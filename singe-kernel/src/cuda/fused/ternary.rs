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

#[cfg(any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64"))]
use crate::cuda::fused::{fused_2d_bias, tune_fused_2d_bias};
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
            Fused1dConfig, Fused2dConfig, FusedTileTuning, fused_1d_ternary, fused_2d_ternary,
            tune_fused_1d_ternary, tune_fused_2d_ternary,
        },
    },
    error::Result,
};

#[cfg(feature = "dtype_bf16")]
autotuned_ternary_fused_fn!(
    add_silu_mul_bf16,
    tune_add_silu_mul_bf16,
    bf16,
    "bf16",
    add_silu_mul_bf16,
    add_silu_mul_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_silu_mul_f16,
    tune_add_silu_mul_f16,
    f16,
    "f16",
    add_silu_mul_f16,
    add_silu_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_silu_mul_f32,
    tune_add_silu_mul_f32,
    f32,
    "f32",
    add_silu_mul_f32,
    add_silu_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_silu_mul_f64,
    tune_add_silu_mul_f64,
    f64,
    "f64",
    add_silu_mul_f64,
    add_silu_mul_f64_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_ternary_fused_2d_fn!(
    add_silu_mul_2d_bf16,
    tune_add_silu_mul_2d_bf16,
    bf16,
    "bf16",
    add_silu_mul_2d_bf16,
    add_silu_mul_2d_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_silu_mul_2d_f16,
    tune_add_silu_mul_2d_f16,
    f16,
    "f16",
    add_silu_mul_2d_f16,
    add_silu_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_silu_mul_2d_f32,
    tune_add_silu_mul_2d_f32,
    f32,
    "f32",
    add_silu_mul_2d_f32,
    add_silu_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_silu_mul_2d_f64,
    tune_add_silu_mul_2d_f64,
    f64,
    "f64",
    add_silu_mul_2d_f64,
    add_silu_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `input + bias[column]` for every row-major matrix element.",
    bias_add_f16,
    tune_bias_add_f16,
    f16,
    "f16",
    bias_add_f16,
    bias_add_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `input + bias[column]` for every row-major matrix element.",
    bias_add_f32,
    tune_bias_add_f32,
    f32,
    "f32",
    bias_add_f32,
    bias_add_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `input + bias[column]` for every row-major matrix element.",
    bias_add_f64,
    tune_bias_add_f64,
    f64,
    "f64",
    bias_add_f64,
    bias_add_f64_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_ternary_fused_fn!(
    add_gelu_mul_bf16,
    tune_add_gelu_mul_bf16,
    bf16,
    "bf16",
    add_gelu_mul_bf16,
    add_gelu_mul_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_gelu_mul_f16,
    tune_add_gelu_mul_f16,
    f16,
    "f16",
    add_gelu_mul_f16,
    add_gelu_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_gelu_mul_f32,
    tune_add_gelu_mul_f32,
    f32,
    "f32",
    add_gelu_mul_f32,
    add_gelu_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_gelu_mul_f64,
    tune_add_gelu_mul_f64,
    f64,
    "f64",
    add_gelu_mul_f64,
    add_gelu_mul_f64_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_ternary_fused_2d_fn!(
    add_gelu_mul_2d_bf16,
    tune_add_gelu_mul_2d_bf16,
    bf16,
    "bf16",
    add_gelu_mul_2d_bf16,
    add_gelu_mul_2d_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_gelu_mul_2d_f16,
    tune_add_gelu_mul_2d_f16,
    f16,
    "f16",
    add_gelu_mul_2d_f16,
    add_gelu_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_gelu_mul_2d_f32,
    tune_add_gelu_mul_2d_f32,
    f32,
    "f32",
    add_gelu_mul_2d_f32,
    add_gelu_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_gelu_mul_2d_f64,
    tune_add_gelu_mul_2d_f64,
    f64,
    "f64",
    add_gelu_mul_2d_f64,
    add_gelu_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_ternary_fused_fn!(
    add_sigmoid_mul_bf16,
    tune_add_sigmoid_mul_bf16,
    bf16,
    "bf16",
    add_sigmoid_mul_bf16,
    add_sigmoid_mul_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_sigmoid_mul_f16,
    tune_add_sigmoid_mul_f16,
    f16,
    "f16",
    add_sigmoid_mul_f16,
    add_sigmoid_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_sigmoid_mul_f32,
    tune_add_sigmoid_mul_f32,
    f32,
    "f32",
    add_sigmoid_mul_f32,
    add_sigmoid_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_sigmoid_mul_f64,
    tune_add_sigmoid_mul_f64,
    f64,
    "f64",
    add_sigmoid_mul_f64,
    add_sigmoid_mul_f64_tuned
);

#[cfg(feature = "dtype_bf16")]
autotuned_ternary_fused_2d_fn!(
    add_sigmoid_mul_2d_bf16,
    tune_add_sigmoid_mul_2d_bf16,
    bf16,
    "bf16",
    add_sigmoid_mul_2d_bf16,
    add_sigmoid_mul_2d_bf16_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_sigmoid_mul_2d_f16,
    tune_add_sigmoid_mul_2d_f16,
    f16,
    "f16",
    add_sigmoid_mul_2d_f16,
    add_sigmoid_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_sigmoid_mul_2d_f32,
    tune_add_sigmoid_mul_2d_f32,
    f32,
    "f32",
    add_sigmoid_mul_2d_f32,
    add_sigmoid_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_sigmoid_mul_2d_f64,
    tune_add_sigmoid_mul_2d_f64,
    f64,
    "f64",
    add_sigmoid_mul_2d_f64,
    add_sigmoid_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_relu_mul_f16,
    tune_add_relu_mul_f16,
    f16,
    "f16",
    add_relu_mul_f16,
    add_relu_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_relu_mul_f32,
    tune_add_relu_mul_f32,
    f32,
    "f32",
    add_relu_mul_f32,
    add_relu_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_relu_mul_f64,
    tune_add_relu_mul_f64,
    f64,
    "f64",
    add_relu_mul_f64,
    add_relu_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_relu_mul_2d_f16,
    tune_add_relu_mul_2d_f16,
    f16,
    "f16",
    add_relu_mul_2d_f16,
    add_relu_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_relu_mul_2d_f32,
    tune_add_relu_mul_2d_f32,
    f32,
    "f32",
    add_relu_mul_2d_f32,
    add_relu_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_relu_mul_2d_f64,
    tune_add_relu_mul_2d_f64,
    f64,
    "f64",
    add_relu_mul_2d_f64,
    add_relu_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_relu6_mul_f16,
    tune_add_relu6_mul_f16,
    f16,
    "f16",
    add_relu6_mul_f16,
    add_relu6_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_relu6_mul_f32,
    tune_add_relu6_mul_f32,
    f32,
    "f32",
    add_relu6_mul_f32,
    add_relu6_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_relu6_mul_f64,
    tune_add_relu6_mul_f64,
    f64,
    "f64",
    add_relu6_mul_f64,
    add_relu6_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_relu6_mul_2d_f16,
    tune_add_relu6_mul_2d_f16,
    f16,
    "f16",
    add_relu6_mul_2d_f16,
    add_relu6_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_relu6_mul_2d_f32,
    tune_add_relu6_mul_2d_f32,
    f32,
    "f32",
    add_relu6_mul_2d_f32,
    add_relu6_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_relu6_mul_2d_f64,
    tune_add_relu6_mul_2d_f64,
    f64,
    "f64",
    add_relu6_mul_2d_f64,
    add_relu6_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_hard_sigmoid_mul_f16,
    tune_add_hard_sigmoid_mul_f16,
    f16,
    "f16",
    add_hard_sigmoid_mul_f16,
    add_hard_sigmoid_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_hard_sigmoid_mul_f32,
    tune_add_hard_sigmoid_mul_f32,
    f32,
    "f32",
    add_hard_sigmoid_mul_f32,
    add_hard_sigmoid_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_hard_sigmoid_mul_f64,
    tune_add_hard_sigmoid_mul_f64,
    f64,
    "f64",
    add_hard_sigmoid_mul_f64,
    add_hard_sigmoid_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_hard_sigmoid_mul_2d_f16,
    tune_add_hard_sigmoid_mul_2d_f16,
    f16,
    "f16",
    add_hard_sigmoid_mul_2d_f16,
    add_hard_sigmoid_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_hard_sigmoid_mul_2d_f32,
    tune_add_hard_sigmoid_mul_2d_f32,
    f32,
    "f32",
    add_hard_sigmoid_mul_2d_f32,
    add_hard_sigmoid_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_hard_sigmoid_mul_2d_f64,
    tune_add_hard_sigmoid_mul_2d_f64,
    f64,
    "f64",
    add_hard_sigmoid_mul_2d_f64,
    add_hard_sigmoid_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_hard_swish_mul_f16,
    tune_add_hard_swish_mul_f16,
    f16,
    "f16",
    add_hard_swish_mul_f16,
    add_hard_swish_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_hard_swish_mul_f32,
    tune_add_hard_swish_mul_f32,
    f32,
    "f32",
    add_hard_swish_mul_f32,
    add_hard_swish_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_hard_swish_mul_f64,
    tune_add_hard_swish_mul_f64,
    f64,
    "f64",
    add_hard_swish_mul_f64,
    add_hard_swish_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_hard_swish_mul_2d_f16,
    tune_add_hard_swish_mul_2d_f16,
    f16,
    "f16",
    add_hard_swish_mul_2d_f16,
    add_hard_swish_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_hard_swish_mul_2d_f32,
    tune_add_hard_swish_mul_2d_f32,
    f32,
    "f32",
    add_hard_swish_mul_2d_f32,
    add_hard_swish_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_hard_swish_mul_2d_f64,
    tune_add_hard_swish_mul_2d_f64,
    f64,
    "f64",
    add_hard_swish_mul_2d_f64,
    add_hard_swish_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_softsign_mul_f16,
    tune_add_softsign_mul_f16,
    f16,
    "f16",
    add_softsign_mul_f16,
    add_softsign_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_softsign_mul_f32,
    tune_add_softsign_mul_f32,
    f32,
    "f32",
    add_softsign_mul_f32,
    add_softsign_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_softsign_mul_f64,
    tune_add_softsign_mul_f64,
    f64,
    "f64",
    add_softsign_mul_f64,
    add_softsign_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_softsign_mul_2d_f16,
    tune_add_softsign_mul_2d_f16,
    f16,
    "f16",
    add_softsign_mul_2d_f16,
    add_softsign_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_softsign_mul_2d_f32,
    tune_add_softsign_mul_2d_f32,
    f32,
    "f32",
    add_softsign_mul_2d_f32,
    add_softsign_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_softsign_mul_2d_f64,
    tune_add_softsign_mul_2d_f64,
    f64,
    "f64",
    add_softsign_mul_2d_f64,
    add_softsign_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_mish_mul_f16,
    tune_add_mish_mul_f16,
    f16,
    "f16",
    add_mish_mul_f16,
    add_mish_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_mish_mul_f32,
    tune_add_mish_mul_f32,
    f32,
    "f32",
    add_mish_mul_f32,
    add_mish_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_mish_mul_f64,
    tune_add_mish_mul_f64,
    f64,
    "f64",
    add_mish_mul_f64,
    add_mish_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_mish_mul_2d_f16,
    tune_add_mish_mul_2d_f16,
    f16,
    "f16",
    add_mish_mul_2d_f16,
    add_mish_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_mish_mul_2d_f32,
    tune_add_mish_mul_2d_f32,
    f32,
    "f32",
    add_mish_mul_2d_f32,
    add_mish_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_mish_mul_2d_f64,
    tune_add_mish_mul_2d_f64,
    f64,
    "f64",
    add_mish_mul_2d_f64,
    add_mish_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_selu_mul_f16,
    tune_add_selu_mul_f16,
    f16,
    "f16",
    add_selu_mul_f16,
    add_selu_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_selu_mul_f32,
    tune_add_selu_mul_f32,
    f32,
    "f32",
    add_selu_mul_f32,
    add_selu_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_selu_mul_f64,
    tune_add_selu_mul_f64,
    f64,
    "f64",
    add_selu_mul_f64,
    add_selu_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_selu_mul_2d_f16,
    tune_add_selu_mul_2d_f16,
    f16,
    "f16",
    add_selu_mul_2d_f16,
    add_selu_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_selu_mul_2d_f32,
    tune_add_selu_mul_2d_f32,
    f32,
    "f32",
    add_selu_mul_2d_f32,
    add_selu_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_selu_mul_2d_f64,
    tune_add_selu_mul_2d_f64,
    f64,
    "f64",
    add_selu_mul_2d_f64,
    add_selu_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_tanhshrink_mul_f16,
    tune_add_tanhshrink_mul_f16,
    f16,
    "f16",
    add_tanhshrink_mul_f16,
    add_tanhshrink_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_tanhshrink_mul_f32,
    tune_add_tanhshrink_mul_f32,
    f32,
    "f32",
    add_tanhshrink_mul_f32,
    add_tanhshrink_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_tanhshrink_mul_f64,
    tune_add_tanhshrink_mul_f64,
    f64,
    "f64",
    add_tanhshrink_mul_f64,
    add_tanhshrink_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_tanhshrink_mul_2d_f16,
    tune_add_tanhshrink_mul_2d_f16,
    f16,
    "f16",
    add_tanhshrink_mul_2d_f16,
    add_tanhshrink_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_tanhshrink_mul_2d_f32,
    tune_add_tanhshrink_mul_2d_f32,
    f32,
    "f32",
    add_tanhshrink_mul_2d_f32,
    add_tanhshrink_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_tanhshrink_mul_2d_f64,
    tune_add_tanhshrink_mul_2d_f64,
    f64,
    "f64",
    add_tanhshrink_mul_2d_f64,
    add_tanhshrink_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_log_sigmoid_mul_f16,
    tune_add_log_sigmoid_mul_f16,
    f16,
    "f16",
    add_log_sigmoid_mul_f16,
    add_log_sigmoid_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_log_sigmoid_mul_f32,
    tune_add_log_sigmoid_mul_f32,
    f32,
    "f32",
    add_log_sigmoid_mul_f32,
    add_log_sigmoid_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_log_sigmoid_mul_f64,
    tune_add_log_sigmoid_mul_f64,
    f64,
    "f64",
    add_log_sigmoid_mul_f64,
    add_log_sigmoid_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_log_sigmoid_mul_2d_f16,
    tune_add_log_sigmoid_mul_2d_f16,
    f16,
    "f16",
    add_log_sigmoid_mul_2d_f16,
    add_log_sigmoid_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_log_sigmoid_mul_2d_f32,
    tune_add_log_sigmoid_mul_2d_f32,
    f32,
    "f32",
    add_log_sigmoid_mul_2d_f32,
    add_log_sigmoid_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_log_sigmoid_mul_2d_f64,
    tune_add_log_sigmoid_mul_2d_f64,
    f64,
    "f64",
    add_log_sigmoid_mul_2d_f64,
    add_log_sigmoid_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_reciprocal_mul_f16,
    tune_add_reciprocal_mul_f16,
    f16,
    "f16",
    add_reciprocal_mul_f16,
    add_reciprocal_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_reciprocal_mul_f32,
    tune_add_reciprocal_mul_f32,
    f32,
    "f32",
    add_reciprocal_mul_f32,
    add_reciprocal_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_reciprocal_mul_f64,
    tune_add_reciprocal_mul_f64,
    f64,
    "f64",
    add_reciprocal_mul_f64,
    add_reciprocal_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_reciprocal_mul_2d_f16,
    tune_add_reciprocal_mul_2d_f16,
    f16,
    "f16",
    add_reciprocal_mul_2d_f16,
    add_reciprocal_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_reciprocal_mul_2d_f32,
    tune_add_reciprocal_mul_2d_f32,
    f32,
    "f32",
    add_reciprocal_mul_2d_f32,
    add_reciprocal_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_reciprocal_mul_2d_f64,
    tune_add_reciprocal_mul_2d_f64,
    f64,
    "f64",
    add_reciprocal_mul_2d_f64,
    add_reciprocal_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_square_mul_f16,
    tune_add_square_mul_f16,
    f16,
    "f16",
    add_square_mul_f16,
    add_square_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_square_mul_f32,
    tune_add_square_mul_f32,
    f32,
    "f32",
    add_square_mul_f32,
    add_square_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_square_mul_f64,
    tune_add_square_mul_f64,
    f64,
    "f64",
    add_square_mul_f64,
    add_square_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_square_mul_2d_f16,
    tune_add_square_mul_2d_f16,
    f16,
    "f16",
    add_square_mul_2d_f16,
    add_square_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_square_mul_2d_f32,
    tune_add_square_mul_2d_f32,
    f32,
    "f32",
    add_square_mul_2d_f32,
    add_square_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_square_mul_2d_f64,
    tune_add_square_mul_2d_f64,
    f64,
    "f64",
    add_square_mul_2d_f64,
    add_square_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_ceil_mul_f16,
    tune_add_ceil_mul_f16,
    f16,
    "f16",
    add_ceil_mul_f16,
    add_ceil_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_ceil_mul_f32,
    tune_add_ceil_mul_f32,
    f32,
    "f32",
    add_ceil_mul_f32,
    add_ceil_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_ceil_mul_f64,
    tune_add_ceil_mul_f64,
    f64,
    "f64",
    add_ceil_mul_f64,
    add_ceil_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_ceil_mul_2d_f16,
    tune_add_ceil_mul_2d_f16,
    f16,
    "f16",
    add_ceil_mul_2d_f16,
    add_ceil_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_ceil_mul_2d_f32,
    tune_add_ceil_mul_2d_f32,
    f32,
    "f32",
    add_ceil_mul_2d_f32,
    add_ceil_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_ceil_mul_2d_f64,
    tune_add_ceil_mul_2d_f64,
    f64,
    "f64",
    add_ceil_mul_2d_f64,
    add_ceil_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_floor_mul_f16,
    tune_add_floor_mul_f16,
    f16,
    "f16",
    add_floor_mul_f16,
    add_floor_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_floor_mul_f32,
    tune_add_floor_mul_f32,
    f32,
    "f32",
    add_floor_mul_f32,
    add_floor_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_floor_mul_f64,
    tune_add_floor_mul_f64,
    f64,
    "f64",
    add_floor_mul_f64,
    add_floor_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_floor_mul_2d_f16,
    tune_add_floor_mul_2d_f16,
    f16,
    "f16",
    add_floor_mul_2d_f16,
    add_floor_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_floor_mul_2d_f32,
    tune_add_floor_mul_2d_f32,
    f32,
    "f32",
    add_floor_mul_2d_f32,
    add_floor_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_floor_mul_2d_f64,
    tune_add_floor_mul_2d_f64,
    f64,
    "f64",
    add_floor_mul_2d_f64,
    add_floor_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_trunc_mul_f16,
    tune_add_trunc_mul_f16,
    f16,
    "f16",
    add_trunc_mul_f16,
    add_trunc_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_trunc_mul_f32,
    tune_add_trunc_mul_f32,
    f32,
    "f32",
    add_trunc_mul_f32,
    add_trunc_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_trunc_mul_f64,
    tune_add_trunc_mul_f64,
    f64,
    "f64",
    add_trunc_mul_f64,
    add_trunc_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_trunc_mul_2d_f16,
    tune_add_trunc_mul_2d_f16,
    f16,
    "f16",
    add_trunc_mul_2d_f16,
    add_trunc_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_trunc_mul_2d_f32,
    tune_add_trunc_mul_2d_f32,
    f32,
    "f32",
    add_trunc_mul_2d_f32,
    add_trunc_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_trunc_mul_2d_f64,
    tune_add_trunc_mul_2d_f64,
    f64,
    "f64",
    add_trunc_mul_2d_f64,
    add_trunc_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_frac_mul_f16,
    tune_add_frac_mul_f16,
    f16,
    "f16",
    add_frac_mul_f16,
    add_frac_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_frac_mul_f32,
    tune_add_frac_mul_f32,
    f32,
    "f32",
    add_frac_mul_f32,
    add_frac_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_frac_mul_f64,
    tune_add_frac_mul_f64,
    f64,
    "f64",
    add_frac_mul_f64,
    add_frac_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_frac_mul_2d_f16,
    tune_add_frac_mul_2d_f16,
    f16,
    "f16",
    add_frac_mul_2d_f16,
    add_frac_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_frac_mul_2d_f32,
    tune_add_frac_mul_2d_f32,
    f32,
    "f32",
    add_frac_mul_2d_f32,
    add_frac_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_frac_mul_2d_f64,
    tune_add_frac_mul_2d_f64,
    f64,
    "f64",
    add_frac_mul_2d_f64,
    add_frac_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_deg2rad_mul_f16,
    tune_add_deg2rad_mul_f16,
    f16,
    "f16",
    add_deg2rad_mul_f16,
    add_deg2rad_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_deg2rad_mul_f32,
    tune_add_deg2rad_mul_f32,
    f32,
    "f32",
    add_deg2rad_mul_f32,
    add_deg2rad_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_deg2rad_mul_f64,
    tune_add_deg2rad_mul_f64,
    f64,
    "f64",
    add_deg2rad_mul_f64,
    add_deg2rad_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_deg2rad_mul_2d_f16,
    tune_add_deg2rad_mul_2d_f16,
    f16,
    "f16",
    add_deg2rad_mul_2d_f16,
    add_deg2rad_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_deg2rad_mul_2d_f32,
    tune_add_deg2rad_mul_2d_f32,
    f32,
    "f32",
    add_deg2rad_mul_2d_f32,
    add_deg2rad_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_deg2rad_mul_2d_f64,
    tune_add_deg2rad_mul_2d_f64,
    f64,
    "f64",
    add_deg2rad_mul_2d_f64,
    add_deg2rad_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_rad2deg_mul_f16,
    tune_add_rad2deg_mul_f16,
    f16,
    "f16",
    add_rad2deg_mul_f16,
    add_rad2deg_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_rad2deg_mul_f32,
    tune_add_rad2deg_mul_f32,
    f32,
    "f32",
    add_rad2deg_mul_f32,
    add_rad2deg_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_rad2deg_mul_f64,
    tune_add_rad2deg_mul_f64,
    f64,
    "f64",
    add_rad2deg_mul_f64,
    add_rad2deg_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_rad2deg_mul_2d_f16,
    tune_add_rad2deg_mul_2d_f16,
    f16,
    "f16",
    add_rad2deg_mul_2d_f16,
    add_rad2deg_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_rad2deg_mul_2d_f32,
    tune_add_rad2deg_mul_2d_f32,
    f32,
    "f32",
    add_rad2deg_mul_2d_f32,
    add_rad2deg_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_rad2deg_mul_2d_f64,
    tune_add_rad2deg_mul_2d_f64,
    f64,
    "f64",
    add_rad2deg_mul_2d_f64,
    add_rad2deg_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_sin_mul_f16,
    tune_add_sin_mul_f16,
    f16,
    "f16",
    add_sin_mul_f16,
    add_sin_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_sin_mul_f32,
    tune_add_sin_mul_f32,
    f32,
    "f32",
    add_sin_mul_f32,
    add_sin_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_sin_mul_f64,
    tune_add_sin_mul_f64,
    f64,
    "f64",
    add_sin_mul_f64,
    add_sin_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_sin_mul_2d_f16,
    tune_add_sin_mul_2d_f16,
    f16,
    "f16",
    add_sin_mul_2d_f16,
    add_sin_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_sin_mul_2d_f32,
    tune_add_sin_mul_2d_f32,
    f32,
    "f32",
    add_sin_mul_2d_f32,
    add_sin_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_sin_mul_2d_f64,
    tune_add_sin_mul_2d_f64,
    f64,
    "f64",
    add_sin_mul_2d_f64,
    add_sin_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_cos_mul_f16,
    tune_add_cos_mul_f16,
    f16,
    "f16",
    add_cos_mul_f16,
    add_cos_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_cos_mul_f32,
    tune_add_cos_mul_f32,
    f32,
    "f32",
    add_cos_mul_f32,
    add_cos_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_cos_mul_f64,
    tune_add_cos_mul_f64,
    f64,
    "f64",
    add_cos_mul_f64,
    add_cos_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_cos_mul_2d_f16,
    tune_add_cos_mul_2d_f16,
    f16,
    "f16",
    add_cos_mul_2d_f16,
    add_cos_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_cos_mul_2d_f32,
    tune_add_cos_mul_2d_f32,
    f32,
    "f32",
    add_cos_mul_2d_f32,
    add_cos_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_cos_mul_2d_f64,
    tune_add_cos_mul_2d_f64,
    f64,
    "f64",
    add_cos_mul_2d_f64,
    add_cos_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_tan_mul_f16,
    tune_add_tan_mul_f16,
    f16,
    "f16",
    add_tan_mul_f16,
    add_tan_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_tan_mul_f32,
    tune_add_tan_mul_f32,
    f32,
    "f32",
    add_tan_mul_f32,
    add_tan_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_tan_mul_f64,
    tune_add_tan_mul_f64,
    f64,
    "f64",
    add_tan_mul_f64,
    add_tan_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_tan_mul_2d_f16,
    tune_add_tan_mul_2d_f16,
    f16,
    "f16",
    add_tan_mul_2d_f16,
    add_tan_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_tan_mul_2d_f32,
    tune_add_tan_mul_2d_f32,
    f32,
    "f32",
    add_tan_mul_2d_f32,
    add_tan_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_tan_mul_2d_f64,
    tune_add_tan_mul_2d_f64,
    f64,
    "f64",
    add_tan_mul_2d_f64,
    add_tan_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_asin_mul_f16,
    tune_add_asin_mul_f16,
    f16,
    "f16",
    add_asin_mul_f16,
    add_asin_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_asin_mul_f32,
    tune_add_asin_mul_f32,
    f32,
    "f32",
    add_asin_mul_f32,
    add_asin_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_asin_mul_f64,
    tune_add_asin_mul_f64,
    f64,
    "f64",
    add_asin_mul_f64,
    add_asin_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_asin_mul_2d_f16,
    tune_add_asin_mul_2d_f16,
    f16,
    "f16",
    add_asin_mul_2d_f16,
    add_asin_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_asin_mul_2d_f32,
    tune_add_asin_mul_2d_f32,
    f32,
    "f32",
    add_asin_mul_2d_f32,
    add_asin_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_asin_mul_2d_f64,
    tune_add_asin_mul_2d_f64,
    f64,
    "f64",
    add_asin_mul_2d_f64,
    add_asin_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_acos_mul_f16,
    tune_add_acos_mul_f16,
    f16,
    "f16",
    add_acos_mul_f16,
    add_acos_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_acos_mul_f32,
    tune_add_acos_mul_f32,
    f32,
    "f32",
    add_acos_mul_f32,
    add_acos_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_acos_mul_f64,
    tune_add_acos_mul_f64,
    f64,
    "f64",
    add_acos_mul_f64,
    add_acos_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_acos_mul_2d_f16,
    tune_add_acos_mul_2d_f16,
    f16,
    "f16",
    add_acos_mul_2d_f16,
    add_acos_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_acos_mul_2d_f32,
    tune_add_acos_mul_2d_f32,
    f32,
    "f32",
    add_acos_mul_2d_f32,
    add_acos_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_acos_mul_2d_f64,
    tune_add_acos_mul_2d_f64,
    f64,
    "f64",
    add_acos_mul_2d_f64,
    add_acos_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_atan_mul_f16,
    tune_add_atan_mul_f16,
    f16,
    "f16",
    add_atan_mul_f16,
    add_atan_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_atan_mul_f32,
    tune_add_atan_mul_f32,
    f32,
    "f32",
    add_atan_mul_f32,
    add_atan_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_atan_mul_f64,
    tune_add_atan_mul_f64,
    f64,
    "f64",
    add_atan_mul_f64,
    add_atan_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_atan_mul_2d_f16,
    tune_add_atan_mul_2d_f16,
    f16,
    "f16",
    add_atan_mul_2d_f16,
    add_atan_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_atan_mul_2d_f32,
    tune_add_atan_mul_2d_f32,
    f32,
    "f32",
    add_atan_mul_2d_f32,
    add_atan_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_atan_mul_2d_f64,
    tune_add_atan_mul_2d_f64,
    f64,
    "f64",
    add_atan_mul_2d_f64,
    add_atan_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_sinh_mul_f16,
    tune_add_sinh_mul_f16,
    f16,
    "f16",
    add_sinh_mul_f16,
    add_sinh_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_sinh_mul_f32,
    tune_add_sinh_mul_f32,
    f32,
    "f32",
    add_sinh_mul_f32,
    add_sinh_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_sinh_mul_f64,
    tune_add_sinh_mul_f64,
    f64,
    "f64",
    add_sinh_mul_f64,
    add_sinh_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_sinh_mul_2d_f16,
    tune_add_sinh_mul_2d_f16,
    f16,
    "f16",
    add_sinh_mul_2d_f16,
    add_sinh_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_sinh_mul_2d_f32,
    tune_add_sinh_mul_2d_f32,
    f32,
    "f32",
    add_sinh_mul_2d_f32,
    add_sinh_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_sinh_mul_2d_f64,
    tune_add_sinh_mul_2d_f64,
    f64,
    "f64",
    add_sinh_mul_2d_f64,
    add_sinh_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_cosh_mul_f16,
    tune_add_cosh_mul_f16,
    f16,
    "f16",
    add_cosh_mul_f16,
    add_cosh_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_cosh_mul_f32,
    tune_add_cosh_mul_f32,
    f32,
    "f32",
    add_cosh_mul_f32,
    add_cosh_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_cosh_mul_f64,
    tune_add_cosh_mul_f64,
    f64,
    "f64",
    add_cosh_mul_f64,
    add_cosh_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_cosh_mul_2d_f16,
    tune_add_cosh_mul_2d_f16,
    f16,
    "f16",
    add_cosh_mul_2d_f16,
    add_cosh_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_cosh_mul_2d_f32,
    tune_add_cosh_mul_2d_f32,
    f32,
    "f32",
    add_cosh_mul_2d_f32,
    add_cosh_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_cosh_mul_2d_f64,
    tune_add_cosh_mul_2d_f64,
    f64,
    "f64",
    add_cosh_mul_2d_f64,
    add_cosh_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_asinh_mul_f16,
    tune_add_asinh_mul_f16,
    f16,
    "f16",
    add_asinh_mul_f16,
    add_asinh_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_asinh_mul_f32,
    tune_add_asinh_mul_f32,
    f32,
    "f32",
    add_asinh_mul_f32,
    add_asinh_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_asinh_mul_f64,
    tune_add_asinh_mul_f64,
    f64,
    "f64",
    add_asinh_mul_f64,
    add_asinh_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_asinh_mul_2d_f16,
    tune_add_asinh_mul_2d_f16,
    f16,
    "f16",
    add_asinh_mul_2d_f16,
    add_asinh_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_asinh_mul_2d_f32,
    tune_add_asinh_mul_2d_f32,
    f32,
    "f32",
    add_asinh_mul_2d_f32,
    add_asinh_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_asinh_mul_2d_f64,
    tune_add_asinh_mul_2d_f64,
    f64,
    "f64",
    add_asinh_mul_2d_f64,
    add_asinh_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_acosh_mul_f16,
    tune_add_acosh_mul_f16,
    f16,
    "f16",
    add_acosh_mul_f16,
    add_acosh_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_acosh_mul_f32,
    tune_add_acosh_mul_f32,
    f32,
    "f32",
    add_acosh_mul_f32,
    add_acosh_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_acosh_mul_f64,
    tune_add_acosh_mul_f64,
    f64,
    "f64",
    add_acosh_mul_f64,
    add_acosh_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_acosh_mul_2d_f16,
    tune_add_acosh_mul_2d_f16,
    f16,
    "f16",
    add_acosh_mul_2d_f16,
    add_acosh_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_acosh_mul_2d_f32,
    tune_add_acosh_mul_2d_f32,
    f32,
    "f32",
    add_acosh_mul_2d_f32,
    add_acosh_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_acosh_mul_2d_f64,
    tune_add_acosh_mul_2d_f64,
    f64,
    "f64",
    add_acosh_mul_2d_f64,
    add_acosh_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_atanh_mul_f16,
    tune_add_atanh_mul_f16,
    f16,
    "f16",
    add_atanh_mul_f16,
    add_atanh_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_atanh_mul_f32,
    tune_add_atanh_mul_f32,
    f32,
    "f32",
    add_atanh_mul_f32,
    add_atanh_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_atanh_mul_f64,
    tune_add_atanh_mul_f64,
    f64,
    "f64",
    add_atanh_mul_f64,
    add_atanh_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_atanh_mul_2d_f16,
    tune_add_atanh_mul_2d_f16,
    f16,
    "f16",
    add_atanh_mul_2d_f16,
    add_atanh_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_atanh_mul_2d_f32,
    tune_add_atanh_mul_2d_f32,
    f32,
    "f32",
    add_atanh_mul_2d_f32,
    add_atanh_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_atanh_mul_2d_f64,
    tune_add_atanh_mul_2d_f64,
    f64,
    "f64",
    add_atanh_mul_2d_f64,
    add_atanh_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_tanh_mul_f16,
    tune_add_tanh_mul_f16,
    f16,
    "f16",
    add_tanh_mul_f16,
    add_tanh_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_tanh_mul_f32,
    tune_add_tanh_mul_f32,
    f32,
    "f32",
    add_tanh_mul_f32,
    add_tanh_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_tanh_mul_f64,
    tune_add_tanh_mul_f64,
    f64,
    "f64",
    add_tanh_mul_f64,
    add_tanh_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_tanh_mul_2d_f16,
    tune_add_tanh_mul_2d_f16,
    f16,
    "f16",
    add_tanh_mul_2d_f16,
    add_tanh_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_tanh_mul_2d_f32,
    tune_add_tanh_mul_2d_f32,
    f32,
    "f32",
    add_tanh_mul_2d_f32,
    add_tanh_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_tanh_mul_2d_f64,
    tune_add_tanh_mul_2d_f64,
    f64,
    "f64",
    add_tanh_mul_2d_f64,
    add_tanh_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_exp_mul_f16,
    tune_add_exp_mul_f16,
    f16,
    "f16",
    add_exp_mul_f16,
    add_exp_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_exp_mul_f32,
    tune_add_exp_mul_f32,
    f32,
    "f32",
    add_exp_mul_f32,
    add_exp_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_exp_mul_f64,
    tune_add_exp_mul_f64,
    f64,
    "f64",
    add_exp_mul_f64,
    add_exp_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_exp_mul_2d_f16,
    tune_add_exp_mul_2d_f16,
    f16,
    "f16",
    add_exp_mul_2d_f16,
    add_exp_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_exp_mul_2d_f32,
    tune_add_exp_mul_2d_f32,
    f32,
    "f32",
    add_exp_mul_2d_f32,
    add_exp_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_exp_mul_2d_f64,
    tune_add_exp_mul_2d_f64,
    f64,
    "f64",
    add_exp_mul_2d_f64,
    add_exp_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_exp2_mul_f16,
    tune_add_exp2_mul_f16,
    f16,
    "f16",
    add_exp2_mul_f16,
    add_exp2_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_exp2_mul_f32,
    tune_add_exp2_mul_f32,
    f32,
    "f32",
    add_exp2_mul_f32,
    add_exp2_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_exp2_mul_f64,
    tune_add_exp2_mul_f64,
    f64,
    "f64",
    add_exp2_mul_f64,
    add_exp2_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_exp2_mul_2d_f16,
    tune_add_exp2_mul_2d_f16,
    f16,
    "f16",
    add_exp2_mul_2d_f16,
    add_exp2_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_exp2_mul_2d_f32,
    tune_add_exp2_mul_2d_f32,
    f32,
    "f32",
    add_exp2_mul_2d_f32,
    add_exp2_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_exp2_mul_2d_f64,
    tune_add_exp2_mul_2d_f64,
    f64,
    "f64",
    add_exp2_mul_2d_f64,
    add_exp2_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_expm1_mul_f16,
    tune_add_expm1_mul_f16,
    f16,
    "f16",
    add_expm1_mul_f16,
    add_expm1_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_expm1_mul_f32,
    tune_add_expm1_mul_f32,
    f32,
    "f32",
    add_expm1_mul_f32,
    add_expm1_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_expm1_mul_f64,
    tune_add_expm1_mul_f64,
    f64,
    "f64",
    add_expm1_mul_f64,
    add_expm1_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_expm1_mul_2d_f16,
    tune_add_expm1_mul_2d_f16,
    f16,
    "f16",
    add_expm1_mul_2d_f16,
    add_expm1_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_expm1_mul_2d_f32,
    tune_add_expm1_mul_2d_f32,
    f32,
    "f32",
    add_expm1_mul_2d_f32,
    add_expm1_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_expm1_mul_2d_f64,
    tune_add_expm1_mul_2d_f64,
    f64,
    "f64",
    add_expm1_mul_2d_f64,
    add_expm1_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_log_mul_f16,
    tune_add_log_mul_f16,
    f16,
    "f16",
    add_log_mul_f16,
    add_log_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_log_mul_f32,
    tune_add_log_mul_f32,
    f32,
    "f32",
    add_log_mul_f32,
    add_log_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_log_mul_f64,
    tune_add_log_mul_f64,
    f64,
    "f64",
    add_log_mul_f64,
    add_log_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_log_mul_2d_f16,
    tune_add_log_mul_2d_f16,
    f16,
    "f16",
    add_log_mul_2d_f16,
    add_log_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_log_mul_2d_f32,
    tune_add_log_mul_2d_f32,
    f32,
    "f32",
    add_log_mul_2d_f32,
    add_log_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_log_mul_2d_f64,
    tune_add_log_mul_2d_f64,
    f64,
    "f64",
    add_log_mul_2d_f64,
    add_log_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_log2_mul_f16,
    tune_add_log2_mul_f16,
    f16,
    "f16",
    add_log2_mul_f16,
    add_log2_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_log2_mul_f32,
    tune_add_log2_mul_f32,
    f32,
    "f32",
    add_log2_mul_f32,
    add_log2_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_log2_mul_f64,
    tune_add_log2_mul_f64,
    f64,
    "f64",
    add_log2_mul_f64,
    add_log2_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_log2_mul_2d_f16,
    tune_add_log2_mul_2d_f16,
    f16,
    "f16",
    add_log2_mul_2d_f16,
    add_log2_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_log2_mul_2d_f32,
    tune_add_log2_mul_2d_f32,
    f32,
    "f32",
    add_log2_mul_2d_f32,
    add_log2_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_log2_mul_2d_f64,
    tune_add_log2_mul_2d_f64,
    f64,
    "f64",
    add_log2_mul_2d_f64,
    add_log2_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_log10_mul_f16,
    tune_add_log10_mul_f16,
    f16,
    "f16",
    add_log10_mul_f16,
    add_log10_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_log10_mul_f32,
    tune_add_log10_mul_f32,
    f32,
    "f32",
    add_log10_mul_f32,
    add_log10_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_log10_mul_f64,
    tune_add_log10_mul_f64,
    f64,
    "f64",
    add_log10_mul_f64,
    add_log10_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_log10_mul_2d_f16,
    tune_add_log10_mul_2d_f16,
    f16,
    "f16",
    add_log10_mul_2d_f16,
    add_log10_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_log10_mul_2d_f32,
    tune_add_log10_mul_2d_f32,
    f32,
    "f32",
    add_log10_mul_2d_f32,
    add_log10_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_log10_mul_2d_f64,
    tune_add_log10_mul_2d_f64,
    f64,
    "f64",
    add_log10_mul_2d_f64,
    add_log10_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_log1p_mul_f16,
    tune_add_log1p_mul_f16,
    f16,
    "f16",
    add_log1p_mul_f16,
    add_log1p_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_log1p_mul_f32,
    tune_add_log1p_mul_f32,
    f32,
    "f32",
    add_log1p_mul_f32,
    add_log1p_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_log1p_mul_f64,
    tune_add_log1p_mul_f64,
    f64,
    "f64",
    add_log1p_mul_f64,
    add_log1p_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_log1p_mul_2d_f16,
    tune_add_log1p_mul_2d_f16,
    f16,
    "f16",
    add_log1p_mul_2d_f16,
    add_log1p_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_log1p_mul_2d_f32,
    tune_add_log1p_mul_2d_f32,
    f32,
    "f32",
    add_log1p_mul_2d_f32,
    add_log1p_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_log1p_mul_2d_f64,
    tune_add_log1p_mul_2d_f64,
    f64,
    "f64",
    add_log1p_mul_2d_f64,
    add_log1p_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_softplus_mul_f16,
    tune_add_softplus_mul_f16,
    f16,
    "f16",
    add_softplus_mul_f16,
    add_softplus_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_softplus_mul_f32,
    tune_add_softplus_mul_f32,
    f32,
    "f32",
    add_softplus_mul_f32,
    add_softplus_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_softplus_mul_f64,
    tune_add_softplus_mul_f64,
    f64,
    "f64",
    add_softplus_mul_f64,
    add_softplus_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_softplus_mul_2d_f16,
    tune_add_softplus_mul_2d_f16,
    f16,
    "f16",
    add_softplus_mul_2d_f16,
    add_softplus_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_softplus_mul_2d_f32,
    tune_add_softplus_mul_2d_f32,
    f32,
    "f32",
    add_softplus_mul_2d_f32,
    add_softplus_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_softplus_mul_2d_f64,
    tune_add_softplus_mul_2d_f64,
    f64,
    "f64",
    add_softplus_mul_2d_f64,
    add_softplus_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_sqrt_mul_f16,
    tune_add_sqrt_mul_f16,
    f16,
    "f16",
    add_sqrt_mul_f16,
    add_sqrt_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_sqrt_mul_f32,
    tune_add_sqrt_mul_f32,
    f32,
    "f32",
    add_sqrt_mul_f32,
    add_sqrt_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_sqrt_mul_f64,
    tune_add_sqrt_mul_f64,
    f64,
    "f64",
    add_sqrt_mul_f64,
    add_sqrt_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_sqrt_mul_2d_f16,
    tune_add_sqrt_mul_2d_f16,
    f16,
    "f16",
    add_sqrt_mul_2d_f16,
    add_sqrt_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_sqrt_mul_2d_f32,
    tune_add_sqrt_mul_2d_f32,
    f32,
    "f32",
    add_sqrt_mul_2d_f32,
    add_sqrt_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_sqrt_mul_2d_f64,
    tune_add_sqrt_mul_2d_f64,
    f64,
    "f64",
    add_sqrt_mul_2d_f64,
    add_sqrt_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_rsqrt_mul_f16,
    tune_add_rsqrt_mul_f16,
    f16,
    "f16",
    add_rsqrt_mul_f16,
    add_rsqrt_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_rsqrt_mul_f32,
    tune_add_rsqrt_mul_f32,
    f32,
    "f32",
    add_rsqrt_mul_f32,
    add_rsqrt_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_rsqrt_mul_f64,
    tune_add_rsqrt_mul_f64,
    f64,
    "f64",
    add_rsqrt_mul_f64,
    add_rsqrt_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_rsqrt_mul_2d_f16,
    tune_add_rsqrt_mul_2d_f16,
    f16,
    "f16",
    add_rsqrt_mul_2d_f16,
    add_rsqrt_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_rsqrt_mul_2d_f32,
    tune_add_rsqrt_mul_2d_f32,
    f32,
    "f32",
    add_rsqrt_mul_2d_f32,
    add_rsqrt_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_rsqrt_mul_2d_f64,
    tune_add_rsqrt_mul_2d_f64,
    f64,
    "f64",
    add_rsqrt_mul_2d_f64,
    add_rsqrt_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_cbrt_mul_f16,
    tune_add_cbrt_mul_f16,
    f16,
    "f16",
    add_cbrt_mul_f16,
    add_cbrt_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_cbrt_mul_f32,
    tune_add_cbrt_mul_f32,
    f32,
    "f32",
    add_cbrt_mul_f32,
    add_cbrt_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_cbrt_mul_f64,
    tune_add_cbrt_mul_f64,
    f64,
    "f64",
    add_cbrt_mul_f64,
    add_cbrt_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_cbrt_mul_2d_f16,
    tune_add_cbrt_mul_2d_f16,
    f16,
    "f16",
    add_cbrt_mul_2d_f16,
    add_cbrt_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_cbrt_mul_2d_f32,
    tune_add_cbrt_mul_2d_f32,
    f32,
    "f32",
    add_cbrt_mul_2d_f32,
    add_cbrt_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_cbrt_mul_2d_f64,
    tune_add_cbrt_mul_2d_f64,
    f64,
    "f64",
    add_cbrt_mul_2d_f64,
    add_cbrt_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_abs_mul_f16,
    tune_add_abs_mul_f16,
    f16,
    "f16",
    add_abs_mul_f16,
    add_abs_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_abs_mul_f32,
    tune_add_abs_mul_f32,
    f32,
    "f32",
    add_abs_mul_f32,
    add_abs_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_abs_mul_f64,
    tune_add_abs_mul_f64,
    f64,
    "f64",
    add_abs_mul_f64,
    add_abs_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_abs_mul_2d_f16,
    tune_add_abs_mul_2d_f16,
    f16,
    "f16",
    add_abs_mul_2d_f16,
    add_abs_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_abs_mul_2d_f32,
    tune_add_abs_mul_2d_f32,
    f32,
    "f32",
    add_abs_mul_2d_f32,
    add_abs_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_abs_mul_2d_f64,
    tune_add_abs_mul_2d_f64,
    f64,
    "f64",
    add_abs_mul_2d_f64,
    add_abs_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_neg_mul_f16,
    tune_add_neg_mul_f16,
    f16,
    "f16",
    add_neg_mul_f16,
    add_neg_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_neg_mul_f32,
    tune_add_neg_mul_f32,
    f32,
    "f32",
    add_neg_mul_f32,
    add_neg_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_neg_mul_f64,
    tune_add_neg_mul_f64,
    f64,
    "f64",
    add_neg_mul_f64,
    add_neg_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_neg_mul_2d_f16,
    tune_add_neg_mul_2d_f16,
    f16,
    "f16",
    add_neg_mul_2d_f16,
    add_neg_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_neg_mul_2d_f32,
    tune_add_neg_mul_2d_f32,
    f32,
    "f32",
    add_neg_mul_2d_f32,
    add_neg_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_neg_mul_2d_f64,
    tune_add_neg_mul_2d_f64,
    f64,
    "f64",
    add_neg_mul_2d_f64,
    add_neg_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    mul_add_f16,
    tune_mul_add_f16,
    f16,
    "f16",
    mul_add_f16,
    mul_add_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    mul_add_f32,
    tune_mul_add_f32,
    f32,
    "f32",
    mul_add_f32,
    mul_add_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    mul_add_f64,
    tune_mul_add_f64,
    f64,
    "f64",
    mul_add_f64,
    mul_add_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    mul_add_2d_f16,
    tune_mul_add_2d_f16,
    f16,
    "f16",
    mul_add_2d_f16,
    mul_add_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    mul_add_2d_f32,
    tune_mul_add_2d_f32,
    f32,
    "f32",
    mul_add_2d_f32,
    mul_add_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    mul_add_2d_f64,
    tune_mul_add_2d_f64,
    f64,
    "f64",
    mul_add_2d_f64,
    mul_add_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    add_mul_f16,
    tune_add_mul_f16,
    f16,
    "f16",
    add_mul_f16,
    add_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    add_mul_f32,
    tune_add_mul_f32,
    f32,
    "f32",
    add_mul_f32,
    add_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    add_mul_f64,
    tune_add_mul_f64,
    f64,
    "f64",
    add_mul_f64,
    add_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    add_mul_2d_f16,
    tune_add_mul_2d_f16,
    f16,
    "f16",
    add_mul_2d_f16,
    add_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    add_mul_2d_f32,
    tune_add_mul_2d_f32,
    f32,
    "f32",
    add_mul_2d_f32,
    add_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    add_mul_2d_f64,
    tune_add_mul_2d_f64,
    f64,
    "f64",
    add_mul_2d_f64,
    add_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    sub_mul_f16,
    tune_sub_mul_f16,
    f16,
    "f16",
    sub_mul_f16,
    sub_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    sub_mul_f32,
    tune_sub_mul_f32,
    f32,
    "f32",
    sub_mul_f32,
    sub_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    sub_mul_f64,
    tune_sub_mul_f64,
    f64,
    "f64",
    sub_mul_f64,
    sub_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    sub_mul_2d_f16,
    tune_sub_mul_2d_f16,
    f16,
    "f16",
    sub_mul_2d_f16,
    sub_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    sub_mul_2d_f32,
    tune_sub_mul_2d_f32,
    f32,
    "f32",
    sub_mul_2d_f32,
    sub_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    sub_mul_2d_f64,
    tune_sub_mul_2d_f64,
    f64,
    "f64",
    sub_mul_2d_f64,
    sub_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    rsub_mul_f16,
    tune_rsub_mul_f16,
    f16,
    "f16",
    rsub_mul_f16,
    rsub_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    rsub_mul_f32,
    tune_rsub_mul_f32,
    f32,
    "f32",
    rsub_mul_f32,
    rsub_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    rsub_mul_f64,
    tune_rsub_mul_f64,
    f64,
    "f64",
    rsub_mul_f64,
    rsub_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    rsub_mul_2d_f16,
    tune_rsub_mul_2d_f16,
    f16,
    "f16",
    rsub_mul_2d_f16,
    rsub_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    rsub_mul_2d_f32,
    tune_rsub_mul_2d_f32,
    f32,
    "f32",
    rsub_mul_2d_f32,
    rsub_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    rsub_mul_2d_f64,
    tune_rsub_mul_2d_f64,
    f64,
    "f64",
    rsub_mul_2d_f64,
    rsub_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    div_mul_f16,
    tune_div_mul_f16,
    f16,
    "f16",
    div_mul_f16,
    div_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    div_mul_f32,
    tune_div_mul_f32,
    f32,
    "f32",
    div_mul_f32,
    div_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    div_mul_f64,
    tune_div_mul_f64,
    f64,
    "f64",
    div_mul_f64,
    div_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    div_mul_2d_f16,
    tune_div_mul_2d_f16,
    f16,
    "f16",
    div_mul_2d_f16,
    div_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    div_mul_2d_f32,
    tune_div_mul_2d_f32,
    f32,
    "f32",
    div_mul_2d_f32,
    div_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    div_mul_2d_f64,
    tune_div_mul_2d_f64,
    f64,
    "f64",
    div_mul_2d_f64,
    div_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    rdiv_mul_f16,
    tune_rdiv_mul_f16,
    f16,
    "f16",
    rdiv_mul_f16,
    rdiv_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    rdiv_mul_f32,
    tune_rdiv_mul_f32,
    f32,
    "f32",
    rdiv_mul_f32,
    rdiv_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    rdiv_mul_f64,
    tune_rdiv_mul_f64,
    f64,
    "f64",
    rdiv_mul_f64,
    rdiv_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    rdiv_mul_2d_f16,
    tune_rdiv_mul_2d_f16,
    f16,
    "f16",
    rdiv_mul_2d_f16,
    rdiv_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    rdiv_mul_2d_f32,
    tune_rdiv_mul_2d_f32,
    f32,
    "f32",
    rdiv_mul_2d_f32,
    rdiv_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    rdiv_mul_2d_f64,
    tune_rdiv_mul_2d_f64,
    f64,
    "f64",
    rdiv_mul_2d_f64,
    rdiv_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    min_add_f16,
    tune_min_add_f16,
    f16,
    "f16",
    min_add_f16,
    min_add_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    min_add_f32,
    tune_min_add_f32,
    f32,
    "f32",
    min_add_f32,
    min_add_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    min_add_f64,
    tune_min_add_f64,
    f64,
    "f64",
    min_add_f64,
    min_add_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    min_add_2d_f16,
    tune_min_add_2d_f16,
    f16,
    "f16",
    min_add_2d_f16,
    min_add_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    min_add_2d_f32,
    tune_min_add_2d_f32,
    f32,
    "f32",
    min_add_2d_f32,
    min_add_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    min_add_2d_f64,
    tune_min_add_2d_f64,
    f64,
    "f64",
    min_add_2d_f64,
    min_add_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    max_add_f16,
    tune_max_add_f16,
    f16,
    "f16",
    max_add_f16,
    max_add_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    max_add_f32,
    tune_max_add_f32,
    f32,
    "f32",
    max_add_f32,
    max_add_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    max_add_f64,
    tune_max_add_f64,
    f64,
    "f64",
    max_add_f64,
    max_add_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    max_add_2d_f16,
    tune_max_add_2d_f16,
    f16,
    "f16",
    max_add_2d_f16,
    max_add_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    max_add_2d_f32,
    tune_max_add_2d_f32,
    f32,
    "f32",
    max_add_2d_f32,
    max_add_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    max_add_2d_f64,
    tune_max_add_2d_f64,
    f64,
    "f64",
    max_add_2d_f64,
    max_add_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    min_mul_f16,
    tune_min_mul_f16,
    f16,
    "f16",
    min_mul_f16,
    min_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    min_mul_f32,
    tune_min_mul_f32,
    f32,
    "f32",
    min_mul_f32,
    min_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    min_mul_f64,
    tune_min_mul_f64,
    f64,
    "f64",
    min_mul_f64,
    min_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    min_mul_2d_f16,
    tune_min_mul_2d_f16,
    f16,
    "f16",
    min_mul_2d_f16,
    min_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    min_mul_2d_f32,
    tune_min_mul_2d_f32,
    f32,
    "f32",
    min_mul_2d_f32,
    min_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    min_mul_2d_f64,
    tune_min_mul_2d_f64,
    f64,
    "f64",
    min_mul_2d_f64,
    min_mul_2d_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_fn!(
    max_mul_f16,
    tune_max_mul_f16,
    f16,
    "f16",
    max_mul_f16,
    max_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_fn!(
    max_mul_f32,
    tune_max_mul_f32,
    f32,
    "f32",
    max_mul_f32,
    max_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_fn!(
    max_mul_f64,
    tune_max_mul_f64,
    f64,
    "f64",
    max_mul_f64,
    max_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_ternary_fused_2d_fn!(
    max_mul_2d_f16,
    tune_max_mul_2d_f16,
    f16,
    "f16",
    max_mul_2d_f16,
    max_mul_2d_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_ternary_fused_2d_fn!(
    max_mul_2d_f32,
    tune_max_mul_2d_f32,
    f32,
    "f32",
    max_mul_2d_f32,
    max_mul_2d_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_ternary_fused_2d_fn!(
    max_mul_2d_f64,
    tune_max_mul_2d_f64,
    f64,
    "f64",
    max_mul_2d_f64,
    max_mul_2d_f64_tuned
);
