#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
#[cfg(any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64"))]
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64"))]
use crate::{
    cuda::{
        autotune::{TuneOptions, TuneReport},
        cutile,
        fused::{
            Fused2dConfig, FusedTileTuning, fused_2d_bias, fused_2d_column_ternary,
            tune_fused_2d_bias, tune_fused_2d_column_ternary,
        },
    },
    error::Result,
};

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `gelu(input + bias[column])` for every row-major matrix element.",
    bias_gelu_f16,
    tune_bias_gelu_f16,
    f16,
    "f16",
    bias_gelu_f16,
    bias_gelu_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `gelu(input + bias[column])` for every row-major matrix element.",
    bias_gelu_f32,
    tune_bias_gelu_f32,
    f32,
    "f32",
    bias_gelu_f32,
    bias_gelu_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `gelu(input + bias[column])` for every row-major matrix element.",
    bias_gelu_f64,
    tune_bias_gelu_f64,
    f64,
    "f64",
    bias_gelu_f64,
    bias_gelu_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `input - bias[column]` for every row-major matrix element.",
    bias_sub_f16,
    tune_bias_sub_f16,
    f16,
    "f16",
    bias_sub_f16,
    bias_sub_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `input - bias[column]` for every row-major matrix element.",
    bias_sub_f32,
    tune_bias_sub_f32,
    f32,
    "f32",
    bias_sub_f32,
    bias_sub_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `input - bias[column]` for every row-major matrix element.",
    bias_sub_f64,
    tune_bias_sub_f64,
    f64,
    "f64",
    bias_sub_f64,
    bias_sub_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `bias[column] - input` for every row-major matrix element.",
    bias_rsub_f16,
    tune_bias_rsub_f16,
    f16,
    "f16",
    bias_rsub_f16,
    bias_rsub_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `bias[column] - input` for every row-major matrix element.",
    bias_rsub_f32,
    tune_bias_rsub_f32,
    f32,
    "f32",
    bias_rsub_f32,
    bias_rsub_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `bias[column] - input` for every row-major matrix element.",
    bias_rsub_f64,
    tune_bias_rsub_f64,
    f64,
    "f64",
    bias_rsub_f64,
    bias_rsub_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `bias(lhs) * rhs` elementwise.",
    bias_mul_f16,
    tune_bias_mul_f16,
    f16,
    "f16",
    bias_mul_f16,
    bias_mul_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `bias(lhs) * rhs` elementwise.",
    bias_mul_f32,
    tune_bias_mul_f32,
    f32,
    "f32",
    bias_mul_f32,
    bias_mul_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `bias(lhs) * rhs` elementwise.",
    bias_mul_f64,
    tune_bias_mul_f64,
    f64,
    "f64",
    bias_mul_f64,
    bias_mul_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `input / bias[column]` for every row-major matrix element.",
    bias_div_f16,
    tune_bias_div_f16,
    f16,
    "f16",
    bias_div_f16,
    bias_div_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `input / bias[column]` for every row-major matrix element.",
    bias_div_f32,
    tune_bias_div_f32,
    f32,
    "f32",
    bias_div_f32,
    bias_div_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `input / bias[column]` for every row-major matrix element.",
    bias_div_f64,
    tune_bias_div_f64,
    f64,
    "f64",
    bias_div_f64,
    bias_div_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `bias[column] / input` for every row-major matrix element.",
    bias_rdiv_f16,
    tune_bias_rdiv_f16,
    f16,
    "f16",
    bias_rdiv_f16,
    bias_rdiv_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `bias[column] / input` for every row-major matrix element.",
    bias_rdiv_f32,
    tune_bias_rdiv_f32,
    f32,
    "f32",
    bias_rdiv_f32,
    bias_rdiv_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `bias[column] / input` for every row-major matrix element.",
    bias_rdiv_f64,
    tune_bias_rdiv_f64,
    f64,
    "f64",
    bias_rdiv_f64,
    bias_rdiv_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_add_f16,
    tune_bias_scale_add_f16,
    f16,
    "f16",
    bias_scale_add_f16,
    bias_scale_add_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_add_f32,
    tune_bias_scale_add_f32,
    f32,
    "f32",
    bias_scale_add_f32,
    bias_scale_add_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_add_f64,
    tune_bias_scale_add_f64,
    f64,
    "f64",
    bias_scale_add_f64,
    bias_scale_add_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_sub_f16,
    tune_bias_scale_sub_f16,
    f16,
    "f16",
    bias_scale_sub_f16,
    bias_scale_sub_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_sub_f32,
    tune_bias_scale_sub_f32,
    f32,
    "f32",
    bias_scale_sub_f32,
    bias_scale_sub_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_sub_f64,
    tune_bias_scale_sub_f64,
    f64,
    "f64",
    bias_scale_sub_f64,
    bias_scale_sub_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_add_scale_f16,
    tune_bias_add_scale_f16,
    f16,
    "f16",
    bias_add_scale_f16,
    bias_add_scale_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_add_scale_f32,
    tune_bias_add_scale_f32,
    f32,
    "f32",
    bias_add_scale_f32,
    bias_add_scale_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_add_scale_f64,
    tune_bias_add_scale_f64,
    f64,
    "f64",
    bias_add_scale_f64,
    bias_add_scale_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_sub_scale_f16,
    tune_bias_sub_scale_f16,
    f16,
    "f16",
    bias_sub_scale_f16,
    bias_sub_scale_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_sub_scale_f32,
    tune_bias_sub_scale_f32,
    f32,
    "f32",
    bias_sub_scale_f32,
    bias_sub_scale_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_sub_scale_f64,
    tune_bias_sub_scale_f64,
    f64,
    "f64",
    bias_sub_scale_f64,
    bias_sub_scale_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_gelu_f16,
    tune_bias_scale_gelu_f16,
    f16,
    "f16",
    bias_scale_gelu_f16,
    bias_scale_gelu_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_gelu_f32,
    tune_bias_scale_gelu_f32,
    f32,
    "f32",
    bias_scale_gelu_f32,
    bias_scale_gelu_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_gelu_f64,
    tune_bias_scale_gelu_f64,
    f64,
    "f64",
    bias_scale_gelu_f64,
    bias_scale_gelu_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_silu_f16,
    tune_bias_scale_silu_f16,
    f16,
    "f16",
    bias_scale_silu_f16,
    bias_scale_silu_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_silu_f32,
    tune_bias_scale_silu_f32,
    f32,
    "f32",
    bias_scale_silu_f32,
    bias_scale_silu_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_silu_f64,
    tune_bias_scale_silu_f64,
    f64,
    "f64",
    bias_scale_silu_f64,
    bias_scale_silu_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_sigmoid_f16,
    tune_bias_scale_sigmoid_f16,
    f16,
    "f16",
    bias_scale_sigmoid_f16,
    bias_scale_sigmoid_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_sigmoid_f32,
    tune_bias_scale_sigmoid_f32,
    f32,
    "f32",
    bias_scale_sigmoid_f32,
    bias_scale_sigmoid_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_sigmoid_f64,
    tune_bias_scale_sigmoid_f64,
    f64,
    "f64",
    bias_scale_sigmoid_f64,
    bias_scale_sigmoid_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_relu_f16,
    tune_bias_scale_relu_f16,
    f16,
    "f16",
    bias_scale_relu_f16,
    bias_scale_relu_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_relu_f32,
    tune_bias_scale_relu_f32,
    f32,
    "f32",
    bias_scale_relu_f32,
    bias_scale_relu_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_relu_f64,
    tune_bias_scale_relu_f64,
    f64,
    "f64",
    bias_scale_relu_f64,
    bias_scale_relu_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_relu6_f16,
    tune_bias_scale_relu6_f16,
    f16,
    "f16",
    bias_scale_relu6_f16,
    bias_scale_relu6_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_relu6_f32,
    tune_bias_scale_relu6_f32,
    f32,
    "f32",
    bias_scale_relu6_f32,
    bias_scale_relu6_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_relu6_f64,
    tune_bias_scale_relu6_f64,
    f64,
    "f64",
    bias_scale_relu6_f64,
    bias_scale_relu6_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_hard_sigmoid_f16,
    tune_bias_scale_hard_sigmoid_f16,
    f16,
    "f16",
    bias_scale_hard_sigmoid_f16,
    bias_scale_hard_sigmoid_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_hard_sigmoid_f32,
    tune_bias_scale_hard_sigmoid_f32,
    f32,
    "f32",
    bias_scale_hard_sigmoid_f32,
    bias_scale_hard_sigmoid_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_hard_sigmoid_f64,
    tune_bias_scale_hard_sigmoid_f64,
    f64,
    "f64",
    bias_scale_hard_sigmoid_f64,
    bias_scale_hard_sigmoid_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_hard_swish_f16,
    tune_bias_scale_hard_swish_f16,
    f16,
    "f16",
    bias_scale_hard_swish_f16,
    bias_scale_hard_swish_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_hard_swish_f32,
    tune_bias_scale_hard_swish_f32,
    f32,
    "f32",
    bias_scale_hard_swish_f32,
    bias_scale_hard_swish_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_hard_swish_f64,
    tune_bias_scale_hard_swish_f64,
    f64,
    "f64",
    bias_scale_hard_swish_f64,
    bias_scale_hard_swish_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_softsign_f16,
    tune_bias_scale_softsign_f16,
    f16,
    "f16",
    bias_scale_softsign_f16,
    bias_scale_softsign_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_softsign_f32,
    tune_bias_scale_softsign_f32,
    f32,
    "f32",
    bias_scale_softsign_f32,
    bias_scale_softsign_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_softsign_f64,
    tune_bias_scale_softsign_f64,
    f64,
    "f64",
    bias_scale_softsign_f64,
    bias_scale_softsign_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_mish_f16,
    tune_bias_scale_mish_f16,
    f16,
    "f16",
    bias_scale_mish_f16,
    bias_scale_mish_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_mish_f32,
    tune_bias_scale_mish_f32,
    f32,
    "f32",
    bias_scale_mish_f32,
    bias_scale_mish_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_mish_f64,
    tune_bias_scale_mish_f64,
    f64,
    "f64",
    bias_scale_mish_f64,
    bias_scale_mish_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_selu_f16,
    tune_bias_scale_selu_f16,
    f16,
    "f16",
    bias_scale_selu_f16,
    bias_scale_selu_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_selu_f32,
    tune_bias_scale_selu_f32,
    f32,
    "f32",
    bias_scale_selu_f32,
    bias_scale_selu_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_selu_f64,
    tune_bias_scale_selu_f64,
    f64,
    "f64",
    bias_scale_selu_f64,
    bias_scale_selu_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_tanhshrink_f16,
    tune_bias_scale_tanhshrink_f16,
    f16,
    "f16",
    bias_scale_tanhshrink_f16,
    bias_scale_tanhshrink_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_tanhshrink_f32,
    tune_bias_scale_tanhshrink_f32,
    f32,
    "f32",
    bias_scale_tanhshrink_f32,
    bias_scale_tanhshrink_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_tanhshrink_f64,
    tune_bias_scale_tanhshrink_f64,
    f64,
    "f64",
    bias_scale_tanhshrink_f64,
    bias_scale_tanhshrink_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_log_sigmoid_f16,
    tune_bias_scale_log_sigmoid_f16,
    f16,
    "f16",
    bias_scale_log_sigmoid_f16,
    bias_scale_log_sigmoid_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_log_sigmoid_f32,
    tune_bias_scale_log_sigmoid_f32,
    f32,
    "f32",
    bias_scale_log_sigmoid_f32,
    bias_scale_log_sigmoid_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_log_sigmoid_f64,
    tune_bias_scale_log_sigmoid_f64,
    f64,
    "f64",
    bias_scale_log_sigmoid_f64,
    bias_scale_log_sigmoid_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_reciprocal_f16,
    tune_bias_scale_reciprocal_f16,
    f16,
    "f16",
    bias_scale_reciprocal_f16,
    bias_scale_reciprocal_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_reciprocal_f32,
    tune_bias_scale_reciprocal_f32,
    f32,
    "f32",
    bias_scale_reciprocal_f32,
    bias_scale_reciprocal_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_reciprocal_f64,
    tune_bias_scale_reciprocal_f64,
    f64,
    "f64",
    bias_scale_reciprocal_f64,
    bias_scale_reciprocal_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_square_f16,
    tune_bias_scale_square_f16,
    f16,
    "f16",
    bias_scale_square_f16,
    bias_scale_square_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_square_f32,
    tune_bias_scale_square_f32,
    f32,
    "f32",
    bias_scale_square_f32,
    bias_scale_square_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_square_f64,
    tune_bias_scale_square_f64,
    f64,
    "f64",
    bias_scale_square_f64,
    bias_scale_square_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_ceil_f16,
    tune_bias_scale_ceil_f16,
    f16,
    "f16",
    bias_scale_ceil_f16,
    bias_scale_ceil_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_ceil_f32,
    tune_bias_scale_ceil_f32,
    f32,
    "f32",
    bias_scale_ceil_f32,
    bias_scale_ceil_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_ceil_f64,
    tune_bias_scale_ceil_f64,
    f64,
    "f64",
    bias_scale_ceil_f64,
    bias_scale_ceil_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_floor_f16,
    tune_bias_scale_floor_f16,
    f16,
    "f16",
    bias_scale_floor_f16,
    bias_scale_floor_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_floor_f32,
    tune_bias_scale_floor_f32,
    f32,
    "f32",
    bias_scale_floor_f32,
    bias_scale_floor_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_floor_f64,
    tune_bias_scale_floor_f64,
    f64,
    "f64",
    bias_scale_floor_f64,
    bias_scale_floor_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_trunc_f16,
    tune_bias_scale_trunc_f16,
    f16,
    "f16",
    bias_scale_trunc_f16,
    bias_scale_trunc_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_trunc_f32,
    tune_bias_scale_trunc_f32,
    f32,
    "f32",
    bias_scale_trunc_f32,
    bias_scale_trunc_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_trunc_f64,
    tune_bias_scale_trunc_f64,
    f64,
    "f64",
    bias_scale_trunc_f64,
    bias_scale_trunc_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_frac_f16,
    tune_bias_scale_frac_f16,
    f16,
    "f16",
    bias_scale_frac_f16,
    bias_scale_frac_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_frac_f32,
    tune_bias_scale_frac_f32,
    f32,
    "f32",
    bias_scale_frac_f32,
    bias_scale_frac_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_frac_f64,
    tune_bias_scale_frac_f64,
    f64,
    "f64",
    bias_scale_frac_f64,
    bias_scale_frac_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_deg2rad_f16,
    tune_bias_scale_deg2rad_f16,
    f16,
    "f16",
    bias_scale_deg2rad_f16,
    bias_scale_deg2rad_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_deg2rad_f32,
    tune_bias_scale_deg2rad_f32,
    f32,
    "f32",
    bias_scale_deg2rad_f32,
    bias_scale_deg2rad_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_deg2rad_f64,
    tune_bias_scale_deg2rad_f64,
    f64,
    "f64",
    bias_scale_deg2rad_f64,
    bias_scale_deg2rad_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_rad2deg_f16,
    tune_bias_scale_rad2deg_f16,
    f16,
    "f16",
    bias_scale_rad2deg_f16,
    bias_scale_rad2deg_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_rad2deg_f32,
    tune_bias_scale_rad2deg_f32,
    f32,
    "f32",
    bias_scale_rad2deg_f32,
    bias_scale_rad2deg_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_rad2deg_f64,
    tune_bias_scale_rad2deg_f64,
    f64,
    "f64",
    bias_scale_rad2deg_f64,
    bias_scale_rad2deg_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_sin_f16,
    tune_bias_scale_sin_f16,
    f16,
    "f16",
    bias_scale_sin_f16,
    bias_scale_sin_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_sin_f32,
    tune_bias_scale_sin_f32,
    f32,
    "f32",
    bias_scale_sin_f32,
    bias_scale_sin_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_sin_f64,
    tune_bias_scale_sin_f64,
    f64,
    "f64",
    bias_scale_sin_f64,
    bias_scale_sin_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_cos_f16,
    tune_bias_scale_cos_f16,
    f16,
    "f16",
    bias_scale_cos_f16,
    bias_scale_cos_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_cos_f32,
    tune_bias_scale_cos_f32,
    f32,
    "f32",
    bias_scale_cos_f32,
    bias_scale_cos_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_cos_f64,
    tune_bias_scale_cos_f64,
    f64,
    "f64",
    bias_scale_cos_f64,
    bias_scale_cos_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_tan_f16,
    tune_bias_scale_tan_f16,
    f16,
    "f16",
    bias_scale_tan_f16,
    bias_scale_tan_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_tan_f32,
    tune_bias_scale_tan_f32,
    f32,
    "f32",
    bias_scale_tan_f32,
    bias_scale_tan_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_tan_f64,
    tune_bias_scale_tan_f64,
    f64,
    "f64",
    bias_scale_tan_f64,
    bias_scale_tan_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_asin_f16,
    tune_bias_scale_asin_f16,
    f16,
    "f16",
    bias_scale_asin_f16,
    bias_scale_asin_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_asin_f32,
    tune_bias_scale_asin_f32,
    f32,
    "f32",
    bias_scale_asin_f32,
    bias_scale_asin_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_asin_f64,
    tune_bias_scale_asin_f64,
    f64,
    "f64",
    bias_scale_asin_f64,
    bias_scale_asin_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_acos_f16,
    tune_bias_scale_acos_f16,
    f16,
    "f16",
    bias_scale_acos_f16,
    bias_scale_acos_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_acos_f32,
    tune_bias_scale_acos_f32,
    f32,
    "f32",
    bias_scale_acos_f32,
    bias_scale_acos_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_acos_f64,
    tune_bias_scale_acos_f64,
    f64,
    "f64",
    bias_scale_acos_f64,
    bias_scale_acos_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_atan_f16,
    tune_bias_scale_atan_f16,
    f16,
    "f16",
    bias_scale_atan_f16,
    bias_scale_atan_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_atan_f32,
    tune_bias_scale_atan_f32,
    f32,
    "f32",
    bias_scale_atan_f32,
    bias_scale_atan_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_atan_f64,
    tune_bias_scale_atan_f64,
    f64,
    "f64",
    bias_scale_atan_f64,
    bias_scale_atan_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_sinh_f16,
    tune_bias_scale_sinh_f16,
    f16,
    "f16",
    bias_scale_sinh_f16,
    bias_scale_sinh_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_sinh_f32,
    tune_bias_scale_sinh_f32,
    f32,
    "f32",
    bias_scale_sinh_f32,
    bias_scale_sinh_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_sinh_f64,
    tune_bias_scale_sinh_f64,
    f64,
    "f64",
    bias_scale_sinh_f64,
    bias_scale_sinh_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_cosh_f16,
    tune_bias_scale_cosh_f16,
    f16,
    "f16",
    bias_scale_cosh_f16,
    bias_scale_cosh_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_cosh_f32,
    tune_bias_scale_cosh_f32,
    f32,
    "f32",
    bias_scale_cosh_f32,
    bias_scale_cosh_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_cosh_f64,
    tune_bias_scale_cosh_f64,
    f64,
    "f64",
    bias_scale_cosh_f64,
    bias_scale_cosh_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_asinh_f16,
    tune_bias_scale_asinh_f16,
    f16,
    "f16",
    bias_scale_asinh_f16,
    bias_scale_asinh_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_asinh_f32,
    tune_bias_scale_asinh_f32,
    f32,
    "f32",
    bias_scale_asinh_f32,
    bias_scale_asinh_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_asinh_f64,
    tune_bias_scale_asinh_f64,
    f64,
    "f64",
    bias_scale_asinh_f64,
    bias_scale_asinh_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_acosh_f16,
    tune_bias_scale_acosh_f16,
    f16,
    "f16",
    bias_scale_acosh_f16,
    bias_scale_acosh_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_acosh_f32,
    tune_bias_scale_acosh_f32,
    f32,
    "f32",
    bias_scale_acosh_f32,
    bias_scale_acosh_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_acosh_f64,
    tune_bias_scale_acosh_f64,
    f64,
    "f64",
    bias_scale_acosh_f64,
    bias_scale_acosh_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_atanh_f16,
    tune_bias_scale_atanh_f16,
    f16,
    "f16",
    bias_scale_atanh_f16,
    bias_scale_atanh_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_atanh_f32,
    tune_bias_scale_atanh_f32,
    f32,
    "f32",
    bias_scale_atanh_f32,
    bias_scale_atanh_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_atanh_f64,
    tune_bias_scale_atanh_f64,
    f64,
    "f64",
    bias_scale_atanh_f64,
    bias_scale_atanh_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_tanh_f16,
    tune_bias_scale_tanh_f16,
    f16,
    "f16",
    bias_scale_tanh_f16,
    bias_scale_tanh_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_tanh_f32,
    tune_bias_scale_tanh_f32,
    f32,
    "f32",
    bias_scale_tanh_f32,
    bias_scale_tanh_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_tanh_f64,
    tune_bias_scale_tanh_f64,
    f64,
    "f64",
    bias_scale_tanh_f64,
    bias_scale_tanh_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_exp_f16,
    tune_bias_scale_exp_f16,
    f16,
    "f16",
    bias_scale_exp_f16,
    bias_scale_exp_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_exp_f32,
    tune_bias_scale_exp_f32,
    f32,
    "f32",
    bias_scale_exp_f32,
    bias_scale_exp_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_exp_f64,
    tune_bias_scale_exp_f64,
    f64,
    "f64",
    bias_scale_exp_f64,
    bias_scale_exp_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_exp2_f16,
    tune_bias_scale_exp2_f16,
    f16,
    "f16",
    bias_scale_exp2_f16,
    bias_scale_exp2_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_exp2_f32,
    tune_bias_scale_exp2_f32,
    f32,
    "f32",
    bias_scale_exp2_f32,
    bias_scale_exp2_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_exp2_f64,
    tune_bias_scale_exp2_f64,
    f64,
    "f64",
    bias_scale_exp2_f64,
    bias_scale_exp2_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_expm1_f16,
    tune_bias_scale_expm1_f16,
    f16,
    "f16",
    bias_scale_expm1_f16,
    bias_scale_expm1_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_expm1_f32,
    tune_bias_scale_expm1_f32,
    f32,
    "f32",
    bias_scale_expm1_f32,
    bias_scale_expm1_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_expm1_f64,
    tune_bias_scale_expm1_f64,
    f64,
    "f64",
    bias_scale_expm1_f64,
    bias_scale_expm1_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_log_f16,
    tune_bias_scale_log_f16,
    f16,
    "f16",
    bias_scale_log_f16,
    bias_scale_log_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_log_f32,
    tune_bias_scale_log_f32,
    f32,
    "f32",
    bias_scale_log_f32,
    bias_scale_log_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_log_f64,
    tune_bias_scale_log_f64,
    f64,
    "f64",
    bias_scale_log_f64,
    bias_scale_log_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_log2_f16,
    tune_bias_scale_log2_f16,
    f16,
    "f16",
    bias_scale_log2_f16,
    bias_scale_log2_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_log2_f32,
    tune_bias_scale_log2_f32,
    f32,
    "f32",
    bias_scale_log2_f32,
    bias_scale_log2_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_log2_f64,
    tune_bias_scale_log2_f64,
    f64,
    "f64",
    bias_scale_log2_f64,
    bias_scale_log2_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_log10_f16,
    tune_bias_scale_log10_f16,
    f16,
    "f16",
    bias_scale_log10_f16,
    bias_scale_log10_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_log10_f32,
    tune_bias_scale_log10_f32,
    f32,
    "f32",
    bias_scale_log10_f32,
    bias_scale_log10_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_log10_f64,
    tune_bias_scale_log10_f64,
    f64,
    "f64",
    bias_scale_log10_f64,
    bias_scale_log10_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_log1p_f16,
    tune_bias_scale_log1p_f16,
    f16,
    "f16",
    bias_scale_log1p_f16,
    bias_scale_log1p_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_log1p_f32,
    tune_bias_scale_log1p_f32,
    f32,
    "f32",
    bias_scale_log1p_f32,
    bias_scale_log1p_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_log1p_f64,
    tune_bias_scale_log1p_f64,
    f64,
    "f64",
    bias_scale_log1p_f64,
    bias_scale_log1p_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_softplus_f16,
    tune_bias_scale_softplus_f16,
    f16,
    "f16",
    bias_scale_softplus_f16,
    bias_scale_softplus_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_softplus_f32,
    tune_bias_scale_softplus_f32,
    f32,
    "f32",
    bias_scale_softplus_f32,
    bias_scale_softplus_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_softplus_f64,
    tune_bias_scale_softplus_f64,
    f64,
    "f64",
    bias_scale_softplus_f64,
    bias_scale_softplus_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_sqrt_f16,
    tune_bias_scale_sqrt_f16,
    f16,
    "f16",
    bias_scale_sqrt_f16,
    bias_scale_sqrt_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_sqrt_f32,
    tune_bias_scale_sqrt_f32,
    f32,
    "f32",
    bias_scale_sqrt_f32,
    bias_scale_sqrt_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_sqrt_f64,
    tune_bias_scale_sqrt_f64,
    f64,
    "f64",
    bias_scale_sqrt_f64,
    bias_scale_sqrt_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_rsqrt_f16,
    tune_bias_scale_rsqrt_f16,
    f16,
    "f16",
    bias_scale_rsqrt_f16,
    bias_scale_rsqrt_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_rsqrt_f32,
    tune_bias_scale_rsqrt_f32,
    f32,
    "f32",
    bias_scale_rsqrt_f32,
    bias_scale_rsqrt_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_rsqrt_f64,
    tune_bias_scale_rsqrt_f64,
    f64,
    "f64",
    bias_scale_rsqrt_f64,
    bias_scale_rsqrt_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_cbrt_f16,
    tune_bias_scale_cbrt_f16,
    f16,
    "f16",
    bias_scale_cbrt_f16,
    bias_scale_cbrt_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_cbrt_f32,
    tune_bias_scale_cbrt_f32,
    f32,
    "f32",
    bias_scale_cbrt_f32,
    bias_scale_cbrt_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_cbrt_f64,
    tune_bias_scale_cbrt_f64,
    f64,
    "f64",
    bias_scale_cbrt_f64,
    bias_scale_cbrt_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_abs_f16,
    tune_bias_scale_abs_f16,
    f16,
    "f16",
    bias_scale_abs_f16,
    bias_scale_abs_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_abs_f32,
    tune_bias_scale_abs_f32,
    f32,
    "f32",
    bias_scale_abs_f32,
    bias_scale_abs_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_abs_f64,
    tune_bias_scale_abs_f64,
    f64,
    "f64",
    bias_scale_abs_f64,
    bias_scale_abs_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_affine_fused_fn!(
    bias_scale_neg_f16,
    tune_bias_scale_neg_f16,
    f16,
    "f16",
    bias_scale_neg_f16,
    bias_scale_neg_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_affine_fused_fn!(
    bias_scale_neg_f32,
    tune_bias_scale_neg_f32,
    f32,
    "f32",
    bias_scale_neg_f32,
    bias_scale_neg_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_affine_fused_fn!(
    bias_scale_neg_f64,
    tune_bias_scale_neg_f64,
    f64,
    "f64",
    bias_scale_neg_f64,
    bias_scale_neg_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `silu(input + bias[column])` for every row-major matrix element.",
    bias_silu_f16,
    tune_bias_silu_f16,
    f16,
    "f16",
    bias_silu_f16,
    bias_silu_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `silu(input + bias[column])` for every row-major matrix element.",
    bias_silu_f32,
    tune_bias_silu_f32,
    f32,
    "f32",
    bias_silu_f32,
    bias_silu_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `silu(input + bias[column])` for every row-major matrix element.",
    bias_silu_f64,
    tune_bias_silu_f64,
    f64,
    "f64",
    bias_silu_f64,
    bias_silu_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `sigmoid(input + bias[column])` for every row-major matrix element.",
    bias_sigmoid_f16,
    tune_bias_sigmoid_f16,
    f16,
    "f16",
    bias_sigmoid_f16,
    bias_sigmoid_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `sigmoid(input + bias[column])` for every row-major matrix element.",
    bias_sigmoid_f32,
    tune_bias_sigmoid_f32,
    f32,
    "f32",
    bias_sigmoid_f32,
    bias_sigmoid_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `sigmoid(input + bias[column])` for every row-major matrix element.",
    bias_sigmoid_f64,
    tune_bias_sigmoid_f64,
    f64,
    "f64",
    bias_sigmoid_f64,
    bias_sigmoid_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `relu(input + bias[column])` for every row-major matrix element.",
    bias_relu_f16,
    tune_bias_relu_f16,
    f16,
    "f16",
    bias_relu_f16,
    bias_relu_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `relu(input + bias[column])` for every row-major matrix element.",
    bias_relu_f32,
    tune_bias_relu_f32,
    f32,
    "f32",
    bias_relu_f32,
    bias_relu_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `relu(input + bias[column])` for every row-major matrix element.",
    bias_relu_f64,
    tune_bias_relu_f64,
    f64,
    "f64",
    bias_relu_f64,
    bias_relu_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `relu6(input + bias[column])` for every row-major matrix element.",
    bias_relu6_f16,
    tune_bias_relu6_f16,
    f16,
    "f16",
    bias_relu6_f16,
    bias_relu6_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `relu6(input + bias[column])` for every row-major matrix element.",
    bias_relu6_f32,
    tune_bias_relu6_f32,
    f32,
    "f32",
    bias_relu6_f32,
    bias_relu6_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `relu6(input + bias[column])` for every row-major matrix element.",
    bias_relu6_f64,
    tune_bias_relu6_f64,
    f64,
    "f64",
    bias_relu6_f64,
    bias_relu6_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `hard_sigmoid(input + bias[column])` for every row-major matrix element.",
    bias_hard_sigmoid_f16,
    tune_bias_hard_sigmoid_f16,
    f16,
    "f16",
    bias_hard_sigmoid_f16,
    bias_hard_sigmoid_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `hard_sigmoid(input + bias[column])` for every row-major matrix element.",
    bias_hard_sigmoid_f32,
    tune_bias_hard_sigmoid_f32,
    f32,
    "f32",
    bias_hard_sigmoid_f32,
    bias_hard_sigmoid_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `hard_sigmoid(input + bias[column])` for every row-major matrix element.",
    bias_hard_sigmoid_f64,
    tune_bias_hard_sigmoid_f64,
    f64,
    "f64",
    bias_hard_sigmoid_f64,
    bias_hard_sigmoid_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `hard_swish(input + bias[column])` for every row-major matrix element.",
    bias_hard_swish_f16,
    tune_bias_hard_swish_f16,
    f16,
    "f16",
    bias_hard_swish_f16,
    bias_hard_swish_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `hard_swish(input + bias[column])` for every row-major matrix element.",
    bias_hard_swish_f32,
    tune_bias_hard_swish_f32,
    f32,
    "f32",
    bias_hard_swish_f32,
    bias_hard_swish_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `hard_swish(input + bias[column])` for every row-major matrix element.",
    bias_hard_swish_f64,
    tune_bias_hard_swish_f64,
    f64,
    "f64",
    bias_hard_swish_f64,
    bias_hard_swish_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `softsign(input + bias[column])` for every row-major matrix element.",
    bias_softsign_f16,
    tune_bias_softsign_f16,
    f16,
    "f16",
    bias_softsign_f16,
    bias_softsign_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `softsign(input + bias[column])` for every row-major matrix element.",
    bias_softsign_f32,
    tune_bias_softsign_f32,
    f32,
    "f32",
    bias_softsign_f32,
    bias_softsign_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `softsign(input + bias[column])` for every row-major matrix element.",
    bias_softsign_f64,
    tune_bias_softsign_f64,
    f64,
    "f64",
    bias_softsign_f64,
    bias_softsign_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `mish(input + bias[column])` for every row-major matrix element.",
    bias_mish_f16,
    tune_bias_mish_f16,
    f16,
    "f16",
    bias_mish_f16,
    bias_mish_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `mish(input + bias[column])` for every row-major matrix element.",
    bias_mish_f32,
    tune_bias_mish_f32,
    f32,
    "f32",
    bias_mish_f32,
    bias_mish_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `mish(input + bias[column])` for every row-major matrix element.",
    bias_mish_f64,
    tune_bias_mish_f64,
    f64,
    "f64",
    bias_mish_f64,
    bias_mish_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `selu(input + bias[column])` for every row-major matrix element.",
    bias_selu_f16,
    tune_bias_selu_f16,
    f16,
    "f16",
    bias_selu_f16,
    bias_selu_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `selu(input + bias[column])` for every row-major matrix element.",
    bias_selu_f32,
    tune_bias_selu_f32,
    f32,
    "f32",
    bias_selu_f32,
    bias_selu_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `selu(input + bias[column])` for every row-major matrix element.",
    bias_selu_f64,
    tune_bias_selu_f64,
    f64,
    "f64",
    bias_selu_f64,
    bias_selu_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `tanhshrink(input + bias[column])` for every row-major matrix element.",
    bias_tanhshrink_f16,
    tune_bias_tanhshrink_f16,
    f16,
    "f16",
    bias_tanhshrink_f16,
    bias_tanhshrink_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `tanhshrink(input + bias[column])` for every row-major matrix element.",
    bias_tanhshrink_f32,
    tune_bias_tanhshrink_f32,
    f32,
    "f32",
    bias_tanhshrink_f32,
    bias_tanhshrink_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `tanhshrink(input + bias[column])` for every row-major matrix element.",
    bias_tanhshrink_f64,
    tune_bias_tanhshrink_f64,
    f64,
    "f64",
    bias_tanhshrink_f64,
    bias_tanhshrink_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `log_sigmoid(input + bias[column])` for every row-major matrix element.",
    bias_log_sigmoid_f16,
    tune_bias_log_sigmoid_f16,
    f16,
    "f16",
    bias_log_sigmoid_f16,
    bias_log_sigmoid_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `log_sigmoid(input + bias[column])` for every row-major matrix element.",
    bias_log_sigmoid_f32,
    tune_bias_log_sigmoid_f32,
    f32,
    "f32",
    bias_log_sigmoid_f32,
    bias_log_sigmoid_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `log_sigmoid(input + bias[column])` for every row-major matrix element.",
    bias_log_sigmoid_f64,
    tune_bias_log_sigmoid_f64,
    f64,
    "f64",
    bias_log_sigmoid_f64,
    bias_log_sigmoid_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `reciprocal(input + bias[column])` for every row-major matrix element.",
    bias_reciprocal_f16,
    tune_bias_reciprocal_f16,
    f16,
    "f16",
    bias_reciprocal_f16,
    bias_reciprocal_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `reciprocal(input + bias[column])` for every row-major matrix element.",
    bias_reciprocal_f32,
    tune_bias_reciprocal_f32,
    f32,
    "f32",
    bias_reciprocal_f32,
    bias_reciprocal_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `reciprocal(input + bias[column])` for every row-major matrix element.",
    bias_reciprocal_f64,
    tune_bias_reciprocal_f64,
    f64,
    "f64",
    bias_reciprocal_f64,
    bias_reciprocal_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `square(input + bias[column])` for every row-major matrix element.",
    bias_square_f16,
    tune_bias_square_f16,
    f16,
    "f16",
    bias_square_f16,
    bias_square_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `square(input + bias[column])` for every row-major matrix element.",
    bias_square_f32,
    tune_bias_square_f32,
    f32,
    "f32",
    bias_square_f32,
    bias_square_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `square(input + bias[column])` for every row-major matrix element.",
    bias_square_f64,
    tune_bias_square_f64,
    f64,
    "f64",
    bias_square_f64,
    bias_square_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `ceil(input + bias[column])` for every row-major matrix element.",
    bias_ceil_f16,
    tune_bias_ceil_f16,
    f16,
    "f16",
    bias_ceil_f16,
    bias_ceil_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `ceil(input + bias[column])` for every row-major matrix element.",
    bias_ceil_f32,
    tune_bias_ceil_f32,
    f32,
    "f32",
    bias_ceil_f32,
    bias_ceil_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `ceil(input + bias[column])` for every row-major matrix element.",
    bias_ceil_f64,
    tune_bias_ceil_f64,
    f64,
    "f64",
    bias_ceil_f64,
    bias_ceil_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `floor(input + bias[column])` for every row-major matrix element.",
    bias_floor_f16,
    tune_bias_floor_f16,
    f16,
    "f16",
    bias_floor_f16,
    bias_floor_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `floor(input + bias[column])` for every row-major matrix element.",
    bias_floor_f32,
    tune_bias_floor_f32,
    f32,
    "f32",
    bias_floor_f32,
    bias_floor_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `floor(input + bias[column])` for every row-major matrix element.",
    bias_floor_f64,
    tune_bias_floor_f64,
    f64,
    "f64",
    bias_floor_f64,
    bias_floor_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `trunc(input + bias[column])` for every row-major matrix element.",
    bias_trunc_f16,
    tune_bias_trunc_f16,
    f16,
    "f16",
    bias_trunc_f16,
    bias_trunc_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `trunc(input + bias[column])` for every row-major matrix element.",
    bias_trunc_f32,
    tune_bias_trunc_f32,
    f32,
    "f32",
    bias_trunc_f32,
    bias_trunc_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `trunc(input + bias[column])` for every row-major matrix element.",
    bias_trunc_f64,
    tune_bias_trunc_f64,
    f64,
    "f64",
    bias_trunc_f64,
    bias_trunc_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `frac(input + bias[column])` for every row-major matrix element.",
    bias_frac_f16,
    tune_bias_frac_f16,
    f16,
    "f16",
    bias_frac_f16,
    bias_frac_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `frac(input + bias[column])` for every row-major matrix element.",
    bias_frac_f32,
    tune_bias_frac_f32,
    f32,
    "f32",
    bias_frac_f32,
    bias_frac_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `frac(input + bias[column])` for every row-major matrix element.",
    bias_frac_f64,
    tune_bias_frac_f64,
    f64,
    "f64",
    bias_frac_f64,
    bias_frac_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `deg2rad(input + bias[column])` for every row-major matrix element.",
    bias_deg2rad_f16,
    tune_bias_deg2rad_f16,
    f16,
    "f16",
    bias_deg2rad_f16,
    bias_deg2rad_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `deg2rad(input + bias[column])` for every row-major matrix element.",
    bias_deg2rad_f32,
    tune_bias_deg2rad_f32,
    f32,
    "f32",
    bias_deg2rad_f32,
    bias_deg2rad_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `deg2rad(input + bias[column])` for every row-major matrix element.",
    bias_deg2rad_f64,
    tune_bias_deg2rad_f64,
    f64,
    "f64",
    bias_deg2rad_f64,
    bias_deg2rad_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `rad2deg(input + bias[column])` for every row-major matrix element.",
    bias_rad2deg_f16,
    tune_bias_rad2deg_f16,
    f16,
    "f16",
    bias_rad2deg_f16,
    bias_rad2deg_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `rad2deg(input + bias[column])` for every row-major matrix element.",
    bias_rad2deg_f32,
    tune_bias_rad2deg_f32,
    f32,
    "f32",
    bias_rad2deg_f32,
    bias_rad2deg_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `rad2deg(input + bias[column])` for every row-major matrix element.",
    bias_rad2deg_f64,
    tune_bias_rad2deg_f64,
    f64,
    "f64",
    bias_rad2deg_f64,
    bias_rad2deg_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `sin(input + bias[column])` for every row-major matrix element.",
    bias_sin_f16,
    tune_bias_sin_f16,
    f16,
    "f16",
    bias_sin_f16,
    bias_sin_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `sin(input + bias[column])` for every row-major matrix element.",
    bias_sin_f32,
    tune_bias_sin_f32,
    f32,
    "f32",
    bias_sin_f32,
    bias_sin_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `sin(input + bias[column])` for every row-major matrix element.",
    bias_sin_f64,
    tune_bias_sin_f64,
    f64,
    "f64",
    bias_sin_f64,
    bias_sin_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `cos(input + bias[column])` for every row-major matrix element.",
    bias_cos_f16,
    tune_bias_cos_f16,
    f16,
    "f16",
    bias_cos_f16,
    bias_cos_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `cos(input + bias[column])` for every row-major matrix element.",
    bias_cos_f32,
    tune_bias_cos_f32,
    f32,
    "f32",
    bias_cos_f32,
    bias_cos_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `cos(input + bias[column])` for every row-major matrix element.",
    bias_cos_f64,
    tune_bias_cos_f64,
    f64,
    "f64",
    bias_cos_f64,
    bias_cos_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `tan(input + bias[column])` for every row-major matrix element.",
    bias_tan_f16,
    tune_bias_tan_f16,
    f16,
    "f16",
    bias_tan_f16,
    bias_tan_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `tan(input + bias[column])` for every row-major matrix element.",
    bias_tan_f32,
    tune_bias_tan_f32,
    f32,
    "f32",
    bias_tan_f32,
    bias_tan_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `tan(input + bias[column])` for every row-major matrix element.",
    bias_tan_f64,
    tune_bias_tan_f64,
    f64,
    "f64",
    bias_tan_f64,
    bias_tan_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `asin(input + bias[column])` for every row-major matrix element.",
    bias_asin_f16,
    tune_bias_asin_f16,
    f16,
    "f16",
    bias_asin_f16,
    bias_asin_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `asin(input + bias[column])` for every row-major matrix element.",
    bias_asin_f32,
    tune_bias_asin_f32,
    f32,
    "f32",
    bias_asin_f32,
    bias_asin_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `asin(input + bias[column])` for every row-major matrix element.",
    bias_asin_f64,
    tune_bias_asin_f64,
    f64,
    "f64",
    bias_asin_f64,
    bias_asin_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `acos(input + bias[column])` for every row-major matrix element.",
    bias_acos_f16,
    tune_bias_acos_f16,
    f16,
    "f16",
    bias_acos_f16,
    bias_acos_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `acos(input + bias[column])` for every row-major matrix element.",
    bias_acos_f32,
    tune_bias_acos_f32,
    f32,
    "f32",
    bias_acos_f32,
    bias_acos_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `acos(input + bias[column])` for every row-major matrix element.",
    bias_acos_f64,
    tune_bias_acos_f64,
    f64,
    "f64",
    bias_acos_f64,
    bias_acos_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `atan(input + bias[column])` for every row-major matrix element.",
    bias_atan_f16,
    tune_bias_atan_f16,
    f16,
    "f16",
    bias_atan_f16,
    bias_atan_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `atan(input + bias[column])` for every row-major matrix element.",
    bias_atan_f32,
    tune_bias_atan_f32,
    f32,
    "f32",
    bias_atan_f32,
    bias_atan_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `atan(input + bias[column])` for every row-major matrix element.",
    bias_atan_f64,
    tune_bias_atan_f64,
    f64,
    "f64",
    bias_atan_f64,
    bias_atan_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `sinh(input + bias[column])` for every row-major matrix element.",
    bias_sinh_f16,
    tune_bias_sinh_f16,
    f16,
    "f16",
    bias_sinh_f16,
    bias_sinh_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `sinh(input + bias[column])` for every row-major matrix element.",
    bias_sinh_f32,
    tune_bias_sinh_f32,
    f32,
    "f32",
    bias_sinh_f32,
    bias_sinh_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `sinh(input + bias[column])` for every row-major matrix element.",
    bias_sinh_f64,
    tune_bias_sinh_f64,
    f64,
    "f64",
    bias_sinh_f64,
    bias_sinh_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `cosh(input + bias[column])` for every row-major matrix element.",
    bias_cosh_f16,
    tune_bias_cosh_f16,
    f16,
    "f16",
    bias_cosh_f16,
    bias_cosh_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `cosh(input + bias[column])` for every row-major matrix element.",
    bias_cosh_f32,
    tune_bias_cosh_f32,
    f32,
    "f32",
    bias_cosh_f32,
    bias_cosh_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `cosh(input + bias[column])` for every row-major matrix element.",
    bias_cosh_f64,
    tune_bias_cosh_f64,
    f64,
    "f64",
    bias_cosh_f64,
    bias_cosh_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `asinh(input + bias[column])` for every row-major matrix element.",
    bias_asinh_f16,
    tune_bias_asinh_f16,
    f16,
    "f16",
    bias_asinh_f16,
    bias_asinh_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `asinh(input + bias[column])` for every row-major matrix element.",
    bias_asinh_f32,
    tune_bias_asinh_f32,
    f32,
    "f32",
    bias_asinh_f32,
    bias_asinh_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `asinh(input + bias[column])` for every row-major matrix element.",
    bias_asinh_f64,
    tune_bias_asinh_f64,
    f64,
    "f64",
    bias_asinh_f64,
    bias_asinh_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `acosh(input + bias[column])` for every row-major matrix element.",
    bias_acosh_f16,
    tune_bias_acosh_f16,
    f16,
    "f16",
    bias_acosh_f16,
    bias_acosh_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `acosh(input + bias[column])` for every row-major matrix element.",
    bias_acosh_f32,
    tune_bias_acosh_f32,
    f32,
    "f32",
    bias_acosh_f32,
    bias_acosh_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `acosh(input + bias[column])` for every row-major matrix element.",
    bias_acosh_f64,
    tune_bias_acosh_f64,
    f64,
    "f64",
    bias_acosh_f64,
    bias_acosh_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `atanh(input + bias[column])` for every row-major matrix element.",
    bias_atanh_f16,
    tune_bias_atanh_f16,
    f16,
    "f16",
    bias_atanh_f16,
    bias_atanh_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `atanh(input + bias[column])` for every row-major matrix element.",
    bias_atanh_f32,
    tune_bias_atanh_f32,
    f32,
    "f32",
    bias_atanh_f32,
    bias_atanh_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `atanh(input + bias[column])` for every row-major matrix element.",
    bias_atanh_f64,
    tune_bias_atanh_f64,
    f64,
    "f64",
    bias_atanh_f64,
    bias_atanh_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `tanh(input + bias[column])` for every row-major matrix element.",
    bias_tanh_f16,
    tune_bias_tanh_f16,
    f16,
    "f16",
    bias_tanh_f16,
    bias_tanh_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `tanh(input + bias[column])` for every row-major matrix element.",
    bias_tanh_f32,
    tune_bias_tanh_f32,
    f32,
    "f32",
    bias_tanh_f32,
    bias_tanh_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `tanh(input + bias[column])` for every row-major matrix element.",
    bias_tanh_f64,
    tune_bias_tanh_f64,
    f64,
    "f64",
    bias_tanh_f64,
    bias_tanh_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `exp(input + bias[column])` for every row-major matrix element.",
    bias_exp_f16,
    tune_bias_exp_f16,
    f16,
    "f16",
    bias_exp_f16,
    bias_exp_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `exp(input + bias[column])` for every row-major matrix element.",
    bias_exp_f32,
    tune_bias_exp_f32,
    f32,
    "f32",
    bias_exp_f32,
    bias_exp_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `exp(input + bias[column])` for every row-major matrix element.",
    bias_exp_f64,
    tune_bias_exp_f64,
    f64,
    "f64",
    bias_exp_f64,
    bias_exp_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `exp2(input + bias[column])` for every row-major matrix element.",
    bias_exp2_f16,
    tune_bias_exp2_f16,
    f16,
    "f16",
    bias_exp2_f16,
    bias_exp2_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `exp2(input + bias[column])` for every row-major matrix element.",
    bias_exp2_f32,
    tune_bias_exp2_f32,
    f32,
    "f32",
    bias_exp2_f32,
    bias_exp2_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `exp2(input + bias[column])` for every row-major matrix element.",
    bias_exp2_f64,
    tune_bias_exp2_f64,
    f64,
    "f64",
    bias_exp2_f64,
    bias_exp2_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `expm1(input + bias[column])` for every row-major matrix element.",
    bias_expm1_f16,
    tune_bias_expm1_f16,
    f16,
    "f16",
    bias_expm1_f16,
    bias_expm1_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `expm1(input + bias[column])` for every row-major matrix element.",
    bias_expm1_f32,
    tune_bias_expm1_f32,
    f32,
    "f32",
    bias_expm1_f32,
    bias_expm1_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `expm1(input + bias[column])` for every row-major matrix element.",
    bias_expm1_f64,
    tune_bias_expm1_f64,
    f64,
    "f64",
    bias_expm1_f64,
    bias_expm1_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `log(input + bias[column])` for every row-major matrix element.",
    bias_log_f16,
    tune_bias_log_f16,
    f16,
    "f16",
    bias_log_f16,
    bias_log_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `log(input + bias[column])` for every row-major matrix element.",
    bias_log_f32,
    tune_bias_log_f32,
    f32,
    "f32",
    bias_log_f32,
    bias_log_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `log(input + bias[column])` for every row-major matrix element.",
    bias_log_f64,
    tune_bias_log_f64,
    f64,
    "f64",
    bias_log_f64,
    bias_log_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `log2(input + bias[column])` for every row-major matrix element.",
    bias_log2_f16,
    tune_bias_log2_f16,
    f16,
    "f16",
    bias_log2_f16,
    bias_log2_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `log2(input + bias[column])` for every row-major matrix element.",
    bias_log2_f32,
    tune_bias_log2_f32,
    f32,
    "f32",
    bias_log2_f32,
    bias_log2_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `log2(input + bias[column])` for every row-major matrix element.",
    bias_log2_f64,
    tune_bias_log2_f64,
    f64,
    "f64",
    bias_log2_f64,
    bias_log2_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `log10(input + bias[column])` for every row-major matrix element.",
    bias_log10_f16,
    tune_bias_log10_f16,
    f16,
    "f16",
    bias_log10_f16,
    bias_log10_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `log10(input + bias[column])` for every row-major matrix element.",
    bias_log10_f32,
    tune_bias_log10_f32,
    f32,
    "f32",
    bias_log10_f32,
    bias_log10_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `log10(input + bias[column])` for every row-major matrix element.",
    bias_log10_f64,
    tune_bias_log10_f64,
    f64,
    "f64",
    bias_log10_f64,
    bias_log10_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `log1p(input + bias[column])` for every row-major matrix element.",
    bias_log1p_f16,
    tune_bias_log1p_f16,
    f16,
    "f16",
    bias_log1p_f16,
    bias_log1p_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `log1p(input + bias[column])` for every row-major matrix element.",
    bias_log1p_f32,
    tune_bias_log1p_f32,
    f32,
    "f32",
    bias_log1p_f32,
    bias_log1p_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `log1p(input + bias[column])` for every row-major matrix element.",
    bias_log1p_f64,
    tune_bias_log1p_f64,
    f64,
    "f64",
    bias_log1p_f64,
    bias_log1p_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `softplus(input + bias[column])` for every row-major matrix element.",
    bias_softplus_f16,
    tune_bias_softplus_f16,
    f16,
    "f16",
    bias_softplus_f16,
    bias_softplus_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `softplus(input + bias[column])` for every row-major matrix element.",
    bias_softplus_f32,
    tune_bias_softplus_f32,
    f32,
    "f32",
    bias_softplus_f32,
    bias_softplus_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `softplus(input + bias[column])` for every row-major matrix element.",
    bias_softplus_f64,
    tune_bias_softplus_f64,
    f64,
    "f64",
    bias_softplus_f64,
    bias_softplus_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `sqrt(input + bias[column])` for every row-major matrix element.",
    bias_sqrt_f16,
    tune_bias_sqrt_f16,
    f16,
    "f16",
    bias_sqrt_f16,
    bias_sqrt_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `sqrt(input + bias[column])` for every row-major matrix element.",
    bias_sqrt_f32,
    tune_bias_sqrt_f32,
    f32,
    "f32",
    bias_sqrt_f32,
    bias_sqrt_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `sqrt(input + bias[column])` for every row-major matrix element.",
    bias_sqrt_f64,
    tune_bias_sqrt_f64,
    f64,
    "f64",
    bias_sqrt_f64,
    bias_sqrt_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `rsqrt(input + bias[column])` for every row-major matrix element.",
    bias_rsqrt_f16,
    tune_bias_rsqrt_f16,
    f16,
    "f16",
    bias_rsqrt_f16,
    bias_rsqrt_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `rsqrt(input + bias[column])` for every row-major matrix element.",
    bias_rsqrt_f32,
    tune_bias_rsqrt_f32,
    f32,
    "f32",
    bias_rsqrt_f32,
    bias_rsqrt_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `rsqrt(input + bias[column])` for every row-major matrix element.",
    bias_rsqrt_f64,
    tune_bias_rsqrt_f64,
    f64,
    "f64",
    bias_rsqrt_f64,
    bias_rsqrt_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `cbrt(input + bias[column])` for every row-major matrix element.",
    bias_cbrt_f16,
    tune_bias_cbrt_f16,
    f16,
    "f16",
    bias_cbrt_f16,
    bias_cbrt_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `cbrt(input + bias[column])` for every row-major matrix element.",
    bias_cbrt_f32,
    tune_bias_cbrt_f32,
    f32,
    "f32",
    bias_cbrt_f32,
    bias_cbrt_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `cbrt(input + bias[column])` for every row-major matrix element.",
    bias_cbrt_f64,
    tune_bias_cbrt_f64,
    f64,
    "f64",
    bias_cbrt_f64,
    bias_cbrt_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `abs(input + bias[column])` for every row-major matrix element.",
    bias_abs_f16,
    tune_bias_abs_f16,
    f16,
    "f16",
    bias_abs_f16,
    bias_abs_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `abs(input + bias[column])` for every row-major matrix element.",
    bias_abs_f32,
    tune_bias_abs_f32,
    f32,
    "f32",
    bias_abs_f32,
    bias_abs_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `abs(input + bias[column])` for every row-major matrix element.",
    bias_abs_f64,
    tune_bias_abs_f64,
    f64,
    "f64",
    bias_abs_f64,
    bias_abs_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `neg(input + bias[column])` for every row-major matrix element.",
    bias_neg_f16,
    tune_bias_neg_f16,
    f16,
    "f16",
    bias_neg_f16,
    bias_neg_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `neg(input + bias[column])` for every row-major matrix element.",
    bias_neg_f32,
    tune_bias_neg_f32,
    f32,
    "f32",
    bias_neg_f32,
    bias_neg_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `neg(input + bias[column])` for every row-major matrix element.",
    bias_neg_f64,
    tune_bias_neg_f64,
    f64,
    "f64",
    bias_neg_f64,
    bias_neg_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `min(input, bias[column])` for every row-major matrix element.",
    bias_min_f16,
    tune_bias_min_f16,
    f16,
    "f16",
    bias_min_f16,
    bias_min_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `min(input, bias[column])` for every row-major matrix element.",
    bias_min_f32,
    tune_bias_min_f32,
    f32,
    "f32",
    bias_min_f32,
    bias_min_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `min(input, bias[column])` for every row-major matrix element.",
    bias_min_f64,
    tune_bias_min_f64,
    f64,
    "f64",
    bias_min_f64,
    bias_min_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_bias_fused_fn!(
    "Computes `max(input, bias[column])` for every row-major matrix element.",
    bias_max_f16,
    tune_bias_max_f16,
    f16,
    "f16",
    bias_max_f16,
    bias_max_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_bias_fused_fn!(
    "Computes `max(input, bias[column])` for every row-major matrix element.",
    bias_max_f32,
    tune_bias_max_f32,
    f32,
    "f32",
    bias_max_f32,
    bias_max_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_bias_fused_fn!(
    "Computes `max(input, bias[column])` for every row-major matrix element.",
    bias_max_f64,
    tune_bias_max_f64,
    f64,
    "f64",
    bias_max_f64,
    bias_max_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_column2_fused_fn!(
    column_clamp_f16,
    tune_column_clamp_f16,
    f16,
    "f16",
    column_clamp_f16,
    column_clamp_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_column2_fused_fn!(
    column_clamp_f32,
    tune_column_clamp_f32,
    f32,
    "f32",
    column_clamp_f32,
    column_clamp_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_column2_fused_fn!(
    column_clamp_f64,
    tune_column_clamp_f64,
    f64,
    "f64",
    column_clamp_f64,
    column_clamp_f64_tuned
);

#[cfg(feature = "dtype_f16")]
autotuned_column2_fused_fn!(
    column_lerp_f16,
    tune_column_lerp_f16,
    f16,
    "f16",
    column_lerp_f16,
    column_lerp_f16_tuned
);

#[cfg(feature = "dtype_f32")]
autotuned_column2_fused_fn!(
    column_lerp_f32,
    tune_column_lerp_f32,
    f32,
    "f32",
    column_lerp_f32,
    column_lerp_f32_tuned
);

#[cfg(feature = "dtype_f64")]
autotuned_column2_fused_fn!(
    column_lerp_f64,
    tune_column_lerp_f64,
    f64,
    "f64",
    column_lerp_f64,
    column_lerp_f64_tuned
);
