#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
use singe_cuda::types::f16;
#[cfg(any(
    all(feature = "dtype_f16", feature = "dtype_u8"),
    all(feature = "dtype_f32", feature = "dtype_u8"),
    all(feature = "dtype_f64", feature = "dtype_u8")
))]
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(any(
    all(feature = "dtype_f16", feature = "dtype_u8"),
    all(feature = "dtype_f32", feature = "dtype_u8"),
    all(feature = "dtype_f64", feature = "dtype_u8")
))]
use crate::{
    cuda::{
        autotune::{TuneOptions, TuneReport},
        cutile,
        fused::{
            Fused1dConfig, Fused2dConfig, FusedTileTuning, fused_1d_conditional,
            fused_2d_conditional, tune_fused_1d_conditional, tune_fused_2d_conditional,
        },
    },
    error::Result,
};

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_mul_add_f16,
    tune_where_mul_add_f16,
    f16,
    "f16",
    where_mul_add_f16,
    where_mul_add_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_mul_add_f32,
    tune_where_mul_add_f32,
    f32,
    "f32",
    where_mul_add_f32,
    where_mul_add_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_mul_add_f64,
    tune_where_mul_add_f64,
    f64,
    "f64",
    where_mul_add_f64,
    where_mul_add_f64_tuned
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_mul_add_2d_f16,
    tune_where_mul_add_2d_f16,
    f16,
    "f16",
    where_mul_add_2d_f16,
    where_mul_add_2d_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_mul_add_2d_f32,
    tune_where_mul_add_2d_f32,
    f32,
    "f32",
    where_mul_add_2d_f32,
    where_mul_add_2d_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_mul_add_2d_f64,
    tune_where_mul_add_2d_f64,
    f64,
    "f64",
    where_mul_add_2d_f64,
    where_mul_add_2d_f64_tuned
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_add_f16,
    tune_where_add_f16,
    f16,
    "f16",
    where_add_f16,
    where_add_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_add_f32,
    tune_where_add_f32,
    f32,
    "f32",
    where_add_f32,
    where_add_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_add_f64,
    tune_where_add_f64,
    f64,
    "f64",
    where_add_f64,
    where_add_f64_tuned
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_add_2d_f16,
    tune_where_add_2d_f16,
    f16,
    "f16",
    where_add_2d_f16,
    where_add_2d_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_add_2d_f32,
    tune_where_add_2d_f32,
    f32,
    "f32",
    where_add_2d_f32,
    where_add_2d_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_add_2d_f64,
    tune_where_add_2d_f64,
    f64,
    "f64",
    where_add_2d_f64,
    where_add_2d_f64_tuned
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_sub_f16,
    tune_where_sub_f16,
    f16,
    "f16",
    where_sub_f16,
    where_sub_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_sub_f32,
    tune_where_sub_f32,
    f32,
    "f32",
    where_sub_f32,
    where_sub_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_sub_f64,
    tune_where_sub_f64,
    f64,
    "f64",
    where_sub_f64,
    where_sub_f64_tuned
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_sub_2d_f16,
    tune_where_sub_2d_f16,
    f16,
    "f16",
    where_sub_2d_f16,
    where_sub_2d_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_sub_2d_f32,
    tune_where_sub_2d_f32,
    f32,
    "f32",
    where_sub_2d_f32,
    where_sub_2d_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_sub_2d_f64,
    tune_where_sub_2d_f64,
    f64,
    "f64",
    where_sub_2d_f64,
    where_sub_2d_f64_tuned
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_mul_f16,
    tune_where_mul_f16,
    f16,
    "f16",
    where_mul_f16,
    where_mul_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_mul_f32,
    tune_where_mul_f32,
    f32,
    "f32",
    where_mul_f32,
    where_mul_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_mul_f64,
    tune_where_mul_f64,
    f64,
    "f64",
    where_mul_f64,
    where_mul_f64_tuned
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_mul_2d_f16,
    tune_where_mul_2d_f16,
    f16,
    "f16",
    where_mul_2d_f16,
    where_mul_2d_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_mul_2d_f32,
    tune_where_mul_2d_f32,
    f32,
    "f32",
    where_mul_2d_f32,
    where_mul_2d_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_mul_2d_f64,
    tune_where_mul_2d_f64,
    f64,
    "f64",
    where_mul_2d_f64,
    where_mul_2d_f64_tuned
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_div_f16,
    tune_where_div_f16,
    f16,
    "f16",
    where_div_f16,
    where_div_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_div_f32,
    tune_where_div_f32,
    f32,
    "f32",
    where_div_f32,
    where_div_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_div_f64,
    tune_where_div_f64,
    f64,
    "f64",
    where_div_f64,
    where_div_f64_tuned
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_div_2d_f16,
    tune_where_div_2d_f16,
    f16,
    "f16",
    where_div_2d_f16,
    where_div_2d_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_div_2d_f32,
    tune_where_div_2d_f32,
    f32,
    "f32",
    where_div_2d_f32,
    where_div_2d_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_div_2d_f64,
    tune_where_div_2d_f64,
    f64,
    "f64",
    where_div_2d_f64,
    where_div_2d_f64_tuned
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_min_f16,
    tune_where_min_f16,
    f16,
    "f16",
    where_min_f16,
    where_min_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_min_f32,
    tune_where_min_f32,
    f32,
    "f32",
    where_min_f32,
    where_min_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_min_f64,
    tune_where_min_f64,
    f64,
    "f64",
    where_min_f64,
    where_min_f64_tuned
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_min_2d_f16,
    tune_where_min_2d_f16,
    f16,
    "f16",
    where_min_2d_f16,
    where_min_2d_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_min_2d_f32,
    tune_where_min_2d_f32,
    f32,
    "f32",
    where_min_2d_f32,
    where_min_2d_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_min_2d_f64,
    tune_where_min_2d_f64,
    f64,
    "f64",
    where_min_2d_f64,
    where_min_2d_f64_tuned
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_max_f16,
    tune_where_max_f16,
    f16,
    "f16",
    where_max_f16,
    where_max_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_max_f32,
    tune_where_max_f32,
    f32,
    "f32",
    where_max_f32,
    where_max_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_fn!(
    where_max_f64,
    tune_where_max_f64,
    f64,
    "f64",
    where_max_f64,
    where_max_f64_tuned
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_max_2d_f16,
    tune_where_max_2d_f16,
    f16,
    "f16",
    where_max_2d_f16,
    where_max_2d_f16_tuned
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_max_2d_f32,
    tune_where_max_2d_f32,
    f32,
    "f32",
    where_max_2d_f32,
    where_max_2d_f32_tuned
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
autotuned_conditional_fused_2d_fn!(
    where_max_2d_f64,
    tune_where_max_2d_f64,
    f64,
    "f64",
    where_max_2d_f64,
    where_max_2d_f64_tuned
);
