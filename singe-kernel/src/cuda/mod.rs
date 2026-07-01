//! CUDA kernels grouped by operation family.

#[cfg(all(
    feature = "kernel-activation",
    any(feature = "dtype-f32", feature = "dtype-f16", feature = "dtype-f64")
))]
pub mod activation;
#[cfg(feature = "kernel-attention")]
pub mod attention;
#[cfg(feature = "kernel-audio")]
pub mod audio;
#[cfg(all(
    feature = "kernel-cast",
    any(
        feature = "dtype-f32",
        feature = "dtype-f16",
        feature = "dtype-f64",
        feature = "dtype-i8",
        feature = "dtype-i32",
        feature = "dtype-i64",
        feature = "dtype-u8",
        feature = "dtype-u32",
        feature = "dtype-u64",
    )
))]
pub mod cast;
#[cfg(feature = "kernel-conv")]
pub mod conv;
#[cfg(all(
    feature = "kernel-creation",
    any(
        feature = "dtype-f32",
        feature = "dtype-f16",
        feature = "dtype-f64",
        feature = "dtype-i8",
        feature = "dtype-i32",
        feature = "dtype-i64",
        feature = "dtype-u8",
        feature = "dtype-u32",
        feature = "dtype-u64",
    )
))]
pub mod creation;
#[cfg(feature = "cutile")]
pub mod cutile;
#[cfg(all(
    feature = "kernel-elementwise",
    any(
        feature = "dtype-f32",
        feature = "dtype-f16",
        feature = "dtype-f64",
        feature = "dtype-i8",
        feature = "dtype-i32",
        feature = "dtype-i64",
        feature = "dtype-u8",
        feature = "dtype-u32",
        feature = "dtype-u64",
    )
))]
pub mod elementwise;
#[cfg(all(
    feature = "kernel-embedding",
    any(feature = "dtype-f32", feature = "dtype-f16", feature = "dtype-f64")
))]
pub mod embedding;
#[cfg(feature = "kernel-fft")]
pub mod fft;
#[cfg(feature = "kernel-fused")]
pub mod fused;
#[cfg(all(
    feature = "kernel-gather",
    any(
        feature = "dtype-f32",
        feature = "dtype-f16",
        feature = "dtype-f64",
        feature = "dtype-i8",
        feature = "dtype-i32",
        feature = "dtype-i64",
        feature = "dtype-u8",
        feature = "dtype-u32",
        feature = "dtype-u64",
    )
))]
pub mod gather;
#[cfg(any(
    feature = "kernel-activation",
    feature = "kernel-attention",
    feature = "kernel-audio",
    feature = "kernel-cast",
    feature = "kernel-conv",
    feature = "kernel-creation",
    feature = "kernel-elementwise",
    feature = "kernel-embedding",
    feature = "kernel-fft",
    feature = "kernel-fused",
    feature = "kernel-gather",
    feature = "kernel-matmul",
    feature = "kernel-moe",
    feature = "kernel-mask",
    feature = "kernel-normalization",
    feature = "kernel-pooling",
    feature = "kernel-positional",
    feature = "kernel-quantization",
    feature = "kernel-reduction",
    feature = "kernel-recurrent",
    feature = "kernel-scalar",
    feature = "kernel-shape",
    feature = "kernel-softmax",
    feature = "kernel-unary",
))]
mod interop;
#[cfg(feature = "kernel-mask")]
pub mod mask;
#[cfg(feature = "kernel-matmul")]
pub mod matmul;
#[cfg(feature = "kernel-moe")]
pub mod moe;
#[cfg(feature = "kernel-normalization")]
pub mod normalization;
#[cfg(all(
    feature = "kernel-pooling",
    any(feature = "dtype-f32", feature = "dtype-f16", feature = "dtype-f64")
))]
pub mod pooling;
#[cfg(feature = "kernel-positional")]
pub mod positional;
#[cfg(feature = "kernel-quantization")]
pub mod quantization;
#[cfg(feature = "kernel-recurrent")]
pub mod recurrent;
#[cfg(all(
    feature = "kernel-reduction",
    any(feature = "dtype-f32", feature = "dtype-f16", feature = "dtype-f64")
))]
pub mod reduction;
#[cfg(all(
    feature = "kernel-scalar",
    any(feature = "dtype-f32", feature = "dtype-f16", feature = "dtype-f64")
))]
pub mod scalar;
#[cfg(all(
    feature = "kernel-shape",
    any(
        feature = "dtype-f32",
        feature = "dtype-f16",
        feature = "dtype-f64",
        feature = "dtype-i8",
        feature = "dtype-i32",
        feature = "dtype-i64",
        feature = "dtype-u8",
        feature = "dtype-u32",
        feature = "dtype-u64",
    )
))]
pub mod shape;
#[cfg(all(
    feature = "kernel-softmax",
    any(feature = "dtype-f32", feature = "dtype-f16", feature = "dtype-f64")
))]
pub mod softmax;
#[cfg(all(
    feature = "kernel-unary",
    any(
        feature = "dtype-f32",
        feature = "dtype-f16",
        feature = "dtype-f64",
        feature = "dtype-i8",
        feature = "dtype-i32",
        feature = "dtype-i64",
        feature = "dtype-u8",
        feature = "dtype-u32",
        feature = "dtype-u64",
    )
))]
pub mod unary;

pub(crate) mod utility;
