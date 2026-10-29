//! Device kernels written in the cuTile DSL.
//!
//! # Safety contract
//!
//! Each `#[cutile::module]` in this tree is device code. `unsafe` blocks inside
//! those modules call raw tile loads, stores and intrinsics whose soundness comes
//! from the launch, not from the kernel body: the host wrapper in
//! `crate::cuda::cutile` validates tensor shapes, strides, lengths and masks before
//! launch, binds the CUDA context, and keeps every device buffer borrowed until the
//! submission completes. Offsets and masks computed inside a kernel only address
//! elements inside those validated extents. Individual device `unsafe` blocks
//! therefore carry no per-site `SAFETY` comment; host-side `unsafe` code does.

#![allow(
    clippy::approx_constant,
    reason = "constants match reference implementations bit for bit"
)]

pub mod tiled;
pub mod utility;

#[cfg(feature = "kernel_normalization")]
pub mod normalization;

#[cfg(any(
    feature = "kernel_cast",
    feature = "kernel_creation",
    feature = "kernel_elementwise",
    feature = "kernel_gather",
    feature = "kernel_matmul",
    feature = "kernel_moe",
    feature = "kernel_mask",
    feature = "kernel_positional",
    feature = "kernel_quantization",
    feature = "kernel_reduction",
    feature = "kernel_recurrent",
    feature = "kernel_scalar",
    feature = "kernel_shape",
    feature = "kernel_softmax",
    feature = "kernel_unary",
))]
pub mod common;

#[cfg(feature = "kernel_attention")]
pub mod attention;
#[cfg(feature = "kernel_audio")]
pub mod audio;
#[cfg(feature = "kernel_conv")]
pub mod conv;
#[cfg(feature = "kernel_fft")]
pub mod fft;
#[cfg(feature = "kernel_fused")]
pub mod fused;
#[cfg(all(feature = "kernel_fused", feature = "kernel_quantization"))]
pub mod fused_quantization;
#[cfg(feature = "kernel_matmul")]
pub mod matmul;
#[cfg(feature = "kernel_moe")]
pub mod moe;
#[cfg(feature = "kernel_moe")]
pub mod moe_routing;
#[cfg(feature = "kernel_positional")]
pub mod positional_structured;
#[cfg(feature = "kernel_recurrent")]
pub mod recurrent;

#[cfg(all(
    feature = "dtype_f32",
    any(
        feature = "kernel_activation",
        feature = "kernel_attention",
        feature = "kernel_audio",
        feature = "kernel_cast",
        feature = "kernel_conv",
        feature = "kernel_creation",
        feature = "kernel_elementwise",
        feature = "kernel_embedding",
        feature = "kernel_fft",
        feature = "kernel_fused",
        feature = "kernel_gather",
        feature = "kernel_matmul",
        feature = "kernel_moe",
        feature = "kernel_normalization",
        feature = "kernel_pooling",
        feature = "kernel_positional",
        feature = "kernel_quantization",
        feature = "kernel_reduction",
        feature = "kernel_scalar",
        feature = "kernel_shape",
        feature = "kernel_softmax",
        feature = "kernel_unary",
    )
))]
pub mod f32;

#[cfg(all(
    any(feature = "dtype_f16", feature = "dtype_bf16"),
    any(
        feature = "kernel_activation",
        feature = "kernel_attention",
        feature = "kernel_cast",
        feature = "kernel_conv",
        feature = "kernel_creation",
        feature = "kernel_elementwise",
        feature = "kernel_embedding",
        feature = "kernel_fft",
        feature = "kernel_fused",
        feature = "kernel_gather",
        feature = "kernel_matmul",
        feature = "kernel_normalization",
        feature = "kernel_pooling",
        feature = "kernel_positional",
        feature = "kernel_quantization",
        feature = "kernel_reduction",
        feature = "kernel_scalar",
        feature = "kernel_shape",
        feature = "kernel_softmax",
        feature = "kernel_unary",
    )
))]
pub mod f16;

#[cfg(all(
    feature = "dtype_f64",
    any(
        feature = "kernel_activation",
        feature = "kernel_cast",
        feature = "kernel_creation",
        feature = "kernel_elementwise",
        feature = "kernel_embedding",
        feature = "kernel_fused",
        feature = "kernel_gather",
        feature = "kernel_matmul",
        feature = "kernel_normalization",
        feature = "kernel_pooling",
        feature = "kernel_positional",
        feature = "kernel_quantization",
        feature = "kernel_reduction",
        feature = "kernel_scalar",
        feature = "kernel_shape",
        feature = "kernel_softmax",
        feature = "kernel_unary",
    )
))]
pub mod f64;

#[cfg(all(feature = "kernel_fused", feature = "dtype_f16"))]
pub mod transformer;
