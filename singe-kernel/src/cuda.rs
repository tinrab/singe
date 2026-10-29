//! CUDA kernels grouped by operation family.
//!
//! # Raw kernel execution
//!
//! Public unsafe launches enqueue work on the supplied stream.
//! Inputs and read-write values must be valid when read, with preceding producers ordered before use.
//! Keep the stream, plans, modules, allocations and scratch in their actual CUDA context alive until work completes on every path and throughout graph replay.
//! Writable ranges require exclusive access and must obey each operation's aliasing and device-index bounds.
//! Unrelated non-atomic writes must be disjoint.
//! Submission and capture recording do not establish output initialization.
//! Promote only the ranges actually produced after successful completion; padding, masked elements and stride gaps require their own initialization.
//! Autotuning applies these requirements to every candidate launch.

#[cfg(all(
    feature = "kernel_activation",
    any(feature = "dtype_f32", feature = "dtype_f16", feature = "dtype_f64")
))]
pub mod activation;
#[cfg(feature = "kernel_attention")]
pub mod attention;
#[cfg(feature = "kernel_audio")]
pub mod audio;
pub mod autotune;
#[cfg(all(
    feature = "kernel_cast",
    any(
        feature = "dtype_f32",
        feature = "dtype_f16",
        feature = "dtype_f64",
        feature = "dtype_i8",
        feature = "dtype_i32",
        feature = "dtype_i64",
        feature = "dtype_u8",
        feature = "dtype_u32",
        feature = "dtype_u64",
    )
))]
pub mod cast;
#[cfg(feature = "kernel_conv")]
pub mod conv;
#[cfg(all(
    feature = "kernel_creation",
    any(
        feature = "dtype_f32",
        feature = "dtype_f16",
        feature = "dtype_f64",
        feature = "dtype_i8",
        feature = "dtype_i32",
        feature = "dtype_i64",
        feature = "dtype_u8",
        feature = "dtype_u32",
        feature = "dtype_u64",
    )
))]
pub mod creation;
#[cfg(feature = "cutile")]
pub mod cutile;
#[cfg(all(
    feature = "kernel_elementwise",
    any(
        feature = "dtype_f32",
        feature = "dtype_f16",
        feature = "dtype_f64",
        feature = "dtype_i8",
        feature = "dtype_i32",
        feature = "dtype_i64",
        feature = "dtype_u8",
        feature = "dtype_u32",
        feature = "dtype_u64",
    )
))]
pub mod elementwise;
#[cfg(all(
    feature = "kernel_embedding",
    any(feature = "dtype_f32", feature = "dtype_f16", feature = "dtype_f64")
))]
pub mod embedding;
#[cfg(feature = "kernel_fft")]
pub mod fft;
#[cfg(feature = "kernel_fused")]
pub mod fused;
#[cfg(all(
    feature = "kernel_gather",
    any(
        feature = "dtype_f32",
        feature = "dtype_f16",
        feature = "dtype_f64",
        feature = "dtype_i8",
        feature = "dtype_i32",
        feature = "dtype_i64",
        feature = "dtype_u8",
        feature = "dtype_u32",
        feature = "dtype_u64",
    )
))]
pub mod gather;
#[cfg(any(
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
    feature = "kernel_mask",
    feature = "kernel_normalization",
    feature = "kernel_pooling",
    feature = "kernel_positional",
    feature = "kernel_quantization",
    feature = "kernel_reduction",
    feature = "kernel_recurrent",
    feature = "kernel_scalar",
    feature = "kernel_shape",
    feature = "kernel_softmax",
    feature = "kernel_unary",
))]
mod interop;
#[cfg(feature = "kernel_mask")]
pub mod mask;
#[cfg(feature = "kernel_matmul")]
pub mod matmul;
#[cfg(feature = "kernel_moe")]
pub mod moe;
#[cfg(feature = "kernel_normalization")]
pub mod normalization;
#[cfg(all(
    feature = "kernel_pooling",
    any(feature = "dtype_f32", feature = "dtype_f16", feature = "dtype_f64")
))]
pub mod pooling;
#[cfg(feature = "kernel_positional")]
pub mod positional;
#[cfg(feature = "kernel_quantization")]
pub mod quantization;
#[cfg(feature = "kernel_recurrent")]
pub mod recurrent;
#[cfg(all(
    feature = "kernel_reduction",
    any(
        feature = "dtype_f32",
        feature = "dtype_f16",
        feature = "dtype_f64",
        feature = "dtype_u8",
        feature = "dtype_i8",
        feature = "dtype_u32",
        feature = "dtype_i32",
        feature = "dtype_u64",
        feature = "dtype_i64",
    )
))]
pub mod reduction;
#[cfg(all(
    feature = "kernel_scalar",
    any(
        feature = "dtype_f32",
        feature = "dtype_f16",
        feature = "dtype_f64",
        feature = "dtype_u8",
        feature = "dtype_i8",
        feature = "dtype_u32",
        feature = "dtype_i32",
        feature = "dtype_u64",
        feature = "dtype_i64",
    )
))]
pub mod scalar;
#[cfg(all(
    feature = "kernel_shape",
    any(
        feature = "dtype_f32",
        feature = "dtype_f16",
        feature = "dtype_f64",
        feature = "dtype_i8",
        feature = "dtype_i32",
        feature = "dtype_i64",
        feature = "dtype_u8",
        feature = "dtype_u32",
        feature = "dtype_u64",
    )
))]
pub mod shape;
#[cfg(all(
    feature = "kernel_softmax",
    any(feature = "dtype_f32", feature = "dtype_f16", feature = "dtype_f64")
))]
pub mod softmax;
#[cfg(all(
    feature = "kernel_unary",
    any(
        feature = "dtype_f32",
        feature = "dtype_f16",
        feature = "dtype_f64",
        feature = "dtype_i8",
        feature = "dtype_i32",
        feature = "dtype_i64",
        feature = "dtype_u8",
        feature = "dtype_u32",
        feature = "dtype_u64",
    )
))]
pub mod unary;

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_cast",
    feature = "kernel_elementwise",
    feature = "kernel_quantization",
    feature = "kernel_fused",
    feature = "kernel_scalar",
    feature = "kernel_unary",
))]
pub(crate) mod utility;

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_cast",
    feature = "kernel_elementwise",
    feature = "kernel_quantization",
    feature = "kernel_scalar",
    feature = "kernel_unary",
))]
pub use crate::{cuda::utility::VectorConfig, cuda::utility::VectorTuning};
