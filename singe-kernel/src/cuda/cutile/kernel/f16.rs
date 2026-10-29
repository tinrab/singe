#[cfg(all(feature = "dtype_f16", feature = "kernel_activation"))]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f16/activation.rs"]
pub mod activation;

#[cfg(all(
    any(feature = "dtype_f16", feature = "dtype_bf16"),
    feature = "kernel_cast"
))]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f16/cast.rs"]
pub mod cast;

#[cfg(all(feature = "dtype_f16", feature = "kernel_elementwise"))]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f16/elementwise.rs"]
pub mod elementwise;

#[cfg(all(feature = "dtype_f16", feature = "kernel_quantization"))]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f16/quantization.rs"]
pub mod quantization;

#[cfg(all(feature = "dtype_f16", feature = "kernel_reduction"))]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f16/reduction.rs"]
pub mod reduction;

#[cfg(all(feature = "dtype_f16", feature = "kernel_scalar"))]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f16/scalar.rs"]
pub mod scalar;

#[cfg(all(feature = "dtype_f16", feature = "kernel_softmax"))]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f16/softmax.rs"]
pub mod softmax;

#[cfg(all(feature = "dtype_f16", feature = "kernel_unary"))]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f16/unary.rs"]
pub mod unary;

#[cfg(all(feature = "dtype_f16", feature = "kernel_embedding"))]
pub mod embedding;
#[cfg(all(feature = "dtype_f16", feature = "kernel_gather"))]
pub mod gather;
#[cfg(feature = "kernel_normalization")]
pub mod normalization;
#[cfg(all(feature = "dtype_f16", feature = "kernel_pooling"))]
pub mod pooling;
#[cfg(all(feature = "dtype_f16", feature = "kernel_positional"))]
pub mod positional;
#[cfg(all(feature = "dtype_f16", feature = "kernel_shape"))]
pub mod shape;

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_cast",
    feature = "kernel_elementwise",
    feature = "kernel_quantization",
    feature = "kernel_reduction",
    feature = "kernel_scalar",
    feature = "kernel_softmax",
    feature = "kernel_unary",
))]
#[doc(hidden)]
pub mod utility;
