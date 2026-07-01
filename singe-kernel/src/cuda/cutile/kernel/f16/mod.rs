#[cfg(all(feature = "dtype-f16", feature = "kernel-activation"))]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f16/activation.rs"]
pub mod activation;
#[cfg(all(
    any(feature = "dtype-f16", feature = "dtype-bf16"),
    feature = "kernel-cast"
))]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f16/cast.rs"]
pub mod cast;
#[cfg(all(feature = "dtype-f16", feature = "kernel-elementwise"))]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f16/elementwise.rs"]
pub mod elementwise;
#[cfg(all(feature = "dtype-f16", feature = "kernel-quantization"))]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f16/quantization.rs"]
pub mod quantization;
#[cfg(all(feature = "dtype-f16", feature = "kernel-reduction"))]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f16/reduction.rs"]
pub mod reduction;
#[cfg(all(feature = "dtype-f16", feature = "kernel-scalar"))]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f16/scalar.rs"]
pub mod scalar;
#[cfg(all(feature = "dtype-f16", feature = "kernel-softmax"))]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f16/softmax.rs"]
pub mod softmax;
#[cfg(all(feature = "dtype-f16", feature = "kernel-unary"))]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f16/unary.rs"]
pub mod unary;

#[cfg(all(feature = "dtype-f16", feature = "kernel-embedding"))]
pub mod embedding;
#[cfg(all(feature = "dtype-f16", feature = "kernel-gather"))]
pub mod gather;
#[cfg(feature = "kernel-normalization")]
pub mod normalization;
#[cfg(all(feature = "dtype-f16", feature = "kernel-pooling"))]
pub mod pooling;
#[cfg(all(feature = "dtype-f16", feature = "kernel-positional"))]
pub mod positional;
#[cfg(all(feature = "dtype-f16", feature = "kernel-shape"))]
pub mod shape;

#[cfg(any(
    feature = "kernel-activation",
    feature = "kernel-cast",
    feature = "kernel-elementwise",
    feature = "kernel-quantization",
    feature = "kernel-reduction",
    feature = "kernel-scalar",
    feature = "kernel-softmax",
    feature = "kernel-unary",
))]
#[doc(hidden)]
pub mod utility;
