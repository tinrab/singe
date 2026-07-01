#[cfg(feature = "kernel-activation")]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f32/activation.rs"]
pub mod activation;
#[cfg(feature = "kernel-cast")]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f32/cast.rs"]
pub mod cast;
#[cfg(feature = "kernel-elementwise")]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f32/elementwise.rs"]
pub mod elementwise;
#[cfg(feature = "kernel-quantization")]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f32/quantization.rs"]
pub mod quantization;
#[cfg(feature = "kernel-reduction")]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f32/reduction.rs"]
pub mod reduction;
#[cfg(feature = "kernel-scalar")]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f32/scalar.rs"]
pub mod scalar;
#[cfg(feature = "kernel-softmax")]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f32/softmax.rs"]
pub mod softmax;
#[cfg(feature = "kernel-unary")]
#[doc(hidden)]
#[path = "../../../../-generated/cutile/f32/unary.rs"]
pub mod unary;

#[cfg(feature = "kernel-embedding")]
pub mod embedding;
#[cfg(feature = "kernel-gather")]
pub mod gather;
#[cfg(feature = "kernel-normalization")]
pub mod normalization;
#[cfg(feature = "kernel-pooling")]
pub mod pooling;
#[cfg(feature = "kernel-positional")]
pub mod positional;
#[cfg(feature = "kernel-shape")]
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
