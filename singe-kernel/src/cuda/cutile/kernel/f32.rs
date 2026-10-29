#[cfg(feature = "kernel_activation")]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f32/activation.rs"]
pub mod activation;

#[cfg(feature = "kernel_cast")]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f32/cast.rs"]
pub mod cast;

#[cfg(feature = "kernel_elementwise")]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f32/elementwise.rs"]
pub mod elementwise;

#[cfg(feature = "kernel_quantization")]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f32/quantization.rs"]
pub mod quantization;

#[cfg(feature = "kernel_reduction")]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f32/reduction.rs"]
pub mod reduction;

#[cfg(feature = "kernel_scalar")]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f32/scalar.rs"]
pub mod scalar;

#[cfg(feature = "kernel_softmax")]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f32/softmax.rs"]
pub mod softmax;

#[cfg(feature = "kernel_unary")]
#[doc(hidden)]
#[path = "../../../-generated/cutile/f32/unary.rs"]
pub mod unary;

#[cfg(feature = "kernel_embedding")]
pub mod embedding;
#[cfg(feature = "kernel_gather")]
pub mod gather;
#[cfg(feature = "kernel_normalization")]
pub mod normalization;
#[cfg(feature = "kernel_pooling")]
pub mod pooling;
#[cfg(feature = "kernel_positional")]
pub mod positional;
#[cfg(feature = "kernel_shape")]
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
