pub mod utility;

#[cfg(any(
    feature = "kernel-cast",
    feature = "kernel-creation",
    feature = "kernel-elementwise",
    feature = "kernel-gather",
    feature = "kernel-matmul",
    feature = "kernel-moe",
    feature = "kernel-mask",
    feature = "kernel-positional",
    feature = "kernel-quantization",
    feature = "kernel-reduction",
    feature = "kernel-recurrent",
    feature = "kernel-scalar",
    feature = "kernel-shape",
    feature = "kernel-softmax",
    feature = "kernel-unary",
))]
pub mod common;

#[cfg(feature = "kernel-attention")]
pub mod attention;
#[cfg(feature = "kernel-audio")]
pub mod audio;
#[cfg(feature = "kernel-conv")]
pub mod conv;
#[cfg(feature = "kernel-fft")]
pub mod fft;
#[cfg(feature = "kernel-fused")]
pub mod fused;
#[cfg(feature = "kernel-matmul")]
pub mod matmul;
#[cfg(feature = "kernel-moe")]
pub mod moe;
#[cfg(feature = "kernel-positional")]
pub mod positional_structured;
#[cfg(feature = "kernel-recurrent")]
pub mod recurrent;

#[cfg(all(
    feature = "dtype-f32",
    any(
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
        feature = "kernel-normalization",
        feature = "kernel-pooling",
        feature = "kernel-positional",
        feature = "kernel-quantization",
        feature = "kernel-reduction",
        feature = "kernel-scalar",
        feature = "kernel-shape",
        feature = "kernel-softmax",
        feature = "kernel-unary",
    )
))]
pub mod f32;

#[cfg(all(
    any(feature = "dtype-f16", feature = "dtype-bf16"),
    any(
        feature = "kernel-activation",
        feature = "kernel-attention",
        feature = "kernel-cast",
        feature = "kernel-conv",
        feature = "kernel-creation",
        feature = "kernel-elementwise",
        feature = "kernel-embedding",
        feature = "kernel-fft",
        feature = "kernel-fused",
        feature = "kernel-gather",
        feature = "kernel-matmul",
        feature = "kernel-normalization",
        feature = "kernel-pooling",
        feature = "kernel-positional",
        feature = "kernel-quantization",
        feature = "kernel-reduction",
        feature = "kernel-scalar",
        feature = "kernel-shape",
        feature = "kernel-softmax",
        feature = "kernel-unary",
    )
))]
pub mod f16;

#[cfg(all(
    feature = "dtype-f64",
    any(
        feature = "kernel-activation",
        feature = "kernel-cast",
        feature = "kernel-creation",
        feature = "kernel-elementwise",
        feature = "kernel-embedding",
        feature = "kernel-fused",
        feature = "kernel-gather",
        feature = "kernel-matmul",
        feature = "kernel-normalization",
        feature = "kernel-pooling",
        feature = "kernel-positional",
        feature = "kernel-quantization",
        feature = "kernel-reduction",
        feature = "kernel-scalar",
        feature = "kernel-shape",
        feature = "kernel-softmax",
        feature = "kernel-unary",
    )
))]
pub mod f64;
