mod adapter;
pub mod operation;
mod runtime;

/// Raw cuTile device address accepted by the public pointer entry points.
pub use ::cutile::tile_kernel::DevicePointer;

pub use runtime::tile_kernel::{TileKernel, clear_context_kernel_cache};

#[cfg(all(
    feature = "kernel_activation",
    any(feature = "dtype_f32", feature = "dtype_f16", feature = "dtype_f64")
))]
pub mod activation;
#[cfg(feature = "kernel_attention")]
pub mod attention;
#[cfg(feature = "kernel_audio")]
pub mod audio;
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
pub mod kernel;
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

pub(crate) mod utility;

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
use std::sync::Arc;

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
use cutile::{
    cuda_async::device_operation::DeviceOp,
    cuda_core::Stream,
    error::{Error as CutileError, Result as CutileResult},
};

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
#[allow(dead_code, reason = "only used by some kernel feature combinations")]
pub(crate) trait DeviceOpExt: DeviceOp + Sized {
    fn enqueue_on(self, stream: &Arc<Stream>) -> CutileResult<<Self as DeviceOp>::Output> {
        // SAFETY: callers own stream ordering and keep raw device allocations alive until the enqueued work has completed.
        unsafe { self.async_on(stream) }.map_err(CutileError::from)
    }
}

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
impl<T> DeviceOpExt for T where T: DeviceOp {}

#[cfg(all(feature = "kernel_fused", feature = "dtype_f16"))]
pub mod transformer;
