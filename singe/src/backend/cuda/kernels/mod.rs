//! The CUDA kernel functions that `backend_kernels!` registers, named after their operations.

mod activation;
mod binary;
mod elementwise;
mod neural;
mod reduction;
mod shape;
mod storage;

pub use self::{
    activation::{log_softmax, softmax},
    binary::{grouped_matmul, matmul, scaled_matmul},
    elementwise::elementwise,
    neural::{
        avg_pool2d, batch_norm, convolution2d, convolution3d, embedding, layer_norm, max_pool2d,
        paged_attention, paged_mla_attention, rms_norm, rotary_embedding,
        scaled_dot_product_attention,
    },
    reduction::{
        argmax, argmin, argsort, cumprod, cumsum, logsumexp, max, mean, min, product, sum, topk,
    },
    shape::{
        concat, contiguous, dequantize, index, index_put, index_select, index_write, quantize,
        repeat_kv, take,
    },
    storage::{assign, copy_to, random},
};
pub use crate::backend::cuda::eager::{
    fourier::{irfft, rfft},
    tensor_algebra::contraction,
};
