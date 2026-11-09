//! The CPU kernel functions that `backend_kernels!` registers, named after their operations.

pub use crate::backend::cpu::{
    activation::{gelu, gelu_exact, log_softmax, relu, sigmoid, silu, softmax},
    binary::{
        add, atan2, div, equal, greater, greater_equal, grouped_matmul, less, less_equal,
        logical_and, logical_or, matmul, maximum, minimum, modulo, mul, not_equal, remainder, sub,
    },
    contraction::contraction,
    convolution::convolution2d,
    elementwise::{cast, clamp, dequantize, lerp, quantize, where_op},
    normalization::{
        batch_norm, layer_norm, paged_attention, paged_mla_attention, rms_norm, rotary_embedding,
        scaled_dot_product_attention,
    },
    pooling::{avg_pool2d, max_pool2d},
    reduction::{
        argmax, argmin, argsort, cumprod, cumsum, logsumexp, max, mean, min, product, sum,
    },
    shape::{
        add_scalar, concat, index, index_put, index_select, index_write, rdiv_scalar, repeat_kv,
        rsub_scalar, scale, sub_scalar,
    },
    storage::{assign, contiguous, copy_to},
    unary::{
        abs, acos, acosh, asin, asinh, atan, atanh, ceil, cos, cosh, erf, exp, floor, log,
        logical_not, neg, reciprocal, rsqrt, sin, sinh, sqrt, square, tan, tanh, truncate,
    },
};
use crate::{
    backend::{cpu::shape::select, registry::Operation},
    error::Result,
    tensor::Tensor,
};

/// Looks up rows of `weight` at `indices`.
pub fn embedding(weight: &Tensor, indices: &Tensor) -> Result<Tensor> {
    select(Operation::Embedding, weight, indices, 0)
}

/// Takes the slices of `input` along `dim` that `indices` name, with ONNX `Gather` semantics.
pub fn take(input: &Tensor, dim: i64, indices: &Tensor) -> Result<Tensor> {
    select(Operation::Take, input, indices, dim)
}
