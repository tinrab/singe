//! Inference-oriented fused operations that combine common model steps.

#[cfg(feature = "dtype-bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceSlice, DeviceSliceMut},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::positional::QkRotaryEmbeddingConfig,
    cuda::{
        interop::{borrowed_stream, input_pointer, output_pointer},
        positional,
        utility::{ensure_binary_lengths, ensure_unary_lengths},
    },
    error::{Error, Result},
    utility::{
        checked_element_count, checked_rank4_len, ensure_len, ensure_rank2_reach,
        ensure_rank4_reach,
    },
};

#[derive(Debug, Clone, Copy)]
pub struct RopeQkCacheUpdateConfig {
    pub rotary: QkRotaryEmbeddingConfig,
    pub cache_max_seq: usize,
    pub cache_position_start: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct BshdRopeQkCacheUpdateConfig {
    pub seq_len: usize,
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub cache_max_seq: usize,
    pub cache_position_start: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct KvCacheCompactConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub source_start: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct KvCacheSlidingWindowAppendConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub cache_start: usize,
    pub cache_len: usize,
    pub append_start: usize,
    pub window: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheSlidingWindowAppendState {
    pub cache_start: usize,
    pub cache_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MhcApplyResidualConfig {
    pub batch: usize,
    pub n: usize,
    pub channels: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MhcGemmRmsScaleConfig {
    pub rows: usize,
    pub columns: usize,
    pub reduction: usize,
    pub n: usize,
    pub alpha_pre: f32,
    pub alpha_post: f32,
    pub alpha_res: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct MhcSplitGemmRmsScaleConfig {
    pub base: MhcGemmRmsScaleConfig,
    pub split_k: usize,
}

#[derive(Debug, Clone, Copy)]
struct KvCacheSlidingWindowAppendPlan {
    pre_compact: Option<KvCacheCompactConfig>,
    append_position: usize,
    post_compact: Option<KvCacheCompactConfig>,
    state: KvCacheSlidingWindowAppendState,
}

macro_rules! binary_fused_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            lhs: &impl DeviceSlice<$ty>,
            rhs: &impl DeviceSlice<$ty>,
        ) -> Result<()> {
            let len = validate_binary_fused_lengths(out.len(), lhs.len(), rhs.len())?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                input_pointer(lhs),
                input_pointer(rhs),
                len,
            )
        }
    };
}

macro_rules! unary_fused_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
        ) -> Result<()> {
            let len = ensure_unary_lengths(out, input)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(&stream, output_pointer(out), input_pointer(input), len)
        }
    };
}

macro_rules! binary_fused_2d_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            lhs: &impl DeviceSlice<$ty>,
            rhs: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(lhs.len(), len)?;
            ensure_len(rhs.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                input_pointer(lhs),
                input_pointer(rhs),
                rows,
                cols,
            )
        }
    };
}

macro_rules! silu_and_mul_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            rows: usize,
            hidden: usize,
        ) -> Result<()> {
            validate_silu_and_mul_packed_lengths(out.len(), input.len(), rows, hidden)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                rows,
                hidden,
            )
        }
    };
}

macro_rules! ternary_fused_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            a: &impl DeviceSlice<$ty>,
            b: &impl DeviceSlice<$ty>,
            c: &impl DeviceSlice<$ty>,
        ) -> Result<()> {
            let len = ensure_binary_lengths(out, a, b)?;
            ensure_len(c.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                input_pointer(a),
                input_pointer(b),
                input_pointer(c),
                len,
            )
        }
    };
}

macro_rules! ternary_fused_2d_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            a: &impl DeviceSlice<$ty>,
            b: &impl DeviceSlice<$ty>,
            c: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(a.len(), len)?;
            ensure_len(b.len(), len)?;
            ensure_len(c.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                input_pointer(a),
                input_pointer(b),
                input_pointer(c),
                rows,
                cols,
            )
        }
    };
}

macro_rules! conditional_fused_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            condition: &impl DeviceSlice<u8>,
            a: &impl DeviceSlice<$ty>,
            b: &impl DeviceSlice<$ty>,
            c: &impl DeviceSlice<$ty>,
        ) -> Result<()> {
            let len = ensure_binary_lengths(out, a, b)?;
            ensure_len(condition.len(), len)?;
            ensure_len(c.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                input_pointer(condition),
                input_pointer(a),
                input_pointer(b),
                input_pointer(c),
                len,
            )
        }
    };
}

macro_rules! conditional_fused_2d_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            condition: &impl DeviceSlice<u8>,
            a: &impl DeviceSlice<$ty>,
            b: &impl DeviceSlice<$ty>,
            c: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(condition.len(), len)?;
            ensure_len(a.len(), len)?;
            ensure_len(b.len(), len)?;
            ensure_len(c.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                input_pointer(condition),
                input_pointer(a),
                input_pointer(b),
                input_pointer(c),
                rows,
                cols,
            )
        }
    };
}

macro_rules! bias_fused_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            bias: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(input.len(), len)?;
            ensure_len(bias.len(), cols)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(bias),
                rows,
                cols,
            )
        }
    };
}

macro_rules! affine_fused_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            scale: &impl DeviceSlice<$ty>,
            bias: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(input.len(), len)?;
            ensure_len(scale.len(), cols)?;
            ensure_len(bias.len(), cols)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(scale),
                input_pointer(bias),
                rows,
                cols,
            )
        }
    };
}

macro_rules! column2_fused_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            lhs: &impl DeviceSlice<$ty>,
            rhs: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(input.len(), len)?;
            ensure_len(lhs.len(), cols)?;
            ensure_len(rhs.len(), cols)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(lhs),
                input_pointer(rhs),
                rows,
                cols,
            )
        }
    };
}

#[cfg(feature = "dtype-bf16")]
binary_fused_fn!(silu_mul_bf16, bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_fn!(silu_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(silu_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(silu_mul_f64, f64);
#[cfg(feature = "dtype-bf16")]
binary_fused_2d_fn!(silu_mul_2d_bf16, bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(silu_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(silu_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(silu_mul_2d_f64, f64);

#[cfg(feature = "dtype-bf16")]
binary_fused_fn!(swiglu_bf16, bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_fn!(swiglu_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(swiglu_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(swiglu_f64, f64);
#[cfg(feature = "dtype-bf16")]
binary_fused_2d_fn!(swiglu_2d_bf16, bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(swiglu_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(swiglu_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(swiglu_2d_f64, f64);

#[cfg(feature = "dtype-bf16")]
binary_fused_fn!(gelu_mul_bf16, bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_fn!(gelu_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(gelu_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(gelu_mul_f64, f64);
#[cfg(feature = "dtype-bf16")]
binary_fused_2d_fn!(gelu_mul_2d_bf16, bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(gelu_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(gelu_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(gelu_mul_2d_f64, f64);

#[cfg(feature = "dtype-bf16")]
binary_fused_fn!(geglu_approx_bf16, bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_fn!(geglu_approx_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(geglu_approx_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(geglu_approx_f64, f64);
#[cfg(feature = "dtype-bf16")]
binary_fused_2d_fn!(geglu_approx_2d_bf16, bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(geglu_approx_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(geglu_approx_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(geglu_approx_2d_f64, f64);

#[cfg(feature = "dtype-f32")]
binary_fused_fn!(exact_gelu_mul_f32, f32);
#[cfg(feature = "dtype-f32")]
unary_fused_fn!(exact_gelu_f32, f32);
#[cfg(feature = "dtype-f16")]
binary_fused_fn!(exact_gelu_mul_f16, f16);
#[cfg(feature = "dtype-f16")]
unary_fused_fn!(exact_gelu_f16, f16);
#[cfg(feature = "dtype-bf16")]
binary_fused_fn!(exact_gelu_mul_bf16, bf16);
#[cfg(feature = "dtype-bf16")]
unary_fused_fn!(exact_gelu_bf16, bf16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(geglu_exact_f32, f32);
#[cfg(feature = "dtype-f16")]
binary_fused_fn!(geglu_exact_f16, f16);
#[cfg(feature = "dtype-bf16")]
binary_fused_fn!(geglu_exact_bf16, bf16);

#[cfg(feature = "dtype-f32")]
silu_and_mul_fn!(silu_and_mul_f32, f32);
#[cfg(feature = "dtype-f16")]
silu_and_mul_fn!(silu_and_mul_f16, f16);
#[cfg(feature = "dtype-bf16")]
silu_and_mul_fn!(silu_and_mul_bf16, bf16);

#[cfg(feature = "dtype-bf16")]
binary_fused_fn!(sigmoid_mul_bf16, bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_fn!(sigmoid_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(sigmoid_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(sigmoid_mul_f64, f64);
#[cfg(feature = "dtype-bf16")]
binary_fused_2d_fn!(sigmoid_mul_2d_bf16, bf16);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(sigmoid_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(sigmoid_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(sigmoid_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(relu_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(relu_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(relu_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(relu_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(relu_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(relu_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(relu6_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(relu6_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(relu6_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(relu6_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(relu6_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(relu6_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(hard_sigmoid_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(hard_sigmoid_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(hard_sigmoid_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(hard_sigmoid_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(hard_sigmoid_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(hard_sigmoid_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(hard_swish_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(hard_swish_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(hard_swish_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(hard_swish_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(hard_swish_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(hard_swish_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(softsign_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(softsign_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(softsign_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(softsign_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(softsign_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(softsign_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(mish_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(mish_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(mish_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(mish_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(mish_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(mish_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(selu_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(selu_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(selu_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(selu_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(selu_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(selu_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(tanhshrink_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(tanhshrink_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(tanhshrink_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(tanhshrink_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(tanhshrink_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(tanhshrink_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(log_sigmoid_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(log_sigmoid_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(log_sigmoid_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(log_sigmoid_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(log_sigmoid_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(log_sigmoid_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(reciprocal_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(reciprocal_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(reciprocal_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(reciprocal_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(reciprocal_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(reciprocal_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(square_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(square_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(square_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(square_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(square_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(square_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(ceil_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(ceil_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(ceil_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(ceil_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(ceil_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(ceil_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(floor_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(floor_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(floor_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(floor_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(floor_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(floor_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(trunc_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(trunc_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(trunc_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(trunc_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(trunc_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(trunc_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(frac_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(frac_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(frac_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(frac_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(frac_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(frac_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(deg2rad_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(deg2rad_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(deg2rad_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(deg2rad_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(deg2rad_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(deg2rad_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(rad2deg_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(rad2deg_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(rad2deg_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(rad2deg_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(rad2deg_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(rad2deg_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(sin_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(sin_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(sin_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(sin_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(sin_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(sin_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(cos_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(cos_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(cos_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(cos_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(cos_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(cos_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(tan_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(tan_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(tan_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(tan_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(tan_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(tan_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(asin_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(asin_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(asin_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(asin_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(asin_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(asin_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(acos_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(acos_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(acos_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(acos_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(acos_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(acos_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(atan_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(atan_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(atan_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(atan_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(atan_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(atan_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(sinh_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(sinh_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(sinh_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(sinh_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(sinh_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(sinh_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(cosh_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(cosh_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(cosh_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(cosh_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(cosh_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(cosh_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(asinh_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(asinh_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(asinh_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(asinh_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(asinh_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(asinh_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(acosh_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(acosh_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(acosh_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(acosh_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(acosh_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(acosh_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(atanh_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(atanh_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(atanh_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(atanh_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(atanh_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(atanh_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(tanh_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(tanh_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(tanh_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(tanh_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(tanh_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(tanh_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(exp_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(exp_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(exp_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(exp_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(exp_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(exp_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(exp2_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(exp2_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(exp2_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(exp2_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(exp2_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(exp2_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(expm1_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(expm1_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(expm1_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(expm1_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(expm1_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(expm1_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(log_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(log_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(log_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(log_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(log_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(log_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(log2_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(log2_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(log2_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(log2_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(log2_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(log2_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(log10_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(log10_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(log10_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(log10_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(log10_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(log10_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(log1p_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(log1p_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(log1p_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(log1p_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(log1p_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(log1p_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(softplus_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(softplus_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(softplus_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(softplus_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(softplus_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(softplus_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(sqrt_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(sqrt_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(sqrt_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(sqrt_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(sqrt_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(sqrt_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(rsqrt_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(rsqrt_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(rsqrt_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(rsqrt_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(rsqrt_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(rsqrt_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(cbrt_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(cbrt_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(cbrt_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(cbrt_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(cbrt_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(cbrt_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(abs_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(abs_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(abs_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(abs_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(abs_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(abs_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
binary_fused_fn!(neg_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_fn!(neg_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_fn!(neg_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
binary_fused_2d_fn!(neg_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
binary_fused_2d_fn!(neg_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
binary_fused_2d_fn!(neg_mul_2d_f64, f64);

#[cfg(feature = "dtype-bf16")]
ternary_fused_fn!(add_silu_mul_bf16, bf16);
#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_silu_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_silu_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_silu_mul_f64, f64);
#[cfg(feature = "dtype-bf16")]
ternary_fused_2d_fn!(add_silu_mul_2d_bf16, bf16);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_silu_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_silu_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_silu_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_add_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_add_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_add_f64, f64);

#[cfg(feature = "dtype-bf16")]
ternary_fused_fn!(add_gelu_mul_bf16, bf16);
#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_gelu_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_gelu_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_gelu_mul_f64, f64);
#[cfg(feature = "dtype-bf16")]
ternary_fused_2d_fn!(add_gelu_mul_2d_bf16, bf16);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_gelu_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_gelu_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_gelu_mul_2d_f64, f64);

#[cfg(feature = "dtype-bf16")]
ternary_fused_fn!(add_sigmoid_mul_bf16, bf16);
#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_sigmoid_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_sigmoid_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_sigmoid_mul_f64, f64);
#[cfg(feature = "dtype-bf16")]
ternary_fused_2d_fn!(add_sigmoid_mul_2d_bf16, bf16);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_sigmoid_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_sigmoid_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_sigmoid_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_relu_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_relu_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_relu_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_relu_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_relu_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_relu_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_relu6_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_relu6_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_relu6_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_relu6_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_relu6_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_relu6_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_hard_sigmoid_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_hard_sigmoid_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_hard_sigmoid_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_hard_sigmoid_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_hard_sigmoid_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_hard_sigmoid_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_hard_swish_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_hard_swish_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_hard_swish_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_hard_swish_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_hard_swish_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_hard_swish_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_softsign_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_softsign_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_softsign_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_softsign_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_softsign_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_softsign_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_mish_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_mish_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_mish_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_mish_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_mish_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_mish_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_selu_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_selu_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_selu_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_selu_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_selu_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_selu_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_tanhshrink_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_tanhshrink_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_tanhshrink_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_tanhshrink_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_tanhshrink_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_tanhshrink_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_log_sigmoid_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_log_sigmoid_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_log_sigmoid_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_log_sigmoid_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_log_sigmoid_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_log_sigmoid_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_reciprocal_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_reciprocal_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_reciprocal_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_reciprocal_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_reciprocal_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_reciprocal_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_square_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_square_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_square_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_square_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_square_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_square_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_ceil_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_ceil_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_ceil_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_ceil_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_ceil_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_ceil_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_floor_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_floor_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_floor_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_floor_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_floor_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_floor_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_trunc_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_trunc_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_trunc_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_trunc_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_trunc_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_trunc_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_frac_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_frac_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_frac_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_frac_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_frac_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_frac_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_deg2rad_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_deg2rad_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_deg2rad_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_deg2rad_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_deg2rad_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_deg2rad_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_rad2deg_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_rad2deg_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_rad2deg_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_rad2deg_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_rad2deg_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_rad2deg_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_sin_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_sin_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_sin_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_sin_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_sin_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_sin_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_cos_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_cos_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_cos_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_cos_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_cos_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_cos_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_tan_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_tan_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_tan_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_tan_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_tan_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_tan_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_asin_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_asin_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_asin_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_asin_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_asin_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_asin_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_acos_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_acos_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_acos_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_acos_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_acos_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_acos_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_atan_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_atan_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_atan_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_atan_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_atan_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_atan_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_sinh_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_sinh_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_sinh_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_sinh_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_sinh_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_sinh_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_cosh_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_cosh_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_cosh_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_cosh_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_cosh_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_cosh_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_asinh_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_asinh_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_asinh_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_asinh_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_asinh_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_asinh_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_acosh_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_acosh_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_acosh_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_acosh_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_acosh_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_acosh_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_atanh_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_atanh_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_atanh_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_atanh_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_atanh_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_atanh_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_tanh_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_tanh_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_tanh_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_tanh_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_tanh_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_tanh_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_exp_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_exp_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_exp_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_exp_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_exp_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_exp_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_exp2_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_exp2_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_exp2_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_exp2_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_exp2_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_exp2_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_expm1_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_expm1_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_expm1_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_expm1_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_expm1_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_expm1_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_log_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_log_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_log_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_log_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_log_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_log_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_log2_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_log2_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_log2_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_log2_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_log2_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_log2_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_log10_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_log10_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_log10_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_log10_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_log10_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_log10_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_log1p_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_log1p_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_log1p_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_log1p_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_log1p_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_log1p_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_softplus_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_softplus_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_softplus_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_softplus_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_softplus_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_softplus_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_sqrt_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_sqrt_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_sqrt_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_sqrt_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_sqrt_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_sqrt_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_rsqrt_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_rsqrt_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_rsqrt_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_rsqrt_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_rsqrt_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_rsqrt_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_cbrt_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_cbrt_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_cbrt_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_cbrt_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_cbrt_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_cbrt_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_abs_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_abs_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_abs_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_abs_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_abs_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_abs_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_neg_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_neg_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_neg_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_neg_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_neg_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_neg_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(mul_add_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(mul_add_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(mul_add_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(mul_add_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(mul_add_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(mul_add_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(add_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(add_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(add_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(add_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(add_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(add_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(sub_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(sub_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(sub_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(sub_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(sub_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(sub_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(rsub_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(rsub_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(rsub_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(rsub_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(rsub_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(rsub_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(div_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(div_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(div_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(div_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(div_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(div_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(rdiv_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(rdiv_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(rdiv_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(rdiv_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(rdiv_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(rdiv_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(min_add_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(min_add_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(min_add_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(min_add_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(min_add_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(min_add_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(max_add_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(max_add_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(max_add_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(max_add_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(max_add_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(max_add_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(min_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(min_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(min_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(min_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(min_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(min_mul_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
ternary_fused_fn!(max_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_fn!(max_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_fn!(max_mul_f64, f64);
#[cfg(feature = "dtype-f16")]
ternary_fused_2d_fn!(max_mul_2d_f16, f16);
#[cfg(feature = "dtype-f32")]
ternary_fused_2d_fn!(max_mul_2d_f32, f32);
#[cfg(feature = "dtype-f64")]
ternary_fused_2d_fn!(max_mul_2d_f64, f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_mul_add_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_mul_add_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_mul_add_f64, f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_mul_add_2d_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_mul_add_2d_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_mul_add_2d_f64, f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_add_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_add_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_add_f64, f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_add_2d_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_add_2d_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_add_2d_f64, f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_sub_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_sub_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_sub_f64, f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_sub_2d_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_sub_2d_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_sub_2d_f64, f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_mul_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_mul_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_mul_f64, f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_mul_2d_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_mul_2d_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_mul_2d_f64, f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_div_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_div_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_div_f64, f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_div_2d_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_div_2d_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_div_2d_f64, f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_min_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_min_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_min_f64, f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_min_2d_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_min_2d_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_min_2d_f64, f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_fn!(where_max_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_fn!(where_max_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_fn!(where_max_f64, f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_max_2d_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_max_2d_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
conditional_fused_2d_fn!(where_max_2d_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_gelu_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_gelu_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_gelu_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_sub_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_sub_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_sub_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_rsub_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_rsub_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_rsub_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_mul_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_mul_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_mul_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_div_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_div_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_div_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_rdiv_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_rdiv_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_rdiv_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_add_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_add_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_add_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_sub_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_sub_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_sub_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_add_scale_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_add_scale_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_add_scale_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_sub_scale_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_sub_scale_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_sub_scale_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_gelu_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_gelu_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_gelu_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_silu_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_silu_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_silu_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_sigmoid_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_sigmoid_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_sigmoid_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_relu_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_relu_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_relu_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_relu6_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_relu6_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_relu6_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_hard_sigmoid_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_hard_sigmoid_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_hard_sigmoid_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_hard_swish_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_hard_swish_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_hard_swish_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_softsign_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_softsign_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_softsign_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_mish_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_mish_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_mish_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_selu_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_selu_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_selu_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_tanhshrink_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_tanhshrink_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_tanhshrink_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_log_sigmoid_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_log_sigmoid_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_log_sigmoid_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_reciprocal_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_reciprocal_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_reciprocal_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_square_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_square_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_square_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_ceil_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_ceil_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_ceil_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_floor_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_floor_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_floor_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_trunc_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_trunc_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_trunc_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_frac_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_frac_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_frac_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_deg2rad_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_deg2rad_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_deg2rad_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_rad2deg_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_rad2deg_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_rad2deg_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_sin_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_sin_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_sin_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_cos_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_cos_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_cos_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_tan_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_tan_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_tan_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_asin_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_asin_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_asin_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_acos_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_acos_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_acos_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_atan_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_atan_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_atan_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_sinh_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_sinh_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_sinh_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_cosh_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_cosh_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_cosh_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_asinh_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_asinh_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_asinh_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_acosh_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_acosh_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_acosh_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_atanh_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_atanh_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_atanh_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_tanh_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_tanh_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_tanh_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_exp_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_exp_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_exp_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_exp2_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_exp2_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_exp2_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_expm1_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_expm1_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_expm1_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_log_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_log_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_log_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_log2_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_log2_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_log2_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_log10_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_log10_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_log10_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_log1p_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_log1p_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_log1p_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_softplus_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_softplus_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_softplus_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_sqrt_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_sqrt_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_sqrt_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_rsqrt_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_rsqrt_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_rsqrt_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_cbrt_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_cbrt_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_cbrt_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_abs_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_abs_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_abs_f64, f64);

#[cfg(feature = "dtype-f16")]
affine_fused_fn!(bias_scale_neg_f16, f16);
#[cfg(feature = "dtype-f32")]
affine_fused_fn!(bias_scale_neg_f32, f32);
#[cfg(feature = "dtype-f64")]
affine_fused_fn!(bias_scale_neg_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_silu_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_silu_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_silu_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_sigmoid_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_sigmoid_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_sigmoid_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_relu_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_relu_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_relu_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_relu6_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_relu6_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_relu6_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_hard_sigmoid_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_hard_sigmoid_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_hard_sigmoid_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_hard_swish_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_hard_swish_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_hard_swish_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_softsign_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_softsign_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_softsign_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_mish_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_mish_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_mish_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_selu_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_selu_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_selu_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_tanhshrink_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_tanhshrink_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_tanhshrink_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_log_sigmoid_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_log_sigmoid_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_log_sigmoid_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_reciprocal_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_reciprocal_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_reciprocal_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_square_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_square_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_square_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_ceil_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_ceil_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_ceil_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_floor_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_floor_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_floor_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_trunc_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_trunc_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_trunc_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_frac_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_frac_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_frac_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_deg2rad_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_deg2rad_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_deg2rad_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_rad2deg_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_rad2deg_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_rad2deg_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_sin_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_sin_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_sin_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_cos_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_cos_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_cos_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_tan_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_tan_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_tan_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_asin_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_asin_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_asin_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_acos_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_acos_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_acos_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_atan_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_atan_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_atan_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_sinh_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_sinh_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_sinh_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_cosh_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_cosh_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_cosh_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_asinh_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_asinh_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_asinh_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_acosh_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_acosh_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_acosh_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_atanh_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_atanh_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_atanh_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_tanh_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_tanh_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_tanh_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_exp_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_exp_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_exp_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_exp2_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_exp2_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_exp2_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_expm1_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_expm1_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_expm1_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_log_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_log_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_log_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_log2_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_log2_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_log2_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_log10_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_log10_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_log10_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_log1p_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_log1p_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_log1p_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_softplus_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_softplus_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_softplus_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_sqrt_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_sqrt_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_sqrt_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_rsqrt_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_rsqrt_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_rsqrt_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_cbrt_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_cbrt_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_cbrt_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_abs_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_abs_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_abs_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_neg_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_neg_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_neg_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_min_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_min_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_min_f64, f64);

#[cfg(feature = "dtype-f16")]
bias_fused_fn!(bias_max_f16, f16);
#[cfg(feature = "dtype-f32")]
bias_fused_fn!(bias_max_f32, f32);
#[cfg(feature = "dtype-f64")]
bias_fused_fn!(bias_max_f64, f64);

#[cfg(feature = "dtype-f16")]
column2_fused_fn!(column_clamp_f16, f16);
#[cfg(feature = "dtype-f32")]
column2_fused_fn!(column_clamp_f32, f32);
#[cfg(feature = "dtype-f64")]
column2_fused_fn!(column_clamp_f64, f64);

#[cfg(feature = "dtype-f16")]
column2_fused_fn!(column_lerp_f16, f16);
#[cfg(feature = "dtype-f32")]
column2_fused_fn!(column_lerp_f32, f32);
#[cfg(feature = "dtype-f64")]
column2_fused_fn!(column_lerp_f64, f64);

macro_rules! rms_norm_add_fn {
    ($name:ident, $ty:ty, $eps_ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            residual_out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            residual: &impl DeviceSlice<$ty>,
            weight: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
            weight_offset: $eps_ty,
        ) -> Result<()> {
            validate_rms_norm_add(
                out.len(),
                residual_out.len(),
                input.len(),
                residual.len(),
                weight.len(),
                rows,
                cols,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                output_pointer(residual_out),
                input_pointer(input),
                input_pointer(residual),
                input_pointer(weight),
                rows,
                cols,
                eps,
                weight_offset,
            )
        }
    };
}

macro_rules! rms_norm_add_wide_fn {
    ($name:ident, $ty:ty, $eps_ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            residual_out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            residual: &impl DeviceSlice<$ty>,
            weight: &impl DeviceSlice<$ty>,
            partial_sum: &mut impl DeviceSliceMut<f32>,
            row_sum: &mut impl DeviceSliceMut<f32>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
            weight_offset: $eps_ty,
        ) -> Result<()> {
            validate_rms_norm_add_wide(
                out.len(),
                residual_out.len(),
                input.len(),
                residual.len(),
                weight.len(),
                partial_sum.len(),
                row_sum.len(),
                rows,
                cols,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                output_pointer(residual_out),
                input_pointer(input),
                input_pointer(residual),
                input_pointer(weight),
                output_pointer(partial_sum),
                output_pointer(row_sum),
                rows,
                cols,
                eps,
                weight_offset,
            )
        }
    };
}

macro_rules! rms_norm_silu_mul_fn {
    ($name:ident, $ty:ty, $eps_ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            weight: &impl DeviceSlice<$ty>,
            up: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
            weight_offset: $eps_ty,
        ) -> Result<()> {
            validate_rms_norm_silu_mul(out.len(), input.len(), weight.len(), up.len(), rows, cols)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(weight),
                input_pointer(up),
                rows,
                cols,
                eps,
                weight_offset,
            )
        }
    };
}

macro_rules! rms_norm_gated_silu_fn {
    ($name:ident, $ty:ty, $eps_ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            gate: &impl DeviceSlice<$ty>,
            weight: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
            weight_offset: $eps_ty,
        ) -> Result<()> {
            validate_rms_norm_gated_silu(
                out.len(),
                input.len(),
                gate.len(),
                weight.len(),
                rows,
                cols,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(gate),
                input_pointer(weight),
                rows,
                cols,
                eps,
                weight_offset,
            )
        }
    };
}

macro_rules! dual_rms_norm_fn {
    ($name:ident, $ty:ty, $eps_ty:ty) => {
        pub fn $name(
            stream: &Stream,
            q_out: &mut impl DeviceSliceMut<$ty>,
            k_out: &mut impl DeviceSliceMut<$ty>,
            q: &impl DeviceSlice<$ty>,
            k: &impl DeviceSlice<$ty>,
            q_weight: &impl DeviceSlice<$ty>,
            k_weight: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
        ) -> Result<()> {
            validate_dual_rms_norm(
                q_out.len(),
                k_out.len(),
                q.len(),
                k.len(),
                q_weight.len(),
                k_weight.len(),
                rows,
                cols,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(q_out),
                output_pointer(k_out),
                input_pointer(q),
                input_pointer(k),
                input_pointer(q_weight),
                input_pointer(k_weight),
                rows,
                cols,
                eps,
            )
        }
    };
}

macro_rules! rms_norm_residual_add_fn {
    ($name:ident, $ty:ty, $eps_ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            residual: &impl DeviceSlice<$ty>,
            weight: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
            eps: $eps_ty,
        ) -> Result<()> {
            validate_rms_norm_residual_add(
                out.len(),
                input.len(),
                residual.len(),
                weight.len(),
                rows,
                cols,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(residual),
                input_pointer(weight),
                rows,
                cols,
                eps,
            )
        }
    };
}

#[cfg(feature = "dtype-f16")]
rms_norm_add_fn!(rms_norm_add_f16, f16, f32);
#[cfg(feature = "dtype-bf16")]
rms_norm_add_fn!(rms_norm_add_bf16, bf16, f32);
#[cfg(feature = "dtype-f32")]
rms_norm_add_fn!(rms_norm_add_f32, f32, f32);
#[cfg(feature = "dtype-f64")]
rms_norm_add_fn!(rms_norm_add_f64, f64, f64);

#[cfg(feature = "dtype-f16")]
rms_norm_add_wide_fn!(rms_norm_add_wide_f16, f16, f32);
#[cfg(feature = "dtype-bf16")]
rms_norm_add_wide_fn!(rms_norm_add_wide_bf16, bf16, f32);
#[cfg(feature = "dtype-f32")]
rms_norm_add_wide_fn!(rms_norm_add_wide_f32, f32, f32);

#[cfg(feature = "dtype-f16")]
rms_norm_silu_mul_fn!(rms_norm_silu_mul_f16, f16, f32);
#[cfg(feature = "dtype-bf16")]
rms_norm_silu_mul_fn!(rms_norm_silu_mul_bf16, bf16, f32);
#[cfg(feature = "dtype-f32")]
rms_norm_silu_mul_fn!(rms_norm_silu_mul_f32, f32, f32);
#[cfg(feature = "dtype-f64")]
rms_norm_silu_mul_fn!(rms_norm_silu_mul_f64, f64, f64);

#[cfg(feature = "dtype-f32")]
rms_norm_gated_silu_fn!(rms_norm_gated_silu_f32, f32, f32);
#[cfg(feature = "dtype-f16")]
rms_norm_gated_silu_fn!(rms_norm_gated_silu_f16, f16, f32);
#[cfg(feature = "dtype-bf16")]
rms_norm_gated_silu_fn!(rms_norm_gated_silu_bf16, bf16, f32);

#[cfg(feature = "dtype-f32")]
dual_rms_norm_fn!(dual_rms_norm_f32, f32, f32);
#[cfg(feature = "dtype-f16")]
dual_rms_norm_fn!(dual_rms_norm_f16, f16, f32);
#[cfg(feature = "dtype-bf16")]
dual_rms_norm_fn!(dual_rms_norm_bf16, bf16, f32);

#[cfg(feature = "dtype-f32")]
rms_norm_residual_add_fn!(rms_norm_residual_add_f32, f32, f32);
#[cfg(feature = "dtype-f16")]
rms_norm_residual_add_fn!(rms_norm_residual_add_f16, f16, f32);
#[cfg(feature = "dtype-bf16")]
rms_norm_residual_add_fn!(rms_norm_residual_add_bf16, bf16, f32);

#[cfg(feature = "dtype-f32")]
pub fn mhc_apply_residual_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    x: &impl DeviceSlice<f32>,
    f_out: &impl DeviceSlice<f32>,
    y: &impl DeviceSlice<f32>,
    config: MhcApplyResidualConfig,
) -> Result<()> {
    validate_mhc_apply_residual(out.len(), x.len(), f_out.len(), y.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::fused::mhc_apply_residual_f32(
        &stream,
        output_pointer(out),
        input_pointer(x),
        input_pointer(f_out),
        input_pointer(y),
        config.batch,
        config.n,
        config.channels,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn mhc_sinkhorn_f32(
    stream: &Stream,
    y: &mut impl DeviceSliceMut<f32>,
    batch: usize,
    n: usize,
) -> Result<()> {
    validate_mhc_y_buffer(y.len(), batch, n)?;
    let stream = borrowed_stream(stream)?;
    cutile::fused::mhc_sinkhorn_f32(&stream, output_pointer(y), batch, n)
}

#[cfg(feature = "dtype-f32")]

pub fn mhc_gemm_rms_scale_f32(
    stream: &Stream,
    y: &mut impl DeviceSliceMut<f32>,
    r: &mut impl DeviceSliceMut<f32>,
    x: &impl DeviceSlice<f32>,
    w: &impl DeviceSlice<f32>,
    bias: &impl DeviceSlice<f32>,
    config: MhcGemmRmsScaleConfig,
) -> Result<()> {
    validate_mhc_gemm_rms_scale(y.len(), r.len(), x.len(), w.len(), bias.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::fused::mhc_gemm_rms_scale_f32(
        &stream,
        output_pointer(y),
        output_pointer(r),
        input_pointer(x),
        input_pointer(w),
        input_pointer(bias),
        config.rows,
        config.columns,
        config.reduction,
        config.n,
        config.alpha_pre,
        config.alpha_post,
        config.alpha_res,
    )
}

#[cfg(feature = "dtype-f32")]

pub fn mhc_split_gemm_rms_f32(
    stream: &Stream,
    y_acc: &mut impl DeviceSliceMut<f32>,
    r_acc: &mut impl DeviceSliceMut<f32>,
    x: &impl DeviceSlice<f32>,
    w: &impl DeviceSlice<f32>,
    config: MhcSplitGemmRmsScaleConfig,
) -> Result<()> {
    validate_mhc_split_gemm_rms(y_acc.len(), r_acc.len(), x.len(), w.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::fused::mhc_split_gemm_rms_f32(
        &stream,
        output_pointer(y_acc),
        output_pointer(r_acc),
        input_pointer(x),
        input_pointer(w),
        config.base.rows,
        config.base.columns,
        config.base.reduction,
        config.split_k,
    )
}

#[cfg(feature = "dtype-f32")]

pub fn mhc_finalize_scale_bias_sigmoid_f32(
    stream: &Stream,
    y: &mut impl DeviceSliceMut<f32>,
    r: &mut impl DeviceSliceMut<f32>,
    y_acc: &impl DeviceSlice<f32>,
    r_acc: &impl DeviceSlice<f32>,
    bias: &impl DeviceSlice<f32>,
    config: MhcSplitGemmRmsScaleConfig,
) -> Result<()> {
    validate_mhc_finalize_scale_bias_sigmoid(
        y.len(),
        r.len(),
        y_acc.len(),
        r_acc.len(),
        bias.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::fused::mhc_finalize_scale_bias_sigmoid_f32(
        &stream,
        output_pointer(y),
        output_pointer(r),
        input_pointer(y_acc),
        input_pointer(r_acc),
        input_pointer(bias),
        config.base.rows,
        config.base.columns,
        config.base.reduction,
        config.base.n,
        config.split_k,
        config.base.alpha_pre,
        config.base.alpha_post,
        config.base.alpha_res,
    )
}

macro_rules! kv_cache_update_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            k_cache: &mut impl DeviceSliceMut<$ty>,
            v_cache: &mut impl DeviceSliceMut<$ty>,
            new_k: &impl DeviceSlice<$ty>,
            new_v: &impl DeviceSlice<$ty>,
            seq_len: usize,
            heads: usize,
            head_dim: usize,
            max_seq: usize,
            position_start: usize,
        ) -> Result<()> {
            let new_len = checked_element_count(checked_element_count(seq_len, heads)?, head_dim)?;
            let cache_len =
                checked_element_count(checked_element_count(heads, max_seq)?, head_dim)?;
            ensure_len(new_k.len(), new_len)?;
            ensure_len(new_v.len(), new_len)?;
            ensure_len(k_cache.len(), cache_len)?;
            ensure_len(v_cache.len(), cache_len)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(k_cache),
                output_pointer(v_cache),
                input_pointer(new_k),
                input_pointer(new_v),
                seq_len,
                heads,
                head_dim,
                max_seq,
                position_start,
            )
        }
    };
}

#[cfg(feature = "dtype-f16")]
kv_cache_update_fn!(kv_cache_update_f16, f16);

#[cfg(feature = "dtype-f16")]

pub fn bshd_rope_qk_cache_update_f16(
    stream: &Stream,
    q_out: &mut impl DeviceSliceMut<f16>,
    k_out: &mut impl DeviceSliceMut<f16>,
    k_cache: &mut impl DeviceSliceMut<f16>,
    v_cache: &mut impl DeviceSliceMut<f16>,
    q_input: &impl DeviceSlice<f16>,
    k_input: &impl DeviceSlice<f16>,
    v_input: &impl DeviceSlice<f16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BshdRopeQkCacheUpdateConfig,
) -> Result<()> {
    validate_bshd_rope_qk_cache_update(
        q_out.len(),
        k_out.len(),
        k_cache.len(),
        v_cache.len(),
        q_input.len(),
        k_input.len(),
        v_input.len(),
        cos.len(),
        sin.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::fused::bshd_rope_qk_cache_update_f16(
        &stream,
        output_pointer(q_out),
        output_pointer(k_out),
        output_pointer(k_cache),
        output_pointer(v_cache),
        input_pointer(q_input),
        input_pointer(k_input),
        input_pointer(v_input),
        input_pointer(cos),
        input_pointer(sin),
        config.seq_len,
        config.query_heads,
        config.key_value_heads,
        config.head_dim,
        config.cache_max_seq,
        config.cache_position_start,
    )
}

#[cfg(feature = "dtype-f32")]
kv_cache_update_fn!(kv_cache_update_f32, f32);
#[cfg(feature = "dtype-f64")]
kv_cache_update_fn!(kv_cache_update_f64, f64);
#[cfg(feature = "dtype-f16")]
kv_cache_update_fn!(kv_cache_update_seq_f16, f16);
#[cfg(feature = "dtype-f32")]
kv_cache_update_fn!(kv_cache_update_seq_f32, f32);
#[cfg(feature = "dtype-f64")]
kv_cache_update_fn!(kv_cache_update_seq_f64, f64);

macro_rules! kv_cache_append_sliding_window_fn {
    ($name:ident, $ty:ty, $update:ident, $compact:ident) => {
        pub fn $name(
            stream: &Stream,
            k_cache: &mut impl DeviceSliceMut<$ty>,
            v_cache: &mut impl DeviceSliceMut<$ty>,
            new_k: &impl DeviceSlice<$ty>,
            new_v: &impl DeviceSlice<$ty>,
            seq_len: usize,
            config: KvCacheSlidingWindowAppendConfig,
        ) -> Result<KvCacheSlidingWindowAppendState> {
            let plan = validate_kv_cache_sliding_window_append(
                k_cache.len(),
                v_cache.len(),
                new_k.len(),
                new_v.len(),
                seq_len,
                config,
            )?;
            if let Some(compact) = plan.pre_compact {
                $compact(stream, k_cache, v_cache, compact)?;
            }
            $update(
                stream,
                k_cache,
                v_cache,
                new_k,
                new_v,
                seq_len,
                config.heads,
                config.head_dim,
                config.max_seq,
                plan.append_position,
            )?;
            if let Some(compact) = plan.post_compact {
                $compact(stream, k_cache, v_cache, compact)?;
            }
            Ok(plan.state)
        }
    };
}

#[cfg(feature = "dtype-f16")]
kv_cache_append_sliding_window_fn!(
    kv_cache_append_sliding_window_f16,
    f16,
    kv_cache_update_f16,
    kv_cache_compact_f16
);
#[cfg(feature = "dtype-f32")]
kv_cache_append_sliding_window_fn!(
    kv_cache_append_sliding_window_f32,
    f32,
    kv_cache_update_f32,
    kv_cache_compact_f32
);
#[cfg(feature = "dtype-f64")]
kv_cache_append_sliding_window_fn!(
    kv_cache_append_sliding_window_f64,
    f64,
    kv_cache_update_f64,
    kv_cache_compact_f64
);

macro_rules! kv_cache_compact_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            k_cache: &mut impl DeviceSliceMut<$ty>,
            v_cache: &mut impl DeviceSliceMut<$ty>,
            config: KvCacheCompactConfig,
        ) -> Result<()> {
            validate_kv_cache_compact(k_cache.len(), v_cache.len(), config)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(k_cache),
                output_pointer(v_cache),
                config.heads,
                config.head_dim,
                config.max_seq,
                config.source_start,
                config.token_count,
            )
        }
    };
}

#[cfg(feature = "dtype-f16")]
kv_cache_compact_fn!(kv_cache_compact_f16, f16);
#[cfg(feature = "dtype-f32")]
kv_cache_compact_fn!(kv_cache_compact_f32, f32);
#[cfg(feature = "dtype-f64")]
kv_cache_compact_fn!(kv_cache_compact_f64, f64);

fn validate_kv_cache_compact(
    k_cache_len: usize,
    v_cache_len: usize,
    config: KvCacheCompactConfig,
) -> Result<()> {
    if config.heads == 0 || config.head_dim == 0 || config.max_seq == 0 || config.token_count == 0 {
        return Err(Error::InvalidLength);
    }
    let source_end = config
        .source_start
        .checked_add(config.token_count)
        .ok_or(Error::SizeOverflow)?;
    if source_end > config.max_seq {
        return Err(Error::InvalidLength);
    }
    let cache_len = checked_element_count(
        checked_element_count(config.heads, config.max_seq)?,
        config.head_dim,
    )?;
    ensure_len(k_cache_len, cache_len)?;
    ensure_len(v_cache_len, cache_len)?;
    Ok(())
}

fn validate_rms_norm_add(
    out_len: usize,
    residual_out_len: usize,
    input_len: usize,
    residual_len: usize,
    weight_len: usize,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    let len = validate_matrix_shape(rows, cols)?;
    ensure_len(out_len, len)?;
    ensure_len(residual_out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(residual_len, len)?;
    ensure_len(weight_len, cols)?;
    Ok(len)
}

fn validate_rms_norm_add_wide(
    out_len: usize,
    residual_out_len: usize,
    input_len: usize,
    residual_len: usize,
    weight_len: usize,
    partial_sum_len: usize,
    row_sum_len: usize,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    let len = validate_rms_norm_add(
        out_len,
        residual_out_len,
        input_len,
        residual_len,
        weight_len,
        rows,
        cols,
    )?;
    let Some(plan) = cutile::fused::rms_norm_wide_plan(rows, cols)? else {
        return Ok(len);
    };
    ensure_len(partial_sum_len, plan.partial_len)?;
    ensure_len(row_sum_len, plan.row_sum_len)?;
    Ok(len)
}

fn validate_binary_fused_lengths(out_len: usize, lhs_len: usize, rhs_len: usize) -> Result<usize> {
    ensure_len(lhs_len, out_len)?;
    ensure_len(rhs_len, out_len)?;
    Ok(out_len)
}

fn validate_silu_and_mul_packed_lengths(
    out_len: usize,
    input_len: usize,
    rows: usize,
    hidden: usize,
) -> Result<usize> {
    let len = validate_matrix_shape(rows, hidden)?;
    let input_len_expected = checked_element_count(len, 2)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, input_len_expected)?;
    Ok(len)
}

fn validate_rms_norm_silu_mul(
    out_len: usize,
    input_len: usize,
    weight_len: usize,
    up_len: usize,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    let len = validate_matrix_shape(rows, cols)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(weight_len, cols)?;
    ensure_len(up_len, len)?;
    Ok(len)
}

fn validate_rms_norm_gated_silu(
    out_len: usize,
    input_len: usize,
    gate_len: usize,
    weight_len: usize,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    let len = validate_matrix_shape(rows, cols)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(gate_len, len)?;
    ensure_len(weight_len, cols)?;
    Ok(len)
}

fn validate_dual_rms_norm(
    q_out_len: usize,
    k_out_len: usize,
    q_len: usize,
    k_len: usize,
    q_weight_len: usize,
    k_weight_len: usize,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    let len = validate_matrix_shape(rows, cols)?;
    ensure_len(q_out_len, len)?;
    ensure_len(k_out_len, len)?;
    ensure_len(q_len, len)?;
    ensure_len(k_len, len)?;
    ensure_len(q_weight_len, cols)?;
    ensure_len(k_weight_len, cols)?;
    Ok(len)
}

fn validate_rms_norm_residual_add(
    out_len: usize,
    input_len: usize,
    residual_len: usize,
    weight_len: usize,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    let len = validate_matrix_shape(rows, cols)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(residual_len, len)?;
    ensure_len(weight_len, cols)?;
    Ok(len)
}

fn validate_mhc_apply_residual(
    out_len: usize,
    x_len: usize,
    f_out_len: usize,
    y_len: usize,
    config: MhcApplyResidualConfig,
) -> Result<usize> {
    if config.batch == 0 || config.n == 0 || config.channels == 0 {
        return Err(Error::InvalidLength);
    }
    let token_len = checked_element_count(config.batch, config.n)?;
    let out_expected = checked_element_count(token_len, config.channels)?;
    let f_out_expected = checked_element_count(config.batch, config.channels)?;
    validate_mhc_y_buffer(y_len, config.batch, config.n)?;
    ensure_len(out_len, out_expected)?;
    ensure_len(x_len, out_expected)?;
    ensure_len(f_out_len, f_out_expected)?;
    Ok(out_expected)
}

fn validate_mhc_y_buffer(y_len: usize, batch: usize, n: usize) -> Result<usize> {
    if batch == 0 || n == 0 {
        return Err(Error::InvalidLength);
    }
    let y_row = checked_element_count(n, n.checked_add(2).ok_or(Error::SizeOverflow)?)?;
    let y_expected = checked_element_count(batch, y_row)?;
    ensure_len(y_len, y_expected)?;
    Ok(y_expected)
}

fn validate_mhc_gemm_rms_scale(
    y_len: usize,
    r_len: usize,
    x_len: usize,
    w_len: usize,
    bias_len: usize,
    config: MhcGemmRmsScaleConfig,
) -> Result<usize> {
    if config.rows == 0 || config.columns == 0 || config.reduction == 0 || config.n == 0 {
        return Err(Error::InvalidLength);
    }
    let min_columns = checked_element_count(2, config.n)?;
    if config.columns < min_columns {
        return Err(Error::LengthMismatch);
    }
    let y_expected = checked_element_count(config.rows, config.columns)?;
    let x_expected = checked_element_count(config.rows, config.reduction)?;
    let w_expected = checked_element_count(config.reduction, config.columns)?;
    ensure_len(y_len, y_expected)?;
    ensure_len(r_len, config.rows)?;
    ensure_len(x_len, x_expected)?;
    ensure_len(w_len, w_expected)?;
    ensure_len(bias_len, config.columns)?;
    Ok(y_expected)
}

fn validate_mhc_split_gemm_rms(
    y_acc_len: usize,
    r_acc_len: usize,
    x_len: usize,
    w_len: usize,
    config: MhcSplitGemmRmsScaleConfig,
) -> Result<usize> {
    if config.split_k == 0 {
        return Err(Error::InvalidLength);
    }
    validate_mhc_base_config(config.base)?;
    let base_len = checked_element_count(config.base.rows, config.base.columns)?;
    let x_expected = checked_element_count(config.base.rows, config.base.reduction)?;
    let w_expected = checked_element_count(config.base.reduction, config.base.columns)?;
    ensure_len(x_len, x_expected)?;
    ensure_len(w_len, w_expected)?;
    let temp_len = checked_element_count(base_len, config.split_k)?;
    ensure_len(y_acc_len, temp_len)?;
    ensure_len(r_acc_len, temp_len)?;
    Ok(temp_len)
}

fn validate_mhc_finalize_scale_bias_sigmoid(
    y_len: usize,
    r_len: usize,
    y_acc_len: usize,
    r_acc_len: usize,
    bias_len: usize,
    config: MhcSplitGemmRmsScaleConfig,
) -> Result<usize> {
    if config.split_k == 0 {
        return Err(Error::InvalidLength);
    }
    validate_mhc_base_config(config.base)?;
    let base_len = checked_element_count(config.base.rows, config.base.columns)?;
    let temp_len = checked_element_count(base_len, config.split_k)?;
    ensure_len(y_len, base_len)?;
    ensure_len(r_len, config.base.rows)?;
    ensure_len(y_acc_len, temp_len)?;
    ensure_len(r_acc_len, temp_len)?;
    ensure_len(bias_len, config.base.columns)?;
    Ok(base_len)
}

fn validate_mhc_base_config(config: MhcGemmRmsScaleConfig) -> Result<()> {
    if config.rows == 0 || config.columns == 0 || config.reduction == 0 || config.n == 0 {
        return Err(Error::InvalidLength);
    }
    let min_columns = checked_element_count(2, config.n)?;
    if config.columns < min_columns {
        return Err(Error::LengthMismatch);
    }
    Ok(())
}

fn validate_matrix_shape(rows: usize, cols: usize) -> Result<usize> {
    if rows == 0 || cols == 0 {
        return Err(Error::InvalidLength);
    }
    checked_element_count(rows, cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_gated_silu_validation_accepts_exact_lengths() -> Result<()> {
        assert_eq!(validate_rms_norm_gated_silu(24, 24, 24, 6, 4, 6)?, 24);
        Ok(())
    }

    #[test]
    fn rms_norm_gated_silu_validation_rejects_short_gate() {
        assert!(matches!(
            validate_rms_norm_gated_silu(24, 24, 23, 6, 4, 6),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn rms_norm_gated_silu_validation_rejects_zero_rows() {
        assert!(matches!(
            validate_rms_norm_gated_silu(0, 0, 0, 6, 0, 6),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn dual_rms_norm_validation_accepts_exact_lengths() -> Result<()> {
        assert_eq!(validate_dual_rms_norm(24, 24, 24, 24, 6, 6, 4, 6)?, 24);
        Ok(())
    }

    #[test]
    fn dual_rms_norm_validation_rejects_short_k_weight() {
        assert!(matches!(
            validate_dual_rms_norm(24, 24, 24, 24, 6, 5, 4, 6),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn rms_norm_residual_add_validation_accepts_exact_lengths() -> Result<()> {
        assert_eq!(validate_rms_norm_residual_add(24, 24, 24, 6, 4, 6)?, 24);
        Ok(())
    }

    #[test]
    fn rms_norm_residual_add_validation_rejects_short_residual() {
        assert!(matches!(
            validate_rms_norm_residual_add(24, 24, 23, 6, 4, 6),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn rms_norm_add_wide_validation_accepts_shared_hidden_width() -> Result<()> {
        assert_eq!(
            validate_rms_norm_add_wide(10_240, 10_240, 10_240, 10_240, 5_120, 10, 2, 2, 5_120,)?,
            10_240
        );
        Ok(())
    }

    #[test]
    fn rms_norm_add_wide_validation_rejects_short_partial_sum() {
        assert!(matches!(
            validate_rms_norm_add_wide(10_240, 10_240, 10_240, 10_240, 5_120, 9, 2, 2, 5_120,),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn rms_norm_add_wide_validation_rejects_short_row_sum() {
        assert!(matches!(
            validate_rms_norm_add_wide(10_240, 10_240, 10_240, 10_240, 5_120, 10, 1, 2, 5_120,),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn mhc_apply_residual_validation_accepts_exact_lengths() -> Result<()> {
        let config = MhcApplyResidualConfig {
            batch: 2,
            n: 3,
            channels: 4,
        };
        assert_eq!(validate_mhc_apply_residual(24, 24, 8, 30, config)?, 24);
        Ok(())
    }

    #[test]
    fn mhc_apply_residual_validation_rejects_short_y() {
        let config = MhcApplyResidualConfig {
            batch: 2,
            n: 3,
            channels: 4,
        };
        assert!(matches!(
            validate_mhc_apply_residual(24, 24, 8, 29, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn mhc_apply_residual_validation_rejects_zero_n() {
        let config = MhcApplyResidualConfig {
            batch: 2,
            n: 0,
            channels: 4,
        };
        assert!(matches!(
            validate_mhc_apply_residual(0, 0, 8, 0, config),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn mhc_sinkhorn_validation_accepts_exact_y() -> Result<()> {
        assert_eq!(validate_mhc_y_buffer(30, 2, 3)?, 30);
        Ok(())
    }

    #[test]
    fn mhc_sinkhorn_validation_rejects_short_y() {
        assert!(matches!(
            validate_mhc_y_buffer(29, 2, 3),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn mhc_sinkhorn_validation_rejects_zero_batch() {
        assert!(matches!(
            validate_mhc_y_buffer(0, 0, 3),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn mhc_gemm_rms_scale_validation_accepts_exact_lengths() -> Result<()> {
        let config = MhcGemmRmsScaleConfig {
            rows: 2,
            columns: 7,
            reduction: 4,
            n: 3,
            alpha_pre: 0.5,
            alpha_post: 1.25,
            alpha_res: 2.0,
        };
        assert_eq!(validate_mhc_gemm_rms_scale(14, 2, 8, 28, 7, config)?, 14);
        Ok(())
    }

    #[test]
    fn mhc_gemm_rms_scale_validation_rejects_short_bias() {
        let config = MhcGemmRmsScaleConfig {
            rows: 2,
            columns: 7,
            reduction: 4,
            n: 3,
            alpha_pre: 0.5,
            alpha_post: 1.25,
            alpha_res: 2.0,
        };
        assert!(matches!(
            validate_mhc_gemm_rms_scale(14, 2, 8, 28, 6, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn mhc_gemm_rms_scale_validation_rejects_too_few_columns() {
        let config = MhcGemmRmsScaleConfig {
            rows: 2,
            columns: 5,
            reduction: 4,
            n: 3,
            alpha_pre: 0.5,
            alpha_post: 1.25,
            alpha_res: 2.0,
        };
        assert!(matches!(
            validate_mhc_gemm_rms_scale(10, 2, 8, 20, 5, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn mhc_split_gemm_rms_validation_accepts_exact_temp_lengths() -> Result<()> {
        let config = MhcSplitGemmRmsScaleConfig {
            base: MhcGemmRmsScaleConfig {
                rows: 2,
                columns: 7,
                reduction: 5,
                n: 3,
                alpha_pre: 0.5,
                alpha_post: 1.25,
                alpha_res: 2.0,
            },
            split_k: 3,
        };
        assert_eq!(validate_mhc_split_gemm_rms(42, 42, 10, 35, config)?, 42);
        assert_eq!(
            validate_mhc_finalize_scale_bias_sigmoid(14, 2, 42, 42, 7, config)?,
            14
        );
        Ok(())
    }

    #[test]
    fn mhc_split_gemm_rms_validation_rejects_short_temp() {
        let config = MhcSplitGemmRmsScaleConfig {
            base: MhcGemmRmsScaleConfig {
                rows: 2,
                columns: 7,
                reduction: 5,
                n: 3,
                alpha_pre: 0.5,
                alpha_post: 1.25,
                alpha_res: 2.0,
            },
            split_k: 3,
        };
        assert!(matches!(
            validate_mhc_split_gemm_rms(41, 42, 10, 35, config),
            Err(Error::LengthMismatch)
        ));
        assert!(matches!(
            validate_mhc_finalize_scale_bias_sigmoid(14, 2, 42, 41, 7, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn mhc_split_gemm_rms_validation_rejects_zero_split_k() {
        let config = MhcSplitGemmRmsScaleConfig {
            base: MhcGemmRmsScaleConfig {
                rows: 2,
                columns: 7,
                reduction: 5,
                n: 3,
                alpha_pre: 0.5,
                alpha_post: 1.25,
                alpha_res: 2.0,
            },
            split_k: 0,
        };
        assert!(matches!(
            validate_mhc_split_gemm_rms(42, 42, 10, 35, config),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn rms_norm_silu_mul_validation_rejects_short_up_projection() {
        assert!(matches!(
            validate_rms_norm_silu_mul(24, 24, 6, 23, 4, 6),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn exact_gelu_mul_validation_accepts_exact_lengths() -> Result<()> {
        assert_eq!(validate_binary_fused_lengths(24, 24, 24)?, 24);
        Ok(())
    }

    #[test]
    fn exact_gelu_mul_validation_rejects_short_gate() {
        assert!(matches!(
            validate_binary_fused_lengths(24, 23, 24),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn exact_gelu_mul_validation_rejects_short_up_projection() {
        assert!(matches!(
            validate_binary_fused_lengths(24, 24, 23),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn silu_and_mul_packed_validation_accepts_exact_lengths() -> Result<()> {
        assert_eq!(validate_silu_and_mul_packed_lengths(24, 48, 4, 6)?, 24);
        Ok(())
    }

    #[test]
    fn silu_and_mul_packed_validation_rejects_short_input() {
        assert!(matches!(
            validate_silu_and_mul_packed_lengths(24, 47, 4, 6),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn silu_and_mul_packed_validation_rejects_zero_hidden() {
        assert!(matches!(
            validate_silu_and_mul_packed_lengths(0, 0, 4, 0),
            Err(Error::InvalidLength)
        ));
    }
}

fn validate_kv_cache_sliding_window_append(
    k_cache_len: usize,
    v_cache_len: usize,
    new_k_len: usize,
    new_v_len: usize,
    seq_len: usize,
    config: KvCacheSlidingWindowAppendConfig,
) -> Result<KvCacheSlidingWindowAppendPlan> {
    if config.heads == 0
        || config.head_dim == 0
        || config.max_seq == 0
        || config.window == 0
        || config.cache_len > config.max_seq
        || config.window > config.max_seq
    {
        return Err(Error::InvalidLength);
    }
    let expected_append_start = config
        .cache_start
        .checked_add(config.cache_len)
        .ok_or(Error::SizeOverflow)?;
    if config.append_start != expected_append_start {
        return Err(Error::InvalidLength);
    }
    if seq_len > config.window {
        return Err(Error::LengthMismatch);
    }
    let cache_len = checked_element_count(
        checked_element_count(config.heads, config.max_seq)?,
        config.head_dim,
    )?;
    let new_len = checked_element_count(
        checked_element_count(seq_len, config.heads)?,
        config.head_dim,
    )?;
    ensure_len(k_cache_len, cache_len)?;
    ensure_len(v_cache_len, cache_len)?;
    ensure_len(new_k_len, new_len)?;
    ensure_len(new_v_len, new_len)?;
    let appended_end = config
        .append_start
        .checked_add(seq_len)
        .ok_or(Error::SizeOverflow)?;
    let total_len = config
        .cache_len
        .checked_add(seq_len)
        .ok_or(Error::SizeOverflow)?;
    let retained_len = total_len.min(config.window);
    let new_cache_start = appended_end
        .checked_sub(retained_len)
        .ok_or(Error::SizeOverflow)?;
    let retained_existing_start = new_cache_start.max(config.cache_start);
    let retained_existing_len = config
        .append_start
        .checked_sub(retained_existing_start)
        .ok_or(Error::SizeOverflow)?;
    let pre_compact_source_start = retained_existing_start
        .checked_sub(config.cache_start)
        .ok_or(Error::SizeOverflow)?;
    let needs_pre_compact = total_len > config.max_seq && retained_existing_len > 0;
    let pre_compact = if needs_pre_compact {
        Some(KvCacheCompactConfig {
            heads: config.heads,
            head_dim: config.head_dim,
            max_seq: config.max_seq,
            source_start: pre_compact_source_start,
            token_count: retained_existing_len,
        })
    } else {
        None
    };
    let append_position = if retained_existing_len == 0 {
        0
    } else if needs_pre_compact {
        retained_existing_len
    } else {
        config.cache_len
    };
    let compacted_len = append_position
        .checked_add(seq_len)
        .ok_or(Error::SizeOverflow)?;
    if compacted_len > config.max_seq {
        return Err(Error::LengthMismatch);
    }
    let post_compact = if compacted_len > retained_len {
        Some(KvCacheCompactConfig {
            heads: config.heads,
            head_dim: config.head_dim,
            max_seq: config.max_seq,
            source_start: compacted_len - retained_len,
            token_count: retained_len,
        })
    } else {
        None
    };
    Ok(KvCacheSlidingWindowAppendPlan {
        pre_compact,
        append_position,
        post_compact,
        state: KvCacheSlidingWindowAppendState {
            cache_start: new_cache_start,
            cache_len: retained_len,
        },
    })
}

macro_rules! kv_cache_update_dynpos_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            k_cache: &mut impl DeviceSliceMut<$ty>,
            v_cache: &mut impl DeviceSliceMut<$ty>,
            new_k: &impl DeviceSlice<$ty>,
            new_v: &impl DeviceSlice<$ty>,
            position_start: &impl DeviceSlice<u32>,
            seq_len: usize,
            heads: usize,
            head_dim: usize,
            max_seq: usize,
        ) -> Result<()> {
            let new_len = checked_element_count(checked_element_count(seq_len, heads)?, head_dim)?;
            let cache_len =
                checked_element_count(checked_element_count(heads, max_seq)?, head_dim)?;
            ensure_len(new_k.len(), new_len)?;
            ensure_len(new_v.len(), new_len)?;
            ensure_len(k_cache.len(), cache_len)?;
            ensure_len(v_cache.len(), cache_len)?;
            ensure_len(position_start.len(), 1)?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(k_cache),
                output_pointer(v_cache),
                input_pointer(new_k),
                input_pointer(new_v),
                input_pointer(position_start),
                seq_len,
                heads,
                head_dim,
                max_seq,
            )
        }
    };
}

#[cfg(all(feature = "dtype-f16", feature = "dtype-u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_dynpos_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_dynpos_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_dynpos_f64, f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_seq_dynpos_f16, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_seq_dynpos_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_seq_dynpos_f64, f64);

macro_rules! rope_qk_fn {
    ($name:ident, $target:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            q_out: &mut impl DeviceSliceMut<$ty>,
            k_out: &mut impl DeviceSliceMut<$ty>,
            q_input: &impl DeviceSlice<$ty>,
            k_input: &impl DeviceSlice<$ty>,
            cos: &impl DeviceSlice<$ty>,
            sin: &impl DeviceSlice<$ty>,
            config: QkRotaryEmbeddingConfig,
        ) -> Result<()> {
            positional::$target(stream, q_out, k_out, q_input, k_input, cos, sin, config)
        }
    };
}

#[cfg(feature = "dtype-f16")]
rope_qk_fn!(rope_qk_f16, rotary_embedding_qk_f16, f16);
#[cfg(feature = "dtype-f32")]
rope_qk_fn!(rope_qk_f32, rotary_embedding_qk_f32, f32);
#[cfg(feature = "dtype-f64")]
rope_qk_fn!(rope_qk_f64, rotary_embedding_qk_f64, f64);

macro_rules! rope_qk_cache_update_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            q_out: &mut impl DeviceSliceMut<$ty>,
            k_out: &mut impl DeviceSliceMut<$ty>,
            k_cache: &mut impl DeviceSliceMut<$ty>,
            q_input: &impl DeviceSlice<$ty>,
            k_input: &impl DeviceSlice<$ty>,
            cos: &impl DeviceSlice<$ty>,
            sin: &impl DeviceSlice<$ty>,
            config: RopeQkCacheUpdateConfig,
        ) -> Result<()> {
            validate_rope_qk_cache_update(
                q_out.len(),
                k_out.len(),
                k_cache.len(),
                q_input.len(),
                k_input.len(),
                cos.len(),
                sin.len(),
                config,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::fused::$name(
                &stream,
                output_pointer(q_out),
                output_pointer(k_out),
                output_pointer(k_cache),
                input_pointer(q_input),
                input_pointer(k_input),
                input_pointer(cos),
                input_pointer(sin),
                config,
            )
        }
    };
}

#[cfg(feature = "dtype-f16")]
rope_qk_cache_update_fn!(rope_qk_cache_update_f16, f16);
#[cfg(feature = "dtype-f32")]
rope_qk_cache_update_fn!(rope_qk_cache_update_f32, f32);
#[cfg(feature = "dtype-f64")]
rope_qk_cache_update_fn!(rope_qk_cache_update_f64, f64);

fn validate_rope_qk_cache_update(
    q_out_len: usize,
    k_out_len: usize,
    k_cache_len: usize,
    q_input_len: usize,
    k_input_len: usize,
    cos_len: usize,
    sin_len: usize,
    config: RopeQkCacheUpdateConfig,
) -> Result<()> {
    let rotary = config.rotary;
    let q_output_len = checked_rank4_len(rotary.q_output_dimensions)?;
    let k_output_len = checked_rank4_len(rotary.k_output_dimensions)?;
    ensure_len(q_out_len, q_output_len)?;
    ensure_len(k_out_len, k_output_len)?;
    ensure_rank4_reach(
        q_input_len,
        rotary.q_output_dimensions,
        rotary.q_input_strides,
    )?;
    ensure_rank4_reach(
        k_input_len,
        rotary.k_output_dimensions,
        rotary.k_input_strides,
    )?;
    if rotary.q_output_dimensions[0] != rotary.k_output_dimensions[0]
        || rotary.q_output_dimensions[2] != rotary.k_output_dimensions[2]
        || rotary.q_output_dimensions[3] != rotary.k_output_dimensions[3]
    {
        return Err(Error::LengthMismatch);
    }
    let cache_end = config
        .cache_position_start
        .checked_add(rotary.k_output_dimensions[2])
        .ok_or(Error::SizeOverflow)?;
    if cache_end > config.cache_max_seq {
        return Err(Error::LengthMismatch);
    }
    let cache_len = checked_rank4_len([
        rotary.k_output_dimensions[0],
        rotary.k_output_dimensions[1],
        config.cache_max_seq,
        rotary.k_output_dimensions[3],
    ])?;
    ensure_len(k_cache_len, cache_len)?;
    if rotary.rotary_pairs == 0 {
        return Ok(());
    }
    if rotary
        .rotary_pairs
        .checked_mul(2)
        .ok_or(Error::SizeOverflow)?
        > rotary.q_output_dimensions[3]
    {
        return Err(Error::LengthMismatch);
    }
    let trig_dimensions = [rotary.q_output_dimensions[2], rotary.rotary_pairs];
    ensure_rank2_reach(cos_len, trig_dimensions, rotary.cos_strides)?;
    ensure_rank2_reach(sin_len, trig_dimensions, rotary.sin_strides)
}

fn validate_bshd_rope_qk_cache_update(
    q_out_len: usize,
    k_out_len: usize,
    k_cache_len: usize,
    v_cache_len: usize,
    q_input_len: usize,
    k_input_len: usize,
    v_input_len: usize,
    cos_len: usize,
    sin_len: usize,
    config: BshdRopeQkCacheUpdateConfig,
) -> Result<()> {
    if config.seq_len == 0
        || config.query_heads == 0
        || config.key_value_heads == 0
        || config.head_dim == 0
        || config.cache_max_seq == 0
        || config.head_dim % 2 != 0
    {
        return Err(Error::InvalidLength);
    }
    let cache_end = config
        .cache_position_start
        .checked_add(config.seq_len)
        .ok_or(Error::SizeOverflow)?;
    if cache_end > config.cache_max_seq {
        return Err(Error::LengthMismatch);
    }
    let q_len = checked_element_count(
        checked_element_count(config.seq_len, config.query_heads)?,
        config.head_dim,
    )?;
    let kv_len = checked_element_count(
        checked_element_count(config.seq_len, config.key_value_heads)?,
        config.head_dim,
    )?;
    let cache_len = checked_element_count(
        checked_element_count(config.key_value_heads, config.cache_max_seq)?,
        config.head_dim,
    )?;
    let trig_len = checked_element_count(config.seq_len, config.head_dim / 2)?;
    ensure_len(q_out_len, q_len)?;
    ensure_len(q_input_len, q_len)?;
    ensure_len(k_out_len, kv_len)?;
    ensure_len(k_input_len, kv_len)?;
    ensure_len(v_input_len, kv_len)?;
    ensure_len(k_cache_len, cache_len)?;
    ensure_len(v_cache_len, cache_len)?;
    ensure_len(cos_len, trig_len)?;
    ensure_len(sin_len, trig_len)
}
