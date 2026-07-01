//! RoPE and positional embedding variants.

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
    cuda::interop::{borrowed_stream, input_pointer, output_pointer},
    error::{Error, Result},
    utility::{
        checked_element_count, checked_rank4_len, ensure_len, ensure_len_at_least,
        ensure_rank2_reach, ensure_rank3_reach, ensure_rank4_reach,
    },
};

pub use crate::cpu::positional::{
    BnsdQkRopeConfig, InterleavedComplexRopeConfig, MultiaxisRopeConfig,
};

#[derive(Debug, Clone, Copy)]
/// Generic interleaved-pair RoPE config.
///
/// This convention rotates adjacent dimensions `(0, 1)`, `(2, 3)`, and so on.
/// It is not the half-split `rotate_half` convention.
pub struct RotaryEmbeddingConfig {
    pub output_dimensions: [usize; 4],
    pub input_strides: [usize; 4],
    pub cos_strides: [usize; 2],
    pub sin_strides: [usize; 2],
    pub rotary_pairs: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct BatchedRotaryEmbeddingConfig {
    pub output_dimensions: [usize; 4],
    pub input_strides: [usize; 4],
    pub cos_strides: [usize; 3],
    pub sin_strides: [usize; 3],
    pub trig_batch: usize,
    pub rotary_pairs: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct QkRotaryEmbeddingConfig {
    pub q_output_dimensions: [usize; 4],
    pub k_output_dimensions: [usize; 4],
    pub q_input_strides: [usize; 4],
    pub k_input_strides: [usize; 4],
    pub cos_strides: [usize; 2],
    pub sin_strides: [usize; 2],
    pub rotary_pairs: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct QkBatchedRotaryEmbeddingConfig {
    pub q_output_dimensions: [usize; 4],
    pub k_output_dimensions: [usize; 4],
    pub q_input_strides: [usize; 4],
    pub k_input_strides: [usize; 4],
    pub cos_strides: [usize; 3],
    pub sin_strides: [usize; 3],
    pub trig_batch: usize,
    pub rotary_pairs: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// BSHD half-split RoPE config.
///
/// This convention rotates dimensions from the first half of the head with the
/// corresponding dimensions in the second half: `(0, head_dim / 2)`, `(1, head_dim / 2 + 1)`, and so on.
pub struct BshdRopeConfig {
    pub batch: usize,
    pub seq_len: usize,
    pub heads: usize,
    pub head_dim: usize,
}

macro_rules! rotary_embedding_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            cos: &impl DeviceSlice<$ty>,
            sin: &impl DeviceSlice<$ty>,
            config: RotaryEmbeddingConfig,
        ) -> Result<()> {
            validate_rotary(out.len(), input.len(), cos.len(), sin.len(), config)?;
            let stream = borrowed_stream(stream)?;
            cutile::positional::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(cos),
                input_pointer(sin),
                config,
            )
        }
    };
}

macro_rules! rotary_embedding_positioned_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            cos: &impl DeviceSlice<$ty>,
            sin: &impl DeviceSlice<$ty>,
            config: RotaryEmbeddingConfig,
            position_start: usize,
        ) -> Result<()> {
            validate_rotary_positioned(
                out.len(),
                input.len(),
                cos.len(),
                sin.len(),
                config,
                position_start,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::positional::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(cos),
                input_pointer(sin),
                config,
                position_start,
            )
        }
    };
}

macro_rules! rotary_embedding_dynpos_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            cos: &impl DeviceSlice<$ty>,
            sin: &impl DeviceSlice<$ty>,
            position_start: &impl DeviceSlice<u32>,
            config: RotaryEmbeddingConfig,
        ) -> Result<()> {
            validate_rotary_dynpos(
                out.len(),
                input.len(),
                cos.len(),
                sin.len(),
                position_start.len(),
                config,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::positional::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(cos),
                input_pointer(sin),
                input_pointer(position_start),
                config,
            )
        }
    };
}

macro_rules! rotary_embedding_batched_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            cos: &impl DeviceSlice<$ty>,
            sin: &impl DeviceSlice<$ty>,
            config: BatchedRotaryEmbeddingConfig,
        ) -> Result<()> {
            validate_rotary_batched(out.len(), input.len(), cos.len(), sin.len(), config)?;
            let stream = borrowed_stream(stream)?;
            cutile::positional::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(cos),
                input_pointer(sin),
                config,
            )
        }
    };
}

macro_rules! rotary_embedding_qk_fn {
    ($name:ident, $ty:ty) => {
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
            validate_rotary_qk(
                q_out.len(),
                k_out.len(),
                q_input.len(),
                k_input.len(),
                cos.len(),
                sin.len(),
                config,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::positional::$name(
                &stream,
                output_pointer(q_out),
                output_pointer(k_out),
                input_pointer(q_input),
                input_pointer(k_input),
                input_pointer(cos),
                input_pointer(sin),
                config,
            )
        }
    };
}

macro_rules! rotary_embedding_qk_batched_fn {
    ($name:ident, $target:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            q_out: &mut impl DeviceSliceMut<$ty>,
            k_out: &mut impl DeviceSliceMut<$ty>,
            q_input: &impl DeviceSlice<$ty>,
            k_input: &impl DeviceSlice<$ty>,
            cos: &impl DeviceSlice<$ty>,
            sin: &impl DeviceSlice<$ty>,
            config: QkBatchedRotaryEmbeddingConfig,
        ) -> Result<()> {
            validate_rotary_qk_batched(
                q_out.len(),
                k_out.len(),
                q_input.len(),
                k_input.len(),
                cos.len(),
                sin.len(),
                config,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::positional::$target(
                &stream,
                output_pointer(q_out),
                output_pointer(k_out),
                input_pointer(q_input),
                input_pointer(k_input),
                input_pointer(cos),
                input_pointer(sin),
                config,
            )
        }
    };
}

macro_rules! rotary_embedding_qk_batched_in_place_fn {
    ($name:ident, $target:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            q: &mut impl DeviceSliceMut<$ty>,
            k: &mut impl DeviceSliceMut<$ty>,
            cos: &impl DeviceSlice<$ty>,
            sin: &impl DeviceSlice<$ty>,
            config: QkBatchedRotaryEmbeddingConfig,
        ) -> Result<()> {
            validate_rotary_qk_batched(
                q.len(),
                k.len(),
                q.len(),
                k.len(),
                cos.len(),
                sin.len(),
                config,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::positional::$target(
                &stream,
                output_pointer(q),
                output_pointer(k),
                input_pointer(cos),
                input_pointer(sin),
                config,
            )
        }
    };
}

macro_rules! rotary_embedding_qk_in_place_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            q: &mut impl DeviceSliceMut<$ty>,
            k: &mut impl DeviceSliceMut<$ty>,
            cos: &impl DeviceSlice<$ty>,
            sin: &impl DeviceSlice<$ty>,
            config: QkRotaryEmbeddingConfig,
        ) -> Result<()> {
            validate_rotary_qk(
                q.len(),
                k.len(),
                q.len(),
                k.len(),
                cos.len(),
                sin.len(),
                config,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::positional::$name(
                &stream,
                output_pointer(q),
                output_pointer(k),
                input_pointer(cos),
                input_pointer(sin),
                config,
            )
        }
    };
}

macro_rules! rope_bshd_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            cos: &impl DeviceSlice<f32>,
            sin: &impl DeviceSlice<f32>,
            config: BshdRopeConfig,
        ) -> Result<()> {
            validate_rope_bshd(out.len(), input.len(), cos.len(), sin.len(), config)?;
            let stream = borrowed_stream(stream)?;
            cutile::positional::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(cos),
                input_pointer(sin),
                config,
            )
        }
    };
}

#[cfg(feature = "dtype-bf16")]

pub fn rotary_embedding_qk_bnsd_bf16(
    stream: &Stream,
    q_out: &mut impl DeviceSliceMut<bf16>,
    k_out: &mut impl DeviceSliceMut<bf16>,
    q_input: &impl DeviceSlice<bf16>,
    k_input: &impl DeviceSlice<bf16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    validate_rope_qk_bnsd(
        q_out.len(),
        k_out.len(),
        q_input.len(),
        k_input.len(),
        cos.len(),
        sin.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::positional::rotary_embedding_qk_bnsd_bf16(
        &stream,
        output_pointer(q_out),
        output_pointer(k_out),
        input_pointer(q_input),
        input_pointer(k_input),
        input_pointer(cos),
        input_pointer(sin),
        config,
    )
}

#[cfg(feature = "dtype-f32")]

pub fn rotary_embedding_qk_bnsd_f32(
    stream: &Stream,
    q_out: &mut impl DeviceSliceMut<f32>,
    k_out: &mut impl DeviceSliceMut<f32>,
    q_input: &impl DeviceSlice<f32>,
    k_input: &impl DeviceSlice<f32>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    validate_rope_qk_bnsd(
        q_out.len(),
        k_out.len(),
        q_input.len(),
        k_input.len(),
        cos.len(),
        sin.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::positional::rotary_embedding_qk_bnsd_f32(
        &stream,
        output_pointer(q_out),
        output_pointer(k_out),
        input_pointer(q_input),
        input_pointer(k_input),
        input_pointer(cos),
        input_pointer(sin),
        config,
    )
}

#[cfg(feature = "dtype-f16")]

pub fn rotary_embedding_qk_bnsd_f16(
    stream: &Stream,
    q_out: &mut impl DeviceSliceMut<f16>,
    k_out: &mut impl DeviceSliceMut<f16>,
    q_input: &impl DeviceSlice<f16>,
    k_input: &impl DeviceSlice<f16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    validate_rope_qk_bnsd(
        q_out.len(),
        k_out.len(),
        q_input.len(),
        k_input.len(),
        cos.len(),
        sin.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::positional::rotary_embedding_qk_bnsd_f16(
        &stream,
        output_pointer(q_out),
        output_pointer(k_out),
        input_pointer(q_input),
        input_pointer(k_input),
        input_pointer(cos),
        input_pointer(sin),
        config,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn rotary_embedding_qk_bnsd_in_place_f32(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<f32>,
    k: &mut impl DeviceSliceMut<f32>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    validate_rope_qk_bnsd(
        q.len(),
        k.len(),
        q.len(),
        k.len(),
        cos.len(),
        sin.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::positional::rotary_embedding_qk_bnsd_in_place_f32(
        &stream,
        output_pointer(q),
        output_pointer(k),
        input_pointer(cos),
        input_pointer(sin),
        config,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn rotary_embedding_qk_bnsd_in_place_f16(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<f16>,
    k: &mut impl DeviceSliceMut<f16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    validate_rope_qk_bnsd(
        q.len(),
        k.len(),
        q.len(),
        k.len(),
        cos.len(),
        sin.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::positional::rotary_embedding_qk_bnsd_in_place_f16(
        &stream,
        output_pointer(q),
        output_pointer(k),
        input_pointer(cos),
        input_pointer(sin),
        config,
    )
}

#[cfg(feature = "dtype-bf16")]
pub fn rotary_embedding_qk_bnsd_in_place_bf16(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<bf16>,
    k: &mut impl DeviceSliceMut<bf16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    validate_rope_qk_bnsd(
        q.len(),
        k.len(),
        q.len(),
        k.len(),
        cos.len(),
        sin.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::positional::rotary_embedding_qk_bnsd_in_place_bf16(
        &stream,
        output_pointer(q),
        output_pointer(k),
        input_pointer(cos),
        input_pointer(sin),
        config,
    )
}

#[cfg(feature = "dtype-bf16")]
/// BNSD Q/K half-split RoPE.
///
/// Rotates dimensions `(pair, rotary_pairs + pair)`, matching half-split
/// `rotate_half` layouts rather than adjacent interleaved pairs.
pub fn rope_embedding_qk_bnsd_bf16(
    stream: &Stream,
    q_out: &mut impl DeviceSliceMut<bf16>,
    k_out: &mut impl DeviceSliceMut<bf16>,
    q_input: &impl DeviceSlice<bf16>,
    k_input: &impl DeviceSlice<bf16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    rotary_embedding_qk_bnsd_bf16(stream, q_out, k_out, q_input, k_input, cos, sin, config)
}

#[cfg(feature = "dtype-f32")]
/// BNSD Q/K half-split RoPE.
///
/// Rotates dimensions `(pair, rotary_pairs + pair)`, matching half-split
/// `rotate_half` layouts rather than adjacent interleaved pairs.
pub fn rope_embedding_qk_bnsd_f32(
    stream: &Stream,
    q_out: &mut impl DeviceSliceMut<f32>,
    k_out: &mut impl DeviceSliceMut<f32>,
    q_input: &impl DeviceSlice<f32>,
    k_input: &impl DeviceSlice<f32>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    rotary_embedding_qk_bnsd_f32(stream, q_out, k_out, q_input, k_input, cos, sin, config)
}

#[cfg(feature = "dtype-f16")]
/// BNSD Q/K half-split RoPE.
///
/// Rotates dimensions `(pair, rotary_pairs + pair)`, matching half-split
/// `rotate_half` layouts rather than adjacent interleaved pairs.
pub fn rope_embedding_qk_bnsd_f16(
    stream: &Stream,
    q_out: &mut impl DeviceSliceMut<f16>,
    k_out: &mut impl DeviceSliceMut<f16>,
    q_input: &impl DeviceSlice<f16>,
    k_input: &impl DeviceSlice<f16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    rotary_embedding_qk_bnsd_f16(stream, q_out, k_out, q_input, k_input, cos, sin, config)
}

#[cfg(feature = "dtype-bf16")]
pub fn rope_qk_half_split_bnsd_bf16(
    stream: &Stream,
    q_out: &mut impl DeviceSliceMut<bf16>,
    k_out: &mut impl DeviceSliceMut<bf16>,
    q_input: &impl DeviceSlice<bf16>,
    k_input: &impl DeviceSlice<bf16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    rotary_embedding_qk_bnsd_bf16(stream, q_out, k_out, q_input, k_input, cos, sin, config)
}

#[cfg(feature = "dtype-f32")]

pub fn rope_qk_half_split_bnsd_f32(
    stream: &Stream,
    q_out: &mut impl DeviceSliceMut<f32>,
    k_out: &mut impl DeviceSliceMut<f32>,
    q_input: &impl DeviceSlice<f32>,
    k_input: &impl DeviceSlice<f32>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    rotary_embedding_qk_bnsd_f32(stream, q_out, k_out, q_input, k_input, cos, sin, config)
}

#[cfg(feature = "dtype-f16")]

pub fn rope_qk_half_split_bnsd_f16(
    stream: &Stream,
    q_out: &mut impl DeviceSliceMut<f16>,
    k_out: &mut impl DeviceSliceMut<f16>,
    q_input: &impl DeviceSlice<f16>,
    k_input: &impl DeviceSlice<f16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    rotary_embedding_qk_bnsd_f16(stream, q_out, k_out, q_input, k_input, cos, sin, config)
}

#[cfg(feature = "dtype-f32")]
pub fn rope_embedding_qk_bnsd_in_place_f32(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<f32>,
    k: &mut impl DeviceSliceMut<f32>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    rotary_embedding_qk_bnsd_in_place_f32(stream, q, k, cos, sin, config)
}

#[cfg(feature = "dtype-f16")]
pub fn rope_embedding_qk_bnsd_in_place_f16(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<f16>,
    k: &mut impl DeviceSliceMut<f16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    rotary_embedding_qk_bnsd_in_place_f16(stream, q, k, cos, sin, config)
}

#[cfg(feature = "dtype-bf16")]
pub fn rope_embedding_qk_bnsd_in_place_bf16(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<bf16>,
    k: &mut impl DeviceSliceMut<bf16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    rotary_embedding_qk_bnsd_in_place_bf16(stream, q, k, cos, sin, config)
}

#[cfg(feature = "dtype-f32")]
pub fn rope_qk_half_split_bnsd_in_place_f32(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<f32>,
    k: &mut impl DeviceSliceMut<f32>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    rotary_embedding_qk_bnsd_in_place_f32(stream, q, k, cos, sin, config)
}

#[cfg(feature = "dtype-f16")]
pub fn rope_qk_half_split_bnsd_in_place_f16(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<f16>,
    k: &mut impl DeviceSliceMut<f16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    rotary_embedding_qk_bnsd_in_place_f16(stream, q, k, cos, sin, config)
}

#[cfg(feature = "dtype-bf16")]
pub fn rope_qk_half_split_bnsd_in_place_bf16(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<bf16>,
    k: &mut impl DeviceSliceMut<bf16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    rotary_embedding_qk_bnsd_in_place_bf16(stream, q, k, cos, sin, config)
}

#[cfg(feature = "dtype-f32")]
rotary_embedding_fn!(rotary_embedding_f32, f32);
#[cfg(feature = "dtype-f16")]
rotary_embedding_fn!(rotary_embedding_f16, f16);
#[cfg(feature = "dtype-f64")]
rotary_embedding_fn!(rotary_embedding_f64, f64);
#[cfg(feature = "dtype-f32")]
rotary_embedding_positioned_fn!(rotary_embedding_positioned_f32, f32);
#[cfg(feature = "dtype-f16")]
rotary_embedding_positioned_fn!(rotary_embedding_positioned_f16, f16);
#[cfg(feature = "dtype-f64")]
rotary_embedding_positioned_fn!(rotary_embedding_positioned_f64, f64);
#[cfg(feature = "dtype-f32")]
rotary_embedding_dynpos_fn!(rotary_embedding_dynpos_f32, f32);
#[cfg(feature = "dtype-f16")]
rotary_embedding_dynpos_fn!(rotary_embedding_dynpos_f16, f16);
#[cfg(feature = "dtype-f64")]
rotary_embedding_dynpos_fn!(rotary_embedding_dynpos_f64, f64);
#[cfg(feature = "dtype-f32")]
rotary_embedding_batched_fn!(rotary_embedding_batched_f32, f32);
#[cfg(feature = "dtype-f16")]
rotary_embedding_batched_fn!(rotary_embedding_batched_f16, f16);
#[cfg(feature = "dtype-f64")]
rotary_embedding_batched_fn!(rotary_embedding_batched_f64, f64);
#[cfg(feature = "dtype-f32")]
rotary_embedding_fn!(rotary_embedding_chunked_f32, f32);
#[cfg(feature = "dtype-f16")]
rotary_embedding_fn!(rotary_embedding_chunked_f16, f16);
#[cfg(feature = "dtype-f64")]
rotary_embedding_fn!(rotary_embedding_chunked_f64, f64);

#[cfg(feature = "dtype-f32")]
rotary_embedding_positioned_fn!(rotary_embedding_chunked_positioned_f32, f32);
#[cfg(feature = "dtype-f16")]
rotary_embedding_positioned_fn!(rotary_embedding_chunked_positioned_f16, f16);
#[cfg(feature = "dtype-f64")]
rotary_embedding_positioned_fn!(rotary_embedding_chunked_positioned_f64, f64);
#[cfg(feature = "dtype-f32")]
rotary_embedding_dynpos_fn!(rotary_embedding_chunked_dynpos_f32, f32);
#[cfg(feature = "dtype-f16")]
rotary_embedding_dynpos_fn!(rotary_embedding_chunked_dynpos_f16, f16);
#[cfg(feature = "dtype-f64")]
rotary_embedding_dynpos_fn!(rotary_embedding_chunked_dynpos_f64, f64);
#[cfg(feature = "dtype-f32")]
rotary_embedding_batched_fn!(rotary_embedding_chunked_batched_f32, f32);
#[cfg(feature = "dtype-f16")]
rotary_embedding_batched_fn!(rotary_embedding_chunked_batched_f16, f16);
#[cfg(feature = "dtype-f64")]
rotary_embedding_batched_fn!(rotary_embedding_chunked_batched_f64, f64);
#[cfg(feature = "dtype-f32")]
rotary_embedding_qk_fn!(rotary_embedding_qk_f32, f32);
#[cfg(feature = "dtype-f16")]
rotary_embedding_qk_fn!(rotary_embedding_qk_f16, f16);
#[cfg(feature = "dtype-f64")]
rotary_embedding_qk_fn!(rotary_embedding_qk_f64, f64);
#[cfg(feature = "dtype-f32")]
rotary_embedding_qk_batched_fn!(
    rotary_embedding_qk_batched_f32,
    rotary_embedding_qk_batched_f32,
    f32
);
#[cfg(feature = "dtype-f16")]
rotary_embedding_qk_batched_fn!(
    rotary_embedding_qk_batched_f16,
    rotary_embedding_qk_batched_f16,
    f16
);
#[cfg(feature = "dtype-f64")]
rotary_embedding_qk_batched_fn!(
    rotary_embedding_qk_batched_f64,
    rotary_embedding_qk_batched_f64,
    f64
);
#[cfg(feature = "dtype-f32")]
rotary_embedding_qk_batched_in_place_fn!(
    rotary_embedding_qk_batched_in_place_f32,
    rotary_embedding_qk_batched_in_place_f32,
    f32
);
#[cfg(feature = "dtype-f16")]
rotary_embedding_qk_batched_in_place_fn!(
    rotary_embedding_qk_batched_in_place_f16,
    rotary_embedding_qk_batched_in_place_f16,
    f16
);
#[cfg(feature = "dtype-f64")]
rotary_embedding_qk_batched_in_place_fn!(
    rotary_embedding_qk_batched_in_place_f64,
    rotary_embedding_qk_batched_in_place_f64,
    f64
);
#[cfg(feature = "dtype-f32")]
rotary_embedding_qk_in_place_fn!(rotary_embedding_qk_in_place_f32, f32);
#[cfg(feature = "dtype-f16")]
rotary_embedding_qk_in_place_fn!(rotary_embedding_qk_in_place_f16, f16);
#[cfg(feature = "dtype-f64")]
rotary_embedding_qk_in_place_fn!(rotary_embedding_qk_in_place_f64, f64);

#[cfg(feature = "dtype-f32")]
rope_bshd_fn!(rope_bshd_f32, f32);
#[cfg(feature = "dtype-f16")]
rope_bshd_fn!(rope_bshd_f16, f16);
#[cfg(feature = "dtype-bf16")]
rope_bshd_fn!(rope_bshd_bf16, bf16);

#[cfg(feature = "dtype-f16")]
pub fn rope_bshd_dynpos_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    input: &impl DeviceSlice<f16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    position_start: &impl DeviceSlice<u32>,
    config: BshdRopeConfig,
) -> Result<()> {
    if config.batch == 0
        || config.seq_len == 0
        || config.heads == 0
        || config.head_dim == 0
        || !config.head_dim.is_multiple_of(2)
    {
        return Err(Error::InvalidLength);
    }
    let tensor_len = checked_element_count(
        checked_element_count(
            checked_element_count(config.batch, config.seq_len)?,
            config.heads,
        )?,
        config.head_dim,
    )?;
    ensure_len(out.len(), tensor_len)?;
    ensure_len(input.len(), tensor_len)?;
    ensure_len_at_least(cos.len(), config.head_dim / 2)?;
    ensure_len_at_least(sin.len(), config.head_dim / 2)?;
    ensure_len(position_start.len(), 1)?;
    let stream = borrowed_stream(stream)?;
    cutile::positional::rope_bshd_dynpos_f16(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(cos),
        input_pointer(sin),
        input_pointer(position_start),
        config,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn multiaxis_rope_qk_f32(
    stream: &Stream,
    query: &mut impl DeviceSliceMut<f32>,
    key: &mut impl DeviceSliceMut<f32>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: MultiaxisRopeConfig,
) -> Result<()> {
    validate_multiaxis_rope_qk(query.len(), key.len(), cos.len(), sin.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::positional::multiaxis_rope_qk_f32(
        &stream,
        output_pointer(query),
        output_pointer(key),
        input_pointer(cos),
        input_pointer(sin),
        config,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn multiaxis_rope_qk_f16(
    stream: &Stream,
    query: &mut impl DeviceSliceMut<f16>,
    key: &mut impl DeviceSliceMut<f16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: MultiaxisRopeConfig,
) -> Result<()> {
    validate_multiaxis_rope_qk(query.len(), key.len(), cos.len(), sin.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::positional::multiaxis_rope_qk_f16(
        &stream,
        output_pointer(query),
        output_pointer(key),
        input_pointer(cos),
        input_pointer(sin),
        config,
    )
}

#[cfg(feature = "dtype-bf16")]
pub fn multiaxis_rope_qk_bf16(
    stream: &Stream,
    query: &mut impl DeviceSliceMut<bf16>,
    key: &mut impl DeviceSliceMut<bf16>,
    cos: &impl DeviceSlice<f32>,
    sin: &impl DeviceSlice<f32>,
    config: MultiaxisRopeConfig,
) -> Result<()> {
    validate_multiaxis_rope_qk(query.len(), key.len(), cos.len(), sin.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::positional::multiaxis_rope_qk_bf16(
        &stream,
        output_pointer(query),
        output_pointer(key),
        input_pointer(cos),
        input_pointer(sin),
        config,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn interleaved_complex_rope_qk_f32(
    stream: &Stream,
    query: &mut impl DeviceSliceMut<f32>,
    key: &mut impl DeviceSliceMut<f32>,
    freqs: &impl DeviceSlice<f32>,
    config: InterleavedComplexRopeConfig,
) -> Result<()> {
    validate_interleaved_complex_rope_qk(query.len(), key.len(), freqs.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::positional::interleaved_complex_rope_qk_f32(
        &stream,
        output_pointer(query),
        output_pointer(key),
        input_pointer(freqs),
        config,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn interleaved_complex_rope_qk_f16(
    stream: &Stream,
    query: &mut impl DeviceSliceMut<f16>,
    key: &mut impl DeviceSliceMut<f16>,
    freqs: &impl DeviceSlice<f32>,
    config: InterleavedComplexRopeConfig,
) -> Result<()> {
    validate_interleaved_complex_rope_qk(query.len(), key.len(), freqs.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::positional::interleaved_complex_rope_qk_f16(
        &stream,
        output_pointer(query),
        output_pointer(key),
        input_pointer(freqs),
        config,
    )
}

#[cfg(feature = "dtype-bf16")]
pub fn interleaved_complex_rope_qk_bf16(
    stream: &Stream,
    query: &mut impl DeviceSliceMut<bf16>,
    key: &mut impl DeviceSliceMut<bf16>,
    freqs: &impl DeviceSlice<f32>,
    config: InterleavedComplexRopeConfig,
) -> Result<()> {
    validate_interleaved_complex_rope_qk(query.len(), key.len(), freqs.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::positional::interleaved_complex_rope_qk_bf16(
        &stream,
        output_pointer(query),
        output_pointer(key),
        input_pointer(freqs),
        config,
    )
}

fn validate_rotary(
    out_len: usize,
    input_len: usize,
    cos_len: usize,
    sin_len: usize,
    config: RotaryEmbeddingConfig,
) -> Result<()> {
    let output_len = checked_rank4_len(config.output_dimensions)?;
    ensure_len(out_len, output_len)?;
    ensure_rank4_reach(input_len, config.output_dimensions, config.input_strides)?;
    if config.rotary_pairs == 0 {
        return Ok(());
    }
    let head_dim = config.output_dimensions[3];
    if config
        .rotary_pairs
        .checked_mul(2)
        .ok_or(Error::SizeOverflow)?
        > head_dim
    {
        return Err(Error::LengthMismatch);
    }
    let trig_dimensions = [config.output_dimensions[2], config.rotary_pairs];
    ensure_rank2_reach(cos_len, trig_dimensions, config.cos_strides)?;
    ensure_rank2_reach(sin_len, trig_dimensions, config.sin_strides)
}

fn validate_rotary_positioned(
    out_len: usize,
    input_len: usize,
    cos_len: usize,
    sin_len: usize,
    config: RotaryEmbeddingConfig,
    position_start: usize,
) -> Result<()> {
    let output_len = checked_rank4_len(config.output_dimensions)?;
    ensure_len(out_len, output_len)?;
    ensure_rank4_reach(input_len, config.output_dimensions, config.input_strides)?;
    if config.rotary_pairs == 0 {
        return Ok(());
    }
    let head_dim = config.output_dimensions[3];
    if config
        .rotary_pairs
        .checked_mul(2)
        .ok_or(Error::SizeOverflow)?
        > head_dim
    {
        return Err(Error::LengthMismatch);
    }
    let trig_rows = position_start
        .checked_add(config.output_dimensions[2])
        .ok_or(Error::SizeOverflow)?;
    let trig_dimensions = [trig_rows, config.rotary_pairs];
    ensure_rank2_reach(cos_len, trig_dimensions, config.cos_strides)?;
    ensure_rank2_reach(sin_len, trig_dimensions, config.sin_strides)
}

fn validate_rotary_dynpos(
    out_len: usize,
    input_len: usize,
    cos_len: usize,
    sin_len: usize,
    position_start_len: usize,
    config: RotaryEmbeddingConfig,
) -> Result<()> {
    ensure_len(position_start_len, 1)?;
    validate_rotary(out_len, input_len, cos_len, sin_len, config)
}

fn validate_rotary_batched(
    out_len: usize,
    input_len: usize,
    cos_len: usize,
    sin_len: usize,
    config: BatchedRotaryEmbeddingConfig,
) -> Result<()> {
    let output_len = checked_rank4_len(config.output_dimensions)?;
    ensure_len(out_len, output_len)?;
    ensure_rank4_reach(input_len, config.output_dimensions, config.input_strides)?;
    if config.rotary_pairs == 0 {
        return Ok(());
    }
    let output_batch = config.output_dimensions[0];
    if config.trig_batch != 1 && config.trig_batch != output_batch {
        return Err(Error::LengthMismatch);
    }
    let head_dim = config.output_dimensions[3];
    if config
        .rotary_pairs
        .checked_mul(2)
        .ok_or(Error::SizeOverflow)?
        > head_dim
    {
        return Err(Error::LengthMismatch);
    }
    let trig_dimensions = [
        config.trig_batch,
        config.output_dimensions[2],
        config.rotary_pairs,
    ];
    ensure_rank3_reach(cos_len, trig_dimensions, config.cos_strides)?;
    ensure_rank3_reach(sin_len, trig_dimensions, config.sin_strides)
}

fn validate_rope_bshd(
    out_len: usize,
    input_len: usize,
    cos_len: usize,
    sin_len: usize,
    config: BshdRopeConfig,
) -> Result<()> {
    if config.batch == 0
        || config.seq_len == 0
        || config.heads == 0
        || config.head_dim == 0
        || !config.head_dim.is_multiple_of(2)
    {
        return Err(Error::InvalidLength);
    }
    let tensor_len = checked_element_count(
        checked_element_count(
            checked_element_count(config.batch, config.seq_len)?,
            config.heads,
        )?,
        config.head_dim,
    )?;
    ensure_len(out_len, tensor_len)?;
    ensure_len(input_len, tensor_len)?;
    let trig_len = checked_element_count(config.seq_len, config.head_dim / 2)?;
    ensure_len(cos_len, trig_len)?;
    ensure_len(sin_len, trig_len)
}

fn validate_rope_qk_bnsd(
    q_out_len: usize,
    k_out_len: usize,
    q_input_len: usize,
    k_input_len: usize,
    cos_len: usize,
    sin_len: usize,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    if config.batch == 0
        || config.seq_len == 0
        || config.query_heads == 0
        || config.key_heads == 0
        || config.head_dim == 0
        || config.trig_batch == 0
        || config.rotary_pairs == 0
        || config.trig_batch != 1 && config.trig_batch != config.batch
    {
        return Err(Error::InvalidLength);
    }
    if config
        .rotary_pairs
        .checked_mul(2)
        .ok_or(Error::SizeOverflow)?
        > config.head_dim
    {
        return Err(Error::LengthMismatch);
    }
    let batch_seq = checked_element_count(config.batch, config.seq_len)?;
    let q_len = checked_element_count(
        checked_element_count(batch_seq, config.query_heads)?,
        config.head_dim,
    )?;
    let k_len = checked_element_count(
        checked_element_count(batch_seq, config.key_heads)?,
        config.head_dim,
    )?;
    ensure_len(q_out_len, q_len)?;
    ensure_len(k_out_len, k_len)?;
    ensure_len(q_input_len, q_len)?;
    ensure_len(k_input_len, k_len)?;
    let trig_len = checked_element_count(
        checked_element_count(config.trig_batch, config.seq_len)?,
        config.rotary_pairs,
    )?;
    ensure_len(cos_len, trig_len)?;
    ensure_len(sin_len, trig_len)
}

fn validate_rotary_qk(
    q_out_len: usize,
    k_out_len: usize,
    q_input_len: usize,
    k_input_len: usize,
    cos_len: usize,
    sin_len: usize,
    config: QkRotaryEmbeddingConfig,
) -> Result<()> {
    let q_output_len = checked_rank4_len(config.q_output_dimensions)?;
    let k_output_len = checked_rank4_len(config.k_output_dimensions)?;
    ensure_len(q_out_len, q_output_len)?;
    ensure_len(k_out_len, k_output_len)?;
    ensure_rank4_reach(
        q_input_len,
        config.q_output_dimensions,
        config.q_input_strides,
    )?;
    ensure_rank4_reach(
        k_input_len,
        config.k_output_dimensions,
        config.k_input_strides,
    )?;
    if config.q_output_dimensions[0] != config.k_output_dimensions[0]
        || config.q_output_dimensions[2] != config.k_output_dimensions[2]
        || config.q_output_dimensions[3] != config.k_output_dimensions[3]
    {
        return Err(Error::LengthMismatch);
    }
    if config.rotary_pairs == 0 {
        return Ok(());
    }
    let head_dim = config.q_output_dimensions[3];
    if config
        .rotary_pairs
        .checked_mul(2)
        .ok_or(Error::SizeOverflow)?
        > head_dim
    {
        return Err(Error::LengthMismatch);
    }
    let trig_dimensions = [config.q_output_dimensions[2], config.rotary_pairs];
    ensure_rank2_reach(cos_len, trig_dimensions, config.cos_strides)?;
    ensure_rank2_reach(sin_len, trig_dimensions, config.sin_strides)
}

fn validate_rotary_qk_batched(
    q_out_len: usize,
    k_out_len: usize,
    q_input_len: usize,
    k_input_len: usize,
    cos_len: usize,
    sin_len: usize,
    config: QkBatchedRotaryEmbeddingConfig,
) -> Result<(BatchedRotaryEmbeddingConfig, BatchedRotaryEmbeddingConfig)> {
    let q_config = BatchedRotaryEmbeddingConfig {
        output_dimensions: config.q_output_dimensions,
        input_strides: config.q_input_strides,
        cos_strides: config.cos_strides,
        sin_strides: config.sin_strides,
        trig_batch: config.trig_batch,
        rotary_pairs: config.rotary_pairs,
    };
    let k_config = BatchedRotaryEmbeddingConfig {
        output_dimensions: config.k_output_dimensions,
        input_strides: config.k_input_strides,
        cos_strides: config.cos_strides,
        sin_strides: config.sin_strides,
        trig_batch: config.trig_batch,
        rotary_pairs: config.rotary_pairs,
    };
    if config.q_output_dimensions[0] != config.k_output_dimensions[0]
        || config.q_output_dimensions[2] != config.k_output_dimensions[2]
        || config.q_output_dimensions[3] != config.k_output_dimensions[3]
    {
        return Err(Error::LengthMismatch);
    }
    validate_rotary_batched(q_out_len, q_input_len, cos_len, sin_len, q_config)?;
    validate_rotary_batched(k_out_len, k_input_len, cos_len, sin_len, k_config)?;
    Ok((q_config, k_config))
}

fn validate_multiaxis_rope_qk(
    query_len: usize,
    key_len: usize,
    cos_len: usize,
    sin_len: usize,
    config: MultiaxisRopeConfig,
) -> Result<()> {
    if config.batch == 0
        || config.seq_len == 0
        || config.query_heads == 0
        || config.key_heads == 0
        || config.head_dim == 0
        || !config.head_dim.is_multiple_of(2)
    {
        return Err(Error::InvalidLength);
    }
    let head_dim_half = config.head_dim / 2;
    let section_sum = config
        .section_t
        .checked_add(config.section_h)
        .and_then(|value| value.checked_add(config.section_w))
        .ok_or(Error::SizeOverflow)?;
    if section_sum != head_dim_half {
        return Err(Error::LengthMismatch);
    }
    let batch_seq = checked_element_count(config.batch, config.seq_len)?;
    let query_len_expected = checked_element_count(
        checked_element_count(batch_seq, config.query_heads)?,
        config.head_dim,
    )?;
    let key_len_expected = checked_element_count(
        checked_element_count(batch_seq, config.key_heads)?,
        config.head_dim,
    )?;
    let trig_len = checked_element_count(checked_element_count(3, batch_seq)?, head_dim_half)?;
    ensure_len(query_len, query_len_expected)?;
    ensure_len(key_len, key_len_expected)?;
    ensure_len(cos_len, trig_len)?;
    ensure_len(sin_len, trig_len)
}

fn validate_interleaved_complex_rope_qk(
    query_len: usize,
    key_len: usize,
    freqs_len: usize,
    config: InterleavedComplexRopeConfig,
) -> Result<()> {
    if config.batch == 0
        || config.seq_len == 0
        || config.query_heads == 0
        || config.key_heads == 0
        || config.head_dim == 0
        || !config.head_dim.is_multiple_of(2)
    {
        return Err(Error::InvalidLength);
    }
    let batch_seq = checked_element_count(config.batch, config.seq_len)?;
    let query_len_expected = checked_element_count(
        checked_element_count(batch_seq, config.query_heads)?,
        config.head_dim,
    )?;
    let key_len_expected = checked_element_count(
        checked_element_count(batch_seq, config.key_heads)?,
        config.head_dim,
    )?;
    let freqs_len_expected = checked_element_count(config.seq_len, config.head_dim)?;
    ensure_len(query_len, query_len_expected)?;
    ensure_len(key_len, key_len_expected)?;
    ensure_len(freqs_len, freqs_len_expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qk_batched_rotary_validation_accepts_shared_and_batched_trig() -> Result<()> {
        let mut config = qk_batched_rotary_validation_config();
        validate_rotary_qk_batched(192, 96, 192, 96, 12, 12, config)?;

        config.trig_batch = 2;
        validate_rotary_qk_batched(192, 96, 192, 96, 24, 24, config)?;
        Ok(())
    }

    #[test]
    fn qk_batched_rotary_validation_rejects_bad_trig_batch() {
        let mut config = qk_batched_rotary_validation_config();
        config.trig_batch = 3;
        assert!(matches!(
            validate_rotary_qk_batched(192, 96, 192, 96, 36, 36, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn qk_batched_rotary_validation_rejects_mismatched_qk_shape() {
        let mut config = qk_batched_rotary_validation_config();
        config.k_output_dimensions[2] = 4;
        assert!(matches!(
            validate_rotary_qk_batched(192, 128, 192, 128, 12, 12, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn qk_batched_rotary_validation_rejects_short_batched_trig() {
        let mut config = qk_batched_rotary_validation_config();
        config.trig_batch = 2;
        assert!(matches!(
            validate_rotary_qk_batched(192, 96, 192, 96, 23, 24, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn rope_bshd_validation_accepts_exact_lengths() -> Result<()> {
        let config = BshdRopeConfig {
            batch: 2,
            seq_len: 3,
            heads: 4,
            head_dim: 8,
        };
        validate_rope_bshd(192, 192, 12, 12, config)
    }

    #[test]
    fn rope_bshd_validation_rejects_odd_head_dim() {
        let config = BshdRopeConfig {
            batch: 2,
            seq_len: 3,
            heads: 4,
            head_dim: 7,
        };
        assert!(matches!(
            validate_rope_bshd(168, 168, 9, 9, config),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn rope_bshd_validation_rejects_short_trig() {
        let config = BshdRopeConfig {
            batch: 2,
            seq_len: 3,
            heads: 4,
            head_dim: 8,
        };
        assert!(matches!(
            validate_rope_bshd(192, 192, 11, 12, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn rope_qk_bnsd_validation_accepts_exact_lengths() -> Result<()> {
        let config = BnsdQkRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 4,
            key_heads: 2,
            head_dim: 8,
            trig_batch: 1,
            rotary_pairs: 3,
        };
        validate_rope_qk_bnsd(192, 96, 192, 96, 9, 9, config)
    }

    #[test]
    fn rope_qk_bnsd_validation_rejects_bad_trig_batch() {
        let config = BnsdQkRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 4,
            key_heads: 2,
            head_dim: 8,
            trig_batch: 3,
            rotary_pairs: 3,
        };
        assert!(matches!(
            validate_rope_qk_bnsd(192, 96, 192, 96, 27, 27, config),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn rope_qk_bnsd_validation_rejects_rotary_overflow() {
        let config = BnsdQkRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 4,
            key_heads: 2,
            head_dim: 8,
            trig_batch: 1,
            rotary_pairs: 5,
        };
        assert!(matches!(
            validate_rope_qk_bnsd(192, 96, 192, 96, 15, 15, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn multiaxis_rope_validation_accepts_exact_lengths() -> Result<()> {
        let config = MultiaxisRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 4,
            key_heads: 2,
            head_dim: 8,
            section_t: 1,
            section_h: 2,
            section_w: 1,
        };
        validate_multiaxis_rope_qk(192, 96, 72, 72, config)
    }

    #[test]
    fn multiaxis_rope_validation_rejects_bad_sections() {
        let config = MultiaxisRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 4,
            key_heads: 2,
            head_dim: 8,
            section_t: 1,
            section_h: 1,
            section_w: 1,
        };
        assert!(matches!(
            validate_multiaxis_rope_qk(192, 96, 72, 72, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn multiaxis_rope_validation_rejects_short_trig() {
        let config = MultiaxisRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 4,
            key_heads: 2,
            head_dim: 8,
            section_t: 1,
            section_h: 2,
            section_w: 1,
        };
        assert!(matches!(
            validate_multiaxis_rope_qk(192, 96, 71, 72, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn interleaved_complex_rope_validation_accepts_exact_lengths() -> Result<()> {
        let config = InterleavedComplexRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 4,
            key_heads: 2,
            head_dim: 8,
        };
        validate_interleaved_complex_rope_qk(192, 96, 24, config)
    }

    #[test]
    fn interleaved_complex_rope_validation_rejects_short_freqs() {
        let config = InterleavedComplexRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 4,
            key_heads: 2,
            head_dim: 8,
        };
        assert!(matches!(
            validate_interleaved_complex_rope_qk(192, 96, 23, config),
            Err(Error::LengthMismatch)
        ));
    }

    fn qk_batched_rotary_validation_config() -> QkBatchedRotaryEmbeddingConfig {
        QkBatchedRotaryEmbeddingConfig {
            q_output_dimensions: [2, 4, 3, 8],
            k_output_dimensions: [2, 2, 3, 8],
            q_input_strides: [96, 24, 8, 1],
            k_input_strides: [48, 24, 8, 1],
            cos_strides: [12, 4, 1],
            sin_strides: [12, 4, 1],
            trig_batch: 1,
            rotary_pairs: 4,
        }
    }
}
