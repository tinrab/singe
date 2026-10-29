//! `RoPE` and positional embedding variants.

use ::cutile::cuda_async::device_buffer::DevicePointer;
#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput, DeviceSliceMut},
};

pub use crate::cpu::positional::{
    BnsdQkRopeConfig, InterleavedComplexRopeConfig, MultiaxisRopeConfig,
};
#[cfg(feature = "cutile")]
use crate::cuda::cutile;
#[cfg(feature = "dtype_f16")]
use crate::utility::ensure_len_at_least;
use crate::{
    cuda::interop::{input_pointer, output_pointer},
    error::{Error, Result},
    utility::{
        checked_element_count, checked_rank4_len, ensure_len, ensure_rank2_reach,
        ensure_rank3_reach, ensure_rank4_reach,
    },
};

#[derive(Debug, Clone, Copy)]
/// Generic interleaved-pair `RoPE` config.
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
/// BSHD half-split `RoPE` config.
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
    ($operation:literal, $name:ident, $ty:ty) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Retain input allocations and exclusive writable ranges in the stream's context
        /// through completion on every path and throughout captured replay.
        /// Order prior accesses before launch. Distinct destinations must not overlap.
        /// Out-of-place destinations must not overlap inputs; in-place query and key buffers
        /// must already be initialized and disjoint from the frequency tables.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            cos: &impl DeviceInput<$ty>,
            sin: &impl DeviceInput<$ty>,
            config: RotaryEmbeddingConfig,
        ) -> Result<()> {
            validate_rotary(out.len(), input.len(), cos.len(), sin.len(), config)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::positional::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    input_pointer(cos),
                    input_pointer(sin),
                    config,
                )
            }
        }
    };
}

macro_rules! rotary_embedding_positioned_fn {
    ($operation:literal, $name:ident, $ty:ty) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Retain input allocations and exclusive writable ranges in the stream's context
        /// through completion on every path and throughout captured replay.
        /// Order prior accesses before launch. Distinct destinations must not overlap.
        /// Out-of-place destinations must not overlap inputs; in-place query and key buffers
        /// must already be initialized and disjoint from the frequency tables.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            cos: &impl DeviceInput<$ty>,
            sin: &impl DeviceInput<$ty>,
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
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::positional::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    input_pointer(cos),
                    input_pointer(sin),
                    config,
                    position_start,
                )
            }
        }
    };
}

macro_rules! rotary_embedding_dynpos_fn {
    ($operation:literal, $name:ident, $ty:ty) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Retain input allocations and exclusive writable ranges in the stream's context
        /// through completion on every path and throughout captured replay.
        /// Order prior accesses before launch. Distinct destinations must not overlap.
        /// Out-of-place destinations must not overlap inputs; in-place query and key buffers
        /// must already be initialized and disjoint from the frequency tables.
        /// The device position plus each sequence index must address a valid row in both
        /// trigonometric tables and fit the kernel indexing width.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            cos: &impl DeviceInput<$ty>,
            sin: &impl DeviceInput<$ty>,
            position_start: &impl DeviceInput<u32>,
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
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::positional::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    input_pointer(cos),
                    input_pointer(sin),
                    input_pointer(position_start),
                    config,
                )
            }
        }
    };
}

macro_rules! rotary_embedding_batched_fn {
    ($operation:literal, $name:ident, $ty:ty) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Retain input allocations and exclusive writable ranges in the stream's context
        /// through completion on every path and throughout captured replay.
        /// Order prior accesses before launch. Distinct destinations must not overlap.
        /// Out-of-place destinations must not overlap inputs; in-place query and key buffers
        /// must already be initialized and disjoint from the frequency tables.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            cos: &impl DeviceInput<$ty>,
            sin: &impl DeviceInput<$ty>,
            config: BatchedRotaryEmbeddingConfig,
        ) -> Result<()> {
            validate_rotary_batched(out.len(), input.len(), cos.len(), sin.len(), config)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::positional::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    input_pointer(cos),
                    input_pointer(sin),
                    config,
                )
            }
        }
    };
}

macro_rules! rotary_embedding_qk_fn {
    ($name:ident, $ty:ty) => {
        /// Writes query and key embeddings by rotating adjacent feature pairs.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Retain input allocations and exclusive writable ranges in the stream's context
        /// through completion on every path and throughout captured replay.
        /// Order prior accesses before launch. Distinct destinations must not overlap.
        /// Out-of-place destinations must not overlap inputs; in-place query and key buffers
        /// must already be initialized and disjoint from the frequency tables.
        pub unsafe fn $name(
            stream: &Stream,
            q_out: &mut impl DeviceOutput<$ty>,
            k_out: &mut impl DeviceOutput<$ty>,
            q_input: &impl DeviceInput<$ty>,
            k_input: &impl DeviceInput<$ty>,
            cos: &impl DeviceInput<$ty>,
            sin: &impl DeviceInput<$ty>,
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
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::positional::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(q_out.as_mut_ptr() as _),
                    DevicePointer::from_cu_deviceptr(k_out.as_mut_ptr() as _),
                    input_pointer(q_input),
                    input_pointer(k_input),
                    input_pointer(cos),
                    input_pointer(sin),
                    config,
                )
            }
        }
    };
}

macro_rules! rotary_embedding_qk_batched_fn {
    ($name:ident, $target:ident, $ty:ty) => {
        /// Writes query and key embeddings by rotating adjacent feature pairs.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Retain input allocations and exclusive writable ranges in the stream's context
        /// through completion on every path and throughout captured replay.
        /// Order prior accesses before launch. Distinct destinations must not overlap.
        /// Out-of-place destinations must not overlap inputs; in-place query and key buffers
        /// must already be initialized and disjoint from the frequency tables.
        pub unsafe fn $name(
            stream: &Stream,
            q_out: &mut impl DeviceOutput<$ty>,
            k_out: &mut impl DeviceOutput<$ty>,
            q_input: &impl DeviceInput<$ty>,
            k_input: &impl DeviceInput<$ty>,
            cos: &impl DeviceInput<$ty>,
            sin: &impl DeviceInput<$ty>,
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
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::positional::$target(
                    &stream,
                    DevicePointer::from_cu_deviceptr(q_out.as_mut_ptr() as _),
                    DevicePointer::from_cu_deviceptr(k_out.as_mut_ptr() as _),
                    input_pointer(q_input),
                    input_pointer(k_input),
                    input_pointer(cos),
                    input_pointer(sin),
                    config,
                )
            }
        }
    };
}

macro_rules! rotary_embedding_qk_batched_in_place_fn {
    ($name:ident, $target:ident, $ty:ty) => {
        /// Applies rotary position encoding to initialized query and key storage in place.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Retain input allocations and exclusive writable ranges in the stream's context
        /// through completion on every path and throughout captured replay.
        /// Order prior accesses before launch. Distinct destinations must not overlap.
        /// Out-of-place destinations must not overlap inputs; in-place query and key buffers
        /// must already be initialized and disjoint from the frequency tables.
        /// Input strides must match the contiguous output layout for in-place rotation.
        pub unsafe fn $name(
            stream: &Stream,
            q: &mut impl DeviceSliceMut<$ty>,
            k: &mut impl DeviceSliceMut<$ty>,
            cos: &impl DeviceInput<$ty>,
            sin: &impl DeviceInput<$ty>,
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
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::positional::$target(
                    &stream,
                    output_pointer(q),
                    output_pointer(k),
                    input_pointer(cos),
                    input_pointer(sin),
                    config,
                )
            }
        }
    };
}

macro_rules! rotary_embedding_qk_in_place_fn {
    ($name:ident, $ty:ty) => {
        /// Applies rotary position encoding to initialized query and key storage in place.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Retain input allocations and exclusive writable ranges in the stream's context
        /// through completion on every path and throughout captured replay.
        /// Order prior accesses before launch. Distinct destinations must not overlap.
        /// Out-of-place destinations must not overlap inputs; in-place query and key buffers
        /// must already be initialized and disjoint from the frequency tables.
        /// Input strides must match the contiguous output layout for in-place rotation.
        pub unsafe fn $name(
            stream: &Stream,
            q: &mut impl DeviceSliceMut<$ty>,
            k: &mut impl DeviceSliceMut<$ty>,
            cos: &impl DeviceInput<$ty>,
            sin: &impl DeviceInput<$ty>,
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
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::positional::$name(
                    &stream,
                    output_pointer(q),
                    output_pointer(k),
                    input_pointer(cos),
                    input_pointer(sin),
                    config,
                )
            }
        }
    };
}

macro_rules! rope_bshd_fn {
    ($name:ident, $ty:ty) => {
        /// Writes half-split rotary embeddings in BSHD layout.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Retain input allocations and exclusive writable ranges in the stream's context
        /// through completion on every path and throughout captured replay.
        /// Order prior accesses before launch. Distinct destinations must not overlap.
        /// Out-of-place destinations must not overlap inputs; in-place query and key buffers
        /// must already be initialized and disjoint from the frequency tables.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            cos: &impl DeviceInput<f32>,
            sin: &impl DeviceInput<f32>,
            config: BshdRopeConfig,
        ) -> Result<()> {
            validate_rope_bshd(out.len(), input.len(), cos.len(), sin.len(), config)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::positional::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    input_pointer(cos),
                    input_pointer(sin),
                    config,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_bf16")]
/// Writes half-split rotary query and key embeddings in BNSD layout.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rotary_embedding_qk_bnsd_bf16(
    stream: &Stream,
    q_out: &mut impl DeviceOutput<bf16>,
    k_out: &mut impl DeviceOutput<bf16>,
    q_input: &impl DeviceInput<bf16>,
    k_input: &impl DeviceInput<bf16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
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
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::positional::rotary_embedding_qk_bnsd_bf16(
            &stream,
            DevicePointer::from_cu_deviceptr(q_out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(k_out.as_mut_ptr() as _),
            input_pointer(q_input),
            input_pointer(k_input),
            input_pointer(cos),
            input_pointer(sin),
            config,
        )
    }
}

#[cfg(feature = "dtype_f32")]
/// Writes half-split rotary query and key embeddings in BNSD layout.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rotary_embedding_qk_bnsd_f32(
    stream: &Stream,
    q_out: &mut impl DeviceOutput<f32>,
    k_out: &mut impl DeviceOutput<f32>,
    q_input: &impl DeviceInput<f32>,
    k_input: &impl DeviceInput<f32>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
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
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::positional::rotary_embedding_qk_bnsd_f32(
            &stream,
            DevicePointer::from_cu_deviceptr(q_out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(k_out.as_mut_ptr() as _),
            input_pointer(q_input),
            input_pointer(k_input),
            input_pointer(cos),
            input_pointer(sin),
            config,
        )
    }
}

#[cfg(feature = "dtype_f16")]
/// Writes half-split rotary query and key embeddings in BNSD layout.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rotary_embedding_qk_bnsd_f16(
    stream: &Stream,
    q_out: &mut impl DeviceOutput<f16>,
    k_out: &mut impl DeviceOutput<f16>,
    q_input: &impl DeviceInput<f16>,
    k_input: &impl DeviceInput<f16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
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
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::positional::rotary_embedding_qk_bnsd_f16(
            &stream,
            DevicePointer::from_cu_deviceptr(q_out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(k_out.as_mut_ptr() as _),
            input_pointer(q_input),
            input_pointer(k_input),
            input_pointer(cos),
            input_pointer(sin),
            config,
        )
    }
}

#[cfg(feature = "dtype_f32")]
/// Applies rotary position encoding to initialized query and key storage in place.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rotary_embedding_qk_bnsd_in_place_f32(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<f32>,
    k: &mut impl DeviceSliceMut<f32>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
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
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::positional::rotary_embedding_qk_bnsd_in_place_f32(
            &stream,
            output_pointer(q),
            output_pointer(k),
            input_pointer(cos),
            input_pointer(sin),
            config,
        )
    }
}

#[cfg(feature = "dtype_f16")]
/// Applies rotary position encoding to initialized query and key storage in place.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rotary_embedding_qk_bnsd_in_place_f16(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<f16>,
    k: &mut impl DeviceSliceMut<f16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
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
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::positional::rotary_embedding_qk_bnsd_in_place_f16(
            &stream,
            output_pointer(q),
            output_pointer(k),
            input_pointer(cos),
            input_pointer(sin),
            config,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
/// Applies rotary position encoding to initialized query and key storage in place.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rotary_embedding_qk_bnsd_in_place_bf16(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<bf16>,
    k: &mut impl DeviceSliceMut<bf16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
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
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::positional::rotary_embedding_qk_bnsd_in_place_bf16(
            &stream,
            output_pointer(q),
            output_pointer(k),
            input_pointer(cos),
            input_pointer(sin),
            config,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
/// BNSD Q/K half-split RoPE.
///
/// Rotates dimensions `(pair, rotary_pairs + pair)`, matching half-split
/// `rotate_half` layouts rather than adjacent interleaved pairs.
/// Writes half-split rotary query and key embeddings in BNSD layout.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rope_embedding_qk_bnsd_bf16(
    stream: &Stream,
    q_out: &mut impl DeviceOutput<bf16>,
    k_out: &mut impl DeviceOutput<bf16>,
    q_input: &impl DeviceInput<bf16>,
    k_input: &impl DeviceInput<bf16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        rotary_embedding_qk_bnsd_bf16(stream, q_out, k_out, q_input, k_input, cos, sin, config)
    }
}

#[cfg(feature = "dtype_f32")]
/// BNSD Q/K half-split `RoPE`.
///
/// Rotates dimensions `(pair, rotary_pairs + pair)`, matching half-split
/// `rotate_half` layouts rather than adjacent interleaved pairs.
/// Writes half-split rotary query and key embeddings in BNSD layout.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rope_embedding_qk_bnsd_f32(
    stream: &Stream,
    q_out: &mut impl DeviceOutput<f32>,
    k_out: &mut impl DeviceOutput<f32>,
    q_input: &impl DeviceInput<f32>,
    k_input: &impl DeviceInput<f32>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        rotary_embedding_qk_bnsd_f32(stream, q_out, k_out, q_input, k_input, cos, sin, config)
    }
}

#[cfg(feature = "dtype_f16")]
/// BNSD Q/K half-split RoPE.
///
/// Rotates dimensions `(pair, rotary_pairs + pair)`, matching half-split
/// `rotate_half` layouts rather than adjacent interleaved pairs.
/// Writes half-split rotary query and key embeddings in BNSD layout.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rope_embedding_qk_bnsd_f16(
    stream: &Stream,
    q_out: &mut impl DeviceOutput<f16>,
    k_out: &mut impl DeviceOutput<f16>,
    q_input: &impl DeviceInput<f16>,
    k_input: &impl DeviceInput<f16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        rotary_embedding_qk_bnsd_f16(stream, q_out, k_out, q_input, k_input, cos, sin, config)
    }
}

#[cfg(feature = "dtype_bf16")]
/// Writes half-split rotary query and key embeddings in BNSD layout.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rope_qk_half_split_bnsd_bf16(
    stream: &Stream,
    q_out: &mut impl DeviceOutput<bf16>,
    k_out: &mut impl DeviceOutput<bf16>,
    q_input: &impl DeviceInput<bf16>,
    k_input: &impl DeviceInput<bf16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        rotary_embedding_qk_bnsd_bf16(stream, q_out, k_out, q_input, k_input, cos, sin, config)
    }
}

#[cfg(feature = "dtype_f32")]
/// Writes half-split rotary query and key embeddings in BNSD layout.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rope_qk_half_split_bnsd_f32(
    stream: &Stream,
    q_out: &mut impl DeviceOutput<f32>,
    k_out: &mut impl DeviceOutput<f32>,
    q_input: &impl DeviceInput<f32>,
    k_input: &impl DeviceInput<f32>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        rotary_embedding_qk_bnsd_f32(stream, q_out, k_out, q_input, k_input, cos, sin, config)
    }
}

#[cfg(feature = "dtype_f16")]
/// Writes half-split rotary query and key embeddings in BNSD layout.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rope_qk_half_split_bnsd_f16(
    stream: &Stream,
    q_out: &mut impl DeviceOutput<f16>,
    k_out: &mut impl DeviceOutput<f16>,
    q_input: &impl DeviceInput<f16>,
    k_input: &impl DeviceInput<f16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        rotary_embedding_qk_bnsd_f16(stream, q_out, k_out, q_input, k_input, cos, sin, config)
    }
}

#[cfg(feature = "dtype_f32")]
/// Applies rotary position encoding to initialized query and key storage in place.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rope_embedding_qk_bnsd_in_place_f32(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<f32>,
    k: &mut impl DeviceSliceMut<f32>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { rotary_embedding_qk_bnsd_in_place_f32(stream, q, k, cos, sin, config) }
}

#[cfg(feature = "dtype_f16")]
/// Applies rotary position encoding to initialized query and key storage in place.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rope_embedding_qk_bnsd_in_place_f16(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<f16>,
    k: &mut impl DeviceSliceMut<f16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { rotary_embedding_qk_bnsd_in_place_f16(stream, q, k, cos, sin, config) }
}

#[cfg(feature = "dtype_bf16")]
/// Applies rotary position encoding to initialized query and key storage in place.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rope_embedding_qk_bnsd_in_place_bf16(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<bf16>,
    k: &mut impl DeviceSliceMut<bf16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { rotary_embedding_qk_bnsd_in_place_bf16(stream, q, k, cos, sin, config) }
}

#[cfg(feature = "dtype_f32")]
/// Applies rotary position encoding to initialized query and key storage in place.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rope_qk_half_split_bnsd_in_place_f32(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<f32>,
    k: &mut impl DeviceSliceMut<f32>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { rotary_embedding_qk_bnsd_in_place_f32(stream, q, k, cos, sin, config) }
}

#[cfg(feature = "dtype_f16")]
/// Applies rotary position encoding to initialized query and key storage in place.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rope_qk_half_split_bnsd_in_place_f16(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<f16>,
    k: &mut impl DeviceSliceMut<f16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { rotary_embedding_qk_bnsd_in_place_f16(stream, q, k, cos, sin, config) }
}

#[cfg(feature = "dtype_bf16")]
/// Applies rotary position encoding to initialized query and key storage in place.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn rope_qk_half_split_bnsd_in_place_bf16(
    stream: &Stream,
    q: &mut impl DeviceSliceMut<bf16>,
    k: &mut impl DeviceSliceMut<bf16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { rotary_embedding_qk_bnsd_in_place_bf16(stream, q, k, cos, sin, config) }
}

#[cfg(feature = "dtype_f32")]
rotary_embedding_fn!(
    "Writes rotary embeddings by rotating adjacent feature pairs and copying unrotated features.",
    rotary_embedding_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
rotary_embedding_fn!(
    "Writes rotary embeddings by rotating adjacent feature pairs and copying unrotated features.",
    rotary_embedding_f16,
    f16
);

#[cfg(feature = "dtype_f64")]
rotary_embedding_fn!(
    "Writes rotary embeddings by rotating adjacent feature pairs and copying unrotated features.",
    rotary_embedding_f64,
    f64
);

#[cfg(feature = "dtype_f32")]
rotary_embedding_positioned_fn!(
    "Writes rotary embeddings by rotating adjacent feature pairs and copying unrotated features.",
    rotary_embedding_positioned_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
rotary_embedding_positioned_fn!(
    "Writes rotary embeddings by rotating adjacent feature pairs and copying unrotated features.",
    rotary_embedding_positioned_f16,
    f16
);

#[cfg(feature = "dtype_f64")]
rotary_embedding_positioned_fn!(
    "Writes rotary embeddings by rotating adjacent feature pairs and copying unrotated features.",
    rotary_embedding_positioned_f64,
    f64
);

#[cfg(feature = "dtype_f32")]
rotary_embedding_dynpos_fn!(
    "Writes rotary embeddings by rotating adjacent feature pairs and copying unrotated features.",
    rotary_embedding_dynpos_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
rotary_embedding_dynpos_fn!(
    "Writes rotary embeddings by rotating adjacent feature pairs and copying unrotated features.",
    rotary_embedding_dynpos_f16,
    f16
);

#[cfg(feature = "dtype_f64")]
rotary_embedding_dynpos_fn!(
    "Writes rotary embeddings by rotating adjacent feature pairs and copying unrotated features.",
    rotary_embedding_dynpos_f64,
    f64
);

#[cfg(feature = "dtype_f32")]
rotary_embedding_batched_fn!(
    "Writes rotary embeddings by rotating adjacent feature pairs and copying unrotated features.",
    rotary_embedding_batched_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
rotary_embedding_batched_fn!(
    "Writes rotary embeddings by rotating adjacent feature pairs and copying unrotated features.",
    rotary_embedding_batched_f16,
    f16
);

#[cfg(feature = "dtype_f64")]
rotary_embedding_batched_fn!(
    "Writes rotary embeddings by rotating adjacent feature pairs and copying unrotated features.",
    rotary_embedding_batched_f64,
    f64
);

#[cfg(feature = "dtype_f32")]
rotary_embedding_fn!(
    "Writes half-split rotary embeddings within the rotary prefix and copies unrotated features.",
    rotary_embedding_chunked_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
rotary_embedding_fn!(
    "Writes half-split rotary embeddings within the rotary prefix and copies unrotated features.",
    rotary_embedding_chunked_f16,
    f16
);

#[cfg(feature = "dtype_f64")]
rotary_embedding_fn!(
    "Writes half-split rotary embeddings within the rotary prefix and copies unrotated features.",
    rotary_embedding_chunked_f64,
    f64
);

#[cfg(feature = "dtype_f32")]
rotary_embedding_positioned_fn!(
    "Writes half-split rotary embeddings within the rotary prefix and copies unrotated features.",
    rotary_embedding_chunked_positioned_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
rotary_embedding_positioned_fn!(
    "Writes half-split rotary embeddings within the rotary prefix and copies unrotated features.",
    rotary_embedding_chunked_positioned_f16,
    f16
);

#[cfg(feature = "dtype_f64")]
rotary_embedding_positioned_fn!(
    "Writes half-split rotary embeddings within the rotary prefix and copies unrotated features.",
    rotary_embedding_chunked_positioned_f64,
    f64
);

#[cfg(feature = "dtype_f32")]
rotary_embedding_dynpos_fn!(
    "Writes half-split rotary embeddings within the rotary prefix and copies unrotated features.",
    rotary_embedding_chunked_dynpos_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
rotary_embedding_dynpos_fn!(
    "Writes half-split rotary embeddings within the rotary prefix and copies unrotated features.",
    rotary_embedding_chunked_dynpos_f16,
    f16
);

#[cfg(feature = "dtype_f64")]
rotary_embedding_dynpos_fn!(
    "Writes half-split rotary embeddings within the rotary prefix and copies unrotated features.",
    rotary_embedding_chunked_dynpos_f64,
    f64
);

#[cfg(feature = "dtype_f32")]
rotary_embedding_batched_fn!(
    "Writes half-split rotary embeddings within the rotary prefix and copies unrotated features.",
    rotary_embedding_chunked_batched_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
rotary_embedding_batched_fn!(
    "Writes half-split rotary embeddings within the rotary prefix and copies unrotated features.",
    rotary_embedding_chunked_batched_f16,
    f16
);

#[cfg(feature = "dtype_f64")]
rotary_embedding_batched_fn!(
    "Writes half-split rotary embeddings within the rotary prefix and copies unrotated features.",
    rotary_embedding_chunked_batched_f64,
    f64
);

#[cfg(feature = "dtype_f32")]
rotary_embedding_qk_fn!(rotary_embedding_qk_f32, f32);

#[cfg(feature = "dtype_f16")]
rotary_embedding_qk_fn!(rotary_embedding_qk_f16, f16);

#[cfg(feature = "dtype_f64")]
rotary_embedding_qk_fn!(rotary_embedding_qk_f64, f64);

#[cfg(feature = "dtype_f32")]
rotary_embedding_qk_batched_fn!(
    rotary_embedding_qk_batched_f32,
    rotary_embedding_qk_batched_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
rotary_embedding_qk_batched_fn!(
    rotary_embedding_qk_batched_f16,
    rotary_embedding_qk_batched_f16,
    f16
);

#[cfg(feature = "dtype_f64")]
rotary_embedding_qk_batched_fn!(
    rotary_embedding_qk_batched_f64,
    rotary_embedding_qk_batched_f64,
    f64
);

#[cfg(feature = "dtype_f32")]
rotary_embedding_qk_batched_in_place_fn!(
    rotary_embedding_qk_batched_in_place_f32,
    rotary_embedding_qk_batched_in_place_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
rotary_embedding_qk_batched_in_place_fn!(
    rotary_embedding_qk_batched_in_place_f16,
    rotary_embedding_qk_batched_in_place_f16,
    f16
);

#[cfg(feature = "dtype_f64")]
rotary_embedding_qk_batched_in_place_fn!(
    rotary_embedding_qk_batched_in_place_f64,
    rotary_embedding_qk_batched_in_place_f64,
    f64
);

#[cfg(feature = "dtype_f32")]
rotary_embedding_qk_in_place_fn!(rotary_embedding_qk_in_place_f32, f32);

#[cfg(feature = "dtype_f16")]
rotary_embedding_qk_in_place_fn!(rotary_embedding_qk_in_place_f16, f16);

#[cfg(feature = "dtype_f64")]
rotary_embedding_qk_in_place_fn!(rotary_embedding_qk_in_place_f64, f64);

#[cfg(feature = "dtype_f32")]
rope_bshd_fn!(rope_bshd_f32, f32);

#[cfg(feature = "dtype_f16")]
rope_bshd_fn!(rope_bshd_f16, f16);

#[cfg(feature = "dtype_bf16")]
rope_bshd_fn!(rope_bshd_bf16, bf16);

#[cfg(feature = "dtype_f16")]
/// Writes half-split rotary embeddings in BSHD layout.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
/// The device position plus each sequence index must address a valid row in both
/// trigonometric tables and fit the kernel indexing width.
pub unsafe fn rope_bshd_dynpos_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    input: &impl DeviceInput<f16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    position_start: &impl DeviceInput<u32>,
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
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::positional::rope_bshd_dynpos_f16(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(input),
            input_pointer(cos),
            input_pointer(sin),
            input_pointer(position_start),
            config,
        )
    }
}

#[cfg(feature = "dtype_f32")]
/// Rotates the configured query and key axes in place.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn multiaxis_rope_qk_f32(
    stream: &Stream,
    query: &mut impl DeviceSliceMut<f32>,
    key: &mut impl DeviceSliceMut<f32>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: MultiaxisRopeConfig,
) -> Result<()> {
    validate_multiaxis_rope_qk(query.len(), key.len(), cos.len(), sin.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::positional::multiaxis_rope_qk_f32(
            &stream,
            output_pointer(query),
            output_pointer(key),
            input_pointer(cos),
            input_pointer(sin),
            config,
        )
    }
}

#[cfg(feature = "dtype_f16")]
/// Rotates the configured query and key axes in place.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn multiaxis_rope_qk_f16(
    stream: &Stream,
    query: &mut impl DeviceSliceMut<f16>,
    key: &mut impl DeviceSliceMut<f16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: MultiaxisRopeConfig,
) -> Result<()> {
    validate_multiaxis_rope_qk(query.len(), key.len(), cos.len(), sin.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::positional::multiaxis_rope_qk_f16(
            &stream,
            output_pointer(query),
            output_pointer(key),
            input_pointer(cos),
            input_pointer(sin),
            config,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
/// Rotates the configured query and key axes in place.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn multiaxis_rope_qk_bf16(
    stream: &Stream,
    query: &mut impl DeviceSliceMut<bf16>,
    key: &mut impl DeviceSliceMut<bf16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    config: MultiaxisRopeConfig,
) -> Result<()> {
    validate_multiaxis_rope_qk(query.len(), key.len(), cos.len(), sin.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::positional::multiaxis_rope_qk_bf16(
            &stream,
            output_pointer(query),
            output_pointer(key),
            input_pointer(cos),
            input_pointer(sin),
            config,
        )
    }
}

#[cfg(feature = "dtype_f32")]
/// Rotates adjacent query and key pairs using complex frequency values.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn interleaved_complex_rope_qk_f32(
    stream: &Stream,
    query: &mut impl DeviceSliceMut<f32>,
    key: &mut impl DeviceSliceMut<f32>,
    freqs: &impl DeviceInput<f32>,
    config: InterleavedComplexRopeConfig,
) -> Result<()> {
    validate_interleaved_complex_rope_qk(query.len(), key.len(), freqs.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::positional::interleaved_complex_rope_qk_f32(
            &stream,
            output_pointer(query),
            output_pointer(key),
            input_pointer(freqs),
            config,
        )
    }
}

#[cfg(feature = "dtype_f16")]
/// Rotates adjacent query and key pairs using complex frequency values.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn interleaved_complex_rope_qk_f16(
    stream: &Stream,
    query: &mut impl DeviceSliceMut<f16>,
    key: &mut impl DeviceSliceMut<f16>,
    freqs: &impl DeviceInput<f32>,
    config: InterleavedComplexRopeConfig,
) -> Result<()> {
    validate_interleaved_complex_rope_qk(query.len(), key.len(), freqs.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::positional::interleaved_complex_rope_qk_f16(
            &stream,
            output_pointer(query),
            output_pointer(key),
            input_pointer(freqs),
            config,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
/// Rotates adjacent query and key pairs using complex frequency values.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Retain input allocations and exclusive writable ranges in the stream's context
/// through completion on every path and throughout captured replay.
/// Order prior accesses before launch. Distinct destinations must not overlap.
/// Out-of-place destinations must not overlap inputs; in-place query and key buffers
/// must already be initialized and disjoint from the frequency tables.
pub unsafe fn interleaved_complex_rope_qk_bf16(
    stream: &Stream,
    query: &mut impl DeviceSliceMut<bf16>,
    key: &mut impl DeviceSliceMut<bf16>,
    freqs: &impl DeviceInput<f32>,
    config: InterleavedComplexRopeConfig,
) -> Result<()> {
    validate_interleaved_complex_rope_qk(query.len(), key.len(), freqs.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::positional::interleaved_complex_rope_qk_bf16(
            &stream,
            output_pointer(query),
            output_pointer(key),
            input_pointer(freqs),
            config,
        )
    }
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

    fn is_length_mismatch<T>(result: Result<T>) -> bool {
        matches!(
            result,
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::RankReachMismatch { .. })
        )
    }

    #[test]
    fn test_qk_batched_rotary_validation_accepts_shared_and_batched_trig() -> Result<()> {
        let mut config = qk_batched_rotary_validation_config();
        validate_rotary_qk_batched(192, 96, 192, 96, 12, 12, config)?;

        config.trig_batch = 2;
        validate_rotary_qk_batched(192, 96, 192, 96, 24, 24, config)?;
        Ok(())
    }

    #[test]
    fn test_qk_batched_rotary_validation_rejects_bad_trig_batch() {
        let mut config = qk_batched_rotary_validation_config();
        config.trig_batch = 3;
        assert!(is_length_mismatch(validate_rotary_qk_batched(
            192, 96, 192, 96, 36, 36, config
        )));
    }

    #[test]
    fn test_qk_batched_rotary_validation_rejects_mismatched_qk_shape() {
        let mut config = qk_batched_rotary_validation_config();
        config.k_output_dimensions[2] = 4;
        assert!(is_length_mismatch(validate_rotary_qk_batched(
            192, 128, 192, 128, 12, 12, config
        )));
    }

    #[test]
    fn test_qk_batched_rotary_validation_rejects_short_batched_trig() {
        let mut config = qk_batched_rotary_validation_config();
        config.trig_batch = 2;
        assert!(is_length_mismatch(validate_rotary_qk_batched(
            192, 96, 192, 96, 23, 24, config
        )));
    }

    #[test]
    fn test_rope_bshd_validation_accepts_exact_lengths() -> Result<()> {
        let config = BshdRopeConfig {
            batch: 2,
            seq_len: 3,
            heads: 4,
            head_dim: 8,
        };
        validate_rope_bshd(192, 192, 12, 12, config)
    }

    #[test]
    fn test_rope_bshd_validation_rejects_odd_head_dim() {
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
    fn test_rope_bshd_validation_rejects_short_trig() {
        let config = BshdRopeConfig {
            batch: 2,
            seq_len: 3,
            heads: 4,
            head_dim: 8,
        };
        assert!(is_length_mismatch(validate_rope_bshd(
            192, 192, 11, 12, config
        )));
    }

    #[test]
    fn test_rope_qk_bnsd_validation_accepts_exact_lengths() -> Result<()> {
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
    fn test_rope_qk_bnsd_validation_rejects_bad_trig_batch() {
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
    fn test_rope_qk_bnsd_validation_rejects_rotary_overflow() {
        let config = BnsdQkRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 4,
            key_heads: 2,
            head_dim: 8,
            trig_batch: 1,
            rotary_pairs: 5,
        };
        assert!(is_length_mismatch(validate_rope_qk_bnsd(
            192, 96, 192, 96, 15, 15, config
        )));
    }

    #[test]
    fn test_multiaxis_rope_validation_accepts_exact_lengths() -> Result<()> {
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
    fn test_multiaxis_rope_validation_rejects_bad_sections() {
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
        assert!(is_length_mismatch(validate_multiaxis_rope_qk(
            192, 96, 72, 72, config
        )));
    }

    #[test]
    fn test_multiaxis_rope_validation_rejects_short_trig() {
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
        assert!(is_length_mismatch(validate_multiaxis_rope_qk(
            192, 96, 71, 72, config
        )));
    }

    #[test]
    fn test_interleaved_complex_rope_validation_accepts_exact_lengths() -> Result<()> {
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
    fn test_interleaved_complex_rope_validation_rejects_short_freqs() {
        let config = InterleavedComplexRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 4,
            key_heads: 2,
            head_dim: 8,
        };
        assert!(is_length_mismatch(validate_interleaved_complex_rope_qk(
            192, 96, 23, config
        )));
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
