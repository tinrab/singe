#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(feature = "dtype_f32")]
use crate::cuda::attention::FusedNeighborhoodAttentionConfig;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16"))]
use crate::cuda::attention::{SlidingWindowAttentionConfig, SwaAttentionConfig};
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use crate::{
    cuda::{
        cutile,
        interop::{input_pointer, write_only_pointer},
    },
    error::Result,
};

#[cfg(feature = "dtype_f16")]
/// Causal left-window attention.
///
/// See [`sliding_window_attention_f32`] for mask and grouped-query semantics.
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn sliding_window_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    config: SlidingWindowAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::sliding_window_attention_f16(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            config.batch,
            config.query_len,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.query_start,
            config.key_start,
            config.window,
            config.query_batch_stride,
            config.key_batch_stride,
            config.value_batch_stride,
            config.output_batch_stride,
            config.query_sequence_stride,
            config.key_sequence_stride,
            config.value_sequence_stride,
            config.output_sequence_stride,
            config.query_head_stride,
            config.key_head_stride,
            config.value_head_stride,
            config.output_head_stride,
            config.scale,
            config.output_scale,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
/// Causal left-window attention.
///
/// See [`sliding_window_attention_f32`] for mask and grouped-query semantics.
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn sliding_window_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    query: &impl DeviceInput<bf16>,
    key: &impl DeviceInput<bf16>,
    value: &impl DeviceInput<bf16>,
    config: SlidingWindowAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::sliding_window_attention_bf16(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            config.batch,
            config.query_len,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.query_start,
            config.key_start,
            config.window,
            config.query_batch_stride,
            config.key_batch_stride,
            config.value_batch_stride,
            config.output_batch_stride,
            config.query_sequence_stride,
            config.key_sequence_stride,
            config.value_sequence_stride,
            config.output_sequence_stride,
            config.query_head_stride,
            config.key_head_stride,
            config.value_head_stride,
            config.output_head_stride,
            config.scale,
            config.output_scale,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn swa_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    config: SwaAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::swa_attention_f16(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            config.batch,
            config.query_len,
            config.key_len,
            config.heads,
            config.head_dim,
            config.effective_window(),
            config.scale,
            config.causal,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn swa_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    query: &impl DeviceInput<bf16>,
    key: &impl DeviceInput<bf16>,
    value: &impl DeviceInput<bf16>,
    config: SwaAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::swa_attention_bf16(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            config.batch,
            config.query_len,
            config.key_len,
            config.heads,
            config.head_dim,
            config.effective_window(),
            config.scale,
            config.causal,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fused_neighborhood_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: FusedNeighborhoodAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fused_neighborhood_attention_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            config.batch,
            config.seq_len,
            config.heads,
            config.head_dim,
            config.kernel_size,
            config.dilation,
            config.scale,
        )
    }
}
