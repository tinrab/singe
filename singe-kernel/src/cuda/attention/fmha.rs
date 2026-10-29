#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(feature = "dtype_bf16")]
use crate::cuda::attention::{fmha_prefill_lse_mma_bf16, fmha_prefill_mma_bf16};
#[cfg(feature = "dtype_f32")]
use crate::cuda::autotune;
#[cfg(feature = "dtype_f32")]
use crate::cuda::{
    attention::{
        AttentionSplitKTuning, FmhaDecodeSplitKConfig, FmhaDecodeSplitKReduceConfig,
        SplitKReduceConfig, attention_splitk_candidates, attention_splitk_tune_key,
        fmha_splitk_decode_config, fmha_splitk_reduce_config, resolve_attention_splitk_tuning,
        validate_attention_splitk_tuning, validate_fmha_decode_splitk_reduce_lengths,
    },
    autotune::{TuneOptions, TuneReport, TuningChoice},
};
#[cfg(feature = "dtype_f16")]
use crate::{
    cuda::attention::{fmha_prefill_lse_mma_f16, fmha_prefill_mma_f16},
    utility::ensure_len,
};
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use crate::{
    cuda::{
        attention::{
            AttentionSinkPrefillConfig, FmhaDecodeConfig, FmhaPrefillConfig,
            SoftcappedWindowAttentionConfig, SoftcappedWindowDecodeConfig,
        },
        cutile,
        interop::{input_pointer, write_only_pointer},
    },
    error::Result,
};

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_prefill_lse_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_prefill_lse_f32(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            config.batch,
            config.query_len,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
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
            config.causal,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_prefill_lse_mma_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_prefill_lse_mma_f32(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            config.batch,
            config.query_len,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
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
            config.causal,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_prefill_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    config.validate_output_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_prefill_f32(
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
            config.causal,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn attention_sink_prefill_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    sinks: &impl DeviceInput<f32>,
    config: AttentionSinkPrefillConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len(), sinks.len())?;
    let attention = config.attention;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::attention_sink_prefill_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            input_pointer(sinks),
            attention.batch,
            attention.query_len,
            attention.key_len,
            attention.heads,
            attention.kv_heads,
            attention.head_dim,
            attention.query_batch_stride,
            attention.key_batch_stride,
            attention.value_batch_stride,
            attention.output_batch_stride,
            attention.query_sequence_stride,
            attention.key_sequence_stride,
            attention.value_sequence_stride,
            attention.output_sequence_stride,
            attention.query_head_stride,
            attention.key_head_stride,
            attention.value_head_stride,
            attention.output_head_stride,
            config.start_q,
            config.window,
            attention.scale,
            attention.causal,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn softcapped_window_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: SoftcappedWindowAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let attention = config.attention;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::softcapped_window_attention_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            attention.batch,
            attention.query_len,
            attention.key_len,
            attention.heads,
            attention.kv_heads,
            attention.head_dim,
            attention.query_batch_stride,
            attention.key_batch_stride,
            attention.value_batch_stride,
            attention.output_batch_stride,
            attention.query_sequence_stride,
            attention.key_sequence_stride,
            attention.value_sequence_stride,
            attention.output_sequence_stride,
            attention.query_head_stride,
            attention.key_head_stride,
            attention.value_head_stride,
            attention.output_head_stride,
            attention.scale,
            attention.causal,
            config.window_size,
            config.soft_cap,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn softcapped_window_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    config: SoftcappedWindowAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let attention = config.attention;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::softcapped_window_attention_f16(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            attention.batch,
            attention.query_len,
            attention.key_len,
            attention.heads,
            attention.kv_heads,
            attention.head_dim,
            attention.query_batch_stride,
            attention.key_batch_stride,
            attention.value_batch_stride,
            attention.output_batch_stride,
            attention.query_sequence_stride,
            attention.key_sequence_stride,
            attention.value_sequence_stride,
            attention.output_sequence_stride,
            attention.query_head_stride,
            attention.key_head_stride,
            attention.value_head_stride,
            attention.output_head_stride,
            attention.scale,
            attention.causal,
            config.window_size,
            config.soft_cap,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn softcapped_window_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    query: &impl DeviceInput<bf16>,
    key: &impl DeviceInput<bf16>,
    value: &impl DeviceInput<bf16>,
    config: SoftcappedWindowAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let attention = config.attention;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::softcapped_window_attention_bf16(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            attention.batch,
            attention.query_len,
            attention.key_len,
            attention.heads,
            attention.kv_heads,
            attention.head_dim,
            attention.query_batch_stride,
            attention.key_batch_stride,
            attention.value_batch_stride,
            attention.output_batch_stride,
            attention.query_sequence_stride,
            attention.key_sequence_stride,
            attention.value_sequence_stride,
            attention.output_sequence_stride,
            attention.query_head_stride,
            attention.key_head_stride,
            attention.value_head_stride,
            attention.output_head_stride,
            attention.scale,
            attention.causal,
            config.window_size,
            config.soft_cap,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_decode_lse_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: FmhaDecodeConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_decode_lse_f32(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            config.batch,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.query_batch_stride,
            config.key_batch_stride,
            config.value_batch_stride,
            config.output_batch_stride,
            config.query_head_stride,
            config.key_sequence_stride,
            config.value_sequence_stride,
            config.output_head_stride,
            config.key_head_stride,
            config.value_head_stride,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn softcapped_window_decode_lse_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: SoftcappedWindowDecodeConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let decode = config.decode;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::softcapped_window_decode_lse_f32(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            decode.batch,
            decode.key_len,
            decode.heads,
            decode.kv_heads,
            decode.head_dim,
            decode.query_batch_stride,
            decode.key_batch_stride,
            decode.value_batch_stride,
            decode.output_batch_stride,
            decode.query_head_stride,
            decode.key_sequence_stride,
            decode.value_sequence_stride,
            decode.output_head_stride,
            decode.key_head_stride,
            decode.value_head_stride,
            decode.scale,
            config.window_size,
            config.soft_cap,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn softcapped_window_decode_lse_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    config: SoftcappedWindowDecodeConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let decode = config.decode;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::softcapped_window_decode_lse_f16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            decode.batch,
            decode.key_len,
            decode.heads,
            decode.kv_heads,
            decode.head_dim,
            decode.query_batch_stride,
            decode.key_batch_stride,
            decode.value_batch_stride,
            decode.output_batch_stride,
            decode.query_head_stride,
            decode.key_sequence_stride,
            decode.value_sequence_stride,
            decode.output_head_stride,
            decode.key_head_stride,
            decode.value_head_stride,
            decode.scale,
            config.window_size,
            config.soft_cap,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn softcapped_window_decode_lse_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<bf16>,
    key: &impl DeviceInput<bf16>,
    value: &impl DeviceInput<bf16>,
    config: SoftcappedWindowDecodeConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let decode = config.decode;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::softcapped_window_decode_lse_bf16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            decode.batch,
            decode.key_len,
            decode.heads,
            decode.kv_heads,
            decode.head_dim,
            decode.query_batch_stride,
            decode.key_batch_stride,
            decode.value_batch_stride,
            decode.output_batch_stride,
            decode.query_head_stride,
            decode.key_sequence_stride,
            decode.value_sequence_stride,
            decode.output_head_stride,
            decode.key_head_stride,
            decode.value_head_stride,
            decode.scale,
            config.window_size,
            config.soft_cap,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_prefill_lse_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    if config.query_len >= 16 && cutile::attention::supports_fmha_prefill_mma(config) {
        // SAFETY: the caller upholds this function's safety contract.
        return unsafe { fmha_prefill_lse_mma_f16(stream, out, lse, query, key, value, config) };
    }

    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_prefill_lse_f16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            config.batch,
            config.query_len,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
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
            config.causal,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_prefill_lse_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<bf16>,
    key: &impl DeviceInput<bf16>,
    value: &impl DeviceInput<bf16>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    if config.query_len >= 16 && cutile::attention::supports_fmha_prefill_mma(config) {
        // SAFETY: the caller upholds this function's safety contract.
        return unsafe { fmha_prefill_lse_mma_bf16(stream, out, lse, query, key, value, config) };
    }

    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_prefill_lse_bf16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            config.batch,
            config.query_len,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
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
            config.causal,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_prefill_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    if config.query_len >= 16 && cutile::attention::supports_fmha_prefill_mma(config) {
        // SAFETY: the caller upholds this function's safety contract.
        return unsafe { fmha_prefill_mma_f16(stream, out, query, key, value, config) };
    }

    config.validate_output_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_prefill_f16(
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
            config.causal,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_prefill_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    query: &impl DeviceInput<bf16>,
    key: &impl DeviceInput<bf16>,
    value: &impl DeviceInput<bf16>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    if config.query_len >= 16 && cutile::attention::supports_fmha_prefill_mma(config) {
        // SAFETY: the caller upholds this function's safety contract.
        return unsafe { fmha_prefill_mma_bf16(stream, out, query, key, value, config) };
    }

    config.validate_output_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_prefill_bf16(
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
            config.causal,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn attention_sink_prefill_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    sinks: &impl DeviceInput<f32>,
    config: AttentionSinkPrefillConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len(), sinks.len())?;
    let attention = config.attention;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::attention_sink_prefill_f16(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            input_pointer(sinks),
            attention.batch,
            attention.query_len,
            attention.key_len,
            attention.heads,
            attention.kv_heads,
            attention.head_dim,
            attention.query_batch_stride,
            attention.key_batch_stride,
            attention.value_batch_stride,
            attention.output_batch_stride,
            attention.query_sequence_stride,
            attention.key_sequence_stride,
            attention.value_sequence_stride,
            attention.output_sequence_stride,
            attention.query_head_stride,
            attention.key_head_stride,
            attention.value_head_stride,
            attention.output_head_stride,
            config.start_q,
            config.window,
            attention.scale,
            attention.causal,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn attention_sink_prefill_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    query: &impl DeviceInput<bf16>,
    key: &impl DeviceInput<bf16>,
    value: &impl DeviceInput<bf16>,
    sinks: &impl DeviceInput<f32>,
    config: AttentionSinkPrefillConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len(), sinks.len())?;
    let attention = config.attention;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::attention_sink_prefill_bf16(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            input_pointer(sinks),
            attention.batch,
            attention.query_len,
            attention.key_len,
            attention.heads,
            attention.kv_heads,
            attention.head_dim,
            attention.query_batch_stride,
            attention.key_batch_stride,
            attention.value_batch_stride,
            attention.output_batch_stride,
            attention.query_sequence_stride,
            attention.key_sequence_stride,
            attention.value_sequence_stride,
            attention.output_sequence_stride,
            attention.query_head_stride,
            attention.key_head_stride,
            attention.value_head_stride,
            attention.output_head_stride,
            config.start_q,
            config.window,
            attention.scale,
            attention.causal,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_decode_lse_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    config: FmhaDecodeConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_decode_lse_f16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            config.batch,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.query_batch_stride,
            config.key_batch_stride,
            config.value_batch_stride,
            config.output_batch_stride,
            config.query_head_stride,
            config.key_sequence_stride,
            config.value_sequence_stride,
            config.output_head_stride,
            config.key_head_stride,
            config.value_head_stride,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_decode_lse_dynpos_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    position_start: &impl DeviceInput<u32>,
    config: FmhaDecodeConfig,
) -> Result<()> {
    ensure_len(position_start.len(), 1)?;
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_decode_lse_dynpos_f16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            input_pointer(position_start),
            config.batch,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.query_batch_stride,
            config.key_batch_stride,
            config.value_batch_stride,
            config.output_batch_stride,
            config.query_head_stride,
            config.key_sequence_stride,
            config.value_sequence_stride,
            config.output_head_stride,
            config.key_head_stride,
            config.value_head_stride,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_decode_lse_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<bf16>,
    key: &impl DeviceInput<bf16>,
    value: &impl DeviceInput<bf16>,
    config: FmhaDecodeConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_decode_lse_bf16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            config.batch,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.query_batch_stride,
            config.key_batch_stride,
            config.value_batch_stride,
            config.output_batch_stride,
            config.query_head_stride,
            config.key_sequence_stride,
            config.value_sequence_stride,
            config.output_head_stride,
            config.key_head_stride,
            config.value_head_stride,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn splitk_reduce_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    attn_splitk: &impl DeviceInput<f32>,
    lse_splitk: &impl DeviceInput<f32>,
    config: SplitKReduceConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), attn_splitk.len(), lse_splitk.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::splitk_reduce_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(attn_splitk),
            input_pointer(lse_splitk),
            config.batch,
            config.heads,
            config.splits,
            config.head_dim,
            config.attn_batch_stride,
            config.attn_head_stride,
            config.attn_split_stride,
            config.lse_batch_stride,
            config.lse_head_stride,
            config.output_batch_stride,
            config.output_head_stride,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_decode_splitk_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: FmhaDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_decode_splitk_f32(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            config.batch,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.splits,
            config.kv_len_per_split,
            config.head_dim,
            config.query_batch_stride,
            config.key_batch_stride,
            config.value_batch_stride,
            config.output_batch_stride,
            config.query_head_stride,
            config.key_sequence_stride,
            config.value_sequence_stride,
            config.output_head_stride,
            config.output_split_stride,
            config.key_head_stride,
            config.value_head_stride,
            config.lse_batch_stride,
            config.lse_head_stride,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f32")]
fn launch_fmha_decode_splitk_reduce_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    attn_scratch: &mut impl DeviceOutput<f32>,
    lse_scratch: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: FmhaDecodeSplitKReduceConfig,
    tuning: AttentionSplitKTuning,
) -> Result<()> {
    validate_attention_splitk_tuning(config, tuning)?;
    let decode_config = fmha_splitk_decode_config(config, tuning);
    let reduce_config = fmha_splitk_reduce_config(config, tuning);
    // SAFETY: the typed device buffers stay borrowed for this stream-ordered launch, and the tuning was validated above.
    unsafe {
        fmha_decode_splitk_f32(
            stream,
            attn_scratch,
            lse_scratch,
            query,
            key,
            value,
            decode_config,
        )
    }?;
    // SAFETY: the typed device buffers stay borrowed for this stream-ordered launch, and the tuning was validated above.
    unsafe { splitk_reduce_f32(stream, out, attn_scratch, lse_scratch, reduce_config) }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_decode_splitk_reduce_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    attn_scratch: &mut impl DeviceOutput<f32>,
    lse_scratch: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: FmhaDecodeSplitKReduceConfig,
) -> Result<()> {
    validate_fmha_decode_splitk_reduce_lengths(
        config,
        out.len(),
        attn_scratch.len(),
        lse_scratch.len(),
        query.len(),
        key.len(),
        value.len(),
    )?;
    let default_tuning =
        AttentionSplitKTuning::create(config.max_splits.min(config.decode.key_len));
    match config.tuning {
        TuningChoice::Default => {
            return launch_fmha_decode_splitk_reduce_f32(
                stream,
                out,
                attn_scratch,
                lse_scratch,
                query,
                key,
                value,
                config,
                default_tuning,
            );
        }
        TuningChoice::Explicit(tuning) => {
            return launch_fmha_decode_splitk_reduce_f32(
                stream,
                out,
                attn_scratch,
                lse_scratch,
                query,
                key,
                value,
                config,
                tuning,
            );
        }
        TuningChoice::Cached => {}
        TuningChoice::Tune(_) | TuningChoice::Retune(_) => {
            autotune::ensure_not_capturing(stream, "fmha_decode_splitk_reduce_f32")?;
        }
    }

    let tune_key =
        attention_splitk_tune_key(stream, "fmha_decode_splitk_reduce_f32", "f32", config)?;
    let tuning = resolve_attention_splitk_tuning(
        stream,
        "fmha_decode_splitk_reduce_f32",
        tune_key.clone(),
        config.tuning,
        || attention_splitk_candidates(config),
        |options, candidates| {
            autotune::tune_candidates(
                stream,
                "fmha_decode_splitk_reduce_f32",
                tune_key,
                options,
                candidates,
                |tuning| {
                    launch_fmha_decode_splitk_reduce_f32(
                        stream,
                        out,
                        attn_scratch,
                        lse_scratch,
                        query,
                        key,
                        value,
                        config,
                        *tuning,
                    )
                },
            )
        },
    )?;
    launch_fmha_decode_splitk_reduce_f32(
        stream,
        out,
        attn_scratch,
        lse_scratch,
        query,
        key,
        value,
        config,
        tuning.unwrap_or(default_tuning),
    )
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn tune_fmha_decode_splitk_reduce_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    attn_scratch: &mut impl DeviceOutput<f32>,
    lse_scratch: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: FmhaDecodeSplitKReduceConfig,
    options: TuneOptions,
) -> Result<TuneReport<AttentionSplitKTuning>> {
    validate_fmha_decode_splitk_reduce_lengths(
        config,
        out.len(),
        attn_scratch.len(),
        lse_scratch.len(),
        query.len(),
        key.len(),
        value.len(),
    )?;
    let tune_key =
        attention_splitk_tune_key(stream, "fmha_decode_splitk_reduce_f32", "f32", config)?;
    let report = autotune::tune_candidates(
        stream,
        "fmha_decode_splitk_reduce_f32",
        tune_key.clone(),
        options,
        attention_splitk_candidates(config),
        |tuning| {
            launch_fmha_decode_splitk_reduce_f32(
                stream,
                out,
                attn_scratch,
                lse_scratch,
                query,
                key,
                value,
                config,
                *tuning,
            )
        },
    )?;
    autotune::cache().insert(tune_key, report.best);
    launch_fmha_decode_splitk_reduce_f32(
        stream,
        out,
        attn_scratch,
        lse_scratch,
        query,
        key,
        value,
        config,
        report.best,
    )?;
    Ok(report)
}
