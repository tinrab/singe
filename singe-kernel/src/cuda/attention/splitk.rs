#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16"))]
use crate::cuda::attention::FmhaDecodeSplitKConfig;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use crate::{
    cuda::{
        attention::{AttentionSinkDecodeSplitKConfig, SoftcappedWindowDecodeSplitKConfig},
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
pub unsafe fn softcapped_window_decode_splitk_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: SoftcappedWindowDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let decode = config.decode;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::softcapped_window_decode_splitk_f32(
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
            decode.splits,
            decode.kv_len_per_split,
            decode.head_dim,
            decode.query_batch_stride,
            decode.key_batch_stride,
            decode.value_batch_stride,
            decode.output_batch_stride,
            decode.query_head_stride,
            decode.key_sequence_stride,
            decode.value_sequence_stride,
            decode.output_head_stride,
            decode.output_split_stride,
            decode.key_head_stride,
            decode.value_head_stride,
            decode.lse_batch_stride,
            decode.lse_head_stride,
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
pub unsafe fn softcapped_window_decode_splitk_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    config: SoftcappedWindowDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let decode = config.decode;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::softcapped_window_decode_splitk_f16(
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
            decode.splits,
            decode.kv_len_per_split,
            decode.head_dim,
            decode.query_batch_stride,
            decode.key_batch_stride,
            decode.value_batch_stride,
            decode.output_batch_stride,
            decode.query_head_stride,
            decode.key_sequence_stride,
            decode.value_sequence_stride,
            decode.output_head_stride,
            decode.output_split_stride,
            decode.key_head_stride,
            decode.value_head_stride,
            decode.lse_batch_stride,
            decode.lse_head_stride,
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
pub unsafe fn softcapped_window_decode_splitk_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<bf16>,
    key: &impl DeviceInput<bf16>,
    value: &impl DeviceInput<bf16>,
    config: SoftcappedWindowDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let decode = config.decode;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::softcapped_window_decode_splitk_bf16(
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
            decode.splits,
            decode.kv_len_per_split,
            decode.head_dim,
            decode.query_batch_stride,
            decode.key_batch_stride,
            decode.value_batch_stride,
            decode.output_batch_stride,
            decode.query_head_stride,
            decode.key_sequence_stride,
            decode.value_sequence_stride,
            decode.output_head_stride,
            decode.output_split_stride,
            decode.key_head_stride,
            decode.value_head_stride,
            decode.lse_batch_stride,
            decode.lse_head_stride,
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
pub unsafe fn fmha_decode_splitk_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    config: FmhaDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_decode_splitk_f16(
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

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fmha_decode_splitk_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<bf16>,
    key: &impl DeviceInput<bf16>,
    value: &impl DeviceInput<bf16>,
    config: FmhaDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fmha_decode_splitk_bf16(
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
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn attention_sink_decode_splitk_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    sinks: &impl DeviceInput<f32>,
    config: AttentionSinkDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        key.len(),
        value.len(),
        sinks.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::attention_sink_decode_splitk_f32(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            input_pointer(sinks),
            config.decode.batch,
            config.decode.key_len,
            config.decode.heads,
            config.decode.kv_heads,
            config.decode.splits,
            config.decode.kv_len_per_split,
            config.decode.head_dim,
            config.decode.query_batch_stride,
            config.decode.key_batch_stride,
            config.decode.value_batch_stride,
            config.decode.output_batch_stride,
            config.decode.query_head_stride,
            config.decode.key_sequence_stride,
            config.decode.value_sequence_stride,
            config.decode.output_head_stride,
            config.decode.output_split_stride,
            config.decode.key_head_stride,
            config.decode.value_head_stride,
            config.decode.lse_batch_stride,
            config.decode.lse_head_stride,
            config.start_q,
            config.window,
            config.decode.scale,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn attention_sink_decode_splitk_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    sinks: &impl DeviceInput<f32>,
    config: AttentionSinkDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        key.len(),
        value.len(),
        sinks.len(),
    )?;
    let decode = config.decode;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::attention_sink_decode_splitk_f16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            input_pointer(sinks),
            decode.batch,
            decode.key_len,
            decode.heads,
            decode.kv_heads,
            decode.splits,
            decode.kv_len_per_split,
            decode.head_dim,
            decode.query_batch_stride,
            decode.key_batch_stride,
            decode.value_batch_stride,
            decode.output_batch_stride,
            decode.query_head_stride,
            decode.key_sequence_stride,
            decode.value_sequence_stride,
            decode.output_head_stride,
            decode.output_split_stride,
            decode.key_head_stride,
            decode.value_head_stride,
            decode.lse_batch_stride,
            decode.lse_head_stride,
            config.start_q,
            config.window,
            decode.scale,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn attention_sink_decode_splitk_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<bf16>,
    key: &impl DeviceInput<bf16>,
    value: &impl DeviceInput<bf16>,
    sinks: &impl DeviceInput<f32>,
    config: AttentionSinkDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        key.len(),
        value.len(),
        sinks.len(),
    )?;
    let decode = config.decode;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::attention_sink_decode_splitk_bf16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            input_pointer(sinks),
            decode.batch,
            decode.key_len,
            decode.heads,
            decode.kv_heads,
            decode.splits,
            decode.kv_len_per_split,
            decode.head_dim,
            decode.query_batch_stride,
            decode.key_batch_stride,
            decode.value_batch_stride,
            decode.output_batch_stride,
            decode.query_head_stride,
            decode.key_sequence_stride,
            decode.value_sequence_stride,
            decode.output_head_stride,
            decode.output_split_stride,
            decode.key_head_stride,
            decode.value_head_stride,
            decode.lse_batch_stride,
            decode.lse_head_stride,
            config.start_q,
            config.window,
            decode.scale,
        )
    }
}
