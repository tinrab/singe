#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use crate::{
    cuda::{
        attention::MultiTokenAttentionConfig,
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
pub unsafe fn multi_token_attention_f32<B>(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    scores: &impl DeviceInput<f32>,
    weight: &impl DeviceInput<f32>,
    bias: Option<&B>,
    config: MultiTokenAttentionConfig,
) -> Result<()>
where
    B: DeviceInput<f32>,
{
    config.validate_lengths(
        out.len(),
        scores.len(),
        weight.len(),
        bias.map(|bias| bias.len()),
    )?;
    let stream = stream.to_cuda_core()?;
    let scores_pointer = input_pointer(scores);
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::multi_token_attention_f32(
            &stream,
            write_only_pointer(out),
            scores_pointer,
            input_pointer(weight),
            bias.map_or(scores_pointer, input_pointer),
            config.batch,
            config.channels_in,
            config.channels_out,
            config.seq_len,
            config.kernel_h,
            config.kernel_w,
            config.stride_h,
            config.stride_w,
            config.padding_h,
            config.padding_w,
            config.dilation_h,
            config.dilation_w,
            config.groups,
            bias.is_some(),
            false,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn multi_token_attention_sparse_f32<B>(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    scores: &impl DeviceInput<f32>,
    weight: &impl DeviceInput<f32>,
    bias: Option<&B>,
    config: MultiTokenAttentionConfig,
) -> Result<()>
where
    B: DeviceInput<f32>,
{
    config.validate_lengths(
        out.len(),
        scores.len(),
        weight.len(),
        bias.map(|bias| bias.len()),
    )?;
    let stream = stream.to_cuda_core()?;
    let scores_pointer = input_pointer(scores);
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::multi_token_attention_f32(
            &stream,
            write_only_pointer(out),
            scores_pointer,
            input_pointer(weight),
            bias.map_or(scores_pointer, input_pointer),
            config.batch,
            config.channels_in,
            config.channels_out,
            config.seq_len,
            config.kernel_h,
            config.kernel_w,
            config.stride_h,
            config.stride_w,
            config.padding_h,
            config.padding_w,
            config.dilation_h,
            config.dilation_w,
            config.groups,
            bias.is_some(),
            true,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn multi_token_attention_f16<B>(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    scores: &impl DeviceInput<f16>,
    weight: &impl DeviceInput<f16>,
    bias: Option<&B>,
    config: MultiTokenAttentionConfig,
) -> Result<()>
where
    B: DeviceInput<f16>,
{
    config.validate_lengths(
        out.len(),
        scores.len(),
        weight.len(),
        bias.map(|bias| bias.len()),
    )?;
    let stream = stream.to_cuda_core()?;
    let scores_pointer = input_pointer(scores);
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::multi_token_attention_f16(
            &stream,
            write_only_pointer(out),
            scores_pointer,
            input_pointer(weight),
            bias.map_or(scores_pointer, input_pointer),
            config.batch,
            config.channels_in,
            config.channels_out,
            config.seq_len,
            config.kernel_h,
            config.kernel_w,
            config.stride_h,
            config.stride_w,
            config.padding_h,
            config.padding_w,
            config.dilation_h,
            config.dilation_w,
            config.groups,
            bias.is_some(),
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn multi_token_attention_sparse_f16<B>(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    scores: &impl DeviceInput<f16>,
    weight: &impl DeviceInput<f16>,
    bias: Option<&B>,
    config: MultiTokenAttentionConfig,
) -> Result<()>
where
    B: DeviceInput<f16>,
{
    config.validate_lengths(
        out.len(),
        scores.len(),
        weight.len(),
        bias.map(|bias| bias.len()),
    )?;
    let stream = stream.to_cuda_core()?;
    let scores_pointer = input_pointer(scores);
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::multi_token_attention_sparse_f16(
            &stream,
            write_only_pointer(out),
            scores_pointer,
            input_pointer(weight),
            bias.map_or(scores_pointer, input_pointer),
            config.batch,
            config.channels_in,
            config.channels_out,
            config.seq_len,
            config.kernel_h,
            config.kernel_w,
            config.stride_h,
            config.stride_w,
            config.padding_h,
            config.padding_w,
            config.dilation_h,
            config.dilation_w,
            config.groups,
            bias.is_some(),
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn multi_token_attention_bf16<B>(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    scores: &impl DeviceInput<bf16>,
    weight: &impl DeviceInput<bf16>,
    bias: Option<&B>,
    config: MultiTokenAttentionConfig,
) -> Result<()>
where
    B: DeviceInput<bf16>,
{
    config.validate_lengths(
        out.len(),
        scores.len(),
        weight.len(),
        bias.map(|bias| bias.len()),
    )?;
    let stream = stream.to_cuda_core()?;
    let scores_pointer = input_pointer(scores);
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::multi_token_attention_bf16(
            &stream,
            write_only_pointer(out),
            scores_pointer,
            input_pointer(weight),
            bias.map_or(scores_pointer, input_pointer),
            config.batch,
            config.channels_in,
            config.channels_out,
            config.seq_len,
            config.kernel_h,
            config.kernel_w,
            config.stride_h,
            config.stride_w,
            config.padding_h,
            config.padding_w,
            config.dilation_h,
            config.dilation_w,
            config.groups,
            bias.is_some(),
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn multi_token_attention_sparse_bf16<B>(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    scores: &impl DeviceInput<bf16>,
    weight: &impl DeviceInput<bf16>,
    bias: Option<&B>,
    config: MultiTokenAttentionConfig,
) -> Result<()>
where
    B: DeviceInput<bf16>,
{
    config.validate_lengths(
        out.len(),
        scores.len(),
        weight.len(),
        bias.map(|bias| bias.len()),
    )?;
    let stream = stream.to_cuda_core()?;
    let scores_pointer = input_pointer(scores);
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::multi_token_attention_sparse_bf16(
            &stream,
            write_only_pointer(out),
            scores_pointer,
            input_pointer(weight),
            bias.map_or(scores_pointer, input_pointer),
            config.batch,
            config.channels_in,
            config.channels_out,
            config.seq_len,
            config.kernel_h,
            config.kernel_w,
            config.stride_h,
            config.stride_w,
            config.padding_h,
            config.padding_w,
            config.dilation_h,
            config.dilation_w,
            config.groups,
            bias.is_some(),
        )
    }
}
