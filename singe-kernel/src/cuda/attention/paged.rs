#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
#[cfg(feature = "dtype_f32")]
use singe_cuda::view::DeviceInputView;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use singe_cuda::{
    stream::Stream,
    view::{DeviceOutput, DeviceRange},
};

#[cfg(feature = "dtype_f32")]
use crate::cuda::attention::{
    PackedQkvAttentionConfig, PagedKvAttentionConfig, sliding_window_attention_f32,
    validate_packed_qkv_attention_config, validate_paged_kv_attention_config,
};
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use crate::{
    cuda::{
        attention::{
            PagedKvDecodeAttentionConfig, PagedKvPrefillAttentionConfig,
            RaggedKvPrefillAttentionConfig,
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
pub unsafe fn sliding_window_attention_packed_qkv_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    qkv: &impl DeviceRange<f32>,
    config: PackedQkvAttentionConfig,
) -> Result<()> {
    validate_packed_qkv_attention_config(qkv.len(), config)?;
    let packed = DeviceInputView::from_buffer(qkv);
    let query = packed.slice(config.query_offset..)?;
    let key = packed.slice(config.key_offset..)?;
    let value = packed.slice(config.value_offset..)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { sliding_window_attention_f32(stream, out, &query, &key, &value, config.attention) }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn sliding_window_attention_paged_kv_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    query: &impl DeviceRange<f32>,
    key_cache: &impl DeviceRange<f32>,
    value_cache: &impl DeviceRange<f32>,
    block_table: &impl DeviceRange<u32>,
    config: PagedKvAttentionConfig,
) -> Result<()> {
    validate_paged_kv_attention_config(
        out.len(),
        query.len(),
        key_cache.len(),
        value_cache.len(),
        block_table.len(),
        config,
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::sliding_window_attention_paged_kv_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key_cache),
            input_pointer(value_cache),
            input_pointer(block_table),
            config.attention.batch,
            config.attention.query_len,
            config.attention.key_len,
            config.attention.heads,
            config.attention.kv_heads,
            config.attention.head_dim,
            config.attention.query_start,
            config.attention.key_start,
            config.attention.window,
            config.attention.query_batch_stride,
            config.attention.output_batch_stride,
            config.attention.query_sequence_stride,
            config.attention.key_sequence_stride,
            config.attention.value_sequence_stride,
            config.attention.output_sequence_stride,
            config.attention.query_head_stride,
            config.attention.key_head_stride,
            config.attention.value_head_stride,
            config.attention.output_head_stride,
            config.block_size,
            config.physical_blocks,
            config.block_table_batch_stride,
            config.key_cache_block_stride,
            config.value_cache_block_stride,
            config.attention.scale,
            config.attention.output_scale,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn paged_kv_decode_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    query: &impl DeviceRange<f32>,
    key_cache: &impl DeviceRange<f32>,
    value_cache: &impl DeviceRange<f32>,
    actual_seq_lens: &impl DeviceRange<i32>,
    block_table: &impl DeviceRange<u32>,
    config: PagedKvDecodeAttentionConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        query.len(),
        key_cache.len(),
        value_cache.len(),
        actual_seq_lens.len(),
        block_table.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::paged_kv_decode_attention_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key_cache),
            input_pointer(value_cache),
            input_pointer(actual_seq_lens),
            input_pointer(block_table),
            config.batch,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.block_size,
            config.physical_blocks,
            config.block_table_batch_stride,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn paged_kv_decode_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    query: &impl DeviceRange<f16>,
    key_cache: &impl DeviceRange<f16>,
    value_cache: &impl DeviceRange<f16>,
    actual_seq_lens: &impl DeviceRange<i32>,
    block_table: &impl DeviceRange<u32>,
    config: PagedKvDecodeAttentionConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        query.len(),
        key_cache.len(),
        value_cache.len(),
        actual_seq_lens.len(),
        block_table.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::paged_kv_decode_attention_f16(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key_cache),
            input_pointer(value_cache),
            input_pointer(actual_seq_lens),
            input_pointer(block_table),
            config.batch,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.block_size,
            config.physical_blocks,
            config.block_table_batch_stride,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn paged_kv_decode_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    query: &impl DeviceRange<bf16>,
    key_cache: &impl DeviceRange<bf16>,
    value_cache: &impl DeviceRange<bf16>,
    actual_seq_lens: &impl DeviceRange<i32>,
    block_table: &impl DeviceRange<u32>,
    config: PagedKvDecodeAttentionConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        query.len(),
        key_cache.len(),
        value_cache.len(),
        actual_seq_lens.len(),
        block_table.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::paged_kv_decode_attention_bf16(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key_cache),
            input_pointer(value_cache),
            input_pointer(actual_seq_lens),
            input_pointer(block_table),
            config.batch,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.block_size,
            config.physical_blocks,
            config.block_table_batch_stride,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn paged_kv_prefill_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceRange<f32>,
    key_cache: &impl DeviceRange<f32>,
    value_cache: &impl DeviceRange<f32>,
    actual_seq_lens_q: &impl DeviceRange<i32>,
    actual_seq_lens_kv: &impl DeviceRange<i32>,
    actual_seq_offsets: &impl DeviceRange<i32>,
    block_table: &impl DeviceRange<u32>,
    config: PagedKvPrefillAttentionConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        key_cache.len(),
        value_cache.len(),
        actual_seq_lens_q.len(),
        actual_seq_lens_kv.len(),
        actual_seq_offsets.len(),
        block_table.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::paged_kv_prefill_attention_f32(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key_cache),
            input_pointer(value_cache),
            input_pointer(actual_seq_lens_q),
            input_pointer(actual_seq_lens_kv),
            input_pointer(actual_seq_offsets),
            input_pointer(block_table),
            config.batch,
            config.total_query_tokens,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.block_size,
            config.physical_blocks,
            config.block_table_batch_stride,
            config.causal,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn paged_kv_prefill_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceRange<f16>,
    key_cache: &impl DeviceRange<f16>,
    value_cache: &impl DeviceRange<f16>,
    actual_seq_lens_q: &impl DeviceRange<i32>,
    actual_seq_lens_kv: &impl DeviceRange<i32>,
    actual_seq_offsets: &impl DeviceRange<i32>,
    block_table: &impl DeviceRange<u32>,
    config: PagedKvPrefillAttentionConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        key_cache.len(),
        value_cache.len(),
        actual_seq_lens_q.len(),
        actual_seq_lens_kv.len(),
        actual_seq_offsets.len(),
        block_table.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::paged_kv_prefill_attention_f16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key_cache),
            input_pointer(value_cache),
            input_pointer(actual_seq_lens_q),
            input_pointer(actual_seq_lens_kv),
            input_pointer(actual_seq_offsets),
            input_pointer(block_table),
            config.batch,
            config.total_query_tokens,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.block_size,
            config.physical_blocks,
            config.block_table_batch_stride,
            config.causal,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn paged_kv_prefill_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceRange<bf16>,
    key_cache: &impl DeviceRange<bf16>,
    value_cache: &impl DeviceRange<bf16>,
    actual_seq_lens_q: &impl DeviceRange<i32>,
    actual_seq_lens_kv: &impl DeviceRange<i32>,
    actual_seq_offsets: &impl DeviceRange<i32>,
    block_table: &impl DeviceRange<u32>,
    config: PagedKvPrefillAttentionConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        key_cache.len(),
        value_cache.len(),
        actual_seq_lens_q.len(),
        actual_seq_lens_kv.len(),
        actual_seq_offsets.len(),
        block_table.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::paged_kv_prefill_attention_bf16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key_cache),
            input_pointer(value_cache),
            input_pointer(actual_seq_lens_q),
            input_pointer(actual_seq_lens_kv),
            input_pointer(actual_seq_offsets),
            input_pointer(block_table),
            config.batch,
            config.total_query_tokens,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.block_size,
            config.physical_blocks,
            config.block_table_batch_stride,
            config.causal,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn ragged_kv_prefill_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceRange<f32>,
    key: &impl DeviceRange<f32>,
    value: &impl DeviceRange<f32>,
    actual_seq_lens_q: &impl DeviceRange<i32>,
    actual_seq_lens_kv: &impl DeviceRange<i32>,
    actual_seq_offsets: &impl DeviceRange<i32>,
    config: RaggedKvPrefillAttentionConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        key.len(),
        value.len(),
        actual_seq_lens_q.len(),
        actual_seq_lens_kv.len(),
        actual_seq_offsets.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::ragged_kv_prefill_attention_f32(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            input_pointer(actual_seq_lens_q),
            input_pointer(actual_seq_lens_kv),
            input_pointer(actual_seq_offsets),
            config.batch,
            config.total_query_tokens,
            config.total_kv_tokens,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.causal,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn ragged_kv_prefill_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceRange<f16>,
    key: &impl DeviceRange<f16>,
    value: &impl DeviceRange<f16>,
    actual_seq_lens_q: &impl DeviceRange<i32>,
    actual_seq_lens_kv: &impl DeviceRange<i32>,
    actual_seq_offsets: &impl DeviceRange<i32>,
    config: RaggedKvPrefillAttentionConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        key.len(),
        value.len(),
        actual_seq_lens_q.len(),
        actual_seq_lens_kv.len(),
        actual_seq_offsets.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::ragged_kv_prefill_attention_f16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            input_pointer(actual_seq_lens_q),
            input_pointer(actual_seq_lens_kv),
            input_pointer(actual_seq_offsets),
            config.batch,
            config.total_query_tokens,
            config.total_kv_tokens,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.causal,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn ragged_kv_prefill_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceRange<bf16>,
    key: &impl DeviceRange<bf16>,
    value: &impl DeviceRange<bf16>,
    actual_seq_lens_q: &impl DeviceRange<i32>,
    actual_seq_lens_kv: &impl DeviceRange<i32>,
    actual_seq_offsets: &impl DeviceRange<i32>,
    config: RaggedKvPrefillAttentionConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        key.len(),
        value.len(),
        actual_seq_lens_q.len(),
        actual_seq_lens_kv.len(),
        actual_seq_offsets.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::ragged_kv_prefill_attention_bf16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            input_pointer(actual_seq_lens_q),
            input_pointer(actual_seq_lens_kv),
            input_pointer(actual_seq_offsets),
            config.batch,
            config.total_query_tokens,
            config.total_kv_tokens,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.causal,
            config.scale,
        )
    }
}
