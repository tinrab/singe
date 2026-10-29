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
use crate::cuda::attention::SparseMlaPrefillConfig;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use crate::{
    cuda::{
        attention::{
            MlaDecodeConfig, MlaDecodeSplitKConfig, MlaPrefillConfig, PagedMlaDecodeAttentionConfig,
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
pub unsafe fn mla_prefill_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    query_pe: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    key_pe: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: MlaPrefillConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        query.len(),
        query_pe.len(),
        key.len(),
        key_pe.len(),
        value.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::mla_prefill_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(query_pe),
            input_pointer(key),
            input_pointer(key_pe),
            input_pointer(value),
            config.batch,
            config.query_len,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.pe_dim,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn mla_prefill_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    query: &impl DeviceInput<f16>,
    query_pe: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    key_pe: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    config: MlaPrefillConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        query.len(),
        query_pe.len(),
        key.len(),
        key_pe.len(),
        value.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::mla_prefill_f16(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(query_pe),
            input_pointer(key),
            input_pointer(key_pe),
            input_pointer(value),
            config.batch,
            config.query_len,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.pe_dim,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn mla_prefill_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    query: &impl DeviceInput<bf16>,
    query_pe: &impl DeviceInput<bf16>,
    key: &impl DeviceInput<bf16>,
    key_pe: &impl DeviceInput<bf16>,
    value: &impl DeviceInput<bf16>,
    config: MlaPrefillConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        query.len(),
        query_pe.len(),
        key.len(),
        key_pe.len(),
        value.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::mla_prefill_bf16(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(query_pe),
            input_pointer(key),
            input_pointer(key_pe),
            input_pointer(value),
            config.batch,
            config.query_len,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.pe_dim,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn sparse_mla_prefill_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    query_pe: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    key_pe: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    indices: &impl DeviceInput<i32>,
    config: SparseMlaPrefillConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        query.len(),
        query_pe.len(),
        key.len(),
        key_pe.len(),
        value.len(),
        indices.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::sparse_mla_prefill_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(query_pe),
            input_pointer(key),
            input_pointer(key_pe),
            input_pointer(value),
            input_pointer(indices),
            config.batch,
            config.query_len,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.pe_dim,
            config.topk,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn mla_decode_lse_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    query_pe: &impl DeviceInput<f32>,
    key_value: &impl DeviceInput<f32>,
    key_pe: &impl DeviceInput<f32>,
    config: MlaDecodeConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        query_pe.len(),
        key_value.len(),
        key_pe.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::mla_decode_lse_f32(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(query_pe),
            input_pointer(key_value),
            input_pointer(key_pe),
            config.batch,
            config.key_len,
            config.heads,
            config.head_dim,
            config.pe_dim,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn paged_mla_decode_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    query_pe: &impl DeviceInput<f32>,
    key_value_cache: &impl DeviceInput<f32>,
    key_pe_cache: &impl DeviceInput<f32>,
    actual_seq_lens: &impl DeviceInput<i32>,
    block_table: &impl DeviceInput<u32>,
    config: PagedMlaDecodeAttentionConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        query_pe.len(),
        key_value_cache.len(),
        key_pe_cache.len(),
        actual_seq_lens.len(),
        block_table.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::paged_mla_decode_attention_f32(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(query_pe),
            input_pointer(key_value_cache),
            input_pointer(key_pe_cache),
            input_pointer(actual_seq_lens),
            input_pointer(block_table),
            config.batch,
            config.heads,
            config.head_dim,
            config.pe_dim,
            config.block_size,
            config.physical_blocks,
            config.block_table_batch_stride,
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
pub unsafe fn paged_mla_decode_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f16>,
    query_pe: &impl DeviceInput<f16>,
    key_value_cache: &impl DeviceInput<f16>,
    key_pe_cache: &impl DeviceInput<f16>,
    actual_seq_lens: &impl DeviceInput<i32>,
    block_table: &impl DeviceInput<u32>,
    config: PagedMlaDecodeAttentionConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        query_pe.len(),
        key_value_cache.len(),
        key_pe_cache.len(),
        actual_seq_lens.len(),
        block_table.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::paged_mla_decode_attention_f16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(query_pe),
            input_pointer(key_value_cache),
            input_pointer(key_pe_cache),
            input_pointer(actual_seq_lens),
            input_pointer(block_table),
            config.batch,
            config.heads,
            config.head_dim,
            config.pe_dim,
            config.block_size,
            config.physical_blocks,
            config.block_table_batch_stride,
            config.scale,
            config.output_scale,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn paged_mla_decode_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<bf16>,
    query_pe: &impl DeviceInput<bf16>,
    key_value_cache: &impl DeviceInput<bf16>,
    key_pe_cache: &impl DeviceInput<bf16>,
    actual_seq_lens: &impl DeviceInput<i32>,
    block_table: &impl DeviceInput<u32>,
    config: PagedMlaDecodeAttentionConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        query_pe.len(),
        key_value_cache.len(),
        key_pe_cache.len(),
        actual_seq_lens.len(),
        block_table.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::paged_mla_decode_attention_bf16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(query_pe),
            input_pointer(key_value_cache),
            input_pointer(key_pe_cache),
            input_pointer(actual_seq_lens),
            input_pointer(block_table),
            config.batch,
            config.heads,
            config.head_dim,
            config.pe_dim,
            config.block_size,
            config.physical_blocks,
            config.block_table_batch_stride,
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
pub unsafe fn mla_decode_lse_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f16>,
    query_pe: &impl DeviceInput<f16>,
    key_value: &impl DeviceInput<f16>,
    key_pe: &impl DeviceInput<f16>,
    config: MlaDecodeConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        query_pe.len(),
        key_value.len(),
        key_pe.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::mla_decode_lse_f16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(query_pe),
            input_pointer(key_value),
            input_pointer(key_pe),
            config.batch,
            config.key_len,
            config.heads,
            config.head_dim,
            config.pe_dim,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn mla_decode_lse_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<bf16>,
    query_pe: &impl DeviceInput<bf16>,
    key_value: &impl DeviceInput<bf16>,
    key_pe: &impl DeviceInput<bf16>,
    config: MlaDecodeConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        query_pe.len(),
        key_value.len(),
        key_pe.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::mla_decode_lse_bf16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(query_pe),
            input_pointer(key_value),
            input_pointer(key_pe),
            config.batch,
            config.key_len,
            config.heads,
            config.head_dim,
            config.pe_dim,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn mla_decode_splitk_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    query_pe: &impl DeviceInput<f32>,
    key_value: &impl DeviceInput<f32>,
    key_pe: &impl DeviceInput<f32>,
    config: MlaDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        query_pe.len(),
        key_value.len(),
        key_pe.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::mla_decode_splitk_f32(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(query_pe),
            input_pointer(key_value),
            input_pointer(key_pe),
            config.batch,
            config.key_len,
            config.heads,
            config.splits,
            config.kv_len_per_split,
            config.head_dim,
            config.pe_dim,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn mla_decode_splitk_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f16>,
    query_pe: &impl DeviceInput<f16>,
    key_value: &impl DeviceInput<f16>,
    key_pe: &impl DeviceInput<f16>,
    config: MlaDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        query_pe.len(),
        key_value.len(),
        key_pe.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::mla_decode_splitk_f16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(query_pe),
            input_pointer(key_value),
            input_pointer(key_pe),
            config.batch,
            config.key_len,
            config.heads,
            config.splits,
            config.kv_len_per_split,
            config.head_dim,
            config.pe_dim,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn mla_decode_splitk_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    lse: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<bf16>,
    query_pe: &impl DeviceInput<bf16>,
    key_value: &impl DeviceInput<bf16>,
    key_pe: &impl DeviceInput<bf16>,
    config: MlaDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        lse.len(),
        query.len(),
        query_pe.len(),
        key_value.len(),
        key_pe.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::mla_decode_splitk_bf16(
            &stream,
            write_only_pointer(out),
            write_only_pointer(lse),
            input_pointer(query),
            input_pointer(query_pe),
            input_pointer(key_value),
            input_pointer(key_pe),
            config.batch,
            config.key_len,
            config.heads,
            config.splits,
            config.kv_len_per_split,
            config.head_dim,
            config.pe_dim,
            config.scale,
        )
    }
}
