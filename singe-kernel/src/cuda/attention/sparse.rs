#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16"))]
use crate::cuda::attention::FusedNeighborhoodAttentionConfig;
#[cfg(feature = "dtype_f32")]
use crate::cuda::attention::{
    CompressedSparseAttentionCompressConfig, CompressedSparseAttentionLightningIndexerConfig,
    CompressedSparseAttentionSharedMqaConfig, CompressedSparseAttentionTopkSelectorConfig,
    HeavilyCompressedAttentionConfig, MiniMaxSparseAttentionBlockMaxConfig,
    MiniMaxSparseAttentionGatheredGqaConfig, MiniMaxSparseAttentionSelectTopkBlocksConfig,
};
use crate::{
    cuda::{
        attention::MiniMaxSparseAttentionSelectedTokenPositionsConfig,
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
pub unsafe fn heavily_compressed_attention_compress_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    hidden: &impl DeviceInput<f32>,
    weight_kv: &impl DeviceInput<f32>,
    weight_z: &impl DeviceInput<f32>,
    bias: &impl DeviceInput<f32>,
    config: HeavilyCompressedAttentionConfig,
) -> Result<()> {
    config.validate_compress_lengths(
        out.len(),
        hidden.len(),
        weight_kv.len(),
        weight_z.len(),
        bias.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::heavily_compressed_attention_compress_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(hidden),
            input_pointer(weight_kv),
            input_pointer(weight_z),
            input_pointer(bias),
            config.batch,
            config.seq_len,
            config.hidden_dim,
            config.head_dim,
            config.compression_block,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn compressed_sparse_attention_compress_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    hidden: &impl DeviceInput<f32>,
    weight_a_kv: &impl DeviceInput<f32>,
    weight_b_kv: &impl DeviceInput<f32>,
    weight_a_z: &impl DeviceInput<f32>,
    weight_b_z: &impl DeviceInput<f32>,
    bias_a: &impl DeviceInput<f32>,
    bias_b: &impl DeviceInput<f32>,
    config: CompressedSparseAttentionCompressConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        hidden.len(),
        weight_a_kv.len(),
        weight_b_kv.len(),
        weight_a_z.len(),
        weight_b_z.len(),
        bias_a.len(),
        bias_b.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::compressed_sparse_attention_compress_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(hidden),
            input_pointer(weight_a_kv),
            input_pointer(weight_b_kv),
            input_pointer(weight_a_z),
            input_pointer(weight_b_z),
            input_pointer(bias_a),
            input_pointer(bias_b),
            config.batch,
            config.seq_len,
            config.hidden_dim,
            config.head_dim,
            config.compression_block,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn compressed_sparse_attention_lightning_indexer_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    indexer_query: &impl DeviceInput<f32>,
    indexer_key: &impl DeviceInput<f32>,
    indexer_weight: &impl DeviceInput<f32>,
    config: CompressedSparseAttentionLightningIndexerConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        indexer_query.len(),
        indexer_key.len(),
        indexer_weight.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::compressed_sparse_attention_lightning_indexer_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(indexer_query),
            input_pointer(indexer_key),
            input_pointer(indexer_weight),
            config.batch,
            config.query_len,
            config.blocks,
            config.index_heads,
            config.index_dim,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn compressed_sparse_attention_topk_selector_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    selected_indices: &mut impl DeviceOutput<i32>,
    scores: &impl DeviceInput<f32>,
    compressed_kv: &impl DeviceInput<f32>,
    query_positions: &impl DeviceInput<i32>,
    config: CompressedSparseAttentionTopkSelectorConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        selected_indices.len(),
        scores.len(),
        compressed_kv.len(),
        query_positions.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::compressed_sparse_attention_topk_selector_f32(
            &stream,
            write_only_pointer(out),
            write_only_pointer(selected_indices),
            input_pointer(scores),
            input_pointer(compressed_kv),
            input_pointer(query_positions),
            config.batch,
            config.query_len,
            config.blocks,
            config.head_dim,
            config.top_k,
            config.compression_block,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn compressed_sparse_attention_shared_mqa_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    kv_entries: &impl DeviceInput<f32>,
    valid_mask: &impl DeviceInput<i32>,
    sink: &impl DeviceInput<f32>,
    config: CompressedSparseAttentionSharedMqaConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        query.len(),
        kv_entries.len(),
        valid_mask.len(),
        sink.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::compressed_sparse_attention_shared_mqa_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(kv_entries),
            input_pointer(valid_mask),
            input_pointer(sink),
            config.batch,
            config.query_len,
            config.heads,
            config.head_dim,
            config.kv_len,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn minimax_sparse_attention_block_max_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    scores: &impl DeviceInput<f32>,
    config: MiniMaxSparseAttentionBlockMaxConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), scores.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::minimax_sparse_attention_block_max_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(scores),
            config.batch,
            config.rows,
            config.key_len,
            config.block_size,
        )
    }
}

///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn minimax_sparse_attention_selected_token_positions_i32(
    stream: &Stream,
    positions: &mut impl DeviceOutput<i32>,
    valid_mask: &mut impl DeviceOutput<i32>,
    selected_blocks: &impl DeviceInput<i32>,
    query_positions: &impl DeviceInput<i32>,
    config: MiniMaxSparseAttentionSelectedTokenPositionsConfig,
) -> Result<()> {
    config.validate_lengths(
        positions.len(),
        valid_mask.len(),
        selected_blocks.len(),
        query_positions.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::minimax_sparse_attention_selected_token_positions_i32(
            &stream,
            write_only_pointer(positions),
            write_only_pointer(valid_mask),
            input_pointer(selected_blocks),
            input_pointer(query_positions),
            config.batch,
            config.rows,
            config.selected_blocks,
            config.block_size,
            config.seq_len,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn minimax_sparse_attention_select_topk_blocks_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<i32>,
    block_scores: &impl DeviceInput<f32>,
    local_blocks: &impl DeviceInput<i32>,
    config: MiniMaxSparseAttentionSelectTopkBlocksConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), block_scores.len(), local_blocks.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::minimax_sparse_attention_select_topk_blocks_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(block_scores),
            input_pointer(local_blocks),
            config.batch,
            config.rows,
            config.blocks,
            config.top_k,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn minimax_sparse_attention_gathered_gqa_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    positions: &impl DeviceInput<i32>,
    valid_mask: &impl DeviceInput<i32>,
    config: MiniMaxSparseAttentionGatheredGqaConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        query.len(),
        key.len(),
        value.len(),
        positions.len(),
        valid_mask.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::minimax_sparse_attention_gathered_gqa_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            input_pointer(positions),
            input_pointer(valid_mask),
            config.batch,
            config.query_len,
            config.key_len,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.selected_keys,
            config.scale,
        )
    }
}

#[cfg(feature = "dtype_f32")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn heavily_compressed_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    compressed_kv: &impl DeviceInput<f32>,
    weight_group: &impl DeviceInput<f32>,
    weight_final: &impl DeviceInput<f32>,
    config: HeavilyCompressedAttentionConfig,
) -> Result<()> {
    config.validate_attention_lengths(
        out.len(),
        query.len(),
        compressed_kv.len(),
        weight_group.len(),
        weight_final.len(),
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::heavily_compressed_attention_f32(
            &stream,
            write_only_pointer(out),
            input_pointer(query),
            input_pointer(compressed_kv),
            input_pointer(weight_group),
            input_pointer(weight_final),
            config.batch,
            config.seq_len,
            config.heads,
            config.head_dim,
            config.compression_block,
            config.groups,
            config.group_dim,
            config.hidden_dim,
            1.0 / (config.head_dim as f32).sqrt(),
        )
    }
}

#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fused_neighborhood_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    config: FusedNeighborhoodAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fused_neighborhood_attention_f16(
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

#[cfg(feature = "dtype_bf16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn fused_neighborhood_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    query: &impl DeviceInput<bf16>,
    key: &impl DeviceInput<bf16>,
    value: &impl DeviceInput<bf16>,
    config: FusedNeighborhoodAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::fused_neighborhood_attention_bf16(
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
