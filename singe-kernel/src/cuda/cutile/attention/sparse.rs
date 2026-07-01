use super::*;

#[cfg(feature = "dtype-f32")]
pub fn fused_neighborhood_attention_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key: DevicePointer<f32>,
    value: DevicePointer<f32>,
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    kernel_size: usize,
    dilation: usize,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(value)?;

    let params = FusedNeighborhoodAttention::create(
        batch,
        seq_len,
        heads,
        head_dim,
        kernel_size,
        dilation,
        scale,
    )?;
    let _batch = params.batch;
    unsafe {
        kernel_attention::fused_neighborhood_attention_f32(
            out,
            query,
            key,
            value,
            params.seq_len,
            params.heads,
            params.head_dim,
            params.kernel_size,
            params.dilation,
            scale,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn heavily_compressed_attention_compress_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    hidden: DevicePointer<f32>,
    weight_kv: DevicePointer<f32>,
    weight_z: DevicePointer<f32>,
    bias: DevicePointer<f32>,
    batch: usize,
    seq_len: usize,
    hidden_dim: usize,
    head_dim: usize,
    compression_block: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(hidden)?;
    checked_device_pointer(weight_kv)?;
    checked_device_pointer(weight_z)?;
    checked_device_pointer(bias)?;

    let params = HeavilyCompressedAttentionCompress::create(
        batch,
        seq_len,
        hidden_dim,
        head_dim,
        compression_block,
    )?;
    let _batch = params.batch;
    unsafe {
        kernel_attention::heavily_compressed_attention_compress_f32(
            out,
            hidden,
            weight_kv,
            weight_z,
            bias,
            params.seq_len,
            params.hidden_dim,
            params.head_dim,
            params.compression_block,
            params.blocks,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn compressed_sparse_attention_lightning_indexer_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    indexer_query: DevicePointer<f32>,
    indexer_key: DevicePointer<f32>,
    indexer_weight: DevicePointer<f32>,
    batch: usize,
    query_len: usize,
    blocks: usize,
    index_heads: usize,
    index_dim: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(indexer_query)?;
    checked_device_pointer(indexer_key)?;
    checked_device_pointer(indexer_weight)?;

    let params = CompressedSparseAttentionLightningIndexer::create(
        batch,
        query_len,
        blocks,
        index_heads,
        index_dim,
    )?;
    let _batch = params.batch;
    unsafe {
        kernel_attention::compressed_sparse_attention_lightning_indexer_f32(
            out,
            indexer_query,
            indexer_key,
            indexer_weight,
            params.query_len,
            params.blocks,
            params.index_heads,
            params.index_dim,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn compressed_sparse_attention_topk_selector_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    selected_indices: DevicePointer<i32>,
    scores: DevicePointer<f32>,
    compressed_kv: DevicePointer<f32>,
    query_positions: DevicePointer<i32>,
    batch: usize,
    query_len: usize,
    blocks: usize,
    head_dim: usize,
    top_k: usize,
    compression_block: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(selected_indices)?;
    checked_device_pointer(scores)?;
    checked_device_pointer(compressed_kv)?;
    checked_device_pointer(query_positions)?;

    let params = CompressedSparseAttentionTopkSelector::create(
        batch,
        query_len,
        blocks,
        head_dim,
        top_k,
        compression_block,
    )?;
    let _batch = params.batch;
    unsafe {
        kernel_attention::compressed_sparse_attention_topk_selector_f32(
            out,
            selected_indices,
            scores,
            compressed_kv,
            query_positions,
            params.query_len,
            params.blocks,
            params.head_dim,
            params.top_k,
            params.compression_block,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn compressed_sparse_attention_shared_mqa_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    query: DevicePointer<f32>,
    kv_entries: DevicePointer<f32>,
    valid_mask: DevicePointer<i32>,
    sink: DevicePointer<f32>,
    batch: usize,
    query_len: usize,
    heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(query)?;
    checked_device_pointer(kv_entries)?;
    checked_device_pointer(valid_mask)?;
    checked_device_pointer(sink)?;

    let params = CompressedSparseAttentionSharedMqa::create(
        batch, query_len, heads, head_dim, kv_len, scale,
    )?;
    let _batch = params.batch;
    unsafe {
        kernel_attention::compressed_sparse_attention_shared_mqa_f32(
            out,
            query,
            kv_entries,
            valid_mask,
            sink,
            params.query_len,
            params.heads,
            params.head_dim,
            params.kv_len,
            scale,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn minimax_sparse_attention_block_max_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    scores: DevicePointer<f32>,
    batch: usize,
    rows: usize,
    key_len: usize,
    block_size: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(scores)?;

    let params = MiniMaxSparseAttentionBlockMax::create(batch, rows, key_len, block_size)?;
    let _batch = params.batch;
    unsafe {
        kernel_attention::minimax_sparse_attention_block_max_f32(
            out,
            scores,
            params.rows,
            params.key_len,
            params.block_size,
            params.blocks,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

pub fn minimax_sparse_attention_selected_token_positions_i32(
    stream: &Arc<Stream>,
    positions: DevicePointer<i32>,
    valid_mask: DevicePointer<i32>,
    selected_blocks: DevicePointer<i32>,
    query_positions: DevicePointer<i32>,
    batch: usize,
    rows: usize,
    selected_block_count: usize,
    block_size: usize,
    seq_len: usize,
) -> Result<()> {
    checked_device_pointer(positions)?;
    checked_device_pointer(valid_mask)?;
    checked_device_pointer(selected_blocks)?;
    checked_device_pointer(query_positions)?;

    let params = MiniMaxSparseAttentionSelectedTokenPositions::create(
        batch,
        rows,
        selected_block_count,
        block_size,
        seq_len,
    )?;
    let _batch = params.batch;
    unsafe {
        kernel_attention::minimax_sparse_attention_selected_token_positions_i32(
            positions,
            valid_mask,
            selected_blocks,
            query_positions,
            params.rows,
            params.selected_blocks,
            params.block_size,
            params.seq_len,
            params.expanded_keys,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn minimax_sparse_attention_select_topk_blocks_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<i32>,
    block_scores: DevicePointer<f32>,
    local_blocks: DevicePointer<i32>,
    batch: usize,
    rows: usize,
    blocks: usize,
    top_k: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(block_scores)?;
    checked_device_pointer(local_blocks)?;

    let params = MiniMaxSparseAttentionSelectTopkBlocks::create(batch, rows, blocks, top_k)?;
    let _batch = params.batch;
    unsafe {
        kernel_attention::minimax_sparse_attention_select_topk_blocks_f32(
            out,
            block_scores,
            local_blocks,
            params.rows,
            params.blocks,
            params.top_k,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn minimax_sparse_attention_gathered_gqa_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key: DevicePointer<f32>,
    value: DevicePointer<f32>,
    positions: DevicePointer<i32>,
    valid_mask: DevicePointer<i32>,
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    selected_keys: usize,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(value)?;
    checked_device_pointer(positions)?;
    checked_device_pointer(valid_mask)?;

    let params = MiniMaxSparseAttentionGatheredGqa::create(
        batch,
        query_len,
        key_len,
        heads,
        kv_heads,
        head_dim,
        selected_keys,
        scale,
    )?;
    let _batch = params.batch;
    unsafe {
        kernel_attention::minimax_sparse_attention_gathered_gqa_f32(
            out,
            query,
            key,
            value,
            positions,
            valid_mask,
            params.query_len,
            params.key_len,
            params.heads,
            params.kv_heads,
            params.head_dim,
            params.selected_keys,
            scale,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn compressed_sparse_attention_compress_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    hidden: DevicePointer<f32>,
    weight_a_kv: DevicePointer<f32>,
    weight_b_kv: DevicePointer<f32>,
    weight_a_z: DevicePointer<f32>,
    weight_b_z: DevicePointer<f32>,
    bias_a: DevicePointer<f32>,
    bias_b: DevicePointer<f32>,
    batch: usize,
    seq_len: usize,
    hidden_dim: usize,
    head_dim: usize,
    compression_block: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(hidden)?;
    checked_device_pointer(weight_a_kv)?;
    checked_device_pointer(weight_b_kv)?;
    checked_device_pointer(weight_a_z)?;
    checked_device_pointer(weight_b_z)?;
    checked_device_pointer(bias_a)?;
    checked_device_pointer(bias_b)?;

    let params = HeavilyCompressedAttentionCompress::create(
        batch,
        seq_len,
        hidden_dim,
        head_dim,
        compression_block,
    )?;
    let _batch = params.batch;
    unsafe {
        kernel_attention::compressed_sparse_attention_compress_f32(
            out,
            hidden,
            weight_a_kv,
            weight_b_kv,
            weight_a_z,
            weight_b_z,
            bias_a,
            bias_b,
            params.seq_len,
            params.hidden_dim,
            params.head_dim,
            params.compression_block,
            params.blocks,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn heavily_compressed_attention_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    query: DevicePointer<f32>,
    compressed_kv: DevicePointer<f32>,
    weight_group: DevicePointer<f32>,
    weight_final: DevicePointer<f32>,
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    compression_block: usize,
    groups: usize,
    group_dim: usize,
    hidden_dim: usize,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(query)?;
    checked_device_pointer(compressed_kv)?;
    checked_device_pointer(weight_group)?;
    checked_device_pointer(weight_final)?;

    let params = HeavilyCompressedAttention::create(
        batch,
        seq_len,
        heads,
        head_dim,
        compression_block,
        groups,
        group_dim,
        hidden_dim,
        scale,
    )?;
    unsafe {
        kernel_attention::heavily_compressed_attention_f32(
            out,
            query,
            compressed_kv,
            weight_group,
            weight_final,
            params.seq_len,
            params.heads,
            params.head_dim,
            params.compression_block,
            params.blocks,
            params.groups,
            params.group_dim,
            params.hidden_dim,
            scale,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

macro_rules! fused_neighborhood_attention_half_fn {
    ($name:ident, $dtype:ty, $kernel:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$dtype>,
            query: DevicePointer<$dtype>,
            key: DevicePointer<$dtype>,
            value: DevicePointer<$dtype>,
            batch: usize,
            seq_len: usize,
            heads: usize,
            head_dim: usize,
            kernel_size: usize,
            dilation: usize,
            scale: f32,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key)?;
            checked_device_pointer(value)?;

            let params = FusedNeighborhoodAttention::create(
                batch,
                seq_len,
                heads,
                head_dim,
                kernel_size,
                dilation,
                scale,
            )?;
            let _batch = params.batch;
            unsafe {
                kernel_attention::$kernel(
                    out,
                    query,
                    key,
                    value,
                    params.seq_len,
                    params.heads,
                    params.head_dim,
                    params.kernel_size,
                    params.dilation,
                    scale,
                    params.output_len,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
fused_neighborhood_attention_half_fn!(
    fused_neighborhood_attention_f16,
    f16,
    fused_neighborhood_attention_f16
);

#[cfg(feature = "dtype-bf16")]
fused_neighborhood_attention_half_fn!(
    fused_neighborhood_attention_bf16,
    bf16,
    fused_neighborhood_attention_bf16
);
