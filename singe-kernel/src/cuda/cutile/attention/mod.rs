mod fmha;
mod local;
mod mla;
mod multi_token;
mod paged;
mod sparse;
mod splitk;

pub use fmha::*;
pub use local::*;
pub use mla::*;
pub use multi_token::*;
pub use paged::*;
pub use sparse::*;
pub use splitk::*;

use std::sync::Arc;

#[cfg(feature = "dtype-bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

use crate::{
    cuda::cutile::{
        DeviceOpExt,
        adapter::TensorAdapter,
        kernel::attention as kernel_attention,
        utility::{checked_device_pointer, raw_vector_grid},
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SlidingWindowAttention {
    batch: i32,
    query_len: i32,
    key_len: i32,
    heads: i32,
    kv_heads: i32,
    query_start: i32,
    key_start: i32,
    window: i32,
    output_len: i32,
    output_values_per_batch: i32,
    query_batch_stride: i32,
    key_batch_stride: i32,
    value_batch_stride: i32,
    output_batch_stride: i32,
    query_sequence_stride: i32,
    key_sequence_stride: i32,
    value_sequence_stride: i32,
    output_sequence_stride: i32,
    query_head_stride: i32,
    key_head_stride: i32,
    value_head_stride: i32,
    output_head_stride: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FusedNeighborhoodAttention {
    batch: i32,
    seq_len: i32,
    heads: i32,
    head_dim: i32,
    kernel_size: i32,
    dilation: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MultiTokenAttention {
    batch: i32,
    channels_in: i32,
    channels_out: i32,
    seq_len: i32,
    output_seq_len: i32,
    kernel_h: i32,
    kernel_w: i32,
    stride_h: i32,
    stride_w: i32,
    padding_h: i32,
    padding_w: i32,
    dilation_h: i32,
    dilation_w: i32,
    groups: i32,
    has_bias: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HeavilyCompressedAttentionCompress {
    batch: i32,
    seq_len: i32,
    hidden_dim: i32,
    head_dim: i32,
    compression_block: i32,
    blocks: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompressedSparseAttentionLightningIndexer {
    batch: i32,
    query_len: i32,
    blocks: i32,
    index_heads: i32,
    index_dim: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompressedSparseAttentionTopkSelector {
    batch: i32,
    query_len: i32,
    blocks: i32,
    head_dim: i32,
    top_k: i32,
    compression_block: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompressedSparseAttentionSharedMqa {
    batch: i32,
    query_len: i32,
    heads: i32,
    head_dim: i32,
    kv_len: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MiniMaxSparseAttentionBlockMax {
    batch: i32,
    rows: i32,
    key_len: i32,
    block_size: i32,
    blocks: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MiniMaxSparseAttentionSelectedTokenPositions {
    batch: i32,
    rows: i32,
    selected_blocks: i32,
    block_size: i32,
    seq_len: i32,
    expanded_keys: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MiniMaxSparseAttentionSelectTopkBlocks {
    batch: i32,
    rows: i32,
    blocks: i32,
    top_k: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MiniMaxSparseAttentionGatheredGqa {
    batch: i32,
    query_len: i32,
    key_len: i32,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    selected_keys: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HeavilyCompressedAttention {
    batch: i32,
    seq_len: i32,
    heads: i32,
    head_dim: i32,
    compression_block: i32,
    blocks: i32,
    groups: i32,
    group_dim: i32,
    hidden_dim: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

impl SlidingWindowAttention {
    fn create(
        batch: usize,
        query_len: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        query_start: usize,
        key_start: usize,
        window: usize,
        query_batch_stride: usize,
        key_batch_stride: usize,
        value_batch_stride: usize,
        output_batch_stride: usize,
        query_sequence_stride: usize,
        key_sequence_stride: usize,
        value_sequence_stride: usize,
        output_sequence_stride: usize,
        query_head_stride: usize,
        key_head_stride: usize,
        value_head_stride: usize,
        output_head_stride: usize,
        scale: f32,
        output_scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || query_len == 0
            || key_len == 0
            || heads == 0
            || kv_heads == 0
            || head_dim == 0
            || window == 0
            || query_batch_stride == 0
            || key_batch_stride == 0
            || value_batch_stride == 0
            || output_batch_stride == 0
            || query_sequence_stride == 0
            || key_sequence_stride == 0
            || value_sequence_stride == 0
            || output_sequence_stride == 0
            || query_head_stride == 0
            || key_head_stride == 0
            || value_head_stride == 0
            || output_head_stride == 0
            || !scale.is_finite()
            || !output_scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }

        let output_values_per_batch =
            checked_element_count(checked_element_count(query_len, heads)?, head_dim)?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        let query_end = query_start
            .checked_add(query_len)
            .ok_or(Error::SizeOverflow)?;
        let key_end = key_start.checked_add(key_len).ok_or(Error::SizeOverflow)?;
        checked_i32_value(query_end)?;
        checked_i32_value(key_end)?;

        Ok(Self {
            batch: checked_i32_value(batch)?,
            query_len: checked_i32_value(query_len)?,
            key_len: checked_i32_value(key_len)?,
            heads: checked_i32_value(heads)?,
            kv_heads: checked_i32_value(kv_heads)?,
            query_start: checked_i32_value(query_start)?,
            key_start: checked_i32_value(key_start)?,
            window: checked_i32_value(window)?,
            output_len: checked_i32_value(output_len)?,
            output_values_per_batch: checked_i32_value(output_values_per_batch)?,
            query_batch_stride: checked_i32_value(query_batch_stride)?,
            key_batch_stride: checked_i32_value(key_batch_stride)?,
            value_batch_stride: checked_i32_value(value_batch_stride)?,
            output_batch_stride: checked_i32_value(output_batch_stride)?,
            query_sequence_stride: checked_i32_value(query_sequence_stride)?,
            key_sequence_stride: checked_i32_value(key_sequence_stride)?,
            value_sequence_stride: checked_i32_value(value_sequence_stride)?,
            output_sequence_stride: checked_i32_value(output_sequence_stride)?,
            query_head_stride: checked_i32_value(query_head_stride)?,
            key_head_stride: checked_i32_value(key_head_stride)?,
            value_head_stride: checked_i32_value(value_head_stride)?,
            output_head_stride: checked_i32_value(output_head_stride)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl FusedNeighborhoodAttention {
    fn create(
        batch: usize,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        kernel_size: usize,
        dilation: usize,
        scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || seq_len == 0
            || heads == 0
            || head_dim == 0
            || kernel_size == 0
            || dilation == 0
            || !scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }

        let output_len = checked_element_count(
            checked_element_count(checked_element_count(batch, heads)?, seq_len)?,
            head_dim,
        )?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            seq_len: checked_i32_value(seq_len)?,
            heads: checked_i32_value(heads)?,
            head_dim: checked_i32_value(head_dim)?,
            kernel_size: checked_i32_value(kernel_size)?,
            dilation: checked_i32_value(dilation)?,
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl MultiTokenAttention {
    fn create(
        batch: usize,
        channels_in: usize,
        channels_out: usize,
        seq_len: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        groups: usize,
        has_bias: bool,
    ) -> Result<Self> {
        if batch == 0
            || channels_in == 0
            || channels_out == 0
            || seq_len == 0
            || kernel_h == 0
            || kernel_w == 0
            || stride_h == 0
            || stride_w == 0
            || dilation_h == 0
            || dilation_w == 0
            || groups == 0
            || !channels_in.is_multiple_of(groups)
            || !channels_out.is_multiple_of(groups)
        {
            return Err(Error::InvalidLength);
        }
        let output_h = conv_output_len(seq_len, kernel_h, stride_h, padding_h, dilation_h)?;
        let output_w = conv_output_len(seq_len, kernel_w, stride_w, padding_w, dilation_w)?;
        if output_h == 0 || output_h != output_w {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(
            checked_element_count(checked_element_count(batch, channels_out)?, output_h)?,
            output_w,
        )?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            channels_in: checked_i32_value(channels_in)?,
            channels_out: checked_i32_value(channels_out)?,
            seq_len: checked_i32_value(seq_len)?,
            output_seq_len: checked_i32_value(output_h)?,
            kernel_h: checked_i32_value(kernel_h)?,
            kernel_w: checked_i32_value(kernel_w)?,
            stride_h: checked_i32_value(stride_h)?,
            stride_w: checked_i32_value(stride_w)?,
            padding_h: checked_i32_value(padding_h)?,
            padding_w: checked_i32_value(padding_w)?,
            dilation_h: checked_i32_value(dilation_h)?,
            dilation_w: checked_i32_value(dilation_w)?,
            groups: checked_i32_value(groups)?,
            has_bias: i32::from(has_bias),
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

fn conv_output_len(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize> {
    let effective_kernel = dilation
        .checked_mul(kernel - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or(Error::SizeOverflow)?;
    let padded = input
        .checked_add(padding)
        .and_then(|value| value.checked_add(padding))
        .ok_or(Error::SizeOverflow)?;
    if padded < effective_kernel {
        return Ok(0);
    }
    Ok((padded - effective_kernel) / stride + 1)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PagedSlidingWindowAttention {
    batch: i32,
    query_len: i32,
    key_len: i32,
    heads: i32,
    kv_heads: i32,
    query_start: i32,
    key_start: i32,
    window: i32,
    output_len: i32,
    output_values_per_batch: i32,
    query_batch_stride: i32,
    output_batch_stride: i32,
    query_sequence_stride: i32,
    key_sequence_stride: i32,
    value_sequence_stride: i32,
    output_sequence_stride: i32,
    query_head_stride: i32,
    key_head_stride: i32,
    value_head_stride: i32,
    output_head_stride: i32,
    block_size: i32,
    physical_blocks: i32,
    block_table_batch_stride: i32,
    key_cache_block_stride: i32,
    value_cache_block_stride: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PagedKvDecodeAttention {
    batch: i32,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    physical_blocks: i32,
    block_table_batch_stride: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PagedKvPrefillAttention {
    batch: i32,
    total_query_tokens: i32,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    physical_blocks: i32,
    block_table_batch_stride: i32,
    causal: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RaggedKvPrefillAttention {
    batch: i32,
    total_query_tokens: i32,
    total_kv_tokens: i32,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    causal: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FmhaPrefill {
    batch: i32,
    query_len: i32,
    key_len: i32,
    heads: i32,
    kv_heads: i32,
    output_len: i32,
    output_values_per_batch: i32,
    query_batch_stride: i32,
    key_batch_stride: i32,
    value_batch_stride: i32,
    output_batch_stride: i32,
    query_sequence_stride: i32,
    key_sequence_stride: i32,
    value_sequence_stride: i32,
    output_sequence_stride: i32,
    query_head_stride: i32,
    key_head_stride: i32,
    value_head_stride: i32,
    output_head_stride: i32,
    causal: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MlaPrefill {
    batch: i32,
    query_len: i32,
    key_len: i32,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    pe_dim: i32,
    output_len: i32,
    output_values_per_batch: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SparseMlaPrefill {
    batch: i32,
    query_len: i32,
    key_len: i32,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    pe_dim: i32,
    topk: i32,
    output_len: i32,
    output_values_per_batch: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MlaDecode {
    batch: i32,
    key_len: i32,
    heads: i32,
    head_dim: i32,
    pe_dim: i32,
    output_len: i32,
    output_values_per_batch: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PagedMlaDecodeAttention {
    batch: i32,
    heads: i32,
    head_dim: i32,
    pe_dim: i32,
    block_size: i32,
    physical_blocks: i32,
    block_table_batch_stride: i32,
    output_len: i32,
    output_values_per_batch: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MlaDecodeSplitK {
    batch: i32,
    key_len: i32,
    heads: i32,
    splits: i32,
    kv_len_per_split: i32,
    head_dim: i32,
    pe_dim: i32,
    output_len: i32,
    output_values_per_batch: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FmhaDecode {
    batch: i32,
    key_len: i32,
    heads: i32,
    kv_heads: i32,
    output_len: i32,
    output_values_per_batch: i32,
    query_batch_stride: i32,
    key_batch_stride: i32,
    value_batch_stride: i32,
    output_batch_stride: i32,
    query_head_stride: i32,
    key_sequence_stride: i32,
    value_sequence_stride: i32,
    output_head_stride: i32,
    key_head_stride: i32,
    value_head_stride: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FmhaDecodeSplitK {
    batch: i32,
    key_len: i32,
    heads: i32,
    kv_heads: i32,
    splits: i32,
    kv_len_per_split: i32,
    output_len: i32,
    output_values_per_batch: i32,
    query_batch_stride: i32,
    key_batch_stride: i32,
    value_batch_stride: i32,
    output_batch_stride: i32,
    query_head_stride: i32,
    key_sequence_stride: i32,
    value_sequence_stride: i32,
    output_head_stride: i32,
    output_split_stride: i32,
    key_head_stride: i32,
    value_head_stride: i32,
    lse_batch_stride: i32,
    lse_head_stride: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SplitKReduce {
    batch: i32,
    heads: i32,
    splits: i32,
    head_dim: i32,
    output_len: i32,
    output_values_per_batch: i32,
    attn_batch_stride: i32,
    attn_head_stride: i32,
    attn_split_stride: i32,
    lse_batch_stride: i32,
    lse_head_stride: i32,
    output_batch_stride: i32,
    output_head_stride: i32,
    grid: (u32, u32, u32),
}

impl PagedSlidingWindowAttention {
    fn create(
        batch: usize,
        query_len: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        query_start: usize,
        key_start: usize,
        window: usize,
        query_batch_stride: usize,
        output_batch_stride: usize,
        query_sequence_stride: usize,
        key_sequence_stride: usize,
        value_sequence_stride: usize,
        output_sequence_stride: usize,
        query_head_stride: usize,
        key_head_stride: usize,
        value_head_stride: usize,
        output_head_stride: usize,
        block_size: usize,
        physical_blocks: usize,
        block_table_batch_stride: usize,
        key_cache_block_stride: usize,
        value_cache_block_stride: usize,
        scale: f32,
        output_scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || query_len == 0
            || key_len == 0
            || heads == 0
            || kv_heads == 0
            || head_dim == 0
            || window == 0
            || query_batch_stride == 0
            || output_batch_stride == 0
            || query_sequence_stride == 0
            || key_sequence_stride == 0
            || value_sequence_stride == 0
            || output_sequence_stride == 0
            || query_head_stride == 0
            || key_head_stride == 0
            || value_head_stride == 0
            || output_head_stride == 0
            || block_size == 0
            || physical_blocks == 0
            || block_table_batch_stride == 0
            || key_cache_block_stride == 0
            || value_cache_block_stride == 0
            || !scale.is_finite()
            || !output_scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }

        let output_values_per_batch =
            checked_element_count(checked_element_count(query_len, heads)?, head_dim)?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        let query_end = query_start
            .checked_add(query_len)
            .ok_or(Error::SizeOverflow)?;
        let key_end = key_start.checked_add(key_len).ok_or(Error::SizeOverflow)?;
        checked_i32_value(query_end)?;
        checked_i32_value(key_end)?;

        Ok(Self {
            batch: checked_i32_value(batch)?,
            query_len: checked_i32_value(query_len)?,
            key_len: checked_i32_value(key_len)?,
            heads: checked_i32_value(heads)?,
            kv_heads: checked_i32_value(kv_heads)?,
            query_start: checked_i32_value(query_start)?,
            key_start: checked_i32_value(key_start)?,
            window: checked_i32_value(window)?,
            output_len: checked_i32_value(output_len)?,
            output_values_per_batch: checked_i32_value(output_values_per_batch)?,
            query_batch_stride: checked_i32_value(query_batch_stride)?,
            output_batch_stride: checked_i32_value(output_batch_stride)?,
            query_sequence_stride: checked_i32_value(query_sequence_stride)?,
            key_sequence_stride: checked_i32_value(key_sequence_stride)?,
            value_sequence_stride: checked_i32_value(value_sequence_stride)?,
            output_sequence_stride: checked_i32_value(output_sequence_stride)?,
            query_head_stride: checked_i32_value(query_head_stride)?,
            key_head_stride: checked_i32_value(key_head_stride)?,
            value_head_stride: checked_i32_value(value_head_stride)?,
            output_head_stride: checked_i32_value(output_head_stride)?,
            block_size: checked_i32_value(block_size)?,
            physical_blocks: checked_i32_value(physical_blocks)?,
            block_table_batch_stride: checked_i32_value(block_table_batch_stride)?,
            key_cache_block_stride: checked_i32_value(key_cache_block_stride)?,
            value_cache_block_stride: checked_i32_value(value_cache_block_stride)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl PagedKvDecodeAttention {
    fn create(
        batch: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        physical_blocks: usize,
        block_table_batch_stride: usize,
        scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || heads == 0
            || kv_heads == 0
            || head_dim == 0
            || block_size == 0
            || physical_blocks == 0
            || block_table_batch_stride == 0
            || !heads.is_multiple_of(kv_heads)
            || !scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(checked_element_count(batch, heads)?, head_dim)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            heads: checked_i32_value(heads)?,
            kv_heads: checked_i32_value(kv_heads)?,
            head_dim: checked_i32_value(head_dim)?,
            block_size: checked_i32_value(block_size)?,
            physical_blocks: checked_i32_value(physical_blocks)?,
            block_table_batch_stride: checked_i32_value(block_table_batch_stride)?,
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl PagedKvPrefillAttention {
    fn create(
        batch: usize,
        total_query_tokens: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        physical_blocks: usize,
        block_table_batch_stride: usize,
        causal: bool,
        scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || total_query_tokens == 0
            || heads == 0
            || kv_heads == 0
            || head_dim == 0
            || block_size == 0
            || physical_blocks == 0
            || block_table_batch_stride == 0
            || !heads.is_multiple_of(kv_heads)
            || !scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        let output_len =
            checked_element_count(checked_element_count(total_query_tokens, heads)?, head_dim)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            total_query_tokens: checked_i32_value(total_query_tokens)?,
            heads: checked_i32_value(heads)?,
            kv_heads: checked_i32_value(kv_heads)?,
            head_dim: checked_i32_value(head_dim)?,
            block_size: checked_i32_value(block_size)?,
            physical_blocks: checked_i32_value(physical_blocks)?,
            block_table_batch_stride: checked_i32_value(block_table_batch_stride)?,
            causal: i32::from(causal),
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl RaggedKvPrefillAttention {
    fn create(
        batch: usize,
        total_query_tokens: usize,
        total_kv_tokens: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        causal: bool,
        scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || total_query_tokens == 0
            || total_kv_tokens == 0
            || heads == 0
            || kv_heads == 0
            || head_dim == 0
            || !heads.is_multiple_of(kv_heads)
            || !scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        let output_len =
            checked_element_count(checked_element_count(total_query_tokens, heads)?, head_dim)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            total_query_tokens: checked_i32_value(total_query_tokens)?,
            total_kv_tokens: checked_i32_value(total_kv_tokens)?,
            heads: checked_i32_value(heads)?,
            kv_heads: checked_i32_value(kv_heads)?,
            head_dim: checked_i32_value(head_dim)?,
            causal: i32::from(causal),
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl FmhaPrefill {
    fn create(
        batch: usize,
        query_len: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        query_batch_stride: usize,
        key_batch_stride: usize,
        value_batch_stride: usize,
        output_batch_stride: usize,
        query_sequence_stride: usize,
        key_sequence_stride: usize,
        value_sequence_stride: usize,
        output_sequence_stride: usize,
        query_head_stride: usize,
        key_head_stride: usize,
        value_head_stride: usize,
        output_head_stride: usize,
        scale: f32,
        causal: bool,
    ) -> Result<Self> {
        if batch == 0
            || query_len == 0
            || key_len == 0
            || heads == 0
            || kv_heads == 0
            || head_dim == 0
            || query_batch_stride == 0
            || key_batch_stride == 0
            || value_batch_stride == 0
            || output_batch_stride == 0
            || query_sequence_stride == 0
            || key_sequence_stride == 0
            || value_sequence_stride == 0
            || output_sequence_stride == 0
            || query_head_stride == 0
            || key_head_stride == 0
            || value_head_stride == 0
            || output_head_stride == 0
            || !scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        if !heads.is_multiple_of(kv_heads) {
            return Err(Error::InvalidLength);
        }

        let output_values_per_batch =
            checked_element_count(checked_element_count(query_len, heads)?, head_dim)?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            query_len: checked_i32_value(query_len)?,
            key_len: checked_i32_value(key_len)?,
            heads: checked_i32_value(heads)?,
            kv_heads: checked_i32_value(kv_heads)?,
            output_len: checked_i32_value(output_len)?,
            output_values_per_batch: checked_i32_value(output_values_per_batch)?,
            query_batch_stride: checked_i32_value(query_batch_stride)?,
            key_batch_stride: checked_i32_value(key_batch_stride)?,
            value_batch_stride: checked_i32_value(value_batch_stride)?,
            output_batch_stride: checked_i32_value(output_batch_stride)?,
            query_sequence_stride: checked_i32_value(query_sequence_stride)?,
            key_sequence_stride: checked_i32_value(key_sequence_stride)?,
            value_sequence_stride: checked_i32_value(value_sequence_stride)?,
            output_sequence_stride: checked_i32_value(output_sequence_stride)?,
            query_head_stride: checked_i32_value(query_head_stride)?,
            key_head_stride: checked_i32_value(key_head_stride)?,
            value_head_stride: checked_i32_value(value_head_stride)?,
            output_head_stride: checked_i32_value(output_head_stride)?,
            causal: i32::from(causal),
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl MlaDecode {
    fn create(
        batch: usize,
        key_len: usize,
        heads: usize,
        head_dim: usize,
        pe_dim: usize,
        scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || key_len == 0
            || heads == 0
            || head_dim == 0
            || pe_dim == 0
            || !scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        let output_values_per_batch = checked_element_count(heads, head_dim)?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            key_len: checked_i32_value(key_len)?,
            heads: checked_i32_value(heads)?,
            head_dim: checked_i32_value(head_dim)?,
            pe_dim: checked_i32_value(pe_dim)?,
            output_len: checked_i32_value(output_len)?,
            output_values_per_batch: checked_i32_value(output_values_per_batch)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl PagedMlaDecodeAttention {
    fn create(
        batch: usize,
        heads: usize,
        head_dim: usize,
        pe_dim: usize,
        block_size: usize,
        physical_blocks: usize,
        block_table_batch_stride: usize,
        scale: f32,
        output_scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || heads == 0
            || head_dim == 0
            || pe_dim == 0
            || block_size == 0
            || physical_blocks == 0
            || block_table_batch_stride == 0
            || !scale.is_finite()
            || !output_scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        let output_values_per_batch = checked_element_count(heads, head_dim)?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            heads: checked_i32_value(heads)?,
            head_dim: checked_i32_value(head_dim)?,
            pe_dim: checked_i32_value(pe_dim)?,
            block_size: checked_i32_value(block_size)?,
            physical_blocks: checked_i32_value(physical_blocks)?,
            block_table_batch_stride: checked_i32_value(block_table_batch_stride)?,
            output_len: checked_i32_value(output_len)?,
            output_values_per_batch: checked_i32_value(output_values_per_batch)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl SparseMlaPrefill {
    fn create(
        batch: usize,
        query_len: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pe_dim: usize,
        topk: usize,
        scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || query_len == 0
            || key_len == 0
            || heads == 0
            || kv_heads == 0
            || head_dim == 0
            || pe_dim == 0
            || topk == 0
            || !scale.is_finite()
            || !heads.is_multiple_of(kv_heads)
        {
            return Err(Error::InvalidLength);
        }
        let output_values_per_batch =
            checked_element_count(checked_element_count(query_len, heads)?, head_dim)?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            query_len: checked_i32_value(query_len)?,
            key_len: checked_i32_value(key_len)?,
            heads: checked_i32_value(heads)?,
            kv_heads: checked_i32_value(kv_heads)?,
            head_dim: checked_i32_value(head_dim)?,
            pe_dim: checked_i32_value(pe_dim)?,
            topk: checked_i32_value(topk)?,
            output_len: checked_i32_value(output_len)?,
            output_values_per_batch: checked_i32_value(output_values_per_batch)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl MlaDecodeSplitK {
    fn create(
        batch: usize,
        key_len: usize,
        heads: usize,
        splits: usize,
        kv_len_per_split: usize,
        head_dim: usize,
        pe_dim: usize,
        scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || key_len == 0
            || heads == 0
            || splits == 0
            || kv_len_per_split == 0
            || head_dim == 0
            || pe_dim == 0
            || !scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        let output_values_per_batch =
            checked_element_count(checked_element_count(heads, splits)?, head_dim)?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            key_len: checked_i32_value(key_len)?,
            heads: checked_i32_value(heads)?,
            splits: checked_i32_value(splits)?,
            kv_len_per_split: checked_i32_value(kv_len_per_split)?,
            head_dim: checked_i32_value(head_dim)?,
            pe_dim: checked_i32_value(pe_dim)?,
            output_len: checked_i32_value(output_len)?,
            output_values_per_batch: checked_i32_value(output_values_per_batch)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl MlaPrefill {
    fn create(
        batch: usize,
        query_len: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pe_dim: usize,
        scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || query_len == 0
            || key_len == 0
            || heads == 0
            || kv_heads == 0
            || head_dim == 0
            || pe_dim == 0
            || !scale.is_finite()
            || !heads.is_multiple_of(kv_heads)
        {
            return Err(Error::InvalidLength);
        }
        let output_values_per_batch =
            checked_element_count(checked_element_count(query_len, heads)?, head_dim)?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            query_len: checked_i32_value(query_len)?,
            key_len: checked_i32_value(key_len)?,
            heads: checked_i32_value(heads)?,
            kv_heads: checked_i32_value(kv_heads)?,
            head_dim: checked_i32_value(head_dim)?,
            pe_dim: checked_i32_value(pe_dim)?,
            output_len: checked_i32_value(output_len)?,
            output_values_per_batch: checked_i32_value(output_values_per_batch)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl FmhaDecode {
    fn create(
        batch: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        query_batch_stride: usize,
        key_batch_stride: usize,
        value_batch_stride: usize,
        output_batch_stride: usize,
        query_head_stride: usize,
        key_sequence_stride: usize,
        value_sequence_stride: usize,
        output_head_stride: usize,
        key_head_stride: usize,
        value_head_stride: usize,
        scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || key_len == 0
            || heads == 0
            || kv_heads == 0
            || head_dim == 0
            || query_batch_stride == 0
            || key_batch_stride == 0
            || value_batch_stride == 0
            || output_batch_stride == 0
            || query_head_stride == 0
            || key_sequence_stride == 0
            || value_sequence_stride == 0
            || output_head_stride == 0
            || key_head_stride == 0
            || value_head_stride == 0
            || !scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        if !heads.is_multiple_of(kv_heads) {
            return Err(Error::InvalidLength);
        }

        let output_values_per_batch = checked_element_count(heads, head_dim)?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            key_len: checked_i32_value(key_len)?,
            heads: checked_i32_value(heads)?,
            kv_heads: checked_i32_value(kv_heads)?,
            output_len: checked_i32_value(output_len)?,
            output_values_per_batch: checked_i32_value(output_values_per_batch)?,
            query_batch_stride: checked_i32_value(query_batch_stride)?,
            key_batch_stride: checked_i32_value(key_batch_stride)?,
            value_batch_stride: checked_i32_value(value_batch_stride)?,
            output_batch_stride: checked_i32_value(output_batch_stride)?,
            query_head_stride: checked_i32_value(query_head_stride)?,
            key_sequence_stride: checked_i32_value(key_sequence_stride)?,
            value_sequence_stride: checked_i32_value(value_sequence_stride)?,
            output_head_stride: checked_i32_value(output_head_stride)?,
            key_head_stride: checked_i32_value(key_head_stride)?,
            value_head_stride: checked_i32_value(value_head_stride)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl FmhaDecodeSplitK {
    fn create(
        batch: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        splits: usize,
        kv_len_per_split: usize,
        head_dim: usize,
        query_batch_stride: usize,
        key_batch_stride: usize,
        value_batch_stride: usize,
        output_batch_stride: usize,
        query_head_stride: usize,
        key_sequence_stride: usize,
        value_sequence_stride: usize,
        output_head_stride: usize,
        output_split_stride: usize,
        key_head_stride: usize,
        value_head_stride: usize,
        lse_batch_stride: usize,
        lse_head_stride: usize,
        scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || key_len == 0
            || heads == 0
            || kv_heads == 0
            || splits == 0
            || kv_len_per_split == 0
            || head_dim == 0
            || query_batch_stride == 0
            || key_batch_stride == 0
            || value_batch_stride == 0
            || output_batch_stride == 0
            || query_head_stride == 0
            || key_sequence_stride == 0
            || value_sequence_stride == 0
            || output_head_stride == 0
            || output_split_stride == 0
            || key_head_stride == 0
            || value_head_stride == 0
            || lse_batch_stride == 0
            || lse_head_stride == 0
            || !scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        if !heads.is_multiple_of(kv_heads) {
            return Err(Error::InvalidLength);
        }

        let output_values_per_batch =
            checked_element_count(checked_element_count(heads, splits)?, head_dim)?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            key_len: checked_i32_value(key_len)?,
            heads: checked_i32_value(heads)?,
            kv_heads: checked_i32_value(kv_heads)?,
            splits: checked_i32_value(splits)?,
            kv_len_per_split: checked_i32_value(kv_len_per_split)?,
            output_len: checked_i32_value(output_len)?,
            output_values_per_batch: checked_i32_value(output_values_per_batch)?,
            query_batch_stride: checked_i32_value(query_batch_stride)?,
            key_batch_stride: checked_i32_value(key_batch_stride)?,
            value_batch_stride: checked_i32_value(value_batch_stride)?,
            output_batch_stride: checked_i32_value(output_batch_stride)?,
            query_head_stride: checked_i32_value(query_head_stride)?,
            key_sequence_stride: checked_i32_value(key_sequence_stride)?,
            value_sequence_stride: checked_i32_value(value_sequence_stride)?,
            output_head_stride: checked_i32_value(output_head_stride)?,
            output_split_stride: checked_i32_value(output_split_stride)?,
            key_head_stride: checked_i32_value(key_head_stride)?,
            value_head_stride: checked_i32_value(value_head_stride)?,
            lse_batch_stride: checked_i32_value(lse_batch_stride)?,
            lse_head_stride: checked_i32_value(lse_head_stride)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl SplitKReduce {
    fn create(
        batch: usize,
        heads: usize,
        splits: usize,
        head_dim: usize,
        attn_batch_stride: usize,
        attn_head_stride: usize,
        attn_split_stride: usize,
        lse_batch_stride: usize,
        lse_head_stride: usize,
        output_batch_stride: usize,
        output_head_stride: usize,
    ) -> Result<Self> {
        if batch == 0
            || heads == 0
            || splits == 0
            || head_dim == 0
            || attn_batch_stride == 0
            || attn_head_stride == 0
            || attn_split_stride == 0
            || lse_batch_stride == 0
            || lse_head_stride == 0
            || output_batch_stride == 0
            || output_head_stride == 0
        {
            return Err(Error::InvalidLength);
        }

        let output_values_per_batch = checked_element_count(heads, head_dim)?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            heads: checked_i32_value(heads)?,
            splits: checked_i32_value(splits)?,
            head_dim: checked_i32_value(head_dim)?,
            output_len: checked_i32_value(output_len)?,
            output_values_per_batch: checked_i32_value(output_values_per_batch)?,
            attn_batch_stride: checked_i32_value(attn_batch_stride)?,
            attn_head_stride: checked_i32_value(attn_head_stride)?,
            attn_split_stride: checked_i32_value(attn_split_stride)?,
            lse_batch_stride: checked_i32_value(lse_batch_stride)?,
            lse_head_stride: checked_i32_value(lse_head_stride)?,
            output_batch_stride: checked_i32_value(output_batch_stride)?,
            output_head_stride: checked_i32_value(output_head_stride)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl HeavilyCompressedAttentionCompress {
    fn create(
        batch: usize,
        seq_len: usize,
        hidden_dim: usize,
        head_dim: usize,
        compression_block: usize,
    ) -> Result<Self> {
        if batch == 0 || seq_len == 0 || hidden_dim == 0 || head_dim == 0 || compression_block == 0
        {
            return Err(Error::InvalidLength);
        }
        let blocks = seq_len / compression_block;
        if blocks == 0 {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(checked_element_count(batch, blocks)?, head_dim)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            seq_len: checked_i32_value(seq_len)?,
            hidden_dim: checked_i32_value(hidden_dim)?,
            head_dim: checked_i32_value(head_dim)?,
            compression_block: checked_i32_value(compression_block)?,
            blocks: checked_i32_value(blocks)?,
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl CompressedSparseAttentionLightningIndexer {
    fn create(
        batch: usize,
        query_len: usize,
        blocks: usize,
        index_heads: usize,
        index_dim: usize,
    ) -> Result<Self> {
        if batch == 0 || query_len == 0 || blocks == 0 || index_heads == 0 || index_dim == 0 {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(checked_element_count(batch, query_len)?, blocks)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            query_len: checked_i32_value(query_len)?,
            blocks: checked_i32_value(blocks)?,
            index_heads: checked_i32_value(index_heads)?,
            index_dim: checked_i32_value(index_dim)?,
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl CompressedSparseAttentionTopkSelector {
    fn create(
        batch: usize,
        query_len: usize,
        blocks: usize,
        head_dim: usize,
        top_k: usize,
        compression_block: usize,
    ) -> Result<Self> {
        if batch == 0
            || query_len == 0
            || blocks == 0
            || head_dim == 0
            || top_k == 0
            || compression_block == 0
        {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(
            checked_element_count(checked_element_count(batch, query_len)?, top_k)?,
            head_dim,
        )?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            query_len: checked_i32_value(query_len)?,
            blocks: checked_i32_value(blocks)?,
            head_dim: checked_i32_value(head_dim)?,
            top_k: checked_i32_value(top_k)?,
            compression_block: checked_i32_value(compression_block)?,
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl CompressedSparseAttentionSharedMqa {
    fn create(
        batch: usize,
        query_len: usize,
        heads: usize,
        head_dim: usize,
        kv_len: usize,
        scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || query_len == 0
            || heads == 0
            || head_dim == 0
            || kv_len == 0
            || !scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(
            checked_element_count(checked_element_count(batch, query_len)?, heads)?,
            head_dim,
        )?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            query_len: checked_i32_value(query_len)?,
            heads: checked_i32_value(heads)?,
            head_dim: checked_i32_value(head_dim)?,
            kv_len: checked_i32_value(kv_len)?,
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl MiniMaxSparseAttentionBlockMax {
    fn create(batch: usize, rows: usize, key_len: usize, block_size: usize) -> Result<Self> {
        if batch == 0 || rows == 0 || key_len == 0 || block_size == 0 {
            return Err(Error::InvalidLength);
        }
        let blocks = key_len.div_ceil(block_size);
        let output_len = checked_element_count(checked_element_count(batch, rows)?, blocks)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            rows: checked_i32_value(rows)?,
            key_len: checked_i32_value(key_len)?,
            block_size: checked_i32_value(block_size)?,
            blocks: checked_i32_value(blocks)?,
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl MiniMaxSparseAttentionSelectedTokenPositions {
    fn create(
        batch: usize,
        rows: usize,
        selected_blocks: usize,
        block_size: usize,
        seq_len: usize,
    ) -> Result<Self> {
        if batch == 0 || rows == 0 || selected_blocks == 0 || block_size == 0 || seq_len == 0 {
            return Err(Error::InvalidLength);
        }
        let expanded_keys = checked_element_count(selected_blocks, block_size)?;
        let output_len = checked_element_count(checked_element_count(batch, rows)?, expanded_keys)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            rows: checked_i32_value(rows)?,
            selected_blocks: checked_i32_value(selected_blocks)?,
            block_size: checked_i32_value(block_size)?,
            seq_len: checked_i32_value(seq_len)?,
            expanded_keys: checked_i32_value(expanded_keys)?,
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl MiniMaxSparseAttentionSelectTopkBlocks {
    fn create(batch: usize, rows: usize, blocks: usize, top_k: usize) -> Result<Self> {
        if batch == 0 || rows == 0 || blocks == 0 || top_k == 0 {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(checked_element_count(batch, rows)?, top_k)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            rows: checked_i32_value(rows)?,
            blocks: checked_i32_value(blocks)?,
            top_k: checked_i32_value(top_k)?,
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl MiniMaxSparseAttentionGatheredGqa {
    fn create(
        batch: usize,
        query_len: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        selected_keys: usize,
        scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || query_len == 0
            || key_len == 0
            || heads == 0
            || kv_heads == 0
            || head_dim == 0
            || selected_keys == 0
            || !heads.is_multiple_of(kv_heads)
            || !scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(
            checked_element_count(checked_element_count(batch, query_len)?, heads)?,
            head_dim,
        )?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            query_len: checked_i32_value(query_len)?,
            key_len: checked_i32_value(key_len)?,
            heads: checked_i32_value(heads)?,
            kv_heads: checked_i32_value(kv_heads)?,
            head_dim: checked_i32_value(head_dim)?,
            selected_keys: checked_i32_value(selected_keys)?,
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl HeavilyCompressedAttention {
    fn create(
        batch: usize,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        compression_block: usize,
        groups: usize,
        group_dim: usize,
        hidden_dim: usize,
        scale: f32,
    ) -> Result<Self> {
        if batch == 0
            || seq_len == 0
            || heads == 0
            || head_dim == 0
            || compression_block == 0
            || groups == 0
            || group_dim == 0
            || hidden_dim == 0
            || !heads.is_multiple_of(groups)
            || !scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        let blocks = seq_len / compression_block;
        if blocks == 0 {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(checked_element_count(batch, seq_len)?, hidden_dim)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            seq_len: checked_i32_value(seq_len)?,
            heads: checked_i32_value(heads)?,
            head_dim: checked_i32_value(head_dim)?,
            compression_block: checked_i32_value(compression_block)?,
            blocks: checked_i32_value(blocks)?,
            groups: checked_i32_value(groups)?,
            group_dim: checked_i32_value(group_dim)?,
            hidden_dim: checked_i32_value(hidden_dim)?,
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}
