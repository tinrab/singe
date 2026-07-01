//! Prefill/decode attention kernels and related attention variants.

#[cfg(feature = "dtype-bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceSlice, DeviceSliceMut, DeviceView},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{borrowed_stream, input_pointer, output_pointer},
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value, ensure_len, ensure_len_at_least},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlidingWindowAttentionConfig {
    pub batch: usize,
    pub query_len: usize,
    pub key_len: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub query_start: usize,
    pub key_start: usize,
    pub window: usize,
    pub query_batch_stride: usize,
    pub key_batch_stride: usize,
    pub value_batch_stride: usize,
    pub output_batch_stride: usize,
    pub query_sequence_stride: usize,
    pub key_sequence_stride: usize,
    pub value_sequence_stride: usize,
    pub output_sequence_stride: usize,
    pub query_head_stride: usize,
    pub key_head_stride: usize,
    pub value_head_stride: usize,
    pub output_head_stride: usize,
    pub scale: f32,
    pub output_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedQkvAttentionConfig {
    pub attention: SlidingWindowAttentionConfig,
    pub query_offset: usize,
    pub key_offset: usize,
    pub value_offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PagedKvAttentionConfig {
    pub attention: SlidingWindowAttentionConfig,
    pub block_size: usize,
    pub physical_blocks: usize,
    pub block_table_batch_stride: usize,
    pub key_cache_block_stride: usize,
    pub value_cache_block_stride: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Decode attention over paged KV caches.
///
/// Query is laid out as one token per batch. K/V cache storage is addressed
/// through a per-batch block table, with `actual_seq_lens` selecting the active
/// prefix length for each batch item.
pub struct PagedKvDecodeAttentionConfig {
    pub batch: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub physical_blocks: usize,
    pub block_table_batch_stride: usize,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Prefill attention over paged KV caches.
///
/// Query tokens are packed across the batch and indexed by sequence offsets.
/// K/V cache tokens are addressed through a block table.
pub struct PagedKvPrefillAttentionConfig {
    pub batch: usize,
    pub total_query_tokens: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub physical_blocks: usize,
    pub block_table_batch_stride: usize,
    pub causal: bool,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Prefill attention over ragged packed K/V tensors.
///
/// Query, key, and value tokens are packed by sequence; per-batch lengths and
/// offsets describe each ragged segment.
pub struct RaggedKvPrefillAttentionConfig {
    pub batch: usize,
    pub total_query_tokens: usize,
    pub total_kv_tokens: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwaAttentionConfig {
    pub batch: usize,
    pub query_len: usize,
    pub key_len: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub window_size: usize,
    pub scale: f32,
    pub causal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusedNeighborhoodAttentionConfig {
    pub batch: usize,
    pub seq_len: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub kernel_size: usize,
    pub dilation: usize,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeavilyCompressedAttentionConfig {
    pub batch: usize,
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub compression_block: usize,
    pub groups: usize,
    pub group_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompressedSparseAttentionCompressConfig {
    pub batch: usize,
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub head_dim: usize,
    pub compression_block: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompressedSparseAttentionLightningIndexerConfig {
    pub batch: usize,
    pub query_len: usize,
    pub blocks: usize,
    pub index_heads: usize,
    pub index_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompressedSparseAttentionTopkSelectorConfig {
    pub batch: usize,
    pub query_len: usize,
    pub blocks: usize,
    pub head_dim: usize,
    pub top_k: usize,
    pub compression_block: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompressedSparseAttentionSharedMqaConfig {
    pub batch: usize,
    pub query_len: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub kv_len: usize,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MiniMaxSparseAttentionBlockMaxConfig {
    pub batch: usize,
    pub rows: usize,
    pub key_len: usize,
    pub block_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MiniMaxSparseAttentionSelectedTokenPositionsConfig {
    pub batch: usize,
    pub rows: usize,
    pub selected_blocks: usize,
    pub block_size: usize,
    pub seq_len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MiniMaxSparseAttentionSelectTopkBlocksConfig {
    pub batch: usize,
    pub rows: usize,
    pub blocks: usize,
    pub top_k: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MiniMaxSparseAttentionGatheredGqaConfig {
    pub batch: usize,
    pub query_len: usize,
    pub key_len: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub selected_keys: usize,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultiTokenAttentionConfig {
    pub batch: usize,
    pub channels_in: usize,
    pub channels_out: usize,
    pub seq_len: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_h: usize,
    pub padding_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub groups: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FmhaPrefillConfig {
    pub batch: usize,
    pub query_len: usize,
    pub key_len: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub query_batch_stride: usize,
    pub key_batch_stride: usize,
    pub value_batch_stride: usize,
    pub output_batch_stride: usize,
    pub query_sequence_stride: usize,
    pub key_sequence_stride: usize,
    pub value_sequence_stride: usize,
    pub output_sequence_stride: usize,
    pub query_head_stride: usize,
    pub key_head_stride: usize,
    pub value_head_stride: usize,
    pub output_head_stride: usize,
    pub scale: f32,
    pub causal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlaPrefillConfig {
    pub batch: usize,
    pub query_len: usize,
    pub key_len: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub pe_dim: usize,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Windowed attention with optional tanh score soft-capping.
///
/// When `soft_cap` is set, attention scores are transformed as
/// `tanh(score / soft_cap) * soft_cap` before softmax.
pub struct SoftcappedWindowAttentionConfig {
    pub attention: FmhaPrefillConfig,
    pub window_size: usize,
    pub soft_cap: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionSinkPrefillConfig {
    pub attention: FmhaPrefillConfig,
    pub start_q: usize,
    pub window: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Decode variant of windowed attention with optional tanh score soft-capping.
pub struct SoftcappedWindowDecodeConfig {
    pub decode: FmhaDecodeConfig,
    pub window_size: usize,
    pub soft_cap: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseMlaPrefillConfig {
    pub batch: usize,
    pub query_len: usize,
    pub key_len: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub pe_dim: usize,
    pub topk: usize,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlaDecodeConfig {
    pub batch: usize,
    pub key_len: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub pe_dim: usize,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Decode attention over paged MLA K/V caches.
///
/// Content and positional key channels are stored separately and combined with
/// query and query positional channels during scoring.
pub struct PagedMlaDecodeAttentionConfig {
    pub batch: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub pe_dim: usize,
    pub block_size: usize,
    pub physical_blocks: usize,
    pub block_table_batch_stride: usize,
    pub scale: f32,
    pub output_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlaDecodeSplitKConfig {
    pub batch: usize,
    pub key_len: usize,
    pub heads: usize,
    pub splits: usize,
    pub kv_len_per_split: usize,
    pub head_dim: usize,
    pub pe_dim: usize,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FmhaDecodeConfig {
    pub batch: usize,
    pub key_len: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub query_batch_stride: usize,
    pub key_batch_stride: usize,
    pub value_batch_stride: usize,
    pub output_batch_stride: usize,
    pub query_head_stride: usize,
    pub key_sequence_stride: usize,
    pub value_sequence_stride: usize,
    pub output_head_stride: usize,
    pub key_head_stride: usize,
    pub value_head_stride: usize,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitKReduceConfig {
    pub batch: usize,
    pub heads: usize,
    pub splits: usize,
    pub head_dim: usize,
    pub attn_batch_stride: usize,
    pub attn_head_stride: usize,
    pub attn_split_stride: usize,
    pub lse_batch_stride: usize,
    pub lse_head_stride: usize,
    pub output_batch_stride: usize,
    pub output_head_stride: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FmhaDecodeSplitKConfig {
    pub batch: usize,
    pub key_len: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub splits: usize,
    pub kv_len_per_split: usize,
    pub head_dim: usize,
    pub query_batch_stride: usize,
    pub key_batch_stride: usize,
    pub value_batch_stride: usize,
    pub output_batch_stride: usize,
    pub query_head_stride: usize,
    pub key_sequence_stride: usize,
    pub value_sequence_stride: usize,
    pub output_head_stride: usize,
    pub output_split_stride: usize,
    pub key_head_stride: usize,
    pub value_head_stride: usize,
    pub lse_batch_stride: usize,
    pub lse_head_stride: usize,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionSinkDecodeSplitKConfig {
    pub decode: FmhaDecodeSplitKConfig,
    pub start_q: usize,
    pub window: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Split-K decode variant of windowed attention with optional score soft-cap.
pub struct SoftcappedWindowDecodeSplitKConfig {
    pub decode: FmhaDecodeSplitKConfig,
    pub window_size: usize,
    pub soft_cap: Option<f32>,
}

impl SlidingWindowAttentionConfig {
    pub fn contiguous(
        batch: usize,
        query_len: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        query_start: usize,
        key_start: usize,
        window: usize,
        scale: f32,
    ) -> Result<Self> {
        let q_features = checked_element_count(heads, head_dim)?;
        let kv_features = checked_element_count(kv_heads, head_dim)?;
        Ok(Self {
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            head_dim,
            query_start,
            key_start,
            window,
            query_batch_stride: checked_element_count(query_len, q_features)?,
            key_batch_stride: checked_element_count(key_len, kv_features)?,
            value_batch_stride: checked_element_count(key_len, kv_features)?,
            output_batch_stride: checked_element_count(query_len, q_features)?,
            query_sequence_stride: q_features,
            key_sequence_stride: kv_features,
            value_sequence_stride: kv_features,
            output_sequence_stride: q_features,
            query_head_stride: head_dim,
            key_head_stride: head_dim,
            value_head_stride: head_dim,
            output_head_stride: head_dim,
            scale,
            output_scale: 1.0,
        })
    }

    pub fn with_output_scale(mut self, output_scale: f32) -> Result<Self> {
        self.output_scale = output_scale;
        validate_sliding_window_attention_config(self)?;
        Ok(self)
    }

    pub fn with_projected_strides(
        mut self,
        query_sequence_stride: usize,
        key_sequence_stride: usize,
        value_sequence_stride: usize,
        output_sequence_stride: usize,
        query_head_stride: usize,
        key_head_stride: usize,
        value_head_stride: usize,
        output_head_stride: usize,
    ) -> Result<Self> {
        self.query_sequence_stride = query_sequence_stride;
        self.key_sequence_stride = key_sequence_stride;
        self.value_sequence_stride = value_sequence_stride;
        self.output_sequence_stride = output_sequence_stride;
        self.query_head_stride = query_head_stride;
        self.key_head_stride = key_head_stride;
        self.value_head_stride = value_head_stride;
        self.output_head_stride = output_head_stride;
        validate_sliding_window_attention_config(self)?;
        Ok(self)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
    ) -> Result<()> {
        validate_sliding_window_attention_config(self)?;
        let q_features = checked_element_count(self.heads, self.head_dim)?;
        let kv_features = checked_element_count(self.kv_heads, self.head_dim)?;
        let query_item_reach = projected_layout_reach(
            self.query_len,
            self.heads,
            self.head_dim,
            self.query_sequence_stride,
            self.query_head_stride,
        )?;
        let key_item_reach = projected_layout_reach(
            self.key_len,
            self.kv_heads,
            self.head_dim,
            self.key_sequence_stride,
            self.key_head_stride,
        )?;
        let value_item_reach = projected_layout_reach(
            self.key_len,
            self.kv_heads,
            self.head_dim,
            self.value_sequence_stride,
            self.value_head_stride,
        )?;
        let output_item_reach = projected_layout_reach(
            self.query_len,
            self.heads,
            self.head_dim,
            self.output_sequence_stride,
            self.output_head_stride,
        )?;
        let _query_logical_len = checked_element_count(self.query_len, q_features)?;
        let _key_logical_len = checked_element_count(self.key_len, kv_features)?;
        ensure_len_at_least(
            query_len,
            batched_reach(self.batch, query_item_reach, self.query_batch_stride)?,
        )?;
        ensure_len_at_least(
            key_len,
            batched_reach(self.batch, key_item_reach, self.key_batch_stride)?,
        )?;
        ensure_len_at_least(
            value_len,
            batched_reach(self.batch, value_item_reach, self.value_batch_stride)?,
        )?;
        ensure_len_at_least(
            out_len,
            batched_reach(self.batch, output_item_reach, self.output_batch_stride)?,
        )?;
        Ok(())
    }
}

impl SwaAttentionConfig {
    pub fn contiguous(
        batch: usize,
        query_len: usize,
        key_len: usize,
        heads: usize,
        head_dim: usize,
        window_size: usize,
        scale: f32,
        causal: bool,
    ) -> Result<Self> {
        let config = Self {
            batch,
            query_len,
            key_len,
            heads,
            head_dim,
            window_size,
            scale,
            causal,
        };
        validate_swa_attention_config(config)?;
        Ok(config)
    }

    fn effective_window(self) -> usize {
        if self.window_size == 0 {
            self.key_len
        } else {
            self.window_size
        }
    }

    fn validate_lengths(
        self,
        out_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
    ) -> Result<()> {
        validate_swa_attention_config(self)?;
        let features = checked_element_count(self.heads, self.head_dim)?;
        let query_values =
            checked_element_count(checked_element_count(self.query_len, features)?, self.batch)?;
        let key_values =
            checked_element_count(checked_element_count(self.key_len, features)?, self.batch)?;
        ensure_len_at_least(out_len, query_values)?;
        ensure_len_at_least(query_len, query_values)?;
        ensure_len_at_least(key_len, key_values)?;
        ensure_len_at_least(value_len, key_values)
    }
}

impl PagedKvDecodeAttentionConfig {
    pub fn contiguous(
        batch: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        physical_blocks: usize,
        block_table_batch_stride: usize,
        scale: f32,
    ) -> Result<Self> {
        let config = Self {
            batch,
            heads,
            kv_heads,
            head_dim,
            block_size,
            physical_blocks,
            block_table_batch_stride,
            scale,
        };
        validate_paged_kv_decode_attention_config(config)?;
        Ok(config)
    }

    fn output_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.heads)?,
            self.head_dim,
        )
    }

    fn validate_lengths(
        self,
        out_len: usize,
        query_len: usize,
        key_cache_len: usize,
        value_cache_len: usize,
        actual_seq_lens_len: usize,
        block_table_len: usize,
    ) -> Result<()> {
        validate_paged_kv_decode_attention_config(self)?;
        let output_len = self.output_len()?;
        let cache_block_len = checked_element_count(
            checked_element_count(self.block_size, self.kv_heads)?,
            self.head_dim,
        )?;
        let cache_len = checked_element_count(self.physical_blocks, cache_block_len)?;
        ensure_len_at_least(out_len, output_len)?;
        ensure_len_at_least(query_len, output_len)?;
        ensure_len_at_least(key_cache_len, cache_len)?;
        ensure_len_at_least(value_cache_len, cache_len)?;
        ensure_len_at_least(actual_seq_lens_len, self.batch)?;
        ensure_len_at_least(
            block_table_len,
            checked_element_count(self.batch, self.block_table_batch_stride)?,
        )
    }
}

impl PagedKvPrefillAttentionConfig {
    pub fn contiguous(
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
        let config = Self {
            batch,
            total_query_tokens,
            heads,
            kv_heads,
            head_dim,
            block_size,
            physical_blocks,
            block_table_batch_stride,
            causal,
            scale,
        };
        validate_paged_kv_prefill_attention_config(config)?;
        Ok(config)
    }

    fn output_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.total_query_tokens, self.heads)?,
            self.head_dim,
        )
    }

    fn lse_len(self) -> Result<usize> {
        checked_element_count(self.total_query_tokens, self.heads)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        lse_len: usize,
        query_len: usize,
        key_cache_len: usize,
        value_cache_len: usize,
        actual_seq_lens_q_len: usize,
        actual_seq_lens_kv_len: usize,
        actual_seq_offsets_len: usize,
        block_table_len: usize,
    ) -> Result<()> {
        validate_paged_kv_prefill_attention_config(self)?;
        let output_len = self.output_len()?;
        let cache_block_len = checked_element_count(
            checked_element_count(self.block_size, self.kv_heads)?,
            self.head_dim,
        )?;
        let cache_len = checked_element_count(self.physical_blocks, cache_block_len)?;
        ensure_len_at_least(out_len, output_len)?;
        ensure_len_at_least(lse_len, self.lse_len()?)?;
        ensure_len_at_least(query_len, output_len)?;
        ensure_len_at_least(key_cache_len, cache_len)?;
        ensure_len_at_least(value_cache_len, cache_len)?;
        ensure_len_at_least(actual_seq_lens_q_len, self.batch)?;
        ensure_len_at_least(actual_seq_lens_kv_len, self.batch)?;
        ensure_len_at_least(actual_seq_offsets_len, self.batch)?;
        ensure_len_at_least(
            block_table_len,
            checked_element_count(self.batch, self.block_table_batch_stride)?,
        )
    }
}

impl RaggedKvPrefillAttentionConfig {
    pub fn contiguous(
        batch: usize,
        total_query_tokens: usize,
        total_kv_tokens: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        causal: bool,
        scale: f32,
    ) -> Result<Self> {
        let config = Self {
            batch,
            total_query_tokens,
            total_kv_tokens,
            heads,
            kv_heads,
            head_dim,
            causal,
            scale,
        };
        validate_ragged_kv_prefill_attention_config(config)?;
        Ok(config)
    }

    fn output_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.total_query_tokens, self.heads)?,
            self.head_dim,
        )
    }

    fn lse_len(self) -> Result<usize> {
        checked_element_count(self.total_query_tokens, self.heads)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        lse_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
        actual_seq_lens_q_len: usize,
        actual_seq_lens_kv_len: usize,
        actual_seq_offsets_len: usize,
    ) -> Result<()> {
        validate_ragged_kv_prefill_attention_config(self)?;
        let output_len = self.output_len()?;
        let kv_len = checked_element_count(
            checked_element_count(self.total_kv_tokens, self.kv_heads)?,
            self.head_dim,
        )?;
        ensure_len_at_least(out_len, output_len)?;
        ensure_len_at_least(lse_len, self.lse_len()?)?;
        ensure_len_at_least(query_len, output_len)?;
        ensure_len_at_least(key_len, kv_len)?;
        ensure_len_at_least(value_len, kv_len)?;
        ensure_len_at_least(actual_seq_lens_q_len, self.batch)?;
        ensure_len_at_least(actual_seq_lens_kv_len, self.batch)?;
        ensure_len_at_least(actual_seq_offsets_len, self.batch)
    }
}

impl FusedNeighborhoodAttentionConfig {
    pub fn contiguous(
        batch: usize,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        kernel_size: usize,
        dilation: usize,
        scale: f32,
    ) -> Result<Self> {
        let config = Self {
            batch,
            seq_len,
            heads,
            head_dim,
            kernel_size,
            dilation,
            scale,
        };
        validate_fused_neighborhood_attention_config(config)?;
        Ok(config)
    }

    fn element_count(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(checked_element_count(self.batch, self.heads)?, self.seq_len)?,
            self.head_dim,
        )
    }

    fn validate_lengths(
        self,
        out_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
    ) -> Result<()> {
        validate_fused_neighborhood_attention_config(self)?;
        let expected = self.element_count()?;
        ensure_len_at_least(out_len, expected)?;
        ensure_len_at_least(query_len, expected)?;
        ensure_len_at_least(key_len, expected)?;
        ensure_len_at_least(value_len, expected)
    }
}

impl HeavilyCompressedAttentionConfig {
    pub fn create(
        batch: usize,
        seq_len: usize,
        hidden_dim: usize,
        heads: usize,
        head_dim: usize,
        compression_block: usize,
        groups: usize,
        group_dim: usize,
    ) -> Result<Self> {
        let config = Self {
            batch,
            seq_len,
            hidden_dim,
            heads,
            head_dim,
            compression_block,
            groups,
            group_dim,
        };
        validate_heavily_compressed_attention_config(config)?;
        Ok(config)
    }

    pub fn blocks(self) -> Result<usize> {
        validate_heavily_compressed_attention_config(self)?;
        Ok(self.seq_len / self.compression_block)
    }

    fn hidden_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.seq_len)?,
            self.hidden_dim,
        )
    }

    fn compressed_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.blocks()?)?,
            self.head_dim,
        )
    }

    fn query_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(checked_element_count(self.batch, self.seq_len)?, self.heads)?,
            self.head_dim,
        )
    }

    fn output_len(self) -> Result<usize> {
        self.hidden_len()
    }

    fn group_weight_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(
                checked_element_count(self.groups, self.heads / self.groups)?,
                self.head_dim,
            )?,
            self.group_dim,
        )
    }

    fn final_weight_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.groups, self.group_dim)?,
            self.hidden_dim,
        )
    }

    fn validate_compress_lengths(
        self,
        out_len: usize,
        hidden_len: usize,
        weight_kv_len: usize,
        weight_z_len: usize,
        bias_len: usize,
    ) -> Result<()> {
        validate_heavily_compressed_attention_config(self)?;
        let projection_len = checked_element_count(self.hidden_dim, self.head_dim)?;
        let bias_expected = checked_element_count(self.compression_block, self.head_dim)?;
        ensure_len_at_least(out_len, self.compressed_len()?)?;
        ensure_len_at_least(hidden_len, self.hidden_len()?)?;
        ensure_len_at_least(weight_kv_len, projection_len)?;
        ensure_len_at_least(weight_z_len, projection_len)?;
        ensure_len_at_least(bias_len, bias_expected)
    }

    fn validate_attention_lengths(
        self,
        out_len: usize,
        query_len: usize,
        compressed_kv_len: usize,
        weight_group_len: usize,
        weight_final_len: usize,
    ) -> Result<()> {
        validate_heavily_compressed_attention_config(self)?;
        ensure_len_at_least(out_len, self.output_len()?)?;
        ensure_len_at_least(query_len, self.query_len()?)?;
        ensure_len_at_least(compressed_kv_len, self.compressed_len()?)?;
        ensure_len_at_least(weight_group_len, self.group_weight_len()?)?;
        ensure_len_at_least(weight_final_len, self.final_weight_len()?)
    }
}

impl CompressedSparseAttentionCompressConfig {
    pub fn create(
        batch: usize,
        seq_len: usize,
        hidden_dim: usize,
        head_dim: usize,
        compression_block: usize,
    ) -> Result<Self> {
        let config = Self {
            batch,
            seq_len,
            hidden_dim,
            head_dim,
            compression_block,
        };
        validate_compressed_sparse_attention_compress_config(config)?;
        Ok(config)
    }

    pub fn blocks(self) -> Result<usize> {
        validate_compressed_sparse_attention_compress_config(self)?;
        Ok(self.seq_len / self.compression_block)
    }

    fn hidden_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.seq_len)?,
            self.hidden_dim,
        )
    }

    fn compressed_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.blocks()?)?,
            self.head_dim,
        )
    }

    fn validate_lengths(
        self,
        out_len: usize,
        hidden_len: usize,
        weight_a_kv_len: usize,
        weight_b_kv_len: usize,
        weight_a_z_len: usize,
        weight_b_z_len: usize,
        bias_a_len: usize,
        bias_b_len: usize,
    ) -> Result<()> {
        validate_compressed_sparse_attention_compress_config(self)?;
        let projection_len = checked_element_count(self.hidden_dim, self.head_dim)?;
        let bias_len = checked_element_count(self.compression_block, self.head_dim)?;
        ensure_len_at_least(out_len, self.compressed_len()?)?;
        ensure_len_at_least(hidden_len, self.hidden_len()?)?;
        ensure_len_at_least(weight_a_kv_len, projection_len)?;
        ensure_len_at_least(weight_b_kv_len, projection_len)?;
        ensure_len_at_least(weight_a_z_len, projection_len)?;
        ensure_len_at_least(weight_b_z_len, projection_len)?;
        ensure_len_at_least(bias_a_len, bias_len)?;
        ensure_len_at_least(bias_b_len, bias_len)
    }
}

impl CompressedSparseAttentionLightningIndexerConfig {
    pub fn create(
        batch: usize,
        query_len: usize,
        blocks: usize,
        index_heads: usize,
        index_dim: usize,
    ) -> Result<Self> {
        let config = Self {
            batch,
            query_len,
            blocks,
            index_heads,
            index_dim,
        };
        validate_compressed_sparse_attention_lightning_indexer_config(config)?;
        Ok(config)
    }

    fn output_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.query_len)?,
            self.blocks,
        )
    }

    fn query_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(
                checked_element_count(self.batch, self.query_len)?,
                self.index_heads,
            )?,
            self.index_dim,
        )
    }

    fn key_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.blocks)?,
            self.index_dim,
        )
    }

    fn weight_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.query_len)?,
            self.index_heads,
        )
    }

    fn validate_lengths(
        self,
        out_len: usize,
        indexer_query_len: usize,
        indexer_key_len: usize,
        indexer_weight_len: usize,
    ) -> Result<()> {
        validate_compressed_sparse_attention_lightning_indexer_config(self)?;
        ensure_len_at_least(out_len, self.output_len()?)?;
        ensure_len_at_least(indexer_query_len, self.query_len()?)?;
        ensure_len_at_least(indexer_key_len, self.key_len()?)?;
        ensure_len_at_least(indexer_weight_len, self.weight_len()?)
    }
}

impl CompressedSparseAttentionTopkSelectorConfig {
    pub fn create(
        batch: usize,
        query_len: usize,
        blocks: usize,
        head_dim: usize,
        top_k: usize,
        compression_block: usize,
    ) -> Result<Self> {
        let config = Self {
            batch,
            query_len,
            blocks,
            head_dim,
            top_k,
            compression_block,
        };
        validate_compressed_sparse_attention_topk_selector_config(config)?;
        Ok(config)
    }

    fn output_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(
                checked_element_count(self.batch, self.query_len)?,
                self.top_k,
            )?,
            self.head_dim,
        )
    }

    fn selected_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.query_len)?,
            self.top_k,
        )
    }

    fn scores_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.query_len)?,
            self.blocks,
        )
    }

    fn compressed_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.blocks)?,
            self.head_dim,
        )
    }

    fn query_positions_len(self) -> Result<usize> {
        checked_element_count(self.batch, self.query_len)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        selected_len: usize,
        scores_len: usize,
        compressed_kv_len: usize,
        query_positions_len: usize,
    ) -> Result<()> {
        validate_compressed_sparse_attention_topk_selector_config(self)?;
        ensure_len_at_least(out_len, self.output_len()?)?;
        ensure_len_at_least(selected_len, self.selected_len()?)?;
        ensure_len_at_least(scores_len, self.scores_len()?)?;
        ensure_len_at_least(compressed_kv_len, self.compressed_len()?)?;
        ensure_len_at_least(query_positions_len, self.query_positions_len()?)
    }
}

impl CompressedSparseAttentionSharedMqaConfig {
    pub fn create(
        batch: usize,
        query_len: usize,
        heads: usize,
        head_dim: usize,
        kv_len: usize,
        scale: f32,
    ) -> Result<Self> {
        let config = Self {
            batch,
            query_len,
            heads,
            head_dim,
            kv_len,
            scale,
        };
        validate_compressed_sparse_attention_shared_mqa_config(config)?;
        Ok(config)
    }

    fn output_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(
                checked_element_count(self.batch, self.query_len)?,
                self.heads,
            )?,
            self.head_dim,
        )
    }

    fn kv_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(
                checked_element_count(self.batch, self.query_len)?,
                self.kv_len,
            )?,
            self.head_dim,
        )
    }

    fn valid_mask_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.query_len)?,
            self.kv_len,
        )
    }

    fn validate_lengths(
        self,
        out_len: usize,
        query_len: usize,
        kv_entries_len: usize,
        valid_mask_len: usize,
        sink_len: usize,
    ) -> Result<()> {
        validate_compressed_sparse_attention_shared_mqa_config(self)?;
        ensure_len_at_least(out_len, self.output_len()?)?;
        ensure_len_at_least(query_len, self.output_len()?)?;
        ensure_len_at_least(kv_entries_len, self.kv_len()?)?;
        ensure_len_at_least(valid_mask_len, self.valid_mask_len()?)?;
        ensure_len_at_least(sink_len, self.heads)
    }
}

impl MiniMaxSparseAttentionBlockMaxConfig {
    pub fn create(batch: usize, rows: usize, key_len: usize, block_size: usize) -> Result<Self> {
        let config = Self {
            batch,
            rows,
            key_len,
            block_size,
        };
        validate_minimax_sparse_attention_block_max_config(config)?;
        Ok(config)
    }

    pub fn blocks(self) -> Result<usize> {
        validate_minimax_sparse_attention_block_max_config(self)?;
        Ok(self.key_len.div_ceil(self.block_size))
    }

    fn input_len(self) -> Result<usize> {
        checked_element_count(checked_element_count(self.batch, self.rows)?, self.key_len)
    }

    fn output_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.rows)?,
            self.blocks()?,
        )
    }

    fn validate_lengths(self, out_len: usize, scores_len: usize) -> Result<()> {
        validate_minimax_sparse_attention_block_max_config(self)?;
        ensure_len_at_least(out_len, self.output_len()?)?;
        ensure_len_at_least(scores_len, self.input_len()?)
    }
}

impl MiniMaxSparseAttentionSelectedTokenPositionsConfig {
    pub fn create(
        batch: usize,
        rows: usize,
        selected_blocks: usize,
        block_size: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let config = Self {
            batch,
            rows,
            selected_blocks,
            block_size,
            seq_len,
        };
        validate_minimax_sparse_attention_selected_token_positions_config(config)?;
        Ok(config)
    }

    pub fn expanded_keys(self) -> Result<usize> {
        validate_minimax_sparse_attention_selected_token_positions_config(self)?;
        checked_element_count(self.selected_blocks, self.block_size)
    }

    fn selected_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.rows)?,
            self.selected_blocks,
        )
    }

    fn query_positions_len(self) -> Result<usize> {
        checked_element_count(self.batch, self.rows)
    }

    fn output_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.rows)?,
            self.expanded_keys()?,
        )
    }

    fn validate_lengths(
        self,
        positions_len: usize,
        valid_mask_len: usize,
        selected_len: usize,
        query_positions_len: usize,
    ) -> Result<()> {
        validate_minimax_sparse_attention_selected_token_positions_config(self)?;
        ensure_len_at_least(positions_len, self.output_len()?)?;
        ensure_len_at_least(valid_mask_len, self.output_len()?)?;
        ensure_len_at_least(selected_len, self.selected_len()?)?;
        ensure_len_at_least(query_positions_len, self.query_positions_len()?)
    }
}

impl MiniMaxSparseAttentionSelectTopkBlocksConfig {
    pub fn create(batch: usize, rows: usize, blocks: usize, top_k: usize) -> Result<Self> {
        let config = Self {
            batch,
            rows,
            blocks,
            top_k,
        };
        validate_minimax_sparse_attention_select_topk_blocks_config(config)?;
        Ok(config)
    }

    fn scores_len(self) -> Result<usize> {
        checked_element_count(checked_element_count(self.batch, self.rows)?, self.blocks)
    }

    fn local_blocks_len(self) -> Result<usize> {
        checked_element_count(self.batch, self.rows)
    }

    fn output_len(self) -> Result<usize> {
        checked_element_count(checked_element_count(self.batch, self.rows)?, self.top_k)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        block_scores_len: usize,
        local_blocks_len: usize,
    ) -> Result<()> {
        validate_minimax_sparse_attention_select_topk_blocks_config(self)?;
        ensure_len_at_least(out_len, self.output_len()?)?;
        ensure_len_at_least(block_scores_len, self.scores_len()?)?;
        ensure_len_at_least(local_blocks_len, self.local_blocks_len()?)
    }
}

impl MiniMaxSparseAttentionGatheredGqaConfig {
    pub fn create(
        batch: usize,
        query_len: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        selected_keys: usize,
        scale: f32,
    ) -> Result<Self> {
        let config = Self {
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            head_dim,
            selected_keys,
            scale,
        };
        validate_minimax_sparse_attention_gathered_gqa_config(config)?;
        Ok(config)
    }

    fn output_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(
                checked_element_count(self.batch, self.query_len)?,
                self.heads,
            )?,
            self.head_dim,
        )
    }

    fn query_len_elements(self) -> Result<usize> {
        self.output_len()
    }

    fn kv_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(
                checked_element_count(self.batch, self.kv_heads)?,
                self.key_len,
            )?,
            self.head_dim,
        )
    }

    fn sparse_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(
                checked_element_count(self.batch, self.kv_heads)?,
                self.query_len,
            )?,
            self.selected_keys,
        )
    }

    fn validate_lengths(
        self,
        out_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
        positions_len: usize,
        valid_mask_len: usize,
    ) -> Result<()> {
        validate_minimax_sparse_attention_gathered_gqa_config(self)?;
        ensure_len_at_least(out_len, self.output_len()?)?;
        ensure_len_at_least(query_len, self.query_len_elements()?)?;
        ensure_len_at_least(key_len, self.kv_len()?)?;
        ensure_len_at_least(value_len, self.kv_len()?)?;
        ensure_len_at_least(positions_len, self.sparse_len()?)?;
        ensure_len_at_least(valid_mask_len, self.sparse_len()?)
    }
}

impl MultiTokenAttentionConfig {
    pub fn same_padding_1x1(
        batch: usize,
        channels_in: usize,
        channels_out: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let config = Self {
            batch,
            channels_in,
            channels_out,
            seq_len,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            padding_h: 0,
            padding_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        };
        validate_multi_token_attention_config(config)?;
        Ok(config)
    }

    pub fn output_seq_len(self) -> Result<usize> {
        validate_multi_token_attention_config(self)?;
        let output_h = conv2d_output_len(
            self.seq_len,
            self.kernel_h,
            self.stride_h,
            self.padding_h,
            self.dilation_h,
        )?;
        let output_w = conv2d_output_len(
            self.seq_len,
            self.kernel_w,
            self.stride_w,
            self.padding_w,
            self.dilation_w,
        )?;
        if output_h == 0 || output_h != output_w {
            return Err(Error::InvalidLength);
        }
        Ok(output_h)
    }

    fn output_len(self) -> Result<usize> {
        let output_seq_len = self.output_seq_len()?;
        checked_element_count(
            checked_element_count(
                checked_element_count(self.batch, self.channels_out)?,
                output_seq_len,
            )?,
            output_seq_len,
        )
    }

    fn validate_lengths(
        self,
        out_len: usize,
        scores_len: usize,
        weight_len: usize,
        bias_len: Option<usize>,
    ) -> Result<()> {
        validate_multi_token_attention_config(self)?;
        let scores_expected = checked_element_count(
            checked_element_count(
                checked_element_count(self.batch, self.channels_in)?,
                self.seq_len,
            )?,
            self.seq_len,
        )?;
        let channels_in_per_group = self.channels_in / self.groups;
        let weight_expected = checked_element_count(
            checked_element_count(
                checked_element_count(self.channels_out, channels_in_per_group)?,
                self.kernel_h,
            )?,
            self.kernel_w,
        )?;
        ensure_len_at_least(scores_len, scores_expected)?;
        ensure_len_at_least(weight_len, weight_expected)?;
        ensure_len_at_least(out_len, self.output_len()?)?;
        if let Some(bias_len) = bias_len {
            ensure_len_at_least(bias_len, self.channels_out)?;
        }
        Ok(())
    }
}

impl FmhaPrefillConfig {
    pub fn contiguous(
        batch: usize,
        query_len: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
        causal: bool,
    ) -> Result<Self> {
        let q_features = checked_element_count(heads, head_dim)?;
        let kv_features = checked_element_count(kv_heads, head_dim)?;
        Ok(Self {
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            head_dim,
            query_batch_stride: checked_element_count(query_len, q_features)?,
            key_batch_stride: checked_element_count(key_len, kv_features)?,
            value_batch_stride: checked_element_count(key_len, kv_features)?,
            output_batch_stride: checked_element_count(query_len, q_features)?,
            query_sequence_stride: q_features,
            key_sequence_stride: kv_features,
            value_sequence_stride: kv_features,
            output_sequence_stride: q_features,
            query_head_stride: head_dim,
            key_head_stride: head_dim,
            value_head_stride: head_dim,
            output_head_stride: head_dim,
            scale,
            causal,
        })
    }

    fn validate_lengths(
        self,
        out_len: usize,
        lse_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
    ) -> Result<()> {
        validate_fmha_prefill_config(self)?;
        let query_item_reach = projected_layout_reach(
            self.query_len,
            self.heads,
            self.head_dim,
            self.query_sequence_stride,
            self.query_head_stride,
        )?;
        let key_item_reach = projected_layout_reach(
            self.key_len,
            self.kv_heads,
            self.head_dim,
            self.key_sequence_stride,
            self.key_head_stride,
        )?;
        let value_item_reach = projected_layout_reach(
            self.key_len,
            self.kv_heads,
            self.head_dim,
            self.value_sequence_stride,
            self.value_head_stride,
        )?;
        let output_item_reach = projected_layout_reach(
            self.query_len,
            self.heads,
            self.head_dim,
            self.output_sequence_stride,
            self.output_head_stride,
        )?;
        ensure_len_at_least(
            query_len,
            batched_reach(self.batch, query_item_reach, self.query_batch_stride)?,
        )?;
        ensure_len_at_least(
            key_len,
            batched_reach(self.batch, key_item_reach, self.key_batch_stride)?,
        )?;
        ensure_len_at_least(
            value_len,
            batched_reach(self.batch, value_item_reach, self.value_batch_stride)?,
        )?;
        ensure_len_at_least(
            out_len,
            batched_reach(self.batch, output_item_reach, self.output_batch_stride)?,
        )?;
        ensure_len_at_least(
            lse_len,
            checked_element_count(
                checked_element_count(self.batch, self.heads)?,
                self.query_len,
            )?,
        )?;
        Ok(())
    }

    fn validate_output_lengths(
        self,
        out_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
    ) -> Result<()> {
        validate_fmha_prefill_config(self)?;
        let query_item_reach = projected_layout_reach(
            self.query_len,
            self.heads,
            self.head_dim,
            self.query_sequence_stride,
            self.query_head_stride,
        )?;
        let key_item_reach = projected_layout_reach(
            self.key_len,
            self.kv_heads,
            self.head_dim,
            self.key_sequence_stride,
            self.key_head_stride,
        )?;
        let value_item_reach = projected_layout_reach(
            self.key_len,
            self.kv_heads,
            self.head_dim,
            self.value_sequence_stride,
            self.value_head_stride,
        )?;
        let output_item_reach = projected_layout_reach(
            self.query_len,
            self.heads,
            self.head_dim,
            self.output_sequence_stride,
            self.output_head_stride,
        )?;
        ensure_len_at_least(
            query_len,
            batched_reach(self.batch, query_item_reach, self.query_batch_stride)?,
        )?;
        ensure_len_at_least(
            key_len,
            batched_reach(self.batch, key_item_reach, self.key_batch_stride)?,
        )?;
        ensure_len_at_least(
            value_len,
            batched_reach(self.batch, value_item_reach, self.value_batch_stride)?,
        )?;
        ensure_len_at_least(
            out_len,
            batched_reach(self.batch, output_item_reach, self.output_batch_stride)?,
        )?;
        Ok(())
    }
}

impl MlaDecodeSplitKConfig {
    pub fn contiguous(
        batch: usize,
        key_len: usize,
        heads: usize,
        splits: usize,
        kv_len_per_split: usize,
        head_dim: usize,
        pe_dim: usize,
        scale: f32,
    ) -> Result<Self> {
        let config = Self {
            batch,
            key_len,
            heads,
            splits,
            kv_len_per_split,
            head_dim,
            pe_dim,
            scale,
        };
        validate_mla_decode_splitk_config(config)?;
        Ok(config)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        lse_len: usize,
        query_len: usize,
        query_pe_len: usize,
        key_value_len: usize,
        key_pe_len: usize,
    ) -> Result<()> {
        validate_mla_decode_splitk_config(self)?;
        let split_values = checked_element_count(self.splits, self.head_dim)?;
        let out_expected =
            checked_element_count(self.batch, checked_element_count(self.heads, split_values)?)?;
        let lse_expected =
            checked_element_count(self.batch, checked_element_count(self.heads, self.splits)?)?;
        let q_len = checked_element_count(
            self.batch,
            checked_element_count(self.heads, self.head_dim)?,
        )?;
        let qpe_len =
            checked_element_count(self.batch, checked_element_count(self.heads, self.pe_dim)?)?;
        let kv_len = checked_element_count(
            self.batch,
            checked_element_count(self.key_len, self.head_dim)?,
        )?;
        let kpe_len = checked_element_count(
            self.batch,
            checked_element_count(self.key_len, self.pe_dim)?,
        )?;
        ensure_len_at_least(out_len, out_expected)?;
        ensure_len_at_least(lse_len, lse_expected)?;
        ensure_len_at_least(query_len, q_len)?;
        ensure_len_at_least(query_pe_len, qpe_len)?;
        ensure_len_at_least(key_value_len, kv_len)?;
        ensure_len_at_least(key_pe_len, kpe_len)?;
        Ok(())
    }
}

impl AttentionSinkPrefillConfig {
    pub fn contiguous(
        batch: usize,
        query_len: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        start_q: usize,
        window: usize,
        scale: f32,
        causal: bool,
    ) -> Result<Self> {
        Ok(Self {
            attention: FmhaPrefillConfig::contiguous(
                batch, query_len, key_len, heads, kv_heads, head_dim, scale, causal,
            )?,
            start_q,
            window,
        })
    }

    fn validate_lengths(
        self,
        out_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
        sinks_len: usize,
    ) -> Result<()> {
        self.attention
            .validate_output_lengths(out_len, query_len, key_len, value_len)?;
        ensure_len_at_least(sinks_len, self.attention.heads)?;
        checked_i32_value(self.start_q)?;
        checked_i32_value(self.window)?;
        Ok(())
    }
}

impl MlaDecodeConfig {
    pub fn contiguous(
        batch: usize,
        key_len: usize,
        heads: usize,
        head_dim: usize,
        pe_dim: usize,
        scale: f32,
    ) -> Result<Self> {
        let config = Self {
            batch,
            key_len,
            heads,
            head_dim,
            pe_dim,
            scale,
        };
        validate_mla_decode_config(config)?;
        Ok(config)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        lse_len: usize,
        query_len: usize,
        query_pe_len: usize,
        key_value_len: usize,
        key_pe_len: usize,
    ) -> Result<()> {
        validate_mla_decode_config(self)?;
        let q_len = checked_element_count(
            self.batch,
            checked_element_count(self.heads, self.head_dim)?,
        )?;
        let qpe_len =
            checked_element_count(self.batch, checked_element_count(self.heads, self.pe_dim)?)?;
        let kv_len = checked_element_count(
            self.batch,
            checked_element_count(self.key_len, self.head_dim)?,
        )?;
        let kpe_len = checked_element_count(
            self.batch,
            checked_element_count(self.key_len, self.pe_dim)?,
        )?;
        ensure_len_at_least(out_len, q_len)?;
        ensure_len_at_least(lse_len, checked_element_count(self.batch, self.heads)?)?;
        ensure_len_at_least(query_len, q_len)?;
        ensure_len_at_least(query_pe_len, qpe_len)?;
        ensure_len_at_least(key_value_len, kv_len)?;
        ensure_len_at_least(key_pe_len, kpe_len)?;
        Ok(())
    }
}

impl PagedMlaDecodeAttentionConfig {
    pub fn contiguous(
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
        let config = Self {
            batch,
            heads,
            head_dim,
            pe_dim,
            block_size,
            physical_blocks,
            block_table_batch_stride,
            scale,
            output_scale,
        };
        validate_paged_mla_decode_attention_config(config)?;
        Ok(config)
    }

    fn output_len(self) -> Result<usize> {
        checked_element_count(
            checked_element_count(self.batch, self.heads)?,
            self.head_dim,
        )
    }

    fn lse_len(self) -> Result<usize> {
        checked_element_count(self.batch, self.heads)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        lse_len: usize,
        query_len: usize,
        query_pe_len: usize,
        key_value_cache_len: usize,
        key_pe_cache_len: usize,
        actual_seq_lens_len: usize,
        block_table_len: usize,
    ) -> Result<()> {
        validate_paged_mla_decode_attention_config(self)?;
        let output_len = self.output_len()?;
        let query_pe_len_expected =
            checked_element_count(checked_element_count(self.batch, self.heads)?, self.pe_dim)?;
        let key_value_cache_len_expected = checked_element_count(
            checked_element_count(self.physical_blocks, self.block_size)?,
            self.head_dim,
        )?;
        let key_pe_cache_len_expected = checked_element_count(
            checked_element_count(self.physical_blocks, self.block_size)?,
            self.pe_dim,
        )?;
        ensure_len_at_least(out_len, output_len)?;
        ensure_len_at_least(lse_len, self.lse_len()?)?;
        ensure_len_at_least(query_len, output_len)?;
        ensure_len_at_least(query_pe_len, query_pe_len_expected)?;
        ensure_len_at_least(key_value_cache_len, key_value_cache_len_expected)?;
        ensure_len_at_least(key_pe_cache_len, key_pe_cache_len_expected)?;
        ensure_len_at_least(actual_seq_lens_len, self.batch)?;
        ensure_len_at_least(
            block_table_len,
            checked_element_count(self.batch, self.block_table_batch_stride)?,
        )
    }
}

impl MlaPrefillConfig {
    pub fn contiguous(
        batch: usize,
        query_len: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pe_dim: usize,
        scale: f32,
    ) -> Result<Self> {
        let config = Self {
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            head_dim,
            pe_dim,
            scale,
        };
        validate_mla_prefill_config(config)?;
        Ok(config)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        query_len: usize,
        query_pe_len: usize,
        key_len: usize,
        key_pe_len: usize,
        value_len: usize,
    ) -> Result<()> {
        validate_mla_prefill_config(self)?;
        let q_features = checked_element_count(self.heads, self.head_dim)?;
        let qpe_features = checked_element_count(self.heads, self.pe_dim)?;
        let kv_features = checked_element_count(self.kv_heads, self.head_dim)?;
        let kpe_features = checked_element_count(self.kv_heads, self.pe_dim)?;
        ensure_len_at_least(
            out_len,
            checked_element_count(
                self.batch,
                checked_element_count(self.query_len, q_features)?,
            )?,
        )?;
        ensure_len_at_least(
            query_len,
            checked_element_count(
                self.batch,
                checked_element_count(self.query_len, q_features)?,
            )?,
        )?;
        ensure_len_at_least(
            query_pe_len,
            checked_element_count(
                self.batch,
                checked_element_count(self.query_len, qpe_features)?,
            )?,
        )?;
        ensure_len_at_least(
            key_len,
            checked_element_count(
                self.batch,
                checked_element_count(self.key_len, kv_features)?,
            )?,
        )?;
        ensure_len_at_least(
            key_pe_len,
            checked_element_count(
                self.batch,
                checked_element_count(self.key_len, kpe_features)?,
            )?,
        )?;
        ensure_len_at_least(
            value_len,
            checked_element_count(
                self.batch,
                checked_element_count(self.key_len, kv_features)?,
            )?,
        )?;
        Ok(())
    }
}

impl SoftcappedWindowAttentionConfig {
    pub fn bnsd_contiguous(
        batch: usize,
        query_len: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
        causal: bool,
        window_size: usize,
        soft_cap: Option<f32>,
    ) -> Result<Self> {
        let query_head_stride = checked_element_count(query_len, head_dim)?;
        let key_head_stride = checked_element_count(key_len, head_dim)?;
        let attention = FmhaPrefillConfig {
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            head_dim,
            query_batch_stride: checked_element_count(heads, query_head_stride)?,
            key_batch_stride: checked_element_count(kv_heads, key_head_stride)?,
            value_batch_stride: checked_element_count(kv_heads, key_head_stride)?,
            output_batch_stride: checked_element_count(heads, query_head_stride)?,
            query_sequence_stride: head_dim,
            key_sequence_stride: head_dim,
            value_sequence_stride: head_dim,
            output_sequence_stride: head_dim,
            query_head_stride,
            key_head_stride,
            value_head_stride: key_head_stride,
            output_head_stride: query_head_stride,
            scale,
            causal,
        };
        let config = Self {
            attention,
            window_size,
            soft_cap,
        };
        validate_softcapped_window_attention_config(config)?;
        Ok(config)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
    ) -> Result<()> {
        validate_softcapped_window_attention_config(self)?;
        self.attention
            .validate_output_lengths(out_len, query_len, key_len, value_len)
    }
}

impl SparseMlaPrefillConfig {
    pub fn contiguous(
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
        let config = Self {
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            head_dim,
            pe_dim,
            topk,
            scale,
        };
        validate_sparse_mla_prefill_config(config)?;
        Ok(config)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        query_len: usize,
        query_pe_len: usize,
        key_len: usize,
        key_pe_len: usize,
        value_len: usize,
        indices_len: usize,
    ) -> Result<()> {
        validate_sparse_mla_prefill_config(self)?;
        let q_features = checked_element_count(self.heads, self.head_dim)?;
        let qpe_features = checked_element_count(self.heads, self.pe_dim)?;
        let kv_features = checked_element_count(self.kv_heads, self.head_dim)?;
        ensure_len_at_least(
            out_len,
            checked_element_count(
                self.batch,
                checked_element_count(self.query_len, q_features)?,
            )?,
        )?;
        ensure_len_at_least(
            query_len,
            checked_element_count(
                self.batch,
                checked_element_count(self.query_len, q_features)?,
            )?,
        )?;
        ensure_len_at_least(
            query_pe_len,
            checked_element_count(
                self.batch,
                checked_element_count(self.query_len, qpe_features)?,
            )?,
        )?;
        ensure_len_at_least(
            key_len,
            checked_element_count(
                self.batch,
                checked_element_count(self.key_len, kv_features)?,
            )?,
        )?;
        ensure_len_at_least(
            key_pe_len,
            checked_element_count(
                self.batch,
                checked_element_count(self.key_len, self.pe_dim)?,
            )?,
        )?;
        ensure_len_at_least(
            value_len,
            checked_element_count(
                self.batch,
                checked_element_count(self.key_len, kv_features)?,
            )?,
        )?;
        ensure_len_at_least(
            indices_len,
            checked_element_count(
                self.batch,
                checked_element_count(
                    self.query_len,
                    checked_element_count(self.kv_heads, self.topk)?,
                )?,
            )?,
        )?;
        Ok(())
    }
}

impl FmhaDecodeConfig {
    pub fn contiguous(
        batch: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Self> {
        let q_features = checked_element_count(heads, head_dim)?;
        let kv_features = checked_element_count(kv_heads, head_dim)?;
        Ok(Self {
            batch,
            key_len,
            heads,
            kv_heads,
            head_dim,
            query_batch_stride: q_features,
            key_batch_stride: checked_element_count(key_len, kv_features)?,
            value_batch_stride: checked_element_count(key_len, kv_features)?,
            output_batch_stride: q_features,
            query_head_stride: head_dim,
            key_sequence_stride: kv_features,
            value_sequence_stride: kv_features,
            output_head_stride: head_dim,
            key_head_stride: head_dim,
            value_head_stride: head_dim,
            scale,
        })
    }

    fn validate_lengths(
        self,
        out_len: usize,
        lse_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
    ) -> Result<()> {
        validate_fmha_decode_config(self)?;
        let query_item_reach =
            projected_layout_reach(1, self.heads, self.head_dim, 1, self.query_head_stride)?;
        let key_item_reach = projected_layout_reach(
            self.key_len,
            self.kv_heads,
            self.head_dim,
            self.key_sequence_stride,
            self.key_head_stride,
        )?;
        let value_item_reach = projected_layout_reach(
            self.key_len,
            self.kv_heads,
            self.head_dim,
            self.value_sequence_stride,
            self.value_head_stride,
        )?;
        let output_item_reach =
            projected_layout_reach(1, self.heads, self.head_dim, 1, self.output_head_stride)?;
        ensure_len_at_least(
            query_len,
            batched_reach(self.batch, query_item_reach, self.query_batch_stride)?,
        )?;
        ensure_len_at_least(
            key_len,
            batched_reach(self.batch, key_item_reach, self.key_batch_stride)?,
        )?;
        ensure_len_at_least(
            value_len,
            batched_reach(self.batch, value_item_reach, self.value_batch_stride)?,
        )?;
        ensure_len_at_least(
            out_len,
            batched_reach(self.batch, output_item_reach, self.output_batch_stride)?,
        )?;
        ensure_len_at_least(lse_len, checked_element_count(self.batch, self.heads)?)?;
        Ok(())
    }
}

impl SplitKReduceConfig {
    pub fn contiguous(batch: usize, heads: usize, splits: usize, head_dim: usize) -> Result<Self> {
        let split_values = checked_element_count(splits, head_dim)?;
        let output_values = checked_element_count(heads, head_dim)?;
        Ok(Self {
            batch,
            heads,
            splits,
            head_dim,
            attn_batch_stride: checked_element_count(heads, split_values)?,
            attn_head_stride: split_values,
            attn_split_stride: head_dim,
            lse_batch_stride: checked_element_count(heads, splits)?,
            lse_head_stride: splits,
            output_batch_stride: output_values,
            output_head_stride: head_dim,
        })
    }

    fn validate_lengths(
        self,
        out_len: usize,
        attn_splitk_len: usize,
        lse_splitk_len: usize,
    ) -> Result<()> {
        validate_splitk_reduce_config(self)?;
        let attn_item_reach = splitk_attn_layout_reach(
            self.heads,
            self.splits,
            self.head_dim,
            self.attn_head_stride,
            self.attn_split_stride,
        )?;
        let lse_item_reach =
            splitk_lse_layout_reach(self.heads, self.splits, self.lse_head_stride)?;
        let output_item_reach =
            projected_layout_reach(1, self.heads, self.head_dim, 1, self.output_head_stride)?;
        ensure_len_at_least(
            attn_splitk_len,
            batched_reach(self.batch, attn_item_reach, self.attn_batch_stride)?,
        )?;
        ensure_len_at_least(
            lse_splitk_len,
            batched_reach(self.batch, lse_item_reach, self.lse_batch_stride)?,
        )?;
        ensure_len_at_least(
            out_len,
            batched_reach(self.batch, output_item_reach, self.output_batch_stride)?,
        )?;
        Ok(())
    }
}

impl SoftcappedWindowDecodeConfig {
    pub fn bnsd_contiguous(
        batch: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
        window_size: usize,
        soft_cap: Option<f32>,
    ) -> Result<Self> {
        let query_batch_stride = checked_element_count(heads, head_dim)?;
        let key_head_stride = checked_element_count(key_len, head_dim)?;
        let decode = FmhaDecodeConfig {
            batch,
            key_len,
            heads,
            kv_heads,
            head_dim,
            query_batch_stride,
            key_batch_stride: checked_element_count(kv_heads, key_head_stride)?,
            value_batch_stride: checked_element_count(kv_heads, key_head_stride)?,
            output_batch_stride: query_batch_stride,
            query_head_stride: head_dim,
            key_sequence_stride: head_dim,
            value_sequence_stride: head_dim,
            output_head_stride: head_dim,
            key_head_stride,
            value_head_stride: key_head_stride,
            scale,
        };
        let config = Self {
            decode,
            window_size,
            soft_cap,
        };
        validate_softcapped_window_decode_config(config)?;
        Ok(config)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        lse_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
    ) -> Result<()> {
        validate_softcapped_window_decode_config(self)?;
        self.decode
            .validate_lengths(out_len, lse_len, query_len, key_len, value_len)
    }
}

impl SoftcappedWindowDecodeSplitKConfig {
    pub fn contiguous(
        batch: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        splits: usize,
        kv_len_per_split: usize,
        head_dim: usize,
        scale: f32,
        window_size: usize,
        soft_cap: Option<f32>,
    ) -> Result<Self> {
        let config = Self {
            decode: FmhaDecodeSplitKConfig::contiguous(
                batch,
                key_len,
                heads,
                kv_heads,
                splits,
                kv_len_per_split,
                head_dim,
                scale,
            )?,
            window_size,
            soft_cap,
        };
        validate_softcapped_window_decode_splitk_config(config)?;
        Ok(config)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        lse_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
    ) -> Result<()> {
        validate_softcapped_window_decode_splitk_config(self)?;
        self.decode
            .validate_lengths(out_len, lse_len, query_len, key_len, value_len)
    }
}

impl FmhaDecodeSplitKConfig {
    pub fn contiguous(
        batch: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        splits: usize,
        kv_len_per_split: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Self> {
        let q_features = checked_element_count(heads, head_dim)?;
        let kv_features = checked_element_count(kv_heads, head_dim)?;
        let split_values = checked_element_count(splits, head_dim)?;
        Ok(Self {
            batch,
            key_len,
            heads,
            kv_heads,
            splits,
            kv_len_per_split,
            head_dim,
            query_batch_stride: q_features,
            key_batch_stride: checked_element_count(key_len, kv_features)?,
            value_batch_stride: checked_element_count(key_len, kv_features)?,
            output_batch_stride: checked_element_count(heads, split_values)?,
            query_head_stride: head_dim,
            key_sequence_stride: kv_features,
            value_sequence_stride: kv_features,
            output_head_stride: split_values,
            output_split_stride: head_dim,
            key_head_stride: head_dim,
            value_head_stride: head_dim,
            lse_batch_stride: checked_element_count(heads, splits)?,
            lse_head_stride: splits,
            scale,
        })
    }

    fn validate_lengths(
        self,
        out_len: usize,
        lse_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
    ) -> Result<()> {
        validate_fmha_decode_splitk_config(self)?;
        let query_item_reach =
            projected_layout_reach(1, self.heads, self.head_dim, 1, self.query_head_stride)?;
        let key_item_reach = projected_layout_reach(
            self.key_len,
            self.kv_heads,
            self.head_dim,
            self.key_sequence_stride,
            self.key_head_stride,
        )?;
        let value_item_reach = projected_layout_reach(
            self.key_len,
            self.kv_heads,
            self.head_dim,
            self.value_sequence_stride,
            self.value_head_stride,
        )?;
        let output_item_reach = splitk_attn_layout_reach(
            self.heads,
            self.splits,
            self.head_dim,
            self.output_head_stride,
            self.output_split_stride,
        )?;
        let lse_item_reach =
            splitk_lse_layout_reach(self.heads, self.splits, self.lse_head_stride)?;
        ensure_len_at_least(
            query_len,
            batched_reach(self.batch, query_item_reach, self.query_batch_stride)?,
        )?;
        ensure_len_at_least(
            key_len,
            batched_reach(self.batch, key_item_reach, self.key_batch_stride)?,
        )?;
        ensure_len_at_least(
            value_len,
            batched_reach(self.batch, value_item_reach, self.value_batch_stride)?,
        )?;
        ensure_len_at_least(
            out_len,
            batched_reach(self.batch, output_item_reach, self.output_batch_stride)?,
        )?;
        ensure_len_at_least(
            lse_len,
            batched_reach(self.batch, lse_item_reach, self.lse_batch_stride)?,
        )?;
        Ok(())
    }
}

impl AttentionSinkDecodeSplitKConfig {
    pub fn contiguous(
        batch: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        splits: usize,
        kv_len_per_split: usize,
        head_dim: usize,
        start_q: usize,
        window: usize,
        scale: f32,
    ) -> Result<Self> {
        Ok(Self {
            decode: FmhaDecodeSplitKConfig::contiguous(
                batch,
                key_len,
                heads,
                kv_heads,
                splits,
                kv_len_per_split,
                head_dim,
                scale,
            )?,
            start_q,
            window,
        })
    }

    fn validate_lengths(
        self,
        out_len: usize,
        lse_len: usize,
        query_len: usize,
        key_len: usize,
        value_len: usize,
        sinks_len: usize,
    ) -> Result<()> {
        validate_attention_sink_decode_splitk_config(self)?;
        self.decode
            .validate_lengths(out_len, lse_len, query_len, key_len, value_len)?;
        ensure_len_at_least(sinks_len, self.decode.heads)
    }
}

#[cfg(feature = "dtype-f32")]
/// Causal left-window attention.
///
/// For query absolute position `q_abs`, only keys satisfying
/// `key_abs <= q_abs` and `key_abs + window > q_abs` are visible. KV heads are
/// mapped with contiguous grouped-query attention semantics.
pub fn sliding_window_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    config: SlidingWindowAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::sliding_window_attention_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]

pub fn mla_prefill_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    query_pe: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    key_pe: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::mla_prefill_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f16")]

pub fn mla_prefill_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    query: &impl DeviceSlice<f16>,
    query_pe: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    key_pe: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::mla_prefill_f16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-bf16")]

pub fn mla_prefill_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    query: &impl DeviceSlice<bf16>,
    query_pe: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    key_pe: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::mla_prefill_bf16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]

pub fn sparse_mla_prefill_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    query_pe: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    key_pe: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    indices: &impl DeviceSlice<i32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::sparse_mla_prefill_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]

pub fn mla_decode_lse_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    query_pe: &impl DeviceSlice<f32>,
    key_value: &impl DeviceSlice<f32>,
    key_pe: &impl DeviceSlice<f32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::mla_decode_lse_f32(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f32")]

pub fn paged_mla_decode_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    query_pe: &impl DeviceSlice<f32>,
    key_value_cache: &impl DeviceSlice<f32>,
    key_pe_cache: &impl DeviceSlice<f32>,
    actual_seq_lens: &impl DeviceSlice<i32>,
    block_table: &impl DeviceSlice<u32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::paged_mla_decode_attention_f32(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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
#[cfg(feature = "dtype-f16")]

pub fn paged_mla_decode_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f16>,
    query_pe: &impl DeviceSlice<f16>,
    key_value_cache: &impl DeviceSlice<f16>,
    key_pe_cache: &impl DeviceSlice<f16>,
    actual_seq_lens: &impl DeviceSlice<i32>,
    block_table: &impl DeviceSlice<u32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::paged_mla_decode_attention_f16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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
#[cfg(feature = "dtype-bf16")]

pub fn paged_mla_decode_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<bf16>,
    query_pe: &impl DeviceSlice<bf16>,
    key_value_cache: &impl DeviceSlice<bf16>,
    key_pe_cache: &impl DeviceSlice<bf16>,
    actual_seq_lens: &impl DeviceSlice<i32>,
    block_table: &impl DeviceSlice<u32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::paged_mla_decode_attention_bf16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f16")]

pub fn mla_decode_lse_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f16>,
    query_pe: &impl DeviceSlice<f16>,
    key_value: &impl DeviceSlice<f16>,
    key_pe: &impl DeviceSlice<f16>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::mla_decode_lse_f16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-bf16")]

pub fn mla_decode_lse_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<bf16>,
    query_pe: &impl DeviceSlice<bf16>,
    key_value: &impl DeviceSlice<bf16>,
    key_pe: &impl DeviceSlice<bf16>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::mla_decode_lse_bf16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f32")]

pub fn mla_decode_splitk_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    query_pe: &impl DeviceSlice<f32>,
    key_value: &impl DeviceSlice<f32>,
    key_pe: &impl DeviceSlice<f32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::mla_decode_splitk_f32(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f16")]

pub fn mla_decode_splitk_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f16>,
    query_pe: &impl DeviceSlice<f16>,
    key_value: &impl DeviceSlice<f16>,
    key_pe: &impl DeviceSlice<f16>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::mla_decode_splitk_f16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-bf16")]

pub fn mla_decode_splitk_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<bf16>,
    query_pe: &impl DeviceSlice<bf16>,
    key_value: &impl DeviceSlice<bf16>,
    key_pe: &impl DeviceSlice<bf16>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::mla_decode_splitk_bf16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f32")]
pub fn fmha_prefill_lse_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_prefill_lse_f32(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f32")]
pub fn fmha_prefill_lse_mma_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_prefill_lse_mma_f32(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f32")]
pub fn fmha_prefill_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    config.validate_output_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_prefill_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn attention_sink_prefill_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    sinks: &impl DeviceSlice<f32>,
    config: AttentionSinkPrefillConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len(), sinks.len())?;
    let attention = config.attention;
    let stream = borrowed_stream(stream)?;
    cutile::attention::attention_sink_prefill_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn softcapped_window_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    config: SoftcappedWindowAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let attention = config.attention;
    let stream = borrowed_stream(stream)?;
    cutile::attention::softcapped_window_attention_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f16")]
pub fn softcapped_window_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    config: SoftcappedWindowAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let attention = config.attention;
    let stream = borrowed_stream(stream)?;
    cutile::attention::softcapped_window_attention_f16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-bf16")]
pub fn softcapped_window_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    query: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
    config: SoftcappedWindowAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let attention = config.attention;
    let stream = borrowed_stream(stream)?;
    cutile::attention::softcapped_window_attention_bf16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn fmha_decode_lse_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    config: FmhaDecodeConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_decode_lse_f32(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f32")]
pub fn softcapped_window_decode_lse_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    config: SoftcappedWindowDecodeConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let decode = config.decode;
    let stream = borrowed_stream(stream)?;
    cutile::attention::softcapped_window_decode_lse_f32(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f16")]
pub fn softcapped_window_decode_lse_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    config: SoftcappedWindowDecodeConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let decode = config.decode;
    let stream = borrowed_stream(stream)?;
    cutile::attention::softcapped_window_decode_lse_f16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-bf16")]
pub fn softcapped_window_decode_lse_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
    config: SoftcappedWindowDecodeConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let decode = config.decode;
    let stream = borrowed_stream(stream)?;
    cutile::attention::softcapped_window_decode_lse_bf16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f16")]
pub fn fmha_prefill_lse_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_prefill_lse_f16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-bf16")]
pub fn fmha_prefill_lse_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_prefill_lse_bf16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f16")]
pub fn fmha_prefill_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    config.validate_output_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_prefill_f16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-bf16")]
pub fn fmha_prefill_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    query: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
    config: FmhaPrefillConfig,
) -> Result<()> {
    config.validate_output_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_prefill_bf16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f16")]
pub fn attention_sink_prefill_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    sinks: &impl DeviceSlice<f32>,
    config: AttentionSinkPrefillConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len(), sinks.len())?;
    let attention = config.attention;
    let stream = borrowed_stream(stream)?;
    cutile::attention::attention_sink_prefill_f16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-bf16")]
pub fn attention_sink_prefill_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    query: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
    sinks: &impl DeviceSlice<f32>,
    config: AttentionSinkPrefillConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len(), sinks.len())?;
    let attention = config.attention;
    let stream = borrowed_stream(stream)?;
    cutile::attention::attention_sink_prefill_bf16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f16")]
pub fn fmha_decode_lse_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    config: FmhaDecodeConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_decode_lse_f16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f16")]
pub fn fmha_decode_lse_dynpos_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    position_start: &impl DeviceSlice<u32>,
    config: FmhaDecodeConfig,
) -> Result<()> {
    ensure_len(position_start.len(), 1)?;
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_decode_lse_dynpos_f16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-bf16")]
pub fn fmha_decode_lse_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
    config: FmhaDecodeConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_decode_lse_bf16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f32")]
pub fn splitk_reduce_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    attn_splitk: &impl DeviceSlice<f32>,
    lse_splitk: &impl DeviceSlice<f32>,
    config: SplitKReduceConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), attn_splitk.len(), lse_splitk.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::splitk_reduce_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn fmha_decode_splitk_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    config: FmhaDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_decode_splitk_f32(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f32")]
pub fn softcapped_window_decode_splitk_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    config: SoftcappedWindowDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let decode = config.decode;
    let stream = borrowed_stream(stream)?;
    cutile::attention::softcapped_window_decode_splitk_f32(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f16")]
pub fn softcapped_window_decode_splitk_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    config: SoftcappedWindowDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let decode = config.decode;
    let stream = borrowed_stream(stream)?;
    cutile::attention::softcapped_window_decode_splitk_f16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-bf16")]
pub fn softcapped_window_decode_splitk_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
    config: SoftcappedWindowDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let decode = config.decode;
    let stream = borrowed_stream(stream)?;
    cutile::attention::softcapped_window_decode_splitk_bf16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f16")]
pub fn fmha_decode_splitk_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    config: FmhaDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_decode_splitk_f16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-bf16")]
pub fn fmha_decode_splitk_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
    config: FmhaDecodeSplitKConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fmha_decode_splitk_bf16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f32")]
pub fn attention_sink_decode_splitk_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    sinks: &impl DeviceSlice<f32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::attention_sink_decode_splitk_f32(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f16")]
pub fn attention_sink_decode_splitk_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    sinks: &impl DeviceSlice<f32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::attention_sink_decode_splitk_f16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-bf16")]
pub fn attention_sink_decode_splitk_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
    sinks: &impl DeviceSlice<f32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::attention_sink_decode_splitk_bf16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f16")]
/// Causal left-window attention.
///
/// See [`sliding_window_attention_f32`] for mask and grouped-query semantics.
pub fn sliding_window_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    config: SlidingWindowAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::sliding_window_attention_f16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-bf16")]
/// Causal left-window attention.
///
/// See [`sliding_window_attention_f32`] for mask and grouped-query semantics.
pub fn sliding_window_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    query: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
    config: SlidingWindowAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::sliding_window_attention_bf16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f16")]
pub fn swa_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    config: SwaAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::swa_attention_f16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-bf16")]
pub fn swa_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    query: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
    config: SwaAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::swa_attention_bf16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn fused_neighborhood_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    config: FusedNeighborhoodAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fused_neighborhood_attention_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn heavily_compressed_attention_compress_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    hidden: &impl DeviceSlice<f32>,
    weight_kv: &impl DeviceSlice<f32>,
    weight_z: &impl DeviceSlice<f32>,
    bias: &impl DeviceSlice<f32>,
    config: HeavilyCompressedAttentionConfig,
) -> Result<()> {
    config.validate_compress_lengths(
        out.len(),
        hidden.len(),
        weight_kv.len(),
        weight_z.len(),
        bias.len(),
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::heavily_compressed_attention_compress_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn compressed_sparse_attention_compress_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    hidden: &impl DeviceSlice<f32>,
    weight_a_kv: &impl DeviceSlice<f32>,
    weight_b_kv: &impl DeviceSlice<f32>,
    weight_a_z: &impl DeviceSlice<f32>,
    weight_b_z: &impl DeviceSlice<f32>,
    bias_a: &impl DeviceSlice<f32>,
    bias_b: &impl DeviceSlice<f32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::compressed_sparse_attention_compress_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn compressed_sparse_attention_lightning_indexer_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    indexer_query: &impl DeviceSlice<f32>,
    indexer_key: &impl DeviceSlice<f32>,
    indexer_weight: &impl DeviceSlice<f32>,
    config: CompressedSparseAttentionLightningIndexerConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        indexer_query.len(),
        indexer_key.len(),
        indexer_weight.len(),
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::compressed_sparse_attention_lightning_indexer_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn compressed_sparse_attention_topk_selector_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    selected_indices: &mut impl DeviceSliceMut<i32>,
    scores: &impl DeviceSlice<f32>,
    compressed_kv: &impl DeviceSlice<f32>,
    query_positions: &impl DeviceSlice<i32>,
    config: CompressedSparseAttentionTopkSelectorConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        selected_indices.len(),
        scores.len(),
        compressed_kv.len(),
        query_positions.len(),
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::compressed_sparse_attention_topk_selector_f32(
        &stream,
        output_pointer(out),
        output_pointer(selected_indices),
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

#[cfg(feature = "dtype-f32")]
pub fn compressed_sparse_attention_shared_mqa_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    kv_entries: &impl DeviceSlice<f32>,
    valid_mask: &impl DeviceSlice<i32>,
    sink: &impl DeviceSlice<f32>,
    config: CompressedSparseAttentionSharedMqaConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        query.len(),
        kv_entries.len(),
        valid_mask.len(),
        sink.len(),
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::compressed_sparse_attention_shared_mqa_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn minimax_sparse_attention_block_max_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    scores: &impl DeviceSlice<f32>,
    config: MiniMaxSparseAttentionBlockMaxConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), scores.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::minimax_sparse_attention_block_max_f32(
        &stream,
        output_pointer(out),
        input_pointer(scores),
        config.batch,
        config.rows,
        config.key_len,
        config.block_size,
    )
}

pub fn minimax_sparse_attention_selected_token_positions_i32(
    stream: &Stream,
    positions: &mut impl DeviceSliceMut<i32>,
    valid_mask: &mut impl DeviceSliceMut<i32>,
    selected_blocks: &impl DeviceSlice<i32>,
    query_positions: &impl DeviceSlice<i32>,
    config: MiniMaxSparseAttentionSelectedTokenPositionsConfig,
) -> Result<()> {
    config.validate_lengths(
        positions.len(),
        valid_mask.len(),
        selected_blocks.len(),
        query_positions.len(),
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::minimax_sparse_attention_selected_token_positions_i32(
        &stream,
        output_pointer(positions),
        output_pointer(valid_mask),
        input_pointer(selected_blocks),
        input_pointer(query_positions),
        config.batch,
        config.rows,
        config.selected_blocks,
        config.block_size,
        config.seq_len,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn minimax_sparse_attention_select_topk_blocks_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<i32>,
    block_scores: &impl DeviceSlice<f32>,
    local_blocks: &impl DeviceSlice<i32>,
    config: MiniMaxSparseAttentionSelectTopkBlocksConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), block_scores.len(), local_blocks.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::minimax_sparse_attention_select_topk_blocks_f32(
        &stream,
        output_pointer(out),
        input_pointer(block_scores),
        input_pointer(local_blocks),
        config.batch,
        config.rows,
        config.blocks,
        config.top_k,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn minimax_sparse_attention_gathered_gqa_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    positions: &impl DeviceSlice<i32>,
    valid_mask: &impl DeviceSlice<i32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::minimax_sparse_attention_gathered_gqa_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn heavily_compressed_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    compressed_kv: &impl DeviceSlice<f32>,
    weight_group: &impl DeviceSlice<f32>,
    weight_final: &impl DeviceSlice<f32>,
    config: HeavilyCompressedAttentionConfig,
) -> Result<()> {
    config.validate_attention_lengths(
        out.len(),
        query.len(),
        compressed_kv.len(),
        weight_group.len(),
        weight_final.len(),
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::heavily_compressed_attention_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f16")]
pub fn fused_neighborhood_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    config: FusedNeighborhoodAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fused_neighborhood_attention_f16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-bf16")]
pub fn fused_neighborhood_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    query: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
    config: FusedNeighborhoodAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::attention::fused_neighborhood_attention_bf16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn multi_token_attention_f32<B>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    scores: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    bias: Option<&B>,
    config: MultiTokenAttentionConfig,
) -> Result<()>
where
    B: DeviceSlice<f32>,
{
    config.validate_lengths(
        out.len(),
        scores.len(),
        weight.len(),
        bias.map(|bias| bias.len()),
    )?;
    let stream = borrowed_stream(stream)?;
    let scores_pointer = input_pointer(scores);
    cutile::attention::multi_token_attention_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn multi_token_attention_sparse_f32<B>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    scores: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    bias: Option<&B>,
    config: MultiTokenAttentionConfig,
) -> Result<()>
where
    B: DeviceSlice<f32>,
{
    config.validate_lengths(
        out.len(),
        scores.len(),
        weight.len(),
        bias.map(|bias| bias.len()),
    )?;
    let stream = borrowed_stream(stream)?;
    let scores_pointer = input_pointer(scores);
    cutile::attention::multi_token_attention_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f16")]
pub fn multi_token_attention_f16<B>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    scores: &impl DeviceSlice<f16>,
    weight: &impl DeviceSlice<f16>,
    bias: Option<&B>,
    config: MultiTokenAttentionConfig,
) -> Result<()>
where
    B: DeviceSlice<f16>,
{
    config.validate_lengths(
        out.len(),
        scores.len(),
        weight.len(),
        bias.map(|bias| bias.len()),
    )?;
    let stream = borrowed_stream(stream)?;
    let scores_pointer = input_pointer(scores);
    cutile::attention::multi_token_attention_f16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f16")]
pub fn multi_token_attention_sparse_f16<B>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    scores: &impl DeviceSlice<f16>,
    weight: &impl DeviceSlice<f16>,
    bias: Option<&B>,
    config: MultiTokenAttentionConfig,
) -> Result<()>
where
    B: DeviceSlice<f16>,
{
    config.validate_lengths(
        out.len(),
        scores.len(),
        weight.len(),
        bias.map(|bias| bias.len()),
    )?;
    let stream = borrowed_stream(stream)?;
    let scores_pointer = input_pointer(scores);
    cutile::attention::multi_token_attention_sparse_f16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-bf16")]
pub fn multi_token_attention_bf16<B>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    scores: &impl DeviceSlice<bf16>,
    weight: &impl DeviceSlice<bf16>,
    bias: Option<&B>,
    config: MultiTokenAttentionConfig,
) -> Result<()>
where
    B: DeviceSlice<bf16>,
{
    config.validate_lengths(
        out.len(),
        scores.len(),
        weight.len(),
        bias.map(|bias| bias.len()),
    )?;
    let stream = borrowed_stream(stream)?;
    let scores_pointer = input_pointer(scores);
    cutile::attention::multi_token_attention_bf16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-bf16")]
pub fn multi_token_attention_sparse_bf16<B>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    scores: &impl DeviceSlice<bf16>,
    weight: &impl DeviceSlice<bf16>,
    bias: Option<&B>,
    config: MultiTokenAttentionConfig,
) -> Result<()>
where
    B: DeviceSlice<bf16>,
{
    config.validate_lengths(
        out.len(),
        scores.len(),
        weight.len(),
        bias.map(|bias| bias.len()),
    )?;
    let stream = borrowed_stream(stream)?;
    let scores_pointer = input_pointer(scores);
    cutile::attention::multi_token_attention_sparse_bf16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn sliding_window_attention_packed_qkv_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    qkv: &impl DeviceSlice<f32>,
    config: PackedQkvAttentionConfig,
) -> Result<()> {
    validate_packed_qkv_attention_config(qkv.len(), config)?;
    let query = unsafe {
        DeviceView::from_raw_parts(
            qkv.as_device_ptr().add(config.query_offset),
            qkv.len() - config.query_offset,
        )
    };
    let key = unsafe {
        DeviceView::from_raw_parts(
            qkv.as_device_ptr().add(config.key_offset),
            qkv.len() - config.key_offset,
        )
    };
    let value = unsafe {
        DeviceView::from_raw_parts(
            qkv.as_device_ptr().add(config.value_offset),
            qkv.len() - config.value_offset,
        )
    };
    sliding_window_attention_f32(stream, out, &query, &key, &value, config.attention)
}

#[cfg(feature = "dtype-f32")]
pub fn sliding_window_attention_paged_kv_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key_cache: &impl DeviceSlice<f32>,
    value_cache: &impl DeviceSlice<f32>,
    block_table: &impl DeviceSlice<u32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::sliding_window_attention_paged_kv_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn paged_kv_decode_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key_cache: &impl DeviceSlice<f32>,
    value_cache: &impl DeviceSlice<f32>,
    actual_seq_lens: &impl DeviceSlice<i32>,
    block_table: &impl DeviceSlice<u32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::paged_kv_decode_attention_f32(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f16")]
pub fn paged_kv_decode_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    query: &impl DeviceSlice<f16>,
    key_cache: &impl DeviceSlice<f16>,
    value_cache: &impl DeviceSlice<f16>,
    actual_seq_lens: &impl DeviceSlice<i32>,
    block_table: &impl DeviceSlice<u32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::paged_kv_decode_attention_f16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-bf16")]
pub fn paged_kv_decode_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    query: &impl DeviceSlice<bf16>,
    key_cache: &impl DeviceSlice<bf16>,
    value_cache: &impl DeviceSlice<bf16>,
    actual_seq_lens: &impl DeviceSlice<i32>,
    block_table: &impl DeviceSlice<u32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::paged_kv_decode_attention_bf16(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]

pub fn paged_kv_prefill_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key_cache: &impl DeviceSlice<f32>,
    value_cache: &impl DeviceSlice<f32>,
    actual_seq_lens_q: &impl DeviceSlice<i32>,
    actual_seq_lens_kv: &impl DeviceSlice<i32>,
    actual_seq_offsets: &impl DeviceSlice<i32>,
    block_table: &impl DeviceSlice<u32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::paged_kv_prefill_attention_f32(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f16")]

pub fn paged_kv_prefill_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f16>,
    key_cache: &impl DeviceSlice<f16>,
    value_cache: &impl DeviceSlice<f16>,
    actual_seq_lens_q: &impl DeviceSlice<i32>,
    actual_seq_lens_kv: &impl DeviceSlice<i32>,
    actual_seq_offsets: &impl DeviceSlice<i32>,
    block_table: &impl DeviceSlice<u32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::paged_kv_prefill_attention_f16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-bf16")]

pub fn paged_kv_prefill_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<bf16>,
    key_cache: &impl DeviceSlice<bf16>,
    value_cache: &impl DeviceSlice<bf16>,
    actual_seq_lens_q: &impl DeviceSlice<i32>,
    actual_seq_lens_kv: &impl DeviceSlice<i32>,
    actual_seq_offsets: &impl DeviceSlice<i32>,
    block_table: &impl DeviceSlice<u32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::paged_kv_prefill_attention_bf16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

#[cfg(feature = "dtype-f32")]

pub fn ragged_kv_prefill_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f32>,
    key: &impl DeviceSlice<f32>,
    value: &impl DeviceSlice<f32>,
    actual_seq_lens_q: &impl DeviceSlice<i32>,
    actual_seq_lens_kv: &impl DeviceSlice<i32>,
    actual_seq_offsets: &impl DeviceSlice<i32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::ragged_kv_prefill_attention_f32(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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
#[cfg(feature = "dtype-f16")]

pub fn ragged_kv_prefill_attention_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<f16>,
    key: &impl DeviceSlice<f16>,
    value: &impl DeviceSlice<f16>,
    actual_seq_lens_q: &impl DeviceSlice<i32>,
    actual_seq_lens_kv: &impl DeviceSlice<i32>,
    actual_seq_offsets: &impl DeviceSlice<i32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::ragged_kv_prefill_attention_f16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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
#[cfg(feature = "dtype-bf16")]

pub fn ragged_kv_prefill_attention_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    lse: &mut impl DeviceSliceMut<f32>,
    query: &impl DeviceSlice<bf16>,
    key: &impl DeviceSlice<bf16>,
    value: &impl DeviceSlice<bf16>,
    actual_seq_lens_q: &impl DeviceSlice<i32>,
    actual_seq_lens_kv: &impl DeviceSlice<i32>,
    actual_seq_offsets: &impl DeviceSlice<i32>,
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
    let stream = borrowed_stream(stream)?;
    cutile::attention::ragged_kv_prefill_attention_bf16(
        &stream,
        output_pointer(out),
        output_pointer(lse),
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

fn validate_packed_qkv_attention_config(
    qkv_len: usize,
    config: PackedQkvAttentionConfig,
) -> Result<()> {
    if config.query_offset > qkv_len || config.key_offset > qkv_len || config.value_offset > qkv_len
    {
        return Err(Error::LengthMismatch);
    }
    config.attention.validate_lengths(
        usize::MAX,
        qkv_len - config.query_offset,
        qkv_len - config.key_offset,
        qkv_len - config.value_offset,
    )
}

fn validate_paged_kv_attention_config(
    out_len: usize,
    query_len: usize,
    key_cache_len: usize,
    value_cache_len: usize,
    block_table_len: usize,
    config: PagedKvAttentionConfig,
) -> Result<()> {
    let attention = config.attention;
    validate_sliding_window_attention_config(attention)?;
    if config.block_size == 0
        || config.physical_blocks == 0
        || config.block_table_batch_stride == 0
        || config.key_cache_block_stride == 0
        || config.value_cache_block_stride == 0
    {
        return Err(Error::InvalidLength);
    }
    let key_blocks = attention
        .key_len
        .checked_add(config.block_size - 1)
        .ok_or(Error::SizeOverflow)?
        / config.block_size;
    ensure_len_at_least(
        block_table_len,
        batched_reach(attention.batch, key_blocks, config.block_table_batch_stride)?,
    )?;
    let query_item_reach = projected_layout_reach(
        attention.query_len,
        attention.heads,
        attention.head_dim,
        attention.query_sequence_stride,
        attention.query_head_stride,
    )?;
    let output_item_reach = projected_layout_reach(
        attention.query_len,
        attention.heads,
        attention.head_dim,
        attention.output_sequence_stride,
        attention.output_head_stride,
    )?;
    let key_block_reach = projected_layout_reach(
        config.block_size,
        attention.kv_heads,
        attention.head_dim,
        attention.key_sequence_stride,
        attention.key_head_stride,
    )?;
    let value_block_reach = projected_layout_reach(
        config.block_size,
        attention.kv_heads,
        attention.head_dim,
        attention.value_sequence_stride,
        attention.value_head_stride,
    )?;
    ensure_len_at_least(
        query_len,
        batched_reach(
            attention.batch,
            query_item_reach,
            attention.query_batch_stride,
        )?,
    )?;
    ensure_len_at_least(
        out_len,
        batched_reach(
            attention.batch,
            output_item_reach,
            attention.output_batch_stride,
        )?,
    )?;
    ensure_len_at_least(
        key_cache_len,
        batched_reach(
            config.physical_blocks,
            key_block_reach,
            config.key_cache_block_stride,
        )?,
    )?;
    ensure_len_at_least(
        value_cache_len,
        batched_reach(
            config.physical_blocks,
            value_block_reach,
            config.value_cache_block_stride,
        )?,
    )
}

fn validate_paged_kv_decode_attention_config(config: PagedKvDecodeAttentionConfig) -> Result<()> {
    if config.batch == 0
        || config.heads == 0
        || config.kv_heads == 0
        || config.head_dim == 0
        || config.block_size == 0
        || config.physical_blocks == 0
        || config.block_table_batch_stride == 0
        || !config.heads.is_multiple_of(config.kv_heads)
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.heads)?;
    checked_i32_value(config.kv_heads)?;
    checked_i32_value(config.head_dim)?;
    checked_i32_value(config.block_size)?;
    checked_i32_value(config.physical_blocks)?;
    checked_i32_value(config.block_table_batch_stride)?;
    Ok(())
}

fn validate_paged_kv_prefill_attention_config(config: PagedKvPrefillAttentionConfig) -> Result<()> {
    if config.batch == 0
        || config.total_query_tokens == 0
        || config.heads == 0
        || config.kv_heads == 0
        || config.head_dim == 0
        || config.block_size == 0
        || config.physical_blocks == 0
        || config.block_table_batch_stride == 0
        || !config.heads.is_multiple_of(config.kv_heads)
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.total_query_tokens)?;
    checked_i32_value(config.heads)?;
    checked_i32_value(config.kv_heads)?;
    checked_i32_value(config.head_dim)?;
    checked_i32_value(config.block_size)?;
    checked_i32_value(config.physical_blocks)?;
    checked_i32_value(config.block_table_batch_stride)?;
    Ok(())
}

fn validate_ragged_kv_prefill_attention_config(
    config: RaggedKvPrefillAttentionConfig,
) -> Result<()> {
    if config.batch == 0
        || config.total_query_tokens == 0
        || config.total_kv_tokens == 0
        || config.heads == 0
        || config.kv_heads == 0
        || config.head_dim == 0
        || !config.heads.is_multiple_of(config.kv_heads)
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.total_query_tokens)?;
    checked_i32_value(config.total_kv_tokens)?;
    checked_i32_value(config.heads)?;
    checked_i32_value(config.kv_heads)?;
    checked_i32_value(config.head_dim)?;
    Ok(())
}

fn validate_sliding_window_attention_config(config: SlidingWindowAttentionConfig) -> Result<()> {
    if config.batch == 0
        || config.query_len == 0
        || config.key_len == 0
        || config.heads == 0
        || config.kv_heads == 0
        || config.head_dim == 0
        || config.window == 0
        || config.query_batch_stride == 0
        || config.key_batch_stride == 0
        || config.value_batch_stride == 0
        || config.output_batch_stride == 0
        || config.query_sequence_stride == 0
        || config.key_sequence_stride == 0
        || config.value_sequence_stride == 0
        || config.output_sequence_stride == 0
        || config.query_head_stride == 0
        || config.key_head_stride == 0
        || config.value_head_stride == 0
        || config.output_head_stride == 0
        || !config.scale.is_finite()
        || !config.output_scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    if !config.heads.is_multiple_of(config.kv_heads) {
        return Err(Error::InvalidLength);
    }
    config
        .query_start
        .checked_add(config.query_len)
        .ok_or(Error::SizeOverflow)?;
    config
        .key_start
        .checked_add(config.key_len)
        .ok_or(Error::SizeOverflow)?;
    Ok(())
}

fn validate_swa_attention_config(config: SwaAttentionConfig) -> Result<()> {
    if config.batch == 0
        || config.query_len == 0
        || config.key_len == 0
        || config.heads == 0
        || config.head_dim == 0
        || !config.scale.is_finite()
        || !config.causal
    {
        return Err(Error::InvalidLength);
    }
    if config.window_size > 0 {
        checked_i32_value(config.window_size)?;
    }
    checked_i32_value(config.effective_window())?;
    Ok(())
}

fn validate_fused_neighborhood_attention_config(
    config: FusedNeighborhoodAttentionConfig,
) -> Result<()> {
    if config.batch == 0
        || config.seq_len == 0
        || config.heads == 0
        || config.head_dim == 0
        || config.kernel_size == 0
        || config.dilation == 0
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.seq_len)?;
    checked_i32_value(config.heads)?;
    checked_i32_value(config.head_dim)?;
    checked_i32_value(config.kernel_size)?;
    checked_i32_value(config.dilation)?;
    Ok(())
}

fn validate_heavily_compressed_attention_config(
    config: HeavilyCompressedAttentionConfig,
) -> Result<()> {
    if config.batch == 0
        || config.seq_len == 0
        || config.hidden_dim == 0
        || config.heads == 0
        || config.head_dim == 0
        || config.compression_block == 0
        || config.groups == 0
        || config.group_dim == 0
        || !config.heads.is_multiple_of(config.groups)
        || config.seq_len / config.compression_block == 0
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.seq_len)?;
    checked_i32_value(config.hidden_dim)?;
    checked_i32_value(config.heads)?;
    checked_i32_value(config.head_dim)?;
    checked_i32_value(config.compression_block)?;
    checked_i32_value(config.groups)?;
    checked_i32_value(config.group_dim)?;
    checked_i32_value(config.seq_len / config.compression_block)?;
    Ok(())
}

fn validate_compressed_sparse_attention_compress_config(
    config: CompressedSparseAttentionCompressConfig,
) -> Result<()> {
    if config.batch == 0
        || config.seq_len == 0
        || config.hidden_dim == 0
        || config.head_dim == 0
        || config.compression_block == 0
        || config.seq_len / config.compression_block == 0
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.seq_len)?;
    checked_i32_value(config.hidden_dim)?;
    checked_i32_value(config.head_dim)?;
    checked_i32_value(config.compression_block)?;
    checked_i32_value(config.seq_len / config.compression_block)?;
    Ok(())
}

fn validate_compressed_sparse_attention_lightning_indexer_config(
    config: CompressedSparseAttentionLightningIndexerConfig,
) -> Result<()> {
    if config.batch == 0
        || config.query_len == 0
        || config.blocks == 0
        || config.index_heads == 0
        || config.index_dim == 0
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.query_len)?;
    checked_i32_value(config.blocks)?;
    checked_i32_value(config.index_heads)?;
    checked_i32_value(config.index_dim)?;
    Ok(())
}

fn validate_compressed_sparse_attention_topk_selector_config(
    config: CompressedSparseAttentionTopkSelectorConfig,
) -> Result<()> {
    if config.batch == 0
        || config.query_len == 0
        || config.blocks == 0
        || config.head_dim == 0
        || config.top_k == 0
        || config.compression_block == 0
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.query_len)?;
    checked_i32_value(config.blocks)?;
    checked_i32_value(config.head_dim)?;
    checked_i32_value(config.top_k)?;
    checked_i32_value(config.compression_block)?;
    Ok(())
}

fn validate_compressed_sparse_attention_shared_mqa_config(
    config: CompressedSparseAttentionSharedMqaConfig,
) -> Result<()> {
    if config.batch == 0
        || config.query_len == 0
        || config.heads == 0
        || config.head_dim == 0
        || config.kv_len == 0
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.query_len)?;
    checked_i32_value(config.heads)?;
    checked_i32_value(config.head_dim)?;
    checked_i32_value(config.kv_len)?;
    Ok(())
}

fn validate_minimax_sparse_attention_block_max_config(
    config: MiniMaxSparseAttentionBlockMaxConfig,
) -> Result<()> {
    if config.batch == 0 || config.rows == 0 || config.key_len == 0 || config.block_size == 0 {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.rows)?;
    checked_i32_value(config.key_len)?;
    checked_i32_value(config.block_size)?;
    checked_i32_value(config.key_len.div_ceil(config.block_size))?;
    Ok(())
}

fn validate_minimax_sparse_attention_selected_token_positions_config(
    config: MiniMaxSparseAttentionSelectedTokenPositionsConfig,
) -> Result<()> {
    if config.batch == 0
        || config.rows == 0
        || config.selected_blocks == 0
        || config.block_size == 0
        || config.seq_len == 0
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.rows)?;
    checked_i32_value(config.selected_blocks)?;
    checked_i32_value(config.block_size)?;
    checked_i32_value(config.seq_len)?;
    checked_i32_value(checked_element_count(
        config.selected_blocks,
        config.block_size,
    )?)?;
    Ok(())
}

fn validate_minimax_sparse_attention_select_topk_blocks_config(
    config: MiniMaxSparseAttentionSelectTopkBlocksConfig,
) -> Result<()> {
    if config.batch == 0 || config.rows == 0 || config.blocks == 0 || config.top_k == 0 {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.rows)?;
    checked_i32_value(config.blocks)?;
    checked_i32_value(config.top_k)?;
    Ok(())
}

fn validate_minimax_sparse_attention_gathered_gqa_config(
    config: MiniMaxSparseAttentionGatheredGqaConfig,
) -> Result<()> {
    if config.batch == 0
        || config.query_len == 0
        || config.key_len == 0
        || config.heads == 0
        || config.kv_heads == 0
        || config.head_dim == 0
        || config.selected_keys == 0
        || !config.heads.is_multiple_of(config.kv_heads)
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.query_len)?;
    checked_i32_value(config.key_len)?;
    checked_i32_value(config.heads)?;
    checked_i32_value(config.kv_heads)?;
    checked_i32_value(config.head_dim)?;
    checked_i32_value(config.selected_keys)?;
    Ok(())
}

fn validate_multi_token_attention_config(config: MultiTokenAttentionConfig) -> Result<()> {
    if config.batch == 0
        || config.channels_in == 0
        || config.channels_out == 0
        || config.seq_len == 0
        || config.kernel_h == 0
        || config.kernel_w == 0
        || config.stride_h == 0
        || config.stride_w == 0
        || config.dilation_h == 0
        || config.dilation_w == 0
        || config.groups == 0
        || !config.channels_in.is_multiple_of(config.groups)
        || !config.channels_out.is_multiple_of(config.groups)
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.channels_in)?;
    checked_i32_value(config.channels_out)?;
    checked_i32_value(config.seq_len)?;
    checked_i32_value(config.kernel_h)?;
    checked_i32_value(config.kernel_w)?;
    checked_i32_value(config.stride_h)?;
    checked_i32_value(config.stride_w)?;
    checked_i32_value(config.padding_h)?;
    checked_i32_value(config.padding_w)?;
    checked_i32_value(config.dilation_h)?;
    checked_i32_value(config.dilation_w)?;
    checked_i32_value(config.groups)?;
    Ok(())
}

fn conv2d_output_len(
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

fn validate_fmha_prefill_config(config: FmhaPrefillConfig) -> Result<()> {
    if config.batch == 0
        || config.query_len == 0
        || config.key_len == 0
        || config.heads == 0
        || config.kv_heads == 0
        || config.head_dim == 0
        || config.query_batch_stride == 0
        || config.key_batch_stride == 0
        || config.value_batch_stride == 0
        || config.output_batch_stride == 0
        || config.query_sequence_stride == 0
        || config.key_sequence_stride == 0
        || config.value_sequence_stride == 0
        || config.output_sequence_stride == 0
        || config.query_head_stride == 0
        || config.key_head_stride == 0
        || config.value_head_stride == 0
        || config.output_head_stride == 0
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    if !config.heads.is_multiple_of(config.kv_heads) {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_mla_prefill_config(config: MlaPrefillConfig) -> Result<()> {
    if config.batch == 0
        || config.query_len == 0
        || config.key_len == 0
        || config.heads == 0
        || config.kv_heads == 0
        || config.head_dim == 0
        || config.pe_dim == 0
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    if !config.heads.is_multiple_of(config.kv_heads) {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_softcapped_window_attention_config(
    config: SoftcappedWindowAttentionConfig,
) -> Result<()> {
    validate_fmha_prefill_config(config.attention)?;
    if let Some(soft_cap) = config.soft_cap
        && (!soft_cap.is_finite() || soft_cap <= 0.0)
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.window_size)?;
    Ok(())
}

fn validate_softcapped_window_decode_config(config: SoftcappedWindowDecodeConfig) -> Result<()> {
    validate_fmha_decode_config(config.decode)?;
    if let Some(soft_cap) = config.soft_cap
        && (!soft_cap.is_finite() || soft_cap <= 0.0)
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.window_size)?;
    Ok(())
}

fn validate_softcapped_window_decode_splitk_config(
    config: SoftcappedWindowDecodeSplitKConfig,
) -> Result<()> {
    validate_fmha_decode_splitk_config(config.decode)?;
    if let Some(soft_cap) = config.soft_cap
        && (!soft_cap.is_finite() || soft_cap <= 0.0)
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.window_size)?;
    Ok(())
}

fn validate_sparse_mla_prefill_config(config: SparseMlaPrefillConfig) -> Result<()> {
    if config.batch == 0
        || config.query_len == 0
        || config.key_len == 0
        || config.heads == 0
        || config.kv_heads == 0
        || config.head_dim == 0
        || config.pe_dim == 0
        || config.topk == 0
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    if !config.heads.is_multiple_of(config.kv_heads) {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_mla_decode_config(config: MlaDecodeConfig) -> Result<()> {
    if config.batch == 0
        || config.key_len == 0
        || config.heads == 0
        || config.head_dim == 0
        || config.pe_dim == 0
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_paged_mla_decode_attention_config(config: PagedMlaDecodeAttentionConfig) -> Result<()> {
    if config.batch == 0
        || config.heads == 0
        || config.head_dim == 0
        || config.pe_dim == 0
        || config.block_size == 0
        || config.physical_blocks == 0
        || config.block_table_batch_stride == 0
        || !config.scale.is_finite()
        || !config.output_scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(config.batch)?;
    checked_i32_value(config.heads)?;
    checked_i32_value(config.head_dim)?;
    checked_i32_value(config.pe_dim)?;
    checked_i32_value(config.block_size)?;
    checked_i32_value(config.physical_blocks)?;
    checked_i32_value(config.block_table_batch_stride)?;
    Ok(())
}

fn validate_mla_decode_splitk_config(config: MlaDecodeSplitKConfig) -> Result<()> {
    if config.batch == 0
        || config.key_len == 0
        || config.heads == 0
        || config.splits == 0
        || config.kv_len_per_split == 0
        || config.head_dim == 0
        || config.pe_dim == 0
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_fmha_decode_config(config: FmhaDecodeConfig) -> Result<()> {
    if config.batch == 0
        || config.key_len == 0
        || config.heads == 0
        || config.kv_heads == 0
        || config.head_dim == 0
        || config.query_batch_stride == 0
        || config.key_batch_stride == 0
        || config.value_batch_stride == 0
        || config.output_batch_stride == 0
        || config.query_head_stride == 0
        || config.key_sequence_stride == 0
        || config.value_sequence_stride == 0
        || config.output_head_stride == 0
        || config.key_head_stride == 0
        || config.value_head_stride == 0
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    if !config.heads.is_multiple_of(config.kv_heads) {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_splitk_reduce_config(config: SplitKReduceConfig) -> Result<()> {
    if config.batch == 0
        || config.heads == 0
        || config.splits == 0
        || config.head_dim == 0
        || config.attn_batch_stride == 0
        || config.attn_head_stride == 0
        || config.attn_split_stride == 0
        || config.lse_batch_stride == 0
        || config.lse_head_stride == 0
        || config.output_batch_stride == 0
        || config.output_head_stride == 0
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_fmha_decode_splitk_config(config: FmhaDecodeSplitKConfig) -> Result<()> {
    if config.batch == 0
        || config.key_len == 0
        || config.heads == 0
        || config.kv_heads == 0
        || config.splits == 0
        || config.kv_len_per_split == 0
        || config.head_dim == 0
        || config.query_batch_stride == 0
        || config.key_batch_stride == 0
        || config.value_batch_stride == 0
        || config.output_batch_stride == 0
        || config.query_head_stride == 0
        || config.key_sequence_stride == 0
        || config.value_sequence_stride == 0
        || config.output_head_stride == 0
        || config.output_split_stride == 0
        || config.key_head_stride == 0
        || config.value_head_stride == 0
        || config.lse_batch_stride == 0
        || config.lse_head_stride == 0
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    if !config.heads.is_multiple_of(config.kv_heads) {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_attention_sink_decode_splitk_config(
    config: AttentionSinkDecodeSplitKConfig,
) -> Result<()> {
    validate_fmha_decode_splitk_config(config.decode)?;
    config.start_q.checked_add(1).ok_or(Error::SizeOverflow)?;
    Ok(())
}

fn splitk_attn_layout_reach(
    heads: usize,
    splits: usize,
    head_dim: usize,
    head_stride: usize,
    split_stride: usize,
) -> Result<usize> {
    let head_span = heads
        .checked_sub(1)
        .ok_or(Error::SizeOverflow)?
        .checked_mul(head_stride)
        .ok_or(Error::SizeOverflow)?;
    let split_span = splits
        .checked_sub(1)
        .ok_or(Error::SizeOverflow)?
        .checked_mul(split_stride)
        .ok_or(Error::SizeOverflow)?;
    head_span
        .checked_add(split_span)
        .and_then(|span| span.checked_add(head_dim))
        .ok_or(Error::SizeOverflow)
}

fn splitk_lse_layout_reach(heads: usize, splits: usize, head_stride: usize) -> Result<usize> {
    let head_span = heads
        .checked_sub(1)
        .ok_or(Error::SizeOverflow)?
        .checked_mul(head_stride)
        .ok_or(Error::SizeOverflow)?;
    head_span.checked_add(splits).ok_or(Error::SizeOverflow)
}

fn projected_layout_reach(
    sequence_len: usize,
    heads: usize,
    head_dim: usize,
    sequence_stride: usize,
    head_stride: usize,
) -> Result<usize> {
    let sequence_span = sequence_len
        .checked_sub(1)
        .ok_or(Error::SizeOverflow)?
        .checked_mul(sequence_stride)
        .ok_or(Error::SizeOverflow)?;
    let head_span = heads
        .checked_sub(1)
        .ok_or(Error::SizeOverflow)?
        .checked_mul(head_stride)
        .ok_or(Error::SizeOverflow)?;
    sequence_span
        .checked_add(head_span)
        .and_then(|span| span.checked_add(head_dim))
        .ok_or(Error::SizeOverflow)
}

fn batched_reach(batch: usize, item_len: usize, batch_stride: usize) -> Result<usize> {
    let batch_span = batch
        .checked_sub(1)
        .ok_or(Error::SizeOverflow)?
        .checked_mul(batch_stride)
        .ok_or(Error::SizeOverflow)?;
    batch_span.checked_add(item_len).ok_or(Error::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn mla_prefill_validation_accepts_exact_lengths() -> Result<()> {
        let config = MlaPrefillConfig::contiguous(2, 3, 5, 4, 2, 8, 3, 0.5)?;
        config.validate_lengths(192, 192, 72, 160, 60, 160)
    }

    #[test]
    fn mla_prefill_validation_rejects_bad_head_mapping() {
        assert!(matches!(
            MlaPrefillConfig::contiguous(2, 3, 5, 3, 2, 8, 3, 0.5),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn mla_prefill_validation_rejects_short_query_pe() -> Result<()> {
        let config = MlaPrefillConfig::contiguous(2, 3, 5, 4, 2, 8, 3, 0.5)?;
        assert!(matches!(
            config.validate_lengths(192, 192, 71, 160, 60, 160),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn mla_decode_validation_accepts_exact_lengths() -> Result<()> {
        let config = MlaDecodeConfig::contiguous(2, 5, 4, 8, 3, 0.5)?;
        config.validate_lengths(64, 8, 64, 24, 80, 30)
    }

    #[test]
    fn mla_decode_validation_rejects_short_lse() -> Result<()> {
        let config = MlaDecodeConfig::contiguous(2, 5, 4, 8, 3, 0.5)?;
        assert!(matches!(
            config.validate_lengths(64, 7, 64, 24, 80, 30),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn paged_mla_decode_attention_validation_accepts_exact_lengths() -> Result<()> {
        let config = PagedMlaDecodeAttentionConfig::contiguous(2, 4, 8, 3, 3, 5, 4, 0.5, 1.0)?;
        config.validate_lengths(64, 8, 64, 24, 120, 45, 2, 8)
    }

    #[test]
    fn paged_mla_decode_attention_validation_rejects_short_key_pe_cache() -> Result<()> {
        let config = PagedMlaDecodeAttentionConfig::contiguous(2, 4, 8, 3, 3, 5, 4, 0.5, 1.0)?;
        assert!(matches!(
            config.validate_lengths(64, 8, 64, 24, 120, 44, 2, 8),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn mla_decode_splitk_validation_accepts_exact_lengths() -> Result<()> {
        let config = MlaDecodeSplitKConfig::contiguous(2, 5, 4, 3, 2, 8, 3, 0.5)?;
        config.validate_lengths(192, 24, 64, 24, 80, 30)
    }

    #[test]
    fn mla_decode_splitk_validation_rejects_short_split_output() -> Result<()> {
        let config = MlaDecodeSplitKConfig::contiguous(2, 5, 4, 3, 2, 8, 3, 0.5)?;
        assert!(matches!(
            config.validate_lengths(191, 24, 64, 24, 80, 30),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn splitk_reduce_validation_accepts_padded_strides() -> Result<()> {
        let mut config = SplitKReduceConfig::contiguous(2, 4, 3, 8)?;
        config.attn_batch_stride = 104;
        config.attn_head_stride = 25;
        config.lse_batch_stride = 16;
        config.output_batch_stride = 40;
        config.validate_lengths(72, 208, 32)
    }

    #[test]
    fn splitk_reduce_validation_rejects_short_lse() -> Result<()> {
        let config = SplitKReduceConfig::contiguous(2, 4, 3, 8)?;
        assert!(matches!(
            config.validate_lengths(64, 192, 23),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn swa_attention_validation_accepts_exact_lengths_and_zero_window() -> Result<()> {
        let config = SwaAttentionConfig::contiguous(2, 3, 5, 4, 8, 0, 0.5, true)?;
        config.validate_lengths(192, 192, 320, 320)
    }

    #[test]
    fn swa_attention_validation_rejects_short_key() -> Result<()> {
        let config = SwaAttentionConfig::contiguous(2, 3, 5, 4, 8, 2, 0.5, true)?;
        assert!(matches!(
            config.validate_lengths(192, 192, 319, 320),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn swa_attention_validation_rejects_non_causal() {
        assert!(matches!(
            SwaAttentionConfig::contiguous(2, 3, 5, 4, 8, 2, 0.5, false),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn fused_neighborhood_attention_validation_accepts_exact_lengths() -> Result<()> {
        let config = FusedNeighborhoodAttentionConfig::contiguous(2, 3, 4, 8, 3, 1, 0.5)?;
        config.validate_lengths(192, 192, 192, 192)
    }

    #[test]
    fn fused_neighborhood_attention_validation_rejects_short_key() -> Result<()> {
        let config = FusedNeighborhoodAttentionConfig::contiguous(2, 3, 4, 8, 3, 1, 0.5)?;
        assert!(matches!(
            config.validate_lengths(192, 192, 191, 192),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn fused_neighborhood_attention_validation_rejects_zero_dilation() {
        assert!(matches!(
            FusedNeighborhoodAttentionConfig::contiguous(2, 3, 4, 8, 3, 0, 0.5),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn multi_token_attention_validation_accepts_exact_lengths() -> Result<()> {
        let config = MultiTokenAttentionConfig::same_padding_1x1(2, 3, 4, 5)?;
        config.validate_lengths(200, 150, 12, Some(4))
    }

    #[test]
    fn multi_token_attention_validation_rejects_short_bias() -> Result<()> {
        let config = MultiTokenAttentionConfig::same_padding_1x1(2, 3, 4, 5)?;
        assert!(matches!(
            config.validate_lengths(200, 150, 12, Some(3)),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn multi_token_attention_validation_rejects_uneven_groups() {
        let config = MultiTokenAttentionConfig {
            groups: 2,
            ..MultiTokenAttentionConfig::same_padding_1x1(2, 3, 4, 5).unwrap()
        };
        assert!(matches!(
            validate_multi_token_attention_config(config),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn paged_kv_attention_validation_accepts_exact_lengths() -> Result<()> {
        let config = paged_kv_validation_config();
        validate_paged_kv_attention_config(192, 192, 96, 96, 7, config)
    }

    #[test]
    fn paged_kv_attention_validation_rejects_short_block_table() {
        let config = paged_kv_validation_config();
        assert!(matches!(
            validate_paged_kv_attention_config(192, 192, 96, 96, 6, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn paged_kv_attention_validation_rejects_short_key_cache() {
        let config = paged_kv_validation_config();
        assert!(matches!(
            validate_paged_kv_attention_config(192, 192, 95, 96, 7, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn paged_kv_attention_validation_rejects_zero_block_size() {
        let mut config = paged_kv_validation_config();
        config.block_size = 0;
        assert!(matches!(
            validate_paged_kv_attention_config(192, 192, 96, 96, 7, config),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn paged_kv_decode_attention_validation_accepts_exact_lengths() -> Result<()> {
        let config = PagedKvDecodeAttentionConfig::contiguous(2, 4, 2, 8, 3, 5, 4, 0.5)?;
        config.validate_lengths(64, 64, 240, 240, 2, 8)
    }

    #[test]
    fn paged_kv_decode_attention_validation_rejects_short_block_table() -> Result<()> {
        let config = PagedKvDecodeAttentionConfig::contiguous(2, 4, 2, 8, 3, 5, 4, 0.5)?;
        assert!(matches!(
            config.validate_lengths(64, 64, 240, 240, 2, 7),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn paged_kv_decode_attention_validation_rejects_invalid_head_mapping() {
        assert!(matches!(
            PagedKvDecodeAttentionConfig::contiguous(2, 3, 2, 8, 3, 5, 4, 0.5),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn paged_kv_prefill_attention_validation_accepts_exact_lengths() -> Result<()> {
        let config = PagedKvPrefillAttentionConfig::contiguous(2, 7, 4, 2, 8, 3, 5, 4, true, 0.5)?;
        config.validate_lengths(224, 28, 224, 240, 240, 2, 2, 2, 8)
    }

    #[test]
    fn paged_kv_prefill_attention_validation_rejects_short_lse() -> Result<()> {
        let config = PagedKvPrefillAttentionConfig::contiguous(2, 7, 4, 2, 8, 3, 5, 4, true, 0.5)?;
        assert!(matches!(
            config.validate_lengths(224, 27, 224, 240, 240, 2, 2, 2, 8),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn paged_kv_prefill_attention_validation_rejects_invalid_head_mapping() {
        assert!(matches!(
            PagedKvPrefillAttentionConfig::contiguous(2, 7, 3, 2, 8, 3, 5, 4, true, 0.5),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn ragged_kv_prefill_attention_validation_accepts_exact_lengths() -> Result<()> {
        let config = RaggedKvPrefillAttentionConfig::contiguous(2, 7, 8, 4, 2, 8, true, 0.5)?;
        config.validate_lengths(224, 28, 224, 128, 128, 2, 2, 2)
    }

    #[test]
    fn ragged_kv_prefill_attention_validation_rejects_short_key() -> Result<()> {
        let config = RaggedKvPrefillAttentionConfig::contiguous(2, 7, 8, 4, 2, 8, true, 0.5)?;
        assert!(matches!(
            config.validate_lengths(224, 28, 224, 127, 128, 2, 2, 2),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn ragged_kv_prefill_attention_validation_rejects_invalid_head_mapping() {
        assert!(matches!(
            RaggedKvPrefillAttentionConfig::contiguous(2, 7, 8, 3, 2, 8, true, 0.5),
            Err(Error::InvalidLength)
        ));
    }

    fn paged_kv_validation_config() -> PagedKvAttentionConfig {
        let batch = 2usize;
        let query_len = 3usize;
        let key_len = 5usize;
        let heads = 4usize;
        let kv_heads = 2usize;
        let head_dim = 8usize;
        let query_features = heads * head_dim;
        let kv_features = kv_heads * head_dim;
        let block_size = 2usize;
        PagedKvAttentionConfig {
            attention: SlidingWindowAttentionConfig {
                batch,
                query_len,
                key_len,
                heads,
                kv_heads,
                head_dim,
                query_start: 0,
                key_start: 0,
                window: key_len,
                query_batch_stride: query_len * query_features,
                key_batch_stride: key_len * kv_features,
                value_batch_stride: key_len * kv_features,
                output_batch_stride: query_len * query_features,
                query_sequence_stride: query_features,
                key_sequence_stride: kv_features,
                value_sequence_stride: kv_features,
                output_sequence_stride: query_features,
                query_head_stride: head_dim,
                key_head_stride: head_dim,
                value_head_stride: head_dim,
                output_head_stride: head_dim,
                scale: 1.0,
                output_scale: 1.0,
            },
            block_size,
            physical_blocks: 3,
            block_table_batch_stride: 4,
            key_cache_block_stride: block_size * kv_features,
            value_cache_block_stride: block_size * kv_features,
        }
    }
}
