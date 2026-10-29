//! Prefill/decode attention kernels and related attention variants.
//!
//! Destinations and split-K scratch accept pending storage. Attention writes
//! the configured active query/head/feature positions; LSE writes the
//! corresponding query/head slots, or slots for each active split. Selectors
//! write their configured index, position, and validity-mask outputs.
//! Stride gaps, inactive rows, and unused capacity are not initialized by these
//! operations. Promote only ranges whose producers have completed successfully.

#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::{
        autotune::{
            self, DeviceFingerprint, TuneOptions, TuneProblemKey, TuneReport, TuningChoice,
        },
        interop::{input_pointer, write_only_pointer},
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value, ensure_len_at_least},
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AttentionSplitKTuning {
    pub splits: usize,
}

impl AttentionSplitKTuning {
    pub const fn create(splits: usize) -> Self {
        Self { splits }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FmhaDecodeSplitKReduceConfig {
    pub decode: FmhaDecodeConfig,
    pub max_splits: usize,
    pub tuning: TuningChoice<AttentionSplitKTuning>,
}

impl FmhaDecodeSplitKReduceConfig {
    pub fn contiguous(
        batch: usize,
        key_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        max_splits: usize,
        scale: f32,
    ) -> Result<Self> {
        let config = Self {
            decode: FmhaDecodeConfig::contiguous(batch, key_len, heads, kv_heads, head_dim, scale)?,
            max_splits,
            tuning: TuningChoice::Default,
        };
        validate_fmha_decode_splitk_reduce_config(config)?;
        Ok(config)
    }

    pub const fn with_tuning(mut self, tuning: TuningChoice<AttentionSplitKTuning>) -> Self {
        self.tuning = tuning;
        self
    }

    pub const fn explicit_tuning(self, tuning: AttentionSplitKTuning) -> Self {
        self.with_tuning(TuningChoice::Explicit(tuning))
    }

    pub const fn cached_tuning(self) -> Self {
        self.with_tuning(TuningChoice::Cached)
    }

    pub const fn tune(self, options: TuneOptions) -> Self {
        self.with_tuning(TuningChoice::Tune(options))
    }

    pub const fn retune(self, options: TuneOptions) -> Self {
        self.with_tuning(TuningChoice::Retune(options))
    }
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

    #[cfg(any(feature = "dtype_f16", feature = "dtype_bf16"))]
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

fn attention_splitk_candidates(config: FmhaDecodeSplitKReduceConfig) -> Vec<AttentionSplitKTuning> {
    let max_splits = config.max_splits.min(config.decode.key_len);
    let mut splits = [1, 2, 4, 8, 16, 32, 64]
        .into_iter()
        .filter(|&splits| splits <= max_splits)
        .collect::<Vec<_>>();
    if max_splits > 0 && splits.last().copied() != Some(max_splits) {
        splits.push(max_splits);
    }
    splits.sort_unstable();
    splits.dedup();
    splits
        .into_iter()
        .map(AttentionSplitKTuning::create)
        .collect()
}

fn attention_splitk_tune_key(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    config: FmhaDecodeSplitKReduceConfig,
) -> Result<autotune::TuneKey> {
    let decode = config.decode;
    let device = DeviceFingerprint::create(stream.context().device())?;
    Ok(autotune::TuneKey::create(
        op,
        "cutile",
        device,
        TuneProblemKey::AttentionSplitK {
            dtype: dtype.to_owned(),
            op_kind: op.to_owned(),
            batch: decode.batch,
            key_len: decode.key_len,
            heads: decode.heads,
            kv_heads: decode.kv_heads,
            head_dim: decode.head_dim,
            max_splits: config.max_splits,
            query_batch_stride: decode.query_batch_stride,
            key_batch_stride: decode.key_batch_stride,
            value_batch_stride: decode.value_batch_stride,
            output_batch_stride: decode.output_batch_stride,
            query_head_stride: decode.query_head_stride,
            key_sequence_stride: decode.key_sequence_stride,
            value_sequence_stride: decode.value_sequence_stride,
            output_head_stride: decode.output_head_stride,
            key_head_stride: decode.key_head_stride,
            value_head_stride: decode.value_head_stride,
        },
    ))
}

fn resolve_attention_splitk_tuning<C, F>(
    stream: &Stream,
    op: &'static str,
    key: autotune::TuneKey,
    choice: TuningChoice<AttentionSplitKTuning>,
    candidates: C,
    tune: F,
) -> Result<Option<AttentionSplitKTuning>>
where
    C: FnOnce() -> Vec<AttentionSplitKTuning>,
    F: FnOnce(TuneOptions, Vec<AttentionSplitKTuning>) -> Result<TuneReport<AttentionSplitKTuning>>,
{
    match choice {
        TuningChoice::Default => Ok(None),
        TuningChoice::Explicit(tuning) => Ok(Some(tuning)),
        TuningChoice::Cached => {
            if let Some(tuning) = autotune::cache().get(&key)? {
                Ok(Some(tuning))
            } else {
                Err(autotune::cache_miss_error(op, stream.capture_status()?))
            }
        }
        TuningChoice::Tune(options) => {
            if let Some(tuning) = autotune::cache().get(&key)? {
                return Ok(Some(tuning));
            }
            let report = tune(options, candidates())?;
            autotune::cache().insert(key, report.best);
            Ok(Some(report.best))
        }
        TuningChoice::Retune(options) => {
            let report = tune(options, candidates())?;
            autotune::cache().insert(key, report.best);
            Ok(Some(report.best))
        }
    }
}

fn fmha_splitk_decode_config(
    config: FmhaDecodeSplitKReduceConfig,
    tuning: AttentionSplitKTuning,
) -> FmhaDecodeSplitKConfig {
    let decode = config.decode;
    let splits = tuning.splits;
    let kv_len_per_split = decode.key_len.div_ceil(splits);
    let split_values = splits * decode.head_dim;
    FmhaDecodeSplitKConfig {
        batch: decode.batch,
        key_len: decode.key_len,
        heads: decode.heads,
        kv_heads: decode.kv_heads,
        splits,
        kv_len_per_split,
        head_dim: decode.head_dim,
        query_batch_stride: decode.query_batch_stride,
        key_batch_stride: decode.key_batch_stride,
        value_batch_stride: decode.value_batch_stride,
        output_batch_stride: decode.heads * split_values,
        query_head_stride: decode.query_head_stride,
        key_sequence_stride: decode.key_sequence_stride,
        value_sequence_stride: decode.value_sequence_stride,
        output_head_stride: split_values,
        output_split_stride: decode.head_dim,
        key_head_stride: decode.key_head_stride,
        value_head_stride: decode.value_head_stride,
        lse_batch_stride: decode.heads * splits,
        lse_head_stride: splits,
        scale: decode.scale,
    }
}

fn fmha_splitk_reduce_config(
    config: FmhaDecodeSplitKReduceConfig,
    tuning: AttentionSplitKTuning,
) -> SplitKReduceConfig {
    let decode = config.decode;
    let splits = tuning.splits;
    let split_values = splits * decode.head_dim;
    SplitKReduceConfig {
        batch: decode.batch,
        heads: decode.heads,
        splits,
        head_dim: decode.head_dim,
        attn_batch_stride: decode.heads * split_values,
        attn_head_stride: split_values,
        attn_split_stride: decode.head_dim,
        lse_batch_stride: decode.heads * splits,
        lse_head_stride: splits,
        output_batch_stride: decode.output_batch_stride,
        output_head_stride: decode.output_head_stride,
    }
}

fn validate_fmha_decode_splitk_reduce_lengths(
    config: FmhaDecodeSplitKReduceConfig,
    out_len: usize,
    attn_scratch_len: usize,
    lse_scratch_len: usize,
    query_len: usize,
    key_len: usize,
    value_len: usize,
) -> Result<()> {
    validate_fmha_decode_splitk_reduce_config(config)?;
    config.decode.validate_lengths(
        out_len,
        checked_element_count(config.decode.batch, config.decode.heads)?,
        query_len,
        key_len,
        value_len,
    )?;
    let effective_splits = config.max_splits.min(config.decode.key_len);
    let scratch_values = checked_element_count(
        checked_element_count(config.decode.batch, config.decode.heads)?,
        effective_splits,
    )?;
    ensure_len_at_least(
        attn_scratch_len,
        checked_element_count(scratch_values, config.decode.head_dim)?,
    )?;
    ensure_len_at_least(lse_scratch_len, scratch_values)?;
    Ok(())
}

fn validate_attention_splitk_tuning(
    config: FmhaDecodeSplitKReduceConfig,
    tuning: AttentionSplitKTuning,
) -> Result<()> {
    if tuning.splits == 0
        || tuning.splits > config.max_splits
        || tuning.splits > config.decode.key_len
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_fmha_decode_splitk_reduce_config(config: FmhaDecodeSplitKReduceConfig) -> Result<()> {
    validate_fmha_decode_config(config.decode)?;
    if config.max_splits == 0 {
        return Err(Error::InvalidLength);
    }
    Ok(())
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

#[cfg(feature = "dtype_f32")]
/// Causal left-window attention.
///
/// For query absolute position `q_abs`, only keys satisfying
/// `key_abs <= q_abs` and `key_abs + window > q_abs` are visible. KV heads are
/// mapped with contiguous grouped-query attention semantics.
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn sliding_window_attention_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    query: &impl DeviceInput<f32>,
    key: &impl DeviceInput<f32>,
    value: &impl DeviceInput<f32>,
    config: SlidingWindowAttentionConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), query.len(), key.len(), value.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::attention::sliding_window_attention_f32(
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

#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
mod fmha;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
mod local;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
mod mla;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
mod multi_token;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
mod paged;
mod sparse;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
mod splitk;

#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
pub use fmha::*;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
pub use local::*;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
pub use mla::*;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
pub use multi_token::*;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
pub use paged::*;
pub use sparse::*;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
pub use splitk::*;

#[cfg(test)]
mod tests;

#[cfg(any(feature = "dtype_f16", feature = "dtype_bf16"))]
macro_rules! fmha_prefill_mma_half_fns {
    ($ty:ty, $plain:ident, $with_lse:ident) => {
        /// Enqueues tiled MMA prefill attention with f32 accumulation.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $plain(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            query: &impl DeviceInput<$ty>,
            key: &impl DeviceInput<$ty>,
            value: &impl DeviceInput<$ty>,
            config: FmhaPrefillConfig,
        ) -> Result<()> {
            config.validate_output_lengths(out.len(), query.len(), key.len(), value.len())?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::attention::fmha_prefill_mma(
                    &stream,
                    write_only_pointer(out),
                    None,
                    input_pointer(query),
                    input_pointer(key),
                    input_pointer(value),
                    config,
                )
            }
        }
        /// Enqueues tiled MMA prefill attention and writes f32 natural-log LSE values.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $with_lse(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lse: &mut impl DeviceOutput<f32>,
            query: &impl DeviceInput<$ty>,
            key: &impl DeviceInput<$ty>,
            value: &impl DeviceInput<$ty>,
            config: FmhaPrefillConfig,
        ) -> Result<()> {
            config.validate_lengths(out.len(), lse.len(), query.len(), key.len(), value.len())?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::attention::fmha_prefill_mma(
                    &stream,
                    write_only_pointer(out),
                    Some(write_only_pointer(lse)),
                    input_pointer(query),
                    input_pointer(key),
                    input_pointer(value),
                    config,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f16")]
fmha_prefill_mma_half_fns!(f16, fmha_prefill_mma_f16, fmha_prefill_lse_mma_f16);

#[cfg(feature = "dtype_bf16")]
fmha_prefill_mma_half_fns!(bf16, fmha_prefill_mma_bf16, fmha_prefill_lse_mma_bf16);
