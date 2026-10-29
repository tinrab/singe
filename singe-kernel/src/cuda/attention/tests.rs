use super::*;
use crate::error::Error;

#[test]
fn test_mla_prefill_validation_accepts_exact_lengths() -> Result<()> {
    let config = MlaPrefillConfig::contiguous(2, 3, 5, 4, 2, 8, 3, 0.5)?;
    config.validate_lengths(192, 192, 72, 160, 60, 160)
}

#[test]
fn test_mla_prefill_validation_rejects_bad_head_mapping() {
    assert!(matches!(
        MlaPrefillConfig::contiguous(2, 3, 5, 3, 2, 8, 3, 0.5),
        Err(Error::InvalidLength)
    ));
}

#[test]
fn test_mla_prefill_validation_rejects_short_query_pe() -> Result<()> {
    let config = MlaPrefillConfig::contiguous(2, 3, 5, 4, 2, 8, 3, 0.5)?;
    assert!(matches!(
        config.validate_lengths(192, 192, 71, 160, 60, 160),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_mla_decode_validation_accepts_exact_lengths() -> Result<()> {
    let config = MlaDecodeConfig::contiguous(2, 5, 4, 8, 3, 0.5)?;
    config.validate_lengths(64, 8, 64, 24, 80, 30)
}

#[test]
fn test_mla_decode_validation_rejects_short_lse() -> Result<()> {
    let config = MlaDecodeConfig::contiguous(2, 5, 4, 8, 3, 0.5)?;
    assert!(matches!(
        config.validate_lengths(64, 7, 64, 24, 80, 30),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_paged_mla_decode_attention_validation_accepts_exact_lengths() -> Result<()> {
    let config = PagedMlaDecodeAttentionConfig::contiguous(2, 4, 8, 3, 3, 5, 4, 0.5, 1.0)?;
    config.validate_lengths(64, 8, 64, 24, 120, 45, 2, 8)
}

#[test]
fn test_paged_mla_decode_attention_validation_rejects_short_key_pe_cache() -> Result<()> {
    let config = PagedMlaDecodeAttentionConfig::contiguous(2, 4, 8, 3, 3, 5, 4, 0.5, 1.0)?;
    assert!(matches!(
        config.validate_lengths(64, 8, 64, 24, 120, 44, 2, 8),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_mla_decode_splitk_validation_accepts_exact_lengths() -> Result<()> {
    let config = MlaDecodeSplitKConfig::contiguous(2, 5, 4, 3, 2, 8, 3, 0.5)?;
    config.validate_lengths(192, 24, 64, 24, 80, 30)
}

#[test]
fn test_mla_decode_splitk_validation_rejects_short_split_output() -> Result<()> {
    let config = MlaDecodeSplitKConfig::contiguous(2, 5, 4, 3, 2, 8, 3, 0.5)?;
    assert!(matches!(
        config.validate_lengths(191, 24, 64, 24, 80, 30),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_splitk_reduce_validation_accepts_padded_strides() -> Result<()> {
    let mut config = SplitKReduceConfig::contiguous(2, 4, 3, 8)?;
    config.attn_batch_stride = 104;
    config.attn_head_stride = 25;
    config.lse_batch_stride = 16;
    config.output_batch_stride = 40;
    config.validate_lengths(72, 208, 32)
}

#[test]
fn test_splitk_reduce_validation_rejects_short_lse() -> Result<()> {
    let config = SplitKReduceConfig::contiguous(2, 4, 3, 8)?;
    assert!(matches!(
        config.validate_lengths(64, 192, 23),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_fmha_decode_splitk_reduce_config_builders_preserve_tuning_choice() -> Result<()> {
    let config = FmhaDecodeSplitKReduceConfig::contiguous(2, 5, 4, 2, 8, 4, 0.5)?;
    let tuning = AttentionSplitKTuning::create(2);
    assert_eq!(
        config.explicit_tuning(tuning).tuning,
        TuningChoice::Explicit(tuning)
    );
    assert_eq!(config.cached_tuning().tuning, TuningChoice::Cached);
    assert!(matches!(
        config.tune(TuneOptions::default()).tuning,
        TuningChoice::Tune(_)
    ));
    assert!(matches!(
        config.retune(TuneOptions::default()).tuning,
        TuningChoice::Retune(_)
    ));
    Ok(())
}

#[test]
fn test_attention_splitk_candidates_include_power_ladder_and_max() -> Result<()> {
    let config = FmhaDecodeSplitKReduceConfig::contiguous(2, 17, 4, 2, 8, 6, 0.5)?;
    let splits = attention_splitk_candidates(config)
        .into_iter()
        .map(|tuning| tuning.splits)
        .collect::<Vec<_>>();
    assert_eq!(splits, vec![1, 2, 4, 6]);

    let config = FmhaDecodeSplitKReduceConfig::contiguous(2, 17, 4, 2, 8, 4, 0.5)?;
    let splits = attention_splitk_candidates(config)
        .into_iter()
        .map(|tuning| tuning.splits)
        .collect::<Vec<_>>();
    assert_eq!(splits, vec![1, 2, 4]);
    Ok(())
}

#[test]
fn test_fmha_decode_splitk_reduce_validation_accepts_exact_lengths() -> Result<()> {
    let config = FmhaDecodeSplitKReduceConfig::contiguous(2, 5, 4, 2, 8, 4, 0.5)?;
    validate_fmha_decode_splitk_reduce_lengths(config, 64, 256, 32, 64, 160, 160)
}

#[test]
fn test_fmha_decode_splitk_reduce_validation_caps_scratch_at_key_len() -> Result<()> {
    let config = FmhaDecodeSplitKReduceConfig::contiguous(2, 3, 4, 2, 8, 8, 0.5)?;
    validate_fmha_decode_splitk_reduce_lengths(config, 64, 192, 24, 64, 96, 96)
}

#[test]
fn test_fmha_decode_splitk_reduce_validation_rejects_short_scratch() -> Result<()> {
    let config = FmhaDecodeSplitKReduceConfig::contiguous(2, 5, 4, 2, 8, 4, 0.5)?;
    assert!(matches!(
        validate_fmha_decode_splitk_reduce_lengths(config, 64, 255, 32, 64, 160, 160),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
    assert!(matches!(
        validate_fmha_decode_splitk_reduce_lengths(config, 64, 256, 31, 64, 160, 160),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_attention_splitk_tuning_rejects_too_many_splits() -> Result<()> {
    let config = FmhaDecodeSplitKReduceConfig::contiguous(2, 5, 4, 2, 8, 4, 0.5)?;
    assert!(matches!(
        validate_attention_splitk_tuning(config, AttentionSplitKTuning::create(0)),
        Err(Error::InvalidLength)
    ));
    assert!(matches!(
        validate_attention_splitk_tuning(config, AttentionSplitKTuning::create(5)),
        Err(Error::InvalidLength)
    ));
    Ok(())
}

#[cfg(any(feature = "dtype_f16", feature = "dtype_bf16"))]
#[test]
fn test_swa_attention_validation_accepts_exact_lengths_and_zero_window() -> Result<()> {
    let config = SwaAttentionConfig::contiguous(2, 3, 5, 4, 8, 0, 0.5, true)?;
    config.validate_lengths(192, 192, 320, 320)
}

#[cfg(any(feature = "dtype_f16", feature = "dtype_bf16"))]
#[test]
fn test_swa_attention_validation_rejects_short_key() -> Result<()> {
    let config = SwaAttentionConfig::contiguous(2, 3, 5, 4, 8, 2, 0.5, true)?;
    assert!(matches!(
        config.validate_lengths(192, 192, 319, 320),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_swa_attention_validation_rejects_non_causal() {
    assert!(matches!(
        SwaAttentionConfig::contiguous(2, 3, 5, 4, 8, 2, 0.5, false),
        Err(Error::InvalidLength)
    ));
}

#[test]
fn test_fused_neighborhood_attention_validation_accepts_exact_lengths() -> Result<()> {
    let config = FusedNeighborhoodAttentionConfig::contiguous(2, 3, 4, 8, 3, 1, 0.5)?;
    config.validate_lengths(192, 192, 192, 192)
}

#[test]
fn test_fused_neighborhood_attention_validation_rejects_short_key() -> Result<()> {
    let config = FusedNeighborhoodAttentionConfig::contiguous(2, 3, 4, 8, 3, 1, 0.5)?;
    assert!(matches!(
        config.validate_lengths(192, 192, 191, 192),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_fused_neighborhood_attention_validation_rejects_zero_dilation() {
    assert!(matches!(
        FusedNeighborhoodAttentionConfig::contiguous(2, 3, 4, 8, 3, 0, 0.5),
        Err(Error::InvalidLength)
    ));
}

#[test]
fn test_multi_token_attention_validation_accepts_exact_lengths() -> Result<()> {
    let config = MultiTokenAttentionConfig::same_padding_1x1(2, 3, 4, 5)?;
    config.validate_lengths(200, 150, 12, Some(4))
}

#[test]
fn test_multi_token_attention_validation_rejects_short_bias() -> Result<()> {
    let config = MultiTokenAttentionConfig::same_padding_1x1(2, 3, 4, 5)?;
    assert!(matches!(
        config.validate_lengths(200, 150, 12, Some(3)),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_multi_token_attention_validation_rejects_uneven_groups() {
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
fn test_paged_kv_attention_validation_accepts_exact_lengths() -> Result<()> {
    let config = paged_kv_validation_config();
    validate_paged_kv_attention_config(192, 192, 96, 96, 7, config)
}

#[test]
fn test_paged_kv_attention_validation_rejects_short_block_table() {
    let config = paged_kv_validation_config();
    assert!(matches!(
        validate_paged_kv_attention_config(192, 192, 96, 96, 6, config),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
}

#[test]
fn test_paged_kv_attention_validation_rejects_short_key_cache() {
    let config = paged_kv_validation_config();
    assert!(matches!(
        validate_paged_kv_attention_config(192, 192, 95, 96, 7, config),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
}

#[test]
fn test_paged_kv_attention_validation_rejects_zero_block_size() {
    let mut config = paged_kv_validation_config();
    config.block_size = 0;
    assert!(matches!(
        validate_paged_kv_attention_config(192, 192, 96, 96, 7, config),
        Err(Error::InvalidLength)
    ));
}

#[test]
fn test_paged_kv_decode_attention_validation_accepts_exact_lengths() -> Result<()> {
    let config = PagedKvDecodeAttentionConfig::contiguous(2, 4, 2, 8, 3, 5, 4, 0.5)?;
    config.validate_lengths(64, 64, 240, 240, 2, 8)
}

#[test]
fn test_paged_kv_decode_attention_validation_rejects_short_block_table() -> Result<()> {
    let config = PagedKvDecodeAttentionConfig::contiguous(2, 4, 2, 8, 3, 5, 4, 0.5)?;
    assert!(matches!(
        config.validate_lengths(64, 64, 240, 240, 2, 7),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_paged_kv_decode_attention_validation_rejects_invalid_head_mapping() {
    assert!(matches!(
        PagedKvDecodeAttentionConfig::contiguous(2, 3, 2, 8, 3, 5, 4, 0.5),
        Err(Error::InvalidLength)
    ));
}

#[test]
fn test_paged_kv_prefill_attention_validation_accepts_exact_lengths() -> Result<()> {
    let config = PagedKvPrefillAttentionConfig::contiguous(2, 7, 4, 2, 8, 3, 5, 4, true, 0.5)?;
    config.validate_lengths(224, 28, 224, 240, 240, 2, 2, 2, 8)
}

#[test]
fn test_paged_kv_prefill_attention_validation_rejects_short_lse() -> Result<()> {
    let config = PagedKvPrefillAttentionConfig::contiguous(2, 7, 4, 2, 8, 3, 5, 4, true, 0.5)?;
    assert!(matches!(
        config.validate_lengths(224, 27, 224, 240, 240, 2, 2, 2, 8),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_paged_kv_prefill_attention_validation_rejects_invalid_head_mapping() {
    assert!(matches!(
        PagedKvPrefillAttentionConfig::contiguous(2, 7, 3, 2, 8, 3, 5, 4, true, 0.5),
        Err(Error::InvalidLength)
    ));
}

#[test]
fn test_ragged_kv_prefill_attention_validation_accepts_exact_lengths() -> Result<()> {
    let config = RaggedKvPrefillAttentionConfig::contiguous(2, 7, 8, 4, 2, 8, true, 0.5)?;
    config.validate_lengths(224, 28, 224, 128, 128, 2, 2, 2)
}

#[test]
fn test_ragged_kv_prefill_attention_validation_rejects_short_key() -> Result<()> {
    let config = RaggedKvPrefillAttentionConfig::contiguous(2, 7, 8, 4, 2, 8, true, 0.5)?;
    assert!(matches!(
        config.validate_lengths(224, 28, 224, 127, 128, 2, 2, 2),
        Err(Error::LengthMismatch
            | Error::LengthMismatchFor { .. }
            | Error::LengthTooShort { .. }
            | Error::RankReachMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_ragged_kv_prefill_attention_validation_rejects_invalid_head_mapping() {
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
