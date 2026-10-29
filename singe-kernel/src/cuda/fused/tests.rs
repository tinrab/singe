use super::*;

fn is_length_mismatch<T>(result: Result<T>) -> bool {
    matches!(
        result,
        Err(Error::LengthMismatch | Error::LengthMismatchFor { .. })
    )
}

#[test]
fn test_fused_config_builders_preserve_tuning_choice() {
    let tuning =
        FusedTileTuning::tile(256).with_compile(KernelCompileConfig::default().occupancy(3));
    assert_eq!(
        Fused1dConfig::create(17).explicit_tuning(tuning).tuning,
        TuningChoice::Explicit(tuning)
    );
    assert_eq!(
        Fused1dConfig::create(17).cached_tuning().tuning,
        TuningChoice::Cached
    );

    let options = TuneOptions::create(0, 1, 2);
    assert_eq!(
        Fused1dConfig::create(17).tune(options).tuning,
        TuningChoice::Tune(options)
    );
    assert_eq!(
        Fused1dConfig::create(17).retune(options).tuning,
        TuningChoice::Retune(options)
    );

    assert_eq!(
        Fused2dConfig::create(3, 5).explicit_tuning(tuning).tuning,
        TuningChoice::Explicit(tuning)
    );
    assert_eq!(
        Fused2dConfig::create(3, 5).cached_tuning().tuning,
        TuningChoice::Cached
    );

    assert_eq!(
        Fused2dConfig::create(3, 5).tune(options).tuning,
        TuningChoice::Tune(options)
    );
    assert_eq!(
        Fused2dConfig::create(3, 5).retune(options).tuning,
        TuningChoice::Retune(options)
    );

    assert_eq!(
        FusedRmsNormConfig::create(3, 64, 1e-5f32, 1.0)
            .explicit_tuning(tuning)
            .tuning,
        TuningChoice::Explicit(tuning)
    );
    assert_eq!(
        FusedRmsNormConfig::create(3, 64, 1e-5f32, 1.0)
            .cached_tuning()
            .tuning,
        TuningChoice::Cached
    );
    assert_eq!(
        FusedRmsNormConfig::create(3, 64, 1e-5f32, 1.0)
            .tune(options)
            .tuning,
        TuningChoice::Tune(options)
    );
    assert_eq!(
        FusedRmsNormConfig::create(3, 64, 1e-5f32, 1.0)
            .retune(options)
            .tuning,
        TuningChoice::Retune(options)
    );
}

#[test]
fn test_fused_tile_candidates_cover_cu_tile_ladder() {
    let candidates = fused_tile_candidates();
    let tile_widths = candidates
        .into_iter()
        .map(|candidate| candidate.tile_width)
        .collect::<Vec<_>>();
    assert_eq!(tile_widths, vec![128, 256, 512, 1024]);
}

#[test]
fn test_rms_norm_fused_tile_candidates_cover_launchable_ladder() {
    let tile_widths = rms_norm_fused_tile_candidates(65)
        .into_iter()
        .map(|candidate| candidate.tile_width)
        .collect::<Vec<_>>();
    assert_eq!(tile_widths, vec![128, 256, 512, 1024]);
}

#[test]
fn test_binary_fused_1d_validation_accepts_exact_lengths() -> Result<()> {
    let config = Fused1dConfig::create(24);
    assert_eq!(validate_binary_fused_1d_lengths(24, 24, 24, config)?, 24);
    Ok(())
}

#[test]
fn test_binary_fused_1d_validation_rejects_short_rhs() {
    let config = Fused1dConfig::create(24);
    assert!(is_length_mismatch(validate_binary_fused_1d_lengths(
        24, 24, 23, config
    )));
}

#[test]
fn test_ternary_fused_1d_validation_accepts_exact_lengths() -> Result<()> {
    let config = Fused1dConfig::create(24);
    assert_eq!(
        validate_ternary_fused_1d_lengths(24, 24, 24, 24, config)?,
        24
    );
    Ok(())
}

#[test]
fn test_ternary_fused_1d_validation_rejects_short_third_input() {
    let config = Fused1dConfig::create(24);
    assert!(is_length_mismatch(validate_ternary_fused_1d_lengths(
        24, 24, 24, 23, config
    )));
}

#[test]
fn test_binary_fused_2d_validation_accepts_exact_lengths() -> Result<()> {
    let config = Fused2dConfig::create(4, 6);
    assert_eq!(validate_binary_fused_2d_lengths(24, 24, 24, config)?, 24);
    Ok(())
}

#[test]
fn test_binary_fused_2d_validation_rejects_short_rhs() {
    let config = Fused2dConfig::create(4, 6);
    assert!(is_length_mismatch(validate_binary_fused_2d_lengths(
        24, 24, 23, config
    )));
}

#[test]
fn test_ternary_fused_2d_validation_accepts_exact_lengths() -> Result<()> {
    let config = Fused2dConfig::create(4, 6);
    assert_eq!(
        validate_ternary_fused_2d_lengths(24, 24, 24, 24, config)?,
        24
    );
    Ok(())
}

#[test]
fn test_ternary_fused_2d_validation_rejects_short_third_input() {
    let config = Fused2dConfig::create(4, 6);
    assert!(is_length_mismatch(validate_ternary_fused_2d_lengths(
        24, 24, 24, 23, config
    )));
}

#[test]
fn test_bias_fused_2d_validation_accepts_column_bias() -> Result<()> {
    let config = Fused2dConfig::create(4, 6);
    assert_eq!(validate_bias_fused_2d_lengths(24, 24, 6, config)?, 24);
    Ok(())
}

#[test]
fn test_bias_fused_2d_validation_rejects_short_bias() {
    let config = Fused2dConfig::create(4, 6);
    assert!(is_length_mismatch(validate_bias_fused_2d_lengths(
        24, 24, 5, config
    )));
}

#[test]
fn test_column_ternary_fused_2d_validation_accepts_column_operands() -> Result<()> {
    let config = Fused2dConfig::create(4, 6);
    assert_eq!(
        validate_column_ternary_fused_2d_lengths(24, 24, 6, 6, config)?,
        24
    );
    Ok(())
}

#[test]
fn test_column_ternary_fused_2d_validation_rejects_short_rhs_column() {
    let config = Fused2dConfig::create(4, 6);
    assert!(is_length_mismatch(
        validate_column_ternary_fused_2d_lengths(24, 24, 6, 5, config)
    ));
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
#[test]
fn test_conditional_fused_1d_validation_accepts_exact_lengths() -> Result<()> {
    let config = Fused1dConfig::create(24);
    assert_eq!(
        validate_conditional_fused_1d_lengths(24, 24, 24, 24, 24, config)?,
        24
    );
    Ok(())
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
#[test]
fn test_conditional_fused_1d_validation_rejects_short_condition() {
    let config = Fused1dConfig::create(24);
    assert!(is_length_mismatch(validate_conditional_fused_1d_lengths(
        24, 23, 24, 24, 24, config
    )));
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
#[test]
fn test_conditional_fused_2d_validation_accepts_exact_lengths() -> Result<()> {
    let config = Fused2dConfig::create(4, 6);
    assert_eq!(
        validate_conditional_fused_2d_lengths(24, 24, 24, 24, 24, config)?,
        24
    );
    Ok(())
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
#[test]
fn test_conditional_fused_2d_validation_rejects_short_condition() {
    let config = Fused2dConfig::create(4, 6);
    assert!(is_length_mismatch(validate_conditional_fused_2d_lengths(
        24, 23, 24, 24, 24, config
    )));
}

#[test]
fn test_rms_norm_add_validation_accepts_exact_lengths() -> Result<()> {
    assert_eq!(validate_rms_norm_add(24, 24, 24, 24, 6, 4, 6)?, 24);
    Ok(())
}

#[test]
fn test_rms_norm_silu_mul_validation_accepts_exact_lengths() -> Result<()> {
    assert_eq!(validate_rms_norm_silu_mul(24, 24, 6, 24, 4, 6)?, 24);
    Ok(())
}

#[test]
fn test_rms_norm_gated_silu_validation_accepts_exact_lengths() -> Result<()> {
    assert_eq!(validate_rms_norm_gated_silu(24, 24, 24, 6, 4, 6)?, 24);
    Ok(())
}

#[test]
fn test_rms_norm_gated_silu_validation_rejects_short_gate() {
    assert!(is_length_mismatch(validate_rms_norm_gated_silu(
        24, 24, 23, 6, 4, 6
    )));
}

#[test]
fn test_rms_norm_gated_silu_validation_rejects_zero_rows() {
    assert!(matches!(
        validate_rms_norm_gated_silu(0, 0, 0, 6, 0, 6),
        Err(Error::InvalidLength)
    ));
}

#[test]
fn test_dual_rms_norm_validation_accepts_exact_lengths() -> Result<()> {
    assert_eq!(validate_dual_rms_norm(24, 24, 24, 24, 6, 6, 4, 6)?, 24);
    Ok(())
}

#[test]
fn test_dual_rms_norm_validation_rejects_short_k_weight() {
    assert!(is_length_mismatch(validate_dual_rms_norm(
        24, 24, 24, 24, 6, 5, 4, 6
    )));
}

#[test]
fn test_rms_norm_residual_add_validation_accepts_exact_lengths() -> Result<()> {
    assert_eq!(validate_rms_norm_residual_add(24, 24, 24, 6, 4, 6)?, 24);
    Ok(())
}

#[test]
fn test_rms_norm_residual_add_validation_rejects_short_residual() {
    assert!(is_length_mismatch(validate_rms_norm_residual_add(
        24, 24, 23, 6, 4, 6
    )));
}

#[test]
fn test_rms_norm_add_wide_validation_accepts_shared_hidden_width() -> Result<()> {
    assert_eq!(
        validate_rms_norm_add_wide(10_240, 10_240, 10_240, 10_240, 5_120, 10, 2, 2, 5_120,)?,
        10_240
    );
    Ok(())
}

#[test]
fn test_rms_norm_add_wide_validation_rejects_short_partial_sum() {
    assert!(is_length_mismatch(validate_rms_norm_add_wide(
        10_240, 10_240, 10_240, 10_240, 5_120, 9, 2, 2, 5_120,
    )));
}

#[test]
fn test_rms_norm_add_wide_validation_rejects_short_row_sum() {
    assert!(is_length_mismatch(validate_rms_norm_add_wide(
        10_240, 10_240, 10_240, 10_240, 5_120, 10, 1, 2, 5_120,
    )));
}

#[test]
fn test_mhc_apply_residual_validation_accepts_exact_lengths() -> Result<()> {
    let config = MhcApplyResidualConfig {
        batch: 2,
        n: 3,
        channels: 4,
    };
    assert_eq!(validate_mhc_apply_residual(24, 24, 8, 30, config)?, 24);
    Ok(())
}

#[test]
fn test_mhc_apply_residual_validation_rejects_short_y() {
    let config = MhcApplyResidualConfig {
        batch: 2,
        n: 3,
        channels: 4,
    };
    assert!(is_length_mismatch(validate_mhc_apply_residual(
        24, 24, 8, 29, config
    )));
}

#[test]
fn test_mhc_apply_residual_validation_rejects_zero_n() {
    let config = MhcApplyResidualConfig {
        batch: 2,
        n: 0,
        channels: 4,
    };
    assert!(matches!(
        validate_mhc_apply_residual(0, 0, 8, 0, config),
        Err(Error::InvalidLength)
    ));
}

#[test]
fn test_mhc_sinkhorn_validation_accepts_exact_y() -> Result<()> {
    assert_eq!(validate_mhc_y_buffer(30, 2, 3)?, 30);
    Ok(())
}

#[test]
fn test_mhc_sinkhorn_validation_rejects_short_y() {
    assert!(is_length_mismatch(validate_mhc_y_buffer(29, 2, 3)));
}

#[test]
fn test_mhc_sinkhorn_validation_rejects_zero_batch() {
    assert!(matches!(
        validate_mhc_y_buffer(0, 0, 3),
        Err(Error::InvalidLength)
    ));
}

#[test]
fn test_mhc_gemm_rms_scale_validation_accepts_exact_lengths() -> Result<()> {
    let config = MhcGemmRmsScaleConfig {
        rows: 2,
        columns: 7,
        reduction: 4,
        n: 3,
        alpha_pre: 0.5,
        alpha_post: 1.25,
        alpha_res: 2.0,
    };
    assert_eq!(validate_mhc_gemm_rms_scale(14, 2, 8, 28, 7, config)?, 14);
    Ok(())
}

#[test]
fn test_mhc_gemm_rms_scale_validation_rejects_short_bias() {
    let config = MhcGemmRmsScaleConfig {
        rows: 2,
        columns: 7,
        reduction: 4,
        n: 3,
        alpha_pre: 0.5,
        alpha_post: 1.25,
        alpha_res: 2.0,
    };
    assert!(is_length_mismatch(validate_mhc_gemm_rms_scale(
        14, 2, 8, 28, 6, config
    )));
}

#[test]
fn test_mhc_gemm_rms_scale_validation_rejects_too_few_columns() {
    let config = MhcGemmRmsScaleConfig {
        rows: 2,
        columns: 5,
        reduction: 4,
        n: 3,
        alpha_pre: 0.5,
        alpha_post: 1.25,
        alpha_res: 2.0,
    };
    assert!(matches!(
        validate_mhc_gemm_rms_scale(10, 2, 8, 20, 5, config),
        Err(Error::LengthMismatch)
    ));
}

#[test]
fn test_mhc_split_gemm_rms_validation_accepts_exact_temp_lengths() -> Result<()> {
    let config = MhcSplitGemmRmsScaleConfig {
        base: MhcGemmRmsScaleConfig {
            rows: 2,
            columns: 7,
            reduction: 5,
            n: 3,
            alpha_pre: 0.5,
            alpha_post: 1.25,
            alpha_res: 2.0,
        },
        split_k: 3,
    };
    assert_eq!(validate_mhc_split_gemm_rms(42, 42, 10, 35, config)?, 42);
    assert_eq!(
        validate_mhc_finalize_scale_bias_sigmoid(14, 2, 42, 42, 7, config)?,
        14
    );
    Ok(())
}

#[test]
fn test_mhc_split_gemm_rms_validation_rejects_short_temp() {
    let config = MhcSplitGemmRmsScaleConfig {
        base: MhcGemmRmsScaleConfig {
            rows: 2,
            columns: 7,
            reduction: 5,
            n: 3,
            alpha_pre: 0.5,
            alpha_post: 1.25,
            alpha_res: 2.0,
        },
        split_k: 3,
    };
    assert!(is_length_mismatch(validate_mhc_split_gemm_rms(
        41, 42, 10, 35, config
    )));
    assert!(is_length_mismatch(
        validate_mhc_finalize_scale_bias_sigmoid(14, 2, 42, 41, 7, config)
    ));
}

#[test]
fn test_mhc_split_gemm_rms_validation_rejects_zero_split_k() {
    let config = MhcSplitGemmRmsScaleConfig {
        base: MhcGemmRmsScaleConfig {
            rows: 2,
            columns: 7,
            reduction: 5,
            n: 3,
            alpha_pre: 0.5,
            alpha_post: 1.25,
            alpha_res: 2.0,
        },
        split_k: 0,
    };
    assert!(matches!(
        validate_mhc_split_gemm_rms(42, 42, 10, 35, config),
        Err(Error::InvalidLength)
    ));
}

#[test]
fn test_rms_norm_silu_mul_validation_rejects_short_up_projection() {
    assert!(is_length_mismatch(validate_rms_norm_silu_mul(
        24, 24, 6, 23, 4, 6
    )));
}

#[test]
fn test_exact_gelu_mul_validation_accepts_exact_lengths() -> Result<()> {
    assert_eq!(validate_binary_fused_lengths(24, 24, 24)?, 24);
    Ok(())
}

#[test]
fn test_exact_gelu_mul_validation_rejects_short_gate() {
    assert!(is_length_mismatch(validate_binary_fused_lengths(
        24, 23, 24
    )));
}

#[test]
fn test_exact_gelu_mul_validation_rejects_short_up_projection() {
    assert!(is_length_mismatch(validate_binary_fused_lengths(
        24, 24, 23
    )));
}

#[test]
fn test_silu_and_mul_packed_validation_accepts_exact_lengths() -> Result<()> {
    assert_eq!(validate_silu_and_mul_packed_lengths(24, 48, 4, 6)?, 24);
    Ok(())
}

#[test]
fn test_silu_and_mul_packed_validation_rejects_short_input() {
    assert!(is_length_mismatch(validate_silu_and_mul_packed_lengths(
        24, 47, 4, 6
    )));
}

#[test]
fn test_silu_and_mul_packed_validation_rejects_zero_hidden() {
    assert!(matches!(
        validate_silu_and_mul_packed_lengths(0, 0, 4, 0),
        Err(Error::InvalidLength)
    ));
}
