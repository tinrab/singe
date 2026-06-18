#[cfg(test)]
use crate::data_type::DataType;
use crate::{
    error::{Error, Result},
    frontend::operation::{AttentionConfig, ReductionConfig},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrontendFeatureSupport {
    pub(crate) feature: &'static str,
    pub(crate) min_version: u64,
    pub(crate) min_version_label: &'static str,
}

impl FrontendFeatureSupport {
    pub(crate) const fn new(
        feature: &'static str,
        min_version: u64,
        min_version_label: &'static str,
    ) -> Self {
        Self {
            feature,
            min_version,
            min_version_label,
        }
    }

    pub(crate) fn is_supported(self, cudnn_version: u64) -> bool {
        cudnn_version >= self.min_version
    }

    pub(crate) fn require_frontend_feature(self, cudnn_version: u64) -> Result<()> {
        if !self.is_supported(cudnn_version) {
            return Err(Error::FrontendFeatureRequiresVersion {
                feature: self.feature.into(),
                min_version: self.min_version_label.into(),
                actual_version: cudnn_version,
            });
        }
        Ok(())
    }

    pub(crate) fn require_descriptor_match(self, cudnn_version: u64) -> Result<()> {
        if !self.is_supported(cudnn_version) {
            return Err(Error::DescriptorMismatch {
                name: format!("{} version", self.feature),
            });
        }
        Ok(())
    }
}

pub(crate) const DYNAMIC_SHAPE_OR_KERNEL_CACHE: FrontendFeatureSupport =
    FrontendFeatureSupport::new("dynamic shape or kernel cache", 90400, "9.4");
pub(crate) const OVERRIDE_SHAPE: FrontendFeatureSupport =
    FrontendFeatureSupport::new("override shape", 92100, "9.21");
pub(crate) const DEVICE_PROPERTIES: FrontendFeatureSupport =
    FrontendFeatureSupport::new("device properties", 90800, "9.8");
pub(crate) const KERNEL_CACHE_SERIALIZATION: FrontendFeatureSupport =
    FrontendFeatureSupport::new("kernel cache serialization", 91000, "9.10");

pub(crate) const CONCAT: FrontendFeatureSupport =
    FrontendFeatureSupport::new("concat", 90700, "9.7");
pub(crate) const BLOCK_SCALE_QUANTIZE: FrontendFeatureSupport =
    FrontendFeatureSupport::new("block scale quantize", 90700, "9.7");
pub(crate) const BLOCK_SCALE_DEQUANTIZE: FrontendFeatureSupport =
    FrontendFeatureSupport::new("block scale dequantize", 90700, "9.7");
pub(crate) const PAGED_CACHE_LOAD: FrontendFeatureSupport =
    FrontendFeatureSupport::new("paged cache load", 90500, "9.5");
pub(crate) const ADAPTIVE_LAYER_NORM: FrontendFeatureSupport =
    FrontendFeatureSupport::new("adaptive layer norm", 90900, "9.9");
pub(crate) const MOE_GROUPED_MATMUL: FrontendFeatureSupport =
    FrontendFeatureSupport::new("moe grouped matmul", 91500, "9.15");
pub(crate) const MOE_GROUPED_MATMUL_BACKWARD: FrontendFeatureSupport =
    FrontendFeatureSupport::new("moe grouped matmul backward", 92200, "9.22");
pub(crate) const DETERMINISTIC_REDUCTION: FrontendFeatureSupport =
    FrontendFeatureSupport::new("reduction deterministic", 91100, "9.11");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UnifiedSdpaForwardSupport {
    cudnn_version: u64,
}

impl UnifiedSdpaForwardSupport {
    pub(crate) const fn new(cudnn_version: u64) -> Self {
        Self { cudnn_version }
    }

    pub(crate) fn supports_config(self, config: &AttentionConfig) -> bool {
        let modifiers = config.modifiers();
        let fused = config.fused();

        let uses_sequence_lengths = modifiers.sequence_length_query.is_some()
            && modifiers.sequence_length_key_value.is_some();
        let uses_pre_softmax_subgraph = modifiers.bias.is_some()
            || config.score_subgraph().is_some()
            || modifiers.alibi_slopes.is_some()
            || modifiers.additive_mask.is_some()
            || modifiers.causal_mask
            || modifiers.causal_bottom_right
            || modifiers.sliding_window.is_some();
        let uses_paged_cache = config.paged_k().is_some()
            || config.paged_v().is_some()
            || config.max_sequence_length_key_value().is_some();
        let uses_921_aux_outputs = fused.has_logit_max() || fused.has_score_sum_exp();
        let uses_unsupported_aux_outputs = fused.has_absolute_max_s() || fused.has_absolute_max_o();

        self.supports_base_unified()
            && self.supports_cudnn_921_feature(uses_pre_softmax_subgraph)
            && self.supports_cudnn_915_feature(uses_sequence_lengths)
            && self.supports_cudnn_915_feature(uses_paged_cache)
            && self.supports_cudnn_914_feature(config.block_mask().is_some())
            && self.supports_cudnn_921_feature(config.dropout().is_some())
            && self.supports_cudnn_921_feature(config.unfuse_fma())
            && self.supports_cudnn_921_feature(modifiers.sink_token.is_some())
            && modifiers.dropout_mask.is_none()
            && modifiers.dropout_scale.is_none()
            && self.supports_cudnn_921_feature(uses_921_aux_outputs)
            && !uses_unsupported_aux_outputs
    }

    const fn supports_base_unified(self) -> bool {
        self.cudnn_version >= 91301
    }

    const fn supports_cudnn_914_feature(self, feature_used: bool) -> bool {
        self.cudnn_version >= 91400 || !feature_used
    }

    const fn supports_cudnn_915_feature(self, feature_used: bool) -> bool {
        self.cudnn_version >= 91500 || !feature_used
    }

    const fn supports_cudnn_921_feature(self, feature_used: bool) -> bool {
        self.cudnn_version >= 92100 || !feature_used
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SdpaBackwardSupport {
    pub(crate) cudnn_version: u64,
    pub(crate) sm_version: Option<i32>,
}

#[cfg(test)]
impl SdpaBackwardSupport {
    pub(crate) const fn new(cudnn_version: u64, sm_version: Option<i32>) -> Self {
        Self {
            cudnn_version,
            sm_version,
        }
    }

    pub(crate) fn should_clear_max_total_sequence_lengths(self, d_qk: i64, d_v: i64) -> bool {
        should_clear_sdpa_backward_max_total_sequence_lengths(
            self.cudnn_version,
            self.sm_version,
            d_qk,
            d_v,
        )
    }

    pub(crate) fn require_data_type(self, data_type: DataType) -> Result<()> {
        require_sdpa_backward_data_type_support(self.cudnn_version, data_type)
    }

    pub(crate) fn require_architecture(
        self,
        data_type: DataType,
        d_qk: i64,
        d_v: i64,
    ) -> Result<()> {
        require_sdpa_backward_architecture_support(
            self.cudnn_version,
            self.sm_version,
            data_type,
            d_qk,
            d_v,
        )
    }

    pub(crate) fn require_ragged_gqa(
        self,
        is_ragged: bool,
        q_head_count: i64,
        k_head_count: i64,
        v_head_count: i64,
    ) -> Result<()> {
        require_sdpa_backward_ragged_gqa_support(
            self.cudnn_version,
            is_ragged,
            q_head_count,
            k_head_count,
            v_head_count,
        )
    }

    pub(crate) fn require_ragged_stats(self, has_ragged_stats: bool) -> Result<()> {
        require_sdpa_backward_ragged_stats_support(
            self.cudnn_version,
            self.sm_version,
            has_ragged_stats,
        )
    }
}

#[cfg(test)]
pub(crate) fn should_clear_sdpa_backward_max_total_sequence_lengths(
    cudnn_version: u64,
    sm_version: Option<i32>,
    d_qk: i64,
    d_v: i64,
) -> bool {
    if cudnn_version < 90600 || d_qk % 16 != 0 || d_v % 16 != 0 {
        return true;
    }

    cudnn_version >= 91801 && sm_version.is_some_and(|sm_version| matches!(sm_version / 10, 8 | 12))
}

#[cfg(test)]
pub(crate) fn require_sdpa_backward_data_type_support(
    cudnn_version: u64,
    data_type: DataType,
) -> Result<()> {
    if matches!(data_type, DataType::F16 | DataType::BF16) && matches!(cudnn_version, 91000 | 91001)
    {
        return Err(Error::DescriptorMismatch {
            name: "sdpa backward cuDNN version".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn require_sdpa_backward_architecture_support(
    cudnn_version: u64,
    sm_version: Option<i32>,
    data_type: DataType,
    d_qk: i64,
    d_v: i64,
) -> Result<()> {
    let Some(sm_version) = sm_version else {
        return Ok(());
    };
    let prop_major = sm_version / 10;

    if matches!(data_type, DataType::F16 | DataType::BF16) && sm_version < 80 {
        return Err(Error::DescriptorMismatch {
            name: "sdpa backward sm version".into(),
        });
    }

    if prop_major == 9 {
        if (91100..91300).contains(&cudnn_version)
            && (128 < d_qk && d_qk <= 192)
            && (64 < d_v && d_v <= 128)
            && (d_qk != 192 || d_v != 128)
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward head dimension".into(),
            });
        }
        if d_qk > 256 || d_qk % 8 != 0 || d_v > 256 || d_v % 8 != 0 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward head dimension".into(),
            });
        }
    } else if prop_major == 10 && cudnn_version >= 91100 {
        if d_qk == 192 {
            if d_v != 128 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa backward head dimension".into(),
                });
            }
        } else if d_qk > 128 || d_qk % 8 != 0 || d_v > 128 || d_v % 8 != 0 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward head dimension".into(),
            });
        }
    } else if d_qk > 128 || d_qk % 8 != 0 || d_v > 128 || d_v % 8 != 0 {
        return Err(Error::DescriptorMismatch {
            name: "sdpa backward head dimension".into(),
        });
    }

    Ok(())
}

#[cfg(test)]
pub(crate) fn require_sdpa_backward_ragged_gqa_support(
    cudnn_version: u64,
    is_ragged: bool,
    q_head_count: i64,
    k_head_count: i64,
    v_head_count: i64,
) -> Result<()> {
    if is_ragged
        && cudnn_version < 90600
        && (q_head_count != k_head_count || q_head_count != v_head_count)
    {
        return Err(Error::DescriptorMismatch {
            name: "sdpa backward ragged gqa".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn require_sdpa_backward_ragged_stats_support(
    cudnn_version: u64,
    sm_version: Option<i32>,
    has_ragged_stats: bool,
) -> Result<()> {
    if has_ragged_stats
        && cudnn_version >= 91801
        && sm_version.is_some_and(|sm_version| matches!(sm_version / 10, 8 | 12))
    {
        return Err(Error::DescriptorMismatch {
            name: "sdpa backward ragged stats".into(),
        });
    }
    Ok(())
}

pub(crate) fn require_reduction_support(cudnn_version: u64, config: ReductionConfig) -> Result<()> {
    if config.is_deterministic() {
        DETERMINISTIC_REDUCTION.require_descriptor_match(cudnn_version)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_descriptor_mismatch(result: Result<()>, expected_name: &str) {
        assert!(matches!(
            result,
            Err(Error::DescriptorMismatch { name }) if name == expected_name
        ));
    }

    #[test]
    fn sdpa_backward_max_total_sequence_lengths_clear_from_support_table() {
        let cases = [
            (90500, None, 16, 16, true),
            (90600, None, 15, 16, true),
            (90600, None, 16, 15, true),
            (91800, Some(80), 16, 16, false),
            (91801, Some(80), 16, 16, true),
            (91801, Some(120), 16, 16, true),
            (91801, Some(90), 16, 16, false),
            (91801, None, 16, 16, false),
        ];

        for (version, sm_version, d_qk, d_v, expected) in cases {
            assert_eq!(
                should_clear_sdpa_backward_max_total_sequence_lengths(
                    version, sm_version, d_qk, d_v
                ),
                expected,
                "version={version} sm={sm_version:?} d_qk={d_qk} d_v={d_v}",
            );
        }
    }

    #[test]
    fn sdpa_backward_data_type_support_rejects_low_precision_on_cudnn_9100_and_9101() {
        for version in [91000, 91001] {
            for data_type in [DataType::F16, DataType::BF16] {
                assert_descriptor_mismatch(
                    require_sdpa_backward_data_type_support(version, data_type),
                    "sdpa backward cuDNN version",
                );
            }
        }

        for data_type in [DataType::F16, DataType::BF16, DataType::F32] {
            require_sdpa_backward_data_type_support(91002, data_type).unwrap();
        }
    }

    #[test]
    fn sdpa_backward_architecture_support_matches_head_dimension_matrix() {
        let accepted = [
            (91000, Some(80), DataType::F16, 128, 128),
            (91100, Some(90), DataType::F16, 128, 128),
            (91300, Some(90), DataType::F16, 192, 128),
            (91100, Some(100), DataType::F16, 192, 128),
            (91100, None, DataType::F16, 512, 512),
        ];
        for (version, sm_version, data_type, d_qk, d_v) in accepted {
            require_sdpa_backward_architecture_support(version, sm_version, data_type, d_qk, d_v)
                .unwrap();
        }

        let rejected = [
            (
                91000,
                Some(70),
                DataType::F16,
                64,
                64,
                "sdpa backward sm version",
            ),
            (
                91100,
                Some(90),
                DataType::F16,
                192,
                96,
                "sdpa backward head dimension",
            ),
            (
                91100,
                Some(100),
                DataType::F16,
                192,
                96,
                "sdpa backward head dimension",
            ),
            (
                91100,
                Some(80),
                DataType::F16,
                136,
                128,
                "sdpa backward head dimension",
            ),
        ];
        for (version, sm_version, data_type, d_qk, d_v, expected_name) in rejected {
            assert_descriptor_mismatch(
                require_sdpa_backward_architecture_support(
                    version, sm_version, data_type, d_qk, d_v,
                ),
                expected_name,
            );
        }
    }

    #[test]
    fn sdpa_backward_ragged_support_rules_are_directly_tested() {
        assert_descriptor_mismatch(
            require_sdpa_backward_ragged_gqa_support(90500, true, 4, 2, 2),
            "sdpa backward ragged gqa",
        );
        require_sdpa_backward_ragged_gqa_support(90600, true, 4, 2, 2).unwrap();
        require_sdpa_backward_ragged_gqa_support(90500, false, 4, 2, 2).unwrap();

        assert_descriptor_mismatch(
            require_sdpa_backward_ragged_stats_support(91801, Some(80), true),
            "sdpa backward ragged stats",
        );
        assert_descriptor_mismatch(
            require_sdpa_backward_ragged_stats_support(91801, Some(120), true),
            "sdpa backward ragged stats",
        );
        require_sdpa_backward_ragged_stats_support(91801, Some(90), true).unwrap();
        require_sdpa_backward_ragged_stats_support(91801, Some(80), false).unwrap();
    }
}
