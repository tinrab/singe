use half::{bf16, f16};

use crate::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorId, TensorPlacement, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        error::{Error as FrontendError, SdpaFeature, SdpaRequirement},
        graph::Graph,
    },
    scalar::ScalarValue,
    utility::{check_range, to_i32},
};

/// Caps a nonnegative window at the largest observable distance before INT32 conversion.
pub fn bounded_sdpa_window(window: Option<i64>, maximum: i64, name: &str) -> Result<Option<i32>> {
    window
        .map(|window| {
            check_range!(name, window >= 0)?;
            to_i32(window.min(maximum), name)
        })
        .transpose()
}

pub const SDPA_SM_VERSION: &str = "sdpa sm version";
pub const SDPA_BACKWARD_SCALE: &str = "sdpa backward scale";
pub const SDPA_DROPOUT_OFFSET: &str = "sdpa dropout offset";
pub const SDPA_DROPOUT_PROBABILITY: &str = "sdpa dropout probability";
pub const SDPA_RNG_DUMP: &str = "sdpa rng dump";
pub const SDPA_BACKWARD_RNG_DUMP: &str = "sdpa backward rng dump";
pub const SDPA_RANK: &str = "sdpa rank";
pub const SDPA_Q_LAYOUT: &str = "sdpa q layout";
pub const SDPA_BIAS_LAYOUT: &str = "sdpa bias layout";
pub const SDPA_K_LAYOUT: &str = "sdpa k layout";
pub const SDPA_OUTPUT_LAYOUT: &str = "sdpa output layout";
pub const SDPA_DATA_TYPE: &str = "sdpa data type";
pub const SDPA_SCALE: &str = "sdpa scale";
pub const SDPA_OUTPUT_SHAPE: &str = "sdpa output shape";
pub const SDPA_STATS_SHAPE: &str = "sdpa stats shape";
pub const SDPA_STATS_DATA_TYPE: &str = "sdpa stats data type";
pub const SDPA_LOGIT_MAX_SHAPE: &str = "sdpa logit max shape";
pub const SDPA_SCORE_SUM_EXP_SHAPE: &str = "sdpa score sum exp shape";
pub const SDPA_PADDING_MASK_SEQ_LEN_DATA_TYPE: &str = "sdpa padding mask seq len data type";
pub const SDPA_SEQ_LEN_DATA_TYPE: &str = "sdpa seq len data type";
pub const SDPA_SEQ_LEN_SHAPE: &str = "sdpa seq len shape";
pub const SDPA_DROPOUT_SCALE: &str = "sdpa dropout scale";
pub const SDPA_ALIBI_DATA_TYPE: &str = "sdpa alibi data type";
pub const SDPA_ALIBI_SHAPE: &str = "sdpa alibi shape";
pub const SDPA_SINK_TOKEN_RANK: &str = "sdpa sink token rank";
pub const SDPA_SINK_TOKEN_SHAPE: &str = "sdpa sink token shape";
pub const SDPA_BLOCK_MASK_DATA_TYPE: &str = "sdpa block mask data type";
pub const SDPA_BLOCK_MASK_LAYOUT: &str = "sdpa block mask layout";
pub const SDPA_MAX_SEQUENCE_LENGTH_KEY_VALUE: &str = "sdpa max_sequence_length_key_value";
pub const SDPA_PAGED_K_PAGE_TABLE_DATA_TYPE: &str = "sdpa paged k page table data type";
pub const SDPA_PAGED_V_PAGE_TABLE_DATA_TYPE: &str = "sdpa paged v page table data type";
pub const SDPA_SCORE_SUBGRAPH: &str = "sdpa score subgraph";
pub const SDPA_BACKWARD_SCORE_SUBGRAPH: &str = "sdpa backward score subgraph";
pub const SDPA_BACKWARD_SCORE_SUBGRAPH_BPROP: &str = "sdpa backward score subgraph bprop";
pub const SDPA_BACKWARD_RANK: &str = "sdpa backward rank";
pub const SDPA_BACKWARD_STATS_SHAPE: &str = "sdpa backward stats shape";
pub const SDPA_BACKWARD_D_O_LAYOUT: &str = "sdpa backward d_o layout";
pub const SDPA_BACKWARD_D_O_SHAPE: &str = "sdpa backward d_o shape";
pub const SDPA_BACKWARD_D_Q_LAYOUT: &str = "sdpa backward d_q layout";
pub const SDPA_BACKWARD_D_Q_SHAPE: &str = "sdpa backward d_q shape";
pub const SDPA_BACKWARD_D_K_LAYOUT: &str = "sdpa backward d_k layout";
pub const SDPA_BACKWARD_D_K_SHAPE: &str = "sdpa backward d_k shape";
pub const SDPA_BACKWARD_D_V_LAYOUT: &str = "sdpa backward d_v layout";
pub const SDPA_BACKWARD_D_V_SHAPE: &str = "sdpa backward d_v shape";
pub const SDPA_BACKWARD_D_SINK_TOKEN_SHAPE: &str = "sdpa backward d_sink_token shape";
pub const SDPA_BACKWARD_D_BIAS_SHAPE: &str = "sdpa backward d_bias shape";
pub const SDPA_FP8_BACKWARD_ATTENTION_SCALE: &str = "sdpa fp8 backward attention_scale";
pub const SDPA_FP8_BACKWARD_SCORE_SUBGRAPH: &str = "sdpa fp8 backward score subgraph";

pub const SDPA_FP8_BACKWARD_SCORE_SUBGRAPH_BPROP: &str = "sdpa fp8 backward score subgraph bprop";

pub const SDPA_MXFP8_BACKWARD_ATTENTION_SCALE: &str = "sdpa mxfp8 backward attention_scale";
pub const SDPA_MXFP8_BACKWARD_Q_LAYOUT: &str = "sdpa mxfp8 backward q layout";
pub const SDPA_MXFP8_BACKWARD_Q_T_LAYOUT: &str = "sdpa mxfp8 backward q_t layout";
pub const SDPA_MXFP8_BACKWARD_K_LAYOUT: &str = "sdpa mxfp8 backward k layout";
pub const SDPA_MXFP8_BACKWARD_K_T_LAYOUT: &str = "sdpa mxfp8 backward k_t layout";
pub const SDPA_MXFP8_BACKWARD_V_LAYOUT: &str = "sdpa mxfp8 backward v layout";
pub const SDPA_MXFP8_BACKWARD_O_LAYOUT: &str = "sdpa mxfp8 backward o layout";
pub const SDPA_MXFP8_BACKWARD_D_O_LAYOUT: &str = "sdpa mxfp8 backward d_o layout";

pub const SDPA_MXFP8_BACKWARD_D_O_QUANTIZED_LAYOUT: &str =
    "sdpa mxfp8 backward d_o_quantized layout";

pub const SDPA_MXFP8_BACKWARD_D_O_T_LAYOUT: &str = "sdpa mxfp8 backward d_o_t layout";
pub const SDPA_MXFP8_BACKWARD_Q_T_SHAPE: &str = "sdpa mxfp8 backward q_t shape";
pub const SDPA_MXFP8_BACKWARD_K_T_SHAPE: &str = "sdpa mxfp8 backward k_t shape";

pub const SDPA_MXFP8_BACKWARD_D_O_QUANTIZED_SHAPE: &str = "sdpa mxfp8 backward d_o_quantized shape";

pub const SDPA_MXFP8_BACKWARD_D_O_T_SHAPE: &str = "sdpa mxfp8 backward d_o_t shape";
pub const SDPA_MXFP8_BACKWARD_O_SHAPE: &str = "sdpa mxfp8 backward o shape";
pub const SDPA_MXFP8_BACKWARD_STATS_SHAPE: &str = "sdpa mxfp8 backward stats shape";
pub const SDPA_MXFP8_BACKWARD_STATS_DATA_TYPE: &str = "sdpa mxfp8 backward stats data type";
pub const SDPA_MXFP8_BACKWARD_SCORE_SUBGRAPH: &str = "sdpa mxfp8 backward score subgraph";

pub const SDPA_MXFP8_BACKWARD_SCORE_SUBGRAPH_BPROP: &str =
    "sdpa mxfp8 backward score subgraph bprop";

pub const SDPA_MXFP8_BACKWARD_SCALE_Q: &str = "sdpa mxfp8 backward scale_q";
pub const SDPA_MXFP8_BACKWARD_SCALE_Q_T: &str = "sdpa mxfp8 backward scale_q_t";
pub const SDPA_MXFP8_BACKWARD_SCALE_K: &str = "sdpa mxfp8 backward scale_k";
pub const SDPA_MXFP8_BACKWARD_SCALE_K_T: &str = "sdpa mxfp8 backward scale_k_t";
pub const SDPA_MXFP8_BACKWARD_SCALE_V: &str = "sdpa mxfp8 backward scale_v";
pub const SDPA_MXFP8_BACKWARD_SCALE_D_O: &str = "sdpa mxfp8 backward scale_d_o";
pub const SDPA_MXFP8_BACKWARD_SCALE_D_O_T: &str = "sdpa mxfp8 backward scale_d_o_t";

pub const SDPA_MXFP8_BACKWARD_SCALE_D_O_T_SHAPE: &str = "sdpa mxfp8 backward scale_d_o_t shape";

pub const SDPA_MXFP8_BACKWARD_SCALE_Q_SHAPE: &str = "sdpa mxfp8 backward scale_q shape";
pub const SDPA_MXFP8_BACKWARD_SCALE_Q_T_SHAPE: &str = "sdpa mxfp8 backward scale_q_t shape";
pub const SDPA_MXFP8_BACKWARD_SCALE_K_SHAPE: &str = "sdpa mxfp8 backward scale_k shape";
pub const SDPA_MXFP8_BACKWARD_SCALE_K_T_SHAPE: &str = "sdpa mxfp8 backward scale_k_t shape";
pub const SDPA_MXFP8_BACKWARD_SCALE_V_SHAPE: &str = "sdpa mxfp8 backward scale_v shape";
pub const SDPA_MXFP8_BACKWARD_SCALE_D_O_SHAPE: &str = "sdpa mxfp8 backward scale_d_o shape";
pub const SDPA_DROPOUT_SEED: &str = "sdpa dropout seed";
pub const SDPA_RNG_OFFSET: &str = "sdpa rng offset";
pub const SDPA_RNG_OFFSET_DATA_TYPE: &str = "sdpa rng offset data type";
pub const SDPA_RNG_SEED: &str = "sdpa rng seed";
pub const SDPA_RNG_SEED_DATA_TYPE: &str = "sdpa rng seed data type";

pub fn sdpa_mask_scalar(data_type: DataType, value: f32) -> Result<TensorSpec> {
    let data_type = match data_type {
        DataType::F16 | DataType::BF16 => data_type,
        _ => DataType::F32,
    };
    sdpa_scalar_tensor(data_type, value)
}

pub fn sdpa_scalar_tensor(data_type: DataType, value: f32) -> Result<TensorSpec> {
    let value = match data_type {
        DataType::F32 => ScalarValue::F32(value),
        DataType::F64 => ScalarValue::F64(f64::from(value)),
        DataType::F16 => ScalarValue::f16_bits(f16::from_f32(value).to_bits()),
        DataType::BF16 => ScalarValue::bf16_bits(bf16::from_f32(value).to_bits()),
        _ => return Err(FrontendError::SdpaScalarDataTypeUnsupported { data_type }.into()),
    };
    Ok(TensorSpec::constant(value).with_shape(Shape::contiguous([1, 1, 1, 1])?))
}

impl Graph {
    pub(crate) fn sdpa_effective_scale(
        &mut self,
        scale: TensorId,
        attention_scale: Option<f32>,
        compute_type: DataType,
    ) -> Result<TensorId> {
        self.validate_tensor_scalar_shape(scale, SDPA_SCALE)?;
        let Some(attention_scale) = attention_scale else {
            return Ok(scale);
        };

        let scale_data_type = self.tensor_spec(scale)?.data_type;
        if let TensorPlacement::Constant(scalar_value) = self.tensor_spec(scale)?.placement {
            let combined = match scalar_value {
                ScalarValue::F32(value) => Some(ScalarValue::F32(value * attention_scale)),
                ScalarValue::F64(value) => {
                    Some(ScalarValue::F64(value * f64::from(attention_scale)))
                }
                ScalarValue::F16(bits) => {
                    let value = f16::from_bits(bits).to_f32() * attention_scale;
                    Some(ScalarValue::f16_bits(f16::from_f32(value).to_bits()))
                }
                ScalarValue::Bf16(bits) => {
                    let value = bf16::from_bits(bits).to_f32() * attention_scale;
                    Some(ScalarValue::bf16_bits(bf16::from_f32(value).to_bits()))
                }
                _ => None,
            };
            if let Some(combined) = combined {
                return Ok(self.tensor(
                    TensorSpec::new(scale_data_type, Shape::contiguous([1, 1, 1, 1])?)
                        .with_constant_value(combined)?,
                ));
            }
        }
        let inline_scale = self.tensor(sdpa_scalar_tensor(scale_data_type, attention_scale)?);
        self.sdpa_binary(
            PointwiseMode::Mul,
            scale,
            inline_scale,
            scale_data_type,
            compute_type,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaSupportContext {
    cudnn_version: u64,
    sm_version: Option<i32>,
}

impl SdpaSupportContext {
    pub(crate) const fn new(cudnn_version: u64, sm_version: Option<i32>) -> Self {
        Self {
            cudnn_version,
            sm_version,
        }
    }

    pub(crate) const fn backward(self) -> SdpaBackwardSupport {
        SdpaBackwardSupport::new(self.cudnn_version, self.sm_version)
    }

    pub(crate) fn require_forward_architecture(
        self,
        data_type: DataType,
        d_qk: i64,
        d_v: i64,
    ) -> Result<()> {
        if matches!(data_type, DataType::F16 | DataType::BF16)
            && self.sm_version.is_some_and(|sm| sm < 80)
        {
            return Err(FrontendError::SdpaArchitectureUnsupported {
                operation: "sdpa",
                reason: SDPA_SM_VERSION,
                cudnn_version: self.cudnn_version,
                sm_version: self.sm_version,
                data_type,
                d_qk,
                d_v,
            }
            .into());
        }
        Ok(())
    }
}

/// Mask alignment restrictions that still apply at the cuDNN 9.27 baseline.
pub fn require_sdpa_forward_mask_lengths(
    query_sequence_length: i64,
    key_sequence_length: i64,
    has_padding_mask: bool,
    has_sliding_window: bool,
    has_causal_bottom_right: bool,
) -> Result<()> {
    if has_padding_mask || query_sequence_length <= key_sequence_length {
        return Ok(());
    }
    if has_sliding_window {
        return Err(FrontendError::SdpaSlidingWindowSequenceLengthUnsupported {
            query_sequence_length,
            key_sequence_length,
        }
        .into());
    }
    if has_causal_bottom_right {
        return Err(
            FrontendError::SdpaCausalBottomRightSequenceLengthUnsupported {
                query_sequence_length,
                key_sequence_length,
            }
            .into(),
        );
    }
    Ok(())
}

#[expect(
    clippy::fn_params_excessive_bools,
    reason = "the support predicate combines independent native attention features"
)]
pub fn requires_sdpa_ragged_mask_or_subgraph(
    is_ragged: bool,
    has_padding_mask: bool,
    has_additive_mask: bool,
    has_score_subgraph: bool,
) -> bool {
    is_ragged && !has_padding_mask && !has_additive_mask && !has_score_subgraph
}

#[expect(
    clippy::fn_params_excessive_bools,
    reason = "the support predicate combines independent native attention features"
)]
pub fn require_sdpa_ragged_mask_or_subgraph(
    is_ragged: bool,
    has_padding_mask: bool,
    has_additive_mask: bool,
    has_score_subgraph: bool,
    operation: &'static str,
) -> Result<()> {
    if requires_sdpa_ragged_mask_or_subgraph(
        is_ragged,
        has_padding_mask,
        has_additive_mask,
        has_score_subgraph,
    ) {
        return Err(FrontendError::SdpaRequires {
            operation,
            requirement: SdpaRequirement::MaskOrScoreSubgraphForRaggedOffsets,
        }
        .into());
    }
    Ok(())
}

#[expect(
    clippy::fn_params_excessive_bools,
    reason = "the support predicate combines independent native attention features"
)]
pub fn requires_sdpa_sequence_lengths_mask_or_subgraph(
    has_sequence_lengths: bool,
    has_padding_mask: bool,
    has_causal_bottom_right: bool,
    has_score_subgraph: bool,
) -> bool {
    has_sequence_lengths && !has_padding_mask && !has_causal_bottom_right && !has_score_subgraph
}

#[expect(
    clippy::fn_params_excessive_bools,
    reason = "the support predicate combines independent native attention features"
)]
pub fn require_sdpa_sequence_lengths_mask_or_subgraph(
    has_sequence_lengths: bool,
    has_padding_mask: bool,
    has_causal_bottom_right: bool,
    has_score_subgraph: bool,
    operation: &'static str,
) -> Result<()> {
    if requires_sdpa_sequence_lengths_mask_or_subgraph(
        has_sequence_lengths,
        has_padding_mask,
        has_causal_bottom_right,
        has_score_subgraph,
    ) {
        return Err(FrontendError::SdpaRequires {
            operation,
            requirement: SdpaRequirement::MaskOrScoreSubgraphForSequenceLengths,
        }
        .into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaBackwardSupport {
    pub(crate) cudnn_version: u64,
    pub(crate) sm_version: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaBackwardCausalBottomRightSupport {
    pub(crate) enabled: bool,
    pub(crate) query_sequence_length: i64,
    pub(crate) key_sequence_length: i64,
    pub(crate) has_dropout: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaBackwardBiasGradientSupport {
    pub(crate) enabled: bool,
    pub(crate) is_ragged: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "native capabilities and tensor properties are independent flags"
)]
pub struct SdpaBackwardDeterministicSupport {
    pub(crate) enabled: bool,
    pub(crate) has_bias_gradient: bool,
    pub(crate) has_dropout: bool,
    pub(crate) has_alibi: bool,
    pub(crate) is_ragged: bool,
}

impl SdpaBackwardSupport {
    pub(crate) const fn new(cudnn_version: u64, sm_version: Option<i32>) -> Self {
        Self {
            cudnn_version,
            sm_version,
        }
    }

    pub(crate) fn rejects_ragged_stats(self, has_ragged_stats: bool) -> bool {
        has_ragged_stats
            && self
                .sm_version
                .is_some_and(|sm_version| matches!(sm_version / 10, 8 | 12))
    }

    pub(crate) fn require_causal_bottom_right(
        self,
        rule: SdpaBackwardCausalBottomRightSupport,
    ) -> Result<()> {
        if !rule.enabled {
            return Ok(());
        }
        if self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 < 10)
        {
            return Err(FrontendError::SdpaUnsupported {
                sm_version: self.sm_version,
                feature: SdpaFeature::CausalBottomRight,
                operation: "sdpa backward",
            }
            .into());
        }
        if rule.query_sequence_length > rule.key_sequence_length {
            return Err(
                FrontendError::SdpaBackwardCausalBottomRightSequenceLengthUnsupported {
                    query_sequence_length: rule.query_sequence_length,
                    key_sequence_length: rule.key_sequence_length,
                }
                .into(),
            );
        }
        if rule.has_dropout {
            return Err(FrontendError::SdpaUnsupported {
                feature: SdpaFeature::Dropout,
                operation: "sdpa backward causal bottom-right",
                sm_version: None,
            }
            .into());
        }
        if rule.query_sequence_length % 64 != 0 || rule.key_sequence_length % 64 != 0 {
            return Err(
                FrontendError::SdpaBackwardCausalBottomRightSequenceLengthAlignmentUnsupported {
                    query_sequence_length: rule.query_sequence_length,
                    key_sequence_length: rule.key_sequence_length,
                }
                .into(),
            );
        }
        Ok(())
    }

    pub(crate) fn require_bias_gradient(self, rule: SdpaBackwardBiasGradientSupport) -> Result<()> {
        if !rule.enabled {
            return Ok(());
        }
        if self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 == 10)
            && rule.is_ragged
        {
            return Err(FrontendError::SdpaUnsupported {
                sm_version: self.sm_version,
                feature: SdpaFeature::RaggedTensors,
                operation: "sdpa backward bias gradient",
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn require_deterministic_algorithm(
        self,
        rule: SdpaBackwardDeterministicSupport,
    ) -> Result<()> {
        if !rule.enabled {
            return Ok(());
        }
        if self.sm_version.is_some_and(|sm| sm / 10 == 10)
            && (rule.has_bias_gradient || rule.has_dropout || rule.has_alibi)
        {
            return Err(FrontendError::SdpaUnsupported {
                sm_version: self.sm_version,
                feature: SdpaFeature::BiasDropoutOrAlibi,
                operation: "sdpa backward deterministic algorithm",
            }
            .into());
        }
        if rule.is_ragged && self.rejects_ragged_stats(true) {
            return Err(FrontendError::SdpaUnsupported {
                sm_version: self.sm_version,
                feature: SdpaFeature::RaggedTensors,
                operation: "sdpa backward deterministic algorithm",
            }
            .into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedSdpaSupport {
    sm_version: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MxFp8BackwardSupport {
    pub(crate) output_io_type: DataType,
    pub(crate) is_ragged: bool,
    pub(crate) has_generated_dropout: bool,
    pub(crate) has_random_number_generator_dump: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "native capabilities and tensor properties are independent flags"
)]
pub struct Fp8BackwardSupport {
    pub(crate) output_io_type: DataType,
    pub(crate) d_qk: i64,
    pub(crate) d_v: i64,
    pub(crate) deterministic_algorithm: bool,
    pub(crate) is_ragged: bool,
    pub(crate) has_generated_dropout: bool,
    pub(crate) has_random_number_generator_dump: bool,
}

impl QuantizedSdpaSupport {
    pub(crate) const fn new(sm_version: Option<i32>) -> Self {
        Self { sm_version }
    }

    pub(crate) fn require_mxfp8_backward(self, rule: MxFp8BackwardSupport) -> Result<()> {
        self.require_quantized_backward_architecture("mxfp8 sdpa backward")?;
        self.require_quantized_backward_ragged("mxfp8 sdpa backward", rule.is_ragged)?;
        self.require_fp8_output_io_type("mxfp8 sdpa backward", rule.output_io_type)?;
        if self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 == 10)
            && rule.has_generated_dropout
        {
            return Err(FrontendError::SdpaUnsupported {
                operation: "mxfp8 sdpa backward",
                sm_version: self.sm_version,
                feature: SdpaFeature::GeneratedDropout,
            }
            .into());
        }
        Self::require_quantized_backward_no_rng_dump(
            "mxfp8 sdpa backward",
            rule.has_random_number_generator_dump,
        )
    }

    pub(crate) fn require_fp8_backward(self, rule: Fp8BackwardSupport) -> Result<()> {
        self.require_quantized_backward_architecture("fp8 sdpa backward")?;
        self.require_quantized_backward_ragged("fp8 sdpa backward", rule.is_ragged)?;
        self.require_fp8_output_io_type("fp8 sdpa backward", rule.output_io_type)?;

        let requires_deterministic_algorithm = self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 == 10)
            && (rule.deterministic_algorithm || (rule.d_qk <= 192 && rule.d_v <= 128));
        if requires_deterministic_algorithm && rule.has_generated_dropout {
            return Err(FrontendError::SdpaUnsupported {
                operation: "fp8 sdpa backward",
                sm_version: self.sm_version,
                feature: SdpaFeature::GeneratedDropout,
            }
            .into());
        }
        Self::require_quantized_backward_no_rng_dump(
            "fp8 sdpa backward",
            rule.has_random_number_generator_dump,
        )
    }

    pub(crate) fn require_fp8_forward_output_type(self, output_io_type: DataType) -> Result<()> {
        self.require_fp8_output_io_type("fp8 sdpa", output_io_type)
    }

    pub(crate) fn require_quantized_backward_head_dimensions(
        self,
        d_qk: i64,
        d_v: i64,
        operation: &'static str,
    ) -> Result<()> {
        let Some(sm_version) = self.sm_version else {
            return Ok(());
        };
        let supported = if sm_version / 10 >= 10 {
            ((d_qk <= 128) && (d_qk % 16 == 0) && (d_v <= 128) && (d_v % 16 == 0))
                || (d_qk == 192 && d_v == 128)
        } else {
            d_qk == 128 && d_v == 128 && d_qk % 16 == 0 && d_v % 16 == 0
        };
        if !supported {
            return Err(
                FrontendError::SdpaQuantizedBackwardHeadDimensionUnsupported {
                    operation,
                    d_qk,
                    d_v,
                    sm_version: self.sm_version,
                }
                .into(),
            );
        }
        Ok(())
    }

    pub(crate) fn require_quantized_backward_output_ragged_support(
        self,
        has_ragged_output: bool,
        operation: &'static str,
    ) -> Result<()> {
        if self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 == 9)
            && has_ragged_output
        {
            return Err(FrontendError::SdpaUnsupported {
                operation,
                sm_version: self.sm_version,
                feature: SdpaFeature::RaggedOutputs,
            }
            .into());
        }
        Ok(())
    }

    fn require_fp8_output_io_type(
        self,
        operation: &'static str,
        output_io_type: DataType,
    ) -> Result<()> {
        if matches!(output_io_type, DataType::F16 | DataType::BF16)
            && self.sm_version.is_some_and(|sm| sm < 100)
        {
            return Err(FrontendError::SdpaQuantizedOutputDataTypeRequiresSm100 {
                operation,
                output_io_type,
                sm_version: self.sm_version,
            }
            .into());
        }
        Ok(())
    }

    fn require_quantized_backward_architecture(self, operation: &'static str) -> Result<()> {
        if self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 < 9)
        {
            return Err(FrontendError::SdpaUnsupported {
                operation,
                sm_version: self.sm_version,
                feature: SdpaFeature::TargetArchitecture,
            }
            .into());
        }
        Ok(())
    }

    fn require_quantized_backward_ragged(
        self,
        operation: &'static str,
        is_ragged: bool,
    ) -> Result<()> {
        if self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 == 9)
            && is_ragged
        {
            return Err(FrontendError::SdpaUnsupported {
                operation,
                sm_version: self.sm_version,
                feature: SdpaFeature::RaggedTensors,
            }
            .into());
        }
        Ok(())
    }

    fn require_quantized_backward_no_rng_dump(
        operation: &'static str,
        has_random_number_generator_dump: bool,
    ) -> Result<()> {
        if has_random_number_generator_dump {
            return Err(FrontendError::SdpaUnsupported {
                operation,
                feature: SdpaFeature::RngDump,
                sm_version: None,
            }
            .into());
        }
        Ok(())
    }
}
