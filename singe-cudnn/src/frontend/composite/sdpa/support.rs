use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::graph::Graph,
    pointwise::PointwiseMode,
    scalar::ScalarValue,
    tensor::{Shape, TensorId, TensorSpec},
};

pub(in crate::frontend::composite::sdpa) const SDPA_DROPOUT_OFFSET: &str = "sdpa dropout offset";
pub(in crate::frontend::composite::sdpa) const SDPA_DROPOUT_PROBABILITY: &str =
    "sdpa dropout probability";
pub(in crate::frontend::composite::sdpa) const SDPA_RNG_DUMP: &str = "sdpa rng dump";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_RNG_DUMP: &str =
    "sdpa backward rng dump";
pub(in crate::frontend::composite::sdpa) const SDPA_RANK: &str = "sdpa rank";
pub(in crate::frontend::composite::sdpa) const SDPA_Q_LAYOUT: &str = "sdpa q layout";
pub(in crate::frontend::composite::sdpa) const SDPA_K_LAYOUT: &str = "sdpa k layout";
pub(in crate::frontend::composite::sdpa) const SDPA_OUTPUT_LAYOUT: &str = "sdpa output layout";
pub(in crate::frontend::composite::sdpa) const SDPA_DATA_TYPE: &str = "sdpa data type";
pub(in crate::frontend::composite::sdpa) const SDPA_SCALE: &str = "sdpa scale";
pub(in crate::frontend::composite::sdpa) const SDPA_OUTPUT_SHAPE: &str = "sdpa output shape";
pub(in crate::frontend::composite::sdpa) const SDPA_STATS: &str = "sdpa stats";
pub(in crate::frontend::composite::sdpa) const SDPA_STATS_SHAPE: &str = "sdpa stats shape";
pub(in crate::frontend::composite::sdpa) const SDPA_STATS_DATA_TYPE: &str = "sdpa stats data type";
pub(in crate::frontend::composite::sdpa) const SDPA_LOGIT_MAX: &str = "sdpa logit max";
pub(in crate::frontend::composite::sdpa) const SDPA_LOGIT_MAX_SHAPE: &str = "sdpa logit max shape";
pub(in crate::frontend::composite::sdpa) const SDPA_LOGIT_MAX_DATA_TYPE: &str =
    "sdpa logit max data type";
pub(in crate::frontend::composite::sdpa) const SDPA_SCORE_SUM_EXP: &str = "sdpa score sum exp";
pub(in crate::frontend::composite::sdpa) const SDPA_SCORE_SUM_EXP_SHAPE: &str =
    "sdpa score sum exp shape";
pub(in crate::frontend::composite::sdpa) const SDPA_SCORE_SUM_EXP_DATA_TYPE: &str =
    "sdpa score sum exp data type";
pub(in crate::frontend::composite::sdpa) const SDPA_BOTTOM_RIGHT_SEQ_LEN_DATA_TYPE: &str =
    "sdpa bottom right seq len data type";
pub(in crate::frontend::composite::sdpa) const SDPA_PADDING_MASK_SEQ_LEN_DATA_TYPE: &str =
    "sdpa padding mask seq len data type";
pub(in crate::frontend::composite::sdpa) const SDPA_SEQ_LEN_DATA_TYPE: &str =
    "sdpa seq len data type";
pub(in crate::frontend::composite::sdpa) const SDPA_SEQ_LEN_SHAPE: &str = "sdpa seq len shape";
pub(in crate::frontend::composite::sdpa) const SDPA_DROPOUT_SCALE: &str = "sdpa dropout scale";
pub(in crate::frontend::composite::sdpa) const SDPA_ALIBI_DATA_TYPE: &str = "sdpa alibi data type";
pub(in crate::frontend::composite::sdpa) const SDPA_ALIBI_SHAPE: &str = "sdpa alibi shape";
pub(in crate::frontend::composite::sdpa) const SDPA_SINK_TOKEN_DATA_TYPE: &str =
    "sdpa sink token data type";
pub(in crate::frontend::composite::sdpa) const SDPA_SINK_TOKEN_RANK: &str = "sdpa sink token rank";
pub(in crate::frontend::composite::sdpa) const SDPA_SINK_TOKEN_SHAPE: &str =
    "sdpa sink token shape";
pub(in crate::frontend::composite::sdpa) const SDPA_BLOCK_MASK_DATA_TYPE: &str =
    "sdpa block mask data type";
pub(in crate::frontend::composite::sdpa) const SDPA_BLOCK_MASK_LAYOUT: &str =
    "sdpa block mask layout";
pub(in crate::frontend::composite::sdpa) const SDPA_MAX_SEQUENCE_LENGTH_KEY_VALUE: &str =
    "sdpa max_sequence_length_key_value";
pub(in crate::frontend::composite::sdpa) const SDPA_PAGED_K_PAGE_TABLE_DATA_TYPE: &str =
    "sdpa paged k page table data type";
pub(in crate::frontend::composite::sdpa) const SDPA_PAGED_V_PAGE_TABLE_DATA_TYPE: &str =
    "sdpa paged v page table data type";
pub(in crate::frontend::composite::sdpa) const SDPA_SCORE_SUBGRAPH: &str = "sdpa score subgraph";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_SCORE_SUBGRAPH: &str =
    "sdpa backward score subgraph";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_SCORE_SUBGRAPH_BPROP: &str =
    "sdpa backward score subgraph bprop";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_RANK: &str = "sdpa backward rank";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_DATA_TYPE: &str =
    "sdpa backward data type";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_STATS_DATA_TYPE: &str =
    "sdpa backward stats data type";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_STATS_SHAPE: &str =
    "sdpa backward stats shape";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_D_O_LAYOUT: &str =
    "sdpa backward d_o layout";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_D_O_SHAPE: &str =
    "sdpa backward d_o shape";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_D_Q_LAYOUT: &str =
    "sdpa backward d_q layout";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_D_Q_SHAPE: &str =
    "sdpa backward d_q shape";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_D_K_LAYOUT: &str =
    "sdpa backward d_k layout";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_D_K_SHAPE: &str =
    "sdpa backward d_k shape";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_D_V_LAYOUT: &str =
    "sdpa backward d_v layout";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_D_V_SHAPE: &str =
    "sdpa backward d_v shape";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_D_SINK_TOKEN_SHAPE: &str =
    "sdpa backward d_sink_token shape";
pub(in crate::frontend::composite::sdpa) const SDPA_BACKWARD_D_BIAS_SHAPE: &str =
    "sdpa backward d_bias shape";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_AUX_OUTPUTS: &str = "sdpa fp8 aux outputs";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_ATTN_SCALE: &str = "sdpa fp8 attn scale";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_DESCALE_Q: &str = "sdpa fp8 descale_q";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_DESCALE_K: &str = "sdpa fp8 descale_k";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_DESCALE_V: &str = "sdpa fp8 descale_v";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_DESCALE_S: &str = "sdpa fp8 descale_s";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_SCALE_S: &str = "sdpa fp8 scale_s";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_SCALE_O: &str = "sdpa fp8 scale_o";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_RNG_DUMP: &str = "sdpa fp8 rng dump";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_OUTPUT: &str = "sdpa fp8 output";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_STATS: &str = "sdpa fp8 stats";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_ABSOLUTE_MAX_S: &str =
    "sdpa fp8 absolute_max_s";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_ABSOLUTE_MAX_O: &str =
    "sdpa fp8 absolute_max_o";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_D_Q: &str =
    "sdpa fp8 backward d_q";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_D_K: &str =
    "sdpa fp8 backward d_k";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_D_V: &str =
    "sdpa fp8 backward d_v";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_D_SINK_TOKEN: &str =
    "sdpa fp8 backward d_sink_token";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_ABSOLUTE_MAX_D_Q: &str =
    "sdpa fp8 backward absolute_max_d_q";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_ABSOLUTE_MAX_D_K: &str =
    "sdpa fp8 backward absolute_max_d_k";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_ABSOLUTE_MAX_D_V: &str =
    "sdpa fp8 backward absolute_max_d_v";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_ABSOLUTE_MAX_D_P: &str =
    "sdpa fp8 backward absolute_max_d_p";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_ATTENTION_SCALE: &str =
    "sdpa fp8 backward attention_scale";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_SCORE_SUBGRAPH: &str =
    "sdpa fp8 backward score subgraph";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_SCORE_SUBGRAPH_BPROP: &str =
    "sdpa fp8 backward score subgraph bprop";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_DESCALE_Q: &str =
    "sdpa fp8 backward descale_q";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_DESCALE_K: &str =
    "sdpa fp8 backward descale_k";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_DESCALE_V: &str =
    "sdpa fp8 backward descale_v";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_DESCALE_O: &str =
    "sdpa fp8 backward descale_o";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_DESCALE_D_O: &str =
    "sdpa fp8 backward descale_d_o";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_DESCALE_S: &str =
    "sdpa fp8 backward descale_s";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_DESCALE_D_P: &str =
    "sdpa fp8 backward descale_d_p";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_SCALE_S: &str =
    "sdpa fp8 backward scale_s";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_SCALE_D_Q: &str =
    "sdpa fp8 backward scale_d_q";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_SCALE_D_K: &str =
    "sdpa fp8 backward scale_d_k";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_SCALE_D_V: &str =
    "sdpa fp8 backward scale_d_v";
pub(in crate::frontend::composite::sdpa) const SDPA_FP8_BACKWARD_SCALE_D_P: &str =
    "sdpa fp8 backward scale_d_p";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_ABSOLUTE_MAX_S: &str =
    "sdpa mxfp8 absolute_max_s";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_AUX_OUTPUTS: &str =
    "sdpa mxfp8 aux outputs";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_ATTN_SCALE: &str =
    "sdpa mxfp8 attn scale";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_OUTPUT: &str = "sdpa mxfp8 output";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_STATS: &str = "sdpa mxfp8 stats";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_ABSOLUTE_MAX_O: &str =
    "sdpa mxfp8 absolute_max_o";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_ATTENTION_SCALE: &str =
    "sdpa mxfp8 backward attention_scale";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_Q_LAYOUT: &str =
    "sdpa mxfp8 backward q layout";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_Q_T_LAYOUT: &str =
    "sdpa mxfp8 backward q_t layout";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_K_LAYOUT: &str =
    "sdpa mxfp8 backward k layout";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_K_T_LAYOUT: &str =
    "sdpa mxfp8 backward k_t layout";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_V_LAYOUT: &str =
    "sdpa mxfp8 backward v layout";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_O_LAYOUT: &str =
    "sdpa mxfp8 backward o layout";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_D_O_LAYOUT: &str =
    "sdpa mxfp8 backward d_o layout";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_D_O_QUANTIZED_LAYOUT: &str =
    "sdpa mxfp8 backward d_o_quantized layout";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_D_O_T_LAYOUT: &str =
    "sdpa mxfp8 backward d_o_t layout";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_Q_T_SHAPE: &str =
    "sdpa mxfp8 backward q_t shape";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_K_T_SHAPE: &str =
    "sdpa mxfp8 backward k_t shape";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_D_O_QUANTIZED_SHAPE: &str =
    "sdpa mxfp8 backward d_o_quantized shape";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_D_O_T_SHAPE: &str =
    "sdpa mxfp8 backward d_o_t shape";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_O_SHAPE: &str =
    "sdpa mxfp8 backward o shape";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_STATS_SHAPE: &str =
    "sdpa mxfp8 backward stats shape";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_STATS_DATA_TYPE: &str =
    "sdpa mxfp8 backward stats data type";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_D_Q: &str =
    "sdpa mxfp8 backward d_q";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_D_K: &str =
    "sdpa mxfp8 backward d_k";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_D_V: &str =
    "sdpa mxfp8 backward d_v";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_ABSOLUTE_MAX_D_Q: &str =
    "sdpa mxfp8 backward absolute_max_d_q";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_ABSOLUTE_MAX_D_K: &str =
    "sdpa mxfp8 backward absolute_max_d_k";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_ABSOLUTE_MAX_D_V: &str =
    "sdpa mxfp8 backward absolute_max_d_v";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCORE_SUBGRAPH: &str =
    "sdpa mxfp8 backward score subgraph";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCORE_SUBGRAPH_BPROP: &str =
    "sdpa mxfp8 backward score subgraph bprop";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_Q: &str =
    "sdpa mxfp8 backward scale_q";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_Q_T: &str =
    "sdpa mxfp8 backward scale_q_t";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_K: &str =
    "sdpa mxfp8 backward scale_k";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_K_T: &str =
    "sdpa mxfp8 backward scale_k_t";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_V: &str =
    "sdpa mxfp8 backward scale_v";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_D_O: &str =
    "sdpa mxfp8 backward scale_d_o";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_D_O_T: &str =
    "sdpa mxfp8 backward scale_d_o_t";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_D_O_T_SHAPE: &str =
    "sdpa mxfp8 backward scale_d_o_t shape";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_Q_SHAPE: &str =
    "sdpa mxfp8 backward scale_q shape";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_Q_T_SHAPE: &str =
    "sdpa mxfp8 backward scale_q_t shape";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_K_SHAPE: &str =
    "sdpa mxfp8 backward scale_k shape";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_K_T_SHAPE: &str =
    "sdpa mxfp8 backward scale_k_t shape";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_V_SHAPE: &str =
    "sdpa mxfp8 backward scale_v shape";
pub(in crate::frontend::composite::sdpa) const SDPA_MXFP8_BACKWARD_SCALE_D_O_SHAPE: &str =
    "sdpa mxfp8 backward scale_d_o shape";
pub(in crate::frontend::composite::sdpa) const SDPA_DROPOUT_SEED: &str = "sdpa dropout seed";
pub(in crate::frontend::composite::sdpa) const SDPA_RNG_OFFSET: &str = "rng offset";
pub(in crate::frontend::composite::sdpa) const SDPA_RNG_OFFSET_DATA_TYPE: &str =
    "rng offset data type";
pub(in crate::frontend::composite::sdpa) const SDPA_RNG_SEED: &str = "rng seed";
pub(in crate::frontend::composite::sdpa) const SDPA_RNG_SEED_DATA_TYPE: &str = "rng seed data type";
pub(in crate::frontend::composite::sdpa) const SDPA_SOFTMAX_P: &str = "sdpa softmax p";
pub(in crate::frontend::composite::sdpa) const SDPA_SOFTMAX_S: &str = "sdpa softmax s";

pub(super) fn sdpa_mask_scalar(data_type: DataType, value: f32) -> Result<TensorSpec> {
    let tensor = TensorSpec::new(data_type, Shape::contiguous([1, 1, 1, 1])?);
    match data_type {
        DataType::F32 => tensor.with_scalar_value(ScalarValue::F32(value)),
        DataType::F16 => tensor.with_scalar_value(ScalarValue::f16_bits(if value == 0.0 {
            0x0000
        } else {
            0xfc00
        })),
        DataType::BF16 => tensor.with_scalar_value(ScalarValue::bf16_bits(if value == 0.0 {
            0x0000
        } else {
            0xff80
        })),
        _ => TensorSpec::scalar_f32(value),
    }
}

pub(super) fn sdpa_scalar_tensor(data_type: DataType, value: f32) -> Result<TensorSpec> {
    let tensor = TensorSpec::new(data_type, Shape::contiguous([1, 1, 1, 1])?);
    match data_type {
        DataType::F32 => tensor.with_scalar_value(ScalarValue::F32(value)),
        DataType::F16 => {
            tensor.with_scalar_value(ScalarValue::f16_bits(half::f16::from_f32(value).to_bits()))
        }
        DataType::BF16 => tensor.with_scalar_value(ScalarValue::bf16_bits(
            half::bf16::from_f32(value).to_bits(),
        )),
        _ => Err(Error::FrontendSdpaScalarDataTypeUnsupported { data_type }),
    }
}

impl Graph {
    pub(super) fn sdpa_effective_scale(
        &mut self,
        scale: TensorId,
        attention_scale: Option<f32>,
    ) -> Result<TensorId> {
        let Some(attention_scale) = attention_scale else {
            return Ok(scale);
        };

        let scale_data_type = self.tensor_config(scale)?.data_type;
        if let Some(scalar_value) = self.tensor_config(scale)?.scalar_value {
            let combined = match scalar_value {
                ScalarValue::F32(value) => Some(ScalarValue::F32(value * attention_scale)),
                ScalarValue::F16(bits) => {
                    let value = half::f16::from_bits(bits).to_f32() * attention_scale;
                    Some(ScalarValue::f16_bits(half::f16::from_f32(value).to_bits()))
                }
                ScalarValue::Bf16(bits) => {
                    let value = half::bf16::from_bits(bits).to_f32() * attention_scale;
                    Some(ScalarValue::bf16_bits(
                        half::bf16::from_f32(value).to_bits(),
                    ))
                }
                _ => None,
            };
            if let Some(combined) = combined {
                return Ok(self.tensor(
                    TensorSpec::new(scale_data_type, Shape::contiguous([1, 1, 1, 1])?)
                        .with_scalar_value(combined)?,
                ));
            }
        }
        let inline_scale = self.tensor(sdpa_scalar_tensor(scale_data_type, attention_scale)?);
        self.pointwise_binary_infer_as(
            scale,
            inline_scale,
            PointwiseMode::Mul,
            self.effective_compute_data_type(DataType::F32),
            scale_data_type,
        )
    }

    pub(super) fn bind_sdpa_inferred_tensor(
        &mut self,
        inferred: TensorId,
        output: TensorId,
        error_name: &str,
    ) -> Result<()> {
        let inferred_tensor = self.tensor_config(inferred)?.clone();
        let output_tensor = self.tensor_config(output)?.clone();
        if inferred_tensor.data_type != output_tensor.data_type {
            return Err(Error::FrontendTensorDataTypeMismatch {
                tensor_id: output,
                operation: error_name.into(),
                expected: inferred_tensor.data_type,
                actual: output_tensor.data_type,
            });
        }
        if inferred_tensor.shape.dimensions() != output_tensor.shape.dimensions() {
            return Err(Error::FrontendTensorDimensionsMismatch {
                tensor_id: output,
                operation: error_name.into(),
                expected: inferred_tensor.shape.dimensions().to_vec(),
                actual: output_tensor.shape.dimensions().to_vec(),
            });
        }
        self.reshape(inferred, output)
    }

    pub(super) fn bind_sdpa_inferred_optional_tensor(
        &mut self,
        inferred: Option<TensorId>,
        output: Option<TensorId>,
        error_name: &str,
    ) -> Result<()> {
        match (inferred, output) {
            (Some(inferred), Some(output)) => {
                self.bind_sdpa_inferred_tensor(inferred, output, error_name)
            }
            (None, None) => Ok(()),
            _ => Err(Error::FrontendSdpaOptionalOutputMismatch {
                output: error_name.into(),
            }),
        }
    }
}
