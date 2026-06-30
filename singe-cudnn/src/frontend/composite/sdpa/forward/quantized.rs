use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::{
            sdpa::{
                SdpaAuxOutputs, SdpaFp8Inputs, SdpaFp8OutputTensors, SdpaMxfp8AuxOutputs,
                SdpaMxfp8Inputs, SdpaMxfp8OutputTensors,
                support::{
                    SDPA_FP8_ABSOLUTE_MAX_O, SDPA_FP8_ABSOLUTE_MAX_S, SDPA_FP8_ATTN_SCALE,
                    SDPA_FP8_AUX_OUTPUTS, SDPA_FP8_DESCALE_K, SDPA_FP8_DESCALE_Q,
                    SDPA_FP8_DESCALE_S, SDPA_FP8_DESCALE_V, SDPA_FP8_OUTPUT, SDPA_FP8_RNG_DUMP,
                    SDPA_FP8_SCALE_O, SDPA_FP8_SCALE_S, SDPA_FP8_STATS, SDPA_MXFP8_ABSOLUTE_MAX_O,
                    SDPA_MXFP8_ABSOLUTE_MAX_S, SDPA_MXFP8_ATTN_SCALE, SDPA_MXFP8_AUX_OUTPUTS,
                    SDPA_MXFP8_OUTPUT, SDPA_MXFP8_STATS,
                },
            },
            softmax::SoftmaxCompositeTensors,
        },
        graph::{Graph, MatmulFp8Inputs, MatmulFp8QuantizeTensors},
        infer::*,
        operation::*,
        shape::{shape_with_strides, transpose_last_two_shape, unit_shape_like},
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    reduction::ReduceTensorOperator,
    scalar::ScalarValue,
    tensor::{Shape, TensorId, TensorSpec},
    version,
};

struct QuantizedSoftmaxOutputs {
    probs: TensorId,
    stats: Option<TensorId>,
}

impl Graph {
    pub(in crate::frontend::composite::sdpa) fn dequantize_tensor_infer(
        &mut self,
        input: TensorId,
        descale: TensorId,
    ) -> Result<TensorId> {
        let intermediate_type = self.effective_intermediate_data_type(DataType::F32);
        let compute_type = self.effective_compute_data_type(DataType::F32);
        let input_fp = self.pointwise_identity_infer(input, intermediate_type, compute_type)?;
        self.pointwise_binary_infer(input_fp, descale, PointwiseMode::Mul, compute_type)
    }

    pub(in crate::frontend::composite::sdpa) fn validate_fp8_scale_tensor(
        &self,
        tensor: TensorId,
        name: &str,
    ) -> Result<()> {
        let tensor_config = self.tensor_config(tensor)?;
        let supported = [DataType::F32, DataType::F16, DataType::BF16];
        if !supported.contains(&tensor_config.data_type) {
            return Err(Error::FrontendTensorDataTypeUnsupported {
                tensor_id: tensor,
                operation: name.into(),
                supported: supported.to_vec(),
                actual: tensor_config.data_type,
            });
        }
        self.validate_sdpa_scalar_element_count(tensor, name)
    }

    fn validate_fp8_forward_scale_tensors(&self, inputs: SdpaFp8Inputs) -> Result<()> {
        self.validate_fp8_scale_tensor(inputs.descale_query, SDPA_FP8_DESCALE_Q)?;
        self.validate_fp8_scale_tensor(inputs.descale_key, SDPA_FP8_DESCALE_K)?;
        self.validate_fp8_scale_tensor(inputs.descale_value, SDPA_FP8_DESCALE_V)?;
        self.validate_fp8_scale_tensor(inputs.descale_scores, SDPA_FP8_DESCALE_S)?;
        self.validate_fp8_scale_tensor(inputs.scale_scores, SDPA_FP8_SCALE_S)?;
        self.validate_fp8_scale_tensor(inputs.scale_output, SDPA_FP8_SCALE_O)
    }

    fn validate_fp8_forward_aux_outputs(&self, config: &SdpaConfig) -> Result<()> {
        let softmax_aux = config.aux_outputs().softmax();
        if softmax_aux.requires_unified_cudnn_921() {
            return Err(Error::FrontendSdpaQuantizedAuxOutputUnsupported {
                operation: "fp8 sdpa".into(),
                output: SDPA_FP8_AUX_OUTPUTS.into(),
            });
        }
        Ok(())
    }

    fn validate_mxfp8_forward_aux_outputs(&self, config: &SdpaConfig) -> Result<()> {
        let aux_outputs = config.aux_outputs();
        let softmax_aux = aux_outputs.softmax();
        let quantized_aux = aux_outputs.quantized();
        if quantized_aux.absolute_max_s() {
            return Err(Error::FrontendSdpaQuantizedAuxOutputUnsupported {
                operation: "mxfp8 sdpa".into(),
                output: SDPA_MXFP8_ABSOLUTE_MAX_S.into(),
            });
        }
        if softmax_aux.requires_unified_cudnn_921() {
            return Err(Error::FrontendSdpaQuantizedAuxOutputUnsupported {
                operation: "mxfp8 sdpa".into(),
                output: SDPA_MXFP8_AUX_OUTPUTS.into(),
            });
        }
        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn quantize_tensor_infer(
        &mut self,
        input: TensorId,
        scale: TensorId,
        output_data_type: DataType,
    ) -> Result<TensorId> {
        let compute_type = self.effective_compute_data_type(DataType::F32);
        let scaled = self.pointwise_binary_infer(input, scale, PointwiseMode::Mul, compute_type)?;
        self.pointwise_identity_infer(scaled, output_data_type, compute_type)
    }

    fn sdpa_apply_fused_score_mask(
        &mut self,
        scores: TensorId,
        mask: SdpaFusedMaskMode,
        compute_type: DataType,
        output_data_type: DataType,
    ) -> Result<TensorId> {
        match mask {
            SdpaFusedMaskMode::None => Ok(scores),
            SdpaFusedMaskMode::CausalTopLeft => self.apply_diagonal_band_mask(scores, 0, Some(0)),
            SdpaFusedMaskMode::CausalBottomRight {
                sequence_length_query,
                sequence_length_key_value,
            } => {
                let mask = self.causal_bottom_right_mask(
                    scores,
                    sequence_length_query,
                    sequence_length_key_value,
                )?;
                self.pointwise_binary_infer_as(
                    scores,
                    mask,
                    PointwiseMode::Add,
                    compute_type,
                    output_data_type,
                )
            }
        }
    }

    fn quantized_sdpa_mul_like(
        &mut self,
        input: TensorId,
        rhs: TensorId,
        output_data_type: DataType,
        compute_type: DataType,
    ) -> Result<TensorId> {
        let output = self.tensor(
            TensorSpec::new(output_data_type, self.tensor_config(input)?.shape.clone())
                .virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Mul,
            lhs: input,
            rhs,
            output,
            compute_type,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });
        Ok(output)
    }

    fn quantized_sdpa_softmax(
        &mut self,
        scores: TensorId,
        softmax_aux: SdpaSoftmaxAuxOutputRequest,
        probability_data_type: DataType,
        aux_data_type: DataType,
        compute_type: DataType,
    ) -> Result<QuantizedSoftmaxOutputs> {
        let probs = self.tensor(
            TensorSpec::new(
                probability_data_type,
                self.tensor_config(scores)?.shape.clone(),
            )
            .virtual_tensor(),
        );
        let aux_shape = reduce_last_axis(self.shape(scores)?)?;
        let stats = softmax_aux
            .stats()
            .then(|| self.tensor(TensorSpec::new(aux_data_type, aux_shape.clone())));
        let softmax_stats = stats.unwrap_or_else(|| {
            self.tensor(TensorSpec::new(aux_data_type, aux_shape.clone()).virtual_tensor())
        });
        let logit_max =
            self.tensor(TensorSpec::new(aux_data_type, aux_shape.clone()).virtual_tensor());
        let score_sum_exp = self.tensor(TensorSpec::new(aux_data_type, aux_shape).virtual_tensor());
        self.softmax_composite(
            SoftmaxCompositeTensors::new(scores, softmax_stats, logit_max, score_sum_exp, probs),
            SoftmaxConfig::new(compute_type),
        )?;
        Ok(QuantizedSoftmaxOutputs { probs, stats })
    }

    /// Adds an FP8 SDPA forward graph and creates output and auxiliary tensors.
    ///
    /// FP8 SDPA uses explicit descale/scale tensors for Q, K, V, scores, and
    /// output as documented in the frontend attention support surface.
    pub fn sdpa_fp8_infer(
        &mut self,
        inputs: SdpaFp8Inputs,
        config: &SdpaConfig,
    ) -> Result<SdpaAuxOutputs> {
        let q = inputs.query;
        let output_io_type = self.tensor_config(q)?.data_type;
        self.sdpa_fp8_infer_with_output_type(inputs, output_io_type, config)
    }

    /// Adds FP8 SDPA forward with an explicit output data type.
    pub fn sdpa_fp8_infer_with_output_type(
        &mut self,
        inputs: SdpaFp8Inputs,
        output_io_type: DataType,
        config: &SdpaConfig,
    ) -> Result<SdpaAuxOutputs> {
        let SdpaFp8Inputs {
            query: q,
            key: k,
            value: v,
            descale_query: descale_q,
            descale_key: descale_k,
            descale_value: descale_v,
            descale_scores: descale_s,
            scale_scores: scale_s,
            scale_output: scale_o,
        } = inputs;
        let compute_type = self.effective_compute_data_type(DataType::F32);
        let intermediate_type = self.effective_intermediate_data_type(DataType::F32);
        let aux_data_type = self.sdpa_aux_data_type();
        self.validate_fp8_forward_support_surface_for_version(version()?.raw(), output_io_type)?;
        self.validate_fp8_forward_scale_tensors(inputs)?;
        self.validate_fp8_forward_aux_outputs(config)?;
        let softmax_aux = config.aux_outputs().softmax();
        let quantized_aux = config.aux_outputs().quantized();
        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, &v_tensor.shape)?;
        let attention_scale = self.tensor(
            TensorSpec::new(compute_type, Shape::contiguous([1, 1, 1, 1])?).with_scalar_value(
                ScalarValue::F32(config.attention_scale().ok_or_else(|| {
                    Error::FrontendSdpaAttentionScaleRequired {
                        operation: SDPA_FP8_ATTN_SCALE.into(),
                    }
                })?),
            )?,
        );
        let k_t = self.transpose_last_two(k)?;
        let scores =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        self.matmul(q, k_t, scores, compute_type)?;
        let scaled_scores =
            self.quantized_sdpa_mul_like(scores, attention_scale, intermediate_type, compute_type)?;
        let q_descaled_scores = self.quantized_sdpa_mul_like(
            scaled_scores,
            descale_q,
            intermediate_type,
            compute_type,
        )?;
        let k_descaled_scores = self.quantized_sdpa_mul_like(
            q_descaled_scores,
            descale_k,
            intermediate_type,
            compute_type,
        )?;
        let masked_scores = self.sdpa_apply_fused_score_mask(
            k_descaled_scores,
            config.mask(),
            compute_type,
            intermediate_type,
        )?;
        let softmax = self.quantized_sdpa_softmax(
            masked_scores,
            softmax_aux,
            intermediate_type,
            aux_data_type,
            compute_type,
        )?;
        let probs = softmax.probs;
        let absolute_max_s = if quantized_aux.absolute_max_s() {
            let output = self.tensor(TensorSpec::new(
                compute_type,
                unit_shape_like(self.shape(probs)?)?,
            ));
            self.reduction(ReductionOperation::Reduce {
                op: ReduceTensorOperator::AbsoluteMax,
                input: probs,
                output,
                compute_type,
                is_deterministic: false,
            });
            Some(output)
        } else {
            None
        };
        let probs_scaled =
            self.quantized_sdpa_mul_like(probs, scale_s, intermediate_type, compute_type)?;
        let probs_fp8 = self.tensor(
            TensorSpec::new(
                output_io_type,
                Shape::contiguous(
                    self.tensor_config(probs_scaled)?
                        .shape
                        .dimensions()
                        .to_vec(),
                )?,
            )
            .virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Identity,
            input: probs_scaled,
            output: probs_fp8,
            compute_type,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            axis: None,
        });
        let output = self.tensor(TensorSpec::new(output_io_type, output_shape.clone()));
        let computed_absolute_max_o = self.tensor(TensorSpec::new(
            compute_type,
            unit_shape_like(&output_shape)?,
        ));
        self.matmul_fp8_quantize(
            MatmulFp8QuantizeTensors::new(
                MatmulFp8Inputs {
                    a: probs_fp8,
                    b: v,
                    descale_a: descale_s,
                    descale_b: descale_v,
                },
                scale_o,
                output,
                computed_absolute_max_o,
            ),
            compute_type,
        )?;
        let absolute_max_o = quantized_aux
            .absolute_max_o()
            .then_some(computed_absolute_max_o);
        Ok(SdpaAuxOutputs::with_aux(
            output,
            softmax.stats,
            absolute_max_s,
            absolute_max_o,
            None,
        ))
    }

    pub fn sdpa_fp8_with_aux(
        &mut self,
        inputs: SdpaFp8Inputs,
        outputs: SdpaFp8OutputTensors,
        config: &SdpaConfig,
    ) -> Result<()> {
        let computed = self.sdpa_fp8_infer_with_output_type(
            inputs,
            self.tensor_config(outputs.output)?.data_type,
            config,
        )?;
        if computed.rng_dump.is_some() {
            return Err(Error::FrontendSdpaQuantizedAuxOutputUnsupported {
                operation: "fp8 sdpa".into(),
                output: SDPA_FP8_RNG_DUMP.into(),
            });
        }

        self.bind_sdpa_inferred_tensor(computed.output, outputs.output, SDPA_FP8_OUTPUT)?;
        self.bind_sdpa_inferred_optional_tensor(computed.stats, outputs.stats, SDPA_FP8_STATS)?;
        self.bind_sdpa_inferred_optional_tensor(
            computed.logit_max,
            outputs.absolute_max_scores,
            SDPA_FP8_ABSOLUTE_MAX_S,
        )?;
        self.bind_sdpa_inferred_optional_tensor(
            computed.score_sum_exp,
            outputs.absolute_max_output,
            SDPA_FP8_ABSOLUTE_MAX_O,
        )?;
        Ok(())
    }

    pub fn sdpa_fp8(
        &mut self,
        inputs: SdpaFp8Inputs,
        outputs: SdpaFp8OutputTensors,
        config: &SdpaConfig,
    ) -> Result<()> {
        self.sdpa_fp8_with_aux(inputs, outputs, config)
    }

    pub fn sdpa_mxfp8_infer(
        &mut self,
        inputs: SdpaMxfp8Inputs,
        config: &SdpaConfig,
    ) -> Result<SdpaMxfp8AuxOutputs> {
        let SdpaMxfp8Inputs {
            query: q,
            key: k,
            value: v,
            scale_query: scale_q,
            scale_key: scale_k,
            scale_value: scale_v,
            attention_scale,
        } = inputs;
        let output_io_type = self.effective_io_data_type(DataType::BF16);
        let intermediate_type = self.effective_intermediate_data_type(DataType::F32);
        let compute_type = self.effective_compute_data_type(DataType::F32);
        self.validate_mxfp8_forward_aux_outputs(config)?;
        let softmax_aux = config.aux_outputs().softmax();
        let quantized_aux = config.aux_outputs().quantized();

        let attn_scale_value = self
            .tensor_config(attention_scale)?
            .scalar_value
            .ok_or_else(|| Error::FrontendSdpaAttentionScaleRequired {
                operation: SDPA_MXFP8_ATTN_SCALE.into(),
            })?;
        let attention_scale = self.tensor(
            TensorSpec::new(compute_type, Shape::contiguous([1, 1, 1, 1])?)
                .with_scalar_value(attn_scale_value)?,
        );

        let q_input_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let k_t_shape = transpose_last_two_shape(&k_tensor.shape)?;
        self.replace_tensor(k, k_tensor.with_shape(k_t_shape))?;

        let sf_k_tensor = self.tensor_config(scale_k)?.clone();
        let sf_k_t_shape = transpose_last_two_shape(&sf_k_tensor.shape)?;
        self.replace_tensor(scale_k, sf_k_tensor.with_shape(sf_k_t_shape))?;

        let q_fp = self.block_scale_dequantize_infer(
            q,
            scale_q,
            BlockScaleDequantizeConfig::new(compute_type, vec![1, 32]),
        )?;
        let k_fp = self.block_scale_dequantize_infer(
            k,
            scale_k,
            BlockScaleDequantizeConfig::new(compute_type, vec![32, 1]),
        )?;

        let q_tensor = self.tensor_config(q_fp)?.clone();
        let k_tensor = self.tensor_config(k_fp)?.clone();
        let v_fp = self.block_scale_dequantize_infer(
            v,
            scale_v,
            BlockScaleDequantizeConfig::new(compute_type, vec![32, 1]),
        )?;
        let v_tensor = self.tensor_config(v_fp)?.clone();
        let scores = self.tensor(
            TensorSpec::new(
                intermediate_type,
                shape_with_strides(
                    vec![
                        q_tensor.shape.dimensions()[0],
                        q_tensor.shape.dimensions()[1],
                        q_tensor.shape.dimensions()[2],
                        k_tensor.shape.dimensions()[3],
                    ],
                    vec![
                        q_tensor.shape.dimensions()[1]
                            * q_tensor.shape.dimensions()[2]
                            * k_tensor.shape.dimensions()[3],
                        q_tensor.shape.dimensions()[2] * k_tensor.shape.dimensions()[3],
                        k_tensor.shape.dimensions()[3],
                        1,
                    ],
                )?,
            )
            .virtual_tensor(),
        );
        self.matmul(q_fp, k_fp, scores, compute_type)?;
        let scaled_scores = self.tensor(
            TensorSpec::new(intermediate_type, self.tensor_config(scores)?.shape.clone())
                .virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Mul,
            lhs: scores,
            rhs: attention_scale,
            output: scaled_scores,
            compute_type,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });
        let scores = self.sdpa_apply_fused_score_mask(
            scaled_scores,
            config.mask(),
            compute_type,
            intermediate_type,
        )?;

        let softmax = self.quantized_sdpa_softmax(
            scores,
            softmax_aux,
            intermediate_type,
            compute_type,
            compute_type,
        )?;
        let probs = softmax.probs;

        let scaled_probs = self.tensor(
            TensorSpec::new(
                q_input_tensor.data_type,
                self.tensor_config(probs)?.shape.clone(),
            )
            .virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Identity,
            input: probs,
            output: scaled_probs,
            compute_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: None,
        });

        let output = self.tensor(TensorSpec::new(
            output_io_type,
            shape_with_strides(
                vec![
                    q_input_tensor.shape.dimensions()[0],
                    q_input_tensor.shape.dimensions()[1],
                    q_input_tensor.shape.dimensions()[2],
                    v_tensor.shape.dimensions()[3],
                ],
                vec![
                    q_input_tensor.shape.dimensions()[2]
                        * q_input_tensor.shape.dimensions()[1]
                        * v_tensor.shape.dimensions()[3],
                    v_tensor.shape.dimensions()[3],
                    q_input_tensor.shape.dimensions()[1] * v_tensor.shape.dimensions()[3],
                    1,
                ],
            )?,
        ));
        self.matmul(scaled_probs, v_fp, output, compute_type)?;

        let absolute_max_o = if quantized_aux.absolute_max_o() {
            Some(self.absolute_max_infer(output, compute_type)?)
        } else {
            None
        };

        Ok(SdpaMxfp8AuxOutputs::new(
            output,
            softmax.stats,
            absolute_max_o,
        ))
    }

    pub fn sdpa_mxfp8_with_aux(
        &mut self,
        inputs: SdpaMxfp8Inputs,
        outputs: SdpaMxfp8OutputTensors,
        config: &SdpaConfig,
    ) -> Result<()> {
        let computed = self.sdpa_mxfp8_infer(inputs, config)?;

        self.bind_sdpa_inferred_tensor(computed.output, outputs.output, SDPA_MXFP8_OUTPUT)?;
        self.bind_sdpa_inferred_optional_tensor(computed.stats, outputs.stats, SDPA_MXFP8_STATS)?;
        self.bind_sdpa_inferred_optional_tensor(
            computed.absolute_max_output,
            outputs.absolute_max_output,
            SDPA_MXFP8_ABSOLUTE_MAX_O,
        )?;
        Ok(())
    }

    pub fn sdpa_mxfp8(
        &mut self,
        inputs: SdpaMxfp8Inputs,
        outputs: SdpaMxfp8OutputTensors,
        config: &SdpaConfig,
    ) -> Result<()> {
        self.sdpa_mxfp8_with_aux(inputs, outputs, config)
    }
}
