use std::iter;

use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::{
            SdpaAuxOutputs, SdpaFp8Inputs, SdpaFp8OutputTensors, SdpaMxfp8AuxOutputs,
            SdpaMxfp8Inputs, SdpaMxfp8OutputTensors,
        },
        graph::{Graph, MatmulFp8Inputs, MatmulFp8QuantizeTensors},
        infer::*,
        operation::*,
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    reduction::ReduceTensorOperator,
    scalar::ScalarValue,
    tensor::{Shape, TensorId, TensorSpec},
    version,
};

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
        let tensor = self.tensor_config(tensor)?;
        match tensor.data_type {
            DataType::F32 | DataType::F16 | DataType::BF16 => {}
            _ => {
                return Err(Error::DescriptorMismatch { name: name.into() });
            }
        }
        if tensor.shape.element_count()? != 1 {
            return Err(Error::DescriptorMismatch { name: name.into() });
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
        self.validate_fp8_scale_tensor(descale_q, "sdpa fp8 descale_q")?;
        self.validate_fp8_scale_tensor(descale_k, "sdpa fp8 descale_k")?;
        self.validate_fp8_scale_tensor(descale_v, "sdpa fp8 descale_v")?;
        self.validate_fp8_scale_tensor(descale_s, "sdpa fp8 descale_s")?;
        self.validate_fp8_scale_tensor(scale_s, "sdpa fp8 scale_s")?;
        self.validate_fp8_scale_tensor(scale_o, "sdpa fp8 scale_o")?;
        if config.has_logit_max() || config.has_score_sum_exp() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa fp8 aux outputs".into(),
            });
        }
        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, &v_tensor.shape)?;
        let attention_scale = self.tensor(
            TensorSpec::new(compute_type, Shape::contiguous([1, 1, 1, 1])?).with_scalar_value(
                ScalarValue::F32(config.attention_scale().ok_or(Error::DescriptorMismatch {
                    name: "sdpa fp8 attn scale".into(),
                })?),
            )?,
        );
        let k_t = self.transpose_last_two(k)?;
        let scores =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        self.matmul(q, k_t, scores, compute_type)?;
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
        let q_descaled_scores = self.tensor(
            TensorSpec::new(
                intermediate_type,
                self.tensor_config(scaled_scores)?.shape.clone(),
            )
            .virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Mul,
            lhs: scaled_scores,
            rhs: descale_q,
            output: q_descaled_scores,
            compute_type,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });
        let k_descaled_scores = self.tensor(
            TensorSpec::new(
                intermediate_type,
                self.tensor_config(q_descaled_scores)?.shape.clone(),
            )
            .virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Mul,
            lhs: q_descaled_scores,
            rhs: descale_k,
            output: k_descaled_scores,
            compute_type,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });
        let mut masked_scores = k_descaled_scores;
        if config.has_causal_bottom_right() {
            let sequence_length_query =
                config
                    .sequence_length_query()
                    .ok_or(Error::DescriptorMismatch {
                        name: "sdpa bottom right seq len".into(),
                    })?;
            let sequence_length_key_value =
                config
                    .sequence_length_key_value()
                    .ok_or(Error::DescriptorMismatch {
                        name: "sdpa bottom right seq len".into(),
                    })?;
            let mask = self.causal_bottom_right_mask(
                masked_scores,
                sequence_length_query,
                sequence_length_key_value,
            )?;
            masked_scores = self.pointwise_binary_infer_as(
                masked_scores,
                mask,
                PointwiseMode::Add,
                compute_type,
                intermediate_type,
            )?;
        } else if config.has_causal_mask() {
            masked_scores = self.apply_diagonal_band_mask(masked_scores, 0, Some(0))?;
        }
        let probs = self.tensor(
            TensorSpec::new(
                intermediate_type,
                self.tensor_config(masked_scores)?.shape.clone(),
            )
            .virtual_tensor(),
        );
        let (stats, _score_sum_exp) = if config.has_stats() {
            let aux_shape = reduce_last_axis(self.shape(masked_scores)?)?;
            let logit_max = self.tensor(
                TensorSpec::new(aux_data_type, aux_shape.clone())
                    .with_id(TensorId::generate())
                    .virtual_tensor(),
            );
            let score_sum_exp =
                self.tensor(TensorSpec::new(aux_data_type, aux_shape).virtual_tensor());
            self.softmax_composite(
                masked_scores,
                logit_max,
                score_sum_exp,
                probs,
                SoftmaxConfig::new(compute_type),
            )?;
            (Some(logit_max), Some(score_sum_exp))
        } else {
            let aux_shape = reduce_last_axis(self.shape(masked_scores)?)?;
            let logit_max =
                self.tensor(TensorSpec::new(aux_data_type, aux_shape.clone()).virtual_tensor());
            let score_sum_exp =
                self.tensor(TensorSpec::new(aux_data_type, aux_shape).virtual_tensor());
            self.softmax_composite(
                masked_scores,
                logit_max,
                score_sum_exp,
                probs,
                SoftmaxConfig::new(compute_type),
            )?;
            (None, None)
        };
        let absolute_max_s = if config.has_absolute_max_s() {
            let absolute_max_s_shape = Shape::contiguous(
                iter::repeat_n(1_i64, self.tensor_config(probs)?.shape.dimensions().len())
                    .collect::<Vec<_>>(),
            )?
            .with_strides(
                iter::repeat_n(1_i64, self.tensor_config(probs)?.shape.strides().len())
                    .collect::<Vec<_>>(),
            )?;
            let output = self.tensor(TensorSpec::new(compute_type, absolute_max_s_shape));
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
        let probs_scaled = self.tensor(
            TensorSpec::new(intermediate_type, self.tensor_config(probs)?.shape.clone())
                .virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Mul,
            lhs: probs,
            rhs: scale_s,
            output: probs_scaled,
            compute_type,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });
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
        let absolute_max_o_shape = Shape::contiguous(
            iter::repeat_n(1_i64, output_shape.dimensions().len()).collect::<Vec<_>>(),
        )?
        .with_strides(iter::repeat_n(1_i64, output_shape.strides().len()).collect::<Vec<_>>())?;
        let computed_absolute_max_o =
            self.tensor(TensorSpec::new(compute_type, absolute_max_o_shape));
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
        let absolute_max_o = config
            .has_absolute_max_o()
            .then_some(computed_absolute_max_o);
        Ok(SdpaAuxOutputs::with_aux(
            output,
            stats,
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
            return Err(Error::DescriptorMismatch {
                name: "sdpa fp8 rng dump".into(),
            });
        }

        self.bind_sdpa_inferred_tensor(computed.output, outputs.output, "sdpa fp8 output")?;
        self.bind_sdpa_inferred_optional_tensor(computed.stats, outputs.stats, "sdpa fp8 stats")?;
        self.bind_sdpa_inferred_optional_tensor(
            computed.logit_max,
            outputs.absolute_max_scores,
            "sdpa fp8 absolute_max_s",
        )?;
        self.bind_sdpa_inferred_optional_tensor(
            computed.score_sum_exp,
            outputs.absolute_max_output,
            "sdpa fp8 absolute_max_o",
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
        if config.has_absolute_max_s() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 absolute_max_s".into(),
            });
        }
        if config.has_logit_max() || config.has_score_sum_exp() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 aux outputs".into(),
            });
        }

        let attn_scale_value =
            self.tensor_config(attention_scale)?
                .scalar_value
                .ok_or(Error::DescriptorMismatch {
                    name: "sdpa mxfp8 attn scale".into(),
                })?;
        let attention_scale = self.tensor(
            TensorSpec::new(compute_type, Shape::contiguous([1, 1, 1, 1])?)
                .with_scalar_value(attn_scale_value)?,
        );

        let q_input_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let mut kt_dimensions = k_tensor.shape.dimensions().to_vec();
        let mut kt_strides = k_tensor.shape.strides().to_vec();
        kt_dimensions.swap(2, 3);
        kt_strides.swap(2, 3);
        self.replace_tensor(
            k,
            k_tensor.with_shape(
                Shape::contiguous(kt_dimensions.clone())?.with_strides(kt_strides.clone())?,
            ),
        )?;

        let sf_k_tensor = self.tensor_config(scale_k)?.clone();
        let mut sf_k_dimensions = sf_k_tensor.shape.dimensions().to_vec();
        let mut sf_k_strides = sf_k_tensor.shape.strides().to_vec();
        sf_k_dimensions.swap(2, 3);
        sf_k_strides.swap(2, 3);
        self.replace_tensor(
            scale_k,
            sf_k_tensor.with_shape(Shape::contiguous(sf_k_dimensions)?.with_strides(sf_k_strides)?),
        )?;

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
                Shape::contiguous(vec![
                    q_tensor.shape.dimensions()[0],
                    q_tensor.shape.dimensions()[1],
                    q_tensor.shape.dimensions()[2],
                    k_tensor.shape.dimensions()[3],
                ])?
                .with_strides(vec![
                    q_tensor.shape.dimensions()[1]
                        * q_tensor.shape.dimensions()[2]
                        * k_tensor.shape.dimensions()[3],
                    q_tensor.shape.dimensions()[2] * k_tensor.shape.dimensions()[3],
                    k_tensor.shape.dimensions()[3],
                    1,
                ])?,
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
        let scores = if config.has_causal_bottom_right() {
            let sequence_length_query =
                config
                    .sequence_length_query()
                    .ok_or(Error::DescriptorMismatch {
                        name: "sdpa bottom right seq len".into(),
                    })?;
            let sequence_length_key_value =
                config
                    .sequence_length_key_value()
                    .ok_or(Error::DescriptorMismatch {
                        name: "sdpa bottom right seq len".into(),
                    })?;
            let score_data_type = intermediate_type;
            let mask = self.causal_bottom_right_mask(
                scaled_scores,
                sequence_length_query,
                sequence_length_key_value,
            )?;
            self.pointwise_binary_infer_as(
                scaled_scores,
                mask,
                PointwiseMode::Add,
                compute_type,
                score_data_type,
            )?
        } else if config.has_causal_mask() {
            self.apply_diagonal_band_mask(scaled_scores, 0, Some(0))?
        } else {
            scaled_scores
        };

        let probs = self.tensor(
            TensorSpec::new(intermediate_type, self.shape(scores)?.clone()).virtual_tensor(),
        );
        let stats = if config.has_stats() {
            Some(self.tensor(TensorSpec::new(
                compute_type,
                reduce_last_axis(self.shape(scores)?)?,
            )))
        } else {
            None
        };
        let logit_max = self.tensor(
            TensorSpec::new(compute_type, reduce_last_axis(self.shape(scores)?)?).virtual_tensor(),
        );
        let score_sum_exp = self.tensor(
            TensorSpec::new(compute_type, reduce_last_axis(self.shape(scores)?)?).virtual_tensor(),
        );
        self.softmax_composite(
            scores,
            if config.has_stats() {
                stats.ok_or_else(|| Error::DescriptorMismatch {
                    name: "sdpa stats".into(),
                })?
            } else {
                logit_max
            },
            score_sum_exp,
            probs,
            SoftmaxConfig::new(compute_type),
        )?;

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
            Shape::contiguous(vec![
                q_input_tensor.shape.dimensions()[0],
                q_input_tensor.shape.dimensions()[1],
                q_input_tensor.shape.dimensions()[2],
                v_tensor.shape.dimensions()[3],
            ])?
            .with_strides(vec![
                q_input_tensor.shape.dimensions()[2]
                    * q_input_tensor.shape.dimensions()[1]
                    * v_tensor.shape.dimensions()[3],
                v_tensor.shape.dimensions()[3],
                q_input_tensor.shape.dimensions()[1] * v_tensor.shape.dimensions()[3],
                1,
            ])?,
        ));
        self.matmul(scaled_probs, v_fp, output, compute_type)?;

        let absolute_max_o = if config.has_absolute_max_o() {
            Some(self.absolute_max_infer(output, compute_type)?)
        } else {
            None
        };

        Ok(SdpaMxfp8AuxOutputs::new(output, stats, absolute_max_o))
    }

    pub fn sdpa_mxfp8_with_aux(
        &mut self,
        inputs: SdpaMxfp8Inputs,
        outputs: SdpaMxfp8OutputTensors,
        config: &SdpaConfig,
    ) -> Result<()> {
        let computed = self.sdpa_mxfp8_infer(inputs, config)?;

        self.bind_sdpa_inferred_tensor(computed.output, outputs.output, "sdpa mxfp8 output")?;
        self.bind_sdpa_inferred_optional_tensor(computed.stats, outputs.stats, "sdpa mxfp8 stats")?;
        self.bind_sdpa_inferred_optional_tensor(
            computed.absolute_max_output,
            outputs.absolute_max_output,
            "sdpa mxfp8 absolute_max_o",
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
