use crate::{
    backend::{
        pointwise::PointwiseMode,
        reduction::ReduceTensorOperator,
        tensor::{BackendTensorReordering, TensorId, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::{
            SdpaForwardOutputTensors, SdpaFp8Inputs, SdpaInputs, SdpaMxfp8Inputs,
            forward::selection::ResolvedAttentionImplementation, infer_sdpa_output_shape,
            infer_sdpa_scores_shape, support::SDPA_SCORE_SUBGRAPH,
        },
        error::{Error as FrontendError, SdpaFeature},
        graph::{
            BlockScaleDequantizeInputs, Graph, MatmulFp8Inputs, MatmulFp8QuantizeOutputs,
            MatmulFp8QuantizeTensors, SoftmaxOutputs, TensorOutputConfig,
        },
        operation::{
            AttentionConfig, AttentionForwardInputs, BlockScaleDequantizeConfig,
            PointwiseOperation, ReductionConfig, SdpaForwardQuantization,
            SdpaSoftmaxAuxOutputRequest,
        },
    },
    math::NanPropagation,
};

struct QuantizedSoftmaxOutputs {
    probs: TensorId,
    stats: Option<TensorId>,
    logit_max: Option<TensorId>,
    score_sum_exp: Option<TensorId>,
    rng_dump: Option<TensorId>,
}

impl Graph {
    fn quantized_sdpa_mul_like(
        &mut self,
        input: TensorId,
        rhs: TensorId,
        output_data_type: DataType,
        compute_type: DataType,
    ) -> Result<TensorId> {
        let output = self.tensor(
            TensorSpec::new(output_data_type, self.tensor_spec(input)?.shape.clone())
                .virtual_tensor(),
        );
        self.pointwise_into(PointwiseOperation::Binary {
            name: None,
            mode: PointwiseMode::Mul,
            lhs: input,
            rhs,
            output,
            compute_type: Some(compute_type),
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            alpha2: 1.0,
        })?;
        Ok(output)
    }

    fn quantized_sdpa_softmax(
        &mut self,
        scores: TensorId,
        softmax_aux: SdpaSoftmaxAuxOutputRequest,
        probability_data_type: DataType,
        config: &AttentionConfig,
    ) -> Result<QuantizedSoftmaxOutputs> {
        let mut softmax_config = config.softmax().clone();
        if let Some(sink) = config.modifiers().sink_token() {
            softmax_config = softmax_config.with_sink(sink);
        }
        let compute_type = self.resolve_compute_data_type(softmax_config.compute_type());
        let scores_shape = self.shape(scores)?.clone();
        let summary = TensorOutputConfig::new()
            .with_data_type(self.effective_io_data_type(probability_data_type))
            .output_tensor();
        let outputs = self.attention_softmax_infer(
            scores,
            SoftmaxOutputs {
                output: TensorOutputConfig::new().with_data_type(probability_data_type),
                stats: softmax_aux.stats().then_some(
                    TensorOutputConfig::new()
                        .with_data_type(DataType::F32)
                        .output_tensor(),
                ),
                max: softmax_aux.logit_max().then_some(summary),
                sum_exp: softmax_aux.score_sum_exp().then_some(summary),
            },
            &softmax_config,
        )?;
        let probs = outputs.output;
        let (probs, rng_dump) = if let Some(dropout) = config.dropout() {
            let mask =
                self.attention_dropout_random_number_generator_dump(dropout, &scores_shape)?;
            let scale = self.tensor(TensorSpec::scalar_f32(dropout.scale()));
            if !config.fused().aux_outputs().rng_dump() {
                self.modify_tensor(mask, TensorSpec::virtual_tensor)?;
            }
            (
                self.sdpa_apply_dropout(probs, Some(mask), Some(scale), compute_type)?,
                config.fused().aux_outputs().rng_dump().then_some(mask),
            )
        } else {
            (
                self.sdpa_apply_dropout(
                    probs,
                    config.modifiers().dropout_mask(),
                    config.modifiers().dropout_scale(),
                    compute_type,
                )?,
                None,
            )
        };
        Ok(QuantizedSoftmaxOutputs {
            probs,
            stats: outputs.stats,
            logit_max: outputs.max,
            score_sum_exp: outputs.sum_exp,
            rng_dump,
        })
    }

    /// Adds FP8 SDPA with independent output and auxiliary placement and encoding.
    /// Stats, RNG dumps and Amax outputs default to FP32 independently of graph policies.
    ///
    /// FP8 SDPA uses explicit descale/scale tensors for Q, K, V, scores, and
    /// output as documented in the frontend attention support surface.
    /// Quantization scales need not be scalars. Composite operations accept shapes
    /// broadcastable at their point of use; native engines determine their supported layouts.
    /// The separate attention-scale operand remains a logical scalar and can be a device tensor,
    /// host by-value input or constant. A configured numeric attention factor multiplies it.
    /// `AttentionConfig` carries sequence lengths, paging, score modifiers, dropout and implementation selection.
    /// Cumulative lengths and block masks require the unified implementation.
    /// Preparation selects an implementation for the target and validates its support restrictions.
    /// Result placement selects I/O or intermediate encoding unless explicitly overridden.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown or incompatible query/key/value or paged-cache geometry, invalid sequence-length combinations, an RNG dump requested without dropout, or unrepresentable inferred score, result, or auxiliary layouts.
    pub fn sdpa_fp8(
        &mut self,
        inputs: SdpaFp8Inputs,
        config: &AttentionConfig,
        outputs: SdpaForwardOutputTensors<TensorOutputConfig>,
    ) -> Result<SdpaForwardOutputTensors> {
        self.record_attention_forward(AttentionForwardInputs::Fp8(inputs), outputs, config.clone())
    }

    pub(crate) fn expand_fp8_attention(
        &mut self,
        inputs: SdpaFp8Inputs,
        output_io_type: DataType,
        implementation: ResolvedAttentionImplementation,
        config: &AttentionConfig,
    ) -> Result<SdpaForwardOutputTensors> {
        if implementation == ResolvedAttentionImplementation::Composite {
            return self.sdpa_fp8_composite_infer(inputs, output_io_type, config);
        }
        self.validate_fp8_forward_support(output_io_type)?;
        self.sdpa_unified_infer(
            SdpaInputs {
                query: inputs.query,
                key: inputs.key,
                value: inputs.value,
                scale: inputs.attention_scale,
            },
            output_io_type,
            SdpaForwardQuantization {
                descale_query: Some(inputs.descale_query),
                descale_key: Some(inputs.descale_key),
                descale_value: Some(inputs.descale_value),
                descale_scores: Some(inputs.descale_scores),
                scale_scores: Some(inputs.scale_scores),
                scale_output: Some(inputs.scale_output),
                ..SdpaForwardQuantization::default()
            },
            config,
        )
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the composite attention operation builds and validates its complete tensor graph"
    )]
    fn sdpa_fp8_composite_infer(
        &mut self,
        inputs: SdpaFp8Inputs,
        output_io_type: DataType,
        config: &AttentionConfig,
    ) -> Result<SdpaForwardOutputTensors> {
        let SdpaFp8Inputs {
            query: q,
            key: k,
            value: v,
            attention_scale,
            descale_query: descale_q,
            descale_key: descale_k,
            descale_value: descale_v,
            descale_scores: descale_s,
            scale_scores: scale_s,
            scale_output: scale_o,
        } = inputs;
        let compute_type = self.resolve_compute_data_type(config.softmax().compute_type());
        let intermediate_type = self.effective_intermediate_data_type(DataType::F32);
        let (score_matmul, output_matmul) =
            Self::sdpa_forward_matmul_configs(config.modifiers(), compute_type);
        self.validate_fp8_forward_support(output_io_type)?;
        let softmax_aux = config.fused().aux_outputs().softmax();
        let quantized_aux = config.fused().aux_outputs().quantized();
        let probability_type = self.tensor_spec(q)?.data_type;
        let prepared_kv = self.prepare_attention_kv(k, v, config)?;
        let k = prepared_kv.key;
        let v = prepared_kv.value;
        let scores_shape = infer_sdpa_scores_shape(self.shape(q)?, self.shape(k)?)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, self.shape(v)?)?;
        let k_t = self.transpose_last_two_view(k)?;
        let attention_scale = self.sdpa_effective_scale(
            attention_scale,
            config.fused().attention_scale(),
            compute_type,
        )?;
        let scores = self.tensor(TensorSpec::new(intermediate_type, scores_shape).virtual_tensor());
        self.matmul_into(q, k_t, scores, score_matmul)?;
        let q_descaled_scores =
            self.quantized_sdpa_mul_like(scores, descale_q, intermediate_type, compute_type)?;
        let k_descaled_scores = self.quantized_sdpa_mul_like(
            q_descaled_scores,
            descale_k,
            intermediate_type,
            compute_type,
        )?;
        let masked_scores = self.sdpa_scores_with_modifiers(
            k_descaled_scores,
            Some(attention_scale),
            config.modifiers(),
            config.score_subgraph(),
            SDPA_SCORE_SUBGRAPH,
            compute_type,
            self.sdpa_score_data_type(),
        )?;
        let softmax =
            self.quantized_sdpa_softmax(masked_scores, softmax_aux, intermediate_type, config)?;
        let probs = softmax.probs;
        let absolute_max_s = if quantized_aux.absolute_max_s() {
            let output = self.tensor(TensorSpec::new(
                DataType::F32,
                self.shape(probs)?.unit_like()?,
            ));
            self.reduction_into(
                probs,
                output,
                &ReductionConfig::new(ReduceTensorOperator::AbsoluteMax)
                    .with_compute_type(compute_type),
            )?;
            Some(output)
        } else {
            None
        };
        let probs_scaled =
            self.quantized_sdpa_mul_like(probs, scale_s, intermediate_type, compute_type)?;
        self.modify_tensor(probs_scaled, |mut tensor| {
            tensor.data_type = probability_type;
            tensor.virtual_tensor()
        })?;
        let output = self.tensor(TensorSpec::new(output_io_type, output_shape.clone()));
        let amax_spec = TensorSpec::new(DataType::F32, output_shape.unit_like()?);
        let computed_absolute_max_o = self.tensor(if quantized_aux.absolute_max_o() {
            amax_spec
        } else {
            amax_spec.virtual_tensor()
        });
        self.matmul_fp8_quantize_into(
            MatmulFp8QuantizeTensors::new(
                MatmulFp8Inputs {
                    a: probs_scaled,
                    b: v,
                    descale_a: descale_s,
                    descale_b: descale_v,
                },
                scale_o,
                MatmulFp8QuantizeOutputs::new(output, computed_absolute_max_o),
            ),
            output_matmul,
        )?;
        let absolute_max_o = quantized_aux
            .absolute_max_o()
            .then_some(computed_absolute_max_o);
        Ok(SdpaForwardOutputTensors {
            output,
            stats: softmax.stats,
            logit_max: softmax.logit_max,
            score_sum_exp: softmax.score_sum_exp,
            rng_dump: softmax.rng_dump,
            absolute_max_probabilities: absolute_max_s,
            absolute_max_output: absolute_max_o,
        })
    }

    /// Records FP8 attention with caller-declared output layouts and auxiliary tensors.
    /// Supplied auxiliary tensors enable their requests without duplicate configuration flags.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown or incompatible query/key/value or paged-cache geometry, invalid sequence-length combinations, an RNG dump requested without dropout, or unrepresentable inferred score, result, or auxiliary layouts. Caller-declared outputs must exist and match inferred output requirements.
    pub fn sdpa_fp8_into(
        &mut self,
        inputs: SdpaFp8Inputs,
        outputs: SdpaForwardOutputTensors,
        config: &AttentionConfig,
    ) -> Result<()> {
        self.record_explicit_attention_forward(AttentionForwardInputs::Fp8(inputs), outputs, config)
    }

    /// Adds block-scaled attention with independent result placement and encoding.
    /// Every requested result selects placement and encoding independently.
    /// Stats, RNG dumps and probability/output Amax default to FP32 independently of graph policies.
    /// Omitted primary encoding follows I/O or intermediate policy, with BF16 as the fallback.
    /// Explicit output types reach native support selection without a frontend-only BF16 restriction.
    /// Paged MXFP8 caches are unsupported by the C++ cuDNN backend path.
    /// Implementation selection and expansion occur during preparation.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown or incompatible query/key/value or paged-cache geometry, invalid sequence-length combinations, an RNG dump requested without dropout, or unrepresentable inferred score, result, or auxiliary layouts. MXFP8 requests cannot use paged caches.
    pub fn sdpa_mxfp8(
        &mut self,
        inputs: SdpaMxfp8Inputs,
        config: &AttentionConfig,
        outputs: SdpaForwardOutputTensors<TensorOutputConfig>,
    ) -> Result<SdpaForwardOutputTensors> {
        self.record_attention_forward(
            AttentionForwardInputs::Mxfp8(inputs),
            outputs,
            config.clone(),
        )
    }

    pub(crate) fn expand_mxfp8_attention(
        &mut self,
        inputs: SdpaMxfp8Inputs,
        output_data_type: DataType,
        implementation: ResolvedAttentionImplementation,
        config: &AttentionConfig,
    ) -> Result<SdpaForwardOutputTensors> {
        if config.paged().is_enabled() {
            return Err(FrontendError::SdpaUnsupported {
                feature: SdpaFeature::PagedCaches,
                operation: "mxfp8 sdpa",
                sm_version: None,
            }
            .into());
        }
        if implementation == ResolvedAttentionImplementation::Composite {
            return self.sdpa_mxfp8_composite_infer(inputs, output_data_type, config);
        }
        for (input, scale, blocks) in [
            (inputs.query, inputs.scale_query, [1, 32]),
            (inputs.key, inputs.scale_key, [1, 32]),
            (inputs.value, inputs.scale_value, [32, 1]),
        ] {
            self.validate_tensor_data_type(scale, DataType::F8UE8M0, "mxfp8 scale encoding")?;
            let scale_tensor = self.tensor_spec(scale)?;
            let input_tensor = self.tensor_spec(input)?;
            if scale_tensor.reordering != Some(BackendTensorReordering::F8_128x4) {
                return Err(FrontendError::TensorLayoutMismatch {
                    tensor_id: scale,
                    operation: "mxfp8 scale layout",
                    reason: "expected F8_128x4 reordering".into(),
                }
                .into());
            }
            let expected_dims = BlockScaleDequantizeConfig::new(blocks)
                .infer_scale_dimensions(&input_tensor.shape)?;
            let dims = scale_tensor.shape.dimensions();
            if dims.len() != 4
                || expected_dims.len() != 4
                || dims[..2] != expected_dims[..2]
                || dims[2] < expected_dims[2]
                || dims[3] < expected_dims[3]
            {
                return Err(FrontendError::TensorLayoutMismatch { tensor_id: scale, operation: "mxfp8 scale layout", reason: "expected matching batch/head dimensions and scale dimensions covering the input".into() }.into());
            }
            let last_stride = scale_tensor.shape.strides()[3];
            if blocks[0] == 1 && last_stride != 1 {
                return Err(FrontendError::TensorLastStrideMismatch {
                    tensor_id: scale,
                    operation: "mxfp8 scale layout",
                    expected: 1,
                    actual: last_stride,
                }
                .into());
            }
        }
        self.sdpa_unified_infer(
            SdpaInputs {
                query: inputs.query,
                key: inputs.key,
                value: inputs.value,
                scale: inputs.attention_scale,
            },
            output_data_type,
            SdpaForwardQuantization {
                descale_query: Some(inputs.scale_query),
                descale_key: Some(inputs.scale_key),
                descale_value: Some(inputs.scale_value),
                ..SdpaForwardQuantization::default()
            },
            config,
        )
    }

    fn sdpa_mxfp8_composite_infer(
        &mut self,
        inputs: SdpaMxfp8Inputs,
        output_data_type: DataType,
        config: &AttentionConfig,
    ) -> Result<SdpaForwardOutputTensors> {
        let SdpaMxfp8Inputs {
            query: q,
            key: k,
            value: v,
            scale_query: scale_q,
            scale_key: scale_k,
            scale_value: scale_v,
            attention_scale,
        } = inputs;
        let intermediate_type = self.effective_intermediate_data_type(DataType::F32);
        let compute_type = self.resolve_compute_data_type(config.softmax().compute_type());
        let (score_matmul, output_matmul) =
            Self::sdpa_forward_matmul_configs(config.modifiers(), compute_type);
        let softmax_aux = config.fused().aux_outputs().softmax();
        let quantized_aux = config.fused().aux_outputs().quantized();

        let attention_scale = self.sdpa_effective_scale(
            attention_scale,
            config.fused().attention_scale(),
            compute_type,
        )?;

        let q_input_tensor = self.tensor_spec(q)?;
        let probability_type = q_input_tensor.data_type;
        let scores_shape = infer_sdpa_scores_shape(&q_input_tensor.shape, self.shape(k)?)?;
        let k = self.transpose_last_two_view(k)?;
        let scale_k = self.transpose_last_two_view(scale_k)?;

        let q_fp = self.block_scale_dequantize(
            BlockScaleDequantizeInputs::new(q, scale_q),
            BlockScaleDequantizeConfig::new(vec![1, 32]).with_compute_type(compute_type),
        )?;
        let k_fp = self.block_scale_dequantize(
            BlockScaleDequantizeInputs::new(k, scale_k),
            BlockScaleDequantizeConfig::new(vec![32, 1]).with_compute_type(compute_type),
        )?;

        let v_fp = self.block_scale_dequantize(
            BlockScaleDequantizeInputs::new(v, scale_v),
            BlockScaleDequantizeConfig::new(vec![32, 1]).with_compute_type(compute_type),
        )?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, self.shape(v_fp)?)?;
        let scores = self.tensor(TensorSpec::new(intermediate_type, scores_shape).virtual_tensor());
        self.matmul_into(q_fp, k_fp, scores, score_matmul)?;
        let scores = self.sdpa_scores_with_modifiers(
            scores,
            Some(attention_scale),
            config.modifiers(),
            config.score_subgraph(),
            SDPA_SCORE_SUBGRAPH,
            compute_type,
            self.sdpa_score_data_type(),
        )?;
        let softmax =
            self.quantized_sdpa_softmax(scores, softmax_aux, intermediate_type, config)?;
        let probs = softmax.probs;

        self.modify_tensor(probs, |mut tensor| {
            tensor.data_type = probability_type;
            tensor.virtual_tensor()
        })?;
        let absolute_max_s = if quantized_aux.absolute_max_s() {
            Some(self.absolute_max_infer(probs, compute_type)?)
        } else {
            None
        };
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        self.matmul_into(probs, v_fp, output, output_matmul)?;

        let absolute_max_o = if quantized_aux.absolute_max_o() {
            Some(self.absolute_max_infer(output, compute_type)?)
        } else {
            None
        };

        Ok(SdpaForwardOutputTensors {
            output,
            stats: softmax.stats,
            logit_max: softmax.logit_max,
            score_sum_exp: softmax.score_sum_exp,
            rng_dump: softmax.rng_dump,
            absolute_max_probabilities: absolute_max_s,
            absolute_max_output: absolute_max_o,
        })
    }

    /// Records MXFP8 attention with caller-declared output layouts and auxiliary tensors.
    /// Supplied auxiliary tensors enable their requests without duplicate configuration flags.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown or incompatible query/key/value or paged-cache geometry, invalid sequence-length combinations, an RNG dump requested without dropout, or unrepresentable inferred score, result, or auxiliary layouts. Caller-declared outputs must exist and match inferred output requirements. MXFP8 requests cannot use paged caches.
    pub fn sdpa_mxfp8_into(
        &mut self,
        inputs: SdpaMxfp8Inputs,
        outputs: SdpaForwardOutputTensors,
        config: &AttentionConfig,
    ) -> Result<()> {
        self.record_explicit_attention_forward(
            AttentionForwardInputs::Mxfp8(inputs),
            outputs,
            config,
        )
    }
}
