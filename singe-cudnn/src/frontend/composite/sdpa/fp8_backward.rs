use crate::{
    backend::{
        pointwise::PointwiseMode,
        reduction::ReduceTensorOperator,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::{
            SdpaFp8BackwardGradientTensors, SdpaFp8BackwardInputs, SdpaFp8BackwardOutputConfig,
            backward::SdpaBackwardProbabilities,
            backward_request::SdpaBackwardMatmulConfigs,
            infer_sdpa_output_shape, infer_sdpa_scores_shape,
            support::{
                SDPA_FP8_BACKWARD_ATTENTION_SCALE, SDPA_FP8_BACKWARD_SCORE_SUBGRAPH,
                SDPA_FP8_BACKWARD_SCORE_SUBGRAPH_BPROP,
            },
        },
        error::{Error as FrontendError, SdpaFeature},
        graph::{
            Graph, MatmulFp8Inputs, MatmulFp8QuantizeOutputs, MatmulFp8QuantizeTensors,
            reduce_last_axis,
        },
        operation::{AttentionBackwardConfig, AttentionBackwardRequest, ReductionConfig},
    },
};

struct Fp8BackwardPreparation {
    scores_shape: Shape,
    stats_shape: Shape,
    compute_type: DataType,
    attention_scale: TensorId,
    q_data_type: DataType,
    d_o_data_type: DataType,
}

impl Graph {
    /// Creates independently typed gradients while retaining FP8 probability intermediates.
    /// Fixed scalar scales and `F16` or `BF16` gradients support current scaling.
    /// Each Amax describes the descaled result before its output scale and conversion.
    /// Expansion occurs during graph preparation, after resolving the target architecture.
    /// Each gradient selects its own placement and encoding; sink gradients default to FP32.
    /// Bias gradients and RNG dumps are unavailable in the C++ FP8 backward operation.
    /// Each required Amax selects its own placement and supported encoding, defaulting to bindable FP32.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported bias-gradient or RNG-dump requests, unknown query, key or value identities, non-four-dimensional inputs, or unrepresentable inferred auxiliary-output layouts. Target and native support validation occurs during preparation.
    pub fn sdpa_fp8_backward(
        &mut self,
        inputs: SdpaFp8BackwardInputs,
        config: AttentionBackwardConfig,
        outputs: SdpaFp8BackwardOutputConfig,
    ) -> Result<SdpaFp8BackwardGradientTensors> {
        outputs.gradients.validate_quantized("fp8 sdpa backward")?;
        let checkpoint = self.mutation_checkpoint();
        let result = (|| {
            let [d_q, d_k, d_v] = self.infer_attention_gradients(
                [inputs.query, inputs.key, inputs.value],
                outputs.gradients,
                None,
                config.name(),
            )?;
            let amax_q = self.infer_attention_absolute_max(
                outputs.absolute_max_query_gradient,
                config.name(),
                "Amax_dQ",
            )?;
            let amax_k = self.infer_attention_absolute_max(
                outputs.absolute_max_key_gradient,
                config.name(),
                "Amax_dK",
            )?;
            let amax_v = self.infer_attention_absolute_max(
                outputs.absolute_max_value_gradient,
                config.name(),
                "Amax_dV",
            )?;
            let amax_p = self.infer_attention_absolute_max(
                outputs.absolute_max_probability_gradient,
                config.name(),
                "Amax_dP",
            )?;
            let sink = self.infer_attention_sink_gradient(
                inputs.query,
                outputs.gradients.sink_token_gradient,
                DataType::F32,
                config.name(),
            )?;
            let gradients = SdpaFp8BackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                sink_token_gradient: sink,
                absolute_max_query_gradient: amax_q,
                absolute_max_key_gradient: amax_k,
                absolute_max_value_gradient: amax_v,
                absolute_max_probability_gradient: amax_p,
            };
            self.sdpa_fp8_backward_into(inputs, gradients, config)?;
            Ok(gradients)
        })();
        if result.is_err() {
            self.rollback_to(checkpoint);
        }
        result
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the composite attention operation builds and validates its complete tensor graph"
    )]
    pub(crate) fn expand_fp8_attention_backward(
        &mut self,
        inputs: SdpaFp8BackwardInputs,
        gradients: SdpaFp8BackwardGradientTensors,
        config: &AttentionBackwardConfig,
    ) -> Result<()> {
        let SdpaFp8BackwardInputs {
            query: q,
            key: k,
            value: v,
            output: o,
            output_gradient: d_o,
            stats,
            descale_query: descale_q,
            descale_key: descale_k,
            descale_value: descale_v,
            descale_output: descale_o,
            descale_output_gradient: descale_d_o,
            descale_scores: descale_s,
            descale_probability_gradient: descale_d_p,
            scale_scores: scale_s,
            scale_query_gradient: scale_d_q,
            scale_key_gradient: scale_d_k,
            scale_value_gradient: scale_d_v,
            scale_probability_gradient: scale_d_p,
            ..
        } = inputs;

        let SdpaFp8BackwardGradientTensors {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            sink_token_gradient: d_sink_token,
            absolute_max_query_gradient: absolute_max_d_q,
            absolute_max_key_gradient: absolute_max_d_k,
            absolute_max_value_gradient: absolute_max_d_v,
            absolute_max_probability_gradient: absolute_max_d_p,
            ..
        } = gradients;

        let Fp8BackwardPreparation {
            scores_shape,
            stats_shape,
            compute_type,
            attention_scale,
            q_data_type,
            d_o_data_type,
        } = self.prepare_fp8_backward(&inputs, &gradients, config)?;

        let intermediate_type = self.effective_intermediate_data_type(DataType::F32);
        let modifiers = self.sdpa_backward_modifiers_with_dropout(
            config,
            None,
            config.aux_gradients(),
            &scores_shape,
        )?;
        let SdpaBackwardMatmulConfigs {
            scores: score_matmul_config,
            query_key: matmul_qk_config,
            key_value: matmul_kv_config,
        } = Self::sdpa_backward_matmul_configs(&modifiers, compute_type);

        // Keep the per-tensor descales after the raw products, as in the C++ fusion.
        let mut softmax_sum =
            self.sdpa_backward_output_dot(d_o, o, stats_shape, compute_type, intermediate_type)?;
        for descale in [descale_d_o, descale_o] {
            softmax_sum = self.sdpa_binary(
                PointwiseMode::Mul,
                softmax_sum,
                descale,
                intermediate_type,
                compute_type,
            )?;
        }
        self.sdpa_add_backward_sink_token_gradient(
            d_sink_token,
            &modifiers,
            config.aux_gradients(),
            stats,
            softmax_sum,
            &scores_shape,
            compute_type,
            intermediate_type,
        )?;
        if modifiers.dropout_scale().is_some() || modifiers.sink_token().is_some() {
            softmax_sum = self.sdpa_backward_scale_softmax_sum(
                softmax_sum,
                config,
                compute_type,
                intermediate_type,
                intermediate_type,
            )?;
        }

        let k_t = self.transpose_last_two_view(k)?;
        let mut scores =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        self.matmul_into(q, k_t, scores, score_matmul_config.clone())?;
        for scale in [attention_scale, descale_q, descale_k] {
            scores = self.sdpa_binary(
                PointwiseMode::Mul,
                scores,
                scale,
                intermediate_type,
                compute_type,
            )?;
        }
        let scaled_scores = self.sdpa_scores_with_modifiers(
            scores,
            None,
            &modifiers,
            config.score_subgraph(),
            SDPA_FP8_BACKWARD_SCORE_SUBGRAPH,
            compute_type,
            intermediate_type,
        )?;
        let SdpaBackwardProbabilities {
            probabilities: probs,
            dropped_probabilities: dropped_probs,
        } = self.reconstruct_sdpa_backward_probabilities(
            scaled_scores,
            stats,
            &scores_shape,
            &modifiers,
            compute_type,
            intermediate_type,
        )?;
        let quantized_probs = self.sdpa_binary(
            PointwiseMode::Mul,
            dropped_probs,
            scale_s,
            q_data_type,
            compute_type,
        )?;
        let probs_t = self.transpose_last_two_view(quantized_probs)?;
        self.matmul_fp8_quantize_into(
            MatmulFp8QuantizeTensors {
                inputs: MatmulFp8Inputs {
                    a: probs_t,
                    b: d_o,
                    descale_a: descale_s,
                    descale_b: descale_d_o,
                },
                scale_output: scale_d_v,
                outputs: MatmulFp8QuantizeOutputs::new(d_v, absolute_max_d_v),
            },
            matmul_kv_config.clone(),
        )?;

        let v_t = self.transpose_last_two_view(v)?;
        let mut d_p =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape).virtual_tensor());
        self.matmul_into(d_o, v_t, d_p, score_matmul_config)?;
        for descale in [descale_d_o, descale_v] {
            d_p = self.sdpa_binary(
                PointwiseMode::Mul,
                d_p,
                descale,
                intermediate_type,
                compute_type,
            )?;
        }
        d_p = self.sdpa_optional_mul(
            d_p,
            modifiers.dropout_mask(),
            compute_type,
            intermediate_type,
        )?;
        d_p = self.sdpa_binary(
            PointwiseMode::Sub,
            d_p,
            softmax_sum,
            intermediate_type,
            compute_type,
        )?;
        d_p = self.sdpa_binary(
            PointwiseMode::Mul,
            d_p,
            probs,
            intermediate_type,
            compute_type,
        )?;
        d_p = self.sdpa_optional_mul(
            d_p,
            modifiers.dropout_scale(),
            compute_type,
            intermediate_type,
        )?;
        if let Some(score_subgraph_bprop) = config.score_subgraph_bprop() {
            d_p = self.import_score_subgraph(
                score_subgraph_bprop,
                d_p,
                SDPA_FP8_BACKWARD_SCORE_SUBGRAPH_BPROP,
                None,
            )?;
        }
        d_p = self.sdpa_binary(
            PointwiseMode::Mul,
            d_p,
            attention_scale,
            intermediate_type,
            compute_type,
        )?;
        self.reduction_into(
            d_p,
            absolute_max_d_p,
            &ReductionConfig::new(ReduceTensorOperator::AbsoluteMax)
                .with_compute_type(DataType::F32),
        )?;
        let quantized_d_p = self.sdpa_binary(
            PointwiseMode::Mul,
            d_p,
            scale_d_p,
            d_o_data_type,
            compute_type,
        )?;
        self.matmul_fp8_quantize_into(
            MatmulFp8QuantizeTensors {
                inputs: MatmulFp8Inputs {
                    a: quantized_d_p,
                    b: k,
                    descale_a: descale_d_p,
                    descale_b: descale_k,
                },
                scale_output: scale_d_q,
                outputs: MatmulFp8QuantizeOutputs::new(d_q, absolute_max_d_q),
            },
            matmul_qk_config,
        )?;
        let d_p_t = self.transpose_last_two_view(quantized_d_p)?;
        self.matmul_fp8_quantize_into(
            MatmulFp8QuantizeTensors {
                inputs: MatmulFp8Inputs {
                    a: d_p_t,
                    b: q,
                    descale_a: descale_d_p,
                    descale_b: descale_q,
                },
                scale_output: scale_d_k,
                outputs: MatmulFp8QuantizeOutputs::new(d_k, absolute_max_d_k),
            },
            matmul_kv_config,
        )?;

        Ok(())
    }

    /// Records FP8 backward with caller-declared gradient layouts and Amax outputs.
    /// A supplied sink-gradient output enables that request without discarding its identity.
    /// Preparation resolves support and deterministic routing for the compilation target.
    ///
    /// # Errors
    ///
    /// This recording step currently returns no errors. Graph preparation validates the deferred operands, layouts, and target-specific support.
    pub fn sdpa_fp8_backward_into(
        &mut self,
        inputs: SdpaFp8BackwardInputs,
        gradients: SdpaFp8BackwardGradientTensors,
        config: AttentionBackwardConfig,
    ) -> Result<()> {
        self.record_attention_backward_request(AttentionBackwardRequest::Fp8 {
            inputs,
            outputs: gradients,
            config,
        });
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the composite attention operation builds and validates its complete tensor graph"
    )]
    fn prepare_fp8_backward(
        &mut self,
        inputs: &SdpaFp8BackwardInputs,
        gradients: &SdpaFp8BackwardGradientTensors,
        config: &AttentionBackwardConfig,
    ) -> Result<Fp8BackwardPreparation> {
        let SdpaFp8BackwardInputs {
            query: q,
            key: k,
            value: v,
            output: o,
            output_gradient: d_o,
            stats,
            attention_scale,
            ..
        } = *inputs;
        let SdpaFp8BackwardGradientTensors {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            sink_token_gradient: d_sink_token,
            ..
        } = *gradients;
        let query_gradient_type = self.tensor_spec(d_q)?.data_type;
        let key_gradient_type = self.tensor_spec(d_k)?.data_type;
        let value_gradient_type = self.tensor_spec(d_v)?.data_type;
        self.promote_index_tensors(config.modifiers().sequence_inputs().tensors())?;
        self.validate_attention_backward_config(q, k, v, config)?;
        let q_tensor = self.tensor_spec(q)?.clone();
        let v_tensor = self.tensor_spec(v)?.clone();
        let q_data_type = q_tensor.data_type;
        self.validate_tensor_scalar_shape(attention_scale, SDPA_FP8_BACKWARD_ATTENTION_SCALE)?;
        let is_ragged = self.validate_attention_backward_ragged_layout(
            config,
            [q, k, v, o, stats, d_o, d_q, d_k, d_v],
        )?;
        if config.aux_gradients().bias() {
            return Err(FrontendError::SdpaUnsupported {
                feature: SdpaFeature::BiasGradient,
                operation: "quantized sdpa backward",
                sm_version: None,
            }
            .into());
        }
        let query_key_width = q_tensor.shape.dimensions()[3];
        let d_v_hidden = v_tensor.shape.dimensions()[3];
        for output_type in [query_gradient_type, key_gradient_type, value_gradient_type] {
            self.validate_fp8_backward_support(
                output_type,
                query_key_width,
                d_v_hidden,
                is_ragged,
                config,
            )?;
        }
        self.validate_quantized_backward_head_dimensions(
            query_key_width,
            d_v_hidden,
            "fp8 sdpa backward",
        )?;
        if config.modifiers().left_window().is_some()
            && !config.modifiers().padding_mask()
            && q_tensor.shape.dimensions()[2] > self.tensor_spec(k)?.shape.dimensions()[2]
        {
            return Err(FrontendError::SdpaSlidingWindowSequenceLengthUnsupported {
                query_sequence_length: q_tensor.shape.dimensions()[2],
                key_sequence_length: self.tensor_spec(k)?.shape.dimensions()[2],
            }
            .into());
        }
        let compute_type = self.resolve_compute_data_type(config.softmax().compute_type());
        let attention_scale =
            self.sdpa_effective_scale(attention_scale, config.attention_scale(), compute_type)?;

        let k_tensor = self.tensor_spec(k)?.clone();
        self.validate_sdpa_tensor_dimensions([
            (
                d_q,
                q_tensor.shape.dimensions(),
                "fp8 attention query gradient",
            ),
            (
                d_k,
                k_tensor.shape.dimensions(),
                "fp8 attention key gradient",
            ),
            (
                d_v,
                v_tensor.shape.dimensions(),
                "fp8 attention value gradient",
            ),
        ])?;
        for (tensor, name) in [
            (d_q, "fp8 attention query gradient"),
            (d_k, "fp8 attention key gradient"),
            (d_v, "fp8 attention value gradient"),
        ] {
            self.validate_sdpa_tensor_layout(tensor, name)?;
        }
        if d_sink_token.is_some() != config.aux_gradients().sink_token() {
            return Err(FrontendError::SdpaOptionalOutputMismatch {
                output: "fp8 attention sink gradient",
            }
            .into());
        }
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, &v_tensor.shape)?;
        let stats_shape = reduce_last_axis(&scores_shape)?;
        self.validate_sdpa_tensor_dimensions([
            (o, output_shape.dimensions(), "fp8 sdpa backward output"),
            (
                d_o,
                output_shape.dimensions(),
                "fp8 sdpa backward output gradient",
            ),
            (
                stats,
                stats_shape.dimensions(),
                "fp8 sdpa backward statistics",
            ),
        ])?;
        self.validate_sdpa_tensor_layout(o, "fp8 sdpa backward output")?;
        self.validate_sdpa_tensor_layout(d_o, "fp8 sdpa backward output gradient")?;
        self.validate_sdpa_tensor_data_type_match(
            stats,
            DataType::F32,
            "fp8 sdpa backward statistics",
        )?;
        let d_o_data_type = self.tensor_spec(d_o)?.data_type;

        Ok(Fp8BackwardPreparation {
            scores_shape,
            stats_shape,
            compute_type,
            attention_scale,
            q_data_type,
            d_o_data_type,
        })
    }
}
