mod inputs;

use crate::{
    backend::{
        pointwise::PointwiseMode,
        reduction::ReduceTensorOperator,
        tensor::{Shape, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::{
            SdpaMxfp8BackwardGradientTensors, SdpaMxfp8BackwardInputs,
            SdpaMxfp8BackwardOutputConfig,
            backward::SdpaBackwardProbabilities,
            backward_request::{GradientMatmulWorkspace, SdpaBackwardMatmulConfigs},
            infer_sdpa_scores_shape,
            support::{
                SDPA_MXFP8_BACKWARD_ATTENTION_SCALE, SDPA_MXFP8_BACKWARD_D_O_LAYOUT,
                SDPA_MXFP8_BACKWARD_D_O_QUANTIZED_LAYOUT, SDPA_MXFP8_BACKWARD_D_O_QUANTIZED_SHAPE,
                SDPA_MXFP8_BACKWARD_D_O_T_LAYOUT, SDPA_MXFP8_BACKWARD_D_O_T_SHAPE,
                SDPA_MXFP8_BACKWARD_K_LAYOUT, SDPA_MXFP8_BACKWARD_K_T_LAYOUT,
                SDPA_MXFP8_BACKWARD_K_T_SHAPE, SDPA_MXFP8_BACKWARD_O_LAYOUT,
                SDPA_MXFP8_BACKWARD_O_SHAPE, SDPA_MXFP8_BACKWARD_Q_LAYOUT,
                SDPA_MXFP8_BACKWARD_Q_T_LAYOUT, SDPA_MXFP8_BACKWARD_Q_T_SHAPE,
                SDPA_MXFP8_BACKWARD_SCALE_D_O, SDPA_MXFP8_BACKWARD_SCALE_D_O_T,
                SDPA_MXFP8_BACKWARD_SCALE_K, SDPA_MXFP8_BACKWARD_SCALE_K_T,
                SDPA_MXFP8_BACKWARD_SCALE_Q, SDPA_MXFP8_BACKWARD_SCALE_Q_T,
                SDPA_MXFP8_BACKWARD_SCALE_V, SDPA_MXFP8_BACKWARD_SCORE_SUBGRAPH,
                SDPA_MXFP8_BACKWARD_SCORE_SUBGRAPH_BPROP, SDPA_MXFP8_BACKWARD_STATS_DATA_TYPE,
                SDPA_MXFP8_BACKWARD_STATS_SHAPE, SDPA_MXFP8_BACKWARD_V_LAYOUT,
            },
        },
        error::{Error as FrontendError, SdpaFeature},
        graph::{Graph, reduce_last_axis},
        operation::{AttentionBackwardConfig, AttentionBackwardRequest, ReductionConfig},
    },
};

struct Mxfp8BackwardPreparation {
    query_shape: Shape,
    key_shape: Shape,
    compute_type: DataType,
    intermediate_type: DataType,
}

impl Graph {
    /// Creates independently typed MXFP8 gradients and requested Amax outputs measured after conversion.
    /// Each gradient selects placement and encoding independently, with a BF16 fallback.
    /// Sink gradients select their own placement and encoding, with an FP32 default.
    /// Bias gradients and RNG dumps remain unsupported for quantized backward inference.
    /// Unrequested Amax outputs create no tensors, bindings or reduction operations.
    /// Preparation resolves the target architecture and deterministic routing.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported bias-gradient or RNG-dump requests, unknown query, key or value identities, non-four-dimensional inputs, or unrepresentable inferred auxiliary-output layouts. Target and native support validation occurs during preparation.
    pub fn sdpa_mxfp8_backward(
        &mut self,
        inputs: SdpaMxfp8BackwardInputs,
        config: AttentionBackwardConfig,
        outputs: SdpaMxfp8BackwardOutputConfig,
    ) -> Result<SdpaMxfp8BackwardGradientTensors> {
        outputs
            .gradients
            .validate_quantized("mxfp8 sdpa backward")?;
        let checkpoint = self.mutation_checkpoint();
        let result = (|| {
            let [d_q, d_k, d_v] = self.infer_attention_gradients(
                [inputs.query, inputs.key, inputs.value],
                outputs.gradients,
                Some(DataType::BF16),
                config.name(),
            )?;
            let amax_q = outputs
                .absolute_max_query_gradient
                .map(|output| self.infer_attention_absolute_max(output, config.name(), "Amax_dQ"))
                .transpose()?;
            let amax_k = outputs
                .absolute_max_key_gradient
                .map(|output| self.infer_attention_absolute_max(output, config.name(), "Amax_dK"))
                .transpose()?;
            let amax_v = outputs
                .absolute_max_value_gradient
                .map(|output| self.infer_attention_absolute_max(output, config.name(), "Amax_dV"))
                .transpose()?;
            let sink = self.infer_attention_sink_gradient(
                inputs.query,
                outputs.gradients.sink_token_gradient,
                DataType::F32,
                config.name(),
            )?;
            let gradients = SdpaMxfp8BackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                sink_token_gradient: sink,
                absolute_max_query_gradient: amax_q,
                absolute_max_key_gradient: amax_k,
                absolute_max_value_gradient: amax_v,
            };
            self.sdpa_mxfp8_backward_into(inputs, gradients, config)?;
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
    pub(crate) fn expand_mxfp8_attention_backward(
        &mut self,
        inputs: SdpaMxfp8BackwardInputs,
        gradients: SdpaMxfp8BackwardGradientTensors,
        config: &AttentionBackwardConfig,
    ) -> Result<()> {
        let SdpaMxfp8BackwardInputs {
            query: q,
            sequence_scaled_query: q_t,
            key: k,
            sequence_scaled_key: k_t,
            value: v,
            output: o,
            output_gradient: d_o,
            quantized_output_gradient: d_o_quantized,
            sequence_scaled_output_gradient: d_o_t,
            stats,
            scale_query: scale_q,
            sequence_scale_query: query_sequence_scale,
            scale_key: scale_k,
            sequence_scale_key: key_sequence_scale,
            scale_value: scale_v,
            scale_output_gradient: scale_d_o,
            sequence_scale_output_gradient: scale_d_o_t,
            attention_scale,
            ..
        } = inputs;

        let Mxfp8BackwardPreparation {
            query_shape,
            key_shape,
            compute_type,
            intermediate_type,
        } = self.prepare_mxfp8_backward(&inputs, &gradients, config)?;

        let attention_scale = self.mxfp8_attention_scale_tensor(
            attention_scale,
            SDPA_MXFP8_BACKWARD_ATTENTION_SCALE,
            compute_type,
        )?;
        let attention_scale =
            self.sdpa_effective_scale(attention_scale, config.attention_scale(), compute_type)?;

        let q_fp =
            self.mxfp8_dequantize_input_infer(q, scale_q, intermediate_type, compute_type)?;
        let k_fp = self.mxfp8_dequantize_transposed_input_infer(
            k,
            scale_k,
            intermediate_type,
            compute_type,
        )?;
        let v_t_fp = self.mxfp8_dequantize_transposed_input_infer(
            v,
            scale_v,
            intermediate_type,
            compute_type,
        )?;
        let d_o_fp = self.mxfp8_dequantize_input_infer(
            d_o_quantized,
            scale_d_o,
            intermediate_type,
            compute_type,
        )?;
        let d_o_t_fp = self.mxfp8_dequantize_sequence_scaled_input_infer(
            d_o_t,
            scale_d_o_t,
            intermediate_type,
            compute_type,
        )?;
        let k_t_fp = self.mxfp8_dequantize_sequence_scaled_input_infer(
            k_t,
            key_sequence_scale,
            intermediate_type,
            compute_type,
        )?;
        let q_t_fp = self.mxfp8_dequantize_sequence_scaled_input_infer(
            q_t,
            query_sequence_scale,
            intermediate_type,
            compute_type,
        )?;
        let scores_shape = infer_sdpa_scores_shape(&query_shape, &key_shape)?;
        let modifiers = self.sdpa_backward_modifiers_with_dropout(
            config,
            None,
            config.aux_gradients(),
            &scores_shape,
        )?;
        let grouped_workspace = GradientMatmulWorkspace {
            data_type: DataType::F32,
            max_total_tokens: None,
        };
        let SdpaBackwardMatmulConfigs {
            scores: score_matmul_config,
            query_key: matmul_qk_config,
            key_value: matmul_kv_config,
        } = Self::sdpa_backward_matmul_configs(&modifiers, compute_type);
        let scores =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        self.matmul_into(q_fp, k_fp, scores, score_matmul_config.clone())?;

        let scaled_scores = self.sdpa_scores_with_modifiers(
            scores,
            Some(attention_scale),
            &modifiers,
            config.score_subgraph(),
            SDPA_MXFP8_BACKWARD_SCORE_SUBGRAPH,
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
        let probs_t = self.transpose_last_two_view(dropped_probs)?;
        let d_v = gradients.value_gradient;
        self.sdpa_backward_gradient_matmul(
            probs_t,
            d_o_t_fp,
            d_v,
            matmul_kv_config.clone(),
            grouped_workspace,
        )?;

        let d_p =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        self.matmul_into(d_o_fp, v_t_fp, d_p, score_matmul_config)?;
        let d_p = self.sdpa_optional_mul(
            d_p,
            modifiers.dropout_mask(),
            compute_type,
            intermediate_type,
        )?;
        let mut softmax_sum = self.sdpa_backward_output_dot(
            d_o,
            o,
            reduce_last_axis(&scores_shape)?,
            compute_type,
            intermediate_type,
        )?;
        self.sdpa_add_backward_sink_token_gradient(
            gradients.sink_token_gradient,
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
        let mut d_s = self.sdpa_binary(
            PointwiseMode::Sub,
            d_p,
            softmax_sum,
            intermediate_type,
            compute_type,
        )?;
        d_s = self.sdpa_binary(
            PointwiseMode::Mul,
            d_s,
            probs,
            intermediate_type,
            compute_type,
        )?;
        d_s = self.sdpa_optional_mul(
            d_s,
            modifiers.dropout_scale(),
            compute_type,
            intermediate_type,
        )?;
        if let Some(score_subgraph_bprop) = config.score_subgraph_bprop() {
            d_s = self.import_score_subgraph(
                score_subgraph_bprop,
                d_s,
                SDPA_MXFP8_BACKWARD_SCORE_SUBGRAPH_BPROP,
                None,
            )?;
        }
        d_s = self.sdpa_binary(
            PointwiseMode::Mul,
            d_s,
            attention_scale,
            intermediate_type,
            compute_type,
        )?;
        let d_q = gradients.query_gradient;
        self.matmul_into(d_s, k_t_fp, d_q, matmul_qk_config)?;
        let transposed_score_gradient = self.transpose_last_two_view(d_s)?;
        let d_k = gradients.key_gradient;
        self.sdpa_backward_gradient_matmul(
            transposed_score_gradient,
            q_t_fp,
            d_k,
            matmul_kv_config,
            grouped_workspace,
        )?;

        for (input, output) in [
            (d_q, gradients.absolute_max_query_gradient),
            (d_k, gradients.absolute_max_key_gradient),
            (d_v, gradients.absolute_max_value_gradient),
        ] {
            if let Some(output) = output {
                self.reduction_into(
                    input,
                    output,
                    &ReductionConfig::new(ReduceTensorOperator::AbsoluteMax)
                        .with_compute_type(compute_type),
                )?;
            }
        }
        Ok(())
    }

    /// Records MXFP8 backward with caller-declared gradient layouts and optional Amax outputs.
    /// Amax reductions observe the gradient values after conversion to their declared encodings.
    ///
    /// # Errors
    ///
    /// This recording step currently returns no errors. Graph preparation validates the deferred operands, layouts, and target-specific support.
    pub fn sdpa_mxfp8_backward_into(
        &mut self,
        inputs: SdpaMxfp8BackwardInputs,
        gradients: SdpaMxfp8BackwardGradientTensors,
        config: AttentionBackwardConfig,
    ) -> Result<()> {
        self.record_attention_backward_request(AttentionBackwardRequest::Mxfp8 {
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
    fn prepare_mxfp8_backward(
        &mut self,
        inputs: &SdpaMxfp8BackwardInputs,
        gradients: &SdpaMxfp8BackwardGradientTensors,
        config: &AttentionBackwardConfig,
    ) -> Result<Mxfp8BackwardPreparation> {
        let SdpaMxfp8BackwardInputs {
            query: q,
            sequence_scaled_query: q_t,
            key: k,
            sequence_scaled_key: k_t,
            value: v,
            output: o,
            output_gradient: d_o,
            quantized_output_gradient: d_o_quantized,
            sequence_scaled_output_gradient: d_o_t,
            stats,
            scale_query: scale_q,
            sequence_scale_query: query_sequence_scale,
            scale_key: scale_k,
            sequence_scale_key: key_sequence_scale,
            scale_value: scale_v,
            scale_output_gradient: scale_d_o,
            sequence_scale_output_gradient: scale_d_o_t,
            ..
        } = *inputs;
        let intermediate_type = self.effective_intermediate_data_type(DataType::F32);
        let compute_type = self.resolve_compute_data_type(config.softmax().compute_type());
        self.promote_index_tensors(config.modifiers().sequence_inputs().tensors())?;
        self.validate_attention_backward_config(q, k, v, config)?;
        if config.aux_gradients().bias() {
            return Err(FrontendError::SdpaUnsupported {
                feature: SdpaFeature::BiasGradient,
                operation: "quantized sdpa backward",
                sm_version: None,
            }
            .into());
        }
        let is_ragged = self.validate_attention_backward_ragged_layout(
            config,
            [
                q,
                k,
                v,
                o,
                stats,
                d_o,
                gradients.query_gradient,
                gradients.key_gradient,
                gradients.value_gradient,
            ],
        )?;
        for gradient in [
            gradients.query_gradient,
            gradients.key_gradient,
            gradients.value_gradient,
        ] {
            self.validate_mxfp8_backward_support(
                self.tensor_spec(gradient)?.data_type,
                is_ragged,
                config,
            )?;
        }

        self.validate_sdpa_tensor_layouts([
            (q, SDPA_MXFP8_BACKWARD_Q_LAYOUT),
            (q_t, SDPA_MXFP8_BACKWARD_Q_T_LAYOUT),
            (k, SDPA_MXFP8_BACKWARD_K_LAYOUT),
            (k_t, SDPA_MXFP8_BACKWARD_K_T_LAYOUT),
            (v, SDPA_MXFP8_BACKWARD_V_LAYOUT),
            (o, SDPA_MXFP8_BACKWARD_O_LAYOUT),
            (d_o, SDPA_MXFP8_BACKWARD_D_O_LAYOUT),
            (d_o_quantized, SDPA_MXFP8_BACKWARD_D_O_QUANTIZED_LAYOUT),
            (d_o_t, SDPA_MXFP8_BACKWARD_D_O_T_LAYOUT),
        ])?;

        for (scale, name) in [
            (scale_q, SDPA_MXFP8_BACKWARD_SCALE_Q),
            (query_sequence_scale, SDPA_MXFP8_BACKWARD_SCALE_Q_T),
            (scale_k, SDPA_MXFP8_BACKWARD_SCALE_K),
            (key_sequence_scale, SDPA_MXFP8_BACKWARD_SCALE_K_T),
            (scale_v, SDPA_MXFP8_BACKWARD_SCALE_V),
            (scale_d_o, SDPA_MXFP8_BACKWARD_SCALE_D_O),
            (scale_d_o_t, SDPA_MXFP8_BACKWARD_SCALE_D_O_T),
        ] {
            self.validate_mxfp8_scale_tensor(scale, name)?;
        }
        self.validate_mxfp8_backward_scale_dimensions(inputs)?;

        let q_tensor = self.tensor_spec(q)?.clone();
        let k_tensor = self.tensor_spec(k)?.clone();
        let v_tensor = self.tensor_spec(v)?.clone();
        let d_o_tensor = self.tensor_spec(d_o)?.clone();
        self.validate_sdpa_tensor_dimensions([
            (
                gradients.query_gradient,
                q_tensor.shape.dimensions(),
                "mxfp8 attention query gradient",
            ),
            (
                gradients.key_gradient,
                k_tensor.shape.dimensions(),
                "mxfp8 attention key gradient",
            ),
            (
                gradients.value_gradient,
                v_tensor.shape.dimensions(),
                "mxfp8 attention value gradient",
            ),
        ])?;
        if config.modifiers().left_window().is_some()
            && !config.modifiers().padding_mask()
            && q_tensor.shape.dimensions()[2] > k_tensor.shape.dimensions()[2]
        {
            return Err(FrontendError::SdpaSlidingWindowSequenceLengthUnsupported {
                query_sequence_length: q_tensor.shape.dimensions()[2],
                key_sequence_length: k_tensor.shape.dimensions()[2],
            }
            .into());
        }

        self.validate_sdpa_tensor_dimensions([
            (
                q_t,
                q_tensor.shape.dimensions(),
                SDPA_MXFP8_BACKWARD_Q_T_SHAPE,
            ),
            (
                k_t,
                k_tensor.shape.dimensions(),
                SDPA_MXFP8_BACKWARD_K_T_SHAPE,
            ),
            (
                d_o_quantized,
                d_o_tensor.shape.dimensions(),
                SDPA_MXFP8_BACKWARD_D_O_QUANTIZED_SHAPE,
            ),
            (
                d_o_t,
                d_o_tensor.shape.dimensions(),
                SDPA_MXFP8_BACKWARD_D_O_T_SHAPE,
            ),
            (
                o,
                d_o_tensor.shape.dimensions(),
                SDPA_MXFP8_BACKWARD_O_SHAPE,
            ),
        ])?;
        let expected_stats_shape = reduce_last_axis(&q_tensor.shape)?;
        self.validate_sdpa_tensor_dimensions([(
            stats,
            expected_stats_shape.dimensions(),
            SDPA_MXFP8_BACKWARD_STATS_SHAPE,
        )])?;
        self.validate_sdpa_tensor_data_types([(
            stats,
            compute_type,
            SDPA_MXFP8_BACKWARD_STATS_DATA_TYPE,
        )])?;
        self.validate_quantized_backward_head_dimensions(
            q_tensor.shape.dimensions()[3],
            v_tensor.shape.dimensions()[3],
            "mxfp8 sdpa backward",
        )?;

        Ok(Mxfp8BackwardPreparation {
            query_shape: q_tensor.shape,
            key_shape: k_tensor.shape,
            compute_type,
            intermediate_type,
        })
    }
}
