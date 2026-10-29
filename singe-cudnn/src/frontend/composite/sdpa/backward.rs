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
            SdpaBackwardGradientTensors, SdpaBackwardInputs, SdpaGradientOutputConfig,
            backward_policy::FloatingBackwardAccumulation,
            backward_request::{GradientMatmulWorkspace, SdpaBackwardMatmulConfigs},
            infer_sdpa_output_shape, infer_sdpa_scores_shape,
            support::{SDPA_BACKWARD_SCORE_SUBGRAPH, SDPA_BACKWARD_SCORE_SUBGRAPH_BPROP},
        },
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph},
        logging,
        operation::{
            AttentionBackwardConfig, AttentionBackwardRequest, AttentionMaxTotalSequenceLengths,
            PointwiseOperation, ReductionConfig, SdpaScoreModifiers,
        },
    },
    math::NanPropagation,
    version,
};

/// Reconstructed softmax values before and after forward dropout.
pub struct SdpaBackwardProbabilities {
    pub(crate) probabilities: TensorId,
    pub(crate) dropped_probabilities: TensorId,
}

struct FloatingBackwardPreparation {
    config: AttentionBackwardConfig,
    scores_shape: Shape,
    expected_stats_shape: Shape,
}

impl Graph {
    /// Records floating attention backward with caller-declared gradients and layouts.
    /// Preparation resolves support and engine selection for the compilation target.
    ///
    /// # Errors
    ///
    /// This recording step currently returns no errors. Graph preparation validates the deferred operands, layouts, and target-specific support.
    pub fn sdpa_backward_into(
        &mut self,
        inputs: SdpaBackwardInputs,
        gradients: SdpaBackwardGradientTensors,
        config: AttentionBackwardConfig,
    ) -> Result<()> {
        self.record_attention_backward_request(AttentionBackwardRequest::Floating {
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
    pub(crate) fn expand_floating_attention_backward(
        &mut self,
        inputs: SdpaBackwardInputs,
        gradients: SdpaBackwardGradientTensors,
        config: AttentionBackwardConfig,
        accumulation: FloatingBackwardAccumulation,
    ) -> Result<()> {
        let SdpaBackwardInputs {
            query: q,
            key: k,
            value: v,
            output: o,
            output_gradient: d_o,
            stats,
            scale,
            ..
        } = inputs;

        let SdpaBackwardGradientTensors {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            bias_gradient: d_bias,
            rng_dump: requested_rng_dump,
            sink_token_gradient: d_sink_token,
            ..
        } = gradients;

        let FloatingBackwardPreparation {
            config,
            scores_shape,
            expected_stats_shape,
        } = self.prepare_floating_backward(&inputs, &gradients, config)?;
        let aux_gradients = config.aux_gradients();
        let modifiers = self.sdpa_backward_modifiers_with_dropout(
            &config,
            requested_rng_dump,
            aux_gradients,
            &scores_shape,
        )?;
        let compute_type = self.resolve_compute_data_type(config.softmax().compute_type());
        let SdpaBackwardMatmulConfigs {
            scores: score_matmul_config,
            query_key: matmul_qk_config,
            key_value: matmul_kv_config,
        } = Self::sdpa_backward_matmul_configs(&modifiers, compute_type);
        let intermediate_type = self.effective_intermediate_data_type(DataType::F32);
        let grouped_workspace = GradientMatmulWorkspace {
            data_type: self.tensor_spec(q)?.data_type,
            max_total_tokens: config.max_total_sequence_lengths().key_value,
        };

        let scores =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        let k_t = self.transpose_last_two_view(k)?;
        let scale = self.sdpa_effective_scale(scale, config.attention_scale(), compute_type)?;
        self.matmul_into(q, k_t, scores, score_matmul_config.clone())?;

        let scaled_scores = self.sdpa_scores_with_modifiers(
            scores,
            Some(scale),
            &modifiers,
            config.score_subgraph(),
            SDPA_BACKWARD_SCORE_SUBGRAPH,
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
        self.sdpa_backward_gradient_matmul(
            probs_t,
            d_o,
            d_v,
            matmul_kv_config.clone(),
            grouped_workspace,
        )?;

        let v_t = self.transpose_last_two_view(v)?;
        let mut d_p =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        self.matmul_into(d_o, v_t, d_p, score_matmul_config)?;
        d_p = self.sdpa_optional_mul(
            d_p,
            modifiers.dropout_mask(),
            compute_type,
            intermediate_type,
        )?;

        let mut softmax_sum = self.sdpa_backward_output_dot(
            d_o,
            o,
            expected_stats_shape,
            compute_type,
            intermediate_type,
        )?;

        self.sdpa_add_backward_sink_token_gradient(
            d_sink_token,
            &modifiers,
            aux_gradients,
            stats,
            softmax_sum,
            &scores_shape,
            compute_type,
            intermediate_type,
        )?;

        softmax_sum = self.sdpa_backward_scale_softmax_sum(
            softmax_sum,
            &config,
            compute_type,
            intermediate_type,
            DataType::F32,
        )?;

        let dot_spec = self.attention_workspace_spec(
            self.tensor_spec(softmax_sum)?.clone(),
            stats,
            config.max_total_sequence_lengths().query,
            1,
        )?;
        self.replace_tensor(softmax_sum, dot_spec)?;

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
        if let Some(d_bias) = d_bias {
            self.reduction_into(
                d_s,
                d_bias,
                &ReductionConfig::new(ReduceTensorOperator::Add).with_compute_type(compute_type),
            )?;
        } else if aux_gradients.bias() {
            return Err(FrontendError::SdpaOutputMissing {
                operation: "sdpa backward",
                output: "bias gradient",
            }
            .into());
        }
        if let Some(score_subgraph_bprop) = config.score_subgraph_bprop() {
            d_s = self.import_score_subgraph(
                score_subgraph_bprop,
                d_s,
                SDPA_BACKWARD_SCORE_SUBGRAPH_BPROP,
                None,
            )?;
        }
        d_s = self.sdpa_binary(
            PointwiseMode::Mul,
            d_s,
            scale,
            intermediate_type,
            compute_type,
        )?;

        let d_s_t = self.transpose_last_two_view(d_s)?;
        self.sdpa_backward_gradient_matmul(d_s_t, q, d_k, matmul_kv_config, grouped_workspace)?;
        if accumulation == FloatingBackwardAccumulation::Probability {
            self.matmul_into(d_s, k, d_q, matmul_qk_config)?;
        } else {
            let spec = self.attention_workspace_spec(
                TensorSpec::new(
                    DataType::F32,
                    Shape::contiguous(self.tensor_spec(d_q)?.shape.dimensions())?,
                ),
                d_q,
                config.max_total_sequence_lengths().query,
                1,
            )?;
            let accumulator = self.tensor(spec);
            self.matmul_into(d_s, k, accumulator, matmul_qk_config)?;
            self.pointwise_into(PointwiseOperation::Unary {
                name: None,
                mode: PointwiseMode::Identity,
                input: accumulator,
                output: d_q,
                compute_type: Some(compute_type),
                nan_propagation: NanPropagation::NotPropagate,
                alpha1: 1.0,
                axis: None,
            })?;
        }

        Ok(())
    }

    /// Creates layout-preserving gradients and independently requested auxiliary outputs.
    /// An inferred bias gradient follows the bias layout, or the score shape when no bias is supplied.
    /// Each gradient selects placement and encoding independently.
    /// Bias gradients and RNG dumps default to FP32; sink gradients default to math precision.
    /// Auxiliary defaults are independent of graph I/O and intermediate policies.
    /// RNG dumps retain their native FP32 encoding contract.
    /// Returns all requested gradients and auxiliary outputs in the group accepted by explicit construction.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown query, key, value or requested bias references, non-four-dimensional attention inputs, incompatible query/key or value geometry, or unrepresentable inferred score, statistic, or auxiliary-gradient layouts.
    pub fn sdpa_backward(
        &mut self,
        inputs: SdpaBackwardInputs,
        config: AttentionBackwardConfig,
        outputs: SdpaGradientOutputConfig,
    ) -> Result<SdpaBackwardGradientTensors> {
        let checkpoint = self.mutation_checkpoint();
        let result = (|| {
            let [q, k, v] = [inputs.query, inputs.key, inputs.value];
            for tensor in [q, k, v] {
                self.validate_tensor_rank(tensor, 4, "attention backward input")?;
            }
            let q_tensor = self.tensor_spec(q)?;
            let k_tensor = self.tensor_spec(k)?;
            let v_tensor = self.tensor_spec(v)?;
            if q_tensor.shape.dimensions()[3] != k_tensor.shape.dimensions()[3] {
                return Err(FrontendError::SdpaQueryKeyShapeMismatch {
                    query: q_tensor.shape.dimensions().to_vec(),
                    key: k_tensor.shape.dimensions().to_vec(),
                }
                .into());
            }
            let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
            infer_sdpa_output_shape(&scores_shape, &v_tensor.shape)?;
            let [d_q, d_k, d_v] =
                self.infer_attention_gradients([q, k, v], outputs, None, config.name())?;
            let d_bias = if let Some(output) = outputs.bias_gradient {
                let shape = config
                    .modifiers()
                    .bias()
                    .map(|bias| self.tensor_spec(bias).map(|tensor| tensor.shape.clone()))
                    .transpose()?
                    .unwrap_or_else(|| scores_shape.clone());
                Some(self.inferred_tensor(
                    output.into_spec(DataTypePolicy::default(), DataType::F32, shape),
                    config.name(),
                    "Dbias",
                ))
            } else {
                None
            };
            let rng_dump = outputs.rng_dump.map(|output| {
                self.inferred_tensor(
                    output.into_spec(
                        DataTypePolicy::default(),
                        DataType::F32,
                        scores_shape.clone(),
                    ),
                    config.name(),
                    "RNG_DUMP",
                )
            });
            let sink_type = self.resolve_compute_data_type(config.softmax().compute_type());
            let d_sink_token = self.infer_attention_sink_gradient(
                q,
                outputs.sink_token_gradient,
                sink_type,
                config.name(),
            )?;
            let gradients = SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: d_bias,
                rng_dump,
                sink_token_gradient: d_sink_token,
            };
            self.sdpa_backward_into(inputs, gradients, config)?;
            Ok(gradients)
        })();
        if result.is_err() {
            self.rollback_to(checkpoint);
        }
        result
    }

    fn prepare_floating_backward(
        &mut self,
        inputs: &SdpaBackwardInputs,
        gradients: &SdpaBackwardGradientTensors,
        config: AttentionBackwardConfig,
    ) -> Result<FloatingBackwardPreparation> {
        let SdpaBackwardInputs {
            query: q,
            key: k,
            value: v,
            output: o,
            output_gradient: d_o,
            stats,
            ..
        } = *inputs;
        let SdpaBackwardGradientTensors {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            ..
        } = *gradients;
        self.promote_index_tensors(config.modifiers().sequence_inputs().tensors())?;
        self.validate_attention_backward_config(q, k, v, &config)?;
        self.validate_attention_backward_ragged_layout(
            &config,
            [q, k, v, o, stats, d_o, d_q, d_k, d_v],
        )?;
        self.validate_attention_backward_support_surface_for_version(
            version()?.raw(),
            q,
            k,
            v,
            o,
            stats,
            d_o,
            d_q,
            d_k,
            d_v,
            &config,
        )?;
        let validated = self.validate_sdpa_backward_tensors(inputs, gradients, &config)?;
        let query_hidden = self.shape(q)?.dimensions()[3];
        let value_hidden = self.shape(v)?.dimensions()[3];
        let sm = self.sm_version();
        // Match C++'s native workspace workarounds after validating the originally requested packed hints.
        let config = if !config.max_total_sequence_lengths().is_empty()
            && (query_hidden % 16 != 0
                || value_hidden % 16 != 0
                || sm.is_some_and(|sm| matches!(sm / 10, 8 | 12)))
        {
            logging::diagnostic!(
                "ignoring sdpa backward total-token hints for sm {sm:?} and head widths ({query_hidden}, {value_hidden}); using full workspace",
            );
            config.with_max_total_sequence_lengths(AttentionMaxTotalSequenceLengths::default())
        } else {
            config
        };
        Ok(FloatingBackwardPreparation {
            config,
            scores_shape: validated.scores_shape,
            expected_stats_shape: validated.expected_stats_shape,
        })
    }

    pub(crate) fn reconstruct_sdpa_backward_probabilities(
        &mut self,
        scaled_scores: TensorId,
        stats: TensorId,
        scores_shape: &Shape,
        modifiers: &SdpaScoreModifiers,
        compute_type: DataType,
        intermediate_type: DataType,
    ) -> Result<SdpaBackwardProbabilities> {
        let shifted_scores = self.sdpa_binary(
            PointwiseMode::Sub,
            scaled_scores,
            stats,
            intermediate_type,
            compute_type,
        )?;
        let probs =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        self.pointwise_into(PointwiseOperation::Unary {
            name: None,
            mode: PointwiseMode::Exp,
            input: shifted_scores,
            output: probs,
            compute_type: Some(compute_type),
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: None,
        })?;
        let dropped_probs = self.sdpa_apply_dropout_with_data_type(
            probs,
            modifiers.dropout_mask(),
            modifiers.dropout_scale(),
            compute_type,
            intermediate_type,
        )?;
        Ok(SdpaBackwardProbabilities {
            probabilities: probs,
            dropped_probabilities: dropped_probs,
        })
    }
}
