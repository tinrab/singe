use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::{
            SdpaBackwardAuxOutputs, SdpaBackwardGradientTensors, SdpaBackwardInputs,
            SdpaBackwardOutputs, SdpaFp8BackwardAuxOutputs, SdpaFp8BackwardGradientTensors,
            SdpaFp8BackwardInputs, SdpaFp8BackwardOutputs, SdpaInputs,
            SdpaMxfp8BackwardGradientTensors, SdpaMxfp8BackwardInputs, SdpaMxfp8BackwardOutputs,
            SdpaOutputTensors,
            support::{sdpa_mask_scalar, sdpa_scalar_tensor},
        },
        graph::Graph,
        infer::*,
        operation::*,
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    reduction::ReduceTensorOperator,
    tensor::{Shape, TensorId, TensorSpec},
    version,
};

impl Graph {
    pub(in crate::frontend::composite::sdpa) fn sdpa_with_modifiers_internal(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        scale: TensorId,
        output: TensorId,
        stats: Option<TensorId>,
        logit_max: Option<TensorId>,
        score_sum_exp: Option<TensorId>,
        modifiers: SdpaScoreModifiers,
        score_subgraph: Option<&SdpaScoreSubgraph>,
        config: SoftmaxConfig,
    ) -> Result<()> {
        self.validate_sdpa_score_modifiers(q, k, v, &modifiers)?;
        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let output_tensor = self.tensor_config(output)?.clone();
        let (k_matmul, k_tensor) =
            if q_tensor.shape.dimensions()[3] == k_tensor.shape.dimensions()[3] {
                let k_t = self.transpose_last_two(k)?;
                (k_t, self.tensor_config(k_t)?.clone())
            } else {
                (k, k_tensor)
            };
        let v_tensor = self.tensor_config(v)?.clone();

        if q_tensor.shape.dimensions().len() != 4
            || k_tensor.shape.dimensions().len() != 4
            || v_tensor.shape.dimensions().len() != 4
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa bias rank".into(),
            });
        }
        let is_ragged = q_tensor.ragged_offset.is_some()
            || self.tensor_config(k)?.ragged_offset.is_some()
            || v_tensor.ragged_offset.is_some()
            || output_tensor.ragged_offset.is_some();
        let has_padding_mask = modifiers.padding_mask;
        let has_seq_lens = modifiers.sequence_length_query.is_some()
            && modifiers.sequence_length_key_value.is_some();
        if has_seq_lens
            && !has_padding_mask
            && !modifiers.causal_bottom_right
            && score_subgraph.is_none()
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa seq lens".into(),
            });
        }
        if is_ragged
            && !has_padding_mask
            && modifiers.additive_mask.is_none()
            && score_subgraph.is_none()
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa ragged offsets".into(),
            });
        }
        if is_ragged
            && self.sm_version().is_some_and(|sm_version| sm_version < 90)
            && version()? < 91801
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa ragged offsets".into(),
            });
        }

        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        let scores =
            self.tensor(TensorSpec::new(q_tensor.data_type, scores_shape).virtual_tensor());
        self.matmul(q, k_matmul, scores, config.compute_type())?;

        let mut scores_with_modifiers =
            self.pointwise_binary_infer(scores, scale, PointwiseMode::Mul, config.compute_type())?;
        if let Some(bias) = modifiers.bias {
            scores_with_modifiers = self.pointwise_binary_infer(
                scores_with_modifiers,
                bias,
                PointwiseMode::Add,
                config.compute_type(),
            )?;
        }
        if let Some(score_subgraph) = score_subgraph {
            scores_with_modifiers = self.import_score_subgraph(
                score_subgraph,
                scores_with_modifiers,
                "sdpa score subgraph",
            )?;
        }
        if let Some(sink_token) = modifiers.sink_token {
            let sink_token_tensor = self.tensor_config(sink_token)?.clone();
            if sink_token_tensor.shape.dimensions().len() != 4 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa sink token rank".into(),
                });
            }
            scores_with_modifiers = self.pointwise_binary_infer(
                scores_with_modifiers,
                sink_token,
                PointwiseMode::Add,
                config.compute_type(),
            )?;
        }
        if let Some(alibi_slopes) = modifiers.alibi_slopes {
            let alibi_bias = self.alibi_bias(scores_with_modifiers, alibi_slopes)?;
            scores_with_modifiers = self.pointwise_binary_infer(
                scores_with_modifiers,
                alibi_bias,
                PointwiseMode::Add,
                config.compute_type(),
            )?;
        }
        if let Some(additive_mask) = modifiers.additive_mask {
            scores_with_modifiers = self.pointwise_binary_infer(
                scores_with_modifiers,
                additive_mask,
                PointwiseMode::Add,
                config.compute_type(),
            )?;
        }
        if modifiers.causal_mask {
            let causal_mask = self.sdpa_causal_mask_infer(q, k_matmul)?;
            scores_with_modifiers = self.pointwise_binary_infer(
                scores_with_modifiers,
                causal_mask,
                PointwiseMode::Add,
                config.compute_type(),
            )?;
        }
        if let Some((left_window, right_window)) = modifiers.sliding_window {
            let sliding_mask =
                self.sdpa_sliding_window_mask_infer(q, k_matmul, left_window, right_window)?;
            scores_with_modifiers = self.pointwise_binary_infer(
                scores_with_modifiers,
                sliding_mask,
                PointwiseMode::Add,
                config.compute_type(),
            )?;
        }
        if let (Some(sequence_length_query), Some(sequence_length_key_value)) = (
            modifiers.sequence_length_query,
            modifiers.sequence_length_key_value,
        ) {
            let mask = if modifiers.causal_bottom_right {
                Some(self.causal_bottom_right_mask(
                    scores_with_modifiers,
                    sequence_length_query,
                    sequence_length_key_value,
                )?)
            } else if modifiers.padding_mask {
                Some(self.padding_mask(
                    scores_with_modifiers,
                    sequence_length_query,
                    sequence_length_key_value,
                )?)
            } else {
                None
            };
            if let Some(mask) = mask {
                scores_with_modifiers = self.pointwise_binary_infer(
                    scores_with_modifiers,
                    mask,
                    PointwiseMode::Add,
                    config.compute_type(),
                )?;
            }
        }

        let softmax_outputs = self.softmax_composite_infer(scores_with_modifiers, config)?;
        let stats_value = softmax_outputs.stats;
        let max = softmax_outputs.max;
        let sum = softmax_outputs.sum;
        let mut probs = softmax_outputs.output;
        if let (Some(dropout_mask), Some(dropout_scale)) =
            (modifiers.dropout_mask, modifiers.dropout_scale)
        {
            let masked_probs = self.pointwise_binary_infer(
                probs,
                dropout_mask,
                PointwiseMode::Mul,
                config.compute_type(),
            )?;
            probs = self.pointwise_binary_infer(
                masked_probs,
                dropout_scale,
                PointwiseMode::Mul,
                config.compute_type(),
            )?;
        }

        let inferred_output_shape = infer_sdpa_output_shape(self.shape(probs)?, &v_tensor.shape)?;
        if output_tensor.shape.dimensions() != inferred_output_shape.dimensions() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa output shape".into(),
            });
        }

        self.matmul(probs, v, output, config.compute_type())?;

        if let Some(stats) = stats {
            self.reshape(stats_value, stats)?;
        }
        if let Some(logit_max) = logit_max {
            self.reshape(max, logit_max)?;
        }
        if let Some(score_sum_exp) = score_sum_exp {
            self.reshape(sum, score_sum_exp)?;
        }

        Ok(())
    }

    pub fn sdpa_with_modifiers(
        &mut self,
        inputs: SdpaInputs,
        outputs: SdpaOutputTensors,
        modifiers: SdpaScoreModifiers,
        config: SoftmaxConfig,
    ) -> Result<()> {
        let SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        } = inputs;
        self.sdpa_with_modifiers_internal(
            q,
            k,
            v,
            scale,
            outputs.output,
            outputs.stats,
            outputs.logit_max,
            outputs.score_sum_exp,
            modifiers,
            None,
            config,
        )
    }

    /// Adds scaled dot product attention backward with explicit gradient outputs.
    ///
    /// SDPA backward computes gradients for Q, K, and V, with optional gradients
    /// for score modifiers such as bias and sink token.
    pub fn sdpa_backward(
        &mut self,
        inputs: SdpaBackwardInputs,
        gradients: SdpaBackwardGradientTensors,
        config: AttentionBackwardConfig,
    ) -> Result<()> {
        let SdpaBackwardInputs {
            query: q,
            key: k,
            value: v,
            output: o,
            output_gradient: d_o,
            stats,
            scale,
        } = inputs;
        let SdpaBackwardGradientTensors {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            bias_gradient: d_bias,
            rng_dump: requested_rng_dump,
            sink_token_gradient: d_sink_token,
        } = gradients;
        self.validate_attention_backward_config(q, k, v, &config)?;
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
        let validated = self.validate_sdpa_backward_tensors(&inputs, &gradients, &config)?;
        let scores_shape = validated.scores_shape;
        let expected_stats_shape = validated.expected_stats_shape;
        let k_is_transposed = validated.k_is_transposed;
        let v_is_transposed = validated.v_is_transposed;
        let modifiers = if let Some(dropout) = config.dropout() {
            let rng_config = self.attention_dropout_random_number_generator_config(
                dropout.probability(),
                dropout.seed_source(),
                config.dropout_offset(),
            )?;
            let rng_dump = self.tensor(TensorSpec::new(DataType::F32, scores_shape.clone()));
            self.random_number_generator(rng_dump, rng_config)?;
            if let Some(requested_rng_dump) = requested_rng_dump {
                self.bind_sdpa_inferred_tensor(
                    rng_dump,
                    requested_rng_dump,
                    "sdpa backward rng dump",
                )?;
            } else if config.has_random_number_generator_dump() {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa backward rng dump".into(),
                });
            }
            let dropout_scale = self.tensor(sdpa_scalar_tensor(
                DataType::F32,
                1.0 / (1.0 - dropout.probability()),
            )?);
            config.modifiers().with_dropout(rng_dump, dropout_scale)
        } else {
            *config.modifiers()
        };
        let mut matmul_qk_config = MatmulConfig::new(config.softmax().compute_type());
        if let Some(m_override) = modifiers.sequence_length_query {
            matmul_qk_config = matmul_qk_config.with_m_override(m_override);
        }
        if let Some(k_override) = modifiers.sequence_length_key_value {
            matmul_qk_config = matmul_qk_config.with_k_override(k_override);
        }
        let mut matmul_kv_config = MatmulConfig::new(config.softmax().compute_type());
        if let Some(m_override) = modifiers.sequence_length_key_value {
            matmul_kv_config = matmul_kv_config.with_m_override(m_override);
        }
        if let Some(k_override) = modifiers.sequence_length_query {
            matmul_kv_config = matmul_kv_config.with_k_override(k_override);
        }
        let intermediate_type = self.effective_intermediate_data_type(DataType::F32);

        let scores =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        let k_t = if k_is_transposed {
            k
        } else {
            self.transpose_last_two(k)?
        };
        self.matmul_with_config(q, k_t, scores, matmul_qk_config.clone())?;

        let mut scaled_scores = self.pointwise_binary_virtual_infer_as(
            scores,
            scale,
            PointwiseMode::Mul,
            config.softmax().compute_type(),
            intermediate_type,
        )?;
        if let Some(score_subgraph) = config.score_subgraph() {
            scaled_scores = self.import_score_subgraph(
                score_subgraph,
                scaled_scores,
                "sdpa backward score subgraph",
            )?;
        }
        if let Some(bias) = modifiers.bias {
            scaled_scores = self.pointwise_binary_virtual_infer_as(
                scaled_scores,
                bias,
                PointwiseMode::Add,
                config.softmax().compute_type(),
                intermediate_type,
            )?;
        }
        if let Some(sink_token) = modifiers.sink_token {
            scaled_scores = self.pointwise_binary_virtual_infer_as(
                scaled_scores,
                sink_token,
                PointwiseMode::Add,
                config.softmax().compute_type(),
                intermediate_type,
            )?;
        }
        if let Some(alibi_slopes) = modifiers.alibi_slopes {
            let alibi_bias = self.alibi_bias(scaled_scores, alibi_slopes)?;
            scaled_scores = self.pointwise_binary_virtual_infer_as(
                scaled_scores,
                alibi_bias,
                PointwiseMode::Add,
                config.softmax().compute_type(),
                intermediate_type,
            )?;
        }
        if let Some(additive_mask) = modifiers.additive_mask {
            scaled_scores = self.pointwise_binary_virtual_infer_as(
                scaled_scores,
                additive_mask,
                PointwiseMode::Add,
                config.softmax().compute_type(),
                intermediate_type,
            )?;
        }
        if modifiers.causal_mask {
            let masked = self.tensor(
                TensorSpec::new(
                    intermediate_type,
                    self.tensor_config(scaled_scores)?.shape.clone(),
                )
                .virtual_tensor(),
            );
            let negative_inf = self.tensor(sdpa_mask_scalar(intermediate_type, f32::NEG_INFINITY)?);
            self.diagonal_band_mask(
                scaled_scores,
                negative_inf,
                masked,
                DiagonalBandMaskConfig::new(PointwiseMode::CmpGe),
            )?;
            scaled_scores = masked;
        }
        if let Some((left_window, right_window)) = modifiers.sliding_window {
            let sliding_mask =
                self.diagonal_band_mask_composite(scaled_scores, left_window, Some(right_window))?;
            scaled_scores = self.pointwise_binary_virtual_infer_as(
                scaled_scores,
                sliding_mask,
                PointwiseMode::Add,
                config.softmax().compute_type(),
                intermediate_type,
            )?;
        }
        if let (Some(sequence_length_query), Some(sequence_length_key_value)) = (
            modifiers.sequence_length_query,
            modifiers.sequence_length_key_value,
        ) {
            let masked_scores = if modifiers.causal_bottom_right {
                Some(self.causal_bottom_right_mask(
                    scaled_scores,
                    sequence_length_query,
                    sequence_length_key_value,
                )?)
            } else if modifiers.padding_mask {
                Some(self.padding_mask_select(
                    scaled_scores,
                    sequence_length_query,
                    sequence_length_key_value,
                )?)
            } else {
                None
            };
            if let Some(masked_scores) = masked_scores {
                scaled_scores = masked_scores;
            }
        }

        let shifted_scores = self.pointwise_binary_virtual_infer_as(
            scaled_scores,
            stats,
            PointwiseMode::Sub,
            config.softmax().compute_type(),
            intermediate_type,
        )?;
        let mut probs =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Exp,
            input: shifted_scores,
            output: probs,
            compute_type: config.softmax().compute_type(),
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: None,
        });
        if let (Some(dropout_mask), Some(dropout_scale)) =
            (modifiers.dropout_mask, modifiers.dropout_scale)
        {
            let masked_probs = self.pointwise_binary_virtual_infer_as(
                probs,
                dropout_mask,
                PointwiseMode::Mul,
                config.softmax().compute_type(),
                intermediate_type,
            )?;
            probs = self.pointwise_binary_virtual_infer_as(
                masked_probs,
                dropout_scale,
                PointwiseMode::Mul,
                config.softmax().compute_type(),
                intermediate_type,
            )?;
        }

        let probs_t = self.transpose_last_two_reshape(probs)?;
        self.matmul_with_config(probs_t, d_o, d_v, matmul_kv_config.clone())?;

        let v_t = if v_is_transposed {
            v
        } else {
            self.transpose_last_two(v)?
        };
        let mut d_p =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        self.matmul_with_config(d_o, v_t, d_p, matmul_qk_config.clone())?;
        if let Some(dropout_mask) = modifiers.dropout_mask {
            d_p = self.pointwise_binary_virtual_infer_as(
                d_p,
                dropout_mask,
                PointwiseMode::Mul,
                config.softmax().compute_type(),
                intermediate_type,
            )?;
        }

        let d_o_mul_o = self.pointwise_binary_virtual_infer_as(
            d_o,
            o,
            PointwiseMode::Mul,
            config.softmax().compute_type(),
            intermediate_type,
        )?;
        let softmax_sum = self.tensor(
            TensorSpec::new(intermediate_type, expected_stats_shape.clone()).virtual_tensor(),
        );
        self.reduction(ReductionOperation::Reduce {
            op: ReduceTensorOperator::Add,
            input: d_o_mul_o,
            output: softmax_sum,
            compute_type: config.softmax().compute_type(),
            is_deterministic: false,
        });

        if let Some(d_sink_token) = d_sink_token {
            let sink_token = if let Some(sink_token) = modifiers.sink_token {
                sink_token
            } else {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa backward sink token".into(),
                });
            };

            let sink_minus_stats = self.pointwise_binary_virtual_infer_as(
                sink_token,
                stats,
                PointwiseMode::Sub,
                config.softmax().compute_type(),
                intermediate_type,
            )?;
            let exp_sink = self.tensor(
                TensorSpec::new(intermediate_type, reduce_last_axis(&scores_shape)?)
                    .virtual_tensor(),
            );
            self.pointwise(PointwiseOperation::Unary {
                mode: PointwiseMode::Exp,
                input: sink_minus_stats,
                output: exp_sink,
                compute_type: config.softmax().compute_type(),
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                axis: None,
            });
            let per_token_grad = self.pointwise_binary_virtual_infer_as(
                exp_sink,
                softmax_sum,
                PointwiseMode::Mul,
                config.softmax().compute_type(),
                intermediate_type,
            )?;
            self.reduction(ReductionOperation::Reduce {
                op: ReduceTensorOperator::Add,
                input: per_token_grad,
                output: d_sink_token,
                compute_type: config.softmax().compute_type(),
                is_deterministic: false,
            });
        } else if config.has_sink_token_gradient() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward d_sink_token".into(),
            });
        }

        let mut d_s = self.pointwise_binary_virtual_infer_as(
            d_p,
            softmax_sum,
            PointwiseMode::Sub,
            config.softmax().compute_type(),
            intermediate_type,
        )?;
        d_s = self.pointwise_binary_virtual_infer_as(
            d_s,
            probs,
            PointwiseMode::Mul,
            config.softmax().compute_type(),
            intermediate_type,
        )?;
        if let Some(dropout_scale) = modifiers.dropout_scale {
            d_s = self.pointwise_binary_virtual_infer_as(
                d_s,
                dropout_scale,
                PointwiseMode::Mul,
                config.softmax().compute_type(),
                intermediate_type,
            )?;
        }
        if let Some(d_bias) = d_bias {
            self.reshape(d_s, d_bias)?;
        } else if config.has_bias_gradient() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward d_bias".into(),
            });
        }
        if let Some(score_subgraph_bprop) = config.score_subgraph_bprop() {
            d_s = self.import_score_subgraph(
                score_subgraph_bprop,
                d_s,
                "sdpa backward score subgraph bprop",
            )?;
        }
        d_s = self.pointwise_binary_virtual_infer_as(
            d_s,
            scale,
            PointwiseMode::Mul,
            config.softmax().compute_type(),
            intermediate_type,
        )?;

        let d_s_t = self.transpose_last_two_reshape(d_s)?;
        self.matmul_with_config(d_s_t, q, d_k, matmul_kv_config)?;
        let k_for_d_q = if k_is_transposed {
            self.transpose_last_two_reshape(k)?
        } else {
            k
        };
        self.matmul_with_config(d_s, k_for_d_q, d_q, matmul_qk_config)?;

        Ok(())
    }

    pub fn sdpa_backward_infer(
        &mut self,
        inputs: SdpaBackwardInputs,
        config: AttentionBackwardConfig,
    ) -> Result<SdpaBackwardOutputs> {
        let outputs = self.sdpa_backward_infer_with_aux(inputs, config)?;
        Ok(outputs.into())
    }

    pub fn sdpa_backward_infer_with_aux(
        &mut self,
        inputs: SdpaBackwardInputs,
        config: AttentionBackwardConfig,
    ) -> Result<SdpaBackwardAuxOutputs> {
        let SdpaBackwardInputs {
            query: q,
            key: k,
            value: v,
            output: o,
            output_gradient: _,
            stats: _,
            scale,
        } = inputs;
        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();
        let o_tensor = self.tensor_config(o)?.clone();
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        let q_dims = q_tensor.shape.dimensions();
        let k_dims = k_tensor.shape.dimensions();
        let v_dims = v_tensor.shape.dimensions();
        let k_is_transposed = q_dims[3] == k_dims[2];
        let v_is_transposed = o_tensor.shape.dimensions()[3] == v_dims[2];

        let d_q = self.tensor(TensorSpec::new(q_tensor.data_type, q_tensor.shape.clone()));
        let d_k_shape = if k_is_transposed {
            Shape::contiguous([k_dims[0], k_dims[1], k_dims[3], q_dims[3]])?
        } else {
            k_tensor.shape.clone()
        };
        let d_v_shape = if v_is_transposed {
            Shape::contiguous([v_dims[0], v_dims[1], v_dims[3], v_dims[2]])?
        } else {
            v_tensor.shape.clone()
        };
        let d_k = self.tensor(TensorSpec::new(k_tensor.data_type, d_k_shape));
        let d_v = self.tensor(TensorSpec::new(v_tensor.data_type, d_v_shape));
        let d_bias = if config.has_bias_gradient() {
            Some(self.tensor(TensorSpec::new(DataType::F32, scores_shape.clone())))
        } else {
            None
        };
        let rng_dump = if config.has_random_number_generator_dump() {
            Some(self.tensor(TensorSpec::new(DataType::F32, scores_shape.clone())))
        } else {
            None
        };
        let d_sink_token = if config.has_sink_token_gradient() {
            Some(self.tensor(TensorSpec::new(
                config.softmax().compute_type(),
                Shape::contiguous([1, q_tensor.shape.dimensions()[1], 1, 1])?,
            )))
        } else {
            None
        };
        let scale = self.sdpa_effective_scale(scale, config.attention_scale())?;

        self.sdpa_backward(
            SdpaBackwardInputs { scale, ..inputs },
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: d_bias,
                rng_dump,
                sink_token_gradient: d_sink_token,
            },
            config,
        )?;

        Ok(SdpaBackwardAuxOutputs::new(
            d_q,
            d_k,
            d_v,
            d_bias,
            rng_dump,
            d_sink_token,
        ))
    }

    pub fn sdpa_mxfp8_backward_infer(
        &mut self,
        inputs: SdpaMxfp8BackwardInputs,
        config: AttentionBackwardConfig,
    ) -> Result<SdpaMxfp8BackwardOutputs> {
        let output_io_type = self.effective_io_data_type(DataType::BF16);
        self.sdpa_mxfp8_backward_infer_with_output_types(
            inputs,
            output_io_type,
            output_io_type,
            output_io_type,
            config,
        )
    }

    pub fn sdpa_mxfp8_backward_infer_with_output_types(
        &mut self,
        inputs: SdpaMxfp8BackwardInputs,
        d_q_output_type: DataType,
        d_k_output_type: DataType,
        d_v_output_type: DataType,
        config: AttentionBackwardConfig,
    ) -> Result<SdpaMxfp8BackwardOutputs> {
        let SdpaMxfp8BackwardInputs {
            query: q,
            query_transposed: q_t,
            key: k,
            key_transposed: k_t,
            value: v,
            output: o,
            output_gradient: d_o,
            quantized_output_gradient: d_o_quantized,
            output_gradient_transposed: d_o_t,
            stats,
            scale_query: scale_q,
            scale_query_transposed: scale_q_t,
            scale_key: scale_k,
            scale_key_transposed: scale_k_t,
            scale_value: scale_v,
            scale_output_gradient: scale_d_o,
            scale_output_gradient_transposed: scale_d_o_t,
            attention_scale,
        } = inputs;
        let intermediate_type = self.effective_intermediate_data_type(DataType::F32);
        let compute_type = self.effective_compute_data_type(DataType::F32);
        self.validate_attention_backward_config(q, k, v, &config)?;
        let is_ragged = self.tensor_config(q)?.ragged_offset.is_some()
            || self.tensor_config(k)?.ragged_offset.is_some()
            || self.tensor_config(v)?.ragged_offset.is_some()
            || self.tensor_config(o)?.ragged_offset.is_some()
            || self.tensor_config(d_o)?.ragged_offset.is_some();
        self.validate_mxfp8_backward_support_surface_for_version(
            version()?.raw(),
            d_q_output_type,
            is_ragged,
            &config,
        )?;
        self.validate_mxfp8_backward_support_surface_for_version(
            version()?.raw(),
            d_k_output_type,
            is_ragged,
            &config,
        )?;
        self.validate_mxfp8_backward_support_surface_for_version(
            version()?.raw(),
            d_v_output_type,
            is_ragged,
            &config,
        )?;

        self.validate_sdpa_tensor_layout(q, "sdpa mxfp8 backward q layout")?;
        self.validate_sdpa_tensor_layout(q_t, "sdpa mxfp8 backward q_t layout")?;
        self.validate_sdpa_tensor_layout(k, "sdpa mxfp8 backward k layout")?;
        self.validate_sdpa_tensor_layout(k_t, "sdpa mxfp8 backward k_t layout")?;
        self.validate_sdpa_tensor_layout(v, "sdpa mxfp8 backward v layout")?;
        self.validate_sdpa_tensor_layout(o, "sdpa mxfp8 backward o layout")?;
        self.validate_sdpa_tensor_layout(d_o, "sdpa mxfp8 backward d_o layout")?;
        self.validate_sdpa_tensor_layout(
            d_o_quantized,
            "sdpa mxfp8 backward d_o_quantized layout",
        )?;
        self.validate_sdpa_tensor_layout(d_o_t, "sdpa mxfp8 backward d_o_t layout")?;

        for (scale, name) in [
            (scale_q, "sdpa mxfp8 backward scale_q"),
            (scale_q_t, "sdpa mxfp8 backward scale_q_t"),
            (scale_k, "sdpa mxfp8 backward scale_k"),
            (scale_k_t, "sdpa mxfp8 backward scale_k_t"),
            (scale_v, "sdpa mxfp8 backward scale_v"),
            (scale_d_o, "sdpa mxfp8 backward scale_d_o"),
            (scale_d_o_t, "sdpa mxfp8 backward scale_d_o_t"),
        ] {
            self.validate_mxfp8_scale_tensor(scale, name)?;
        }
        self.validate_mxfp8_backward_scale_dimensions(
            q,
            k,
            v,
            d_o,
            scale_q,
            scale_q_t,
            scale_k,
            scale_k_t,
            scale_v,
            scale_d_o,
            scale_d_o_t,
        )?;

        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();
        let o_tensor = self.tensor_config(o)?.clone();
        let d_o_tensor = self.tensor_config(d_o)?.clone();
        let d_o_quantized_tensor = self.tensor_config(d_o_quantized)?.clone();
        let d_o_t_tensor = self.tensor_config(d_o_t)?.clone();
        let stats_tensor = self.tensor_config(stats)?.clone();
        if config.modifiers().sliding_window.is_some()
            && !config.modifiers().padding_mask
            && q_tensor.shape.dimensions()[2] > k_tensor.shape.dimensions()[2]
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward sliding window sequence lengths".into(),
            });
        }

        if self.tensor_config(q_t)?.shape.dimensions() != q_tensor.shape.dimensions() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward q_t shape".into(),
            });
        }
        if self.tensor_config(k_t)?.shape.dimensions() != k_tensor.shape.dimensions() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward k_t shape".into(),
            });
        }
        if d_o_quantized_tensor.shape.dimensions() != d_o_tensor.shape.dimensions() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward d_o_quantized shape".into(),
            });
        }
        if d_o_t_tensor.shape.dimensions() != d_o_tensor.shape.dimensions() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward d_o_t shape".into(),
            });
        }
        if o_tensor.shape.dimensions() != d_o_tensor.shape.dimensions() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward o shape".into(),
            });
        }
        if stats_tensor.shape.dimensions() != reduce_last_axis(&q_tensor.shape)?.dimensions() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward stats shape".into(),
            });
        }
        if stats_tensor.data_type != compute_type {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward stats data type".into(),
            });
        }
        self.validate_quantized_backward_head_dimensions(
            q_tensor.shape.dimensions()[3],
            v_tensor.shape.dimensions()[3],
            "sdpa mxfp8 backward head dimension",
        )?;

        let attention_scale = self.mxfp8_attn_scale_tensor(
            attention_scale,
            "sdpa mxfp8 backward attention_scale",
            compute_type,
        )?;
        let attention_scale =
            self.sdpa_effective_scale(attention_scale, config.attention_scale())?;

        let q_fp = self.mxfp8_dequantize_input_infer(q, scale_q, intermediate_type)?;
        let k_fp = self.mxfp8_dequantize_transposed_input_infer(k, scale_k, intermediate_type)?;
        let v_t_fp = self.mxfp8_dequantize_transposed_input_infer(v, scale_v, intermediate_type)?;
        let d_o_fp =
            self.mxfp8_dequantize_input_infer(d_o_quantized, scale_d_o, intermediate_type)?;
        let d_o_t_fp = self.mxfp8_dequantize_sequence_scaled_input_infer(
            d_o_t,
            scale_d_o_t,
            intermediate_type,
        )?;
        let k_t_fp =
            self.mxfp8_dequantize_sequence_scaled_input_infer(k_t, scale_k_t, intermediate_type)?;
        let q_t_fp =
            self.mxfp8_dequantize_sequence_scaled_input_infer(q_t, scale_q_t, intermediate_type)?;
        self.validate_attention_backward_support_surface_for_version(
            version()?.raw(),
            q_fp,
            k_fp,
            v,
            o,
            stats,
            d_o,
            q,
            k,
            v,
            &config,
        )?;

        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        let scores =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        self.matmul(q_fp, k_fp, scores, compute_type)?;

        let mut scaled_scores =
            self.pointwise_binary_infer(scores, attention_scale, PointwiseMode::Mul, compute_type)?;

        let modifiers = config.modifiers();
        if let Some(score_subgraph) = config.score_subgraph() {
            scaled_scores = self.import_score_subgraph(
                score_subgraph,
                scaled_scores,
                "sdpa mxfp8 backward score subgraph",
            )?;
        }
        if let Some(bias) = modifiers.bias {
            scaled_scores = self.pointwise_binary_infer_as(
                scaled_scores,
                bias,
                PointwiseMode::Add,
                compute_type,
                intermediate_type,
            )?;
        }
        if let Some(sink_token) = modifiers.sink_token {
            scaled_scores = self.pointwise_binary_infer_as(
                scaled_scores,
                sink_token,
                PointwiseMode::Add,
                compute_type,
                intermediate_type,
            )?;
        }
        if let Some(alibi_slopes) = modifiers.alibi_slopes {
            let alibi_bias = self.alibi_bias(scaled_scores, alibi_slopes)?;
            scaled_scores = self.pointwise_binary_infer_as(
                scaled_scores,
                alibi_bias,
                PointwiseMode::Add,
                compute_type,
                intermediate_type,
            )?;
        }
        if let Some(additive_mask) = modifiers.additive_mask {
            scaled_scores = self.pointwise_binary_infer_as(
                scaled_scores,
                additive_mask,
                PointwiseMode::Add,
                compute_type,
                intermediate_type,
            )?;
        }
        if modifiers.causal_mask {
            let causal_mask = self.sdpa_causal_mask_infer(q_fp, k_fp)?;
            scaled_scores = self.pointwise_binary_infer_as(
                scaled_scores,
                causal_mask,
                PointwiseMode::Add,
                compute_type,
                intermediate_type,
            )?;
        }
        if let Some((left_window, right_window)) = modifiers.sliding_window {
            let sliding_mask =
                self.sdpa_sliding_window_mask_infer(q_fp, k_fp, left_window, right_window)?;
            scaled_scores = self.pointwise_binary_infer_as(
                scaled_scores,
                sliding_mask,
                PointwiseMode::Add,
                compute_type,
                intermediate_type,
            )?;
        }
        if let (Some(sequence_length_query), Some(sequence_length_key_value)) = (
            modifiers.sequence_length_query,
            modifiers.sequence_length_key_value,
        ) {
            let mask = if modifiers.causal_bottom_right {
                Some(self.causal_bottom_right_mask(
                    scaled_scores,
                    sequence_length_query,
                    sequence_length_key_value,
                )?)
            } else if modifiers.padding_mask {
                Some(self.padding_mask(
                    scaled_scores,
                    sequence_length_query,
                    sequence_length_key_value,
                )?)
            } else {
                None
            };
            if let Some(mask) = mask {
                scaled_scores = self.pointwise_binary_infer_as(
                    scaled_scores,
                    mask,
                    PointwiseMode::Add,
                    compute_type,
                    intermediate_type,
                )?;
            }
        }

        let shifted_scores =
            self.pointwise_binary_infer(scaled_scores, stats, PointwiseMode::Sub, compute_type)?;
        let mut probs =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Exp,
            input: shifted_scores,
            output: probs,
            compute_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: None,
        });
        if let (Some(dropout_mask), Some(dropout_scale)) =
            (modifiers.dropout_mask, modifiers.dropout_scale)
        {
            let masked_probs =
                self.pointwise_binary_infer(probs, dropout_mask, PointwiseMode::Mul, compute_type)?;
            probs = self.pointwise_binary_infer(
                masked_probs,
                dropout_scale,
                PointwiseMode::Mul,
                compute_type,
            )?;
        }

        let probs_t = self.transpose_last_two(probs)?;
        let d_v_fp = self.tensor(TensorSpec::new(intermediate_type, v_tensor.shape.clone()));
        self.matmul(probs_t, d_o_t_fp, d_v_fp, compute_type)?;

        let mut d_p =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        self.matmul(d_o_fp, v_t_fp, d_p, compute_type)?;
        if let Some(dropout_mask) = modifiers.dropout_mask {
            d_p =
                self.pointwise_binary_infer(d_p, dropout_mask, PointwiseMode::Mul, compute_type)?;
        }

        let d_o_mul_o = self.pointwise_binary_infer(d_o, o, PointwiseMode::Mul, compute_type)?;
        let softmax_sum = self.reduction_infer(
            d_o_mul_o,
            ReductionConfig::new(ReduceTensorOperator::Add, compute_type),
        )?;

        let mut d_s =
            self.pointwise_binary_infer(d_p, softmax_sum, PointwiseMode::Sub, compute_type)?;
        d_s = self.pointwise_binary_infer(d_s, probs, PointwiseMode::Mul, compute_type)?;
        if let Some(dropout_scale) = modifiers.dropout_scale {
            d_s =
                self.pointwise_binary_infer(d_s, dropout_scale, PointwiseMode::Mul, compute_type)?;
        }
        if let Some(score_subgraph_bprop) = config.score_subgraph_bprop() {
            d_s = self.import_score_subgraph(
                score_subgraph_bprop,
                d_s,
                "sdpa mxfp8 backward score subgraph bprop",
            )?;
        }
        d_s =
            self.pointwise_binary_infer(d_s, attention_scale, PointwiseMode::Mul, compute_type)?;

        let d_q_fp = self.tensor(TensorSpec::new(intermediate_type, q_tensor.shape.clone()));
        self.matmul(d_s, k_t_fp, d_q_fp, compute_type)?;

        let d_s_t = self.transpose_last_two(d_s)?;
        let d_k_fp = self.tensor(TensorSpec::new(intermediate_type, k_tensor.shape.clone()));
        self.matmul(d_s_t, q_t_fp, d_k_fp, compute_type)?;

        let d_q = self.convert_mxfp8_backward_output(d_q_fp, d_q_output_type, compute_type)?;
        let d_k = self.convert_mxfp8_backward_output(d_k_fp, d_k_output_type, compute_type)?;
        let d_v = self.convert_mxfp8_backward_output(d_v_fp, d_v_output_type, compute_type)?;

        let absolute_max_d_q = self.absolute_max_infer(d_q, DataType::F32)?;
        let absolute_max_d_k = self.absolute_max_infer(d_k, DataType::F32)?;
        let absolute_max_d_v = self.absolute_max_infer(d_v, DataType::F32)?;

        Ok(SdpaMxfp8BackwardOutputs::new(
            d_q,
            d_k,
            d_v,
            absolute_max_d_q,
            absolute_max_d_k,
            absolute_max_d_v,
        ))
    }

    pub fn sdpa_mxfp8_backward(
        &mut self,
        inputs: SdpaMxfp8BackwardInputs,
        gradients: SdpaMxfp8BackwardGradientTensors,
        config: AttentionBackwardConfig,
    ) -> Result<()> {
        let SdpaMxfp8BackwardGradientTensors {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            absolute_max_query_gradient: absolute_max_d_q,
            absolute_max_key_gradient: absolute_max_d_k,
            absolute_max_value_gradient: absolute_max_d_v,
        } = gradients;
        self.validate_quantized_backward_output_ragged_support(
            d_q,
            d_k,
            d_v,
            "sdpa mxfp8 backward ragged outputs",
        )?;
        let result = self.sdpa_mxfp8_backward_infer_with_output_types(
            inputs,
            self.tensor_config(d_q)?.data_type,
            self.tensor_config(d_k)?.data_type,
            self.tensor_config(d_v)?.data_type,
            config,
        );
        let result = result?;

        self.bind_sdpa_inferred_tensor(result.query_gradient, d_q, "sdpa mxfp8 backward d_q")?;
        self.bind_sdpa_inferred_tensor(result.key_gradient, d_k, "sdpa mxfp8 backward d_k")?;
        self.bind_sdpa_inferred_tensor(result.value_gradient, d_v, "sdpa mxfp8 backward d_v")?;
        self.bind_sdpa_inferred_tensor(
            result.absolute_max_query_gradient,
            absolute_max_d_q,
            "sdpa mxfp8 backward absolute_max_d_q",
        )?;
        self.bind_sdpa_inferred_tensor(
            result.absolute_max_key_gradient,
            absolute_max_d_k,
            "sdpa mxfp8 backward absolute_max_d_k",
        )?;
        self.bind_sdpa_inferred_tensor(
            result.absolute_max_value_gradient,
            absolute_max_d_v,
            "sdpa mxfp8 backward absolute_max_d_v",
        )?;
        Ok(())
    }

    pub fn sdpa_fp8_backward_infer(
        &mut self,
        inputs: SdpaFp8BackwardInputs,
        config: AttentionBackwardConfig,
    ) -> Result<SdpaFp8BackwardOutputs> {
        let outputs = self.sdpa_fp8_backward_infer_with_aux(inputs, config)?;
        Ok(outputs.into())
    }

    pub fn sdpa_fp8_backward_infer_with_aux(
        &mut self,
        inputs: SdpaFp8BackwardInputs,
        config: AttentionBackwardConfig,
    ) -> Result<SdpaFp8BackwardAuxOutputs> {
        let q = inputs.query;
        let q_data_type = self.tensor_config(q)?.data_type;
        let output_io_type = self.io_data_type().unwrap_or(q_data_type);
        self.sdpa_fp8_backward_infer_with_aux_and_output_types(
            inputs,
            output_io_type,
            output_io_type,
            output_io_type,
            config,
        )
    }

    pub fn sdpa_fp8_backward_infer_with_aux_and_output_types(
        &mut self,
        inputs: SdpaFp8BackwardInputs,
        d_q_output_type: DataType,
        d_k_output_type: DataType,
        d_v_output_type: DataType,
        config: AttentionBackwardConfig,
    ) -> Result<SdpaFp8BackwardAuxOutputs> {
        let SdpaFp8BackwardInputs {
            query: q,
            key: k,
            value: v,
            output: o,
            output_gradient: d_o,
            stats,
            attention_scale,
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
        } = inputs;
        let q_data_type = self.tensor_config(q)?.data_type;
        self.validate_fp8_scale_tensor(attention_scale, "sdpa fp8 backward attention_scale")?;
        self.validate_fp8_scale_tensor(descale_q, "sdpa fp8 backward descale_q")?;
        self.validate_fp8_scale_tensor(descale_k, "sdpa fp8 backward descale_k")?;
        self.validate_fp8_scale_tensor(descale_v, "sdpa fp8 backward descale_v")?;
        self.validate_fp8_scale_tensor(descale_o, "sdpa fp8 backward descale_o")?;
        self.validate_fp8_scale_tensor(descale_d_o, "sdpa fp8 backward descale_d_o")?;
        self.validate_fp8_scale_tensor(descale_s, "sdpa fp8 backward descale_s")?;
        self.validate_fp8_scale_tensor(descale_d_p, "sdpa fp8 backward descale_d_p")?;
        self.validate_fp8_scale_tensor(scale_s, "sdpa fp8 backward scale_s")?;
        self.validate_fp8_scale_tensor(scale_d_q, "sdpa fp8 backward scale_d_q")?;
        self.validate_fp8_scale_tensor(scale_d_k, "sdpa fp8 backward scale_d_k")?;
        self.validate_fp8_scale_tensor(scale_d_v, "sdpa fp8 backward scale_d_v")?;
        self.validate_fp8_scale_tensor(scale_d_p, "sdpa fp8 backward scale_d_p")?;
        let is_ragged = self.tensor_config(q)?.ragged_offset.is_some()
            || self.tensor_config(k)?.ragged_offset.is_some()
            || self.tensor_config(v)?.ragged_offset.is_some()
            || self.tensor_config(o)?.ragged_offset.is_some()
            || self.tensor_config(d_o)?.ragged_offset.is_some();
        if config.has_bias_gradient() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa fp8 backward d_bias".into(),
            });
        }
        self.validate_fp8_backward_support_surface_for_version(
            version()?.raw(),
            d_q_output_type,
            self.tensor_config(q)?.shape.dimensions()[3],
            self.tensor_config(v)?.shape.dimensions()[3],
            is_ragged,
            &config,
        )?;
        self.validate_fp8_backward_support_surface_for_version(
            version()?.raw(),
            d_k_output_type,
            self.tensor_config(q)?.shape.dimensions()[3],
            self.tensor_config(v)?.shape.dimensions()[3],
            is_ragged,
            &config,
        )?;
        self.validate_fp8_backward_support_surface_for_version(
            version()?.raw(),
            d_v_output_type,
            self.tensor_config(q)?.shape.dimensions()[3],
            self.tensor_config(v)?.shape.dimensions()[3],
            is_ragged,
            &config,
        )?;
        self.validate_quantized_backward_head_dimensions(
            self.tensor_config(q)?.shape.dimensions()[3],
            self.tensor_config(v)?.shape.dimensions()[3],
            "sdpa fp8 backward head dimension",
        )?;
        if config.modifiers().sliding_window.is_some()
            && !config.modifiers().padding_mask
            && self.tensor_config(q)?.shape.dimensions()[2]
                > self.tensor_config(k)?.shape.dimensions()[2]
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa fp8 backward sliding window sequence lengths".into(),
            });
        }
        let attention_scale =
            self.sdpa_effective_scale(attention_scale, config.attention_scale())?;

        let q_fp = self.dequantize_tensor_infer(q, descale_q)?;
        let k_fp = self.dequantize_tensor_infer(k, descale_k)?;
        let v_fp = self.dequantize_tensor_infer(v, descale_v)?;
        let o_fp = self.dequantize_tensor_infer(o, descale_o)?;
        let d_o_fp = self.dequantize_tensor_infer(d_o, descale_d_o)?;

        let fp_outputs = self.sdpa_backward_infer_with_aux(
            SdpaBackwardInputs::new(q_fp, k_fp, v_fp, o_fp, d_o_fp, stats, attention_scale),
            config.clone(),
        )?;
        let d_q_fp = fp_outputs.query_gradient;
        let d_k_fp = fp_outputs.key_gradient;
        let d_v_fp = fp_outputs.value_gradient;
        let d_sink_token = fp_outputs.sink_token_gradient;

        let absolute_max_d_q = self.absolute_max_infer(d_q_fp, DataType::F32)?;
        let absolute_max_d_k = self.absolute_max_infer(d_k_fp, DataType::F32)?;
        let absolute_max_d_v = self.absolute_max_infer(d_v_fp, DataType::F32)?;

        let d_q = self.convert_fp8_backward_output(d_q_fp, scale_d_q, d_q_output_type)?;
        let d_k = self.convert_fp8_backward_output(d_k_fp, scale_d_k, d_k_output_type)?;
        let d_v = self.convert_fp8_backward_output(d_v_fp, scale_d_v, d_v_output_type)?;

        let k_t_fp = self.transpose_last_two(k_fp)?;
        let scores = self.tensor(
            TensorSpec::new(
                DataType::F32,
                infer_sdpa_scores_shape(self.shape(q_fp)?, self.shape(k_fp)?)?,
            )
            .virtual_tensor(),
        );
        self.matmul(q_fp, k_t_fp, scores, DataType::F32)?;
        let mut scaled_scores = self.pointwise_binary_infer_as(
            scores,
            attention_scale,
            PointwiseMode::Mul,
            DataType::F32,
            DataType::F32,
        )?;

        let modifiers = config.modifiers();
        if let Some(score_subgraph) = config.score_subgraph() {
            scaled_scores = self.import_score_subgraph(
                score_subgraph,
                scaled_scores,
                "sdpa fp8 backward score subgraph",
            )?;
        }
        if let Some(bias) = modifiers.bias {
            scaled_scores = self.pointwise_binary_infer_as(
                scaled_scores,
                bias,
                PointwiseMode::Add,
                DataType::F32,
                DataType::F32,
            )?;
        }
        if let Some(sink_token) = modifiers.sink_token {
            scaled_scores = self.pointwise_binary_infer_as(
                scaled_scores,
                sink_token,
                PointwiseMode::Add,
                DataType::F32,
                DataType::F32,
            )?;
        }
        if let Some(alibi_slopes) = modifiers.alibi_slopes {
            let alibi_bias = self.alibi_bias(scaled_scores, alibi_slopes)?;
            scaled_scores = self.pointwise_binary_infer_as(
                scaled_scores,
                alibi_bias,
                PointwiseMode::Add,
                DataType::F32,
                DataType::F32,
            )?;
        }
        if let Some(additive_mask) = modifiers.additive_mask {
            scaled_scores = self.pointwise_binary_infer_as(
                scaled_scores,
                additive_mask,
                PointwiseMode::Add,
                DataType::F32,
                DataType::F32,
            )?;
        }
        if modifiers.causal_mask {
            let causal_mask = self.sdpa_causal_mask_infer(q_fp, k_fp)?;
            scaled_scores = self.pointwise_binary_infer_as(
                scaled_scores,
                causal_mask,
                PointwiseMode::Add,
                DataType::F32,
                DataType::F32,
            )?;
        }
        if let Some((left_window, right_window)) = modifiers.sliding_window {
            let sliding_mask =
                self.sdpa_sliding_window_mask_infer(q_fp, k_fp, left_window, right_window)?;
            scaled_scores = self.pointwise_binary_infer_as(
                scaled_scores,
                sliding_mask,
                PointwiseMode::Add,
                DataType::F32,
                DataType::F32,
            )?;
        }
        if let (Some(sequence_length_query), Some(sequence_length_key_value)) = (
            modifiers.sequence_length_query,
            modifiers.sequence_length_key_value,
        ) {
            let mask = if modifiers.causal_bottom_right {
                Some(self.causal_bottom_right_mask(
                    scaled_scores,
                    sequence_length_query,
                    sequence_length_key_value,
                )?)
            } else if modifiers.padding_mask {
                Some(self.padding_mask(
                    scaled_scores,
                    sequence_length_query,
                    sequence_length_key_value,
                )?)
            } else {
                None
            };
            if let Some(mask) = mask {
                scaled_scores = self.pointwise_binary_infer_as(
                    scaled_scores,
                    mask,
                    PointwiseMode::Add,
                    DataType::F32,
                    DataType::F32,
                )?;
            }
        }

        let shifted_scores =
            self.pointwise_binary_infer(scaled_scores, stats, PointwiseMode::Sub, DataType::F32)?;
        let mut probs = self.tensor(
            TensorSpec::new(
                DataType::F32,
                self.tensor_config(shifted_scores)?.shape.clone(),
            )
            .virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Exp,
            input: shifted_scores,
            output: probs,
            compute_type: DataType::F32,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: None,
        });
        if let (Some(dropout_mask), Some(dropout_scale)) =
            (modifiers.dropout_mask, modifiers.dropout_scale)
        {
            let masked_probs = self.pointwise_binary_infer(
                probs,
                dropout_mask,
                PointwiseMode::Mul,
                DataType::F32,
            )?;
            probs = self.pointwise_binary_infer(
                masked_probs,
                dropout_scale,
                PointwiseMode::Mul,
                DataType::F32,
            )?;
        }

        let v_t_fp = self.transpose_last_two(v_fp)?;
        let mut d_p = self.tensor(
            TensorSpec::new(DataType::F32, self.tensor_config(probs)?.shape.clone())
                .virtual_tensor(),
        );
        self.matmul(d_o_fp, v_t_fp, d_p, DataType::F32)?;
        if let Some(dropout_mask) = modifiers.dropout_mask {
            d_p =
                self.pointwise_binary_infer(d_p, dropout_mask, PointwiseMode::Mul, DataType::F32)?;
        }

        let d_o_mul_o =
            self.pointwise_binary_infer(d_o_fp, o_fp, PointwiseMode::Mul, DataType::F32)?;
        let softmax_sum = self.reduction_infer(
            d_o_mul_o,
            ReductionConfig::new(ReduceTensorOperator::Add, DataType::F32),
        )?;

        d_p = self.pointwise_binary_infer(d_p, softmax_sum, PointwiseMode::Sub, DataType::F32)?;
        d_p = self.pointwise_binary_infer(d_p, probs, PointwiseMode::Mul, DataType::F32)?;
        if let Some(dropout_scale) = modifiers.dropout_scale {
            d_p =
                self.pointwise_binary_infer(d_p, dropout_scale, PointwiseMode::Mul, DataType::F32)?;
        }
        if let Some(score_subgraph_bprop) = config.score_subgraph_bprop() {
            d_p = self.import_score_subgraph(
                score_subgraph_bprop,
                d_p,
                "sdpa fp8 backward score subgraph bprop",
            )?;
        }
        d_p =
            self.pointwise_binary_infer(d_p, attention_scale, PointwiseMode::Mul, DataType::F32)?;

        let absolute_max_d_p = self.absolute_max_infer(d_p, DataType::F32)?;

        let _ = self.quantize_tensor_infer(d_p, scale_d_p, q_data_type)?;
        let _ = descale_s;
        let _ = descale_d_p;
        let _ = scale_s;

        Ok(SdpaFp8BackwardAuxOutputs::new(
            d_q,
            d_k,
            d_v,
            d_sink_token,
            absolute_max_d_q,
            absolute_max_d_k,
            absolute_max_d_v,
            absolute_max_d_p,
        ))
    }

    pub fn sdpa_fp8_backward_with_aux(
        &mut self,
        inputs: SdpaFp8BackwardInputs,
        gradients: SdpaFp8BackwardGradientTensors,
        config: AttentionBackwardConfig,
    ) -> Result<()> {
        let SdpaFp8BackwardGradientTensors {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            sink_token_gradient: d_sink_token,
            absolute_max_query_gradient: absolute_max_d_q,
            absolute_max_key_gradient: absolute_max_d_k,
            absolute_max_value_gradient: absolute_max_d_v,
            absolute_max_probability_gradient: absolute_max_d_p,
        } = gradients;
        self.validate_quantized_backward_output_ragged_support(
            d_q,
            d_k,
            d_v,
            "sdpa fp8 backward ragged outputs",
        )?;
        let computed = self.sdpa_fp8_backward_infer_with_aux_and_output_types(
            inputs,
            self.tensor_config(d_q)?.data_type,
            self.tensor_config(d_k)?.data_type,
            self.tensor_config(d_v)?.data_type,
            config,
        )?;

        self.bind_sdpa_inferred_tensor(computed.query_gradient, d_q, "sdpa fp8 backward d_q")?;
        self.bind_sdpa_inferred_tensor(computed.key_gradient, d_k, "sdpa fp8 backward d_k")?;
        self.bind_sdpa_inferred_tensor(computed.value_gradient, d_v, "sdpa fp8 backward d_v")?;
        self.bind_sdpa_inferred_optional_tensor(
            computed.sink_token_gradient,
            d_sink_token,
            "sdpa fp8 backward d_sink_token",
        )?;
        self.bind_sdpa_inferred_tensor(
            computed.absolute_max_query_gradient,
            absolute_max_d_q,
            "sdpa fp8 backward absolute_max_d_q",
        )?;
        self.bind_sdpa_inferred_tensor(
            computed.absolute_max_key_gradient,
            absolute_max_d_k,
            "sdpa fp8 backward absolute_max_d_k",
        )?;
        self.bind_sdpa_inferred_tensor(
            computed.absolute_max_value_gradient,
            absolute_max_d_v,
            "sdpa fp8 backward absolute_max_d_v",
        )?;
        self.bind_sdpa_inferred_tensor(
            computed.absolute_max_probability_gradient,
            absolute_max_d_p,
            "sdpa fp8 backward absolute_max_d_p",
        )?;
        Ok(())
    }

    pub fn sdpa_fp8_backward(
        &mut self,
        inputs: SdpaFp8BackwardInputs,
        gradients: SdpaFp8BackwardGradientTensors,
        config: AttentionBackwardConfig,
    ) -> Result<()> {
        self.sdpa_fp8_backward_with_aux(
            inputs,
            SdpaFp8BackwardGradientTensors {
                sink_token_gradient: None,
                ..gradients
            },
            config,
        )
    }
}
