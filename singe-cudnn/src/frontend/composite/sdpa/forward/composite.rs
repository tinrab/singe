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
            support::{
                SDPA_BACKWARD_RNG_DUMP, SDPA_BACKWARD_SCORE_SUBGRAPH,
                SDPA_BACKWARD_SCORE_SUBGRAPH_BPROP, SDPA_FP8_BACKWARD_ABSOLUTE_MAX_D_K,
                SDPA_FP8_BACKWARD_ABSOLUTE_MAX_D_P, SDPA_FP8_BACKWARD_ABSOLUTE_MAX_D_Q,
                SDPA_FP8_BACKWARD_ABSOLUTE_MAX_D_V, SDPA_FP8_BACKWARD_ATTENTION_SCALE,
                SDPA_FP8_BACKWARD_D_K, SDPA_FP8_BACKWARD_D_Q, SDPA_FP8_BACKWARD_D_SINK_TOKEN,
                SDPA_FP8_BACKWARD_D_V, SDPA_FP8_BACKWARD_DESCALE_D_O,
                SDPA_FP8_BACKWARD_DESCALE_D_P, SDPA_FP8_BACKWARD_DESCALE_K,
                SDPA_FP8_BACKWARD_DESCALE_O, SDPA_FP8_BACKWARD_DESCALE_Q,
                SDPA_FP8_BACKWARD_DESCALE_S, SDPA_FP8_BACKWARD_DESCALE_V,
                SDPA_FP8_BACKWARD_SCALE_D_K, SDPA_FP8_BACKWARD_SCALE_D_P,
                SDPA_FP8_BACKWARD_SCALE_D_Q, SDPA_FP8_BACKWARD_SCALE_D_V,
                SDPA_FP8_BACKWARD_SCALE_S, SDPA_FP8_BACKWARD_SCORE_SUBGRAPH,
                SDPA_FP8_BACKWARD_SCORE_SUBGRAPH_BPROP, SDPA_MXFP8_BACKWARD_ABSOLUTE_MAX_D_K,
                SDPA_MXFP8_BACKWARD_ABSOLUTE_MAX_D_Q, SDPA_MXFP8_BACKWARD_ABSOLUTE_MAX_D_V,
                SDPA_MXFP8_BACKWARD_ATTENTION_SCALE, SDPA_MXFP8_BACKWARD_D_K,
                SDPA_MXFP8_BACKWARD_D_O_LAYOUT, SDPA_MXFP8_BACKWARD_D_O_QUANTIZED_LAYOUT,
                SDPA_MXFP8_BACKWARD_D_O_QUANTIZED_SHAPE, SDPA_MXFP8_BACKWARD_D_O_T_LAYOUT,
                SDPA_MXFP8_BACKWARD_D_O_T_SHAPE, SDPA_MXFP8_BACKWARD_D_Q, SDPA_MXFP8_BACKWARD_D_V,
                SDPA_MXFP8_BACKWARD_K_LAYOUT, SDPA_MXFP8_BACKWARD_K_T_LAYOUT,
                SDPA_MXFP8_BACKWARD_K_T_SHAPE, SDPA_MXFP8_BACKWARD_O_LAYOUT,
                SDPA_MXFP8_BACKWARD_O_SHAPE, SDPA_MXFP8_BACKWARD_Q_LAYOUT,
                SDPA_MXFP8_BACKWARD_Q_T_LAYOUT, SDPA_MXFP8_BACKWARD_Q_T_SHAPE,
                SDPA_MXFP8_BACKWARD_SCALE_D_O, SDPA_MXFP8_BACKWARD_SCALE_D_O_T,
                SDPA_MXFP8_BACKWARD_SCALE_K, SDPA_MXFP8_BACKWARD_SCALE_K_T,
                SDPA_MXFP8_BACKWARD_SCALE_Q, SDPA_MXFP8_BACKWARD_SCALE_Q_T,
                SDPA_MXFP8_BACKWARD_SCALE_V, SDPA_MXFP8_BACKWARD_SCORE_SUBGRAPH,
                SDPA_MXFP8_BACKWARD_SCORE_SUBGRAPH_BPROP, SDPA_MXFP8_BACKWARD_STATS_DATA_TYPE,
                SDPA_MXFP8_BACKWARD_STATS_SHAPE, SDPA_MXFP8_BACKWARD_V_LAYOUT, SDPA_OUTPUT_SHAPE,
                SDPA_RANK, SDPA_SCORE_SUBGRAPH, SDPA_SINK_TOKEN_RANK, sdpa_mask_scalar,
                sdpa_scalar_tensor,
            },
        },
        composite::softmax::SoftmaxCompositeOutputs,
        graph::Graph,
        infer::*,
        operation::*,
        support,
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    reduction::ReduceTensorOperator,
    tensor::{Shape, TensorId, TensorSpec},
    version,
};

struct SdpaForwardInputTensors {
    q: TensorSpec,
    k_matmul: TensorId,
    k_matmul_tensor: TensorSpec,
    v: TensorSpec,
}

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
        let inputs =
            self.prepare_sdpa_forward_inputs(q, k, v, output, &modifiers, score_subgraph)?;

        let scores_shape = infer_sdpa_scores_shape(&inputs.q.shape, &inputs.k_matmul_tensor.shape)?;
        let scores =
            self.tensor(TensorSpec::new(inputs.q.data_type, scores_shape).virtual_tensor());
        self.matmul(q, inputs.k_matmul, scores, config.compute_type())?;

        let scores_with_modifiers = self.sdpa_forward_scores_with_modifiers(
            scores,
            scale,
            q,
            inputs.k_matmul,
            &modifiers,
            score_subgraph,
            config.compute_type(),
        )?;

        let softmax_outputs = self.softmax_composite_infer(scores_with_modifiers, config)?;
        let probs = self.sdpa_apply_dropout(
            softmax_outputs.output,
            modifiers.dropout_mask(),
            modifiers.dropout_scale(),
            config.compute_type(),
        )?;

        let inferred_output_shape = infer_sdpa_output_shape(self.shape(probs)?, &inputs.v.shape)?;
        self.validate_sdpa_tensor_dimensions_match(
            output,
            inferred_output_shape.dimensions(),
            SDPA_OUTPUT_SHAPE,
        )?;

        self.matmul(probs, v, output, config.compute_type())?;
        self.bind_sdpa_softmax_outputs(softmax_outputs, stats, logit_max, score_sum_exp)?;

        Ok(())
    }

    fn prepare_sdpa_forward_inputs(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        output: TensorId,
        modifiers: &SdpaScoreModifiers,
        score_subgraph: Option<&SdpaScoreSubgraph>,
    ) -> Result<SdpaForwardInputTensors> {
        self.validate_sdpa_score_modifiers(q, k, v, modifiers)?;
        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();
        self.validate_sdpa_forward_ranks(q, &q_tensor, k, &k_tensor, v, &v_tensor)?;

        let (k_matmul, k_matmul_tensor) =
            if q_tensor.shape.dimensions()[3] == k_tensor.shape.dimensions()[3] {
                let k_t = self.transpose_last_two(k)?;
                (k_t, self.tensor_config(k_t)?.clone())
            } else {
                (k, k_tensor)
            };

        self.validate_sdpa_forward_modifier_support(q, k, v, output, modifiers, score_subgraph)?;

        Ok(SdpaForwardInputTensors {
            q: q_tensor,
            k_matmul,
            k_matmul_tensor,
            v: v_tensor,
        })
    }

    fn validate_sdpa_forward_ranks(
        &self,
        q: TensorId,
        q_tensor: &TensorSpec,
        k: TensorId,
        k_tensor: &TensorSpec,
        v: TensorId,
        v_tensor: &TensorSpec,
    ) -> Result<()> {
        for (tensor_id, tensor) in [(q, q_tensor), (k, k_tensor), (v, v_tensor)] {
            let actual = tensor.shape.dimensions().len();
            if actual != 4 {
                return Err(Error::FrontendTensorRankMismatch {
                    tensor_id,
                    operation: SDPA_RANK.into(),
                    expected: "4".into(),
                    actual,
                });
            }
        }
        Ok(())
    }

    fn validate_sdpa_forward_modifier_support(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        output: TensorId,
        modifiers: &SdpaScoreModifiers,
        score_subgraph: Option<&SdpaScoreSubgraph>,
    ) -> Result<()> {
        let is_ragged = self.has_any_ragged_offset([q, k, v, output])?;
        let has_padding_mask = modifiers.padding_mask();
        support::require_sdpa_sequence_lengths_mask_or_subgraph(
            modifiers.sequence_lengths().is_some(),
            has_padding_mask,
            modifiers.causal_bottom_right(),
            score_subgraph.is_some(),
            "sdpa",
        )?;
        support::require_sdpa_ragged_mask_or_subgraph(
            is_ragged,
            has_padding_mask,
            modifiers.additive_mask().is_some(),
            score_subgraph.is_some(),
            "sdpa",
        )?;
        support::SdpaSupportContext::new(version()?.raw(), self.sm_version())
            .require_ragged_offsets_arch(is_ragged)
    }

    fn bind_sdpa_softmax_outputs(
        &mut self,
        inferred: SoftmaxCompositeOutputs,
        stats: Option<TensorId>,
        logit_max: Option<TensorId>,
        score_sum_exp: Option<TensorId>,
    ) -> Result<()> {
        if let Some(stats) = stats {
            self.reshape(inferred.stats, stats)?;
        }
        if let Some(logit_max) = logit_max {
            self.reshape(inferred.max, logit_max)?;
        }
        if let Some(score_sum_exp) = score_sum_exp {
            self.reshape(inferred.sum, score_sum_exp)?;
        }
        Ok(())
    }

    fn sdpa_forward_scores_with_modifiers(
        &mut self,
        scores: TensorId,
        scale: TensorId,
        q: TensorId,
        k: TensorId,
        modifiers: &SdpaScoreModifiers,
        score_subgraph: Option<&SdpaScoreSubgraph>,
        compute_type: DataType,
    ) -> Result<TensorId> {
        let mut scores =
            self.pointwise_binary_infer(scores, scale, PointwiseMode::Mul, compute_type)?;
        scores = self.sdpa_optional_add(scores, modifiers.bias(), compute_type)?;
        if let Some(score_subgraph) = score_subgraph {
            scores = self.import_score_subgraph(score_subgraph, scores, SDPA_SCORE_SUBGRAPH)?;
        }
        if let Some(sink_token) = modifiers.sink_token() {
            let sink_token_tensor = self.tensor_config(sink_token)?.clone();
            if sink_token_tensor.shape.dimensions().len() != 4 {
                return Err(Error::FrontendTensorRankMismatch {
                    tensor_id: sink_token,
                    operation: SDPA_SINK_TOKEN_RANK.into(),
                    expected: "4".into(),
                    actual: sink_token_tensor.shape.dimensions().len(),
                });
            }
            scores = self.sdpa_optional_add(scores, Some(sink_token), compute_type)?;
        }
        let alibi_bias = self.sdpa_optional_alibi_bias(scores, modifiers)?;
        scores = self.sdpa_optional_add(scores, alibi_bias, compute_type)?;
        scores = self.sdpa_optional_add(scores, modifiers.additive_mask(), compute_type)?;
        let causal_mask = self.sdpa_optional_causal_mask(q, k, modifiers)?;
        scores = self.sdpa_optional_add(scores, causal_mask, compute_type)?;
        let sliding_mask = self.sdpa_optional_sliding_window_mask(q, k, modifiers)?;
        scores = self.sdpa_optional_add(scores, sliding_mask, compute_type)?;
        if let Some((sequence_length_query, sequence_length_key_value)) =
            modifiers.sequence_lengths()
        {
            let mask = self.sdpa_sequence_length_additive_mask(
                scores,
                modifiers,
                sequence_length_query,
                sequence_length_key_value,
            )?;
            scores = self.sdpa_optional_add(scores, mask, compute_type)?;
        }
        Ok(scores)
    }

    fn sdpa_optional_add(
        &mut self,
        tensor: TensorId,
        rhs: Option<TensorId>,
        compute_type: DataType,
    ) -> Result<TensorId> {
        match rhs {
            Some(rhs) => self.pointwise_binary_infer(tensor, rhs, PointwiseMode::Add, compute_type),
            None => Ok(tensor),
        }
    }

    fn sdpa_backward_scores_with_modifiers(
        &mut self,
        scores: TensorId,
        scale: TensorId,
        modifiers: &SdpaScoreModifiers,
        score_subgraph: Option<&SdpaScoreSubgraph>,
        compute_type: DataType,
        intermediate_type: DataType,
    ) -> Result<TensorId> {
        let mut scores = self.pointwise_binary_virtual_infer_as(
            scores,
            scale,
            PointwiseMode::Mul,
            compute_type,
            intermediate_type,
        )?;
        if let Some(score_subgraph) = score_subgraph {
            scores =
                self.import_score_subgraph(score_subgraph, scores, SDPA_BACKWARD_SCORE_SUBGRAPH)?;
        }
        scores = self.sdpa_optional_add_virtual_as(
            scores,
            modifiers.bias(),
            compute_type,
            intermediate_type,
        )?;
        scores = self.sdpa_optional_add_virtual_as(
            scores,
            modifiers.sink_token(),
            compute_type,
            intermediate_type,
        )?;
        let alibi_bias = self.sdpa_optional_alibi_bias(scores, modifiers)?;
        scores =
            self.sdpa_optional_add_virtual_as(scores, alibi_bias, compute_type, intermediate_type)?;
        scores = self.sdpa_optional_add_virtual_as(
            scores,
            modifiers.additive_mask(),
            compute_type,
            intermediate_type,
        )?;
        if modifiers.causal_mask() {
            let masked = self.tensor(
                TensorSpec::new(intermediate_type, self.tensor_config(scores)?.shape.clone())
                    .virtual_tensor(),
            );
            let negative_inf = self.tensor(sdpa_mask_scalar(intermediate_type, f32::NEG_INFINITY)?);
            self.diagonal_band_mask(
                scores,
                negative_inf,
                masked,
                DiagonalBandMaskConfig::new(PointwiseMode::CmpGe),
            )?;
            scores = masked;
        }
        if let Some((left_window, right_window)) = modifiers.sliding_window() {
            let sliding_mask =
                self.diagonal_band_mask_composite(scores, left_window, Some(right_window))?;
            scores = self.sdpa_optional_add_virtual_as(
                scores,
                Some(sliding_mask),
                compute_type,
                intermediate_type,
            )?;
        }
        if let Some((sequence_length_query, sequence_length_key_value)) =
            modifiers.sequence_lengths()
        {
            let masked_scores = self.sdpa_sequence_length_select_mask(
                scores,
                modifiers,
                sequence_length_query,
                sequence_length_key_value,
            )?;
            if let Some(masked_scores) = masked_scores {
                scores = masked_scores;
            }
        }
        Ok(scores)
    }

    fn sdpa_backward_additive_scores_with_modifiers(
        &mut self,
        scores: TensorId,
        scale: TensorId,
        q: TensorId,
        k: TensorId,
        modifiers: &SdpaScoreModifiers,
        score_subgraph: Option<&SdpaScoreSubgraph>,
        score_subgraph_error_name: &str,
        compute_type: DataType,
        output_type: DataType,
    ) -> Result<TensorId> {
        let mut scores = self.pointwise_binary_infer_as(
            scores,
            scale,
            PointwiseMode::Mul,
            compute_type,
            output_type,
        )?;
        if let Some(score_subgraph) = score_subgraph {
            scores =
                self.import_score_subgraph(score_subgraph, scores, score_subgraph_error_name)?;
        }
        scores = self.sdpa_optional_add_as(scores, modifiers.bias(), compute_type, output_type)?;
        scores =
            self.sdpa_optional_add_as(scores, modifiers.sink_token(), compute_type, output_type)?;
        let alibi_bias = self.sdpa_optional_alibi_bias(scores, modifiers)?;
        scores = self.sdpa_optional_add_as(scores, alibi_bias, compute_type, output_type)?;
        scores = self.sdpa_optional_add_as(
            scores,
            modifiers.additive_mask(),
            compute_type,
            output_type,
        )?;
        let causal_mask = self.sdpa_optional_causal_mask(q, k, modifiers)?;
        scores = self.sdpa_optional_add_as(scores, causal_mask, compute_type, output_type)?;
        let sliding_mask = self.sdpa_optional_sliding_window_mask(q, k, modifiers)?;
        scores = self.sdpa_optional_add_as(scores, sliding_mask, compute_type, output_type)?;
        if let Some((sequence_length_query, sequence_length_key_value)) =
            modifiers.sequence_lengths()
        {
            let mask = self.sdpa_sequence_length_additive_mask(
                scores,
                modifiers,
                sequence_length_query,
                sequence_length_key_value,
            )?;
            scores = self.sdpa_optional_add_as(scores, mask, compute_type, output_type)?;
        }
        Ok(scores)
    }

    fn sdpa_sequence_length_additive_mask(
        &mut self,
        scores: TensorId,
        modifiers: &SdpaScoreModifiers,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Result<Option<TensorId>> {
        if modifiers.causal_bottom_right() {
            return Ok(Some(self.causal_bottom_right_mask(
                scores,
                sequence_length_query,
                sequence_length_key_value,
            )?));
        }
        if modifiers.padding_mask() {
            return Ok(Some(self.padding_mask(
                scores,
                sequence_length_query,
                sequence_length_key_value,
            )?));
        }
        Ok(None)
    }

    fn sdpa_sequence_length_select_mask(
        &mut self,
        scores: TensorId,
        modifiers: &SdpaScoreModifiers,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Result<Option<TensorId>> {
        if modifiers.causal_bottom_right() {
            return Ok(Some(self.causal_bottom_right_mask(
                scores,
                sequence_length_query,
                sequence_length_key_value,
            )?));
        }
        if modifiers.padding_mask() {
            return Ok(Some(self.padding_mask_select(
                scores,
                sequence_length_query,
                sequence_length_key_value,
            )?));
        }
        Ok(None)
    }

    fn sdpa_optional_alibi_bias(
        &mut self,
        scores: TensorId,
        modifiers: &SdpaScoreModifiers,
    ) -> Result<Option<TensorId>> {
        modifiers
            .alibi_slopes()
            .map(|alibi_slopes| self.alibi_bias(scores, alibi_slopes))
            .transpose()
    }

    fn sdpa_optional_causal_mask(
        &mut self,
        q: TensorId,
        k: TensorId,
        modifiers: &SdpaScoreModifiers,
    ) -> Result<Option<TensorId>> {
        modifiers
            .causal_mask()
            .then(|| self.sdpa_causal_mask_infer(q, k))
            .transpose()
    }

    fn sdpa_optional_sliding_window_mask(
        &mut self,
        q: TensorId,
        k: TensorId,
        modifiers: &SdpaScoreModifiers,
    ) -> Result<Option<TensorId>> {
        modifiers
            .sliding_window()
            .map(|(left_window, right_window)| {
                self.sdpa_sliding_window_mask_infer(q, k, left_window, right_window)
            })
            .transpose()
    }

    fn sdpa_optional_add_virtual_as(
        &mut self,
        tensor: TensorId,
        rhs: Option<TensorId>,
        compute_type: DataType,
        output_type: DataType,
    ) -> Result<TensorId> {
        match rhs {
            Some(rhs) => self.pointwise_binary_virtual_infer_as(
                tensor,
                rhs,
                PointwiseMode::Add,
                compute_type,
                output_type,
            ),
            None => Ok(tensor),
        }
    }

    fn sdpa_optional_add_as(
        &mut self,
        tensor: TensorId,
        rhs: Option<TensorId>,
        compute_type: DataType,
        output_type: DataType,
    ) -> Result<TensorId> {
        match rhs {
            Some(rhs) => self.pointwise_binary_infer_as(
                tensor,
                rhs,
                PointwiseMode::Add,
                compute_type,
                output_type,
            ),
            None => Ok(tensor),
        }
    }

    fn sdpa_apply_dropout(
        &mut self,
        probs: TensorId,
        dropout_mask: Option<TensorId>,
        dropout_scale: Option<TensorId>,
        compute_type: DataType,
    ) -> Result<TensorId> {
        if dropout_mask.is_none() || dropout_scale.is_none() {
            return Ok(probs);
        }
        let probs = self.sdpa_apply_dropout_mask(probs, dropout_mask, compute_type)?;
        self.sdpa_apply_dropout_scale(probs, dropout_scale, compute_type)
    }

    fn sdpa_apply_dropout_virtual_as(
        &mut self,
        probs: TensorId,
        dropout_mask: Option<TensorId>,
        dropout_scale: Option<TensorId>,
        compute_type: DataType,
        output_type: DataType,
    ) -> Result<TensorId> {
        if dropout_mask.is_none() || dropout_scale.is_none() {
            return Ok(probs);
        }
        let probs = self.sdpa_apply_dropout_mask_virtual_as(
            probs,
            dropout_mask,
            compute_type,
            output_type,
        )?;
        self.sdpa_apply_dropout_scale_virtual_as(probs, dropout_scale, compute_type, output_type)
    }

    fn sdpa_apply_dropout_mask(
        &mut self,
        tensor: TensorId,
        dropout_mask: Option<TensorId>,
        compute_type: DataType,
    ) -> Result<TensorId> {
        self.sdpa_optional_mul(tensor, dropout_mask, compute_type)
    }

    fn sdpa_apply_dropout_mask_virtual_as(
        &mut self,
        tensor: TensorId,
        dropout_mask: Option<TensorId>,
        compute_type: DataType,
        output_type: DataType,
    ) -> Result<TensorId> {
        self.sdpa_optional_mul_virtual_as(tensor, dropout_mask, compute_type, output_type)
    }

    fn sdpa_apply_dropout_scale(
        &mut self,
        tensor: TensorId,
        dropout_scale: Option<TensorId>,
        compute_type: DataType,
    ) -> Result<TensorId> {
        self.sdpa_optional_mul(tensor, dropout_scale, compute_type)
    }

    fn sdpa_apply_dropout_scale_virtual_as(
        &mut self,
        tensor: TensorId,
        dropout_scale: Option<TensorId>,
        compute_type: DataType,
        output_type: DataType,
    ) -> Result<TensorId> {
        self.sdpa_optional_mul_virtual_as(tensor, dropout_scale, compute_type, output_type)
    }

    fn sdpa_optional_mul(
        &mut self,
        tensor: TensorId,
        rhs: Option<TensorId>,
        compute_type: DataType,
    ) -> Result<TensorId> {
        match rhs {
            Some(rhs) => self.pointwise_binary_infer(tensor, rhs, PointwiseMode::Mul, compute_type),
            None => Ok(tensor),
        }
    }

    fn sdpa_optional_mul_virtual_as(
        &mut self,
        tensor: TensorId,
        rhs: Option<TensorId>,
        compute_type: DataType,
        output_type: DataType,
    ) -> Result<TensorId> {
        match rhs {
            Some(rhs) => self.pointwise_binary_virtual_infer_as(
                tensor,
                rhs,
                PointwiseMode::Mul,
                compute_type,
                output_type,
            ),
            None => Ok(tensor),
        }
    }

    fn sdpa_add_backward_sink_token_gradient(
        &mut self,
        d_sink_token: Option<TensorId>,
        modifiers: &SdpaScoreModifiers,
        aux_gradients: AttentionBackwardAuxGradientRequest,
        stats: TensorId,
        softmax_sum: TensorId,
        scores_shape: &Shape,
        compute_type: DataType,
        intermediate_type: DataType,
    ) -> Result<()> {
        let Some(d_sink_token) = d_sink_token else {
            if aux_gradients.sink_token() {
                return Err(Error::FrontendSdpaBackwardSinkGradientRequiresSink);
            }
            return Ok(());
        };
        let sink_token = if let Some(sink_token) = modifiers.sink_token() {
            sink_token
        } else {
            return Err(Error::FrontendSdpaBackwardSinkGradientRequiresSink);
        };

        let sink_minus_stats = self.pointwise_binary_virtual_infer_as(
            sink_token,
            stats,
            PointwiseMode::Sub,
            compute_type,
            intermediate_type,
        )?;
        let exp_sink = self.tensor(
            TensorSpec::new(intermediate_type, reduce_last_axis(scores_shape)?).virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Exp,
            input: sink_minus_stats,
            output: exp_sink,
            compute_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: None,
        });
        let per_token_grad = self.pointwise_binary_virtual_infer_as(
            exp_sink,
            softmax_sum,
            PointwiseMode::Mul,
            compute_type,
            intermediate_type,
        )?;
        self.reduction(ReductionOperation::Reduce {
            op: ReduceTensorOperator::Add,
            input: per_token_grad,
            output: d_sink_token,
            compute_type,
            is_deterministic: false,
        });
        Ok(())
    }

    fn sdpa_backward_modifiers_with_dropout(
        &mut self,
        config: &AttentionBackwardConfig,
        requested_rng_dump: Option<TensorId>,
        aux_gradients: AttentionBackwardAuxGradientRequest,
        scores_shape: &Shape,
    ) -> Result<SdpaScoreModifiers> {
        let Some(dropout) = config.dropout() else {
            return Ok(*config.modifiers());
        };

        let rng_dump = self.attention_dropout_random_number_generator_dump(
            dropout,
            config.dropout_offset(),
            scores_shape,
        )?;
        if let Some(requested_rng_dump) = requested_rng_dump {
            self.bind_sdpa_inferred_tensor(rng_dump, requested_rng_dump, SDPA_BACKWARD_RNG_DUMP)?;
        } else if aux_gradients.random_number_generator_dump() {
            return Err(Error::FrontendSdpaBackwardRngDumpOutputMissing);
        }

        let dropout_scale = self.tensor(sdpa_scalar_tensor(
            DataType::F32,
            1.0 / (1.0 - dropout.probability()),
        )?);
        Ok(config.modifiers().with_dropout(rng_dump, dropout_scale))
    }

    fn sdpa_backward_matmul_configs(
        &self,
        modifiers: &SdpaScoreModifiers,
        compute_type: DataType,
    ) -> (MatmulConfig, MatmulConfig) {
        let mut qk = MatmulConfig::new(compute_type);
        if let Some(m_override) = modifiers.sequence_length_query() {
            qk = qk.with_m_override(m_override);
        }
        if let Some(k_override) = modifiers.sequence_length_key_value() {
            qk = qk.with_k_override(k_override);
        }

        let mut kv = MatmulConfig::new(compute_type);
        if let Some(m_override) = modifiers.sequence_length_key_value() {
            kv = kv.with_m_override(m_override);
        }
        if let Some(k_override) = modifiers.sequence_length_query() {
            kv = kv.with_k_override(k_override);
        }

        (qk, kv)
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
        let aux_gradients = config.aux_gradients();
        let modifiers = self.sdpa_backward_modifiers_with_dropout(
            &config,
            requested_rng_dump,
            aux_gradients,
            &scores_shape,
        )?;
        let compute_type = config.softmax().compute_type();
        let (matmul_qk_config, matmul_kv_config) =
            self.sdpa_backward_matmul_configs(&modifiers, compute_type);
        let intermediate_type = self.effective_intermediate_data_type(DataType::F32);

        let scores =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        let k_t = if k_is_transposed {
            k
        } else {
            self.transpose_last_two(k)?
        };
        self.matmul_with_config(q, k_t, scores, matmul_qk_config.clone())?;

        let scaled_scores = self.sdpa_backward_scores_with_modifiers(
            scores,
            scale,
            &modifiers,
            config.score_subgraph(),
            compute_type,
            intermediate_type,
        )?;

        let shifted_scores = self.pointwise_binary_virtual_infer_as(
            scaled_scores,
            stats,
            PointwiseMode::Sub,
            compute_type,
            intermediate_type,
        )?;
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
        probs = self.sdpa_apply_dropout_virtual_as(
            probs,
            modifiers.dropout_mask(),
            modifiers.dropout_scale(),
            compute_type,
            intermediate_type,
        )?;

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
        d_p = self.sdpa_apply_dropout_mask_virtual_as(
            d_p,
            modifiers.dropout_mask(),
            compute_type,
            intermediate_type,
        )?;

        let d_o_mul_o = self.pointwise_binary_virtual_infer_as(
            d_o,
            o,
            PointwiseMode::Mul,
            compute_type,
            intermediate_type,
        )?;
        let softmax_sum = self.tensor(
            TensorSpec::new(intermediate_type, expected_stats_shape.clone()).virtual_tensor(),
        );
        self.reduction(ReductionOperation::Reduce {
            op: ReduceTensorOperator::Add,
            input: d_o_mul_o,
            output: softmax_sum,
            compute_type,
            is_deterministic: false,
        });

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

        let mut d_s = self.pointwise_binary_virtual_infer_as(
            d_p,
            softmax_sum,
            PointwiseMode::Sub,
            compute_type,
            intermediate_type,
        )?;
        d_s = self.pointwise_binary_virtual_infer_as(
            d_s,
            probs,
            PointwiseMode::Mul,
            compute_type,
            intermediate_type,
        )?;
        d_s = self.sdpa_apply_dropout_scale_virtual_as(
            d_s,
            modifiers.dropout_scale(),
            compute_type,
            intermediate_type,
        )?;
        if let Some(d_bias) = d_bias {
            self.reshape(d_s, d_bias)?;
        } else if aux_gradients.bias() {
            return Err(Error::FrontendSdpaBackwardBiasGradientOutputMissing);
        }
        if let Some(score_subgraph_bprop) = config.score_subgraph_bprop() {
            d_s = self.import_score_subgraph(
                score_subgraph_bprop,
                d_s,
                SDPA_BACKWARD_SCORE_SUBGRAPH_BPROP,
            )?;
        }
        d_s = self.pointwise_binary_virtual_infer_as(
            d_s,
            scale,
            PointwiseMode::Mul,
            compute_type,
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
        let aux_gradients = config.aux_gradients();
        let d_bias = if aux_gradients.bias() {
            Some(self.tensor(TensorSpec::new(DataType::F32, scores_shape.clone())))
        } else {
            None
        };
        let rng_dump = if aux_gradients.random_number_generator_dump() {
            Some(self.tensor(TensorSpec::new(DataType::F32, scores_shape.clone())))
        } else {
            None
        };
        let d_sink_token = if aux_gradients.sink_token() {
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
        let is_ragged = self.has_any_ragged_offset([q, k, v, o, d_o])?;
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
            (scale_q_t, SDPA_MXFP8_BACKWARD_SCALE_Q_T),
            (scale_k, SDPA_MXFP8_BACKWARD_SCALE_K),
            (scale_k_t, SDPA_MXFP8_BACKWARD_SCALE_K_T),
            (scale_v, SDPA_MXFP8_BACKWARD_SCALE_V),
            (scale_d_o, SDPA_MXFP8_BACKWARD_SCALE_D_O),
            (scale_d_o_t, SDPA_MXFP8_BACKWARD_SCALE_D_O_T),
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
        let d_o_tensor = self.tensor_config(d_o)?.clone();
        if config.modifiers().sliding_window().is_some()
            && !config.modifiers().padding_mask()
            && q_tensor.shape.dimensions()[2] > k_tensor.shape.dimensions()[2]
        {
            return Err(Error::FrontendSdpaSlidingWindowSequenceLengthUnsupported {
                query_sequence_length: q_tensor.shape.dimensions()[2],
                key_sequence_length: k_tensor.shape.dimensions()[2],
            });
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

        let attention_scale = self.mxfp8_attn_scale_tensor(
            attention_scale,
            SDPA_MXFP8_BACKWARD_ATTENTION_SCALE,
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

        let modifiers = config.modifiers();
        let scaled_scores = self.sdpa_backward_additive_scores_with_modifiers(
            scores,
            attention_scale,
            q_fp,
            k_fp,
            modifiers,
            config.score_subgraph(),
            SDPA_MXFP8_BACKWARD_SCORE_SUBGRAPH,
            compute_type,
            intermediate_type,
        )?;

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
        probs = self.sdpa_apply_dropout(
            probs,
            modifiers.dropout_mask(),
            modifiers.dropout_scale(),
            compute_type,
        )?;

        let probs_t = self.transpose_last_two(probs)?;
        let d_v_fp = self.tensor(TensorSpec::new(intermediate_type, v_tensor.shape.clone()));
        self.matmul(probs_t, d_o_t_fp, d_v_fp, compute_type)?;

        let mut d_p =
            self.tensor(TensorSpec::new(intermediate_type, scores_shape.clone()).virtual_tensor());
        self.matmul(d_o_fp, v_t_fp, d_p, compute_type)?;
        d_p = self.sdpa_apply_dropout_mask(d_p, modifiers.dropout_mask(), compute_type)?;

        let d_o_mul_o = self.pointwise_binary_infer(d_o, o, PointwiseMode::Mul, compute_type)?;
        let softmax_sum = self.reduction_infer(
            d_o_mul_o,
            ReductionConfig::new(ReduceTensorOperator::Add, compute_type),
        )?;

        let mut d_s =
            self.pointwise_binary_infer(d_p, softmax_sum, PointwiseMode::Sub, compute_type)?;
        d_s = self.pointwise_binary_infer(d_s, probs, PointwiseMode::Mul, compute_type)?;
        d_s = self.sdpa_apply_dropout_scale(d_s, modifiers.dropout_scale(), compute_type)?;
        if let Some(score_subgraph_bprop) = config.score_subgraph_bprop() {
            d_s = self.import_score_subgraph(
                score_subgraph_bprop,
                d_s,
                SDPA_MXFP8_BACKWARD_SCORE_SUBGRAPH_BPROP,
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
            "mxfp8 sdpa backward",
        )?;
        let result = self.sdpa_mxfp8_backward_infer_with_output_types(
            inputs,
            self.tensor_config(d_q)?.data_type,
            self.tensor_config(d_k)?.data_type,
            self.tensor_config(d_v)?.data_type,
            config,
        );
        let result = result?;

        self.bind_sdpa_inferred_tensor(result.query_gradient, d_q, SDPA_MXFP8_BACKWARD_D_Q)?;
        self.bind_sdpa_inferred_tensor(result.key_gradient, d_k, SDPA_MXFP8_BACKWARD_D_K)?;
        self.bind_sdpa_inferred_tensor(result.value_gradient, d_v, SDPA_MXFP8_BACKWARD_D_V)?;
        self.bind_sdpa_inferred_tensor(
            result.absolute_max_query_gradient,
            absolute_max_d_q,
            SDPA_MXFP8_BACKWARD_ABSOLUTE_MAX_D_Q,
        )?;
        self.bind_sdpa_inferred_tensor(
            result.absolute_max_key_gradient,
            absolute_max_d_k,
            SDPA_MXFP8_BACKWARD_ABSOLUTE_MAX_D_K,
        )?;
        self.bind_sdpa_inferred_tensor(
            result.absolute_max_value_gradient,
            absolute_max_d_v,
            SDPA_MXFP8_BACKWARD_ABSOLUTE_MAX_D_V,
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
        let output_io_type = self.data_type_policy().io().unwrap_or(q_data_type);
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
        let q_tensor = self.tensor_config(q)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();
        let q_data_type = q_tensor.data_type;
        for (scale, name) in [
            (attention_scale, SDPA_FP8_BACKWARD_ATTENTION_SCALE),
            (descale_q, SDPA_FP8_BACKWARD_DESCALE_Q),
            (descale_k, SDPA_FP8_BACKWARD_DESCALE_K),
            (descale_v, SDPA_FP8_BACKWARD_DESCALE_V),
            (descale_o, SDPA_FP8_BACKWARD_DESCALE_O),
            (descale_d_o, SDPA_FP8_BACKWARD_DESCALE_D_O),
            (descale_s, SDPA_FP8_BACKWARD_DESCALE_S),
            (descale_d_p, SDPA_FP8_BACKWARD_DESCALE_D_P),
            (scale_s, SDPA_FP8_BACKWARD_SCALE_S),
            (scale_d_q, SDPA_FP8_BACKWARD_SCALE_D_Q),
            (scale_d_k, SDPA_FP8_BACKWARD_SCALE_D_K),
            (scale_d_v, SDPA_FP8_BACKWARD_SCALE_D_V),
            (scale_d_p, SDPA_FP8_BACKWARD_SCALE_D_P),
        ] {
            self.validate_fp8_scale_tensor(scale, name)?;
        }
        let is_ragged = self.has_any_ragged_offset([q, k, v, o, d_o])?;
        if config.aux_gradients().bias() {
            return Err(Error::FrontendSdpaFp8BackwardBiasGradientUnsupported);
        }
        let cudnn_version = version()?.raw();
        let d_qk = q_tensor.shape.dimensions()[3];
        let d_v = v_tensor.shape.dimensions()[3];
        for output_type in [d_q_output_type, d_k_output_type, d_v_output_type] {
            self.validate_fp8_backward_support_surface_for_version(
                cudnn_version,
                output_type,
                d_qk,
                d_v,
                is_ragged,
                &config,
            )?;
        }
        self.validate_quantized_backward_head_dimensions(d_qk, d_v, "fp8 sdpa backward")?;
        if config.modifiers().sliding_window().is_some()
            && !config.modifiers().padding_mask()
            && q_tensor.shape.dimensions()[2] > self.tensor_config(k)?.shape.dimensions()[2]
        {
            return Err(Error::FrontendSdpaSlidingWindowSequenceLengthUnsupported {
                query_sequence_length: q_tensor.shape.dimensions()[2],
                key_sequence_length: self.tensor_config(k)?.shape.dimensions()[2],
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

        let modifiers = config.modifiers();
        let scaled_scores = self.sdpa_backward_additive_scores_with_modifiers(
            scores,
            attention_scale,
            q_fp,
            k_fp,
            modifiers,
            config.score_subgraph(),
            SDPA_FP8_BACKWARD_SCORE_SUBGRAPH,
            DataType::F32,
            DataType::F32,
        )?;

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
        probs = self.sdpa_apply_dropout(
            probs,
            modifiers.dropout_mask(),
            modifiers.dropout_scale(),
            DataType::F32,
        )?;

        let v_t_fp = self.transpose_last_two(v_fp)?;
        let mut d_p = self.tensor(
            TensorSpec::new(DataType::F32, self.tensor_config(probs)?.shape.clone())
                .virtual_tensor(),
        );
        self.matmul(d_o_fp, v_t_fp, d_p, DataType::F32)?;
        d_p = self.sdpa_apply_dropout_mask(d_p, modifiers.dropout_mask(), DataType::F32)?;

        let d_o_mul_o =
            self.pointwise_binary_infer(d_o_fp, o_fp, PointwiseMode::Mul, DataType::F32)?;
        let softmax_sum = self.reduction_infer(
            d_o_mul_o,
            ReductionConfig::new(ReduceTensorOperator::Add, DataType::F32),
        )?;

        d_p = self.pointwise_binary_infer(d_p, softmax_sum, PointwiseMode::Sub, DataType::F32)?;
        d_p = self.pointwise_binary_infer(d_p, probs, PointwiseMode::Mul, DataType::F32)?;
        d_p = self.sdpa_apply_dropout_scale(d_p, modifiers.dropout_scale(), DataType::F32)?;
        if let Some(score_subgraph_bprop) = config.score_subgraph_bprop() {
            d_p = self.import_score_subgraph(
                score_subgraph_bprop,
                d_p,
                SDPA_FP8_BACKWARD_SCORE_SUBGRAPH_BPROP,
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
        self.validate_quantized_backward_output_ragged_support(d_q, d_k, d_v, "fp8 sdpa backward")?;
        let computed = self.sdpa_fp8_backward_infer_with_aux_and_output_types(
            inputs,
            self.tensor_config(d_q)?.data_type,
            self.tensor_config(d_k)?.data_type,
            self.tensor_config(d_v)?.data_type,
            config,
        )?;

        self.bind_sdpa_inferred_tensor(computed.query_gradient, d_q, SDPA_FP8_BACKWARD_D_Q)?;
        self.bind_sdpa_inferred_tensor(computed.key_gradient, d_k, SDPA_FP8_BACKWARD_D_K)?;
        self.bind_sdpa_inferred_tensor(computed.value_gradient, d_v, SDPA_FP8_BACKWARD_D_V)?;
        self.bind_sdpa_inferred_optional_tensor(
            computed.sink_token_gradient,
            d_sink_token,
            SDPA_FP8_BACKWARD_D_SINK_TOKEN,
        )?;
        self.bind_sdpa_inferred_tensor(
            computed.absolute_max_query_gradient,
            absolute_max_d_q,
            SDPA_FP8_BACKWARD_ABSOLUTE_MAX_D_Q,
        )?;
        self.bind_sdpa_inferred_tensor(
            computed.absolute_max_key_gradient,
            absolute_max_d_k,
            SDPA_FP8_BACKWARD_ABSOLUTE_MAX_D_K,
        )?;
        self.bind_sdpa_inferred_tensor(
            computed.absolute_max_value_gradient,
            absolute_max_d_v,
            SDPA_FP8_BACKWARD_ABSOLUTE_MAX_D_V,
        )?;
        self.bind_sdpa_inferred_tensor(
            computed.absolute_max_probability_gradient,
            absolute_max_d_p,
            SDPA_FP8_BACKWARD_ABSOLUTE_MAX_D_P,
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
