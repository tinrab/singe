use crate::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{TensorId, TensorSpec},
    },
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::{
            SdpaForwardOutputTensors, SdpaInputs, SdpaOutputTensors, infer_sdpa_output_shape,
            infer_sdpa_scores_shape, support,
            support::{SDPA_OUTPUT_SHAPE, SDPA_RANK, SDPA_SCORE_SUBGRAPH, SDPA_STATS_DATA_TYPE},
        },
        error::{Error as FrontendError, RankExpectation, SdpaFeature},
        graph::{Graph, SoftmaxOutputs, TensorOutputConfig},
        operation::{
            AlibiBias, MatmulConfig, PointwiseOperation, SdpaQuantizedAuxOutputRequest,
            SdpaScoreModifiers, SdpaScoreSubgraph, SoftmaxConfig,
        },
    },
};

struct SdpaForwardInputTensors {
    scores: TensorSpec,
    k_matmul: TensorId,
    v: TensorSpec,
}

impl Graph {
    pub(crate) fn sdpa_with_modifiers_internal(
        &mut self,
        inputs: SdpaInputs,
        outputs: SdpaOutputTensors,
        modifiers: SdpaScoreModifiers,
        score_subgraph: Option<&SdpaScoreSubgraph>,
        quantized_aux: SdpaQuantizedAuxOutputRequest,
        config: SoftmaxConfig,
    ) -> Result<SdpaForwardOutputTensors> {
        let SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        } = inputs;
        let SdpaOutputTensors {
            output,
            stats,
            logit_max,
            score_sum_exp,
        } = outputs;
        if let Some(stats) = stats {
            self.validate_sdpa_tensor_data_type_match(stats, DataType::F32, SDPA_STATS_DATA_TYPE)?;
        }
        let compute_type = self.resolve_compute_data_type(config.compute_type());
        let (score_matmul, output_matmul) =
            Self::sdpa_forward_matmul_configs(&modifiers, compute_type);
        if modifiers.sequence_inputs().has_cumulative() {
            return Err(FrontendError::SdpaUnsupported {
                operation: "composite sdpa",
                feature: SdpaFeature::CumulativeSequenceLengths,
                sm_version: None,
            }
            .into());
        }
        self.promote_index_tensors(modifiers.sequence_inputs().tensors())?;
        let inputs =
            self.prepare_sdpa_forward_inputs(q, k, v, output, &modifiers, score_subgraph)?;

        let scores = self.tensor(inputs.scores);
        self.matmul_into(q, inputs.k_matmul, scores, score_matmul)?;

        let scores_with_modifiers = self.sdpa_scores_with_modifiers(
            scores,
            Some(scale),
            &modifiers,
            score_subgraph,
            SDPA_SCORE_SUBGRAPH,
            compute_type,
            self.sdpa_score_data_type(),
        )?;

        let config = match modifiers.sink_token() {
            Some(sink) => config.with_sink(sink),
            None => config,
        };
        let probability_shape = self.shape(scores_with_modifiers)?.clone();
        let intermediate_type = self.effective_intermediate_data_type(DataType::F32);
        let probabilities = self.inferred_tensor(
            TensorSpec::new(intermediate_type, probability_shape).virtual_tensor(),
            config.name(),
            "S",
        );
        self.attention_softmax(
            scores_with_modifiers,
            SoftmaxOutputs {
                output: probabilities,
                stats,
                max: logit_max,
                sum_exp: score_sum_exp,
            },
            &config,
        )?;
        let probs = self.sdpa_apply_dropout(
            probabilities,
            modifiers.dropout_mask(),
            modifiers.dropout_scale(),
            compute_type,
        )?;
        let query_type = self.tensor_spec(q)?.data_type;
        // C++ narrows the final probability/dropout result to Q's encoding before BMM2.
        self.modify_tensor(probs, |mut tensor| {
            tensor.data_type = query_type;
            tensor.virtual_tensor()
        })?;
        let absolute_max_probabilities = if quantized_aux.absolute_max_s() {
            Some(self.absolute_max_infer(probs, compute_type)?)
        } else {
            None
        };

        let inferred_output_shape = infer_sdpa_output_shape(self.shape(probs)?, &inputs.v.shape)?;
        self.validate_sdpa_tensor_dimensions_match(
            output,
            inferred_output_shape.dimensions(),
            SDPA_OUTPUT_SHAPE,
        )?;

        self.matmul_into(probs, v, output, output_matmul)?;
        let absolute_max_output = if quantized_aux.absolute_max_o() {
            Some(self.absolute_max_infer(output, compute_type)?)
        } else {
            None
        };

        Ok(SdpaForwardOutputTensors {
            output,
            stats,
            logit_max,
            score_sum_exp,
            rng_dump: None,
            absolute_max_probabilities,
            absolute_max_output,
        })
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
        let q_tensor = self.tensor_spec(q)?.clone();
        let k_tensor = self.tensor_spec(k)?.clone();
        let v_tensor = self.tensor_spec(v)?.clone();
        Self::validate_sdpa_forward_ranks(q, &q_tensor, k, &k_tensor, v, &v_tensor)?;
        self.validate_sdpa_score_modifiers(q, k, v, modifiers)?;
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        self.validate_sdpa_forward_modifier_support(q, k, v, output, modifiers, score_subgraph)?;
        let k_matmul = self.transpose_last_two_view(k)?;

        Ok(SdpaForwardInputTensors {
            scores: TensorSpec::new(
                self.effective_intermediate_data_type(DataType::F32),
                scores_shape,
            )
            .virtual_tensor(),
            k_matmul,
            v: v_tensor,
        })
    }

    fn validate_sdpa_forward_ranks(
        q: TensorId,
        q_tensor: &TensorSpec,
        k: TensorId,
        k_tensor: &TensorSpec,
        v: TensorId,
        v_tensor: &TensorSpec,
    ) -> Result<()> {
        for (tensor_id, tensor) in [(q, q_tensor), (k, k_tensor), (v, v_tensor)] {
            let actual = tensor.shape.rank();
            if actual != 4 {
                return Err(FrontendError::TensorRankMismatch {
                    tensor_id,
                    operation: SDPA_RANK,
                    expected: RankExpectation::Exact(4),
                    actual,
                }
                .into());
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
            modifiers.has_sequence_lengths(),
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
        Ok(())
    }

    pub(crate) fn sdpa_optional_alibi_bias(
        &mut self,
        scores: TensorId,
        modifiers: &SdpaScoreModifiers,
    ) -> Result<Option<TensorId>> {
        let slopes = match modifiers.alibi() {
            None => return Ok(None),
            Some(AlibiBias::Slopes(slopes)) => slopes,
            Some(AlibiBias::Standard) => {
                let heads = *self
                    .shape(scores)?
                    .dimensions()
                    .get(1)
                    .ok_or(Error::InvalidDataShape)?;
                self.standard_alibi_slopes(heads)?
            }
        };
        self.alibi_bias(scores, slopes).map(Some)
    }

    pub(crate) fn sdpa_optional_add(
        &mut self,
        tensor: TensorId,
        rhs: Option<TensorId>,
        compute_type: DataType,
        output_type: DataType,
    ) -> Result<TensorId> {
        rhs.map_or_else(
            || Ok(tensor),
            |rhs| self.sdpa_binary(PointwiseMode::Add, tensor, rhs, output_type, compute_type),
        )
    }

    pub(crate) fn sdpa_apply_dropout(
        &mut self,
        probs: TensorId,
        dropout_mask: Option<TensorId>,
        dropout_scale: Option<TensorId>,
        compute_type: DataType,
    ) -> Result<TensorId> {
        let output_type = self.tensor_spec(probs)?.data_type;
        self.sdpa_apply_dropout_with_data_type(
            probs,
            dropout_mask,
            dropout_scale,
            compute_type,
            output_type,
        )
    }

    pub(crate) fn sdpa_apply_dropout_with_data_type(
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
        let probs = self.sdpa_optional_mul(probs, dropout_mask, compute_type, output_type)?;
        self.sdpa_optional_mul(probs, dropout_scale, compute_type, output_type)
    }

    pub(crate) fn sdpa_optional_mul(
        &mut self,
        tensor: TensorId,
        rhs: Option<TensorId>,
        compute_type: DataType,
        output_type: DataType,
    ) -> Result<TensorId> {
        rhs.map_or_else(
            || Ok(tensor),
            |rhs| self.sdpa_binary(PointwiseMode::Mul, tensor, rhs, output_type, compute_type),
        )
    }

    /// Records a binary pointwise step with an explicit output and compute type.
    pub(crate) fn sdpa_binary(
        &mut self,
        mode: PointwiseMode,
        lhs: TensorId,
        rhs: TensorId,
        output_type: DataType,
        compute_type: DataType,
    ) -> Result<TensorId> {
        self.pointwise(
            PointwiseOperation::binary(
                mode,
                lhs,
                rhs,
                TensorOutputConfig::new().with_data_type(output_type),
            )
            .with_compute_type(compute_type),
        )
    }

    pub(crate) fn sdpa_forward_matmul_configs(
        modifiers: &SdpaScoreModifiers,
        compute_type: DataType,
    ) -> (MatmulConfig, MatmulConfig) {
        let query_length = modifiers.sequence_length_query();
        let key_value_length = modifiers.sequence_length_key_value();
        (
            MatmulConfig::new()
                .with_compute_type(compute_type)
                .with_m_override(query_length)
                .with_n_override(key_value_length),
            MatmulConfig::new()
                .with_compute_type(compute_type)
                .with_m_override(query_length)
                .with_k_override(key_value_length),
        )
    }

    /// Composes attention with explicit score modifiers and result tensors.
    /// Rejected composition preserves existing tensor metadata, bounds and operations.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown or incompatible attention operands or output dimensions, unsupported cumulative sequence lengths, invalid mask or sequence metadata, incompatible score modifiers or softmax summaries, or failure to construct constituent matrix, pointwise, reduction, or dropout operations.
    pub fn sdpa_with_modifiers(
        &mut self,
        inputs: SdpaInputs,
        outputs: SdpaOutputTensors,
        modifiers: SdpaScoreModifiers,
        config: SoftmaxConfig,
    ) -> Result<()> {
        modifiers.sequence_inputs().validate_forward()?;
        let mut composed = self.clone();
        composed.sdpa_with_modifiers_internal(
            inputs,
            outputs,
            modifiers,
            None,
            SdpaQuantizedAuxOutputRequest::default(),
            config,
        )?;
        *self = composed;
        Ok(())
    }
}
