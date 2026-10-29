use std::collections::BTreeSet;

use crate::{
    backend::{
        execution::attention::SoftmaxStatsBase,
        tensor::{Shape, TensorId},
    },
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::{
            SdpaBackwardGradientTensors, SdpaBackwardInputs, attention_heads_compatible,
            infer_sdpa_output_shape, infer_sdpa_scores_shape, support,
            support::{
                SDPA_ALIBI_DATA_TYPE, SDPA_ALIBI_SHAPE, SDPA_BACKWARD_D_BIAS_SHAPE,
                SDPA_BACKWARD_D_K_LAYOUT, SDPA_BACKWARD_D_K_SHAPE, SDPA_BACKWARD_D_O_LAYOUT,
                SDPA_BACKWARD_D_O_SHAPE, SDPA_BACKWARD_D_Q_LAYOUT, SDPA_BACKWARD_D_Q_SHAPE,
                SDPA_BACKWARD_D_SINK_TOKEN_SHAPE, SDPA_BACKWARD_D_V_LAYOUT,
                SDPA_BACKWARD_D_V_SHAPE, SDPA_BACKWARD_RANK, SDPA_BACKWARD_SCALE,
                SDPA_BACKWARD_SCORE_SUBGRAPH, SDPA_BACKWARD_SCORE_SUBGRAPH_BPROP,
                SDPA_BACKWARD_STATS_SHAPE, SDPA_BIAS_LAYOUT, SDPA_BLOCK_MASK_DATA_TYPE,
                SDPA_BLOCK_MASK_LAYOUT, SDPA_DROPOUT_PROBABILITY, SDPA_DROPOUT_SCALE,
                SDPA_K_LAYOUT, SDPA_MAX_SEQUENCE_LENGTH_KEY_VALUE, SDPA_OUTPUT_LAYOUT,
                SDPA_PAGED_K_PAGE_TABLE_DATA_TYPE, SDPA_PAGED_V_PAGE_TABLE_DATA_TYPE,
                SDPA_Q_LAYOUT, SDPA_RANK, SDPA_RNG_OFFSET, SDPA_RNG_OFFSET_DATA_TYPE,
                SDPA_RNG_SEED, SDPA_RNG_SEED_DATA_TYPE, SDPA_SCORE_SUBGRAPH,
                SDPA_SEQ_LEN_DATA_TYPE, SDPA_SEQ_LEN_SHAPE, SDPA_SINK_TOKEN_RANK,
                SDPA_SINK_TOKEN_SHAPE,
            },
        },
        error::{Error as FrontendError, RankExpectation, SdpaFeature, SdpaRequirement},
        graph::{Graph, reduce_last_axis},
        operation::{
            AttentionBackwardConfig, AttentionConfig, AttentionDropoutConfig,
            AttentionImplementation, AttentionMaxTotalSequenceLengths, AttentionPagedKvCache,
            AttentionSequenceInputs, DirectSdpaForwardConfig, Operation, SdpaScoreModifiers,
            SdpaScoreSubgraph,
        },
    },
    utility::check_range,
    version,
};

pub struct SdpaBackwardValidation {
    pub scores_shape: Shape,
    pub expected_stats_shape: Shape,
}

impl Graph {
    pub(crate) fn validate_sdpa_rank4_scalar_data_type_supported(
        &self,
        tensor_id: TensorId,
        supported_data_types: &[DataType],
        data_type_error_name: &'static str,
        shape_error_name: &'static str,
    ) -> Result<()> {
        let tensor = self.tensor_spec(tensor_id)?;
        if !supported_data_types.contains(&tensor.data_type) {
            return Err(FrontendError::TensorDataTypeUnsupported {
                tensor_id,
                operation: data_type_error_name,
                actual: tensor.data_type,
                supported: supported_data_types.to_vec(),
            }
            .into());
        }
        let shape = tensor.shape_with_scalar_rank(4)?;
        if shape.dimensions() != [1, 1, 1, 1] {
            return Err(FrontendError::TensorDimensionsMismatch {
                tensor_id,
                operation: shape_error_name,
                expected: vec![1, 1, 1, 1],
                actual: tensor.shape.dimensions().to_vec(),
            }
            .into());
        }
        self.validate_tensor_scalar_shape(tensor_id, shape_error_name)
    }

    /// Forward hints admit ragged Q/K/V/O, matching C++ even when only the declared output is packed.
    pub(crate) fn validate_attention_forward_ragged_layout(
        &self,
        lengths: AttentionMaxTotalSequenceLengths,
        tensors: [TensorId; 4],
    ) -> Result<()> {
        if !lengths.is_empty() && !self.has_any_ragged_offset(tensors)? {
            return Err(FrontendError::SdpaRequires {
                operation: "sdpa max total sequence lengths",
                requirement: SdpaRequirement::RaggedTensors,
            }
            .into());
        }
        Ok(())
    }

    /// Uses the same complete operand set as C++ for packed-layout admission and architecture restrictions.
    pub(crate) fn validate_attention_backward_ragged_layout(
        &self,
        config: &AttentionBackwardConfig,
        tensors: [TensorId; 9],
    ) -> Result<bool> {
        let is_ragged = self.has_any_ragged_offset(tensors)?;
        if !config.max_total_sequence_lengths().is_empty() && !is_ragged {
            return Err(FrontendError::SdpaRequires {
                operation: "sdpa max total sequence lengths",
                requirement: SdpaRequirement::RaggedTensors,
            }
            .into());
        }
        Ok(is_ragged)
    }

    pub(crate) fn has_any_ragged_offset<const N: usize>(
        &self,
        tensors: [TensorId; N],
    ) -> Result<bool> {
        for tensor in tensors {
            if self.tensor_spec(tensor)?.ragged_offset.is_some() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn validate_sdpa_sequence_length_data_types(
        &self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
        error_name: &'static str,
    ) -> Result<()> {
        self.validate_sdpa_tensor_data_types([
            (sequence_length_query, DataType::I32, error_name),
            (sequence_length_key_value, DataType::I32, error_name),
        ])
    }

    pub(crate) fn validate_sdpa_tensor_layout(
        &self,
        tensor: TensorId,
        name: &'static str,
    ) -> Result<()> {
        let tensor_spec = self.tensor_spec(tensor)?;
        let actual_rank = tensor_spec.shape.rank();
        if actual_rank != 4 {
            return Err(FrontendError::TensorRankMismatch {
                tensor_id: tensor,
                operation: SDPA_RANK,
                expected: RankExpectation::Exact(4),
                actual: actual_rank,
            }
            .into());
        }
        let actual = tensor_spec.shape.strides()[3];
        if actual != 1 {
            return Err(FrontendError::TensorLastStrideMismatch {
                tensor_id: tensor,
                operation: name,
                expected: 1,
                actual,
            }
            .into());
        }

        Ok(())
    }

    pub(crate) fn validate_sdpa_tensor_layouts<const N: usize>(
        &self,
        tensors: [(TensorId, &'static str); N],
    ) -> Result<()> {
        for (tensor, name) in tensors {
            self.validate_sdpa_tensor_layout(tensor, name)?;
        }
        Ok(())
    }

    pub(crate) fn validate_sdpa_k_layout(&self, tensor: TensorId) -> Result<()> {
        let tensor_spec = self.tensor_spec(tensor)?;
        let actual_rank = tensor_spec.shape.rank();
        if actual_rank != 4 {
            return Err(FrontendError::TensorRankMismatch {
                tensor_id: tensor,
                operation: SDPA_RANK,
                expected: RankExpectation::Exact(4),
                actual: actual_rank,
            }
            .into());
        }
        if tensor_spec.shape.strides()[2] != 1 && tensor_spec.shape.strides()[3] != 1 {
            return Err(FrontendError::TensorLayoutMismatch {
                tensor_id: tensor,
                operation: SDPA_K_LAYOUT,
                reason: "expected either the last or second-last stride to be 1".into(),
            }
            .into());
        }

        Ok(())
    }

    pub(crate) fn validate_sdpa_direct_ragged_support(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        o: TensorId,
        direct_config: &DirectSdpaForwardConfig,
        has_custom_score_subgraph: bool,
    ) -> Result<()> {
        let is_ragged = self.has_any_ragged_offset([q, k, v, o])?;
        let has_padding_mask =
            direct_config.padding_mask() || !direct_config.sequence_inputs().is_empty();
        support::require_sdpa_ragged_mask_or_subgraph(
            is_ragged,
            has_padding_mask,
            false,
            has_custom_score_subgraph,
            "sdpa",
        )?;

        Ok(())
    }

    pub(crate) fn validate_sdpa_score_modifiers(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        modifiers: &SdpaScoreModifiers,
    ) -> Result<()> {
        self.validate_sdpa_tensor_layout(q, SDPA_Q_LAYOUT)?;
        self.validate_sdpa_k_layout(k)?;
        self.validate_sdpa_k_layout(v)?;

        let q_tensor = self.tensor_spec(q)?;
        let k_tensor = self.tensor_spec(k)?;
        let v_tensor = self.tensor_spec(v)?;
        let q_dims = q_tensor.shape.dimensions();
        let k_dims = k_tensor.shape.dimensions();
        let v_dims = v_tensor.shape.dimensions();

        if !attention_heads_compatible(q_dims[1], k_dims[1])
            || !attention_heads_compatible(q_dims[1], v_dims[1])
        {
            return Err(FrontendError::SdpaGroupedQueryAttentionHeadsMismatch {
                query_heads: q_dims[1],
                key_heads: k_dims[1],
                value_heads: v_dims[1],
                operation: "sdpa",
            }
            .into());
        }

        self.validate_sdpa_modifier_sequence_lengths(modifiers, q_dims[0])?;
        if let Some(dropout) = modifiers.custom_dropout() {
            self.validate_tensor_scalar_shape(dropout.scale, SDPA_DROPOUT_SCALE)?;
        }

        if let Some(bias) = modifiers.bias() {
            self.validate_sdpa_tensor_layout(bias, SDPA_BIAS_LAYOUT)?;
            if matches!(
                self.tensor_spec(bias)?.data_type,
                DataType::Boolean | DataType::ByteBoolean
            ) {
                return Err(FrontendError::SdpaUnsupported {
                    operation: "sdpa",
                    feature: SdpaFeature::NonNumericBias,
                    sm_version: None,
                }
                .into());
            }
        }

        self.validate_sdpa_modifier_alibi(modifiers, q_dims[1])?;
        self.validate_sdpa_modifier_sink_token(modifiers, q_dims[1], q_dims[2])?;

        if modifiers.causal_bottom_right()
            && modifiers.has_causal_bottom_right_incompatible_modifier()
        {
            return Err(sdpa_causal_bottom_right_modifier_conflict(modifiers));
        }
        if let Some(left_window) = modifiers.left_window() {
            check_range!("left_window", left_window >= 0)?;
        }
        if let Some(right_window) = modifiers.right_window() {
            check_range!("right_window", right_window >= 0)?;
        }

        Ok(())
    }

    /// Normalizes native ragged-offset descriptors without touching unused declarations.
    pub(crate) fn promote_attention_ragged_offsets(&mut self) -> Result<()> {
        let mut offsets = BTreeSet::new();
        for operation in &self.operations {
            let uses = match operation {
                Operation::AttentionForward(request) => request.direct_tensor_uses(),
                Operation::AttentionBackward(_) | Operation::Sdpa(_) => {
                    operation.tensor_uses(self.config.data_type_policy)
                }
                _ => continue,
            };
            for id in uses.reads.into_iter().chain(uses.writes) {
                // Score subgraphs also list their own external bindings here. They are
                // imported later; only operands declared in this graph belong to this pass.
                if let Some(offset) = self
                    .tensors
                    .get(&id)
                    .and_then(|tensor| tensor.ragged_offset)
                {
                    offsets.insert(offset);
                }
            }
        }
        self.promote_index_tensors(offsets)
    }

    pub(crate) fn validate_attention_sequence_inputs(
        &self,
        sequence: AttentionSequenceInputs<TensorId>,
        batch_size: i64,
    ) -> Result<()> {
        for lengths in sequence.query.into_iter().chain(sequence.key_value) {
            let count = batch_size
                .checked_add(i64::from(lengths.is_cumulative()))
                .ok_or_else(|| Error::out_of_range("attention sequence length count"))?;
            let id = lengths.tensor();
            self.validate_tensor_data_type(id, DataType::I32, SDPA_SEQ_LEN_DATA_TYPE)?;
            let tensor = self.tensor_spec(id)?;
            if !lengths.is_cumulative()
                && tensor.scalar_value().is_some()
                && tensor.has_scalar_shape()?
            {
                continue;
            }
            if tensor.shape.rank() == 1 {
                self.validate_tensor_dimensions(id, &[count], SDPA_SEQ_LEN_SHAPE)?;
            } else {
                self.validate_tensor_dimensions(id, &[count, 1, 1, 1], SDPA_SEQ_LEN_SHAPE)?;
            }
        }
        Ok(())
    }

    fn validate_sdpa_modifier_sequence_lengths(
        &self,
        modifiers: &SdpaScoreModifiers,
        batch_size: i64,
    ) -> Result<()> {
        let sequence = modifiers.sequence_inputs();
        if modifiers.padding_mask() && (sequence.query.is_none() || sequence.key_value.is_none()) {
            return Err(FrontendError::SdpaRequires {
                operation: "attention mask",
                requirement: SdpaRequirement::SequenceLengths,
            }
            .into());
        }
        self.validate_attention_sequence_inputs(sequence, batch_size)
    }

    fn validate_sdpa_modifier_alibi(
        &self,
        modifiers: &SdpaScoreModifiers,
        attention_heads: i64,
    ) -> Result<()> {
        if !modifiers.has_alibi() {
            return Ok(());
        }

        if let Some(alibi_slopes) = modifiers.alibi_slopes() {
            self.validate_sdpa_tensor_data_type_match(
                alibi_slopes,
                DataType::F32,
                SDPA_ALIBI_DATA_TYPE,
            )?;
            self.validate_sdpa_tensor_dimensions_match(
                alibi_slopes,
                &[1, attention_heads, 1, 1],
                SDPA_ALIBI_SHAPE,
            )?;
        }
        let has_alibi_right_bound = modifiers.right_window() == Some(0);
        if !has_alibi_right_bound {
            return Err(FrontendError::SdpaUnsupported {
                feature: SdpaFeature::UnalignedAlibi,
                operation: "sdpa",
                sm_version: None,
            }
            .into());
        }
        Ok(())
    }

    fn validate_sdpa_modifier_sink_token(
        &self,
        modifiers: &SdpaScoreModifiers,
        attention_heads: i64,
        query_sequence_length: i64,
    ) -> Result<()> {
        let Some(sink_token) = modifiers.sink_token() else {
            return Ok(());
        };

        let sink_tensor = self.tensor_spec(sink_token)?;
        if sink_tensor.shape.rank() != 4 {
            return Err(FrontendError::TensorRankMismatch {
                tensor_id: sink_token,
                operation: SDPA_SINK_TOKEN_RANK,
                expected: RankExpectation::Exact(4),
                actual: sink_tensor.shape.rank(),
            }
            .into());
        }
        let expected_sink_shape = [1, attention_heads, 1, 1];
        self.validate_sdpa_tensor_dimensions_match(
            sink_token,
            &expected_sink_shape,
            SDPA_SINK_TOKEN_SHAPE,
        )?;
        if query_sequence_length == 1 {
            return Err(FrontendError::SdpaSinkTokenRequiresNonDecode {
                query_sequence_length,
            }
            .into());
        }
        Ok(())
    }

    fn validate_attention_dropout_config(
        &self,
        dropout: Option<AttentionDropoutConfig>,
        modifiers: &SdpaScoreModifiers,
    ) -> Result<()> {
        if let Some(dropout) = dropout {
            check_range!(
                SDPA_DROPOUT_PROBABILITY,
                (0.0..1.0).contains(&dropout.probability())
            )?;
            if modifiers.has_custom_dropout() {
                return Err(FrontendError::SdpaDropoutModeConflict.into());
            }
            if modifiers.causal_bottom_right() {
                return Err(FrontendError::SdpaCausalBottomRightModifierConflict {
                    modifier: "dropout",
                }
                .into());
            }
            if let Some(offset) = dropout.offset() {
                self.validate_attention_rng_offset(offset)?;
            }
            if let Some(seed) = dropout.seed_tensor() {
                self.validate_attention_rng_seed(seed)?;
            }
        }
        Ok(())
    }

    fn validate_attention_rng_offset(&self, offset: TensorId) -> Result<()> {
        self.validate_sdpa_rank4_scalar_data_type_supported(
            offset,
            &[DataType::I32, DataType::I64],
            SDPA_RNG_OFFSET_DATA_TYPE,
            SDPA_RNG_OFFSET,
        )
    }

    fn validate_attention_rng_seed(&self, seed: TensorId) -> Result<()> {
        self.validate_sdpa_rank4_scalar_data_type_supported(
            seed,
            &[DataType::I32, DataType::I64],
            SDPA_RNG_SEED_DATA_TYPE,
            SDPA_RNG_SEED,
        )
    }

    fn validate_attention_support_modifier_compatibility(
        modifiers: &SdpaScoreModifiers,
        has_custom_score_subgraph: bool,
        operation: &'static str,
    ) -> Result<()> {
        let has_padding_mask = modifiers.padding_mask();
        let has_causal_like_masking = modifiers.causal_mask() || modifiers.causal_bottom_right();
        if has_custom_score_subgraph
            && (modifiers.has_alibi()
                || has_padding_mask
                || has_causal_like_masking
                || modifiers.left_window().is_some())
        {
            return Err(FrontendError::SdpaScoreSubgraphModifierConflict {
                operation,
                modifier: sdpa_score_subgraph_modifier_conflict(modifiers),
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn validate_attention_config(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: &AttentionConfig,
    ) -> Result<()> {
        if config.fused().aux_outputs().rng_dump() && config.dropout().is_none() {
            return Err(FrontendError::SdpaRequires {
                requirement: SdpaRequirement::PhiloxDropout,
                operation: "sdpa rng dump",
            }
            .into());
        }
        self.validate_sdpa_score_modifiers(q, k, v, config.modifiers())?;
        self.validate_attention_support_surface_for_version(version()?.raw(), q, k, v, config)?;
        if let Some(score_subgraph) = config.score_subgraph() {
            let q_tensor = self.tensor_spec(q)?;
            let kv_shapes = self.attention_kv_shapes(k, v, config)?;
            let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &kv_shapes.key)?;
            self.validate_score_subgraph(score_subgraph, &scores_shape, SDPA_SCORE_SUBGRAPH)?;
        }
        self.validate_attention_block_mask_config(q, k, v, config)?;
        self.validate_attention_paged_config(q, config.paged(), config.modifiers())?;
        self.validate_attention_dropout_config(config.dropout(), config.modifiers())?;
        Ok(())
    }

    fn validate_attention_block_mask_config(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: &AttentionConfig,
    ) -> Result<()> {
        let Some(block_mask) = config.block_mask() else {
            return Ok(());
        };

        if !self.can_lower_attention_as_unified_with_tensors(q, k, v, config)? {
            return Err(FrontendError::SdpaRequires {
                requirement: SdpaRequirement::UnifiedImplementation,
                operation: "sdpa block mask",
            }
            .into());
        }

        self.validate_sdpa_tensor_data_type_match(
            block_mask,
            DataType::U8,
            SDPA_BLOCK_MASK_DATA_TYPE,
        )?;
        let block_mask_tensor = self.tensor_spec(block_mask)?;
        let block_mask_rank = block_mask_tensor.shape.rank();
        if block_mask_rank != 4 {
            return Err(FrontendError::TensorRankMismatch {
                tensor_id: block_mask,
                operation: SDPA_BLOCK_MASK_LAYOUT,
                expected: RankExpectation::Exact(4),
                actual: block_mask_rank,
            }
            .into());
        }
        let last_stride = block_mask_tensor.shape.strides()[3];
        if last_stride != 1 {
            return Err(FrontendError::TensorLastStrideMismatch {
                tensor_id: block_mask,
                operation: SDPA_BLOCK_MASK_LAYOUT,
                expected: 1,
                actual: last_stride,
            }
            .into());
        }
        Ok(())
    }

    fn validate_attention_paged_config(
        &self,
        query: TensorId,
        paged: AttentionPagedKvCache,
        modifiers: &SdpaScoreModifiers,
    ) -> Result<()> {
        let has_seq_lens = modifiers.sequence_inputs().key_value.is_some();
        if paged.is_enabled() && !has_seq_lens {
            return Err(FrontendError::SdpaRequires {
                requirement: SdpaRequirement::SequenceLengths,
                operation: "sdpa paged attention",
            }
            .into());
        }
        if let Some(max_sequence_length_key_value) = paged.max_sequence_length_key_value() {
            check_range!(
                SDPA_MAX_SEQUENCE_LENGTH_KEY_VALUE,
                max_sequence_length_key_value > 0
            )?;
            let quantized = matches!(
                self.tensor_spec(query)?.data_type,
                DataType::F8E4M3 | DataType::F8E5M2
            );
            if !paged.is_enabled() && !quantized {
                return Err(FrontendError::SdpaRequires {
                    requirement: SdpaRequirement::PagedAttention,
                    operation: "sdpa max sequence length key/value",
                }
                .into());
            }
            if let Some(bias) = modifiers.bias() {
                let bias_tensor = self.tensor_spec(bias)?.clone();
                if bias_tensor.shape.rank() == 4
                    && bias_tensor.shape.dimensions()[3] != max_sequence_length_key_value
                {
                    return Err(FrontendError::SdpaMaxSequenceLengthBiasMismatch {
                        max_sequence_length_key_value,
                        bias_key_value_dimension: bias_tensor.shape.dimensions()[3],
                    }
                    .into());
                }
            }
        }
        if let Some(cache) = paged.k() {
            self.validate_sdpa_tensor_data_type_match(
                cache.page_table(),
                DataType::I32,
                SDPA_PAGED_K_PAGE_TABLE_DATA_TYPE,
            )?;
        }
        if let Some(cache) = paged.v() {
            self.validate_sdpa_tensor_data_type_match(
                cache.page_table(),
                DataType::I32,
                SDPA_PAGED_V_PAGE_TABLE_DATA_TYPE,
            )?;
        }
        Ok(())
    }

    pub(crate) fn validate_score_subgraph(
        &self,
        score_subgraph: &SdpaScoreSubgraph,
        scores_shape: &Shape,
        error_name: &'static str,
    ) -> Result<()> {
        let graph = score_subgraph.graph();
        let input = score_subgraph.input();
        let output = score_subgraph.output();
        graph.validate_sdpa_tensor_data_type_match(
            input,
            self.sdpa_score_data_type(),
            error_name,
        )?;
        graph.validate_sdpa_tensor_dimensions_match(
            input,
            scores_shape.dimensions(),
            error_name,
        )?;
        let input_shape = &graph.tensor_spec(input)?.shape;
        if input_shape.strides() != scores_shape.strides() {
            return Err(FrontendError::TensorStridesMismatch {
                tensor_id: input,
                operation: error_name,
                expected: scores_shape.strides().to_vec(),
                actual: input_shape.strides().to_vec(),
            }
            .into());
        }
        graph.validate_sdpa_tensor_dimensions_match(output, scores_shape.dimensions(), error_name)
    }

    fn validate_sdpa_tensor_shape(
        &self,
        tensor: TensorId,
        expected_shape: &Shape,
        error_name: &'static str,
    ) -> Result<()> {
        self.validate_sdpa_tensor_dimensions_match(
            tensor,
            expected_shape.dimensions(),
            error_name,
        )?;
        Ok(())
    }

    pub(crate) fn validate_sdpa_tensor_dimensions_match(
        &self,
        tensor: TensorId,
        expected_dimensions: &[i64],
        error_name: &'static str,
    ) -> Result<()> {
        let actual = self.tensor_spec(tensor)?.shape.dimensions();
        if actual != expected_dimensions {
            return Err(FrontendError::TensorDimensionsMismatch {
                tensor_id: tensor,
                operation: error_name,
                expected: expected_dimensions.to_vec(),
                actual: actual.to_vec(),
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn validate_sdpa_tensor_dimensions<const N: usize>(
        &self,
        tensors: [(TensorId, &[i64], &'static str); N],
    ) -> Result<()> {
        for (tensor, expected_dimensions, error_name) in tensors {
            self.validate_sdpa_tensor_dimensions_match(tensor, expected_dimensions, error_name)?;
        }
        Ok(())
    }

    pub(crate) fn validate_sdpa_tensor_dimensions_and_data_type(
        &self,
        tensor: TensorId,
        expected_dimensions: &[i64],
        expected_data_type: DataType,
        error_name: &'static str,
    ) -> Result<()> {
        self.validate_sdpa_tensor_dimensions_match(tensor, expected_dimensions, error_name)?;
        self.validate_sdpa_tensor_data_type_match(tensor, expected_data_type, error_name)
    }

    pub(crate) fn validate_sdpa_tensor_data_type_match(
        &self,
        tensor: TensorId,
        expected_data_type: DataType,
        error_name: &'static str,
    ) -> Result<()> {
        let actual = self.tensor_spec(tensor)?.data_type;
        if actual != expected_data_type {
            return Err(FrontendError::TensorDataTypeMismatch {
                tensor_id: tensor,
                operation: error_name,
                expected: expected_data_type,
                actual,
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn validate_sdpa_tensor_data_types<const N: usize>(
        &self,
        tensors: [(TensorId, DataType, &'static str); N],
    ) -> Result<()> {
        for (tensor, expected_data_type, error_name) in tensors {
            self.validate_sdpa_tensor_data_type_match(tensor, expected_data_type, error_name)?;
        }
        Ok(())
    }

    pub(crate) fn validate_attention_support_surface_for_version(
        &self,
        cudnn_version: u64,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: &AttentionConfig,
    ) -> Result<()> {
        let support_context = support::SdpaSupportContext::new(cudnn_version, self.sm_version());
        let q_tensor = self.tensor_spec(q)?;
        let kv_shapes = self.attention_kv_shapes(k, v, config)?;
        let q_dims = q_tensor.shape.dimensions();
        let k_dims = kv_shapes.key.dimensions();
        let v_dims = kv_shapes.value.dimensions();
        let has_padding_mask = config.modifiers().padding_mask()
            || (config.implementation() != AttentionImplementation::Composite
                && config.modifiers().has_sequence_lengths());
        let has_custom_score_subgraph = config.score_subgraph().is_some();
        let is_ragged = self.has_any_ragged_offset([q, k, v])?;

        if !attention_heads_compatible(q_dims[1], k_dims[1])
            || !attention_heads_compatible(q_dims[1], v_dims[1])
        {
            return Err(FrontendError::SdpaGroupedQueryAttentionHeadsMismatch {
                query_heads: q_dims[1],
                key_heads: k_dims[1],
                value_heads: v_dims[1],
                operation: "sdpa",
            }
            .into());
        }
        Self::validate_attention_support_modifier_compatibility(
            config.modifiers(),
            has_custom_score_subgraph,
            "sdpa",
        )?;
        support_context.require_forward_architecture(q_tensor.data_type, q_dims[3], v_dims[3])?;

        support::require_sdpa_forward_mask_lengths(
            q_dims[2],
            k_dims[2],
            has_padding_mask,
            config.modifiers().left_window().is_some(),
            config.modifiers().causal_bottom_right(),
        )?;

        support::require_sdpa_ragged_mask_or_subgraph(
            is_ragged,
            has_padding_mask,
            config.modifiers().additive_mask().is_some(),
            config.score_subgraph().is_some(),
            "sdpa",
        )?;

        Ok(())
    }

    pub(crate) fn validate_attention_backward_config(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: &AttentionBackwardConfig,
    ) -> Result<()> {
        if config.modifiers().sequence_inputs().has_cumulative() {
            return Err(FrontendError::SdpaUnsupported {
                operation: "sdpa backward",
                feature: SdpaFeature::CumulativeSequenceLengths,
                sm_version: None,
            }
            .into());
        }
        if config.softmax().stats_base() != SoftmaxStatsBase::NaturalLog {
            return Err(FrontendError::SdpaBackwardStatsBaseUnsupported {
                actual: config.softmax().stats_base(),
            }
            .into());
        }
        self.validate_sdpa_score_modifiers(q, k, v, config.modifiers())?;
        if config.score_subgraph_bprop().is_some() && config.score_subgraph().is_none() {
            return Err(FrontendError::SdpaRequires {
                requirement: SdpaRequirement::ScoreSubgraph,
                operation: "sdpa backward score subgraph bprop",
            }
            .into());
        }
        let q_tensor = self.tensor_spec(q)?;
        let k_tensor = self.tensor_spec(k)?;
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        if let Some(score_subgraph) = config.score_subgraph() {
            self.validate_score_subgraph(
                score_subgraph,
                &scores_shape,
                SDPA_BACKWARD_SCORE_SUBGRAPH,
            )?;
        }
        if let Some(score_subgraph_bprop) = config.score_subgraph_bprop() {
            self.validate_score_subgraph(
                score_subgraph_bprop,
                &scores_shape,
                SDPA_BACKWARD_SCORE_SUBGRAPH_BPROP,
            )?;
        }
        self.validate_attention_dropout_config(config.dropout(), config.modifiers())?;
        if let Some(inverse) = config.dropout_scale_inverse() {
            self.validate_tensor_scalar_shape(inverse, "attention inverse dropout scale")?;
            self.validate_tensor_data_type_supported(
                inverse,
                &[DataType::F32, DataType::F16, DataType::BF16],
                "attention inverse dropout scale",
            )?;
        }
        let aux_gradients = config.aux_gradients();
        if aux_gradients.random_number_generator_dump() && config.dropout().is_none() {
            return Err(FrontendError::SdpaRequires {
                requirement: SdpaRequirement::Dropout,
                operation: "sdpa backward rng dump",
            }
            .into());
        }

        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the complete attention tensor contract is validated together"
    )]
    pub(crate) fn validate_sdpa_backward_tensors(
        &self,
        inputs: &SdpaBackwardInputs,
        gradients: &SdpaBackwardGradientTensors,
        config: &AttentionBackwardConfig,
    ) -> Result<SdpaBackwardValidation> {
        let q = inputs.query;
        let k = inputs.key;
        let v = inputs.value;
        let o = inputs.output;
        let d_o = inputs.output_gradient;
        let stats = inputs.stats;
        let scale = inputs.scale;
        let d_q = gradients.query_gradient;
        let d_k = gradients.key_gradient;
        let d_v = gradients.value_gradient;

        let q_tensor = self.tensor_spec(q)?.clone();
        let k_tensor = self.tensor_spec(k)?.clone();
        let v_tensor = self.tensor_spec(v)?.clone();
        let o_tensor = self.tensor_spec(o)?.clone();
        let stats_tensor = self.tensor_spec(stats)?.clone();

        self.validate_sdpa_tensor_layout(q, SDPA_Q_LAYOUT)?;
        self.validate_sdpa_k_layout(k)?;
        self.validate_sdpa_k_layout(v)?;
        self.validate_sdpa_tensor_layouts([
            (o, SDPA_OUTPUT_LAYOUT),
            (d_o, SDPA_BACKWARD_D_O_LAYOUT),
            (d_q, SDPA_BACKWARD_D_Q_LAYOUT),
            (d_k, SDPA_BACKWARD_D_K_LAYOUT),
            (d_v, SDPA_BACKWARD_D_V_LAYOUT),
        ])?;

        let stats_rank = stats_tensor.shape.rank();
        if stats_rank != 4 {
            return Err(FrontendError::TensorRankMismatch {
                tensor_id: stats,
                operation: SDPA_BACKWARD_RANK,
                expected: RankExpectation::Exact(4),
                actual: stats_rank,
            }
            .into());
        }
        self.validate_tensor_scalar_shape(scale, SDPA_BACKWARD_SCALE)?;
        self.validate_sdpa_tensor_dimensions_match(
            d_o,
            o_tensor.shape.dimensions(),
            SDPA_BACKWARD_D_O_SHAPE,
        )?;

        let q_dims = q_tensor.shape.dimensions();
        let k_dims = k_tensor.shape.dimensions();
        let v_dims = v_tensor.shape.dimensions();
        if !attention_heads_compatible(q_dims[1], k_dims[1])
            || !attention_heads_compatible(q_dims[1], v_dims[1])
        {
            return Err(FrontendError::SdpaGroupedQueryAttentionHeadsMismatch {
                query_heads: q_dims[1],
                key_heads: k_dims[1],
                value_heads: v_dims[1],
                operation: "sdpa backward",
            }
            .into());
        }
        if q_dims[3] != k_dims[3] {
            return Err(FrontendError::SdpaQueryKeyShapeMismatch {
                query: q_dims.to_vec(),
                key: k_dims.to_vec(),
            }
            .into());
        }
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        let expected_stats_shape = reduce_last_axis(&scores_shape)?;

        self.validate_sdpa_tensor_shape(stats, &expected_stats_shape, SDPA_BACKWARD_STATS_SHAPE)?;
        self.validate_sdpa_tensor_shape(d_q, &q_tensor.shape, SDPA_BACKWARD_D_Q_SHAPE)?;

        let expected_output = infer_sdpa_output_shape(&scores_shape, &v_tensor.shape)?;
        self.validate_sdpa_tensor_dimensions_match(
            o,
            expected_output.dimensions(),
            SDPA_BACKWARD_D_O_SHAPE,
        )?;
        self.validate_sdpa_tensor_shape(d_k, &k_tensor.shape, SDPA_BACKWARD_D_K_SHAPE)?;
        self.validate_sdpa_tensor_shape(d_v, &v_tensor.shape, SDPA_BACKWARD_D_V_SHAPE)?;

        let expected_d_sink_token_shape = Shape::contiguous([1, q_dims[1], 1, 1])?;
        if let Some(bias_gradient) = gradients.bias_gradient {
            self.validate_tensor_rank(bias_gradient, 4, SDPA_BACKWARD_D_BIAS_SHAPE)?;
            let shape = &self.tensor_spec(bias_gradient)?.shape;
            if shape
                .dimensions()
                .iter()
                .zip(scores_shape.dimensions())
                .any(|(&actual, &score)| actual != 1 && actual != score)
            {
                return Err(FrontendError::TensorLayoutMismatch {
                    tensor_id: bias_gradient,
                    operation: SDPA_BACKWARD_D_BIAS_SHAPE,
                    reason: "expected dimensions reducible from the attention scores".into(),
                }
                .into());
            }
        }
        if let Some(gradient) = gradients.sink_token_gradient {
            self.validate_sdpa_tensor_dimensions_match(
                gradient,
                expected_d_sink_token_shape.dimensions(),
                SDPA_BACKWARD_D_SINK_TOKEN_SHAPE,
            )?;
        }
        if gradients.rng_dump.is_some() && config.dropout().is_none() {
            return Err(FrontendError::SdpaRequires {
                requirement: SdpaRequirement::Dropout,
                operation: "sdpa backward rng dump",
            }
            .into());
        }

        Ok(SdpaBackwardValidation {
            scores_shape,
            expected_stats_shape,
        })
    }

    pub(crate) fn validate_attention_backward_support_surface_for_version(
        &self,
        cudnn_version: u64,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        o: TensorId,
        stats: TensorId,
        d_o: TensorId,
        d_q: TensorId,
        d_k: TensorId,
        d_v: TensorId,
        config: &AttentionBackwardConfig,
    ) -> Result<()> {
        let support_context = support::SdpaSupportContext::new(cudnn_version, self.sm_version());
        let q_dims = self.tensor_spec(q)?.shape.dimensions().to_vec();
        let k_dims = self.tensor_spec(k)?.shape.dimensions().to_vec();
        let has_custom_score_subgraph = config.score_subgraph().is_some();
        let is_ragged = self.has_any_ragged_offset([q, k, v, o, stats, d_o, d_q, d_k, d_v])?;
        Self::validate_attention_support_modifier_compatibility(
            config.modifiers(),
            has_custom_score_subgraph,
            "sdpa backward",
        )?;
        let support = support_context.backward();
        let has_dropout = config.dropout().is_some() || config.modifiers().has_custom_dropout();
        let aux_gradients = config.aux_gradients();
        support.require_causal_bottom_right(support::SdpaBackwardCausalBottomRightSupport {
            enabled: config.modifiers().causal_bottom_right(),
            query_sequence_length: q_dims[2],
            key_sequence_length: k_dims[2],
            has_dropout: config.dropout().is_some(),
        })?;
        support.require_bias_gradient(support::SdpaBackwardBiasGradientSupport {
            enabled: aux_gradients.bias(),
            is_ragged,
        })?;
        support.require_deterministic_algorithm(support::SdpaBackwardDeterministicSupport {
            enabled: config.deterministic_algorithm(),
            has_bias_gradient: aux_gradients.bias(),
            has_dropout,
            has_alibi: config.modifiers().has_alibi(),
            is_ragged,
        })?;

        Ok(())
    }

    pub(crate) fn validate_mxfp8_backward_support(
        &self,
        output_io_type: DataType,
        is_ragged: bool,
        config: &AttentionBackwardConfig,
    ) -> Result<()> {
        let aux_gradients = config.aux_gradients();
        support::QuantizedSdpaSupport::new(self.sm_version()).require_mxfp8_backward(
            support::MxFp8BackwardSupport {
                output_io_type,
                is_ragged,
                has_generated_dropout: config.dropout().is_some(),
                has_random_number_generator_dump: aux_gradients.random_number_generator_dump(),
            },
        )
    }

    pub(crate) fn validate_fp8_backward_support(
        &self,
        output_io_type: DataType,
        d_qk: i64,
        d_v: i64,
        is_ragged: bool,
        config: &AttentionBackwardConfig,
    ) -> Result<()> {
        let aux_gradients = config.aux_gradients();
        support::QuantizedSdpaSupport::new(self.sm_version()).require_fp8_backward(
            support::Fp8BackwardSupport {
                output_io_type,
                d_qk,
                d_v,
                deterministic_algorithm: config.deterministic_algorithm(),
                is_ragged,
                has_generated_dropout: config.dropout().is_some(),
                has_random_number_generator_dump: aux_gradients.random_number_generator_dump(),
            },
        )
    }

    pub(crate) fn validate_fp8_forward_support(&self, output_io_type: DataType) -> Result<()> {
        support::QuantizedSdpaSupport::new(self.sm_version())
            .require_fp8_forward_output_type(output_io_type)
    }

    pub(crate) fn validate_quantized_backward_head_dimensions(
        &self,
        d_qk: i64,
        d_v: i64,
        operation: &'static str,
    ) -> Result<()> {
        support::QuantizedSdpaSupport::new(self.sm_version())
            .require_quantized_backward_head_dimensions(d_qk, d_v, operation)
    }

    pub(crate) fn validate_quantized_backward_output_ragged_support(
        &self,
        d_q: TensorId,
        d_k: TensorId,
        d_v: TensorId,
        operation: &'static str,
    ) -> Result<()> {
        let has_ragged_output = self.has_any_ragged_offset([d_q, d_k, d_v])?;
        support::QuantizedSdpaSupport::new(self.sm_version())
            .require_quantized_backward_output_ragged_support(has_ragged_output, operation)
    }
}

fn sdpa_causal_bottom_right_modifier_conflict(modifiers: &SdpaScoreModifiers) -> Error {
    let modifier = if modifiers.bias().is_some() {
        "bias"
    } else if modifiers.dropout_mask().is_some() || modifiers.dropout_scale().is_some() {
        "custom dropout"
    } else if modifiers.has_alibi() {
        "alibi slopes"
    } else {
        "score modifier"
    };
    FrontendError::SdpaCausalBottomRightModifierConflict { modifier }.into()
}

fn sdpa_score_subgraph_modifier_conflict(modifiers: &SdpaScoreModifiers) -> &'static str {
    if modifiers.has_alibi() {
        "alibi slopes"
    } else if modifiers.padding_mask() {
        "padding mask"
    } else if modifiers.causal_bottom_right() {
        "causal bottom-right mask"
    } else if modifiers.causal_mask() {
        "causal top-left mask"
    } else if modifiers.left_window().is_some() {
        "sliding window"
    } else {
        "score modifier"
    }
}
