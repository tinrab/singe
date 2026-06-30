use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::{
            SdpaBackwardGradientTensors, SdpaBackwardInputs,
            support::{
                SDPA_ALIBI_DATA_TYPE, SDPA_ALIBI_SHAPE, SDPA_BACKWARD_D_BIAS_SHAPE,
                SDPA_BACKWARD_D_K_LAYOUT, SDPA_BACKWARD_D_K_SHAPE, SDPA_BACKWARD_D_O_LAYOUT,
                SDPA_BACKWARD_D_O_SHAPE, SDPA_BACKWARD_D_Q_LAYOUT, SDPA_BACKWARD_D_Q_SHAPE,
                SDPA_BACKWARD_D_SINK_TOKEN_SHAPE, SDPA_BACKWARD_D_V_LAYOUT,
                SDPA_BACKWARD_D_V_SHAPE, SDPA_BACKWARD_DATA_TYPE, SDPA_BACKWARD_RANK,
                SDPA_BACKWARD_SCORE_SUBGRAPH, SDPA_BACKWARD_SCORE_SUBGRAPH_BPROP,
                SDPA_BACKWARD_STATS_DATA_TYPE, SDPA_BACKWARD_STATS_SHAPE,
                SDPA_BLOCK_MASK_DATA_TYPE, SDPA_BLOCK_MASK_LAYOUT, SDPA_DROPOUT_PROBABILITY,
                SDPA_DROPOUT_SCALE, SDPA_K_LAYOUT, SDPA_MAX_SEQUENCE_LENGTH_KEY_VALUE,
                SDPA_MXFP8_BACKWARD_SCALE_D_O_SHAPE, SDPA_MXFP8_BACKWARD_SCALE_D_O_T_SHAPE,
                SDPA_MXFP8_BACKWARD_SCALE_K_SHAPE, SDPA_MXFP8_BACKWARD_SCALE_K_T_SHAPE,
                SDPA_MXFP8_BACKWARD_SCALE_Q_SHAPE, SDPA_MXFP8_BACKWARD_SCALE_Q_T_SHAPE,
                SDPA_MXFP8_BACKWARD_SCALE_V_SHAPE, SDPA_OUTPUT_LAYOUT,
                SDPA_PAGED_K_PAGE_TABLE_DATA_TYPE, SDPA_PAGED_V_PAGE_TABLE_DATA_TYPE,
                SDPA_Q_LAYOUT, SDPA_RANK, SDPA_RNG_OFFSET, SDPA_RNG_OFFSET_DATA_TYPE,
                SDPA_RNG_SEED, SDPA_RNG_SEED_DATA_TYPE, SDPA_SCORE_SUBGRAPH,
                SDPA_SEQ_LEN_DATA_TYPE, SDPA_SEQ_LEN_SHAPE, SDPA_SINK_TOKEN_DATA_TYPE,
                SDPA_SINK_TOKEN_RANK, SDPA_SINK_TOKEN_SHAPE,
            },
        },
        graph::Graph,
        infer::*,
        operation::*,
        support::{self, SDPA_BACKWARD_SCALE, SDPA_FP8_BACKWARD_OUTPUT_DATA_TYPE},
    },
    tensor::{Shape, TensorId, TensorSpec},
    utility::check_range,
    version,
};

pub(in crate::frontend::composite::sdpa) struct SdpaBackwardValidation {
    pub scores_shape: Shape,
    pub expected_stats_shape: Shape,
    pub k_is_transposed: bool,
    pub v_is_transposed: bool,
}

impl Graph {
    pub(crate) fn validate_sdpa_scalar_element_count(
        &self,
        tensor: TensorId,
        error_name: &str,
    ) -> Result<()> {
        let actual = self.tensor_config(tensor)?.shape.element_count()?;
        if actual != 1 {
            return Err(Error::FrontendTensorElementCountMismatch {
                tensor_id: tensor,
                operation: error_name.into(),
                expected: 1,
                actual,
            });
        }
        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_rank4_scalar_data_type_supported(
        &self,
        tensor_id: TensorId,
        supported_data_types: &[DataType],
        data_type_error_name: &str,
        shape_error_name: &str,
    ) -> Result<()> {
        let tensor = self.tensor_config(tensor_id)?;
        if !supported_data_types.contains(&tensor.data_type) {
            return Err(Error::FrontendTensorDataTypeUnsupported {
                tensor_id,
                operation: data_type_error_name.into(),
                actual: tensor.data_type,
                supported: supported_data_types.to_vec(),
            });
        }
        if tensor.shape.dimensions() != [1, 1, 1, 1] {
            return Err(Error::FrontendTensorDimensionsMismatch {
                tensor_id,
                operation: shape_error_name.into(),
                expected: vec![1, 1, 1, 1],
                actual: tensor.shape.dimensions().to_vec(),
            });
        }
        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn has_any_ragged_offset<const N: usize>(
        &self,
        tensors: [TensorId; N],
    ) -> Result<bool> {
        for tensor in tensors {
            if self.tensor_config(tensor)?.ragged_offset.is_some() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_sequence_length_data_types(
        &self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
        error_name: &str,
    ) -> Result<()> {
        self.validate_sdpa_tensor_data_types([
            (sequence_length_query, DataType::I32, error_name),
            (sequence_length_key_value, DataType::I32, error_name),
        ])
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_tensor_layout(
        &self,
        tensor: TensorId,
        name: &str,
    ) -> Result<()> {
        let tensor_config = self.tensor_config(tensor)?;
        let actual_rank = tensor_config.shape.dimensions().len();
        if actual_rank != 4 {
            return Err(Error::FrontendTensorRankMismatch {
                tensor_id: tensor,
                operation: SDPA_RANK.into(),
                expected: "4".into(),
                actual: actual_rank,
            });
        }
        let actual = tensor_config.shape.strides()[3];
        if actual != 1 {
            return Err(Error::FrontendTensorLastStrideMismatch {
                tensor_id: tensor,
                operation: name.into(),
                expected: 1,
                actual,
            });
        }

        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_tensor_layouts<const N: usize>(
        &self,
        tensors: [(TensorId, &str); N],
    ) -> Result<()> {
        for (tensor, name) in tensors {
            self.validate_sdpa_tensor_layout(tensor, name)?;
        }
        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_k_layout(
        &self,
        tensor: TensorId,
    ) -> Result<()> {
        let tensor_config = self.tensor_config(tensor)?;
        let actual_rank = tensor_config.shape.dimensions().len();
        if actual_rank != 4 {
            return Err(Error::FrontendTensorRankMismatch {
                tensor_id: tensor,
                operation: SDPA_RANK.into(),
                expected: "4".into(),
                actual: actual_rank,
            });
        }
        if tensor_config.shape.strides()[2] != 1 && tensor_config.shape.strides()[3] != 1 {
            return Err(Error::FrontendTensorLayoutMismatch {
                tensor_id: tensor,
                operation: SDPA_K_LAYOUT.into(),
                reason: "expected either the last or second-last stride to be 1".into(),
            });
        }

        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_direct_ragged_support_for_version(
        &self,
        cudnn_version: u64,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        o: TensorId,
        direct_config: &DirectSdpaForwardConfig,
        has_custom_score_subgraph: bool,
    ) -> Result<()> {
        let support_context = support::SdpaSupportContext::new(cudnn_version, self.sm_version());
        let is_ragged = self.has_any_ragged_offset([q, k, v, o])?;
        let has_padding_mask = direct_config.sequence_mode().is_padding_mask();
        support::require_sdpa_ragged_mask_or_subgraph(
            is_ragged,
            has_padding_mask,
            false,
            has_custom_score_subgraph,
            "sdpa",
        )?;
        support_context.require_ragged_offsets_arch(is_ragged)?;

        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_score_modifiers(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        modifiers: &SdpaScoreModifiers,
    ) -> Result<()> {
        self.validate_sdpa_tensor_layout(q, SDPA_Q_LAYOUT)?;
        self.validate_sdpa_k_layout(k)?;
        self.validate_sdpa_k_layout(v)?;

        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();
        let q_dims = q_tensor.shape.dimensions();
        let k_dims = k_tensor.shape.dimensions();
        let v_dims = v_tensor.shape.dimensions();

        if q_dims[1] % k_dims[1] != 0 || q_dims[1] % v_dims[1] != 0 {
            return Err(Error::FrontendSdpaGroupedQueryAttentionHeadsMismatch {
                query_heads: q_dims[1],
                key_heads: k_dims[1],
                value_heads: v_dims[1],
            });
        }

        self.validate_sdpa_modifier_sequence_lengths(modifiers, q_dims[0])?;
        self.validate_sdpa_modifier_custom_dropout(modifiers)?;

        if let Some(bias) = modifiers.bias()
            && self.tensor_config(bias)?.data_type == DataType::Boolean
        {
            return Err(Error::FrontendSdpaBiasDataTypeUnsupported);
        }

        self.validate_sdpa_modifier_alibi(modifiers, q_dims[1])?;
        self.validate_sdpa_modifier_sink_token(modifiers, q_dims[1], q_dims[2])?;

        if modifiers.causal_bottom_right()
            && modifiers.has_causal_bottom_right_incompatible_modifier()
        {
            return Err(sdpa_causal_bottom_right_modifier_conflict(modifiers));
        }
        if let Some((left_window, right_window)) = modifiers.sliding_window() {
            check_range!("left_window", left_window >= 0)?;
            check_range!("right_window", right_window >= 0)?;
        }

        Ok(())
    }

    fn validate_sdpa_modifier_sequence_lengths(
        &self,
        modifiers: &SdpaScoreModifiers,
        batch_size: i64,
    ) -> Result<()> {
        let Some((sequence_length_query, sequence_length_key_value)) = modifiers.sequence_lengths()
        else {
            if modifiers.causal_bottom_right() {
                return Err(Error::FrontendSdpaCausalBottomRightRequiresSequenceLengths);
            }
            return Ok(());
        };

        self.validate_sdpa_sequence_length_data_types(
            sequence_length_query,
            sequence_length_key_value,
            SDPA_SEQ_LEN_DATA_TYPE,
        )?;

        let expected_seq_shape = [batch_size, 1, 1, 1];
        self.validate_sdpa_tensor_dimensions([
            (
                sequence_length_query,
                &expected_seq_shape,
                SDPA_SEQ_LEN_SHAPE,
            ),
            (
                sequence_length_key_value,
                &expected_seq_shape,
                SDPA_SEQ_LEN_SHAPE,
            ),
        ])?;
        Ok(())
    }

    fn validate_sdpa_modifier_custom_dropout(&self, modifiers: &SdpaScoreModifiers) -> Result<()> {
        match (modifiers.dropout_mask(), modifiers.dropout_scale()) {
            (Some(_), Some(dropout_scale)) => {
                self.validate_sdpa_scalar_element_count(dropout_scale, SDPA_DROPOUT_SCALE)?;
            }
            (None, None) => {}
            _ => {
                return Err(Error::FrontendSdpaCustomDropoutIncomplete);
            }
        }
        Ok(())
    }

    fn validate_sdpa_modifier_alibi(
        &self,
        modifiers: &SdpaScoreModifiers,
        attention_heads: i64,
    ) -> Result<()> {
        let Some(alibi_slopes) = modifiers.alibi_slopes() else {
            return Ok(());
        };

        let expected_alibi_shape = [1, attention_heads, 1, 1];
        self.validate_sdpa_tensor_data_type_match(
            alibi_slopes,
            DataType::F32,
            SDPA_ALIBI_DATA_TYPE,
        )?;
        self.validate_sdpa_tensor_dimensions_match(
            alibi_slopes,
            &expected_alibi_shape,
            SDPA_ALIBI_SHAPE,
        )?;
        let has_alibi_right_bound = modifiers.causal_mask()
            || modifiers
                .sliding_window()
                .is_some_and(|(_, right)| right == 0);
        if !has_alibi_right_bound {
            return Err(Error::FrontendSdpaAlibiAlignmentUnsupported);
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

        let sink_tensor = self.tensor_config(sink_token)?;
        self.validate_sdpa_tensor_data_type_match(
            sink_token,
            DataType::F32,
            SDPA_SINK_TOKEN_DATA_TYPE,
        )?;
        if sink_tensor.shape.dimensions().len() != 4 {
            return Err(Error::FrontendTensorRankMismatch {
                tensor_id: sink_token,
                operation: SDPA_SINK_TOKEN_RANK.into(),
                expected: "4".into(),
                actual: sink_tensor.shape.dimensions().len(),
            });
        }
        let expected_sink_shape = [1, attention_heads, 1, 1];
        self.validate_sdpa_tensor_dimensions_match(
            sink_token,
            &expected_sink_shape,
            SDPA_SINK_TOKEN_SHAPE,
        )?;
        if query_sequence_length == 1 {
            return Err(Error::FrontendSdpaSinkTokenRequiresNonDecode {
                query_sequence_length,
            });
        }
        Ok(())
    }

    fn validate_attention_dropout_config(
        &self,
        dropout: Option<AttentionDropoutConfig>,
        dropout_offset: Option<TensorId>,
        modifiers: &SdpaScoreModifiers,
    ) -> Result<()> {
        if let Some(dropout) = dropout {
            check_range!(
                SDPA_DROPOUT_PROBABILITY,
                (0.0..1.0).contains(&dropout.probability())
            )?;
            if dropout_offset.is_some() && modifiers.dropout_mask().is_some() {
                return Err(Error::FrontendSdpaDropoutModeConflict);
            }
            if modifiers.has_custom_dropout() {
                return Err(Error::FrontendSdpaDropoutModeConflict);
            }
            if modifiers.causal_bottom_right() {
                return Err(Error::FrontendSdpaCausalBottomRightModifierConflict {
                    modifier: String::from("dropout"),
                });
            }
            if let Some(offset) = dropout_offset {
                self.validate_attention_rng_offset(offset)?;
            }
            if let Some(seed) = dropout.seed_tensor() {
                self.validate_attention_rng_seed(seed)?;
                if dropout_offset.is_none() {
                    return Err(Error::FrontendSdpaDropoutSeedRequiresOffset);
                }
            }
        } else if dropout_offset.is_some() {
            return Err(Error::FrontendSdpaDropoutOffsetRequiresDropout);
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
            &[DataType::I64],
            SDPA_RNG_SEED_DATA_TYPE,
            SDPA_RNG_SEED,
        )
    }

    fn validate_attention_support_modifier_compatibility(
        modifiers: &SdpaScoreModifiers,
        has_custom_score_subgraph: bool,
        operation: &str,
    ) -> Result<()> {
        let has_padding_mask = modifiers.padding_mask();
        let has_causal_like_masking = modifiers.causal_mask() || modifiers.causal_bottom_right();
        if has_custom_score_subgraph
            && (modifiers.alibi_slopes().is_some()
                || has_padding_mask
                || has_causal_like_masking
                || modifiers.sliding_window().is_some())
        {
            return Err(Error::FrontendSdpaScoreSubgraphModifierConflict {
                operation: operation.into(),
                modifier: sdpa_score_subgraph_modifier_conflict(modifiers).into(),
            });
        }
        support::require_sdpa_sequence_lengths_mask_or_subgraph(
            modifiers.sequence_lengths().is_some(),
            has_padding_mask,
            modifiers.causal_bottom_right(),
            has_custom_score_subgraph,
            operation,
        )?;
        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_attention_config(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: AttentionConfig,
    ) -> Result<()> {
        self.validate_sdpa_score_modifiers(q, k, v, config.modifiers())?;
        self.validate_attention_support_surface_for_version(
            version()?.raw(),
            q,
            k,
            v,
            config.clone(),
        )?;
        if let Some(score_subgraph) = config.score_subgraph() {
            let q_tensor = self.tensor_config(q)?.clone();
            let k_tensor = self.tensor_config(k)?.clone();
            let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
            self.validate_score_subgraph(score_subgraph, &scores_shape, SDPA_SCORE_SUBGRAPH)?;
        }
        self.validate_attention_block_mask_config(q, k, v, &config)?;
        self.validate_attention_paged_config(config.paged(), config.modifiers())?;
        self.validate_attention_dropout_config(
            config.dropout(),
            config.dropout_offset(),
            config.modifiers(),
        )?;
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

        if !self.can_lower_attention_as_unified_with_tensors_for_version(
            version()?.raw(),
            q,
            k,
            v,
            config.clone(),
        )? {
            return Err(Error::FrontendSdpaBlockMaskRequiresUnified);
        }

        self.validate_sdpa_tensor_data_type_match(
            block_mask,
            DataType::U8,
            SDPA_BLOCK_MASK_DATA_TYPE,
        )?;
        let block_mask_tensor = self.tensor_config(block_mask)?;
        let block_mask_rank = block_mask_tensor.shape.dimensions().len();
        if block_mask_rank != 4 {
            return Err(Error::FrontendTensorRankMismatch {
                tensor_id: block_mask,
                operation: SDPA_BLOCK_MASK_LAYOUT.into(),
                expected: "4".into(),
                actual: block_mask_rank,
            });
        }
        let last_stride = block_mask_tensor.shape.strides()[3];
        if last_stride != 1 {
            return Err(Error::FrontendTensorLastStrideMismatch {
                tensor_id: block_mask,
                operation: SDPA_BLOCK_MASK_LAYOUT.into(),
                expected: 1,
                actual: last_stride,
            });
        }
        Ok(())
    }

    fn validate_attention_paged_config(
        &self,
        paged: AttentionPagedKvCache,
        modifiers: &SdpaScoreModifiers,
    ) -> Result<()> {
        let has_seq_lens = modifiers.sequence_lengths().is_some();
        if paged.is_enabled() && !has_seq_lens {
            return Err(Error::FrontendSdpaPagedAttentionRequiresSequenceLengths);
        }
        if let Some(max_sequence_length_key_value) = paged.max_sequence_length_key_value() {
            check_range!(
                SDPA_MAX_SEQUENCE_LENGTH_KEY_VALUE,
                max_sequence_length_key_value > 0
            )?;
            if !paged.is_enabled() {
                return Err(Error::FrontendSdpaMaxSequenceLengthRequiresPagedAttention);
            }
            if let Some(bias) = modifiers.bias() {
                let bias_tensor = self.tensor_config(bias)?.clone();
                if bias_tensor.shape.dimensions().len() == 4
                    && bias_tensor.shape.dimensions()[3] != max_sequence_length_key_value
                {
                    return Err(Error::FrontendSdpaMaxSequenceLengthBiasMismatch {
                        max_sequence_length_key_value,
                        bias_key_value_dimension: bias_tensor.shape.dimensions()[3],
                    });
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

    pub(in crate::frontend::composite::sdpa) fn validate_score_subgraph(
        &self,
        score_subgraph: &SdpaScoreSubgraph,
        scores_shape: &Shape,
        error_name: &str,
    ) -> Result<()> {
        let graph = score_subgraph.graph();
        let data_type = self.sdpa_aux_data_type();
        self.validate_sdpa_graph_tensor_shape_and_data_type(
            graph,
            score_subgraph.input(),
            scores_shape,
            data_type,
            error_name,
        )?;
        self.validate_sdpa_graph_tensor_shape_and_data_type(
            graph,
            score_subgraph.output(),
            scores_shape,
            data_type,
            error_name,
        )?;
        Ok(())
    }

    fn validate_sdpa_tensor_shape(
        &self,
        tensor: TensorId,
        expected_shape: &Shape,
        error_name: &str,
    ) -> Result<()> {
        self.validate_sdpa_tensor_dimensions_match(
            tensor,
            expected_shape.dimensions(),
            error_name,
        )?;
        self.validate_sdpa_tensor_strides_match(tensor, expected_shape.strides(), error_name)?;
        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_tensor_dimensions_match(
        &self,
        tensor: TensorId,
        expected_dimensions: &[i64],
        error_name: &str,
    ) -> Result<()> {
        let actual = self.tensor_config(tensor)?.shape.dimensions();
        if actual != expected_dimensions {
            return Err(Error::FrontendTensorDimensionsMismatch {
                tensor_id: tensor,
                operation: error_name.into(),
                expected: expected_dimensions.to_vec(),
                actual: actual.to_vec(),
            });
        }
        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_tensor_strides_match(
        &self,
        tensor: TensorId,
        expected_strides: &[i64],
        error_name: &str,
    ) -> Result<()> {
        let actual = self.tensor_config(tensor)?.shape.strides();
        if actual != expected_strides {
            return Err(Error::FrontendTensorStridesMismatch {
                tensor_id: tensor,
                operation: error_name.into(),
                expected: expected_strides.to_vec(),
                actual: actual.to_vec(),
            });
        }
        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_tensor_dimensions<const N: usize>(
        &self,
        tensors: [(TensorId, &[i64], &str); N],
    ) -> Result<()> {
        for (tensor, expected_dimensions, error_name) in tensors {
            self.validate_sdpa_tensor_dimensions_match(tensor, expected_dimensions, error_name)?;
        }
        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_tensor_dimensions_and_data_type(
        &self,
        tensor: TensorId,
        expected_dimensions: &[i64],
        expected_data_type: DataType,
        error_name: &str,
    ) -> Result<()> {
        self.validate_sdpa_tensor_dimensions_match(tensor, expected_dimensions, error_name)?;
        self.validate_sdpa_tensor_data_type_match(tensor, expected_data_type, error_name)
    }

    fn validate_sdpa_graph_tensor_shape_and_data_type(
        &self,
        graph: &Graph,
        tensor: TensorId,
        expected_shape: &Shape,
        expected_data_type: DataType,
        error_name: &str,
    ) -> Result<()> {
        let tensor_config = graph.tensor_config(tensor)?;
        if tensor_config.data_type != expected_data_type {
            return Err(Error::FrontendTensorDataTypeMismatch {
                tensor_id: tensor,
                operation: error_name.into(),
                expected: expected_data_type,
                actual: tensor_config.data_type,
            });
        }
        if tensor_config.shape.dimensions() != expected_shape.dimensions() {
            return Err(Error::FrontendTensorDimensionsMismatch {
                tensor_id: tensor,
                operation: error_name.into(),
                expected: expected_shape.dimensions().to_vec(),
                actual: tensor_config.shape.dimensions().to_vec(),
            });
        }
        if tensor_config.shape.strides() != expected_shape.strides() {
            return Err(Error::FrontendTensorStridesMismatch {
                tensor_id: tensor,
                operation: error_name.into(),
                expected: expected_shape.strides().to_vec(),
                actual: tensor_config.shape.strides().to_vec(),
            });
        }
        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_optional_sdpa_tensor_dimensions_and_data_type(
        &self,
        tensor: Option<TensorId>,
        requested: bool,
        expected_dimensions: &[i64],
        expected_data_type: DataType,
        missing_error_name: &str,
        shape_error_name: &str,
        data_type_error_name: &str,
    ) -> Result<()> {
        let Some(tensor) = tensor else {
            if requested {
                return Err(Error::FrontendSdpaAuxOutputRequestMismatch {
                    output: missing_error_name.into(),
                    requested,
                    provided: false,
                });
            }
            return Ok(());
        };

        self.validate_sdpa_tensor_dimensions_match(tensor, expected_dimensions, shape_error_name)?;
        self.validate_sdpa_tensor_data_type_match(tensor, expected_data_type, data_type_error_name)
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_tensor_data_type_match(
        &self,
        tensor: TensorId,
        expected_data_type: DataType,
        error_name: &str,
    ) -> Result<()> {
        let actual = self.tensor_config(tensor)?.data_type;
        if actual != expected_data_type {
            return Err(Error::FrontendTensorDataTypeMismatch {
                tensor_id: tensor,
                operation: error_name.into(),
                expected: expected_data_type,
                actual,
            });
        }
        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_tensor_data_types<const N: usize>(
        &self,
        tensors: [(TensorId, DataType, &str); N],
    ) -> Result<()> {
        for (tensor, expected_data_type, error_name) in tensors {
            self.validate_sdpa_tensor_data_type_match(tensor, expected_data_type, error_name)?;
        }
        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_tensor_shape_and_data_types(
        &self,
        tensor: TensorId,
        expected_shape: &Shape,
        allowed_data_types: &[DataType],
        error_name: &str,
    ) -> Result<()> {
        let tensor_config = self.tensor_config(tensor)?;
        if !allowed_data_types.contains(&tensor_config.data_type) {
            return Err(Error::FrontendTensorDataTypeUnsupported {
                tensor_id: tensor,
                operation: error_name.into(),
                actual: tensor_config.data_type,
                supported: allowed_data_types.to_vec(),
            });
        }
        self.validate_sdpa_tensor_dimensions_match(
            tensor,
            expected_shape.dimensions(),
            error_name,
        )?;
        self.validate_sdpa_tensor_strides_match(tensor, expected_shape.strides(), error_name)?;
        Ok(())
    }

    fn validate_optional_sdpa_tensor_shape_and_data_type(
        &self,
        tensor: Option<TensorId>,
        expected_shape: &Shape,
        allowed_data_types: &[DataType],
        error_name: &str,
    ) -> Result<()> {
        let Some(tensor) = tensor else {
            return Ok(());
        };
        self.validate_sdpa_tensor_shape_and_data_types(
            tensor,
            expected_shape,
            allowed_data_types,
            error_name,
        )
    }

    pub(in crate::frontend::composite::sdpa) fn validate_attention_support_surface_for_version(
        &self,
        cudnn_version: u64,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: AttentionConfig,
    ) -> Result<()> {
        let support_context = support::SdpaSupportContext::new(cudnn_version, self.sm_version());
        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();
        let q_dims = q_tensor.shape.dimensions();
        let k_dims = k_tensor.shape.dimensions();
        let v_dims = v_tensor.shape.dimensions();
        let has_padding_mask = config.modifiers().padding_mask();
        let has_custom_score_subgraph = config.score_subgraph().is_some();
        let has_causal_like_masking =
            config.modifiers().causal_mask() || config.modifiers().causal_bottom_right();
        let has_dropout = config.dropout().is_some() || config.modifiers().dropout_mask().is_some();
        let is_decode_only = q_dims[2] == 1;
        let is_paged = config.paged().is_enabled();
        let is_ragged = self.has_any_ragged_offset([q, k, v])?;

        if q_dims[1] % k_dims[1] != 0 || q_dims[1] % v_dims[1] != 0 {
            return Err(Error::FrontendSdpaGroupedQueryAttentionHeadsMismatch {
                query_heads: q_dims[1],
                key_heads: k_dims[1],
                value_heads: v_dims[1],
            });
        }
        Self::validate_attention_support_modifier_compatibility(
            config.modifiers(),
            has_custom_score_subgraph,
            "sdpa",
        )?;
        if config.modifiers().sink_token().is_some() {
            support::SDPA_SINK_TOKEN.require_frontend_feature(cudnn_version)?;
        }
        if is_paged {
            support::SDPA_PAGED_ATTENTION.require_frontend_feature(cudnn_version)?;
        }

        support::require_sdpa_forward_data_type_support(cudnn_version, q_tensor.data_type)?;
        support_context.require_forward_architecture(
            q_tensor.data_type,
            q_dims[3],
            v_dims[3],
            is_decode_only,
            is_paged,
        )?;

        let forward_support = support_context.forward();
        if let Some((left_window, _)) = config.modifiers().sliding_window() {
            forward_support.require_sliding_window(support::SdpaSlidingWindowSupport {
                left_window,
                query_sequence_length: q_dims[2],
                key_sequence_length: k_dims[2],
                has_padding_mask,
                has_causal_like_masking,
                has_dropout,
                has_bias: config.modifiers().bias().is_some(),
                has_causal_bottom_right: config.modifiers().causal_bottom_right(),
                is_decode_only,
            })?;
        }
        forward_support.require_causal_bottom_right(
            config.modifiers().causal_bottom_right(),
            has_padding_mask,
            config.modifiers().sliding_window().is_some(),
            q_dims[2],
            k_dims[2],
        )?;

        support::require_sdpa_ragged_mask_or_subgraph(
            is_ragged,
            has_padding_mask,
            config.modifiers().additive_mask().is_some(),
            config.score_subgraph().is_some(),
            "sdpa",
        )?;

        support_context.require_ragged_offsets_arch(is_ragged)?;
        forward_support.require_paged_ragged_offsets(is_paged, is_ragged)?;
        let paged = config.paged();
        if let Some(cache) = paged.k() {
            forward_support.require_packed_page_table(
                support::SdpaPagedTable::Key,
                self.tensor_config(cache.page_table())?
                    .ragged_offset
                    .is_some(),
            )?;
        }
        if let Some(cache) = paged.v() {
            forward_support.require_packed_page_table(
                support::SdpaPagedTable::Value,
                self.tensor_config(cache.page_table())?
                    .ragged_offset
                    .is_some(),
            )?;
        }

        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_attention_backward_config(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: &AttentionBackwardConfig,
    ) -> Result<()> {
        self.validate_sdpa_score_modifiers(q, k, v, config.modifiers())?;
        if config.score_subgraph_bprop().is_some() && config.score_subgraph().is_none() {
            return Err(Error::FrontendSdpaBackwardScoreSubgraphBpropRequiresForward);
        }
        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
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
        self.validate_attention_dropout_config(
            config.dropout(),
            config.dropout_offset(),
            config.modifiers(),
        )?;
        let aux_gradients = config.aux_gradients();
        if aux_gradients.random_number_generator_dump() && config.dropout().is_none() {
            return Err(Error::FrontendSdpaBackwardRngDumpRequiresDropout);
        }

        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_backward_tensors(
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

        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();
        let o_tensor = self.tensor_config(o)?.clone();
        let stats_tensor = self.tensor_config(stats)?.clone();

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

        let stats_rank = stats_tensor.shape.dimensions().len();
        if stats_rank != 4 {
            return Err(Error::FrontendTensorRankMismatch {
                tensor_id: stats,
                operation: SDPA_BACKWARD_RANK.into(),
                expected: "4".into(),
                actual: stats_rank,
            });
        }
        self.validate_sdpa_tensor_data_types([
            (o, q_tensor.data_type, SDPA_BACKWARD_DATA_TYPE),
            (d_o, q_tensor.data_type, SDPA_BACKWARD_DATA_TYPE),
            (d_q, q_tensor.data_type, SDPA_BACKWARD_DATA_TYPE),
            (d_k, k_tensor.data_type, SDPA_BACKWARD_DATA_TYPE),
            (d_v, v_tensor.data_type, SDPA_BACKWARD_DATA_TYPE),
        ])?;
        self.validate_sdpa_tensor_data_type_match(
            stats,
            config.softmax().compute_type(),
            SDPA_BACKWARD_STATS_DATA_TYPE,
        )?;
        self.validate_sdpa_scalar_element_count(scale, SDPA_BACKWARD_SCALE)?;
        self.validate_sdpa_tensor_dimensions_match(
            d_o,
            o_tensor.shape.dimensions(),
            SDPA_BACKWARD_D_O_SHAPE,
        )?;

        let q_dims = q_tensor.shape.dimensions();
        let k_dims = k_tensor.shape.dimensions();
        let v_dims = v_tensor.shape.dimensions();
        let o_dims = o_tensor.shape.dimensions();
        if q_dims[1] % k_dims[1] != 0 || q_dims[1] % v_dims[1] != 0 {
            return Err(
                Error::FrontendSdpaBackwardGroupedQueryAttentionHeadsMismatch {
                    query_heads: q_dims[1],
                    key_heads: k_dims[1],
                    value_heads: v_dims[1],
                },
            );
        }
        let k_is_transposed = q_dims[3] == k_dims[2];
        let v_is_transposed = o_dims[3] == v_dims[2];
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        let expected_stats_shape = reduce_last_axis(&scores_shape)?;

        self.validate_sdpa_tensor_shape(stats, &expected_stats_shape, SDPA_BACKWARD_STATS_SHAPE)?;
        self.validate_sdpa_tensor_shape(d_q, &q_tensor.shape, SDPA_BACKWARD_D_Q_SHAPE)?;

        let expected_d_k_shape = Shape::contiguous(vec![
            k_dims[0],
            k_dims[1],
            scores_shape.dimensions()[3],
            q_dims[3],
        ])?;
        self.validate_sdpa_tensor_shape(d_k, &expected_d_k_shape, SDPA_BACKWARD_D_K_SHAPE)?;
        let expected_d_v_shape = Shape::contiguous(vec![
            v_dims[0],
            v_dims[1],
            scores_shape.dimensions()[3],
            o_dims[3],
        ])?;
        self.validate_sdpa_tensor_shape(d_v, &expected_d_v_shape, SDPA_BACKWARD_D_V_SHAPE)?;

        let expected_d_sink_token_shape = Shape::contiguous([1, q_dims[1], 1, 1])?;
        self.validate_optional_sdpa_tensor_shape_and_data_type(
            gradients.bias_gradient,
            &scores_shape,
            &[q_tensor.data_type, config.softmax().compute_type()],
            SDPA_BACKWARD_D_BIAS_SHAPE,
        )?;
        self.validate_optional_sdpa_tensor_shape_and_data_type(
            gradients.sink_token_gradient,
            &expected_d_sink_token_shape,
            &[config.softmax().compute_type()],
            SDPA_BACKWARD_D_SINK_TOKEN_SHAPE,
        )?;
        if gradients.rng_dump.is_some() && config.dropout().is_none() {
            return Err(Error::FrontendSdpaBackwardRngDumpRequiresDropout);
        }

        Ok(SdpaBackwardValidation {
            scores_shape,
            expected_stats_shape,
            k_is_transposed,
            v_is_transposed,
        })
    }

    pub(in crate::frontend::composite::sdpa) fn validate_attention_backward_support_surface_for_version(
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
        let q_dims = self.tensor_config(q)?.shape.dimensions().to_vec();
        let k_dims = self.tensor_config(k)?.shape.dimensions().to_vec();
        let has_padding_mask = config.modifiers().padding_mask();
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
        support.require_sink_token(config.modifiers().sink_token().is_some())?;
        support.require_bias_gradient(support::SdpaBackwardBiasGradientSupport {
            enabled: aux_gradients.bias(),
            has_padding_mask,
            query_sequence_length: q_dims[2],
            key_sequence_length: k_dims[2],
            is_ragged,
        })?;
        support.require_deterministic_algorithm(support::SdpaBackwardDeterministicSupport {
            enabled: config.deterministic_algorithm(),
            has_bias_gradient: aux_gradients.bias(),
            has_dropout,
            has_alibi: config.modifiers().alibi_slopes().is_some(),
            is_ragged,
        })?;

        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_mxfp8_backward_support_surface_for_version(
        &self,
        cudnn_version: u64,
        output_io_type: DataType,
        is_ragged: bool,
        config: &AttentionBackwardConfig,
    ) -> Result<()> {
        let support_context = support::SdpaSupportContext::new(cudnn_version, self.sm_version());
        let aux_gradients = config.aux_gradients();
        let has_dropout = config.dropout().is_some() || config.modifiers().has_custom_dropout();
        support_context
            .quantized()
            .require_mxfp8_backward(support::MxFp8BackwardSupport {
                output_io_type,
                is_ragged,
                has_dropout,
                has_random_number_generator_dump: aux_gradients.random_number_generator_dump(),
            })
    }

    pub(in crate::frontend::composite::sdpa) fn convert_mxfp8_backward_output(
        &mut self,
        output: TensorId,
        output_data_type: DataType,
        compute_type: DataType,
    ) -> Result<TensorId> {
        self.pointwise_identity_infer(output, output_data_type, compute_type)
    }

    pub(in crate::frontend::composite::sdpa) fn validate_fp8_backward_support_surface_for_version(
        &self,
        cudnn_version: u64,
        output_io_type: DataType,
        d_qk: i64,
        d_v: i64,
        is_ragged: bool,
        config: &AttentionBackwardConfig,
    ) -> Result<()> {
        let support_context = support::SdpaSupportContext::new(cudnn_version, self.sm_version());
        let aux_gradients = config.aux_gradients();
        let has_dropout = config.dropout().is_some() || config.modifiers().has_custom_dropout();
        support_context
            .quantized()
            .require_fp8_backward(support::Fp8BackwardSupport {
                output_io_type,
                d_qk,
                d_v,
                is_ragged,
                has_dropout,
                has_random_number_generator_dump: aux_gradients.random_number_generator_dump(),
            })
    }

    pub(in crate::frontend::composite::sdpa) fn validate_fp8_forward_support_surface_for_version(
        &self,
        cudnn_version: u64,
        output_io_type: DataType,
    ) -> Result<()> {
        support::SdpaSupportContext::new(cudnn_version, self.sm_version())
            .quantized()
            .require_fp8_forward_output_type(output_io_type)
    }

    pub(in crate::frontend::composite::sdpa) fn convert_fp8_backward_output(
        &mut self,
        output: TensorId,
        scale: TensorId,
        output_data_type: DataType,
    ) -> Result<TensorId> {
        match output_data_type {
            DataType::F16 | DataType::BF16 => {
                self.pointwise_identity_infer(output, output_data_type, DataType::F32)
            }
            DataType::F8E4M3 | DataType::F8E5M2 => {
                self.quantize_tensor_infer(output, scale, output_data_type)
            }
            _ => Err(Error::FrontendTensorDataTypeUnsupported {
                tensor_id: output,
                operation: SDPA_FP8_BACKWARD_OUTPUT_DATA_TYPE.into(),
                supported: vec![
                    DataType::F16,
                    DataType::BF16,
                    DataType::F8E4M3,
                    DataType::F8E5M2,
                ],
                actual: output_data_type,
            }),
        }
    }

    pub(in crate::frontend::composite::sdpa) fn validate_quantized_backward_head_dimensions(
        &self,
        d_qk: i64,
        d_v: i64,
        operation: &str,
    ) -> Result<()> {
        support::SdpaSupportContext::new(version()?.raw(), self.sm_version())
            .quantized()
            .require_quantized_backward_head_dimensions(d_qk, d_v, operation)
    }

    pub(in crate::frontend::composite::sdpa) fn validate_quantized_backward_output_ragged_support(
        &self,
        d_q: TensorId,
        d_k: TensorId,
        d_v: TensorId,
        operation: &str,
    ) -> Result<()> {
        let has_ragged_output = self.has_any_ragged_offset([d_q, d_k, d_v])?;
        support::SdpaSupportContext::new(version()?.raw(), self.sm_version())
            .quantized()
            .require_quantized_backward_output_ragged_support(has_ragged_output, operation)
    }

    pub(in crate::frontend::composite::sdpa) fn validate_mxfp8_backward_scale_dimensions(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        d_o: TensorId,
        scale_q: TensorId,
        scale_q_t: TensorId,
        scale_k: TensorId,
        scale_k_t: TensorId,
        scale_v: TensorId,
        scale_d_o: TensorId,
        scale_d_o_t: TensorId,
    ) -> Result<()> {
        let q_dims = self.tensor_config(q)?.shape.dimensions().to_vec();
        let k_dims = self.tensor_config(k)?.shape.dimensions().to_vec();
        let v_dims = self.tensor_config(v)?.shape.dimensions().to_vec();
        let d_o_dims = self.tensor_config(d_o)?.shape.dimensions().to_vec();
        let d_qk_scale = (q_dims[3] + 31) / 32;
        let d_v_scale = (v_dims[3] + 31) / 32;
        let s_q_scale = (q_dims[2] + 31) / 32;
        let s_kv_scale = (k_dims[2] + 31) / 32;

        let validate = |tensor_id: TensorId,
                        tensor: &TensorSpec,
                        name: &str,
                        batch: i64,
                        heads: i64,
                        min_seq_scale: Option<i64>,
                        min_d_scale: Option<i64>|
         -> Result<()> {
            let dims = tensor.shape.dimensions();
            if dims.len() != 4 {
                return Err(Error::FrontendTensorRankMismatch {
                    tensor_id,
                    operation: name.into(),
                    expected: "4".into(),
                    actual: dims.len(),
                });
            }
            if dims[0] != batch || dims[1] != heads {
                return Err(Error::FrontendTensorLayoutMismatch {
                    tensor_id,
                    operation: name.into(),
                    reason: format!(
                        "expected batch/head dimensions [{batch}, {heads}], got [{}, {}]",
                        dims[0], dims[1]
                    ),
                });
            }
            if let Some(min_seq_scale) = min_seq_scale
                && dims[2] < min_seq_scale
            {
                return Err(Error::FrontendTensorLayoutMismatch {
                    tensor_id,
                    operation: name.into(),
                    reason: format!(
                        "expected sequence scale dimension at least {min_seq_scale}, got {}",
                        dims[2]
                    ),
                });
            }
            if let Some(min_d_scale) = min_d_scale
                && dims[3] < min_d_scale
            {
                return Err(Error::FrontendTensorLayoutMismatch {
                    tensor_id,
                    operation: name.into(),
                    reason: format!(
                        "expected head-dimension scale at least {min_d_scale}, got {}",
                        dims[3]
                    ),
                });
            }
            Ok(())
        };

        validate(
            scale_q,
            self.tensor_config(scale_q)?,
            SDPA_MXFP8_BACKWARD_SCALE_Q_SHAPE,
            q_dims[0],
            q_dims[1],
            None,
            Some(d_qk_scale),
        )?;
        validate(
            scale_q_t,
            self.tensor_config(scale_q_t)?,
            SDPA_MXFP8_BACKWARD_SCALE_Q_T_SHAPE,
            q_dims[0],
            q_dims[1],
            Some(s_q_scale),
            None,
        )?;
        validate(
            scale_k,
            self.tensor_config(scale_k)?,
            SDPA_MXFP8_BACKWARD_SCALE_K_SHAPE,
            k_dims[0],
            k_dims[1],
            None,
            Some(d_qk_scale),
        )?;
        validate(
            scale_k_t,
            self.tensor_config(scale_k_t)?,
            SDPA_MXFP8_BACKWARD_SCALE_K_T_SHAPE,
            k_dims[0],
            k_dims[1],
            Some(s_kv_scale),
            None,
        )?;
        validate(
            scale_v,
            self.tensor_config(scale_v)?,
            SDPA_MXFP8_BACKWARD_SCALE_V_SHAPE,
            v_dims[0],
            v_dims[1],
            Some(s_kv_scale),
            None,
        )?;
        validate(
            scale_d_o,
            self.tensor_config(scale_d_o)?,
            SDPA_MXFP8_BACKWARD_SCALE_D_O_SHAPE,
            d_o_dims[0],
            d_o_dims[1],
            None,
            Some(d_v_scale),
        )?;
        validate(
            scale_d_o_t,
            self.tensor_config(scale_d_o_t)?,
            SDPA_MXFP8_BACKWARD_SCALE_D_O_T_SHAPE,
            d_o_dims[0],
            d_o_dims[1],
            Some(s_q_scale),
            None,
        )?;

        Ok(())
    }
}

fn sdpa_causal_bottom_right_modifier_conflict(modifiers: &SdpaScoreModifiers) -> Error {
    let modifier = if modifiers.dropout_mask().is_some() || modifiers.dropout_scale().is_some() {
        "custom dropout"
    } else if modifiers.causal_mask() {
        "causal top-left mask"
    } else if modifiers.padding_mask() {
        "padding mask"
    } else if modifiers.sliding_window().is_some() {
        "sliding window"
    } else if modifiers.alibi_slopes().is_some() {
        "alibi slopes"
    } else {
        "score modifier"
    };
    Error::FrontendSdpaCausalBottomRightModifierConflict {
        modifier: modifier.into(),
    }
}

fn sdpa_score_subgraph_modifier_conflict(modifiers: &SdpaScoreModifiers) -> &'static str {
    if modifiers.alibi_slopes().is_some() {
        "alibi slopes"
    } else if modifiers.padding_mask() {
        "padding mask"
    } else if modifiers.causal_bottom_right() {
        "causal bottom-right mask"
    } else if modifiers.causal_mask() {
        "causal top-left mask"
    } else if modifiers.sliding_window().is_some() {
        "sliding window"
    } else {
        "score modifier"
    }
}
