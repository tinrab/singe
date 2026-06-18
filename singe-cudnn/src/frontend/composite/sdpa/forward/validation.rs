use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::{SdpaBackwardGradientTensors, SdpaBackwardInputs},
        graph::Graph,
        infer::*,
        operation::*,
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
    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_tensor_layout(
        &self,
        tensor: TensorId,
        name: &str,
    ) -> Result<()> {
        let tensor = self.tensor_config(tensor)?;
        if tensor.shape.dimensions().len() != 4 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa rank".into(),
            });
        }
        if tensor.shape.strides()[3] != 1 {
            return Err(Error::DescriptorMismatch { name: name.into() });
        }

        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_k_layout(
        &self,
        tensor: TensorId,
    ) -> Result<()> {
        let tensor = self.tensor_config(tensor)?;
        if tensor.shape.dimensions().len() != 4 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa rank".into(),
            });
        }
        if tensor.shape.strides()[2] != 1 && tensor.shape.strides()[3] != 1 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa k layout".into(),
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
        let is_ragged = self.tensor_config(q)?.ragged_offset.is_some()
            || self.tensor_config(k)?.ragged_offset.is_some()
            || self.tensor_config(v)?.ragged_offset.is_some()
            || self.tensor_config(o)?.ragged_offset.is_some();
        let has_padding_mask = direct_config.padding_mask();
        if is_ragged && !has_padding_mask && !has_custom_score_subgraph {
            return Err(Error::DescriptorMismatch {
                name: "sdpa ragged offsets".into(),
            });
        }
        if is_ragged
            && self.sm_version().is_some_and(|sm_version| sm_version < 90)
            && cudnn_version < 91801
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa ragged offsets".into(),
            });
        }

        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_sdpa_score_modifiers(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        modifiers: &SdpaScoreModifiers,
    ) -> Result<()> {
        self.validate_sdpa_tensor_layout(q, "sdpa q layout")?;
        self.validate_sdpa_k_layout(k)?;
        self.validate_sdpa_k_layout(v)?;

        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();
        let q_dims = q_tensor.shape.dimensions();
        let k_dims = k_tensor.shape.dimensions();
        let v_dims = v_tensor.shape.dimensions();

        if q_dims[1] % k_dims[1] != 0 || q_dims[1] % v_dims[1] != 0 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa attention heads".into(),
            });
        }

        match (
            modifiers.sequence_length_query,
            modifiers.sequence_length_key_value,
        ) {
            (Some(sequence_length_query), Some(sequence_length_key_value)) => {
                let seq_len_q_tensor = self.tensor_config(sequence_length_query)?.clone();
                let seq_len_kv_tensor = self.tensor_config(sequence_length_key_value)?.clone();
                if seq_len_q_tensor.data_type != DataType::I32
                    || seq_len_kv_tensor.data_type != DataType::I32
                {
                    return Err(Error::DescriptorMismatch {
                        name: "sdpa seq len data type".into(),
                    });
                }
                let expected_seq_shape = [q_dims[0], 1, 1, 1];
                if seq_len_q_tensor.shape.dimensions() != expected_seq_shape
                    || seq_len_kv_tensor.shape.dimensions() != expected_seq_shape
                {
                    return Err(Error::DescriptorMismatch {
                        name: "sdpa seq len shape".into(),
                    });
                }
            }
            (None, None) => {
                if modifiers.causal_bottom_right {
                    return Err(Error::DescriptorMismatch {
                        name: "sdpa causal bottom right seq lens".into(),
                    });
                }
            }
            _ => {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa seq lens".into(),
                });
            }
        }

        match (modifiers.dropout_mask, modifiers.dropout_scale) {
            (Some(_), Some(dropout_scale)) => {
                if self.tensor_config(dropout_scale)?.shape.element_count()? != 1 {
                    return Err(Error::DescriptorMismatch {
                        name: "sdpa dropout scale".into(),
                    });
                }
            }
            (None, None) => {}
            _ => {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa dropout".into(),
                });
            }
        }

        if let Some(bias) = modifiers.bias
            && self.tensor_config(bias)?.data_type == DataType::Boolean
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa bias data type".into(),
            });
        }
        if let Some(alibi_slopes) = modifiers.alibi_slopes {
            let alibi_tensor = self.tensor_config(alibi_slopes)?.clone();
            if alibi_tensor.data_type != DataType::F32 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa alibi data type".into(),
                });
            }
            let expected_alibi_shape = [1, q_dims[1], 1, 1];
            if alibi_tensor.shape.dimensions() != expected_alibi_shape {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa alibi shape".into(),
                });
            }
            let has_alibi_right_bound = modifiers.causal_mask
                || modifiers
                    .sliding_window
                    .is_some_and(|(_, right)| right == 0);
            if !has_alibi_right_bound {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa alibi alignment".into(),
                });
            }
        }

        if let Some(sink_token) = modifiers.sink_token {
            let sink_tensor = self.tensor_config(sink_token)?.clone();
            if sink_tensor.data_type != DataType::F32 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa sink token data type".into(),
                });
            }
            if sink_tensor.shape.dimensions().len() != 4 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa sink token rank".into(),
                });
            }
            let expected_sink_shape = [1, q_dims[1], 1, 1];
            if sink_tensor.shape.dimensions() != expected_sink_shape {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa sink token shape".into(),
                });
            }
            if q_dims[2] == 1 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa sink token decode".into(),
                });
            }
        }

        if modifiers.causal_bottom_right
            && (modifiers.bias.is_some()
                || modifiers.alibi_slopes.is_some()
                || modifiers.dropout_mask.is_some())
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa causal bottom right modifiers".into(),
            });
        }
        if modifiers.causal_mask && modifiers.causal_bottom_right {
            return Err(Error::DescriptorMismatch {
                name: "sdpa causal alignment".into(),
            });
        }
        if let Some((left_window, right_window)) = modifiers.sliding_window {
            check_range!("left_window", left_window >= 0)?;
            check_range!("right_window", right_window >= 0)?;
            if modifiers.causal_bottom_right {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa sliding window alignment".into(),
                });
            }
        }

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
            self.validate_score_subgraph(score_subgraph, &scores_shape, "sdpa score subgraph")?;
        }
        if config.block_mask().is_some()
            && !self.can_lower_attention_as_unified_with_tensors_for_version(
                version()?.raw(),
                q,
                k,
                v,
                config.clone(),
            )?
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa block mask".into(),
            });
        }

        let has_seq_lens = config.modifiers().sequence_length_query.is_some()
            && config.modifiers().sequence_length_key_value.is_some();
        if (config.paged_k().is_some() || config.paged_v().is_some()) && !has_seq_lens {
            return Err(Error::DescriptorMismatch {
                name: "sdpa paged seq lens".into(),
            });
        }
        if let Some(max_sequence_length_key_value) = config.max_sequence_length_key_value() {
            check_range!(
                "sdpa max_sequence_length_key_value",
                max_sequence_length_key_value > 0
            )?;
            if config.paged_k().is_none() && config.paged_v().is_none() {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa max_sequence_length_key_value".into(),
                });
            }
            if let Some(bias) = config.modifiers().bias {
                let bias_tensor = self.tensor_config(bias)?.clone();
                if bias_tensor.shape.dimensions().len() == 4
                    && bias_tensor.shape.dimensions()[3] != max_sequence_length_key_value
                {
                    return Err(Error::DescriptorMismatch {
                        name: "sdpa max_sequence_length_key_value bias".into(),
                    });
                }
            }
        }
        if let Some(cache) = config.paged_k() {
            let page_table_k_tensor = self.tensor_config(cache.page_table)?.clone();
            if page_table_k_tensor.data_type != DataType::I32 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa paged k page table data type".into(),
                });
            }
        }
        if let Some(cache) = config.paged_v() {
            let page_table_v_tensor = self.tensor_config(cache.page_table)?.clone();
            if page_table_v_tensor.data_type != DataType::I32 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa paged v page table data type".into(),
                });
            }
        }
        if let Some(block_mask) = config.block_mask() {
            let block_mask_tensor = self.tensor_config(block_mask)?.clone();
            if block_mask_tensor.data_type != DataType::U8 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa block mask data type".into(),
                });
            }
            if block_mask_tensor.shape.dimensions().len() != 4
                || block_mask_tensor.shape.strides()[3] != 1
            {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa block mask layout".into(),
                });
            }
        }
        if let Some(dropout) = config.dropout() {
            check_range!(
                "sdpa dropout probability",
                (0.0..1.0).contains(&dropout.probability())
            )?;
            if config.dropout_offset().is_some() && config.modifiers().dropout_mask.is_some() {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa dropout".into(),
                });
            }
            if config.modifiers().dropout_mask.is_some()
                || config.modifiers().dropout_scale.is_some()
            {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa dropout".into(),
                });
            }
            if config.modifiers().causal_bottom_right {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa causal bottom right modifiers".into(),
                });
            }
            if let Some(offset) = config.dropout_offset() {
                let offset_tensor = self.tensor_config(offset)?.clone();
                if !matches!(offset_tensor.data_type, DataType::I32 | DataType::I64) {
                    return Err(Error::DescriptorMismatch {
                        name: "rng offset data type".into(),
                    });
                }
                if offset_tensor.shape.dimensions() != [1, 1, 1, 1] {
                    return Err(Error::DescriptorMismatch {
                        name: "rng offset".into(),
                    });
                }
            }
            if let Some(seed) = config.dropout().and_then(|dropout| dropout.seed_tensor()) {
                let seed_tensor = self.tensor_config(seed)?.clone();
                if seed_tensor.data_type != DataType::I64 {
                    return Err(Error::DescriptorMismatch {
                        name: "rng seed data type".into(),
                    });
                }
                if seed_tensor.shape.dimensions() != [1, 1, 1, 1] {
                    return Err(Error::DescriptorMismatch {
                        name: "rng seed".into(),
                    });
                }
                if config.dropout_offset().is_none() {
                    return Err(Error::DescriptorMismatch {
                        name: "sdpa dropout offset".into(),
                    });
                }
            }
        } else if config.dropout_offset().is_some() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa dropout offset".into(),
            });
        }
        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_score_subgraph(
        &self,
        score_subgraph: &SdpaScoreSubgraph,
        scores_shape: &Shape,
        error_name: &str,
    ) -> Result<()> {
        let input = score_subgraph
            .graph()
            .tensor_config(score_subgraph.input())?
            .clone();
        let output = score_subgraph
            .graph()
            .tensor_config(score_subgraph.output())?
            .clone();
        if input.data_type != self.sdpa_aux_data_type()
            || output.data_type != self.sdpa_aux_data_type()
            || input.shape.dimensions() != scores_shape.dimensions()
            || input.shape.strides() != scores_shape.strides()
            || output.shape.dimensions() != scores_shape.dimensions()
            || output.shape.strides() != scores_shape.strides()
        {
            return Err(Error::DescriptorMismatch {
                name: error_name.into(),
            });
        }

        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_attention_support_surface_for_version(
        &self,
        cudnn_version: u64,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: AttentionConfig,
    ) -> Result<()> {
        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();
        let q_dims = q_tensor.shape.dimensions();
        let k_dims = k_tensor.shape.dimensions();
        let v_dims = v_tensor.shape.dimensions();
        let has_seq_lens = config.modifiers().sequence_length_query.is_some()
            && config.modifiers().sequence_length_key_value.is_some();
        let has_padding_mask = config.modifiers().padding_mask;
        let has_custom_score_subgraph = config.score_subgraph().is_some();
        let has_causal_like_masking =
            config.modifiers().causal_mask || config.modifiers().causal_bottom_right;
        let has_dropout = config.dropout().is_some() || config.modifiers().dropout_mask.is_some();
        let is_decode_only = q_dims[2] == 1;
        let is_paged = config.paged_k().is_some() || config.paged_v().is_some();
        let is_ragged = q_tensor.ragged_offset.is_some()
            || k_tensor.ragged_offset.is_some()
            || v_tensor.ragged_offset.is_some();

        if q_dims[1] % k_dims[1] != 0 || q_dims[1] % v_dims[1] != 0 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa gqa heads".into(),
            });
        }
        if has_custom_score_subgraph
            && (config.modifiers().alibi_slopes.is_some()
                || has_padding_mask
                || has_causal_like_masking
                || config.modifiers().sliding_window.is_some())
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa score subgraph modifiers".into(),
            });
        }
        if !has_padding_mask
            && !has_custom_score_subgraph
            && has_seq_lens
            && !config.modifiers().causal_bottom_right
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa seq lens".into(),
            });
        }
        if config.modifiers().sink_token.is_some() && cudnn_version < 91300 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa sink token version".into(),
            });
        }

        if is_paged && cudnn_version < 90500 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa paged attention version".into(),
            });
        }

        if matches!(q_tensor.data_type, DataType::F16 | DataType::BF16)
            && matches!(cudnn_version, 91000 | 91001)
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa cuDNN version".into(),
            });
        }

        if let Some(sm_version) = self.sm_version() {
            let prop_major = sm_version / 10;
            let d_qk = q_dims[3];
            let d_v = v_dims[3];

            if matches!(q_tensor.data_type, DataType::F16 | DataType::BF16) && sm_version < 80 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa sm version".into(),
                });
            }

            if prop_major == 8 && cudnn_version <= 90900 && (d_qk > 128 || d_v > 128) {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa head dimension".into(),
                });
            }
            if prop_major == 9 && cudnn_version <= 90900 && (d_qk > 256 || d_v > 256) {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa head dimension".into(),
                });
            }
            if prop_major == 10 && cudnn_version < 90900 && (d_qk > 128 || d_v > 128) {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa head dimension".into(),
                });
            }
            if is_decode_only
                && prop_major == 10
                && cudnn_version <= 90900
                && (d_qk > 128 || d_v > 128)
            {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa decode head dimension".into(),
                });
            }
            if is_paged && cudnn_version <= 90900 && (d_qk > 128 || d_v > 128) {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa paged attention head dimension".into(),
                });
            }
        }

        if let Some((left_window, _)) = config.modifiers().sliding_window {
            check_range!("left_window", cudnn_version >= 91000 || left_window > 0)?;
            if cudnn_version < 91002
                && (!has_causal_like_masking || has_dropout || config.modifiers().bias.is_some())
            {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa sliding window modifiers".into(),
                });
            }
            if cudnn_version <= 90900
                && self
                    .sm_version()
                    .is_some_and(|sm_version| sm_version / 10 == 9)
                && config.modifiers().causal_bottom_right
                && q_dims[2] * left_window == k_dims[2] * left_window
            {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa sliding window alignment".into(),
                });
            }
            if !has_padding_mask && q_dims[2] > k_dims[2] {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa sliding window sequence lengths".into(),
                });
            }
        }
        if is_decode_only && cudnn_version <= 90900 && config.modifiers().sliding_window.is_some() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa decode masking".into(),
            });
        }

        if cudnn_version == 91400
            && config.modifiers().sliding_window.is_some()
            && k_dims[2] > 1024
            && !has_causal_like_masking
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa cuDNN version".into(),
            });
        }

        if config.modifiers().causal_bottom_right && !has_padding_mask && q_dims[2] > k_dims[2] {
            return Err(Error::DescriptorMismatch {
                name: "sdpa causal bottom right sequence lengths".into(),
            });
        }
        if config.modifiers().causal_bottom_right
            && cudnn_version < 90600
            && (q_dims[2] % 64 != 0 || k_dims[2] % 64 != 0)
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa causal bottom right sequence lengths".into(),
            });
        }
        if config.modifiers().causal_bottom_right
            && config.modifiers().sliding_window.is_some()
            && cudnn_version < 90600
            && q_dims[2] != k_dims[2]
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa sliding window alignment".into(),
            });
        }

        if is_ragged
            && !has_padding_mask
            && config.modifiers().additive_mask.is_none()
            && config.score_subgraph().is_none()
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa ragged offsets".into(),
            });
        }

        if is_ragged
            && self.sm_version().is_some_and(|sm_version| sm_version < 90)
            && cudnn_version < 91801
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa ragged offsets".into(),
            });
        }
        if is_paged && is_ragged && cudnn_version < 90700 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa paged attention ragged offsets".into(),
            });
        }
        if cudnn_version < 91002 {
            if let Some(cache) = config.paged_k()
                && self
                    .tensor_config(cache.page_table)?
                    .ragged_offset
                    .is_some()
            {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa paged k packed page table version".into(),
                });
            }
            if let Some(cache) = config.paged_v()
                && self
                    .tensor_config(cache.page_table)?
                    .ragged_offset
                    .is_some()
            {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa paged v packed page table version".into(),
                });
            }
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
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward score subgraph bprop".into(),
            });
        }
        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        if let Some(score_subgraph) = config.score_subgraph() {
            self.validate_score_subgraph(
                score_subgraph,
                &scores_shape,
                "sdpa backward score subgraph",
            )?;
        }
        if let Some(score_subgraph_bprop) = config.score_subgraph_bprop() {
            self.validate_score_subgraph(
                score_subgraph_bprop,
                &scores_shape,
                "sdpa backward score subgraph bprop",
            )?;
        }
        if let Some(dropout) = config.dropout() {
            check_range!(
                "sdpa dropout probability",
                (0.0..1.0).contains(&dropout.probability())
            )?;
            if config.dropout_offset().is_some() && config.modifiers().dropout_mask.is_some() {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa dropout".into(),
                });
            }
            if config.modifiers().dropout_mask.is_some()
                || config.modifiers().dropout_scale.is_some()
            {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa dropout".into(),
                });
            }
            if config.modifiers().causal_bottom_right {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa causal bottom right modifiers".into(),
                });
            }
            if let Some(offset) = config.dropout_offset() {
                let offset_tensor = self.tensor_config(offset)?.clone();
                if !matches!(offset_tensor.data_type, DataType::I32 | DataType::I64) {
                    return Err(Error::DescriptorMismatch {
                        name: "rng offset data type".into(),
                    });
                }
                if offset_tensor.shape.dimensions() != [1, 1, 1, 1] {
                    return Err(Error::DescriptorMismatch {
                        name: "rng offset".into(),
                    });
                }
            }
            if let Some(seed) = config.dropout().and_then(|dropout| dropout.seed_tensor()) {
                let seed_tensor = self.tensor_config(seed)?.clone();
                if seed_tensor.data_type != DataType::I64 {
                    return Err(Error::DescriptorMismatch {
                        name: "rng seed data type".into(),
                    });
                }
                if seed_tensor.shape.dimensions() != [1, 1, 1, 1] {
                    return Err(Error::DescriptorMismatch {
                        name: "rng seed".into(),
                    });
                }
                if config.dropout_offset().is_none() {
                    return Err(Error::DescriptorMismatch {
                        name: "sdpa dropout offset".into(),
                    });
                }
            }
        } else if config.dropout_offset().is_some() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa dropout offset".into(),
            });
        }
        if config.has_random_number_generator_dump() && config.dropout().is_none() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward rng dump".into(),
            });
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
        let d_o_tensor = self.tensor_config(d_o)?.clone();
        let stats_tensor = self.tensor_config(stats)?.clone();
        let scale_tensor = self.tensor_config(scale)?.clone();

        self.validate_sdpa_tensor_layout(q, "sdpa q layout")?;
        self.validate_sdpa_k_layout(k)?;
        self.validate_sdpa_k_layout(v)?;
        self.validate_sdpa_tensor_layout(o, "sdpa output layout")?;
        self.validate_sdpa_tensor_layout(d_o, "sdpa backward d_o layout")?;
        self.validate_sdpa_tensor_layout(d_q, "sdpa backward d_q layout")?;
        self.validate_sdpa_tensor_layout(d_k, "sdpa backward d_k layout")?;
        self.validate_sdpa_tensor_layout(d_v, "sdpa backward d_v layout")?;

        if stats_tensor.shape.dimensions().len() != 4 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward rank".into(),
            });
        }
        if q_tensor.data_type != o_tensor.data_type
            || q_tensor.data_type != d_o_tensor.data_type
            || q_tensor.data_type != self.tensor_config(d_q)?.data_type
            || k_tensor.data_type != self.tensor_config(d_k)?.data_type
            || v_tensor.data_type != self.tensor_config(d_v)?.data_type
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward data type".into(),
            });
        }
        if stats_tensor.data_type != config.softmax().compute_type() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward stats data type".into(),
            });
        }
        if q_tensor.shape.dimensions()[1] % k_tensor.shape.dimensions()[1] != 0
            || q_tensor.shape.dimensions()[1] % v_tensor.shape.dimensions()[1] != 0
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa attention heads".into(),
            });
        }
        if scale_tensor.shape.element_count()? != 1 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward scale".into(),
            });
        }
        if o_tensor.shape.dimensions() != d_o_tensor.shape.dimensions() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward d_o shape".into(),
            });
        }

        let q_dims = q_tensor.shape.dimensions();
        let k_dims = k_tensor.shape.dimensions();
        let v_dims = v_tensor.shape.dimensions();
        let o_dims = o_tensor.shape.dimensions();
        let k_is_transposed = q_dims[3] == k_dims[2];
        let v_is_transposed = o_dims[3] == v_dims[2];
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        let expected_stats_shape = reduce_last_axis(&scores_shape)?;

        if stats_tensor.shape.dimensions() != expected_stats_shape.dimensions()
            || stats_tensor.shape.strides() != expected_stats_shape.strides()
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward stats shape".into(),
            });
        }
        if self.tensor_config(d_q)?.shape.dimensions() != q_tensor.shape.dimensions()
            || self.tensor_config(d_q)?.shape.strides() != q_tensor.shape.strides()
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward d_q shape".into(),
            });
        }

        let expected_d_k_shape = Shape::contiguous(vec![
            k_dims[0],
            k_dims[1],
            scores_shape.dimensions()[3],
            q_dims[3],
        ])?;
        if self.tensor_config(d_k)?.shape.dimensions() != expected_d_k_shape.dimensions()
            || self.tensor_config(d_k)?.shape.strides() != expected_d_k_shape.strides()
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward d_k shape".into(),
            });
        }
        let expected_d_v_shape = Shape::contiguous(vec![
            v_dims[0],
            v_dims[1],
            scores_shape.dimensions()[3],
            o_dims[3],
        ])?;
        if self.tensor_config(d_v)?.shape.dimensions() != expected_d_v_shape.dimensions()
            || self.tensor_config(d_v)?.shape.strides() != expected_d_v_shape.strides()
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward d_v shape".into(),
            });
        }

        let expected_d_sink_token_shape = Shape::contiguous([1, q_dims[1], 1, 1])?;
        if let Some(d_bias) = gradients.bias_gradient
            && (self.tensor_config(d_bias)?.data_type != q_tensor.data_type
                && self.tensor_config(d_bias)?.data_type != config.softmax().compute_type()
                || self.tensor_config(d_bias)?.shape.dimensions() != scores_shape.dimensions()
                || self.tensor_config(d_bias)?.shape.strides() != scores_shape.strides())
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward d_bias shape".into(),
            });
        }
        if let Some(d_sink_token) = gradients.sink_token_gradient
            && (self.tensor_config(d_sink_token)?.data_type != config.softmax().compute_type()
                || self.tensor_config(d_sink_token)?.shape.dimensions() != [1, q_dims[1], 1, 1]
                || self.tensor_config(d_sink_token)?.shape.strides()
                    != expected_d_sink_token_shape.strides())
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward d_sink_token shape".into(),
            });
        }
        if gradients.rng_dump.is_some() && config.dropout().is_none() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward rng dump".into(),
            });
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
        let q_dims = self.tensor_config(q)?.shape.dimensions().to_vec();
        let k_dims = self.tensor_config(k)?.shape.dimensions().to_vec();
        let has_seq_lens = config.modifiers().sequence_length_query.is_some()
            && config.modifiers().sequence_length_key_value.is_some();
        let has_padding_mask = config.modifiers().padding_mask;
        let has_custom_score_subgraph = config.score_subgraph().is_some();
        let is_ragged = self.tensor_config(q)?.ragged_offset.is_some()
            || self.tensor_config(k)?.ragged_offset.is_some()
            || self.tensor_config(v)?.ragged_offset.is_some()
            || self.tensor_config(o)?.ragged_offset.is_some()
            || self.tensor_config(stats)?.ragged_offset.is_some()
            || self.tensor_config(d_o)?.ragged_offset.is_some()
            || self.tensor_config(d_q)?.ragged_offset.is_some()
            || self.tensor_config(d_k)?.ragged_offset.is_some()
            || self.tensor_config(d_v)?.ragged_offset.is_some();
        let has_causal_like_masking =
            config.modifiers().causal_mask || config.modifiers().causal_bottom_right;

        if has_custom_score_subgraph
            && (config.modifiers().alibi_slopes.is_some()
                || has_padding_mask
                || has_causal_like_masking
                || config.modifiers().sliding_window.is_some())
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward score subgraph modifiers".into(),
            });
        }
        if !has_padding_mask
            && !has_custom_score_subgraph
            && has_seq_lens
            && !config.modifiers().causal_bottom_right
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward seq lens".into(),
            });
        }
        if config.modifiers().causal_bottom_right {
            if cudnn_version < 90700 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa backward causal bottom right version".into(),
                });
            }
            if self
                .sm_version()
                .is_some_and(|sm_version| sm_version / 10 < 10)
            {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa backward causal bottom right architecture".into(),
                });
            }
            if q_dims[2] > k_dims[2] {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa backward causal bottom right sequence lengths".into(),
                });
            }
            if config.dropout().is_some() {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa backward causal bottom right dropout".into(),
                });
            }
            if q_dims[2] % 64 != 0 || k_dims[2] % 64 != 0 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa backward causal bottom right multiples of 64".into(),
                });
            }
        }

        if config.modifiers().sink_token.is_some() && cudnn_version < 91300 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward sink token version".into(),
            });
        }
        if cudnn_version < 90500 && config.has_bias_gradient() && has_padding_mask {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward d_bias version".into(),
            });
        }
        if cudnn_version < 90500
            && config.has_bias_gradient()
            && (q_dims[2] % 64 != 0 || k_dims[2] % 64 != 0)
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward d_bias sequence lengths".into(),
            });
        }
        if self
            .sm_version()
            .is_some_and(|sm_version| sm_version / 10 == 10)
            && is_ragged
            && config.has_bias_gradient()
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward d_bias ragged".into(),
            });
        }
        if config.deterministic_algorithm()
            && self
                .sm_version()
                .is_some_and(|sm_version| sm_version / 10 == 10)
        {
            if cudnn_version < 91800 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa backward deterministic algorithm version".into(),
                });
            }
            if config.has_bias_gradient()
                || config.dropout().is_some()
                || config.modifiers().dropout_mask.is_some()
                || config.modifiers().dropout_scale.is_some()
                || config.modifiers().alibi_slopes.is_some()
            {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa backward deterministic algorithm modifiers".into(),
                });
            }
        }
        if config.deterministic_algorithm()
            && is_ragged
            && cudnn_version >= 91801
            && self
                .sm_version()
                .is_some_and(|sm_version| matches!(sm_version / 10, 8 | 12))
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward deterministic algorithm ragged".into(),
            });
        }

        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_mxfp8_backward_support_surface_for_version(
        &self,
        cudnn_version: u64,
        output_io_type: DataType,
        is_ragged: bool,
        config: &AttentionBackwardConfig,
    ) -> Result<()> {
        if self
            .sm_version()
            .is_some_and(|sm_version| sm_version / 10 < 9)
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward sm version".into(),
            });
        }
        if self
            .sm_version()
            .is_some_and(|sm_version| sm_version / 10 == 9)
            && is_ragged
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward ragged".into(),
            });
        }
        if cudnn_version < 92100 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward version".into(),
            });
        }

        if matches!(output_io_type, DataType::F16 | DataType::BF16)
            && (cudnn_version < 91300
                || self
                    .sm_version()
                    .is_some_and(|sm_version| sm_version / 10 < 10))
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward output data type".into(),
            });
        }

        let has_dropout = config.dropout().is_some()
            || config.modifiers().dropout_mask.is_some()
            || config.modifiers().dropout_scale.is_some();
        if self
            .sm_version()
            .is_some_and(|sm_version| sm_version / 10 == 10)
            && has_dropout
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward deterministic algorithm dropout".into(),
            });
        }
        if config.has_random_number_generator_dump() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa mxfp8 backward rng_dump".into(),
            });
        }

        Ok(())
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
        if cudnn_version < 90100 || cudnn_version == 91000 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa fp8 backward version".into(),
            });
        }
        if self
            .sm_version()
            .is_some_and(|sm_version| sm_version / 10 < 9)
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa fp8 backward sm version".into(),
            });
        }
        if self
            .sm_version()
            .is_some_and(|sm_version| sm_version / 10 == 9)
            && is_ragged
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa fp8 backward ragged".into(),
            });
        }
        if matches!(output_io_type, DataType::F16 | DataType::BF16)
            && (cudnn_version < 91300
                || self
                    .sm_version()
                    .is_some_and(|sm_version| sm_version / 10 < 10))
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa fp8 backward output data type".into(),
            });
        }

        let requires_deterministic_algorithm = self
            .sm_version()
            .is_some_and(|sm_version| sm_version / 10 == 10)
            && d_qk <= 192
            && d_v <= 128;
        let has_dropout = config.dropout().is_some()
            || config.modifiers().dropout_mask.is_some()
            || config.modifiers().dropout_scale.is_some();
        if requires_deterministic_algorithm {
            if cudnn_version < 91900 {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa fp8 backward deterministic algorithm version".into(),
                });
            }
            if has_dropout {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa fp8 backward deterministic algorithm dropout".into(),
                });
            }
        }
        if config.has_random_number_generator_dump() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa fp8 backward rng_dump".into(),
            });
        }

        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_fp8_forward_support_surface_for_version(
        &self,
        cudnn_version: u64,
        output_io_type: DataType,
    ) -> Result<()> {
        if matches!(output_io_type, DataType::F16 | DataType::BF16)
            && (cudnn_version < 91300
                || self
                    .sm_version()
                    .is_some_and(|sm_version| sm_version / 10 < 10))
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa fp8 output data type".into(),
            });
        }

        Ok(())
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
            _ => Err(Error::DescriptorMismatch {
                name: "sdpa fp8 backward output data type".into(),
            }),
        }
    }

    pub(in crate::frontend::composite::sdpa) fn validate_quantized_backward_head_dimensions(
        &self,
        d_qk: i64,
        d_v: i64,
        error_name: &str,
    ) -> Result<()> {
        if let Some(sm_version) = self.sm_version() {
            let prop_major = sm_version / 10;
            if prop_major >= 10 {
                let supported_blackwell_dims =
                    ((d_qk <= 128) && (d_qk % 16 == 0) && (d_v <= 128) && (d_v % 16 == 0))
                        || (d_qk == 192 && d_v == 128);
                if !supported_blackwell_dims {
                    return Err(Error::DescriptorMismatch {
                        name: error_name.into(),
                    });
                }
            } else {
                if d_qk != 128 || d_v != 128 || d_qk % 16 != 0 || d_v % 16 != 0 {
                    return Err(Error::DescriptorMismatch {
                        name: error_name.into(),
                    });
                }
            }
        }

        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn validate_quantized_backward_output_ragged_support(
        &self,
        d_q: TensorId,
        d_k: TensorId,
        d_v: TensorId,
        error_name: &str,
    ) -> Result<()> {
        let has_ragged_output = self.tensor_config(d_q)?.ragged_offset.is_some()
            || self.tensor_config(d_k)?.ragged_offset.is_some()
            || self.tensor_config(d_v)?.ragged_offset.is_some();
        if self
            .sm_version()
            .is_some_and(|sm_version| sm_version / 10 == 9)
            && has_ragged_output
        {
            return Err(Error::DescriptorMismatch {
                name: error_name.into(),
            });
        }

        Ok(())
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

        let validate = |tensor: &TensorSpec,
                        name: &str,
                        batch: i64,
                        heads: i64,
                        min_seq_scale: Option<i64>,
                        min_d_scale: Option<i64>|
         -> Result<()> {
            let dims = tensor.shape.dimensions();
            if dims.len() != 4 || dims[0] != batch || dims[1] != heads {
                return Err(Error::DescriptorMismatch { name: name.into() });
            }
            if let Some(min_seq_scale) = min_seq_scale
                && dims[2] < min_seq_scale
            {
                return Err(Error::DescriptorMismatch { name: name.into() });
            }
            if let Some(min_d_scale) = min_d_scale
                && dims[3] < min_d_scale
            {
                return Err(Error::DescriptorMismatch { name: name.into() });
            }
            Ok(())
        };

        validate(
            self.tensor_config(scale_q)?,
            "sdpa mxfp8 backward scale_q shape",
            q_dims[0],
            q_dims[1],
            None,
            Some(d_qk_scale),
        )?;
        validate(
            self.tensor_config(scale_q_t)?,
            "sdpa mxfp8 backward scale_q_t shape",
            q_dims[0],
            q_dims[1],
            Some(s_q_scale),
            None,
        )?;
        validate(
            self.tensor_config(scale_k)?,
            "sdpa mxfp8 backward scale_k shape",
            k_dims[0],
            k_dims[1],
            None,
            Some(d_qk_scale),
        )?;
        validate(
            self.tensor_config(scale_k_t)?,
            "sdpa mxfp8 backward scale_k_t shape",
            k_dims[0],
            k_dims[1],
            Some(s_kv_scale),
            None,
        )?;
        validate(
            self.tensor_config(scale_v)?,
            "sdpa mxfp8 backward scale_v shape",
            v_dims[0],
            v_dims[1],
            Some(s_kv_scale),
            None,
        )?;
        validate(
            self.tensor_config(scale_d_o)?,
            "sdpa mxfp8 backward scale_d_o shape",
            d_o_dims[0],
            d_o_dims[1],
            None,
            Some(d_v_scale),
        )?;
        validate(
            self.tensor_config(scale_d_o_t)?,
            "sdpa mxfp8 backward scale_d_o_t shape",
            d_o_dims[0],
            d_o_dims[1],
            Some(s_q_scale),
            None,
        )?;

        Ok(())
    }
}
