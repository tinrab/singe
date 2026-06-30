use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::{
            SdpaAuxOutputs,
            support::{
                SDPA_DATA_TYPE, SDPA_DROPOUT_OFFSET, SDPA_DROPOUT_PROBABILITY, SDPA_DROPOUT_SEED,
                SDPA_LOGIT_MAX, SDPA_LOGIT_MAX_DATA_TYPE, SDPA_LOGIT_MAX_SHAPE, SDPA_OUTPUT_LAYOUT,
                SDPA_OUTPUT_SHAPE, SDPA_Q_LAYOUT, SDPA_RNG_DUMP, SDPA_SCALE, SDPA_SCORE_SUM_EXP,
                SDPA_SCORE_SUM_EXP_DATA_TYPE, SDPA_SCORE_SUM_EXP_SHAPE, SDPA_SINK_TOKEN_SHAPE,
                SDPA_SOFTMAX_P, SDPA_SOFTMAX_S, SDPA_STATS, SDPA_STATS_DATA_TYPE, SDPA_STATS_SHAPE,
            },
        },
        graph::Graph,
        infer::*,
        operation::*,
        support,
    },
    scalar::ScalarValue,
    tensor::{Shape, TensorId, TensorSpec},
    utility::check_range,
    version,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::frontend::composite::sdpa) struct DirectSdpaKvShapes {
    pub key: Shape,
    pub value: Shape,
}

impl DirectSdpaKvShapes {
    pub fn new(key: Shape, value: Shape) -> Self {
        Self { key, value }
    }
}

impl Graph {
    pub(in crate::frontend::composite::sdpa) fn can_lower_attention_as_unified_for_version(
        &self,
        cudnn_version: u64,
        config: AttentionConfig,
    ) -> bool {
        let config = self.normalize_attention_config_for_forward_version(cudnn_version, config);
        support::UnifiedSdpaForwardSupport::new(cudnn_version).supports_config(&config)
    }

    pub(in crate::frontend::composite::sdpa) fn can_lower_attention_as_unified_with_tensors_for_version(
        &self,
        cudnn_version: u64,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: AttentionConfig,
    ) -> Result<bool> {
        if self.dynamic_shape_enabled() || self.override_shape_enabled() {
            return Ok(false);
        }

        if !self.can_lower_attention_as_unified_for_version(cudnn_version, config.clone()) {
            return Ok(false);
        }

        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let q_data_type = q_tensor.data_type;
        let k_data_type = k_tensor.data_type;
        let v_data_type = self.tensor_config(v)?.data_type;

        if !matches!(q_data_type, DataType::F16 | DataType::BF16)
            || q_data_type != k_data_type
            || q_data_type != v_data_type
            || config.softmax().compute_type() != DataType::F32
        {
            return Ok(false);
        }

        let paged = config.paged();
        if let Some(cache) = paged.k()
            && self
                .tensor_config(cache.page_table())?
                .ragged_offset
                .is_some()
        {
            return Ok(false);
        }
        if let Some(cache) = paged.v()
            && self
                .tensor_config(cache.page_table())?
                .ragged_offset
                .is_some()
        {
            return Ok(false);
        }
        Ok(q_tensor.shape.dimensions()[3] % 8 == 0 && k_tensor.shape.dimensions()[3] % 8 == 0)
    }

    pub fn resolve_attention_implementation(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: AttentionConfig,
    ) -> Result<AttentionImplementation> {
        self.resolve_attention_implementation_for_version(version()?.raw(), q, k, v, config)
    }

    pub(crate) fn resolve_attention_implementation_for_version(
        &self,
        cudnn_version: u64,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: AttentionConfig,
    ) -> Result<AttentionImplementation> {
        let config = self.normalize_attention_config_for_forward_version(cudnn_version, config);
        match config.implementation() {
            AttentionImplementation::Auto => {
                if self.can_lower_attention_as_unified_with_tensors_for_version(
                    cudnn_version,
                    q,
                    k,
                    v,
                    config.clone(),
                )? || (self.override_shape_enabled()
                    && !self.dynamic_shape_enabled()
                    && self.can_lower_attention_as_unified_for_version(cudnn_version, config))
                {
                    Ok(AttentionImplementation::Unified)
                } else {
                    Ok(AttentionImplementation::Composite)
                }
            }
            AttentionImplementation::Unified => {
                if self.can_lower_attention_as_unified_with_tensors_for_version(
                    cudnn_version,
                    q,
                    k,
                    v,
                    config,
                )? {
                    Ok(AttentionImplementation::Unified)
                } else {
                    Err(Error::FrontendSdpaUnifiedImplementationUnavailable)
                }
            }
            AttentionImplementation::Composite => Ok(AttentionImplementation::Composite),
        }
    }

    pub(in crate::frontend::composite::sdpa) fn infer_direct_sdpa_kv_shapes(
        &self,
        k: TensorId,
        v: TensorId,
        direct_config: &DirectSdpaForwardConfig,
        max_sequence_length_key_value: Option<i64>,
    ) -> Result<DirectSdpaKvShapes> {
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();

        let mut k_shape = if let Some(page_table_k) = direct_config.page_table_k() {
            let sequence_length_key_value = direct_config
                .sequence_length_key_value()
                .ok_or(Error::FrontendSdpaPagedAttentionRequiresSequenceLengths)?;
            let seq_len_kv_tensor = self.tensor_config(sequence_length_key_value)?.clone();
            let page_table_k_tensor = self.tensor_config(page_table_k)?.clone();
            infer_paged_cache_output(
                &k_tensor.shape,
                &seq_len_kv_tensor.shape,
                &page_table_k_tensor.shape,
                true,
            )?
        } else {
            k_tensor.shape.clone()
        };

        let mut v_shape = if let Some(page_table_v) = direct_config.page_table_v() {
            let sequence_length_key_value = direct_config
                .sequence_length_key_value()
                .ok_or(Error::FrontendSdpaPagedAttentionRequiresSequenceLengths)?;
            let seq_len_kv_tensor = self.tensor_config(sequence_length_key_value)?.clone();
            let page_table_v_tensor = self.tensor_config(page_table_v)?.clone();
            infer_paged_cache_output(
                &v_tensor.shape,
                &seq_len_kv_tensor.shape,
                &page_table_v_tensor.shape,
                false,
            )?
        } else {
            v_tensor.shape.clone()
        };

        if let Some(max_sequence_length_key_value) = max_sequence_length_key_value {
            if let Some(page_table_k) = direct_config.page_table_k() {
                let _ = page_table_k;
                let k_dims = k_shape.dimensions();
                k_shape = infer_slice_output(
                    &k_shape,
                    &[
                        (0, k_dims[0]),
                        (0, k_dims[1]),
                        (0, k_dims[2]),
                        (0, max_sequence_length_key_value),
                    ],
                )?;
            }
            if let Some(page_table_v) = direct_config.page_table_v() {
                let _ = page_table_v;
                let v_dims = v_shape.dimensions();
                v_shape = infer_slice_output(
                    &v_shape,
                    &[
                        (0, v_dims[0]),
                        (0, v_dims[1]),
                        (0, max_sequence_length_key_value),
                        (0, v_dims[3]),
                    ],
                )?;
            }
        }

        Ok(DirectSdpaKvShapes::new(k_shape, v_shape))
    }

    pub(in crate::frontend::composite::sdpa) fn sdpa_unified_infer(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        scale: TensorId,
        config: AttentionConfig,
    ) -> Result<SdpaAuxOutputs> {
        let config = self.normalize_attention_config_for_forward_version(version()?.raw(), config);
        let fused = config.fused();
        let mut direct_config = DirectSdpaForwardConfig::new();
        if config.unfuse_fma() {
            direct_config = direct_config.with_unfused_fma();
        }
        if let Some((sequence_length_query, sequence_length_key_value)) =
            config.modifiers().sequence_lengths()
        {
            let sequence = if config.modifiers().padding_mask() {
                DirectSdpaSequenceMode::padding_mask(
                    sequence_length_query,
                    sequence_length_key_value,
                )
            } else {
                DirectSdpaSequenceMode::sequence_lengths(
                    sequence_length_query,
                    sequence_length_key_value,
                )
            };
            direct_config = direct_config.with_sequence_mode(sequence);
        }
        let paged = config.paged();
        if let Some(cache) = paged.k() {
            direct_config = direct_config.with_page_table_k(cache.page_table());
        }
        if let Some(cache) = paged.v() {
            direct_config = direct_config.with_page_table_v(cache.page_table());
        }
        if let Some(block_mask) = config.block_mask() {
            direct_config = direct_config.with_block_mask(block_mask);
        }
        let q_tensor = self.tensor_config(q)?.clone();
        let kv_shapes = self.infer_direct_sdpa_kv_shapes(
            k,
            v,
            &direct_config,
            paged.max_sequence_length_key_value(),
        )?;
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &kv_shapes.key)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, &kv_shapes.value)?;
        let output = self.tensor(TensorSpec::new(q_tensor.data_type, output_shape));
        let aux_data_type = self.sdpa_aux_data_type();
        let aux_shape = reduce_last_axis(&scores_shape)?;
        let outputs = self.sdpa_requested_aux_output_tensors(
            output,
            fused.aux_outputs().softmax(),
            aux_data_type,
            &aux_shape,
        );
        let sink = config.modifiers().sink_token();
        let (softmax_p, softmax_s) = if support::SDPA_UNIFIED_AUX_SOFTMAX
            .is_supported(version()?.raw())
            && (outputs.stats.is_some()
                || outputs.logit_max.is_some()
                || outputs.score_sum_exp.is_some()
                || sink.is_some())
        {
            let softmax_shape = Shape::contiguous(scores_shape.dimensions().to_vec())?;
            (
                Some(self.tensor(
                    TensorSpec::new(aux_data_type, softmax_shape.clone()).virtual_tensor(),
                )),
                Some(self.tensor(TensorSpec::new(aux_data_type, softmax_shape).virtual_tensor())),
            )
        } else {
            (None, None)
        };
        let rng_dump = if config.dropout().is_some() {
            Some(self.tensor(TensorSpec::new(DataType::F32, scores_shape.clone())))
        } else {
            None
        };
        if let Some(dropout) = config.dropout() {
            let seed = if let Some(seed) = dropout.seed_tensor() {
                seed
            } else {
                let seed_value = dropout
                    .seed()
                    .ok_or(Error::FrontendSdpaDropoutSeedRequired)?;
                self.tensor(
                    TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
                        .with_scalar_value(ScalarValue::I64(seed_value))?,
                )
            };
            let mut direct_dropout = DirectSdpaDropout::new(dropout).with_seed(seed);
            if let Some(rng_dump) = rng_dump {
                direct_dropout = direct_dropout.with_random_number_generator_dump(rng_dump);
            }
            direct_config = direct_config.with_dropout(direct_dropout);
        }
        self.sdpa_direct(
            q,
            k,
            v,
            scale,
            output,
            outputs.stats,
            outputs.logit_max,
            outputs.score_sum_exp,
            softmax_p,
            softmax_s,
            sink,
            *config.modifiers(),
            config.score_subgraph().cloned(),
            fused,
            direct_config,
            config.paged().max_sequence_length_key_value(),
        )?;
        Ok(SdpaAuxOutputs::with_aux(
            output,
            outputs.stats,
            outputs.logit_max,
            outputs.score_sum_exp,
            rng_dump,
        ))
    }

    fn validate_direct_sdpa_dropout(
        &self,
        direct_config: &DirectSdpaForwardConfig,
        scores_shape: &Shape,
    ) -> Result<()> {
        let Some(dropout) = direct_config.dropout() else {
            return Ok(());
        };
        let dropout_config = dropout.config();

        check_range!(
            SDPA_DROPOUT_PROBABILITY,
            (0.0..1.0).contains(&dropout_config.probability())
        )?;
        if let Some(dropout_seed) = dropout.seed() {
            self.validate_direct_sdpa_rng_scalar(dropout_seed, SDPA_DROPOUT_SEED)?;
        }
        if let Some(dropout_offset) = dropout_config.offset() {
            self.validate_direct_sdpa_rng_scalar(dropout_offset, SDPA_DROPOUT_OFFSET)?;
        }
        if let Some(rng_dump) = dropout.random_number_generator_dump() {
            self.validate_sdpa_tensor_dimensions_and_data_type(
                rng_dump,
                scores_shape.dimensions(),
                DataType::F32,
                SDPA_RNG_DUMP,
            )?;
        }
        Ok(())
    }

    fn validate_direct_sdpa_softmax_descriptors(
        &self,
        softmax_p: Option<TensorId>,
        softmax_s: Option<TensorId>,
        expected_shape: &Shape,
        expected_data_type: DataType,
    ) -> Result<bool> {
        let (Some(softmax_p), Some(softmax_s)) = (softmax_p, softmax_s) else {
            if softmax_p.is_some() || softmax_s.is_some() {
                return Err(Error::FrontendSdpaUnifiedSoftmaxDescriptorsIncomplete);
            }
            return Ok(false);
        };

        self.validate_direct_sdpa_softmax_tensor(
            softmax_p,
            expected_shape,
            expected_data_type,
            SDPA_SOFTMAX_P,
        )?;
        self.validate_direct_sdpa_softmax_tensor(
            softmax_s,
            expected_shape,
            expected_data_type,
            SDPA_SOFTMAX_S,
        )?;
        Ok(true)
    }

    fn validate_direct_sdpa_softmax_tensor(
        &self,
        tensor: TensorId,
        expected_shape: &Shape,
        expected_data_type: DataType,
        error_name: &str,
    ) -> Result<()> {
        self.validate_sdpa_tensor_shape_and_data_types(
            tensor,
            expected_shape,
            &[expected_data_type],
            error_name,
        )
    }

    fn validate_direct_sdpa_sink_token(
        &self,
        sink: Option<TensorId>,
        attention_heads: i64,
        has_softmax_descriptors: bool,
    ) -> Result<()> {
        let Some(sink) = sink else {
            return Ok(());
        };
        if !has_softmax_descriptors {
            return Err(Error::FrontendSdpaUnifiedSoftmaxDescriptorsIncomplete);
        }

        let expected_sink_shape = Shape::contiguous([1, attention_heads, 1, 1])?;
        self.validate_sdpa_tensor_shape_and_data_types(
            sink,
            &expected_sink_shape,
            &[DataType::F32],
            SDPA_SINK_TOKEN_SHAPE,
        )
    }

    fn validate_direct_sdpa_rng_scalar(&self, tensor: TensorId, error_name: &str) -> Result<()> {
        self.validate_sdpa_rank4_scalar_data_type_supported(
            tensor,
            &[DataType::I32, DataType::I64],
            error_name,
            error_name,
        )
    }

    pub(in crate::frontend::composite::sdpa) fn sdpa_direct(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        scale: TensorId,
        o: TensorId,
        stats: Option<TensorId>,
        logit_max: Option<TensorId>,
        score_sum_exp: Option<TensorId>,
        softmax_p: Option<TensorId>,
        softmax_s: Option<TensorId>,
        sink: Option<TensorId>,
        score_modifiers: SdpaScoreModifiers,
        score_subgraph: Option<SdpaScoreSubgraph>,
        config: &SdpaConfig,
        direct_config: DirectSdpaForwardConfig,
        max_sequence_length_key_value: Option<i64>,
    ) -> Result<()> {
        let scale = self.sdpa_effective_scale(scale, config.attention_scale())?;

        let q_tensor = self.tensor_config(q)?.clone();
        let kv_shapes =
            self.infer_direct_sdpa_kv_shapes(k, v, &direct_config, max_sequence_length_key_value)?;

        self.validate_sdpa_tensor_data_type_match(o, q_tensor.data_type, SDPA_DATA_TYPE)?;
        self.validate_sdpa_tensor_layout(q, SDPA_Q_LAYOUT)?;
        self.validate_sdpa_k_layout(k)?;
        self.validate_sdpa_k_layout(v)?;
        self.validate_sdpa_tensor_layout(o, SDPA_OUTPUT_LAYOUT)?;
        if q_tensor.shape.dimensions()[1] % kv_shapes.key.dimensions()[1] != 0
            || q_tensor.shape.dimensions()[1] % kv_shapes.value.dimensions()[1] != 0
        {
            return Err(Error::FrontendSdpaGroupedQueryAttentionHeadsMismatch {
                query_heads: q_tensor.shape.dimensions()[1],
                key_heads: kv_shapes.key.dimensions()[1],
                value_heads: kv_shapes.value.dimensions()[1],
            });
        }
        self.validate_sdpa_direct_ragged_support_for_version(
            version()?.raw(),
            q,
            k,
            v,
            o,
            &direct_config,
            score_subgraph.is_some(),
        )?;
        self.validate_sdpa_scalar_element_count(scale, SDPA_SCALE)?;
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &kv_shapes.key)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, &kv_shapes.value)?;
        let aux_data_type = self.sdpa_aux_data_type();
        let expected_softmax_shape = Shape::contiguous(scores_shape.dimensions().to_vec())?;
        self.validate_sdpa_tensor_dimensions_match(
            o,
            output_shape.dimensions(),
            SDPA_OUTPUT_SHAPE,
        )?;
        let expected_aux_shape = reduce_last_axis(&scores_shape)?;
        let softmax_aux = config.aux_outputs().softmax();
        self.validate_optional_sdpa_tensor_dimensions_and_data_type(
            stats,
            softmax_aux.stats(),
            expected_aux_shape.dimensions(),
            aux_data_type,
            SDPA_STATS,
            SDPA_STATS_SHAPE,
            SDPA_STATS_DATA_TYPE,
        )?;
        self.validate_optional_sdpa_tensor_dimensions_and_data_type(
            logit_max,
            softmax_aux.logit_max(),
            expected_aux_shape.dimensions(),
            aux_data_type,
            SDPA_LOGIT_MAX,
            SDPA_LOGIT_MAX_SHAPE,
            SDPA_LOGIT_MAX_DATA_TYPE,
        )?;
        self.validate_optional_sdpa_tensor_dimensions_and_data_type(
            score_sum_exp,
            softmax_aux.score_sum_exp(),
            expected_aux_shape.dimensions(),
            aux_data_type,
            SDPA_SCORE_SUM_EXP,
            SDPA_SCORE_SUM_EXP_SHAPE,
            SDPA_SCORE_SUM_EXP_DATA_TYPE,
        )?;
        self.validate_direct_sdpa_dropout(&direct_config, &scores_shape)?;
        let has_softmax_descriptors = self.validate_direct_sdpa_softmax_descriptors(
            softmax_p,
            softmax_s,
            &expected_softmax_shape,
            aux_data_type,
        )?;
        self.validate_direct_sdpa_sink_token(
            sink,
            q_tensor.shape.dimensions()[1],
            has_softmax_descriptors,
        )?;

        self.operations
            .push(Operation::Sdpa(SdpaOperation::Forward {
                q,
                k,
                v,
                o,
                scale,
                stats,
                logit_max,
                score_sum_exp,
                softmax_p,
                softmax_s,
                sink,
                score_modifiers,
                score_subgraph: Box::new(score_subgraph),
                config: Box::new(direct_config),
            }));

        Ok(())
    }
}
