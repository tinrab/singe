use std::borrow::Cow;

use crate::{
    backend::tensor::{Shape, TensorId, TensorSpec},
    context::Context,
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::{
            SdpaForwardOutputTensors, SdpaInputs, SdpaOutputTensors, attention_heads_compatible,
            infer_sdpa_output_shape, infer_sdpa_scores_shape,
            support::{
                SDPA_DATA_TYPE, SDPA_DROPOUT_OFFSET, SDPA_DROPOUT_PROBABILITY, SDPA_DROPOUT_SEED,
                SDPA_LOGIT_MAX_SHAPE, SDPA_OUTPUT_LAYOUT, SDPA_OUTPUT_SHAPE, SDPA_Q_LAYOUT,
                SDPA_RNG_DUMP, SDPA_SCORE_SUM_EXP_SHAPE, SDPA_SINK_TOKEN_SHAPE,
                SDPA_STATS_DATA_TYPE, SDPA_STATS_SHAPE,
            },
        },
        error::{Error as FrontendError, SdpaFeature, SdpaRequirement},
        graph::{Graph, infer_paged_cache_output, reduce_last_axis},
        operation::{
            AttentionConfig, AttentionForwardRequest, AttentionImplementation, DirectSdpaDropout,
            DirectSdpaForwardConfig, Operation, PagedCacheLayout, PagedCacheLoadConfig,
            RandomNumberSeedSource, SdpaConfig, SdpaForwardQuantization, SdpaForwardTensors,
            SdpaOperation, SdpaScoreModifiers, SdpaScoreSubgraph,
        },
        require_baseline,
    },
    utility::check_range,
    version,
};

/// Forward expansion only accepts a concrete choice made from the complete request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedAttentionImplementation {
    Unified,
    Composite,
}

impl From<ResolvedAttentionImplementation> for AttentionImplementation {
    fn from(implementation: ResolvedAttentionImplementation) -> Self {
        match implementation {
            ResolvedAttentionImplementation::Unified => Self::Unified,
            ResolvedAttentionImplementation::Composite => Self::Composite,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectSdpaKvShapes {
    pub key: Shape,
    pub value: Shape,
}

impl DirectSdpaKvShapes {
    pub fn new(key: Shape, value: Shape) -> Self {
        Self { key, value }
    }
}

impl Graph {
    pub(crate) fn can_lower_attention_as_unified_with_tensors(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: &AttentionConfig,
    ) -> Result<bool> {
        if self.unified_attention_config_error(config).is_some() {
            return Ok(false);
        }

        let tensors = [
            self.tensor_spec(q)?,
            self.tensor_spec(k)?,
            self.tensor_spec(v)?,
        ];
        if !tensors.iter().all(|tensor| {
            matches!(
                tensor.data_type,
                DataType::F16 | DataType::BF16 | DataType::F8E4M3 | DataType::F8E5M2
            )
        }) {
            return Ok(false);
        }

        Ok(tensors
            .iter()
            .all(|tensor| tensor.shape.dimensions()[3] % 8 == 0))
    }

    /// Resolves a request using the same target architecture as graph preparation.
    /// An explicit graph target takes precedence over attached device properties and the supplied live context.
    /// Pass `None` for deviceless resolution with an explicit target or properties that retain their architecture.
    /// Leaves the graph recipe unchanged and does not construct native descriptors or query engine heuristics.
    /// Includes every direct operand and declared output, including quantization scales,
    /// addressing metadata and optional outputs. Score-subgraph bindings remain separate.
    /// Auxiliary presence follows the actual outputs, including caller-built or restored requests.
    ///
    /// # Errors
    /// Returns an error when the target architecture is unavailable or the requested implementation is unsupported.
    pub fn resolve_attention_implementation(
        &self,
        request: &AttentionForwardRequest,
        context: Option<&Context>,
    ) -> Result<AttentionImplementation> {
        let sm_version = self.compilation_sm_version(context)?;
        let mut graph = Cow::Borrowed(self);
        if sm_version != self.sm_version() {
            graph.to_mut().set_sm_version(sm_version);
        }
        graph.resolve_attention_request(request).map(Into::into)
    }

    pub(crate) fn resolve_attention_request(
        &self,
        request: &AttentionForwardRequest,
    ) -> Result<ResolvedAttentionImplementation> {
        request
            .config
            .modifiers()
            .sequence_inputs()
            .validate_forward()?;
        let [q, k, v] = request.inputs.query_key_value();
        self.validate_attention_forward_ragged_layout(
            request.config.max_total_sequence_lengths(),
            [q, k, v, request.outputs.output],
        )?;
        let output_data_type = self.tensor_spec(request.outputs.output)?.data_type;
        let uses = request.direct_tensor_uses();
        let mut config =
            self.normalized_attention_forward_config(&request.config, request.outputs)?;
        for tensor in uses.reads.into_iter().chain(uses.writes) {
            if self.tensor_spec(tensor)?.ragged_offset_multiplier()? != 1 {
                if config.implementation() == AttentionImplementation::Composite {
                    return Err(FrontendError::SdpaRequires {
                        requirement: SdpaRequirement::UnifiedImplementation,
                        operation: "sdpa ragged offset multipliers",
                    }
                    .into());
                }
                if config.implementation() != AttentionImplementation::Unified {
                    config = Cow::Owned(
                        config
                            .into_owned()
                            .with_implementation(AttentionImplementation::Unified),
                    );
                }
            }
        }
        if self.sm_version().is_none_or(|sm| sm <= 0) {
            return Err(FrontendError::SdpaRequires {
                requirement: SdpaRequirement::TargetArchitecture,
                operation: "sdpa implementation selection",
            }
            .into());
        }
        require_baseline(version()?.raw())?;
        self.validate_attention_config(q, k, v, &config)?;
        let native_output = matches!(
            output_data_type,
            DataType::F16 | DataType::BF16 | DataType::F8E4M3 | DataType::F8E5M2
        );
        match config.implementation() {
            AttentionImplementation::Auto => {
                if native_output
                    && self.can_lower_attention_as_unified_with_tensors(q, k, v, &config)?
                {
                    Ok(ResolvedAttentionImplementation::Unified)
                } else {
                    Self::validate_composite_attention_config(&config)?;
                    Ok(ResolvedAttentionImplementation::Composite)
                }
            }
            AttentionImplementation::Unified => {
                if let Some(error) = self.unified_attention_config_error(&config) {
                    return Err(error);
                }
                if native_output
                    && self.can_lower_attention_as_unified_with_tensors(q, k, v, &config)?
                {
                    Ok(ResolvedAttentionImplementation::Unified)
                } else {
                    Err(FrontendError::SdpaUnifiedImplementationUnavailable.into())
                }
            }
            AttentionImplementation::Composite => {
                Self::validate_composite_attention_config(&config)?;
                Ok(ResolvedAttentionImplementation::Composite)
            }
        }
    }

    fn unified_attention_config_error(&self, config: &AttentionConfig) -> Option<Error> {
        if self.dynamic_shape_enabled() {
            Some(
                FrontendError::SdpaUnsupported {
                    feature: SdpaFeature::DynamicShape,
                    operation: "unified sdpa",
                    sm_version: None,
                }
                .into(),
            )
        } else if config.modifiers().has_custom_dropout() {
            Some(
                FrontendError::SdpaUnsupported {
                    feature: SdpaFeature::CustomDropout,
                    operation: "unified sdpa",
                    sm_version: None,
                }
                .into(),
            )
        } else if config.dropout().is_some() && config.fused().aux_outputs().stats() {
            Some(
                FrontendError::SdpaUnsupported {
                    feature: SdpaFeature::DropoutWithGeneratedStats,
                    operation: "unified sdpa",
                    sm_version: None,
                }
                .into(),
            )
        } else {
            None
        }
    }

    fn validate_composite_attention_config(config: &AttentionConfig) -> Result<()> {
        if config.modifiers().sequence_inputs().has_cumulative() {
            return Err(FrontendError::SdpaUnsupported {
                operation: "composite sdpa",
                feature: SdpaFeature::CumulativeSequenceLengths,
                sm_version: None,
            }
            .into());
        }
        if config.block_mask().is_some() {
            return Err(FrontendError::SdpaRequires {
                requirement: SdpaRequirement::UnifiedImplementation,
                operation: "sdpa block mask",
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn infer_direct_sdpa_kv_shapes(
        &self,
        k: TensorId,
        v: TensorId,
        direct_config: &DirectSdpaForwardConfig,
        sequence_length_hint: Option<TensorId>,
    ) -> Result<DirectSdpaKvShapes> {
        self.validate_tensor_rank(k, 4, "attention key")?;
        self.validate_tensor_rank(v, 4, "attention value")?;
        let key = self.tensor_spec(k)?;
        let value = self.tensor_spec(v)?;
        let page_table_k = direct_config.key_page_table();
        let page_table_v = direct_config.value_page_table();
        if page_table_k.is_none() && page_table_v.is_none() {
            return Ok(DirectSdpaKvShapes::new(
                key.shape.clone(),
                value.shape.clone(),
            ));
        }
        for page_table in [page_table_k, page_table_v].into_iter().flatten() {
            self.validate_tensor_rank(page_table, 4, "attention page table")?;
        }

        // Keep K and V logical here. Only the composite key matmul consumes K^T.
        let sequence_length = match (
            direct_config.max_sequence_length_key_value(),
            page_table_k,
            page_table_v,
            sequence_length_hint,
        ) {
            (Some(length), ..) => length,
            (None, None, ..) => key.shape.dimensions()[2],
            (None, Some(_), None, _) => value.shape.dimensions()[2],
            (None, Some(_), Some(_), Some(hint)) => {
                self.validate_tensor_rank(hint, 4, "attention sequence length hint")?;
                self.tensor_spec(hint)?.shape.dimensions()[3]
            }
            (None, Some(k_table), Some(v_table), None) => {
                let capacity = |container: &TensorSpec, table: TensorId| -> Result<i64> {
                    container.shape.dimensions()[2]
                        .checked_mul(self.tensor_spec(table)?.shape.dimensions()[2])
                        .ok_or_else(|| Error::out_of_range("paged cache sequence length"))
                };
                capacity(key, k_table)?.min(capacity(value, v_table)?)
            }
        };
        let reconstruct = |container: &TensorSpec, table: Option<TensorId>| -> Result<Shape> {
            let Some(table) = table else {
                return Ok(container.shape.clone());
            };
            let table = self.tensor_spec(table)?;
            let sequence = Shape::contiguous([table.shape.dimensions()[0], 1, 1, 1])?;
            infer_paged_cache_output(
                &container.shape,
                &sequence,
                &table.shape,
                &PagedCacheLoadConfig::new(PagedCacheLayout::SequenceHidden)
                    .with_max_sequence_length(sequence_length),
            )
        };
        Ok(DirectSdpaKvShapes::new(
            reconstruct(key, page_table_k)?,
            reconstruct(value, page_table_v)?,
        ))
    }

    pub(crate) fn attention_kv_shapes(
        &self,
        k: TensorId,
        v: TensorId,
        config: &AttentionConfig,
    ) -> Result<DirectSdpaKvShapes> {
        let mut direct = DirectSdpaForwardConfig::new();
        let paged = config.paged();
        if let Some(cache) = paged.k() {
            direct = direct.with_key_page_table(cache.page_table());
        }
        if let Some(cache) = paged.v() {
            direct = direct.with_value_page_table(cache.page_table());
        }
        if let Some(length) = paged.max_sequence_length_key_value() {
            direct = direct.with_max_sequence_length_key_value(length);
        }
        self.infer_direct_sdpa_kv_shapes(k, v, &direct, config.modifiers().bias())
    }

    pub(crate) fn sdpa_unified_infer(
        &mut self,
        inputs: SdpaInputs,
        output_data_type: DataType,
        mut quantization: SdpaForwardQuantization<TensorId>,
        config: &AttentionConfig,
    ) -> Result<SdpaForwardOutputTensors> {
        let SdpaInputs {
            query: q,
            key: k,
            value: v,
            ..
        } = inputs;
        let fused = config.fused();
        let mut direct_config = DirectSdpaForwardConfig::new()
            .with_stats_base(config.softmax().stats_base())
            .with_unfuse_fma(config.unfuse_fma())
            .with_sequence_inputs(config.modifiers().sequence_inputs())
            .with_padding_mask(config.modifiers().padding_mask());
        let compute_type = self.resolve_compute_data_type(config.softmax().compute_type());
        direct_config
            .inherit_data_type_policy(self.config.data_type_policy.with_compute(compute_type));
        let paged = config.paged();
        if let Some(cache) = paged.k() {
            direct_config = direct_config.with_key_page_table(cache.page_table());
        }
        if let Some(cache) = paged.v() {
            direct_config = direct_config.with_value_page_table(cache.page_table());
        }
        if let Some(length) = paged.max_sequence_length_key_value() {
            direct_config = direct_config.with_max_sequence_length_key_value(length);
        }
        if let Some(block_mask) = config.block_mask() {
            direct_config = direct_config.with_block_mask(block_mask);
        }
        let q_tensor = self.tensor_spec(q)?.clone();
        let kv_shapes = self.attention_kv_shapes(k, v, config)?;
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &kv_shapes.key)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, &kv_shapes.value)?;
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        let aux_shape = reduce_last_axis(&scores_shape)?;
        let outputs =
            self.sdpa_requested_aux_output_tensors(output, fused, q_tensor.data_type, &aux_shape);
        if fused.aux_outputs().absolute_max_s() && quantization.absolute_max_probabilities.is_none()
        {
            quantization.absolute_max_probabilities = Some(self.tensor(TensorSpec::new(
                DataType::F32,
                Shape::contiguous([1, 1, 1, 1])?,
            )));
        }
        if fused.aux_outputs().absolute_max_o() && quantization.absolute_max_output.is_none() {
            quantization.absolute_max_output = Some(self.tensor(TensorSpec::new(
                DataType::F32,
                Shape::contiguous([1, 1, 1, 1])?,
            )));
        }
        let absolute_max_probabilities = quantization.absolute_max_probabilities;
        let absolute_max_output = quantization.absolute_max_output;
        if quantization != SdpaForwardQuantization::default() {
            direct_config = direct_config.with_quantization(quantization);
        }
        let rng_dump = if config.dropout().is_some() && fused.aux_outputs().rng_dump() {
            Some(self.tensor(TensorSpec::new(DataType::F32, scores_shape)))
        } else {
            None
        };
        if let Some(dropout) = config.dropout() {
            let (seed, offset) = match dropout.seed_source() {
                RandomNumberSeedSource::Device { tensor, offset } => (tensor, offset),
                RandomNumberSeedSource::Host { value, offset } => {
                    let seed = self.tensor(
                        TensorSpec::constant(value).with_shape(Shape::contiguous([1, 1, 1, 1])?),
                    );
                    let offset = match offset {
                        Some(offset) => offset,
                        None => self.tensor(
                            TensorSpec::constant(0_i64)
                                .with_shape(Shape::contiguous([1, 1, 1, 1])?),
                        ),
                    };
                    (seed, offset)
                }
            };
            let mut direct_dropout = DirectSdpaDropout::new(dropout.probability(), seed, offset);
            if let Some(rng_dump) = rng_dump {
                direct_dropout = direct_dropout.with_random_number_generator_dump(rng_dump);
            }
            direct_config = direct_config.with_dropout(direct_dropout);
        }
        self.sdpa_direct(
            inputs,
            outputs,
            *config.modifiers(),
            config.score_subgraph().cloned(),
            fused,
            &direct_config,
        )?;
        Ok(SdpaForwardOutputTensors {
            output,
            stats: outputs.stats,
            logit_max: outputs.logit_max,
            score_sum_exp: outputs.score_sum_exp,
            rng_dump,
            absolute_max_probabilities,
            absolute_max_output,
        })
    }

    fn validate_direct_sdpa_dropout(
        &self,
        direct_config: &DirectSdpaForwardConfig,
        scores_shape: &Shape,
    ) -> Result<()> {
        let Some(dropout) = direct_config.dropout() else {
            return Ok(());
        };
        check_range!(
            SDPA_DROPOUT_PROBABILITY,
            (0.0..1.0).contains(&dropout.probability())
        )?;
        self.validate_direct_sdpa_rng_scalar(dropout.seed(), SDPA_DROPOUT_SEED)?;
        self.validate_direct_sdpa_rng_scalar(dropout.offset(), SDPA_DROPOUT_OFFSET)?;
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

    fn validate_direct_sdpa_sink_token(
        &self,
        sink: Option<TensorId>,
        attention_heads: i64,
    ) -> Result<()> {
        let Some(sink) = sink else {
            return Ok(());
        };
        let expected_sink_shape = Shape::contiguous([1, attention_heads, 1, 1])?;
        self.validate_sdpa_tensor_dimensions_match(
            sink,
            expected_sink_shape.dimensions(),
            SDPA_SINK_TOKEN_SHAPE,
        )
    }

    fn validate_direct_sdpa_rng_scalar(
        &self,
        tensor: TensorId,
        error_name: &'static str,
    ) -> Result<()> {
        self.validate_sdpa_rank4_scalar_data_type_supported(
            tensor,
            &[DataType::I32, DataType::I64],
            error_name,
            error_name,
        )
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the composite attention operation builds and validates its complete tensor graph"
    )]
    pub(crate) fn sdpa_direct(
        &mut self,
        inputs: SdpaInputs,
        outputs: SdpaOutputTensors,
        score_modifiers: SdpaScoreModifiers,
        score_subgraph: Option<SdpaScoreSubgraph>,
        config: &SdpaConfig,
        direct_config: &DirectSdpaForwardConfig,
    ) -> Result<()> {
        direct_config.sequence_inputs().validate_forward()?;
        let SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        } = inputs;
        let SdpaOutputTensors {
            output: o,
            stats,
            logit_max,
            score_sum_exp,
        } = outputs;
        let sink = score_modifiers.sink_token();
        self.promote_index_tensors(direct_config.sequence_inputs().tensors())?;
        self.validate_attention_sequence_inputs(
            direct_config.sequence_inputs(),
            self.tensor_spec(q)?.shape.dimensions()[0],
        )?;
        let compute_type = direct_config
            .data_type_policy()
            .with_defaults(self.config.data_type_policy)
            .resolve_compute(None);
        let scale = self.sdpa_effective_scale(scale, config.attention_scale(), compute_type)?;

        let q_tensor = self.tensor_spec(q)?;
        let sequence_length_hint = score_modifiers.bias().or_else(|| {
            direct_config
                .dropout()
                .and_then(DirectSdpaDropout::random_number_generator_dump)
        });
        let kv_shapes =
            self.infer_direct_sdpa_kv_shapes(k, v, direct_config, sequence_length_hint)?;

        self.validate_tensor_data_type_supported(
            o,
            &[
                DataType::F16,
                DataType::BF16,
                DataType::F8E4M3,
                DataType::F8E5M2,
            ],
            SDPA_DATA_TYPE,
        )?;
        self.validate_sdpa_tensor_layout(q, SDPA_Q_LAYOUT)?;
        self.validate_sdpa_k_layout(k)?;
        self.validate_sdpa_k_layout(v)?;
        self.validate_sdpa_tensor_layout(o, SDPA_OUTPUT_LAYOUT)?;
        if !attention_heads_compatible(
            q_tensor.shape.dimensions()[1],
            kv_shapes.key.dimensions()[1],
        ) || !attention_heads_compatible(
            q_tensor.shape.dimensions()[1],
            kv_shapes.value.dimensions()[1],
        ) {
            return Err(FrontendError::SdpaGroupedQueryAttentionHeadsMismatch {
                query_heads: q_tensor.shape.dimensions()[1],
                key_heads: kv_shapes.key.dimensions()[1],
                value_heads: kv_shapes.value.dimensions()[1],
                operation: "sdpa",
            }
            .into());
        }
        self.validate_sdpa_direct_ragged_support(
            q,
            k,
            v,
            o,
            direct_config,
            score_subgraph.is_some(),
        )?;
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &kv_shapes.key)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, &kv_shapes.value)?;
        self.validate_sdpa_tensor_dimensions_match(
            o,
            output_shape.dimensions(),
            SDPA_OUTPUT_SHAPE,
        )?;
        let expected_aux_shape = reduce_last_axis(&scores_shape)?;
        for (tensor, name) in [
            (stats, SDPA_STATS_SHAPE),
            (logit_max, SDPA_LOGIT_MAX_SHAPE),
            (score_sum_exp, SDPA_SCORE_SUM_EXP_SHAPE),
        ] {
            if let Some(tensor) = tensor {
                self.validate_sdpa_tensor_dimensions_match(
                    tensor,
                    expected_aux_shape.dimensions(),
                    name,
                )?;
            }
        }
        if let Some(stats) = stats {
            self.validate_sdpa_tensor_data_type_match(stats, DataType::F32, SDPA_STATS_DATA_TYPE)?;
        }
        self.validate_direct_sdpa_dropout(direct_config, &scores_shape)?;
        let needs_softmax =
            stats.is_some() || logit_max.is_some() || score_sum_exp.is_some() || sink.is_some();
        self.validate_direct_sdpa_sink_token(sink, q_tensor.shape.dimensions()[1])?;
        let (softmax_p, softmax_s) = if needs_softmax {
            let (input, output) = self.sdpa_softmax_descriptor_tensors(
                scores_shape,
                self.effective_intermediate_data_type(DataType::F32),
            );
            (Some(input), Some(output))
        } else {
            (None, None)
        };

        self.operations
            .push(Operation::Sdpa(SdpaOperation::Forward {
                tensors: SdpaForwardTensors {
                    query: q,
                    key: k,
                    value: v,
                    output: o,
                    scale,
                },
                stats,
                logit_max,
                score_sum_exp,
                softmax_p,
                softmax_s,
                sink,
                score_modifiers,
                score_subgraph: score_subgraph.map(Box::new),
                config: Box::new(*direct_config),
            }));

        Ok(())
    }

    pub(crate) fn sdpa_softmax_descriptor_tensors(
        &mut self,
        shape: Shape,
        data_type: DataType,
    ) -> (TensorId, TensorId) {
        let tensor = TensorSpec::new(data_type, shape).virtual_tensor();
        (self.tensor(tensor.clone()), self.tensor(tensor))
    }
}
