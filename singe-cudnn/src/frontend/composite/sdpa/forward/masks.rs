use crate::{
    backend::{
        execution::tensor_ops::ReshapeMode,
        pointwise::PointwiseMode,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::{
            SdpaForwardOutputTensors, SdpaInputs, SdpaOutputTensors,
            forward::selection::ResolvedAttentionImplementation,
            infer_sdpa_output_shape, infer_sdpa_scores_shape,
            support::{
                SDPA_PADDING_MASK_SEQ_LEN_DATA_TYPE, bounded_sdpa_window, sdpa_mask_scalar,
                sdpa_scalar_tensor,
            },
        },
        error::{Error as FrontendError, SdpaRequirement},
        graph::{DataTypePolicy, Graph, TensorOutputConfig, reduce_last_axis},
        operation::{
            AttentionConfig, AttentionDiagonalAlignment, AttentionDiagonalBand,
            AttentionDropoutMask, AttentionForwardInputs, AttentionMaskMode,
            AttentionSequenceInputs, DirectSdpaForwardConfig, PagedCacheLayout,
            PagedCacheLoadConfig, PointwiseOperation, ReshapeConfig, SdpaConfig,
            SdpaForwardQuantization, SdpaFusedMaskMode, SdpaQuantizedAuxOutputRequest,
            SdpaScoreModifiers, SoftmaxConfig,
        },
    },
    math::NanPropagation,
    scalar::ScalarValue,
    utility::{check_range, to_i64},
};

/// Prepared K/V operands in logical `[batch, heads, sequence, hidden]` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedAttentionKv {
    pub key: TensorId,
    pub value: TensorId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SdpaScoreIndices {
    row: TensorId,
    col: TensorId,
}

impl PreparedAttentionKv {
    pub fn new(key: TensorId, value: TensorId) -> Self {
        Self { key, value }
    }
}

impl Graph {
    fn sdpa_score_indices(
        &mut self,
        scores: TensorId,
        index_shape: &Shape,
        row_nan_propagation: NanPropagation,
        col_nan_propagation: NanPropagation,
    ) -> Result<SdpaScoreIndices> {
        let row = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        let col = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        self.pointwise_into(PointwiseOperation::Unary {
            name: None,
            mode: PointwiseMode::GenIndex,
            input: scores,
            output: row,
            compute_type: Some(DataType::I32),
            nan_propagation: row_nan_propagation,
            alpha1: 1.0,
            axis: Some(2),
        })?;
        self.pointwise_into(PointwiseOperation::Unary {
            name: None,
            mode: PointwiseMode::GenIndex,
            input: scores,
            output: col,
            compute_type: Some(DataType::I32),
            nan_propagation: col_nan_propagation,
            alpha1: 1.0,
            axis: Some(3),
        })?;
        Ok(SdpaScoreIndices { row, col })
    }

    pub(crate) fn sdpa_requested_aux_output_tensors(
        &mut self,
        output: TensorId,
        config: &SdpaConfig,
        default_io_type: DataType,
        shape: &Shape,
    ) -> SdpaOutputTensors {
        let request = config.aux_outputs().softmax();
        let data_type = self.effective_io_data_type(default_io_type);
        let stats = request.stats().then(|| {
            self.inferred_tensor(
                TensorSpec::new(DataType::F32, shape.clone()),
                config.name(),
                "Stats",
            )
        });
        let logit_max = request.logit_max().then(|| {
            self.inferred_tensor(
                TensorSpec::new(data_type, shape.clone()),
                config.name(),
                "Max",
            )
        });
        let score_sum_exp = request.score_sum_exp().then(|| {
            self.inferred_tensor(
                TensorSpec::new(data_type, shape.clone()),
                config.name(),
                "Sum_exp",
            )
        });
        SdpaOutputTensors {
            output,
            stats,
            logit_max,
            score_sum_exp,
        }
    }

    /// Reinterprets the trailing axes through a native view-only reshape for every placement.
    pub(crate) fn transpose_last_two_view(&mut self, input: TensorId) -> Result<TensorId> {
        let input_tensor = self.tensor_spec(input)?;
        let shape = input_tensor.shape.transpose_last_two()?;
        let rank = to_i64(input_tensor.shape.rank(), "tensor rank")?;
        let vectorization = input_tensor.vectorization()?.map(|(count, dimension)| {
            let dimension = if dimension == rank - 1 {
                rank - 2
            } else if dimension == rank - 2 {
                rank - 1
            } else {
                dimension
            };
            (count, dimension)
        });
        let mut tensor = input_tensor.clone().with_shape(shape).virtual_tensor();
        tensor.name = None;
        tensor.vector_count = vectorization.map(|(count, _)| count);
        tensor.vectorized_dimension = vectorization.map(|(_, dimension)| dimension);
        let checkpoint = self.mutation_checkpoint();
        let output = self.tensor(tensor);
        if let Err(error) =
            self.reshape_into(input, output, ReshapeConfig::new(ReshapeMode::ViewOnly))
        {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    pub(crate) fn prepare_attention_kv(
        &mut self,
        k: TensorId,
        v: TensorId,
        config: &AttentionConfig,
    ) -> Result<PreparedAttentionKv> {
        let paged = config.paged();
        let shapes = self.attention_kv_shapes(k, v, config)?;
        let sequence_length = if paged.k().is_some() {
            shapes.key.dimensions()[2]
        } else {
            shapes.value.dimensions()[2]
        };
        let sequence = config.modifiers().sequence_length_key_value();
        let k_loaded = if let Some(cache) = paged.k() {
            self.paged_cache_load(
                k,
                sequence.ok_or(FrontendError::SdpaRequires {
                    requirement: SdpaRequirement::SequenceLengths,
                    operation: "sdpa paged attention",
                })?,
                cache.page_table(),
                PagedCacheLoadConfig::new(PagedCacheLayout::SequenceHidden)
                    .with_max_sequence_length(sequence_length),
                TensorOutputConfig::new().with_data_type(self.tensor_spec(k)?.data_type),
            )?
        } else {
            k
        };
        let v_loaded = if let Some(cache) = paged.v() {
            self.paged_cache_load(
                v,
                sequence.ok_or(FrontendError::SdpaRequires {
                    requirement: SdpaRequirement::SequenceLengths,
                    operation: "sdpa paged attention",
                })?,
                cache.page_table(),
                PagedCacheLoadConfig::new(PagedCacheLayout::SequenceHidden)
                    .with_max_sequence_length(sequence_length),
                TensorOutputConfig::new().with_data_type(self.tensor_spec(v)?.data_type),
            )?
        } else {
            v
        };

        Ok(PreparedAttentionKv::new(k_loaded, v_loaded))
    }

    fn sdpa_mask_select(
        &mut self,
        scores: TensorId,
        unmasked: TensorId,
        valid: TensorId,
    ) -> Result<TensorId> {
        let tensor = self.tensor_spec(scores)?.clone();
        let negative_inf = self.tensor(sdpa_mask_scalar(tensor.data_type, f32::NEG_INFINITY)?);
        let output = self.tensor(tensor.virtual_tensor());
        self.pointwise_into(PointwiseOperation::Ternary {
            name: None,
            mode: PointwiseMode::BinarySelect,
            x: unmasked,
            b: negative_inf,
            t: valid,
            output,
            compute_type: None,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            alpha2: 1.0,
        })?;
        Ok(output)
    }

    fn sdpa_additive_mask(&mut self, scores: TensorId, valid: TensorId) -> Result<TensorId> {
        let data_type = self.tensor_spec(scores)?.data_type;
        let zero = self.tensor(sdpa_mask_scalar(data_type, 0.0)?);
        self.sdpa_mask_select(scores, zero, valid)
    }

    pub(crate) fn diagonal_band_mask_composite(
        &mut self,
        scores: TensorId,
        left_window: Option<i64>,
        right_window: Option<i64>,
    ) -> Result<TensorId> {
        let valid = self.diagonal_band_compare(scores, left_window, right_window)?;
        self.sdpa_additive_mask(scores, valid)
    }

    fn diagonal_band_compare(
        &mut self,
        scores: TensorId,
        left_window: Option<i64>,
        right_window: Option<i64>,
    ) -> Result<TensorId> {
        self.validate_tensor_rank(scores, 4, "attention mask score rank")?;
        let shape = self.tensor_spec(scores)?.shape.clone();
        let left_window =
            bounded_sdpa_window(left_window, shape.dimensions()[2], "attention left window")?;
        let right_window = bounded_sdpa_window(
            right_window,
            shape.dimensions()[3] - 1,
            "attention right window",
        )?;
        let SdpaScoreIndices { row, col } = self.sdpa_score_indices(
            scores,
            &shape,
            NanPropagation::NotPropagate,
            NanPropagation::NotPropagate,
        )?;
        // Subtract indices before comparing bounds, avoiding overflow in index + window.
        let relative =
            self.sdpa_binary(PointwiseMode::Sub, col, row, DataType::I32, DataType::I32)?;
        let left = left_window
            .map(|window| {
                // C++'s left window counts the current token, so the lower edge is exclusive.
                let bound = self.tensor(TensorSpec::scalar_i32(-window));
                self.sdpa_binary(
                    PointwiseMode::CmpGt,
                    relative,
                    bound,
                    DataType::Boolean,
                    DataType::Boolean,
                )
            })
            .transpose()?;
        let right = right_window
            .map(|window| {
                let bound = self.tensor(TensorSpec::scalar_i32(window));
                self.sdpa_binary(
                    PointwiseMode::CmpLe,
                    relative,
                    bound,
                    DataType::Boolean,
                    DataType::Boolean,
                )
            })
            .transpose()?;
        match (left, right) {
            (Some(left), Some(right)) => self.sdpa_binary(
                PointwiseMode::LogicalAnd,
                left,
                right,
                DataType::Boolean,
                DataType::Boolean,
            ),
            (Some(valid), None) | (None, Some(valid)) => Ok(valid),
            (None, None) => Ok(self.tensor(TensorSpec::constant(ScalarValue::Boolean(true)))),
        }
    }

    fn padding_mask_predicate(
        &mut self,
        scores: TensorId,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Result<TensorId> {
        self.validate_sdpa_sequence_length_data_types(
            sequence_length_query,
            sequence_length_key_value,
            SDPA_PADDING_MASK_SEQ_LEN_DATA_TYPE,
        )?;
        let shape = self.tensor_spec(scores)?.shape.clone();
        let SdpaScoreIndices { row, col } = self.sdpa_score_indices(
            scores,
            &shape,
            NanPropagation::NotPropagate,
            NanPropagation::NotPropagate,
        )?;
        let row_valid = self.sdpa_binary(
            PointwiseMode::CmpLt,
            row,
            sequence_length_query,
            DataType::Boolean,
            DataType::Boolean,
        )?;
        let col_valid = self.sdpa_binary(
            PointwiseMode::CmpLt,
            col,
            sequence_length_key_value,
            DataType::Boolean,
            DataType::Boolean,
        )?;
        self.sdpa_binary(
            PointwiseMode::LogicalAnd,
            row_valid,
            col_valid,
            DataType::Boolean,
            DataType::Boolean,
        )
    }

    pub(crate) fn padding_mask_select(
        &mut self,
        scores: TensorId,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Result<TensorId> {
        let valid =
            self.padding_mask_predicate(scores, sequence_length_query, sequence_length_key_value)?;
        self.sdpa_mask_select(scores, scores, valid)
    }

    pub(crate) fn alibi_bias(
        &mut self,
        scores: TensorId,
        alibi_slopes: TensorId,
    ) -> Result<TensorId> {
        let scores_tensor = self.tensor_spec(scores)?.clone();
        let index_shape = scores_tensor.shape.clone();

        let SdpaScoreIndices { row, col } = self.sdpa_score_indices(
            scores,
            &index_shape,
            NanPropagation::Propagate,
            NanPropagation::Propagate,
        )?;

        let relative_position =
            self.sdpa_binary(PointwiseMode::Sub, col, row, DataType::I32, DataType::I32)?;
        self.pointwise(
            PointwiseOperation::binary(
                PointwiseMode::Mul,
                relative_position,
                alibi_slopes,
                TensorOutputConfig::new()
                    .with_data_type(self.effective_intermediate_data_type(scores_tensor.data_type)),
            )
            .with_compute_type(DataType::F32),
        )
    }

    fn sdpa_mask_reference(&mut self, q: TensorId, k: TensorId) -> Result<TensorId> {
        let shape = infer_sdpa_scores_shape(self.shape(q)?, self.shape(k)?)?;
        let data_type = self.effective_intermediate_data_type(DataType::F32);
        let zero = self.tensor(sdpa_mask_scalar(data_type, 0.0)?);
        let scores = self.tensor(TensorSpec::new(data_type, shape).virtual_tensor());
        self.pointwise_into(PointwiseOperation::Unary {
            name: None,
            mode: PointwiseMode::Identity,
            input: zero,
            output: scores,
            compute_type: None,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            axis: None,
        })?;
        Ok(scores)
    }

    /// Creates an additive top-left causal mask: zero for `key <= query`, negative infinity otherwise.
    /// Query and key supply geometry only. No extra score-buffer binding is required.
    ///
    /// # Errors
    ///
    /// Returns an error if query or key is unknown, their ranks or batch/head/contraction dimensions cannot form attention scores, or a required mask index, comparison, or pointwise operation cannot be constructed.
    pub fn sdpa_causal_mask(&mut self, q: TensorId, k: TensorId) -> Result<TensorId> {
        let checkpoint = self.mutation_checkpoint();
        let result = (|| {
            let scores = self.sdpa_mask_reference(q, k)?;
            self.diagonal_band_mask_composite(scores, None, Some(0))
        })();
        if result.is_err() {
            self.rollback_to(checkpoint);
        }
        result
    }

    /// Creates an additive window mask with C++'s asymmetric boundary conventions.
    /// `left_window` counts visible tokens including self; `right_window` counts future tokens.
    /// Positions satisfy `query - left_window < key <= query + right_window`.
    ///
    /// # Errors
    ///
    /// Returns an error for negative window bounds, unknown or incompatible query/key geometry, or failure to construct mask index, comparison, or pointwise operations.
    pub fn sdpa_sliding_window_mask(
        &mut self,
        q: TensorId,
        k: TensorId,
        left_window: i64,
        right_window: i64,
    ) -> Result<TensorId> {
        check_range!("left_window", left_window >= 0)?;
        check_range!("right_window", right_window >= 0)?;
        let checkpoint = self.mutation_checkpoint();
        let result = (|| {
            let scores = self.sdpa_mask_reference(q, k)?;
            self.diagonal_band_mask_composite(scores, Some(left_window), Some(right_window))
        })();
        if result.is_err() {
            self.rollback_to(checkpoint);
        }
        result
    }

    /// Adds scaled dot product attention with explicit output and optional softmax summaries.
    ///
    /// SDPA computes `softmax(QK^T / sqrt(d))V` and may generate stats for
    /// backward training computation.
    /// Records direct attention, or composes the configured fused mask.
    /// Rejected requests discard tentative scale tensors and operations.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid direct-attention operands or summaries, unknown or incompatible tensor layouts, or failure to construct configured masks, attention scaling, or constituent composite operations.
    pub fn sdpa_into(
        &mut self,
        inputs: SdpaInputs,
        outputs: SdpaOutputTensors,
        config: &SdpaConfig,
    ) -> Result<()> {
        if !matches!(config.mask(), SdpaFusedMaskMode::None) {
            return self.sdpa_composite(inputs, outputs, config);
        }
        let checkpoint = self.mutation_checkpoint();
        let result = self.sdpa_direct(
            inputs,
            outputs,
            SdpaScoreModifiers::new(),
            None,
            config,
            &DirectSdpaForwardConfig::new().with_stats_base(config.stats_base()),
        );
        if result.is_err() {
            self.rollback_to(checkpoint);
        }
        result
    }

    /// Composes SDPA from matmul, softmax and configured masking operations.
    /// Unlike the unmasked `sdpa` path, this always expands the composite implementation.
    /// Supplied outputs determine which auxiliary results are produced.
    /// Rejected composition preserves existing declarations and removes all tentative mask/scale work.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown or incompatible attention operands or output dimensions, unsupported cumulative sequence lengths, invalid mask or sequence metadata, incompatible score modifiers or softmax summaries, or failure to construct constituent matrix, pointwise, reduction, or dropout operations.
    pub fn sdpa_composite(
        &mut self,
        inputs: SdpaInputs,
        outputs: SdpaOutputTensors,
        config: &SdpaConfig,
    ) -> Result<()> {
        let SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        } = inputs;
        let mut composed = self.clone();
        let modifiers = match config.mask() {
            SdpaFusedMaskMode::None => SdpaScoreModifiers::new(),
            SdpaFusedMaskMode::CausalTopLeft => {
                let causal_mask = composed.sdpa_causal_mask(q, k)?;
                SdpaScoreModifiers::new().with_additive_mask(causal_mask)
            }
            SdpaFusedMaskMode::CausalBottomRight {
                sequence_length_query,
                sequence_length_key_value,
            } => SdpaScoreModifiers::new()
                .with_sequence_inputs(AttentionSequenceInputs::per_batch(
                    sequence_length_query,
                    sequence_length_key_value,
                ))
                .with_mask_mode(AttentionMaskMode::DiagonalBand(
                    AttentionDiagonalBand::causal(AttentionDiagonalAlignment::BottomRight),
                )),
        };
        let compute_type = composed.effective_compute_data_type(DataType::F32);
        let scale = composed.sdpa_effective_scale(scale, config.attention_scale(), compute_type)?;

        composed.sdpa_with_modifiers_internal(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale,
            },
            outputs,
            modifiers,
            None,
            SdpaQuantizedAuxOutputRequest::default(),
            SoftmaxConfig::new().with_stats_base(config.stats_base()),
        )?;
        *self = composed;
        Ok(())
    }

    /// Composes SDPA with independently configured result and auxiliary tensors.
    /// Uses the same composite implementation as [`Self::sdpa_composite`].
    /// Each output selects placement and encoding; saved Stats defaults to FP32.
    /// Result placement selects I/O or intermediate encoding unless explicitly overridden.
    /// Rejected construction leaves the graph unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown or incompatible attention operands or output dimensions, unsupported cumulative sequence lengths, invalid mask or sequence metadata, incompatible score modifiers or softmax summaries, or failure to construct constituent matrix, pointwise, reduction, or dropout operations. Inferred result and auxiliary layouts must also be representable.
    pub fn sdpa(
        &mut self,
        inputs: SdpaInputs,
        config: &SdpaConfig,
        outputs: SdpaOutputTensors<TensorOutputConfig>,
    ) -> Result<SdpaForwardOutputTensors> {
        let SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: _,
        } = inputs;
        let q_tensor = self.tensor_spec(q)?;
        let k_tensor = self.tensor_spec(k)?;
        let v_tensor = self.tensor_spec(v)?;
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, &v_tensor.shape)?;
        let aux_data_type = q_tensor.data_type;
        let aux_shape = reduce_last_axis(&q_tensor.shape)?;
        let output_spec =
            outputs
                .output
                .into_spec(self.config.data_type_policy, aux_data_type, output_shape);
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output_spec, config.name(), "O");
        let outputs = SdpaOutputTensors {
            output,
            stats: outputs.stats.map(|request| {
                self.inferred_tensor(
                    request.into_spec(DataTypePolicy::default(), DataType::F32, aux_shape.clone()),
                    config.name(),
                    "Stats",
                )
            }),
            logit_max: outputs.logit_max.map(|request| {
                self.inferred_tensor(
                    request.into_spec(
                        self.config.data_type_policy,
                        aux_data_type,
                        aux_shape.clone(),
                    ),
                    config.name(),
                    "Max",
                )
            }),
            score_sum_exp: outputs.score_sum_exp.map(|request| {
                self.inferred_tensor(
                    request.into_spec(
                        self.config.data_type_policy,
                        aux_data_type,
                        aux_shape.clone(),
                    ),
                    config.name(),
                    "Sum_exp",
                )
            }),
        };

        if let Err(error) = self.sdpa_composite(inputs, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }

        Ok(SdpaForwardOutputTensors {
            output,
            stats: outputs.stats,
            logit_max: outputs.logit_max,
            score_sum_exp: outputs.score_sum_exp,
            rng_dump: None,
            absolute_max_probabilities: None,
            absolute_max_output: None,
        })
    }

    /// Records floating attention with caller-declared result tensors and the complete attention configuration.
    /// Supplied auxiliary tensors enable their requests without duplicate configuration flags.
    /// Explicit configuration requests still require corresponding output tensors.
    /// Preparation selects the configured automatic, unified or composite implementation for the target.
    /// Rejected requests preserve existing declarations, layouts and operations.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown or incompatible query/key/value or paged-cache geometry, invalid sequence-length combinations, an RNG dump requested without dropout, or unrepresentable inferred score, result, or auxiliary layouts. Caller-declared outputs must exist and match inferred output requirements.
    pub fn sdpa_auto_into(
        &mut self,
        inputs: SdpaInputs,
        outputs: SdpaForwardOutputTensors,
        config: &AttentionConfig,
    ) -> Result<()> {
        self.record_explicit_attention_forward(
            AttentionForwardInputs::Floating(inputs),
            outputs,
            config,
        )
    }

    /// Records attention with independent placement and encoding for every requested result.
    /// Stats, RNG dumps and Amax outputs default to FP32 independently of graph policies.
    /// Preparation resolves the implementation against the live or explicitly declared target.
    /// Caller changes to result tensor layouts are retained during expansion.
    /// Returns every requested auxiliary output instead of discarding side-output identities.
    /// Result placement selects I/O or intermediate encoding unless explicitly overridden.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown or incompatible query/key/value or paged-cache geometry, invalid sequence-length combinations, an RNG dump requested without dropout, or unrepresentable inferred score, result, or auxiliary layouts.
    pub fn sdpa_auto(
        &mut self,
        inputs: SdpaInputs,
        config: AttentionConfig,
        outputs: SdpaForwardOutputTensors<TensorOutputConfig>,
    ) -> Result<SdpaForwardOutputTensors> {
        self.record_attention_forward(AttentionForwardInputs::Floating(inputs), outputs, config)
    }

    pub(crate) fn expand_floating_attention(
        &mut self,
        inputs: SdpaInputs,
        output_data_type: DataType,
        implementation: ResolvedAttentionImplementation,
        config: &AttentionConfig,
    ) -> Result<SdpaForwardOutputTensors> {
        let SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        } = inputs;
        if implementation == ResolvedAttentionImplementation::Unified {
            return self.sdpa_unified_infer(
                inputs,
                output_data_type,
                SdpaForwardQuantization::default(),
                config,
            );
        }

        let compute_type = self.resolve_compute_data_type(config.softmax().compute_type());
        let scale =
            self.sdpa_effective_scale(scale, config.fused().attention_scale(), compute_type)?;

        let prepared_kv = self.prepare_attention_kv(k, v, config)?;
        let k = prepared_kv.key;
        let v = prepared_kv.value;

        let q_tensor = self.tensor_spec(q)?.clone();
        let k_tensor = self.tensor_spec(k)?.clone();
        let v_tensor = self.tensor_spec(v)?.clone();
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, &v_tensor.shape)?;
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        let aux_data_type = q_tensor.data_type;
        let aux_shape = reduce_last_axis(&scores_shape)?;
        let outputs = self.sdpa_requested_aux_output_tensors(
            output,
            config.fused(),
            aux_data_type,
            &aux_shape,
        );
        let rng_dump = if let Some(dropout) = config.dropout() {
            Some(self.attention_dropout_random_number_generator_dump(dropout, &scores_shape)?)
        } else {
            None
        };
        if !config.fused().aux_outputs().rng_dump()
            && let Some(mask) = rng_dump
        {
            self.modify_tensor(mask, TensorSpec::virtual_tensor)?;
        }
        let modifiers = if let Some(dropout) = config.dropout() {
            let dropout_scale = self.tensor(sdpa_scalar_tensor(DataType::F32, dropout.scale())?);
            let rng_dump = rng_dump.ok_or(FrontendError::SdpaOutputMissing {
                operation: "sdpa",
                output: "rng dump",
            })?;
            config
                .modifiers()
                .with_dropout(AttentionDropoutMask::new(rng_dump, dropout_scale))
        } else {
            *config.modifiers()
        };

        let computed = self.sdpa_with_modifiers_internal(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale,
            },
            outputs,
            modifiers,
            config.score_subgraph(),
            config.fused().aux_outputs().quantized(),
            config.softmax().clone(),
        )?;

        Ok(SdpaForwardOutputTensors {
            output,
            stats: outputs.stats,
            logit_max: outputs.logit_max,
            score_sum_exp: outputs.score_sum_exp,
            rng_dump: rng_dump.filter(|_| config.fused().aux_outputs().rng_dump()),
            absolute_max_probabilities: computed.absolute_max_probabilities,
            absolute_max_output: computed.absolute_max_output,
        })
    }

    /// Composes attention with a scalar attention scale and an additive bias, writing caller-declared output and optional statistics.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown or incompatible attention operands or output dimensions, unsupported cumulative sequence lengths, invalid mask or sequence metadata, incompatible score modifiers or softmax summaries, or failure to construct constituent matrix, pointwise, reduction, or dropout operations.
    pub fn sdpa_bias(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        stats: Option<TensorId>,
        config: SoftmaxConfig,
    ) -> Result<()> {
        self.sdpa_with_modifiers(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale,
            },
            SdpaOutputTensors {
                output,
                stats,
                logit_max: None,
                score_sum_exp: None,
            },
            SdpaScoreModifiers::new().with_bias(bias),
            config,
        )
    }
}
