use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
};

use crate::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{BackendTensorReordering, Shape, TensorId, TensorSpec},
    },
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::{
            UnifiedSdpaSubgraph, infer_sdpa_scores_shape,
            support::{SDPA_SCORE_SUBGRAPH, bounded_sdpa_window, sdpa_mask_scalar},
        },
        error::Error as FrontendError,
        graph::{Graph, RuntimeShapeConstraints, StorageView, TensorOutputConfig},
        operation::{
            AlibiBias, AttentionBackwardHeuristicOverride, AttentionDiagonalAlignment,
            AttentionDiagonalBand, AttentionDropoutConfig, AttentionSequenceInputs,
            DiagonalBandMaskBound, DiagonalBandMaskConfig, DirectSdpaDropout,
            DirectSdpaForwardConfig, Operation, PointwiseOperation, RandomNumberDistribution,
            RandomNumberGeneratorConfig, SdpaScoreModifiers, SdpaScoreSubgraph, SequenceLength,
        },
    },
    utility::to_i32,
};

impl Graph {
    pub(crate) fn sdpa_score_data_type(&self) -> DataType {
        self.effective_intermediate_data_type(DataType::F32)
    }

    pub(crate) fn attention_dropout_random_number_generator_dump(
        &mut self,
        dropout: AttentionDropoutConfig,
        scores_shape: &Shape,
    ) -> Result<TensorId> {
        let rng_config = RandomNumberGeneratorConfig::new(
            RandomNumberDistribution::Bernoulli {
                probability: dropout.keep_probability(),
            },
            dropout.seed_source(),
        );
        let rng_dump = self.tensor(TensorSpec::new(DataType::F32, scores_shape.clone()));
        self.random_number_generator_into(rng_dump, rng_config)?;
        Ok(rng_dump)
    }

    pub(crate) fn capture_tensor_into_subgraph(
        &self,
        subgraph: &mut Self,
        tensor_captures: &mut BTreeMap<TensorId, TensorId>,
        shared_tensors: &BTreeSet<TensorId>,
        tensor: TensorId,
    ) -> Result<TensorId> {
        if let Some(captured_tensor) = tensor_captures.get(&tensor).copied() {
            return Ok(captured_tensor);
        }

        let source = self.tensor_spec(tensor)?.clone();
        let ragged_offset = source.ragged_offset;
        let captured_tensor = if shared_tensors.contains(&tensor) {
            subgraph.insert_tensor(tensor, source.with_ragged_offset_reference(None))?
        } else {
            subgraph.tensor(source.with_ragged_offset_reference(None))
        };
        tensor_captures.insert(tensor, captured_tensor);

        if let Some(ragged_offset) = ragged_offset {
            let captured_ragged_offset = self.capture_tensor_into_subgraph(
                subgraph,
                tensor_captures,
                shared_tensors,
                ragged_offset,
            )?;
            subgraph.set_ragged_offset(captured_tensor, captured_ragged_offset)?;
        }

        Ok(captured_tensor)
    }

    pub(crate) fn sdpa_diagonal_band_mask(
        &mut self,
        scores: TensorId,
        band: AttentionDiagonalBand,
        mut sequence: AttentionSequenceInputs<TensorId>,
    ) -> Result<TensorId> {
        if !band.has_bounds() {
            return Ok(scores);
        }
        self.validate_tensor_rank(scores, 4, "attention mask score rank")?;
        if band.alignment() == AttentionDiagonalAlignment::TopLeft {
            sequence = AttentionSequenceInputs::default();
        } else {
            let shape = self.shape(scores)?;
            let fixed_lengths = [shape.dimensions()[2], shape.dimensions()[3]];
            for (lengths, fixed) in [&mut sequence.query, &mut sequence.key_value]
                .into_iter()
                .zip(fixed_lengths)
            {
                if lengths.is_none() {
                    let length = to_i32(fixed, "attention fixed sequence length")?;
                    let tensor = self.tensor(
                        TensorSpec::constant(length).with_shape(Shape::contiguous([1, 1, 1, 1])?),
                    );
                    *lengths = Some(SequenceLength::PerBatch(tensor));
                }
            }
        }
        self.sdpa_native_window_mask(scores, sequence, band.left_window(), band.right_window())
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "unbound sequence masks use f32 tensors to represent diagonal offsets"
    )]
    fn sdpa_native_window_mask(
        &mut self,
        mut scores: TensorId,
        sequence: AttentionSequenceInputs<TensorId>,
        left_window: Option<i64>,
        right_window: Option<i64>,
    ) -> Result<TensorId> {
        self.validate_tensor_rank(scores, 4, "attention mask score rank")?;
        let shape = self.shape(scores)?;
        let (query, key_value) = (shape.dimensions()[2], shape.dimensions()[3]);
        let (left_maximum, right_maximum) = if sequence.is_empty() {
            (query, key_value - 1)
        } else {
            let extent = query
                .checked_add(key_value)
                .ok_or_else(|| Error::out_of_range("attention shifted window extent"))?;
            (extent, extent - 1)
        };
        let left_window = bounded_sdpa_window(left_window, left_maximum, "attention left window")?;
        let right_window =
            bounded_sdpa_window(right_window, right_maximum, "attention right window")?;
        if let Some(right) = right_window {
            let tensor = self.tensor_spec(scores)?.clone();
            let negative_inf = self.tensor(sdpa_mask_scalar(tensor.data_type, f32::NEG_INFINITY)?);
            let mut config =
                DiagonalBandMaskConfig::new(PointwiseMode::CmpGe).with_sequence_inputs(sequence);
            if right != 0 {
                let shift = self.tensor(
                    TensorSpec::constant(right).with_shape(Shape::contiguous([1, 1, 1, 1])?),
                );
                config = config.with_bound(DiagonalBandMaskBound::Right { shift: Some(shift) });
            }
            let output = self.tensor(tensor.virtual_tensor());
            self.diagonal_band_mask_into(scores, negative_inf, output, config)?;
            scores = output;
        }
        if let Some(left) = left_window {
            let tensor = self.tensor_spec(scores)?.clone();
            let negative_inf = self.tensor(sdpa_mask_scalar(tensor.data_type, f32::NEG_INFINITY)?);
            let bound = if sequence.is_empty() {
                TensorSpec::constant(left as f32).with_shape(Shape::contiguous([1, 1, 1, 1])?)
            } else {
                TensorSpec::constant(left).with_shape(Shape::contiguous([1, 1, 1, 1])?)
            };
            let bound = self.tensor(bound);
            let output = self.tensor(tensor.virtual_tensor());
            self.diagonal_band_mask_into(
                scores,
                negative_inf,
                output,
                DiagonalBandMaskConfig::new(PointwiseMode::CmpGt)
                    .with_sequence_inputs(sequence)
                    .with_bound(DiagonalBandMaskBound::Left(bound)),
            )?;
            scores = output;
        }
        Ok(scores)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the composite attention operation builds and validates its complete tensor graph"
    )]
    pub(crate) fn build_sdpa_unified_subgraph(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: &DirectSdpaForwardConfig,
        score_modifiers: SdpaScoreModifiers,
        score_subgraph: Option<&SdpaScoreSubgraph>,
    ) -> Result<Option<UnifiedSdpaSubgraph>> {
        if !score_modifiers.has_pre_softmax_subgraph_modifier() && score_subgraph.is_none() {
            return Ok(None);
        }
        let mut tensor_captures = BTreeMap::new();
        let mut shared_tensors = self
            .tensor_records()?
            .into_iter()
            .filter_map(|(id, record)| record.access.is_some().then_some(id))
            .collect::<BTreeSet<_>>();
        shared_tensors.extend(
            self.tensors
                .iter()
                .filter_map(|(&id, tensor)| tensor.is_virtual().then_some(id)),
        );
        shared_tensors.retain(|id| {
            self.tensors
                .get(id)
                .is_some_and(|tensor| !tensor.is_by_value() && !tensor.is_workspace())
        });
        let q_tensor = self.tensor_spec(q)?.clone();
        let sequence_length_hint = score_modifiers.bias().or_else(|| {
            config
                .dropout()
                .and_then(DirectSdpaDropout::random_number_generator_dump)
        });
        let kv = self.infer_direct_sdpa_kv_shapes(k, v, config, sequence_length_hint)?;

        let mut subgraph = self.new_subgraph();
        if let Some(score_subgraph) = score_subgraph {
            subgraph
                .runtime
                .reserved_tensor_ids
                .extend(score_subgraph.graph().tensors.keys().copied());
        }
        subgraph.config.sm_version = self.config.sm_version;
        subgraph.config.data_type_policy = config
            .data_type_policy()
            .with_defaults(self.config.data_type_policy);
        subgraph.config.dynamic_shape_enabled = self.config.dynamic_shape_enabled;
        subgraph.config.override_shape_enabled = self.config.override_shape_enabled;

        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &kv.key)?;
        let score_data_type = subgraph.sdpa_score_data_type();
        let input = subgraph
            .tensor(TensorSpec::new(score_data_type, scores_shape.clone()).virtual_tensor());
        let mut output = input;

        if let Some(score_subgraph) = score_subgraph {
            subgraph.validate_score_subgraph(score_subgraph, &scores_shape, SDPA_SCORE_SUBGRAPH)?;
            for id in score_subgraph.captured_tensor_ids() {
                let source = if self.tensors.contains_key(&id) {
                    self
                } else {
                    score_subgraph
                        .graph()
                        .captured_tensor_owner(id)
                        .ok_or(FrontendError::TensorNotFound(id))?
                };
                source.capture_tensor_into_subgraph(
                    &mut subgraph,
                    &mut tensor_captures,
                    &shared_tensors,
                    id,
                )?;
            }
            output = subgraph.import_score_subgraph(
                score_subgraph,
                output,
                SDPA_SCORE_SUBGRAPH,
                Some(&tensor_captures),
            )?;
        }
        if let Some(bias) = score_modifiers.bias() {
            let bias = self.capture_tensor_into_subgraph(
                &mut subgraph,
                &mut tensor_captures,
                &shared_tensors,
                bias,
            )?;
            output = subgraph.pointwise(
                PointwiseOperation::binary(
                    PointwiseMode::Add,
                    output,
                    bias,
                    TensorOutputConfig::new().with_data_type(DataType::F32),
                )
                .with_compute_type(DataType::F32),
            )?;
        }
        if let Some(alibi) = score_modifiers.alibi() {
            let alibi_slopes = match alibi {
                AlibiBias::Slopes(slopes) => self.capture_tensor_into_subgraph(
                    &mut subgraph,
                    &mut tensor_captures,
                    &shared_tensors,
                    slopes,
                )?,
                AlibiBias::Standard => {
                    subgraph.standard_alibi_slopes(q_tensor.shape.dimensions()[1])?
                }
            };
            let alibi_bias = subgraph.alibi_bias(output, alibi_slopes)?;
            output = subgraph.pointwise(
                PointwiseOperation::binary(
                    PointwiseMode::Add,
                    output,
                    alibi_bias,
                    TensorOutputConfig::new().with_data_type(DataType::F32),
                )
                .with_compute_type(DataType::F32),
            )?;
        }
        if let Some(additive_mask) = score_modifiers.additive_mask() {
            let additive_mask = self.capture_tensor_into_subgraph(
                &mut subgraph,
                &mut tensor_captures,
                &shared_tensors,
                additive_mask,
            )?;
            output = subgraph.pointwise(
                PointwiseOperation::binary(
                    PointwiseMode::Add,
                    output,
                    additive_mask,
                    TensorOutputConfig::new().with_data_type(DataType::F32),
                )
                .with_compute_type(DataType::F32),
            )?;
        }
        if let Some(band) = score_modifiers.diagonal_band() {
            let sequence = if band.has_bounds()
                && band.alignment() == AttentionDiagonalAlignment::BottomRight
            {
                score_modifiers.sequence_inputs().try_map(|tensor| {
                    self.capture_tensor_into_subgraph(
                        &mut subgraph,
                        &mut tensor_captures,
                        &shared_tensors,
                        tensor,
                    )
                })?
            } else {
                AttentionSequenceInputs::default()
            };
            output = subgraph.sdpa_diagonal_band_mask(output, band, sequence)?;
        }

        if subgraph.operations.is_empty() && output == input {
            return Ok(None);
        }

        let captured_tensors = tensor_captures
            .keys()
            .map(|&id| {
                let tensor = self
                    .tensors
                    .get(&id)
                    .or_else(|| {
                        score_subgraph.and_then(|graph| {
                            graph
                                .graph()
                                .captured_tensor_owner(id)
                                .and_then(|owner| owner.tensors.get(&id))
                        })
                    })
                    .ok_or(FrontendError::TensorNotFound(id))?;
                Ok((id, tensor.clone()))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Some(UnifiedSdpaSubgraph {
            graph: subgraph,
            input,
            output,
            tensor_captures,
            captured_tensors,
        }))
    }

    pub(crate) fn remap_score_subgraph_tensor(
        tensor_map: &BTreeMap<TensorId, TensorId>,
        tensor: TensorId,
        error_name: &str,
    ) -> Result<TensorId> {
        tensor_map.get(&tensor).copied().ok_or_else(|| {
            FrontendError::Compile(format!("{error_name} tensor remap failed")).into()
        })
    }

    pub(crate) fn import_score_subgraph(
        &mut self,
        score_subgraph: &SdpaScoreSubgraph,
        replacement_input: TensorId,
        error_name: &str,
        captures: Option<&BTreeMap<TensorId, TensorId>>,
    ) -> Result<TensorId> {
        let mut combined = self.clone();
        let output = combined.import_score_subgraph_in_place(
            score_subgraph,
            replacement_input,
            error_name,
            captures,
        )?;
        *self = combined;
        Ok(output)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the composite attention operation builds and validates its complete tensor graph"
    )]
    fn import_score_subgraph_in_place(
        &mut self,
        score_subgraph: &SdpaScoreSubgraph,
        replacement_input: TensorId,
        error_name: &str,
        captures: Option<&BTreeMap<TensorId, TensorId>>,
    ) -> Result<TensorId> {
        let source_policy = score_subgraph.graph().config.data_type_policy;
        let mut source_graph = Cow::Borrowed(score_subgraph.graph());
        if source_graph.operations.iter().any(|operation| {
            matches!(
                operation,
                Operation::AttentionForward(_) | Operation::AttentionBackward(_)
            )
        }) {
            let source = source_graph.to_mut();
            source.config.sm_version = self.config.sm_version.or(source.config.sm_version);
            source.config.dynamic_shape_enabled |= self.config.dynamic_shape_enabled;
            source.config.override_shape_enabled |= self.config.override_shape_enabled;
            source
                .runtime
                .reserved_tensor_ids
                .extend(self.tensors.keys().copied());
            source
                .runtime
                .reserved_tensor_ids
                .extend(self.runtime.reserved_tensor_ids.iter().copied());
            source.config.data_type_policy = source
                .config
                .data_type_policy
                .with_defaults(self.config.data_type_policy);
            source.retain_attention_alignment()?;
            source.promote_attention_ragged_offsets()?;
            source.expand_attention_requests()?;
            source.expand_attention_backward_requests()?;
        }
        source_graph.tensor_spec(score_subgraph.input())?;
        source_graph.tensor_spec(score_subgraph.output())?;

        let mut tensor_map = BTreeMap::new();
        tensor_map.insert(score_subgraph.input(), replacement_input);
        let view_targets = source_graph
            .storage_views
            .iter()
            .map(|binding| binding.target)
            .collect::<BTreeSet<_>>();

        for (&source_tensor_id, source_tensor) in &source_graph.tensors {
            if source_tensor_id == score_subgraph.input() {
                continue;
            }
            if view_targets.contains(&source_tensor_id) {
                let view = self.tensor(source_tensor.clone().with_ragged_offset_reference(None));
                tensor_map.insert(source_tensor_id, view);
                continue;
            }

            let captured_tensor = captures
                .and_then(|captures| captures.get(&source_tensor_id))
                .copied()
                .unwrap_or(source_tensor_id);
            if let Ok(existing) = self.tensor_spec(captured_tensor) {
                let existing_tensor = existing.clone();
                let promoted_layout =
                    Self::attention_index_layout_is_promotion(&existing_tensor, source_tensor)?;
                let existing_promoted_layout =
                    Self::attention_index_layout_is_promotion(source_tensor, &existing_tensor)?;
                if existing_tensor.data_type != source_tensor.data_type
                    || (!promoted_layout
                        && !existing_promoted_layout
                        && (existing_tensor.shape.dimensions() != source_tensor.shape.dimensions()
                            || existing_tensor.shape.strides() != source_tensor.shape.strides()))
                {
                    return Err(FrontendError::Compile(format!(
                        "{error_name} captured_tensor tensor id {source_tensor_id} shape mismatch"
                    ))
                    .into());
                }
                if promoted_layout {
                    self.promote_index_tensors([captured_tensor])?;
                }
                self.tensor_spec_mut(captured_tensor)?.byte_alignment = existing_tensor
                    .byte_alignment
                    .max(source_tensor.byte_alignment);
                tensor_map.insert(source_tensor_id, captured_tensor);
            } else if !source_tensor.is_virtual()
                && !source_tensor.is_constant()
                && !source_tensor.is_workspace()
            {
                let captured_tensor = self.insert_tensor(
                    source_tensor_id,
                    source_tensor.clone().with_ragged_offset_reference(None),
                )?;
                tensor_map.insert(source_tensor_id, captured_tensor);
            } else {
                let cloned = self.tensor(source_tensor.clone().with_ragged_offset_reference(None));
                tensor_map.insert(source_tensor_id, cloned);
            }
        }

        for (&source_tensor_id, source_tensor) in &source_graph.tensors {
            if source_tensor_id == score_subgraph.input() {
                continue;
            }

            if let Some(ragged_offset) = source_tensor.ragged_offset {
                let mapped_tensor =
                    Self::remap_score_subgraph_tensor(&tensor_map, source_tensor_id, error_name)?;
                let mapped_ragged_offset =
                    Self::remap_score_subgraph_tensor(&tensor_map, ragged_offset, error_name)?;
                self.set_ragged_offset(mapped_tensor, mapped_ragged_offset)?;
            }
        }

        for binding in &source_graph.storage_views {
            self.storage_views.push(StorageView {
                source: Self::remap_score_subgraph_tensor(&tensor_map, binding.source, error_name)?,
                target: Self::remap_score_subgraph_tensor(&tensor_map, binding.target, error_name)?,
                byte_offset: binding.byte_offset,
            });
        }

        let imported_operations = source_graph
            .operations
            .iter()
            .map(|operation| {
                let mut imported = operation.try_map_tensor_ids(
                    &|tensor| Self::remap_score_subgraph_tensor(&tensor_map, tensor, error_name),
                    &|tensor| {
                        tensor_map
                            .get(&tensor)
                            .or_else(|| captures.and_then(|captures| captures.get(&tensor)))
                            .map(|&target| self.capture_target(target))
                            .transpose()
                    },
                )?;
                imported.inherit_data_type_policy(source_policy);
                Ok(imported)
            })
            .collect::<Result<Vec<_>>>()?;
        for (&source_tensor_id, bounds) in &source_graph.runtime_shape_constraints {
            let tensor =
                Self::remap_score_subgraph_tensor(&tensor_map, source_tensor_id, error_name)?;
            let source = source_graph.tensor_spec(source_tensor_id)?;
            bounds.validate_for(source)?;
            let bounds =
                if Self::attention_index_layout_is_promotion(source, self.tensor_spec(tensor)?)? {
                    RuntimeShapeConstraints::create(
                        [bounds.min_dimensions[0], 1, 1, 1],
                        [bounds.max_dimensions[0], 1, 1, 1],
                    )?
                } else {
                    bounds.clone()
                };
            let bounds = match self.runtime_shape_constraints.get(&tensor) {
                Some(existing) => bounds.intersection(existing)?,
                None => bounds,
            };
            bounds.validate_for(self.tensor_spec(tensor)?)?;
            self.runtime_shape_constraints.insert(tensor, bounds);
        }
        for &source_tensor_id in &source_graph.vector_aligned_tensors {
            self.vector_aligned_tensors
                .insert(Self::remap_score_subgraph_tensor(
                    &tensor_map,
                    source_tensor_id,
                    error_name,
                )?);
        }
        AttentionBackwardHeuristicOverride::retain_into(
            &mut self.attention_backward_heuristic_overrides,
            source_graph.attention_backward_heuristic_overrides.clone(),
        );
        self.operations.extend(imported_operations);

        Self::remap_score_subgraph_tensor(&tensor_map, score_subgraph.output(), error_name)
    }

    pub(crate) fn validate_mxfp8_scale_tensor(
        &self,
        scale: TensorId,
        name: &'static str,
    ) -> Result<()> {
        let scale_tensor = self.tensor_spec(scale)?;
        if scale_tensor.data_type != DataType::F8UE8M0 {
            return Err(FrontendError::TensorDataTypeMismatch {
                tensor_id: scale,
                operation: name,
                expected: DataType::F8UE8M0,
                actual: scale_tensor.data_type,
            }
            .into());
        }
        if scale_tensor.reordering != Some(BackendTensorReordering::F8_128x4) {
            return Err(FrontendError::TensorLayoutMismatch {
                tensor_id: scale,
                operation: name,
                reason: "expected f8 128x4 reordering".into(),
            }
            .into());
        }

        Ok(())
    }
}
