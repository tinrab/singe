use std::collections::BTreeMap;

use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::{UnifiedSdpaSubgraph, support::sdpa_mask_scalar},
        graph::{AliasBinding, Graph},
        infer::*,
        operation::*,
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    tensor::{BackendTensorReordering, Shape, TensorId, TensorSpec},
};

impl Graph {
    pub(in crate::frontend::composite::sdpa) fn sdpa_aux_data_type(&self) -> DataType {
        self.effective_compute_data_type(DataType::F32)
    }

    pub(in crate::frontend::composite::sdpa) fn pointwise_binary_virtual_infer_as(
        &mut self,
        lhs: TensorId,
        rhs: TensorId,
        mode: PointwiseMode,
        compute_type: DataType,
        output_data_type: DataType,
    ) -> Result<TensorId> {
        let lhs_tensor = self.tensor_config(lhs)?.clone();
        let rhs_tensor = self.tensor_config(rhs)?.clone();
        let output_shape = infer_binary_pointwise_output(&lhs_tensor.shape, &rhs_tensor.shape)?;
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape).virtual_tensor());
        self.pointwise(PointwiseOperation::Binary {
            mode,
            lhs,
            rhs,
            output,
            compute_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });
        Ok(output)
    }

    pub(in crate::frontend::composite::sdpa) fn normalize_attention_config_for_forward_version(
        &self,
        cudnn_version: u64,
        config: AttentionConfig,
    ) -> AttentionConfig {
        if cudnn_version > 8902
            && config
                .dropout()
                .is_some_and(|dropout| dropout.probability() == 0.0)
        {
            config.clear_dropout_probability().clear_dropout_offset()
        } else {
            config
        }
    }

    pub(in crate::frontend::composite::sdpa) fn attention_dropout_random_number_generator_config(
        &self,
        probability: f32,
        seed_source: AttentionDropoutSeedSource,
        offset: Option<TensorId>,
    ) -> Result<RandomNumberGeneratorConfig> {
        let probability = f64::from(1.0 - probability);
        let rng_config = match seed_source {
            AttentionDropoutSeedSource::Host(seed) => {
                let mut rng_config = RandomNumberGeneratorConfig::bernoulli(probability, seed);
                if let Some(offset) = offset {
                    rng_config = rng_config.with_offset(offset);
                }
                rng_config
            }
            AttentionDropoutSeedSource::Device(seed) => {
                let offset = offset.ok_or(Error::DescriptorMismatch {
                    name: "sdpa dropout offset".into(),
                })?;
                RandomNumberGeneratorConfig::bernoulli(probability, 0)
                    .with_seed_tensor(seed, offset)
            }
        };

        Ok(rng_config)
    }

    pub(in crate::frontend::composite::sdpa) fn clone_tensor_into_subgraph(
        &self,
        subgraph: &mut Graph,
        cloned_tensors: &mut BTreeMap<TensorId, TensorId>,
        tensor: TensorId,
    ) -> Result<TensorId> {
        if let Some(cloned) = cloned_tensors.get(&tensor).copied() {
            return Ok(cloned);
        }

        let source = self.tensor_config(tensor)?.clone();
        let ragged_offset = source.ragged_offset;
        let cloned = subgraph.tensor(source.clear_ragged_offset());
        cloned_tensors.insert(tensor, cloned);

        if let Some(ragged_offset) = ragged_offset {
            let cloned_ragged_offset =
                self.clone_tensor_into_subgraph(subgraph, cloned_tensors, ragged_offset)?;
            subgraph.set_ragged_offset(cloned, cloned_ragged_offset)?;
        }

        Ok(cloned)
    }

    pub(crate) fn build_sdpa_unified_subgraph(
        &self,
        q: TensorId,
        k: TensorId,
        config: DirectSdpaForwardConfig,
        score_modifiers: SdpaScoreModifiers,
        score_subgraph: Option<&SdpaScoreSubgraph>,
    ) -> Result<Option<UnifiedSdpaSubgraph>> {
        let mut cloned_tensors = BTreeMap::new();
        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let effective_k_shape = if let Some(page_table_k) = config.page_table_k() {
            let sequence_length_key_value =
                config
                    .sequence_length_key_value()
                    .ok_or(Error::DescriptorMismatch {
                        name: "sdpa paged seq lens".into(),
                    })?;
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
        let has_subgraph_modifiers = score_modifiers.bias.is_some()
            || score_modifiers.alibi_slopes.is_some()
            || score_modifiers.additive_mask.is_some()
            || score_modifiers.causal_mask
            || score_modifiers.causal_bottom_right
            || score_modifiers.sliding_window.is_some();
        if !has_subgraph_modifiers && score_subgraph.is_none() {
            return Ok(None);
        }

        let mut subgraph = Graph::new();
        subgraph.sm_version = self.sm_version;
        subgraph.data_type_policy.intermediate = self.data_type_policy.intermediate;
        subgraph.data_type_policy.compute = self.data_type_policy.compute;
        subgraph.dynamic_shape_enabled = self.dynamic_shape_enabled;
        subgraph.override_shape_enabled = self.override_shape_enabled;

        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &effective_k_shape)?;
        let input = subgraph.tensor(
            TensorSpec::new(self.sdpa_aux_data_type(), scores_shape.clone()).virtual_tensor(),
        );
        let mut output = input;

        if let Some(bias) = score_modifiers.bias {
            let score_subgraph_captures_bias = score_subgraph
                .is_some_and(|score_subgraph| score_subgraph.graph().tensor_config(bias).is_ok());
            let bias = self.clone_tensor_into_subgraph(&mut subgraph, &mut cloned_tensors, bias)?;
            if !score_subgraph_captures_bias {
                output = subgraph.pointwise_binary_infer(
                    output,
                    bias,
                    PointwiseMode::Add,
                    DataType::F32,
                )?;
            }
        }
        if let Some(alibi_slopes) = score_modifiers.alibi_slopes {
            let alibi_slopes =
                self.clone_tensor_into_subgraph(&mut subgraph, &mut cloned_tensors, alibi_slopes)?;
            let alibi_bias = subgraph.alibi_bias(output, alibi_slopes)?;
            output = subgraph.pointwise_binary_infer(
                output,
                alibi_bias,
                PointwiseMode::Add,
                DataType::F32,
            )?;
        }
        if let Some(additive_mask) = score_modifiers.additive_mask {
            let additive_mask =
                self.clone_tensor_into_subgraph(&mut subgraph, &mut cloned_tensors, additive_mask)?;
            output = subgraph.pointwise_binary_infer(
                output,
                additive_mask,
                PointwiseMode::Add,
                DataType::F32,
            )?;
        }
        if let Some(score_subgraph) = score_subgraph {
            self.validate_score_subgraph(score_subgraph, &scores_shape, "sdpa score subgraph")?;
            output =
                subgraph.import_score_subgraph(score_subgraph, output, "sdpa score subgraph")?;
        }
        if score_modifiers.causal_mask {
            let output_tensor = subgraph.tensor_config(output)?.clone();
            let masked = subgraph.tensor(
                TensorSpec::new(output_tensor.data_type, output_tensor.shape.clone())
                    .virtual_tensor(),
            );
            let negative_inf = subgraph.tensor(sdpa_mask_scalar(
                output_tensor.data_type,
                f32::NEG_INFINITY,
            )?);
            subgraph.diagonal_band_mask(
                output,
                negative_inf,
                masked,
                DiagonalBandMaskConfig::new(PointwiseMode::CmpGe),
            )?;
            output = masked;
        }
        if let Some((left_window, right_window)) = score_modifiers.sliding_window {
            let sliding_mask =
                subgraph.diagonal_band_mask_composite(output, left_window, Some(right_window))?;
            output = subgraph.pointwise_binary_infer(
                output,
                sliding_mask,
                PointwiseMode::Add,
                DataType::F32,
            )?;
        }
        if score_modifiers.causal_bottom_right {
            let sequence_length_query = score_modifiers
                .sequence_length_query
                .ok_or(Error::DescriptorMismatch {
                    name: "sdpa bottom right seq len".into(),
                })
                .and_then(|tensor| {
                    self.clone_tensor_into_subgraph(&mut subgraph, &mut cloned_tensors, tensor)
                })?;
            let sequence_length_key_value = score_modifiers
                .sequence_length_key_value
                .ok_or(Error::DescriptorMismatch {
                    name: "sdpa bottom right seq len".into(),
                })
                .and_then(|tensor| {
                    self.clone_tensor_into_subgraph(&mut subgraph, &mut cloned_tensors, tensor)
                })?;
            let mask = subgraph.causal_bottom_right_mask(
                output,
                sequence_length_query,
                sequence_length_key_value,
            )?;
            output =
                subgraph.pointwise_binary_infer(output, mask, PointwiseMode::Add, DataType::F32)?;
        }

        let output_tensor = subgraph.tensor_config(output)?.clone();
        if !output_tensor.is_virtual {
            subgraph.replace_tensor(output, output_tensor.virtual_tensor())?;
        }

        Ok(Some(UnifiedSdpaSubgraph {
            graph: subgraph,
            input,
            output,
            cloned_tensors,
        }))
    }

    pub(in crate::frontend::composite::sdpa) fn remap_score_subgraph_tensor(
        tensor_map: &BTreeMap<TensorId, TensorId>,
        tensor: TensorId,
        error_name: &str,
    ) -> Result<TensorId> {
        tensor_map
            .get(&tensor)
            .copied()
            .ok_or_else(|| Error::FrontendCompile(format!("{error_name} tensor remap failed")))
    }

    pub(in crate::frontend::composite::sdpa) fn remap_score_subgraph_optional_tensor(
        tensor_map: &BTreeMap<TensorId, TensorId>,
        tensor: Option<TensorId>,
        error_name: &str,
    ) -> Result<Option<TensorId>> {
        tensor
            .map(|tensor| Self::remap_score_subgraph_tensor(tensor_map, tensor, error_name))
            .transpose()
    }

    pub(in crate::frontend::composite::sdpa) fn import_score_subgraph_operation(
        &self,
        operation: &Operation,
        tensor_map: &BTreeMap<TensorId, TensorId>,
        error_name: &str,
    ) -> Result<Operation> {
        let remap = |tensor| Self::remap_score_subgraph_tensor(tensor_map, tensor, error_name);
        let remap_optional =
            |tensor| Self::remap_score_subgraph_optional_tensor(tensor_map, tensor, error_name);

        match operation {
            Operation::Matmul { a, b, c, config } => Ok(Operation::Matmul {
                a: remap(*a)?,
                b: remap(*b)?,
                c: remap(*c)?,
                config: {
                    let mut config = config.clone();
                    if let Some(m_override) = remap_optional(config.m_override())? {
                        config = config.with_m_override(m_override);
                    }
                    if let Some(k_override) = remap_optional(config.k_override())? {
                        config = config.with_k_override(k_override);
                    }
                    config
                },
            }),
            Operation::Pointwise(pointwise) => Ok(Operation::Pointwise(match pointwise {
                PointwiseOperation::Unary {
                    mode,
                    input,
                    output,
                    compute_type,
                    nan_propagation,
                    alpha1,
                    axis,
                } => PointwiseOperation::Unary {
                    mode: *mode,
                    input: remap(*input)?,
                    output: remap(*output)?,
                    compute_type: *compute_type,
                    nan_propagation: *nan_propagation,
                    alpha1: *alpha1,
                    axis: *axis,
                },
                PointwiseOperation::ReluForward {
                    input,
                    output,
                    compute_type,
                    nan_propagation,
                    lower_clip,
                    upper_clip,
                    lower_clip_slope,
                    axis,
                } => PointwiseOperation::ReluForward {
                    input: remap(*input)?,
                    output: remap(*output)?,
                    compute_type: *compute_type,
                    nan_propagation: *nan_propagation,
                    lower_clip: *lower_clip,
                    upper_clip: *upper_clip,
                    lower_clip_slope: *lower_clip_slope,
                    axis: *axis,
                },
                PointwiseOperation::Binary {
                    mode,
                    lhs,
                    rhs,
                    output,
                    compute_type,
                    nan_propagation,
                    alpha1,
                    alpha2,
                } => PointwiseOperation::Binary {
                    mode: *mode,
                    lhs: remap(*lhs)?,
                    rhs: remap(*rhs)?,
                    output: remap(*output)?,
                    compute_type: *compute_type,
                    nan_propagation: *nan_propagation,
                    alpha1: *alpha1,
                    alpha2: *alpha2,
                },
                PointwiseOperation::Ternary {
                    mode,
                    x,
                    b,
                    t,
                    output,
                    compute_type,
                    nan_propagation,
                    alpha1,
                    alpha2,
                } => PointwiseOperation::Ternary {
                    mode: *mode,
                    x: remap(*x)?,
                    b: remap(*b)?,
                    t: remap(*t)?,
                    output: remap(*output)?,
                    compute_type: *compute_type,
                    nan_propagation: *nan_propagation,
                    alpha1: *alpha1,
                    alpha2: *alpha2,
                },
            })),
            Operation::Reduction(reduction) => Ok(Operation::Reduction(match reduction {
                ReductionOperation::Reduce {
                    op,
                    input,
                    output,
                    compute_type,
                    is_deterministic,
                } => ReductionOperation::Reduce {
                    op: *op,
                    input: remap(*input)?,
                    output: remap(*output)?,
                    compute_type: *compute_type,
                    is_deterministic: *is_deterministic,
                },
            })),
            Operation::Reshape { input, output } => Ok(Operation::Reshape {
                input: remap(*input)?,
                output: remap(*output)?,
            }),
            Operation::Slice {
                input,
                output,
                starts,
                limits,
                strides,
                byte_offset,
            } => Ok(Operation::Slice {
                input: remap(*input)?,
                output: remap(*output)?,
                starts: starts.clone(),
                limits: limits.clone(),
                strides: strides.clone(),
                byte_offset: *byte_offset,
            }),
            Operation::Transpose {
                input,
                output,
                permutation,
            } => Ok(Operation::Transpose {
                input: remap(*input)?,
                output: remap(*output)?,
                permutation: permutation.clone(),
            }),
            Operation::Concat {
                inputs,
                output,
                axis,
                in_place,
            } => Ok(Operation::Concat {
                inputs: inputs
                    .iter()
                    .copied()
                    .map(remap)
                    .collect::<Result<Vec<_>>>()?,
                output: remap(*output)?,
                axis: *axis,
                in_place: *in_place,
            }),
            Operation::DiagonalBandMask {
                x,
                b,
                y,
                comparison_mode,
                sequence_length_query,
                sequence_length_key_value,
                left_bound,
                shift_right_bound,
            } => Ok(Operation::DiagonalBandMask {
                x: remap(*x)?,
                b: remap(*b)?,
                y: remap(*y)?,
                comparison_mode: *comparison_mode,
                sequence_length_query: remap_optional(*sequence_length_query)?,
                sequence_length_key_value: remap_optional(*sequence_length_key_value)?,
                left_bound: remap_optional(*left_bound)?,
                shift_right_bound: remap_optional(*shift_right_bound)?,
            }),
            Operation::Softmax {
                x,
                y,
                stats,
                max,
                sum_exp,
                sink,
            } => Ok(Operation::Softmax {
                x: remap(*x)?,
                y: remap(*y)?,
                stats: remap_optional(*stats)?,
                max: remap_optional(*max)?,
                sum_exp: remap_optional(*sum_exp)?,
                sink: remap_optional(*sink)?,
            }),
            _ => Err(Error::FrontendCompile(format!(
                "{error_name} contains unsupported operation"
            ))),
        }
    }

    pub(in crate::frontend::composite::sdpa) fn import_score_subgraph(
        &mut self,
        score_subgraph: &SdpaScoreSubgraph,
        replacement_input: TensorId,
        error_name: &str,
    ) -> Result<TensorId> {
        let source_graph = score_subgraph.graph();
        source_graph.tensor_config(score_subgraph.input())?;
        source_graph.tensor_config(score_subgraph.output())?;

        let mut tensor_map = BTreeMap::new();
        tensor_map.insert(score_subgraph.input(), replacement_input);

        for (&source_tensor_id, source_tensor) in &source_graph.tensors {
            if source_tensor_id == score_subgraph.input() {
                continue;
            }

            if let Ok(existing) = self.tensor_config(source_tensor_id) {
                let existing_tensor = existing.clone();
                if existing_tensor.data_type != source_tensor.data_type
                    || existing_tensor.shape.dimensions() != source_tensor.shape.dimensions()
                    || existing_tensor.shape.strides() != source_tensor.shape.strides()
                {
                    return Err(Error::FrontendCompile(format!(
                        "{error_name} captured tensor id {source_tensor_id} shape mismatch"
                    )));
                }
                tensor_map.insert(source_tensor_id, source_tensor_id);
            } else {
                let cloned = self.tensor(
                    source_tensor
                        .clone()
                        .with_id(TensorId::generate())
                        .clear_ragged_offset(),
                );
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

        for binding in &source_graph.alias_bindings {
            self.alias_bindings.push(AliasBinding {
                source: Self::remap_score_subgraph_tensor(&tensor_map, binding.source, error_name)?,
                target: Self::remap_score_subgraph_tensor(&tensor_map, binding.target, error_name)?,
                byte_offset: binding.byte_offset,
            });
        }

        let imported_operations = source_graph
            .operations
            .iter()
            .map(|operation| {
                self.import_score_subgraph_operation(operation, &tensor_map, error_name)
            })
            .collect::<Result<Vec<_>>>()?;
        self.operations.extend(imported_operations);

        Self::remap_score_subgraph_tensor(&tensor_map, score_subgraph.output(), error_name)
    }

    pub(in crate::frontend::composite::sdpa) fn validate_mxfp8_scale_tensor(
        &self,
        scale: TensorId,
        name: &str,
    ) -> Result<()> {
        let scale_tensor = self.tensor_config(scale)?;
        if scale_tensor.data_type != DataType::F8E8M0 {
            return Err(Error::DescriptorMismatch { name: name.into() });
        }
        if scale_tensor.reordering != Some(BackendTensorReordering::F8_128x4) {
            return Err(Error::DescriptorMismatch { name: name.into() });
        }

        Ok(())
    }

    pub(in crate::frontend::composite::sdpa) fn mxfp8_attn_scale_tensor(
        &mut self,
        attention_scale: TensorId,
        name: &str,
        compute_type: DataType,
    ) -> Result<TensorId> {
        let attn_scale_value = self
            .tensor_config(attention_scale)?
            .scalar_value
            .ok_or(Error::DescriptorMismatch { name: name.into() })?;
        Ok(self.tensor(
            TensorSpec::new(compute_type, Shape::contiguous([1, 1, 1, 1])?)
                .with_scalar_value(attn_scale_value)?,
        ))
    }

    pub(in crate::frontend::composite::sdpa) fn mxfp8_dequantize_input_infer(
        &mut self,
        input: TensorId,
        scale: TensorId,
        output_data_type: DataType,
    ) -> Result<TensorId> {
        self.block_scale_dequantize_infer(
            input,
            scale,
            BlockScaleDequantizeConfig::new(output_data_type, vec![1, 32]),
        )
    }

    pub(in crate::frontend::composite::sdpa) fn mxfp8_dequantize_transposed_input_infer(
        &mut self,
        input: TensorId,
        scale: TensorId,
        output_data_type: DataType,
    ) -> Result<TensorId> {
        let input_t = self.transpose_last_two(input)?;
        let scale_t = self.transpose_last_two(scale)?;
        self.block_scale_dequantize_infer(
            input_t,
            scale_t,
            BlockScaleDequantizeConfig::new(output_data_type, vec![32, 1]),
        )
    }

    pub(in crate::frontend::composite::sdpa) fn mxfp8_dequantize_sequence_scaled_input_infer(
        &mut self,
        input: TensorId,
        scale: TensorId,
        output_data_type: DataType,
    ) -> Result<TensorId> {
        let input_t = self.transpose_last_two(input)?;
        let scale_t = self.transpose_last_two(scale)?;
        let dequantized_t = self.block_scale_dequantize_infer(
            input_t,
            scale_t,
            BlockScaleDequantizeConfig::new(output_data_type, vec![1, 32]),
        )?;
        self.transpose_last_two(dequantized_t)
    }
}
