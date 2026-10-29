use crate::{
    backend::{
        engine::EngineIndex,
        execution::{EngineChoice, KnobChoiceInfo},
        knob::BackendKnobType,
        reduction::ReduceTensorOperator,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::{
            SdpaGradientOutputConfig, attention_heads_compatible,
            backward_policy::FloatingBackwardAccumulation,
            support::{SDPA_BACKWARD_RNG_DUMP, sdpa_scalar_tensor},
        },
        error::{Error as FrontendError, SdpaRequirement},
        graph::{DataTypePolicy, Graph, TensorOutputConfig},
        operation::{
            AttentionBackwardAuxGradients, AttentionBackwardConfig,
            AttentionBackwardHeuristicOverride, AttentionBackwardRequest, AttentionDropoutMask,
            MatmulConfig, Operation, ReductionConfig, SdpaOperation, SdpaScoreModifiers,
        },
    },
};

#[derive(Debug, Clone, Copy)]
pub struct GradientMatmulWorkspace {
    pub data_type: DataType,
    pub max_total_tokens: Option<i64>,
}

impl Graph {
    /// Records auxiliary presence from outputs and only operands consumed by the final dropout mode.
    pub(crate) fn record_attention_backward_request(
        &mut self,
        mut request: AttentionBackwardRequest,
    ) {
        request.normalize_configuration();
        self.operations.push(Operation::AttentionBackward(request));
    }

    pub(crate) fn infer_attention_sink_gradient(
        &mut self,
        query: TensorId,
        output: Option<TensorOutputConfig>,
        default_data_type: DataType,
        scope: Option<&str>,
    ) -> Result<Option<TensorId>> {
        let Some(output) = output else {
            return Ok(None);
        };
        let heads = self.tensor_spec(query)?.shape.dimensions()[1];
        let spec = output.into_spec(
            DataTypePolicy::default(),
            default_data_type,
            Shape::contiguous([1, heads, 1, 1])?,
        );
        Ok(Some(self.inferred_tensor(spec, scope, "DSINK_TOKEN")))
    }

    pub(crate) fn retain_attention_alignment(&mut self) -> Result<()> {
        for operation in &self.operations {
            match operation {
                Operation::AttentionForward(request) => {
                    self.vector_aligned_tensors
                        .extend(request.inputs.query_key_value());
                    self.vector_aligned_tensors.insert(request.outputs.output);
                }
                Operation::AttentionBackward(AttentionBackwardRequest::Floating {
                    inputs,
                    outputs,
                    ..
                }) => {
                    self.vector_aligned_tensors.extend([
                        inputs.query,
                        inputs.key,
                        inputs.value,
                        inputs.output,
                        inputs.output_gradient,
                        outputs.query_gradient,
                        outputs.key_gradient,
                        outputs.value_gradient,
                    ]);
                }
                Operation::Sdpa(SdpaOperation::Forward { tensors, .. }) => {
                    self.vector_aligned_tensors.extend([
                        tensors.query,
                        tensors.key,
                        tensors.value,
                        tensors.output,
                    ]);
                }
                Operation::Sdpa(SdpaOperation::Backward { tensors, .. }) => {
                    self.vector_aligned_tensors
                        .extend(tensors.activations_and_gradients());
                }
                _ => {}
            }
        }
        for id in &self.vector_aligned_tensors {
            let tensor = self
                .tensors
                .get_mut(id)
                .ok_or(FrontendError::TensorNotFound(*id))?;
            tensor.byte_alignment = tensor.byte_alignment.max(16);
            tensor.validate_vector_stride_alignment(*id, &tensor.shape)?;
        }
        Ok(())
    }

    pub(crate) fn sdpa_backward_modifiers_with_dropout(
        &mut self,
        config: &AttentionBackwardConfig,
        requested_rng_dump: Option<TensorId>,
        aux_gradients: AttentionBackwardAuxGradients,
        scores_shape: &Shape,
    ) -> Result<SdpaScoreModifiers> {
        let Some(dropout) = config.dropout() else {
            return Ok(*config.modifiers());
        };

        let rng_dump =
            self.attention_dropout_random_number_generator_dump(dropout, scores_shape)?;
        let rng_dump = if let Some(requested_rng_dump) = requested_rng_dump {
            self.bind_sdpa_inferred_tensor(rng_dump, requested_rng_dump, SDPA_BACKWARD_RNG_DUMP)?;
            requested_rng_dump
        } else if aux_gradients.random_number_generator_dump() {
            return Err(FrontendError::SdpaOutputMissing {
                operation: "sdpa backward",
                output: "rng dump",
            }
            .into());
        } else {
            self.modify_tensor(rng_dump, TensorSpec::virtual_tensor)?;
            rng_dump
        };

        let dropout_scale = self.tensor(sdpa_scalar_tensor(DataType::F32, dropout.scale())?);
        Ok(config
            .modifiers()
            .with_dropout(AttentionDropoutMask::new(rng_dump, dropout_scale)))
    }

    pub(crate) fn sdpa_backward_matmul_configs(
        modifiers: &SdpaScoreModifiers,
        compute_type: DataType,
    ) -> SdpaBackwardMatmulConfigs {
        let query = modifiers.sequence_length_query();
        let key_value = modifiers.sequence_length_key_value();
        SdpaBackwardMatmulConfigs {
            scores: MatmulConfig::new()
                .with_compute_type(compute_type)
                .with_m_override(query)
                .with_n_override(key_value),
            query_key: MatmulConfig::new()
                .with_compute_type(compute_type)
                .with_m_override(query)
                .with_k_override(key_value),
            key_value: MatmulConfig::new()
                .with_compute_type(compute_type)
                .with_m_override(key_value)
                .with_k_override(query),
        }
    }

    pub(crate) fn attention_workspace_spec(
        &self,
        mut tensor: TensorSpec,
        layout_source: TensorId,
        max_total_tokens: Option<i64>,
        head_multiplier: i64,
    ) -> Result<TensorSpec> {
        let source = self.tensor_spec(layout_source)?;
        let (Some(offsets), Some(total)) = (source.ragged_offset, max_total_tokens) else {
            return Ok(tensor.workspace_tensor());
        };
        if total <= 0 {
            return Err(Error::out_of_range("attention total token bound"));
        }
        let mut strides = source.shape.strides().to_vec();
        for axis in [0, 2] {
            strides[axis] = strides[axis]
                .checked_mul(head_multiplier)
                .ok_or_else(|| Error::out_of_range("packed workspace stride"))?;
        }
        let dimensions = tensor.shape.dimensions().to_vec();
        let packed_span = Shape::with_strides(
            vec![1, dimensions[1], total, dimensions[3]],
            strides.clone(),
        )?;
        // C++ reserves FP32 accumulation capacity even for narrowed grouped-head descriptors.
        let capacity = TensorSpec::new(DataType::F32, packed_span).storage_byte_len(None)?;
        tensor.shape = Shape::with_strides(dimensions, strides)?;
        tensor.ragged_offset = Some(offsets);
        tensor.ragged_offset_multiplier = source.ragged_offset_multiplier;
        Ok(tensor.with_workspace_byte_capacity(capacity))
    }

    pub(crate) fn sdpa_backward_gradient_matmul(
        &mut self,
        a: TensorId,
        b: TensorId,
        output: TensorId,
        config: MatmulConfig,
        workspace: GradientMatmulWorkspace,
    ) -> Result<()> {
        let a_heads = self.tensor_spec(a)?.shape.dimensions()[1];
        let b_heads = self.tensor_spec(b)?.shape.dimensions()[1];
        let output_tensor = self.tensor_spec(output)?.clone();
        let output_heads = output_tensor.shape.dimensions()[1];
        if a_heads != b_heads
            || a_heads == output_heads
            || !attention_heads_compatible(a_heads, output_heads)
        {
            return self.matmul_into(a, b, output, config);
        }
        let mut dimensions = output_tensor.shape.dimensions().to_vec();
        dimensions[1] = a_heads;
        let mut spec =
            TensorSpec::new(workspace.data_type, Shape::contiguous(dimensions)?).virtual_tensor();
        if output_tensor.ragged_offset.is_some() && workspace.max_total_tokens.is_some() {
            spec = self.attention_workspace_spec(
                spec,
                output,
                workspace.max_total_tokens,
                a_heads / output_heads,
            )?;
        }
        let full_heads = self.tensor(spec);
        let compute_type = config.compute_type();
        self.matmul_into(a, b, full_heads, config)?;
        self.reduction_into(
            full_heads,
            output,
            &ReductionConfig::new(ReduceTensorOperator::Add).with_compute_type(compute_type),
        )?;
        Ok(())
    }

    pub(crate) fn infer_attention_gradients(
        &mut self,
        inputs: [TensorId; 3],
        outputs: SdpaGradientOutputConfig,
        fallback: Option<DataType>,
        scope: Option<&str>,
    ) -> Result<[TensorId; 3]> {
        for input in inputs {
            self.validate_tensor_rank(input, 4, "attention backward input")?;
        }
        let [query, key, value] = [
            (inputs[0], outputs.query_gradient),
            (inputs[1], outputs.key_gradient),
            (inputs[2], outputs.value_gradient),
        ]
        .map(|(input, output)| -> Result<_> {
            let source = self.tensor_spec(input)?;
            let mut gradient = source.clone();
            gradient.data_type = output.resolve_data_type(
                self.config.data_type_policy,
                fallback.unwrap_or(source.data_type),
            );
            if gradient.data_type != source.data_type {
                gradient.ragged_byte_capacity = None;
            }
            Ok(if output.is_virtual() {
                gradient.virtual_tensor()
            } else {
                gradient.output_tensor()
            })
        });
        let [query, key, value] = [query?, key?, value?];
        Ok([
            self.inferred_tensor(query, scope, "dQ"),
            self.inferred_tensor(key, scope, "dK"),
            self.inferred_tensor(value, scope, "dV"),
        ])
    }

    pub(crate) fn infer_attention_absolute_max(
        &mut self,
        output: TensorOutputConfig,
        scope: Option<&str>,
        port: &str,
    ) -> Result<TensorId> {
        let spec = output.into_spec(
            DataTypePolicy::default(),
            DataType::F32,
            Shape::contiguous([1, 1, 1, 1])?,
        );
        Ok(self.inferred_tensor(spec, scope, port))
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the composite attention operation builds and validates its complete tensor graph"
    )]
    pub(crate) fn expand_attention_backward_requests(&mut self) -> Result<()> {
        if !self
            .operations
            .iter()
            .any(|op| matches!(op, Operation::AttentionBackward(_)))
        {
            return Ok(());
        }
        let sm = self
            .sm_version()
            .filter(|sm| *sm > 0)
            .ok_or(FrontendError::SdpaRequires {
                requirement: SdpaRequirement::TargetArchitecture,
                operation: "sdpa implementation selection",
            })?;
        for operation in self.take_operations() {
            let Operation::AttentionBackward(mut request) = operation else {
                self.operations.push(operation);
                continue;
            };
            request.normalize_configuration();
            let [q, k, v] = request.query_key_value();
            self.validate_tensor_rank(q, 4, "attention backward query")?;
            self.validate_tensor_rank(k, 4, "attention backward key")?;
            self.validate_tensor_rank(v, 4, "attention backward value")?;
            let d_qk = self.tensor_spec(q)?.shape.dimensions()[3];
            let d_v = self.tensor_spec(v)?.shape.dimensions()[3];
            let mxfp8 = matches!(&request, AttentionBackwardRequest::Mxfp8 { .. });
            let floating = matches!(&request, AttentionBackwardRequest::Floating { .. });
            let d256 = d_qk == 256 && d_v == 256;
            let automatic_determinism = if floating {
                d256
            } else {
                mxfp8 || (d_qk <= 192 && d_v <= 128)
            };
            let deterministic = sm / 10 == 10
                && (automatic_determinism || request.config().deterministic_algorithm());
            // C++ frontend-only engines cover these shapes without a native engine pin.
            let frontend_only = if floating {
                d_qk > 256 && d_v > 256
            } else {
                d256 && (mxfp8 || (107..=119).contains(&sm))
            };
            let mut choice =
                (deterministic && !frontend_only).then(|| AttentionBackwardHeuristicOverride {
                    choice: EngineChoice::new(
                        EngineIndex::new(if sm > 103 { 17 } else { 5 }),
                        vec![
                            KnobChoiceInfo::new(BackendKnobType::TileM, 3),
                            KnobChoiceInfo::new(BackendKnobType::TileN, 2),
                            KnobChoiceInfo::new(BackendKnobType::KernelCfg, 2),
                            KnobChoiceInfo::new(BackendKnobType::StreamK, 0),
                            KnobChoiceInfo::new(BackendKnobType::TileCgaM, 0),
                            KnobChoiceInfo::new(
                                BackendKnobType::Stages,
                                if floating && d256 { 3 } else { 2 },
                            ),
                        ],
                    ),
                });
            match request {
                AttentionBackwardRequest::Floating {
                    inputs,
                    outputs,
                    config,
                } => {
                    let config = if deterministic {
                        config.with_deterministic_algorithm(true)
                    } else {
                        config
                    };
                    let accumulation = self.floating_backward_accumulation(
                        inputs.query,
                        inputs.key,
                        outputs.bias_gradient.is_some(),
                        &config,
                        sm,
                    )?;
                    if accumulation == FloatingBackwardAccumulation::OrderedQuery {
                        choice = Some(AttentionBackwardHeuristicOverride {
                            choice: EngineChoice::new(
                                EngineIndex::new(6),
                                vec![
                                    KnobChoiceInfo::new(BackendKnobType::TileM, 2),
                                    KnobChoiceInfo::new(BackendKnobType::TileN, 1),
                                    KnobChoiceInfo::new(BackendKnobType::KernelCfg, 2),
                                    KnobChoiceInfo::new(BackendKnobType::StreamK, 0),
                                    KnobChoiceInfo::new(BackendKnobType::TileCgaM, 0),
                                    KnobChoiceInfo::new(BackendKnobType::Stages, 4),
                                ],
                            ),
                        });
                    }
                    self.expand_floating_attention_backward(inputs, outputs, config, accumulation)?;
                }
                AttentionBackwardRequest::Fp8 {
                    inputs,
                    outputs,
                    config,
                } => {
                    self.validate_quantized_backward_output_ragged_support(
                        outputs.query_gradient,
                        outputs.key_gradient,
                        outputs.value_gradient,
                        "quantized attention backward",
                    )?;
                    let config = if deterministic {
                        config.with_deterministic_algorithm(true)
                    } else {
                        config
                    };
                    self.expand_fp8_attention_backward(inputs, outputs, &config)?;
                }
                AttentionBackwardRequest::Mxfp8 {
                    inputs,
                    outputs,
                    config,
                } => {
                    self.validate_quantized_backward_output_ragged_support(
                        outputs.query_gradient,
                        outputs.key_gradient,
                        outputs.value_gradient,
                        "quantized attention backward",
                    )?;
                    let config = if deterministic {
                        config.with_deterministic_algorithm(true)
                    } else {
                        config
                    };
                    self.expand_mxfp8_attention_backward(inputs, outputs, &config)?;
                }
            }
            AttentionBackwardHeuristicOverride::retain_into(
                &mut self.attention_backward_heuristic_overrides,
                choice,
            );
        }
        Ok(())
    }
}

/// Matmul settings shared by the SDPA backward expansions.
pub struct SdpaBackwardMatmulConfigs {
    /// `Q K^T` and `dO V^T`, sized by the query and key-value sequence lengths.
    pub(crate) scores: MatmulConfig,
    /// Gradients with respect to the query.
    pub(crate) query_key: MatmulConfig,
    /// Gradients with respect to the key and value.
    pub(crate) key_value: MatmulConfig,
}
