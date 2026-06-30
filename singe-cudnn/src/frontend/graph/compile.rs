use std::{collections::HashMap, slice::from_ref};

use crate::{
    behavior::{BackendBehaviorNote, BackendNumericalNote},
    context::Context,
    engine::{Engine, EngineIndex},
    error::{Error, Result},
    execution::{EngineConfig, EngineHeuristics, EngineKnobInfo, OperationGraph},
    frontend::{
        composite::sdpa::{SdpaInputs, SdpaOutputTensors},
        lower::{LoweredGraph, LoweredOperation, LoweredSdpaSubgraph},
        operation::{
            AttentionMaskMode, AttentionPrimitiveOperation, CompileConfig, DirectSdpaForwardConfig,
            DirectSdpaSequenceMode, HeuristicMode, Operation, SdpaOperation, SdpaScoreModifiers,
            SoftmaxConfig,
        },
        plan::{
            BindingReplacement, BuildPlanPolicy, CachedPlanChoice, CompiledGraph,
            EngineConfigMetadata, PlanCandidates,
        },
        support,
    },
    knob::BackendKnobType,
    utility::to_i64,
    version,
};

use super::{Graph, PreparedGraph};

impl Graph {
    fn engine_config_matches_compile_config(
        operation_graph: &OperationGraph,
        engine_config: &EngineConfig,
        compile_config: &CompileConfig,
    ) -> Result<bool> {
        Ok(
            EngineConfigMetadata::create(operation_graph, engine_config)?
                .matches_compile_config(compile_config),
        )
    }

    fn filter_engine_configs(
        operation_graph: &OperationGraph,
        engine_configs: Vec<EngineConfig>,
        compile_config: &CompileConfig,
    ) -> Result<Vec<EngineConfig>> {
        if !compile_config.has_engine_filters() {
            return Ok(engine_configs);
        }

        let mut filtered = Vec::new();
        for engine_config in engine_configs {
            if Self::engine_config_matches_compile_config(
                operation_graph,
                &engine_config,
                compile_config,
            )? {
                filtered.push(engine_config);
            }
        }
        Ok(filtered)
    }

    fn collect_engine_configs(
        ctx: Option<&Context>,
        prepared: &PreparedGraph,
        compile_config: &CompileConfig,
    ) -> Result<Vec<EngineConfig>> {
        let device_properties = prepared.expanded.resolved_device_properties()?;
        let heuristic_modes = if compile_config.heuristic_modes().is_empty() {
            vec![HeuristicMode::Instant]
        } else {
            compile_config.heuristic_modes().to_vec()
        };
        let sm_count_target = compile_config
            .sm_count_target()
            .or(prepared.expanded.sm_count_target());

        let mut engine_configs = Vec::new();
        let mut last_error = None;
        for mode in heuristic_modes {
            let backend_mode = mode.backend_mode();
            let heuristics = match match ctx {
                Some(ctx) => EngineHeuristics::create_with_config(
                    ctx,
                    &prepared.operation_graph,
                    backend_mode,
                    sm_count_target,
                    device_properties.as_deref(),
                ),
                None => {
                    let device_properties = device_properties
                        .as_deref()
                        .ok_or(Error::FrontendDevicelessRequiresDeviceProperties);
                    device_properties.and_then(|device_properties| {
                        EngineHeuristics::create_deviceless(
                            &prepared.operation_graph,
                            backend_mode,
                            sm_count_target,
                            device_properties,
                        )
                    })
                }
            } {
                Ok(heuristics) => heuristics,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };
            match heuristics.engine_configs() {
                Ok(configs) => engine_configs.extend(configs),
                Err(error) => last_error = Some(error),
            }
        }

        if engine_configs.is_empty() {
            let engine_indices = match ctx {
                Some(_) => Engine::engine_indices_for_operation_graph(
                    prepared.operation_graph.descriptor(),
                )?,
                None => (0..prepared.operation_graph.engine_count()?)
                    .map(|index| to_i64(index, "engine_index"))
                    .collect::<Result<Vec<_>>>()?,
            };
            if !engine_indices.is_empty() {
                let mut explicit_configs = Vec::new();
                let mut explicit_last_error = None;
                for engine_index in engine_indices {
                    match match ctx {
                        Some(ctx) => EngineConfig::create_with_config(
                            ctx,
                            &prepared.operation_graph,
                            EngineIndex::new(engine_index),
                            &[],
                            device_properties.as_deref(),
                        ),
                        None => {
                            let device_properties = device_properties
                                .as_deref()
                                .ok_or(Error::FrontendDevicelessRequiresDeviceProperties);
                            device_properties.and_then(|device_properties| {
                                EngineConfig::create_deviceless(
                                    &prepared.operation_graph,
                                    EngineIndex::new(engine_index),
                                    &[],
                                    device_properties,
                                )
                            })
                        }
                    } {
                        Ok(engine_config)
                            if Self::engine_config_matches_compile_config(
                                &prepared.operation_graph,
                                &engine_config,
                                compile_config,
                            )? =>
                        {
                            explicit_configs.push(engine_config);
                        }
                        Ok(_) => {}
                        Err(error) => explicit_last_error = Some(error),
                    }
                }

                if !explicit_configs.is_empty() {
                    return Ok(explicit_configs);
                }

                if explicit_last_error.is_some() {
                    last_error = explicit_last_error;
                }
            }

            return Err(last_error.unwrap_or(Error::NoAvailableEngines));
        }

        Self::filter_engine_configs(&prepared.operation_graph, engine_configs, compile_config)
    }

    pub(crate) fn expand_for_runtime(&self, cudnn_version: u64) -> Result<Self> {
        if !support::runtime_requires_legacy_attention_primitives(cudnn_version) {
            return Ok(self.clone());
        }

        let mut expanded = self.clone();
        let operations = std::mem::take(&mut expanded.operations);
        for operation in operations {
            match operation {
                Operation::AttentionPrimitive(AttentionPrimitiveOperation::Softmax {
                    x,
                    y,
                    stats,
                    max,
                    sum_exp,
                    sink,
                }) => {
                    expanded.expand_softmax_for_legacy_runtime(x, y, stats, max, sum_exp, sink)?
                }
                Operation::AttentionPrimitive(AttentionPrimitiveOperation::DiagonalBandMask {
                    x,
                    b,
                    y,
                    comparison_mode,
                    sequence_length_query,
                    sequence_length_key_value,
                    left_bound,
                    shift_right_bound,
                }) => expanded.expand_diagonal_band_mask_for_legacy_runtime(
                    x,
                    b,
                    y,
                    comparison_mode,
                    sequence_length_query,
                    sequence_length_key_value,
                    left_bound,
                    shift_right_bound,
                )?,
                Operation::Sdpa(SdpaOperation::Forward {
                    q,
                    k,
                    v,
                    o,
                    scale,
                    stats,
                    logit_max,
                    score_sum_exp,
                    ..
                }) if support::runtime_requires_composite_sdpa_forward(cudnn_version) => expanded
                    .sdpa_with_modifiers(
                    SdpaInputs {
                        query: q,
                        key: k,
                        value: v,
                        scale,
                    },
                    SdpaOutputTensors {
                        output: o,
                        stats,
                        logit_max,
                        score_sum_exp,
                    },
                    SdpaScoreModifiers::new(),
                    SoftmaxConfig::new(expanded.tensor_config(q)?.data_type),
                )?,
                other => expanded.operations.push(other),
            }
        }

        Ok(expanded)
    }

    pub(crate) fn prepare(&self, ctx: &Context) -> Result<PreparedGraph> {
        self.prepare_with_context(Some(ctx))
    }

    pub(crate) fn prepare_deviceless(&self) -> Result<PreparedGraph> {
        self.prepare_with_context(None)
    }

    pub(crate) fn prepare_with_context(&self, ctx: Option<&Context>) -> Result<PreparedGraph> {
        if self.operations.is_empty() {
            return Err(Error::FrontendGraphEmpty);
        }

        self.validate_runtime_feature_support_for_version(version()?.raw())?;
        let mut expanded = self.expand_for_runtime(version()?.raw())?;
        expanded.expand_fused_scalar_shapes_for_pointwise()?;
        expanded.validate_pointwise_operations()?;
        let mut lowered = LoweredGraph::lower(&expanded)?;
        let mut extra_binding_template_ids = Vec::new();
        let mut extra_scalar_bindings = Vec::new();
        let mut extra_binding_replacements = Vec::new();
        let mut lowered_operation_index = 0_usize;
        for operation in &expanded.operations {
            let lowers = !matches!(
                operation,
                Operation::Matmul(crate::frontend::operation::MatmulOperation::Fp8 { .. })
                    | Operation::Tensor(crate::frontend::operation::TensorOperation::Slice { .. })
                    | Operation::Tensor(
                        crate::frontend::operation::TensorOperation::Transpose { .. }
                    )
            );
            if !lowers {
                continue;
            }
            if let Operation::Sdpa(SdpaOperation::Forward {
                q,
                k,
                config,
                score_modifiers,
                score_subgraph,
                ..
            }) = operation
            {
                if let Some(subgraph) = expanded.build_sdpa_unified_subgraph(
                    *q,
                    *k,
                    **config,
                    *score_modifiers,
                    score_subgraph.as_ref().as_ref(),
                )? {
                    let prepared_subgraph = subgraph.graph.prepare_with_context(ctx)?;
                    let subgraph_input_id =
                        prepared_subgraph.lowered.tensor_record(subgraph.input)?.id;
                    let subgraph_output_id =
                        prepared_subgraph.lowered.tensor_record(subgraph.output)?.id;
                    let cloned_tensor_lookup = subgraph
                        .cloned_tensors
                        .iter()
                        .map(|(source, target)| (*target, source))
                        .collect::<HashMap<_, _>>();
                    for (subgraph_tensor_ref, subgraph_record) in &prepared_subgraph.lowered.tensors
                    {
                        if subgraph_record.tensor.is_virtual {
                            continue;
                        }
                        if let Some(source) = cloned_tensor_lookup.get(subgraph_tensor_ref) {
                            extra_binding_template_ids.push(subgraph_record.id);
                            extra_binding_replacements.push(BindingReplacement {
                                source_id: lowered.tensor_record(**source)?.id,
                                target_id: subgraph_record.id,
                                byte_offset: 0,
                            });
                        } else if let Some(scalar_value) = subgraph_record.tensor.scalar_value {
                            extra_scalar_bindings.push((subgraph_record.id, scalar_value));
                        }
                    }
                    if let Operation::Sdpa(op) = operation {
                        lowered.operations[lowered_operation_index] = op.lower_with_tensors(
                            &lowered.backend_tensors,
                            Some(LoweredSdpaSubgraph {
                                operation_graph: &prepared_subgraph.operation_graph,
                                input_id: subgraph_input_id,
                                output_id: subgraph_output_id,
                            }),
                        )?;
                    }
                }
            } else if let Operation::Sdpa(SdpaOperation::Backward { q, k, config, .. }) = operation
            {
                let mask_mode = if let DirectSdpaSequenceMode::PaddingMask {
                    sequence_length_query,
                    sequence_length_key_value,
                } = config.sequence_mode()
                {
                    if config.causal_mask() {
                        AttentionMaskMode::CausalTopLeftWithPadding {
                            sequence_length_query,
                            sequence_length_key_value,
                        }
                    } else {
                        AttentionMaskMode::Padding {
                            sequence_length_query,
                            sequence_length_key_value,
                        }
                    }
                } else if config.causal_mask() {
                    AttentionMaskMode::CausalTopLeft
                } else {
                    AttentionMaskMode::None
                };
                let score_modifiers = SdpaScoreModifiers::new().with_mask_mode(mask_mode);
                if config.causal_mask() || config.score_subgraph().is_some() {
                    let mut subgraph_config = DirectSdpaForwardConfig::new();
                    if matches!(
                        config.sequence_mode(),
                        DirectSdpaSequenceMode::SequenceLengths { .. }
                            | DirectSdpaSequenceMode::PaddingMask { .. }
                    ) {
                        subgraph_config =
                            subgraph_config.with_sequence_mode(config.sequence_mode());
                    }
                    if let Some(subgraph) = expanded.build_sdpa_unified_subgraph(
                        *q,
                        *k,
                        subgraph_config,
                        score_modifiers,
                        config.score_subgraph(),
                    )? {
                        let prepared_subgraph = subgraph.graph.prepare_with_context(ctx)?;
                        let subgraph_input_id =
                            prepared_subgraph.lowered.tensor_record(subgraph.input)?.id;
                        let subgraph_output_id =
                            prepared_subgraph.lowered.tensor_record(subgraph.output)?.id;
                        let cloned_tensor_lookup = subgraph
                            .cloned_tensors
                            .iter()
                            .map(|(source, target)| (*target, source))
                            .collect::<HashMap<_, _>>();
                        for (subgraph_tensor_ref, subgraph_record) in
                            &prepared_subgraph.lowered.tensors
                        {
                            if subgraph_record.tensor.is_virtual {
                                continue;
                            }
                            if let Some(source) = cloned_tensor_lookup.get(subgraph_tensor_ref) {
                                extra_binding_template_ids.push(subgraph_record.id);
                                extra_binding_replacements.push(BindingReplacement {
                                    source_id: lowered.tensor_record(**source)?.id,
                                    target_id: subgraph_record.id,
                                    byte_offset: 0,
                                });
                            } else if let Some(scalar_value) = subgraph_record.tensor.scalar_value {
                                extra_scalar_bindings.push((subgraph_record.id, scalar_value));
                            }
                        }
                        if let Operation::Sdpa(op) = operation {
                            lowered.operations[lowered_operation_index] = op.lower_with_tensors(
                                &lowered.backend_tensors,
                                Some(LoweredSdpaSubgraph {
                                    operation_graph: &prepared_subgraph.operation_graph,
                                    input_id: subgraph_input_id,
                                    output_id: subgraph_output_id,
                                }),
                            )?;
                        }
                    }
                }
            }
            lowered_operation_index += 1;
        }
        for id in extra_binding_template_ids {
            if !lowered.binding_template_ids.contains(&id) {
                lowered.binding_template_ids.push(id);
            }
        }
        for &(id, _) in &extra_scalar_bindings {
            if !lowered.binding_template_ids.contains(&id) {
                lowered.binding_template_ids.push(id);
            }
        }
        let alias_replacements = expanded
            .alias_bindings
            .iter()
            .map(|binding| {
                Ok(BindingReplacement {
                    source_id: lowered.tensor_record(binding.source)?.id,
                    target_id: lowered.tensor_record(binding.target)?.id,
                    byte_offset: binding.byte_offset,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let operation_descriptors = lowered
            .operations
            .iter()
            .map(LoweredOperation::descriptor)
            .collect::<Vec<_>>();
        let operation_graph = match ctx {
            Some(ctx) => OperationGraph::create_with_shape_config(
                ctx,
                &operation_descriptors,
                expanded.dynamic_shape_enabled,
                expanded.override_shape_enabled,
            )?,
            None => OperationGraph::create_deviceless_with_shape_config(
                &operation_descriptors,
                expanded.dynamic_shape_enabled,
                expanded.override_shape_enabled,
            )?,
        };
        let binding_replacements = std::mem::take(&mut lowered.binding_replacements);

        Ok(PreparedGraph {
            expanded,
            lowered,
            scalar_bindings: extra_scalar_bindings,
            binding_replacements: binding_replacements
                .into_iter()
                .chain(extra_binding_replacements)
                .chain(alias_replacements)
                .collect(),
            operation_graph,
        })
    }

    pub(crate) fn validate_runtime_feature_support_for_version(
        &self,
        cudnn_version: u64,
    ) -> Result<()> {
        if self.kernel_cache_enabled && !self.dynamic_shape_enabled {
            return Err(Error::FrontendKernelCacheRequiresDynamicShape);
        }

        if self.dynamic_shape_enabled || self.kernel_cache_enabled {
            support::DYNAMIC_SHAPE_OR_KERNEL_CACHE.require_frontend_feature(cudnn_version)?;
        }

        if self.override_shape_enabled {
            support::OVERRIDE_SHAPE.require_frontend_feature(cudnn_version)?;
        }

        if self.resolved_device_properties()?.is_some() {
            support::DEVICE_PROPERTIES.require_frontend_feature(cudnn_version)?;
        }

        if self.kernel_cache_json.is_some() {
            support::KERNEL_CACHE_SERIALIZATION.require_frontend_feature(cudnn_version)?;
        }

        Ok(())
    }

    fn candidates_from_engine_configs(
        &self,
        prepared: PreparedGraph,
        engine_configs: Vec<EngineConfig>,
    ) -> Result<PlanCandidates> {
        if engine_configs.is_empty() {
            return Err(Error::NoAvailableEngines);
        }
        let engine_metadata = engine_configs
            .iter()
            .map(|engine_config| {
                EngineConfigMetadata::create(&prepared.operation_graph, engine_config)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(PlanCandidates {
            graph: prepared.expanded,
            graph_name: self.name.clone(),
            tensors: prepared.lowered.tensors,
            binding_template_ids: prepared.lowered.binding_template_ids,
            scalar_bindings: prepared.scalar_bindings,
            binding_replacements: prepared.binding_replacements,
            operation_graph: prepared.operation_graph,
            engine_configs,
            engine_metadata,
        })
    }

    /// Compiles this graph using the requested heuristic modes.
    ///
    /// cuDNN heuristics return engine configurations ranked by expected
    /// performance; the usual workflow is to build the first supported choice.
    pub fn compile(&self, ctx: &Context, heuristics: &[HeuristicMode]) -> Result<CompiledGraph> {
        self.compile_with_config(
            ctx,
            &CompileConfig::new().with_heuristic_modes(heuristics.to_vec()),
        )
    }

    /// Compiles this graph using a full compile configuration.
    pub fn compile_with_config(
        &self,
        ctx: &Context,
        compile_config: &CompileConfig,
    ) -> Result<CompiledGraph> {
        let Some(plan_cache) = self.runtime.plan_cache.as_ref() else {
            return self
                .plan_candidates_with_config(ctx, compile_config)?
                .compile_with_config(ctx, compile_config);
        };

        let cache_key = self.plan_cache_key_with_context(compile_config, Some(ctx))?;
        let cached_choices = {
            let cache = plan_cache
                .lock()
                .map_err(|_| Error::FrontendPlanCacheLockPoisoned)?;
            cache.get(&cache_key).map(<[CachedPlanChoice]>::to_vec)
        };
        if let Some(cached_choices) = cached_choices
            && let Ok(compiled) =
                self.compile_with_cached_plan_choices(ctx, &cached_choices, compile_config)
        {
            return Ok(compiled);
        }

        let compiled = self
            .plan_candidates_with_config(ctx, compile_config)?
            .compile_with_config(ctx, compile_config)?;
        let choices = compiled.cached_plan_choices()?;
        let mut cache = plan_cache
            .lock()
            .map_err(|_| Error::FrontendPlanCacheLockPoisoned)?;
        cache.insert(cache_key, choices);
        Ok(compiled)
    }

    /// Compiles this graph without a live device context where supported.
    pub fn compile_deviceless(&self, heuristics: &[HeuristicMode]) -> Result<CompiledGraph> {
        self.compile_deviceless_with_config(
            &CompileConfig::new().with_heuristic_modes(heuristics.to_vec()),
        )
    }

    /// Compiles this graph deviceless using a full compile configuration.
    pub fn compile_deviceless_with_config(
        &self,
        compile_config: &CompileConfig,
    ) -> Result<CompiledGraph> {
        let Some(plan_cache) = self.runtime.plan_cache.as_ref() else {
            return self
                .plan_candidates_deviceless_with_config(compile_config)?
                .compile_deviceless_with_config(compile_config);
        };

        let cache_key = self.plan_cache_key_with_context(compile_config, None)?;
        let cached_choices = {
            let cache = plan_cache
                .lock()
                .map_err(|_| Error::FrontendPlanCacheLockPoisoned)?;
            cache.get(&cache_key).map(<[CachedPlanChoice]>::to_vec)
        };
        if let Some(cached_choices) = cached_choices
            && let Ok(compiled) =
                self.compile_deviceless_with_cached_plan_choices(&cached_choices, compile_config)
        {
            return Ok(compiled);
        }

        let compiled = self
            .plan_candidates_deviceless_with_config(compile_config)?
            .compile_deviceless_with_config(compile_config)?;
        let choices = compiled.cached_plan_choices()?;
        let mut cache = plan_cache
            .lock()
            .map_err(|_| Error::FrontendPlanCacheLockPoisoned)?;
        cache.insert(cache_key, choices);
        Ok(compiled)
    }

    /// Returns heuristic engine configurations for this graph.
    pub fn engine_configs(
        &self,
        ctx: &Context,
        heuristics: &[HeuristicMode],
    ) -> Result<Vec<EngineConfig>> {
        self.engine_configs_with_config(
            ctx,
            &CompileConfig::new().with_heuristic_modes(heuristics.to_vec()),
        )
    }

    /// Returns engine configurations using a full compile configuration.
    pub fn engine_configs_with_config(
        &self,
        ctx: &Context,
        compile_config: &CompileConfig,
    ) -> Result<Vec<EngineConfig>> {
        let prepared = self.prepare(ctx)?;
        Self::collect_engine_configs(Some(ctx), &prepared, compile_config)
    }

    pub fn engine_configs_deviceless(
        &self,
        heuristics: &[HeuristicMode],
    ) -> Result<Vec<EngineConfig>> {
        self.engine_configs_deviceless_with_config(
            &CompileConfig::new().with_heuristic_modes(heuristics.to_vec()),
        )
    }

    pub fn engine_configs_deviceless_with_config(
        &self,
        compile_config: &CompileConfig,
    ) -> Result<Vec<EngineConfig>> {
        let prepared = self.prepare_deviceless()?;
        Self::collect_engine_configs(None, &prepared, compile_config)
    }

    /// Returns plan candidates created from heuristic engine configurations.
    pub fn plan_candidates(
        &self,
        ctx: &Context,
        heuristics: &[HeuristicMode],
    ) -> Result<PlanCandidates> {
        self.plan_candidates_with_config(
            ctx,
            &CompileConfig::new().with_heuristic_modes(heuristics.to_vec()),
        )
    }

    /// Returns plan candidates using a full compile configuration.
    pub fn plan_candidates_with_config(
        &self,
        ctx: &Context,
        options: &CompileConfig,
    ) -> Result<PlanCandidates> {
        let prepared = self.prepare(ctx)?;
        let engine_configs = Self::collect_engine_configs(Some(ctx), &prepared, options)?;
        self.candidates_from_engine_configs(prepared, engine_configs)
    }

    pub fn plan_candidates_deviceless(
        &self,
        heuristics: &[HeuristicMode],
    ) -> Result<PlanCandidates> {
        self.plan_candidates_deviceless_with_config(
            &CompileConfig::new().with_heuristic_modes(heuristics.to_vec()),
        )
    }

    pub fn plan_candidates_deviceless_with_config(
        &self,
        compile_config: &CompileConfig,
    ) -> Result<PlanCandidates> {
        let prepared = self.prepare_deviceless()?;
        let engine_configs = Self::collect_engine_configs(None, &prepared, compile_config)?;
        self.candidates_from_engine_configs(prepared, engine_configs)
    }

    pub fn compile_with_engine_config(
        &self,
        ctx: &Context,
        engine_config: &EngineConfig,
    ) -> Result<CompiledGraph> {
        self.compile_with_engine_configs(ctx, from_ref(engine_config))
    }

    pub fn compile_with_engine_configs(
        &self,
        ctx: &Context,
        engine_configs: &[EngineConfig],
    ) -> Result<CompiledGraph> {
        let prepared = self.prepare(ctx)?;
        let device_properties = self.resolved_device_properties()?;

        let engine_configs = engine_configs
            .iter()
            .map(|engine_config| {
                let engine = engine_config.engine()?;
                let knob_choices = engine_config
                    .knob_choices()?
                    .into_iter()
                    .map(|choice| (choice.knob_type, choice.value))
                    .collect::<Vec<_>>();
                EngineConfig::create_with_config(
                    ctx,
                    &prepared.operation_graph,
                    engine.index(),
                    &knob_choices,
                    device_properties.as_deref(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        self.candidates_from_engine_configs(prepared, engine_configs)?
            .compile(ctx)
    }

    pub(crate) fn compile_with_cached_plan_choices(
        &self,
        ctx: &Context,
        choices: &[CachedPlanChoice],
        compile_config: &CompileConfig,
    ) -> Result<CompiledGraph> {
        let engine_configs = choices
            .iter()
            .map(|choice| (choice.engine_index, choice.knob_choices.clone()))
            .collect::<Vec<_>>();
        let cached_policy = match compile_config.build_policy() {
            BuildPlanPolicy::AllSupported => BuildPlanPolicy::AllSupported,
            _ => BuildPlanPolicy::FirstSupported,
        };
        self.candidates_with_engines(ctx, &engine_configs)?
            .compile_with_config(
                ctx,
                &compile_config.clone().with_build_policy(cached_policy),
            )
    }

    fn compile_deviceless_with_cached_plan_choices(
        &self,
        choices: &[CachedPlanChoice],
        compile_config: &CompileConfig,
    ) -> Result<CompiledGraph> {
        let engine_configs = choices
            .iter()
            .map(|choice| (choice.engine_index, choice.knob_choices.clone()))
            .collect::<Vec<_>>();
        let cached_policy = match compile_config.build_policy() {
            BuildPlanPolicy::AllSupported => BuildPlanPolicy::AllSupported,
            _ => BuildPlanPolicy::FirstSupported,
        };
        self.candidates_deviceless_with_engines(&engine_configs)?
            .compile_deviceless_with_config(
                &compile_config.clone().with_build_policy(cached_policy),
            )
    }

    /// Compiles this graph with a specific engine index and knob choices.
    ///
    /// This corresponds to the custom execution plan workflow for creating
    /// plans from a chosen engine and knobs.
    pub fn compile_with_engine(
        &self,
        ctx: &Context,
        engine_index: i64,
        knob_choices: &[(BackendKnobType, i64)],
    ) -> Result<CompiledGraph> {
        let prepared = self.prepare(ctx)?;
        let device_properties = self.resolved_device_properties()?;
        let engine_config = EngineConfig::create_with_config(
            ctx,
            &prepared.operation_graph,
            EngineIndex::new(engine_index),
            knob_choices,
            device_properties.as_deref(),
        )?;

        self.candidates_from_engine_configs(prepared, vec![engine_config])?
            .compile(ctx)
    }

    pub fn candidates_with_engines(
        &self,
        ctx: &Context,
        engine_configs: &[(i64, Vec<(BackendKnobType, i64)>)],
    ) -> Result<PlanCandidates> {
        let prepared = self.prepare(ctx)?;
        let device_properties = self.resolved_device_properties()?;
        let mut created = Vec::new();
        let mut last_error = None;
        for (engine_index, knob_choices) in engine_configs {
            match EngineConfig::create_with_config(
                ctx,
                &prepared.operation_graph,
                EngineIndex::new(*engine_index),
                knob_choices,
                device_properties.as_deref(),
            ) {
                Ok(engine_config) => created.push(engine_config),
                Err(error) => last_error = Some(error),
            }
        }

        if created.is_empty() {
            return Err(last_error.unwrap_or(Error::NoAvailableEngines));
        }

        self.candidates_from_engine_configs(prepared, created)
    }

    pub fn candidates_deviceless_with_engines(
        &self,
        engine_configs: &[(i64, Vec<(BackendKnobType, i64)>)],
    ) -> Result<PlanCandidates> {
        let prepared = self.prepare_deviceless()?;
        let device_properties = self
            .resolved_device_properties()?
            .ok_or(Error::FrontendDevicelessRequiresDeviceProperties)?;
        let mut created = Vec::new();
        let mut last_error = None;
        for (engine_index, knob_choices) in engine_configs {
            match EngineConfig::create_deviceless(
                &prepared.operation_graph,
                EngineIndex::new(*engine_index),
                knob_choices,
                device_properties.as_ref(),
            ) {
                Ok(engine_config) => created.push(engine_config),
                Err(error) => last_error = Some(error),
            }
        }

        if created.is_empty() {
            return Err(last_error.unwrap_or(Error::NoAvailableEngines));
        }

        self.candidates_from_engine_configs(prepared, created)
    }

    /// Returns the number of engines available for this operation graph.
    pub fn engine_count(&self, ctx: &Context) -> Result<usize> {
        Ok(self.engine_indices(ctx)?.len())
    }

    /// Returns the engine indices available for this operation graph.
    pub fn engine_indices(&self, ctx: &Context) -> Result<Vec<i64>> {
        let prepared = self.prepare(ctx)?;
        Engine::engine_indices_for_operation_graph(prepared.operation_graph.descriptor())
    }

    /// Returns the adjustable knobs supported by an engine for this graph.
    pub fn knobs_for_engine(
        &self,
        ctx: &Context,
        engine_index: i64,
    ) -> Result<Vec<EngineKnobInfo>> {
        let prepared = self.prepare(ctx)?;
        Engine::from_index(EngineIndex::new(engine_index))
            .knob_infos(ctx, prepared.operation_graph.descriptor())
    }

    pub fn numerical_notes_for_engine(
        &self,
        ctx: &Context,
        engine_index: i64,
    ) -> Result<Vec<BackendNumericalNote>> {
        let prepared = self.prepare(ctx)?;
        Engine::from_index(EngineIndex::new(engine_index))
            .numerical_notes(prepared.operation_graph.descriptor())
    }

    pub fn behavior_notes_for_engine(
        &self,
        ctx: &Context,
        engine_index: i64,
    ) -> Result<Vec<BackendBehaviorNote>> {
        let prepared = self.prepare(ctx)?;
        Engine::from_index(EngineIndex::new(engine_index))
            .behavior_notes(prepared.operation_graph.descriptor())
    }
}
