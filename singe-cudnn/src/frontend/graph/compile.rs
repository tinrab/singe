use std::{borrow::Borrow, collections::BTreeMap};

use crate::{
    backend::execution::{
        EngineConfig, OperationGraph, OperationGraphConfig, attention::SdpaSubgraph,
    },
    context::Context,
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, PreparedGraph, ValidatedGraph, subgraphs::ScoreSubgraph},
        logging,
        lower::{LoweredGraph, LoweredOperation, tensor_at},
        operation::AttentionBackwardHeuristicOverride,
        plan::{CompileConfig, CompiledGraph, EngineChoice, PlanCandidates},
        require_baseline,
    },
};

impl Graph {
    /// Validates this recipe and creates native descriptors for engine discovery and planning.
    /// Pass Some(context) for a live target or None with attached deviceless device properties.
    /// The returned graph retains native tensors, nested score graphs, constants and workspace metadata.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported runtime features, invalid declarations, operation geometry, references, runtime bounds or shared layouts, an empty graph, target-resolution failure, or native descriptor construction failure.
    pub fn prepare(&self, context: Option<&Context>) -> Result<PreparedGraph> {
        self.validate(context)?.prepare(context)
    }

    pub(crate) fn validate_runtime_feature_support_for_version(
        &self,
        cudnn_version: u64,
    ) -> Result<()> {
        require_baseline(cudnn_version)?;
        if self.kernel_cache_enabled && !self.config.dynamic_shape_enabled {
            return Err(FrontendError::KernelCacheRequiresDynamicShape.into());
        }

        Ok(())
    }

    /// Compiles this graph for a live context, or deviceless with `None`.
    ///
    /// cuDNN heuristics return engine configurations ranked by expected
    /// performance; the usual workflow is to build the first supported choice.
    /// An attached plan cache is consulted before preparation and updated after compilation.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported runtime features, invalid declarations, operation geometry, references, runtime bounds or shared layouts, an empty graph, target-resolution failure, or native descriptor construction failure, plan-cache key construction or a poisoned cache mutex, target-property resolution, incompatible attention planning overrides, engine discovery or configuration metadata failure, no candidates surviving filters, or failure to build a usable plan.
    pub fn compile(
        &self,
        context: Option<&Context>,
        compile_config: &CompileConfig,
    ) -> Result<CompiledGraph> {
        let cache = self
            .runtime
            .plan_cache
            .as_ref()
            .map(|cache| Ok::<_, Error>((cache, self.plan_cache_key(compile_config, context)?)))
            .transpose()?;
        if let Some((cache, key)) = &cache {
            let cached_choices = {
                let cache = cache
                    .lock()
                    .map_err(|_| FrontendError::PlanCacheLockPoisoned)?;
                cache.get(key).map(<[EngineChoice]>::to_vec)
            };
            if let Some(cached_choices) = cached_choices
                && let Ok(compiled) =
                    self.compile_with_cached_plan_choices(context, &cached_choices, compile_config)
            {
                return Ok(compiled);
            }
        }

        let compiled = self
            .prepare(context)?
            .plan_candidates(context, compile_config)?
            .build_candidates(context, compile_config)?
            .compile()?;
        if let Some((cache, key)) = cache {
            cache
                .lock()
                .map_err(|_| FrontendError::PlanCacheLockPoisoned)?
                .insert(key, compiled.cached_plan_choices());
        }
        Ok(compiled)
    }

    /// Returns heuristic engine configurations for a live context, or deviceless with `None`.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported runtime features, invalid declarations, operation geometry, references, runtime bounds or shared layouts, an empty graph, target-resolution failure, or native descriptor construction failure, or target-property resolution, incompatible attention planning overrides, engine discovery or configuration metadata failure. Successfully queried configurations excluded by filters yield an empty result.
    pub fn engine_configs(
        &self,
        context: Option<&Context>,
        compile_config: &CompileConfig,
    ) -> Result<Vec<EngineConfig>> {
        self.prepare(context)?
            .engine_configs(context, compile_config)
    }

    /// Returns plan candidates for a live context, or deviceless with `None`.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported runtime features, invalid declarations, operation geometry, references, runtime bounds or shared layouts, an empty graph, target-resolution failure, or native descriptor construction failure, target-property resolution, incompatible attention planning overrides, engine discovery or configuration metadata failure, or no candidates surviving filters.
    pub fn plan_candidates(
        &self,
        context: Option<&Context>,
        compile_config: &CompileConfig,
    ) -> Result<PlanCandidates> {
        self.prepare(context)?
            .plan_candidates(context, compile_config)
    }

    /// Compiles using choices and resource targets from owned or shared native configurations.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported runtime features, invalid declarations, operation geometry, references, runtime bounds or shared layouts, an empty graph, target-resolution failure, or native descriptor construction failure, source configuration metadata or target incompatibility, engine reconstruction or candidate metadata failure, no candidates surviving filters, or failure to build a usable plan.
    pub fn compile_with_engine_configs(
        &self,
        context: &Context,
        engine_configs: &[impl Borrow<EngineConfig>],
    ) -> Result<CompiledGraph> {
        self.prepare(Some(context))?
            .candidates_with_engine_configs(Some(context), engine_configs, &CompileConfig::new())?
            .compile(Some(context), &CompileConfig::new())
    }

    pub(crate) fn compile_with_cached_plan_choices(
        &self,
        context: Option<&Context>,
        choices: &[EngineChoice],
        compile_config: &CompileConfig,
    ) -> Result<CompiledGraph> {
        let prepared = self.prepare(context)?;
        let engine_configs = {
            let required = AttentionBackwardHeuristicOverride::resolve(
                &prepared.expanded.attention_backward_heuristic_overrides,
            )?;
            choices
                .iter()
                .filter(|choice| {
                    required
                        .as_deref()
                        .is_none_or(|required| required.matches(choice))
                })
                .cloned()
                .collect::<Vec<_>>()
        };
        prepared
            .candidates_with_engines(context, &engine_configs, compile_config)?
            .build_candidates(context, compile_config)?
            .compile()
    }

    /// Compiles this graph with specific engine indices and knob choices.
    ///
    /// This corresponds to the custom execution plan workflow for creating
    /// plans from chosen engines and knobs.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported runtime features, invalid declarations, operation geometry, references, runtime bounds or shared layouts, an empty graph, target-resolution failure, or native descriptor construction failure, failure to construct the requested engine choices or read their metadata, no candidates surviving filters, or failure to build a usable plan.
    pub fn compile_with_engines(
        &self,
        context: &Context,
        choices: &[EngineChoice],
    ) -> Result<CompiledGraph> {
        self.candidates_with_engines(Some(context), choices)?
            .compile(Some(context), &CompileConfig::new())
    }

    /// Returns plan candidates for explicit engine choices on a live context, or deviceless with `None`.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported runtime features, invalid declarations, operation geometry, references, runtime bounds or shared layouts, an empty graph, target-resolution failure, or native descriptor construction failure, target-property resolution failure, failure to construct any requested engine configuration, metadata-query failure, or no candidates surviving filters.
    pub fn candidates_with_engines(
        &self,
        context: Option<&Context>,
        choices: &[EngineChoice],
    ) -> Result<PlanCandidates> {
        self.prepare(context)?
            .candidates_with_engines(context, choices, &CompileConfig::new())
    }
}

impl ValidatedGraph {
    /// Creates native descriptors from this validated target-specific snapshot.
    /// Reuses reconciled declarations and binding requirements without repeating expansion or validation.
    /// Use the validation target or a compatible live context, or None for deviceless preparation.
    ///
    /// # Errors
    /// Returns an error for an empty operation graph or rejected native descriptor construction.
    /// Engine discovery and support selection remain separate from preparation.
    pub fn prepare(self, context: Option<&Context>) -> Result<PreparedGraph> {
        let Self {
            recipe,
            expanded,
            layout,
            subgraphs,
            reducers,
        } = self;
        if expanded.operations.is_empty() {
            return Err(FrontendError::GraphEmpty.into());
        }
        logging::diagnostic!(
            "preparing validated graph {:?}: {} operations, deviceless={}",
            recipe.name(),
            expanded.operations.len(),
            context.is_none(),
        );
        let subgraphs = subgraphs
            .into_iter()
            .map(|(index, subgraph)| {
                Ok((
                    index,
                    ScoreSubgraph {
                        graph: subgraph.graph.prepare(context)?,
                        input: subgraph.input,
                        output: subgraph.output,
                        tensor_captures: subgraph.tensor_captures,
                        captured_tensors: subgraph.captured_tensors,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mut lowered = {
            let native_subgraphs = subgraphs
                .iter()
                .map(|(&index, subgraph)| {
                    Ok((
                        index,
                        SdpaSubgraph {
                            operation_graph: &subgraph.graph.operation_graph,
                            input_id: tensor_at(
                                &subgraph.graph.lowered.backend_tensors,
                                subgraph.input,
                            )?
                            .id(),
                            output_id: tensor_at(
                                &subgraph.graph.lowered.backend_tensors,
                                subgraph.output,
                            )?
                            .id(),
                        },
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?;
            LoweredGraph::lower(&expanded, layout, &native_subgraphs, reducers, context)?
        };
        lowered
            .retained_subgraphs
            .extend(subgraphs.into_values().map(|subgraph| subgraph.graph));
        let operation_descriptors = lowered
            .operations
            .iter()
            .map(LoweredOperation::descriptor)
            .collect::<Vec<_>>();
        let config = OperationGraphConfig {
            dynamic_shape_enabled: expanded.config.dynamic_shape_enabled,
            override_shape_enabled: expanded.config.override_shape_enabled,
        };
        let operation_graph = match context {
            Some(context) => OperationGraph::create(context, &operation_descriptors, config)?,
            None => OperationGraph::create_deviceless(&operation_descriptors, config)?,
        };
        logging::diagnostic!(
            "prepared graph {:?}: {} native tensors, {} bindings, {} frontend workspace bytes",
            recipe.name(),
            lowered.backend_tensors.len(),
            lowered.layout.binding_template_ids.len(),
            lowered.layout.workspace_layout.byte_len(),
        );
        Ok(PreparedGraph {
            recipe,
            operation_graph,
            expanded,
            lowered,
        })
    }
}
