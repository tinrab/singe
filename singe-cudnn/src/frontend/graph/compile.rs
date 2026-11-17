use std::collections::BTreeMap;

use singe_core::LibraryVersion;

use crate::{
    backend::{
        execution::{EngineChoice, OperationGraph, OperationGraphOptions},
        operation::attention::SdpaSubgraph,
    },
    context::Context,
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, PreparedGraph, ValidatedGraph, subgraphs::ScoreSubgraph},
        logging,
        lower::{LoweredGraph, LoweredOperation, tensor_at},
        operation::AttentionBackwardHeuristicOverride,
        plan::{CompileConfig, CompiledGraph},
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
    /// Returns an error if a runtime feature is unsupported, a declaration, operation geometry, reference, runtime bound, or shared layout is invalid, the graph is empty, resolving the target fails, or building a native descriptor fails.
    pub fn prepare(&self, context: Option<&Context>) -> Result<PreparedGraph> {
        self.validate(context)?.prepare(context)
    }

    pub(crate) fn validate_runtime_feature_support_for_version(
        &self,
        cudnn_version: LibraryVersion,
    ) -> Result<()> {
        require_baseline(cudnn_version)?;
        if self.kernel_cache_enabled && !self.config.dynamic_shape_enabled {
            return Err(FrontendError::KernelCacheRequiresDynamicShape.into());
        }

        Ok(())
    }

    /// Compiles this graph for a live context, or deviceless with `None`.
    ///
    /// cuDNN heuristics return engine configurations ranked by expected performance; the usual workflow is to build the first supported choice.
    /// An attached plan cache is consulted before preparation and updated after compilation.
    ///
    /// # Errors
    ///
    /// Returns an error if a runtime feature is unsupported, a declaration, operation geometry, reference, runtime bound, or shared layout is invalid, the graph is empty, resolving the target or its properties fails, building a native descriptor or the plan-cache key fails, the cache mutex is poisoned, attention planning overrides are incompatible, engine discovery or configuration metadata fails, no candidate survives the filters, or no usable plan builds.
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
                let cache = cache.lock().map_err(|_| Error::LockPoisoned {
                    name: "plan cache".into(),
                })?;
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
                .map_err(|_| Error::LockPoisoned {
                    name: "plan cache".into(),
                })?
                .insert(key, compiled.cached_plan_choices());
        }
        Ok(compiled)
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
}

impl ValidatedGraph {
    /// Creates native descriptors from this validated target-specific snapshot.
    /// Reuses reconciled declarations and binding requirements without repeating expansion or validation.
    /// Use the validation target or a compatible live context, or None for deviceless preparation.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation graph is empty or cuDNN rejects building the native descriptor.
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
        let config = OperationGraphOptions {
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
