use std::{borrow::Borrow, collections::HashSet, sync::Arc};

use crate::{
    backend::{
        engine::{Engine, EngineIndex},
        execution::{
            EngineChoice, EngineConfig, EngineHeuristics, EngineHeuristicsResult,
            EnginePlanningConfig, OperationGraph,
        },
        heuristic::{BackendHeuristicMode, HeuristicQueryPolicy},
    },
    context::Context,
    error::{Error, Result},
    frontend::{
        graph::Graph,
        logging,
        lower::LoweredGraph,
        operation::AttentionBackwardHeuristicOverride,
        plan::{
            CompileConfig, PlanArtifacts, PlanCandidates, PreparedEngineConfig, RequiredTensor,
        },
    },
};

impl PreparedGraph {
    /// Original recipe, before target-specific expansion and native lowering.
    #[must_use]
    pub fn recipe(&self) -> &Graph {
        &self.recipe
    }

    /// Expanded graph whose operations and layouts produced the native descriptors.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.expanded
    }

    /// Finalized native operation graph shared by engine queries and candidate construction.
    #[must_use]
    pub fn operation_graph(&self) -> &OperationGraph {
        &self.operation_graph
    }

    /// Caller storage and native uses resolved during preparation, before engine discovery.
    /// Includes alias offsets, access roles and addressing dependencies.
    /// Virtual tensors, constants, host scalars and frontend workspace do not require device bindings.
    #[must_use]
    pub fn required_tensor_bindings(&self) -> &[RequiredTensor] {
        &self.lowered.layout.required_tensors
    }

    /// Global native engine count for this prepared graph and target.
    ///
    /// # Errors
    ///
    /// Returns an error if the retained CUDA context cannot be bound, the native engine-count query fails, or the count cannot fit `usize`.
    pub fn engine_count(&self) -> Result<usize> {
        self.operation_graph.engine_count()
    }

    /// Global native engine indices for this prepared graph and target.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding, the native engine-count query, or a checked count conversion fails.
    pub fn engine_indices(&self) -> Result<Vec<EngineIndex>> {
        Engine::engine_indices_for_operation_graph(&self.operation_graph)
    }

    /// Finalizes an engine against these retained native descriptors and target properties.
    /// Omitted resource targets inherit the graph defaults.
    ///
    /// # Errors
    ///
    /// Returns an error if target properties cannot be resolved, resource targets are invalid, targets disagree or are unavailable, context binding fails, or native engine creation, configuration, or finalization fails.
    pub fn engine(&self, index: EngineIndex, planning: EnginePlanningConfig) -> Result<Engine> {
        let properties = self.expanded.resolved_device_properties()?;
        let planning = EnginePlanningConfig {
            sm_count_target: planning
                .sm_count_target
                .or_else(|| self.expanded.sm_count_target()),
            ..planning
        };
        Engine::create_with_context(
            None,
            &self.operation_graph,
            index,
            properties.as_deref(),
            planning,
        )
    }

    /// Consumes this prepared graph into heuristic plan candidates without repeating native lowering.
    /// Supply a compatible live handle, or None to reuse this graph's retained target.
    ///
    /// # Errors
    ///
    /// Returns an error if target properties or attention planning overrides cannot be resolved, native discovery or metadata queries fail without a usable fallback, or no candidate survives filtering.
    pub fn plan_candidates(
        self,
        context: Option<&Context>,
        config: &CompileConfig,
    ) -> Result<PlanCandidates> {
        let configs = self.prepared_engine_configs(context, config)?;
        self.into_candidates(configs)
    }

    fn prepare_engine_configs(
        configs: Vec<EngineConfig>,
        compile_config: &CompileConfig,
        frontend_workspace_bytes: usize,
    ) -> Result<Vec<PreparedEngineConfig>> {
        let mut prepared = Vec::with_capacity(configs.len());
        let mut queried = false;
        let mut last_error = None;
        for (position, config) in configs.into_iter().enumerate() {
            match PreparedEngineConfig::create(config, frontend_workspace_bytes) {
                Ok(config) => {
                    queried = true;
                    if config.metadata.matches_compile_config(compile_config) {
                        prepared.push(config);
                    }
                }
                Err(error) => {
                    logging::diagnostic!(
                        "engine configuration at position {position} failed during metadata query: {error}",
                    );
                    last_error = Some(error);
                }
            }
        }
        if !queried && let Some(error) = last_error {
            return Err(error);
        }
        Ok(prepared)
    }

    /// Queries heuristic configurations against this prepared operation graph.
    /// Supply a compatible live handle, or None to reuse the retained CUDA context or attached device properties.
    /// Failed configuration metadata queries are logged while usable alternatives retain heuristic order.
    /// If every heuristic metadata query fails, explicit engine enumeration supplies the fallback.
    /// Excluding all successfully queried configurations through filters returns an empty collection.
    ///
    /// # Errors
    ///
    /// Returns an error if target properties or attention planning overrides cannot be resolved, required native engine construction fails, or heuristic and enumeration fallback cannot produce configurations with queryable metadata. Filtering all successfully queried configurations yields an empty result.
    pub fn engine_configs(
        &self,
        context: Option<&Context>,
        compile_config: &CompileConfig,
    ) -> Result<Vec<EngineConfig>> {
        Ok(self
            .prepared_engine_configs(context, compile_config)?
            .into_iter()
            .map(|config| config.engine_config)
            .collect())
    }

    /// Queries requested heuristic modes while retaining each attempted native outcome.
    /// Uses the compilation's mode list and resource targets, inheriting an unset SM target from the graph.
    /// An empty compilation mode list uses Instant, matching automatic discovery.
    /// Supply a compatible live handle or None to reuse this preparation's retained target.
    /// This explicit query does not apply attention engine overrides, metadata filters or enumeration fallback.
    /// Use `candidates_with_engine_configs` to turn returned configurations into filtered candidates.
    ///
    /// # Errors
    /// Returns target-property resolution errors before querying modes.
    /// Native mode failures remain available in the returned statuses alongside successful configurations.
    pub fn query_heuristics(
        &self,
        context: Option<&Context>,
        compile_config: &CompileConfig,
        policy: HeuristicQueryPolicy,
    ) -> Result<EngineHeuristicsResult> {
        let device_properties = self.expanded.resolved_device_properties()?;
        let modes = match compile_config.heuristic_modes() {
            [] => vec![BackendHeuristicMode::Instant],
            modes => modes
                .iter()
                .copied()
                .map(BackendHeuristicMode::from)
                .collect(),
        };
        let planning = EnginePlanningConfig {
            sm_count_target: compile_config
                .sm_count_target()
                .or_else(|| self.expanded.sm_count_target()),
            shared_memory_limit: compile_config.shared_memory_limit(),
        };
        Ok(EngineHeuristics::query_modes(
            context,
            &self.operation_graph,
            &modes,
            device_properties.as_deref(),
            planning,
            policy,
        ))
    }

    fn prepared_engine_configs(
        &self,
        context: Option<&Context>,
        compile_config: &CompileConfig,
    ) -> Result<Vec<PreparedEngineConfig>> {
        let device_properties = self.expanded.resolved_device_properties()?;
        let frontend_workspace_bytes = self.lowered.layout.workspace_layout.byte_len();
        let planning = EnginePlanningConfig {
            sm_count_target: compile_config
                .sm_count_target()
                .or_else(|| self.expanded.sm_count_target()),
            shared_memory_limit: compile_config.shared_memory_limit(),
        };
        if let Some(required) = AttentionBackwardHeuristicOverride::resolve(
            &self.expanded.attention_backward_heuristic_overrides,
        )? {
            let config = EngineConfig::create_for_target(
                context,
                &self.operation_graph,
                &required.choice,
                device_properties.as_deref(),
                planning,
            )?;
            return Self::prepare_engine_configs(
                vec![config],
                compile_config,
                frontend_workspace_bytes,
            );
        }
        let queried =
            self.query_heuristics(context, compile_config, HeuristicQueryPolicy::EvaluateAll)?;
        let engine_configs = queried.engine_configs;
        let mut last_error = None;
        for status in queried.statuses {
            if let Err(error) = status.result {
                logging::diagnostic!("heuristic mode {} failed: {error}", status.mode);
                last_error = Some(error);
            }
        }

        if !engine_configs.is_empty() {
            match Self::prepare_engine_configs(
                engine_configs,
                compile_config,
                frontend_workspace_bytes,
            ) {
                Ok(mut configs) => {
                    let mut seen = HashSet::new();
                    configs.retain(|config| seen.insert(config.metadata.name.clone()));
                    return Ok(configs);
                }
                Err(error) => last_error = Some(error),
            }
        }

        let engine_indices = self.engine_indices()?;
        let mut explicit_configs = Vec::new();
        for engine_index in engine_indices {
            let result = EngineConfig::create_for_target(
                context,
                &self.operation_graph,
                &EngineChoice::new(engine_index, []),
                device_properties.as_deref(),
                planning,
            );
            match result {
                Ok(config) => explicit_configs.push(config),
                Err(error) => last_error = Some(error),
            }
        }
        if !explicit_configs.is_empty() {
            match Self::prepare_engine_configs(
                explicit_configs,
                compile_config,
                frontend_workspace_bytes,
            ) {
                Ok(configs) => return Ok(configs),
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or(Error::NoAvailableEngines))
    }

    /// Creates candidates from explicit engine and knob choices without preparing the graph again.
    /// Resource overrides and engine filters apply to the configurations built here.
    /// Explicit choices bypass automatic attention heuristic overrides, as in C++ custom planning.
    ///
    /// # Errors
    ///
    /// Returns an error if target properties cannot be resolved, no requested engine configuration can be constructed, configuration metadata cannot be queried, or no candidate survives filtering.
    pub fn candidates_with_engines(
        self,
        context: Option<&Context>,
        engine_configs: &[EngineChoice],
        compile_config: &CompileConfig,
    ) -> Result<PlanCandidates> {
        let device_properties = self.expanded.resolved_device_properties()?;
        let planning = EnginePlanningConfig {
            sm_count_target: compile_config
                .sm_count_target()
                .or_else(|| self.expanded.sm_count_target()),
            shared_memory_limit: compile_config.shared_memory_limit(),
        };
        let mut created = Vec::new();
        let mut last_error = None;
        for choice in engine_configs {
            let result = EngineConfig::create_for_target(
                context,
                &self.operation_graph,
                choice,
                device_properties.as_deref(),
                planning,
            );
            match result {
                Ok(engine_config) => created.push(engine_config),
                Err(error) => last_error = Some(error),
            }
        }

        if created.is_empty() {
            return Err(last_error.unwrap_or(Error::NoAvailableEngines));
        }

        let created = Self::prepare_engine_configs(
            created,
            compile_config,
            self.lowered.layout.workspace_layout.byte_len(),
        )?;
        self.into_candidates(created)
    }

    /// Copies existing engine choices and resource targets onto this prepared operation graph.
    /// Accepts owned or shared configurations, including snapshots returned by native execution plans.
    /// Compile controls override each source configuration's targets; otherwise those targets are retained.
    /// An unset SM target inherits the graph default. Explicit choices bypass attention heuristic overrides.
    /// All source configurations must finalize for the supplied target before candidate filters apply.
    /// Native construction and metadata-query failures from any source configuration propagate.
    ///
    /// # Errors
    ///
    /// Returns an error if target properties or source configuration choices or resource targets cannot be read, any engine reconstruction or metadata query fails, or no candidate survives filtering.
    pub fn candidates_with_engine_configs(
        self,
        context: Option<&Context>,
        engine_configs: &[impl Borrow<EngineConfig>],
        compile_config: &CompileConfig,
    ) -> Result<PlanCandidates> {
        let device_properties = self.expanded.resolved_device_properties()?;
        let created = engine_configs
            .iter()
            .map(|engine_config| {
                let engine_config: &EngineConfig = engine_config.borrow();
                let choice = engine_config.choice()?;
                let inherited = engine_config.planning_config()?;
                let planning = EnginePlanningConfig {
                    sm_count_target: compile_config
                        .sm_count_target()
                        .or(inherited.sm_count_target)
                        .or_else(|| self.expanded.sm_count_target()),
                    shared_memory_limit: compile_config
                        .shared_memory_limit()
                        .or(inherited.shared_memory_limit),
                };
                EngineConfig::create_for_target(
                    context,
                    &self.operation_graph,
                    choice,
                    device_properties.as_deref(),
                    planning,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let frontend_workspace_bytes = self.lowered.layout.workspace_layout.byte_len();
        let mut created = created
            .into_iter()
            .map(|config| PreparedEngineConfig::create(config, frontend_workspace_bytes))
            .collect::<Result<Vec<_>>>()?;
        created.retain(|config| config.metadata.matches_compile_config(compile_config));
        self.into_candidates(created)
    }

    pub(crate) fn into_candidates(
        self,
        engine_configs: Vec<PreparedEngineConfig>,
    ) -> Result<PlanCandidates> {
        if engine_configs.is_empty() {
            return Err(Error::NoAvailableEngines);
        }
        let entries = PlanCandidates::create_entries(engine_configs);

        let graph_name = self.recipe.config.name.clone();
        Ok(PlanCandidates {
            recipe: self.recipe,
            graph: self.expanded,
            artifacts: PlanArtifacts {
                graph_name,
                tensors: self.lowered.layout.tensors.clone(),
                scalar_bindings: self.lowered.layout.scalar_bindings.clone(),
                binding_replacements: self.lowered.layout.binding_replacements.clone(),
                required_tensors: self.lowered.layout.required_tensors.clone(),
            },
            binding_template_ids: self.lowered.layout.binding_template_ids.clone(),
            operation_graph: self.operation_graph,
            lowered: Arc::new(self.lowered),
            entries,
        })
    }
}

/// Retained native operation graph produced by graph validation and preparation.
/// Query engines and configurations before consuming it into plan candidates.
/// Owns the original recipe, expanded operations, native tensors and captured graph resources.
#[derive(Debug)]
pub struct PreparedGraph {
    pub(crate) recipe: Graph,
    pub(crate) operation_graph: OperationGraph,
    pub(crate) expanded: Graph,
    pub(crate) lowered: LoweredGraph,
}
