use std::collections::BTreeMap;

macro_rules! impl_engine_config_filter_methods {
    () => {
        pub fn filter(
            mut self,
            filter: crate::frontend::plan::metadata::EngineConfigFilter<'_>,
        ) -> crate::error::Result<Self> {
            self.apply_filter(filter);
            Ok(self)
        }

        pub fn deselect_workspace_greater_than(
            mut self,
            max_workspace_size: usize,
        ) -> crate::error::Result<Self> {
            let filter = crate::frontend::plan::metadata::EngineConfigFilter::new()
                .with_max_workspace_size(max_workspace_size);
            self.apply_filter(filter);
            Ok(self)
        }

        pub fn deselect_shared_memory_greater_than(
            mut self,
            max_shared_memory_size: usize,
        ) -> crate::error::Result<Self> {
            let filter = crate::frontend::plan::metadata::EngineConfigFilter::new()
                .with_max_shared_memory_size(max_shared_memory_size);
            self.apply_filter(filter);
            Ok(self)
        }

        pub fn deselect_engines(
            mut self,
            blocked_name_substrings: impl Into<Vec<String>>,
        ) -> crate::error::Result<Self> {
            let blocked_name_substrings = blocked_name_substrings.into();
            let filter = crate::frontend::plan::metadata::EngineConfigFilter::new()
                .with_excluded_engine_name_substrings(&blocked_name_substrings);
            self.apply_filter(filter);
            Ok(self)
        }

        pub fn require_numerical_notes(
            mut self,
            required_notes: impl Into<Vec<crate::behavior::BackendNumericalNote>>,
        ) -> crate::error::Result<Self> {
            let required_notes = required_notes.into();
            let filter = crate::frontend::plan::metadata::EngineConfigFilter::new()
                .with_included_numerical_notes(&required_notes);
            self.apply_filter(filter);
            Ok(self)
        }

        pub fn exclude_numerical_notes(
            mut self,
            excluded_notes: impl Into<Vec<crate::behavior::BackendNumericalNote>>,
        ) -> crate::error::Result<Self> {
            let excluded_notes = excluded_notes.into();
            let filter = crate::frontend::plan::metadata::EngineConfigFilter::new()
                .with_excluded_numerical_notes(&excluded_notes);
            self.apply_filter(filter);
            Ok(self)
        }

        pub fn require_behavior_notes(
            mut self,
            required_notes: impl Into<Vec<crate::behavior::BackendBehaviorNote>>,
        ) -> crate::error::Result<Self> {
            let required_notes = required_notes.into();
            let filter = crate::frontend::plan::metadata::EngineConfigFilter::new()
                .with_included_behavior_notes(&required_notes);
            self.apply_filter(filter);
            Ok(self)
        }

        pub fn exclude_behavior_notes(
            mut self,
            excluded_notes: impl Into<Vec<crate::behavior::BackendBehaviorNote>>,
        ) -> crate::error::Result<Self> {
            let excluded_notes = excluded_notes.into();
            let filter = crate::frontend::plan::metadata::EngineConfigFilter::new()
                .with_excluded_behavior_notes(&excluded_notes);
            self.apply_filter(filter);
            Ok(self)
        }
    };
}

mod bindings;
mod build;
mod built;
mod cache;
mod compiled;
mod config;
mod metadata;

use self::build::{
    build_execution_plan_choice, build_execution_plan_choice_parallel_window,
    build_execution_plan_entries, lock_kernel_cache, materialize_graph_kernel_cache,
};
pub use self::{
    cache::{CachedPlanChoice, PlanCache},
    compiled::CompiledGraph,
    config::{
        AliasTensorBinding, AutotuneConfig, AutotuneResult, BuildPlanPolicy, PlanSupport,
        PlanTiming, RequiredTensor,
    },
    metadata::EngineConfigFilter,
};
pub(crate) use self::{config::BindingReplacement, metadata::EngineConfigMetadata};

use crate::{
    behavior::{BackendBehaviorNote, BackendNumericalNote},
    context::Context,
    engine::EngineIndex,
    error::{Error, Result},
    execution::{
        EngineConfig, EngineKnobInfo, ExecutionPlan, IntermediateInfo, KnobChoiceInfo,
        OperationGraph,
    },
    frontend::{
        graph::{Graph, TensorRecord},
        operation::CompileConfig,
        plan::built::BuiltPlanCandidates,
    },
    scalar::ScalarValue,
    tensor::TensorId,
};

/// Heuristic engine configurations for a lowered frontend graph.
///
/// cuDNN heuristics return candidate engine configurations sorted by expected performance.
/// The recommended workflow is to query heuristics, check support, and build the first supported configuration or autotune multiple supported candidates.
#[derive(Debug)]
pub struct PlanCandidates {
    pub(crate) graph: Graph,
    pub(crate) graph_name: Option<String>,
    pub(crate) tensors: BTreeMap<TensorId, TensorRecord>,
    pub(crate) binding_template_ids: Vec<TensorId>,
    pub(crate) scalar_bindings: Vec<(TensorId, ScalarValue)>,
    pub(crate) binding_replacements: Vec<BindingReplacement>,
    pub(crate) operation_graph: OperationGraph,
    pub(crate) engine_configs: Vec<EngineConfig>,
    pub(crate) engine_metadata: Vec<EngineConfigMetadata>,
}

impl PlanCandidates {
    pub(crate) fn format_name(
        graph_name: Option<&str>,
        engine_config: &EngineConfig,
    ) -> Result<String> {
        let engine_name = Self::name_for_engine_config(engine_config)?;
        Ok(match graph_name {
            Some(graph_name) => format!("{graph_name}/{engine_name}"),
            None => engine_name,
        })
    }

    pub(crate) fn name_for_engine_config(engine_config: &EngineConfig) -> Result<String> {
        let engine_index = engine_config.engine_index()?;
        let mut name = format!("eng{engine_index}");
        if let Ok(knob_choices) = engine_config.knob_choices() {
            for choice in knob_choices {
                name.push_str(&format!("_k{}={}", choice.knob_type as i32, choice.value));
            }
        }
        Ok(name)
    }

    pub fn operation_graph(&self) -> &OperationGraph {
        &self.operation_graph
    }

    pub fn graph_name(&self) -> Option<&str> {
        self.graph_name.as_deref()
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn engine_configs(&self) -> &[EngineConfig] {
        &self.engine_configs
    }

    pub fn engine_config_at(&self, plan_index: usize) -> Result<&EngineConfig> {
        self.engine_configs
            .get(plan_index)
            .ok_or(Error::OutOfRange {
                name: "plan_index".into(),
            })
    }

    fn engine_metadata_at(&self, plan_index: usize) -> Result<&EngineConfigMetadata> {
        self.engine_metadata
            .get(plan_index)
            .ok_or(Error::OutOfRange {
                name: "plan_index".into(),
            })
    }

    pub fn candidate_count(&self) -> usize {
        self.engine_configs.len()
    }

    pub fn engine_index_at(&self, plan_index: usize) -> Result<EngineIndex> {
        Ok(EngineIndex::new(
            self.engine_metadata_at(plan_index)?.engine_index,
        ))
    }

    pub fn engine_numerical_notes_at(
        &self,
        plan_index: usize,
    ) -> Result<Vec<BackendNumericalNote>> {
        Ok(self.engine_metadata_at(plan_index)?.numerical_notes.clone())
    }

    pub fn engine_behavior_notes_at(&self, plan_index: usize) -> Result<Vec<BackendBehaviorNote>> {
        Ok(self.engine_metadata_at(plan_index)?.behavior_notes.clone())
    }

    pub fn engine_workspace_size_at(&self, plan_index: usize) -> Result<usize> {
        Ok(self.engine_metadata_at(plan_index)?.workspace_size)
    }

    pub fn engine_knob_infos_at(&self, plan_index: usize) -> Result<Vec<EngineKnobInfo>> {
        self.engine_config_at(plan_index)?.knob_infos()
    }

    pub fn engine_knob_choices_at(&self, plan_index: usize) -> Result<Vec<KnobChoiceInfo>> {
        Ok(self.engine_metadata_at(plan_index)?.knob_choices.clone())
    }

    pub fn engine_intermediate_infos_at(&self, plan_index: usize) -> Result<Vec<IntermediateInfo>> {
        self.engine_config_at(plan_index)?.intermediate_infos()
    }

    fn retain_engine_configs(&mut self, mut keep: impl FnMut(&EngineConfigMetadata) -> bool) {
        let mut kept_configs = Vec::with_capacity(self.engine_configs.len());
        let mut kept_metadata = Vec::with_capacity(self.engine_metadata.len());
        for (engine_config, metadata) in self
            .engine_configs
            .drain(..)
            .zip(self.engine_metadata.drain(..))
        {
            if keep(&metadata) {
                kept_configs.push(engine_config);
                kept_metadata.push(metadata);
            }
        }
        self.engine_configs = kept_configs;
        self.engine_metadata = kept_metadata;
    }

    pub(crate) fn apply_filter(&mut self, filter: EngineConfigFilter<'_>) {
        self.retain_engine_configs(|metadata| filter.accepts(metadata));
    }

    impl_engine_config_filter_methods!();

    pub fn name_at(&self, plan_index: usize) -> Result<String> {
        let engine_name = &self.engine_metadata_at(plan_index)?.name;
        Ok(match self.graph_name.as_deref() {
            Some(graph_name) => format!("{graph_name}/{engine_name}"),
            None => engine_name.clone(),
        })
    }

    pub fn names(&self) -> Result<Vec<String>> {
        (0..self.candidate_count())
            .map(|plan_index| self.name_at(plan_index))
            .collect()
    }

    pub fn support(&self, ctx: &Context) -> Vec<PlanSupport> {
        (0..self.candidate_count())
            .map(|plan_index| match self.support_at(ctx, plan_index) {
                Ok(()) => PlanSupport {
                    plan_index,
                    supported: true,
                    error: None,
                },
                Err(error) => PlanSupport {
                    plan_index,
                    supported: false,
                    error: Some(error),
                },
            })
            .collect()
    }

    pub fn supported_indices(&self, ctx: &Context) -> Vec<usize> {
        self.support(ctx)
            .into_iter()
            .filter(|support| support.supported)
            .map(|support| support.plan_index)
            .collect()
    }

    pub fn support_at(&self, ctx: &Context, plan_index: usize) -> Result<()> {
        if self.graph.kernel_cache_enabled() && !self.graph.dynamic_shape_enabled() {
            return Err(Error::FrontendKernelCacheRequiresDynamicShape);
        }

        let engine_config = self.engine_config_at(plan_index)?;
        let kernel_cache = materialize_graph_kernel_cache(&self.graph, &self.operation_graph)?;
        let device_properties = self.graph.resolved_device_properties()?;
        let _ = ExecutionPlan::create_with_config(
            ctx,
            engine_config,
            lock_kernel_cache(kernel_cache.as_ref())?.as_deref(),
            device_properties.as_deref(),
        )?;
        Ok(())
    }

    pub fn compile(self, ctx: &Context) -> Result<CompiledGraph> {
        self.compile_with_config(ctx, &CompileConfig::new())
    }

    pub fn build_candidates(self, ctx: &Context) -> Result<BuiltPlanCandidates> {
        self.build_candidates_with_config(ctx, &CompileConfig::new())
    }

    pub fn build_candidates_deviceless(self) -> Result<BuiltPlanCandidates> {
        self.build_candidates_deviceless_with_config(&CompileConfig::new())
    }

    pub fn build_candidates_with_config(
        self,
        ctx: &Context,
        config: &CompileConfig,
    ) -> Result<BuiltPlanCandidates> {
        self.build_candidates_impl(Some(ctx), config)
    }

    pub fn build_candidates_deviceless_with_config(
        self,
        config: &CompileConfig,
    ) -> Result<BuiltPlanCandidates> {
        self.build_candidates_impl(None, config)
    }

    fn build_candidates_impl(
        self,
        ctx: Option<&Context>,
        config: &CompileConfig,
    ) -> Result<BuiltPlanCandidates> {
        if self.graph.kernel_cache_enabled() && !self.graph.dynamic_shape_enabled() {
            return Err(Error::FrontendKernelCacheRequiresDynamicShape);
        }
        let device_properties = self.graph.resolved_device_properties()?;

        let candidates = self
            .engine_configs
            .into_iter()
            .zip(self.engine_metadata)
            .collect::<Vec<_>>();

        let entries = match config.build_policy() {
            BuildPlanPolicy::FirstSupported => build_execution_plan_choice(
                ctx,
                &self.operation_graph,
                self.graph.kernel_cache_enabled(),
                self.graph.kernel_cache_json(),
                self.graph.runtime_kernel_cache(),
                device_properties.as_deref(),
                candidates,
                config,
            )?,
            BuildPlanPolicy::FirstSupportedParallel { window_size } => {
                build_execution_plan_choice_parallel_window(
                    ctx,
                    &self.operation_graph,
                    self.graph.kernel_cache_enabled(),
                    self.graph.kernel_cache_json(),
                    self.graph.runtime_kernel_cache(),
                    self.graph.device_properties_json(),
                    device_properties.as_deref(),
                    candidates,
                    config,
                    window_size,
                )?
            }
            BuildPlanPolicy::AllSupported => build_execution_plan_entries(
                ctx,
                &self.operation_graph,
                self.graph.kernel_cache_enabled(),
                self.graph.kernel_cache_json(),
                self.graph.runtime_kernel_cache(),
                device_properties.as_deref(),
                candidates,
                config,
            )?,
        };

        Ok(BuiltPlanCandidates {
            graph: self.graph,
            graph_name: self.graph_name,
            tensors: self.tensors,
            binding_template_ids: self.binding_template_ids,
            scalar_bindings: self.scalar_bindings,
            binding_replacements: self.binding_replacements,
            operation_graph: self.operation_graph,
            entries,
        })
    }

    pub fn compile_with_config(
        self,
        ctx: &Context,
        config: &CompileConfig,
    ) -> Result<CompiledGraph> {
        self.build_candidates_with_config(ctx, config)?.compile()
    }

    pub fn compile_deviceless_with_config(self, config: &CompileConfig) -> Result<CompiledGraph> {
        self.build_candidates_deviceless_with_config(config)?
            .compile()
    }

    pub fn compile_at(self, ctx: &Context, plan_index: usize) -> Result<CompiledGraph> {
        let engine_config = self
            .engine_configs
            .get(plan_index)
            .ok_or(Error::OutOfRange {
                name: "plan_index".into(),
            })?;

        let engine = engine_config.engine()?;
        let knob_choices = engine_config
            .knob_choices()?
            .into_iter()
            .map(|choice| (choice.knob_type, choice.value))
            .collect::<Vec<_>>();
        let device_properties = self.graph.resolved_device_properties()?;
        let engine_config = EngineConfig::create_with_config(
            ctx,
            &self.operation_graph,
            engine.index(),
            &knob_choices,
            device_properties.as_deref(),
        )?;
        let engine_metadata = EngineConfigMetadata::create(&self.operation_graph, &engine_config)?;

        PlanCandidates {
            graph: self.graph,
            graph_name: self.graph_name,
            tensors: self.tensors,
            binding_template_ids: self.binding_template_ids,
            scalar_bindings: self.scalar_bindings,
            binding_replacements: self.binding_replacements,
            operation_graph: self.operation_graph,
            engine_configs: vec![engine_config],
            engine_metadata: vec![engine_metadata],
        }
        .compile_with_config(
            ctx,
            &CompileConfig::new().with_build_policy(BuildPlanPolicy::FirstSupported),
        )
    }
}
