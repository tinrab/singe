macro_rules! impl_plan_metadata_methods {
    () => {
        /// Current candidate identities in collection order, initially the heuristic order.
        pub fn plan_ids(&self) -> Vec<PlanId> {
            self.candidates()
                .map(|candidate| candidate.id.clone())
                .collect()
        }

        /// Looks up an identity by collection position; positions are not plan identities.
        ///
        /// # Errors
        ///
        /// Returns an error if `position` is outside the current candidate collection.
        pub fn plan_id_at(&self, position: usize) -> Result<PlanId> {
            self.candidates()
                .nth(position)
                .map(|candidate| candidate.id.clone())
                .ok_or(Error::out_of_range("plan position"))
        }

        /// Returns the number of candidates in the current collection order.
        pub fn candidate_count(&self) -> usize {
            self.candidates().len()
        }

        /// Iterates over retained engine configurations in current collection order without querying cuDNN.
        pub fn engine_configs(&self) -> impl ExactSizeIterator<Item = &EngineConfig> {
            self.candidates()
                .map(|candidate| candidate.engine_config.as_ref())
        }

        /// Borrows the engine configuration for a stable plan identity.
        ///
        /// # Errors
        ///
        /// Returns an error if this collection does not contain `plan`.
        pub fn engine_config_for(&self, plan: &PlanId) -> Result<&EngineConfig> {
            Ok(self.candidate_for(plan)?.engine_config.as_ref())
        }

        /// Cached native metadata used by filters and errata matching.
        ///
        /// # Errors
        ///
        /// Returns an error if this collection does not contain `plan`.
        pub fn engine_metadata_for(&self, plan: &PlanId) -> Result<&EngineConfigMetadata> {
            Ok(&self.candidate_for(plan)?.metadata)
        }

        /// Copies the candidate's engine index and canonical knob choices for explicit planning.
        ///
        /// # Errors
        /// Returns an error when this collection does not contain the plan identity.
        pub fn engine_choice_for(&self, plan: &PlanId) -> Result<EngineChoice> {
            Ok(self.candidate_for(plan)?.choice())
        }

        /// Returns the cached global native engine index for a stable plan identity.
        ///
        /// # Errors
        ///
        /// Returns an error if this collection does not contain `plan`.
        pub fn engine_index_for(&self, plan: &PlanId) -> Result<EngineIndex> {
            Ok(self.candidate_for(plan)?.metadata.engine_index)
        }

        /// Borrows the candidate’s cached numerical notes without querying cuDNN.
        ///
        /// # Errors
        ///
        /// Returns an error if this collection does not contain `plan`.
        pub fn engine_numerical_notes_for(&self, plan: &PlanId) -> Result<&[BackendNumericalNote]> {
            Ok(&self.candidate_for(plan)?.metadata.numerical_notes)
        }

        /// Borrows the candidate’s cached behavior notes without querying cuDNN.
        ///
        /// # Errors
        ///
        /// Returns an error if this collection does not contain `plan`.
        pub fn engine_behavior_notes_for(&self, plan: &PlanId) -> Result<&[BackendBehaviorNote]> {
            Ok(&self.candidate_for(plan)?.metadata.behavior_notes)
        }

        /// Returns the candidate’s cached workspace requirement in bytes, including frontend workspace.
        ///
        /// # Errors
        ///
        /// Returns an error if this collection does not contain `plan`.
        pub fn engine_workspace_size_for(&self, plan: &PlanId) -> Result<usize> {
            Ok(self.candidate_for(plan)?.metadata.workspace_size)
        }

        /// Borrows supported knobs from the selected candidate's retained native engine.
        ///
        /// # Errors
        ///
        /// Returns an error if the plan identity is absent, or native engine extraction, context binding, or knob-metadata querying or decoding fails.
        pub fn engine_knob_infos_for(&self, plan: &PlanId) -> Result<&[EngineKnobInfo]> {
            self.engine_config_for(plan)?.knob_infos()
        }

        /// Borrows the candidate’s cached tuning-knob choices without querying cuDNN.
        ///
        /// # Errors
        ///
        /// Returns an error if this collection does not contain `plan`.
        pub fn engine_knob_choices_for(&self, plan: &PlanId) -> Result<&[KnobChoiceInfo]> {
            Ok(&self.candidate_for(plan)?.metadata.knob_choices)
        }

        /// Queries reusable intermediates and their storage and tensor dependencies from the selected engine configuration.
        ///
        /// # Errors
        ///
        /// Returns an error if the plan identity is absent, context binding or native attribute querying or decoding fails, or an intermediate size cannot be represented as `usize`.
        pub fn engine_intermediate_infos_for(
            &self,
            plan: &PlanId,
        ) -> Result<Vec<IntermediateInfo>> {
            self.engine_config_for(plan)?.intermediate_infos()
        }

        /// Returns the candidate’s native name, prefixed by the retained graph name and `/` when present.
        ///
        /// # Errors
        ///
        /// Returns an error if this collection does not contain `plan`.
        pub fn name_for(&self, plan: &PlanId) -> Result<String> {
            Ok(self.candidate_for(plan)?.name(self.graph_name()))
        }

        /// Returns candidate names in current collection order, including the graph-name prefix when present.
        pub fn names(&self) -> Vec<String> {
            self.candidates()
                .map(|candidate| candidate.name(self.graph_name()))
                .collect()
        }
    };
}

macro_rules! impl_engine_config_filter_methods {
    () => {
        /// Applies an in-memory engine filter without making native calls.
        #[must_use]
        pub fn filter(mut self, filter: &EngineConfigFilter) -> Self {
            self.apply_exclusions(|metadata| filter.exclusion(metadata));
            self
        }

        /// Filters cached engine metadata with a custom predicate.
        /// Built collections retain each excluded candidate's build outcome and stable identity.
        #[must_use]
        pub fn filter_by(mut self, mut accept: impl FnMut(&EngineConfigMetadata) -> bool) -> Self {
            self.apply_exclusions(|metadata| (!accept(metadata)).then_some(PlanExclusion::Filter));
            self
        }
    };
}

mod artifacts;
mod autotune;
mod bindings;
mod build;
mod built;
mod cache;
mod compiled;
mod config;
mod entry;
mod execute;
mod metadata;
mod ordering;
mod serialization;
mod tensors;
mod tuned;
mod warmup;
mod workspace;

use std::sync::Arc;

pub(crate) use crate::frontend::plan::{
    artifacts::PlanArtifacts,
    bindings::{BindingReplacement, required_tensor_bindings},
    metadata::PreparedEngineConfig,
};
pub use crate::{
    backend::execution::EngineChoice,
    frontend::plan::{
        autotune::{AutotuneConfig, AutotuneResult, AutotuneSampling, AutotuneSession, PlanTiming},
        bindings::{AliasTensorBinding, AliasTensorKind, RequiredTensor, RequiredTensorBinding},
        built::BuiltPlanCandidates,
        cache::PlanCache,
        compiled::CompiledGraph,
        config::{BuildPlanPolicy, CompileConfig, HeuristicMode, PlanSupport},
        entry::{
            PlanBuildReport, PlanBuildStatus, PlanExclusion, PlanFailure, PlanFailureStage, PlanId,
        },
        execute::ExecuteArgs,
        metadata::{EngineConfigFilter, EngineConfigMetadata},
        serialization::{
            NativePlanRestoreTarget, NativePlanSerializationConfig, NativePlanSet,
            NativePlanStructure,
        },
        tensors::{TensorInfo, TensorInfoIter},
        tuned::TunedPlan,
    },
};
use crate::{
    backend::{
        behavior::{BackendBehaviorNote, BackendNumericalNote},
        engine::EngineIndex,
        execution::{
            EngineConfig, EngineKnobInfo, IntermediateInfo, KnobChoiceInfo, OperationGraph,
        },
        tensor::TensorId,
    },
    context::Context,
    error::{Error, Result},
    frontend::{
        bindings::BindingTemplate,
        error::Error as FrontendError,
        graph::Graph,
        lower::LoweredGraph,
        plan::{
            build::materialize_graph_kernel_cache,
            entry::{BuiltPlanEntry, PlanCandidate},
        },
    },
};

/// Heuristic candidates whose identities remain stable through filtering and compilation.
#[derive(Debug)]
pub struct PlanCandidates {
    pub(crate) recipe: Graph,
    pub(crate) entries: Vec<PlanCandidate>,
    pub(crate) graph: Graph,
    pub(crate) artifacts: PlanArtifacts,
    pub(crate) binding_template_ids: Vec<TensorId>,
    pub(crate) operation_graph: OperationGraph,
    pub(crate) lowered: Arc<LoweredGraph>,
}

impl PlanCandidates {
    pub(crate) fn create_entries(configs: Vec<PreparedEngineConfig>) -> Vec<PlanCandidate> {
        PlanId::collection(configs.len())
            .zip(configs)
            .map(|(id, config)| PlanCandidate {
                id,
                engine_config: Arc::new(config.engine_config),
                metadata: config.metadata,
            })
            .collect()
    }

    /// Borrows the native operation graph shared by these candidates.
    #[must_use]
    pub fn operation_graph(&self) -> &OperationGraph {
        &self.operation_graph
    }

    /// Returns the retained optional graph name.
    #[must_use]
    pub fn graph_name(&self) -> Option<&str> {
        self.artifacts.graph_name.as_deref()
    }

    /// Returns the resolved graph whose operations were lowered for this target.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Returns the declared graph before target-specific attention expansion.
    #[must_use]
    pub fn recipe(&self) -> &Graph {
        &self.recipe
    }

    /// Retained caller binding requirements shared by every candidate in this collection.
    #[must_use]
    pub fn required_tensor_bindings(&self) -> &[RequiredTensor] {
        &self.artifacts.required_tensors
    }

    fn candidates(&self) -> impl ExactSizeIterator<Item = &PlanCandidate> {
        self.entries.iter()
    }

    fn candidate_for(&self, plan: &PlanId) -> Result<&PlanCandidate> {
        self.entries
            .iter()
            .find(|candidate| &candidate.id == plan)
            .ok_or_else(|| FrontendError::PlanNotFound { plan: plan.clone() }.into())
    }

    impl_plan_metadata_methods!();

    impl_engine_config_filter_methods!();

    /// Retains the requested candidates in the supplied order without changing their identities.
    /// First-supported and bounded builds follow this order. Omitted candidates are removed.
    /// Engine descriptors and cached metadata move together.
    ///
    /// # Errors
    /// Returns an error for an empty order, duplicate identities or identities absent from this collection.
    /// An invalid order leaves the collection unchanged.
    pub fn reorder_plans(&mut self, plans: &[PlanId]) -> Result<()> {
        ordering::reorder_plan_entries(&mut self.entries, plans, |entry| &entry.id)
    }

    fn apply_exclusions(
        &mut self,
        mut exclusion: impl FnMut(&EngineConfigMetadata) -> Option<PlanExclusion>,
    ) {
        self.entries
            .retain(|candidate| exclusion(&candidate.metadata).is_none());
    }

    /// Reports cached configuration eligibility in collection order without changing the collection.
    /// Matches C++ support queries at the cuDNN 9.27 baseline: no plan compilation,
    /// CUDA context binding or kernel-cache mutation occurs.
    /// Resource exclusion limits and note/name filters come from `filter`.
    /// Yielded references borrow this collection independently of the temporary filter.
    pub fn support<'candidates>(
        &'candidates self,
        filter: &EngineConfigFilter,
    ) -> impl ExactSizeIterator<Item = PlanSupport<'candidates>> {
        let filter = filter.clone();
        self.candidates().map(move |candidate| PlanSupport {
            plan: &candidate.id,
            metadata: &candidate.metadata,
            exclusion: filter.exclusion(&candidate.metadata),
        })
    }

    /// Reports eligibility for one stable candidate identity without compiling it.
    /// Returns an error only when this collection does not contain the identity.
    ///
    /// # Errors
    ///
    /// Returns an error if this collection does not contain `plan`.
    pub fn support_for(
        &self,
        plan: &PlanId,
        filter: &EngineConfigFilter,
    ) -> Result<PlanSupport<'_>> {
        let candidate = self.candidate_for(plan)?;
        Ok(PlanSupport {
            plan: &candidate.id,
            metadata: &candidate.metadata,
            exclusion: filter.exclusion(&candidate.metadata),
        })
    }

    /// Builds candidates and selects the first available plan in collection order.
    /// Pass `None` for deviceless compilation.
    ///
    /// # Errors
    ///
    /// Returns an error if candidate preparation, kernel-cache setup, binding-source resolution, or target-property resolution fails, or no compiled, eligible plan is available. An unavailable-plan error includes per-candidate build reports.
    pub fn compile(
        self,
        context: Option<&Context>,
        config: &CompileConfig,
    ) -> Result<CompiledGraph> {
        self.build_candidates(context, config)?.compile()
    }

    /// Builds candidates and returns their retained outcomes when at least one plan is available.
    /// Pass `None` to build without a live handle; failed batches return the same reports.
    /// A batch with no usable plan returns [`FrontendError::NoAvailablePlans`] with every candidate's report.
    /// Use [`Self::into_build_candidates`] to retain the collection for retries after an unsuccessful batch.
    ///
    /// # Errors
    ///
    /// Returns an error if kernel-cache or binding-template preparation or target-property resolution fails, or no candidate produces an available plan under the build policy. The latter error includes every candidate’s build report.
    pub fn build_candidates(
        self,
        context: Option<&Context>,
        config: &CompileConfig,
    ) -> Result<BuiltPlanCandidates> {
        let mut built = self.into_build_candidates(config)?;
        built.build_plans(context, config.build_policy())?;
        Ok(built)
    }

    /// Retains every candidate for individually requested builds without building a native plan.
    /// Filters and resource limits apply to every later build in this collection.
    /// Kernel-cache preparation may call cuDNN; plan compilation starts with
    /// [`BuiltPlanCandidates::build_plan`].
    ///
    /// # Errors
    ///
    /// Returns an error if a kernel cache is enabled without dynamic shapes, cache locking, creation, restoration, or finalization fails, or resolving binding aliases fails.
    pub fn into_build_candidates(self, config: &CompileConfig) -> Result<BuiltPlanCandidates> {
        if self.graph.kernel_cache_enabled() && !self.graph.dynamic_shape_enabled() {
            return Err(FrontendError::KernelCacheRequiresDynamicShape.into());
        }
        let kernel_cache = materialize_graph_kernel_cache(&self.graph, &self.operation_graph)?;
        let binding_template = BindingTemplate::create(
            self.binding_template_ids,
            &self.artifacts.binding_replacements,
        )?;
        let entries = self
            .entries
            .into_iter()
            .map(|candidate| BuiltPlanEntry::unbuilt(candidate, config))
            .collect();
        Ok(BuiltPlanCandidates {
            recipe: self.recipe,
            entries,
            graph: self.graph,
            artifacts: self.artifacts,
            binding_template,
            operation_graph: self.operation_graph,
            lowered: self.lowered,
            build_config: config.clone(),
            kernel_cache,
        })
    }

    /// Builds only the requested candidate, preserving its identity.
    ///
    /// # Errors
    ///
    /// Returns an error if `plan` is absent, candidate preparation, kernel-cache setup, binding-source resolution, or target-property resolution fails, or the requested plan is excluded, unbuilt, or failed during native construction.
    pub fn compile_one(self, context: &Context, plan: &PlanId) -> Result<CompiledGraph> {
        self.candidate_for(plan)?;
        let mut built = self.into_build_candidates(&CompileConfig::new())?;
        built.build_plan(Some(context), plan)?;
        built.compile_selected(plan)
    }
}
