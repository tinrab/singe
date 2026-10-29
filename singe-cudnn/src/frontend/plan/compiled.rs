use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        behavior::{BackendBehaviorNote, BackendNumericalNote},
        engine::EngineIndex,
        execution::{
            EngineConfig, EngineKnobInfo, ExecutionPlan, IntermediateId, IntermediateInfo,
            KernelCacheJson, KnobChoiceInfo, OperationGraph, VariantPack,
        },
        tensor::TensorId,
    },
    context::Context,
    error::{Error, Result},
    frontend::{
        bindings::BindingTemplate,
        error::Error as FrontendError,
        graph::Graph,
        logging,
        lower::LoweredGraph,
        plan::{
            AliasTensorBinding, BindingReplacement, EngineChoice, EngineConfigMetadata,
            PlanArtifacts, PlanBuildReport, PlanId, RequiredTensor,
            bindings::alias_tensor_bindings,
            config::RuntimeShapePolicy,
            entry::{CompiledPlanEntry, PlanCandidate},
            ordering::plan_positions,
        },
        workspace::WorkspaceLayout,
    },
};

/// An ordering and selection handle over immutable, shared executable plans.
/// Cloning shares native plans while keeping ordering and default selection independent.
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    pub(crate) core: Arc<CompiledGraphCore>,
    pub(crate) plan_order: Vec<usize>,
    /// Position in this handle's plan order, independent of shared resource storage.
    pub(crate) default_plan_index: usize,
}

#[derive(Debug)]
pub struct CompiledGraphCore {
    pub structure: Option<CompiledGraphStructure>,
    pub shape_policy: RuntimeShapePolicy,
    pub entries: Vec<CompiledPlanEntry>,
    pub reports: Vec<PlanBuildReport>,
    pub operation_graph: Option<OperationGraph>,
    pub artifacts: PlanArtifacts,
    pub binding_template: BindingTemplate,
    pub workspace_layout: WorkspaceLayout,
    /// Keeps lowered native descriptors alive for plans compiled in this process.
    pub _lowered: Option<Arc<LoweredGraph>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledGraphStructure {
    pub recipe: Graph,
    pub graph: Graph,
}

impl CompiledGraph {
    pub(crate) unsafe fn log_execution(
        &self,
        context: &Context,
        pack: &VariantPack,
        plan: &PlanId,
    ) -> Result<()> {
        logging::diagnostic!("executing graph {:?}, plan {plan}", self.graph_name());
        // SAFETY: callers uphold the same pointer validity and initialized-input contracts as native execution.
        unsafe { logging::log_variant_pack(context, pack, &self.core.artifacts.tensors) }
    }

    fn candidates(&self) -> impl ExactSizeIterator<Item = &PlanCandidate> {
        self.ordered_entries().map(|entry| &entry.candidate)
    }

    pub(crate) fn ordered_entries(&self) -> impl ExactSizeIterator<Item = &CompiledPlanEntry> {
        self.plan_order
            .iter()
            .map(|&position| &self.core.entries[position])
    }

    fn candidate_for(&self, plan: &PlanId) -> Result<&PlanCandidate> {
        Ok(&self.entry_for(plan)?.candidate)
    }

    fn entry_for(&self, plan: &PlanId) -> Result<&CompiledPlanEntry> {
        self.ordered_entries()
            .find(|entry| &entry.candidate.id == plan)
            .ok_or_else(|| FrontendError::PlanNotFound { plan: plan.clone() }.into())
    }

    pub(crate) fn default_entry(&self) -> &CompiledPlanEntry {
        &self.core.entries[self.plan_order[self.default_plan_index]]
    }

    /// Returns the optional graph name retained by these plans.
    #[must_use]
    pub fn graph_name(&self) -> Option<&str> {
        self.core.artifacts.graph_name.as_deref()
    }

    /// Returns the resolved graph whose operations were lowered for this target.
    /// Plan-only restored artifacts have no structural graph.
    #[must_use]
    pub fn graph(&self) -> Option<&Graph> {
        self.core
            .structure
            .as_ref()
            .map(|structure| &structure.graph)
    }

    /// Returns the declared graph before target-specific attention expansion.
    /// Plan-only restored artifacts have no declared recipe.
    #[must_use]
    pub fn recipe(&self) -> Option<&Graph> {
        self.core
            .structure
            .as_ref()
            .map(|structure| &structure.recipe)
    }

    /// The native operation graph retained by a fresh compilation.
    /// Restored native artifacts execute directly and do not reconstruct this descriptor.
    #[must_use]
    pub fn operation_graph(&self) -> Option<&OperationGraph> {
        self.core.operation_graph.as_ref()
    }

    impl_plan_metadata_methods!();

    /// Retains executable plans in the requested order for this handle alone.
    /// Metadata, workspace queries, tuning and all-plan serialization use the same order.
    /// Keeps the current default when retained; otherwise selects the first requested plan.
    /// Omitted plans remain owned by shared resources but cannot be selected through this handle.
    /// Cloned handles, selected plans and queued executions retain their original resources and selections.
    ///
    /// # Errors
    /// Returns an error for an empty order, duplicate identities or identities absent from this handle.
    /// An invalid order leaves ordering and selection unchanged.
    pub fn reorder_plans(&mut self, plans: &[PlanId]) -> Result<()> {
        let positions = plan_positions(self.candidates().map(|candidate| &candidate.id), plans)?;
        let default_plan_index = plans
            .iter()
            .position(|plan| plan == self.default_plan())
            .unwrap_or(0);
        self.plan_order = positions
            .into_iter()
            .map(|position| self.plan_order[position])
            .collect();
        self.default_plan_index = default_plan_index;
        Ok(())
    }

    /// Exports the selected plan first, followed by the other plans in their retained order.
    /// Rebuilding the choices therefore prefers this handle's selection without reordering its plans.
    pub fn cached_plan_choices(&self) -> Vec<EngineChoice> {
        let mut choices = self
            .candidates()
            .map(PlanCandidate::choice)
            .collect::<Vec<_>>();
        choices[..=self.default_plan_index].rotate_right(1);
        choices
    }

    /// Original build diagnostics, including failed candidates and plans omitted by this handle.
    /// Reports retain their identities and compilation order independently of executable ordering.
    #[must_use]
    pub fn build_reports(&self) -> &[PlanBuildReport] {
        &self.core.reports
    }

    /// Iterates over native execution plans in this handle’s current order.
    pub fn execution_plans(&self) -> impl ExactSizeIterator<Item = &ExecutionPlan> {
        self.ordered_entries().map(|entry| &entry.execution_plan)
    }

    /// Borrows the default plan’s retained native engine configuration.
    #[must_use]
    pub fn engine_config(&self) -> &EngineConfig {
        self.default_entry().candidate.engine_config.as_ref()
    }

    /// Borrows the default native execution plan.
    #[must_use]
    pub fn execution_plan(&self) -> &ExecutionPlan {
        &self.default_entry().execution_plan
    }

    /// Borrows the stable identity selected as this handle’s default plan.
    #[must_use]
    pub fn default_plan(&self) -> &PlanId {
        &self.default_entry().candidate.id
    }

    /// Cached engine properties for the selected plan, including restored plan-only artifacts.
    /// Reading these properties makes no native queries.
    #[must_use]
    pub fn engine_metadata(&self) -> &EngineConfigMetadata {
        &self.default_entry().candidate.metadata
    }

    /// Selects a retained plan as this handle’s default without changing plan order.
    ///
    /// # Errors
    ///
    /// Returns an error if this handle does not retain `plan`.
    pub fn set_default_plan(&mut self, plan: &PlanId) -> Result<()> {
        let position = self
            .ordered_entries()
            .position(|entry| &entry.candidate.id == plan)
            .ok_or_else(|| FrontendError::PlanNotFound { plan: plan.clone() })?;
        self.default_plan_index = position;
        Ok(())
    }

    /// Selects a retained default plan and returns the updated handle.
    ///
    /// # Errors
    ///
    /// Returns an error if this handle does not retain `plan`.
    pub fn with_default_plan(mut self, plan: &PlanId) -> Result<Self> {
        self.set_default_plan(plan)?;
        Ok(self)
    }

    /// Borrows the native execution plan for a retained stable identity.
    ///
    /// # Errors
    ///
    /// Returns an error if this handle does not retain `plan`.
    pub fn execution_plan_for(&self, plan: &PlanId) -> Result<&ExecutionPlan> {
        Ok(&self.entry_for(plan)?.execution_plan)
    }

    /// Frontend scratch prefix, retained in the supplied workspace allocation.
    /// Its tensors are bound automatically and excluded from required caller bindings.
    #[must_use]
    pub fn frontend_workspace_layout(&self) -> &WorkspaceLayout {
        &self.core.workspace_layout
    }

    /// Cached total allocation size for the selected plan, including frontend scratch.
    /// Build and restoration resolve the finalized requirement before retaining a plan.
    /// Use runtime workspace queries when applying shape overrides.
    #[must_use]
    pub fn workspace_size(&self) -> usize {
        self.engine_metadata().workspace_size
    }

    /// Cached static workspace for one retained plan, including frontend scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if this handle does not retain `plan`.
    pub fn workspace_size_for(&self, plan: &PlanId) -> Result<usize> {
        Ok(self.candidate_for(plan)?.metadata.workspace_size)
    }

    /// Maximum cached static requirement in this handle's current plan order.
    #[must_use]
    pub fn max_workspace_size(&self) -> usize {
        self.candidates()
            .map(|candidate| candidate.metadata.workspace_size)
            .max()
            .unwrap_or(0)
    }

    /// Returns the default candidate’s name, prefixed by the graph name when present.
    #[must_use]
    pub fn name(&self) -> String {
        self.default_entry().candidate.name(self.graph_name())
    }

    /// Returns the default candidate’s cached global native engine index.
    #[must_use]
    pub fn engine_index(&self) -> EngineIndex {
        self.engine_metadata().engine_index
    }

    /// Copies the selected plan's retained engine index and canonical knob choices for explicit planning.
    /// Requires no native query and copies no alternative plans.
    #[must_use]
    pub fn engine_choice(&self) -> EngineChoice {
        self.default_entry().candidate.choice()
    }

    /// Borrows the default candidate’s cached numerical notes without querying cuDNN.
    #[must_use]
    pub fn engine_numerical_notes(&self) -> &[BackendNumericalNote] {
        &self.engine_metadata().numerical_notes
    }

    /// Borrows the default candidate’s cached behavior notes without querying cuDNN.
    #[must_use]
    pub fn engine_behavior_notes(&self) -> &[BackendBehaviorNote] {
        &self.engine_metadata().behavior_notes
    }

    /// Borrows supported knobs from the default plan's retained native engine.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding, native engine extraction, or knob-metadata querying or decoding fails.
    pub fn engine_knob_infos(&self) -> Result<&[EngineKnobInfo]> {
        self.engine_config().knob_infos()
    }

    /// Borrows the default candidate’s cached canonical tuning-knob choices.
    #[must_use]
    pub fn engine_knob_choices(&self) -> &[KnobChoiceInfo] {
        &self.engine_metadata().knob_choices
    }

    /// Queries reusable intermediates and their storage and tensor dependencies for the default engine configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding, native descriptor or attribute querying, or metadata decoding fails, or an intermediate size does not fit `usize`.
    pub fn engine_intermediate_infos(&self) -> Result<Vec<IntermediateInfo>> {
        self.engine_config().intermediate_infos()
    }

    /// Queries native computed-intermediate identities for the default plan.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding or the native attribute query fails, including invalid or persistently changing payload counts.
    pub fn computed_intermediate_ids(&self) -> Result<Vec<IntermediateId>> {
        self.execution_plan().computed_intermediate_ids()
    }

    /// Queries native computed-intermediate identities for a retained plan identity.
    ///
    /// # Errors
    ///
    /// Returns an error if this handle does not retain `plan`. Also returns an error if context binding or the native attribute query fails, including invalid or persistently changing payload counts.
    pub fn computed_intermediate_ids_for(&self, plan: &PlanId) -> Result<Vec<IntermediateId>> {
        self.execution_plan_for(plan)?.computed_intermediate_ids()
    }

    /// Queries native run-only intermediate identities for the default plan.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding or the native attribute query fails, including invalid or persistently changing payload counts.
    pub fn run_only_intermediate_ids(&self) -> Result<Vec<IntermediateId>> {
        self.execution_plan().run_only_intermediate_ids()
    }

    /// Queries native run-only intermediate identities for a retained plan identity.
    ///
    /// # Errors
    ///
    /// Returns an error if this handle does not retain `plan`. Also returns an error if context binding or the native attribute query fails, including invalid or persistently changing payload counts.
    pub fn run_only_intermediate_ids_for(&self, plan: &PlanId) -> Result<Vec<IntermediateId>> {
        self.execution_plan_for(plan)?.run_only_intermediate_ids()
    }

    /// Queries native serialized execution-plan JSON for the default plan.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding or the native attribute query fails, including invalid or persistently changing payload counts.
    pub fn native_plan_json(&self) -> Result<String> {
        self.execution_plan().json_representation()
    }

    /// Queries native serialized execution-plan JSON for a retained plan identity.
    ///
    /// # Errors
    ///
    /// Returns an error if this handle does not retain `plan`. Also returns an error if context binding or the native attribute query fails, including invalid or persistently changing payload counts.
    pub fn native_plan_json_for(&self, plan: &PlanId) -> Result<String> {
        self.execution_plan_for(plan)?.json_representation()
    }

    /// Queries optional native kernel-cache JSON for the default plan.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding or the native attribute query fails, including invalid or persistently changing payload counts. A missing cache is returned as `None`.
    pub fn kernel_cache_json(&self) -> Result<Option<KernelCacheJson>> {
        self.execution_plan().kernel_cache_json()
    }

    /// Queries optional native kernel-cache JSON for a retained plan identity.
    ///
    /// # Errors
    ///
    /// Returns an error if this handle does not retain `plan`. Also returns an error if context binding or the native attribute query fails, including invalid or persistently changing payload counts.
    pub fn kernel_cache_json_for(&self, plan: &PlanId) -> Result<Option<KernelCacheJson>> {
        self.execution_plan_for(plan)?.kernel_cache_json()
    }

    /// Returns caller-storage tensor identities from the retained binding requirements.
    #[must_use]
    pub fn required_tensor_ids(&self) -> Vec<TensorId> {
        self.required_tensor_bindings()
            .iter()
            .map(|binding| binding.id)
            .collect()
    }

    /// Caller storage, actual native uses, alias offsets, access roles and indexing dependencies.
    /// Excludes unused declarations, frontend workspace, virtual tensors and host scalars.
    /// Resolved once during preparation or artifact restoration; this performs no native queries.
    #[must_use]
    pub fn required_tensor_bindings(&self) -> &[RequiredTensor] {
        &self.core.artifacts.required_tensors
    }

    /// Returns retained storage-view and alias edges, including byte offsets and storage-only sources.
    #[must_use]
    pub fn alias_tensor_bindings(&self) -> Vec<AliasTensorBinding> {
        alias_tensor_bindings(
            &self.core.artifacts.tensors,
            &self.core.artifacts.binding_replacements,
        )
    }

    pub(crate) fn binding_replacements(&self) -> &[BindingReplacement] {
        &self.core.artifacts.binding_replacements
    }
}
