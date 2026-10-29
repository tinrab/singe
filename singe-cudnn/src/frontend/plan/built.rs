use std::sync::{Arc, Mutex};

use crate::{
    backend::{
        behavior::{BackendBehaviorNote, BackendNumericalNote},
        engine::EngineIndex,
        execution::{
            EngineConfig, EngineKnobInfo, ExecutionPlan, IntermediateInfo, KernelCache,
            KnobChoiceInfo, OperationGraph,
        },
        tensor::TensorId,
    },
    context::Context,
    error::{Error, Result},
    frontend::{
        bindings::{
            BindingTemplate, Bindings, RuntimeOverrideBuilder, RuntimeOverrides,
            prepare_variant_pack,
        },
        error::Error as FrontendError,
        graph::Graph,
        logging,
        lower::LoweredGraph,
        plan::{
            AliasTensorBinding, BuildPlanPolicy, CompileConfig, CompiledGraph, EngineChoice,
            EngineConfigFilter, EngineConfigMetadata, ExecuteArgs, PlanArtifacts, PlanBuildReport,
            PlanExclusion, PlanId, PlanSupport, RequiredTensor,
            bindings::alias_tensor_bindings,
            build::{build_execution_plan_entries, build_execution_plan_entry},
            compiled::{CompiledGraphCore, CompiledGraphStructure},
            config::RuntimeShapePolicy,
            entry::{BuiltPlanEntry, CompiledPlanEntry, PlanBuildOutcome, PlanCandidate},
            ordering::reorder_plan_entries,
        },
    },
};

/// Build outcomes for every candidate, including failed, excluded and unattempted entries.
#[derive(Debug)]
pub struct BuiltPlanCandidates {
    pub(crate) recipe: Graph,
    pub(crate) entries: Vec<BuiltPlanEntry>,
    pub(crate) graph: Graph,
    pub(crate) artifacts: PlanArtifacts,
    pub(crate) binding_template: BindingTemplate,
    pub(crate) operation_graph: OperationGraph,
    pub(crate) lowered: Arc<LoweredGraph>,
    pub(crate) build_config: CompileConfig,
    pub(crate) kernel_cache: Option<Arc<Mutex<KernelCache>>>,
}

impl BuiltPlanCandidates {
    /// Returns the optional graph name retained by these candidates.
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

    fn candidates(&self) -> impl ExactSizeIterator<Item = &PlanCandidate> {
        self.entries.iter().map(|entry| &entry.candidate)
    }

    fn candidate_for(&self, plan: &PlanId) -> Result<&PlanCandidate> {
        Ok(&self.entry_for(plan)?.candidate)
    }

    fn entry_for(&self, plan: &PlanId) -> Result<&BuiltPlanEntry> {
        self.entries
            .iter()
            .find(|entry| &entry.candidate.id == plan)
            .ok_or_else(|| FrontendError::PlanNotFound { plan: plan.clone() }.into())
    }

    impl_plan_metadata_methods!();

    impl_engine_config_filter_methods!();

    /// Retains candidates in the requested order, including their build outcomes and exclusions.
    /// Omitted entries are removed. Compilation selects the first available entry in this order.
    ///
    /// # Errors
    /// Returns an error for an empty order, duplicate identities or identities absent from this collection.
    /// An invalid order leaves the collection unchanged.
    pub fn reorder_plans(&mut self, plans: &[PlanId]) -> Result<()> {
        reorder_plan_entries(&mut self.entries, plans, |entry| &entry.candidate.id)
    }

    /// Returns the retained build outcome and exclusion for a stable candidate identity.
    ///
    /// # Errors
    ///
    /// Returns an error if this collection does not contain `plan`.
    pub fn build_report(&self, plan: &PlanId) -> Result<PlanBuildReport> {
        Ok(self.entry_for(plan)?.report(self.graph_name()))
    }

    /// Returns every retained candidate’s build outcome and independent exclusion in collection order.
    #[must_use]
    pub fn build_reports(&self) -> Vec<PlanBuildReport> {
        self.entries
            .iter()
            .map(|entry| entry.report(self.graph_name()))
            .collect()
    }

    /// Reports eligibility in retained candidate order using the latest cached native requirements.
    /// Preserves existing exclusions, then applies the supplied filter without mutating the collection.
    /// Does not compile plans, bind a CUDA context or change the kernel cache.
    /// Build outcomes remain independent and are available through [`Self::build_reports`].
    pub fn support<'candidates>(
        &'candidates self,
        filter: &EngineConfigFilter,
    ) -> impl ExactSizeIterator<Item = PlanSupport<'candidates>> {
        let filter = filter.clone();
        self.entries.iter().map(move |entry| entry.support(&filter))
    }

    /// Reports eligibility for one retained identity without compiling it or clearing exclusions.
    ///
    /// # Errors
    /// Returns an error when this collection does not contain the plan identity.
    pub fn support_for(
        &self,
        plan: &PlanId,
        filter: &EngineConfigFilter,
    ) -> Result<PlanSupport<'_>> {
        Ok(self.entry_for(plan)?.support(filter))
    }

    /// Builds candidates in their current order using the requested batch policy.
    /// Reuses built plans, retries failed candidates and retains every candidate's identity and outcome.
    /// Only build-limit exclusions are reconsidered; filters and resource exclusions remain in effect.
    /// Planning settings, resource budgets and the all-supported plan-count limit remain those of this collection.
    /// A first-supported policy admits one plan; a later all-supported call can admit its retained alternatives.
    /// Pass `Some(context)` for a live build or `None` for deviceless compilation.
    ///
    /// # Errors
    /// Returns an error if target device properties cannot be resolved or no unexcluded plan is built.
    /// Individual candidate failures do not abort the batch while other candidates remain.
    /// A batch with no usable plan reports [`FrontendError::NoAvailablePlans`] with all build reports.
    /// The collection retains its outcomes for inspection and later retries even when the batch returns an error.
    pub fn build_plans(
        &mut self,
        context: Option<&Context>,
        policy: BuildPlanPolicy,
    ) -> Result<()> {
        let config = self.build_config.clone().with_build_policy(policy);
        let device_properties = self.graph.resolved_device_properties()?;
        build_execution_plan_entries(
            context,
            &self.operation_graph,
            self.lowered.layout.workspace_layout.byte_len(),
            device_properties.as_deref(),
            &mut self.entries,
            self.kernel_cache.as_ref(),
            &config,
        );
        if self.available_plans().next().is_none() {
            return Err(FrontendError::NoAvailablePlans {
                reports: self.build_reports(),
            }
            .into());
        }
        Ok(())
    }

    /// Builds the requested candidate without discarding alternatives or changing their outcomes.
    /// Pass `Some(context)` for a live build or `None` for deviceless compilation.
    /// Uses this collection's filters, resource limits and shared kernel cache.
    /// Batch build policies and plan-count limits do not prevent explicit requests.
    ///
    /// Already built candidates are reused. Failed candidates are retried, replacing only
    /// their own diagnostics. Filtered candidates remain excluded. A native build failure
    /// is recorded in the returned report; setup and logging failures return `Err`.
    ///
    /// # Errors
    ///
    /// Returns an error if the plan identity is absent or target device properties cannot be resolved. Candidate construction failures are retained in the returned build report.
    pub fn build_plan(
        &mut self,
        context: Option<&Context>,
        plan: &PlanId,
    ) -> Result<PlanBuildReport> {
        let position = self
            .entries
            .iter()
            .position(|entry| &entry.candidate.id == plan)
            .ok_or_else(|| FrontendError::PlanNotFound { plan: plan.clone() })?;
        let entry = &mut self.entries[position];
        if matches!(entry.exclusion, Some(PlanExclusion::BuildLimit { .. })) {
            entry.exclusion = None;
        }
        if entry.exclusion.is_some() || matches!(entry.outcome, PlanBuildOutcome::Built(_)) {
            return Ok(entry.report(self.artifacts.graph_name.as_deref()));
        }
        let device_properties = self.graph.resolved_device_properties()?;
        build_execution_plan_entry(
            context,
            &self.operation_graph,
            entry,
            self.lowered.layout.workspace_layout.byte_len(),
            device_properties.as_deref(),
            self.kernel_cache.as_ref(),
            &self.build_config,
        );
        Ok(entry.report(self.artifacts.graph_name.as_deref()))
    }

    /// Borrows successfully built, unexcluded plans and their stable identities in candidate order.
    /// Does not allocate, query cuDNN or discard unavailable candidates' reports.
    pub fn available_plans(&self) -> impl Iterator<Item = (&PlanId, &ExecutionPlan)> {
        self.entries.iter().filter_map(|entry| {
            entry
                .available_plan()
                .map(|plan| (&entry.candidate.id, plan))
        })
    }

    /// Returns stable identities of built, unexcluded candidates in collection order.
    #[must_use]
    pub fn available_plan_ids(&self) -> Vec<PlanId> {
        self.available_plans().map(|(id, _)| id.clone()).collect()
    }

    /// Borrows the native execution plan for an available candidate.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity is absent or its candidate is excluded, unbuilt, or failed.
    pub fn execution_plan_for(&self, plan: &PlanId) -> Result<&ExecutionPlan> {
        self.entry_for(plan)?.execution_plan(self.graph_name())
    }

    /// Returns the available candidate’s cached static workspace size in bytes, including frontend scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity is absent or its candidate is excluded, unbuilt, or failed.
    pub fn workspace_size_for(&self, plan: &PlanId) -> Result<usize> {
        let entry = self.entry_for(plan)?;
        entry.execution_plan(self.graph_name())?;
        Ok(entry.candidate.metadata.workspace_size)
    }

    /// Maximum cached static requirement among built, non-excluded plans.
    /// Returns zero when no executable plan is available.
    #[must_use]
    pub fn max_workspace_size(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.is_available())
            .map(|entry| entry.candidate.metadata.workspace_size)
            .max()
            .unwrap_or(0)
    }

    /// Executes a built candidate before final selection.
    /// Built collections have no default plan, so `args.plan` is required.
    ///
    /// # Safety
    /// Uphold the allocation, initialization, access and lifetime requirements on [`Bindings`].
    ///
    /// # Errors
    ///
    /// Returns an error if `args.plan` is absent or unavailable, runtime overrides are disabled or invalid, binding or workspace requirements are not met, or CUDA context binding, diagnostic reads, or native preparation or execution fails.
    pub unsafe fn execute(&self, context: &Context, args: ExecuteArgs<'_, '_>) -> Result<()> {
        let plan = args.plan.ok_or(FrontendError::OperationFieldMissing {
            operation: "built candidate execution",
            field: "plan",
        })?;
        let no_overrides = RuntimeOverrides::new();
        let overrides = args.overrides.unwrap_or(&no_overrides);
        overrides.validate_enabled(
            self.graph.dynamic_shape_enabled() || self.graph.override_shape_enabled(),
        )?;
        let execution_plan = self.execution_plan_for(plan)?;
        let pack = prepare_variant_pack(
            context.cuda_context(),
            &self.artifacts.binding_replacements,
            args.bindings,
            args.workspace,
            overrides,
            &self.binding_template,
            &self.artifacts.tensors,
            &self.graph.runtime_shape_constraints,
            &self.lowered.layout.workspace_layout,
        )?;
        logging::diagnostic!("executing candidate {plan}");
        // SAFETY: diagnostics read the same initialized metadata and binding ranges as native execution.
        unsafe {
            logging::log_variant_pack(context, &pack, &self.artifacts.tensors)?;
        }
        // SAFETY: the caller guarantees valid bindings, ordered accesses and retained storage through completion or replay.
        unsafe { execution_plan.execute(context, &pack) }
    }

    /// Builds validated runtime overrides using this collection's tensors and dynamic bounds.
    #[must_use]
    pub fn runtime_overrides(&self) -> RuntimeOverrideBuilder<'_> {
        RuntimeOverrideBuilder::new(
            &self.artifacts.tensors,
            &self.artifacts.binding_replacements,
            &self.graph.runtime_shape_constraints,
        )
    }

    /// Creates graph-aware bindings populated with retained runtime-scalar defaults and scalar bindings.
    ///
    /// # Errors
    ///
    /// Returns an error if a retained scalar cannot be bound to its declared tensor, including identity, encoding, shape, or placement mismatches.
    pub fn bindings(&self) -> Result<Bindings<'_>> {
        let mut bindings = Bindings::with_graph_template(
            &self.artifacts.tensors,
            &self.artifacts.binding_replacements,
        );
        for record in self.artifacts.tensors.values() {
            if let Some(value) = record.tensor.runtime_scalar_value() {
                bindings.set_scalar(record.id, value)?;
            }
        }
        for &(id, value) in &self.artifacts.scalar_bindings {
            bindings.set_scalar(id, value)?;
        }
        Ok(bindings)
    }

    /// Caller storage and actual native uses shared by candidate execution and compiled plans.
    #[must_use]
    pub fn required_tensor_bindings(&self) -> &[RequiredTensor] {
        &self.artifacts.required_tensors
    }

    /// Returns caller-storage tensor identities from the retained binding requirements.
    #[must_use]
    pub fn required_tensor_ids(&self) -> Vec<TensorId> {
        self.required_tensor_bindings()
            .iter()
            .map(|binding| binding.id)
            .collect()
    }

    /// Returns retained alias and storage-view edges, including byte offsets and storage-only sources.
    #[must_use]
    pub fn alias_tensor_bindings(&self) -> Vec<AliasTensorBinding> {
        alias_tensor_bindings(
            &self.artifacts.tensors,
            &self.artifacts.binding_replacements,
        )
    }

    fn apply_exclusions(
        &mut self,
        mut exclusion: impl FnMut(&EngineConfigMetadata) -> Option<PlanExclusion>,
    ) {
        for entry in &mut self.entries {
            if let Some(reason) = exclusion(&entry.candidate.metadata) {
                entry.exclusion = Some(reason);
            }
        }
    }

    /// Consumes this collection into executable plans with the requested available candidate selected as default.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity is absent or its candidate is excluded, unbuilt, or failed.
    pub fn compile_selected(self, plan: &PlanId) -> Result<CompiledGraph> {
        self.execution_plan_for(plan)?;
        self.compile()?.with_default_plan(plan)
    }

    /// Consumes this collection, retaining built, unexcluded plans and selecting the first as default.
    ///
    /// # Errors
    ///
    /// Returns an error if no built, unexcluded candidate remains; the error includes all build reports.
    pub fn compile(self) -> Result<CompiledGraph> {
        let reports = self.build_reports();
        let entries = self
            .entries
            .into_iter()
            .filter_map(|entry| {
                if entry.exclusion.is_some() {
                    return None;
                }
                match entry.outcome {
                    PlanBuildOutcome::Built(execution_plan) => Some(CompiledPlanEntry {
                        execution_plan,
                        candidate: entry.candidate,
                    }),
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return Err(FrontendError::NoAvailablePlans { reports }.into());
        }
        Ok(CompiledGraph {
            plan_order: (0..entries.len()).collect(),
            core: Arc::new(CompiledGraphCore {
                shape_policy: RuntimeShapePolicy::from(&self.graph),
                structure: Some(CompiledGraphStructure {
                    recipe: self.recipe,
                    graph: self.graph,
                }),
                entries,
                reports,
                operation_graph: Some(self.operation_graph),
                artifacts: self.artifacts,
                binding_template: self.binding_template,
                workspace_layout: self.lowered.layout.workspace_layout.clone(),
                _lowered: Some(self.lowered),
            }),
            default_plan_index: 0,
        })
    }
}
