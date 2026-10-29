use std::collections::BTreeMap;

use crate::{
    backend::{
        execution::{ExecutionPlan, VariantPack},
        tensor::TensorId,
    },
    context::Context,
    error::{Error, Result},
    frontend::{
        bindings::RuntimeOverrides,
        graph::{RuntimeShapeConstraints, TensorRecord},
        plan::{BindingReplacement, BuiltPlanCandidates, CompiledGraph, PlanId},
        workspace::WorkspaceLayout,
    },
};

/// One owned runtime-layout pack shared by all workspace queries in a collection.
struct RuntimeWorkspaceQuery<'a> {
    context: &'a Context,
    pack: VariantPack,
    layout: WorkspaceLayout,
}

impl<'a> RuntimeWorkspaceQuery<'a> {
    fn create(
        context: &'a Context,
        overrides_enabled: bool,
        bounds: &BTreeMap<TensorId, RuntimeShapeConstraints>,
        tensors: &BTreeMap<TensorId, TensorRecord>,
        replacements: &[BindingReplacement],
        native_ids: &[TensorId],
        overrides: &RuntimeOverrides,
        static_layout: &WorkspaceLayout,
    ) -> Result<Self> {
        overrides.validate_enabled(overrides_enabled)?;
        let overrides = overrides.resolve_native_layouts(tensors, bounds, replacements)?;
        let layout = static_layout.with_runtime_overrides(tensors, native_ids, &overrides)?;
        // SAFETY: the query-only pack owns layout metadata and contains no device pointees.
        let pack = unsafe {
            VariantPack::create_with_overrides(context.cuda_context(), &[], None, &overrides)?
        };
        Ok(Self {
            context,
            pack,
            layout,
        })
    }

    fn for_plan(&self, plan: &ExecutionPlan) -> Result<usize> {
        self.layout
            .total_byte_len(plan.workspace_size_for(self.context, &self.pack)?)
    }

    fn maximum<'p>(&self, plans: impl IntoIterator<Item = &'p ExecutionPlan>) -> Result<usize> {
        let native = plans.into_iter().try_fold(0_usize, |bytes, plan| {
            Ok::<_, Error>(bytes.max(plan.workspace_size_for(self.context, &self.pack)?))
        })?;
        self.layout.total_byte_len(native)
    }
}

impl CompiledGraph {
    /// Required workspace for the default plan and effective runtime layouts, including frontend scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if overrides are disabled, tensor identities, shapes, strides, bounds, or alias layouts are invalid, frontend workspace sizing overflows, CUDA contexts disagree or cannot be bound, or native variant-pack creation or workspace querying fails.
    pub fn workspace_size_with_runtime_overrides(
        &self,
        context: &Context,
        overrides: &RuntimeOverrides,
    ) -> Result<usize> {
        self.workspace_size_for_with_runtime_overrides(context, self.default_plan(), overrides)
    }

    /// Required workspace for one retained plan and the supplied runtime layouts.
    ///
    /// # Errors
    ///
    /// Returns an error if overrides are disabled, tensor identities, shapes, strides, bounds, or alias layouts are invalid, frontend workspace sizing overflows, CUDA contexts disagree or cannot be bound, or native variant-pack creation or workspace querying fails. Returns an error if this handle does not retain `plan`.
    pub fn workspace_size_for_with_runtime_overrides(
        &self,
        context: &Context,
        plan: &PlanId,
        overrides: &RuntimeOverrides,
    ) -> Result<usize> {
        let plan = self.execution_plan_for(plan)?;
        self.runtime_workspace_query(context, overrides)?
            .for_plan(plan)
    }

    /// Maximum required workspace across all retained plans for these runtime layouts.
    /// Propagates any plan's requirement-query failure rather than underestimating capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if overrides are disabled, tensor identities, shapes, strides, bounds, or alias layouts are invalid, frontend workspace sizing overflows, CUDA contexts disagree or cannot be bound, or native variant-pack creation or workspace querying fails.
    pub fn max_workspace_size_with_runtime_overrides(
        &self,
        context: &Context,
        overrides: &RuntimeOverrides,
    ) -> Result<usize> {
        self.runtime_workspace_query(context, overrides)?
            .maximum(self.execution_plans())
    }

    fn runtime_workspace_query<'a>(
        &self,
        context: &'a Context,
        overrides: &RuntimeOverrides,
    ) -> Result<RuntimeWorkspaceQuery<'a>> {
        RuntimeWorkspaceQuery::create(
            context,
            self.core.shape_policy.overrides_enabled,
            &self.core.shape_policy.bounds,
            &self.core.artifacts.tensors,
            self.binding_replacements(),
            self.core.binding_template.ids(),
            overrides,
            &self.core.workspace_layout,
        )
    }
}

impl BuiltPlanCandidates {
    /// Required workspace for one available built candidate at the supplied runtime layouts.
    ///
    /// # Errors
    ///
    /// Returns an error if overrides are disabled, tensor identities, shapes, strides, bounds, or alias layouts are invalid, frontend workspace sizing overflows, CUDA contexts disagree or cannot be bound, or native variant-pack creation or workspace querying fails. Returns an error if the identity is absent or its candidate is excluded, unbuilt, or failed.
    pub fn workspace_size_for_with_runtime_overrides(
        &self,
        context: &Context,
        plan: &PlanId,
        overrides: &RuntimeOverrides,
    ) -> Result<usize> {
        let plan = self.execution_plan_for(plan)?;
        self.runtime_workspace_query(context, overrides)?
            .for_plan(plan)
    }

    /// Maximum runtime workspace across available candidates; failed and excluded candidates are omitted.
    ///
    /// # Errors
    ///
    /// Returns an error if overrides are disabled, tensor identities, shapes, strides, bounds, or alias layouts are invalid, frontend workspace sizing overflows, CUDA contexts disagree or cannot be bound, or native variant-pack creation or workspace querying fails.
    pub fn max_workspace_size_with_runtime_overrides(
        &self,
        context: &Context,
        overrides: &RuntimeOverrides,
    ) -> Result<usize> {
        self.runtime_workspace_query(context, overrides)?
            .maximum(self.available_plans().map(|(_, plan)| plan))
    }

    fn runtime_workspace_query<'a>(
        &self,
        context: &'a Context,
        overrides: &RuntimeOverrides,
    ) -> Result<RuntimeWorkspaceQuery<'a>> {
        RuntimeWorkspaceQuery::create(
            context,
            self.graph.dynamic_shape_enabled() || self.graph.override_shape_enabled(),
            &self.graph.runtime_shape_constraints,
            &self.artifacts.tensors,
            &self.artifacts.binding_replacements,
            self.binding_template.ids(),
            overrides,
            &self.lowered.layout.workspace_layout,
        )
    }
}
