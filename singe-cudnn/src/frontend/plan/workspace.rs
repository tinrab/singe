use std::collections::BTreeMap;

use crate::{
    backend::execution::{ExecutionPlan, VariantPack},
    context::Context,
    error::{Error, Result},
    frontend::{
        bindings::RuntimeOverrides,
        error::Error as FrontendError,
        graph::{RuntimeShapeConstraints, TensorRecord},
        plan::{BindingReplacement, BuiltPlanCandidates, CompiledGraph, PlanId},
        workspace::WorkspaceLayout,
    },
    tensor::TensorId,
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

/// Selects the plans a workspace query covers.
#[derive(Debug, Clone, Copy)]
pub enum WorkspacePlans<'a> {
    /// The default plan of a compiled graph; built candidates have none.
    Default,
    /// One retained or available plan.
    Plan(&'a PlanId),
    /// The maximum over every retained plan of a compiled graph, or every available built candidate.
    All,
}

impl CompiledGraph {
    /// Returns the workspace size in bytes, including frontend scratch, for `plans` at the static layouts or at `runtime`'s layouts.
    ///
    /// Static queries read cached requirements; runtime queries ask cuDNN for each plan.
    /// [`Self::workspace_size`] is the static default-plan query without a `Result`.
    ///
    /// # Errors
    ///
    /// Returns an error if this handle does not retain the selected plan, overrides are disabled, tensor identities, shapes, strides, bounds, or alias layouts are invalid, frontend workspace sizing overflows, CUDA contexts disagree or cannot be bound, or native variant-pack creation or workspace querying fails.
    pub fn workspace_size_for(
        &self,
        plans: WorkspacePlans<'_>,
        runtime: Option<(&Context, &RuntimeOverrides)>,
    ) -> Result<usize> {
        let Some((context, overrides)) = runtime else {
            return match plans {
                WorkspacePlans::Default => Ok(self.workspace_size()),
                WorkspacePlans::Plan(plan) => self.static_workspace_size(plan),
                WorkspacePlans::All => Ok(self.max_static_workspace_size()),
            };
        };

        let query = self.runtime_workspace_query(context, overrides)?;
        match plans {
            WorkspacePlans::Default => {
                query.for_plan(self.execution_plan_for(self.default_plan())?)
            }
            WorkspacePlans::Plan(plan) => query.for_plan(self.execution_plan_for(plan)?),
            WorkspacePlans::All => query.maximum(self.execution_plans()),
        }
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
    /// Returns the workspace size in bytes, including frontend scratch, for `plans` at the static layouts or at `runtime`'s layouts.
    ///
    /// [`WorkspacePlans::All`] covers the available candidates and is zero when none is available; failed and excluded candidates are omitted.
    ///
    /// # Errors
    ///
    /// Returns an error if `plans` is [`WorkspacePlans::Default`], the selected identity is absent or its candidate is excluded, unbuilt, or failed, overrides are disabled, tensor identities, shapes, strides, bounds, or alias layouts are invalid, frontend workspace sizing overflows, CUDA contexts disagree or cannot be bound, or native variant-pack creation or workspace querying fails.
    pub fn workspace_size_for(
        &self,
        plans: WorkspacePlans<'_>,
        runtime: Option<(&Context, &RuntimeOverrides)>,
    ) -> Result<usize> {
        let plan = match plans {
            WorkspacePlans::Default => {
                return Err(FrontendError::OperationFieldMissing {
                    operation: "built candidate workspace query",
                    field: "plan",
                }
                .into());
            }
            WorkspacePlans::Plan(plan) => Some(plan),
            WorkspacePlans::All => None,
        };

        let Some((context, overrides)) = runtime else {
            return plan.map_or_else(
                || Ok(self.max_static_workspace_size()),
                |plan| self.static_workspace_size(plan),
            );
        };

        let query = self.runtime_workspace_query(context, overrides)?;
        match plan {
            Some(plan) => query.for_plan(self.execution_plan_for(plan)?),
            None => query.maximum(self.available_plans().map(|(_, plan)| plan)),
        }
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
