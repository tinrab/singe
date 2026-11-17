use singe_cuda::{
    graph::{ExecutableGraph, Graph as CudaGraph, GraphNode},
    view::DeviceOutputView,
};

use crate::{
    context::Context,
    error::{Error, Result},
    frontend::{
        bindings::{Bindings, RuntimeOverrides},
        plan::{CompiledGraph, PlanId},
    },
    utility::validate_context_pair,
};

/// Bindings, workspace, plan selection and runtime overrides for one native execution.
///
/// Omitted plans select the handle's default plan.
/// Omitted overrides execute the compiled shapes.
#[derive(Debug)]
pub struct ExecuteArgs<'a, 'graph> {
    /// Tensor pointer registrations and owned scalar values for this execution.
    pub bindings: &'a Bindings<'graph>,
    /// Optional device scratch allocation shared by frontend intermediates and native workspace.
    pub workspace: Option<DeviceOutputView<'a, u8>>,
    /// Optional identity in this compiled collection; `None` selects its default plan.
    pub plan: Option<&'a PlanId>,
    /// Optional execution-time shapes and strides; `None` uses the compiled layouts.
    pub overrides: Option<&'a RuntimeOverrides>,
}

impl<'a, 'graph> ExecuteArgs<'a, 'graph> {
    /// Uses supplied tensor bindings, the default plan, compiled layouts, and no workspace.
    #[must_use]
    pub fn new(bindings: &'a Bindings<'graph>) -> Self {
        Self {
            bindings,
            workspace: None,
            plan: None,
            overrides: None,
        }
    }

    /// Replaces the optional scratch allocation without checking its context or capacity.
    #[must_use]
    pub fn with_workspace(
        mut self,
        workspace: impl Into<Option<DeviceOutputView<'a, u8>>>,
    ) -> Self {
        self.workspace = workspace.into();
        self
    }

    /// Selects an optional plan identity; `None` restores this handle's default selection.
    #[must_use]
    pub fn with_plan(mut self, plan: impl Into<Option<&'a PlanId>>) -> Self {
        self.plan = plan.into();
        self
    }

    /// Replaces optional runtime layouts; `None` restores the compiled shapes and strides.
    #[must_use]
    pub fn with_overrides(mut self, overrides: impl Into<Option<&'a RuntimeOverrides>>) -> Self {
        self.overrides = overrides.into();
        self
    }
}

impl CompiledGraph {
    /// Executes the selected plan with the supplied tensor bindings.
    ///
    /// # Safety
    ///
    /// Uphold the allocation, initialization, access and lifetime requirements on [`Bindings`].
    ///
    /// # Errors
    ///
    /// Returns an error if the plan is unknown, runtime overrides are disabled or invalid, bindings are missing or incompatible, aliases or alignment are invalid, the workspace is too small, building the variant pack fails, preparing the execution context or retaining resources fails, diagnostic logging fails, initializing the workspace fails, or native execution fails.
    pub unsafe fn execute_async(&self, context: &Context, args: ExecuteArgs<'_, '_>) -> Result<()> {
        let plan = args.plan.unwrap_or_else(|| self.default_plan());
        let execution_plan = self.execution_plan_for(plan)?;
        let no_overrides = RuntimeOverrides::new();
        let variant_pack = self.variant_pack_with_overrides(
            context.cuda_context(),
            args.bindings,
            args.workspace,
            args.overrides.unwrap_or(&no_overrides),
        )?;
        // SAFETY: `variant_pack` holds `args.bindings`, which this function's `Bindings` contract keeps live and initialized at entry.
        unsafe {
            self.log_execution(context, &variant_pack, plan)?;
        }
        // SAFETY: `variant_pack_with_overrides` checked contexts, capacities and alignment, and this function's `Bindings` contract supplies the initialization, producer ordering and lifetime that `ExecutionPlan::execute_async` requires.
        unsafe { execution_plan.execute_async(context, &variant_pack) }
    }

    /// Populates an empty CUDA graph with the selected plan.
    ///
    /// # Safety
    ///
    /// Uphold the allocation, initialization, access and lifetime requirements on [`Bindings`].
    ///
    /// # Errors
    ///
    /// Returns an error if the plan is unknown, runtime overrides are disabled or invalid, bindings are missing or incompatible, aliases or alignment are invalid, the workspace is too small, building the variant pack fails, the destination graph is not empty, the CUDA contexts are incompatible, retaining resources or preparing workspace initializers fails, or the CUDA graph query or native population fails.
    pub unsafe fn populate_cuda_graph(
        &self,
        context: &Context,
        args: ExecuteArgs<'_, '_>,
        graph: &mut CudaGraph,
    ) -> Result<()> {
        let plan = args.plan.unwrap_or_else(|| self.default_plan());
        let no_overrides = RuntimeOverrides::new();
        let variant_pack = self.variant_pack_with_overrides(
            context.cuda_context(),
            args.bindings,
            args.workspace,
            args.overrides.unwrap_or(&no_overrides),
        )?;
        if !graph.root_nodes()?.is_empty() {
            return Err(Error::CudaGraphPopulateRequiresEmptyGraph);
        }
        // SAFETY: `variant_pack_with_overrides` checked the bindings' layouts, and this function's `Bindings` contract keeps the captured addresses valid and initialized for every replay of the populated graph.
        unsafe {
            self.execution_plan_for(plan)?
                .populate_graph(context, &variant_pack, graph)
        }
    }

    /// Adds the selected plan to `graph` as a child-graph node after `dependencies`.
    ///
    /// # Safety
    ///
    /// Uphold the allocation, initialization, access and lifetime requirements on [`Bindings`].
    ///
    /// # Errors
    ///
    /// Returns an error if the parent graph context is incompatible, creating or populating the child graph fails, or CUDA rejects the dependencies or child-node insertion.
    /// Population can fail for an unknown plan, invalid overrides or bindings, insufficient workspace, or native graph preparation failure.
    pub unsafe fn append_to_cuda_graph(
        &self,
        context: &Context,
        args: ExecuteArgs<'_, '_>,
        graph: &mut CudaGraph,
        dependencies: &[GraphNode],
    ) -> Result<GraphNode> {
        validate_context_pair(
            Some(context.cuda_context()),
            Some(graph.context()),
            "cuda graph",
        )?;
        let mut child_graph = context.cuda_context().create_graph()?;
        // SAFETY: `populate_cuda_graph` has the same `Bindings` contract as this function, and `child_graph` is new and empty.
        unsafe { self.populate_cuda_graph(context, args, &mut child_graph) }?;
        Ok(graph.add_child_graph_node(dependencies, &child_graph)?)
    }

    /// Updates a CUDA graph previously populated with this compiled graph.
    ///
    /// # Safety
    ///
    /// Uphold the allocation, initialization, access and lifetime requirements on [`Bindings`].
    ///
    /// # Errors
    ///
    /// Returns an error if the plan is unknown, runtime overrides are disabled or invalid, bindings are missing or incompatible, aliases or alignment are invalid, the workspace is too small, building the variant pack fails, the CUDA contexts are incompatible, retaining resources or preparing workspace initializers fails, or a CUDA graph operation or the native update fails.
    pub unsafe fn update_cuda_graph(
        &self,
        context: &Context,
        args: ExecuteArgs<'_, '_>,
        graph: &mut CudaGraph,
    ) -> Result<()> {
        let plan = args.plan.unwrap_or_else(|| self.default_plan());
        let no_overrides = RuntimeOverrides::new();
        let overrides = args.overrides.unwrap_or(&no_overrides);
        overrides.validate_enabled(self.core.shape_policy.overrides_enabled)?;
        let variant_pack = self.variant_pack_with_overrides(
            context.cuda_context(),
            args.bindings,
            args.workspace,
            overrides,
        )?;
        // SAFETY: `variant_pack_with_overrides` checked the bindings' layouts, and this function's `Bindings` contract keeps the new addresses valid and older executables' storage retained for every replay.
        unsafe {
            self.execution_plan_for(plan)?
                .update_graph(context, &variant_pack, graph)
        }
    }

    /// Updates a child-graph node of `graph` created by [`Self::append_to_cuda_graph`].
    /// When `exec` is supplied, the instantiated graph receives the same update.
    ///
    /// # Safety
    ///
    /// Uphold the allocation, initialization, access and lifetime requirements on [`Bindings`].
    ///
    /// # Errors
    ///
    /// Returns an error if child-graph lookup or cloning fails, the node or executable cannot accept the update, execution bindings or layouts are invalid, workspace is insufficient, contexts disagree, native graph preparation or update fails, or installing the updated child graph fails.
    pub unsafe fn update_cuda_graph_node(
        &self,
        context: &Context,
        args: ExecuteArgs<'_, '_>,
        graph: &mut CudaGraph,
        node: &GraphNode,
        exec: Option<&mut ExecutableGraph>,
    ) -> Result<()> {
        let mut child_graph = graph.child_graph(node)?.try_clone()?;
        if let Some(exec) = exec.as_deref() {
            exec.validate_child_graph_node_update(node, &child_graph)?;
        }

        // SAFETY: `update_cuda_graph` has the same `Bindings` contract as this function, applied to a clone of the node's child graph.
        unsafe { self.update_cuda_graph(context, args, &mut child_graph) }?;
        graph.set_child_graph(node, &child_graph)?;
        if let Some(exec) = exec {
            exec.set_child_graph_node(node, &child_graph)?;
        }
        Ok(())
    }
}
