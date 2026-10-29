use std::mem::MaybeUninit;

use singe_cuda::{
    graph::{ExecutableGraph, Graph as CudaGraph, GraphNode},
    memory::DeviceMemory,
};

use crate::{
    context::Context,
    error::Result,
    frontend::{
        bindings::{Bindings, RuntimeOverrides},
        error::Error as FrontendError,
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
    pub workspace: Option<&'a mut DeviceMemory<MaybeUninit<u8>>>,
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
        workspace: impl Into<Option<&'a mut DeviceMemory<MaybeUninit<u8>>>>,
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
    /// Uphold the allocation, initialization, access and lifetime requirements on [`Bindings`].
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown plan, disabled or invalid runtime overrides, missing or incompatible bindings, invalid aliases or alignment, insufficient workspace, or variant-pack construction failure, execution-context or resource-retention failure, diagnostic logging failure, workspace initialization failure, or native execution failure.
    pub unsafe fn execute(&self, context: &Context, args: ExecuteArgs<'_, '_>) -> Result<()> {
        let plan = args.plan.unwrap_or_else(|| self.default_plan());
        let execution_plan = self.execution_plan_for(plan)?;
        let no_overrides = RuntimeOverrides::new();
        let variant_pack = self.variant_pack_with_overrides(
            context.cuda_context(),
            args.bindings,
            args.workspace,
            args.overrides.unwrap_or(&no_overrides),
        )?;
        // SAFETY: the variant pack uses the same established bindings and runtime layouts as the native execution below.
        unsafe {
            self.log_execution(context, &variant_pack, plan)?;
        }
        // SAFETY: the caller upholds the Bindings execution contract.
        unsafe { execution_plan.execute(context, &variant_pack) }
    }

    /// Populates an empty CUDA graph with the selected plan.
    ///
    /// # Safety
    /// Uphold the allocation, initialization, access and lifetime requirements on [`Bindings`].
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown plan, disabled or invalid runtime overrides, missing or incompatible bindings, invalid aliases or alignment, insufficient workspace, or variant-pack construction failure, a nonempty destination graph, incompatible CUDA contexts, resource-retention or workspace-initializer preparation failure, or CUDA graph query or native population failure.
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
            return Err(FrontendError::CudaGraphPopulateRequiresEmptyGraph.into());
        }
        // SAFETY: the caller upholds the Bindings execution contract.
        unsafe {
            self.execution_plan_for(plan)?
                .populate_graph(context, &variant_pack, graph)
        }
    }

    /// Adds the selected plan to `graph` as a child-graph node after `dependencies`.
    ///
    /// # Safety
    /// Uphold the allocation, initialization, access and lifetime requirements on [`Bindings`].
    ///
    /// # Errors
    ///
    /// Returns an error if the parent graph context is incompatible, creating or populating the child graph fails, or CUDA rejects the dependencies or child-node insertion. Population can fail for an unknown plan, invalid overrides or bindings, insufficient workspace, or native graph preparation failure.
    pub unsafe fn append_to_cuda_graph(
        &self,
        context: &Context,
        args: ExecuteArgs<'_, '_>,
        graph: &mut CudaGraph,
        dependencies: &[GraphNode],
    ) -> Result<GraphNode> {
        validate_context_pair(Some(context.cuda_context()), graph.context(), "cuda graph")?;
        let mut child_graph = context.cuda_context().create_graph()?;
        // SAFETY: the caller upholds the Bindings execution contract.
        unsafe { self.populate_cuda_graph(context, args, &mut child_graph) }?;
        Ok(graph.add_child_graph_node(dependencies, &child_graph)?)
    }

    /// Updates a CUDA graph previously populated with this compiled graph.
    ///
    /// # Safety
    /// Uphold the allocation, initialization, access and lifetime requirements on [`Bindings`].
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown plan, disabled or invalid runtime overrides, missing or incompatible bindings, invalid aliases or alignment, insufficient workspace, or variant-pack construction failure, incompatible CUDA contexts, resource-retention or workspace-initializer preparation failure, or CUDA graph operations or native update failure.
    pub unsafe fn update_cuda_graph(
        &self,
        context: &Context,
        args: ExecuteArgs<'_, '_>,
        graph: &CudaGraph,
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
        // SAFETY: the caller upholds the Bindings execution contract.
        unsafe {
            self.execution_plan_for(plan)?
                .update_graph(context, &variant_pack, graph)
        }
    }

    /// Updates a child-graph node created by [`Self::append_to_cuda_graph`].
    /// When `exec` is supplied, the instantiated graph receives the same update.
    ///
    /// # Safety
    /// Uphold the allocation, initialization, access and lifetime requirements on [`Bindings`].
    ///
    /// # Errors
    ///
    /// Returns an error if child-graph lookup or cloning fails, the node or executable cannot accept the update, execution bindings or layouts are invalid, workspace is insufficient, contexts disagree, native graph preparation or update fails, or installing the updated child graph fails.
    pub unsafe fn update_cuda_graph_node(
        &self,
        context: &Context,
        args: ExecuteArgs<'_, '_>,
        node: &GraphNode,
        exec: Option<&mut ExecutableGraph>,
    ) -> Result<()> {
        let child_graph = node.child_graph()?.try_clone()?;
        if let Some(exec) = exec.as_deref() {
            exec.validate_child_graph_node_update(node, &child_graph)?;
        }
        // SAFETY: the caller upholds the Bindings execution contract.
        unsafe { self.update_cuda_graph(context, args, &child_graph) }?;
        node.set_child_graph(&child_graph)?;
        if let Some(exec) = exec {
            exec.set_child_graph_node(node, &child_graph)?;
        }
        Ok(())
    }
}
