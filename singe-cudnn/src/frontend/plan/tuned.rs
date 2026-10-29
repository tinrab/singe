use std::{
    mem::MaybeUninit,
    sync::{Arc, Mutex},
};

use singe_cuda::{
    context::Context as CudaContext, error::Error as CudaError, graph::Graph as CudaGraph,
    memory::DeviceMemory, submission::Resources, view::DeviceRegionProvider,
};

use crate::{
    backend::execution::ExecutionPlan,
    context::Context,
    error::{Error, Result},
    frontend::{
        bindings::{Bindings, RuntimeOverrides},
        plan::{CompiledGraph, ExecuteArgs, PlanId},
    },
    operation::FrontendExecution,
    utility::validate_context_pair,
};

/// An autotuned native winner scoped to its measured CUDA context and runtime layouts.
/// Retains the actual compiled collection, including its target, engine choices and build policy results.
/// Clones share native resources without changing another graph handle's default selection.
#[derive(Debug, Clone)]
pub struct TunedPlan {
    graph: CompiledGraph,
    context: Arc<CudaContext>,
    overrides: RuntimeOverrides,
}

impl TunedPlan {
    /// Called only after successful measurements with these validated layouts and selected graph handle.
    pub(crate) fn new(
        graph: CompiledGraph,
        context: Arc<CudaContext>,
        overrides: RuntimeOverrides,
    ) -> Self {
        Self {
            graph,
            context,
            overrides,
        }
    }

    /// Borrows the selected winner's stable identity in its retained compiled collection.
    #[must_use]
    pub fn id(&self) -> &PlanId {
        self.graph.default_plan()
    }

    /// The retained selection handle and candidate metadata for the measured compilation.
    /// Cloning this handle permits an explicit independent selection without changing this winner.
    #[must_use]
    pub fn compiled_graph(&self) -> &CompiledGraph {
        &self.graph
    }

    /// Borrows the CUDA context against which this winner was measured.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Runtime overrides used for every operation on this winner.
    /// Empty overrides refer to the retained graph's immutable base layouts.
    #[must_use]
    pub fn overrides(&self) -> &RuntimeOverrides {
        &self.overrides
    }

    /// The raw native plan, whose low-level use retains its separate layout and lifetime contracts.
    #[must_use]
    pub fn execution_plan(&self) -> &ExecutionPlan {
        self.graph.execution_plan()
    }

    /// Creates graph-aware tensor bindings with this winner's retained scalar defaults.
    ///
    /// # Errors
    ///
    /// Returns an error if a scalar default references an unknown or incompatible tensor, scalar alignment or alias resolution fails, or aligned host scalar storage cannot be created.
    pub fn bindings(&self) -> Result<Bindings<'_>> {
        self.graph.bindings()
    }

    /// Workspace for this winner's measured layouts, including frontend scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if the context differs from the measured target, runtime layouts or frontend workspace sizing fail, or the native workspace query or byte-count conversion fails.
    pub fn workspace_size(&self, context: &Context) -> Result<usize> {
        self.require_context(context)?;
        self.graph
            .workspace_size_with_runtime_overrides(context, &self.overrides)
    }

    /// Executes with the measured layouts.
    ///
    /// # Safety
    /// Retain this winner, bindings and workspace through completion or every captured replay.
    /// Buffers must satisfy native initialization, aliasing and producer-ordering contracts.
    ///
    /// # Errors
    ///
    /// Returns an error if the context differs from the measured target, bindings or measured layouts are invalid, workspace is insufficient, resource retention or diagnostic logging fails, workspace initialization fails, or native execution fails.
    pub unsafe fn execute(
        &self,
        context: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<MaybeUninit<u8>>>,
    ) -> Result<()> {
        self.require_context(context)?;
        // SAFETY: the caller upholds this function's safety contract.
        // SAFETY: the caller guarantees valid bindings, ordered accesses and retained storage through completion or replay.
        unsafe {
            self.graph.execute(
                context,
                ExecuteArgs::new(bindings)
                    .with_workspace(workspace)
                    .with_overrides(&self.overrides),
            )
        }
    }

    /// Populates a native CUDA graph using this winner and its measured layouts.
    ///
    /// # Safety
    /// Retain this winner, bindings and workspace through every replay and satisfy the native execution contract.
    ///
    /// # Errors
    ///
    /// Returns an error for a context mismatch, invalid bindings or measured layouts, insufficient workspace, a nonempty destination graph, resource-retention or workspace-initializer preparation failure, or CUDA graph query or native population failure.
    pub unsafe fn populate_cuda_graph(
        &self,
        context: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<MaybeUninit<u8>>>,
        graph: &mut CudaGraph,
    ) -> Result<()> {
        self.require_context(context)?;
        // SAFETY: the caller upholds this function's safety contract.
        // SAFETY: the caller guarantees valid bindings, ordered accesses and retained storage through completion or replay.
        unsafe {
            self.graph.populate_cuda_graph(
                context,
                ExecuteArgs::new(bindings)
                    .with_workspace(workspace)
                    .with_overrides(&self.overrides),
                graph,
            )
        }
    }

    /// Updates a populated CUDA graph using the measured layouts.
    ///
    /// # Safety
    /// Retain this winner and new binding/workspace storage through their replays.
    /// Previous bindings remain valid through any executable snapshots or pending launches that still use them.
    ///
    /// # Errors
    ///
    /// Returns an error for a context mismatch, invalid bindings or measured layouts, insufficient workspace, resource-retention or workspace-initializer preparation failure, or CUDA graph operations or native update failure.
    pub unsafe fn update_cuda_graph(
        &self,
        context: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<MaybeUninit<u8>>>,
        graph: &CudaGraph,
    ) -> Result<()> {
        self.require_context(context)?;
        // SAFETY: the caller upholds this function's safety contract.
        // SAFETY: the caller guarantees valid bindings, ordered accesses and retained storage through completion or replay.
        unsafe {
            self.graph.update_cuda_graph(
                context,
                ExecuteArgs::new(bindings)
                    .with_workspace(workspace)
                    .with_overrides(&self.overrides),
                graph,
            )
        }
    }

    /// Prepares owned submission or recording with the retained winner and measured layouts.
    /// The prepared operation remains valid after this winner is dropped.
    ///
    /// # Safety
    /// Retain borrowed and raw allocations in `resources` through completion or replay.
    /// Typed, encoded-byte and retained-region bindings derive graph access declarations automatically.
    /// Raw pointers require explicit access declarations and producer dependencies.
    /// Retained-region bindings also supply allocation owners.
    /// Establish producer dependencies not represented by recorded readiness or tracked CUDA accesses.
    /// Outputs must remain pending until successful execution and satisfy the measured native layouts.
    /// Keep the shared cuDNN handle associated with the measured CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if the context mutex is poisoned, context or workspace targets differ from the measured target, bindings or measured layouts are invalid, workspace is insufficient, or owned execution-resource preparation fails.
    pub unsafe fn prepare_owned<T: 'static>(
        &self,
        context: Arc<Mutex<Context>>,
        bindings: Bindings<'_>,
        workspace: DeviceMemory<MaybeUninit<u8>>,
        resources: Resources,
        output: T,
    ) -> Result<FrontendExecution<T>> {
        {
            let context = context.lock().map_err(|_| {
                Error::Cuda(CudaError::ResourcePoisoned {
                    name: "cudnn context",
                })
            })?;
            self.require_context(&context)?;
        }
        validate_context_pair(
            Some(&self.context),
            Some(workspace.region().context()),
            "autotuned workspace",
        )?;
        // SAFETY: this method's caller establishes the ordinary owned-execution resource contract.
        unsafe {
            FrontendExecution::create(
                context,
                Arc::new(self.graph.clone()),
                bindings,
                workspace,
                &self.overrides,
                resources,
                output,
            )
        }
    }

    fn require_context(&self, context: &Context) -> Result<()> {
        validate_context_pair(
            Some(&self.context),
            Some(context.cuda_context()),
            "autotuned plan",
        )
    }
}
