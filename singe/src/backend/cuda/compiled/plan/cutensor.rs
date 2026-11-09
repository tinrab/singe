use std::{collections::HashMap, hash::RandomState, mem::MaybeUninit, sync::Arc};

use singe_cuda::{
    graph::Graph as NativeCudaGraph, memory::DeviceMemory, stream::StreamCaptureMode,
};

use crate::{
    backend::{
        cuda::{
            CudaStream,
            compiled::plan::lowering::LoweringProvider,
            cutensor::{
                CachedCutensorContractionPlan, cached_cutensor_contraction_plan,
                cutensor_workspace, launch_cutensor_contraction_f32,
            },
            graph_binding::validate_graph_node_value_tensor,
            graph_value::CudaGraphValue,
            runtime::CudaRuntime,
        },
        registry::{CudaGraphProvider, Operation, operation_support},
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    utility::Locked,
};

/// A captured cuTENSOR contraction graph with its retained execution plan and optional workspace.
/// Input and output allocations are retained separately by the compiled graph.
#[derive(Debug)]
pub struct CutensorContractionGraph {
    child_graph: NativeCudaGraph,
    workspace: Option<DeviceMemory<MaybeUninit<u8>>>,
    plan: CachedCutensorContractionPlan,
}

impl CutensorContractionGraph {
    fn new(
        child_graph: NativeCudaGraph,
        workspace: Option<DeviceMemory<MaybeUninit<u8>>>,
        plan: CachedCutensorContractionPlan,
    ) -> Self {
        Self {
            child_graph,
            workspace,
            plan,
        }
    }

    /// Returns the captured native CUDA graph.
    #[must_use]
    pub fn child_graph(&self) -> &NativeCudaGraph {
        &self.child_graph
    }

    /// Returns the workspace allocation byte count, or zero if no workspace is needed.
    pub fn workspace_byte_size(&self) -> usize {
        self.workspace.as_ref().map_or(0, DeviceMemory::len)
    }

    /// Transfers the captured graph, optional workspace, and cached execution plan to the caller.
    /// Keep the plan, workspace, and externally retained tensor bindings alive through graph execution.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        NativeCudaGraph,
        Option<DeviceMemory<MaybeUninit<u8>>>,
        CachedCutensorContractionPlan,
    ) {
        (self.child_graph, self.workspace, self.plan)
    }
}

/// Captures a contraction with coefficients `alpha` and `beta` into a child CUDA graph.
/// Operand tuples supply cached node IDs and mode labels; `d_mode` labels the output.
/// The returned object retains the plan and workspace, while the compiled graph retains tensor allocations.
///
/// # Errors
///
/// Returns an error if an operand is missing, tensors disagree with IR metadata, mode or region validation fails, cache or plan mutexes are poisoned, or stream creation, plan selection, workspace allocation, or captured execution fails.
pub fn build_cutensor_contraction_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    a: (ValueId, &[i64]),
    b: (ValueId, &[i64]),
    c: (ValueId, &[i64]),
    d_mode: &[i64],
    alpha: f32,
    beta: f32,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<CutensorContractionGraph> {
    let a_tensor = cache
        .get(&a.0)
        .ok_or_else(|| Error::missing_input("build cutensor contraction graph"))?;
    let b_tensor = cache
        .get(&b.0)
        .ok_or_else(|| Error::missing_input("build cutensor contraction graph"))?;
    let c_tensor = cache
        .get(&c.0)
        .ok_or_else(|| Error::missing_input("build cutensor contraction graph"))?;
    validate_graph_node_value_tensor(ir, a.0, a_tensor)?;
    validate_graph_node_value_tensor(ir, b.0, b_tensor)?;
    validate_graph_node_value_tensor(ir, c.0, c_tensor)?;
    validate_graph_node_value_tensor(ir, id, output)?;

    let output_value = id;
    let stream = CudaStream::create(ir.required_value(output_value)?.device())?;
    let cached_plan = cached_cutensor_contraction_plan(
        runtime,
        a_tensor,
        a.1,
        b_tensor,
        b.1,
        c_tensor,
        c.1,
        d_mode,
        output.region()?,
    )?;
    let mut workspace = cutensor_workspace(runtime.context(), cached_plan.workspace_size())?;

    let (graph, ()) = stream.capture(StreamCaptureMode::ThreadLocal, || {
        let plan = cached_plan.plan().locked("cuda runtime cache")?;
        // SAFETY: the cached plan matches these tensors and compiled graph execution retains their storage and workspace for replay.
        unsafe {
            launch_cutensor_contraction_f32(
                &plan,
                &stream,
                a_tensor,
                b_tensor,
                c_tensor,
                output.region()?,
                alpha,
                beta,
                workspace.as_mut(),
            )
        }
    })?;

    Ok(CutensorContractionGraph::new(
        graph.into_native(),
        workspace,
        cached_plan,
    ))
}

/// CUDA graph lowering provider for operations supported by cuTENSOR contraction plans.
#[derive(Debug, Clone, Copy)]
pub struct CutensorProvider;

impl LoweringProvider for CutensorProvider {
    fn kind(&self) -> CudaGraphProvider {
        CudaGraphProvider::Cutensor
    }

    fn cost(&self) -> u32 {
        30
    }

    fn supports_operation(&self, operation: Operation) -> bool {
        operation_support(operation).supports_cuda_graph_provider(CudaGraphProvider::Cutensor)
    }
}
