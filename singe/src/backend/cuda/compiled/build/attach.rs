use singe_cuda::graph::{Graph as NativeCudaGraph, GraphNode};
use singe_cufft::plan::Plan;

use crate::{
    backend::cuda::{
        compiled::{
            build::realized::record_realized_cuda_lowering,
            executable::{
                CudaCompiledGraph, CudaCompiledGraphStats, CudnnGraphStep, CutensorGraphStep,
                LibraryResource,
            },
            plan::{
                cublas::CublasLtMatmulGraph,
                cutensor::CutensorContractionGraph,
                lowering::{CudaGraphProvider, RealizedLoweringPlan},
            },
        },
        cudnn_backend::CudnnBackendExecutable,
        graph_value::CudaGraphValue,
    },
    error::Result,
    graph::ir::{Graph, ValueId},
};

// Adds a generated kernel's or cuBLAS's child graph, which differ in what they count and report.
fn attach_child_graph(
    native_graph: &mut NativeCudaGraph,
    dependencies: &[GraphNode],
    stats: &mut CudaCompiledGraphStats,
    realized_lowering: &mut RealizedLoweringPlan,
    library_resources: &mut Vec<LibraryResource>,
    ir: &Graph,
    id: ValueId,
    child_graph: NativeCudaGraph,
    provider: CudaGraphProvider,
) -> Result<GraphNode> {
    let graph_node = native_graph.add_child_graph_node(dependencies, &child_graph)?;
    library_resources.push(LibraryResource::Graph(child_graph));
    if provider == CudaGraphProvider::Kernel {
        stats.record_kernel_child_graph_node();
    } else {
        stats.record_cublas_child_graph_node();
    }

    record_realized_cuda_lowering(realized_lowering, id, ir, provider)?;
    Ok(graph_node)
}

pub fn attach_kernel_graph(
    native_graph: &mut NativeCudaGraph,
    dependencies: &[GraphNode],
    stats: &mut CudaCompiledGraphStats,
    realized_lowering: &mut RealizedLoweringPlan,
    library_resources: &mut Vec<LibraryResource>,
    ir: &Graph,
    id: ValueId,
    child_graph: NativeCudaGraph,
) -> Result<GraphNode> {
    attach_child_graph(
        native_graph,
        dependencies,
        stats,
        realized_lowering,
        library_resources,
        ir,
        id,
        child_graph,
        CudaGraphProvider::Kernel,
    )
}

pub fn attach_cublaslt_graph(
    native_graph: &mut NativeCudaGraph,
    dependencies: &[GraphNode],
    stats: &mut CudaCompiledGraphStats,
    realized_lowering: &mut RealizedLoweringPlan,
    library_resources: &mut Vec<LibraryResource>,
    ir: &Graph,
    id: ValueId,
    graph: CublasLtMatmulGraph,
) -> Result<GraphNode> {
    stats.add_cublaslt_workspace_bytes(graph.workspace_byte_size())?;
    let graph_node = native_graph.add_child_graph_node(dependencies, graph.child_graph())?;
    library_resources.push(LibraryResource::Graph(graph.into_native()));
    stats.record_cublas_child_graph_node();
    record_realized_cuda_lowering(realized_lowering, id, ir, CudaGraphProvider::CublasLt)?;
    Ok(graph_node)
}

pub fn attach_cublas_graph(
    native_graph: &mut NativeCudaGraph,
    dependencies: &[GraphNode],
    stats: &mut CudaCompiledGraphStats,
    realized_lowering: &mut RealizedLoweringPlan,
    library_resources: &mut Vec<LibraryResource>,
    ir: &Graph,
    id: ValueId,
    child_graph: NativeCudaGraph,
) -> Result<GraphNode> {
    attach_child_graph(
        native_graph,
        dependencies,
        stats,
        realized_lowering,
        library_resources,
        ir,
        id,
        child_graph,
        CudaGraphProvider::Cublas,
    )
}

pub struct CudnnGraphAttachment {
    pub(crate) child_graph: NativeCudaGraph,
    pub(crate) executable: CudnnBackendExecutable,
    pub(crate) update_on_replay: bool,
}

pub fn attach_cudnn_graph(
    native_graph: &mut NativeCudaGraph,
    dependencies: &[GraphNode],
    stats: &mut CudaCompiledGraphStats,
    realized_lowering: &mut RealizedLoweringPlan,
    cudnn_steps: &mut Vec<CudnnGraphStep>,
    ir: &Graph,
    id: ValueId,
    output: &CudaGraphValue,
    attachment: CudnnGraphAttachment,
) -> Result<GraphNode> {
    CudaCompiledGraph::record_cudnn_workspace(stats, &attachment.executable)?;
    let graph_node = native_graph.add_child_graph_node(dependencies, &attachment.child_graph)?;
    cudnn_steps.push(CudnnGraphStep::new(
        graph_node.clone(),
        attachment.child_graph,
        output.clone(),
        attachment.executable,
        attachment.update_on_replay,
    ));
    stats.record_cudnn_backend_graph_node();
    record_realized_cuda_lowering(realized_lowering, id, ir, CudaGraphProvider::CudnnBackend)?;
    Ok(graph_node)
}

pub fn attach_cutensor_graph(
    native_graph: &mut NativeCudaGraph,
    dependencies: &[GraphNode],
    stats: &mut CudaCompiledGraphStats,
    realized_lowering: &mut RealizedLoweringPlan,
    cutensor_steps: &mut Vec<CutensorGraphStep>,
    ir: &Graph,
    id: ValueId,
    output: &CudaGraphValue,
    graph: CutensorContractionGraph,
) -> Result<GraphNode> {
    stats.add_cutensor_workspace_bytes(graph.workspace_byte_size())?;
    let graph_node = native_graph.add_child_graph_node(dependencies, graph.child_graph())?;
    let (child_graph, workspace, plan) = graph.into_parts();
    cutensor_steps.push(CutensorGraphStep::new(
        graph_node.clone(),
        child_graph,
        output.clone(),
        workspace,
        plan,
    ));
    stats.record_cutensor_child_graph_node();
    record_realized_cuda_lowering(realized_lowering, id, ir, CudaGraphProvider::Cutensor)?;
    Ok(graph_node)
}

// Attaches a captured library child graph that provider `kind` built, keeping it and the cuFFT plan its capture reads, if any, alive with the compiled graph.
pub fn attach_library_graph(
    native_graph: &mut NativeCudaGraph,
    dependencies: &[GraphNode],
    realized_lowering: &mut RealizedLoweringPlan,
    library_resources: &mut Vec<LibraryResource>,
    ir: &Graph,
    id: ValueId,
    kind: CudaGraphProvider,
    child_graph: NativeCudaGraph,
    plan: Option<Plan>,
) -> Result<GraphNode> {
    let graph_node = native_graph.add_child_graph_node(dependencies, &child_graph)?;
    library_resources.push(LibraryResource::Graph(child_graph));
    library_resources.extend(plan.map(LibraryResource::FftPlan));
    record_realized_cuda_lowering(realized_lowering, id, ir, kind)?;
    Ok(graph_node)
}
