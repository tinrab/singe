use singe_cuda::graph::{Graph as NativeCudaGraph, GraphNode};

use crate::{
    backend::cuda::{
        compiled::{
            build::{fusion::GraphFusionLowering, realized::record_realized_cuda_lowering},
            executable::{CudaCompiledGraph, CudnnGraphStep, LibraryResource},
            plan::{cublas::CublasLtMatmulGraph, lowering::CudaGraphProvider},
        },
        cudnn_backend::CudnnBackendExecutable,
        graph_binding::{graph_native_dependencies, push_unique_cuda_graph_dependency},
        graph_value::CudaGraphValue,
    },
    error::Result,
    graph::ir::{Graph, ValueId},
};

pub struct GraphFusionCompletion<'a> {
    pub(crate) ir: &'a Graph,
    pub(crate) dependency_ids: &'a [ValueId],
    pub(crate) allocation_dependencies: Vec<GraphNode>,
    pub(crate) output_id: ValueId,
    pub(crate) output: CudaGraphValue,
    pub(crate) realized_nodes: &'a [ValueId],
    pub(crate) skipped_nodes: &'a [ValueId],
    pub(crate) release_nodes: &'a [(ValueId, usize)],
}

impl GraphFusionLowering<'_> {
    fn dependencies(
        &self,
        dependency_ids: &[ValueId],
        allocation_dependencies: Vec<GraphNode>,
    ) -> Vec<GraphNode> {
        let mut dependencies = graph_native_dependencies(dependency_ids, self.native_nodes);
        for dependency in allocation_dependencies {
            push_unique_cuda_graph_dependency(&mut dependencies, dependency);
        }
        dependencies
    }

    pub(crate) fn finish_cudnn_backend_fusion(
        &mut self,
        completion: GraphFusionCompletion<'_>,
        child_graph: NativeCudaGraph,
        executable: CudnnBackendExecutable,
    ) -> Result<()> {
        CudaCompiledGraph::record_cudnn_workspace(self.stats, &executable)?;
        let dependencies = self.dependencies(
            completion.dependency_ids,
            completion.allocation_dependencies,
        );
        let graph_node = self
            .native_graph
            .add_child_graph_node(&dependencies, &child_graph)?;
        self.cudnn_steps.push(CudnnGraphStep::new(
            graph_node.clone(),
            child_graph,
            completion.output.clone(),
            executable,
            true,
        ));
        self.cache
            .insert(completion.output_id, completion.output.clone());
        self.values.insert(completion.output_id, completion.output);
        self.native_nodes
            .insert(completion.output_id, graph_node.clone());
        for &node in completion.realized_nodes {
            record_realized_cuda_lowering(
                self.realized_lowering,
                node,
                completion.ir,
                CudaGraphProvider::CudnnBackend,
            )?;
        }

        self.skipped_nodes
            .extend(completion.skipped_nodes.iter().copied());
        self.stats.record_cudnn_backend_graph_node();
        for &(node, index) in completion.release_nodes {
            self.memory_planner
                .release_dependencies(node, index, self.values, &graph_node);
        }
        Ok(())
    }

    pub(crate) fn finish_cublaslt_fusion(
        &mut self,
        completion: GraphFusionCompletion<'_>,
        cublaslt_graph: CublasLtMatmulGraph,
    ) -> Result<()> {
        self.stats
            .add_cublaslt_workspace_bytes(cublaslt_graph.workspace_byte_size())?;
        let dependencies = self.dependencies(
            completion.dependency_ids,
            completion.allocation_dependencies,
        );
        let graph_node = self
            .native_graph
            .add_child_graph_node(&dependencies, cublaslt_graph.child_graph())?;
        let child_graph = cublaslt_graph.into_native();
        self.library_resources
            .push(LibraryResource::Graph(child_graph));
        self.cache
            .insert(completion.output_id, completion.output.clone());
        self.values.insert(completion.output_id, completion.output);
        self.native_nodes
            .insert(completion.output_id, graph_node.clone());
        for &node in completion.realized_nodes {
            record_realized_cuda_lowering(
                self.realized_lowering,
                node,
                completion.ir,
                CudaGraphProvider::CublasLt,
            )?;
        }

        self.skipped_nodes
            .extend(completion.skipped_nodes.iter().copied());
        self.stats.record_cublas_child_graph_node();
        for &(node, index) in completion.release_nodes {
            self.memory_planner
                .release_dependencies(node, index, self.values, &graph_node);
        }
        Ok(())
    }
}
