mod cudnn;
mod cutensor;
mod dispatch;
mod fft;
mod kernel;
mod matmul;

use std::{collections::HashMap, sync::Arc};

#[cfg(test)]
pub use self::dispatch::has_provider_lowering;
use singe_cuda::graph::{Graph as NativeCudaGraph, GraphNode};

pub use self::dispatch::try_add_operation_provider_node;
use crate::{
    backend::{
        cuda::{
            compiled::{
                executable::{
                    CudaCompiledGraphStats, CudnnGraphStep, CutensorGraphStep, GraphPrivateBuffers,
                    LibraryResource,
                },
                plan::{
                    kernel::NodeKernelPlans,
                    lowering::{
                        CudaGraphProvider, LoweringCandidate, LoweringDecisionRecord, LoweringPlan,
                        RealizedLoweringPlan,
                    },
                },
            },
            diagnostics::{
                GraphProviderFailures, describe_cuda_graph_lowering_failure,
                record_cuda_graph_provider_failure,
            },
            graph_value::CudaGraphValue,
            kernel_plan::KernelPlanner,
            runtime::CudaRuntime,
            tuning::{ForcedLibrary, LibraryRecorder},
        },
        registry::OperationSupport,
    },
    data_type::DataType,
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    shape::Shape,
};

pub enum GraphProviderNode {
    Native(GraphNode),
    OutputOnly,
}

pub struct GraphProviderLowering<'a> {
    pub(crate) native_graph: &'a mut NativeCudaGraph,
    pub(crate) stats: &'a mut CudaCompiledGraphStats,
    pub(crate) realized_lowering: &'a mut RealizedLoweringPlan,
    pub(crate) cudnn_steps: &'a mut Vec<CudnnGraphStep>,
    pub(crate) library_resources: &'a mut Vec<LibraryResource>,
    pub(crate) cutensor_steps: &'a mut Vec<CutensorGraphStep>,
    pub(crate) provider_failures: &'a mut GraphProviderFailures,
    pub(crate) private: &'a mut GraphPrivateBuffers,
    pub(crate) planner: &'a KernelPlanner,
    // Each node's kernels, planned before lowering so they compile together; lowering takes them as it reaches the node.
    pub(crate) node_plans: &'a mut HashMap<ValueId, NodeKernelPlans>,
}

pub struct GraphProviderDiagnostics<'a> {
    pub(crate) operation_name: &'static str,
    pub(crate) output_data_type: DataType,
    pub(crate) output_shape: &'a Shape,
    pub(crate) lowering_candidates: &'a [LoweringCandidate],
    pub(crate) lowering_decision: Option<&'a LoweringDecisionRecord>,
    pub(crate) operation_support: Option<&'a OperationSupport>,
    pub(crate) input_data_types: &'a [DataType],
}

#[derive(Clone, Copy)]
pub struct GraphProviderRequest<'a> {
    pub(crate) runtime: &'a Arc<CudaRuntime>,
    pub(crate) ir: &'a Graph,
    pub(crate) id: ValueId,
    pub(crate) output_data_type: DataType,
    pub(crate) output_shape: &'a Shape,
    pub(crate) output: &'a CudaGraphValue,
    pub(crate) cache: &'a HashMap<ValueId, CudaGraphValue>,
    pub(crate) dependencies: &'a [GraphNode],
    pub(crate) lowering_plan: &'a LoweringPlan,
    pub(crate) diagnostics: &'a GraphProviderDiagnostics<'a>,
    // What records say a library should run this node with.
    pub(crate) forced: Option<&'a ForcedLibrary>,
    // Where to list what libraries could run this node with, for tuning.
    pub(crate) library_recorder: Option<&'a LibraryRecorder>,
}

// Runs `build` when the plan allows `provider`, and `attach` on what it made.
// A failed build records why, so the next provider gets its turn; a failed attach is an error.
pub fn try_provider<T>(
    request: &GraphProviderRequest<'_>,
    lowering: &mut GraphProviderLowering<'_>,
    provider: CudaGraphProvider,
    build: impl FnOnce(&mut GraphProviderLowering<'_>) -> Result<T>,
    attach: impl FnOnce(T, &mut GraphProviderLowering<'_>) -> Result<GraphNode>,
) -> Result<Option<GraphNode>> {
    if !request
        .lowering_plan
        .should_try_provider(request.id, provider)
    {
        return Ok(None);
    }
    match build(lowering) {
        Ok(value) => attach(value, lowering).map(Some),
        Err(error) => {
            record_cuda_graph_provider_failure(lowering.provider_failures, provider, error)?;
            Ok(None)
        }
    }
}

impl GraphProviderDiagnostics<'_> {
    pub(crate) fn lowering_failure(&self, provider_failures: &GraphProviderFailures) -> Error {
        describe_cuda_graph_lowering_failure(
            self.operation_name,
            self.output_data_type,
            self.output_shape,
            self.lowering_candidates,
            self.lowering_decision,
            self.operation_support,
            self.input_data_types,
            provider_failures.as_slice(),
        )
    }
}
