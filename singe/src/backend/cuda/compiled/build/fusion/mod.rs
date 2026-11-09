mod contraction;
mod conv2d;
mod elementwise;
mod finish;
mod matmul;
mod module;

use std::collections::{HashMap, HashSet};

use singe_cuda::graph::{Graph as NativeCudaGraph, GraphNode};

pub use self::{
    contraction::try_add_contraction_fusion,
    conv2d::try_add_cudnn_conv2d_bias_relu_fusion,
    elementwise::try_add_elementwise_fusion,
    finish::GraphFusionCompletion,
    matmul::{
        residual_add, try_add_cudnn_matmul_fusion, try_cublaslt_matmul_activation_fusion,
        try_cublaslt_matmul_residual_fusion,
    },
    module::try_add_module_fusion,
};
use crate::{
    backend::cuda::{
        compiled::{
            executable::{
                CudaCompiledGraphStats, CudnnGraphStep, GraphPrivateBuffers, LibraryResource,
            },
            plan::{lowering::RealizedLoweringPlan, memory::GraphMemoryPlanner},
        },
        context::CudaExecutionContext,
        diagnostics::GraphProviderFailures,
        graph_value::CudaGraphValue,
        kernel_plan::KernelPlanner,
    },
    graph::ir::ValueId,
};

pub struct GraphFusionLowering<'a> {
    pub(crate) context: &'a CudaExecutionContext,
    pub(crate) native_graph: &'a mut NativeCudaGraph,
    pub(crate) cache: &'a mut HashMap<ValueId, CudaGraphValue>,
    pub(crate) values: &'a mut HashMap<ValueId, CudaGraphValue>,
    pub(crate) native_nodes: &'a mut HashMap<ValueId, GraphNode>,
    pub(crate) skipped_nodes: &'a mut HashSet<ValueId>,
    pub(crate) realized_lowering: &'a mut RealizedLoweringPlan,
    pub(crate) memory_planner: &'a mut GraphMemoryPlanner,
    pub(crate) cudnn_steps: &'a mut Vec<CudnnGraphStep>,
    pub(crate) library_resources: &'a mut Vec<LibraryResource>,
    pub(crate) stats: &'a mut CudaCompiledGraphStats,
    pub(crate) provider_failures: &'a mut GraphProviderFailures,
    pub(crate) planner: &'a KernelPlanner,
    pub(crate) private: &'a mut GraphPrivateBuffers,
}
