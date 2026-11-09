mod fusion;
mod output;
mod paged_cudnn;
pub mod placement;
mod provider;
pub mod results;
mod special;

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use self::{
    fusion::{try_lower_cublaslt_residual, try_lower_cudnn_fusions, try_lower_fusion_group},
    output::prepare_cuda_graph_lowering_output,
    placement::{NodePlacement, place_by_shared_rules},
    provider::{GraphLoweredProviderOutput, try_lower_provider_output},
    results::try_lower_kernel_result,
    special::{try_lower_device_copy, try_lower_empty_reduction, try_lower_zero_work},
};
use crate::{
    backend::cuda::{
        compiled::{
            build::{
                alias::GraphAliasLowering, prepare::validate_graph_node_support,
                provider::GraphProviderDiagnostics, state::GraphLoweringState,
            },
            plan::lowering::LoweringPlan,
        },
        context::CudaExecutionContext,
        diagnostics::GraphProviderFailures,
        runtime::CudaRuntime,
    },
    error::Result,
    graph::ir::{Graph, ValueId},
};

pub fn lower_cuda_graph_operation(
    request: GraphOperationLoweringRequest<'_>,
    state: &mut GraphLoweringState,
) -> Result<()> {
    let ir = request.ir;
    let id = request.id;
    let index = request.index;
    let lowering_plan = request.lowering_plan;
    let mut provider_failures = GraphProviderFailures::default();
    if place_by_shared_rules(
        &mut GraphPlacement {
            request: &request,
            state: &mut *state,
            provider_failures: &mut provider_failures,
        },
        id,
    )? {
        return Ok(());
    }

    let output_value = id;
    let output_data_type = ir.required_value_data_type(output_value)?;
    let output_shape = ir.required_value_shape(output_value)?;
    let support = validate_graph_node_support(ir, id, lowering_plan, &state.cache)?;
    let operation_support = support.operation_support;
    let provider_diagnostics = GraphProviderDiagnostics {
        operation_name: support.operation_name,
        output_data_type,
        output_shape,
        lowering_candidates: support.lowering_candidates,
        lowering_decision: support.lowering_decision,
        operation_support: operation_support.as_ref(),
        input_data_types: &support.input_data_types,
    };

    if try_lower_cudnn_fusions(&request, state, &mut provider_failures)? {
        return Ok(());
    }

    if try_lower_cublaslt_residual(&request, state, &mut provider_failures)? {
        return Ok(());
    }

    let output = prepare_cuda_graph_lowering_output(&request, state)?;
    if try_lower_device_copy(ir, id, &output.tensor, &output.dependencies, state)? {
        return Ok(());
    }

    if try_lower_zero_work(
        ir,
        id,
        output.shape,
        &output.tensor,
        operation_support,
        lowering_plan,
        state,
    )? {
        return Ok(());
    }

    let graph_node = match try_lower_provider_output(
        &request,
        output.data_type,
        output.shape,
        &output.tensor,
        &output.dependencies,
        &provider_diagnostics,
        &mut provider_failures,
        state,
    )? {
        GraphLoweredProviderOutput::Native(graph_node) => graph_node,
        GraphLoweredProviderOutput::OutputOnly => {
            state.cache.insert(id, output.tensor.clone());
            state.values.insert(id, output.tensor);
            return Ok(());
        }
    };
    if !provider_failures.is_empty() {
        state
            .provider_failure_notes
            .insert(id, provider_failures.clone());
    }

    state.cache.insert(id, output.tensor.clone());
    state.values.insert(id, output.tensor.clone());
    state.native_nodes.insert(id, graph_node.clone());
    state
        .memory_planner
        .release_dependencies(id, index, &state.values, &graph_node);
    Ok(())
}

// The shared placement rules, as a graph's lowering runs them.
struct GraphPlacement<'a, 'b> {
    request: &'a GraphOperationLoweringRequest<'b>,
    state: &'a mut GraphLoweringState,
    provider_failures: &'a mut GraphProviderFailures,
}

impl NodePlacement for GraphPlacement<'_, '_> {
    // A group lowers whole before any check of its first member alone, which may have no lowering of its own, as a dequantization a matmul reads does not.
    fn fusion_group(&mut self, _id: ValueId) -> Result<bool> {
        try_lower_fusion_group(self.request, self.state, self.provider_failures)
    }

    fn kernel_result(&mut self, _id: ValueId) -> Result<bool> {
        try_lower_kernel_result(self.request, self.state, self.provider_failures)
    }

    fn alias(&mut self, id: ValueId) -> Result<bool> {
        let (ir, lowering_plan) = (self.request.ir, self.request.lowering_plan);
        let support = validate_graph_node_support(ir, id, lowering_plan, &self.state.cache)?;
        match self.state.try_add_alias_node(id, ir, lowering_plan)? {
            GraphAliasLowering::Added => Ok(true),
            GraphAliasLowering::NotAlias => Ok(false),
            GraphAliasLowering::Rejected => {
                let diagnostics = GraphProviderDiagnostics {
                    operation_name: support.operation_name,
                    output_data_type: ir.required_value_data_type(id)?,
                    output_shape: ir.required_value_shape(id)?,
                    lowering_candidates: support.lowering_candidates,
                    lowering_decision: support.lowering_decision,
                    operation_support: support.operation_support.as_ref(),
                    input_data_types: &support.input_data_types,
                };
                Err(diagnostics.lowering_failure(&GraphProviderFailures::default()))
            }
        }
    }

    fn empty_reduction(&mut self, id: ValueId) -> Result<bool> {
        try_lower_empty_reduction(self.request.ir, id, self.state)
    }
}

#[derive(Clone, Copy)]
pub struct GraphOperationLoweringRequest<'a> {
    pub(crate) runtime: &'a Arc<CudaRuntime>,
    pub(crate) context: &'a CudaExecutionContext,
    pub(crate) ir: &'a Graph,
    pub(crate) id: ValueId,
    pub(crate) index: usize,
    pub(crate) positions: &'a HashMap<ValueId, usize>,
    pub(crate) users: &'a HashMap<ValueId, Vec<ValueId>>,
    pub(crate) graph_outputs: &'a HashSet<ValueId>,
    pub(crate) lowering_plan: &'a LoweringPlan,
}
