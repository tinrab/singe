mod alias;
mod attach;
mod entry;
mod finalize;
mod fusion;
mod graph_node;
pub mod lower;
pub mod operation;
pub mod prepare;
mod provider;
mod realized;
mod state;
mod terminal;
mod zero_work;

use std::collections::HashMap;

pub use self::entry::compile_graph;
#[cfg(feature = "onnx")]
pub use self::entry::compile_onnx_model;
#[cfg(test)]
pub use self::provider::has_provider_lowering;
use self::{
    finalize::{GraphFinalizeInput, finalize_cuda_graph_compile},
    graph_node::add_input_write_nodes,
    lower::{GraphOperationLoweringRequest, lower_cuda_graph_operation},
    prepare::prepare_cuda_graph_compile,
    state::GraphLoweringState,
    terminal::try_add_cuda_graph_terminal_node,
};

use crate::{
    backend::{
        cuda::{
            self,
            codegen::{KernelDestination, PlanningContext},
            compiled::{
                executable::CudaCompiledGraph,
                plan::{
                    fusion::plan_fusion,
                    kernel::{NodeKernelPlans, node_kernel_plans},
                    lowering::CudaGraphProvider,
                },
            },
            graph_binding::{
                InPlaceWrites, WritableInputs, graph_node_positions, graph_users,
                validate_graph_input_value_tensor,
            },
            kernel_plan::KernelPlanner,
            nvtx::CudaNvtxRange,
            tuning::TunedWay,
        },
        registry::Operation,
    },
    error::Result,
    graph::{CompiledGraph, GraphCudaOptions},
};

/// Compiles a logical graph into a replayable CUDA graph with the providers and shape policy of `options`.
///
/// When `options` holds tuning records, they choose among what the graph can run; asks for kernel tuning, generated kernels with no recorded schedule are tuned as they are planned.
/// Replays read the runtime inputs `options` binds in place where their tensors are, when those tensors fit the graph's input metadata.
///
/// # Errors
///
/// Returns an error if a provider, allocation, or native graph construction fails.
pub fn compile(compiled: &CompiledGraph, options: &GraphCudaOptions) -> Result<CudaCompiledGraph> {
    let _compile_range = CudaNvtxRange::create("singe cuda graph compile");
    let prepared = {
        let _prepare_range = CudaNvtxRange::create("singe cuda graph compile prepare");
        prepare_cuda_graph_compile(
            compiled,
            &options.shapes,
            options.lowering.clone(),
            |device| {
                let target = options.target_architecture;
                let context = PlanningContext {
                    language: options.kernel_language,
                    records: options.tuning_records.clone(),
                    recorder: options.kernel_recorder.clone(),
                    // A record may choose a generated convolution over cuDNN.
                    generated_convolutions: !options
                        .lowering
                        .allows(CudaGraphProvider::CudnnBackend, Operation::Conv2d)
                        || options
                            .tuning_records
                            .as_ref()
                            .is_some_and(|records| records.chooses(TunedWay::Generated)),
                    destination: KernelDestination::Singe,
                    target: Some(match target {
                        Some(target) => target,
                        None => cuda::execution_context(device)?.runtime().architecture(),
                    }),
                };
                context.check_records()?;
                Ok(context)
            },
        )?
    };
    let execution_order = prepared.execution_order;
    let lowering_plan = prepared.lowering_plan;
    let ir = prepared.ir;
    let device = prepared.device;
    let guards = prepared.guards;
    let cache_key = prepared.cache_key;
    let context = cuda::execution_context(device)?;
    context.bind()?;
    let runtime = context.runtime();
    let planner = KernelPlanner::new(prepared.planning);
    let mut reusable_plans = prepared.kernels.take_plans()?;
    let users = graph_users(&ir, &execution_order)?;
    let graph_outputs = ir.output_value_set();
    let fusion_plan = {
        let _plan_range = CudaNvtxRange::create("singe cuda graph compile plan fusion");
        plan_fusion(
            &ir,
            &execution_order,
            &users,
            &graph_outputs,
            &lowering_plan,
            None,
            planner.context().records(),
        )?
    };
    // Lowering visits each fusion group's members together.
    let execution_order = fusion_plan.order().to_vec();
    let node_plans;
    {
        let _compile_kernels_range = CudaNvtxRange::create("singe cuda graph compile kernels");
        // Nodes fusion may take are planned and compiled as they lower, so a kernel a fused group replaces is never compiled or tuned.
        node_plans = execution_order
            .iter()
            .filter(|&&id| {
                lowering_plan.should_try_provider(id, CudaGraphProvider::Kernel)
                    && !lowering_plan.should_try_provider(id, CudaGraphProvider::Fusion)
                    && !fusion_plan.is_planned(id)
            })
            .filter_map(|&id| {
                let plans = reusable_plans
                    .remove(&id)
                    .or_else(|| node_kernel_plans(&planner, &ir, id))?;
                Some((id, plans))
            })
            .collect::<HashMap<_, _>>();
        runtime.prepare_kernels(node_plans.values().flat_map(NodeKernelPlans::artifacts))?;
    }

    let native_graph = runtime.context().create_graph()?;
    let positions = graph_node_positions(&execution_order);
    let mut lowering_state =
        GraphLoweringState::create(&ir, &execution_order, native_graph, planner)?;
    lowering_state
        .memory_planner
        .set_arena(options.arena.clone());
    lowering_state.fusion_plan = fusion_plan;
    lowering_state.node_plans = node_plans;
    // A tensor that does not fit this specialization's input is copied, as unbound inputs are.
    for (id, name) in ir.input_values() {
        if let Some(tensor) = options.inputs_in_place.get(name)
            && !ir.required_value_binding_details(id)?.is_static_input()
            && validate_graph_input_value_tensor(&ir, id, tensor).is_ok()
        {
            lowering_state
                .private
                .inputs_in_place
                .insert(id, tensor.clone());
        }
    }

    lowering_state
        .private
        .library_recorder
        .clone_from(&options.library_recorder);
    lowering_state.private.in_place_writes = InPlaceWrites::analyze(
        &ir,
        &execution_order,
        &users,
        &graph_outputs,
        WritableInputs::Bound(&lowering_state.private.inputs_in_place),
    )?;

    {
        let _lower_range = CudaNvtxRange::create("singe cuda graph compile lower");
        for (index, &id) in execution_order.iter().enumerate() {
            if lowering_state.skipped_nodes.contains(&id) {
                continue;
            }

            if try_add_cuda_graph_terminal_node(&context, id, &ir, &mut lowering_state)? {
                continue;
            }

            lower_cuda_graph_operation(
                GraphOperationLoweringRequest {
                    runtime,
                    context: &context,
                    ir: &ir,
                    id,
                    index,
                    positions: &positions,
                    users: &users,
                    graph_outputs: &graph_outputs,
                    lowering_plan: &lowering_plan,
                },
                &mut lowering_state,
            )
            .map_err(|error| {
                let operation = ir.required_operation_for_value(id).ok();
                error.with_graph_source(operation.and_then(|operation| operation.source_location()))
            })?;
        }
    }

    add_input_write_nodes(&ir, &mut lowering_state)?;
    let has_native_nodes = !lowering_state.native_nodes.is_empty();
    finalize_cuda_graph_compile(GraphFinalizeInput {
        ir,
        cache_key,
        device,
        guards,
        execution_order,
        lowering_plan,
        realized_lowering: lowering_state.realized_lowering,
        provider_failure_notes: lowering_state.provider_failure_notes,
        stats: lowering_state.stats,
        runtime,
        native_graph: lowering_state.native_graph,
        has_native_nodes,
        cudnn_steps: lowering_state.cudnn_steps,
        cutensor_steps: lowering_state.cutensor_steps,
        library_resources: lowering_state.library_resources,
        values: lowering_state.values,
        memory_planner: lowering_state.memory_planner,
        private: lowering_state.private,
        fusion_plan: lowering_state.fusion_plan,
    })
}
