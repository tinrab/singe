// A kernel with two results runs once however many of them a graph reads.
// Normalized top-k is one operation with both results; graph capture records each result of latent attention as its own operation, so the first to lower runs the kernel and its sibling, if any, only publishes its result.
// A result nothing reads goes to a buffer the compiled graph keeps.

use singe_cuda::graph::GraphNode;

use super::paged_cudnn::try_lower_cudnn_paged_attention;
use crate::{
    backend::{
        cuda::{
            compiled::{
                build::{
                    lower::GraphOperationLoweringRequest, realized::record_realized_cuda_lowering,
                    state::GraphLoweringState,
                },
                executable::LibraryResource,
                plan::lowering::CudaGraphProvider,
            },
            cudnn_backend::CudnnPaging,
            diagnostics::GraphProviderFailures,
            graph_binding::{
                graph_native_dependencies, operation_dependencies,
                push_unique_cuda_graph_dependency, validate_graph_node_value_tensor,
            },
            graph_value::CudaGraphValue,
            kernel_plan::{self, GraphStep, KernelInput, KernelPlan, KernelPlanner, cudnn_paging},
            tensor_region::CudaTensorSource,
        },
        registry::{Operation, ScaledDotProductAttentionAttributes},
    },
    error::{Error, Result},
    graph::ir::{OperationNode, ValueId},
    tensor::Tensor,
};

// What cuDNN needs to run this paged attention, if the plan allows cuDNN and it takes these options and tensors.
fn paged_cudnn(
    request: &GraphOperationLoweringRequest<'_>,
    operation: &OperationNode,
    metadata: &[KernelInput],
) -> Result<Option<(CudnnPaging, ScaledDotProductAttentionAttributes)>> {
    if !matches!(
        operation.opcode(),
        Some(Operation::PagedAttentionOutput | Operation::PagedAttentionLse)
    ) || !request
        .lowering_plan
        .should_try_provider(request.id, CudaGraphProvider::CudnnBackend)
    {
        return Ok(None);
    }

    let attention = operation
        .attributes()
        .scaled_dot_product_attention()
        .copied()
        .ok_or_else(|| Error::missing_node("paged cudnn"))?;
    Ok(cudnn_paging(metadata, &attention)?.map(|paging| (paging, attention)))
}

// The position of `operation` among its kernel's results.
pub fn result_of(operation: Operation) -> Option<usize> {
    match operation {
        Operation::TopK | Operation::PagedAttentionOutput | Operation::PagedMlaAttentionOutput => {
            Some(0)
        }
        Operation::PagedAttentionLse | Operation::PagedMlaAttentionLse => Some(1),
        _ => None,
    }
}

// The operation of attention's other result.
pub fn attention_sibling(operation: Operation) -> Operation {
    match operation {
        Operation::PagedAttentionOutput => Operation::PagedAttentionLse,
        Operation::PagedAttentionLse => Operation::PagedAttentionOutput,
        Operation::PagedMlaAttentionOutput => Operation::PagedMlaAttentionLse,
        _ => Operation::PagedMlaAttentionOutput,
    }
}

// The value of the other result: top-k's own second result, or the not yet lowered attention node that computes it from the same operands.
fn other_result(request: &GraphOperationLoweringRequest<'_>) -> Result<Option<ValueId>> {
    let ir = request.ir;
    let operation = ir.required_operation_for_value(request.id)?;
    let other = match operation.opcode() {
        Some(Operation::TopK) => return Ok(operation.result_values().nth(1)),
        Some(opcode) => attention_sibling(opcode),
        None => return Ok(None),
    };
    let Some(first) = operation.inputs().first() else {
        return Ok(None);
    };
    let Some(users) = request.users.get(first) else {
        return Ok(None);
    };
    for &user in users {
        let later = request
            .positions
            .get(&user)
            .is_some_and(|&position| position > request.index);
        if !later {
            continue;
        }

        let candidate = ir.required_operation_for_value(user)?;
        if candidate.opcode() == Some(other)
            && candidate.inputs() == operation.inputs()
            && candidate.attributes() == operation.attributes()
        {
            return Ok(Some(user));
        }
    }
    Ok(None)
}

// Plans the two-result kernel of `operation`, which top-k's selection and latent attention each run for both results.
pub fn plan_kernel_result(
    planner: &KernelPlanner,
    operation: &OperationNode,
    inputs: &[KernelInput],
) -> Result<KernelPlan> {
    let attributes = operation.attributes();
    if operation.opcode() == Some(Operation::TopK) {
        let topk = attributes
            .topk()
            .copied()
            .ok_or_else(|| Error::missing_node("plan kernel result"))?;
        let input = inputs
            .first()
            .ok_or_else(|| Error::missing_input("plan kernel result"))?;
        return KernelPlan::topk_select(input, &topk)?.ok_or_else(|| {
            Error::unsupported_operation("cuda graph kernel results of these operands")
        });
    }

    let attention = attributes
        .scaled_dot_product_attention()
        .copied()
        .ok_or_else(|| Error::missing_node("plan kernel result"))?;
    if matches!(
        operation.opcode(),
        Some(Operation::PagedAttentionOutput | Operation::PagedAttentionLse)
    ) {
        planner.paged_attention(inputs, attention)
    } else {
        planner.paged_mla_attention(inputs, attention)
    }
}

fn record(state: &mut GraphLoweringState, id: ValueId, output: CudaGraphValue, node: GraphNode) {
    state.cache.insert(id, output.clone());
    state.values.insert(id, output);
    state.native_nodes.insert(id, node);
}

fn publish(
    request: &GraphOperationLoweringRequest<'_>,
    state: &mut GraphLoweringState,
    output: CudaGraphValue,
    node: &GraphNode,
    provider: CudaGraphProvider,
) -> Result<()> {
    record(state, request.id, output, node.clone());
    record_realized_cuda_lowering(
        &mut state.realized_lowering,
        request.id,
        request.ir,
        provider,
    )?;
    state
        .memory_planner
        .release_dependencies(request.id, request.index, &state.values, node);
    Ok(())
}

pub fn try_lower_kernel_result(
    request: &GraphOperationLoweringRequest<'_>,
    state: &mut GraphLoweringState,
    failures: &mut GraphProviderFailures,
) -> Result<bool> {
    let ir = request.ir;
    let operation = ir.required_operation_for_value(request.id)?;
    let Some(kept) = operation.opcode().and_then(result_of) else {
        return Ok(false);
    };
    if let Some((output, node, provider)) = state.sibling_results.remove(&request.id) {
        publish(request, state, output, &node, provider)?;
        return Ok(true);
    }

    let inputs = operation
        .inputs()
        .iter()
        .map(|input| {
            state
                .cache
                .get(input)
                .cloned()
                .ok_or_else(|| Error::missing_input("try lower kernel result"))
        })
        .collect::<Result<Vec<_>>>()?;
    let metadata = inputs
        .iter()
        .map(KernelInput::from_source)
        .collect::<Vec<_>>();
    let plan = plan_kernel_result(&state.planner, operation, &metadata)?;
    let mut dependencies = graph_native_dependencies(
        &operation_dependencies(ir, request.id)?,
        &state.native_nodes,
    );
    let allocation = state
        .memory_planner
        .allocate(request.context.allocator(), ir, request.id)?;
    for dependency in allocation.dependencies() {
        push_unique_cuda_graph_dependency(&mut dependencies, dependency);
    }

    let output = allocation.into_value();
    validate_graph_node_value_tensor(ir, request.id, &output)?;
    let sibling = other_result(request)?;
    let other = if let Some(sibling) = sibling {
        let allocation = state
            .memory_planner
            .allocate(request.context.allocator(), ir, sibling)?;
        for dependency in allocation.dependencies() {
            push_unique_cuda_graph_dependency(&mut dependencies, dependency);
        }

        let value = allocation.into_value();
        validate_graph_node_value_tensor(ir, sibling, &value)?;
        value
    } else {
        let spec = plan
            .outputs
            .get(1 - kept)
            .ok_or_else(|| Error::missing_output("try lower kernel result"))?;
        let buffer = Tensor::zeroed_on(output.device(), spec.data_type(), spec.shape())?;
        let value = CudaGraphValue::from_tensor(buffer.clone())?;
        state.private.scratch.push(buffer);
        value
    };
    let outputs = if kept == 0 {
        [&output, &other]
    } else {
        [&other, &output]
    };
    let paged = paged_cudnn(request, operation, &metadata)?;
    let library = if let Some((paging, attention)) = &paged {
        try_lower_cudnn_paged_attention(
            request,
            state,
            &inputs,
            &metadata,
            (paging, attention),
            outputs,
            kept,
            &dependencies,
            failures,
        )?
    } else {
        None
    };
    let provider = if library.is_some() {
        CudaGraphProvider::CudnnBackend
    } else {
        CudaGraphProvider::Kernel
    };
    let graph_node = if let Some(node) = library {
        node
    } else {
        let step = GraphStep {
            plan: &plan,
            inputs: inputs
                .iter()
                .map(|input| input as &dyn CudaTensorSource)
                .collect(),
        };
        let child_graph = kernel_plan::build_graph_steps(&[step], &outputs, &mut state.private)?;
        let graph_node = state
            .native_graph
            .add_child_graph_node(&dependencies, &child_graph)?;
        state
            .library_resources
            .push(LibraryResource::Graph(child_graph));
        state.stats.record_kernel_child_graph_node();
        graph_node
    };
    if let Some(sibling) = sibling {
        if operation.opcode() == Some(Operation::TopK) {
            record(state, sibling, other, graph_node.clone());
        } else {
            state
                .sibling_results
                .insert(sibling, (other, graph_node.clone(), provider));
        }
    }

    publish(request, state, output, &graph_node, provider)?;
    Ok(true)
}
