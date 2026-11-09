use crate::{
    backend::{
        cuda::{
            compiled::{
                build::{
                    fusion::GraphFusionLowering, lower::GraphOperationLoweringRequest,
                    realized::record_realized_cuda_lowering,
                },
                executable::LibraryResource,
                plan::{
                    fusion::FusionGroup, group::plan_group_kernel, lowering::CudaGraphProvider,
                },
            },
            graph_binding::{
                graph_native_dependencies, push_unique_cuda_graph_dependency,
                validate_graph_node_value_tensor,
            },
            kernel_plan::{self, GraphStep, KernelInput},
            tensor_region::CudaTensorSource,
        },
        registry::Operation,
    },
    error::{Error, Result},
};

// Lowers a recognized module as its kernel.
// A cache write writes its cache where it is when nothing else needs the old contents, as a lone `index_put` does, and otherwise a copy made first.
//
// # Errors
//
// Returns an error if a module's operand is missing or its kernel's node cannot be added.
pub fn try_add_module_fusion(
    request: &GraphOperationLoweringRequest<'_>,
    group: &FusionGroup,
    lowering: &mut GraphFusionLowering<'_>,
) -> Result<bool> {
    let ir = request.ir;
    let Some(kernel) = plan_group_kernel(lowering.planner, ir, group)? else {
        return Ok(false);
    };
    let (plan, external, produced) = (kernel.plan, kernel.inputs, kernel.outputs);
    let mut dependencies = graph_native_dependencies(&external, lowering.native_nodes);
    let mut outputs = Vec::with_capacity(produced.len());
    for &id in &produced {
        let operation = ir.required_operation_for_value(id)?;
        let base = (operation.opcode() == Some(Operation::IndexPut)).then(|| operation.inputs()[0]);
        let in_place = base.and_then(|_| lowering.private.in_place_writes.value(id));
        if let Some(base) = in_place {
            let tensor = lowering
                .cache
                .get(&base)
                .cloned()
                .ok_or_else(|| Error::missing_input("try add module fusion"))?;
            for node in graph_native_dependencies(&[base], lowering.native_nodes) {
                push_unique_cuda_graph_dependency(&mut dependencies, node);
            }

            validate_graph_node_value_tensor(ir, id, &tensor)?;
            outputs.push(tensor);
            continue;
        }

        let allocation = lowering
            .memory_planner
            .allocate(lowering.context.allocator(), ir, id)?;
        let allocated = allocation.dependencies();
        let output = allocation.into_value();
        validate_graph_node_value_tensor(ir, id, &output)?;
        for dependency in &allocated {
            push_unique_cuda_graph_dependency(&mut dependencies, dependency.clone());
        }

        if let Some(base) = base {
            let source = lowering
                .cache
                .get(&base)
                .ok_or_else(|| Error::missing_input("try add module fusion"))?;
            let copy = lowering.planner.assign(
                &KernelInput::from_source(&output),
                &KernelInput::from_source(source),
            )?;
            let child_graph = kernel_plan::build_graph_steps(
                &[GraphStep {
                    plan: &copy,
                    inputs: vec![source as &dyn CudaTensorSource],
                }],
                &[&output],
                lowering.private,
            )?;
            let mut before = graph_native_dependencies(&[base], lowering.native_nodes);
            for dependency in allocated {
                push_unique_cuda_graph_dependency(&mut before, dependency);
            }

            let node = lowering
                .native_graph
                .add_child_graph_node(&before, &child_graph)?;
            lowering
                .library_resources
                .push(LibraryResource::Graph(child_graph));
            lowering.stats.record_kernel_child_graph_node();
            push_unique_cuda_graph_dependency(&mut dependencies, node);
        }

        outputs.push(output);
    }

    let values = external
        .iter()
        .map(|node| {
            lowering
                .cache
                .get(node)
                .map(|value| value as &dyn CudaTensorSource)
                .ok_or_else(|| Error::missing_input("try add module fusion"))
        })
        .collect::<Result<Vec<_>>>()?;
    let child_graph = kernel_plan::build_graph_steps(
        &[GraphStep {
            plan: &plan,
            inputs: values,
        }],
        &outputs.iter().collect::<Vec<_>>(),
        lowering.private,
    )?;
    let graph_node = lowering
        .native_graph
        .add_child_graph_node(&dependencies, &child_graph)?;
    lowering
        .library_resources
        .push(LibraryResource::Graph(child_graph));
    for (id, output) in produced.into_iter().zip(outputs) {
        lowering.cache.insert(id, output.clone());
        lowering.values.insert(id, output);
        lowering.native_nodes.insert(id, graph_node.clone());
    }

    let members = group.members.iter().copied();
    for id in members.clone() {
        record_realized_cuda_lowering(
            lowering.realized_lowering,
            id,
            ir,
            CudaGraphProvider::Fusion,
        )?;
    }

    lowering.skipped_nodes.extend(members.clone());
    lowering.stats.record_kernel_child_graph_node();
    for id in members {
        let index = *request
            .positions
            .get(&id)
            .ok_or_else(|| Error::missing_node("try add module fusion"))?;
        lowering
            .memory_planner
            .release_dependencies(id, index, lowering.values, &graph_node);
    }
    Ok(true)
}
