use singe_cuda::graph::GraphNode;

use crate::{
    backend::cuda::{
        compiled::build::{lower::GraphOperationLoweringRequest, state::GraphLoweringState},
        graph_binding::{
            graph_native_dependencies, operation_dependencies, push_unique_cuda_graph_dependency,
            validate_graph_node_value_tensor,
        },
        graph_value::CudaGraphValue,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
};

pub struct GraphLoweringOutput<'a> {
    pub(crate) data_type: DataType,
    pub(crate) shape: &'a Shape,
    pub(crate) tensor: CudaGraphValue,
    pub(crate) dependencies: Vec<GraphNode>,
}

pub fn prepare_cuda_graph_lowering_output<'a>(
    request: &GraphOperationLoweringRequest<'a>,
    state: &mut GraphLoweringState,
) -> Result<GraphLoweringOutput<'a>> {
    let output_value = request.id;
    let data_type = request.ir.required_value_data_type(output_value)?;
    let shape = request.ir.required_value_shape(output_value)?;
    let in_place = state.private.in_place_writes.value(request.id);
    let (tensor, allocation_dependencies) = if let Some(base) = in_place {
        let tensor = state
            .cache
            .get(&base)
            .cloned()
            .ok_or_else(|| Error::missing_input("prepare cuda graph lowering output"))?;
        (tensor, Vec::new())
    } else {
        let allocation =
            state
                .memory_planner
                .allocate(request.context.allocator(), request.ir, request.id)?;
        let dependencies = allocation.dependencies();
        (allocation.into_value(), dependencies)
    };
    validate_graph_node_value_tensor(request.ir, request.id, &tensor)?;
    let dependency_ids = operation_dependencies(request.ir, request.id)?;
    let mut dependencies = graph_native_dependencies(&dependency_ids, &state.native_nodes);
    for dependency in allocation_dependencies {
        push_unique_cuda_graph_dependency(&mut dependencies, dependency);
    }
    Ok(GraphLoweringOutput {
        data_type,
        shape,
        tensor,
        dependencies,
    })
}
