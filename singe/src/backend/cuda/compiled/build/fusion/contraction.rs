use crate::{
    backend::cuda::{
        compiled::{
            build::{
                fusion::GraphFusionLowering, lower::GraphOperationLoweringRequest,
                realized::record_realized_cuda_lowering,
            },
            executable::LibraryResource,
            plan::{
                fusion::{FusionGroup, GroupKind},
                group::plan_group_kernel,
                lowering::CudaGraphProvider,
            },
        },
        graph_binding::{
            graph_native_dependencies, push_unique_cuda_graph_dependency,
            validate_graph_node_value_tensor,
        },
        kernel_plan::{self, GraphStep},
        tensor_region::CudaTensorSource,
    },
    error::{Error, Result},
};

/// Lowers a planned contraction group: matmul `id` as one generated kernel with its dequantized weight `weight` and its epilogue `epilogue`, and returns `false` without changing anything when the kernel cannot be planned.
///
/// # Errors
///
/// Returns an error if a fused operand's value is missing or the fused kernel's node cannot be added.
pub fn try_add_contraction_fusion(
    request: &GraphOperationLoweringRequest<'_>,
    group: &FusionGroup,
    lowering: &mut GraphFusionLowering<'_>,
) -> Result<bool> {
    let ir = request.ir;
    if !matches!(group.kind, GroupKind::Contraction { .. }) {
        return Ok(false);
    }

    let Some(kernel) = plan_group_kernel(lowering.planner, ir, group)? else {
        return Ok(false);
    };
    let (plan, external) = (kernel.plan, kernel.inputs);
    let produced = kernel
        .outputs
        .first()
        .copied()
        .ok_or_else(|| Error::missing_output("try add contraction fusion"))?;
    let values = external
        .iter()
        .map(|node| {
            lowering
                .cache
                .get(node)
                .cloned()
                .ok_or_else(|| Error::missing_input("try add contraction fusion"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut dependencies = graph_native_dependencies(&external, lowering.native_nodes);
    let allocation =
        lowering
            .memory_planner
            .allocate(lowering.context.allocator(), ir, produced)?;
    for dependency in allocation.dependencies() {
        push_unique_cuda_graph_dependency(&mut dependencies, dependency);
    }

    let output = allocation.into_value();
    validate_graph_node_value_tensor(ir, produced, &output)?;
    let step = GraphStep {
        plan: &plan,
        inputs: values
            .iter()
            .map(|value| value as &dyn CudaTensorSource)
            .collect(),
    };
    let child_graph = kernel_plan::build_graph_steps(&[step], &[&output], lowering.private)?;
    let graph_node = lowering
        .native_graph
        .add_child_graph_node(&dependencies, &child_graph)?;
    lowering
        .library_resources
        .push(LibraryResource::Graph(child_graph));
    lowering.cache.insert(produced, output.clone());
    lowering.values.insert(produced, output);
    lowering.native_nodes.insert(produced, graph_node.clone());
    let members = group.members.clone();
    for &member in &members {
        record_realized_cuda_lowering(
            lowering.realized_lowering,
            member,
            ir,
            CudaGraphProvider::Fusion,
        )?;
    }

    lowering.skipped_nodes.extend(members.iter().copied());
    lowering.stats.record_kernel_child_graph_node();
    for &member in &members {
        if let Some(&index) = request.positions.get(&member) {
            lowering.memory_planner.release_dependencies(
                member,
                index,
                lowering.values,
                &graph_node,
            );
        }
    }
    Ok(true)
}
