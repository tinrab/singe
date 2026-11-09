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

/// Lowers a planned point-wise, row, or concatenation group as one generated kernel, and returns `false` without changing anything when the kernel cannot be planned.
///
/// A row kernel computes the group's element-wise members as it loads its input, and stores that input too when the plan says something else reads it; a concatenation's kernel stores each of its inputs into its slice.
///
/// # Errors
///
/// Returns an error if a group's operand is missing or the kernel's node cannot be added.
pub fn try_add_elementwise_fusion(
    request: &GraphOperationLoweringRequest<'_>,
    group: &FusionGroup,
    lowering: &mut GraphFusionLowering<'_>,
) -> Result<bool> {
    let ir = request.ir;
    if matches!(
        group.kind,
        GroupKind::Contraction { .. } | GroupKind::Module(_)
    ) {
        return Ok(false);
    }

    let Some(kernel) = plan_group_kernel(lowering.planner, ir, group)? else {
        return Ok(false);
    };
    let (plan, external, produced) = (kernel.plan, kernel.inputs, kernel.outputs);
    let values = external
        .iter()
        .map(|node| {
            lowering
                .cache
                .get(node)
                .ok_or_else(|| Error::missing_input("try add elementwise fusion"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut dependencies = graph_native_dependencies(&external, lowering.native_nodes);
    let mut outputs = Vec::with_capacity(produced.len());
    for &id in &produced {
        let allocation = lowering
            .memory_planner
            .allocate(lowering.context.allocator(), ir, id)?;
        for dependency in allocation.dependencies() {
            push_unique_cuda_graph_dependency(&mut dependencies, dependency);
        }

        let output = allocation.into_value();
        validate_graph_node_value_tensor(ir, id, &output)?;
        outputs.push(output);
    }

    let step = GraphStep {
        plan: &plan,
        inputs: values
            .iter()
            .map(|value| *value as &dyn CudaTensorSource)
            .collect(),
    };
    let child_graph = kernel_plan::build_graph_steps(
        &[step],
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
            .ok_or_else(|| Error::missing_node("try add elementwise fusion"))?;
        lowering
            .memory_planner
            .release_dependencies(id, index, lowering.values, &graph_node);
    }
    Ok(true)
}
