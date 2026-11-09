use crate::{backend::registry::BackendExecution, graph::ir::Graph};

pub fn assign_cuda_schema_provider_partitions(ir: &mut Graph) {
    let providers = ir
        .operations()
        .iter()
        .map(|operation| {
            let schema = operation.schema()?;
            if !schema.support().supports(BackendExecution::CudaGraph) {
                return None;
            }
            schema.support().cuda_graph_providers().first().copied()
        })
        .collect::<Vec<_>>();
    ir.assign_provider_partitions_from_provider_list(&providers);
}
