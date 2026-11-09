use super::*;
use crate::{graph::outputs::GraphOutput, layout::Layout};

pub fn assert_output_nodes(graph: &Graph, nodes: &[ValueId]) {
    let outputs = graph
        .output_projections()
        .iter()
        .map(GraphOutput::node)
        .collect::<Vec<_>>();
    assert_eq!(outputs, nodes);
}

pub fn value_count(graph: &Graph) -> usize {
    let ir = graph.to_ir().unwrap();
    ir.values().len()
}

pub fn assert_value_count(graph: &Graph, count: usize) {
    assert_eq!(value_count(graph), count);
}

pub fn replace_with_stale_shape_metadata(graph: &mut Graph, node: ValueId) {
    let stale_shape = Shape::create(vec![99]).unwrap();
    graph.replace_node_output_layout_for_test(node, Layout::from_shape(&stale_shape));
}

pub fn remove_redundant_ops_pipeline() -> GraphPassPipeline {
    GraphPassPipeline::new(vec![GraphPassKind::RemoveRedundantOps])
}

pub fn validating_remove_redundant_ops_pipeline() -> GraphPassPipeline {
    GraphPassPipeline::new(vec![
        GraphPassKind::RemoveRedundantOps,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ])
}

/// Returns the node `node` was remapped to.
pub fn remapped(remap: &[(ValueId, ValueId)], node: ValueId) -> ValueId {
    remap
        .iter()
        .find(|(old, _)| *old == node)
        .map(|(_, new)| *new)
        .unwrap()
}

/// Checks that nodes sharing a class were merged into one node and nodes in different classes were not.
pub fn assert_remap_classes(remap: &[(ValueId, ValueId)], classes: &[(ValueId, usize)]) {
    for (node, class) in classes {
        for (other, other_class) in classes {
            assert_eq!(
                remapped(remap, *node) == remapped(remap, *other),
                class == other_class,
                "nodes {node:?} and {other:?}"
            );
        }
    }
}
