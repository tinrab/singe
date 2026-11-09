mod batch;
mod standard;

use super::*;

fn fold_and_assert_checked_nodes(graph: &Graph, expected_checked_nodes: usize) -> Graph {
    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::FoldSmallConstants,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (folded, results) = pipeline.normalize(graph).unwrap();

    assert_eq!(results[0].pass_name, "fold_small_constants");
    assert_eq!(results[0].checked_nodes, expected_checked_nodes);

    folded
}
