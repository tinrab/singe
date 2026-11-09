mod arithmetic;
mod expressions;
mod mixed;

use super::*;

fn fold_small_constants(graph: &Graph, checked_nodes: usize) -> Graph {
    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::FoldSmallConstants,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (folded, results) = pipeline.normalize(graph).unwrap();

    assert_eq!(results[0].pass_name, "fold_small_constants");
    assert_eq!(results[0].checked_nodes, checked_nodes);

    folded
}

fn assert_folded_constants(
    graph: &Graph,
    expected: impl IntoIterator<Item = (ValueId, &'static [f32])>,
) {
    for (node, values) in expected {
        let tensor = folded_constant_tensor(graph, node);
        assert_close!(&tensor.to_vec::<f32>().unwrap(), values);
    }
}

fn add_quantization(
    graph: &mut Graph,
    operation: RegistryOperation,
    input: ValueId,
    output: &Tensor,
) -> ValueId {
    graph
        .add_named_quantization_operation(
            None,
            operation,
            input,
            0.5,
            0,
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap()
}
