use crate::graph::{node::NodeKind, test_support::*};

#[test]
fn test_remove_redundant_ops_pass_removes_single_input_concat_ops() {
    let shape = Shape::create(vec![2, 2]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape, Device::Cpu);
    let concat = graph
        .add_named_operation(
            None,
            NodeKind::concat(vec![input], 1),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.set_outputs(vec![concat]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(
        results[0].remapped_nodes,
        [(input, ValueId::new(0)), (concat, ValueId::new(0))],
    );
    assert_output_nodes(&normalized, &[ValueId::new(0)]);
}
