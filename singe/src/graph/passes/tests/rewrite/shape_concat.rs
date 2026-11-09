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

#[test]
fn test_remove_redundant_ops_pass_rejects_empty_nested_concat_chains() {
    let input_shape = Shape::create(vec![2, 1]).unwrap();
    let first_output_shape = Shape::create(vec![2, 2]).unwrap();
    let final_output_shape = Shape::create(vec![2, 3]).unwrap();
    let first_output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &first_output_shape).unwrap();
    let final_output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &final_output_shape).unwrap();
    let mut graph = Graph::new();
    let first = graph.add_input("first", DataType::F32, input_shape.clone(), Device::Cpu);
    let second = graph.add_input("second", DataType::F32, input_shape.clone(), Device::Cpu);
    let third = graph.add_input("third", DataType::F32, input_shape, Device::Cpu);
    let inner = graph
        .add_named_operation(
            None,
            NodeKind::concat(vec![first, second], 1),
            first_output.data_type(),
            first_output.shape().clone(),
            first_output.device(),
        )
        .unwrap();
    let outer = graph
        .add_named_operation(
            None,
            NodeKind::concat(vec![inner, third], 1),
            final_output.data_type(),
            final_output.shape().clone(),
            final_output.device(),
        )
        .unwrap();
    graph.replace_node_concat_for_test(inner, Vec::new(), 1);
    graph.set_outputs(vec![outer]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    assert!(matches!(
        pipeline.normalize(&graph),
        Err(Error::InvalidShape
            | Error::InvalidDimension { .. }
            | Error::InvalidArgument { .. }
            | Error::OutOfRange { .. }
            | Error::LengthMismatch { .. }
            | Error::InvalidTensorRank { .. })
    ));
}
