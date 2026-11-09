use crate::graph::test_support::*;

#[test]
fn test_remove_redundant_ops_pass_removes_full_range_slice_ops() {
    let shape = Shape::create(vec![2, 3]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let slice = graph
        .add_named_slice_operation(
            None,
            input,
            1,
            0,
            3,
            output.data_type(),
            output.shape(),
            output.device(),
        )
        .unwrap();
    graph.set_outputs(vec![slice]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(
        results[0].remapped_nodes,
        [(input, ValueId::new(0)), (slice, ValueId::new(0)),],
    );
    assert_output_nodes(&normalized, &[ValueId::new(0)]);

    let compiled = normalized.compile().unwrap();
    let values = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let outputs = compiled
        .run(&HashMap::from([(
            "x".to_string(),
            Tensor::from_vec(Device::Cpu, &shape, values.clone()).unwrap(),
        )]))
        .unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &values);
}

#[test]
fn test_remove_redundant_ops_pass_rejects_invalid_full_range_slice_axis() {
    let input_shape = Shape::create(vec![2, 3]).unwrap();
    let output_shape = Shape::create(vec![2, 3]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let result = graph.add_named_slice_operation(
        None,
        input,
        2,
        0,
        3,
        output.data_type(),
        output.shape(),
        output.device(),
    );
    assert!(matches!(result, Err(Error::DimensionOutOfBounds)));
}

#[test]
fn test_remove_redundant_ops_pass_rejects_overflowed_slice_chain_start() {
    let input_shape = Shape::create(vec![2, 8, 3]).unwrap();
    let input_output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &input_shape).unwrap();
    let first_slice_output = input_output.narrow(1, 1, 6).unwrap();
    let second_slice_output = first_slice_output.narrow(1, 1, 2).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let first_slice = graph
        .add_named_slice_operation(
            None,
            input,
            1,
            1,
            6,
            first_slice_output.data_type(),
            first_slice_output.shape(),
            first_slice_output.device(),
        )
        .unwrap();
    let second_slice = graph
        .add_named_slice_operation(
            None,
            first_slice,
            1,
            1,
            2,
            second_slice_output.data_type(),
            second_slice_output.shape(),
            second_slice_output.device(),
        )
        .unwrap();
    graph.replace_node_slice_for_test(first_slice, input, 1, i64::MAX, 6);
    graph.set_outputs(vec![second_slice]).unwrap();

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

#[test]
fn test_remove_redundant_ops_pass_rejects_out_of_range_parent_slice_chain_window() {
    let input_shape = Shape::create(vec![2, 8, 3]).unwrap();
    let input_output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &input_shape).unwrap();
    let first_slice_output = input_output.narrow(1, 1, 6).unwrap();
    let second_slice_output = first_slice_output.narrow(1, 1, 2).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let first_slice = graph
        .add_named_slice_operation(
            None,
            input,
            1,
            1,
            6,
            first_slice_output.data_type(),
            first_slice_output.shape(),
            first_slice_output.device(),
        )
        .unwrap();
    let second_slice = graph
        .add_named_slice_operation(
            None,
            first_slice,
            1,
            1,
            2,
            second_slice_output.data_type(),
            second_slice_output.shape(),
            second_slice_output.device(),
        )
        .unwrap();
    graph.replace_node_slice_for_test(first_slice, input, 1, 7, 6);
    graph.set_outputs(vec![second_slice]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    assert!(matches!(
        pipeline.normalize(&graph),
        Err(Error::IndexOutOfBounds)
    ));
}

#[test]
fn test_remove_redundant_ops_pass_rejects_out_of_range_child_slice_chain_window() {
    let input_shape = Shape::create(vec![2, 8, 3]).unwrap();
    let input_output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &input_shape).unwrap();
    let first_slice_output = input_output.narrow(1, 1, 6).unwrap();
    let second_slice_output = first_slice_output.narrow(1, 1, 2).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let first_slice = graph
        .add_named_slice_operation(
            None,
            input,
            1,
            1,
            6,
            first_slice_output.data_type(),
            first_slice_output.shape(),
            first_slice_output.device(),
        )
        .unwrap();
    let second_slice = graph
        .add_named_slice_operation(
            None,
            first_slice,
            1,
            1,
            2,
            second_slice_output.data_type(),
            second_slice_output.shape(),
            second_slice_output.device(),
        )
        .unwrap();
    graph.replace_node_slice_for_test(second_slice, first_slice, 1, 5, 2);
    graph.set_outputs(vec![second_slice]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    assert!(matches!(
        pipeline.normalize(&graph),
        Err(Error::IndexOutOfBounds)
    ));
}
