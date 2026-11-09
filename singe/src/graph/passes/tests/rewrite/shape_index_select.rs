use crate::graph::{node::NodeKind, test_support::*};

#[test]
fn test_remove_redundant_ops_pass_removes_identity_index_select_ops() {
    let input_shape = Shape::create(vec![2, 3]).unwrap();
    let indices_shape = Shape::create(vec![3]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &input_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape.clone(), Device::Cpu);
    let indices = graph
        .add_constant(Tensor::from_vec(Device::Cpu, &indices_shape, vec![0_i64, 1, 2]).unwrap());
    let select = graph
        .add_named_operation(
            None,
            NodeKind::index_select(input, -1, indices),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.set_outputs(vec![select]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(
        results[0].remapped_nodes,
        [
            (input, ValueId::new(0)),
            (indices, ValueId::new(1)),
            (select, ValueId::new(0)),
        ],
    );
    assert_output_nodes(&normalized, &[ValueId::new(0)]);

    let values = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let compiled = normalized.compile().unwrap();
    let outputs = compiled
        .run(&HashMap::from([(
            "x".to_string(),
            Tensor::from_vec(Device::Cpu, &input_shape, values.clone()).unwrap(),
        )]))
        .unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &values);
}

#[test]
fn test_remove_redundant_ops_pass_rejects_invalid_identity_index_select_axis() {
    let input_shape = Shape::create(vec![2, 3]).unwrap();
    let indices_shape = Shape::create(vec![3]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &input_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let indices = graph
        .add_constant(Tensor::from_vec(Device::Cpu, &indices_shape, vec![0_i64, 1, 2]).unwrap());
    let select = add_operation_fixture(
        &mut graph,
        Operation::IndexSelect,
        OperationAttributes::IndexSelect(2),
        vec![input, indices],
        &output,
    );
    graph.set_outputs(vec![select]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    assert!(matches!(
        pipeline.normalize(&graph),
        Err(Error::InvalidShape
            | Error::DimensionOutOfBounds
            | Error::InvalidDimension { .. }
            | Error::InvalidArgument { .. }
            | Error::OutOfRange { .. }
            | Error::LengthMismatch { .. }
            | Error::InvalidTensorRank { .. })
    ));
}
