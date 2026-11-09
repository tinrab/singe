use super::*;

fn assert_metadata(
    graph: &Graph,
    nodes: impl IntoIterator<Item = (ValueId, DataType)>,
    shape: &Shape,
) {
    for (node, data_type) in nodes {
        assert_value_metadata(
            graph,
            node,
            data_type,
            shape.dimensions(),
            shape.dimensions(),
            Device::Cpu,
        );
    }
}

#[test]
fn test_infer_value_metadata_pass_infers_reduction_shape() {
    let input_shape = Shape::create(vec![2, 3, 4]).unwrap();
    let output_shape = Shape::create(vec![2, 1, 4]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let sum = add_reduction_operation(
        &mut graph,
        RegistryOperation::Sum,
        input,
        vec![1],
        true,
        &output,
    );
    replace_with_stale_shape_metadata(&mut graph, sum);
    graph.set_outputs(vec![sum]).unwrap();

    let normalized = infer_and_validate_value_metadata(&graph);
    assert_metadata(&normalized, [(sum, DataType::F32)], &output_shape);
}

#[test]
fn test_infer_value_metadata_pass_infers_index_reduction_shape() {
    let input_shape = Shape::create(vec![2, 3, 4]).unwrap();
    let output_shape = Shape::create(vec![2, 4]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::I64, &output_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let argmax = add_dimension_operation(&mut graph, RegistryOperation::ArgMax, input, 1, &output);
    replace_with_stale_shape_metadata(&mut graph, argmax);
    graph.set_outputs(vec![argmax]).unwrap();

    let normalized = infer_and_validate_value_metadata(&graph);
    assert_metadata(&normalized, [(argmax, DataType::I64)], &output_shape);
}

#[test]
fn test_infer_value_metadata_pass_infers_topk_shapes() {
    let input_shape = Shape::create(vec![2, 3, 4]).unwrap();
    let output_shape = Shape::create(vec![2, 2, 4]).unwrap();
    let values_output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let indices_output = Tensor::zeroed_on(Device::Cpu, DataType::I64, &output_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let values = add_topk_operation(
        &mut graph,
        RegistryOperation::TopKValues,
        input,
        2,
        -2,
        true,
        true,
        &values_output,
    );
    let indices = add_topk_operation(
        &mut graph,
        RegistryOperation::TopKIndices,
        input,
        2,
        -2,
        true,
        true,
        &indices_output,
    );
    for node in [values, indices] {
        replace_with_stale_shape_metadata(&mut graph, node);
    }
    graph.replace_node_output_data_type_for_test(indices, DataType::F32);
    graph.set_outputs(vec![values, indices]).unwrap();

    let normalized = infer_and_validate_value_metadata(&graph);
    assert_metadata(
        &normalized,
        [(values, DataType::F32), (indices, DataType::I64)],
        &output_shape,
    );
}
