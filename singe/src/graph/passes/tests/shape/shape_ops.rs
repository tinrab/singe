use super::*;

use crate::graph::node::NodeKind;

#[test]
fn test_infer_value_metadata_pass_infers_concat_shape() {
    let lhs_shape = Shape::create(vec![2, 3]).unwrap();
    let rhs_shape = Shape::create(vec![2, 4]).unwrap();
    let output_shape = Shape::create(vec![2, 7]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", DataType::F32, lhs_shape, Device::Cpu);
    let rhs = graph.add_input("rhs", DataType::F32, rhs_shape, Device::Cpu);
    let concat = graph
        .add_named_operation(
            None,
            NodeKind::concat(vec![lhs, rhs], 1),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    replace_with_stale_shape_metadata(&mut graph, concat);
    graph.set_outputs(vec![concat]).unwrap();

    let normalized = infer_and_validate_value_metadata(&graph);
    assert_value_metadata(
        &normalized,
        concat,
        DataType::F32,
        output_shape.dimensions(),
        output_shape.dimensions(),
        Device::Cpu,
    );
}

#[test]
fn test_infer_value_metadata_pass_infers_index_select_and_embedding_shapes() {
    let input_shape = Shape::create(vec![2, 3, 4]).unwrap();
    let weight_shape = Shape::create(vec![5, 6]).unwrap();
    let indices_shape = Shape::create(vec![7]).unwrap();
    let select_shape = Shape::create(vec![2, 7, 4]).unwrap();
    let embedding_shape = Shape::create(vec![7, 6]).unwrap();
    let select_output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &select_shape).unwrap();
    let embedding_output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &embedding_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let weight = graph.add_input("weight", DataType::F32, weight_shape, Device::Cpu);
    let indices = graph.add_input("indices", DataType::I64, indices_shape, Device::Cpu);
    let select = graph
        .add_named_operation(
            None,
            NodeKind::index_select(input, 1, indices),
            select_output.data_type(),
            select_output.shape().clone(),
            select_output.device(),
        )
        .unwrap();
    let embedding = graph
        .add_named_operation(
            None,
            NodeKind::embedding(weight, indices),
            embedding_output.data_type(),
            embedding_output.shape().clone(),
            embedding_output.device(),
        )
        .unwrap();
    for node in [select, embedding] {
        replace_with_stale_shape_metadata(&mut graph, node);
    }
    graph.set_outputs(vec![select, embedding]).unwrap();

    let normalized = infer_and_validate_value_metadata(&graph);
    assert_value_metadata(
        &normalized,
        select,
        DataType::F32,
        select_shape.dimensions(),
        select_shape.dimensions(),
        Device::Cpu,
    );
    assert_value_metadata(
        &normalized,
        embedding,
        DataType::F32,
        embedding_shape.dimensions(),
        embedding_shape.dimensions(),
        Device::Cpu,
    );
}

#[test]
fn test_infer_value_metadata_pass_infers_pad_shape() {
    let input_shape = Shape::create(vec![2, 3]).unwrap();
    let output_shape = Shape::create(vec![3, 5]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let pad = graph
        .add_named_operation(
            None,
            NodeKind::pad(input, vec![(1, 0), (1, 1)], -1.0),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    replace_with_stale_shape_metadata(&mut graph, pad);
    graph.set_outputs(vec![pad]).unwrap();

    let normalized = infer_and_validate_value_metadata(&graph);
    assert_value_metadata(
        &normalized,
        pad,
        DataType::F32,
        output_shape.dimensions(),
        output_shape.dimensions(),
        Device::Cpu,
    );
}

#[test]
fn test_infer_value_metadata_pass_infers_repeat_kv_shape() {
    let input_shape = Shape::create(vec![2, 3, 4, 5]).unwrap();
    let output_shape = Shape::create(vec![2, 12, 4, 5]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let repeat = graph
        .add_named_operation(
            None,
            NodeKind::repeat_kv(input, 4),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    replace_with_stale_shape_metadata(&mut graph, repeat);
    graph.set_outputs(vec![repeat]).unwrap();

    let normalized = infer_and_validate_value_metadata(&graph);
    assert_value_metadata(
        &normalized,
        repeat,
        DataType::F32,
        output_shape.dimensions(),
        output_shape.dimensions(),
        Device::Cpu,
    );
}
