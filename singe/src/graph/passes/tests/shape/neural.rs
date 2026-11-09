use super::*;

#[test]
fn test_infer_value_metadata_pass_infers_attention_shape() {
    let query_shape = Shape::create(vec![2, 3, 4, 5]).unwrap();
    let key_shape = Shape::create(vec![2, 3, 6, 5]).unwrap();
    let value_shape = Shape::create(vec![2, 3, 6, 7]).unwrap();
    let output_shape = Shape::create(vec![2, 3, 4, 7]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let query = graph.add_input("query", DataType::F32, query_shape, Device::Cpu);
    let key = graph.add_input("key", DataType::F32, key_shape, Device::Cpu);
    let value = graph.add_input("value", DataType::F32, value_shape, Device::Cpu);
    let attention = add_scaled_dot_product_attention_operation(
        &mut graph,
        query,
        key,
        value,
        Some(0.5),
        false,
        &output,
    );
    replace_with_stale_shape_metadata(&mut graph, attention);
    graph.set_outputs(vec![attention]).unwrap();

    let normalized = infer_and_validate_value_metadata(&graph);
    assert_value_metadata(
        &normalized,
        attention,
        DataType::F32,
        output_shape.dimensions(),
        output_shape.dimensions(),
        Device::Cpu,
    );
}

#[test]
fn test_infer_value_metadata_pass_infers_normalization_shape() {
    let input_shape = Shape::create(vec![2, 3, 4]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &input_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape.clone(), Device::Cpu);
    let norm = add_layer_normalization_operation(
        &mut graph,
        RegistryOperation::LayerNorm,
        input,
        vec![4],
        None,
        None,
        1e-5,
        &output,
    );
    replace_with_stale_shape_metadata(&mut graph, norm);
    graph.set_outputs(vec![norm]).unwrap();

    let normalized = infer_and_validate_value_metadata(&graph);
    assert_value_metadata(
        &normalized,
        norm,
        DataType::F32,
        input_shape.dimensions(),
        input_shape.dimensions(),
        Device::Cpu,
    );
}

#[test]
fn test_infer_value_metadata_pass_validates_batch_norm_parameter_count() {
    let input_shape = Shape::create(vec![2, 3, 4]).unwrap();
    let parameter_shape = Shape::create(vec![4]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &input_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let weight = graph.add_input("weight", DataType::F32, parameter_shape, Device::Cpu);
    let batch_norm = add_batch_normalization_operation(
        &mut graph,
        input,
        Some(weight),
        None,
        None,
        None,
        false,
        0.1,
        1e-5,
        &output,
    );
    replace_with_stale_shape_metadata(&mut graph, batch_norm);
    graph.set_outputs(vec![batch_norm]).unwrap();

    let error = infer_value_metadata_error(&graph);
    assert!(matches!(
        error,
        Error::LengthMismatch { name, expected: 3, actual: 4 }
            if name == "operation batch norm weight elements"
    ));
}

#[test]
fn test_infer_value_metadata_pass_infers_conv2d_shape() {
    let input_shape = Shape::create(vec![2, 3, 5, 5]).unwrap();
    let weight_shape = Shape::create(vec![4, 3, 3, 3]).unwrap();
    let output_shape = Shape::create(vec![2, 4, 3, 3]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let weight = graph.add_input("w", DataType::F32, weight_shape, Device::Cpu);
    let conv = add_conv2d_operation(&mut graph, input, weight, None, &output);
    replace_with_stale_shape_metadata(&mut graph, conv);
    graph.set_outputs(vec![conv]).unwrap();

    let normalized = infer_and_validate_value_metadata(&graph);
    assert_value_metadata(
        &normalized,
        conv,
        DataType::F32,
        output_shape.dimensions(),
        output_shape.dimensions(),
        Device::Cpu,
    );
}

#[test]
fn test_infer_value_metadata_pass_infers_pool2d_shape() {
    let input_shape = Shape::create(vec![2, 3, 5, 5]).unwrap();
    let output_shape = Shape::create(vec![2, 3, 2, 2]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let pool = add_pool2d_operation(
        &mut graph,
        RegistryOperation::MaxPool2d,
        input,
        (2, 2),
        (2, 2),
        (0, 0, 0, 0),
        (1, 1),
        false,
        false,
        &output,
    );
    replace_with_stale_shape_metadata(&mut graph, pool);
    graph.set_outputs(vec![pool]).unwrap();

    let normalized = infer_and_validate_value_metadata(&graph);
    assert_value_metadata(
        &normalized,
        pool,
        DataType::F32,
        output_shape.dimensions(),
        output_shape.dimensions(),
        Device::Cpu,
    );
}
