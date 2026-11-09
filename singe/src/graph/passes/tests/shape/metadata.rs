use super::*;
use crate::{graph::node::NodeKind, layout::Layout};

#[test]
fn test_infer_value_metadata_pass_normalizes_schema_metadata() {
    let input_shape = Shape::create(vec![2]).unwrap();
    let cast_shape = input_shape.clone();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape.clone(), Device::Cpu);
    let cast_output = Tensor::zeroed_on(Device::Cpu, DataType::I32, &cast_shape).unwrap();
    let cast = graph
        .add_named_operation(
            None,
            NodeKind::cast(input, DataType::I32),
            DataType::I32,
            cast_output.shape().clone(),
            cast_output.device(),
        )
        .unwrap();
    graph.replace_node_output_data_type_for_test(cast, DataType::F32);
    graph.set_outputs(vec![cast]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::InferValueMetadata,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(results[0].pass_name, "infer_value_metadata");
    assert_eq!(results[0].checked_nodes, 2);
    assert_value_metadata(
        &normalized,
        input,
        DataType::F32,
        input_shape.dimensions(),
        input_shape.dimensions(),
        Device::Cpu,
    );
    assert_value_metadata(
        &normalized,
        cast,
        DataType::I32,
        cast_shape.dimensions(),
        cast_shape.dimensions(),
        Device::Cpu,
    );
}

#[test]
fn test_input_declaration_has_one_metadata_owner() {
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, Shape::create([2]).unwrap(), Device::Cpu);
    graph.replace_node_output_metadata_for_test(
        input,
        DataType::F64,
        Layout::contiguous([3]).unwrap(),
        Device::Cuda(0),
    );
    graph.set_outputs(vec![input]).unwrap();
    let pipeline = GraphPassPipeline::new(vec![GraphPassKind::ValidateValueMetadata]);
    let (normalized, _) = pipeline.normalize(&graph).unwrap();
    let node = normalized.required_node(input).unwrap();
    let (name, dtype, shape, device) = node.input_metadata().unwrap();
    assert_eq!(name, "x");
    assert_eq!(dtype, DataType::F64);
    assert_eq!(shape.dimensions(), [3]);
    assert_eq!(device, Device::Cuda(0));
    assert_eq!(node.output_spec().data_type(), dtype);
}

#[test]
fn test_infer_value_metadata_pass_normalizes_same_as_input_metadata() {
    let shape = Shape::create(vec![2, 3]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let relu = add_unary_operation(&mut graph, RegistryOperation::Relu, input, &output);
    let stale_shape = Shape::create(vec![99]).unwrap();
    graph.replace_node_output_metadata_for_test(
        relu,
        DataType::F64,
        Layout::from_shape(&stale_shape),
        Device::Cuda(0),
    );
    graph.set_outputs(vec![relu]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::InferValueMetadata,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (normalized, _) = pipeline.normalize(&graph).unwrap();

    assert_value_metadata(
        &normalized,
        relu,
        DataType::F32,
        shape.dimensions(),
        shape.dimensions(),
        Device::Cpu,
    );
}

#[test]
fn test_infer_value_metadata_pass_infers_same_as_input_metadata() {
    let shape = Shape::create(vec![2, 3]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let relu = add_unary_operation(&mut graph, RegistryOperation::Relu, input, &output);
    replace_with_stale_shape_metadata(&mut graph, relu);
    graph.replace_node_output_data_type_for_test(relu, DataType::I32);
    graph.set_outputs(vec![relu]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::InferValueMetadata,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (normalized, _) = pipeline.normalize(&graph).unwrap();

    assert_value_metadata(
        &normalized,
        relu,
        DataType::F32,
        shape.dimensions(),
        shape.dimensions(),
        Device::Cpu,
    );
}

#[test]
fn test_infer_value_metadata_pass_normalizes_contiguous_layout() {
    let input_shape = Shape::create(vec![2, 3]).unwrap();
    let slice_shape = Shape::create(vec![2, 2]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::I64, &slice_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::I64, input_shape, Device::Cpu);
    let sliced = graph
        .add_named_slice_operation(
            None,
            input,
            1,
            1,
            2,
            output.data_type(),
            output.shape(),
            output.device(),
        )
        .unwrap();
    let contiguous = graph
        .add_named_operation(
            None,
            NodeKind::contiguous(sliced),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.replace_node_output_layout_for_test(
        contiguous,
        Layout::from_shape(&Shape::create(vec![99]).unwrap()),
    );
    graph.set_outputs(vec![contiguous]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::InferValueMetadata,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (normalized, _) = pipeline.normalize(&graph).unwrap();

    assert_value_layout(&normalized, sliced, slice_shape.dimensions(), &[3, 1], 1);
    assert_value_layout(
        &normalized,
        contiguous,
        slice_shape.dimensions(),
        &[2, 1],
        0,
    );
}

#[test]
fn test_infer_value_metadata_pass_normalizes_copy_to_device_and_materializes_contiguous_layout() {
    let input_shape = Shape::create(vec![2, 3]).unwrap();
    let slice_shape = Shape::create(vec![2, 2]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::I64, &slice_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::I64, input_shape, Device::Cpu);
    let sliced = graph
        .add_named_slice_operation(
            None,
            input,
            1,
            1,
            2,
            output.data_type(),
            output.shape(),
            output.device(),
        )
        .unwrap();
    let copied = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(sliced, Device::Cpu),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.replace_node_copy_to_for_test(copied, sliced, Device::Cuda(0));
    graph.replace_node_output_device_for_test(copied, Device::Cpu);
    graph.replace_node_output_layout_for_test(
        copied,
        Layout::from_shape(&Shape::create(vec![99]).unwrap()),
    );
    graph.set_outputs(vec![copied]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::InferValueMetadata,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (normalized, _) = pipeline.normalize(&graph).unwrap();

    assert_value_metadata(
        &normalized,
        copied,
        DataType::I64,
        slice_shape.dimensions(),
        slice_shape.dimensions(),
        Device::Cuda(0),
    );
    assert_value_layout(&normalized, copied, slice_shape.dimensions(), &[2, 1], 0);
}

#[test]
fn test_infer_value_metadata_pass_normalizes_alias_shape_from_layout() {
    let input_shape = Shape::create(vec![2, 2]).unwrap();
    let output_shape = Shape::create(vec![4]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let view = graph
        .add_named_operation(
            None,
            NodeKind::reshape(input, output_shape.clone()),
            output.data_type(),
            output_shape.clone(),
            output.device(),
        )
        .unwrap();
    replace_with_stale_shape_metadata(&mut graph, view);
    graph.set_outputs(vec![view]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::InferValueMetadata,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (normalized, _) = pipeline.normalize(&graph).unwrap();

    assert_value_shape(&normalized, view, output_shape.dimensions());
}

#[test]
fn test_infer_value_metadata_pass_normalizes_quantization_metadata() {
    let shape = Shape::create(vec![2]).unwrap();
    let input_tensor = Tensor::from_vec(Device::Cpu, &shape, vec![-1.0_f32, 2.0]).unwrap();
    let quantized_output = input_tensor.quantize(0.5, 0, DataType::I8).unwrap();
    let dequantized_output = quantized_output.dequantize(0.5, 0).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let quantized = graph
        .add_named_quantization_operation(
            None,
            RegistryOperation::Quantize,
            input,
            0.5,
            0,
            quantized_output.data_type(),
            quantized_output.shape().clone(),
            quantized_output.device(),
        )
        .unwrap();
    let dequantized = graph
        .add_named_quantization_operation(
            None,
            RegistryOperation::Dequantize,
            quantized,
            0.5,
            0,
            dequantized_output.data_type(),
            dequantized_output.shape().clone(),
            dequantized_output.device(),
        )
        .unwrap();
    let stale_shape = Shape::create(vec![99]).unwrap();
    graph.replace_node_output_metadata_for_test(
        quantized,
        DataType::F32,
        Layout::from_shape(&stale_shape),
        quantized_output.device(),
    );
    graph.replace_node_output_metadata_for_test(
        dequantized,
        DataType::I8,
        Layout::from_shape(&stale_shape),
        dequantized_output.device(),
    );
    graph.set_outputs(vec![dequantized]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::InferValueMetadata,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (normalized, _) = pipeline.normalize(&graph).unwrap();

    assert_value_metadata(
        &normalized,
        quantized,
        DataType::I8,
        shape.dimensions(),
        shape.dimensions(),
        Device::Cpu,
    );
    assert_value_metadata(
        &normalized,
        dequantized,
        DataType::F32,
        shape.dimensions(),
        shape.dimensions(),
        Device::Cpu,
    );
}
