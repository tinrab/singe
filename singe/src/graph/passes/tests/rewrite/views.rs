use crate::{
    backend::registry::Operation as RegistryOperation,
    graph::{node::NodeKind, test_support::*},
    layout::Layout,
};

#[test]
fn test_remove_redundant_ops_pass_removes_metadata_preserving_noops() {
    let shape = Shape::create(vec![2]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let cast = graph
        .add_named_operation(
            None,
            NodeKind::cast(input, DataType::F32),
            DataType::F32,
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    let copy = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(cast, Device::Cpu),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    let contiguous = graph
        .add_named_operation(
            None,
            NodeKind::contiguous(copy),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    let reshape = graph
        .add_named_operation(
            None,
            NodeKind::reshape(contiguous, shape.clone()),
            output.data_type(),
            shape.clone(),
            output.device(),
        )
        .unwrap();
    let transpose = graph
        .add_named_operation(
            None,
            NodeKind::transpose(reshape, 0, 0),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    let permute = graph
        .add_named_operation(
            None,
            NodeKind::permute(transpose, vec![0]),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    let flatten = graph
        .add_named_operation(
            None,
            NodeKind::flatten(permute, 0, 0),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    let expand = graph
        .add_named_operation(
            None,
            NodeKind::expand(flatten, shape.clone()),
            output.data_type(),
            shape.clone(),
            output.device(),
        )
        .unwrap();
    let concat = graph
        .add_named_operation(
            None,
            NodeKind::concat(vec![expand], 0),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    let live = add_scalar_operation(&mut graph, RegistryOperation::Scale, concat, 2.0, &output);
    graph.set_outputs(vec![live]).unwrap();

    let pipeline = remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(results[0].pass_name, "remove_redundant_ops");
    assert_eq!(results[0].checked_nodes, 11);
    assert_eq!(
        results[0].remapped_nodes,
        [
            (input, ValueId::new(0)),
            (cast, ValueId::new(0)),
            (copy, ValueId::new(0)),
            (contiguous, ValueId::new(0)),
            (reshape, ValueId::new(0)),
            (transpose, ValueId::new(0)),
            (permute, ValueId::new(0)),
            (flatten, ValueId::new(0)),
            (expand, ValueId::new(0)),
            (concat, ValueId::new(0)),
            (live, ValueId::new(1)),
        ],
    );
    assert_value_count(&normalized, 2);
    assert_output_nodes(&normalized, &[ValueId::new(1)]);
    assert_operation(
        &normalized,
        ValueId::new(1),
        Operation::Scale,
        &[ValueId::new(0)],
        &OperationAttributes::Scalar((2.0).into()),
    );

    let compiled = normalized.compile().unwrap();
    let outputs = compiled
        .run(&HashMap::from([(
            "x".to_string(),
            cpu_tensor(&shape, vec![3.0_f32, -4.0]),
        )]))
        .unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[6.0, -8.0]);
}

#[test]
fn test_remove_redundant_ops_pass_preserves_empty_reshape_and_slice_aliases() {
    let empty_shape = Shape::create(vec![0, 3]).unwrap();
    let strided_empty_shape = Layout::with_strides(vec![0, 3], vec![100, 1]).unwrap();
    let i64_input_shape = Shape::create(vec![2, 3]).unwrap();
    let f32_output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &empty_shape).unwrap();
    let i64_output = Tensor::zeroed_on(Device::Cpu, DataType::I64, &empty_shape).unwrap();
    let mut graph = Graph::new();
    let f32_input = graph.add_input("f32", DataType::F32, empty_shape.clone(), Device::Cpu);
    graph.replace_node_output_layout_for_test(f32_input, strided_empty_shape.clone());
    let i64_input = graph.add_input("i64", DataType::I64, i64_input_shape, Device::Cpu);
    let reshape = graph
        .add_named_operation(
            None,
            NodeKind::reshape(f32_input, empty_shape.clone()),
            f32_output.data_type(),
            empty_shape.clone(),
            f32_output.device(),
        )
        .unwrap();
    let slice = graph
        .add_named_slice_operation(
            None,
            i64_input,
            0,
            1,
            0,
            i64_output.data_type(),
            i64_output.shape(),
            i64_output.device(),
        )
        .unwrap();
    graph.set_outputs(vec![reshape, slice]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(
        results[0].remapped_nodes,
        [
            (f32_input, ValueId::new(0)),
            (i64_input, ValueId::new(1)),
            (reshape, ValueId::new(0)),
            (slice, ValueId::new(2)),
        ],
    );
    assert_eq!(
        normalized.node(ValueId::new(0)).unwrap().output_layout(),
        &strided_empty_shape
    );
    assert_operation(
        &normalized,
        ValueId::new(2),
        Operation::Slice,
        &[ValueId::new(1)],
        &OperationAttributes::Slice(SliceAttributes {
            dim: 0,
            start: 1,
            length: 0,
            step: 1,
        }),
    );
    assert_value_metadata(
        &normalized,
        ValueId::new(2),
        DataType::I64,
        empty_shape.dimensions(),
        empty_shape.dimensions(),
        Device::Cpu,
    );
    assert_output_nodes(&normalized, &[ValueId::new(0), ValueId::new(2)]);
}

#[test]
fn test_graph_compile_removes_redundant_ops_by_default() {
    let shape = Shape::create(vec![2]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let cast = graph
        .add_named_operation(
            None,
            NodeKind::cast(input, DataType::F32),
            DataType::F32,
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    let live = add_scalar_operation(&mut graph, RegistryOperation::Scale, cast, 3.0, &output);
    graph.set_outputs(vec![live]).unwrap();

    let compiled = graph.compile().unwrap();
    let compiled_ir = compiled.to_ir();
    assert_eq!(compiled_ir.values().len(), 2);
    assert_eq!(compiled_ir.output_values(), &[ValueId::new(1)]);
    let scale = compiled_ir
        .operation(compiled_ir.result_for_value(ValueId::new(1)).unwrap().0)
        .unwrap();
    assert_eq!(scale.opcode(), Some(Operation::Scale));
    assert_eq!(scale.inputs(), &[ValueId::new(0)]);
    assert_eq!(
        scale.attributes(),
        &OperationAttributes::Scalar((3.0).into())
    );

    let outputs = compiled
        .run(&HashMap::from([(
            "x".to_string(),
            cpu_tensor(&shape, vec![2.0_f32, 4.0]),
        )]))
        .unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[6.0, 12.0]);
}

#[test]
fn test_remove_redundant_ops_pass_canonicalizes_copy_to_chains() {
    let shape = Shape::create(vec![2]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let first_copy = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(input, Device::Cpu),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    let second_copy = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(first_copy, Device::Cpu),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.replace_node_copy_to_for_test(first_copy, input, Device::Cuda(0));
    graph.replace_node_output_device_for_test(first_copy, Device::Cuda(0));
    let live = add_scalar_operation(
        &mut graph,
        RegistryOperation::Scale,
        second_copy,
        3.0,
        &output,
    );
    graph.set_outputs(vec![live]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(
        results[0].remapped_nodes,
        [
            (input, ValueId::new(0)),
            (first_copy, ValueId::new(1)),
            (second_copy, ValueId::new(0)),
            (live, ValueId::new(2)),
        ],
    );
    assert_operation(
        &normalized,
        ValueId::new(2),
        Operation::Scale,
        &[ValueId::new(0)],
        &OperationAttributes::Scalar((3.0).into()),
    );

    let compiled = normalized.compile().unwrap();
    let outputs = compiled
        .run(&HashMap::from([(
            "x".to_string(),
            cpu_tensor(&shape, vec![2.0_f32, 4.0]),
        )]))
        .unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[6.0, 12.0]);
}

#[test]
fn test_remove_redundant_ops_pass_canonicalizes_multi_hop_copy_to_chains() {
    let shape = Shape::create(vec![2]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let first_copy = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(input, Device::Cpu),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.replace_node_copy_to_for_test(first_copy, input, Device::Cuda(0));
    graph.replace_node_output_device_for_test(first_copy, Device::Cuda(0));
    let second_copy = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(first_copy, Device::Cpu),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    let third_copy = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(second_copy, Device::Cpu),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.replace_node_copy_to_for_test(third_copy, second_copy, Device::Cuda(0));
    graph.replace_node_output_device_for_test(third_copy, Device::Cuda(0));
    let fourth_copy = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(third_copy, Device::Cpu),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    let live = add_scalar_operation(
        &mut graph,
        RegistryOperation::Scale,
        fourth_copy,
        4.0,
        &output,
    );
    graph.set_outputs(vec![live]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(
        results[0].remapped_nodes,
        [
            (input, ValueId::new(0)),
            (first_copy, ValueId::new(1)),
            (second_copy, ValueId::new(0)),
            (third_copy, ValueId::new(1)),
            (fourth_copy, ValueId::new(0)),
            (live, ValueId::new(2)),
        ],
    );
    assert_operation(
        &normalized,
        ValueId::new(2),
        Operation::Scale,
        &[ValueId::new(0)],
        &OperationAttributes::Scalar((4.0).into()),
    );

    let compiled = normalized.compile().unwrap();
    let outputs = compiled
        .run(&HashMap::from([(
            "x".to_string(),
            cpu_tensor(&shape, vec![2.0_f32, 4.0]),
        )]))
        .unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[8.0, 16.0]);
}

#[test]
fn test_remove_redundant_ops_pass_preserves_final_copy_to_device_in_bounce_chains() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let first_cuda = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(input, Device::Cuda(0)),
            DataType::F32,
            shape.clone(),
            Device::Cuda(0),
        )
        .unwrap();
    let back_to_cpu = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(first_cuda, Device::Cpu),
            DataType::F32,
            shape.clone(),
            Device::Cpu,
        )
        .unwrap();
    let final_cuda = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(back_to_cpu, Device::Cuda(0)),
            DataType::F32,
            shape,
            Device::Cuda(0),
        )
        .unwrap();
    graph.set_outputs(vec![final_cuda]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(
        results[0].remapped_nodes,
        [
            (input, ValueId::new(0)),
            (first_cuda, ValueId::new(1)),
            (back_to_cpu, ValueId::new(0)),
            (final_cuda, ValueId::new(1)),
        ],
    );
    assert_operation(
        &normalized,
        ValueId::new(1),
        Operation::CopyTo,
        &[ValueId::new(0)],
        &OperationAttributes::CopyTo(Device::Cuda(0)),
    );
    assert_output_nodes(&normalized, &[ValueId::new(1)]);
}

#[test]
fn test_remove_redundant_ops_pass_folds_inverse_view_pairs() {
    let shape = Shape::create(vec![2, 3]).unwrap();
    let unsqueezed_shape = shape.unsqueeze(0).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let unsqueezed_output =
        Tensor::zeroed_on(Device::Cpu, DataType::F32, &unsqueezed_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape, Device::Cpu);
    let unsqueeze = add_dimension_operation(
        &mut graph,
        RegistryOperation::Unsqueeze,
        input,
        0,
        &unsqueezed_output,
    );
    let squeeze = add_dimension_operation(
        &mut graph,
        RegistryOperation::Squeeze,
        unsqueeze,
        0,
        &output,
    );
    let live = add_scalar_operation(&mut graph, RegistryOperation::Scale, squeeze, 2.0, &output);
    graph.set_outputs(vec![live]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(
        results[0].remapped_nodes,
        [
            (input, ValueId::new(0)),
            (unsqueeze, ValueId::new(1)),
            (squeeze, ValueId::new(0)),
            (live, ValueId::new(2)),
        ],
    );
    assert_value_count(&normalized, 3);
    assert_operation(
        &normalized,
        ValueId::new(2),
        Operation::Scale,
        &[ValueId::new(0)],
        &OperationAttributes::Scalar((2.0).into()),
    );
}

#[test]
fn test_remove_redundant_ops_pass_folds_squeeze_unsqueeze_inverse_pairs() {
    let shape = Shape::create(vec![1, 2, 3]).unwrap();
    let squeezed_shape = shape.squeeze(0).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let squeezed_output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &squeezed_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape, Device::Cpu);
    let squeeze = add_dimension_operation(
        &mut graph,
        RegistryOperation::Squeeze,
        input,
        0,
        &squeezed_output,
    );
    let unsqueeze = add_dimension_operation(
        &mut graph,
        RegistryOperation::Unsqueeze,
        squeeze,
        0,
        &output,
    );
    let live = add_scalar_operation(
        &mut graph,
        RegistryOperation::Scale,
        unsqueeze,
        2.0,
        &output,
    );
    graph.set_outputs(vec![live]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(
        results[0].remapped_nodes,
        [
            (input, ValueId::new(0)),
            (squeeze, ValueId::new(1)),
            (unsqueeze, ValueId::new(0)),
            (live, ValueId::new(2)),
        ],
    );
    assert_value_count(&normalized, 3);
    assert_operation(
        &normalized,
        ValueId::new(2),
        Operation::Scale,
        &[ValueId::new(0)],
        &OperationAttributes::Scalar((2.0).into()),
    );
}
