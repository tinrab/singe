use crate::{backend::registry::Operation as RegistryOperation, graph::test_support::*};

#[test]
fn test_remove_redundant_ops_pass_deduplicates_repeated_constants() {
    let shape = Shape::create(vec![2]).unwrap();
    let constant = cpu_tensor(&shape, vec![1.0_f32, 2.0]);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let first = graph.add_constant(constant.clone());
    let second = graph.add_constant(constant);
    let add = add_binary_operation(&mut graph, RegistryOperation::Add, [first, second], &output);
    graph.set_outputs(vec![add]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(
        results[0].remapped_nodes,
        [
            (first, ValueId::new(0)),
            (second, ValueId::new(0)),
            (add, ValueId::new(1)),
        ],
    );
    assert_value_count(&normalized, 2);
    assert_operation(
        &normalized,
        ValueId::new(1),
        Operation::Add,
        &[ValueId::new(0), ValueId::new(0)],
        &OperationAttributes::None,
    );

    let compiled = normalized.compile().unwrap();
    let outputs = compiled.run(&HashMap::new()).unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[2.0, 4.0]);
}

#[test]
fn test_remove_redundant_ops_pass_deduplicates_pure_deterministic_ops() {
    let shape = Shape::create(vec![2]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let first = add_scalar_operation(&mut graph, RegistryOperation::Scale, input, 2.0, &output);
    let second = add_scalar_operation(&mut graph, RegistryOperation::Scale, input, 2.0, &output);
    let add = add_binary_operation(&mut graph, RegistryOperation::Add, [first, second], &output);
    graph.set_outputs(vec![add]).unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(
        results[0].remapped_nodes,
        [
            (input, ValueId::new(0)),
            (first, ValueId::new(1)),
            (second, ValueId::new(1)),
            (add, ValueId::new(2)),
        ],
    );
    assert_value_count(&normalized, 3);
    assert_operation(
        &normalized,
        ValueId::new(2),
        Operation::Add,
        &[ValueId::new(1), ValueId::new(1)],
        &OperationAttributes::None,
    );

    let compiled = graph.compile().unwrap();
    let compiled_ir = compiled.to_ir();
    assert_eq!(compiled_ir.values().len(), 3);
    let outputs = compiled
        .run(&HashMap::from([(
            "x".to_string(),
            cpu_tensor(&shape, vec![1.0_f32, 3.0]),
        )]))
        .unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[4.0, 12.0]);
}

#[test]
fn test_remove_redundant_ops_pass_deduplicates_equivalent_quantization_attributes() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let f32_input = graph.add_input("f32", DataType::F32, shape.clone(), Device::Cpu);
    let u8_input = graph.add_input("u8", DataType::U8, shape.clone(), Device::Cpu);

    let first_quantize = graph
        .add_named_quantization_operation(
            None,
            RegistryOperation::Quantize,
            f32_input,
            0.5,
            2,
            DataType::U8,
            shape.clone(),
            Device::Cpu,
        )
        .unwrap();
    let equivalent_quantize = graph
        .add_named_quantization_operation(
            None,
            RegistryOperation::Quantize,
            f32_input,
            0.5,
            2,
            DataType::U8,
            shape.clone(),
            Device::Cpu,
        )
        .unwrap();
    let distinct_quantize_zero_point = graph
        .add_named_quantization_operation(
            None,
            RegistryOperation::Quantize,
            f32_input,
            0.5,
            3,
            DataType::U8,
            shape.clone(),
            Device::Cpu,
        )
        .unwrap();
    let distinct_quantize_data_type = graph
        .add_named_quantization_operation(
            None,
            RegistryOperation::Quantize,
            f32_input,
            0.5,
            2,
            DataType::I8,
            shape.clone(),
            Device::Cpu,
        )
        .unwrap();

    let first_dequantize = graph
        .add_named_quantization_operation(
            None,
            RegistryOperation::Dequantize,
            u8_input,
            0.5,
            2,
            DataType::F32,
            shape.clone(),
            Device::Cpu,
        )
        .unwrap();
    let equivalent_dequantize = graph
        .add_named_quantization_operation(
            None,
            RegistryOperation::Dequantize,
            u8_input,
            0.5,
            2,
            DataType::F32,
            shape.clone(),
            Device::Cpu,
        )
        .unwrap();
    let distinct_dequantize_scale = graph
        .add_named_quantization_operation(
            None,
            RegistryOperation::Dequantize,
            u8_input,
            0.25,
            2,
            DataType::F32,
            shape,
            Device::Cpu,
        )
        .unwrap();
    graph
        .set_outputs(vec![
            first_quantize,
            equivalent_quantize,
            distinct_quantize_zero_point,
            distinct_quantize_data_type,
            first_dequantize,
            equivalent_dequantize,
            distinct_dequantize_scale,
        ])
        .unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_remap_classes(
        &results[0].remapped_nodes,
        &[
            (f32_input, 0),
            (u8_input, 1),
            (first_quantize, 2),
            (equivalent_quantize, 2),
            (distinct_quantize_zero_point, 3),
            (distinct_quantize_data_type, 4),
            (first_dequantize, 5),
            (equivalent_dequantize, 5),
            (distinct_dequantize_scale, 6),
        ],
    );
    assert_output_nodes(
        &normalized,
        &[
            remapped(&results[0].remapped_nodes, first_quantize),
            remapped(&results[0].remapped_nodes, first_quantize),
            remapped(&results[0].remapped_nodes, distinct_quantize_zero_point),
            remapped(&results[0].remapped_nodes, distinct_quantize_data_type),
            remapped(&results[0].remapped_nodes, first_dequantize),
            remapped(&results[0].remapped_nodes, first_dequantize),
            remapped(&results[0].remapped_nodes, distinct_dequantize_scale),
        ],
    );
}

#[test]
fn test_remove_redundant_ops_pass_deduplicates_equivalent_quantized_matmul_attributes() {
    let lhs_shape = Shape::create([2, 3]).unwrap();
    let rhs_shape = Shape::create([3, 2]).unwrap();
    let output_shape = Shape::create([2, 2]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", DataType::I8, lhs_shape, Device::Cpu);
    let rhs = graph.add_input("rhs", DataType::U8, rhs_shape, Device::Cpu);
    let first = add_quantized_matmul_operation(&mut graph, [lhs, rhs], 0.5, 1, 0.25, 2, &output);
    let equivalent =
        add_quantized_matmul_operation(&mut graph, [lhs, rhs], 0.5, 1, 0.25, 2, &output);
    let distinct = add_quantized_matmul_operation(&mut graph, [lhs, rhs], 0.5, 1, 0.25, 3, &output);
    let equivalent_sum = add_binary_operation(
        &mut graph,
        RegistryOperation::Add,
        [first, equivalent],
        &output,
    );
    let distinct_sum = add_binary_operation(
        &mut graph,
        RegistryOperation::Add,
        [first, distinct],
        &output,
    );
    graph
        .set_outputs(vec![equivalent_sum, distinct_sum])
        .unwrap();

    let pipeline = validating_remove_redundant_ops_pipeline();
    let (normalized, results) = pipeline.normalize(&graph).unwrap();

    assert_remap_classes(
        &results[0].remapped_nodes,
        &[
            (lhs, 0),
            (rhs, 1),
            (first, 2),
            (equivalent, 2),
            (distinct, 3),
            (equivalent_sum, 4),
            (distinct_sum, 5),
        ],
    );
    assert_operation(
        &normalized,
        remapped(&results[0].remapped_nodes, equivalent_sum),
        Operation::Add,
        &[
            remapped(&results[0].remapped_nodes, first),
            remapped(&results[0].remapped_nodes, first),
        ],
        &OperationAttributes::None,
    );
    assert_operation(
        &normalized,
        remapped(&results[0].remapped_nodes, distinct_sum),
        Operation::Add,
        &[
            remapped(&results[0].remapped_nodes, first),
            remapped(&results[0].remapped_nodes, distinct),
        ],
        &OperationAttributes::None,
    );
}
