use super::*;

#[test]
fn test_graph_compile_prunes_dead_nodes_and_remaps_quantization_attributes() {
    let mut graph = Graph::new();
    add_dead_scale(&mut graph);

    let shape = Shape::create([4]).unwrap();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let quantized = graph
        .add_named_quantization_operation(
            None,
            RegistryOperation::Quantize,
            input,
            0.5,
            2,
            DataType::U8,
            shape.clone(),
            Device::Cpu,
        )
        .unwrap();
    let dequantized = graph
        .add_named_quantization_operation(
            None,
            RegistryOperation::Dequantize,
            quantized,
            0.25,
            3,
            DataType::F32,
            shape,
            Device::Cpu,
        )
        .unwrap();
    graph.set_outputs(vec![dequantized]).unwrap();

    let compiled = compile_and_assert_pruned_order(&graph, 7);
    let ir = compiled.to_ir();
    let quantization_attributes = ir
        .operations()
        .iter()
        .filter_map(|operation| {
            matches!(
                operation.opcode(),
                Some(RegistryOperation::Quantize | RegistryOperation::Dequantize)
            )
            .then(|| {
                (
                    operation.opcode().unwrap(),
                    operation.attributes().quantization().copied(),
                )
            })
            .and_then(|(opcode, attributes)| attributes.map(|attributes| (opcode, attributes)))
        })
        .collect::<Vec<_>>();
    let expected = [
        (RegistryOperation::Quantize, Some(DataType::U8)),
        (RegistryOperation::Dequantize, None),
    ];
    assert_eq!(quantization_attributes.len(), expected.len());
    for ((opcode, attributes), (expected_opcode, data_type)) in
        quantization_attributes.iter().zip(expected)
    {
        assert_eq!(*opcode, expected_opcode);
        assert_eq!(attributes.data_type, data_type);
    }
}
