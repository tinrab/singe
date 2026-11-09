use super::*;

#[test]
fn test_graph_compile_prunes_dead_nodes_from_execution_order() {
    let mut graph = Graph::new();
    let shape = Shape::create(vec![2]).unwrap();
    let a = graph.add_input("a", DataType::F32, shape.clone(), Device::Cpu);
    let b = graph.add_input("b", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let live = add_binary_operation(&mut graph, RegistryOperation::Add, [a, b], &output);

    let dead_left_shape = Shape::create(vec![2, 3]).unwrap();
    let dead_right_shape = Shape::create(vec![4, 2]).unwrap();
    let dead_output_shape = Shape::create(vec![2, 2]).unwrap();
    let dead_lhs = graph.add_input("dead_lhs", DataType::F32, dead_left_shape, Device::Cpu);
    let dead_rhs = graph.add_input("dead_rhs", DataType::F32, dead_right_shape, Device::Cpu);
    let dead_output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &dead_output_shape).unwrap();
    let _dead = add_binary_operation(
        &mut graph,
        RegistryOperation::MatMul,
        [dead_lhs, dead_rhs],
        &dead_output,
    );
    graph.set_outputs(vec![live]).unwrap();

    let compiled = compile_and_assert_pruned_order(&graph, 3);
    let compiled_ir = compiled.to_ir();
    assert_eq!(compiled_ir.values().len(), 3);
    assert_eq!(compiled_ir.output_values(), &[ValueId::new(2)]);

    assert!(matches!(
        graph.compile_with_options(GraphCompileOptions::new().with_dead_node_pruning(false)),
        Err(Error::LengthMismatch { name, .. })
            if name == "operation matmul inner dimension"
    ));

    let outputs = compiled
        .run(&HashMap::from([
            ("a".to_string(), cpu_tensor(&shape, vec![1.0_f32, 2.0])),
            ("b".to_string(), cpu_tensor(&shape, vec![3.0_f32, 4.0])),
            (
                "dead_lhs".to_string(),
                cpu_tensor(&Shape::create(vec![2, 3]).unwrap(), vec![0.0_f32; 6]),
            ),
            (
                "dead_rhs".to_string(),
                cpu_tensor(&Shape::create(vec![4, 2]).unwrap(), vec![0.0_f32; 8]),
            ),
        ]))
        .unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[4.0, 6.0], 1e-6);
}
