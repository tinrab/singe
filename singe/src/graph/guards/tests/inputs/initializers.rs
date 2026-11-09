use super::*;

#[test]
fn test_graph_guards_resolve_initializer_and_runtime_override() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let x = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let bias = graph.add_input("bias", DataType::F32, shape.clone(), Device::Cpu);
    add_f32_initializer(&mut graph, "bias", &shape, vec![10.0_f32, 20.0]);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let sum = add_binary_operation(&mut graph, RegistryOperation::Add, [x, bias], &output);
    graph.set_outputs(vec![sum]).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), cpu_tensor(&shape, vec![1.0_f32, 2.0]));
    let outputs = graph.run(&inputs).unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[11.0, 22.0], 1e-6);

    inputs.insert("bias".to_string(), cpu_tensor(&shape, vec![3.0_f32, 4.0]));
    let outputs = graph.run(&inputs).unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[4.0, 6.0], 1e-6);
}

#[test]
fn test_graph_guards_reject_static_initializer_override() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let x = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let bias = graph.add_input("bias", DataType::F32, shape.clone(), Device::Cpu);
    add_static_f32_initializer(&mut graph, "bias", &shape, vec![10.0_f32, 20.0]);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let sum = add_binary_operation(&mut graph, RegistryOperation::Add, [x, bias], &output);
    graph.set_outputs(vec![sum]).unwrap();

    let outputs = graph
        .run(&single_input("x", cpu_tensor(&shape, vec![1.0_f32, 2.0])))
        .unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[11.0, 22.0], 1e-6);

    let compiled = graph.compile().unwrap();
    let guards = create_guards_from_ir(&compiled);
    let bias_guard = guards
        .inputs()
        .iter()
        .find(|guard| guard.name() == "bias")
        .unwrap();
    assert!(bias_guard.is_static_initializer());
    let static_storage = bias_guard.static_storage().unwrap();
    assert_ne!(static_storage.storage_identity(), 0);
    assert!(static_storage.data_pointer_identity().is_some());
    assert_eq!(static_storage.storage_offset(), 0);
    assert_eq!(static_storage.storage_byte_size(), 8);

    let error = graph
        .run(&HashMap::from([
            ("x".to_string(), cpu_tensor(&shape, vec![1.0_f32, 2.0])),
            ("bias".to_string(), cpu_tensor(&shape, vec![3.0_f32, 4.0])),
        ]))
        .unwrap_err();
    assert!(matches!(
        error,
        Error::Unsupported {
            operation,
            details,
            ..
        } if operation == "static graph input override"
            && details.contains("input `bias` is a static initializer")
    ));
}
