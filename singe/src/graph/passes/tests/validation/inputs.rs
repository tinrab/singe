use super::*;
use crate::error::UnsupportedTarget;

#[test]
fn test_graph_input_classifies_runtime_initializer_and_static_inputs() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    graph.add_input("runtime", DataType::F32, shape.clone(), Device::Cpu);
    graph.add_input("initializer", DataType::F32, shape.clone(), Device::Cpu);
    graph.add_input("static", DataType::F32, shape.clone(), Device::Cpu);
    graph
        .add_initializer(
            "initializer",
            Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap(),
        )
        .unwrap();
    graph
        .add_static_initializer(
            "static",
            Tensor::from_vec(Device::Cpu, &shape, vec![3.0_f32, 4.0]).unwrap(),
        )
        .unwrap();

    assert_eq!(
        graph.input_class("runtime").unwrap(),
        GraphInputClass::Runtime
    );
    assert_eq!(
        graph.input_class("initializer").unwrap(),
        GraphInputClass::Initializer
    );
    assert_eq!(
        graph.input_class("static").unwrap(),
        GraphInputClass::Static
    );
    assert!(matches!(
        graph.input_class("missing"),
        Err(Error::MissingInput { .. })
    ));

    graph
        .add_initializer(
            "static",
            Tensor::from_vec(Device::Cpu, &shape, vec![5.0_f32, 6.0]).unwrap(),
        )
        .unwrap();
    assert_eq!(
        graph.input_class("static").unwrap(),
        GraphInputClass::Initializer
    );
}

#[test]
fn test_graph_add_initializer_rejects_non_input_name() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let a = graph.add_input("a", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let b = graph.add_constant(Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap());
    let sum = add_binary_operation(&mut graph, RegistryOperation::Add, [a, b], &output);
    graph.set_outputs(vec![sum]).unwrap();

    assert!(matches!(
        graph.add_initializer(
            "bias",
            Tensor::from_vec(Device::Cpu, &shape, vec![10.0_f32, 20.0]).unwrap(),
        ),
        Err(Error::MissingInput { .. })
    ));
}

#[test]
fn test_graph_compile_rejects_initializer_metadata_mismatch() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    graph
        .add_initializer(
            "x",
            Tensor::from_vec(Device::Cpu, &shape, vec![1_i32, 2]).unwrap(),
        )
        .unwrap();
    graph.set_outputs(vec![input]).unwrap();

    assert!(matches!(
        graph.compile(),
        Err(Error::DataTypeMismatch { name, .. })
            if name == "x"
    ));
}

#[test]
fn test_graph_run_rejects_duplicate_input_names() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let a = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let b = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let sum = add_binary_operation(&mut graph, RegistryOperation::Add, [a, b], &output);
    graph.set_outputs(vec![sum]).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap(),
    );
    assert!(matches!(graph.run(&inputs), Err(Error::DuplicateInput)));
}

#[test]
fn test_graph_compile_rejects_duplicate_input_names() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let a = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let b = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let sum = add_binary_operation(&mut graph, RegistryOperation::Add, [a, b], &output);
    graph.set_outputs(vec![sum]).unwrap();

    let error = graph.compile();
    assert!(matches!(error, Err(Error::DuplicateInput)));
}

#[test]
fn test_graph_compile_rejects_duplicate_input_names_even_if_dead() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let live = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let dead_a = graph.add_input("y", DataType::F32, shape.clone(), Device::Cpu);
    let dead_b = graph.add_input("y", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let live_output =
        add_binary_operation(&mut graph, RegistryOperation::Add, [live, live], &output);
    let _unused = add_binary_operation(
        &mut graph,
        RegistryOperation::Add,
        [dead_a, dead_b],
        &output,
    );
    graph.set_outputs(vec![live_output]).unwrap();

    let error = graph.compile();
    assert!(matches!(error, Err(Error::DuplicateInput)));
}

#[test]
fn test_graph_run_rejects_unsupported_dtype() {
    let mut graph = Graph::new();

    let shape = Shape::create(vec![2]).unwrap();
    let a = graph.add_input("a", DataType::F4E2M1, shape.clone(), Device::Cpu);
    let b = graph.add_input("b", DataType::F4E2M1, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F4E2M1, &shape).unwrap();
    let c = add_binary_operation(&mut graph, RegistryOperation::Add, [a, b], &output);

    graph.set_outputs(vec![c]).unwrap();

    let ta = Tensor::zeroed_on(Device::Cpu, DataType::F4E2M1, &shape).unwrap();
    let tb = Tensor::zeroed_on(Device::Cpu, DataType::F4E2M1, &shape).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), ta);
    inputs.insert("b".to_string(), tb);

    assert!(matches!(
        graph.run(&inputs),
        Err(Error::Unsupported {
            target: UnsupportedTarget::DataType,
            ..
        })
    ));
}

#[test]
fn test_graph_run_rejects_unexpected_input_names() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let a = graph.add_input("a", DataType::F32, shape.clone(), Device::Cpu);
    let b = graph.add_input("b", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let sum = add_binary_operation(&mut graph, RegistryOperation::Add, [a, b], &output);
    graph.set_outputs(vec![sum]).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap(),
    );
    inputs.insert(
        "b".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![3.0_f32, 4.0]).unwrap(),
    );
    inputs.insert(
        "extra".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![5.0_f32, 6.0]).unwrap(),
    );

    let error = graph.run(&inputs);
    assert!(matches!(
        error,
        Err(Error::UnexpectedInput(name)) if name == "extra"
    ));
}
