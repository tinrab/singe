use super::*;

fn cpu_tensor(shape: &Shape, value: f32) -> Tensor {
    let element_count = usize::try_from(shape.dimensions().iter().product::<i64>()).unwrap();
    Tensor::from_vec(Device::Cpu, shape, vec![value; element_count]).unwrap()
}

fn single_input(name: &str, tensor: Tensor) -> HashMap<String, Tensor> {
    HashMap::from([(name.to_string(), tensor)])
}

fn guarded_identity_graph(axis: usize, name: &'static str) -> (Graph, Shape) {
    let static_shape = Shape::create(vec![2, 3]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, static_shape.clone(), Device::Cpu);
    graph
        .add_input_dimension_guard("x", axis, name, Some(1), Some(8))
        .unwrap();
    graph.set_outputs(vec![input]).unwrap();

    (graph, static_shape)
}

#[test]
fn test_graph_symbolic_input_dimension_allows_runtime_size_change() {
    let static_shape = Shape::create(vec![2, 3]).unwrap();
    let runtime_shape = Shape::create(vec![4, 3]).unwrap();
    let captured_input = cpu_tensor(&static_shape, 1.0);
    let (mut captured, _) =
        Graph::capture([("x", captured_input)], |inputs| -> Result<Vec<Tensor>> {
            Ok(vec![inputs[0].clone()])
        })
        .unwrap();
    captured
        .add_input_dimension_guard("x", 0, "batch", Some(1), Some(8))
        .unwrap();

    let outputs = captured
        .run(&single_input("x", cpu_tensor(&runtime_shape, 1.0)))
        .unwrap();

    assert_eq!(outputs[0].shape().dimensions(), &[4, 3]);
}

#[test]
fn test_graph_symbolic_input_dimension_propagates_to_same_shape_output() {
    let static_shape = Shape::create(vec![2, 3]).unwrap();
    let runtime_shape = Shape::create(vec![4, 3]).unwrap();
    let captured_input = cpu_tensor(&static_shape, 1.0);
    let (mut captured, _) =
        Graph::capture([("x", captured_input)], |inputs| -> Result<Vec<Tensor>> {
            Ok(vec![inputs[0].mul_scalar(2.0)?])
        })
        .unwrap();
    captured
        .add_input_dimension_guard("x", 0, "batch", Some(1), Some(8))
        .unwrap();

    let outputs = captured
        .run(&single_input("x", cpu_tensor(&runtime_shape, 1.0)))
        .unwrap();

    assert_eq!(outputs[0].shape().dimensions(), &[4, 3]);
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[2.0_f32; 12], 1e-6);
}

#[test]
fn test_graph_symbolic_inner_dimension_allows_contiguous_stride_change() {
    let runtime_shape = Shape::create(vec![2, 5]).unwrap();
    let (graph, _) = guarded_identity_graph(1, "features");

    let outputs = graph
        .run(&single_input("x", cpu_tensor(&runtime_shape, 1.0)))
        .unwrap();

    assert_eq!(outputs[0].shape().dimensions(), &[2, 5]);
    assert_eq!(outputs[0].layout().strides(), &[5, 1]);
}

#[test]
fn test_graph_symbolic_input_dimension_enforces_range() {
    let runtime_shape = Shape::create(vec![9, 3]).unwrap();
    let (graph, _) = guarded_identity_graph(0, "batch");

    assert!(matches!(
        graph.run(&single_input("x", cpu_tensor(&runtime_shape, 1.0))),
        Err(Error::OutOfRange { name })
            if name == "x dimension `batch` axis 0 expected range 1..=8 got 9"
    ));

    let compiled = graph.compile().unwrap();
    let guards = create_guards_from_ir(&compiled);
    assert!(matches!(
        guards.check_inputs_with_failure(&single_input("x", cpu_tensor(&runtime_shape, 1.0))),
        Err(GraphGuardFailure::InputDimensionOutOfRange {
            input,
            axis: 0,
            name,
            min: Some(1),
            max: Some(8),
            actual: 9,
        }) if input == "x" && name == "batch"
    ));
}

#[test]
fn test_graph_symbolic_input_dimension_enforces_equality() {
    let shape = Shape::create(vec![2, 3]).unwrap();
    let matching_shape = Shape::create(vec![4, 3]).unwrap();
    let mismatched_shape = Shape::create(vec![5, 3]).unwrap();
    let mut graph = Graph::new();
    let a = graph.add_input("a", DataType::F32, shape.clone(), Device::Cpu);
    let b = graph.add_input("b", DataType::F32, shape, Device::Cpu);
    graph
        .add_input_dimension_guard("a", 0, "batch", Some(1), Some(8))
        .unwrap();
    graph
        .add_input_dimension_guard("b", 0, "batch", Some(1), Some(8))
        .unwrap();
    graph.set_outputs(vec![a, b]).unwrap();

    let outputs = graph
        .run(&HashMap::from([
            ("a".to_string(), cpu_tensor(&matching_shape, 1.0)),
            ("b".to_string(), cpu_tensor(&matching_shape, 2.0)),
        ]))
        .unwrap();
    assert_eq!(outputs[0].shape().dimensions(), &[4, 3]);

    assert!(matches!(
        graph.run(&HashMap::from([
            ("a".to_string(), cpu_tensor(&matching_shape, 1.0)),
            ("b".to_string(), cpu_tensor(&mismatched_shape, 2.0)),
        ])),
        Err(Error::ShapeMismatch { name, expected, actual })
            if name == "symbolic dimension `batch`"
                && expected == vec![4]
                && actual == vec![5]
    ));
}
