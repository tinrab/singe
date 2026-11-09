use super::*;

#[test]
fn test_graph_fingerprint_marks_partial_storage_constant_uncacheable() {
    let storage = CpuStorage::from_bytes(DataType::F32, vec![0_u8; 16]).unwrap();
    let shape = Shape::create(vec![2]).unwrap();
    // SAFETY: the storage was built from values computed for exactly this shape and data type.
    let constant = unsafe { Tensor::from_parts_unchecked(storage, DataType::F32, shape) };
    let mut graph = Graph::new();
    let constant = graph.add_constant(constant);
    graph.set_outputs(vec![constant]).unwrap();

    let fingerprint = graph.fingerprint().unwrap();

    assert!(!fingerprint.is_cacheable());
}

#[test]
fn test_graph_fingerprint_hashes_initializer_content() {
    fn graph_with_initializer(values: Vec<f32>) -> Graph {
        let shape = Shape::create(vec![2]).unwrap();
        let mut graph = Graph::new();
        let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
        graph
            .add_initializer("x", Tensor::from_vec(Device::Cpu, &shape, values).unwrap())
            .unwrap();
        graph.set_outputs(vec![input]).unwrap();
        graph
    }

    let first = graph_with_initializer(vec![1.0, 2.0]);
    let second = graph_with_initializer(vec![1.0, 3.0]);

    assert_ne!(first.fingerprint().unwrap(), second.fingerprint().unwrap());
    assert!(first.fingerprint().unwrap().is_cacheable());
}

#[test]
fn test_graph_fingerprint_marks_partial_storage_initializer_uncacheable() {
    let storage = CpuStorage::from_bytes(DataType::F32, vec![0_u8; 16]).unwrap();
    let shape = Shape::create(vec![2]).unwrap();
    // SAFETY: the storage was built from values computed for exactly this shape and data type.
    let initializer =
        unsafe { Tensor::from_parts_unchecked(storage, DataType::F32, shape.clone()) };
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape, Device::Cpu);
    graph.add_initializer("x", initializer).unwrap();
    graph.set_outputs(vec![input]).unwrap();

    assert!(!graph.fingerprint().unwrap().is_cacheable());
}

#[test]
fn test_graph_fingerprint_ignores_runtime_tensor_values_and_storage_identity() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    graph.set_outputs(vec![input]).unwrap();

    let first = graph.fingerprint().unwrap();
    let first_input = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap();
    let second_input = Tensor::from_vec(Device::Cpu, &shape, vec![10.0_f32, 20.0]).unwrap();

    let _ = graph
        .run(&HashMap::from([("x".to_string(), first_input)]))
        .unwrap();
    let after_first_run = graph.fingerprint().unwrap();
    let _ = graph
        .run(&HashMap::from([("x".to_string(), second_input)]))
        .unwrap();
    let after_second_run = graph.fingerprint().unwrap();

    assert_eq!(first, after_first_run);
    assert_eq!(first, after_second_run);
}
