use super::*;

#[test]
fn test_graph_guards_hash_cpu_constant_content() {
    let shape = Shape::create(vec![2]).unwrap();

    let mut first_graph = Graph::new();
    let first_constant = first_graph
        .add_constant(Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap());
    first_graph.set_outputs(vec![first_constant]).unwrap();
    let first_compiled = first_graph.compile().unwrap();
    let first_guards = create_guards_from_ir(&first_compiled);

    let mut second_graph = Graph::new();
    let second_constant = second_graph
        .add_constant(Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 3.0]).unwrap());
    second_graph.set_outputs(vec![second_constant]).unwrap();
    let second_compiled = second_graph.compile().unwrap();
    let second_guards = create_guards_from_ir(&second_compiled);

    let first_guard = &first_guards.constants()[0];
    let second_guard = &second_guards.constants()[0];
    let first_digest = first_guard
        .host_content_digest()
        .expect("expected host-content constant guard");
    let second_digest = second_guard
        .host_content_digest()
        .expect("expected host-content constant guard");

    assert_eq!(first_guard.data_type(), DataType::F32);
    assert_eq!(first_guard.layout().dimensions(), &[2]);
    assert_ne!(first_digest, second_digest);
}

#[test]
fn test_graph_guards_hash_initializer_content() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    graph
        .add_initializer(
            "x",
            Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap(),
        )
        .unwrap();
    graph.set_outputs(vec![input]).unwrap();

    let compiled = graph.compile().unwrap();
    let guards = create_guards_from_ir(&compiled);

    let guard = &guards.constants()[0];
    assert_eq!(guard.data_type(), DataType::F32);
    assert_eq!(guard.layout().dimensions(), &[2]);
    assert!(guard.host_content_digest().is_some());
}

#[test]
fn test_graph_guards_keep_non_contiguous_constant_graph_local() {
    let shape = Shape::create(vec![2, 2]).unwrap();
    let constant = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0, 3.0, 4.0])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let mut graph = Graph::new();
    let constant = graph.add_constant(constant);
    graph.set_outputs(vec![constant]).unwrap();

    let compiled = graph.compile().unwrap();
    let guards = create_guards_from_ir(&compiled);

    let guard = &guards.constants()[0];
    assert_eq!(guard.value(), constant);
    assert_eq!(guard.data_type(), DataType::F32);
    assert_eq!(guard.graph_local_device(), Some(Device::Cpu));
}

#[test]
fn test_graph_guards_keep_partial_storage_constant_graph_local() {
    let storage = CpuStorage::from_bytes(DataType::F32, vec![0_u8; 16]).unwrap();
    let shape = Shape::create(vec![2]).unwrap();
    // SAFETY: the storage was built from values computed for exactly this shape and data type.
    let constant = unsafe { Tensor::from_parts_unchecked(storage, DataType::F32, shape) };
    let mut graph = Graph::new();
    let constant = graph.add_constant(constant);
    graph.set_outputs(vec![constant]).unwrap();

    let compiled = graph.compile().unwrap();
    let guards = create_guards_from_ir(&compiled);

    let guard = &guards.constants()[0];
    assert_eq!(guard.value(), constant);
    assert_eq!(guard.data_type(), DataType::F32);
    assert_eq!(guard.graph_local_device(), Some(Device::Cpu));
}
