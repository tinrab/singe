use super::*;

#[test]
fn test_graph_fingerprint_ignores_pruned_dead_node_ids() {
    fn live_scale_graph(with_dead_prefix: bool) -> Graph {
        let shape = Shape::create(vec![2]).unwrap();
        let mut graph = Graph::new();
        if with_dead_prefix {
            let dead =
                graph.add_constant(Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap());
            let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
            let _dead_output =
                add_scalar_operation(&mut graph, RegistryOperation::Scale, dead, 99.0, &output);
        }
        let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
        let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
        let scaled =
            add_scalar_operation(&mut graph, RegistryOperation::Scale, input, 2.0, &output);
        graph.set_outputs(vec![scaled]).unwrap();
        graph
    }

    fn live_graph_local_constant_graph(with_dead_prefix: bool) -> Graph {
        let shape = Shape::create(vec![2, 2]).unwrap();
        let constant = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0, 3.0, 4.0])
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let mut graph = Graph::new();
        if with_dead_prefix {
            let dead =
                graph.add_constant(Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap());
            let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
            let _dead_output =
                add_scalar_operation(&mut graph, RegistryOperation::Scale, dead, 99.0, &output);
        }
        let constant = graph.add_constant(constant);
        graph.set_outputs(vec![constant]).unwrap();
        graph
    }

    assert_eq!(
        live_scale_graph(false).fingerprint().unwrap(),
        live_scale_graph(true).fingerprint().unwrap()
    );
    let graph_local = live_graph_local_constant_graph(false)
        .fingerprint()
        .unwrap();
    assert!(!graph_local.is_cacheable());
    assert_eq!(
        graph_local,
        live_graph_local_constant_graph(true).fingerprint().unwrap()
    );
}

#[test]
fn test_graph_fingerprint_hashes_symbolic_dimension_instead_of_static_size() {
    let mut graph_a = Graph::new();
    let shape_a = Shape::create(vec![2, 3]).unwrap();
    let input_a = graph_a.add_input("x", DataType::F32, shape_a, Device::Cpu);
    graph_a
        .add_input_dimension_guard("x", 0, "batch", Some(1), Some(8))
        .unwrap();
    graph_a.set_outputs(vec![input_a]).unwrap();

    let mut graph_b = Graph::new();
    let shape_b = Shape::create(vec![5, 3]).unwrap();
    let input_b = graph_b.add_input("x", DataType::F32, shape_b, Device::Cpu);
    graph_b
        .add_input_dimension_guard("x", 0, "batch", Some(1), Some(8))
        .unwrap();
    graph_b.set_outputs(vec![input_b]).unwrap();

    let mut graph_c = graph_b.clone();
    graph_c
        .add_input_dimension_guard("x", 0, "batch", Some(1), Some(16))
        .unwrap();

    assert_eq!(
        graph_a.fingerprint().unwrap(),
        graph_b.fingerprint().unwrap()
    );
    assert_ne!(
        graph_a.fingerprint().unwrap(),
        graph_c.fingerprint().unwrap()
    );
}

#[test]
fn test_graph_fingerprint_hashes_symbolic_inner_dimension_instead_of_contiguous_stride() {
    let mut graph_a = Graph::new();
    let shape_a = Shape::create(vec![2, 3]).unwrap();
    let input_a = graph_a.add_input("x", DataType::F32, shape_a, Device::Cpu);
    graph_a
        .add_input_dimension_guard("x", 1, "features", Some(1), Some(8))
        .unwrap();
    graph_a.set_outputs(vec![input_a]).unwrap();

    let mut graph_b = Graph::new();
    let shape_b = Shape::create(vec![2, 5]).unwrap();
    let input_b = graph_b.add_input("x", DataType::F32, shape_b, Device::Cpu);
    graph_b
        .add_input_dimension_guard("x", 1, "features", Some(1), Some(8))
        .unwrap();
    graph_b.set_outputs(vec![input_b]).unwrap();

    assert_eq!(
        graph_a.fingerprint().unwrap(),
        graph_b.fingerprint().unwrap()
    );
}

#[test]
fn test_graph_fingerprint_propagates_symbolic_dimension_to_same_shape_output() {
    fn captured_scale_graph_fingerprint(batch: i64, max: i64) -> GraphFingerprint {
        let shape = Shape::create(vec![batch, 3]).unwrap();
        let element_count = usize::try_from(batch).unwrap().checked_mul(3).unwrap();
        let input = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32; element_count]).unwrap();
        let (mut captured, _) = Graph::capture([("x", input)], |inputs| -> Result<Vec<Tensor>> {
            Ok(vec![inputs[0].mul_scalar(2.0)?])
        })
        .unwrap();
        captured
            .add_input_dimension_guard("x", 0, "batch", Some(1), Some(max))
            .unwrap();
        captured.fingerprint().unwrap()
    }

    let fingerprint_a = captured_scale_graph_fingerprint(2, 8);
    let fingerprint_b = captured_scale_graph_fingerprint(5, 8);
    let fingerprint_c = captured_scale_graph_fingerprint(5, 16);

    assert_eq!(fingerprint_a, fingerprint_b);
    assert_ne!(fingerprint_a, fingerprint_c);
}
