use super::*;

#[test]
fn test_graph_topo_sort_rejects_missing_dependency() {
    let mut graph = Graph::new();
    let shape = Shape::create(vec![2]).unwrap();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();

    assert!(matches!(
        graph.add_operation_request(GraphOperationRequest::binary(
            None,
            RegistryOperation::Add,
            [input, ValueId::new(99)],
            graph_operation_output(&output),
        )),
        Err(Error::MissingNode { .. })
    ));
}

#[test]
fn test_graph_set_outputs_rejects_missing_output_node() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    graph.add_input("x", DataType::F32, shape, Device::Cpu);

    assert!(matches!(
        graph.set_outputs(vec![ValueId::new(99)]),
        Err(Error::MissingNode { .. })
    ));
}
