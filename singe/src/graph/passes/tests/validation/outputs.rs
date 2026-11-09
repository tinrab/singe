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
fn test_graph_add_unary_rejects_missing_node_dependency() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();

    assert!(matches!(
        graph.add_operation_request(GraphOperationRequest::scalar(
            None,
            RegistryOperation::Scale,
            ValueId::new(99),
            2.0,
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

#[test]
fn test_graph_validate_outputs_rejects_missing_ir_output_value() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape, Device::Cpu);
    graph.set_outputs(vec![input]).unwrap();
    graph.set_output_projection_unchecked_for_test(ValueId::new(99));

    let error = GraphPassPipeline::new(vec![GraphPassKind::ValidateOutputs])
        .normalize(&graph)
        .unwrap_err();

    assert!(matches!(
        error,
        Error::Unsupported { operation, details, .. }
            if operation == "graph pass validate_outputs"
                && details.contains("missing graph output node 99")
    ));
}
