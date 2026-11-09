use super::*;

#[test]
fn test_graph_fingerprint_hashes_ordered_duplicate_ir_output_values() {
    fn graph_with_outputs(outputs: impl FnOnce(ValueId, ValueId) -> Vec<ValueId>) -> Graph {
        let shape = Shape::create(vec![2]).unwrap();
        let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
        let mut graph = Graph::new();
        let lhs = graph.add_input("lhs", DataType::F32, shape.clone(), Device::Cpu);
        let rhs = graph.add_input("rhs", DataType::F32, shape, Device::Cpu);
        let add = add_binary_operation(&mut graph, RegistryOperation::Add, [lhs, rhs], &output);
        let sub = add_binary_operation(&mut graph, RegistryOperation::Sub, [lhs, rhs], &output);
        graph.set_outputs(outputs(add, sub)).unwrap();
        graph
    }

    let add_sub_add = graph_with_outputs(|add, sub| vec![add, sub, add]);
    let add_add_sub = graph_with_outputs(|add, sub| vec![add, add, sub]);
    let add_sub = graph_with_outputs(|add, sub| vec![add, sub]);

    assert_ne!(
        add_sub_add.fingerprint().unwrap(),
        add_add_sub.fingerprint().unwrap()
    );
    assert_ne!(
        add_sub_add.fingerprint().unwrap(),
        add_sub.fingerprint().unwrap()
    );
    assert_eq!(
        add_sub_add.fingerprint().unwrap(),
        graph_with_outputs(|add, sub| vec![add, sub, add])
            .fingerprint()
            .unwrap()
    );
}

#[test]
fn test_graph_fingerprint_hashes_schema_index_output_metadata() {
    let input_shape = Shape::create(vec![2, 3]).unwrap();
    let value_shape = Shape::create(vec![2]).unwrap();
    let value_output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &value_shape).unwrap();
    let index_output = Tensor::zeroed_on(Device::Cpu, DataType::I64, &value_shape).unwrap();

    let mut max_graph = Graph::new();
    let input = max_graph.add_input("x", DataType::F32, input_shape.clone(), Device::Cpu);
    let max = add_reduction_operation(
        &mut max_graph,
        RegistryOperation::Max,
        input,
        vec![1],
        false,
        &value_output,
    );
    max_graph.set_outputs(vec![max]).unwrap();

    let mut argmax_graph = Graph::new();
    let input = argmax_graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let argmax = add_dimension_operation(
        &mut argmax_graph,
        RegistryOperation::ArgMax,
        input,
        1,
        &index_output,
    );
    argmax_graph.set_outputs(vec![argmax]).unwrap();

    assert_ne!(
        max_graph.fingerprint().unwrap(),
        argmax_graph.fingerprint().unwrap()
    );
    assert!(argmax_graph.fingerprint().unwrap().is_cacheable());
}
