use super::*;

#[test]
fn test_graph_add_node_rejects_declared_output_shape_mismatch() {
    let input_shape = Shape::create(vec![2]).unwrap();
    let output_shape = Shape::create(vec![3]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();

    assert!(matches!(
        graph.add_reshape_with_declared_output_for_test(input, input_shape, &output),
        Err(Error::ShapeMismatch { name, expected, actual })
            if name == "graph add_node node 1 shape"
                && expected == vec![2]
                && actual == vec![3]
    ));
}
