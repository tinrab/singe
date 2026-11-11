use crate::{backend::registry::Operation as RegistryOperation, graph::test_support::*};

#[test]
fn test_graph_run_preserves_duplicate_output_order() {
    let mut graph = Graph::new();

    let shape = Shape::create(vec![2]).unwrap();
    let a = graph.add_input("a", DataType::F32, shape.clone(), Device::Cpu);
    let b = graph.add_input("b", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let add = add_binary_operation(&mut graph, RegistryOperation::Add, [a, b], &output);
    let relu = add_unary_operation(&mut graph, RegistryOperation::Relu, add, &output);
    graph.set_outputs(vec![relu, add, relu]).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![-2.0_f32, 1.0]).unwrap(),
    );
    inputs.insert(
        "b".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap(),
    );

    let outputs = graph.run(&inputs).unwrap();
    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].to_vec::<f32>().unwrap(), vec![0.0, 3.0]);
    assert_eq!(outputs[1].to_vec::<f32>().unwrap(), vec![-1.0, 3.0]);
    assert_eq!(outputs[2].to_vec::<f32>().unwrap(), vec![0.0, 3.0]);
}
