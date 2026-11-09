use crate::{
    backend::registry::Operation as RegistryOperation,
    graph::{node::NodeKind, test_support::*},
};

#[test]
fn test_graph_add() {
    let mut graph = Graph::new();

    let shape = Shape::create(vec![2, 2]).unwrap();
    let a = graph.add_input("a", DataType::F32, shape.clone(), Device::Cpu);
    let b = graph.add_input("b", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let c = add_binary_operation(&mut graph, RegistryOperation::Add, [a, b], &output);

    graph.set_outputs(vec![c]).unwrap();

    let ta = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let tb = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), ta);
    inputs.insert("b".to_string(), tb);

    let outputs = graph.run(&inputs).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape().dimensions(), &[2, 2]);
}

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

#[test]
fn test_graph_run_concat_matches_cpu_registry_support() {
    let mut graph = Graph::new();

    let shape = Shape::create(vec![1]).unwrap();
    let output_shape = Shape::create(vec![2]).unwrap();
    let a = graph.add_input("a", DataType::F32, shape.clone(), Device::Cpu);
    let b = graph.add_input("b", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let concat = graph
        .add_named_operation(
            None,
            NodeKind::concat(vec![a, b], 0),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.set_outputs(vec![concat]).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32]).unwrap(),
    );
    inputs.insert(
        "b".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![2.0_f32]).unwrap(),
    );

    let outputs = graph.run(&inputs).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape().dimensions(), &[2]);
    assert_eq!(outputs[0].to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn test_graph_topo_sort() {
    let mut graph = Graph::new();
    let shape = Shape::create(vec![2]).unwrap();
    let a = graph.add_input("a", DataType::F32, shape.clone(), Device::Cpu);
    let b = graph.add_input("b", DataType::F32, shape.clone(), Device::Cpu);
    let c_out = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let c = add_binary_operation(&mut graph, RegistryOperation::Add, [a, b], &c_out);
    let d_out = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let d = add_unary_operation(&mut graph, RegistryOperation::Relu, c, &d_out);

    graph.set_outputs(vec![d]).unwrap();

    let order = graph.topological_sort().unwrap();
    assert_eq!(order.len(), 4);
    assert!(
        order.iter().position(|&id| id == a).unwrap()
            < order.iter().position(|&id| id == c).unwrap()
    );
    assert!(
        order.iter().position(|&id| id == c).unwrap()
            < order.iter().position(|&id| id == d).unwrap()
    );
}
