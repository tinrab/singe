mod metadata;
mod operations;
mod outputs;

use super::*;

fn input_graph(name: &str, data_type: DataType, shape: Shape, device: Device) -> Graph {
    let mut graph = Graph::new();
    let input = graph.add_input(name, data_type, shape, device);
    graph.set_outputs(vec![input]).unwrap();

    graph
}

fn scalar_graph(shape: &Shape, value: f32) -> Graph {
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let scaled = add_scalar_operation(&mut graph, RegistryOperation::Scale, input, value, &output);
    graph.set_outputs(vec![scaled]).unwrap();

    graph
}

fn constant_graph(shape: &Shape, values: Vec<f32>) -> Graph {
    let mut graph = Graph::new();
    let constant = graph.add_constant(Tensor::from_vec(Device::Cpu, shape, values).unwrap());
    graph.set_outputs(vec![constant]).unwrap();

    graph
}
