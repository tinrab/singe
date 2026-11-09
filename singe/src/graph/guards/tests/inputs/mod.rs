mod initializers;
mod metadata;
mod reachable;
mod storage;

use super::*;

fn single_input(name: &str, tensor: Tensor) -> HashMap<String, Tensor> {
    HashMap::from([(name.to_string(), tensor)])
}

fn add_f32_initializer(graph: &mut Graph, name: &str, shape: &Shape, values: Vec<f32>) {
    graph
        .add_initializer(name, cpu_tensor(shape, values))
        .unwrap();
}

fn add_static_f32_initializer(graph: &mut Graph, name: &str, shape: &Shape, values: Vec<f32>) {
    graph
        .add_static_initializer(name, cpu_tensor(shape, values))
        .unwrap();
}
