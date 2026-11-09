mod execution;
mod quantization;
mod quantized_matmul;

use crate::{
    backend::registry::Operation as RegistryOperation,
    graph::{CompiledGraph, test_support::*},
};

fn add_dead_scale(graph: &mut Graph) {
    let dead_shape = Shape::create([2]).unwrap();
    let dead_input = graph.add_input("dead", DataType::F32, dead_shape.clone(), Device::Cpu);
    let dead_output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &dead_shape).unwrap();
    let _dead = add_scalar_operation(
        graph,
        RegistryOperation::Scale,
        dead_input,
        2.0,
        &dead_output,
    );
}

fn compile_and_assert_pruned_order(graph: &Graph, live_nodes: usize) -> CompiledGraph {
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.value_order().len(), live_nodes);

    compiled
}
