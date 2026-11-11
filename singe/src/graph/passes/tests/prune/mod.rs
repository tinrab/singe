mod execution;

use crate::{
    backend::registry::Operation as RegistryOperation,
    graph::{CompiledGraph, test_support::*},
};

fn compile_and_assert_pruned_order(graph: &Graph, live_nodes: usize) -> CompiledGraph {
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.value_order().len(), live_nodes);

    compiled
}
