mod constants;
mod inputs;
mod pruning;
mod symbolic;

use crate::{
    backend::registry::Operation as RegistryOperation,
    graph::{CompiledGraph, test_support::*},
};

fn create_guards_from_ir(compiled: &CompiledGraph) -> GraphGuards {
    GraphGuards::create_from_ir(compiled.ir()).unwrap()
}
