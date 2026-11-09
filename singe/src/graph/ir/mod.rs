mod infer;
mod nodes;
mod normalize;
mod projection;
mod prune;
mod query;
mod request;
mod structure;
mod validate;
mod value;

use std::collections::{HashMap, HashSet};

pub use self::{
    projection::operation_attributes,
    prune::ValueRemap,
    structure::{
        Block, BlockId, OperationId, OperationNode, OperationResult, Partition, PartitionId,
    },
    value::{
        InitializerInputBinding, Value, ValueAlias, ValueBinding, ValueBindingDetails,
        ValueBindingKind, ValueId,
    },
};
use crate::{
    graph::{
        CaptureInfo, InputDimensionGuard, StateBindingKind, generators::GraphGenerator,
        shape_program::ShapeProgram,
    },
    tensor::Tensor,
};

/// A tensor program of values and the operations that produce them.
///
/// Construction appends one operation per node, and each operation produces one value with the same index.
/// Execution works on the normalized form from `to_ir`, where operations run in dependency order and an operation may produce several results.
#[derive(Debug, Clone)]
pub struct Graph {
    values: Vec<Value>,
    operations: Vec<OperationNode>,
    blocks: Vec<Block>,
    partitions: Vec<Partition>,
    constant_tensors: Vec<Option<Tensor>>,
    value_results: Vec<Option<(OperationId, usize)>>,
    /// Public name of each graph output, in output order.
    output_names: Vec<Option<String>>,
    pub(crate) initializers: HashMap<String, Tensor>,
    pub(crate) static_initializers: HashSet<String>,
    pub(crate) state_kinds: HashMap<String, StateBindingKind>,
    pub(crate) input_dimension_guards: HashMap<(String, usize), InputDimensionGuard>,
    pub(crate) shape_program: ShapeProgram,
    /// Inputs that the trailing outputs, in order, write back after each run.
    pub(crate) input_writes: Vec<String>,
    // Outputs whose leading extent depends on values: each is computed at a bound and narrowed after the run to a count, which a trailing output after the input writes holds, in this order.
    pub(crate) counted_outputs: Vec<usize>,
    /// Generators the graph reads through inputs, supplied and advanced on each run.
    pub(crate) generators: Vec<GraphGenerator>,
    pub(crate) capture_info: Option<CaptureInfo>,
}
