mod allocation;
mod arena;
mod create;
mod stats;

use std::collections::{HashMap, HashSet};

use singe_cuda::{graph::GraphNode, view::AllocationId};

use self::allocation::GraphReusableBuffer;

pub use self::arena::GraphArena;

use crate::{
    backend::cuda::graph_value::CudaGraphValue, data_type::DataType, graph::ir::ValueId,
    layout::Layout,
};

/// Data type and exact layout used to group compatible graph reuse buffers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryPlanKey {
    data_type: DataType,
    layout: Layout,
}

/// Tracks CUDA graph buffer allocation, alias-group lifetimes, and reuse dependencies.
/// Protected input, constant, and output groups cannot enter the planner's free pool.
#[derive(Debug)]
pub struct GraphMemoryPlanner {
    dependencies: HashMap<ValueId, Vec<ValueId>>,
    release_use: HashMap<ValueId, usize>,
    release_group: HashMap<ValueId, ValueId>,
    protected_groups: HashSet<ValueId>,
    protected_input_groups: HashSet<ValueId>,
    protected_constant_groups: HashSet<ValueId>,
    protected_output_groups: HashSet<ValueId>,
    released_groups: HashSet<ValueId>,
    /// Graph nodes that read each alias group, all of which must finish before its storage is written again.
    group_readers: HashMap<ValueId, Vec<GraphNode>>,
    free: HashMap<MemoryPlanKey, Vec<GraphReusableBuffer>>,
    reused_buffers: usize,
    allocated_bytes: usize,
    reused_buffer_bytes: usize,
    /// The buffers this graph shares with the other specializations of its program, and those it holds already.
    arena: Option<GraphArena>,
    arena_taken: Vec<AllocationId>,
}

impl MemoryPlanKey {
    pub(crate) fn new(data_type: DataType, layout: Layout) -> Self {
        Self { data_type, layout }
    }

    pub(crate) fn data_type(&self) -> DataType {
        self.data_type
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }

    pub(crate) fn for_tensor(tensor: &CudaGraphValue) -> Self {
        Self::new(tensor.data_type(), tensor.layout().clone())
    }
}

pub fn graph_alias_group_root(parent: &HashMap<ValueId, ValueId>, id: ValueId) -> ValueId {
    let mut root = id;
    while let Some(&next) = parent.get(&root) {
        if next == root {
            return root;
        }

        root = next;
    }
    root
}

pub fn union_cuda_graph_alias_group(
    parent: &mut HashMap<ValueId, ValueId>,
    a: ValueId,
    b: ValueId,
) {
    let a_root = graph_alias_group_root(parent, a);
    let b_root = graph_alias_group_root(parent, b);
    if a_root != b_root {
        parent.insert(a_root, b_root);
    }
}
