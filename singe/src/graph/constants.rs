use crate::{
    graph::{Graph, ir::ValueId, node::Node},
    tensor::Tensor,
};

impl Graph {
    /// Adds a named constant node that retains the supplied tensor allocation.
    /// The tensor data is not copied, and the name is stored without a uniqueness check.
    pub(crate) fn add_named_constant(&mut self, name: String, tensor: Tensor) -> ValueId {
        self.push_node(Node::constant_node(tensor).with_name(name))
    }

    /// Adds a constant tensor node.
    pub fn add_constant(&mut self, tensor: Tensor) -> ValueId {
        self.push_node(Node::constant_node(tensor))
    }
}
