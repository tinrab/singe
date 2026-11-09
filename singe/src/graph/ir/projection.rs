use crate::{backend::registry::OperationAttributes, graph::node::NodeRef};

/// Borrows the canonical typed attributes declared by a construction graph node.
#[must_use]
pub fn operation_attributes(node: &NodeRef<'_>) -> OperationAttributes {
    node.attributes()
}
