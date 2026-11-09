mod build;
mod kind;
mod metadata;
mod query;
mod rewrite;
mod view;

pub(crate) use self::kind::NodeKind;
pub use self::view::{NodeRef, NodeRefKind};

use crate::{data_type::DataType, device::Device, layout::Layout};

/// Typed, shaped operation node stored in a graph.
#[derive(Debug, Clone)]
pub struct Node {
    /// Operation represented by this node.
    kind: NodeKind,
    /// Optional graph-local name.
    name: Option<String>,
    source_location: Option<String>,
    /// Output data type produced by this node.
    output_data_type: DataType,
    /// Output device produced by this node.
    output_device: Device,
    /// Output layout produced by this node.
    output_layout: Layout,
}

#[cfg(test)]
mod tests;
