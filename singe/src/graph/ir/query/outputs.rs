use crate::graph::ir::{Block, Graph, ValueId};

impl Graph {
    /// Returns ordered graph output values from the entry block.
    pub fn output_values(&self) -> &[ValueId] {
        self.blocks.first().map_or(&[], Block::outputs)
    }
}
