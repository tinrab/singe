mod operations;
mod outputs;
mod partitions;
mod program;
mod values;

use crate::graph::ir::{Block, Graph};

impl Graph {
    /// Returns ordered IR blocks.
    #[must_use]
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }
}
