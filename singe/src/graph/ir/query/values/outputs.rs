#[cfg(feature = "cuda")]
use std::collections::HashSet;

use crate::graph::ir::Graph;
#[cfg(feature = "cuda")]
use crate::graph::ir::ValueId;

impl Graph {
    /// Returns entry-block output values as a set.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn output_value_set(&self) -> HashSet<ValueId> {
        self.output_values().iter().copied().collect()
    }

    /// Returns whether the entry block has graph outputs.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn has_outputs(&self) -> bool {
        !self.output_values().is_empty()
    }

    /// Returns the number of entry-block graph outputs.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn output_value_count(&self) -> usize {
        self.output_values().len()
    }

    /// Returns entry-block output values as owned ids.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn output_values_vec(&self) -> Vec<ValueId> {
        self.output_values().to_vec()
    }

    /// Visits entry-block output values in graph output order.
    #[cfg(feature = "cuda")]
    pub fn for_each_output_value(&self, mut visit: impl FnMut(ValueId)) {
        for &value in self.output_values() {
            visit(value);
        }
    }
}
