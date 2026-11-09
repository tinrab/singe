use crate::graph::{ir::ValueId, node::NodeRef};

impl NodeRef<'_> {
    /// Returns the input for unary elementwise nodes.
    pub(crate) fn unary_elementwise_input(&self) -> Option<ValueId> {
        let operation = self.operation()?;
        if !operation.is_unary_elementwise() {
            return None;
        }
        self.inputs().first().copied()
    }
}
