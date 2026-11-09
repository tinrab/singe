use crate::{
    error::Result,
    graph::{Graph, GraphOperationRequest, ir::ValueId},
};

impl Graph {
    /// Validates and inserts an operation request with its declared output metadata.
    /// Returns the construction graph node ID.
    ///
    /// # Errors
    ///
    /// Returns an error if node validation, operand lookup, logical operation inference, or declared output metadata checks fail.
    pub fn add_operation_request(&mut self, request: GraphOperationRequest) -> Result<ValueId> {
        self.add_named_node_with_layout(request.into_node())
    }
}
