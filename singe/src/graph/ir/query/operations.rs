use crate::{
    error::{Error, Result},
    graph::ir::{Block, Graph, OperationId, OperationNode, OperationResult, ValueId},
};

impl Graph {
    /// Returns all IR operations.
    #[must_use]
    pub fn operations(&self) -> &[OperationNode] {
        &self.operations
    }

    /// Returns the primary result of each operation in execution order, the order backends lower values in.
    ///
    /// # Errors
    ///
    /// Returns a missing-output error if an operation in the entry block has no result.
    pub fn primary_result_order(&self) -> Result<Vec<ValueId>> {
        self.blocks
            .first()
            .map_or(&[][..], Block::operations)
            .iter()
            .map(|&id| {
                self.required_operation(id)?
                    .result_values()
                    .next()
                    .ok_or_else(|| Error::missing_output("primary result order"))
            })
            .collect()
    }

    /// Returns one IR operation by id.
    #[must_use]
    pub fn operation(&self, id: OperationId) -> Option<&OperationNode> {
        self.operations.get(id.index())
    }

    /// Returns one required IR operation by id.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` does not identify an IR operation.
    pub fn required_operation(&self, id: OperationId) -> Result<&OperationNode> {
        self.operation(id)
            .ok_or_else(|| Error::missing_node("required operation"))
    }

    /// Returns the operation result that produced `value`.
    #[must_use]
    pub fn result_for_value(&self, value: ValueId) -> Option<(OperationId, &OperationResult)> {
        let (operation, result) = self.result_index_for_value(value)?;
        Some((
            operation,
            self.operations
                .get(operation.index())?
                .results()
                .get(result)?,
        ))
    }

    /// Returns the operation that produced `value`.
    #[must_use]
    pub fn operation_for_value(&self, value: ValueId) -> Option<&OperationNode> {
        let (operation, _) = self.result_for_value(value)?;
        self.operation(operation)
    }

    /// Returns the required operation that produced `value`.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if the value has no producing operation or its producer mapping is invalid.
    pub fn required_operation_for_value(&self, value: ValueId) -> Result<&OperationNode> {
        self.operation_for_value(value)
            .ok_or_else(|| Error::missing_node("required operation for value"))
    }

    fn result_index_for_value(&self, value: ValueId) -> Option<(OperationId, usize)> {
        self.value_results.get(value.index()).copied().flatten()
    }
}
