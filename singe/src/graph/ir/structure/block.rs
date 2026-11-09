use crate::graph::ir::{ValueId, structure::operation::OperationId};

/// Identifier for an ordered graph block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(usize);

impl BlockId {
    /// Creates a block id from its graph-local index.
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the graph-local block index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

/// Ordered operation block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// Block identifier.
    id: BlockId,
    /// Operations in dependency order.
    operations: Vec<OperationId>,
    /// Values returned by this block.
    outputs: Vec<ValueId>,
}

impl Block {
    /// Creates an operation block.
    #[must_use]
    pub fn new(id: BlockId, operations: Vec<OperationId>, outputs: Vec<ValueId>) -> Self {
        Self {
            id,
            operations,
            outputs,
        }
    }

    /// Returns the block identifier.
    #[must_use]
    pub fn id(&self) -> BlockId {
        self.id
    }

    /// Returns operations in dependency order.
    #[must_use]
    pub fn operations(&self) -> &[OperationId] {
        &self.operations
    }

    /// Returns values returned by this block.
    #[must_use]
    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    pub(crate) fn push_operation(&mut self, operation: OperationId) {
        self.operations.push(operation);
    }

    pub(crate) fn set_outputs(&mut self, outputs: Vec<ValueId>) {
        self.outputs = outputs;
    }

    /// Inserts `operation` before `before`, or does nothing when `before` is not in this sequence.
    pub(crate) fn insert_operation_before(&mut self, before: OperationId, operation: OperationId) {
        if let Some(index) = self.operations.iter().position(|&id| id == before) {
            self.operations.insert(index, operation);
        }
    }
}
