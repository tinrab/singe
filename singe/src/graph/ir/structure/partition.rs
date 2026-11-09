use crate::{backend::registry::CudaGraphProvider, graph::ir::structure::OperationId};

/// Identifier for a provider partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PartitionId(usize);

impl PartitionId {
    /// Creates a partition id from its graph-local index.
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the graph-local partition index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

/// Provider partition over graph operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    /// Partition identifier.
    id: PartitionId,
    /// Operations owned by this partition.
    operations: Vec<OperationId>,
    /// Lowering provider assigned to this partition.
    provider: Option<CudaGraphProvider>,
}

impl Partition {
    /// Creates a provider partition.
    #[must_use]
    pub fn new(
        id: PartitionId,
        operations: Vec<OperationId>,
        provider: Option<CudaGraphProvider>,
    ) -> Self {
        Self {
            id,
            operations,
            provider,
        }
    }

    /// Returns the partition identifier.
    #[must_use]
    pub fn id(&self) -> PartitionId {
        self.id
    }

    /// Returns operations owned by this partition.
    #[must_use]
    pub fn operations(&self) -> &[OperationId] {
        &self.operations
    }

    /// Inserts `operation` before `before`, or does nothing when `before` is not in this sequence.
    pub(crate) fn insert_operation_before(&mut self, before: OperationId, operation: OperationId) {
        if let Some(index) = self.operations.iter().position(|&id| id == before) {
            self.operations.insert(index, operation);
        }
    }

    /// Returns the lowering provider assigned to this partition.
    #[must_use]
    pub fn provider(&self) -> Option<CudaGraphProvider> {
        self.provider
    }

    /// Appends an operation to this partition.
    pub fn push_operation(&mut self, operation: OperationId) {
        self.operations.push(operation);
    }
}
