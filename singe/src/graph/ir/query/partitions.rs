use crate::graph::ir::{Graph, Partition};

#[cfg(feature = "cuda")]
use crate::{backend::registry::CudaGraphProvider, graph::ir::OperationId, graph::ir::PartitionId};

impl Graph {
    /// Returns provider partitions.
    #[must_use]
    pub fn partitions(&self) -> &[Partition] {
        &self.partitions
    }

    /// Regroups operations into contiguous provider partitions.
    #[cfg(feature = "cuda")]
    pub fn assign_provider_partitions(
        &mut self,
        provider_for_operation: impl FnMut(OperationId) -> Option<CudaGraphProvider>,
    ) {
        self.assign_provider_partitions_with(provider_for_operation);
    }

    /// Regroups operations into contiguous provider partitions from operation-indexed providers.
    #[cfg(feature = "cuda")]
    pub fn assign_provider_partitions_from_provider_list(
        &mut self,
        providers: &[Option<CudaGraphProvider>],
    ) {
        self.assign_provider_partitions(|operation| {
            providers.get(operation.index()).copied().flatten()
        });
    }

    /// Regroups operations into contiguous provider partitions.
    #[cfg(feature = "cuda")]
    fn assign_provider_partitions_with(
        &mut self,
        mut provider_for_operation: impl FnMut(OperationId) -> Option<CudaGraphProvider>,
    ) {
        let mut partitions: Vec<Partition> = Vec::new();
        for block in &self.blocks {
            for &operation in block.operations() {
                let provider = provider_for_operation(operation);
                if let Some(partition) = partitions.last_mut()
                    && partition.provider() == provider
                {
                    partition.push_operation(operation);
                    continue;
                }

                partitions.push(Partition::new(
                    PartitionId::new(partitions.len()),
                    vec![operation],
                    provider,
                ));
            }
        }

        self.partitions = partitions;
    }
}
