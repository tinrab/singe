use std::{collections::HashSet, iter};

use crate::{
    error::{Error, Result},
    graph::prepare::{GraphExecutor, PreparedBackend, PreparedPlan},
    tensor::Tensor,
};

/// Allocation bytes retained by this instance's active and cached plans.
/// State and literal storage is deduplicated by retained storage identity.
/// Driver resources, provider caches, allocator free blocks, and caller-owned output snapshots are outside these instance totals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[expect(
    clippy::struct_field_names,
    reason = "bytes records the unit of every memory statistic"
)]
pub struct GraphExecutionMemoryStats {
    bound_allocation_bytes: usize,
    shared_state_allocation_bytes: usize,
    active_workspace_bytes: usize,
    inactive_workspace_bytes: usize,
    active_replay_allocation_bytes: usize,
    inactive_replay_allocation_bytes: usize,
}

impl GraphExecutionMemoryStats {
    /// State and literals, including those retained only by inactive plans.
    #[must_use]
    pub fn bound_allocation_bytes(self) -> usize {
        self.bound_allocation_bytes
    }

    /// State's share of `bound_allocation_bytes`, potentially shared by instances.
    #[must_use]
    pub fn shared_state_allocation_bytes(self) -> usize {
        self.shared_state_allocation_bytes
    }

    /// Returns native workspace bytes retained by the active plan.
    #[must_use]
    pub fn active_workspace_bytes(self) -> usize {
        self.active_workspace_bytes
    }

    /// Returns summed native workspace bytes retained by inactive cached plans.
    #[must_use]
    pub fn inactive_workspace_bytes(self) -> usize {
        self.inactive_workspace_bytes
    }

    /// Returns replay-storage bytes retained by the active plan.
    #[must_use]
    pub fn active_replay_allocation_bytes(self) -> usize {
        self.active_replay_allocation_bytes
    }

    /// Returns summed replay-storage bytes retained by inactive cached plans.
    #[must_use]
    pub fn inactive_replay_allocation_bytes(self) -> usize {
        self.inactive_replay_allocation_bytes
    }

    /// Instance-owned storage, counting shared state/literals once in this instance.
    ///
    /// # Errors
    ///
    /// Returns an error if summing bound, workspace, and replay-storage byte totals overflows `usize`.
    pub fn retained_allocation_bytes(self) -> Result<usize> {
        [
            self.active_workspace_bytes,
            self.inactive_workspace_bytes,
            self.active_replay_allocation_bytes,
            self.inactive_replay_allocation_bytes,
        ]
        .into_iter()
        .try_fold(self.bound_allocation_bytes, checked_sum)
    }
}

impl GraphExecutor {
    /// Accounts for retained state, literals, workspaces, and replay storage across active and inactive plans.
    ///
    /// Tensor storage identities are deduplicated.
    /// Provider caches, driver resources, free allocator blocks, and caller-owned output snapshots are excluded.
    ///
    /// # Errors
    ///
    /// Returns an error if lowering retained logical metadata fails or summing retained allocation bytes overflows `usize`.
    pub fn memory_stats(&self) -> Result<GraphExecutionMemoryStats> {
        let mut allocations = HashSet::new();
        let mut stats = GraphExecutionMemoryStats::default();
        for (_, tensor) in self.state.bindings() {
            retain_tensor(tensor, &mut allocations, &mut stats.bound_allocation_bytes)?;
        }

        stats.shared_state_allocation_bytes = stats.bound_allocation_bytes;
        for graph in [self.program.logical_graph(), &self.template] {
            for tensor in graph
                .values()
                .iter()
                .filter_map(|value| graph.constant_tensor(value.id()))
            {
                retain_tensor(tensor, &mut allocations, &mut stats.bound_allocation_bytes)?;
            }
        }

        for compiled in self
            .plans()
            .map(|plan| &plan.compiled)
            .chain(self.program.retained_analyses())
        {
            for tensor in compiled
                .signature()
                .inputs()
                .iter()
                .filter_map(|input| input.initializer.as_ref())
                .chain(
                    compiled
                        .ir()
                        .values()
                        .iter()
                        .filter_map(|value| compiled.ir().constant_tensor(value.id())),
                )
            {
                retain_tensor(tensor, &mut allocations, &mut stats.bound_allocation_bytes)?;
            }
        }

        (
            stats.active_workspace_bytes,
            stats.active_replay_allocation_bytes,
        ) = native_bytes(&self.active);
        for entry in &self.inactive {
            let (workspace, replay) = native_bytes(&entry.plan);
            stats.inactive_workspace_bytes =
                checked_sum(stats.inactive_workspace_bytes, workspace)?;
            stats.inactive_replay_allocation_bytes =
                checked_sum(stats.inactive_replay_allocation_bytes, replay)?;
        }
        Ok(stats)
    }

    fn plans(&self) -> impl Iterator<Item = &PreparedPlan> {
        iter::once(&self.active).chain(self.inactive.iter().map(|entry| &entry.plan))
    }
}

fn retain_tensor(
    tensor: &Tensor,
    allocations: &mut HashSet<usize>,
    bytes: &mut usize,
) -> Result<()> {
    if allocations.insert(tensor.storage().storage_identity()) {
        *bytes = checked_sum(*bytes, tensor.storage().allocation_byte_size())?;
    }
    Ok(())
}

fn native_bytes(plan: &PreparedPlan) -> (usize, usize) {
    match &plan.backend {
        PreparedBackend::Cpu => (0, 0),
        #[cfg(feature = "cuda")]
        PreparedBackend::CudaEager { .. } => (0, 0),
        #[cfg(feature = "cuda")]
        PreparedBackend::Cuda { graph, .. } => (
            graph.library_workspace_bytes(),
            graph.graph_allocation_bytes(),
        ),
    }
}

fn checked_sum(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| Error::out_of_range("graph retained allocation bytes"))
}
