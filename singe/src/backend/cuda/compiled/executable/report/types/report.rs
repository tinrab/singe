use crate::{
    backend::cuda::compiled::{
        executable::{
            report::types::{
                GraphCompileAliasGroupReport, GraphCompileOperationReport,
                GraphCompilePartitionReport,
            },
            stats::CudaCompiledGraphStats,
        },
        plan::fusion::FusionPlan,
    },
    graph::ir::ValueId,
};

/// Successful CUDA graph compilation report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphCompileReport {
    /// Number of executable graph nodes after graph pruning and normalization.
    pub(crate) graph_nodes: usize,
    /// Number of explicit IR values after graph pruning and normalization.
    pub(crate) graph_values: usize,
    /// Number of explicit IR operations after graph pruning and normalization.
    pub(crate) graph_operations: usize,
    /// Number of explicit IR blocks after graph pruning and normalization.
    pub(crate) graph_blocks: usize,
    /// Number of provider partitions reported by realized lowering.
    pub(crate) graph_partitions: usize,
    /// Number of graph outputs.
    pub(crate) graph_outputs: usize,
    /// Whether the graph fingerprint can be reused by the CUDA graph cache.
    pub(crate) cacheable: bool,
    /// Per-operation lowering candidates, planned decisions, and realized decisions.
    pub(crate) operations: Vec<GraphCompileOperationReport>,
    /// Unique input values constrained by CUDA graph layout policy across all operations.
    pub(crate) layout_constrained_input_values: Vec<ValueId>,
    /// Unique constrained input values whose graph metadata is not contiguous at offset zero.
    pub(crate) non_contiguous_layout_input_values: Vec<ValueId>,
    /// Provider partitions selected directly from operation schema provider lists.
    pub(crate) planned_partitions: Vec<GraphCompilePartitionReport>,
    /// Provider partitions realized during lowering.
    pub(crate) partitions: Vec<GraphCompilePartitionReport>,
    /// Alias groups inferred from graph view/alias relationships.
    pub(crate) alias_groups: Vec<GraphCompileAliasGroupReport>,
    /// Lowering, allocation, workspace, and native graph node statistics.
    pub(crate) stats: CudaCompiledGraphStats,
    /// The operations that share kernels, as the fusion planner grouped them.
    pub(crate) fusion_plan: FusionPlan,
}

impl GraphCompileReport {
    /// Returns the operations that share kernels, as the fusion planner grouped them; its `Display` lists each group.
    #[must_use]
    pub fn fusion_plan(&self) -> &FusionPlan {
        &self.fusion_plan
    }

    /// Returns executable graph nodes after graph pruning and normalization.
    #[must_use]
    pub fn graph_nodes(&self) -> usize {
        self.graph_nodes
    }

    /// Returns explicit IR values after graph pruning and normalization.
    #[must_use]
    pub fn graph_values(&self) -> usize {
        self.graph_values
    }

    /// Returns explicit IR operations after graph pruning and normalization.
    #[must_use]
    pub fn graph_operations(&self) -> usize {
        self.graph_operations
    }

    /// Returns explicit IR blocks after graph pruning and normalization.
    #[must_use]
    pub fn graph_blocks(&self) -> usize {
        self.graph_blocks
    }

    /// Returns provider partitions reported by realized lowering.
    #[must_use]
    pub fn graph_partitions(&self) -> usize {
        self.graph_partitions
    }

    /// Returns graph outputs.
    #[must_use]
    pub fn graph_outputs(&self) -> usize {
        self.graph_outputs
    }

    /// Returns whether the graph fingerprint can be reused by the CUDA graph cache.
    #[must_use]
    pub fn is_cacheable(&self) -> bool {
        self.cacheable
    }

    /// Returns per-operation lowering candidates and decisions.
    #[must_use]
    pub fn operations(&self) -> &[GraphCompileOperationReport] {
        &self.operations
    }

    /// Returns graph input values constrained by any operation's CUDA graph layout policy.
    #[must_use]
    pub fn layout_constrained_input_values(&self) -> &[ValueId] {
        &self.layout_constrained_input_values
    }

    /// Returns constrained input values that are not contiguous at offset zero.
    #[must_use]
    pub fn non_contiguous_layout_input_values(&self) -> &[ValueId] {
        &self.non_contiguous_layout_input_values
    }

    /// Returns provider partitions selected from operation schema provider lists.
    #[must_use]
    pub fn planned_partitions(&self) -> &[GraphCompilePartitionReport] {
        &self.planned_partitions
    }

    /// Returns provider partitions realized during lowering.
    #[must_use]
    pub fn partitions(&self) -> &[GraphCompilePartitionReport] {
        &self.partitions
    }

    /// Returns alias groups inferred from graph view/alias relationships.
    #[must_use]
    pub fn alias_groups(&self) -> &[GraphCompileAliasGroupReport] {
        &self.alias_groups
    }

    /// Returns lowering, allocation, workspace, and native graph node statistics.
    #[must_use]
    pub fn stats(&self) -> CudaCompiledGraphStats {
        self.stats
    }
}
