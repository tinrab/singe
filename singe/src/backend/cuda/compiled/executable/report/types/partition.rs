use crate::{
    backend::{
        cuda::compiled::{
            executable::report::types::{
                GraphCompileInputLayoutRequirement, input_layout_requirement_for_support,
            },
            plan::lowering::CudaGraphProvider,
        },
        registry::LayoutSupport,
    },
    graph::ir::{OperationId, PartitionId, ValueId},
};

pub struct GraphCompileIrPartitionFields {
    pub(crate) operations: Vec<OperationId>,
    pub(crate) layout_support: LayoutSupport,
    pub(crate) input_values: Vec<ValueId>,
    pub(crate) layout_constrained_input_values: Vec<ValueId>,
    pub(crate) non_contiguous_layout_input_values: Vec<ValueId>,
    pub(crate) internal_edges: Vec<(ValueId, OperationId)>,
    pub(crate) boundary_output_values: Vec<ValueId>,
    pub(crate) graph_output_values: Vec<ValueId>,
    pub(crate) output_values: Vec<ValueId>,
}

/// A provider partition, as planned from operation support metadata or as lowering realized it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphCompilePartitionReport {
    /// Partition id in planning or lowering order.
    pub(crate) partition: PartitionId,
    /// Provider selected for this partition.
    pub(crate) provider: CudaGraphProvider,
    /// Strongest layout policy required by operations in this partition.
    pub(crate) layout_support: LayoutSupport,
    /// IR operations in this partition.
    pub(crate) operations: Vec<OperationId>,
    /// IR values consumed by this partition and produced outside the partition.
    pub(crate) input_values: Vec<ValueId>,
    /// Input values whose layout is constrained by any operation in this partition.
    pub(crate) layout_constrained_input_values: Vec<ValueId>,
    /// Layout-constrained input values whose graph metadata is not contiguous at offset zero.
    pub(crate) non_contiguous_layout_input_values: Vec<ValueId>,
    /// Producer value to consumer operation edges inside this partition.
    pub(crate) internal_edges: Vec<(ValueId, OperationId)>,
    /// IR values produced by this partition and consumed outside it or returned as graph outputs.
    pub(crate) boundary_output_values: Vec<ValueId>,
    /// IR values produced by this partition and returned as graph outputs.
    pub(crate) graph_output_values: Vec<ValueId>,
    /// IR values produced by this partition.
    pub(crate) output_values: Vec<ValueId>,
}

impl GraphCompilePartitionReport {
    /// Returns the partition id in realized lowering order.
    #[must_use]
    pub fn partition(&self) -> PartitionId {
        self.partition
    }

    /// Returns the provider selected for this partition.
    #[must_use]
    pub fn provider(&self) -> CudaGraphProvider {
        self.provider
    }

    /// Returns the strongest layout policy required by this partition.
    #[must_use]
    pub fn layout_support(&self) -> LayoutSupport {
        self.layout_support
    }

    /// Returns the strongest input layout required by this partition.
    #[must_use]
    pub fn input_layout_requirement(&self) -> GraphCompileInputLayoutRequirement {
        input_layout_requirement_for_support(self.layout_support)
    }

    /// Returns IR operations in this partition.
    #[must_use]
    pub fn operations(&self) -> &[OperationId] {
        &self.operations
    }

    /// Returns IR values consumed by this partition and produced outside it.
    #[must_use]
    pub fn input_values(&self) -> &[ValueId] {
        &self.input_values
    }

    /// Returns input values whose layout is constrained by this partition.
    #[must_use]
    pub fn layout_constrained_input_values(&self) -> &[ValueId] {
        &self.layout_constrained_input_values
    }

    /// Returns layout-constrained inputs that are not contiguous at offset zero.
    #[must_use]
    pub fn non_contiguous_layout_input_values(&self) -> &[ValueId] {
        &self.non_contiguous_layout_input_values
    }

    /// Returns producer value to consumer operation edges inside this partition.
    #[must_use]
    pub fn internal_edges(&self) -> &[(ValueId, OperationId)] {
        &self.internal_edges
    }

    /// Returns values produced by this partition and consumed outside it or returned.
    #[must_use]
    pub fn boundary_output_values(&self) -> &[ValueId] {
        &self.boundary_output_values
    }

    /// Returns values produced by this partition and returned as graph outputs.
    #[must_use]
    pub fn graph_output_values(&self) -> &[ValueId] {
        &self.graph_output_values
    }

    /// Returns IR values produced by this partition.
    #[must_use]
    pub fn output_values(&self) -> &[ValueId] {
        &self.output_values
    }
}
