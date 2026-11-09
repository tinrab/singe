use crate::{
    backend::{
        cuda::compiled::executable::report::types::{
            GraphCompileInputLayoutRequirement, GraphCompileValueMaterialization,
            input_layout_requirement_for_support,
            provider::{
                GraphCompileProviderCandidate, GraphCompileProviderDecision,
                GraphCompileProviderFailure,
            },
        },
        registry::LayoutSupport,
    },
    data_type::DataType,
    graph::ir::{OperationId, ValueId},
    layout::Layout,
};

/// Per-operation CUDA graph lowering details.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphCompileOperationReport {
    /// Explicit IR operation id corresponding to this report.
    pub(crate) operation_id: OperationId,
    /// Explicit IR input values consumed by this operation.
    pub(crate) input_values: Vec<ValueId>,
    /// Data type metadata for `input_values`.
    pub(crate) input_data_types: Vec<DataType>,
    /// Layout metadata for `input_values`.
    pub(crate) input_layouts: Vec<Layout>,
    /// Explicit IR output values produced by this operation.
    pub(crate) output_values: Vec<ValueId>,
    /// Data type metadata for `output_values`.
    pub(crate) output_data_types: Vec<DataType>,
    /// Layout metadata for `output_values`.
    pub(crate) output_layouts: Vec<Layout>,
    /// Source value aliased by this operation, when the operation may alias another graph value.
    pub(crate) alias_input_value: Option<ValueId>,
    /// Root value of the alias group that contains this operation output, when it aliases.
    pub(crate) alias_group_root_value: Option<ValueId>,
    /// Human-readable operation name.
    pub(crate) operation: &'static str,
    /// Registry layout behavior required by the selected operation.
    pub(crate) layout_support: LayoutSupport,
    /// Input values whose layout is constrained by this operation's CUDA graph layout policy.
    pub(crate) layout_constrained_input_values: Vec<ValueId>,
    /// Layout-constrained input values whose graph metadata is not contiguous at offset zero.
    pub(crate) non_contiguous_layout_input_values: Vec<ValueId>,
    /// How this operation's output is represented in the compiled CUDA graph.
    pub(crate) output_materialization: GraphCompileValueMaterialization,
    /// Providers that advertised support for this operation under the active lowering policy.
    pub(crate) candidates: Vec<GraphCompileProviderCandidate>,
    /// Planned provider decision from the lowering planner.
    pub(crate) planned_decision: Option<GraphCompileProviderDecision>,
    /// Provider that actually materialized in the native CUDA graph lowering.
    pub(crate) realized_decision: Option<GraphCompileProviderDecision>,
    /// Provider algorithm or execution plan selected for this operation.
    pub(crate) selected_algorithm: Option<String>,
    /// Provider attempts that failed before a rejected candidate materialized the node.
    pub(crate) provider_failures: Vec<GraphCompileProviderFailure>,
}

impl GraphCompileOperationReport {
    /// Returns the explicit IR operation id corresponding to this report.
    #[must_use]
    pub fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    /// Returns explicit IR input values consumed by this operation.
    #[must_use]
    pub fn input_values(&self) -> &[ValueId] {
        &self.input_values
    }

    /// Returns data type metadata for input values.
    #[must_use]
    pub fn input_data_types(&self) -> &[DataType] {
        &self.input_data_types
    }

    /// Returns layout metadata for input values.
    #[must_use]
    pub fn input_layouts(&self) -> &[Layout] {
        &self.input_layouts
    }

    /// Returns explicit IR output values produced by this operation.
    #[must_use]
    pub fn output_values(&self) -> &[ValueId] {
        &self.output_values
    }

    /// Returns data type metadata for output values.
    #[must_use]
    pub fn output_data_types(&self) -> &[DataType] {
        &self.output_data_types
    }

    /// Returns layout metadata for output values.
    #[must_use]
    pub fn output_layouts(&self) -> &[Layout] {
        &self.output_layouts
    }

    /// Returns the source value aliased by this operation, when any.
    #[must_use]
    pub fn alias_input_value(&self) -> Option<ValueId> {
        self.alias_input_value
    }

    /// Returns the root value of the alias group containing this output, when any.
    #[must_use]
    pub fn alias_group_root_value(&self) -> Option<ValueId> {
        self.alias_group_root_value
    }

    /// Returns the human-readable operation name.
    #[must_use]
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// Returns registry layout behavior required by the selected operation.
    #[must_use]
    pub fn layout_support(&self) -> LayoutSupport {
        self.layout_support
    }

    /// Returns the input layout required by this operation's CUDA graph provider.
    #[must_use]
    pub fn input_layout_requirement(&self) -> GraphCompileInputLayoutRequirement {
        input_layout_requirement_for_support(self.layout_support)
    }

    /// Returns input values constrained by this operation's CUDA graph layout policy.
    #[must_use]
    pub fn layout_constrained_input_values(&self) -> &[ValueId] {
        &self.layout_constrained_input_values
    }

    /// Returns layout-constrained input values that are not contiguous at offset zero.
    #[must_use]
    pub fn non_contiguous_layout_input_values(&self) -> &[ValueId] {
        &self.non_contiguous_layout_input_values
    }

    /// Returns how this operation's output is represented in the compiled CUDA graph.
    #[must_use]
    pub fn output_materialization(&self) -> GraphCompileValueMaterialization {
        self.output_materialization
    }

    /// Returns providers that advertised support under the active lowering policy.
    #[must_use]
    pub fn candidates(&self) -> &[GraphCompileProviderCandidate] {
        &self.candidates
    }

    /// Returns the planned provider decision from the lowering planner.
    #[must_use]
    pub fn planned_decision(&self) -> Option<&GraphCompileProviderDecision> {
        self.planned_decision.as_ref()
    }

    /// Returns the provider that materialized in native CUDA graph lowering.
    #[must_use]
    pub fn realized_decision(&self) -> Option<&GraphCompileProviderDecision> {
        self.realized_decision.as_ref()
    }

    /// Returns the provider algorithm or execution plan selected for this operation.
    #[must_use]
    pub fn selected_algorithm(&self) -> Option<&str> {
        self.selected_algorithm.as_deref()
    }

    /// Returns provider attempts that failed before another candidate materialized the node.
    #[must_use]
    pub fn provider_failures(&self) -> &[GraphCompileProviderFailure] {
        &self.provider_failures
    }
}
