use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        execution::normalization::{
            GenStatsMode, GenStatsOperation as BackendGenStatsOperation, GenStatsOutputs,
        },
        tensor::TensorId,
    },
    data_type::DataType,
    error::Result,
    frontend::{
        graph::{Graph, infer_normalization_channel_dimensions},
        lower::{LoweredOperation, LoweringContext, tensor_at},
    },
};

/// Frontend generate-stats operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenStatsOperation {
    /// Logical identity of the tensor reduced to per-channel statistics.
    pub input: TensorId,
    /// Declared per-channel sum and square-sum output identities.
    pub outputs: GenStatsOutputs,
    /// Diagnostic scope and optional arithmetic-precision override.
    pub config: GenStatsConfig,
}

/// Scope and math precision for per-channel statistics, independent of input and result storage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenStatsConfig {
    name: Option<String>,
    compute_type: Option<DataType>,
}

impl GenStatsConfig {
    /// Inherits graph compute precision and leaves inferred statistic outputs unnamed.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Names inferred results `name_sum_output` and `name_sq_sum_output`, matching C++.
    /// Explicit tensor names are retained, and an empty scope disables naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty diagnostic scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Overrides graph math precision. `None` restores inheritance.
    /// When the graph leaves math precision unset, use the input encoding as in C++.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.compute_type = compute_type.into();
        self
    }

    /// Returns the explicit math-precision override, or `None` to inherit graph policy and then input encoding.
    #[must_use]
    pub fn compute_type(&self) -> Option<DataType> {
        self.compute_type
    }
}

impl_inherited_compute_type!(GenStatsConfig);

impl GenStatsOperation {
    /// Records per-channel statistics operands and settings without graph validation.
    #[must_use]
    pub fn new(input: TensorId, outputs: GenStatsOutputs, config: GenStatsConfig) -> Self {
        Self {
            input,
            outputs,
            config,
        }
    }
}

impl GenStatsOperation {
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        let expected = infer_normalization_channel_dimensions(graph.shape(self.input)?)?;
        graph.validate_tensors_dimensions(
            [
                (self.outputs.sum, "genstats sum shape"),
                (self.outputs.square_sum, "genstats square-sum shape"),
            ],
            &expected,
        )
    }

    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        let input = tensor_at(tensors, self.input)?;
        Ok(LoweredOperation::GenStats(
            BackendGenStatsOperation::create(
                GenStatsMode::SumSqsum,
                context
                    .compute_data_type_with_fallback(self.config.compute_type(), input.data_type()),
                input,
                self.outputs.try_map(|id| tensor_at(tensors, id))?,
            )?,
        ))
    }
}
