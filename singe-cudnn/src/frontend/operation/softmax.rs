use serde::{Deserialize, Serialize};

use crate::{
    backend::{execution::attention::SoftmaxStatsBase, tensor::TensorId},
    data_type::DataType,
    math::NanPropagation,
};

/// Naming, math and sink attributes for composite softmax and SDPA score computation.
///
/// SDPA uses softmax over attention scores and may emit statistics needed by
/// backward training computation. This config stores the compute type and NaN
/// propagation for the composite helper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxConfig {
    name: Option<String>,
    stats_base: SoftmaxStatsBase,
    compute_type: Option<DataType>,
    nan_propagation: NanPropagation,
    sink: Option<TensorId>,
}

impl SoftmaxConfig {
    /// Names inferred results by their softmax output ports, such as `name::S` and `name::Stats`.
    /// Private composite intermediates use the same diagnostic scope.
    /// Explicit tensor names remain unchanged, and an empty name leaves inferred results unnamed.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty result-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Selects the encoding of emitted log-sum-exp statistics.
    #[must_use]
    pub fn with_stats_base(mut self, base: SoftmaxStatsBase) -> Self {
        self.stats_base = base;
        self
    }

    /// Returns the encoding base for saved log-sum-exp statistics.
    #[must_use]
    pub fn stats_base(&self) -> SoftmaxStatsBase {
        self.stats_base
    }

    /// Inherits graph compute precision, defaulting to `F32`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: None,
            compute_type: None,
            nan_propagation: NanPropagation::Propagate,
            stats_base: SoftmaxStatsBase::NaturalLog,
            sink: None,
        }
    }

    /// Selects compute precision, or inherits the graph policy when cleared.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.compute_type = compute_type.into();
        self
    }

    /// Sets the NaN behavior used by composite softmax calculations.
    #[must_use]
    pub fn with_nan_propagation(mut self, nan_propagation: NanPropagation) -> Self {
        self.nan_propagation = nan_propagation;
        self
    }

    /// Selects sink logits broadcastable to the row summaries, or removes the sink.
    #[must_use]
    pub fn with_sink(mut self, sink: impl Into<Option<TensorId>>) -> Self {
        self.sink = sink.into();
        self
    }

    /// Returns the optional sink-logit tensor identity.
    #[must_use]
    pub fn sink(&self) -> Option<TensorId> {
        self.sink
    }

    /// Returns the compute-precision override, or `None` to inherit graph policy.
    #[must_use]
    pub fn compute_type(&self) -> Option<DataType> {
        self.compute_type
    }

    /// Returns the requested composite softmax NaN behavior.
    #[must_use]
    pub fn nan_propagation(&self) -> NanPropagation {
        self.nan_propagation
    }
}

impl Default for SoftmaxConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftmaxConfig {
    pub(crate) fn sink_mut(&mut self) -> &mut Option<TensorId> {
        &mut self.sink
    }
}

impl_inherited_compute_type!(SoftmaxConfig);
