use serde::{Deserialize, Serialize};

pub use crate::backend::execution::moe::MoeGroupedMatmulRouting;
use crate::{
    backend::{
        execution::moe::{
            MoeGroupedMatmulBackwardOperation, MoeGroupedMatmulBackwardTensors,
            MoeGroupedMatmulOperation as BackendMoeGroupedMatmulOperation, MoeGroupedMatmulTensors,
        },
        tensor::TensorId,
    },
    data_type::DataType,
    error::Result,
    frontend::{
        graph::Graph,
        lower::{LoweredOperation, LoweringContext, tensor_at},
    },
};

/// Frontend `MoE` grouped matmul operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MoeGroupedMatmulOperation {
    /// Expert-grouped forward matmul with optional gather/scatter routing.
    Forward {
        /// Required forward operands and output.
        tensors: MoeGroupedMatmulTensors,
        /// Math precision, diagnostic scope, and routing request.
        config: MoeGroupedMatmulConfig,
    },
    /// Expert-grouped weight gradients from token activations and output gradients.
    Backward {
        /// Required token, output-gradient, offset, and weight-gradient identities.
        tensors: MoeGroupedMatmulBackwardTensors,
        /// Math precision, diagnostic scope, and expert-count inference hint.
        config: MoeGroupedMatmulBackwardConfig,
    },
}

impl MoeGroupedMatmulOperation {
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        match self {
            Self::Forward { tensors, config } => {
                let expected = graph
                    .infer_moe_grouped_matmul_output_dimensions(tensors.inputs, config.routing())?;
                graph.validate_tensor_dimensions(
                    tensors.output,
                    &expected,
                    "moe grouped matmul output shape",
                )?;
            }
            Self::Backward { tensors, .. } => {
                graph.validate_tensor_rank(
                    tensors.weight_gradient,
                    3,
                    "moe grouped matmul backward weight gradient rank",
                )?;
                let experts = graph.shape(tensors.weight_gradient)?.dimensions()[0];
                let expected = graph.infer_moe_grouped_matmul_backward_weight_gradient_dimensions(
                    tensors.inputs,
                    Some(experts),
                )?;
                graph.validate_tensor_dimensions(
                    tensors.weight_gradient,
                    &expected,
                    "moe grouped matmul backward weight gradient shape",
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let native = context.backend_tensors();
        match self {
            Self::Forward { tensors, config } => {
                let routing = config.routing().try_map(|id| tensor_at(native, id))?;
                Ok(LoweredOperation::MoeGroupedMatmul(
                    BackendMoeGroupedMatmulOperation::create(
                        routing,
                        context.compute_data_type(config.compute_type()),
                        (*tensors).try_map(|id| tensor_at(native, id))?,
                    )?,
                ))
            }
            Self::Backward { tensors, config } => Ok(LoweredOperation::MoeGroupedMatmulBackward(
                MoeGroupedMatmulBackwardOperation::create(
                    context.compute_data_type(config.compute_type()),
                    (*tensors).try_map(|id| tensor_at(native, id))?,
                )?,
            )),
        }
    }
}

/// Attributes for cuDNN frontend `MoE` grouped matmul.
///
/// `MoE` grouped matmul performs matrix multiplications across experts.
/// Routing groups each mode with its required index tensors and top-k count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeGroupedMatmulConfig {
    name: Option<String>,
    routing: MoeGroupedMatmulRouting<TensorId>,
    compute_type: Option<DataType>,
}

impl MoeGroupedMatmulConfig {
    /// Uses the requested routing and inherits graph compute precision, defaulting to `F32`.
    #[must_use]
    pub fn new(routing: MoeGroupedMatmulRouting<TensorId>) -> Self {
        Self {
            name: None,
            routing,
            compute_type: None,
        }
    }

    /// Names an inferred result `name::Output` without changing explicitly declared tensor names.
    /// An empty name leaves inferred tensors unnamed.
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

    /// Overrides compute precision, or restores graph inheritance with `None`.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.compute_type = compute_type.into();
        self
    }

    /// Replaces the complete gather/scatter request while preserving name and compute precision.
    #[must_use]
    pub fn with_routing(mut self, routing: MoeGroupedMatmulRouting<TensorId>) -> Self {
        self.routing = routing;
        self
    }

    /// Returns the complete gather/scatter request and its required routing tensors.
    #[must_use]
    pub fn routing(&self) -> MoeGroupedMatmulRouting<TensorId> {
        self.routing
    }

    /// Returns the math-precision override, or `None` to inherit graph policy.
    #[must_use]
    pub fn compute_type(&self) -> Option<DataType> {
        self.compute_type
    }
}

/// Attributes for cuDNN frontend `MoE` grouped matmul backward.
///
/// The native cuDNN node computes grouped weight gradients from routed token
/// activations, output gradients, and the first-token offset table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MoeGroupedMatmulBackwardConfig {
    name: Option<String>,
    compute_type: Option<DataType>,
    expert_count: Option<i64>,
}

impl MoeGroupedMatmulBackwardConfig {
    /// Inherits graph compute precision, defaulting to `F32`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Names an inferred gradient `name::DWeight` without changing explicitly declared tensor names.
    /// An empty name leaves inferred tensors unnamed.
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

    /// Overrides compute precision, or restores graph inheritance with `None`.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.compute_type = compute_type.into();
        self
    }

    /// Selects the expert axis of an inferred weight gradient.
    /// Use this with explicit terminal boundaries or shared experts across routing groups.
    /// Without a hint, inference uses offset-table length, matching C++'s implicit-format inference.
    /// Explicit weight-gradient tensors retain their declared expert dimension.
    /// Passing `None` restores inference from the offset-table length.
    #[must_use]
    pub fn with_expert_count(mut self, expert_count: impl Into<Option<i64>>) -> Self {
        self.expert_count = expert_count.into();
        self
    }

    /// Returns the explicit expert-count hint, or `None` to infer it from the offset-table length.
    #[must_use]
    pub fn expert_count(&self) -> Option<i64> {
        self.expert_count
    }

    /// Returns the math-precision override, or `None` to inherit graph policy.
    #[must_use]
    pub fn compute_type(&self) -> Option<DataType> {
        self.compute_type
    }
}

impl_inherited_compute_type!(MoeGroupedMatmulConfig, MoeGroupedMatmulBackwardConfig);
