use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        execution::matmul::{
            MatmulDescriptor, MatmulDescriptorConfig, MatmulOperation as BackendMatmulOperation,
            MatmulOperationConfig,
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

/// One native matmul operation with its logical operand layouts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatmulOperation {
    /// Logical identity of the left matrix operand A.
    pub a: TensorId,
    /// Logical identity of the right matrix operand B.
    pub b: TensorId,
    /// Logical identity of the declared matrix-product output C.
    pub c: TensorId,
    /// Diagnostic scope, compute policy, optional dimension operands, and padding value.
    pub config: MatmulConfig,
}

impl MatmulOperation {
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        graph.validate_matmul_output_shape(
            self.a,
            self.b,
            self.c,
            &self.config,
            false,
            "matmul output shape",
        )?;
        Ok(())
    }

    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        let padding_value = self.config.padding_value();
        let descriptor = MatmulDescriptor::create(
            MatmulDescriptorConfig::new(context.compute_data_type(self.config.compute_type()))
                .with_padding_value((padding_value != 0.0).then_some(padding_value)),
        )?;
        Ok(LoweredOperation::Matmul(BackendMatmulOperation::create(
            &descriptor,
            tensor_at(tensors, self.a)?,
            tensor_at(tensors, self.b)?,
            tensor_at(tensors, self.c)?,
            MatmulOperationConfig {
                m_override: self.config.m_override(),
                n_override: self.config.n_override(),
                k_override: self.config.k_override(),
                irregularly_strided_batch_count: None,
            }
            .try_map(|id| tensor_at(tensors, id))?,
        )?))
    }
}

/// Attributes for a cuDNN frontend matmul operation.
///
/// The operation computes `C[M, N] = A[M, K] * B[K, N]`; the last two input
/// dimensions are interpreted as the matrix dimensions and preceding dimensions
/// as batch dimensions.
/// Omitted compute precision inherits graph policy during lowering.
/// An explicit override remains independent of result encoding and FP8 epilogue precision.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MatmulConfig {
    name: Option<String>,
    compute_type: Option<DataType>,
    m_override: Option<TensorId>,
    n_override: Option<TensorId>,
    k_override: Option<TensorId>,
    #[serde(with = "crate::scalar::float::f64_bits")]
    padding_value: f64,
}

impl MatmulConfig {
    /// Inherits graph compute precision, defaulting to `F32` when no graph policy is set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects native math precision independently of graph defaults and result storage.
    /// Passing `None` restores graph compute inheritance.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.compute_type = compute_type.into();
        self
    }

    /// Sets the diagnostic scope for inferred matmul products, `name::C`, and FP8 Amax, `name::Amax_C`.
    /// Explicitly declared tensors retain their existing names.
    /// An empty scope restores ordinary unnamed inference.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets or clears the device tensor supplying the effective M dimension.
    #[must_use]
    pub fn with_m_override(mut self, m_override: impl Into<Option<TensorId>>) -> Self {
        self.m_override = m_override.into();
        self
    }

    /// Uses a device tensor for the effective N dimension of each matrix product.
    /// Passing `None` clears the override.
    #[must_use]
    pub fn with_n_override(mut self, n_override: impl Into<Option<TensorId>>) -> Self {
        self.n_override = n_override.into();
        self
    }

    /// Sets the native padding value for elements outside overridden matrix dimensions.
    /// The default is zero.
    #[must_use]
    pub fn with_padding_value(mut self, padding_value: f64) -> Self {
        self.padding_value = padding_value;
        self
    }

    /// Sets or clears the device tensor supplying the effective K dimension.
    #[must_use]
    pub fn with_k_override(mut self, k_override: impl Into<Option<TensorId>>) -> Self {
        self.k_override = k_override.into();
        self
    }

    /// Returns the nonempty diagnostic scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Returns the explicit math-precision override, or `None` to inherit graph policy.
    #[must_use]
    pub fn compute_type(&self) -> Option<DataType> {
        self.compute_type
    }

    /// Returns the optional tensor supplying the effective output row count M.
    #[must_use]
    pub fn m_override(&self) -> Option<TensorId> {
        self.m_override
    }

    /// Returns the optional tensor supplying the effective output column count N.
    #[must_use]
    pub fn n_override(&self) -> Option<TensorId> {
        self.n_override
    }

    /// Returns the requested value for elements outside overridden matrix dimensions; defaults to zero.
    #[must_use]
    pub fn padding_value(&self) -> f64 {
        self.padding_value
    }

    /// Returns the optional tensor supplying the effective contraction dimension K.
    #[must_use]
    pub fn k_override(&self) -> Option<TensorId> {
        self.k_override
    }
}

impl_inherited_compute_type!(MatmulConfig);
