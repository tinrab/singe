use serde::{Deserialize, Serialize};

use crate::{data_type::DataType, execution::advanced::MoeGroupedMatmulMode, tensor::TensorId};

/// Attributes for a cuDNN frontend matmul operation.
///
/// The operation computes `C[M, N] = A[M, K] * B[K, N]`; the last two input
/// dimensions are interpreted as the matrix dimensions and preceding dimensions
/// as batch dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatmulConfig {
    name: Option<String>,
    compute_type: DataType,
    m_override: Option<TensorId>,
    k_override: Option<TensorId>,
}

impl MatmulConfig {
    pub fn new(compute_type: DataType) -> Self {
        Self {
            name: None,
            compute_type,
            m_override: None,
            k_override: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn clear_name(mut self) -> Self {
        self.name = None;
        self
    }

    pub fn with_m_override(mut self, m_override: TensorId) -> Self {
        self.m_override = Some(m_override);
        self
    }

    pub fn clear_m_override(mut self) -> Self {
        self.m_override = None;
        self
    }

    pub fn with_k_override(mut self, k_override: TensorId) -> Self {
        self.k_override = Some(k_override);
        self
    }

    pub fn clear_k_override(mut self) -> Self {
        self.k_override = None;
        self
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn compute_type(&self) -> DataType {
        self.compute_type
    }

    pub fn m_override(&self) -> Option<TensorId> {
        self.m_override
    }

    pub fn k_override(&self) -> Option<TensorId> {
        self.k_override
    }
}

/// Attributes for an FP8 matmul frontend operation.
///
/// This mirrors [`MatmulConfig`] with the same naming and dynamic-dimension
/// override fields, and is used by FP8-specific lowering paths documented as
/// part of cuDNN frontend matmul and fusion support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatmulFp8Config {
    name: Option<String>,
    compute_type: DataType,
    m_override: Option<TensorId>,
    k_override: Option<TensorId>,
}

impl MatmulFp8Config {
    pub fn new(compute_type: DataType) -> Self {
        Self {
            name: None,
            compute_type,
            m_override: None,
            k_override: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn clear_name(mut self) -> Self {
        self.name = None;
        self
    }

    pub fn with_m_override(mut self, m_override: TensorId) -> Self {
        self.m_override = Some(m_override);
        self
    }

    pub fn with_k_override(mut self, k_override: TensorId) -> Self {
        self.k_override = Some(k_override);
        self
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn compute_type(&self) -> DataType {
        self.compute_type
    }

    pub fn m_override(&self) -> Option<TensorId> {
        self.m_override
    }

    pub fn k_override(&self) -> Option<TensorId> {
        self.k_override
    }
}

/// Attributes for cuDNN frontend MoE grouped matmul.
///
/// MoE grouped matmul performs grouped matrix multiplications across experts,
/// with tokens routed to expert weight matrices by the configured mode and
/// optional routing tensors.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MoeGroupedMatmulConfig {
    mode: MoeGroupedMatmulMode,
    compute_type: DataType,
    token_index: Option<TensorId>,
    token_ks: Option<TensorId>,
    top_k: Option<i32>,
}

impl MoeGroupedMatmulConfig {
    pub fn new(mode: MoeGroupedMatmulMode, compute_type: DataType) -> Self {
        Self {
            mode,
            compute_type,
            token_index: None,
            token_ks: None,
            top_k: None,
        }
    }

    pub fn with_token_index(mut self, token_index: TensorId) -> Self {
        self.token_index = Some(token_index);
        self
    }

    pub fn with_token_ks(mut self, token_ks: TensorId) -> Self {
        self.token_ks = Some(token_ks);
        self
    }

    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn mode(&self) -> MoeGroupedMatmulMode {
        self.mode
    }

    pub fn compute_type(&self) -> DataType {
        self.compute_type
    }

    pub fn token_index(&self) -> Option<TensorId> {
        self.token_index
    }

    pub fn token_ks(&self) -> Option<TensorId> {
        self.token_ks
    }

    pub fn top_k(&self) -> Option<i32> {
        self.top_k
    }
}

/// Attributes for cuDNN frontend MoE grouped matmul backward.
///
/// The native cuDNN node computes grouped weight gradients from routed token
/// activations, output gradients, and the first-token offset table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeGroupedMatmulBackwardConfig {
    compute_type: DataType,
}

impl MoeGroupedMatmulBackwardConfig {
    pub fn new(compute_type: DataType) -> Self {
        Self { compute_type }
    }

    pub fn compute_type(&self) -> DataType {
        self.compute_type
    }
}
