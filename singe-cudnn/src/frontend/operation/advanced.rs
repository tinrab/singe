use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType,
    error::{Error, Result},
    execution::advanced::{
        MatmulDescriptor, MatmulOperation as BackendMatmulOperation, MoeGroupedMatmulMode,
        MoeGroupedMatmulOperation as BackendMoeGroupedMatmulOperation, PagedCacheLoadOperation,
    },
    frontend::{
        lower::{LoweredOperation, LoweringContext, LoweringOutput, tensor_at},
        operation::FrontendOperationTensors,
    },
    tensor::TensorId,
};

/// Frontend paged-cache load operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PagedCacheLoadSpec {
    pub container: TensorId,
    pub output: TensorId,
    pub sequence: TensorId,
    pub page_table: TensorId,
}

impl PagedCacheLoadSpec {
    pub fn new(
        container: TensorId,
        output: TensorId,
        sequence: TensorId,
        page_table: TensorId,
    ) -> Self {
        Self {
            container,
            output,
            sequence,
            page_table,
        }
    }
}

impl FrontendOperationTensors for PagedCacheLoadSpec {
    fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>) {
        tensors.extend([self.container, self.output, self.sequence, self.page_table]);
    }
}

impl PagedCacheLoadSpec {
    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        Ok(LoweredOperation::PagedCacheLoad(
            PagedCacheLoadOperation::create(
                tensor_at(tensors, self.container)?,
                tensor_at(tensors, self.output)?,
                tensor_at(tensors, self.sequence)?,
                tensor_at(tensors, self.page_table)?,
            )?,
        ))
    }
}

/// Frontend MoE grouped matmul operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MoeGroupedMatmulOperation {
    Forward {
        token: TensorId,
        weight: TensorId,
        first_token_offset: TensorId,
        output: TensorId,
        config: MoeGroupedMatmulConfig,
    },
    Backward {
        output_gradient: TensorId,
        token: TensorId,
        first_token_offset: TensorId,
        weight_gradient: TensorId,
        config: MoeGroupedMatmulBackwardConfig,
    },
}

impl FrontendOperationTensors for MoeGroupedMatmulOperation {
    fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>) {
        match self {
            Self::Forward {
                token,
                weight,
                first_token_offset,
                output,
                config,
            } => {
                tensors.extend([*token, *weight, *first_token_offset, *output]);
                tensors.extend(config.token_index());
                tensors.extend(config.token_ks());
            }
            Self::Backward {
                output_gradient,
                token,
                first_token_offset,
                weight_gradient,
                ..
            } => {
                tensors.extend([
                    *output_gradient,
                    *token,
                    *first_token_offset,
                    *weight_gradient,
                ]);
            }
        }
    }
}

impl MoeGroupedMatmulOperation {
    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        match self {
            Self::Forward {
                token,
                weight,
                first_token_offset,
                output,
                config,
            } => Ok(LoweredOperation::MoeGroupedMatmul(
                BackendMoeGroupedMatmulOperation::create(
                    config.mode(),
                    config.compute_type(),
                    tensor_at(tensors, *token)?,
                    tensor_at(tensors, *weight)?,
                    tensor_at(tensors, *first_token_offset)?,
                    tensor_at(tensors, *output)?,
                    config
                        .token_index()
                        .map(|tensor| tensor_at(tensors, tensor))
                        .transpose()?,
                    config
                        .token_ks()
                        .map(|tensor| tensor_at(tensors, tensor))
                        .transpose()?,
                    config.top_k(),
                )?,
            )),
            Self::Backward { .. } => Err(Error::FrontendFeatureUnavailable {
                feature: "moe grouped matmul backward".into(),
                reason: "singe-cudnn is built against cuDNN 9.21 bindings, but the native backend descriptor requires cuDNN 9.22 bindings".into(),
            }),
        }
    }
}

/// Frontend matmul operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MatmulOperation {
    Matmul {
        a: TensorId,
        b: TensorId,
        c: TensorId,
        config: MatmulConfig,
    },
    Fp8 {
        a: TensorId,
        b: TensorId,
        descale_a: TensorId,
        descale_b: TensorId,
        scale_c: TensorId,
        c: TensorId,
        absolute_max_c: TensorId,
        config: MatmulFp8Config,
    },
}

impl FrontendOperationTensors for MatmulOperation {
    fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>) {
        match self {
            Self::Matmul { a, b, c, config } => {
                tensors.extend([*a, *b, *c]);
                tensors.extend(config.m_override());
                tensors.extend(config.k_override());
            }
            Self::Fp8 {
                a,
                b,
                descale_a,
                descale_b,
                scale_c,
                c,
                absolute_max_c,
                config,
            } => {
                tensors.extend([
                    *a,
                    *b,
                    *descale_a,
                    *descale_b,
                    *scale_c,
                    *c,
                    *absolute_max_c,
                ]);
                tensors.extend(config.m_override());
                tensors.extend(config.k_override());
            }
        }
    }
}

impl MatmulOperation {
    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweringOutput> {
        let tensors = context.backend_tensors();
        match self {
            Self::Matmul { a, b, c, config } => {
                let descriptor = MatmulDescriptor::create(config.compute_type())?;
                Ok(LoweringOutput::Operation(LoweredOperation::Matmul(
                    BackendMatmulOperation::create_with_overrides(
                        &descriptor,
                        tensor_at(tensors, *a)?,
                        tensor_at(tensors, *b)?,
                        tensor_at(tensors, *c)?,
                        config
                            .m_override()
                            .map(|tensor| tensor_at(tensors, tensor))
                            .transpose()?,
                        None,
                        config
                            .k_override()
                            .map(|tensor| tensor_at(tensors, tensor))
                            .transpose()?,
                    )?,
                )))
            }
            Self::Fp8 { .. } => Ok(LoweringOutput::Expanded),
        }
    }
}

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
