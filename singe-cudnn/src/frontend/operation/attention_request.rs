use serde::{Deserialize, Serialize};

use crate::{
    backend::tensor::TensorId,
    error::Result,
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaFp8Inputs, SdpaInputs, SdpaMxfp8Inputs},
        error::Error as FrontendError,
        lower::{LoweredOperation, LoweringContext},
        operation::AttentionConfig,
    },
};

/// Attention operands retained until preparation chooses a target-specific implementation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AttentionForwardInputs {
    /// Floating-point query, key, and value operands with optional score modifiers.
    Floating(SdpaInputs),
    /// Per-tensor FP8 attention operands with explicit scaling and quantization inputs.
    Fp8(SdpaFp8Inputs),
    /// Block-scaled MXFP8 attention operands with per-block scale inputs.
    Mxfp8(SdpaMxfp8Inputs),
}

impl AttentionForwardInputs {
    /// Returns query, key, and value identities in that order, independently of the quantization variant.
    #[must_use]
    pub fn query_key_value(self) -> [TensorId; 3] {
        match self {
            Self::Floating(inputs) => [inputs.query, inputs.key, inputs.value],
            Self::Fp8(inputs) => [inputs.query, inputs.key, inputs.value],
            Self::Mxfp8(inputs) => [inputs.query, inputs.key, inputs.value],
        }
    }
}

/// A declarative forward request, including the caller's output identities and layouts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionForwardRequest {
    /// Variant-specific query, key, value, score, and quantization operands.
    pub inputs: AttentionForwardInputs,
    /// Caller-declared attention result, statistics, and variant-specific auxiliary output identities.
    pub outputs: SdpaForwardOutputTensors,
    /// Attention, score, dropout, and implementation-selection settings retained until preparation.
    pub config: AttentionConfig,
}

impl AttentionForwardRequest {
    #[expect(
        clippy::unused_self,
        reason = "declarative requests share the operation lowering protocol but must be expanded before lowering"
    )]
    pub(crate) fn lower(&self, _context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        Err(FrontendError::SdpaRequestNotExpanded.into())
    }
}
