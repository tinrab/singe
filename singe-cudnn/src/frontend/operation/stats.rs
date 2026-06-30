use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    execution::normalization::{GenStatsMode, GenStatsOperation as BackendGenStatsOperation},
    frontend::{
        lower::{LoweredOperation, LoweringContext, tensor_at},
        operation::FrontendOperationTensors,
    },
    tensor::TensorId,
};

/// Frontend generate-stats operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenStatsOperation {
    pub input: TensorId,
    pub sum: TensorId,
    pub square_sum: TensorId,
    pub config: GenStatsConfig,
}

impl GenStatsOperation {
    pub fn new(
        input: TensorId,
        sum: TensorId,
        square_sum: TensorId,
        config: GenStatsConfig,
    ) -> Self {
        Self {
            input,
            sum,
            square_sum,
            config,
        }
    }
}

impl FrontendOperationTensors for GenStatsOperation {
    fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>) {
        tensors.extend([self.input, self.sum, self.square_sum]);
    }
}

impl GenStatsOperation {
    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        Ok(LoweredOperation::GenStats(
            BackendGenStatsOperation::create(
                GenStatsMode::SumSqsum,
                tensor_at(tensors, self.input)?,
                tensor_at(tensors, self.sum)?,
                tensor_at(tensors, self.square_sum)?,
            )?,
        ))
    }
}

/// Attributes for a GenStats operation.
///
/// GenStats computes per-channel sum and sum-of-squares outputs for batch
/// normalization finalize.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct GenStatsConfig;

impl GenStatsConfig {
    pub fn new() -> Self {
        Self
    }
}
