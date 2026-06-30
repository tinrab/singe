use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType,
    error::Result,
    execution::advanced::{
        BlockScaleDequantizeOperation as BackendBlockScaleDequantizeOperation,
        BlockScaleQuantizeOperation as BackendBlockScaleQuantizeOperation,
    },
    frontend::{
        lower::{LoweredOperation, LoweringContext, tensor_at},
        operation::FrontendOperationTensors,
    },
    tensor::TensorId,
    utility::to_i32,
};

/// Frontend block-scale operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BlockScaleOperation {
    Quantize {
        input: TensorId,
        output: TensorId,
        scale: TensorId,
        config: BlockScaleQuantizeConfig,
    },
    Dequantize {
        input: TensorId,
        scale: TensorId,
        output: TensorId,
        config: BlockScaleDequantizeConfig,
    },
}

impl FrontendOperationTensors for BlockScaleOperation {
    fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>) {
        match self {
            Self::Quantize {
                input,
                output,
                scale,
                ..
            }
            | Self::Dequantize {
                input,
                scale,
                output,
                ..
            } => {
                tensors.extend([*input, *scale, *output]);
            }
        }
    }
}

impl BlockScaleOperation {
    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        match self {
            Self::Quantize {
                input,
                output,
                scale,
                config,
            } => Ok(LoweredOperation::BlockScaleQuantize(
                BackendBlockScaleQuantizeOperation::create(
                    tensor_at(tensors, *input)?,
                    tensor_at(tensors, *output)?,
                    tensor_at(tensors, *scale)?,
                    config.compute_type(),
                    to_i32(config.block_size(), "block size")?,
                )?,
            )),
            Self::Dequantize {
                input,
                scale,
                output,
                config,
            } => Ok(LoweredOperation::BlockScaleDequantize(
                BackendBlockScaleDequantizeOperation::create(
                    tensor_at(tensors, *input)?,
                    tensor_at(tensors, *scale)?,
                    tensor_at(tensors, *output)?,
                    config.compute_type(),
                    config.block_sizes(),
                    config.is_negative_scale(),
                )?,
            )),
        }
    }
}

/// Attributes for block scale quantization.
///
/// Block scale quantize produces a quantized output tensor and a scale tensor
/// from a higher precision tensor. MXFP8 and NVFP4 recipes quantize fixed-size
/// blocks along an axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockScaleQuantizeConfig {
    compute_type: DataType,
    block_size: i64,
    axis: i64,
    transpose: bool,
}

impl BlockScaleQuantizeConfig {
    pub fn new(compute_type: DataType, block_size: i64) -> Self {
        Self {
            compute_type,
            block_size,
            axis: -1,
            transpose: false,
        }
    }

    pub fn with_axis(mut self, axis: i64) -> Self {
        self.axis = axis;
        self
    }

    pub fn with_transpose(mut self) -> Self {
        self.transpose = true;
        self
    }

    pub fn compute_type(&self) -> DataType {
        self.compute_type
    }

    pub fn block_size(&self) -> i64 {
        self.block_size
    }

    pub fn axis(&self) -> i64 {
        self.axis
    }

    pub fn transpose(&self) -> bool {
        self.transpose
    }
}

/// Attributes for block scale dequantization.
///
/// Block scale dequantize combines quantized values with a block-broadcast scale
/// tensor to produce a dequantized output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockScaleDequantizeConfig {
    compute_type: Option<DataType>,
    block_sizes: Vec<i32>,
    is_negative_scale: bool,
}

impl BlockScaleDequantizeConfig {
    pub fn new(compute_type: DataType, block_sizes: impl Into<Vec<i32>>) -> Self {
        Self {
            compute_type: Some(compute_type),
            block_sizes: block_sizes.into(),
            is_negative_scale: false,
        }
    }

    pub fn with_default_compute_type(block_sizes: impl Into<Vec<i32>>) -> Self {
        Self {
            compute_type: None,
            block_sizes: block_sizes.into(),
            is_negative_scale: false,
        }
    }

    pub fn with_negative_scale(mut self) -> Self {
        self.is_negative_scale = true;
        self
    }

    pub fn compute_type(&self) -> Option<DataType> {
        self.compute_type
    }

    pub fn block_sizes(&self) -> &[i32] {
        &self.block_sizes
    }

    pub fn is_negative_scale(&self) -> bool {
        self.is_negative_scale
    }

    pub fn block_size_at(&self, axis: usize) -> i32 {
        self.block_sizes.get(axis).copied().unwrap_or(1)
    }
}
