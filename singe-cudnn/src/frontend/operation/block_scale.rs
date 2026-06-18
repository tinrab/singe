use serde::{Deserialize, Serialize};

use crate::data_type::DataType;

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

    pub fn without_transpose(mut self) -> Self {
        self.transpose = false;
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

    pub fn without_compute_type(block_sizes: impl Into<Vec<i32>>) -> Self {
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

    pub fn without_negative_scale(mut self) -> Self {
        self.is_negative_scale = false;
        self
    }

    pub fn set_compute_type(&mut self, compute_type: DataType) {
        self.compute_type = Some(compute_type);
    }

    pub fn clear_compute_type(&mut self) {
        self.compute_type = None;
    }

    pub fn with_block_size(mut self, value: i32, index: usize) -> Self {
        if self.block_sizes.len() <= index {
            self.block_sizes.resize(index + 1, 1);
        }
        self.block_sizes[index] = value;
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
