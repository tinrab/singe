use serde::{Deserialize, Serialize};

use crate::{convolution::ConvolutionMode, data_type::DataType};

/// Attributes shared by frontend convolution operations.
///
/// Used for forward convolution, data-gradient convolution, and weight-gradient
/// convolution. It corresponds to the C++ frontend convolution attribute
/// setters for padding, stride, dilation, convolution mode, group count, and
/// compute data type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvolutionConfig {
    compute_type: DataType,
    mode: ConvolutionMode,
    pre_paddings: Vec<i64>,
    post_paddings: Vec<i64>,
    strides: Vec<i64>,
    dilations: Vec<i64>,
    group_count: usize,
}

impl ConvolutionConfig {
    pub fn new(compute_type: DataType, spatial_dimensions: usize) -> Self {
        Self {
            compute_type,
            mode: ConvolutionMode::CrossCorrelation,
            pre_paddings: vec![0; spatial_dimensions],
            post_paddings: vec![0; spatial_dimensions],
            strides: vec![1; spatial_dimensions],
            dilations: vec![1; spatial_dimensions],
            group_count: 1,
        }
    }

    pub fn with_mode(mut self, mode: ConvolutionMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_padding(mut self, paddings: impl Into<Vec<i64>>) -> Self {
        let paddings = paddings.into();
        self.pre_paddings = paddings.clone();
        self.post_paddings = paddings;
        self
    }

    pub fn with_pre_paddings(mut self, pre_paddings: impl Into<Vec<i64>>) -> Self {
        self.pre_paddings = pre_paddings.into();
        self
    }

    pub fn with_post_paddings(mut self, post_paddings: impl Into<Vec<i64>>) -> Self {
        self.post_paddings = post_paddings.into();
        self
    }

    pub fn with_strides(mut self, strides: impl Into<Vec<i64>>) -> Self {
        self.strides = strides.into();
        self
    }

    pub fn with_dilations(mut self, dilations: impl Into<Vec<i64>>) -> Self {
        self.dilations = dilations.into();
        self
    }

    pub fn with_group_count(mut self, group_count: usize) -> Self {
        self.group_count = group_count;
        self
    }

    pub fn compute_type(&self) -> DataType {
        self.compute_type
    }

    pub fn mode(&self) -> ConvolutionMode {
        self.mode
    }

    pub fn pre_paddings(&self) -> &[i64] {
        &self.pre_paddings
    }

    pub fn post_paddings(&self) -> &[i64] {
        &self.post_paddings
    }

    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    pub fn dilations(&self) -> &[i64] {
        &self.dilations
    }

    pub fn group_count(&self) -> usize {
        self.group_count
    }
}
