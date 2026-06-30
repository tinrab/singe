use serde::{Deserialize, Serialize};

use crate::{
    convolution::ConvolutionMode,
    data_type::DataType,
    error::Result,
    execution::convolution::{
        ConvolutionBackwardDataOperation as BackendConvolutionBackwardDataOperation,
        ConvolutionBackwardFilterOperation as BackendConvolutionBackwardFilterOperation,
        ConvolutionDescriptor, ConvolutionForwardOperation as BackendConvolutionForwardOperation,
    },
    frontend::{
        lower::{LoweredOperation, LoweringContext, tensor_at},
        operation::FrontendOperationTensors,
    },
    tensor::TensorId,
};

/// Frontend convolution operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ConvolutionOperation {
    Forward {
        x: TensorId,
        w: TensorId,
        y: TensorId,
        config: ConvolutionConfig,
    },
    BackwardData {
        w: TensorId,
        dy: TensorId,
        dx: TensorId,
        config: ConvolutionConfig,
    },
    BackwardFilter {
        x: TensorId,
        dy: TensorId,
        dw: TensorId,
        config: ConvolutionConfig,
    },
}

impl FrontendOperationTensors for ConvolutionOperation {
    fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>) {
        match self {
            Self::Forward { x, w, y, .. }
            | Self::BackwardData {
                w: x, dy: w, dx: y, ..
            }
            | Self::BackwardFilter {
                x, dy: w, dw: y, ..
            } => {
                tensors.extend([*x, *w, *y]);
            }
        }
    }
}

impl ConvolutionOperation {
    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        match self {
            Self::Forward { x, w, y, config } => {
                let descriptor = convolution_descriptor(config)?;
                Ok(LoweredOperation::ConvolutionForward(
                    BackendConvolutionForwardOperation::create(
                        &descriptor,
                        tensor_at(tensors, *x)?,
                        tensor_at(tensors, *w)?,
                        tensor_at(tensors, *y)?,
                        1.0,
                        0.0,
                    )?,
                ))
            }
            Self::BackwardData { w, dy, dx, config } => {
                let descriptor = convolution_descriptor(config)?;
                Ok(LoweredOperation::ConvolutionBackwardData(
                    BackendConvolutionBackwardDataOperation::create(
                        &descriptor,
                        tensor_at(tensors, *w)?,
                        tensor_at(tensors, *dy)?,
                        tensor_at(tensors, *dx)?,
                        1.0,
                        0.0,
                    )?,
                ))
            }
            Self::BackwardFilter { x, dy, dw, config } => {
                let descriptor = convolution_descriptor(config)?;
                Ok(LoweredOperation::ConvolutionBackwardFilter(
                    BackendConvolutionBackwardFilterOperation::create(
                        &descriptor,
                        tensor_at(tensors, *x)?,
                        tensor_at(tensors, *dy)?,
                        tensor_at(tensors, *dw)?,
                        1.0,
                        0.0,
                    )?,
                ))
            }
        }
    }
}

fn convolution_descriptor(config: &ConvolutionConfig) -> Result<ConvolutionDescriptor> {
    ConvolutionDescriptor::create(
        config.compute_type(),
        config.mode(),
        config.pre_paddings(),
        config.post_paddings(),
        config.dilations(),
        config.strides(),
    )
}

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
