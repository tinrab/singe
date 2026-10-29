use crate::scalar::native_f32;

mod geometry;
mod problem;

pub use crate::backend::execution::convolution::{
    geometry::ConvolutionGeometry, problem::ConvolutionProblemKey,
};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        convolution::ConvolutionMode,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::Tensor,
    },
    data_type::DataType,
    error::Result,
    utility::to_i64,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ConvolutionScaling {
    F32 { alpha: u32, beta: u32 },
    F64 { alpha: u64, beta: u64 },
}

/// Resolved native convolution attributes, independent of operand encodings and scaling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConvolutionDescriptorConfig {
    /// Native math precision, independent of activation and filter storage encodings.
    pub compute_type: DataType,
    /// Convolution or cross-correlation mode.
    pub mode: ConvolutionMode,
    /// Validated spatial pre/post padding, stride, and dilation arrays.
    pub geometry: ConvolutionGeometry,
}

impl ConvolutionDescriptorConfig {
    /// Records native compute precision, convolution mode, and validated spatial geometry.
    #[must_use]
    pub fn new(
        compute_type: DataType,
        mode: ConvolutionMode,
        geometry: ConvolutionGeometry,
    ) -> Self {
        Self {
            compute_type,
            mode,
            geometry,
        }
    }
}

/// Finalized native convolution descriptor retaining its complete configuration.
#[derive(Debug)]
pub struct ConvolutionDescriptor {
    descriptor: BackendDescriptor,
    config: ConvolutionDescriptorConfig,
}

impl ConvolutionDescriptor {
    /// Creates a descriptor shared by forward, data-gradient and weight-gradient operations.
    ///
    /// # Errors
    /// Returns an error for unrepresentable spatial rank or native attribute/finalization failure.
    pub fn create(config: ConvolutionDescriptorConfig) -> Result<Self> {
        let geometry = &config.geometry;
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Convolution)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::ConvolutionCompType,
            config.compute_type,
        )?;
        descriptor.set_attribute_enum(BackendAttributeName::ConvolutionConvMode, config.mode)?;
        descriptor.set_attribute_i64(
            BackendAttributeName::ConvolutionSpatialDims,
            to_i64(geometry.spatial_dims(), "spatial_dim_count")?,
        )?;
        descriptor.set_attribute_i64_slice(
            BackendAttributeName::ConvolutionPrePaddings,
            geometry.pre_paddings(),
        )?;
        descriptor.set_attribute_i64_slice(
            BackendAttributeName::ConvolutionPostPaddings,
            geometry.post_paddings(),
        )?;
        descriptor.set_attribute_i64_slice(
            BackendAttributeName::ConvolutionDilations,
            geometry.dilations(),
        )?;
        descriptor.set_attribute_i64_slice(
            BackendAttributeName::ConvolutionFilterStrides,
            geometry.strides(),
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor, config })
    }

    /// Borrows the finalized native convolution descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    /// Retained attributes and spatial geometry without native queries or allocation.
    #[must_use]
    pub fn config(&self) -> &ConvolutionDescriptorConfig {
        &self.config
    }

    fn set_scaling(
        &self,
        operation: &mut BackendDescriptor,
        alpha_attribute: BackendAttributeName,
        beta_attribute: BackendAttributeName,
        alpha: f64,
        beta: f64,
    ) -> Result<ConvolutionScaling> {
        let scaling = if self.config.compute_type == DataType::F64 {
            ConvolutionScaling::F64 {
                alpha: alpha.to_bits(),
                beta: beta.to_bits(),
            }
        } else {
            ConvolutionScaling::F32 {
                alpha: native_f32(alpha).to_bits(),
                beta: native_f32(beta).to_bits(),
            }
        };
        match scaling {
            ConvolutionScaling::F64 { alpha, beta } => {
                operation.set_attribute_f64(alpha_attribute, f64::from_bits(alpha))?;
                operation.set_attribute_f64(beta_attribute, f64::from_bits(beta))?;
            }
            ConvolutionScaling::F32 { alpha, beta } => {
                operation.set_attribute_f32(alpha_attribute, f32::from_bits(alpha))?;
                operation.set_attribute_f32(beta_attribute, f32::from_bits(beta))?;
            }
        }
        Ok(scaling)
    }
}

/// Finalized native forward convolution with retained problem metadata for plan-cache keys.
#[derive(Debug)]
pub struct ConvolutionForwardOperation {
    descriptor: BackendDescriptor,
    problem_key: ConvolutionProblemKey,
}

impl ConvolutionForwardOperation {
    /// Creates `y = alpha * convolution(x, w) + beta * y`.
    /// Scaling attributes use double precision for `F64` computation and float otherwise.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects the operation or its attributes.
    pub fn create(
        convolution: &ConvolutionDescriptor,
        x: &Tensor,
        w: &Tensor,
        y: &Tensor,
        alpha: f64,
        beta: f64,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationConvolutionForward)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConvolutionForwardConvDesc,
            convolution.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConvolutionForwardX,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConvolutionForwardW,
            w.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConvolutionForwardY,
            y.descriptor(),
        )?;
        let scaling = convolution.set_scaling(
            &mut descriptor,
            BackendAttributeName::OperationConvolutionForwardAlpha,
            BackendAttributeName::OperationConvolutionForwardBeta,
            alpha,
            beta,
        )?;
        descriptor.finalize()?;

        let problem_key = ConvolutionProblemKey::new(
            descriptor.descriptor_type(),
            convolution,
            [x, w, y],
            scaling,
        );
        Ok(Self {
            descriptor,
            problem_key,
        })
    }

    /// Borrows the finalized native convolution operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    /// Retained problem identity for native plan caching, independent of device buffer addresses.
    /// Scope the cache to the compatible context, complete graph and compilation policy.
    #[must_use]
    pub fn problem_key(&self) -> &ConvolutionProblemKey {
        &self.problem_key
    }
}

/// Finalized native input-gradient convolution with retained problem metadata for plan-cache keys.
#[derive(Debug)]
pub struct ConvolutionBackwardDataOperation {
    descriptor: BackendDescriptor,
    problem_key: ConvolutionProblemKey,
}

impl ConvolutionBackwardDataOperation {
    /// Creates `dx = alpha * convolution_backward_data(w, dy) + beta * dx`.
    /// Scaling attributes follow the convolution compute precision.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects the operation or its attributes.
    pub fn create(
        convolution: &ConvolutionDescriptor,
        w: &Tensor,
        output_gradient: &Tensor,
        input_gradient: &Tensor,
        alpha: f64,
        beta: f64,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationConvolutionBackwardData)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConvolutionBwdDataConvDesc,
            convolution.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConvolutionBwdDataW,
            w.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConvolutionBwdDataDy,
            output_gradient.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConvolutionBwdDataDx,
            input_gradient.descriptor(),
        )?;
        let scaling = convolution.set_scaling(
            &mut descriptor,
            BackendAttributeName::OperationConvolutionBwdDataAlpha,
            BackendAttributeName::OperationConvolutionBwdDataBeta,
            alpha,
            beta,
        )?;
        descriptor.finalize()?;

        let problem_key = ConvolutionProblemKey::new(
            descriptor.descriptor_type(),
            convolution,
            [input_gradient, w, output_gradient],
            scaling,
        );
        Ok(Self {
            descriptor,
            problem_key,
        })
    }

    /// Borrows the finalized native convolution operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    /// Retained problem identity for native plan caching, independent of device buffer addresses.
    /// Scope the cache to the compatible context, complete graph and compilation policy.
    #[must_use]
    pub fn problem_key(&self) -> &ConvolutionProblemKey {
        &self.problem_key
    }
}

/// Finalized native filter-gradient convolution with retained problem metadata for plan-cache keys.
#[derive(Debug)]
pub struct ConvolutionBackwardFilterOperation {
    descriptor: BackendDescriptor,
    problem_key: ConvolutionProblemKey,
}

impl ConvolutionBackwardFilterOperation {
    /// Creates `dw = alpha * convolution_backward_filter(x, dy) + beta * dw`.
    /// Scaling attributes follow the convolution compute precision.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects the operation or its attributes.
    pub fn create(
        convolution: &ConvolutionDescriptor,
        x: &Tensor,
        output_gradient: &Tensor,
        filter_gradient: &Tensor,
        alpha: f64,
        beta: f64,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationConvolutionBackwardFilter)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConvolutionBwdFilterConvDesc,
            convolution.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConvolutionBwdFilterX,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConvolutionBwdFilterDy,
            output_gradient.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConvolutionBwdFilterDw,
            filter_gradient.descriptor(),
        )?;
        let scaling = convolution.set_scaling(
            &mut descriptor,
            BackendAttributeName::OperationConvolutionBwdFilterAlpha,
            BackendAttributeName::OperationConvolutionBwdFilterBeta,
            alpha,
            beta,
        )?;
        descriptor.finalize()?;

        let problem_key = ConvolutionProblemKey::new(
            descriptor.descriptor_type(),
            convolution,
            [x, filter_gradient, output_gradient],
            scaling,
        );
        Ok(Self {
            descriptor,
            problem_key,
        })
    }

    /// Borrows the finalized native convolution operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    /// Retained problem identity for native plan caching, independent of device buffer addresses.
    /// Scope the cache to the compatible context, complete graph and compilation policy.
    #[must_use]
    pub fn problem_key(&self) -> &ConvolutionProblemKey {
        &self.problem_key
    }
}
