use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    convolution::ConvolutionMode,
    data_type::DataType,
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::{Error, Result},
    tensor::Tensor,
    utility::to_i64,
};

#[derive(Debug)]
pub struct ConvolutionDescriptor {
    descriptor: BackendDescriptor,
}

impl ConvolutionDescriptor {
    pub fn create(
        compute_type: DataType,
        mode: ConvolutionMode,
        pre_paddings: &[i64],
        post_paddings: &[i64],
        dilations: &[i64],
        filter_strides: &[i64],
    ) -> Result<Self> {
        let spatial_dim_count = pre_paddings.len();
        if spatial_dim_count == 0 {
            return Err(Error::InvalidDataShape);
        }
        if post_paddings.len() != spatial_dim_count {
            return Err(Error::LengthMismatch {
                name: "post_paddings".into(),
                expected: spatial_dim_count,
                actual: post_paddings.len(),
            });
        }
        if dilations.len() != spatial_dim_count {
            return Err(Error::LengthMismatch {
                name: "dilations".into(),
                expected: spatial_dim_count,
                actual: dilations.len(),
            });
        }
        if filter_strides.len() != spatial_dim_count {
            return Err(Error::LengthMismatch {
                name: "filter_strides".into(),
                expected: spatial_dim_count,
                actual: filter_strides.len(),
            });
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Convolution)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::ConvolutionCompType,
            BackendAttributeType::DataType,
            compute_type,
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::ConvolutionConvMode,
            BackendAttributeType::ConvolutionMode,
            mode,
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::ConvolutionSpatialDims,
            to_i64(spatial_dim_count, "spatial_dim_count")?,
        )?;
        descriptor
            .set_attribute_i64_slice(BackendAttributeName::ConvolutionPrePaddings, pre_paddings)?;
        descriptor.set_attribute_i64_slice(
            BackendAttributeName::ConvolutionPostPaddings,
            post_paddings,
        )?;
        descriptor
            .set_attribute_i64_slice(BackendAttributeName::ConvolutionDilations, dilations)?;
        descriptor.set_attribute_i64_slice(
            BackendAttributeName::ConvolutionFilterStrides,
            filter_strides,
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct ConvolutionForwardOperation {
    descriptor: BackendDescriptor,
}

impl ConvolutionForwardOperation {
    pub fn create(
        convolution: &ConvolutionDescriptor,
        x: &Tensor,
        w: &Tensor,
        y: &Tensor,
        alpha: f32,
        beta: f32,
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
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationConvolutionForwardAlpha,
            alpha,
        )?;
        descriptor
            .set_attribute_f32(BackendAttributeName::OperationConvolutionForwardBeta, beta)?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct ConvolutionBackwardDataOperation {
    descriptor: BackendDescriptor,
}

impl ConvolutionBackwardDataOperation {
    pub fn create(
        convolution: &ConvolutionDescriptor,
        w: &Tensor,
        output_gradient: &Tensor,
        input_gradient: &Tensor,
        alpha: f32,
        beta: f32,
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
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationConvolutionBwdDataAlpha,
            alpha,
        )?;
        descriptor
            .set_attribute_f32(BackendAttributeName::OperationConvolutionBwdDataBeta, beta)?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct ConvolutionBackwardFilterOperation {
    descriptor: BackendDescriptor,
}

impl ConvolutionBackwardFilterOperation {
    pub fn create(
        convolution: &ConvolutionDescriptor,
        x: &Tensor,
        output_gradient: &Tensor,
        filter_gradient: &Tensor,
        alpha: f32,
        beta: f32,
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
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationConvolutionBwdFilterAlpha,
            alpha,
        )?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationConvolutionBwdFilterBeta,
            beta,
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
