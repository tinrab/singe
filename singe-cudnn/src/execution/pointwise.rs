use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    data_type::DataType,
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::Result,
    math::NanPropagation,
    pointwise::PointwiseMode,
    tensor::Tensor,
};

#[derive(Debug)]
pub struct PointwiseDescriptor {
    descriptor: BackendDescriptor,
}

impl PointwiseDescriptor {
    pub fn create(
        mode: PointwiseMode,
        compute_type: DataType,
        nan_propagation: NanPropagation,
    ) -> Result<Self> {
        Self::create_with_relu_clips(mode, compute_type, nan_propagation, None)
    }

    pub fn create_with_relu_clips(
        mode: PointwiseMode,
        compute_type: DataType,
        nan_propagation: NanPropagation,
        relu_clips: Option<(f64, f64, f64)>,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Pointwise)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::PointwiseMode,
            BackendAttributeType::PointwiseMode,
            mode,
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::PointwiseMathPrec,
            BackendAttributeType::DataType,
            compute_type,
        )?;
        if matches!(mode, PointwiseMode::SwishFwd | PointwiseMode::SwishBwd) {
            descriptor.set_attribute_f64(BackendAttributeName::PointwiseSwishBeta, 1.0)?;
        }
        if matches!(mode, PointwiseMode::ReluFwd | PointwiseMode::ReluBwd) {
            descriptor.set_attribute_enum(
                BackendAttributeName::PointwiseNanPropagation,
                BackendAttributeType::NanPropagation,
                nan_propagation,
            )?;
            let (lower_clip, upper_clip, lower_clip_slope) = relu_clips.unwrap_or_else(|| {
                let upper_clip = if compute_type == DataType::F32 {
                    f64::from(f32::MAX)
                } else {
                    f64::MAX
                };
                (0.0, upper_clip, 0.0)
            });
            descriptor
                .set_attribute_f64(BackendAttributeName::PointwiseReluLowerClip, lower_clip)?;
            descriptor
                .set_attribute_f64(BackendAttributeName::PointwiseReluUpperClip, upper_clip)?;
            descriptor.set_attribute_f64(
                BackendAttributeName::PointwiseReluLowerClipSlope,
                lower_clip_slope,
            )?;
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn create_with_axis(
        mode: PointwiseMode,
        compute_type: DataType,
        nan_propagation: NanPropagation,
        axis: i64,
    ) -> Result<Self> {
        Self::create_with_axis_and_relu_clips(mode, compute_type, nan_propagation, axis, None)
    }

    pub fn create_with_axis_and_relu_clips(
        mode: PointwiseMode,
        compute_type: DataType,
        nan_propagation: NanPropagation,
        axis: i64,
        relu_clips: Option<(f64, f64, f64)>,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Pointwise)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::PointwiseMode,
            BackendAttributeType::PointwiseMode,
            mode,
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::PointwiseMathPrec,
            BackendAttributeType::DataType,
            compute_type,
        )?;
        if matches!(mode, PointwiseMode::SwishFwd | PointwiseMode::SwishBwd) {
            descriptor.set_attribute_f64(BackendAttributeName::PointwiseSwishBeta, 1.0)?;
        }
        if matches!(mode, PointwiseMode::ReluFwd | PointwiseMode::ReluBwd) {
            descriptor.set_attribute_enum(
                BackendAttributeName::PointwiseNanPropagation,
                BackendAttributeType::NanPropagation,
                nan_propagation,
            )?;
            let (lower_clip, upper_clip, lower_clip_slope) = relu_clips.unwrap_or_else(|| {
                let upper_clip = if compute_type == DataType::F32 {
                    f64::from(f32::MAX)
                } else {
                    f64::MAX
                };
                (0.0, upper_clip, 0.0)
            });
            descriptor
                .set_attribute_f64(BackendAttributeName::PointwiseReluLowerClip, lower_clip)?;
            descriptor
                .set_attribute_f64(BackendAttributeName::PointwiseReluUpperClip, upper_clip)?;
            descriptor.set_attribute_f64(
                BackendAttributeName::PointwiseReluLowerClipSlope,
                lower_clip_slope,
            )?;
        }
        descriptor.set_attribute_i64(BackendAttributeName::PointwiseAxis, axis)?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct PointwiseOperation {
    descriptor: BackendDescriptor,
}

impl PointwiseOperation {
    pub fn activation_backward(
        pointwise: &PointwiseDescriptor,
        dy: &Tensor,
        x: &Tensor,
        dx: &Tensor,
        alpha1: f64,
        alpha2: f64,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationPointwise)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwisePwDescriptor,
            pointwise.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwiseXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwiseDyDesc,
            dy.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwiseDxDesc,
            dx.descriptor(),
        )?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationPointwiseAlpha1,
            alpha1 as f32,
        )?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationPointwiseAlpha2,
            alpha2 as f32,
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn unary(
        pointwise: &PointwiseDescriptor,
        x: &Tensor,
        y: &Tensor,
        alpha1: f64,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationPointwise)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwisePwDescriptor,
            pointwise.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwiseXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwiseYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationPointwiseAlpha1,
            alpha1 as f32,
        )?;
        descriptor.set_attribute_f32(BackendAttributeName::OperationPointwiseAlpha2, 1.0)?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn binary(
        pointwise: &PointwiseDescriptor,
        x: &Tensor,
        b: &Tensor,
        y: &Tensor,
        alpha1: f64,
        alpha2: f64,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationPointwise)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwisePwDescriptor,
            pointwise.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwiseXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwiseBDesc,
            b.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwiseYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationPointwiseAlpha1,
            alpha1 as f32,
        )?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationPointwiseAlpha2,
            alpha2 as f32,
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn ternary(
        pointwise: &PointwiseDescriptor,
        x: &Tensor,
        b: &Tensor,
        t: &Tensor,
        y: &Tensor,
        alpha1: f64,
        alpha2: f64,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationPointwise)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwisePwDescriptor,
            pointwise.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwiseXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwiseBDesc,
            b.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwiseTDesc,
            t.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwiseYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationPointwiseAlpha1,
            alpha1 as f32,
        )?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationPointwiseAlpha2,
            alpha2 as f32,
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
