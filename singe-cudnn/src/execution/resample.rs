use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cudnn_sys as sys;

use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    data_type::DataType,
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::{Error, Result},
    math::NanPropagation,
    tensor::Tensor,
    utility::{check_range, to_i64},
};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Resample mode for backend resample operations.
#[repr(u32)]
#[non_exhaustive]
pub enum ResampleMode {
    Nearest = sys::cudnnResampleMode_t::CUDNN_RESAMPLE_NEAREST as _,
    Bilinear = sys::cudnnResampleMode_t::CUDNN_RESAMPLE_BILINEAR as _,
    AvgPoolIncludePadding = sys::cudnnResampleMode_t::CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING as _,
    AvgPoolExcludePadding = sys::cudnnResampleMode_t::CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING as _,
    MaxPool = sys::cudnnResampleMode_t::CUDNN_RESAMPLE_MAXPOOL as _,
}

impl_enum_conversion!(sys::cudnnResampleMode_t, ResampleMode);

impl_enum_display!(ResampleMode, {
    Self::Nearest => "CUDNN_RESAMPLE_NEAREST",
    Self::Bilinear => "CUDNN_RESAMPLE_BILINEAR",
    Self::AvgPoolIncludePadding => "CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING",
    Self::AvgPoolExcludePadding => "CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING",
    Self::MaxPool => "CUDNN_RESAMPLE_MAXPOOL",
});

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Padding mode for backend resample operations.
#[repr(u32)]
#[non_exhaustive]
pub enum PaddingMode {
    Zero = sys::cudnnPaddingMode_t::CUDNN_ZERO_PAD as _,
    NegInf = sys::cudnnPaddingMode_t::CUDNN_NEG_INF_PAD as _,
    EdgeVal = sys::cudnnPaddingMode_t::CUDNN_EDGE_VAL_PAD as _,
}

impl_enum_conversion!(sys::cudnnPaddingMode_t, PaddingMode);

impl_enum_display!(PaddingMode, {
    Self::Zero => "CUDNN_ZERO_PAD",
    Self::NegInf => "CUDNN_NEG_INF_PAD",
    Self::EdgeVal => "CUDNN_EDGE_VAL_PAD",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Fraction {
    numerator: i64,
    denominator: i64,
}

impl Fraction {
    pub const fn new(numerator: i64, denominator: i64) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    pub const fn whole(value: i64) -> Self {
        Self::new(value, 1)
    }

    pub const fn numerator(self) -> i64 {
        self.numerator
    }

    pub const fn denominator(self) -> i64 {
        self.denominator
    }

    pub(crate) fn validate(self, name: &str) -> Result<()> {
        check_range!(name, self.denominator > 0)?;
        Ok(())
    }

    pub(crate) fn raw(self) -> sys::cudnnFraction_t {
        sys::cudnnFraction_t {
            numerator: self.numerator,
            denominator: self.denominator,
        }
    }
}

impl From<i32> for Fraction {
    fn from(value: i32) -> Self {
        Self::whole(i64::from(value))
    }
}

impl From<i64> for Fraction {
    fn from(value: i64) -> Self {
        Self::whole(value)
    }
}

impl From<(i32, i32)> for Fraction {
    fn from((numerator, denominator): (i32, i32)) -> Self {
        Self::new(i64::from(numerator), i64::from(denominator))
    }
}

impl From<(i64, i64)> for Fraction {
    fn from((numerator, denominator): (i64, i64)) -> Self {
        Self::new(numerator, denominator)
    }
}

#[derive(Debug)]
pub struct ResampleDescriptor {
    descriptor: BackendDescriptor,
}

impl ResampleDescriptor {
    pub fn create(
        mode: ResampleMode,
        compute_type: DataType,
        nan_propagation: NanPropagation,
        padding_mode: PaddingMode,
        pre_paddings: &[Fraction],
        post_paddings: &[Fraction],
        strides: &[Fraction],
        window_shape: &[Fraction],
    ) -> Result<Self> {
        let spatial_dims = pre_paddings.len();
        if spatial_dims == 0 {
            return Err(Error::InvalidDataShape);
        }
        if post_paddings.len() != spatial_dims {
            return Err(Error::LengthMismatch {
                name: "post_paddings".into(),
                expected: spatial_dims,
                actual: post_paddings.len(),
            });
        }
        if strides.len() != spatial_dims {
            return Err(Error::LengthMismatch {
                name: "strides".into(),
                expected: spatial_dims,
                actual: strides.len(),
            });
        }
        if window_shape.len() != spatial_dims {
            return Err(Error::LengthMismatch {
                name: "window_dims".into(),
                expected: spatial_dims,
                actual: window_shape.len(),
            });
        }
        for value in pre_paddings {
            value.validate("resample pre paddings")?;
        }
        for value in post_paddings {
            value.validate("resample post paddings")?;
        }
        for value in strides {
            value.validate("resample strides")?;
        }
        for value in window_shape {
            value.validate("resample window dims")?;
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Resample)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::ResampleMode,
            BackendAttributeType::ResampleMode,
            mode,
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::ResampleCompType,
            BackendAttributeType::DataType,
            compute_type,
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::ResampleSpatialDims,
            to_i64(spatial_dims, "spatial_dims")?,
        )?;
        let pre_paddings = pre_paddings
            .iter()
            .copied()
            .map(Fraction::raw)
            .collect::<Vec<_>>();
        let post_paddings = post_paddings
            .iter()
            .copied()
            .map(Fraction::raw)
            .collect::<Vec<_>>();
        let strides = strides
            .iter()
            .copied()
            .map(Fraction::raw)
            .collect::<Vec<_>>();
        let window_dimensions = window_shape
            .iter()
            .copied()
            .map(Fraction::raw)
            .collect::<Vec<_>>();
        descriptor.set_attribute_fraction_slice(
            BackendAttributeName::ResamplePrePaddings,
            &pre_paddings,
        )?;
        descriptor.set_attribute_fraction_slice(
            BackendAttributeName::ResamplePostPaddings,
            &post_paddings,
        )?;
        descriptor.set_attribute_fraction_slice(BackendAttributeName::ResampleStrides, &strides)?;
        descriptor.set_attribute_fraction_slice(
            BackendAttributeName::ResampleWindowDims,
            &window_dimensions,
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::ResampleNanPropagation,
            BackendAttributeType::NanPropagation,
            nan_propagation,
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::ResamplePaddingMode,
            BackendAttributeType::PaddingMode,
            padding_mode,
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct ResampleForwardOperation {
    descriptor: BackendDescriptor,
}

impl ResampleForwardOperation {
    pub fn create(
        resample: &ResampleDescriptor,
        x: &Tensor,
        y: &Tensor,
        indices: Option<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationResampleFwd)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleFwdDesc,
            resample.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleFwdXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleFwdYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_f32(BackendAttributeName::OperationResampleFwdAlpha, 1.0)?;
        descriptor.set_attribute_f32(BackendAttributeName::OperationResampleFwdBeta, 0.0)?;
        if let Some(indices) = indices {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationResampleFwdIdxDesc,
                indices.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct ResampleBackwardOperation {
    descriptor: BackendDescriptor,
}

impl ResampleBackwardOperation {
    pub fn create(
        resample: &ResampleDescriptor,
        x: &Tensor,
        y: &Tensor,
        dy: &Tensor,
        dx: &Tensor,
        indices: Option<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationResampleBwd)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleBwdDesc,
            resample.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleBwdXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleBwdYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleBwdDyDesc,
            dy.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleBwdDxDesc,
            dx.descriptor(),
        )?;
        descriptor.set_attribute_f32(BackendAttributeName::OperationResampleBwdAlpha, 1.0)?;
        descriptor.set_attribute_f32(BackendAttributeName::OperationResampleBwdBeta, 0.0)?;
        if let Some(indices) = indices {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationResampleBwdIdxDesc,
                indices.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
