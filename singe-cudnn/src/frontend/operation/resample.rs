use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType,
    error::{Error, Result},
    execution::resample::{Fraction, PaddingMode, ResampleMode},
    math::NanPropagation,
};

/// Attributes for a frontend resampling operation.
///
/// cuDNN resampling changes the spatial dimensions of an image and covers modes
/// such as average pooling, max pooling, bilinear, or cubic interpolation. The
/// C++ frontend exposes padding mode, window, stride, and pre/post padding
/// setters; this Rust config stores the same spatial attributes as integer or
/// fractional values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResampleConfig {
    mode: ResampleMode,
    compute_type: DataType,
    nan_propagation: NanPropagation,
    padding_mode: PaddingMode,
    pre_paddings: Vec<Fraction>,
    post_paddings: Vec<Fraction>,
    strides: Vec<Fraction>,
    window_dimensions: Vec<Fraction>,
}

impl ResampleConfig {
    pub fn new(mode: ResampleMode, compute_type: DataType) -> Self {
        Self {
            mode,
            compute_type,
            nan_propagation: NanPropagation::Propagate,
            padding_mode: PaddingMode::Zero,
            pre_paddings: Vec::new(),
            post_paddings: Vec::new(),
            strides: Vec::new(),
            window_dimensions: Vec::new(),
        }
    }

    pub fn with_nan_propagation(mut self, nan_propagation: NanPropagation) -> Self {
        self.nan_propagation = nan_propagation;
        self
    }

    pub fn with_padding_mode(mut self, padding_mode: PaddingMode) -> Self {
        self.padding_mode = padding_mode;
        self
    }

    pub fn with_pre_paddings<T>(mut self, pre_paddings: impl IntoIterator<Item = T>) -> Self
    where
        T: Into<Fraction>,
    {
        self.pre_paddings = pre_paddings.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_post_paddings<T>(mut self, post_paddings: impl IntoIterator<Item = T>) -> Self
    where
        T: Into<Fraction>,
    {
        self.post_paddings = post_paddings.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_strides<T>(mut self, strides: impl IntoIterator<Item = T>) -> Self
    where
        T: Into<Fraction>,
    {
        self.strides = strides.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_window_dims<T>(mut self, window_dimensions: impl IntoIterator<Item = T>) -> Self
    where
        T: Into<Fraction>,
    {
        self.window_dimensions = window_dimensions.into_iter().map(Into::into).collect();
        self
    }

    pub fn mode(&self) -> ResampleMode {
        self.mode
    }

    pub fn compute_type(&self) -> DataType {
        self.compute_type
    }

    pub fn nan_propagation(&self) -> NanPropagation {
        self.nan_propagation
    }

    pub fn padding_mode(&self) -> PaddingMode {
        self.padding_mode
    }

    pub fn pre_paddings(&self) -> &[Fraction] {
        &self.pre_paddings
    }

    pub fn post_paddings(&self) -> &[Fraction] {
        &self.post_paddings
    }

    pub fn strides(&self) -> &[Fraction] {
        &self.strides
    }

    pub fn window_dims(&self) -> &[Fraction] {
        &self.window_dimensions
    }

    fn resolve_spatial_values(
        values: &[Fraction],
        spatial_dims: usize,
        default_value: Fraction,
        name: &str,
    ) -> Result<Vec<Fraction>> {
        if values.is_empty() {
            return Ok(vec![default_value; spatial_dims]);
        }
        if values.len() != spatial_dims {
            return Err(Error::LengthMismatch {
                name: name.into(),
                expected: spatial_dims,
                actual: values.len(),
            });
        }
        Ok(values.to_vec())
    }

    pub fn effective_pre_paddings(&self, spatial_dims: usize) -> Result<Vec<Fraction>> {
        Self::resolve_spatial_values(
            &self.pre_paddings,
            spatial_dims,
            Fraction::whole(0),
            "resample pre paddings",
        )
    }

    pub fn effective_post_paddings(&self, spatial_dims: usize) -> Result<Vec<Fraction>> {
        Self::resolve_spatial_values(
            &self.post_paddings,
            spatial_dims,
            Fraction::whole(0),
            "resample post paddings",
        )
    }

    pub fn effective_strides(&self, spatial_dims: usize) -> Result<Vec<Fraction>> {
        Self::resolve_spatial_values(
            &self.strides,
            spatial_dims,
            Fraction::whole(1),
            "resample strides",
        )
    }

    pub fn effective_window_dims(&self, spatial_dims: usize) -> Result<Vec<Fraction>> {
        Self::resolve_spatial_values(
            &self.window_dimensions,
            spatial_dims,
            Fraction::whole(1),
            "resample window dims",
        )
    }
}
