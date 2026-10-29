use serde::{Deserialize, Serialize};

pub use crate::backend::execution::resample::ResampleBackwardInputs;
use crate::{
    backend::{
        execution::resample::{
            PaddingMode, ResampleBackwardOperation as BackendResampleBackwardOperation,
            ResampleDescriptor, ResampleDescriptorConfig,
            ResampleForwardOperation as BackendResampleForwardOperation, ResampleGeometry,
            ResampleMode, ResampleOutputs,
        },
        tensor::{Tensor, TensorId},
    },
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        graph::Graph,
        lower::{LoweredOperation, LoweringContext, tensor_at},
    },
    math::{Fraction, NanPropagation},
};

/// Frontend resample operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ResampleOperation {
    /// Pooling or interpolation with an optional saved-index output.
    Forward {
        /// Input activation tensor identity.
        input: TensorId,
        /// Resampled activation and optional saved-index destinations.
        #[serde(flatten)]
        outputs: ResampleOutputs,
        /// Spatial geometry, math precision, padding, NaN behavior, and output scaling.
        config: ResampleConfig,
    },
    /// Backward resampling with optional saved forward tensors or indices.
    Backward {
        /// Output gradient and optional saved forward activations or indices.
        inputs: ResampleBackwardInputs<TensorId>,
        /// Input-gradient destination tensor identity.
        input_gradient: TensorId,
        /// Spatial geometry, math precision, padding, NaN behavior, and gradient scaling.
        config: ResampleConfig,
    },
}

impl ResampleOperation {
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        match self {
            Self::Forward {
                input,
                outputs,
                config,
            } => {
                graph.validate_accumulation_output(
                    outputs.output,
                    config.beta() != 0.0,
                    "resample",
                )?;
                graph.validate_resample_geometry(*input, outputs.output, config)?;
                graph.validate_resample_indices(
                    outputs.indices,
                    graph.shape(outputs.output)?.dimensions(),
                    config.mode(),
                    "resample indices shape",
                )?;
            }
            Self::Backward {
                inputs,
                input_gradient,
                config,
            } => {
                graph.validate_accumulation_output(
                    *input_gradient,
                    config.beta() != 0.0,
                    "resample backward",
                )?;
                graph.validate_resample_geometry(
                    *input_gradient,
                    inputs.output_gradient,
                    config,
                )?;
                if let Some(input) = inputs.input {
                    graph.validate_tensor_dimensions(
                        input,
                        graph.shape(*input_gradient)?.dimensions(),
                        "resample backward input shape",
                    )?;
                }
                if let Some(output) = inputs.output {
                    graph.validate_tensor_dimensions(
                        output,
                        graph.shape(inputs.output_gradient)?.dimensions(),
                        "resample backward output shape",
                    )?;
                }
                graph.validate_resample_indices(
                    inputs.indices,
                    graph.shape(inputs.output_gradient)?.dimensions(),
                    config.mode(),
                    "resample backward indices shape",
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        match self {
            Self::Forward {
                input,
                outputs,
                config,
            } => {
                let input = tensor_at(tensors, *input)?;
                let descriptor = resample_descriptor(
                    input,
                    config,
                    context.compute_data_type(config.compute_type()),
                )?;
                Ok(LoweredOperation::Resample(
                    BackendResampleForwardOperation::create(
                        &descriptor,
                        input,
                        (*outputs).try_map(|id| tensor_at(tensors, id))?,
                        config.alpha(),
                        config.beta(),
                    )?,
                ))
            }
            Self::Backward {
                inputs,
                input_gradient,
                config,
            } => {
                let input_gradient = tensor_at(tensors, *input_gradient)?;
                let descriptor = resample_descriptor(
                    input_gradient,
                    config,
                    context.compute_data_type(config.compute_type()),
                )?;
                Ok(LoweredOperation::ResampleBackward(
                    BackendResampleBackwardOperation::create(
                        &descriptor,
                        (*inputs).try_map(|id| tensor_at(tensors, id))?,
                        input_gradient,
                        config.alpha(),
                        config.beta(),
                    )?,
                ))
            }
        }
    }
}

fn resample_descriptor(
    input: &Tensor,
    config: &ResampleConfig,
    compute_type: DataType,
) -> Result<ResampleDescriptor> {
    let spatial_dims = input
        .dimensions()
        .len()
        .checked_sub(2)
        .filter(|&rank| rank > 0)
        .ok_or(Error::InvalidDataShape)?;
    ResampleDescriptor::create(ResampleDescriptorConfig {
        mode: config.mode(),
        compute_type,
        nan_propagation: config.nan_propagation(),
        padding_mode: config.padding_mode(),
        geometry: config.resolve_geometry(spatial_dims)?,
    })
}

/// Attributes for a frontend resampling operation.
///
/// cuDNN resampling changes spatial dimensions using pooling or interpolation.
/// Window, stride, and pre/post padding accept integer or fractional geometry.
/// Scaling computes `output = alpha * result + beta * output` in either direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResampleConfig {
    #[serde(default)]
    name: Option<String>,
    mode: ResampleMode,
    compute_type: Option<DataType>,
    nan_propagation: NanPropagation,
    padding_mode: PaddingMode,
    pre_paddings: Vec<Fraction>,
    post_paddings: Vec<Fraction>,
    strides: Vec<Fraction>,
    window_dimensions: Vec<Fraction>,
    #[serde(with = "crate::scalar::float::f64_bits")]
    alpha: f64,
    #[serde(with = "crate::scalar::float::f64_bits")]
    beta: f64,
}

impl ResampleConfig {
    /// Resolves omitted geometry to zero padding and unit windows and strides.
    /// The result can be used directly by native descriptor construction or inspected without a CUDA handle.
    ///
    /// # Errors
    /// Returns an error for zero spatial rank, mismatched array lengths or invalid fractions.
    pub fn resolve_geometry(&self, spatial_dims: usize) -> Result<ResampleGeometry> {
        ResampleGeometry::create(
            self.effective_pre_paddings(spatial_dims)?,
            self.effective_post_paddings(spatial_dims)?,
            self.effective_strides(spatial_dims)?,
            self.effective_window_dims(spatial_dims)?,
        )
    }

    /// Inherits graph compute precision, defaulting to `F32`.
    #[must_use]
    pub fn new(mode: ResampleMode) -> Self {
        Self {
            name: None,
            mode,
            compute_type: None,
            nan_propagation: NanPropagation::Propagate,
            padding_mode: PaddingMode::Zero,
            pre_paddings: Vec::new(),
            post_paddings: Vec::new(),
            strides: Vec::new(),
            window_dimensions: Vec::new(),
            alpha: 1.0,
            beta: 0.0,
        }
    }

    /// Sets the NaN propagation policy for resampling.
    #[must_use]
    pub fn with_nan_propagation(mut self, nan_propagation: NanPropagation) -> Self {
        self.nan_propagation = nan_propagation;
        self
    }

    /// Names inferred forward outputs `name::Y` and `name::Index`, and input gradients `name::DX`.
    /// Explicit declarations retain their names; an empty scope disables naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty result-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Overrides compute precision, or restores graph inheritance with `None`.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.compute_type = compute_type.into();
        self
    }

    /// Sets result scaling and accumulation into the existing destination.
    /// A nonzero double-precision `beta` requires initialized bindable destination storage.
    /// Both inferred and explicit results are checked again during graph preparation.
    #[must_use]
    pub fn with_scaling(mut self, alpha: f64, beta: f64) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Returns the multiplier applied to the computed forward result or backward gradient.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the multiplier applied to the existing destination before accumulation.
    #[must_use]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Selects the values used beyond the spatial input boundary.
    #[must_use]
    pub fn with_padding_mode(mut self, padding_mode: PaddingMode) -> Self {
        self.padding_mode = padding_mode;
        self
    }

    /// Replaces per-axis padding before the input; an empty list restores zero-padding defaults.
    #[must_use]
    pub fn with_pre_paddings<T>(mut self, pre_paddings: impl IntoIterator<Item = T>) -> Self
    where
        T: Into<Fraction>,
    {
        self.pre_paddings = pre_paddings.into_iter().map(Into::into).collect();
        self
    }

    /// Replaces per-axis padding after the input; an empty list restores zero-padding defaults.
    #[must_use]
    pub fn with_post_paddings<T>(mut self, post_paddings: impl IntoIterator<Item = T>) -> Self
    where
        T: Into<Fraction>,
    {
        self.post_paddings = post_paddings.into_iter().map(Into::into).collect();
        self
    }

    /// Replaces per-axis fractional strides; an empty list restores unit-stride defaults.
    #[must_use]
    pub fn with_strides<T>(mut self, strides: impl IntoIterator<Item = T>) -> Self
    where
        T: Into<Fraction>,
    {
        self.strides = strides.into_iter().map(Into::into).collect();
        self
    }

    /// Replaces per-axis fractional window extents; an empty list restores unit-window defaults.
    #[must_use]
    pub fn with_window_dims<T>(mut self, window_dimensions: impl IntoIterator<Item = T>) -> Self
    where
        T: Into<Fraction>,
    {
        self.window_dimensions = window_dimensions.into_iter().map(Into::into).collect();
        self
    }

    /// Returns the pooling or interpolation mode.
    #[must_use]
    pub fn mode(&self) -> ResampleMode {
        self.mode
    }

    /// Returns the compute-precision override, or `None` to inherit graph policy.
    #[must_use]
    pub fn compute_type(&self) -> Option<DataType> {
        self.compute_type
    }

    /// Returns the configured NaN propagation policy.
    #[must_use]
    pub fn nan_propagation(&self) -> NanPropagation {
        self.nan_propagation
    }

    /// Returns the values used beyond the spatial input boundary.
    #[must_use]
    pub fn padding_mode(&self) -> PaddingMode {
        self.padding_mode
    }

    /// Borrows explicit fractional pre-padding values; an empty slice requests zero defaults.
    #[must_use]
    pub fn pre_paddings(&self) -> &[Fraction] {
        &self.pre_paddings
    }

    /// Borrows explicit fractional post-padding values; an empty slice requests zero defaults.
    #[must_use]
    pub fn post_paddings(&self) -> &[Fraction] {
        &self.post_paddings
    }

    /// Borrows explicit fractional spatial strides; an empty slice requests unit defaults.
    #[must_use]
    pub fn strides(&self) -> &[Fraction] {
        &self.strides
    }

    /// Borrows explicit fractional window extents; an empty slice requests unit defaults.
    #[must_use]
    pub fn window_dims(&self) -> &[Fraction] {
        &self.window_dimensions
    }

    fn resolve_spatial_values(
        values: &[Fraction],
        spatial_dims: usize,
        default_value: Fraction,
        name: &'static str,
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

    /// Returns explicit pre-padding fractions or zero defaults for each of `spatial_dims` axes.
    ///
    /// # Errors
    ///
    /// Returns an error if the explicit nonempty array length differs from `spatial_dims`. Fraction validity is checked by [`Self::resolve_geometry`].
    pub fn effective_pre_paddings(&self, spatial_dims: usize) -> Result<Vec<Fraction>> {
        Self::resolve_spatial_values(
            &self.pre_paddings,
            spatial_dims,
            Fraction::whole(0),
            "resample pre paddings",
        )
    }

    /// Returns explicit post-padding fractions or zero defaults for each of `spatial_dims` axes.
    ///
    /// # Errors
    ///
    /// Returns an error if the explicit nonempty array length differs from `spatial_dims`. Fraction validity is checked by [`Self::resolve_geometry`].
    pub fn effective_post_paddings(&self, spatial_dims: usize) -> Result<Vec<Fraction>> {
        Self::resolve_spatial_values(
            &self.post_paddings,
            spatial_dims,
            Fraction::whole(0),
            "resample post paddings",
        )
    }

    /// Returns explicit stride fractions or unit defaults for each of `spatial_dims` axes.
    ///
    /// # Errors
    ///
    /// Returns an error if the explicit nonempty array length differs from `spatial_dims`. Fraction validity is checked by [`Self::resolve_geometry`].
    pub fn effective_strides(&self, spatial_dims: usize) -> Result<Vec<Fraction>> {
        Self::resolve_spatial_values(
            &self.strides,
            spatial_dims,
            Fraction::whole(1),
            "resample strides",
        )
    }

    /// Returns explicit window extent fractions or unit defaults for each of `spatial_dims` axes.
    ///
    /// # Errors
    ///
    /// Returns an error if the explicit nonempty array length differs from `spatial_dims`. Fraction validity is checked by [`Self::resolve_geometry`].
    pub fn effective_window_dims(&self, spatial_dims: usize) -> Result<Vec<Fraction>> {
        Self::resolve_spatial_values(
            &self.window_dimensions,
            spatial_dims,
            Fraction::whole(1),
            "resample window dims",
        )
    }
}

impl_inherited_compute_type!(ResampleConfig);
