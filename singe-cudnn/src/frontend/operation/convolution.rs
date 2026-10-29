use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        convolution::ConvolutionMode,
        execution::convolution::{
            ConvolutionBackwardDataOperation as BackendConvolutionBackwardDataOperation,
            ConvolutionBackwardFilterOperation as BackendConvolutionBackwardFilterOperation,
            ConvolutionDescriptor, ConvolutionDescriptorConfig,
            ConvolutionForwardOperation as BackendConvolutionForwardOperation, ConvolutionGeometry,
        },
        tensor::{Shape, TensorId},
    },
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph},
        lower::{LoweredOperation, LoweringContext, tensor_at},
    },
    scalar::native_f32,
    utility::to_usize,
};

/// Frontend convolution operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ConvolutionOperation {
    /// Forward convolution or cross-correlation.
    Forward {
        /// Input activation tensor identity.
        x: TensorId,
        /// Filter-weight tensor identity.
        w: TensorId,
        /// Forward output tensor identity.
        y: TensorId,
        /// Spatial geometry, convolution mode, computation precision, and destination scaling.
        config: ConvolutionConfig,
    },
    /// Convolution gradient with respect to the input activation.
    BackwardData {
        /// Filter-weight tensor identity.
        w: TensorId,
        /// Gradient tensor identity for the forward output.
        dy: TensorId,
        /// Input-gradient destination tensor identity.
        dx: TensorId,
        /// Spatial geometry, convolution mode, computation precision, and destination scaling.
        config: ConvolutionConfig,
    },
    /// Convolution gradient with respect to the filter weights.
    BackwardFilter {
        /// Input activation tensor identity.
        x: TensorId,
        /// Gradient tensor identity for the forward output.
        dy: TensorId,
        /// Filter-gradient destination tensor identity.
        dw: TensorId,
        /// Spatial geometry, convolution mode, computation precision, and destination scaling.
        config: ConvolutionConfig,
    },
}

impl ConvolutionOperation {
    fn destination_and_config(&self) -> (TensorId, &ConvolutionConfig) {
        match self {
            Self::Forward { y, config, .. } => (*y, config),
            Self::BackwardData { dx, config, .. } => (*dx, config),
            Self::BackwardFilter { dw, config, .. } => (*dw, config),
        }
    }

    /// Returns the destination when the graph precision policy makes the convolution read it.
    pub(crate) fn reads_destination(&self, policy: DataTypePolicy) -> Option<TensorId> {
        let (destination, config) = self.destination_and_config();
        config.reads_destination(policy).then_some(destination)
    }

    /// Implicit-GEMM kernels address 2D and 3D filters with signed 32-bit offsets.
    pub(crate) fn validate_forward_filter_size(
        tensor_id: TensorId,
        shape: &Shape,
        vector_count: i64,
    ) -> Result<()> {
        if !matches!(shape.rank(), 4 | 5) {
            return Ok(());
        }
        let elements = shape
            .element_count()?
            .checked_mul(to_usize(vector_count, "tensor vector count")?)
            .ok_or_else(|| Error::out_of_range("convolution filter element count"))?;
        if i32::try_from(elements).is_err() {
            return Err(FrontendError::ConvolutionFilterTooLarge {
                tensor_id,
                elements,
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        let (input, filter, response) = match self {
            Self::Forward { x, w, y, .. } => (*x, *w, *y),
            Self::BackwardData { w, dy, dx, .. } => (*dx, *w, *dy),
            Self::BackwardFilter { x, dy, dw, .. } => (*x, *dw, *dy),
        };
        let (destination, config) = self.destination_and_config();
        let input_tensor = graph.tensor_spec(input)?;
        let filter_tensor = graph.tensor_spec(filter)?;
        graph.validate_accumulation_output(
            destination,
            config.reads_destination(graph.config.data_type_policy),
            "convolution",
        )?;
        let input_dimensions = input_tensor.logical_dimensions()?;
        let filter_dimensions = filter_tensor.logical_dimensions()?;
        let expected = config
            .resolve_geometry()?
            .infer_forward_dimensions(&input_dimensions, &filter_dimensions)?;
        let actual = graph.tensor_spec(response)?.logical_dimensions()?;
        if actual != expected {
            return Err(FrontendError::TensorDimensionsMismatch {
                tensor_id: response,
                operation: "convolution response shape",
                expected,
                actual,
            }
            .into());
        }

        if matches!(self, Self::Forward { .. }) {
            Self::validate_forward_filter_size(
                filter,
                &filter_tensor.shape,
                filter_tensor.vectorization()?.map_or(1, |(count, _)| count),
            )?;
        }
        Ok(())
    }

    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        let (_, config) = self.destination_and_config();
        let descriptor = ConvolutionDescriptor::create(ConvolutionDescriptorConfig::new(
            context.compute_data_type(config.compute_type()),
            config.mode(),
            config.resolve_geometry()?,
        ))?;
        match self {
            Self::Forward { x, w, y, .. } => Ok(LoweredOperation::ConvolutionForward(
                BackendConvolutionForwardOperation::create(
                    &descriptor,
                    tensor_at(tensors, *x)?,
                    tensor_at(tensors, *w)?,
                    tensor_at(tensors, *y)?,
                    config.alpha(),
                    config.beta(),
                )?,
            )),
            Self::BackwardData { w, dy, dx, .. } => Ok(LoweredOperation::ConvolutionBackwardData(
                BackendConvolutionBackwardDataOperation::create(
                    &descriptor,
                    tensor_at(tensors, *w)?,
                    tensor_at(tensors, *dy)?,
                    tensor_at(tensors, *dx)?,
                    config.alpha(),
                    config.beta(),
                )?,
            )),
            Self::BackwardFilter { x, dy, dw, .. } => {
                Ok(LoweredOperation::ConvolutionBackwardFilter(
                    BackendConvolutionBackwardFilterOperation::create(
                        &descriptor,
                        tensor_at(tensors, *x)?,
                        tensor_at(tensors, *dy)?,
                        tensor_at(tensors, *dw)?,
                        config.alpha(),
                        config.beta(),
                    )?,
                ))
            }
        }
    }
}

/// Attributes shared by frontend convolution operations.
///
/// Used for forward convolution, data-gradient convolution, and weight-gradient
/// convolution. It corresponds to the C++ frontend convolution attribute
/// setters for padding, stride, dilation, convolution mode, and
/// compute data type.
/// Groups are derived from the input and filter channel dimensions, as in C++.
/// Omitted compute precision inherits graph policy for every convolution direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvolutionConfig {
    name: Option<String>,
    compute_type: Option<DataType>,
    #[serde(with = "crate::scalar::float::f64_bits")]
    alpha: f64,
    #[serde(with = "crate::scalar::float::f64_bits")]
    beta: f64,
    mode: ConvolutionMode,
    pre_paddings: Vec<i64>,
    post_paddings: Vec<i64>,
    strides: Vec<i64>,
    dilations: Vec<i64>,
}

impl ConvolutionConfig {
    /// Uses graph compute precision, defaulting to `F32`, with unit strides and dilation and zero padding.
    #[must_use]
    pub fn new(spatial_dimensions: usize) -> Self {
        Self {
            name: None,
            compute_type: None,
            alpha: 1.0,
            beta: 0.0,
            mode: ConvolutionMode::CrossCorrelation,
            pre_paddings: vec![0; spatial_dimensions],
            post_paddings: vec![0; spatial_dimensions],
            strides: vec![1; spatial_dimensions],
            dilations: vec![1; spatial_dimensions],
        }
    }

    /// Names inferred results `name::Y`, `name::DX` or `name::DW` for the selected direction.
    /// Explicit tensor names are retained, and an empty scope disables naming.
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

    /// Selects convolution or cross-correlation mode.
    #[must_use]
    pub fn with_mode(mut self, mode: ConvolutionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Materializes validated spatial geometry for shape inference or native descriptor construction.
    /// The result owns the exact pre/post padding, stride and dilation values.
    ///
    /// # Errors
    /// Returns an error for empty or mismatched arrays, or nonpositive strides and dilations.
    pub fn resolve_geometry(&self) -> Result<ConvolutionGeometry> {
        ConvolutionGeometry::create(
            self.pre_paddings.clone(),
            self.post_paddings.clone(),
            self.strides.clone(),
            self.dilations.clone(),
        )
    }

    /// Overrides graph compute precision without changing operand or result encodings.
    /// Passing `None` restores graph compute inheritance.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.compute_type = compute_type.into();
        self
    }

    /// Computes `destination = alpha * result + beta * destination` in any direction.
    /// A nonzero effective `beta` requires initialized bindable destination storage.
    /// FP32 computation rounds coefficients to FP32 before deciding whether the destination is read.
    /// The compute policy is resolved again during preparation.
    #[must_use]
    pub fn with_scaling(mut self, alpha: f64, beta: f64) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Requested result coefficient, before native precision conversion.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Requested destination coefficient, before native precision conversion.
    #[must_use]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    pub(crate) fn reads_destination(&self, policy: DataTypePolicy) -> bool {
        if policy.resolve_compute(self.compute_type) == DataType::F64 {
            self.beta != 0.0
        } else {
            native_f32(self.beta) != 0.0
        }
    }

    /// Sets equal pre- and post-padding arrays for the spatial axes.
    #[must_use]
    pub fn with_padding(mut self, paddings: impl Into<Vec<i64>>) -> Self {
        let paddings = paddings.into();
        self.pre_paddings.clone_from(&paddings);
        self.post_paddings = paddings;
        self
    }

    /// Replaces spatial pre-padding values without changing post-padding.
    #[must_use]
    pub fn with_pre_paddings(mut self, pre_paddings: impl Into<Vec<i64>>) -> Self {
        self.pre_paddings = pre_paddings.into();
        self
    }

    /// Replaces spatial post-padding values without changing pre-padding.
    #[must_use]
    pub fn with_post_paddings(mut self, post_paddings: impl Into<Vec<i64>>) -> Self {
        self.post_paddings = post_paddings.into();
        self
    }

    /// Replaces the spatial stride values; geometry resolution validates them.
    #[must_use]
    pub fn with_strides(mut self, strides: impl Into<Vec<i64>>) -> Self {
        self.strides = strides.into();
        self
    }

    /// Replaces spatial dilation factors; geometry resolution validates them.
    #[must_use]
    pub fn with_dilations(mut self, dilations: impl Into<Vec<i64>>) -> Self {
        self.dilations = dilations.into();
        self
    }

    /// Returns the compute-precision override, or `None` to inherit graph policy.
    #[must_use]
    pub fn compute_type(&self) -> Option<DataType> {
        self.compute_type
    }

    /// Returns the convolution or cross-correlation mode.
    #[must_use]
    pub fn mode(&self) -> ConvolutionMode {
        self.mode
    }

    /// Borrows the spatial padding before the input.
    #[must_use]
    pub fn pre_paddings(&self) -> &[i64] {
        &self.pre_paddings
    }

    /// Borrows the spatial padding after the input.
    #[must_use]
    pub fn post_paddings(&self) -> &[i64] {
        &self.post_paddings
    }

    /// Borrows the spatial stride values.
    #[must_use]
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    /// Borrows the spatial dilation factors.
    #[must_use]
    pub fn dilations(&self) -> &[i64] {
        &self.dilations
    }
}

impl_inherited_compute_type!(ConvolutionConfig);
