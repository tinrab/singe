use std::{convert::Infallible, result::Result as StdResult};

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        pointwise::{PointwiseMode, PointwiseOperandKind},
        tensor::{BackendTensorUid, Tensor, TensorId},
    },
    data_type::DataType,
    error::{Error, Result},
    math::NanPropagation,
    scalar::native_f32,
};

/// Activation parameters shared by forward and derivative operations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    /// Clipped or leaky rectified linear activation.
    Relu {
        /// Lower clipping threshold for the native `ReLU` function.
        #[serde(with = "crate::scalar::float::f64_bits")]
        lower_clip: f64,
        /// Upper clipping threshold; F32 compute caps this at `f32::MAX`.
        #[serde(with = "crate::scalar::float::f64_bits")]
        upper_clip: f64,
        /// Slope below the lower clipping threshold.
        #[serde(with = "crate::scalar::float::f64_bits")]
        lower_clip_slope: f64,
    },
    /// Exponential linear activation with a configurable alpha parameter.
    Elu {
        /// Native ELU alpha parameter.
        #[serde(with = "crate::scalar::float::f64_bits")]
        alpha: f64,
    },
    /// Softplus activation with a configurable beta parameter.
    Softplus {
        /// Native softplus beta parameter.
        #[serde(with = "crate::scalar::float::f64_bits")]
        beta: f64,
    },
    /// Swish activation with a configurable beta parameter.
    Swish {
        /// Native swish beta parameter.
        #[serde(with = "crate::scalar::float::f64_bits")]
        beta: f64,
    },
    /// Hyperbolic tangent activation.
    Tanh,
    /// Logistic sigmoid activation.
    Sigmoid,
    /// Gaussian error linear activation.
    Gelu,
    /// Gaussian error linear activation using the hyperbolic tangent approximation.
    GeluApproxTanh,
}

/// Selects the forward activation or its derivative mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationDirection {
    /// Evaluates the activation function.
    Forward,
    /// Evaluates the activation derivative using gradient tensor ports.
    Backward,
}

impl Activation {
    /// `ReLU` with C++'s default clipping and zero negative slope.
    #[must_use]
    pub const fn relu() -> Self {
        Self::Relu {
            lower_clip: 0.0,
            upper_clip: f64::MAX,
            lower_clip_slope: 0.0,
        }
    }

    /// Returns the native pointwise mode for this activation and execution direction.
    #[must_use]
    pub fn mode(self, direction: ActivationDirection) -> PointwiseMode {
        match (self, direction) {
            (Self::Relu { .. }, ActivationDirection::Forward) => PointwiseMode::ReluFwd,
            (Self::Relu { .. }, ActivationDirection::Backward) => PointwiseMode::ReluBwd,
            (Self::Elu { .. }, ActivationDirection::Forward) => PointwiseMode::EluFwd,
            (Self::Elu { .. }, ActivationDirection::Backward) => PointwiseMode::EluBwd,
            (Self::Softplus { .. }, ActivationDirection::Forward) => PointwiseMode::SoftplusFwd,
            (Self::Softplus { .. }, ActivationDirection::Backward) => PointwiseMode::SoftplusBwd,
            (Self::Swish { .. }, ActivationDirection::Forward) => PointwiseMode::SwishFwd,
            (Self::Swish { .. }, ActivationDirection::Backward) => PointwiseMode::SwishBwd,
            (Self::Tanh, ActivationDirection::Forward) => PointwiseMode::TanhFwd,
            (Self::Tanh, ActivationDirection::Backward) => PointwiseMode::TanhBwd,
            (Self::Sigmoid, ActivationDirection::Forward) => PointwiseMode::SigmoidFwd,
            (Self::Sigmoid, ActivationDirection::Backward) => PointwiseMode::SigmoidBwd,
            (Self::Gelu, ActivationDirection::Forward) => PointwiseMode::GeluFwd,
            (Self::Gelu, ActivationDirection::Backward) => PointwiseMode::GeluBwd,
            (Self::GeluApproxTanh, ActivationDirection::Forward) => {
                PointwiseMode::GeluApproxTanhFwd
            }
            (Self::GeluApproxTanh, ActivationDirection::Backward) => {
                PointwiseMode::GeluApproxTanhBwd
            }
        }
    }
}

/// Resolved native pointwise attributes with mode-compatible activation parameters.
/// Construction resolves activation defaults, the index axis and FP32 `ReLU` clipping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointwiseDescriptorConfig {
    mode: PointwiseMode,
    compute_type: DataType,
    nan_propagation: NanPropagation,
    axis: Option<i64>,
    activation: Option<Activation>,
}

impl PointwiseDescriptorConfig {
    /// Uses C++ activation defaults, propagating `ReLU` NaNs and selecting native index axis `-1`.
    #[must_use]
    pub fn new(mode: PointwiseMode, compute_type: DataType) -> Self {
        let activation = match mode {
            PointwiseMode::ReluFwd | PointwiseMode::ReluBwd => Some(Activation::relu()),
            PointwiseMode::EluFwd | PointwiseMode::EluBwd => Some(Activation::Elu { alpha: 1.0 }),
            PointwiseMode::SoftplusFwd | PointwiseMode::SoftplusBwd => {
                Some(Activation::Softplus { beta: 1.0 })
            }
            PointwiseMode::SwishFwd | PointwiseMode::SwishBwd => {
                Some(Activation::Swish { beta: 1.0 })
            }
            PointwiseMode::TanhFwd | PointwiseMode::TanhBwd => Some(Activation::Tanh),
            PointwiseMode::SigmoidFwd | PointwiseMode::SigmoidBwd => Some(Activation::Sigmoid),
            PointwiseMode::GeluFwd | PointwiseMode::GeluBwd => Some(Activation::Gelu),
            PointwiseMode::GeluApproxTanhFwd | PointwiseMode::GeluApproxTanhBwd => {
                Some(Activation::GeluApproxTanh)
            }
            _ => None,
        };
        Self::from_parts(mode, compute_type, activation)
    }

    /// Selects forward or derivative mode from the activation and retains its effective parameters.
    #[must_use]
    pub fn for_activation(
        activation: Activation,
        direction: ActivationDirection,
        compute_type: DataType,
    ) -> Self {
        Self::from_parts(activation.mode(direction), compute_type, Some(activation))
    }

    fn from_parts(
        mode: PointwiseMode,
        compute_type: DataType,
        mut activation: Option<Activation>,
    ) -> Self {
        if compute_type == DataType::F32
            && let Some(Activation::Relu { upper_clip, .. }) = &mut activation
            && *upper_clip > f64::from(f32::MAX)
        {
            *upper_clip = f64::from(f32::MAX);
        }
        Self {
            mode,
            compute_type,
            nan_propagation: NanPropagation::Propagate,
            axis: (mode == PointwiseMode::GenIndex).then_some(-1),
            activation,
        }
    }

    /// Sets the NaN behavior emitted for native `ReLU` attributes.
    #[must_use]
    pub fn with_nan_propagation(mut self, nan_propagation: NanPropagation) -> Self {
        self.nan_propagation = nan_propagation;
        self
    }

    /// Sets the native index axis, or restores `-1` with `None`.
    ///
    /// # Errors
    /// Returns an error when an explicit axis is supplied for another mode.
    pub fn with_axis(mut self, axis: impl Into<Option<i64>>) -> Result<Self> {
        let axis = axis.into();
        if self.mode == PointwiseMode::GenIndex {
            self.axis = Some(axis.unwrap_or(-1));
        } else if axis.is_some() {
            return Err(Error::PointwiseAxisRequiresIndexGeneration);
        }
        Ok(self)
    }

    /// Returns the resolved native pointwise mode.
    #[must_use]
    pub fn mode(&self) -> PointwiseMode {
        self.mode
    }

    /// Returns the resolved native computation precision.
    #[must_use]
    pub fn compute_type(&self) -> DataType {
        self.compute_type
    }

    /// Requested NaN behavior, used by the native `ReLU` attributes.
    #[must_use]
    pub fn nan_propagation(&self) -> NanPropagation {
        self.nan_propagation
    }

    /// Effective axis for index generation; other modes have no axis attribute.
    #[must_use]
    pub fn axis(&self) -> Option<i64> {
        self.axis
    }

    /// Effective activation parameters, including defaults and native `ReLU` clipping.
    #[must_use]
    pub fn activation(&self) -> Option<Activation> {
        self.activation
    }

    /// Number of native input and output tensor ports, including activation gradient ports.
    #[must_use]
    pub fn port_count(&self) -> usize {
        self.mode.operand_count() + 1
    }
}

/// Finalized native descriptor with its resolved configuration retained for introspection.
#[derive(Debug)]
pub struct PointwiseDescriptor {
    descriptor: BackendDescriptor,
    config: PointwiseDescriptorConfig,
}

impl PointwiseDescriptor {
    /// Creates a descriptor with resolved mode, precision and optional activation/index attributes.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects descriptor creation, attributes or finalization.
    pub fn create(config: PointwiseDescriptorConfig) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Pointwise)?;
        descriptor.set_attribute_enum(BackendAttributeName::PointwiseMode, config.mode)?;
        descriptor
            .set_attribute_enum(BackendAttributeName::PointwiseMathPrec, config.compute_type)?;
        match config.activation {
            Some(Activation::Swish { beta }) => {
                descriptor.set_attribute_f64(BackendAttributeName::PointwiseSwishBeta, beta)?;
            }
            Some(Activation::Elu { alpha }) => {
                descriptor.set_attribute_f64(BackendAttributeName::PointwiseEluAlpha, alpha)?;
            }
            Some(Activation::Softplus { beta }) => {
                descriptor.set_attribute_f64(BackendAttributeName::PointwiseSoftplusBeta, beta)?;
            }
            Some(Activation::Relu {
                lower_clip,
                upper_clip,
                lower_clip_slope,
            }) => {
                descriptor.set_attribute_enum(
                    BackendAttributeName::PointwiseNanPropagation,
                    config.nan_propagation,
                )?;
                descriptor
                    .set_attribute_f64(BackendAttributeName::PointwiseReluLowerClip, lower_clip)?;
                descriptor
                    .set_attribute_f64(BackendAttributeName::PointwiseReluUpperClip, upper_clip)?;
                descriptor.set_attribute_f64(
                    BackendAttributeName::PointwiseReluLowerClipSlope,
                    lower_clip_slope,
                )?;
            }
            _ => {}
        }
        if let Some(axis) = config.axis {
            descriptor.set_attribute_i64(BackendAttributeName::PointwiseAxis, axis)?;
        }
        descriptor.finalize()?;
        Ok(Self { descriptor, config })
    }

    /// Effective attributes without native queries or allocation.
    #[must_use]
    pub fn config(&self) -> &PointwiseDescriptorConfig {
        &self.config
    }

    /// Borrows the finalized native backend descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    fn set_operation_scaling(
        &self,
        descriptor: &mut BackendDescriptor,
        alpha1: f64,
        alpha2: f64,
    ) -> Result<()> {
        if self.config.compute_type == DataType::F64 {
            descriptor.set_attribute_f64(BackendAttributeName::OperationPointwiseAlpha1, alpha1)?;
            descriptor.set_attribute_f64(BackendAttributeName::OperationPointwiseAlpha2, alpha2)?;
        } else {
            descriptor.set_attribute_f32(
                BackendAttributeName::OperationPointwiseAlpha1,
                native_f32(alpha1),
            )?;
            descriptor.set_attribute_f32(
                BackendAttributeName::OperationPointwiseAlpha2,
                native_f32(alpha2),
            )?;
        }
        Ok(())
    }
}

/// Mode-specific native tensor ports and input scaling.
/// Unary forms use a unit second scale, while derivative forms name DY, X and DX explicitly.
/// Ports can carry frontend identities, native tensor references or retained native UIDs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointwiseOperationConfig<T = TensorId> {
    /// Native X input and Y output ports for a unary mode.
    Unary {
        /// Tensor bound to the native X input port.
        x: T,
        /// Tensor bound to the native Y output port.
        y: T,
        /// Requested first native input scaling coefficient.
        alpha1: f64,
    },
    /// Native X and B input ports and Y output port for a binary mode.
    Binary {
        /// Tensor bound to the native X input port.
        x: T,
        /// Tensor bound to the native B input port.
        b: T,
        /// Tensor bound to the native Y output port.
        y: T,
        /// Requested first native input scaling coefficient.
        alpha1: f64,
        /// Requested second native input scaling coefficient.
        alpha2: f64,
    },
    /// Native X, B, and T input ports and Y output port for a ternary mode.
    Ternary {
        /// Tensor bound to the native X input port.
        x: T,
        /// Tensor bound to the native B input port.
        b: T,
        /// Tensor bound to the native T input port.
        t: T,
        /// Tensor bound to the native Y output port.
        y: T,
        /// Requested first native input scaling coefficient.
        alpha1: f64,
        /// Requested second native input scaling coefficient.
        alpha2: f64,
    },
    /// Forward-input X, output-gradient DY, and input-gradient DX ports.
    ActivationBackward {
        /// Gradient tensor for the forward output, bound to native DY.
        dy: T,
        /// Tensor bound to the native X input port.
        x: T,
        /// Gradient tensor for the forward input, bound to native DX.
        dx: T,
        /// Requested first native input scaling coefficient.
        alpha1: f64,
        /// Requested second native input scaling coefficient.
        alpha2: f64,
    },
}

impl<T> PointwiseOperationConfig<T> {
    /// Borrows tensor owners while preserving native port roles and requested scaling.
    pub fn as_ref(&self) -> PointwiseOperationConfig<&T> {
        match self {
            Self::Unary { x, y, alpha1 } => PointwiseOperationConfig::Unary {
                x,
                y,
                alpha1: *alpha1,
            },
            Self::Binary {
                x,
                b,
                y,
                alpha1,
                alpha2,
            } => PointwiseOperationConfig::Binary {
                x,
                b,
                y,
                alpha1: *alpha1,
                alpha2: *alpha2,
            },
            Self::Ternary {
                x,
                b,
                t,
                y,
                alpha1,
                alpha2,
            } => PointwiseOperationConfig::Ternary {
                x,
                b,
                t,
                y,
                alpha1: *alpha1,
                alpha2: *alpha2,
            },
            Self::ActivationBackward {
                dy,
                x,
                dx,
                alpha1,
                alpha2,
            } => PointwiseOperationConfig::ActivationBackward {
                dy,
                x,
                dx,
                alpha1: *alpha1,
                alpha2: *alpha2,
            },
        }
    }

    /// Maps tensor ports without changing their roles or requested scaling.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> PointwiseOperationConfig<U> {
        match self.try_map::<_, Infallible>(|tensor| Ok(map(tensor))) {
            Ok(config) => config,
            Err(never) => match never {},
        }
    }

    /// Resolves every tensor port through a fallible mapping.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor-port mapping, without mapping later ports.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<PointwiseOperationConfig<U>, E> {
        Ok(match self {
            Self::Unary { x, y, alpha1 } => PointwiseOperationConfig::Unary {
                x: map(x)?,
                y: map(y)?,
                alpha1,
            },
            Self::Binary {
                x,
                b,
                y,
                alpha1,
                alpha2,
            } => PointwiseOperationConfig::Binary {
                x: map(x)?,
                b: map(b)?,
                y: map(y)?,
                alpha1,
                alpha2,
            },
            Self::Ternary {
                x,
                b,
                t,
                y,
                alpha1,
                alpha2,
            } => PointwiseOperationConfig::Ternary {
                x: map(x)?,
                b: map(b)?,
                t: map(t)?,
                y: map(y)?,
                alpha1,
                alpha2,
            },
            Self::ActivationBackward {
                dy,
                x,
                dx,
                alpha1,
                alpha2,
            } => PointwiseOperationConfig::ActivationBackward {
                dy: map(dy)?,
                x: map(x)?,
                dx: map(dx)?,
                alpha1,
                alpha2,
            },
        })
    }
}

/// Retained native descriptor settings, tensor-port identities and requested input scales.
/// Scaling attributes use FP64 for F64 compute and FP32-rounded values for other compute types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointwiseOperationMetadata {
    /// Resolved mode, compute precision, and activation or axis settings.
    pub descriptor: PointwiseDescriptorConfig,
    /// Retained native tensor identities and requested scaling coefficients before native precision rounding.
    pub config: PointwiseOperationConfig<BackendTensorUid>,
}

/// A native pointwise operation with descriptor-selected scaling precision.
/// Scaling values use FP64 for F64 compute and FP32 for other compute types.
#[derive(Debug)]
pub struct PointwiseOperation {
    descriptor: BackendDescriptor,
    metadata: PointwiseOperationMetadata,
}

impl PointwiseOperation {
    /// Creates one operation with mode-compatible tensor ports and scaling.
    ///
    /// # Errors
    /// Returns an error for a tensor-port form inconsistent with the descriptor's mode or a native descriptor failure.
    pub fn create(
        pointwise: &PointwiseDescriptor,
        config: PointwiseOperationConfig<&Tensor>,
    ) -> Result<Self> {
        let (kind, ports, alpha1, alpha2) = match config {
            PointwiseOperationConfig::Unary { x, y, alpha1 } => (
                PointwiseOperandKind::Unary,
                [
                    Some((BackendAttributeName::OperationPointwiseXDesc, x)),
                    Some((BackendAttributeName::OperationPointwiseYDesc, y)),
                    None,
                    None,
                ],
                alpha1,
                1.0,
            ),
            PointwiseOperationConfig::Binary {
                x,
                b,
                y,
                alpha1,
                alpha2,
            } => (
                PointwiseOperandKind::Binary,
                [
                    Some((BackendAttributeName::OperationPointwiseXDesc, x)),
                    Some((BackendAttributeName::OperationPointwiseBDesc, b)),
                    Some((BackendAttributeName::OperationPointwiseYDesc, y)),
                    None,
                ],
                alpha1,
                alpha2,
            ),
            PointwiseOperationConfig::Ternary {
                x,
                b,
                t,
                y,
                alpha1,
                alpha2,
            } => (
                PointwiseOperandKind::Ternary,
                [
                    Some((BackendAttributeName::OperationPointwiseXDesc, x)),
                    Some((BackendAttributeName::OperationPointwiseBDesc, b)),
                    Some((BackendAttributeName::OperationPointwiseTDesc, t)),
                    Some((BackendAttributeName::OperationPointwiseYDesc, y)),
                ],
                alpha1,
                alpha2,
            ),
            PointwiseOperationConfig::ActivationBackward {
                dy,
                x,
                dx,
                alpha1,
                alpha2,
            } => (
                PointwiseOperandKind::ActivationBackward,
                [
                    Some((BackendAttributeName::OperationPointwiseXDesc, x)),
                    Some((BackendAttributeName::OperationPointwiseDyDesc, dy)),
                    Some((BackendAttributeName::OperationPointwiseDxDesc, dx)),
                    None,
                ],
                alpha1,
                alpha2,
            ),
        };
        if pointwise.config.mode.operand_kind() != kind {
            return Err(Error::DescriptorMismatch {
                name: "pointwise tensor ports".into(),
            });
        }
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationPointwise)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPointwisePwDescriptor,
            pointwise.descriptor(),
        )?;
        for (attribute, tensor) in ports.into_iter().flatten() {
            descriptor.set_attribute_descriptor(attribute, tensor.descriptor())?;
        }
        pointwise.set_operation_scaling(&mut descriptor, alpha1, alpha2)?;
        descriptor.finalize()?;
        Ok(Self {
            descriptor,
            metadata: PointwiseOperationMetadata {
                descriptor: pointwise.config,
                config: config.map(Tensor::id),
            },
        })
    }

    /// Retained descriptor settings, native tensor UIDs and requested scales.
    /// Remains available after the original tensor and pointwise descriptor wrappers are released.
    /// Does not query cuDNN or allocate.
    #[must_use]
    pub fn metadata(&self) -> &PointwiseOperationMetadata {
        &self.metadata
    }

    /// Borrows the finalized native backend descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
