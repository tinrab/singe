use std::borrow::Cow;

use serde::{Deserialize, Serialize};

pub use crate::backend::execution::pointwise::Activation;
use crate::{
    backend::{
        execution::pointwise::{
            ActivationDirection, PointwiseDescriptor, PointwiseDescriptorConfig,
            PointwiseOperation as BackendPointwiseOperation, PointwiseOperationConfig,
        },
        pointwise::{PointwiseMode, PointwiseOperandKind},
        tensor::TensorId,
    },
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, infer_binary_pointwise_dimensions},
        lower::{LoweredOperation, LoweringContext, tensor_at},
    },
    math::NanPropagation,
};

/// Frontend pointwise operation variants.
///
/// cuDNN pointwise operations apply unary, binary, or ternary elementwise modes
/// selected by [`PointwiseMode`]. Some modes use additional attributes such as
/// `ReLU` clipping parameters, axis, scaling factors, compute type, and NaN
/// propagation.
/// Omitted compute precision inherits graph policy during lowering, with `F32` as the no-policy fallback.
/// The result parameter is a tensor identity for recorded operations or an output configuration for inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PointwiseOperation<O = TensorId> {
    /// One-input pointwise mode with an optional index-generation axis.
    Unary {
        /// Optional operation scope used to name inferred output tensors.
        #[serde(default)]
        name: Option<String>,
        /// Native pointwise mode; its required tensor-port form must match this variant.
        mode: PointwiseMode,
        /// Frontend identity of the input tensor.
        input: TensorId,
        /// Result tensor identity or output inference configuration.
        output: O,
        /// Compute precision override; `None` inherits the graph policy during lowering.
        compute_type: Option<DataType>,
        /// Requested NaN behavior, applied to native `ReLU` attributes.
        nan_propagation: NanPropagation,
        /// First native input scaling coefficient, rounded to FP32 unless compute precision is F64.
        #[serde(with = "crate::scalar::float::f64_bits")]
        alpha1: f64,
        /// Index-generation axis; `None` selects the native `-1` default. Explicit axes require index-generation mode.
        axis: Option<i64>,
    },
    /// Forward activation with mode-specific parameters.
    ActivationForward {
        /// Optional operation scope used to name inferred output tensors.
        #[serde(default)]
        name: Option<String>,
        /// Activation function and its mode-specific parameters.
        activation: Activation,
        /// Frontend identity of the input tensor.
        input: TensorId,
        /// Result tensor identity or output inference configuration.
        output: O,
        /// Compute precision override; `None` inherits the graph policy during lowering.
        compute_type: Option<DataType>,
        /// Requested NaN behavior, applied to native `ReLU` attributes.
        nan_propagation: NanPropagation,
        /// First native input scaling coefficient, rounded to FP32 unless compute precision is F64.
        #[serde(with = "crate::scalar::float::f64_bits")]
        alpha1: f64,
    },
    /// Activation derivative producing an input gradient from the output gradient and forward input.
    ActivationBackward {
        /// Optional operation scope used to name inferred output tensors.
        #[serde(default)]
        name: Option<String>,
        /// Activation function and its mode-specific parameters.
        activation: Activation,
        /// Frontend identity of the gradient with respect to the forward output.
        output_gradient: TensorId,
        /// Frontend identity of the forward input tensor.
        input: TensorId,
        /// Input-gradient tensor identity or output inference configuration.
        input_gradient: O,
        /// Compute precision override; `None` inherits the graph policy during lowering.
        compute_type: Option<DataType>,
        /// Requested NaN behavior, applied to native `ReLU` attributes.
        nan_propagation: NanPropagation,
        /// First native input scaling coefficient, rounded to FP32 unless compute precision is F64.
        #[serde(with = "crate::scalar::float::f64_bits")]
        alpha1: f64,
        /// Second native input scaling coefficient, rounded to FP32 unless compute precision is F64.
        #[serde(with = "crate::scalar::float::f64_bits")]
        alpha2: f64,
    },
    /// Two-input pointwise mode with independent native scaling coefficients.
    Binary {
        /// Optional operation scope used to name inferred output tensors.
        #[serde(default)]
        name: Option<String>,
        /// Native pointwise mode; its required tensor-port form must match this variant.
        mode: PointwiseMode,
        /// Frontend identity bound to the native X input port.
        lhs: TensorId,
        /// Frontend identity bound to the native B input port.
        rhs: TensorId,
        /// Result tensor identity or output inference configuration.
        output: O,
        /// Compute precision override; `None` inherits the graph policy during lowering.
        compute_type: Option<DataType>,
        /// Requested NaN behavior, applied to native `ReLU` attributes.
        nan_propagation: NanPropagation,
        /// First native input scaling coefficient, rounded to FP32 unless compute precision is F64.
        #[serde(with = "crate::scalar::float::f64_bits")]
        alpha1: f64,
        /// Second native input scaling coefficient, rounded to FP32 unless compute precision is F64.
        #[serde(with = "crate::scalar::float::f64_bits")]
        alpha2: f64,
    },
    /// Three-input pointwise mode with explicit native X, B, and T ports.
    Ternary {
        /// Optional operation scope used to name inferred output tensors.
        #[serde(default)]
        name: Option<String>,
        /// Native pointwise mode; its required tensor-port form must match this variant.
        mode: PointwiseMode,
        /// Frontend identity bound to the native X input port.
        x: TensorId,
        /// Frontend identity bound to the native B input port.
        b: TensorId,
        /// Frontend identity bound to the native T input port.
        t: TensorId,
        /// Result tensor identity or output inference configuration.
        output: O,
        /// Compute precision override; `None` inherits the graph policy during lowering.
        compute_type: Option<DataType>,
        /// Requested NaN behavior, applied to native `ReLU` attributes.
        nan_propagation: NanPropagation,
        /// First native input scaling coefficient, rounded to FP32 unless compute precision is F64.
        #[serde(with = "crate::scalar::float::f64_bits")]
        alpha1: f64,
        /// Second native input scaling coefficient, rounded to FP32 unless compute precision is F64.
        #[serde(with = "crate::scalar::float::f64_bits")]
        alpha2: f64,
    },
}

impl<O> PointwiseOperation<O> {
    /// Creates a unary request with inherited compute precision, unit scaling and an omitted index axis.
    pub fn unary(mode: PointwiseMode, input: TensorId, output: O) -> Self {
        Self::Unary {
            name: None,
            mode,
            input,
            output,
            compute_type: None,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: None,
        }
    }

    /// Creates a parameterized forward activation with inherited compute precision and unit scaling.
    pub fn activation_forward(activation: Activation, input: TensorId, output: O) -> Self {
        Self::ActivationForward {
            name: None,
            activation,
            input,
            output,
            compute_type: None,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
        }
    }

    /// Creates an activation derivative from the output gradient and forward input.
    pub fn activation_backward(
        activation: Activation,
        output_gradient: TensorId,
        input: TensorId,
        input_gradient: O,
    ) -> Self {
        Self::ActivationBackward {
            name: None,
            activation,
            output_gradient,
            input,
            input_gradient,
            compute_type: None,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        }
    }

    /// Creates a binary request with inherited compute precision and unit input scaling.
    pub fn binary(mode: PointwiseMode, lhs: TensorId, rhs: TensorId, output: O) -> Self {
        Self::Binary {
            name: None,
            mode,
            lhs,
            rhs,
            output,
            compute_type: None,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        }
    }

    /// Creates a three-input request with inherited compute precision and unit input scaling.
    pub fn ternary(mode: PointwiseMode, x: TensorId, b: TensorId, t: TensorId, output: O) -> Self {
        Self::Ternary {
            name: None,
            mode,
            x,
            b,
            t,
            output,
            compute_type: None,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        }
    }

    /// Names inferred results `scope::OUT_0` without replacing explicit output tensor names.
    /// An empty scope leaves inferred results unnamed.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        let (Self::Unary { name: scope, .. }
        | Self::ActivationForward { name: scope, .. }
        | Self::ActivationBackward { name: scope, .. }
        | Self::Binary { name: scope, .. }
        | Self::Ternary { name: scope, .. }) = &mut self;
        *scope = Some(name.into());
        self
    }

    /// Returns the nonempty operation scope, or `None` when no nonempty name is set.
    pub fn name(&self) -> Option<&str> {
        let (Self::Unary { name, .. }
        | Self::ActivationForward { name, .. }
        | Self::ActivationBackward { name, .. }
        | Self::Binary { name, .. }
        | Self::Ternary { name, .. }) = self;
        name.as_deref().filter(|name| !name.is_empty())
    }

    /// Selects compute precision or restores graph inheritance with `None`, independently of result storage.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        let (Self::Unary {
            compute_type: value,
            ..
        }
        | Self::ActivationForward {
            compute_type: value,
            ..
        }
        | Self::ActivationBackward {
            compute_type: value,
            ..
        }
        | Self::Binary {
            compute_type: value,
            ..
        }
        | Self::Ternary {
            compute_type: value,
            ..
        }) = &mut self;
        *value = compute_type.into();
        self
    }

    /// Requested compute precision, or `None` to inherit the graph policy during lowering.
    pub fn compute_type(&self) -> Option<DataType> {
        let (Self::Unary { compute_type, .. }
        | Self::ActivationForward { compute_type, .. }
        | Self::ActivationBackward { compute_type, .. }
        | Self::Binary { compute_type, .. }
        | Self::Ternary { compute_type, .. }) = self;
        *compute_type
    }

    /// Sets the requested NaN behavior used by native `ReLU` attributes.
    #[must_use]
    pub fn with_nan_propagation(mut self, nan_propagation: NanPropagation) -> Self {
        match &mut self {
            Self::Unary {
                nan_propagation: value,
                ..
            }
            | Self::ActivationForward {
                nan_propagation: value,
                ..
            }
            | Self::ActivationBackward {
                nan_propagation: value,
                ..
            }
            | Self::Binary {
                nan_propagation: value,
                ..
            }
            | Self::Ternary {
                nan_propagation: value,
                ..
            } => *value = nan_propagation,
        }
        self
    }

    /// Sets alpha1 and, for forms with a second operand, alpha2.
    #[must_use]
    pub fn with_scaling(mut self, alpha1: f64, alpha2: f64) -> Self {
        match &mut self {
            Self::Unary { alpha1: value, .. } | Self::ActivationForward { alpha1: value, .. } => {
                *value = alpha1;
            }
            Self::ActivationBackward {
                alpha1: first,
                alpha2: second,
                ..
            }
            | Self::Binary {
                alpha1: first,
                alpha2: second,
                ..
            }
            | Self::Ternary {
                alpha1: first,
                alpha2: second,
                ..
            } => {
                *first = alpha1;
                *second = alpha2;
            }
        }
        self
    }

    /// Sets the index-generation axis, or restores the native `-1` default with `None`.
    ///
    /// # Errors
    /// Returns an error when an explicit axis is supplied for another mode or tensor-port form.
    pub fn with_axis(mut self, axis: impl Into<Option<i64>>) -> Result<Self> {
        let axis = axis.into();
        if let Self::Unary {
            mode, axis: target, ..
        } = &mut self
        {
            if axis.is_some() && *mode != PointwiseMode::GenIndex {
                return Err(Error::PointwiseAxisRequiresIndexGeneration);
            }
            *target = axis;
        } else if axis.is_some() {
            return Err(Error::PointwiseAxisRequiresIndexGeneration);
        }
        Ok(self)
    }

    /// Result identity or inference configuration, independent of the operand identities.
    pub fn output(&self) -> &O {
        match self {
            Self::ActivationBackward { input_gradient, .. } => input_gradient,
            Self::Unary { output, .. }
            | Self::ActivationForward { output, .. }
            | Self::Binary { output, .. }
            | Self::Ternary { output, .. } => output,
        }
    }

    /// Requires the selected mode to match this operation's tensor ports.
    pub(crate) fn validate_tensor_ports(&self) -> Result<()> {
        let (mode, kind) = match self {
            Self::Unary { mode, axis, .. } => {
                if axis.is_some() && *mode != PointwiseMode::GenIndex {
                    return Err(Error::PointwiseAxisRequiresIndexGeneration);
                }
                (*mode, PointwiseOperandKind::Unary)
            }
            Self::Binary { mode, .. } => (*mode, PointwiseOperandKind::Binary),
            Self::Ternary { mode, .. } => (*mode, PointwiseOperandKind::Ternary),
            Self::ActivationForward { .. } | Self::ActivationBackward { .. } => return Ok(()),
        };
        if mode.operand_kind() != kind {
            return Err(Error::DescriptorMismatch {
                name: "pointwise tensor ports".into(),
            });
        }
        Ok(())
    }

    /// Primary operand followed by additional operands in native port order.
    /// The primary operand is DY for activation backward and X for other forms.
    pub(crate) fn input_ids(&self) -> (TensorId, [Option<TensorId>; 2]) {
        match self {
            Self::Unary { input, .. } | Self::ActivationForward { input, .. } => {
                (*input, [None, None])
            }
            Self::ActivationBackward {
                output_gradient,
                input,
                ..
            } => (*output_gradient, [Some(*input), None]),
            Self::Binary { lhs, rhs, .. } => (*lhs, [Some(*rhs), None]),
            Self::Ternary { x, b, t, .. } => (*x, [Some(*b), Some(*t)]),
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the operation table enumerates every supported variant and its tensor contract"
    )]
    pub(crate) fn map_output<P>(self, map: impl FnOnce(O) -> P) -> PointwiseOperation<P> {
        match self {
            Self::Unary {
                name,
                mode,
                input,
                output,
                compute_type,
                nan_propagation,
                alpha1,
                axis,
            } => PointwiseOperation::Unary {
                name,
                mode,
                input,
                output: map(output),
                compute_type,
                nan_propagation,
                alpha1,
                axis,
            },
            Self::ActivationForward {
                name,
                activation,
                input,
                output,
                compute_type,
                nan_propagation,
                alpha1,
            } => PointwiseOperation::ActivationForward {
                name,
                activation,
                input,
                output: map(output),
                compute_type,
                nan_propagation,
                alpha1,
            },
            Self::ActivationBackward {
                name,
                activation,
                output_gradient,
                input,
                input_gradient,
                compute_type,
                nan_propagation,
                alpha1,
                alpha2,
            } => PointwiseOperation::ActivationBackward {
                name,
                activation,
                output_gradient,
                input,
                input_gradient: map(input_gradient),
                compute_type,
                nan_propagation,
                alpha1,
                alpha2,
            },
            Self::Binary {
                name,
                mode,
                lhs,
                rhs,
                output,
                compute_type,
                nan_propagation,
                alpha1,
                alpha2,
            } => PointwiseOperation::Binary {
                name,
                mode,
                lhs,
                rhs,
                output: map(output),
                compute_type,
                nan_propagation,
                alpha1,
                alpha2,
            },
            Self::Ternary {
                name,
                mode,
                x,
                b,
                t,
                output,
                compute_type,
                nan_propagation,
                alpha1,
                alpha2,
            } => PointwiseOperation::Ternary {
                name,
                mode,
                x,
                b,
                t,
                output: map(output),
                compute_type,
                nan_propagation,
                alpha1,
                alpha2,
            },
        }
    }
}

impl PointwiseOperation {
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        self.validate_tensor_ports()?;
        let (first, others) = self.input_ids();
        let mut expected = Cow::Borrowed(graph.shape(first)?.dimensions());
        for input in others.into_iter().flatten() {
            expected = Cow::Owned(infer_binary_pointwise_dimensions(
                &expected,
                graph.shape(input)?.dimensions(),
            )?);
        }
        let actual = graph.shape(*self.output())?.dimensions();
        let offset = actual.len().checked_sub(expected.len());
        let broadcasts_to_output = offset.is_some_and(|offset| {
            expected
                .iter()
                .zip(&actual[offset..])
                .all(|(&input, &output)| input == 1 || input == output)
        });
        if !broadcasts_to_output {
            let operation = match self {
                Self::Unary { .. } => "unary",
                Self::ActivationForward { .. } => "activation forward",
                Self::ActivationBackward { .. } => "activation backward",
                Self::Binary { .. } => "binary",
                Self::Ternary { .. } => "ternary",
            };
            return Err(FrontendError::PointwiseOutputShapeMismatch {
                operation,
                expected: expected.into_owned(),
                actual: actual.to_vec(),
            }
            .into());
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the operation table enumerates every supported variant and its tensor contract"
    )]
    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let (descriptor_config, operation_config) = match self {
            Self::Unary {
                mode,
                input,
                output,
                compute_type,
                nan_propagation,
                alpha1,
                axis,
                ..
            } => (
                PointwiseDescriptorConfig::new(*mode, context.compute_data_type(*compute_type))
                    .with_nan_propagation(*nan_propagation)
                    .with_axis(*axis)?,
                PointwiseOperationConfig::Unary {
                    x: *input,
                    y: *output,
                    alpha1: *alpha1,
                },
            ),
            Self::ActivationForward {
                activation,
                input,
                output,
                compute_type,
                nan_propagation,
                alpha1,
                ..
            } => (
                PointwiseDescriptorConfig::for_activation(
                    *activation,
                    ActivationDirection::Forward,
                    context.compute_data_type(*compute_type),
                )
                .with_nan_propagation(*nan_propagation),
                PointwiseOperationConfig::Unary {
                    x: *input,
                    y: *output,
                    alpha1: *alpha1,
                },
            ),
            Self::ActivationBackward {
                activation,
                output_gradient,
                input,
                input_gradient,
                compute_type,
                nan_propagation,
                alpha1,
                alpha2,
                ..
            } => (
                PointwiseDescriptorConfig::for_activation(
                    *activation,
                    ActivationDirection::Backward,
                    context.compute_data_type(*compute_type),
                )
                .with_nan_propagation(*nan_propagation),
                PointwiseOperationConfig::ActivationBackward {
                    dy: *output_gradient,
                    x: *input,
                    dx: *input_gradient,
                    alpha1: *alpha1,
                    alpha2: *alpha2,
                },
            ),
            Self::Binary {
                mode,
                lhs,
                rhs,
                output,
                compute_type,
                nan_propagation,
                alpha1,
                alpha2,
                ..
            } => (
                PointwiseDescriptorConfig::new(*mode, context.compute_data_type(*compute_type))
                    .with_nan_propagation(*nan_propagation),
                PointwiseOperationConfig::Binary {
                    x: *lhs,
                    b: *rhs,
                    y: *output,
                    alpha1: *alpha1,
                    alpha2: *alpha2,
                },
            ),
            Self::Ternary {
                mode,
                x,
                b,
                t,
                output,
                compute_type,
                nan_propagation,
                alpha1,
                alpha2,
                ..
            } => (
                PointwiseDescriptorConfig::new(*mode, context.compute_data_type(*compute_type))
                    .with_nan_propagation(*nan_propagation),
                PointwiseOperationConfig::Ternary {
                    x: *x,
                    b: *b,
                    t: *t,
                    y: *output,
                    alpha1: *alpha1,
                    alpha2: *alpha2,
                },
            ),
        };
        let descriptor = PointwiseDescriptor::create(descriptor_config)?;
        let config = operation_config.try_map(|id| tensor_at(context.backend_tensors(), id))?;
        Ok(LoweredOperation::Pointwise(
            BackendPointwiseOperation::create(&descriptor, config)?,
        ))
    }
}
