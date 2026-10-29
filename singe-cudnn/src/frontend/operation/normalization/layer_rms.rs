use serde::{Deserialize, Serialize};

use crate::{
    backend::{normalization::BackendNormalizationForwardPhase, tensor::TensorId},
    frontend::operation::normalization::{NormalizationSavedStatistics, NormalizationStatistics},
};

/// Affine operands accepted by `LayerNorm`.
/// A bias requires a scale, matching the native frontend contract.
/// Each operand retains its declared layout; bias broadcasts independently of scale.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerNormalizationAffine {
    /// Normalizes without an affine scale or bias.
    #[default]
    None,
    /// Applies the supplied scale tensor after normalization.
    Scale(TensorId),
    /// Applies the supplied scale and independently broadcast bias after normalization.
    ScaleBias {
        /// Affine scale tensor identity.
        scale: TensorId,
        /// Independently broadcast affine bias tensor identity.
        bias: TensorId,
    },
}

impl LayerNormalizationAffine {
    /// Returns the optional affine scale tensor identity.
    #[must_use]
    pub fn scale(self) -> Option<TensorId> {
        match self {
            Self::None => None,
            Self::Scale(scale) | Self::ScaleBias { scale, .. } => Some(scale),
        }
    }

    /// Returns the optional affine bias tensor identity.
    #[must_use]
    pub fn bias(self) -> Option<TensorId> {
        match self {
            Self::ScaleBias { bias, .. } => Some(bias),
            _ => None,
        }
    }
}

/// Layer normalization input activation and optional affine operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerNormalizationInputs {
    /// Input activation tensor identity.
    pub input: TensorId,
    /// Optional affine scale and bias operands.
    pub affine: LayerNormalizationAffine,
}

/// Layer normalization activation output and training-statistics request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerNormalizationOutputs<T = TensorId> {
    /// Normalized activation tensor identity or its inference policy.
    pub output: T,
    /// Training-statistics outputs or an inference-phase request.
    pub statistics: NormalizationStatistics<T>,
}

/// Root-mean-square normalization input and independently optional scale and bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmsNormalizationInputs {
    /// Input activation tensor identity.
    pub input: TensorId,
    /// Optional affine scale tensor identity.
    pub scale: Option<TensorId>,
    /// Independently broadcast bias; its dimensions do not select normalization axes.
    pub bias: Option<TensorId>,
}

/// Selects RMS inference or training with a saved reciprocal normalization factor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RmsNormalizationStatistics<T = TensorId> {
    /// Omits the saved reciprocal normalization factor.
    #[default]
    Inference,
    /// Produces a saved reciprocal normalization factor for backward execution.
    Training {
        /// Saved reciprocal normalization factor tensor identity or its inference policy.
        inv_variance: T,
    },
}

impl<T> RmsNormalizationStatistics<T> {
    /// Returns the native inference or training phase selected by this statistics request.
    pub fn phase(self) -> BackendNormalizationForwardPhase {
        match self {
            Self::Inference => BackendNormalizationForwardPhase::Inference,
            Self::Training { .. } => BackendNormalizationForwardPhase::Training,
        }
    }

    /// Returns the saved reciprocal-factor output in training, or `None` for inference.
    pub fn inv_variance(self) -> Option<T> {
        match self {
            Self::Inference => None,
            Self::Training { inv_variance } => Some(inv_variance),
        }
    }
}

/// RMS normalization activation output and optional saved training statistic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmsNormalizationOutputs<T = TensorId> {
    /// Normalized activation tensor identity or its inference policy.
    pub output: T,
    /// Training-statistics outputs or an inference-phase request.
    pub statistics: RmsNormalizationStatistics<T>,
}

macro_rules! normalization_forward_config {
    ($name:ident) => {
        /// Epsilon and optional explicit normalization axes.
        /// Without axes, affine or saved-statistics dimensions describe the reduction.
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct $name {
            name: Option<String>,
            epsilon: TensorId,
            axes: Option<Vec<usize>>,
        }

        impl $name {
            /// Records the epsilon tensor with inferred normalization axes and no output-name scope.
            pub fn new(epsilon: TensorId) -> Self {
                Self {
                    name: None,
                    epsilon,
                    axes: None,
                }
            }

            /// Names inferred activations and saved statistics using C++ output scopes.
            /// Explicit tensor names are retained, and an empty scope disables naming.
            #[must_use]
            pub fn with_name(mut self, name: impl Into<String>) -> Self {
                self.name = Some(name.into());
                self
            }

            /// Returns the nonempty output-name scope, or `None` when unnamed.
            pub fn name(&self) -> Option<&str> {
                self.name.as_deref().filter(|name| !name.is_empty())
            }

            /// Selects normalized axes through affine or saved-statistics dimensions.
            /// Scale-free inference still follows the backend's rank-based axis rules.
            /// `None` restores axis inference without changing epsilon or the diagnostic scope.
            /// An explicit empty list remains invalid when constructing an operation.
            #[must_use]
            pub fn with_axes(mut self, axes: impl Into<Option<Vec<usize>>>) -> Self {
                self.axes = axes.into();
                self
            }

            /// Returns the scalar epsilon tensor used for numerical stability.
            pub fn epsilon(&self) -> TensorId {
                self.epsilon
            }

            /// Borrows the explicit zero-based normalization axes, or `None` to infer them.
            pub fn axes(&self) -> Option<&[usize]> {
                self.axes.as_deref()
            }

            pub(crate) fn epsilon_mut(&mut self) -> &mut TensorId {
                &mut self.epsilon
            }
        }
    };
}

normalization_forward_config!(LayerNormalizationConfig);

normalization_forward_config!(RmsNormalizationConfig);

/// Layer normalization backward operands, including saved forward statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerNormalizationBackwardInputs {
    /// Input activation tensor identity.
    pub input: TensorId,
    /// Saved mean and reciprocal normalization factor from forward training.
    pub statistics: NormalizationSavedStatistics,
    /// Gradient tensor identity for the forward output.
    pub output_gradient: TensorId,
    /// Optional affine scale tensor identity.
    pub scale: Option<TensorId>,
}

/// `LayerNorm` with a scale produces both affine gradients.
/// Explicit declarations retain independent broadcast layouts; inference defaults to the scale layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerNormalizationAffineGradients<T = TensorId> {
    /// Scale-gradient tensor identity or its inference policy.
    pub scale_gradient: T,
    /// Bias-gradient tensor identity or its inference policy.
    pub bias_gradient: T,
}

/// Layer normalization input gradient and optional affine gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerNormalizationBackwardOutputs<T = TensorId> {
    /// Input-gradient tensor identity or its inference policy.
    pub input_gradient: T,
    /// Optional affine-gradient outputs; `None` omits affine parameter gradients.
    pub affine: Option<LayerNormalizationAffineGradients<T>>,
}

/// Diagnostic name and optional explicit reduction axes for layer normalization backward.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerNormalizationBackwardConfig {
    name: Option<String>,
    axes: Option<Vec<usize>>,
}

impl LayerNormalizationBackwardConfig {
    /// Creates an unnamed backward request with axes inferred from scale and saved statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Names inferred gradients `name::DX`, `name::DSCALE` and `name::DBIAS`.
    /// Explicit tensor names are retained, and an empty scope disables naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty gradient-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Selects normalized axes, or restores inference from scale and saved statistics with `None`.
    /// The diagnostic scope is retained, and an explicit empty list remains invalid.
    #[must_use]
    pub fn with_axes(mut self, axes: impl Into<Option<Vec<usize>>>) -> Self {
        self.axes = axes.into();
        self
    }

    /// Borrows the explicitly selected zero-based reduction axes, or `None` to infer them.
    #[must_use]
    pub fn axes(&self) -> Option<&[usize]> {
        self.axes.as_deref()
    }
}

/// RMS normalization backward operands and saved reciprocal normalization factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmsNormalizationBackwardInputs {
    /// Input activation tensor identity.
    pub input: TensorId,
    /// Saved reciprocal normalization factor tensor identity or its inference policy.
    pub inv_variance: TensorId,
    /// Gradient tensor identity for the forward output.
    pub output_gradient: TensorId,
    /// Optional affine scale tensor identity.
    pub scale: Option<TensorId>,
}

/// RMS affine gradients with a required scale gradient and independently optional bias gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmsNormalizationAffineGradients<T = TensorId> {
    /// Explicit declarations retain their broadcast layout; inference defaults to the scale layout.
    pub scale_gradient: T,
    /// Independently declared bias gradient, or `None` to omit it.
    pub bias_gradient: Option<T>,
}

/// RMS normalization input gradient and optional affine-gradient outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmsNormalizationBackwardOutputs<T = TensorId> {
    /// Input-gradient tensor identity or its inference policy.
    pub input_gradient: T,
    /// Optional affine-gradient outputs; `None` omits affine parameter gradients.
    pub affine: Option<RmsNormalizationAffineGradients<T>>,
}

/// Diagnostic name and optional explicit reduction axes for RMS normalization backward.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RmsNormalizationBackwardConfig {
    name: Option<String>,
    axes: Option<Vec<usize>>,
}

impl RmsNormalizationBackwardConfig {
    /// Creates an unnamed backward request with axes inferred from scale and saved statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Names inferred gradients `name::DX`, `name::Dscale` and optional `name::Dbias`.
    /// Explicit tensor names are retained, and an empty scope disables naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty gradient-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Selects normalized axes, or restores inference from scale and saved statistics with `None`.
    /// The diagnostic scope is retained, and an explicit empty list remains invalid.
    #[must_use]
    pub fn with_axes(mut self, axes: impl Into<Option<Vec<usize>>>) -> Self {
        self.axes = axes.into();
        self
    }

    /// Borrows the explicitly selected zero-based reduction axes, or `None` to infer them.
    #[must_use]
    pub fn axes(&self) -> Option<&[usize]> {
        self.axes.as_deref()
    }
}
