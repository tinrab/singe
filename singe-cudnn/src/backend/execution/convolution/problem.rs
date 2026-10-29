use crate::backend::{
    descriptor::BackendDescriptorType,
    execution::convolution::{
        ConvolutionDescriptor, ConvolutionDescriptorConfig, ConvolutionScaling,
    },
    tensor::{Tensor, TensorProblemKey},
};

/// Hashable native convolution problem, including its direction and effective scaling.
///
/// Native operations retain this metadata without querying cuDNN. Clone a key for use
/// with [`crate::backend::execution::ExecutionPlanCache`]. Forward, input-gradient and
/// filter-gradient operations use distinct keys with the same logical X/W/Y roles.
/// Tensor identities, layouts, encodings, vectorization, reordering, alignment,
/// placement, constants and ragged relationships participate in equality.
///
/// Scope the cache to the compatible device/context, complete operation graph and
/// compilation policy. Additional fused operations and runtime shape overrides
/// require additional key information. This key describes the declared operation.
/// It is independent of device buffer addresses and owns no native descriptors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConvolutionProblemKey {
    operation_type: BackendDescriptorType,
    config: ConvolutionDescriptorConfig,
    tensors: [TensorProblemKey; 3],
    scaling: ConvolutionScaling,
}

impl ConvolutionProblemKey {
    pub(crate) fn new(
        operation_type: BackendDescriptorType,
        convolution: &ConvolutionDescriptor,
        tensors: [&Tensor; 3],
        scaling: ConvolutionScaling,
    ) -> Self {
        Self {
            operation_type,
            config: convolution.config().clone(),
            tensors: tensors.map(TensorProblemKey::from),
            scaling,
        }
    }

    /// Returns the native descriptor type distinguishing forward, input-gradient, and filter-gradient problems.
    #[must_use]
    pub fn operation_type(&self) -> BackendDescriptorType {
        self.operation_type
    }

    /// Retained convolution math and complete spatial geometry, including both padding sides.
    #[must_use]
    pub fn config(&self) -> &ConvolutionDescriptorConfig {
        &self.config
    }

    /// Effective result scale supplied to cuDNN, widened to FP64 when its native type is FP32.
    /// Does not query native descriptors or allocate.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        match self.scaling {
            ConvolutionScaling::F32 { alpha, .. } => f64::from(f32::from_bits(alpha)),
            ConvolutionScaling::F64 { alpha, .. } => f64::from_bits(alpha),
        }
    }

    /// Effective destination scale supplied to cuDNN, widened to FP64 when its native type is FP32.
    /// A nonzero coefficient means execution consumes the destination's initial values.
    /// Does not query native descriptors or allocate.
    #[must_use]
    pub fn beta(&self) -> f64 {
        match self.scaling {
            ConvolutionScaling::F32 { beta, .. } => f64::from(f32::from_bits(beta)),
            ConvolutionScaling::F64 { beta, .. } => f64::from_bits(beta),
        }
    }

    /// Tensor snapshots in logical `[X, W, Y]` order for every direction.
    /// Data gradients use `[dX, W, dY]`; filter gradients use `[X, dW, dY]`.
    /// Does not allocate or query native descriptors.
    #[must_use]
    pub fn tensors(&self) -> &[TensorProblemKey; 3] {
        &self.tensors
    }
}
