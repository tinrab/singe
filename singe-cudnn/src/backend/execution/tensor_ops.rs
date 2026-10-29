//! Native tensor transformations and signaling operations.

mod concat;
mod reshape;
mod signal;
mod slice;
mod transpose;

pub use concat::{ConcatConfig, ConcatInPlaceMode, ConcatOperation, ConcatOperationMetadata};
pub use reshape::{ReshapeConfig, ReshapeMode, ReshapeOperation};
pub use signal::{SignalMode, SignalOperation, SignalOperationConfig};
pub use slice::{SliceConfig, SliceOperation};
pub use transpose::{TransposeConfig, TransposeOperation};

use crate::backend::tensor::BackendTensorUid;

/// Native operand identities and retained configuration for a single-input tensor transformation.
/// Reading a snapshot requires no native queries or allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorTransformMetadata<C> {
    /// Native identity of the transformation input.
    pub input: BackendTensorUid,
    /// Native identity of the transformation output.
    pub output: BackendTensorUid,
    /// Complete retained transformation geometry, mode, and diagnostic scope.
    pub config: C,
}
