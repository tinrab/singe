//! Native SDPA, softmax and diagonal-band operations with shared sequence and score-graph operands.

mod diagonal_band;
mod sdpa_backward;
mod sdpa_forward;
mod sequence;
mod softmax;

pub use crate::backend::execution::attention::{
    diagonal_band::{
        DiagonalBandMaskBound, DiagonalBandMaskConfig, DiagonalBandMaskOperation,
        DiagonalBandMaskOperationMetadata,
    },
    sdpa_backward::{
        SdpaBackwardConfig, SdpaBackwardOperation, SdpaBackwardOperationMetadata, SdpaBackwardSink,
        SdpaBackwardTensors,
    },
    sdpa_forward::{
        SdpaForwardConfig, SdpaForwardDropout, SdpaForwardOperation, SdpaForwardOperationMetadata,
        SdpaForwardQuantization, SdpaForwardSoftmax, SdpaForwardTensors,
    },
    sequence::{AttentionSequenceInputs, SequenceLength},
    softmax::{SoftmaxOperation, SoftmaxOperationConfig, SoftmaxOutputs, SoftmaxStatsBase},
};

use crate::backend::{execution::OperationGraph, tensor::BackendTensorUid};

/// Prepared score operation graph and the native tensor UIDs connecting it to SDPA.
#[derive(Debug, Clone, Copy)]
pub struct SdpaSubgraph<'a> {
    /// Prepared native graph implementing the score transformation.
    pub operation_graph: &'a OperationGraph,
    /// Native tensor identity where attention supplies the score-graph input.
    pub input_id: BackendTensorUid,
    /// Native tensor identity whose transformed scores return to attention.
    pub output_id: BackendTensorUid,
}

/// Native tensor identities connecting a score operation graph to attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaSubgraphPorts {
    /// Native identity of the score-transformation input endpoint.
    pub input_id: BackendTensorUid,
    /// Native identity of the transformed-score output endpoint.
    pub output_id: BackendTensorUid,
}

impl SdpaSubgraph<'_> {
    /// Copies the score-graph endpoints without retaining the operation graph wrapper.
    #[must_use]
    pub fn ports(&self) -> SdpaSubgraphPorts {
        SdpaSubgraphPorts {
            input_id: self.input_id,
            output_id: self.output_id,
        }
    }
}
